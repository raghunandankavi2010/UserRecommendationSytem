"""Learning-loop jobs. Run from the repo root with the venv, against the database in DATABASE_URL:

    .venv/bin/python ml/jobs.py nightly                    # merge/promote provisional interests, refresh all user vectors
    .venv/bin/python ml/jobs.py retrain [--promote]        # fine-tune on collected labels, gate, register (and promote)
    .venv/bin/python ml/jobs.py promote VERSION            # build vectors for VERSION, make it active
    .venv/bin/python ml/jobs.py rollback                   # re-activate the previously active version
    .venv/bin/python ml/jobs.py report                     # wave / mutual-accept rate per model version

Suggested schedule: nightly every night, retrain weekly (it exits early without enough new labels).
Promotion is safe while the API runs: vectors are built first, the API switches within 30 s, then a
catch-up pass covers anything written in between.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_linking import bench, load as load_eval  # noqa: E402

from matcher import db, interests, versions  # noqa: E402
from matcher.linking import key, st_encoder  # noqa: E402

MODELS_DIR = Path("models")
MIN_LABELS = 200          # positive labels needed before a retrain is worth it
HOLDOUT_EVERY = 5         # 1 in 5 labelled texts (by hash) is held out for the gate
GATE = {"min_auto_precision": 0.95, "max_end_to_end_drop": 0.01, "max_novel_auto_increase": 0.02}


def connect(url: str) -> psycopg.Connection:
    c = psycopg.connect(url, row_factory=dict_row)
    register_vector(c)
    return c


def log(**kv) -> None:
    print(json.dumps(kv, default=str), flush=True)


# ---------- nightly ----------

def nightly(c) -> dict:
    m = versions.load(c)
    embedded = interests.embed_missing(c, m)
    changes = interests.nightly_merge(c, m)
    users = interests.rebuild_user_vectors(c, m.version)
    c.commit()
    return {"job": "nightly", "version": m.version, "embedded": embedded, "users_refreshed": users, **changes}


# ---------- promote / rollback ----------

def build_vectors(c, version: str) -> dict:
    m = versions.load(c, version)
    out = {"interests_embedded": interests.embed_missing(c, m), "users": interests.rebuild_user_vectors(c, version)}
    c.commit()
    return out


def promote(c, version: str, wait: float = versions.RELOAD_SECONDS + 5) -> dict:
    row = c.execute("SELECT status FROM model_versions WHERE version = %s", (version,)).fetchone()
    if row is None:
        raise SystemExit(f"unknown version {version}")
    if row["status"] == "active":
        return {"job": "promote", "version": version, "note": "already active"}
    if row["status"] == "rejected":
        raise SystemExit(f"{version} failed the gate; it cannot be promoted")
    built = build_vectors(c, version)
    c.execute("UPDATE model_versions SET status = 'retired' WHERE status = 'active'")
    c.execute("UPDATE model_versions SET status = 'active', activated_at = now() WHERE version = %s", (version,))
    c.commit()
    log(job="promote", version=version, step="activated, waiting for the API to reload", seconds=wait)
    time.sleep(wait)
    caught_up = build_vectors(c, version)  # interests/users written under the old version meanwhile
    return {"job": "promote", "version": version, "built": built, "catch_up": caught_up}


def rollback(c, wait: float = versions.RELOAD_SECONDS + 5) -> dict:
    prev = c.execute("SELECT version FROM model_versions WHERE status = 'retired' "
                     "ORDER BY activated_at DESC NULLS LAST LIMIT 1").fetchone()
    if prev is None:
        raise SystemExit("no previous version to roll back to")
    return {**promote(c, prev["version"], wait), "job": "rollback"}


# ---------- retrain ----------

def collect_labels(c) -> tuple[list[dict], list[dict]]:
    """(train, holdout) rows of {text, label, y}: y=1 user picked that interest, y=0 user rejected it."""
    rows = c.execute("SELECT DISTINCT lower(l.raw_text) AS text, i.label, (l.label = 1)::int AS y "
                     "FROM interest_labels l JOIN interests i ON i.id = l.interest_id "
                     "WHERE i.status <> 'merged'").fetchall()
    train, holdout = [], []
    for r in rows:
        (holdout if zlib.crc32(r["text"].encode()) % HOLDOUT_EVERY == 0 else train).append(r)
    return train, holdout


def fine_tune(base_path: str, train: list[dict], out: Path, epochs: int) -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(base_path, device="cpu")
    for name, p in model.named_parameters():
        if "embed_tokens" in name:  # the token table is most of the weights: freezing it keeps CPU training light
            p.requires_grad = False
    pos = [r for r in train if r["y"] == 1]
    neg = [r for r in train if r["y"] == 0]
    datasets = {"picked": Dataset.from_dict({"anchor": [r["text"] for r in pos], "positive": [r["label"] for r in pos]})}
    loss = {"picked": losses.MultipleNegativesRankingLoss(model)}
    if neg:  # rejected suggestions: push the text away from the interest the user said it is not
        datasets["rejected"] = Dataset.from_dict({"sentence1": [r["text"] for r in neg],
                                                  "sentence2": [r["label"] for r in neg],
                                                  "label": [0.0] * len(neg)})
        loss["rejected"] = losses.ContrastiveLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "_trainer"), num_train_epochs=epochs, per_device_train_batch_size=16,
        learning_rate=2e-5, warmup_steps=0.1, save_strategy="no", report_to="none", logging_steps=10,
        use_cpu=True, prompts=model.prompts.get("STS"),            # train with the prompt used at inference
        batch_sampler=BatchSamplers.NO_DUPLICATES)                  # same interest twice in a batch = false negative
    SentenceTransformerTrainer(model=model, args=args, train_dataset=datasets, loss=loss).train()
    model.save(str(out))


def real_holdout_scores(c, path: str, auto_min: float, holdout: list[dict]) -> dict:
    """On held-out real labels: is the picked interest top-1 / top-3, and is a rejected one auto-linked?"""
    labels = [r["label"] for r in c.execute("SELECT label FROM interests WHERE status <> 'merged' ORDER BY id")]
    enc = st_encoder(path)
    S = enc([r["text"] for r in holdout]) @ enc(labels).T
    order = np.argsort(-S, axis=1)
    pos = [(i, r) for i, r in enumerate(holdout) if r["y"] == 1]
    neg = [(i, r) for i, r in enumerate(holdout) if r["y"] == 0]
    top1 = np.mean([labels[order[i, 0]] == r["label"] for i, r in pos]) if pos else None
    top3 = np.mean([r["label"] in {labels[j] for j in order[i, :3]} for i, r in pos]) if pos else None
    bad = np.mean([labels[order[i, 0]] == r["label"] and S[i, order[i, 0]] >= auto_min for i, r in neg]) if neg else 0.0
    return {"n_pos": len(pos), "n_neg": len(neg), "top1": top1, "top3": top3, "rejected_auto_linked": float(bad)}


def gate(c, active: versions.Model, cand_path: str, train: list[dict], holdout: list[dict]) -> tuple[bool, dict]:
    labels, texts, expected, dev = load_eval()
    trained = {r["text"] for r in train}
    keep = np.array([key(t) not in trained for t in texts])  # never score the gate on text it trained on
    sub = lambda xs: [x for x, k in zip(xs, keep) if k]  # noqa: E731
    args = (labels, sub(texts), sub(expected), dev[keep])
    old, new = bench(active.path, *args), bench(cand_path, *args)
    new_auto, new_ask = round(new["auto_min"], 2), round(min(new["ask_min"], new["auto_min"]), 2)
    real_old = real_holdout_scores(c, active.path, active.auto_min, holdout)
    real_new = real_holdout_scores(c, cand_path, new_auto, holdout)
    checks = {
        "eval_rows": int(keep.sum()),
        "auto_precision_ok": new["auto_precision"] >= GATE["min_auto_precision"],
        "end_to_end_ok": new["end_to_end"] >= old["end_to_end"] - GATE["max_end_to_end_drop"],
        "novel_ok": new["novel_auto_linked"] <= old["novel_auto_linked"] + GATE["max_novel_auto_increase"],
        "real_labels_improved": real_new["n_pos"] > 0 and (real_new["top1"] > real_old["top1"]
                                                           or real_new["top3"] > real_old["top3"]),
        "rejections_ok": real_new["rejected_auto_linked"] <= real_old["rejected_auto_linked"],
    }
    checks = {k: v if k == "eval_rows" else bool(v) for k, v in checks.items()}
    passed = all(v for k, v in checks.items() if k != "eval_rows")
    keep_cols = ("top1_acc", "top3_acc", "auto_rate", "auto_precision", "novel_auto_linked", "end_to_end")
    metrics = {"passed": passed, "checks": checks, "auto_min": new_auto, "ask_min": new_ask,
               "eval_active": {k: old[k] for k in keep_cols}, "eval_candidate": {k: new[k] for k in keep_cols},
               "real_active": real_old, "real_candidate": real_new, "n_train": len(train), "n_holdout": len(holdout)}
    return passed, metrics


def retrain(c, min_labels: int, epochs: int, do_promote: bool) -> dict:
    active = versions.load(c)
    train, holdout = collect_labels(c)
    n_pos = sum(r["y"] for r in train)
    if n_pos < min_labels:
        return {"job": "retrain", "skipped": f"{n_pos} positive training labels, need {min_labels}"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version = f"{active.version.split('/')[-1].split('@')[0]}-ft-{stamp}"
    out = MODELS_DIR / version
    log(job="retrain", step="fine-tuning", base=active.version, train=len(train), holdout=len(holdout))
    fine_tune(active.path, train, out, epochs)
    passed, metrics = gate(c, active, str(out), train, holdout)
    c.execute("INSERT INTO model_versions (version, path, auto_min, ask_min, status, metrics) VALUES (%s, %s, %s, %s, %s, %s)",
              (version, str(out), metrics["auto_min"], metrics["ask_min"], "candidate" if passed else "rejected",
               json.dumps(metrics, default=float)))
    c.commit()
    if not passed:  # keep the metrics row, drop the ~1 GB of weights
        shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(out / "_trainer", ignore_errors=True)
    result = {"job": "retrain", "version": version, "passed": passed, "metrics": metrics}
    if passed and do_promote:
        result["promote"] = promote(c, version)
    return result


# ---------- report ----------

def report(c) -> list[dict]:
    """Outcome of shown recommendations per model version (the app sends model_version with 'shown')."""
    return c.execute("""
        SELECT s.model_version, count(*) AS shown,
               round(avg((w.id IS NOT NULL)::int), 4) AS wave_rate,
               round(avg((cn.user_id IS NOT NULL)::int), 4) AS mutual_rate
        FROM (SELECT DISTINCT ON (actor_id, target_id) * FROM events WHERE type = 'shown'
              ORDER BY actor_id, target_id, created_at) s
        LEFT JOIN LATERAL (SELECT id FROM events e WHERE e.type = 'wave' AND e.actor_id = s.actor_id
                           AND e.target_id = s.target_id AND e.created_at >= s.created_at LIMIT 1) w ON true
        LEFT JOIN connections cn ON cn.user_id = s.actor_id AND cn.other_id = s.target_id
        GROUP BY s.model_version ORDER BY s.model_version""").fetchall()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nightly")
    r = sub.add_parser("retrain")
    r.add_argument("--promote", action="store_true")
    r.add_argument("--min-labels", type=int, default=MIN_LABELS)
    r.add_argument("--epochs", type=int, default=3)
    pr = sub.add_parser("promote")
    pr.add_argument("version")
    sub.add_parser("rollback")
    sub.add_parser("report")
    a = p.parse_args(argv)
    url = os.environ["DATABASE_URL"]
    db.migrate(url)
    with connect(url) as c:
        if a.cmd == "nightly":
            log(**nightly(c))
        elif a.cmd == "retrain":
            log(**retrain(c, a.min_labels, a.epochs, a.promote))
        elif a.cmd == "promote":
            log(**promote(c, a.version))
        elif a.cmd == "rollback":
            log(**rollback(c))
        else:
            for row in report(c):
                log(**row)


if __name__ == "__main__":
    main()
