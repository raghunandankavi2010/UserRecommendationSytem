"""Pick the embedding model and thresholds for interest linking.

    .venv/bin/python ml/benchmark_linking.py [model ...]

Data: ml/data/taxonomy.json (canonical interests), ml/data/linking_eval.csv
(text -> expected label, "A|B" = either is right, empty = novel, should become "new").
Thresholds are tuned on one half and reported on the other, both ways (2-fold), averaged.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from matcher.linking import InterestLinker, st_encoder

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results" / "linking_benchmark.json"
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "Snowflake/snowflake-arctic-embed-s",
    "sentence-transformers/all-mpnet-base-v2",
]
AUTO_PRECISION = 0.95  # a silent wrong link is the worst outcome; demand >= 95% precision


def load():
    taxonomy = json.loads((DATA / "taxonomy.json").read_text())
    labels = [label for group in taxonomy.values() for label in group]
    rows = list(csv.DictReader((DATA / "linking_eval.csv").open()))
    texts = [r["text"] for r in rows]
    expected = [set(r["expected"].split("|")) if r["expected"] else set() for r in rows]
    rng = np.random.default_rng(7)
    dev = np.zeros(len(rows), bool)
    for is_novel in (True, False):  # stratified half split
        idx = np.array([i for i, e in enumerate(expected) if (not e) == is_novel])
        dev[rng.permutation(idx)[: len(idx) // 2]] = True
    return labels, texts, expected, dev


def outcomes(best, top1_ok, topk_ok, novel, auto_min, ask_min):
    auto, ask = best >= auto_min, (best >= ask_min) & (best < auto_min)
    pos = ~novel
    n_auto = auto.sum()
    return {
        "auto_rate": auto[pos].mean(),
        "auto_precision": (auto & top1_ok & pos).sum() / n_auto if n_auto else 1.0,
        "ask_rate": ask[pos].mean(),
        "ask_hit_rate": (ask & topk_ok)[pos].sum() / max(ask[pos].sum(), 1),  # right answer among suggestions
        "lost_rate": (best < ask_min)[pos].mean(),          # real match created as "new"
        "novel_to_new": (best < ask_min)[novel].mean(),
        "novel_auto_linked": auto[novel].mean(),            # worst error
        # end to end: the user ends up on the right interest. link() always returns the top 3, so a
        # real interest that is not auto-linked is fine if it is among them; a novel one is fine if it
        # is not auto-linked. A silent wrong link is the only unrecoverable outcome.
        "end_to_end": ((auto & top1_ok) | (~auto & topk_ok))[pos].sum() / pos.sum() * pos.mean()
                      + (~auto)[novel].mean() * novel.mean(),
    }


def tune(best, top1_ok, novel):
    grid = np.round(np.arange(0.40, 0.96, 0.01), 2)
    pos = ~novel
    auto_min = grid[-1]
    for t in grid:  # lowest threshold that still meets the precision bar
        sel = best >= t
        if sel.sum() and (sel & top1_ok & pos).sum() / sel.sum() >= AUTO_PRECISION:
            auto_min = t
            break
    # ask_min: best split between real interests and novel ones (Youden's J)
    j = [(best[pos] >= t).mean() - (best[novel] >= t).mean() for t in grid]
    ask_min = min(grid[int(np.argmax(j))], auto_min)
    return float(auto_min), float(ask_min)


def bench(model, labels, texts, expected, dev):
    enc = st_encoder(model)
    t0 = time.perf_counter()
    linker = InterestLinker(labels, enc)
    S = linker.scores(texts)
    batch_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in texts[:50]:
        enc([t])
    single_ms = (time.perf_counter() - t0) / 50 * 1000

    order = np.argsort(-S, axis=1)
    best = S[np.arange(len(texts)), order[:, 0]]
    top1 = [labels[i] for i in order[:, 0]]
    top1_ok = np.array([lab in e for lab, e in zip(top1, expected)])
    topk_ok = np.array([bool({labels[i] for i in row[:3]} & e) for row, e in zip(order, expected)])
    novel = np.array([not e for e in expected])

    # 2-fold: tune on one half, report on the other, then swap and average (each half is small)
    folds = []
    for tune_on in (dev, ~dev):
        test = ~tune_on
        auto_min, ask_min = tune(best[tune_on], top1_ok[tune_on], novel[tune_on])
        folds.append({"auto_min": auto_min, "ask_min": ask_min,
                      **outcomes(best[test], top1_ok[test], topk_ok[test], novel[test], auto_min, ask_min)})
    res = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    pos = ~novel
    misses = [(texts[i], top1[i], round(float(best[i]), 2)) for i in np.where(pos & ~top1_ok)[0]]
    return {"model": model, "top1_acc": float(top1_ok[pos].mean()), "top3_acc": float(topk_ok[pos].mean()),
            **res, "folds": folds,
            "encode_one_ms": round(single_ms, 1), "batch_s": round(batch_s, 1),
            "test_misses": misses}


def main(models):
    labels, texts, expected, dev = load()
    print(f"{len(labels)} labels, {len(texts)} eval rows ({dev.sum()} dev / {(~dev).sum()} test)")
    results = [bench(m, labels, texts, expected, dev) for m in models]
    cols = ["top1_acc", "top3_acc", "auto_min", "ask_min", "auto_rate", "auto_precision",
            "ask_hit_rate", "lost_rate", "novel_to_new", "novel_auto_linked", "end_to_end", "encode_one_ms"]
    print("model".ljust(40) + "".join(c[:12].rjust(13) for c in cols))
    for r in sorted(results, key=lambda r: -r["end_to_end"]):
        print(r["model"].ljust(40) + "".join(f"{r[c]:13.2f}" for c in cols))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
