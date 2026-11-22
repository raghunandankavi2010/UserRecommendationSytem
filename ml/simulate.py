"""Exercise the learning loop before the app sends real data.

    DATABASE_URL=postgresql://matcher:<pw>@127.0.0.1:5432/matcher .venv/bin/python ml/simulate.py [n_users]

Creates a throwaway `matcher_sim` database and fills it through the same code the API runs:
synthetic users type phrases from the *dev half* of ml/data/linking_eval.csv and answer suggestions
with the known right answer (pick it, or "new" for novel interests); others see recommendations and
wave / accept. Prints the URL to run ml/jobs.py against. The test half is never typed, so the
retrain gate still has unseen text to judge on.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_linking import load as load_eval  # noqa: E402
from jobs import connect  # noqa: E402

from matcher import db, interests, matching, versions  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rnd = random.Random(0)
base = os.environ["DATABASE_URL"]
url = base.rsplit("/", 1)[0] + "/matcher_sim"
with psycopg.connect(base, autocommit=True) as c:
    c.execute("DROP DATABASE IF EXISTS matcher_sim WITH (FORCE)")
    c.execute("CREATE DATABASE matcher_sim")
db.migrate(url)

labels, texts, expected, dev = load_eval()
rows = [(t, e) for t, e, d in zip(texts, expected, dev) if d]
with connect(url) as c:
    m = versions.load(c)
    interests.seed_taxonomy(c, m, Path("ml/data/taxonomy.json"))
    ids = {r["label"]: r["id"] for r in c.execute("SELECT id, label FROM interests")}
    stats = {"linked": 0, "picked": 0, "new": 0}
    for u in range(1, N + 1):
        uid = c.execute("INSERT INTO users (external_id, name, email) VALUES (%s, %s, %s) RETURNING id",
                        (f"sim{u}", f"Sim {u}", f"sim{u}@sim.test")).fetchone()["id"]
        for text, exp in rnd.sample(rows, rnd.randint(3, 6)):
            r = interests.add_interest(c, m, uid, text)
            if r["attached"]:
                stats["linked"] += 1
                continue
            if not exp:
                interests.add_interest(c, m, uid, text, "new")
                stats["new"] += 1
                continue
            shown = [s for s in r["suggestions"] if s["label"] in exp]
            pick = shown[0]["id"] if shown else ids[sorted(exp)[0]]  # not suggested: the user searches for it
            interests.add_interest(c, m, uid, text, pick)
            stats["picked"] += 1
    c.commit()
    by_ext = {r["external_id"]: r["id"] for r in c.execute("SELECT id, external_id FROM users")}
    for u in by_ext.values():
        for rec in matching.recommend(c, m.version, u, limit=5):
            other = by_ext[rec["user_id"]]
            c.execute("INSERT INTO events (actor_id, target_id, type, model_version) VALUES (%s, %s, 'shown', %s)",
                      (u, other, m.version))
            if rnd.random() < (0.35 if rec["common_interests"] else 0.05):
                c.execute("INSERT INTO events (actor_id, target_id, type) VALUES (%s, %s, 'wave')", (u, other))
                if rnd.random() < 0.5:
                    c.execute("INSERT INTO events (actor_id, target_id, type) VALUES (%s, %s, 'accept')", (other, u))
    c.commit()
    n_labels = c.execute("SELECT label, count(*) AS n FROM interest_labels GROUP BY label").fetchall()
print({"users": N, **stats, "labels": {r["label"]: r["n"] for r in n_labels}})
print(f"DATABASE_URL={url.split('@')[0].rsplit(':', 1)[0]}:<password>@{url.split('@')[1]}")
