"""Matching latency at scale on a throwaway `matcher_load` database.

    DATABASE_URL=postgresql://matcher:<pw>@127.0.0.1:5432/matcher .venv/bin/python ml/load_test.py [n_users]

Synthetic users get 3-10 seeded interests with Zipf-like popularity (a few very common, a long tail).
Reports p50/p95 of matching.recommend() for 200 random users, with and without candidate_ids,
and how many ANN candidates come back (should be 200: checks the hnsw.ef_search setting).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from matcher import db, interests, matching, versions

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
rng = np.random.default_rng(0)
base = os.environ["DATABASE_URL"]
url = base.rsplit("/", 1)[0] + "/matcher_load"

with psycopg.connect(base, autocommit=True) as c:
    c.execute("DROP DATABASE IF EXISTS matcher_load WITH (FORCE)")
    c.execute("CREATE DATABASE matcher_load")
db.migrate(url)

with psycopg.connect(url, row_factory=dict_row) as c:
    register_vector(c)
    m = versions.load(c)
    interests.seed_taxonomy(c, m, Path("ml/data/taxonomy.json"))
    rows = c.execute("SELECT interest_id, embedding FROM interest_vectors WHERE model_version = %s "
                     "ORDER BY interest_id", (m.version,)).fetchall()
    ids = np.array([r["interest_id"] for r in rows])
    E = np.stack([r["embedding"].to_numpy() for r in rows])
    pop = 1 / np.arange(1, len(ids) + 1) ** 0.8
    pop = rng.permutation(pop / pop.sum())

    t0 = time.perf_counter()
    picks = [rng.choice(len(ids), size=rng.integers(3, 11), replace=False, p=pop) for _ in range(N)]
    df = np.bincount(np.concatenate(picks), minlength=len(ids))
    w = np.log((1 + N) / (1 + df)) + 1
    c.execute("DROP INDEX user_vectors_embedding_hnsw")  # bulk load first, index after: much faster
    c.execute("ALTER TABLE user_interests DISABLE TRIGGER user_interests_count")  # backfilled below instead
    with c.cursor() as cur:
        with cur.copy("COPY users (id, external_id, name, email) FROM STDIN") as cp:
            for u in range(1, N + 1):
                cp.write_row((u, f"u{u}", f"User {u}", f"u{u}@load.test"))
        with cur.copy("COPY user_interests (user_id, interest_id, raw_text) FROM STDIN") as cp:
            for u, p in enumerate(picks, 1):
                for j in p:
                    cp.write_row((u, int(ids[j]), "synthetic"))
        with cur.copy("COPY user_vectors (user_id, model_version, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "text", "vector"])
            for u, p in enumerate(picks, 1):
                v = (w[p, None] * E[p]).sum(0)
                cp.write_row((u, m.version, v / np.linalg.norm(v)))
    c.execute("SELECT setval('users_id_seq', %s)", (N,))
    c.execute("UPDATE interests i SET user_count = d.n FROM (SELECT interest_id, count(*) AS n FROM user_interests "
              "GROUP BY interest_id) d WHERE d.interest_id = i.id")
    c.execute("ALTER TABLE user_interests ENABLE TRIGGER user_interests_count")
    c.commit()
    print(f"loaded {N} users in {time.perf_counter() - t0:.0f}s, building HNSW index...")
    t0 = time.perf_counter()
    c.execute("SET maintenance_work_mem = '1GB'")
    c.execute("SET max_parallel_maintenance_workers = 0")  # docker's 64 MB /dev/shm breaks parallel builds
    c.execute("CREATE INDEX user_vectors_embedding_hnsw ON user_vectors USING hnsw (embedding vector_cosine_ops)")
    c.execute("ANALYZE")
    c.commit()
    print(f"index built in {time.perf_counter() - t0:.0f}s")

    sample = rng.choice(np.arange(1, N + 1), size=200, replace=False)
    for label, cand in (("full ANN + rare-interest retrieval", None), ("app sends 300 nearby ids", 300)):
        times, n_ann = [], []
        for u in sample:
            ext = None if cand is None else [f"u{x}" for x in rng.integers(1, N + 1, size=cand)]
            t0 = time.perf_counter()
            out = matching.recommend(c, m.version, int(u), ext, limit=20)
            times.append((time.perf_counter() - t0) * 1000)
            if cand is None:
                vec = c.execute("SELECT embedding FROM user_vectors WHERE user_id = %s", (int(u),)).fetchone()["embedding"]
                n_ann.append(len(matching._candidates(c, m.version, int(u), vec, None)))
            c.rollback()  # end the transaction so SET LOCAL does not leak between runs
            assert len(out) == 20
        p50, p95 = np.percentile(times, [50, 95])
        extra = f", candidates per query min {min(n_ann)}" if n_ann else ""
        print(f"{label}: p50 {p50:.0f} ms, p95 {p95:.0f} ms{extra}")
