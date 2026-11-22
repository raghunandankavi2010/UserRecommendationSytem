"""Learning-loop mechanics on a throwaway `matcher_jobs_test` database (needs the compose Postgres)."""
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="needs the compose Postgres")
sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))
BASE = "google/embeddinggemma-300m@base"


@pytest.fixture
def c():
    import jobs
    from matcher import db, interests, versions
    base = os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(base, autocommit=True) as a:
        a.execute("DROP DATABASE IF EXISTS matcher_jobs_test WITH (FORCE)")
        a.execute("CREATE DATABASE matcher_jobs_test")
    url = base.rsplit("/", 1)[0] + "/matcher_jobs_test"
    db.migrate(url)
    conn = jobs.connect(url)
    m = versions.load(conn)
    interests.seed_taxonomy(conn, m, Path("ml/data/taxonomy.json"))
    for ext, texts in {"a": ["Trekking", "Filter coffee"], "b": ["hiking", "Coffee"], "c": ["Cricket"]}.items():
        uid = conn.execute("INSERT INTO users (external_id, name, email) VALUES (%s, %s, %s) RETURNING id",
                           (ext, ext, f"{ext}@example.com")).fetchone()["id"]
        for t in texts:
            interests.add_interest(conn, m, uid, t)
    conn.commit()
    conn.url = url
    yield conn
    conn.close()


def count(c, sql, *args):
    return c.execute(sql, args).fetchone()["n"]


def test_promote_rollback_and_live_reload(c, monkeypatch):
    import jobs
    from matcher import db, matching, versions
    c.execute("INSERT INTO model_versions (version, path, auto_min, ask_min, status) "
              "VALUES ('cand', 'google/embeddinggemma-300m', 0.83, 0.81, 'candidate')")
    c.commit()
    pool = db.make_pool(c.url)
    live = versions.Live(pool)
    assert live.get().version == BASE

    jobs.promote(c, "cand", wait=0)
    assert versions.active_version(c) == "cand"
    live_interests = count(c, "SELECT count(*) AS n FROM interests WHERE status <> 'merged'")
    assert count(c, "SELECT count(*) AS n FROM interest_vectors WHERE model_version = 'cand'") == live_interests
    assert count(c, "SELECT count(*) AS n FROM user_vectors WHERE model_version = 'cand'") == 3
    a = c.execute("SELECT id FROM users WHERE external_id = 'a'").fetchone()["id"]
    assert [r["user_id"] for r in matching.recommend(c, "cand", a)][0] == "b"
    c.rollback()

    monkeypatch.setattr(versions, "RELOAD_SECONDS", 0)
    assert live.get().version == "cand"          # the API picks up the promotion
    jobs.rollback(c, wait=0)
    assert versions.active_version(c) == BASE
    assert live.get().version == BASE            # ... and the rollback
    c.execute("UPDATE model_versions SET status = 'rejected' WHERE version = 'cand'")
    c.commit()
    with pytest.raises(SystemExit):
        jobs.promote(c, "cand", wait=0)
    pool.close()


def test_bulk_rebuild_matches_single_refresh(c):
    from matcher import interests
    b = c.execute("SELECT id FROM users WHERE external_id = 'b'").fetchone()["id"]
    interests.refresh_user_vector(c, BASE, b)
    before = c.execute("SELECT embedding FROM user_vectors WHERE user_id = %s", (b,)).fetchone()["embedding"].to_numpy()
    assert interests.rebuild_user_vectors(c, BASE) == 3
    after = c.execute("SELECT embedding FROM user_vectors WHERE user_id = %s", (b,)).fetchone()["embedding"].to_numpy()
    assert np.allclose(before, after, atol=1e-5)


def test_nightly_merges_duplicates_but_respects_rejections(c):
    from matcher import interests, versions
    m = versions.load(c)
    ids = {}
    for label in ("beekeeping", "bee keeping", "tai chi"):
        ids[label] = c.execute("INSERT INTO interests (label, label_key, status) VALUES (%s, %s, 'provisional') "
                               "RETURNING id", (label, label)).fetchone()["id"]
    interests.embed_missing(c, m)
    users = {r["external_id"]: r["id"] for r in c.execute("SELECT id, external_id FROM users")}
    for ext, label in (("a", "beekeeping"), ("b", "beekeeping"), ("c", "bee keeping"), ("c", "tai chi")):
        c.execute("INSERT INTO user_interests (user_id, interest_id, raw_text) VALUES (%s, %s, %s)",
                  (users[ext], ids[label], label))
    martial = c.execute("SELECT id FROM interests WHERE label = 'Martial arts'").fetchone()["id"]
    c.execute("INSERT INTO interest_labels (raw_text, interest_id, label, source) VALUES ('tai chi', %s, -1, 'test')",
              (martial,))
    m = versions.Model(m.version, m.path, m.encode, 0.80, 0.70)  # loose thresholds so tai chi ~ Martial arts qualifies

    out = interests.nightly_merge(c, m)
    assert {"from": "bee keeping", "into": "beekeeping"} in [{k: x[k] for k in ("from", "into")} for x in out["merged"]]
    assert all(x["from"] != "tai chi" for x in out["merged"])  # a user said tai chi is not Martial arts
    assert count(c, "SELECT user_count AS n FROM interests WHERE label = 'beekeeping'") == 3
    assert c.execute("SELECT interest_id FROM interest_aliases WHERE alias_key = 'bee keeping'").fetchone()["interest_id"] \
        == ids["beekeeping"]


def test_labels_split_and_report(c):
    import jobs
    users = {r["external_id"]: r["id"] for r in c.execute("SELECT id, external_id FROM users")}
    trek = c.execute("SELECT id FROM interests WHERE label = 'Trekking'").fetchone()["id"]
    for i in range(30):
        c.execute("INSERT INTO interest_labels (raw_text, interest_id, label, source) VALUES (%s, %s, 1, 'test')",
                  (f"trek variant {i}", trek))
    train, holdout = jobs.collect_labels(c)
    assert len(train) + len(holdout) == 30 and 0 < len(holdout) < 15
    assert jobs.collect_labels(c) == (train, holdout)  # deterministic split
    a, b = users["a"], users["b"]
    for actor, target, t in ((a, b, "shown"), (a, b, "wave"), (b, a, "accept"), (users["c"], a, "shown")):
        c.execute("INSERT INTO events (actor_id, target_id, type, model_version) VALUES (%s, %s, %s, %s)",
                  (actor, target, t, BASE if t == "shown" else None))
    (row,) = jobs.report(c)
    assert row["model_version"] == BASE and row["shown"] == 2
    assert float(row["wave_rate"]) == 0.5 and float(row["mutual_rate"]) == 0.5
