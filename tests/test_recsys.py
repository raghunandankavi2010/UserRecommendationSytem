import importlib.util

import pandas as pd
import pytest

from recsys import db
from recsys.models import DEFAULT_WEIGHTS, Recommender


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    c = db.connect(db.build_db(tmp_path_factory.mktemp("d") / "t.db"))
    yield c
    c.close()


def profiles_of(rows):
    return pd.DataFrame([{"user_id": i, "name": f"u{i}", "interests": sorted(t)} for i, t in rows])


def test_candidates_exclude_self_and_connections(conn):
    conns = db.load_connections(conn)
    u = int(conns.iloc[0]["user_a"])
    friends = set(conns.query("user_a == @u")["user_b"]) | set(conns.query("user_b == @u")["user_a"])
    cands = set(db.candidate_ids(conn, u))
    assert friends and u not in cands and not (cands & friends)
    assert len(cands) == conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] - 1 - len(friends)


def test_recommendations_are_valid_and_explained(conn):
    profiles = db.load_profiles(conn)
    sets = dict(zip(profiles["user_id"], map(set, profiles["interests"])))
    model = Recommender("tfidf", weights=DEFAULT_WEIGHTS).fit(
        profiles, db.load_connections(conn), db.load_signals(conn))
    for u in profiles["user_id"]:
        cands = db.candidate_ids(conn, u)
        recs = model.recommend(u, 5, cands)
        assert len(recs) == 5 and set(recs["user_id"]) <= set(cands)
        assert recs["score"].is_monotonic_decreasing
        assert recs["interest_score"].between(0, 1).all()
        for _, r in recs.iterrows():
            assert set(r["shared_interests"]) == sets[u] & sets[r["user_id"]]
    assert not {"phone", "email"} & set(recs.columns)       # PII never leaves the DB layer


def test_tfidf_rewards_rare_shared_interest_over_common_one():
    p = profiles_of([(1, {"chess", "music"}), (2, {"chess", "cooking"}), (3, {"music", "cooking"}),
                     (4, {"music", "yoga"}), (5, {"music", "running"})])
    recs = Recommender("tfidf").fit(p).recommend(1, 2)
    assert recs["user_id"].tolist()[0] == 2                 # shares rare "chess", not ubiquitous "music"
    j = Recommender("jaccard").fit(p).recommend(1, 4)
    assert j.set_index("user_id").loc[2, "score"] == j.set_index("user_id").loc[3, "score"]  # jaccard can't tell


def test_mutual_friend_breaks_tie():
    p = profiles_of([(1, {"chess"}), (2, {"chess"}), (3, {"chess"}), (4, {"go"})])
    conns = pd.DataFrame({"user_a": [1, 3], "user_b": [4, 4]})  # 1 and 3 share friend 4
    sig = pd.DataFrame({"user_id": [1, 2, 3, 4], "days_inactive": [0] * 4, "places": [[]] * 4})
    recs = Recommender("tfidf", weights=DEFAULT_WEIGHTS).fit(p, conns, sig).recommend(1, 2, [2, 3])
    assert recs["user_id"].tolist() == [3, 2] and recs.iloc[0]["mutual_friends"] == 1


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="no embeddings")
def test_embeddings_match_synonyms():
    p = profiles_of([(1, {"watching sci-fi movies"}), (2, {"star trek marathons"}), (3, {"home baking"})])
    assert Recommender("tfidf").fit(p).recommend(1, 2)["score"].max() == 0      # no exact overlap
    recs = Recommender("embedding").fit(p).recommend(1, 2)
    assert recs.iloc[0]["user_id"] == 2 and recs.iloc[0]["similar_interests"]
