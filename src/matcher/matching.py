"""Rank people for a user by common interests (production port of recsys.models.Recommender).

  1. candidates: the ids the app sends (people nearby), or ANN on user vectors ∪ users sharing my rarest interests
  2. drop: self, blocked either way, dismissed by me in the last 30 days, already connected
  3. score = ALPHA·idf_overlap + (1-ALPHA)·semantic + W_MUTUAL·min(mutual,3)/3 + W_RECENCY·exp(-days/30)
  4. why: common interests (rarest first) and close-but-different pairs
"""
from __future__ import annotations

import numpy as np

from matcher.interests import MODEL_VERSION, idf_from_counts
from matcher.linking import ASK_MIN

ALPHA = 0.5                       # prototype: hybrid keeps both rare exact matches and synonyms
W_MUTUAL, W_RECENCY = 0.3, 0.05   # prototype weights; retune on real accept data (Phase 3)
RETRIEVE = 200
RARE_INTERESTS = 3                # exact-overlap retrieval uses only my rarest interests (popular ones are in ANN)
DISMISS_DAYS = 30
MAX_SIMILAR_PAIRS = 3

# Interest embeddings never change for a given id within one model version, so cache them per process.
# shortcut: unbounded; fine for tens of thousands of interests (768 floats each), add an LRU beyond that
_vec_cache: dict[int, np.ndarray] = {}


def _interest_vectors(conn, ids: set[int]) -> dict[int, np.ndarray]:
    missing = [i for i in ids if i not in _vec_cache]
    if missing:
        for r in conn.execute("SELECT id, embedding FROM interests WHERE id = ANY(%s)", (missing,)):
            _vec_cache[r["id"]] = r["embedding"].to_numpy()
    return {i: _vec_cache[i] for i in ids}


def _candidates(conn, me: int, my_vec, candidate_ext_ids: list[str] | None) -> list[int]:
    if candidate_ext_ids is not None:
        return [r["id"] for r in conn.execute("SELECT id FROM users WHERE external_id = ANY(%s) AND id <> %s",
                                              (candidate_ext_ids, me))]
    # HNSW returns at most ef_search rows (default 40); raise it for this transaction so LIMIT is honoured
    conn.execute(f"SET LOCAL hnsw.ef_search = {max(RETRIEVE, 40)}")
    ann = conn.execute("SELECT user_id FROM user_vectors WHERE model_version = %s AND user_id <> %s "
                       "ORDER BY embedding <=> %s LIMIT %s", (MODEL_VERSION, me, my_vec, RETRIEVE))
    rare = conn.execute(
        "SELECT DISTINCT ui.user_id FROM user_interests ui WHERE ui.user_id <> %s AND ui.interest_id IN ("
        "  SELECT i.id FROM user_interests mine JOIN interests i ON i.id = mine.interest_id "
        "  WHERE mine.user_id = %s ORDER BY i.user_count LIMIT %s) LIMIT %s", (me, me, RARE_INTERESTS, RETRIEVE))
    return list(dict.fromkeys([r["user_id"] for r in ann] + [r["user_id"] for r in rare]))


def _excluded(conn, me: int) -> set[int]:
    rows = conn.execute(
        "SELECT target_id AS id FROM events WHERE actor_id = %(me)s AND (type = 'block' "
        "   OR (type = 'dismiss' AND created_at > now() - make_interval(days => %(days)s))) "
        "UNION SELECT actor_id FROM events WHERE target_id = %(me)s AND type = 'block' "
        "UNION SELECT other_id FROM connections WHERE user_id = %(me)s",
        {"me": me, "days": DISMISS_DAYS})
    return {r["id"] for r in rows}


def recommend(conn, me: int, candidate_ext_ids: list[str] | None = None, limit: int = 20) -> list[dict]:
    row = conn.execute("SELECT embedding FROM user_vectors WHERE user_id = %s AND model_version = %s",
                       (me, MODEL_VERSION)).fetchone()
    if row is None:
        return []  # no interests yet: nothing to match on
    my_vec = row["embedding"]
    excluded = _excluded(conn, me)
    cands = [c for c in _candidates(conn, me, my_vec, candidate_ext_ids) if c not in excluded]
    if not cands:
        return []

    # semantic similarity computed in Postgres: no candidate vectors cross the wire
    people = {r["id"]: r for r in conn.execute(
        "SELECT u.id, u.external_id, u.name, u.photo_url, 1 - (uv.embedding <=> %s) AS semantic, "
        "       extract(epoch FROM now() - u.last_active_at) / 86400 AS days_inactive "
        "FROM users u JOIN user_vectors uv ON uv.user_id = u.id AND uv.model_version = %s "
        "WHERE u.id = ANY(%s)", (my_vec, MODEL_VERSION, cands))}
    cands = [c for c in cands if c in people]  # candidates without interests cannot be scored
    if not cands:
        return []
    ints: dict[int, dict[int, str]] = {u: {} for u in [me] + cands}
    df: dict[int, int] = {}
    for r in conn.execute("SELECT ui.user_id, i.id, i.label, i.user_count FROM user_interests ui "
                          "JOIN interests i ON i.id = ui.interest_id WHERE ui.user_id = ANY(%s)", ([me] + cands,)):
        ints[r["user_id"]][r["id"]] = r["label"]
        df[r["id"]] = r["user_count"]
    w = idf_from_counts(conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"], df)
    mutual = {r["user_id"]: r["n"] for r in conn.execute(
        "SELECT c2.user_id, count(*) AS n FROM connections c1 JOIN connections c2 ON c2.other_id = c1.other_id "
        "WHERE c1.user_id = %s AND c2.user_id = ANY(%s) GROUP BY c2.user_id", (me, cands))}

    mine = ints[me]
    my_norm = np.sqrt(sum(w[i] ** 2 for i in mine))
    results = []
    for c in cands:
        theirs = ints[c]
        shared = mine.keys() & theirs.keys()
        overlap = sum(w[i] ** 2 for i in shared) / (my_norm * np.sqrt(sum(w[i] ** 2 for i in theirs)))
        m = mutual.get(c, 0)
        p = people[c]
        score = (ALPHA * overlap + (1 - ALPHA) * float(p["semantic"])
                 + W_MUTUAL * min(m, 3) / 3 + W_RECENCY * float(np.exp(-float(p["days_inactive"]) / 30)))
        results.append((score, c, shared, m))
    results.sort(key=lambda t: -t[0])
    top = results[:limit]

    vec = _interest_vectors(conn, set(mine) | {i for _, c, _, _ in top for i in ints[c]})
    out = []
    for score, c, shared, m in top:
        theirs = ints[c]
        pairs = sorted(((float(vec[a] @ vec[b]), mine[a], theirs[b])
                        for a in mine.keys() - shared for b in theirs.keys() - shared), reverse=True)
        p = people[c]
        out.append({
            "user_id": p["external_id"], "name": p["name"], "photo_url": p["photo_url"],
            "score": round(score, 4),
            "common_interests": [mine[i] for i in sorted(shared, key=lambda i: -w[i])],
            "similar_interests": [{"you": a, "them": b, "similarity": round(s, 2)}
                                  for s, a, b in pairs if s >= ASK_MIN][:MAX_SIMILAR_PAIRS],
            "mutual_connections": m, "model_version": MODEL_VERSION,
        })
    return out
