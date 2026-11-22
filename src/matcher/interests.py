"""Interests in the database: seeding, linking free text, attaching to users, user vectors, merging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from matcher.linking import TOP_K, decision, key, normalize
from matcher.versions import Model

MAX_TEXT = 80
PROMOTE_MIN_USERS = 5   # a provisional interest this many users chose becomes a regular one


class NotFound(LookupError):
    pass


def embed_missing(conn, m: Model) -> int:
    """Embed every live interest that has no vector for this model version yet."""
    rows = conn.execute("SELECT i.id, i.label FROM interests i WHERE i.status <> 'merged' AND NOT EXISTS "
                        "(SELECT 1 FROM interest_vectors v WHERE v.interest_id = i.id AND v.model_version = %s)",
                        (m.version,)).fetchall()
    if rows:
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO interest_vectors (interest_id, model_version, embedding) VALUES (%s, %s, %s) "
                            "ON CONFLICT DO NOTHING",
                            [(r["id"], m.version, v) for r, v in zip(rows, m.encode([r["label"] for r in rows]))])
    return len(rows)


def seed_taxonomy(conn, m: Model, path: Path) -> int:
    """Insert canonical interests that are not in the table yet, then embed them. Returns how many were added."""
    labels = [label for group in json.loads(path.read_text()).values() for label in group]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO interests (label, label_key, status) VALUES (%s, %s, 'active') "
                        "ON CONFLICT (label_key) DO NOTHING", [(label, key(label)) for label in labels])
    return embed_missing(conn, m)


def _exact(conn, text: str) -> dict | None:
    k = key(text)
    return conn.execute(
        "SELECT id, label FROM interests WHERE label_key = %s AND status <> 'merged' "
        "UNION ALL "
        "SELECT i.id, i.label FROM interest_aliases a JOIN interests i ON i.id = a.interest_id "
        "WHERE a.alias_key = %s AND i.status <> 'merged' LIMIT 1", (k, k)).fetchone()


def _nearest(conn, version: str, q, k: int = TOP_K, exclude: int = -1) -> list[dict]:
    return conn.execute(
        "SELECT i.id, i.label, i.status, 1 - (v.embedding <=> %s) AS score FROM interest_vectors v "
        "JOIN interests i ON i.id = v.interest_id WHERE v.model_version = %s AND i.status <> 'merged' AND i.id <> %s "
        "ORDER BY v.embedding <=> %s LIMIT %s", (q, version, exclude, q, k)).fetchall()


def lookup(conn, m: Model, text: str) -> dict:
    """Decision for free text: linked / did_you_mean / new, with the nearest interests."""
    if (row := _exact(conn, text)) is not None:
        return {"decision": "linked", "interest": row, "score": 1.0, "suggestions": [{**row, "score": 1.0}]}
    suggestions = _nearest(conn, m.version, m.encode([text])[0])
    best = suggestions[0]["score"] if suggestions else -1.0
    d = decision(best, m.auto_min, m.ask_min)
    return {"decision": d, "interest": suggestions[0] if d == "linked" else None,
            "score": best, "suggestions": suggestions}


def clean(text: str) -> str:
    text = normalize(text)
    if not text or len(text) > MAX_TEXT:
        raise ValueError(f"interest must be 1-{MAX_TEXT} characters")
    return text


def _attach(conn, user_id: int, interest_id: int, raw_text: str) -> None:
    conn.execute("INSERT INTO user_interests (user_id, interest_id, raw_text) VALUES (%s, %s, %s) "
                 "ON CONFLICT DO NOTHING", (user_id, interest_id, raw_text))


def add_interest(conn, m: Model, user_id: int, text: str, choice: int | str | None = None) -> dict:
    """choice None: link if confident, else return suggestions without attaching.
    choice <interest id>: the user picked it (stored as alias + positive label).
    choice "new": the user rejected the suggestions (provisional interest + negative labels)."""
    text = clean(text)
    if choice is None:
        result = lookup(conn, m, text)
        if result["decision"] == "linked":
            _attach(conn, user_id, result["interest"]["id"], text)
            refresh_user_vector(conn, m.version, user_id)
        return {**result, "attached": result["decision"] == "linked"}

    if choice == "new":
        if (row := _exact(conn, text)) is not None:  # already exists under this exact name
            interest = row
        else:
            q = m.encode([text])[0]
            shown = _nearest(conn, m.version, q)
            interest = conn.execute(
                "INSERT INTO interests (label, label_key, status) VALUES (%s, %s, 'provisional') "
                "ON CONFLICT (label_key) DO UPDATE SET label_key = EXCLUDED.label_key RETURNING id, label",
                (text, key(text))).fetchone()
            conn.execute("INSERT INTO interest_vectors (interest_id, model_version, embedding) VALUES (%s, %s, %s) "
                         "ON CONFLICT DO NOTHING", (interest["id"], m.version, q))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO interest_labels (user_id, raw_text, interest_id, label, source) "
                    "VALUES (%s, %s, %s, -1, 'rejected_suggestion')",
                    [(user_id, text, s["id"]) for s in shown if s["id"] != interest["id"]])
    else:
        interest = conn.execute("SELECT id, label FROM interests WHERE id = %s AND status <> 'merged'",
                                (choice,)).fetchone()
        if interest is None:
            raise NotFound("interest not found")
        if key(text) != key(interest["label"]):
            # shortcut: first confirmation wins an alias; switch to majority vote over interest_labels if they conflict
            conn.execute("INSERT INTO interest_aliases (alias_key, interest_id) VALUES (%s, %s) "
                         "ON CONFLICT DO NOTHING", (key(text), interest["id"]))
            conn.execute("INSERT INTO interest_labels (user_id, raw_text, interest_id, label, source) "
                         "VALUES (%s, %s, %s, 1, 'picked_suggestion')", (user_id, text, interest["id"]))
    _attach(conn, user_id, interest["id"], text)
    refresh_user_vector(conn, m.version, user_id)
    return {"decision": "linked", "interest": interest, "score": 1.0, "suggestions": [], "attached": True}


def remove_interest(conn, version: str, user_id: int, interest_id: int) -> None:
    if conn.execute("DELETE FROM user_interests WHERE user_id = %s AND interest_id = %s",
                    (user_id, interest_id)).rowcount == 0:
        raise NotFound("interest not on this user")
    refresh_user_vector(conn, version, user_id)


def idf_from_counts(n_users: int, df: dict[int, int]) -> dict[int, float]:
    """Smoothed IDF over users, same formula as the prototype: log((1+N)/(1+df)) + 1."""
    return {i: float(np.log((1 + n_users) / (1 + d)) + 1) for i, d in df.items()}


def refresh_user_vector(conn, version: str, user_id: int) -> None:
    """User vector = IDF-weighted mean of their interest embeddings (unit length)."""
    # IDF drifts as users join; the nightly rebuild_user_vectors() brings every vector up to date
    rows = conn.execute("SELECT i.id, i.user_count, v.embedding FROM user_interests ui "
                        "JOIN interests i ON i.id = ui.interest_id "
                        "JOIN interest_vectors v ON v.interest_id = i.id AND v.model_version = %s "
                        "WHERE ui.user_id = %s", (version, user_id)).fetchall()
    if not rows:
        conn.execute("DELETE FROM user_vectors WHERE user_id = %s AND model_version = %s", (user_id, version))
        return
    n = conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"]
    w = idf_from_counts(n, {r["id"]: r["user_count"] for r in rows})
    v = sum(w[r["id"]] * r["embedding"].to_numpy() for r in rows)
    conn.execute("INSERT INTO user_vectors (user_id, model_version, embedding) VALUES (%s, %s, %s) "
                 "ON CONFLICT (user_id, model_version) DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()",
                 (user_id, version, v / np.linalg.norm(v)))


def rebuild_user_vectors(conn, version: str) -> int:
    """Recompute every user's vector for a version in bulk (fresh IDF). Returns users written."""
    vecs = {r["interest_id"]: r["embedding"].to_numpy() for r in conn.execute(
        "SELECT interest_id, embedding FROM interest_vectors WHERE model_version = %s", (version,))}
    df = {r["id"]: r["user_count"] for r in conn.execute("SELECT id, user_count FROM interests")}
    w = idf_from_counts(conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"], df)
    acc: dict[int, np.ndarray] = {}
    with conn.cursor(name="ui_scan") as cur:  # server-side cursor: does not load every row at once
        cur.execute("SELECT user_id, interest_id FROM user_interests")
        for r in cur:
            if (e := vecs.get(r["interest_id"])) is not None:
                acc[r["user_id"]] = acc.get(r["user_id"], 0) + w[r["interest_id"]] * e
    conn.execute("CREATE TEMP TABLE new_vectors (user_id bigint, embedding vector(768)) ON COMMIT DROP")
    with conn.cursor() as cur:
        with cur.copy("COPY new_vectors (user_id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "vector"])
            for u, v in acc.items():
                cp.write_row((u, v / np.linalg.norm(v)))
    conn.execute("DELETE FROM user_vectors uv WHERE model_version = %s AND NOT EXISTS "
                 "(SELECT 1 FROM new_vectors n WHERE n.user_id = uv.user_id)", (version,))
    conn.execute("INSERT INTO user_vectors (user_id, model_version, embedding) "
                 "SELECT user_id, %s, embedding FROM new_vectors "
                 "ON CONFLICT (user_id, model_version) DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()",
                 (version,))
    return len(acc)


def merge(conn, src: int, dst: int) -> None:
    """Fold interest src into dst: users, aliases and labels move; src stays as a 'merged' tombstone."""
    conn.execute("DELETE FROM user_interests WHERE interest_id = %s AND user_id IN "
                 "(SELECT user_id FROM user_interests WHERE interest_id = %s)", (src, dst))
    conn.execute("UPDATE user_interests SET interest_id = %s WHERE interest_id = %s", (dst, src))
    conn.execute("UPDATE interest_aliases SET interest_id = %s WHERE interest_id = %s", (dst, src))
    conn.execute("INSERT INTO interest_aliases (alias_key, interest_id) SELECT label_key, %s FROM interests "
                 "WHERE id = %s ON CONFLICT DO NOTHING", (dst, src))
    conn.execute("UPDATE interest_labels SET interest_id = %s WHERE interest_id = %s", (dst, src))
    conn.execute("UPDATE interests SET status = 'merged', merged_into = %s WHERE id = %s", (dst, src))
    # the count trigger only sees INSERT/DELETE, so recount the two rows the UPDATE moved users between
    conn.execute("UPDATE interests i SET user_count = (SELECT count(*) FROM user_interests ui WHERE ui.interest_id = i.id) "
                 "WHERE i.id IN (%s, %s)", (src, dst))


def nightly_merge(conn, m: Model) -> dict:
    """Merge provisional interests into a near-identical one (never into one a user rejected for that text),
    and promote provisional interests enough users chose. Returns what changed, for the job log."""
    merged, review = [], []
    for p in conn.execute("SELECT i.id, i.label, i.label_key, v.embedding FROM interests i "
                          "JOIN interest_vectors v ON v.interest_id = i.id AND v.model_version = %s "
                          "WHERE i.status = 'provisional' ORDER BY i.user_count, i.id", (m.version,)).fetchall():
        if conn.execute("SELECT status FROM interests WHERE id = %s", (p["id"],)).fetchone()["status"] != "provisional":
            continue  # already merged earlier in this run
        best = _nearest(conn, m.version, p["embedding"], k=1, exclude=p["id"])
        if not best:
            continue
        b = best[0]
        rejected = conn.execute("SELECT 1 FROM interest_labels WHERE label = -1 AND interest_id = %s "
                                "AND lower(raw_text) = %s", (b["id"], p["label_key"])).fetchone()
        if b["score"] >= m.auto_min and not rejected:
            merge(conn, p["id"], b["id"])
            merged.append({"from": p["label"], "into": b["label"], "score": round(float(b["score"]), 3)})
        elif b["score"] >= m.ask_min:
            review.append({"interest": p["label"], "nearest": b["label"], "score": round(float(b["score"]), 3),
                           "user_rejected": bool(rejected)})
    promoted = [r["label"] for r in conn.execute(
        "UPDATE interests SET status = 'active' WHERE status = 'provisional' AND user_count >= %s RETURNING label",
        (PROMOTE_MIN_USERS,))]
    return {"merged": merged, "promoted": promoted, "needs_review": review}
