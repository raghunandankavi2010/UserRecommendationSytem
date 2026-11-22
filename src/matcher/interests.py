"""Interests in the database: seeding, linking free text, attaching to users, user vectors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import psycopg

from matcher.linking import EMBED_MODEL, TOP_K, Encoder, decision, key, normalize

MODEL_VERSION = f"{EMBED_MODEL}@base"
MAX_TEXT = 80


class NotFound(LookupError):
    pass


def seed_taxonomy(conn: psycopg.Connection, encode: Encoder, path: Path) -> int:
    """Insert canonical interests that are not in the table yet. Returns how many were added."""
    labels = [label for group in json.loads(path.read_text()).values() for label in group]
    existing = {r["label_key"] for r in conn.execute("SELECT label_key FROM interests")}
    new = [label for label in labels if key(label) not in existing]
    if new:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO interests (label, label_key, status, embedding, model_version) "
                "VALUES (%s, %s, 'active', %s, %s) ON CONFLICT (label_key) DO NOTHING",
                [(label, key(label), v, MODEL_VERSION) for label, v in zip(new, encode(new))])
    return len(new)


def _exact(conn, text: str) -> dict | None:
    k = key(text)
    return conn.execute(
        "SELECT id, label FROM interests WHERE label_key = %s AND status <> 'merged' "
        "UNION ALL "
        "SELECT i.id, i.label FROM interest_aliases a JOIN interests i ON i.id = a.interest_id "
        "WHERE a.alias_key = %s AND i.status <> 'merged' LIMIT 1", (k, k)).fetchone()


def _nearest(conn, q: np.ndarray) -> list[dict]:
    return conn.execute(
        "SELECT id, label, 1 - (embedding <=> %s) AS score FROM interests WHERE status <> 'merged' "
        "ORDER BY embedding <=> %s LIMIT %s", (q, q, TOP_K)).fetchall()


def lookup(conn, encode: Encoder, text: str) -> dict:
    """Decision for free text: linked / did_you_mean / new, with the nearest interests."""
    if (row := _exact(conn, text)) is not None:
        return {"decision": "linked", "interest": row, "score": 1.0, "suggestions": [{**row, "score": 1.0}]}
    suggestions = _nearest(conn, encode([text])[0])
    best = suggestions[0]["score"] if suggestions else -1.0
    d = decision(best)
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


def add_interest(conn, encode: Encoder, user_id: int, text: str, choice: int | str | None = None) -> dict:
    """choice None: link if confident, else return suggestions without attaching.
    choice <interest id>: the user picked it (stored as alias + positive label).
    choice "new": the user rejected the suggestions (provisional interest + negative labels)."""
    text = clean(text)
    if choice is None:
        result = lookup(conn, encode, text)
        if result["decision"] == "linked":
            _attach(conn, user_id, result["interest"]["id"], text)
            refresh_user_vector(conn, user_id)
        return {**result, "attached": result["decision"] == "linked"}

    if choice == "new":
        if (row := _exact(conn, text)) is not None:  # already exists under this exact name
            interest = row
        else:
            q = encode([text])[0]
            shown = _nearest(conn, q)
            interest = conn.execute(
                "INSERT INTO interests (label, label_key, status, embedding, model_version) "
                "VALUES (%s, %s, 'provisional', %s, %s) "
                "ON CONFLICT (label_key) DO UPDATE SET label_key = EXCLUDED.label_key RETURNING id, label",
                (text, key(text), q, MODEL_VERSION)).fetchone()
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
            # shortcut: first confirmation wins an alias; use majority vote once labels pile up (Phase 3)
            conn.execute("INSERT INTO interest_aliases (alias_key, interest_id) VALUES (%s, %s) "
                         "ON CONFLICT DO NOTHING", (key(text), interest["id"]))
            conn.execute("INSERT INTO interest_labels (user_id, raw_text, interest_id, label, source) "
                         "VALUES (%s, %s, %s, 1, 'picked_suggestion')", (user_id, text, interest["id"]))
    _attach(conn, user_id, interest["id"], text)
    refresh_user_vector(conn, user_id)
    return {"decision": "linked", "interest": interest, "score": 1.0, "suggestions": [], "attached": True}


def remove_interest(conn, user_id: int, interest_id: int) -> None:
    if conn.execute("DELETE FROM user_interests WHERE user_id = %s AND interest_id = %s",
                    (user_id, interest_id)).rowcount == 0:
        raise NotFound("interest not on this user")
    refresh_user_vector(conn, user_id)


def idf(conn, interest_ids: list[int]) -> dict[int, float]:
    """Smoothed IDF over users, same formula as the prototype: log((1+N)/(1+df)) + 1."""
    n = conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"]
    rows = conn.execute("SELECT id, user_count FROM interests WHERE id = ANY(%s)", (list(interest_ids),)).fetchall()
    return idf_from_counts(n, {r["id"]: r["user_count"] for r in rows})


def idf_from_counts(n_users: int, df: dict[int, int]) -> dict[int, float]:
    return {i: float(np.log((1 + n_users) / (1 + d)) + 1) for i, d in df.items()}


def refresh_user_vector(conn, user_id: int) -> None:
    """User vector = IDF-weighted mean of their interest embeddings (unit length)."""
    # shortcut: IDF drifts as users join; a nightly full recompute (Phase 3) keeps old vectors current
    rows = conn.execute("SELECT i.id, i.embedding FROM user_interests ui JOIN interests i ON i.id = ui.interest_id "
                        "WHERE ui.user_id = %s", (user_id,)).fetchall()
    if not rows:
        conn.execute("DELETE FROM user_vectors WHERE user_id = %s", (user_id,))
        return
    w = idf(conn, [r["id"] for r in rows])
    v = sum(w[r["id"]] * r["embedding"].to_numpy() for r in rows)
    v = v / np.linalg.norm(v)
    conn.execute("INSERT INTO user_vectors (user_id, model_version, embedding) VALUES (%s, %s, %s) "
                 "ON CONFLICT (user_id, model_version) DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()",
                 (user_id, MODEL_VERSION, v))
