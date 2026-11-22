"""HTTP API for the Peoplo app. Users are addressed by the app's own id (external_id).

Env: DATABASE_URL, API_KEY (sent as X-API-Key), TAXONOMY_PATH (default ml/data/taxonomy.json).
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from matcher import db, interests, matching
from matcher.linking import st_encoder

EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE = r"^\+?[0-9 -]{6,20}$"
URL = r"^https?://"


class NewUser(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=254, pattern=EMAIL)
    phone: str | None = Field(None, max_length=20, pattern=PHONE)
    photo_url: str | None = Field(None, max_length=2048, pattern=URL)
    interests: list[str] = Field(default_factory=list, max_length=50)


class UserPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    email: str | None = Field(None, max_length=254, pattern=EMAIL)
    phone: str | None = Field(None, max_length=20, pattern=PHONE)
    photo_url: str | None = Field(None, max_length=2048, pattern=URL)


class InterestIn(BaseModel):
    text: str = Field(min_length=1, max_length=interests.MAX_TEXT)
    choice: int | Literal["new"] | None = None  # answer to a did_you_mean / new suggestion


class MatchQuery(BaseModel):
    candidate_ids: list[str] | None = Field(None, max_length=1000)  # people the app found nearby
    limit: int = Field(20, ge=1, le=100)


class EventIn(BaseModel):
    actor_id: str = Field(min_length=1, max_length=64)
    target_id: str = Field(min_length=1, max_length=64)
    type: Literal["shown", "wave", "accept", "dismiss", "block"]
    model_version: str | None = Field(None, max_length=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.environ["DATABASE_URL"]
    if not os.environ.get("API_KEY"):
        raise RuntimeError("API_KEY must be set")
    db.migrate(url)
    app.state.pool = db.make_pool(url)
    app.state.encode = st_encoder()
    with app.state.pool.connection() as c:
        interests.seed_taxonomy(c, app.state.encode, Path(os.environ.get("TAXONOMY_PATH", "ml/data/taxonomy.json")))
    yield
    app.state.pool.close()


app = FastAPI(title="Peoplo interest matcher", lifespan=lifespan)


def require_key(x_api_key: str = Header("")) -> None:
    if not secrets.compare_digest(x_api_key.encode(), os.environ["API_KEY"].encode()):
        raise HTTPException(401, "bad API key")


def conn(request: Request):
    with request.app.state.pool.connection() as c:  # commits on success, rolls back on error
        yield c


def user_id(c, external_id: str) -> int:
    row = c.execute("SELECT id FROM users WHERE external_id = %s", (external_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "user not found")
    return row["id"]


def _interest_out(r: dict) -> dict:
    return {"decision": r["decision"], "attached": r["attached"], "score": round(float(r["score"]), 4),
            "interest": r["interest"] and {"id": r["interest"]["id"], "label": r["interest"]["label"]},
            "suggestions": [{"id": s["id"], "label": s["label"], "score": round(float(s["score"]), 4)}
                            for s in r["suggestions"]]}


def _add(request: Request, c, uid: int, text: str, choice=None) -> dict:
    try:
        return {"text": text, **_interest_out(interests.add_interest(c, request.app.state.encode, uid, text, choice))}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except interests.NotFound as e:
        raise HTTPException(404, str(e))


@app.exception_handler(psycopg.errors.UniqueViolation)
def _conflict(request, exc):
    return JSONResponse({"detail": "external_id, email or phone already in use"}, status_code=409)


auth = [Depends(require_key)]


@app.post("/users", status_code=201, dependencies=auth)
def create_user(body: NewUser, request: Request, c=Depends(conn)):
    uid = c.execute("INSERT INTO users (external_id, name, email, phone, photo_url) VALUES (%s, %s, %s, %s, %s) "
                    "RETURNING id", (body.external_id, body.name, body.email, body.phone, body.photo_url)
                    ).fetchone()["id"]
    return {"user_id": body.external_id, "interests": [_add(request, c, uid, t) for t in body.interests]}


@app.get("/users/{external_id}", dependencies=auth)
def get_user(external_id: str, c=Depends(conn)):
    u = c.execute("SELECT id, external_id, name, email, photo_url FROM users WHERE external_id = %s",
                  (external_id,)).fetchone()
    if u is None:
        raise HTTPException(404, "user not found")
    ints = c.execute("SELECT i.id, i.label, ui.raw_text FROM user_interests ui JOIN interests i ON i.id = ui.interest_id "
                     "WHERE ui.user_id = %s ORDER BY ui.created_at", (u["id"],)).fetchall()
    return {"user_id": u["external_id"], "name": u["name"], "email": u["email"], "photo_url": u["photo_url"],
            "interests": ints}


@app.patch("/users/{external_id}", status_code=204, dependencies=auth)
def update_user(external_id: str, body: UserPatch, c=Depends(conn)):
    fields = body.model_dump(exclude_unset=True)
    uid = user_id(c, external_id)
    for col in ("name", "email", "phone", "photo_url"):  # fixed column names: no user input reaches SQL text
        if col in fields:
            c.execute(f"UPDATE users SET {col} = %s WHERE id = %s", (fields[col], uid))
    return Response(status_code=204)


@app.delete("/users/{external_id}", status_code=204, dependencies=auth)
def delete_user(external_id: str, c=Depends(conn)):
    # cascades to interest links, vector, events and labels (DPDP: right to erasure)
    c.execute("DELETE FROM users WHERE id = %s", (user_id(c, external_id),))
    return Response(status_code=204)


@app.post("/users/{external_id}/interests", dependencies=auth)
def add_interest(external_id: str, body: InterestIn, request: Request, c=Depends(conn)):
    return _add(request, c, user_id(c, external_id), body.text, body.choice)


@app.delete("/users/{external_id}/interests/{interest_id}", status_code=204, dependencies=auth)
def remove_interest(external_id: str, interest_id: int, c=Depends(conn)):
    try:
        interests.remove_interest(c, user_id(c, external_id), interest_id)
    except interests.NotFound as e:
        raise HTTPException(404, str(e))
    return Response(status_code=204)


@app.get("/interests/suggest", dependencies=auth)
def suggest(q: str = Query(min_length=1, max_length=interests.MAX_TEXT), c=Depends(conn)):
    # shortcut: substring scan, fine for thousands of interests; add a pg_trgm index beyond ~100k
    like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    return c.execute("SELECT i.id, i.label FROM interests i LEFT JOIN user_interests ui ON ui.interest_id = i.id "
                     "WHERE i.status <> 'merged' AND i.label ILIKE %s GROUP BY i.id "
                     "ORDER BY count(ui.user_id) DESC, i.label LIMIT 10", (like,)).fetchall()


@app.post("/users/{external_id}/matches", dependencies=auth)
def matches(external_id: str, body: MatchQuery, c=Depends(conn)):
    return {"matches": matching.recommend(c, user_id(c, external_id), body.candidate_ids, body.limit)}


@app.post("/events", status_code=201, dependencies=auth)
def event(body: EventIn, c=Depends(conn)):
    if body.actor_id == body.target_id:
        raise HTTPException(422, "actor and target must differ")
    actor, target = user_id(c, body.actor_id), user_id(c, body.target_id)
    c.execute("INSERT INTO events (actor_id, target_id, type, model_version) VALUES (%s, %s, %s, %s)",
              (actor, target, body.type, body.model_version))
    c.execute("UPDATE users SET last_active_at = now() WHERE id = %s", (actor,))
    return {"ok": True}


@app.get("/users/{external_id}/connections/{other_id}", dependencies=auth)
def connection(external_id: str, other_id: str, c=Depends(conn)):
    me, other = user_id(c, external_id), user_id(c, other_id)
    if c.execute("SELECT 1 FROM connections WHERE user_id = %s AND other_id = %s", (me, other)).fetchone() is None:
        raise HTTPException(404, "no mutual connection")
    u = c.execute("SELECT external_id, name, email, photo_url FROM users WHERE id = %s", (other,)).fetchone()
    common = c.execute("SELECT i.label FROM user_interests a JOIN user_interests b ON b.interest_id = a.interest_id "
                       "JOIN interests i ON i.id = a.interest_id WHERE a.user_id = %s AND b.user_id = %s "
                       "ORDER BY i.label", (me, other)).fetchall()
    return {"user_id": u["external_id"], "name": u["name"], "email": u["email"], "photo_url": u["photo_url"],
            "common_interests": [r["label"] for r in common]}


@app.get("/health")
def health(request: Request):
    try:
        with request.app.state.pool.connection(timeout=2) as c:
            row = c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    except Exception as e:  # pool missing/timeout or DB error: report it, never crash the probe
        return JSONResponse({"status": "down", "error": type(e).__name__}, status_code=503)
    if row is None:
        return JSONResponse({"status": "down", "error": "pgvector extension missing"}, status_code=503)
    return {"status": "ok", "pgvector": row["extversion"]}
