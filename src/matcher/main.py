import os

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Peoplo interest matcher")


@app.get("/health")
def health():
    # shortcut: new connection per call, replace with the pool added in Phase 2
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2) as conn:
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    except (KeyError, psycopg.Error) as e:
        return JSONResponse({"status": "down", "error": type(e).__name__}, status_code=503)
    if row is None:
        return JSONResponse({"status": "down", "error": "pgvector extension missing"}, status_code=503)
    return {"status": "ok", "pgvector": row[0]}
