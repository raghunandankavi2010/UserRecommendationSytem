"""Postgres access: connection pool and numbered SQL migrations (src/matcher/migrations/NNN_*.sql)."""
from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

MIGRATIONS = Path(__file__).parent / "migrations"


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.row_factory = dict_row


def migrate(url: str) -> list[str]:
    """Apply migrations not yet recorded in schema_migrations, each in its own transaction."""
    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        raise RuntimeError(f"no migrations found in {MIGRATIONS} (package data missing?)")
    applied = []
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for f in files:
            if f.name in done:
                continue
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
            applied.append(f.name)
    return applied


def make_pool(url: str) -> ConnectionPool:
    # migrate() must run first: register_vector needs the vector type to exist
    return ConnectionPool(url, min_size=1, max_size=10, configure=_configure, open=True)
