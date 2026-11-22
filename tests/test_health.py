import os

import pytest
from fastapi.testclient import TestClient

from matcher.main import app

client = TestClient(app)


def test_health_down_without_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x")
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "down"


@pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="needs the compose Postgres")
def test_health_ok_with_pgvector(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
