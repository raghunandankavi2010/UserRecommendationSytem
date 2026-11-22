from fastapi.testclient import TestClient

from matcher.main import app


def test_health_down_without_db():
    # no lifespan run, so no pool: the probe must answer 503, not crash
    r = TestClient(app).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "down"
