"""End-to-end API tests on a throwaway `matcher_test` database (needs the compose Postgres)."""
import os

import psycopg
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="needs the compose Postgres")
KEY = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    base = os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(base, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS matcher_test WITH (FORCE)")
        c.execute("CREATE DATABASE matcher_test")
    os.environ["DATABASE_URL"] = base.rsplit("/", 1)[0] + "/matcher_test"
    os.environ["API_KEY"] = "test-key"
    from matcher.main import app
    with TestClient(app) as cl:
        yield cl


@pytest.fixture(autouse=True)
def clean(client):
    yield
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
        c.execute("TRUNCATE users, interest_aliases, interest_labels CASCADE")
        c.execute("DELETE FROM interests WHERE status = 'provisional'")
        c.execute("UPDATE interests SET user_count = 0")  # TRUNCATE skips row triggers


def user(client, ext, interests, email=None, phone=None):
    r = client.post("/users", headers=KEY, json={"external_id": ext, "name": ext.title(), "phone": phone,
                                                  "email": email or f"{ext}@example.com", "interests": interests})
    assert r.status_code == 201, r.text
    return {i["text"]: i for i in r.json()["interests"]}


def matches(client, ext, **body):
    r = client.post(f"/users/{ext}/matches", headers=KEY, json=body)
    assert r.status_code == 200, r.text
    return r.json()["matches"]


def event(client, actor, target, type_):
    assert client.post("/events", headers=KEY, json={"actor_id": actor, "target_id": target, "type": type_}).status_code == 201


def test_health_and_auth(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/users/x/matches", json={}).status_code == 401
    assert client.post("/users/x/matches", headers={"X-API-Key": "wrong"}, json={}).status_code == 401


def test_onboarding_links_asks_and_creates(client):
    res = user(client, "asha", ["hiking", "filter kaapi", "beekeeping", "movie marathons"])
    assert res["hiking"]["decision"] == "linked" and res["hiking"]["interest"]["label"] == "Trekking"
    assert res["hiking"]["attached"] and res["filter kaapi"]["interest"]["label"] == "Filter coffee"
    assert res["beekeeping"]["decision"] == "new" and not res["beekeeping"]["attached"]
    assert len(res["beekeeping"]["suggestions"]) == 3
    assert res["movie marathons"]["decision"] != "linked"  # the lexical trap must not be linked silently
    labels = [i["label"] for i in client.get("/users/asha", headers=KEY).json()["interests"]]
    assert sorted(labels) == ["Filter coffee", "Trekking"]


def test_choices_grow_the_vocabulary(client):
    user(client, "asha", [])
    r = client.post("/users/asha/interests", headers=KEY, json={"text": "beekeeping", "choice": "new"}).json()
    assert r["attached"] and r["interest"]["label"] == "beekeeping"
    # the next user typing it links to the same provisional interest
    user(client, "ravi", [])
    r2 = client.post("/users/ravi/interests", headers=KEY, json={"text": "Beekeeping"}).json()
    assert r2["decision"] == "linked" and r2["interest"]["id"] == r["interest"]["id"]

    sug = client.post("/users/asha/interests", headers=KEY, json={"text": "weekend treks in the sahyadris"}).json()
    trek = next(s for s in sug["suggestions"] if s["label"] == "Trekking")
    picked = client.post("/users/asha/interests", headers=KEY,
                         json={"text": "weekend treks in the sahyadris", "choice": trek["id"]}).json()
    assert picked["attached"] and picked["interest"]["label"] == "Trekking"
    alias = client.post("/users/ravi/interests", headers=KEY, json={"text": "Weekend treks in the Sahyadris"}).json()
    assert alias["decision"] == "linked" and alias["interest"]["label"] == "Trekking" and alias["score"] == 1.0

    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        got = dict(c.execute("SELECT label, count(*) FROM interest_labels GROUP BY label").fetchall())
    assert got[1] == 1 and got[-1] >= 1  # one pick, and the suggestions rejected for beekeeping

    assert client.post("/users/asha/interests", headers=KEY, json={"text": "x", "choice": 999999}).status_code == 404
    assert client.post("/users/asha/interests", headers=KEY, json={"text": "   "}).status_code == 422


def test_matching_ranks_and_explains_without_pii(client):
    user(client, "asha", ["Trekking", "Filter coffee", "Bird watching"], phone="+91 98450 00001")
    user(client, "dev", ["Trekking", "Filter coffee", "Bird watching"], phone="+91 98450 00002")
    user(client, "ravi", ["Mountaineering", "Coffee", "Wildlife photography"])
    user(client, "kiran", ["Cricket", "Video games", "Stock market"])
    got = matches(client, "asha")
    order = [m["user_id"] for m in got]
    assert order[:2] == ["dev", "ravi"] and order[-1] == "kiran"
    assert set(got[0]["common_interests"]) == {"Trekking", "Filter coffee", "Bird watching"}
    assert got[1]["common_interests"] == [] and got[1]["similar_interests"]  # e.g. Trekking ~ Mountaineering
    text = str(got)
    assert "example.com" not in text and "98450" not in text and "email" not in text and "phone" not in text
    assert [m["user_id"] for m in matches(client, "asha", candidate_ids=["kiran", "ravi", "nobody"])] == ["ravi", "kiran"]


def test_exclusions_and_mutual_connection(client):
    for ext in ("asha", "dev", "ravi", "kiran"):
        user(client, ext, ["Trekking", "Filter coffee"], phone=None)
    assert client.get("/users/asha/connections/dev", headers=KEY).status_code == 404
    event(client, "asha", "ravi", "dismiss")
    event(client, "kiran", "asha", "block")
    event(client, "asha", "dev", "wave")
    assert {m["user_id"] for m in matches(client, "asha")} == {"dev"}
    event(client, "dev", "asha", "accept")
    r = client.get("/users/asha/connections/dev", headers=KEY)
    assert r.status_code == 200 and r.json()["email"] == "dev@example.com" and "phone" not in r.json()
    assert r.json()["common_interests"] == ["Filter coffee", "Trekking"]
    assert matches(client, "asha") == []  # connected, dismissed, blocked: nobody left
    event(client, "dev", "asha", "block")
    assert client.get("/users/asha/connections/dev", headers=KEY).status_code == 404


def test_conflicts_removal_and_deletion(client):
    user(client, "asha", ["Trekking"])
    r = client.post("/users", headers=KEY, json={"external_id": "other", "name": "O", "email": "asha@example.com"})
    assert r.status_code == 409
    user(client, "dev", ["Trekking"])
    assert len(matches(client, "asha")) == 1
    tid = client.get("/users/asha", headers=KEY).json()["interests"][0]["id"]
    assert client.delete(f"/users/asha/interests/{tid}", headers=KEY).status_code == 204
    assert matches(client, "asha") == []  # no interests left, nothing to match on
    event(client, "dev", "asha", "wave")
    assert client.delete("/users/asha", headers=KEY).status_code == 204
    assert client.get("/users/asha", headers=KEY).status_code == 404
    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        assert c.execute("SELECT count(*) FROM events").fetchone()[0] == 0
        c.execute("INSERT INTO users (external_id, name, email) VALUES ('tmp', 'T', 't@example.com')")
        c.execute("INSERT INTO user_interests (user_id, interest_id, raw_text) "
                  "SELECT u.id, i.id, 'x' FROM users u, interests i WHERE u.external_id = 'tmp' AND i.label = 'Trekking'")
        c.execute("DELETE FROM users WHERE external_id = 'tmp'")  # cascade must decrement via the trigger
        assert c.execute("SELECT user_count FROM interests WHERE label = 'Trekking'").fetchone()[0] == 1  # dev
    assert client.patch("/users/dev", headers=KEY, json={"name": "Dev K"}).status_code == 204
    assert client.get("/users/dev", headers=KEY).json()["name"] == "Dev K"
