"""SQLite schema, synthetic seed data and the SQL the recommender reads.

Production (PostgreSQL + pgvector) swap, in one place:
  * driver: `sqlite3.connect(path)` -> `psycopg.connect(dsn)`; placeholders `?` -> `%s`.
  * aggregation: `GROUP_CONCAT(i.label, '|')` -> `string_agg(i.label, '|' ORDER BY i.label)`.
  * ids: `INTEGER PRIMARY KEY` -> `BIGINT GENERATED ALWAYS AS IDENTITY`.
  * vectors: `CREATE EXTENSION vector; ALTER TABLE users ADD COLUMN embedding vector(384);`
    `CREATE INDEX ON users USING hnsw (embedding vector_cosine_ops);`
    then candidate retrieval becomes `... ORDER BY u.embedding <=> %(q)s LIMIT 200`
    (with the same NOT EXISTS exclusion below) and Python only re-ranks those 200.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "recsys.db"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE users (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    phone      TEXT NOT NULL UNIQUE,   -- PII: stored, never used as a feature or returned
    email      TEXT NOT NULL UNIQUE,   -- PII: same
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at TEXT NOT NULL       -- recency signal
);
CREATE TABLE places (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL                  -- cafe, gym, venue, ...
);
CREATE TABLE checkins (                  -- "both go to the same coffee shop" signal
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    place_id   INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    visited_at TEXT NOT NULL,
    PRIMARY KEY (user_id, place_id, visited_at)
);
CREATE TABLE interests (
    id    INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE          -- free text, normalised to lower-case/trimmed
);
CREATE TABLE user_interests (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    interest_id INTEGER NOT NULL REFERENCES interests(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, interest_id)
);
CREATE INDEX idx_user_interests_interest ON user_interests(interest_id);
CREATE TABLE connections (               -- undirected, stored once with user_a < user_b
    user_a INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_b INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_a, user_b),
    CHECK (user_a < user_b)
);
CREATE INDEX idx_connections_b ON connections(user_b);
"""

# Hidden "personas" drive the synthetic data. They are NOT stored in the DB:
# models only ever see interests, and connections act as ground truth.
# Each persona deliberately mixes near-synonyms ("watching sci-fi movies" vs
# "star trek marathons") so exact-match and semantic methods behave differently.
PERSONAS = {
    "scifi":   ["watching sci-fi movies", "star trek marathons", "reading science fiction novels",
                "astronomy and stargazing", "marvel movie nights", "space documentaries"],
    "reader":  ["drinking filter coffee", "cafe hopping", "reading the newspaper",
                "reading literary fiction", "book club meetups", "journaling"],
    "fitness": ["running marathons", "weight training at the gym", "morning yoga",
                "cycling on weekends", "swimming laps", "crossfit workouts"],
    "foodie":  ["trying street food", "home baking", "cooking south indian food",
                "exploring new restaurants", "making sourdough bread", "food blogging"],
    "travel":  ["backpacking trips", "solo travel", "road trips",
                "travel photography", "camping under the stars", "trekking in the himalayas"],
    "music":   ["playing guitar", "carnatic music concerts", "singing karaoke",
                "going to live gigs", "learning the piano", "collecting vinyl records"],
    "gamer":   ["playing video games", "board game nights", "playing chess online",
                "esports tournaments", "playing dungeons and dragons", "retro gaming"],
    "tech":    ["coding side projects", "building robots with arduino", "reading tech blogs",
                "learning machine learning", "3d printing", "attending hackathons"],
}
# Places each persona tends to frequent (name, kind); fictional venues.
PLACES = {
    "scifi":   [("Orbit Comics & Games, Jayanagar", "store"), ("Planetarium Night Shows", "venue")],
    "reader":  [("Brew Lab Cafe, Indiranagar", "cafe"), ("Blossom Book House", "store"), ("Filter Kaapi Corner, Basavanagudi", "cafe")],
    "fitness": [("Cubbon Park Running Track", "park"), ("IronCore Gym, HSR", "gym")],
    "foodie":  [("VV Puram Food Street", "street"), ("Sourdough Society Bakery", "cafe")],
    "travel":  [("Trek Hub Outfitters", "store"), ("Nomad Hostel Common Room", "hostel")],
    "music":   [("The Gig Room, Church Street", "venue"), ("Gayana Samaja Hall", "venue")],
    "gamer":   [("Dice & Dragons Board Game Cafe", "cafe"), ("Level Up Arcade", "venue")],
    "tech":    [("Hacker House Koramangala", "cowork"), ("Maker Space BLR", "cowork")],
}
POPULAR_PLACE = ("Phoenix City Mall", "mall")  # everyone goes here: should carry little weight
TODAY = "2026-11-20"

# Interests almost everyone has: TF-IDF should learn they say little.
COMMON = ["listening to music", "watching movies", "hanging out with friends", "scrolling social media"]

FIRST = ["Aarav", "Diya", "Kabir", "Ananya", "Rohan", "Meera", "Arjun", "Isha", "Vikram", "Sneha",
         "Rahul", "Priya", "Karthik", "Lakshmi", "Nikhil", "Pooja", "Siddharth", "Kavya", "Aditya", "Nisha",
         "Varun", "Divya", "Manoj", "Shreya", "Harsha", "Anjali", "Suresh", "Deepa", "Ravi", "Tara",
         "Ajay", "Sana"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Patel", "Rao", "Gupta", "Menon", "Kulkarni", "Shetty",
        "Das", "Joshi", "Bhat", "Pillai", "Verma", "Hegde"]


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build_db(path: str | Path = DEFAULT_DB, users_per_persona: int = 25, seed: int = 42) -> Path:
    """(Re)create the DB with reproducible synthetic users, interests and connections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    rng = random.Random(seed)
    conn = connect(path)
    with conn:
        conn.executescript(SCHEMA)
        labels = sorted({l.strip().lower() for ls in PERSONAS.values() for l in ls} | set(COMMON))
        conn.executemany("INSERT INTO interests(label) VALUES (?)", [(l,) for l in labels])
        iid = dict(conn.execute("SELECT label, id FROM interests"))
        conn.executemany("INSERT INTO places(name, kind) VALUES (?, ?)",
                         [p for ps in PLACES.values() for p in ps] + [POPULAR_PLACE])
        pid = dict(conn.execute("SELECT name, id FROM places"))
        all_places = list(pid)

        persona_of: dict[int, str] = {}
        uid = 0
        for persona, own in PERSONAS.items():
            for _ in range(users_per_persona):
                uid += 1
                first, last = rng.choice(FIRST), rng.choice(LAST)
                conn.execute(
                    "INSERT INTO users(id, name, phone, email, last_active_at) VALUES (?, ?, ?, ?, date(?, ?))",
                    (uid, f"{first} {last}", f"+91-90000-{uid:05d}",
                     f"{first.lower()}.{last.lower()}{uid}@example.com",
                     TODAY, f"-{int(rng.expovariate(1 / 20))} days"))
                visits = {n for n, _ in rng.sample(PLACES[persona], rng.randint(0, len(PLACES[persona])))}
                if rng.random() < 0.3:                                             # a random other place
                    visits.add(rng.choice(all_places))
                if rng.random() < 0.7:
                    visits.add(POPULAR_PLACE[0])
                conn.executemany("INSERT INTO checkins VALUES (?, ?, date(?, ?))",
                                 [(uid, pid[v], TODAY, f"-{rng.randint(0, 90)} days") for v in visits])
                picks = set(rng.sample(own, rng.randint(2, 4)))                    # core persona tastes
                if rng.random() < 0.5:                                             # one off-persona hobby
                    other = rng.choice([p for p in PERSONAS if p != persona])
                    picks.add(rng.choice(PERSONAS[other]))
                picks |= set(rng.sample(COMMON, rng.randint(1, 3)))               # generic stuff
                conn.executemany("INSERT INTO user_interests VALUES (?, ?)",
                                 [(uid, iid[p]) for p in picks])
                persona_of[uid] = persona

        # Ground truth, mimicking how real friendships form:
        #  1. shared taste: a couple of links to people of the same persona
        #  2. met somewhere: people who frequent the same (non-mall) place sometimes connect
        #  3. triadic closure: friends of friends become friends
        #  4. noise: a random link
        edges = set()
        add = lambda u, v: edges.add((min(u, v), max(u, v))) if u != v else None
        for u, p in persona_of.items():
            for v in rng.sample([v for v, q in persona_of.items() if q == p and v != u], 2):
                add(u, v)
            if rng.random() < 0.3:
                add(u, rng.choice(list(persona_of)))
        regulars = conn.execute("SELECT place_id, GROUP_CONCAT(user_id) FROM checkins "
                                "WHERE place_id != ? GROUP BY place_id", (pid[POPULAR_PLACE[0]],)).fetchall()
        for _, ids in regulars:
            ids = sorted(map(int, ids.split(",")))
            for k, u in enumerate(ids):
                for v in ids[k + 1:]:
                    if rng.random() < 0.08:
                        add(u, v)
        for _ in range(2):
            nb = {u: set() for u in persona_of}
            for a, b in edges:
                nb[a].add(b); nb[b].add(a)
            for u in persona_of:
                fof = sorted({w for v in nb[u] for w in nb[v]} - nb[u] - {u})
                if fof and rng.random() < 0.5:
                    add(u, rng.choice(fof))
        conn.executemany("INSERT INTO connections(user_a, user_b) VALUES (?, ?)", sorted(edges))
    conn.close()
    return path


# One row per user: id, name, '|'-joined interests. No PII columns selected.
PROFILES_SQL = """
SELECT u.id AS user_id, u.name, GROUP_CONCAT(i.label, '|') AS interests
FROM users u
JOIN user_interests ui ON ui.user_id = u.id
JOIN interests i       ON i.id = ui.interest_id
GROUP BY u.id, u.name
ORDER BY u.id
"""

# Users the target may be recommended: not themself, not already connected (either direction).
CANDIDATES_SQL = """
SELECT u.id AS user_id
FROM users u
WHERE u.id != :target
  AND NOT EXISTS (
      SELECT 1 FROM connections c
      WHERE (c.user_a = :target AND c.user_b = u.id)
         OR (c.user_b = :target AND c.user_a = u.id))
ORDER BY u.id
"""


def load_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(PROFILES_SQL, conn)
    df["interests"] = df["interests"].str.split("|").apply(sorted)
    return df


def candidate_ids(conn: sqlite3.Connection, target: int) -> list[int]:
    return [r[0] for r in conn.execute(CANDIDATES_SQL, {"target": target})]


def load_connections(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT user_a, user_b FROM connections", conn)


# Secondary ranking signals: places visited (last 90 days) and days since last activity.
SIGNALS_SQL = """
SELECT u.id AS user_id,
       CAST(julianday(:today) - julianday(u.last_active_at) AS INTEGER) AS days_inactive,
       (SELECT GROUP_CONCAT(name, '|') FROM (
            SELECT DISTINCT p.name FROM checkins c JOIN places p ON p.id = c.place_id
            WHERE c.user_id = u.id AND c.visited_at >= date(:today, '-90 days'))) AS places
FROM users u
ORDER BY u.id
"""


def load_signals(conn: sqlite3.Connection, today: str = TODAY) -> pd.DataFrame:
    df = pd.read_sql_query(SIGNALS_SQL, conn, params={"today": today})
    df["places"] = df["places"].fillna("").str.split("|").apply(lambda xs: sorted(x for x in xs if x))
    return df
