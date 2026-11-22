CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id             bigserial PRIMARY KEY,
    external_id    text UNIQUE NOT NULL,            -- the Peoplo app's user id; the API speaks only this
    name           text NOT NULL,
    email          text UNIQUE NOT NULL,            -- PII: never used by the model, never in /matches
    phone          text UNIQUE,                     -- PII: never returned by any endpoint
    photo_url      text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    last_active_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE interests (
    id            bigserial PRIMARY KEY,
    label         text NOT NULL,
    label_key     text UNIQUE NOT NULL,             -- lower-cased, whitespace-collapsed label
    status        text NOT NULL CHECK (status IN ('active', 'provisional', 'merged')),
    merged_into   bigint REFERENCES interests(id),
    embedding     vector(768) NOT NULL,
    model_version text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX interests_embedding_hnsw ON interests USING hnsw (embedding vector_cosine_ops);

-- phrases users confirmed for an interest ("weekend treks" -> Trekking): next time they link directly
CREATE TABLE interest_aliases (
    alias_key   text PRIMARY KEY,
    interest_id bigint NOT NULL REFERENCES interests(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE user_interests (
    user_id     bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    interest_id bigint NOT NULL REFERENCES interests(id),
    raw_text    text NOT NULL,                      -- what the user typed, kept for retraining
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, interest_id)
);
CREATE INDEX user_interests_interest ON user_interests (interest_id);

CREATE TABLE user_vectors (
    user_id       bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_version text NOT NULL,
    embedding     vector(768) NOT NULL,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, model_version)
);
CREATE INDEX user_vectors_embedding_hnsw ON user_vectors USING hnsw (embedding vector_cosine_ops);

CREATE TABLE events (
    id            bigserial PRIMARY KEY,
    actor_id      bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_id     bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type          text NOT NULL CHECK (type IN ('shown', 'wave', 'accept', 'dismiss', 'block')),
    model_version text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (actor_id <> target_id)
);
CREATE INDEX events_actor ON events (actor_id, type, target_id);
CREATE INDEX events_target ON events (target_id, type, actor_id);

-- training labels for fine-tuning: +1 user picked this interest for the text, -1 user rejected it
CREATE TABLE interest_labels (
    id          bigserial PRIMARY KEY,
    user_id     bigint REFERENCES users(id) ON DELETE CASCADE,
    raw_text    text NOT NULL,
    interest_id bigint NOT NULL REFERENCES interests(id) ON DELETE CASCADE,
    label       smallint NOT NULL CHECK (label IN (-1, 1)),
    source      text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- mutual connections: both sides waved or accepted, and nobody blocked the other
CREATE VIEW connections AS
SELECT DISTINCT a.actor_id AS user_id, a.target_id AS other_id
FROM events a
JOIN events b ON b.actor_id = a.target_id AND b.target_id = a.actor_id
WHERE a.type IN ('wave', 'accept') AND b.type IN ('wave', 'accept')
  AND NOT EXISTS (SELECT 1 FROM events x WHERE x.type = 'block'
                  AND ((x.actor_id = a.actor_id AND x.target_id = a.target_id)
                    OR (x.actor_id = a.target_id AND x.target_id = a.actor_id)));
