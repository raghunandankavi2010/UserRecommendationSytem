-- Model versions: exactly one is active. A candidate gets its own interest and user vectors built
-- next to the active ones, is activated by flipping status, and rollback flips back.
CREATE TABLE model_versions (
    version      text PRIMARY KEY,
    path         text NOT NULL,                 -- Hugging Face id or local directory (models/<version>)
    auto_min     real NOT NULL,
    ask_min      real NOT NULL,
    status       text NOT NULL CHECK (status IN ('candidate', 'active', 'retired', 'rejected')),
    metrics      jsonb NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz,
    CHECK (ask_min <= auto_min)
);
CREATE UNIQUE INDEX model_versions_one_active ON model_versions ((true)) WHERE status = 'active';

INSERT INTO model_versions (version, path, auto_min, ask_min, status, activated_at)
VALUES ('google/embeddinggemma-300m@base', 'google/embeddinggemma-300m', 0.83, 0.81, 'active', now());

CREATE TABLE interest_vectors (
    interest_id   bigint NOT NULL REFERENCES interests(id) ON DELETE CASCADE,
    model_version text NOT NULL REFERENCES model_versions(version) ON DELETE CASCADE,
    embedding     vector(768) NOT NULL,         -- shortcut: fixed 768-d; a base model of another size needs a new column
    PRIMARY KEY (model_version, interest_id)
);
INSERT INTO interest_vectors (interest_id, model_version, embedding)
SELECT id, model_version, embedding FROM interests;

DROP INDEX interests_embedding_hnsw;           -- interests number in the thousands: an exact scan is fast
ALTER TABLE interests DROP COLUMN embedding, DROP COLUMN model_version;

ALTER TABLE user_vectors
    ADD CONSTRAINT user_vectors_model_version_fkey FOREIGN KEY (model_version)
    REFERENCES model_versions(version) ON DELETE CASCADE;
