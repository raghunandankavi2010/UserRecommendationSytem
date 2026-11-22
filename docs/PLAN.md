# Plan: interest matching backend + model pipeline

Why each choice: see `RESEARCH.md`. This plan turns the prototype (`src/recsys`) into a service.

## Scope (decided 2026-11-22)
- **In:** backend API + the model pipeline that keeps improving as new interests and feedback arrive.
- **Out (for now):** any web frontend, login/OTP. The client is the **Peoplo Android app**: it
  onboards the user (name, email, phone, photo, interests), does location matching, and owns the
  wave → accept → chat → meet flow. The backend ranks people by common interests and returns
  profile info to the app.
- English only. Runs locally in Docker. Sized for up to 100k users in year one.

## Stack
| Layer | Choice | Why |
|---|---|---|
| Database + vectors | PostgreSQL 16 + pgvector (HNSW) | One store for users, interests, vectors, feedback |
| API | Python, FastAPI, SQLAlchemy + Alembic | Same language as the model; prototype code reused |
| Model | sentence-transformers `google/embeddinggemma-300m` (chosen in Phase 1, see `PHASE1_RESULTS.md`) | English, small, fine-tunable, works on unseen interests |
| Background work | One worker (`arq` on Redis) + scheduled jobs | Embed on write, taxonomy merge, retrain |
| Packaging | Docker Compose: db, redis, api, worker | Local now, portable to a VM later |
| Observability | Structured logs, Prometheus `/metrics` | Latency, errors, model version per request |

## How the app and the backend split the work
```
 Peoplo Android                                   Backend (this project)
 ───────────────                                   ──────────────────────
 onboarding (name,email,phone,photo,interests) ──► POST /users, POST /users/{id}/interests
                                                     └─ worker: embed → link to taxonomy → user vector
 finds people nearby (location) ── nearby ids ────► POST /users/{id}/matches {candidate_ids?}
                                ◄── ranked people: name, photo, common + similar interests, score
 wave / accept / dismiss / block ────────────────► POST /events   (training signal for fine-tuning)
 both accepted                   ────────────────► GET /users/{id}/connections/{other}
                                ◄── name, email, photo, common interests (phone never)
 chat / meet: app's own concern
```
- `candidate_ids` is optional: the app passes the people it found nearby and the backend ranks
  only those; without it the backend ranks across everyone (for "people like you" lists).
- **The app must send the wave/accept/dismiss events.** They are the main fine-tuning data; without
  them the model only learns from taxonomy answers.

## Data model
```
users(id, external_id UNIQUE, name, email, phone, photo_url, created_at, last_active_at, status)
interests(id, label, status[provisional|active|merged], merged_into, user_count,
          embedding vector(d), model_version)
user_interests(user_id, interest_id, raw_text, created_at)        -- raw text kept for retraining
user_vectors(user_id, model_version, embedding vector(d))         -- HNSW index per version
events(id, actor_id, target_id, type[shown|wave|accept|dismiss|block], model_version, ts)
interest_labels(raw_text, interest_id, label[+1|-1], source[did_you_mean|curator|auto])
model_versions(version, path, metrics jsonb, is_active, created_at)
```
- `external_id` = the Peoplo app's user id, so both systems talk about the same person.
- Email/phone: stored, encrypted at rest, never used by the model, never in `/matches`.
- Indexes: HNSW on both vector columns; btree on `user_interests(interest_id)` for exact overlap.

## API (v1)
Service-to-service API key for now (the app or its backend calls us); real user auth comes with the app.

| Method | Path | Does |
|---|---|---|
| POST | `/users` | Onboard: name, email, phone, photo_url, external_id, interests[] |
| PATCH / DELETE | `/users/{id}` | Update profile / delete everything incl. vectors and events |
| POST | `/users/{id}/interests` | Add free text → `linked` / `did_you_mean` / `created` |
| POST | `/users/{id}/interests/{iid}/confirm` | Answer "did you mean" (training label) |
| DELETE | `/users/{id}/interests/{iid}` | Remove |
| GET | `/interests/suggest?q=` | Autocomplete for the app's onboarding |
| POST | `/users/{id}/matches` | Body `{candidate_ids?, limit}` → ranked people with "why" |
| POST | `/events` | wave / accept / dismiss / block / shown |
| GET | `/users/{id}/connections/{other_id}` | Only if both accepted: name, email, photo, common interests |
| GET | `/health`, `/metrics` | Ops |

Match item:
```json
{"user_id": 42, "name": "Asha", "photo_url": "...", "score": 0.81,
 "common_interests": ["Trekking", "Filter coffee"],
 "similar_interests": [{"you": "hiking", "them": "mountain trails", "similarity": 0.78}],
 "model_version": "minilm-ft-2026-12-01"}
```

## Model pipeline
1. **On every interest write (online, ms):** normalise text → embed (cached by text) → nearest
   canonical interest: ≥ 0.85 link, 0.60–0.85 "did you mean", < 0.60 new provisional interest →
   recompute the user's vector (IDF-weighted mean).
2. **Matching:** retrieve ~200 (HNSW on user vectors ∪ users sharing a rare interest, or just the
   given `candidate_ids`) → score `α·idf_overlap + (1-α)·semantic + small bonuses` (prototype weights)
   → exclude self, blocked, dismissed recently → "why".
3. **Nightly:** cluster provisional interests, auto-merge very close ones, list the rest for review
   (CLI/JSON report; no admin UI).
4. **Weekly retrain (when ≥ N new labels):**
   build pairs from `interest_labels` + interests of mutually accepted pairs (hard negatives from
   "no" and dismisses) → fine-tune with `MultipleNegativesRankingLoss` → **gate**: must beat the
   active model on held-out pairs and offline nDCG@10 of accepted matches → re-embed interests,
   rebuild user vectors under the new `model_version` → activate. Rollback = flip `is_active`.
5. **A/B:** hash(user_id) chooses model version; report wave and mutual-accept rate per version.

## Phases

### Phase 0: setup (done 2026-11-22)
- Git repo here; service code in `src/matcher/` next to the prototype `src/recsys/`; `docker-compose.yml`.
- You install Docker (commands provided; needs sudo).
- Done when: `docker compose up` gives a healthy API and Postgres with pgvector.

### Phase 1: model choice + interest linking (done 2026-11-22, `PHASE1_RESULTS.md`)
- Labelled set of ~300–500 English interest pairs (same / related / different), Indian context included.
- Benchmark MiniLM vs EmbeddingGemma-300m (pair AUC, CPU latency); pick; tune thresholds.
- Seed taxonomy of ~300 canonical interests.
- Done when: linking precision is measured and tested.

### Phase 2: backend core (3–4 days)
- Schema + migrations, onboarding/interest endpoints, worker, matching, events, connections.
- Interest aliases: every confirmed "did you mean" phrase is stored and linked directly next time.
- Port `Recommender` scoring from the prototype.
- Done when: API tests cover the three linking paths, matching with and without `candidate_ids`,
  exclusions, PII never in `/matches`, contact only after mutual accept, p95 < 150 ms on 100k users.

### Phase 3: learning loop (2–3 days)
- Nightly taxonomy job, weekly retrain with eval gate, versioned re-embed, rollback, A/B report.
- Synthetic event generator so the loop can be exercised before the app sends real events.
- Done when: a retrain can fail the gate safely, or ship and roll back with one command.

### Phase 4: hardening (1–2 days)
- Replace the placeholder Postgres password (still `change-me-local-only` locally), PII encryption, API key auth, rate limits, load test at 100k users, HNSW tuning
  (`m`, `ef_search`), backups, health checks.
- Hand-off doc for the Android team: endpoints, payloads, which events to send.

### Later, only if numbers ask for it
- Learned ranker (LightGBM LambdaRank) after ~10k labelled events.
- Read replica / dedicated vector DB beyond ~1M users.

## Still open
- Path to the Peoplo Android project (not found on this machine), so the API matches its models.
- Does the app have its own backend (chat, users), or is this service its only backend?
