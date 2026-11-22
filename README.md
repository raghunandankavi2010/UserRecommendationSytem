# Recommendation System: who should you connect with?

Recommends people to a user from **shared interests** (primary signal), then re-ranks with
**mutual friends**, **places in common** ("we both go to the same coffee shop") and **activity recency**.
Every recommendation explains itself: shared interests, semantically similar interests, mutual friend
count, common places, days since active. Phone and email are stored but never used or returned.

## Service (production version, in progress)
Plan and research: `docs/PLAN.md`, `docs/RESEARCH.md`, `docs/PHASE1_RESULTS.md`. Code: `src/matcher/` (API, `linking.py`), `ml/` (data, benchmarks).
```bash
cp .env.example .env          # then set POSTGRES_PASSWORD
docker compose up -d --build
curl localhost:8000/health    # {"status":"ok","pgvector":"..."}
TEST_DATABASE_URL=postgresql://matcher:<password>@127.0.0.1:5432/matcher .venv/bin/pytest -q tests
```

## Run it (prototype)
```bash
cd ~/MachineLearning/Recommendation System
source .venv/bin/activate
python -m recsys.cli --user 30            # top 5, hybrid + signals
python -m recsys.cli --user 30 --method tfidf --no-signals
pytest -q tests                           # 5 checks
jupyter lab notebooks/                    # 01 data+SQL, 02 methods side by side, 03 evaluation
```
Recreate from scratch: `python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -r requirements.txt`

## Layout
| Path | What |
|---|---|
| `src/recsys/db.py` | SQLite schema, synthetic data (200 users), the SQL: profiles, candidate exclusion, signals. PostgreSQL/pgvector swap notes at the top |
| `src/recsys/models.py` | `Recommender(method, alpha, weights)`: jaccard, tfidf, embedding, hybrid, random, plus the signal re-rank |
| `src/recsys/evaluate.py` | Hold-out evaluation: hide 30% of connections and measure how many come back in the top 5 |
| `src/recsys/cli.py` | Command line |
| `notebooks/` | Executed notebooks with outputs and charts |
| `data/recsys.db` | Generated database (deterministic, seed 42) |

## Pipeline
```
SQL: profiles (user, interests)   ──►  interest similarity (N×N)  ──►  + 0.3·mutual friends   ──►  top-N, excluding
SQL: candidates (NOT EXISTS conn.)      jaccard | tfidf | MiniLM          + 0.3·common places         self & existing
SQL: signals (places 90d, recency)      hybrid = α·tfidf + (1-α)·emb     + 0.05·recency              connections, + "why"
```

## Results: which works best
Five random hold-out splits, 200 users, 779 connections. nDCG@5 is the main metric (higher is better).

| Method | P@5 | R@5 | nDCG@5 | MRR@5 |
|---|---|---|---|---|
| **Hybrid + mutual + places + recency** | **0.127** | **0.262** | **0.211** | **0.277** |
| Hybrid + mutual friends only | 0.116 | 0.243 | 0.198 | 0.265 |
| TF-IDF + all signals | 0.116 | 0.231 | 0.184 | 0.245 |
| Hybrid + places only | 0.106 | 0.212 | 0.166 | 0.228 |
| Jaccard (interests only) | 0.090 | 0.182 | 0.142 | 0.189 |
| Embedding / hybrid (interests only) | 0.090 | 0.175 | 0.131 | 0.176 |
| TF-IDF (interests only) | 0.086 | 0.173 | 0.128 | 0.168 |
| Random | 0.013 | 0.024 | 0.020 | 0.028 |

**Winner: hybrid interest score + social re-rank, about 60% better than interests alone.**
- **Every interest method is about 7× better than random**, but they finish within about 0.01 nDCG of each other. Which interest method comes first changes with the data: on an earlier version of the dataset, embeddings won (0.168 vs TF-IDF 0.156). Keep the hybrid and retune α on real data (notebook 03), because it is the only one that handles both a rare exact match and synonyms.
- **Mutual friends is the strongest extra signal, and places is second.** Recency barely predicts friendships. It is kept at a small weight so that, among near-ties, active people are shown first.
- **Interests can find someone's tribe, but not their actual friends.** Social signals do that part, which is why the two stages complement each other.

### Be honest about the data
It is synthetic. Connections were generated from (1) shared persona, (2) meeting at the same place, (3) friends of friends, and (4) noise. The signals help *because* the data was built that way. This mirrors well-known real network behaviour, but measure it again on real connections before trusting the weights. In an earlier version with random-within-group connections, the social signals added nothing.

### Known limits
- MiniLM matches surface words: it scores "star trek marathons ~ running marathons" at 0.65, but "sci-fi movies ~ star trek" only at 0.33. So `similar_interests` can contain a false pair. bge-small and mpnet were tried and did not rank better. To fix it, fine-tune on real co-interest data or map free text to a curated interest taxonomy.
- The similarity is a dense N×N matrix, fine up to about 20k users. Beyond that, store user vectors in pgvector (HNSW) and re-rank only the nearest ~200 candidates (see `db.py` header).
- Cold start: users with no interests are skipped by the profiles query.
