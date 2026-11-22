# Research: matching people by free-text interests

Question: users type interests in their own words ("filter coffee", "weekend treks in Sahyadris",
"street food", "bird photography"). New interests appear all the time. How do we recommend people
who share them, in production, and keep getting better?

Starting point: the prototype in this folder (`src/recsys`, Nov 2026) already compared methods on
synthetic data. Its numbers are the baseline this research builds on (see README "Results").

## 1. The core problem has three parts

| Part | Hard because | What works |
|---|---|---|
| **Understand an interest** | Same thing, many words: "trekking" ≈ "hiking" ≈ "mountain trails". Typos. New ones daily. | Sentence embeddings + a growing canonical interest list (taxonomy) |
| **Find matching people fast** | N users × N users does not scale (prototype: fine to ~20k) | Vector index (pgvector HNSW) + exact-overlap SQL, then re-rank ~200 candidates |
| **Get better over time** | We start with zero feedback (cold start) | Zero-shot first, then fine-tune on our own accept/confirm signals |

## 2. Options compared

### Representing interests
| Option | Verdict | Why |
|---|---|---|
| Exact match / Jaccard | Keep as one signal | Precise, explainable, but "hiking" ≠ "trekking". Prototype: 7× random. |
| TF-IDF over interests | Keep as one signal | Down-weights interests everyone has ("music", "movies"). Rare shared interests are the strongest bond. |
| **Sentence embeddings** | **Primary semantic signal** | Handles synonyms and never-seen-before interests with no retraining. This is what makes "users keep adding new interests" a non-problem on day one. |
| LLM call per match | No (online). Yes (offline helper) | Too slow/costly per request. Useful offline to name/merge taxonomy clusters and generate training pairs. |
| Collaborative filtering / matrix factorisation | Later, maybe | Needs lots of interaction history; useless for brand-new users, which is exactly the onboarding case. |
| Graph neural networks | No | Heavy to run and operate; only pays off with a large social graph. |

**Finding from the prototype:** all interest methods land within ~0.01 nDCG of each other; the
**hybrid (α·TF-IDF + (1-α)·embedding)** is chosen not because it wins every split but because it is
the only one that handles both a rare exact match and a synonym. Adding social signals (mutual
connections, common places) gave +60% nDCG@5 over interests alone.

**Known weakness (prototype):** MiniLM matches surface words: "star trek marathons ~ running
marathons" = 0.65 but "sci-fi movies ~ star trek" = 0.33. Fix = taxonomy (section 3) + fine-tuning
on our own data (section 5). bge-small and mpnet did not rank better on the synthetic set.

### Embedding model candidates
Pick by **our own benchmark** (labelled interest pairs), not by leaderboard, because MTEB measures
documents/RAG, not 2–5 word hobby phrases.

| Model | Size | Notes |
|---|---|---|
| all-MiniLM-L6-v2 | 22M, 384-d | Current baseline. Fast on CPU. English only. |
| EmbeddingGemma-300m | 300M, 768-d (Matryoshka, can cut to 256) | Reported best open model under 500M params on MTEB; multilingual; fine-tunes with sentence-transformers. Strong candidate. |
| bge-m3 / multilingual-e5 | ~570M, 1024-d | Choose if users write in Hindi/Kannada/Tamil etc. |
| Qwen3-Embedding (0.6B+) | larger | Top open scores; probably overkill for short phrases, test only if the small ones fail. |

Interests are few (thousands–tens of thousands of distinct phrases) compared with users, so even a
300M model is cheap: we embed each **distinct interest once**, not each user.

### Storing and searching vectors
| Option | Verdict | Why |
|---|---|---|
| **PostgreSQL + pgvector (HNSW)** | **Use** | One database for users, interests, feedback and vectors; filters (city, age, blocked users) in the same SQL query; HNSW is the standard default for small–medium production workloads. Limit 2,000 dims (fine). |
| Qdrant / Weaviate / Pinecone | Not now | Extra system to run and keep in sync. Worth it only beyond ~10M+ vectors or heavy filtered-search load. |
| FAISS in memory | No | Fast but no persistence, filtering or transactions; we would rebuild what Postgres gives us. |

Watch-outs from pgvector production write-ups: keep the HNSW index in RAM; filter **inside** the
query (or over-fetch then filter) to avoid the "post-filter returns 3 results" trap; measure recall,
not only latency.

## 3. New interests: taxonomy that grows itself

Free text is kept, but every interest is linked to a **canonical interest**:

```
user types "weekend treks"  →  embed  →  nearest canonical interest
   cos ≥ 0.85  → auto-link to "Trekking"
   0.60–0.85  → UI asks "Did you mean Trekking?"  (the answer is a free training label)
   < 0.60     → create new canonical interest "weekend treks" (status = provisional)
```
- Provisional interests become real when ≥ N users add them; a nightly job clusters provisional
  interests and proposes merges (optional LLM to name the cluster; a human or rule approves).
- Autocomplete in the onboarding UI suggests existing canonical interests first, which keeps the
  vocabulary tidy without forcing users into a fixed list.
- Thresholds (0.85 / 0.60) are starting guesses; tune them on labelled pairs.

This gives exact overlap on canonical IDs (precise, explainable "you both like Trekking") **and**
semantic similarity for everything in between.

## 4. Ranking people

Two stages, like every large recommender (retrieve cheaply, then rank carefully):

1. **Candidate retrieval (~200 people)**
   - ANN: user vector (IDF-weighted mean of their interest embeddings) via pgvector HNSW.
   - Exact: people sharing ≥1 rare canonical interest (SQL join on `user_interests`).
   - Exclude self, already connected, blocked, dismissed recently.
2. **Score**: `α·idf_overlap + (1-α)·semantic + w1·mutual + w2·place/city + w3·recency`
   (weights from the prototype, re-tuned on real data). Each result carries a **"why"**: shared
   interests and close pairs ("you: hiking · them: trekking").
3. **Later (≥ ~10k labelled impressions):** replace the hand weights with a learned ranker
   (LightGBM LambdaRank) on the same features plus feedback features.

People matching is **two-sided**: a match only helps if both accept. Score is symmetric, the
"connect" flow is mutual opt-in, and we avoid showing the same popular user to everyone
(cap impressions per candidate per day).

## 5. "Fine-tuning" as users add interests

Two separate loops; most of the "keep learning" comes from the cheap one.

| Loop | Frequency | What changes | Cost |
|---|---|---|---|
| **Online** | Every write | New interest embedded, linked to taxonomy, user vector updated | Milliseconds; no training |
| **Offline fine-tune** | Weekly/monthly, when enough new labels | Embedding model itself | Minutes on one GPU; hours on CPU |

Training data we get for free:
- "Did you mean X?" → yes = positive pair, no = hard negative.
- Taxonomy merges approved by a curator → positive pairs.
- Two users who **accepted** each other → their interests co-occur → weak positive pairs.

Method: sentence-transformers `MultipleNegativesRankingLoss` (contrastive, in-batch negatives) on
(interest, interest) pairs, plus hard negatives from "no" answers. Gate before release: the new
model must beat the current one on a **held-out** set of labelled pairs and on offline nDCG@10 of
accepted matches. Then re-embed all distinct interests (cheap), recompute user vectors into a new
column/version, switch, keep the old one for rollback.

## 6. Privacy and safety (not optional)
- Email and phone are **never** model inputs and **never** in recommendation responses. After both
  users accept, the backend returns name, email, photo and common interests; phone is never returned.
  (The prototype already enforces the first half.)
- Store PII in its own table, encrypted at rest; audit access. India's DPDP Act 2023 applies:
  consent, purpose limitation, deletion on request (delete must also remove vectors and feedback).
- Block/report, rate limits on connect requests, minimum-age check.

## 7. Evaluation
- Offline: nDCG@10, recall@50 of the retrieval stage, coverage (share of users who ever get shown).
- Online: connect-request rate and **mutual accept rate** of shown recommendations (the real goal),
  per model version (A/B by user-id hash).
- Interest linking: precision of auto-links, "did you mean" accept rate.

## Sources
- [Baseten: best open-source embedding models](https://www.baseten.co/blog/the-best-open-source-embedding-models/)
- [AIMultiple: open-source embedding models benchmark](https://aimultiple.com/de/open-source-embedding-models)
- [Modal: top models on the MTEB leaderboard](https://frontend.modal.com/blog/mteb-leaderboard-article)
- [BentoML: guide to open-source embedding models](https://bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [Zilliz: choosing an embedding model in 2026](https://www.zilliz.com/blog/choose-embedding-model-rag-2026)
- [AWS: running pgvector in production on Aurora](https://aws.amazon.com/blogs/database/running-pgvector-in-production-on-amazon-aurora-postgresql/)
- [DigitalOcean: advanced PostgreSQL vector workloads](https://docs.digitalocean.com/products/vector-databases/postgresql/concepts/advanced-workloads/)
- [ClickHouse: scaling vector search in Postgres](https://clickhouse.com/pt-BR/resources/engineering/scale-vector-search-postgres)
- [The New Stack: why pgvector benchmarks lie](https://thenewstack.io/why-pgvector-benchmarks-lie/)
- [Google: fine-tune EmbeddingGemma with Sentence Transformers](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers?authuser=19)
- [Zeroentropy: Multiple Negatives Ranking Loss](https://www.zeroentropy.dev/concepts/multiple-negatives-ranking-loss/)
- [Pinecone community: training with MNR loss](https://community.pinecone.io/t/training-sentence-transformers-with-mnr-loss/85)
