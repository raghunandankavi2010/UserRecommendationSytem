# Phase 1 results: embedding model and interest linking

Run: `.venv/bin/python ml/benchmark_linking.py` → `ml/results/linking_benchmark.json`.
EmbeddingGemma is gated: accept its licence on Hugging Face and run `.venv/bin/hf auth login` first.

## Data
- `ml/data/taxonomy.json`: 255 canonical interests in 12 categories (Indian context included:
  filter coffee, kabaddi, rangoli, Bharatanatyam, Tamil cinema...).
- `ml/data/linking_eval.csv`: 331 hand-written user phrasings → expected interest
  (283 real, 48 "novel" that the taxonomy should not claim, e.g. beekeeping, falconry).
- Thresholds are tuned on one half and tested on the other, both ways (2-fold), then averaged.
  The auto-link threshold must give ≥ 95% precision, because a silent wrong link is the worst error.

## Results (6 open models, CPU)
| Model | Top-1 | Top-3 | Auto-linked | Auto precision | Novel silently linked | **End to end** | ms / interest | Dims |
|---|---|---|---|---|---|---|---|---|
| **google/embeddinggemma-300m** (STS prompt) | **0.89** | **0.98** | **0.64** | **0.97** | 0.02 | **0.96** | 55 | 768 |
| all-mpnet-base-v2 | 0.76 | 0.91 | 0.38 | 0.95 | 0.00 | 0.91 | 29 | 768 |
| thenlper/gte-small | 0.80 | 0.89 | 0.38 | 0.95 | 0.04 | 0.89 | 13 | 384 |
| BAAI/bge-small-en-v1.5 | 0.80 | 0.90 | 0.52 | 0.91 | 0.06 | 0.88 | 13 | 384 |
| all-MiniLM-L6-v2 (prototype) | 0.73 | 0.84 | 0.33 | 0.94 | 0.00 | 0.86 | 8 | 384 |
| Snowflake arctic-embed-s | 0.72 | 0.80 | 0.38 | 0.90 | 0.02 | 0.80 | 13 | 384 |

End to end = share of rows where the user ends up on the right interest. `link()` always returns
the 3 nearest interests, so a real interest counts as right if it is auto-linked correctly or is
among the 3 suggestions; a novel one counts as right if it is not auto-linked.
Without its `STS` prompt EmbeddingGemma drops to top-1 0.85 / top-3 0.93, so the prompt matters.

## Decision: EmbeddingGemma-300m, auto-link ≥ 0.83, "did you mean" ≥ 0.81
- Best on every quality column: 64% of interests link with no question at 97% precision
  (gte-small: 38%), and the right interest is in the top 3 for 98%.
- Cost: 55 ms per interest on CPU, 768-d vectors. Fine here: each distinct interest is embedded
  once, in the background worker; a user vector is an average, no model call. pgvector limit is 2,000.
- Licence: Gemma Terms of Use (commercial use allowed, with Google's prohibited-use policy).
  Fallback if that is a problem: all-mpnet-base-v2 (Apache 2.0), end to end 0.91.
- Thresholds are **model-specific**: every new model version (including fine-tuned ones) must
  re-tune them in the Phase 3 eval gate.

## Hard cases, before → after
| Text | gte-small | EmbeddingGemma |
|---|---|---|
| hiking | did you mean: Walking first | **linked Trekking** |
| filter kaapi | new | **linked Filter coffee** |
| movie marathons | **silently linked Marathons** | did you mean (Marathons, Film festivals, Movies) |
| street art | Street photography | **linked Graffiti** |
| gaming | PC gaming | **linked Video games** |
| weekend treks in the sahyadris | new | new, Trekking first suggestion |

## Still weak (what Phase 2–3 fix)
- Long, specific phrases ("weekend treks in the sahyadris", "drinking coffee with friends") score
  below the threshold; the right interest is the first suggestion, so the user picks it once.
  Phase 2 stores that pick as an **alias**, so the next user typing it links directly.
- "movie marathons" still ranks Marathons first; fine-tuning with such hard negatives (Phase 3).
- The eval set is hand-written by us; real phrasing will differ. Re-run on real
  "did you mean" answers once the app sends them.
