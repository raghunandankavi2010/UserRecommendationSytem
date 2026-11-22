# Phase 1 results: embedding model and interest linking

Run: `.venv/bin/python ml/benchmark_linking.py` → `ml/results/linking_benchmark.json`.

## Data
- `ml/data/taxonomy.json`: 255 canonical interests in 12 categories (Indian context included:
  filter coffee, kabaddi, rangoli, Bharatanatyam, Tamil cinema...).
- `ml/data/linking_eval.csv`: 331 hand-written user phrasings → expected interest
  (283 real, 48 "novel" that the taxonomy should not claim, e.g. beekeeping, falconry).
- Thresholds are tuned on one half and tested on the other, both ways (2-fold), then averaged.
  The auto-link threshold must give ≥ 95% precision, because a silent wrong link is the worst error.

## Results (5 open models, CPU)
| Model | Top-1 | Top-3 | Auto-link rate | Auto precision | Right answer in "did you mean" | Novel silently linked | **End to end** | ms / interest |
|---|---|---|---|---|---|---|---|---|
| **thenlper/gte-small** | **0.80** | 0.89 | 0.38 | 0.95 | 0.96 | 0.04 | **0.74** | 13 |
| all-mpnet-base-v2 | 0.76 | 0.91 | 0.38 | 0.95 | 0.95 | 0.00 | 0.68 | 29 |
| all-MiniLM-L6-v2 (prototype) | 0.73 | 0.84 | 0.33 | 0.94 | 0.96 | 0.00 | 0.67 | 7 |
| BAAI/bge-small-en-v1.5 | 0.80 | 0.90 | 0.52 | 0.91 | 0.47 | 0.06 | 0.66 | 14 |
| Snowflake arctic-embed-s | 0.72 | 0.80 | 0.38 | 0.90 | 0.75 | 0.02 | 0.63 | 12 |

End to end = share of all rows where the user ends up on the right interest: auto-linked correctly,
picked from the suggestions, or (novel) created as new.

## Decision: gte-small, auto-link ≥ 0.91, "did you mean" ≥ 0.86
- Best end to end (+6 points over the prototype's MiniLM), tied best top-1, 384-d (small vectors),
  MIT licence, 13 ms per interest on CPU.
- Its scores sit in a narrow band (0.82–0.95), so the thresholds are **model-specific**: every new
  model version (including fine-tuned ones) must re-tune them in the Phase 3 eval gate.
- `link()` always returns the 3 nearest interests, even for "new", so the app can offer them first.

## Known errors (what fine-tuning must fix)
- `movie marathons` → silently linked to **Marathons** (0.91). Same lexical trap the prototype found.
- `street art` → Street photography; `gaming` → PC gaming (should be Video games).
- `hiking` → Walking ranked above Trekking (still offered in "did you mean").
- About 24% of real interests score below 0.86 and would be created as "new"
  (e.g. `weekend treks in the sahyadris`, `filter kaapi`). Mitigations, cheapest first:
  1. Show suggestions for "new" too (done), and save each confirmed phrase as an **alias** of the
     interest so the next user typing it links directly (Phase 2).
  2. Nightly merge of provisional interests into close canonical ones (Phase 3).
  3. Fine-tune on confirmations and these hard negatives (Phase 3).

## Not tested
- **EmbeddingGemma-300m**: gated on Hugging Face (licence acceptance + login needed). To add it:
  accept the licence at huggingface.co/google/embeddinggemma-300m, run `hf auth login`, then
  `.venv/bin/python ml/benchmark_linking.py google/embeddinggemma-300m thenlper/gte-small`.
- The eval set is hand-written by us; real users' phrasing will differ. Re-run on real
  "did you mean" answers once the app sends them.
