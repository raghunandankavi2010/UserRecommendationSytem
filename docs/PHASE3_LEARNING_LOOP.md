# Phase 3: the learning loop

How the system gets better as users add interests, and how a new model reaches production safely.

## What feeds it
| Signal | Where it comes from | Used for |
|---|---|---|
| **Picked suggestion** (+1) | user answers "did you mean" / "new" with an existing interest | fine-tuning positives, alias (next user links directly) |
| **Rejected suggestions** (−1) | user chose "new" over the 3 suggestions | fine-tuning hard negatives; blocks a nightly merge into that interest |
| **New interests** | user chose "new" | provisional vocabulary; promoted after 5 users |
| **shown / wave / accept** | the app's events | per-version outcome report; ranking-weight tuning later |

Deliberately **not** used for fine-tuning: interests of users who connected. People who trek often
like coffee, but "trekking" must not move closer to "coffee"; linking needs synonymy, not co-occurrence.

## Jobs (`ml/jobs.py`)
| Job | Schedule | Does |
|---|---|---|
| `nightly` | every night | embed missing interests · merge near-duplicate provisional interests (never into one a user rejected for that text) · promote popular ones · list ambiguous ones for review · rebuild every user vector with fresh IDF |
| `retrain [--promote]` | weekly | skip if < 200 positive labels · fine-tune the active model (MNRL on picks, contrastive on rejections, STS prompt, token table frozen for CPU) · **gate** · register as `candidate` or `rejected` · optionally promote |
| `promote VERSION` | on demand | build interest + user vectors for VERSION next to the live ones · flip it active · wait for the API to reload (≤ 30 s) · catch-up pass |
| `rollback` | on demand | promote the previously active version again |
| `report` | any time | wave rate and mutual-accept rate of shown recommendations, per model version |

## The gate (a candidate must pass every check)
- **Hand-labelled eval set**, excluding any text it trained on: auto-link precision ≥ 0.95, end to end
  no more than 1 point below the active model, novel interests silently linked no more than +2 points.
- **Held-out real labels** (1 in 5 labelled texts, by hash, never trained on): top-1 or top-3 better
  than the active model, and no more rejected suggestions auto-linked.
- Thresholds are re-tuned for the candidate (fine-tuning moves the score scale: in the first run the
  best auto-link threshold went from 0.83 to 0.52). Rejected candidates keep their metrics row; weights are deleted.

## First run (simulated, `ml/simulate.py`, 300 users)
51 picks + 69 rejections collected; 102 rows trained, 18 held out. Result: **rejected**, correctly.
Real held-out top-1 fell 0.83 → 0.67 and auto-link precision was 0.947 (< 0.95): with ~100 rows the model
over-fits (training loss → 0.003). Expect the first useful retrain after a few hundred real picks.
Nightly in the same run promoted 5 popular provisional interests and refused to merge "tai chi" into
Martial arts because a user had rejected exactly that.

## Not built (on purpose, for now)
- **Live A/B serving.** Promotion is all-or-nothing with instant rollback, and `report` compares versions
  over time. Serving two models side by side needs both encoders loaded (~2 GB) and vectors kept for
  both; worth it once there is enough traffic to read a difference.
- **Learned ranking weights.** Mutual/recency weights are still the prototype's; tune them from
  `events` once there are ~10k labelled impressions.
