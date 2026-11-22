"""Hold-out evaluation: hide some existing connections and check whether each method
recommends them back. Users only see the remaining (visible) connections, so the
mutual-friends signal cannot peek at the answer."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .models import Recommender


def split_edges(conns: pd.DataFrame, hide_frac: float = 0.3, seed: int = 0):
    hidden = conns.sample(frac=hide_frac, random_state=seed)
    return conns.drop(hidden.index), hidden


def _adj(conns: pd.DataFrame) -> dict[int, set[int]]:
    nb = defaultdict(set)
    for a, b in conns[["user_a", "user_b"]].itertuples(index=False):
        nb[a].add(b)
        nb[b].add(a)
    return nb


def ranking_metrics(recommended: list[int], relevant: set[int], k: int) -> dict:
    hits = [1.0 if r in relevant else 0.0 for r in recommended[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    first = next((i for i, h in enumerate(hits) if h), None)
    return {"precision@k": sum(hits) / k, "recall@k": sum(hits) / len(relevant),
            "ndcg@k": dcg / idcg, "mrr@k": 0.0 if first is None else 1 / (first + 1)}


def evaluate(make: Callable[[], Recommender], profiles: pd.DataFrame, conns: pd.DataFrame,
             signals: pd.DataFrame | None = None, k: int = 5, seeds=range(1, 6), hide_frac: float = 0.3) -> dict:
    """Mean metrics over several random hide-splits. `make` builds a fresh, unfitted Recommender."""
    all_ids = set(profiles["user_id"])
    per_seed = []
    for seed in seeds:
        visible, hidden = split_edges(conns, hide_frac, seed)
        model = make().fit(profiles, visible, signals)
        seen, truth = _adj(visible), _adj(hidden)
        rows, shown = [], set()
        for u, relevant in truth.items():
            cands = sorted(all_ids - seen[u] - {u})
            recs = model.recommend(u, k, cands)["user_id"].tolist()
            shown.update(recs)
            rows.append(ranking_metrics(recs, relevant, k))
        m = pd.DataFrame(rows).mean().to_dict()
        m["coverage"] = len(shown) / len(all_ids)   # share of users who ever get recommended
        per_seed.append(m)
    return pd.DataFrame(per_seed).mean().round(4).to_dict()


def compare(configs: dict[str, Callable[[], Recommender]], profiles, conns, signals=None, **kw) -> pd.DataFrame:
    return (pd.DataFrame({name: evaluate(make, profiles, conns, signals, **kw) for name, make in configs.items()})
            .T.sort_values("ndcg@k", ascending=False))


def tune_alpha(profiles, conns, grid=np.linspace(0, 1, 11), seed: int = 0) -> pd.Series:
    """nDCG of the hybrid for each alpha on a separate validation split (seed 0; reports use seeds 1-5)."""
    return pd.Series({round(a, 2): evaluate(lambda a=a: Recommender("hybrid", alpha=a), profiles, conns,
                                            seeds=[seed])["ndcg@k"] for a in grid}, name="ndcg@k")
