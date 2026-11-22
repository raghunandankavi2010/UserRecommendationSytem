"""Link a user's free-text interest to a canonical interest.

    score >= auto_min  -> "linked"        (attach to that interest silently)
    score >= ask_min   -> "did_you_mean"  (app shows the top suggestions; the answer is a training label)
    otherwise          -> "new"           (create a provisional interest unless the user picks a suggestion)

score = cosine similarity between the text and the nearest canonical label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# Chosen by ml/benchmark_linking.py (2-fold, docs/PHASE1_RESULTS.md). Model-specific: retune per model version.
EMBED_MODEL = "thenlper/gte-small"  # 384-d, MIT, ~13 ms per interest on CPU
AUTO_LINK_MIN = 0.91
ASK_MIN = 0.86
TOP_K = 3

Encoder = Callable[[list[str]], np.ndarray]  # texts -> unit-length row vectors


def normalize(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class LinkResult:
    decision: str                 # linked | did_you_mean | new
    interest: str | None          # canonical label when linked
    score: float
    suggestions: tuple[str, ...]  # nearest k labels, nearest first


class InterestLinker:
    def __init__(self, labels: Sequence[str], encode: Encoder,
                 auto_min: float = AUTO_LINK_MIN, ask_min: float = ASK_MIN, k: int = TOP_K):
        if not labels:
            raise ValueError("need at least one canonical interest")
        if ask_min > auto_min:
            raise ValueError("ask_min must be <= auto_min")
        self.labels = list(labels)
        self.encode, self.auto_min, self.ask_min, self.k = encode, auto_min, ask_min, k
        self.vectors = encode(self.labels)
        self._exact = {label.lower(): label for label in self.labels}

    def scores(self, texts: list[str]) -> np.ndarray:
        """Cosine similarity, shape (len(texts), len(labels))."""
        return self.encode([normalize(t) for t in texts]) @ self.vectors.T

    def link(self, text: str) -> LinkResult:
        text = normalize(text)
        if not text:
            raise ValueError("empty interest")
        if (label := self._exact.get(text.lower())) is not None:
            return LinkResult("linked", label, 1.0, (label,))
        return self.decide(self.scores([text])[0])

    def decide(self, row: np.ndarray) -> LinkResult:
        top = np.argsort(-row)[: self.k]
        best = float(row[top[0]])
        suggestions = tuple(self.labels[i] for i in top)
        if best >= self.auto_min:
            return LinkResult("linked", suggestions[0], best, suggestions)
        if best >= self.ask_min:
            return LinkResult("did_you_mean", None, best, suggestions)
        return LinkResult("new", None, best, suggestions)  # app can still offer them before creating


_models: dict = {}


def st_encoder(model: str = EMBED_MODEL) -> Encoder:
    """sentence-transformers encoder, loaded once per process."""
    if model not in _models:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed
        _models[model] = SentenceTransformer(model, device="cpu")
    m = _models[model]
    return lambda texts: m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
