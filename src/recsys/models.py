"""Four interchangeable recommenders over user interest profiles.

Each method turns the profiles into an N x N user-similarity matrix; `recommend`
ranks candidates by that row and explains the result with shared interests.

  jaccard   exact overlap / union (baseline)
  tfidf     one token per interest, IDF down-weights interests everyone has
  embedding sentence-transformer per interest phrase, IDF-weighted mean per user
  hybrid    alpha * tfidf + (1 - alpha) * embedding
  random    sanity baseline

Optional second stage (`weights=`): interest score stays primary, small bonuses are added for
  mutual   mutual friends (from the connections graph), saturating at 3
  place    places in common, e.g. the same coffee shop (TF-IDF cosine, so the mall everyone visits counts little)
  recency  candidate's activity recency, exp(-days_inactive / 30)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

METHODS = ("random", "jaccard", "tfidf", "embedding", "hybrid")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, ~90 MB, CPU friendly
# MiniLM puts true synonyms at ~0.25-0.5 ("sci-fi movies ~ star trek" 0.33). shortcut: it also scores
# shared words high ("star trek marathons ~ running marathons" 0.65), so explanations can include a
# lexical false pair; upgrade by fine-tuning on real co-follow data or curating an interest taxonomy.
SIMILAR_PAIR_MIN = 0.3
MAX_SIMILAR_PAIRS = 3
# Tuned on a validation split (notebook 03). Bigger mutual/place weights score slightly higher on the
# synthetic data but let social signals override interests, which are meant to stay primary.
# Recency barely predicts links; it is kept small so active people surface first among near-ties.
DEFAULT_WEIGHTS = {"mutual": 0.3, "place": 0.3, "recency": 0.05}

_encoders: dict = {}


def encode(phrases: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Unit-length embeddings for interest phrases (each model loaded once per process)."""
    if model not in _encoders:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed
        _encoders[model] = SentenceTransformer(model, device="cpu")
    return _encoders[model].encode(phrases, normalize_embeddings=True, show_progress_bar=False)


class Recommender:
    def __init__(self, method: str = "hybrid", alpha: float = 0.5, seed: int = 0, weights: dict | None = None,
                 embed_model: str = EMBED_MODEL):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.alpha, self.seed, self.weights = method, alpha, seed, weights
        self.embed_model = embed_model

    def fit(self, profiles: pd.DataFrame, connections: pd.DataFrame | None = None,
            signals: pd.DataFrame | None = None) -> "Recommender":
        """profiles: user_id, name, interests (list[str]) — db.load_profiles().
        connections (user_a, user_b) and signals (user_id, days_inactive, places) are only
        needed when weights are set. In evaluation pass only the *visible* connections."""
        self.profiles = profiles.reset_index(drop=True)
        self.row = {u: i for i, u in enumerate(self.profiles["user_id"])}
        self.sets = [set(x) for x in self.profiles["interests"]]
        tags = self.profiles["interests"].tolist()

        # Each whole interest is one token; idf = log((1+N)/(1+df)) + 1, so rare shared tastes weigh more.
        self.tfidf = TfidfVectorizer(analyzer=lambda doc: doc)
        X = self.tfidf.fit_transform(tags)

        if self.method in ("embedding", "hybrid"):
            vocab = self.tfidf.get_feature_names_out().tolist()
            self.vocab = {p: i for i, p in enumerate(vocab)}
            self.E = encode(vocab, self.embed_model)       # (V, d)
            U = X @ self.E                                 # tfidf-weighted sum of phrase vectors
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            self.user_vecs = U                             # this is what pgvector would store per user

        # shortcut: dense N x N matrix is O(N^2) memory; fine to ~20k users, beyond that use pgvector ANN.
        self.sim = self._similarity(X, tags)
        self.final = self.sim.copy()
        if self.weights:
            self._add_signals(connections, signals)
        np.fill_diagonal(self.final, -np.inf)              # never recommend yourself
        return self

    def _add_signals(self, connections: pd.DataFrame, signals: pd.DataFrame) -> None:
        n, w = len(self.row), self.weights
        A = np.zeros((n, n))
        for a, b in connections[["user_a", "user_b"]].itertuples(index=False):
            if a in self.row and b in self.row:
                A[self.row[a], self.row[b]] = A[self.row[b], self.row[a]] = 1
        self.mutual = A @ A                                # common-neighbour counts
        sig = signals.set_index("user_id").reindex(self.profiles["user_id"])
        self.places = [set(p) if isinstance(p, list) else set() for p in sig["places"]]
        if any(self.places):
            P = TfidfVectorizer(analyzer=lambda doc: doc).fit_transform([sorted(p) for p in self.places])
            self.place_sim = cosine_similarity(P)
        else:                                              # no check-ins yet: TF-IDF would have an empty vocabulary
            self.place_sim = np.zeros((n, n))
        self.days_inactive = sig["days_inactive"].fillna(365).to_numpy()
        recency = np.exp(-self.days_inactive / 30)
        self.final += (w.get("mutual", 0) * np.minimum(self.mutual, 3) / 3
                       + w.get("place", 0) * self.place_sim
                       + w.get("recency", 0) * recency[None, :])   # column-wise: candidate's recency

    def _similarity(self, X, tags) -> np.ndarray:
        m = self.method
        if m == "random":
            n = len(tags)
            return np.random.default_rng(self.seed).random((n, n))
        if m == "jaccard":
            B = MultiLabelBinarizer(sparse_output=True).fit_transform(tags).astype(float)
            inter = (B @ B.T).toarray()
            size = np.asarray(B.sum(axis=1)).ravel()
            return inter / (size[:, None] + size[None, :] - inter)
        tfidf_sim = cosine_similarity(X)
        if m == "tfidf":
            return tfidf_sim
        emb_sim = np.clip(self.user_vecs @ self.user_vecs.T, 0.0, 1.0)
        if m == "embedding":
            return emb_sim
        return self.alpha * tfidf_sim + (1 - self.alpha) * emb_sim

    def recommend(self, user_id: int, n: int = 5, candidate_ids=None) -> pd.DataFrame:
        """Top-n for user_id. candidate_ids: who may be suggested (e.g. db.candidate_ids(conn, user_id)
        which already removes self and existing connections); default = everyone else."""
        i = self.row[user_id]
        rows = (np.arange(len(self.row)) if candidate_ids is None
                else np.array([self.row[c] for c in candidate_ids if c in self.row and c != user_id], dtype=int))
        rows = rows[rows != i]
        top = rows[np.argsort(-self.final[i, rows], kind="stable")[:n]]
        out = []
        for j in top:
            rec = {
                "user_id": int(self.profiles.at[j, "user_id"]),
                "name": self.profiles.at[j, "name"],
                "score": round(float(self.final[i, j]), 4),
                "shared_interests": sorted(self.sets[i] & self.sets[j]),
            }
            if self.method in ("embedding", "hybrid"):
                rec["similar_interests"] = self._similar_pairs(i, j)
            if self.weights:
                rec["interest_score"] = round(float(self.sim[i, j]), 4)
                rec["mutual_friends"] = int(self.mutual[i, j])
                rec["common_places"] = sorted(self.places[i] & self.places[j])
                rec["days_inactive"] = int(self.days_inactive[j])
            out.append(rec)
        return pd.DataFrame(out)

    def _similar_pairs(self, i: int, j: int) -> list[str]:
        """Non-identical interest pairs that mean nearly the same thing, e.g. 'sci-fi movies ~ star trek'."""
        a = sorted(self.sets[i] - self.sets[j])
        b = sorted(self.sets[j] - self.sets[i])
        if not a or not b:
            return []
        S = self.E[[self.vocab[x] for x in a]] @ self.E[[self.vocab[y] for y in b]].T
        pairs = sorted(zip(*np.where(S >= SIMILAR_PAIR_MIN)), key=lambda rc: -S[rc])[:MAX_SIMILAR_PAIRS]
        return [f"{a[r]} ~ {b[c]} ({S[r, c]:.2f})" for r, c in pairs]
