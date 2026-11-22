"""Model versions: which encoder and thresholds are live (table model_versions)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from matcher import linking
from matcher.linking import Encoder, st_encoder

RELOAD_SECONDS = 30


@dataclass(frozen=True)
class Model:
    version: str
    path: str
    encode: Encoder
    auto_min: float
    ask_min: float


def load(conn, version: str | None = None) -> Model:
    """The active model, or a named version (e.g. a candidate being built)."""
    if version is None:
        row = conn.execute("SELECT * FROM model_versions WHERE status = 'active'").fetchone()
    else:
        row = conn.execute("SELECT * FROM model_versions WHERE version = %s", (version,)).fetchone()
    if row is None:
        raise LookupError(f"model version not found: {version or 'active'}")
    return Model(row["version"], row["path"], st_encoder(row["path"]), float(row["auto_min"]), float(row["ask_min"]))


def active_version(conn) -> str:
    return conn.execute("SELECT version FROM model_versions WHERE status = 'active'").fetchone()["version"]


class Live:
    """The active model for a long-running process; picks up a promotion or rollback within RELOAD_SECONDS."""

    def __init__(self, pool):
        self.pool, self.model, self._checked, self._lock = pool, None, 0.0, threading.Lock()

    def get(self) -> Model:
        if self.model is not None and time.monotonic() - self._checked < RELOAD_SECONDS:
            return self.model
        with self._lock:
            if self.model is None or time.monotonic() - self._checked >= RELOAD_SECONDS:
                with self.pool.connection() as c:
                    version = active_version(c)
                    if self.model is None or version != self.model.version:
                        old, self.model = self.model, load(c, version)
                        if old is not None and old.path != self.model.path:
                            linking._models.pop(old.path, None)  # free the old encoder (~1 GB)
                self._checked = time.monotonic()
        return self.model
