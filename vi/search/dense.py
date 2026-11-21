"""Cosine scoring and explicit synthetic and real query encoders."""

from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np

from vi.search.index import MODEL_DIMENSION, MODEL_ID


@dataclass(frozen=True)
class EncoderIdentity:
    model: str
    revision: str
    dimension: int
    query_prompt: str | None
    synthetic: bool


class QueryEncoder(Protocol):
    identity: EncoderIdentity

    def __call__(self, text: str) -> np.ndarray: ...


def cosine_scores(embeddings: np.ndarray, encoded: np.ndarray) -> np.ndarray:
    vector = np.asarray(encoded)
    if vector.shape != (embeddings.shape[1],) or vector.dtype.kind not in "iuf":
        raise ValueError("Query encoder must return a real vector of the transcript embedding dimension")
    vector = vector.astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Query embedding must be finite")
    scale = np.max(np.abs(vector))
    if not scale:
        raise ValueError("Query embedding must have nonzero norm")
    vector /= scale
    vector /= np.linalg.norm(vector)
    return embeddings @ vector


class SyntheticEncoder:
    """Deterministic arbitrary-text fixture encoder with no semantic meaning."""

    def __init__(self, dimension: int):
        self.identity = EncoderIdentity("synthetic-sha256", "v1", dimension, None, True)

    def __call__(self, text: str) -> np.ndarray:
        raw = hashlib.shake_256(text.encode("utf-8")).digest(2 * self.identity.dimension)
        return np.frombuffer(raw, dtype="<u2").astype(np.float64) - 32767.5


class QwenQueryEncoder:
    """Lazily load the pinned local model once and reuse it across video indexes."""

    def __init__(self, revision: str):
        self.identity = EncoderIdentity(MODEL_ID, revision, MODEL_DIMENSION, "query", False)
        self._model = None

    def __call__(self, text: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.identity.model, revision=self.identity.revision,
                    truncate_dim=self.identity.dimension, local_files_only=True, device="cpu",
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"Real B1 encoder {self.identity.model}@{self.identity.revision} is unavailable. "
                    "Install sentence-transformers and cache the pinned model on this machine."
                ) from exc
        return self._model.encode(
            text, prompt_name=self.identity.query_prompt, normalize_embeddings=True,
        )
