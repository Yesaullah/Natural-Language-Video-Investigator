"""Reusable transcript-only search over a preloaded per-video index."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.special import expit

from vi.schema import CandidateWindow, Evidence, SearchResult
from vi.search.dense import EncoderIdentity, cosine_scores
from vi.search.index import TranscriptIndex
from vi.search.lexical import LexicalIndex, tokenize
from vi.search.settings import SearchSettings, Standardization
from vi.search.windows import extract_windows, project_passages, standardize_bins


@dataclass(frozen=True)
class PassageScores:
    bm25: np.ndarray | None
    cosine: np.ndarray | None
    combined: np.ndarray
    lexical_term_count: int
    encode_ms: float


class TranscriptSearch:
    def __init__(self, index: TranscriptIndex, settings: SearchSettings):
        self.index = index
        self.settings = settings
        self.lexical = LexicalIndex(
            [passage.text for passage in index.passages], settings.lexical,
        ) if settings.dense_weight < 1 else None

    def score(self, query: str, encoder: Callable[[str], np.ndarray] | None) -> PassageScores:
        if not query.strip():
            raise ValueError("B1 query must contain non-whitespace text")
        terms = tokenize(query)
        weight = self.settings.dense_weight
        identity = getattr(encoder, "identity", None)
        provenance = self.index.provenance
        if weight and provenance is not None:
            expected = EncoderIdentity(
                self.index.meta.models["text_embed"], provenance.encoder_revision,
                self.index.embeddings.shape[1], "query", False,
            )
            if identity != expected:
                raise ValueError("Real B1 query encoder provenance mismatch")
        elif weight and identity is not None and (
            not identity.synthetic or identity.dimension != self.index.embeddings.shape[1]
        ):
            raise ValueError("Synthetic B1 query encoder provenance mismatch")

        bm25 = self.lexical.score(terms) if self.lexical is not None else None
        cosine = None
        encode_ms = 0.0
        if weight:
            if self.index.passages:
                if encoder is None:
                    raise ValueError("B1 dense scoring requires a query encoder")
                started = perf_counter()
                encoded = encoder(query)
                encode_ms = (perf_counter() - started) * 1000
                cosine = cosine_scores(self.index.embeddings, encoded)
            else:
                cosine = np.empty(0)

        combined = np.zeros(len(self.index.passages))
        if bm25 is not None and terms:
            combined += (1 - weight) * bm25 / len(terms)
        if cosine is not None:
            combined += weight * cosine
        return PassageScores(bm25, cosine, combined, len(terms), encode_ms)

    def search(
        self, query: str, encoder: Callable[[str], np.ndarray] | None, fit: Standardization,
    ) -> SearchResult:
        started = perf_counter()
        scores = self.score(query, encoder)
        duration = self.index.video.duration_s
        raw_bins = project_passages(self.index.passages, scores.combined, duration)
        bins = standardize_bins(raw_bins, fit)
        candidates = []
        for window in extract_windows(bins, duration, self.settings):
            excerpts = []
            for passage, score in zip(self.index.passages, scores.combined, strict=True):
                start = max(passage.start_s, window.start_s)
                end = min(passage.end_s, window.end_s)
                if start < end and np.any(raw_bins[math.floor(start):math.ceil(end)] == score):
                    excerpts.append(passage.text)
            candidates.append(CandidateWindow(
                start_s=window.start_s, end_s=window.end_s, score=window.score,
                path_scores={"transcript": window.score}, evidence=Evidence(transcript=excerpts),
            ))
        return SearchResult(
            abstained=not candidates,
            confidence=float(expit(candidates[0].score)) if candidates else 0.0,
            route="transcript", results=candidates,
            timings_ms={
                "encode": scores.encode_ms, "search": (perf_counter() - started) * 1000 - scores.encode_ms,
                "rerank": 0.0,
            },
        )
