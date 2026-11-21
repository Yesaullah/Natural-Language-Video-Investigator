"""Per-video BM25 with shared corpus and query tokenization."""

import re

import bm25s
import numpy as np
from pydantic import BaseModel, Field


class LexicalSettings(BaseModel, frozen=True, extra="forbid", allow_inf_nan=False):
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


TOKENIZER = {"pattern": r"[^\W_]+", "casefold": True, "stemming": False, "stopwords": None}
BM25_METHOD = "lucene"


def tokenize(text: str) -> list[str]:
    return re.findall(TOKENIZER["pattern"], text.casefold())


class LexicalIndex:
    def __init__(self, texts: list[str], settings: LexicalSettings):
        self.size = len(texts)
        corpus = [tokenize(passage) for passage in texts]
        self.retriever = None
        if any(corpus):
            self.retriever = bm25s.BM25(method=BM25_METHOD, k1=settings.k1, b=settings.b)
            self.retriever.index(corpus, show_progress=False)

    def score(self, tokens: list[str]) -> np.ndarray:
        if not tokens or self.retriever is None:
            return np.zeros(self.size, dtype=np.float64)
        return self.retriever.get_scores(tokens).astype(np.float64)
