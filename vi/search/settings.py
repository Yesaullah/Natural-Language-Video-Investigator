"""B1 scoring settings and covered-bin standardization provenance."""

from collections.abc import Iterable
from typing import Literal

import numpy as np

from pydantic import BaseModel, Field

from vi.search.index import RealIndexProvenance
from vi.search.lexical import BM25_METHOD, TOKENIZER, LexicalSettings


class SearchSettings(BaseModel, frozen=True, extra="forbid", allow_inf_nan=False):
    lexical: LexicalSettings = Field(default_factory=LexicalSettings)
    dense_weight: float = Field(ge=0, le=1)
    alpha: float = Field(default=0.7, gt=0, le=1)
    include_cutoff: bool = False

    @property
    def variant(self) -> str:
        if self.dense_weight == 0:
            return "lexical_only"
        if self.dense_weight == 1:
            return "dense_only"
        return "combined"

    def scoring_identity(self) -> dict:
        return {
            "bm25s": "0.3.16", "method": BM25_METHOD, "tokenizer": dict(TOKENIZER),
            **self.lexical.model_dump(),
            "lexical_normalization": "divide_by_query_token_count",
            "dense_weight": self.dense_weight,
            "projection": "max_per_covered_1s_bin",
        }


class Standardization(BaseModel, frozen=True, extra="forbid", allow_inf_nan=False):
    mean: float
    sd: float = Field(ge=0)
    covered_bins: int = Field(ge=0)

    @property
    def scale(self) -> float:
        return self.sd or 1.0



def fit_standardization(timelines: Iterable[np.ndarray]) -> Standardization:
    """Fit equal weight per covered bin across development queries."""
    covered = []
    for timeline in timelines:
        if np.isnan(timeline).any() or np.isposinf(timeline).any():
            raise ValueError("Invalid raw B1 timeline")
        covered.append(timeline[np.isfinite(timeline)])
    if not any(values.size for values in covered):
        raise ValueError("Cannot fit B1 standardization without covered bins")
    values = np.concatenate(covered)
    sd = 0.0 if np.all(values == values[0]) else float(values.std(ddof=0))
    return Standardization(mean=float(values.mean()), sd=sd, covered_bins=values.size)


class RealSettings(BaseModel, frozen=True, extra="forbid"):
    search: SearchSettings
    standardization: Standardization
    index: RealIndexProvenance
    dataset: Literal["team", "charades-sta"]
    fit_population: Literal["dev_query_video_pairs"]
    fit_weighting: Literal["equal_covered_bins_population_sd"]
    fitted_scoring: dict

    def validate_fit(self) -> None:
        if self.fitted_scoring != self.search.scoring_identity():
            raise ValueError("Real B1 fitted scoring configuration mismatch")
        if not self.standardization.covered_bins:
            raise ValueError("Real B1 standardization requires a nonempty covered-bin fit")


FIXTURE_SETTINGS = SearchSettings(dense_weight=0.5)
FIXTURE_STANDARDIZATION = Standardization(mean=0, sd=1, covered_bins=0)
