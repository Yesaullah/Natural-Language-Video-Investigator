"""One-second projection, peak growth and temporal suppression."""

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np

from vi.eval.metrics import temporal_iou
from vi.schema import TranscriptUnit
from vi.search.settings import SearchSettings, Standardization


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    score: float


def project_passages(
    passages: Sequence[TranscriptUnit], scores: np.ndarray, duration_s: float,
) -> np.ndarray:
    """Use -inf for uncovered bins, independently of the sign of covered scores."""
    bins = np.full(math.ceil(duration_s), -np.inf)
    for passage, score in zip(passages, scores, strict=True):
        span = bins[math.floor(passage.start_s):math.ceil(passage.end_s)]
        np.maximum(span, score, out=span)
    return bins


def standardize_bins(bins: np.ndarray, fit: Standardization) -> np.ndarray:
    covered = np.isfinite(bins)
    result = bins.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result[covered] = (bins[covered] - fit.mean) / fit.scale
    if not np.isfinite(result[covered]).all():
        raise ValueError("B1 standardization produced nonfinite scores")
    return result


def extract_windows(
    bins: np.ndarray, duration_s: float, settings: SearchSettings,
) -> list[Window]:
    """Use the first bin of each maximal plateau. A peak always keeps its seed."""
    padded = np.pad(bins, 1, constant_values=-np.inf)
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    windows = set()
    for start, stop in zip(edges[:-1], edges[1:]):
        peak = padded[start + 1]
        if peak <= padded[start] or peak <= padded[stop + 1]:
            continue
        cutoff = settings.alpha * peak
        supports = np.greater_equal if settings.include_cutoff else np.greater
        left, right = int(start), int(start) + 1
        while left > 0 and supports(bins[left - 1], cutoff):
            left -= 1
        while right < len(bins) and supports(bins[right], cutoff):
            right += 1
        windows.add(Window(float(left), min(float(right), duration_s), float(peak)))

    ranked = []
    for window in sorted(windows, key=lambda w: (-w.score, w.start_s, w.end_s)):
        interval = window.start_s, window.end_s
        if any(temporal_iou(interval, (w.start_s, w.end_s)) > 0.5 for w in ranked):
            continue
        ranked.append(window)
        if len(ranked) == 5:
            break
    return ranked
