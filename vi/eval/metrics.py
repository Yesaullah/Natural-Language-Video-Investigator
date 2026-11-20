"""Pure metric functions for Natural-Language Video Investigator.

Implements temporal IoU, raw and abstention-aware retrieval metrics,
abstention rates, balanced accuracy, risk-coverage area, and Wilson intervals.
Conforms to video_investigator_plan.md §6.1 and plans/M1-evaluation-foundation.md §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats

from vi.schema import CandidateWindow, QueryLabel, SearchResult


@dataclass(frozen=True)
class RateResult:
    """Represents a metric rate along with its numerator and denominator."""

    numerator: int
    denominator: int
    rate: Optional[float]


@dataclass(frozen=True)
class StartTimeSummary:
    """Median and quartiles for start-time absolute errors."""

    median: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    iqr: Optional[float]
    count: int


def temporal_iou(
    window_a: Tuple[float, float],
    window_b: Tuple[float, float],
) -> float:
    """Calculate temporal Intersection over Union (IoU) for two half-open intervals.

    Intervals are represented as (start_s, end_s).
    """
    s_a, e_a = float(window_a[0]), float(window_a[1])
    s_b, e_b = float(window_b[0]), float(window_b[1])

    intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
    len_a = max(0.0, e_a - s_a)
    len_b = max(0.0, e_b - s_b)
    union = len_a + len_b - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def best_iou_with_gold(
    candidate: Tuple[float, float],
    gold_windows: Sequence[Tuple[float, float]],
) -> float:
    """Return the greatest IoU between candidate and any gold window."""
    if not gold_windows:
        return 0.0
    return max(temporal_iou(candidate, gold) for gold in gold_windows)


def _get_visible_candidates(
    result: SearchResult,
    abstention_aware: bool,
) -> List[CandidateWindow]:
    """Retrieve visible candidates based on abstention-aware setting.

    If abstention_aware is True and the result abstained, treat visible
    candidates as empty.
    """
    if abstention_aware and result.abstained:
        return []
    return result.results


def recall_at_k(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
    k: int,
    iou_threshold: float,
    abstention_aware: bool = False,
) -> RateResult:
    """Compute Recall@K at a given IoU threshold over answerable queries.

    Threshold equality counts as a hit (IoU >= threshold).
    An empty result contributes zero.
    """
    if len(query_labels) != len(search_results):
        raise ValueError(
            f"Mismatched counts: {len(query_labels)} labels vs {len(search_results)} results"
        )

    ans_count = 0
    hits = 0

    for q, res in zip(query_labels, search_results):
        if not q.answerable:
            continue
        ans_count += 1
        candidates = _get_visible_candidates(res, abstention_aware)[:k]
        hit = False
        for c in candidates:
            if best_iou_with_gold((c.start_s, c.end_s), q.gold_windows) >= iou_threshold:
                hit = True
                break
        if hit:
            hits += 1

    if ans_count == 0:
        return RateResult(numerator=0, denominator=0, rate=None)
    return RateResult(numerator=hits, denominator=ans_count, rate=hits / ans_count)


def mean_reciprocal_rank(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
    max_rank: int = 5,
    iou_threshold: float = 0.5,
    abstention_aware: bool = False,
) -> Optional[float]:
    """Compute Reciprocal Rank through rank 5 at IoU threshold over answerable queries.

    Threshold equality counts as a hit (IoU >= threshold).
    An empty or missed result contributes 0.0.
    """
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts between query_labels and search_results")

    ans_count = 0
    rr_sum = 0.0

    for q, res in zip(query_labels, search_results):
        if not q.answerable:
            continue
        ans_count += 1
        candidates = _get_visible_candidates(res, abstention_aware)[:max_rank]
        for rank_idx, c in enumerate(candidates, start=1):
            if best_iou_with_gold((c.start_s, c.end_s), q.gold_windows) >= iou_threshold:
                rr_sum += 1.0 / rank_idx
                break

    if ans_count == 0:
        return None
    return rr_sum / ans_count


def top_window_mean_iou(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
    abstention_aware: bool = False,
) -> Optional[float]:
    """Compute mean IoU of the top-ranked candidate over answerable queries.

    An empty candidate list contributes 0.0 IoU.
    """
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts between query_labels and search_results")

    ans_count = 0
    iou_sum = 0.0

    for q, res in zip(query_labels, search_results):
        if not q.answerable:
            continue
        ans_count += 1
        candidates = _get_visible_candidates(res, abstention_aware)
        if candidates:
            top_c = candidates[0]
            iou_sum += best_iou_with_gold((top_c.start_s, top_c.end_s), q.gold_windows)
        else:
            iou_sum += 0.0

    if ans_count == 0:
        return None
    return iou_sum / ans_count


def start_time_error(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
    abstention_aware: bool = False,
) -> StartTimeSummary:
    """Summarize absolute start-time errors for top candidates that overlap a gold window.

    Tie-breaking rule:
    Use the best-overlapping gold window for the top candidate. If multiple gold windows
    achieve the greatest IoU (> 0), break ties by choosing the gold window with the earliest
    start time.
    """
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts between query_labels and search_results")

    eligible_errors: List[float] = []

    for q, res in zip(query_labels, search_results):
        if not q.answerable:
            continue
        candidates = _get_visible_candidates(res, abstention_aware)
        if not candidates:
            continue

        top_cand = candidates[0]
        c_span = (top_cand.start_s, top_cand.end_s)

        # Find best overlapping gold window with IoU > 0
        overlapping = [
            (temporal_iou(c_span, g), g[0], g[1])
            for g in q.gold_windows
            if temporal_iou(c_span, g) > 0.0
        ]
        if not overlapping:
            continue

        # Sort by: highest IoU descending, then earliest start ascending
        overlapping.sort(key=lambda item: (-item[0], item[1]))
        best_gold_start = overlapping[0][1]

        err = abs(top_cand.start_s - best_gold_start)
        eligible_errors.append(float(err))

    if not eligible_errors:
        return StartTimeSummary(median=None, q25=None, q75=None, iqr=None, count=0)

    errors_arr = np.array(eligible_errors, dtype=np.float64)
    q25, median, q75 = np.percentile(errors_arr, [25, 50, 75])
    iqr = q75 - q25

    return StartTimeSummary(
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        iqr=float(iqr),
        count=len(eligible_errors),
    )


def rejection_rate(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
) -> RateResult:
    """Share of unanswerable queries on which the system abstains."""
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts")

    unans_count = 0
    abstained = 0

    for q, res in zip(query_labels, search_results):
        if q.answerable:
            continue
        unans_count += 1
        if res.abstained:
            abstained += 1

    if unans_count == 0:
        return RateResult(numerator=0, denominator=0, rate=None)
    return RateResult(numerator=abstained, denominator=unans_count, rate=abstained / unans_count)


def false_answer_rate(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
) -> RateResult:
    """Share of unanswerable queries on which the system returns a result (does not abstain)."""
    rej = rejection_rate(query_labels, search_results)
    if rej.denominator == 0 or rej.rate is None:
        return RateResult(numerator=0, denominator=0, rate=None)
    ans_count = rej.denominator - rej.numerator
    return RateResult(numerator=ans_count, denominator=rej.denominator, rate=1.0 - rej.rate)


def false_abstain_rate(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
) -> RateResult:
    """Share of answerable queries on which the system abstains."""
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts")

    ans_count = 0
    abstained = 0

    for q, res in zip(query_labels, search_results):
        if not q.answerable:
            continue
        ans_count += 1
        if res.abstained:
            abstained += 1

    if ans_count == 0:
        return RateResult(numerator=0, denominator=0, rate=None)
    return RateResult(numerator=abstained, denominator=ans_count, rate=abstained / ans_count)


def balanced_accuracy(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
) -> Optional[float]:
    """Average of rejection rate and the share of answerable queries answered.

    Equals: 0.5 * (rejection_rate + (1 - false_abstain_rate)).
    Returns None if either denominator is zero.
    """
    rej = rejection_rate(query_labels, search_results)
    fa = false_abstain_rate(query_labels, search_results)

    if rej.rate is None or fa.rate is None:
        return None

    share_answered = 1.0 - fa.rate
    return 0.5 * (rej.rate + share_answered)


def risk_coverage_curve(
    query_labels: Sequence[QueryLabel],
    search_results: Sequence[SearchResult],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute risk-coverage curve and its area under the curve (AURC).

    Convention:
    Group tied confidence scores into one threshold. For the area use the sum of
    each coverage increment times the risk at its right endpoint.

    An answer is correct when the query is answerable and its top window reaches
    IoU >= iou_threshold with at least one gold window. Otherwise, the answer is wrong
    (including unanswerable queries answered or answerable queries with top window IoU < threshold).
    """
    if len(query_labels) != len(search_results):
        raise ValueError("Mismatched counts between query_labels and search_results")

    total_queries = len(query_labels)
    if total_queries == 0:
        return {"area": None, "curve": []}

    # Extract confidence and correctness for each query
    items: List[Tuple[float, bool]] = []
    for q, res in zip(query_labels, search_results):
        conf = float(res.confidence)
        is_correct = False
        if q.answerable and res.results:
            top_c = res.results[0]
            if best_iou_with_gold((top_c.start_s, top_c.end_s), q.gold_windows) >= iou_threshold:
                is_correct = True
        items.append((conf, is_correct))

    # Group by confidence descending
    conf_groups: Dict[float, List[bool]] = {}
    for conf, correct in items:
        conf_groups.setdefault(conf, []).append(correct)

    sorted_confs = sorted(conf_groups.keys(), reverse=True)

    curve: List[Dict[str, float]] = []
    prev_coverage = 0.0
    area = 0.0

    cumulative_answered = 0
    cumulative_wrong = 0

    for conf in sorted_confs:
        correctness_list = conf_groups[conf]
        cumulative_answered += len(correctness_list)
        cumulative_wrong += sum(1 for c in correctness_list if not c)

        coverage = cumulative_answered / total_queries
        risk = cumulative_wrong / cumulative_answered if cumulative_answered > 0 else 0.0

        cov_increment = coverage - prev_coverage
        area += cov_increment * risk
        prev_coverage = coverage

        curve.append({
            "threshold": conf,
            "coverage": coverage,
            "risk": risk,
        })

    return {
        "area": area,
        "curve": curve,
    }


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the two-sided Wilson score interval without continuity correction.

    Returns (lower, upper). Returns (None, None) if trials is 0.
    """
    if trials <= 0:
        return (None, None)
    if successes < 0 or successes > trials:
        raise ValueError(f"successes ({successes}) must be in [0, trials ({trials})]")

    alpha = 1.0 - confidence
    z = float(scipy.stats.norm.ppf(1.0 - alpha / 2.0))

    p = float(successes) / float(trials)
    z2 = z * z
    n = float(trials)

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)

