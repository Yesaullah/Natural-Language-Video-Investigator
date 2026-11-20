"""Video-level cluster bootstrap for Natural-Language Video Investigator.

Implements percentile bootstrap intervals with cluster resampling over whole videos,
preserving multiplicity, paired difference evaluation with boundary alignment,
and graceful handling of undefined replicates.
Conforms to video_investigator_plan.md §6.2 and plans/M1-evaluation-foundation.md §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vi.eval.metrics import RateResult
from vi.schema import EvaluationPrediction, QueryLabel, SearchResult


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of video-level bootstrap sampling."""

    point_estimate: Optional[float]
    interval: Tuple[Optional[float], Optional[float]]
    n_resamples: int
    n_valid: int
    n_undefined: int


def _extract_metric_value(val: Any) -> Optional[float]:
    """Helper to extract a float metric value from float, RateResult, or None."""
    if val is None:
        return None
    if isinstance(val, RateResult):
        return val.rate
    if isinstance(val, (int, float)):
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return f_val
    return None


def _extract_prediction_key_and_result(item: Any) -> Tuple[Tuple[str, str], SearchResult]:
    """Extract (video_id, qid) key and SearchResult payload from a prediction record."""
    if isinstance(item, EvaluationPrediction):
        return (item.video_id, item.qid), item.result
    if hasattr(item, "video_id") and hasattr(item, "qid") and hasattr(item, "result"):
        res = item.result
        if isinstance(res, dict):
            res = SearchResult.model_validate(res)
        return (str(item.video_id), str(item.qid)), res
    if isinstance(item, tuple):
        if len(item) == 3:
            res = item[2]
            if isinstance(res, dict):
                res = SearchResult.model_validate(res)
            return (str(item[0]), str(item[1])), res
        if len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 2:
            res = item[1]
            if isinstance(res, dict):
                res = SearchResult.model_validate(res)
            return (str(item[0][0]), str(item[0][1])), res
    if isinstance(item, dict):
        if "video_id" in item and "qid" in item and "result" in item:
            res = item["result"]
            if isinstance(res, dict):
                res = SearchResult.model_validate(res)
            return (str(item["video_id"]), str(item["qid"])), res
    raise TypeError(
        f"Each system's predictions must carry (video_id, qid) at the evaluation boundary. "
        f"Expected EvaluationPrediction, tuple (video_id, qid, SearchResult), or dict, got {type(item)}"
    )


def video_level_bootstrap(
    query_labels: Sequence[QueryLabel],
    predictions: Sequence[Any],
    metric_fn: Callable[[Sequence[QueryLabel], Sequence[SearchResult]], Any],
    n_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Compute percentile bootstrap confidence interval with video-level resampling.

    Whole videos are sampled with replacement. When a video is drawn twice, all
    of its queries are included with multiplicity preserved.
    """
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    # Determine if predictions carry keys
    has_keys = False
    if predictions and not isinstance(predictions[0], SearchResult):
        has_keys = True

    if has_keys:
        pred_dict: Dict[Tuple[str, str], SearchResult] = {}
        for item in predictions:
            k, res = _extract_prediction_key_and_result(item)
            if k in pred_dict:
                raise ValueError(f"Duplicate prediction key {k} in predictions")
            pred_dict[k] = res

        q_keys = set()
        for q in query_labels:
            k = (q.video_id, q.qid)
            if k in q_keys:
                raise ValueError(f"Duplicate query key {k} in query_labels")
            q_keys.add(k)

        if set(pred_dict.keys()) != q_keys:
            raise ValueError(
                f"Prediction keys do not match query label keys. "
                f"Symmetric difference: {set(pred_dict.keys()) ^ q_keys}"
            )
        aligned_results = [pred_dict[(q.video_id, q.qid)] for q in query_labels]
    else:
        if len(query_labels) != len(predictions):
            raise ValueError(
                f"Mismatched lengths: {len(query_labels)} queries vs {len(predictions)} results"
            )
        aligned_results = list(predictions)

    # Compute point estimate on full data
    point_val = _extract_metric_value(metric_fn(query_labels, aligned_results))

    if not query_labels:
        return BootstrapResult(
            point_estimate=point_val,
            interval=(None, None),
            n_resamples=n_resamples,
            n_valid=0,
            n_undefined=n_resamples,
        )

    # Group queries and results by video_id
    unique_videos = sorted(list(set(q.video_id for q in query_labels)))
    video_clusters: Dict[str, List[Tuple[QueryLabel, SearchResult]]] = {
        vid: [] for vid in unique_videos
    }
    for q, res in zip(query_labels, aligned_results):
        video_clusters[q.video_id].append((q, res))

    n_videos = len(unique_videos)
    rng = np.random.default_rng(seed)

    replicates: List[float] = []
    n_undefined = 0

    for _ in range(n_resamples):
        sampled_indices = rng.integers(0, n_videos, size=n_videos)
        resampled_queries: List[QueryLabel] = []
        resampled_results: List[SearchResult] = []

        for idx in sampled_indices:
            vid = unique_videos[idx]
            for q, res in video_clusters[vid]:
                resampled_queries.append(q)
                resampled_results.append(res)

        val = _extract_metric_value(metric_fn(resampled_queries, resampled_results))
        if val is None:
            n_undefined += 1
        else:
            replicates.append(val)

    if not replicates:
        return BootstrapResult(
            point_estimate=point_val,
            interval=(None, None),
            n_resamples=n_resamples,
            n_valid=0,
            n_undefined=n_undefined,
        )

    alpha = 1.0 - confidence
    lower = float(np.percentile(replicates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapResult(
        point_estimate=point_val,
        interval=(lower, upper),
        n_resamples=n_resamples,
        n_valid=len(replicates),
        n_undefined=n_undefined,
    )


def paired_video_level_bootstrap(
    query_labels: Sequence[QueryLabel],
    predictions_a: Sequence[Any],
    predictions_b: Sequence[Any],
    metric_fn: Callable[[Sequence[QueryLabel], Sequence[SearchResult]], Any],
    n_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Compute paired percentile bootstrap interval for difference between two systems.

    Each system's predictions must carry (video_id, qid) at the evaluation boundary.
    Validates key uniqueness within each system, exact key-set equality with query_labels,
    and aligns both systems before computing the point estimate and bootstrap resamples.
    """
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    # Validate query labels uniqueness
    q_keys = set()
    for q in query_labels:
        k = (q.video_id, q.qid)
        if k in q_keys:
            raise ValueError(f"Duplicate query key {k} in query_labels")
        q_keys.add(k)

    # Validate uniqueness and extract predictions for System A
    dict_a: Dict[Tuple[str, str], SearchResult] = {}
    for item in predictions_a:
        k, res = _extract_prediction_key_and_result(item)
        if k in dict_a:
            raise ValueError(f"Duplicate prediction key {k} found in System A predictions")
        dict_a[k] = res

    # Validate uniqueness and extract predictions for System B
    dict_b: Dict[Tuple[str, str], SearchResult] = {}
    for item in predictions_b:
        k, res = _extract_prediction_key_and_result(item)
        if k in dict_b:
            raise ValueError(f"Duplicate prediction key {k} found in System B predictions")
        dict_b[k] = res

    # Exact key-set equality
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())

    if keys_a != q_keys:
        diff_a = keys_a ^ q_keys
        raise ValueError(
            f"System A prediction keys do not match query label keys. "
            f"Symmetric difference: {diff_a}"
        )

    if keys_b != q_keys:
        diff_b = keys_b ^ q_keys
        raise ValueError(
            f"System B prediction keys do not match query label keys. "
            f"Symmetric difference: {diff_b}"
        )

    # Align predictions before both point estimate and bootstrap
    aligned_a = [dict_a[(q.video_id, q.qid)] for q in query_labels]
    aligned_b = [dict_b[(q.video_id, q.qid)] for q in query_labels]

    # Point estimates on aligned data
    val_a = _extract_metric_value(metric_fn(query_labels, aligned_a))
    val_b = _extract_metric_value(metric_fn(query_labels, aligned_b))
    point_diff = (val_a - val_b) if (val_a is not None and val_b is not None) else None

    if not query_labels:
        return BootstrapResult(
            point_estimate=point_diff,
            interval=(None, None),
            n_resamples=n_resamples,
            n_valid=0,
            n_undefined=n_resamples,
        )

    # Group aligned triples by video_id
    unique_videos = sorted(list(set(q.video_id for q in query_labels)))
    clusters: Dict[str, List[Tuple[QueryLabel, SearchResult, SearchResult]]] = {
        vid: [] for vid in unique_videos
    }
    for q, res_a, res_b in zip(query_labels, aligned_a, aligned_b):
        clusters[q.video_id].append((q, res_a, res_b))

    n_videos = len(unique_videos)
    rng = np.random.default_rng(seed)

    replicates: List[float] = []
    n_undefined = 0

    for _ in range(n_resamples):
        sampled_indices = rng.integers(0, n_videos, size=n_videos)
        resampled_q: List[QueryLabel] = []
        resampled_a: List[SearchResult] = []
        resampled_b: List[SearchResult] = []

        for idx in sampled_indices:
            vid = unique_videos[idx]
            for q, res_a, res_b in clusters[vid]:
                resampled_q.append(q)
                resampled_a.append(res_a)
                resampled_b.append(res_b)

        m_a = _extract_metric_value(metric_fn(resampled_q, resampled_a))
        m_b = _extract_metric_value(metric_fn(resampled_q, resampled_b))

        if m_a is None or m_b is None:
            n_undefined += 1
        else:
            replicates.append(m_a - m_b)

    if not replicates:
        return BootstrapResult(
            point_estimate=point_diff,
            interval=(None, None),
            n_resamples=n_resamples,
            n_valid=0,
            n_undefined=n_undefined,
        )

    alpha = 1.0 - confidence
    lower = float(np.percentile(replicates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapResult(
        point_estimate=point_diff,
        interval=(lower, upper),
        n_resamples=n_resamples,
        n_valid=len(replicates),
        n_undefined=n_undefined,
    )
