"""Development evaluation for B0 and B1, with synthetic fixtures by default.

B0 keeps the M3 seeded trial, confidence, ranking and JSON conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
from vi.schema import CandidateWindow, QueryLabel, SearchResult, VideoMetadata
from vi.eval.metrics import (
    RateResult,
    StartTimeSummary,
    balanced_accuracy,
    false_abstain_rate,
    false_answer_rate,
    mean_reciprocal_rank,
    recall_at_k,
    rejection_rate,
    risk_coverage_curve,
    start_time_error,
    top_window_mean_iou,
    wilson_score_interval,
)
from vi.eval.bootstrap import video_level_bootstrap
from vi.eval.validate_labels import (
    build_gold_length_pool,
    load_labels_from_jsonl,
    reject_unsupported_split,
    reject_unsupported_system,
    validate_labels_and_indexes,
)

# ── constants ──────────────────────────────────────────────────────────────────
SUPPORTED_SYSTEMS = ("B0", "B1")
SUPPORTED_SPLITS = ("dev",)

# Default fixture paths (relative to repo root)
_FIXTURE_LABELS: Dict[str, Path] = {
    "dev": Path("data/synthetic/dev_labels.jsonl"),
}
_STUB_INDEX_ROOT = Path("index")
_STUB_VIDEO_IDS = ["stub-spoken", "stub-visual"]

# B0 hyper-parameters
B0_N_TRIALS = 1_000
B0_N_CANDIDATES = 5
B0_CONFIDENCE = 0.5  # single common confidence for all results (no ranking signal)
B0_BOOTSTRAP_SEED = 42
B0_BOOTSTRAP_N = 10_000
B0_GEN_SEED = 0  # outer seed for trial generation

# Wilson z for 95 % CI (pre-computed; also derived inside wilson_score_interval)
_IOU_THRESHOLDS = (0.3, 0.5, 0.7)

# ── helper serialisers ─────────────────────────────────────────────────────────

def _rate(r: RateResult) -> Dict[str, Any]:
    """Serialise a RateResult."""
    return {"hits": r.numerator, "denominator": r.denominator, "rate": r.rate}


def _wilson(hits: int, n: int) -> Dict[str, Any]:
    lo, hi = wilson_score_interval(hits, n)
    return {"lower": lo, "upper": hi}


def _start_summary(s: StartTimeSummary) -> Dict[str, Any]:
    return {
        "median": s.median,
        "q25": s.q25,
        "q75": s.q75,
        "iqr": s.iqr,
        "eligible_count": s.count,
    }


def _boot(br) -> Dict[str, Any]:
    return {
        "point": br.point_estimate,
        "ci_lower": br.interval[0],
        "ci_upper": br.interval[1],
        "n_resamples": br.n_resamples,
        "n_valid": br.n_valid,
        "n_undefined": br.n_undefined,
    }


# ── B0 generator ───────────────────────────────────────────────────────────────

def _generate_b0_trial(
    rng: np.random.Generator,
    duration_s: float,
    length_pool: List[float],
    n_candidates: int,
) -> List[Tuple[float, float]]:
    """Generate n_candidates windows for one trial of one query.

    Algorithm (per brief §B0 definition):
      For each candidate:
        1. Sample a length uniformly WITH replacement from the pooled gold-window
           lengths. Cap at video duration.
        2. Sample a start uniformly from [0, duration - length].
           A full-duration window (length == duration) starts at 0.
    """
    windows: List[Tuple[float, float]] = []
    pool_arr = np.asarray(length_pool, dtype=np.float64)
    for _ in range(n_candidates):
        # Sample length from pool, cap at duration
        idx = int(rng.integers(0, len(pool_arr)))
        length = min(float(pool_arr[idx]), duration_s)
        # Valid start range: [0, duration - length]
        max_start = duration_s - length
        start = float(rng.uniform(0.0, max_start)) if max_start > 0.0 else 0.0
        end = start + length
        # Clamp end to duration to absorb tiny float drift
        end = min(end, duration_s)
        windows.append((start, end))
    return windows


def generate_b0_results(
    labels: Sequence[QueryLabel],
    video_meta: Dict[str, VideoMetadata],
    length_pool: List[float],
    n_trials: int = B0_N_TRIALS,
    n_candidates: int = B0_N_CANDIDATES,
    gen_seed: int = B0_GEN_SEED,
) -> Dict[Tuple[str, str], List[SearchResult]]:
    """Generate B0 results for every query over n_trials trials.

    Returns a dict keyed by (video_id, qid) → list[SearchResult] of length n_trials.

    Stability rules (brief §B0):
    - Each (video_id, qid) pair gets a deterministic sub-RNG derived from gen_seed
      and a stable hash of its key so that reordering labels leaves results identical.
    - The sub-RNG advances through all n_trials × n_candidates windows independently.
    """
    results: Dict[Tuple[str, str], List[SearchResult]] = {}

    for q in labels:
        duration_s = video_meta[q.video_id].duration_s
        # Deterministic per-query seed derived from gen_seed + stable key hash
        key_bytes = json.dumps([gen_seed, q.video_id, q.qid], separators=(",", ":")).encode("utf-8")
        q_seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:16], "big")
        rng = np.random.default_rng(q_seed)

        trial_results: List[SearchResult] = []
        for _trial in range(n_trials):
            windows = _generate_b0_trial(rng, duration_s, length_pool, n_candidates)
            candidates = [
                CandidateWindow(
                    start_s=s,
                    end_s=e,
                    score=float(n_candidates - i),   # preserve random order via descending rank score
                    path_scores={},
                    evidence={},
                )
                for i, (s, e) in enumerate(windows)
            ]
            sr = SearchResult(
                abstained=False,          # B0 always answers
                confidence=B0_CONFIDENCE, # single common confidence
                route="random",
                results=candidates,
                timings_ms={},
            )
            trial_results.append(sr)

        results[(q.video_id, q.qid)] = trial_results

    return results


# ── per-query trial-average helpers ───────────────────────────────────────────

def _query_trial_avg_r_at_k(
    q: QueryLabel,
    trial_results: List[SearchResult],
    k: int,
    iou_threshold: float,
) -> Optional[float]:
    """Average R@K over trials for a single answerable query."""
    if not q.answerable:
        return None
    total = 0.0
    for sr in trial_results:
        candidates = sr.results[:k]
        hit = any(
            _best_iou((c.start_s, c.end_s), q.gold_windows) >= iou_threshold
            for c in candidates
        )
        total += 1.0 if hit else 0.0
    return total / len(trial_results) if trial_results else None


def _best_iou(candidate: Tuple[float, float], gold_windows: List[Tuple[float, float]]) -> float:
    if not gold_windows:
        return 0.0
    s_c, e_c = candidate
    best = 0.0
    for s_g, e_g in gold_windows:
        inter = max(0.0, min(e_c, e_g) - max(s_c, s_g))
        union = (e_c - s_c) + (e_g - s_g) - inter
        if union > 0.0:
            iou = inter / union
            if iou > best:
                best = iou
    return best


def _query_trial_avg_mrr(
    q: QueryLabel,
    trial_results: List[SearchResult],
    iou_threshold: float = 0.5,
    max_rank: int = 5,
) -> Optional[float]:
    if not q.answerable:
        return None
    total = 0.0
    for sr in trial_results:
        for rank, c in enumerate(sr.results[:max_rank], start=1):
            if _best_iou((c.start_s, c.end_s), q.gold_windows) >= iou_threshold:
                total += 1.0 / rank
                break
    return total / len(trial_results) if trial_results else None


def _query_trial_avg_top_iou(
    q: QueryLabel,
    trial_results: List[SearchResult],
) -> Optional[float]:
    if not q.answerable:
        return None
    total = 0.0
    for sr in trial_results:
        if sr.results:
            c = sr.results[0]
            total += _best_iou((c.start_s, c.end_s), q.gold_windows)
    return total / len(trial_results) if trial_results else None


# ── aggregate helpers over a set of queries + per-query trial averages ─────────

def _aggregate_r_at_k(
    labels: Sequence[QueryLabel],
    per_query_avg: Dict[Tuple[str, str], float],
) -> RateResult:
    """R@K from query-level trial averages.

    The mean over answerable queries of their per-query trial-average hit rate.
    Numerator = sum of per-query averages; denominator = number of answerable queries.
    For bootstrap we use this same function over resampled query sets.
    """
    hits_sum = 0.0
    n_ans = 0
    for q in labels:
        if not q.answerable:
            continue
        v = per_query_avg.get((q.video_id, q.qid))
        if v is not None:
            hits_sum += v
            n_ans += 1
    if n_ans == 0:
        return RateResult(numerator=0, denominator=0, rate=None)
    # Return as a RateResult with float numerator and integer denominator
    # rate = mean of per-query averages
    return RateResult(
        numerator=round(hits_sum),   # integer approximation for Wilson
        denominator=n_ans,
        rate=hits_sum / n_ans,
    )


def _first_trial_results(
    labels: Sequence[QueryLabel],
    b0_dict: Dict[Tuple[str, str], List[SearchResult]],
) -> List[SearchResult]:
    """Extract the first trial's SearchResult for each label (in label order)."""
    return [b0_dict[(q.video_id, q.qid)][0] for q in labels]


# ── start-time error over first trial (trial-invariant structure) ──────────────

def run_b0_evaluation(
    labels: List[QueryLabel],
    video_meta: Dict[str, VideoMetadata],
    length_pool: List[float],
    n_trials: int = B0_N_TRIALS,
    bootstrap_n: int = B0_BOOTSTRAP_N,
    gen_seed: int = B0_GEN_SEED,
    bootstrap_seed: int = B0_BOOTSTRAP_SEED,
    *, details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the B0 baseline and return a JSON-serialisable metrics dict."""

    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    t_start = time.monotonic()
    labels = sorted(labels, key=lambda q: (q.video_id, q.qid))

    # ── 1. Generate B0 predictions ────────────────────────────────────────────
    b0_dict = generate_b0_results(
        labels, video_meta, length_pool,
        n_trials=n_trials, gen_seed=gen_seed,
    )

    if b0_dict.keys() != {(q.video_id, q.qid) for q in labels} or len(b0_dict) != len(labels):
        raise ValueError("B0 prediction keys do not match unique evaluation query keys")

    # ── 2. First-trial predictions (for trial-invariant metrics) ──────────────
    first_trial = _first_trial_results(labels, b0_dict)

    # ── 3. Per-query trial averages for retrieval metrics ─────────────────────
    # Additive metrics: cache query-level trial averages for bootstrap
    pq_r1_03: Dict[Tuple[str, str], float] = {}
    pq_r5_03: Dict[Tuple[str, str], float] = {}
    pq_r1_05: Dict[Tuple[str, str], float] = {}
    pq_r5_05: Dict[Tuple[str, str], float] = {}
    pq_r1_07: Dict[Tuple[str, str], float] = {}
    pq_r5_07: Dict[Tuple[str, str], float] = {}
    pq_mrr: Dict[Tuple[str, str], float] = {}
    pq_top_iou: Dict[Tuple[str, str], float] = {}

    for q in labels:
        key = (q.video_id, q.qid)
        trials = b0_dict[key]
        if q.answerable:
            pq_r1_03[key] = _query_trial_avg_r_at_k(q, trials, 1, 0.3) or 0.0
            pq_r5_03[key] = _query_trial_avg_r_at_k(q, trials, 5, 0.3) or 0.0
            pq_r1_05[key] = _query_trial_avg_r_at_k(q, trials, 1, 0.5) or 0.0
            pq_r5_05[key] = _query_trial_avg_r_at_k(q, trials, 5, 0.5) or 0.0
            pq_r1_07[key] = _query_trial_avg_r_at_k(q, trials, 1, 0.7) or 0.0
            pq_r5_07[key] = _query_trial_avg_r_at_k(q, trials, 5, 0.7) or 0.0
            pq_mrr[key] = _query_trial_avg_mrr(q, trials) or 0.0
            pq_top_iou[key] = _query_trial_avg_top_iou(q, trials) or 0.0

    # ── 4. Compute mean metrics over labels ───────────────────────────────────
    # Retrieve metrics (trial-averaged)
    def _mean_r(pq_dict: Dict) -> Tuple[float, int]:
        vals = [pq_dict[k] for k in pq_dict]
        return (sum(vals) / len(vals) if vals else 0.0), len(vals)

    r1_03_mean, n_ans = _mean_r(pq_r1_03)
    r5_03_mean, _     = _mean_r(pq_r5_03)
    r1_05_mean, _     = _mean_r(pq_r1_05)
    r5_05_mean, _     = _mean_r(pq_r5_05)
    r1_07_mean, _     = _mean_r(pq_r1_07)
    r5_07_mean, _     = _mean_r(pq_r5_07)
    mrr_mean, _       = _mean_r(pq_mrr)
    top_iou_mean, _   = _mean_r(pq_top_iou)

    # Trial-invariant metrics (from first trial)
    n_queries = len(labels)
    n_answerable = sum(1 for q in labels if q.answerable)
    n_unanswerable = n_queries - n_answerable

    rej = rejection_rate(labels, first_trial)
    fa  = false_answer_rate(labels, first_trial)
    fas = false_abstain_rate(labels, first_trial)
    bal = balanced_accuracy(labels, first_trial)
    ste = start_time_error(labels, first_trial)

    # Counts for first-trial Wilson intervals
    # B0 never abstains → rejection = 0, false-answer = all unanswerable
    rej_first_hits = rej.numerator
    rej_first_denom = rej.denominator
    fa_first_hits = fa.numerator
    fa_first_denom = fa.denominator
    fas_first_hits = fas.numerator
    fas_first_denom = fas.denominator

    # ── 5. Bootstrap intervals for headline retrieval rates ───────────────────
    # Build synthetic SearchResult lists from per-query averages for bootstrap
    # The bootstrap resamples VIDEOS; each video's queries are taken with their
    # per-query trial averages as a single-probability SearchResult.
    # To satisfy video_level_bootstrap's interface we pass the actual first-trial
    # results but replace the metric_fn with one that uses per-query averages.

    def _make_r_metric(pq_avgs: Dict[Tuple[str, str], float]):
        """Factory: a metric_fn over labels + SearchResults that ignores results
        and uses pre-computed per-query averages. Bootstrap passes all labels for
        a resampled set of videos, so we look up each query's pre-cached average."""
        def metric_fn(qs, _results):
            vals = [pq_avgs[(q.video_id, q.qid)] for q in qs
                    if q.answerable and (q.video_id, q.qid) in pq_avgs]
            if not vals:
                return None
            return sum(vals) / len(vals)
        return metric_fn

    boot_r1_03 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r1_03),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_r5_03 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r5_03),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_r1_05 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r1_05),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_r5_05 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r5_05),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_r1_07 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r1_07),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_r5_07 = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_r5_07),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_mrr   = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_mrr),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)
    boot_tiou  = video_level_bootstrap(labels, first_trial, _make_r_metric(pq_top_iou),
                                        n_resamples=bootstrap_n, seed=bootstrap_seed)

    # ── 6. First-trial retrieval rates with Wilson intervals ──────────────────
    first_rates = {}
    for k in (1, 5):
        for threshold in _IOU_THRESHOLDS:
            rate = recall_at_k(labels, first_trial, k=k, iou_threshold=threshold)
            first_rates[f"R@{k}_iou{threshold}"] = {
                **_rate(rate),
                "wilson_95": _wilson(rate.numerator, rate.denominator),
            }

    # B0 always answers with a common confidence, so each trial has one
    # full-coverage step. Its area equals its risk at that step.
    risks = []
    eligible = []
    for idx in range(n_trials):
        predictions = [b0_dict[(q.video_id, q.qid)][idx] for q in labels]
        risks.append(risk_coverage_curve(labels, predictions)["area"])
        summary = start_time_error(labels, predictions)
        if summary.count:
            eligible.append(summary)
    mean_risk = float(np.mean(risks))
    start_err_summary = {
        field: float(np.mean([getattr(summary, field) for summary in eligible])) if eligible else None
        for field in ("q25", "median", "q75", "iqr")
    }
    start_err_summary.update(
        eligible_count=sum(summary.count for summary in eligible),
        n_trials_with_eligible=len(eligible),
        note="Means of per-trial summaries over trials with eligible overlaps.",
    )
    if not eligible:
        start_err_summary["reason"] = "No trial had a top window overlapping a gold window."

    unique_videos = sorted({q.video_id for q in labels})

    elapsed_s = time.monotonic() - t_start

    # ── 9. Build the report dict ──────────────────────────────────────────────
    report: Dict[str, Any] = {
        # ── metadata ───────────────────────────────────────────────────────────
        "dataset": "synthetic",
        "split": "dev",
        "synthetic": True,
        "fixture_mode": "synthetic",
        "fixture_note": (
            "All labels and indexes are deterministically generated synthetic fixtures. "
            "These numbers do not represent real retrieval performance."
        ),
        "systems": ["B0"],
        "b0_conventions": {
            "always_answers": True,
            "abstention": False,
            "confidence": B0_CONFIDENCE,
            "confidence_note": "Single common confidence for all results; no ranking signal.",
            "candidate_order": "random_generation_order",
            "raw_equals_abstention_aware": True,
        },
        # ── seeds and counts ───────────────────────────────────────────────────
        "seeds": {
            "generation": gen_seed,
            "bootstrap": bootstrap_seed,
        },
        "trial_count": n_trials,
        "bootstrap_count": bootstrap_n,
        "candidates_per_trial": B0_N_CANDIDATES,
        # ── coverage ──────────────────────────────────────────────────────────
        "video_count": len(unique_videos),
        "video_ids": unique_videos,
        "query_count": n_queries,
        "answerable_count": n_answerable,
        "unanswerable_count": n_unanswerable,
        "answerability_mix": {
            "answerable_fraction": n_answerable / n_queries if n_queries else None,
            "unanswerable_fraction": n_unanswerable / n_queries if n_queries else None,
        },
        # ── metric definitions ─────────────────────────────────────────────────
        "metric_definitions": {
            "R@K_iou_T": "Share of answerable queries with a top-K candidate IoU >= T. "
                         "Trial-averaged, then averaged over answerable queries.",
            "MRR": "Mean of 1/rank of first candidate with IoU >= 0.5, over answerable queries. Trial-averaged.",
            "top_window_mean_iou": "Mean IoU of the top candidate over answerable queries. Trial-averaged.",
            "start_time_error": "Absolute difference between predicted and best-overlapping gold start, "
                                "for top candidates that overlap any gold window (IoU > 0).",
            "rejection_accuracy": "Share of unanswerable queries on which the system abstains.",
            "false_answer_rate": "Share of unanswerable queries on which the system returns a result.",
            "false_abstain_rate": "Share of answerable queries on which the system abstains.",
            "balanced_accuracy": "Mean of rejection_accuracy and (1 - false_abstain_rate).",
            "risk_coverage_area": "Area under the risk-coverage curve (sum of coverage-increment × risk "
                                  "at right endpoint). Lower is better.",
            "bootstrap_interval": "Percentile 95% CI from video-level resampling (whole videos drawn "
                                  "with replacement, all queries retained, multiplicity preserved).",
            "wilson_interval": "Two-sided 95% Wilson score interval without continuity correction.",
        },
        # ── retrieval metrics ─────────────────────────────────────────────────
        "retrieval": {
            "trial_mean": {
                "R@1_iou0.3": r1_03_mean,
                "R@5_iou0.3": r5_03_mean,
                "R@1_iou0.5": r1_05_mean,
                "R@5_iou0.5": r5_05_mean,
                "R@1_iou0.7": r1_07_mean,
                "R@5_iou0.7": r5_07_mean,
                "MRR": mrr_mean,
                "top_window_mean_iou": top_iou_mean,
                "answerable_query_count": n_ans,
            },
            "trial_mean_cluster_bootstrap_95": {
                "R@1_iou0.3": _boot(boot_r1_03),
                "R@5_iou0.3": _boot(boot_r5_03),
                "R@1_iou0.5": _boot(boot_r1_05),
                "R@5_iou0.5": _boot(boot_r5_05),
                "R@1_iou0.7": _boot(boot_r1_07),
                "R@5_iou0.7": _boot(boot_r5_07),
                "MRR": _boot(boot_mrr),
                "top_window_mean_iou": _boot(boot_tiou),
            },
            "first_trial": first_rates,
        },
        # ── abstention metrics ────────────────────────────────────────────────
        # B0 never abstains, so rejection = 0, false_answer = 1, false_abstain = 0,
        # balanced_accuracy = 0.5.  These are trial-invariant (same for every trial).
        "abstention": {
            "note": "B0 never abstains. Rejection=0, false_answer=1, false_abstain=0, balanced_accuracy=0.5.",
            "rejection_accuracy": {
                **_rate(rej),
                "wilson_95": _wilson(rej_first_hits, rej_first_denom),
            },
            "false_answer_rate": {
                **_rate(fa),
                "wilson_95": _wilson(fa_first_hits, fa_first_denom),
            },
            "false_abstain_rate": {
                **_rate(fas),
                "wilson_95": _wilson(fas_first_hits, fas_first_denom),
            },
            "balanced_accuracy": bal,
        },
        # ── risk-coverage ─────────────────────────────────────────────────────
        "risk_coverage": {
            "area": mean_risk,
            "curve": [{"threshold": B0_CONFIDENCE, "coverage": 1.0, "risk": mean_risk}],
            "note": (
                "B0 uses a single common confidence (0.5) for all queries. "
                "All queries are grouped into one threshold step and risk is averaged over trials. "
                "Area = coverage_increment × risk. With uniform confidence, area = risk at full coverage."
            ),
        },
        # ── start-time error ──────────────────────────────────────────────────
        "start_time_error_all_trials": start_err_summary,
        "start_time_error_first_trial": _start_summary(ste),
        # ── routing / GPU (not applicable for B0) ─────────────────────────────
        "routing_scores": None,
        "routing_scores_note": "Not applicable: B0 is a random baseline with no routing model.",
        "gpu_measurements": None,
        "gpu_measurements_note": "Not applicable: B0 runs on CPU only. GPU measurements belong to M4+.",
        # ── subgroup counts ───────────────────────────────────────────────────
        "subgroup_counts": {
            "per_video": {
                vid: {
                    "answerable": sum(1 for q in labels if q.video_id == vid and q.answerable),
                    "unanswerable": sum(1 for q in labels if q.video_id == vid and not q.answerable),
                }
                for vid in unique_videos
            },
        },
        # ── timing (CPU baseline, separate from reproducible score fields) ────
        "cpu_timing": {
            "elapsed_s": elapsed_s,
            "note": "Wall-clock time for B0 generation + evaluation on CPU. Not a reproducible score field.",
        },
    }

    if details is not None:
        details.update(
            query_keys=sorted(b0_dict),
            first_trial=dict(zip(((q.video_id, q.qid) for q in labels), first_trial, strict=True)),
            trial_means={
                "R@1_iou0.3": pq_r1_03, "R@1_iou0.5": pq_r1_05, "R@1_iou0.7": pq_r1_07,
                "R@5_iou0.3": pq_r5_03, "R@5_iou0.5": pq_r5_05, "R@5_iou0.7": pq_r5_07,
                "MRR": pq_mrr, "top_window_mean_iou": pq_top_iou,
            },
        )
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development evaluation harness.  Default run is synthetic (stub indexes + synthetic labels).\n"
            "Example: python -m vi.eval.run_eval --split dev --systems B0"
        )
    )
    parser.add_argument(
        "--split", default="dev",
        help="Evaluation split (only 'dev' is supported; default: dev).",
    )
    parser.add_argument(
        "--systems", default="B0",
        help="Comma-separated list of systems to evaluate (B0, B1, or B0,B1; default: B0).",
    )
    parser.add_argument(
        "--index-root", type=Path,
        help=f"Root directory containing per-video index folders (default: {_STUB_INDEX_ROOT}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for JSON output (default: results/).",
    )
    parser.add_argument(
        "--label-file", type=Path, default=None,
        help="Override path to the label JSONL fixture (default: data/synthetic/dev_labels.jsonl).",
    )
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--manifest", type=Path, help="M2 JSON split manifest for real development.")
    parser.add_argument("--b1-settings", type=Path, help="Fitted JSON settings required to score real B1.")
    parser.add_argument(
        "--n-trials", type=int, default=B0_N_TRIALS,
        help=f"Number of B0 trials per query (default: {B0_N_TRIALS}).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=B0_BOOTSTRAP_N,
        help=f"Number of bootstrap resamples (default: {B0_BOOTSTRAP_N}).",
    )
    parser.add_argument(
        "--gen-seed", type=int, default=B0_GEN_SEED,
        help=f"Seed for B0 window generation (default: {B0_GEN_SEED}).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=B0_BOOTSTRAP_SEED,
        help=f"Seed for bootstrap resampling (default: {B0_BOOTSTRAP_SEED}).",
    )
    return parser.parse_args(argv)


def _write_report(output: Path, report: dict) -> None:
    if os.name == "nt":
        # Use extended-length paths for directory creation, temp files and replacement.
        absolute = os.path.abspath(output)
        if absolute.startswith(("\\\\?\\", "\\\\.\\")):
            output = Path(absolute)
        elif absolute.startswith("\\\\"):
            output = Path("\\\\?\\UNC\\" + absolute[2:])
        else:
            output = Path("\\\\?\\" + absolute)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".json.tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
        temp_path.replace(output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    split = args.split.strip().lower()
    reject_unsupported_split(split)
    systems = sorted(s.strip() for s in args.systems.split(","))
    for system in systems:
        reject_unsupported_system(system, SUPPORTED_SYSTEMS)
    if len(systems) != len(set(systems)):
        raise ValueError("Duplicate evaluation systems")
    if args.n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if "B0" in systems and args.n_trials < 1:
        raise ValueError("n_trials must be positive")

    real = args.mode == "real"
    manifest_hash = None
    fitted = None
    if real:
        from vi.manifest import verify_manifest_integrity
        from vi.schema import SplitManifest

        if not all((args.manifest, args.label_file, args.index_root)):
            raise ValueError("Real development requires --manifest, --label-file and --index-root")
        manifest_bytes = args.manifest.read_bytes()
        manifest = SplitManifest.model_validate_json(manifest_bytes)
        dataset = manifest.dataset
        if dataset not in {"team", "charades-sta"}:
            raise ValueError(f"Unsupported development dataset {dataset!r}")
        verify_manifest_integrity(manifest)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        video_ids = sorted(r.video_id for r in manifest.records if r.split == split)
        index_root = args.index_root
        label_path = args.label_file
    else:
        if args.manifest or args.b1_settings:
            raise ValueError("--manifest and --b1-settings require explicit --mode real")
        dataset = "synthetic"
        video_ids = _STUB_VIDEO_IDS
        index_root = args.index_root or _STUB_INDEX_ROOT
        label_path = args.label_file or _FIXTURE_LABELS[split]

    index_dirs = {video: index_root / video for video in video_ids}
    if not real:
        for directory in index_dirs.values():
            if not directory.exists():
                raise FileNotFoundError(
                    f"Index directory not found: {directory}. Run 'python scripts/make_stub_index.py' first."
                )
    labels = sorted(load_labels_from_jsonl(label_path), key=lambda q: (q.video_id, q.qid))
    video_meta = validate_labels_and_indexes(labels, index_dirs)
    expected_split = split if real else "synthetic"
    for video, metadata in video_meta.items():
        if (metadata.dataset, metadata.split) != (dataset, expected_split):
            raise ValueError(f"Index dataset or split mismatch for {video}")

    applicability = {}
    applicable = True
    report_name = f"{split}_{','.join(systems)}"
    if "B1" in systems:
        from vi.eval.transcript_eval import (
            VideoDraws, b0_subgroups, charades_applicability, load_b1_indexes, run_b1, spoken_comparison,
        )

        loaded_indexes, index_load_seconds = load_b1_indexes(index_dirs, video_meta, synthetic=not real)
        if real and dataset == "charades-sta":
            applicability = charades_applicability(loaded_indexes)
            applicable = applicability["applicable"]
        if real and applicable:
            from vi.search.settings import RealSettings

            if not args.b1_settings:
                raise ValueError("Real B1 development requires --b1-settings")
            fitted = RealSettings.model_validate_json(args.b1_settings.read_bytes())
            fitted.validate_fit()
            if fitted.dataset != dataset:
                raise ValueError("B1 fitted settings dataset does not match the manifest")
            canonical = json.dumps(fitted.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
            report_name += f"_{fitted.search.variant}_{digest}"
        elif not applicable:
            report_name += "_inapplicable"

    b0_details = {} if systems == ["B0", "B1"] else None
    if "B0" in systems:
        b0 = run_b0_evaluation(
            labels, video_meta, build_gold_length_pool(labels),
            n_trials=args.n_trials, bootstrap_n=args.n_bootstrap,
            gen_seed=args.gen_seed, bootstrap_seed=args.bootstrap_seed, details=b0_details,
        )
        b0["n_candidates_per_trial"] = B0_N_CANDIDATES
        if real:
            b0.update(dataset=dataset, synthetic=False, fixture_mode=None)
            del b0["fixture_note"]

    if "B1" in systems:
        if applicable or b0_details is not None:
            draws = VideoDraws(labels, args.n_bootstrap, args.bootstrap_seed)
        b1 = dict(applicability)
        if applicable:
            metrics, predictions = run_b1(
                labels, loaded_indexes, fitted, draws, args.bootstrap_seed, index_load_seconds=index_load_seconds,
            )
            b1.update(metrics)
        answerable = sum(q.answerable for q in labels)
        run_metadata = {
            "dataset": dataset, "split": split, "synthetic": not real,
            "fixture_mode": None if real else "synthetic", "systems": systems,
            "video_ids": video_ids, "video_count": len(video_ids), "query_count": len(labels),
            "answerable_count": answerable, "unanswerable_count": len(labels) - answerable,
            "query_keys": [(q.video_id, q.qid) for q in labels],
            "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "manifest_sha256": manifest_hash,
            "seeds": {"bootstrap": args.bootstrap_seed},
            "bootstrap_count": args.n_bootstrap,
            "bootstrap_population": (
                "Sample all selected dev videos with replacement. Include each sampled video's "
                "eligible queries with multiplicity. Weight queries equally. Draws without "
                "eligible queries are undefined. Variation from rebuilding indexes is unmeasured."
            ),
        }
        if not real:
            run_metadata["fixture_note"] = "Synthetic labels and embeddings check integration only."
        if b0_details is None:
            report = {**run_metadata, **b1}
        else:
            report = {
                **run_metadata, "system_reports": {"B0": b0, "B1": b1},
                "B0_subgroups": b0_subgroups(labels, b0_details, draws),
            }
            if applicable:
                if b0_details["query_keys"] != b1["query_keys"]:
                    raise ValueError("B0 and B1 query keys differ")
                comparison = spoken_comparison(
                    labels, b0_details["trial_means"]["R@1_iou0.5"], predictions, draws,
                )
                comparison["B1_variant"] = b1["settings"]["variant"]
                report["spoken_comparison"] = comparison
    else:
        report = b0
        if real:
            report["manifest_sha256"] = manifest_hash

    output_dir = args.output_dir / dataset if real else args.output_dir
    output_path = output_dir / f"{report_name}.json"
    _write_report(output_path, report)
    print(f"[run_eval] Report written to {output_path}")
    print(f"[run_eval] Videos: {report['video_count']}, Queries: {report['query_count']} "
          f"(answerable={report['answerable_count']}, unanswerable={report['unanswerable_count']})")
    if "B0" in systems:
        print(f"[run_eval] B0 R@1 IoU0.5 (trial mean): {b0['retrieval']['trial_mean']['R@1_iou0.5']:.4f}")
    if "B1" in systems:
        if applicable:
            rate = b1["retrieval"]["raw"]["R@1_iou0.5"]["rate"]
            print(f"[run_eval] B1 raw R@1 IoU0.5: {rate}")
        else:
            print(f"[run_eval] B1 inapplicable. {b1['reason']}")


if __name__ == "__main__":
    main()
