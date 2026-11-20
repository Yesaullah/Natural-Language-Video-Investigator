"""Tests for video-level cluster bootstrap and paired comparisons."""

import numpy as np
import pytest

from vi.eval.bootstrap import paired_video_level_bootstrap, video_level_bootstrap
from vi.eval.metrics import RateResult, recall_at_k, rejection_rate
from vi.schema import CandidateWindow, EvaluationPrediction, QueryLabel, SearchResult


def test_constant_metric_and_reproducible_seed() -> None:
    # Case: Constant metric, repeated seed, identical paired systems
    # Expected: Zero-width interval, reproducible draws, paired difference always zero
    q1 = QueryLabel(qid="q1", video_id="v1", query="query 1", answerable=True, gold_windows=[[0.0, 5.0]])
    q2 = QueryLabel(qid="q2", video_id="v2", query="query 2", answerable=True, gold_windows=[[0.0, 5.0]])

    cand = CandidateWindow(start_s=0.0, end_s=5.0, score=1.0)
    res1 = SearchResult(abstained=False, confidence=0.9, route="visual", results=[cand])
    res2 = SearchResult(abstained=False, confidence=0.9, route="visual", results=[cand])

    pred1 = EvaluationPrediction(video_id="v1", qid="q1", result=res1)
    pred2 = EvaluationPrediction(video_id="v2", qid="q2", result=res2)

    # Single system bootstrap on a metric that is always 1.0 (both videos have hit)
    res_b1 = video_level_bootstrap(
        [q1, q2],
        [pred1, pred2],
        metric_fn=lambda q, r: recall_at_k(q, r, k=1, iou_threshold=0.5),
        n_resamples=1000,
        seed=123,
    )
    assert res_b1.interval == (1.0, 1.0)
    assert res_b1.point_estimate == 1.0
    assert res_b1.n_valid == 1000
    assert res_b1.n_undefined == 0

    # Repeat run with identical seed gives exact same result
    res_b2 = video_level_bootstrap(
        [q1, q2],
        [pred1, pred2],
        metric_fn=lambda q, r: recall_at_k(q, r, k=1, iou_threshold=0.5),
        n_resamples=1000,
        seed=123,
    )
    assert res_b1.interval == res_b2.interval

    # Paired systems: identical systems
    paired_res = paired_video_level_bootstrap(
        [q1, q2],
        [pred1, pred2],
        [pred1, pred2],
        metric_fn=lambda q, r: recall_at_k(q, r, k=1, iou_threshold=0.5),
        n_resamples=1000,
        seed=456,
    )
    assert paired_res.interval == (0.0, 0.0)
    assert paired_res.point_estimate == 0.0
    assert paired_res.n_valid == 1000
    assert paired_res.n_undefined == 0


def test_paired_bootstrap_reordered_identical_predictions() -> None:
    # Regression test: predictions provided in different orders must be aligned
    # by (video_id, qid) before point estimate and bootstrap.
    q1 = QueryLabel(qid="q1", video_id="v1", query="query 1", answerable=True, gold_windows=[[0.0, 5.0]])
    q2 = QueryLabel(qid="q2", video_id="v1", query="query 2", answerable=True, gold_windows=[[10.0, 15.0]])
    q3 = QueryLabel(qid="q3", video_id="v2", query="query 3", answerable=True, gold_windows=[[0.0, 10.0]])

    cand_hit = CandidateWindow(start_s=0.0, end_s=5.0, score=2.0)
    cand_miss = CandidateWindow(start_s=50.0, end_s=60.0, score=1.0)

    res_hit = SearchResult(abstained=False, confidence=0.9, route="visual", results=[cand_hit])
    res_miss = SearchResult(abstained=False, confidence=0.3, route="visual", results=[cand_miss])

    p_a1 = EvaluationPrediction(video_id="v1", qid="q1", result=res_hit)
    p_a2 = EvaluationPrediction(video_id="v1", qid="q2", result=res_miss)
    p_a3 = EvaluationPrediction(video_id="v2", qid="q3", result=res_hit)

    preds_a = [p_a1, p_a2, p_a3]
    # Reverse the order for system B
    preds_b = [p_a3, p_a1, p_a2]

    paired_res = paired_video_level_bootstrap(
        [q1, q2, q3],
        preds_a,
        preds_b,
        metric_fn=lambda q, r: recall_at_k(q, r, k=1, iou_threshold=0.5),
        n_resamples=1000,
        seed=777,
    )

    assert paired_res.point_estimate == 0.0
    assert paired_res.interval == (0.0, 0.0)
    assert paired_res.n_valid == 1000
    assert paired_res.n_undefined == 0


def test_paired_bootstrap_duplicate_keys() -> None:
    # Regression test: duplicate keys within a system must be rejected
    q1 = QueryLabel(qid="q1", video_id="v1", query="query 1", answerable=True, gold_windows=[[0.0, 5.0]])
    q2 = QueryLabel(qid="q2", video_id="v1", query="query 2", answerable=True, gold_windows=[[10.0, 15.0]])

    res = SearchResult(abstained=False, confidence=0.5, route="visual")
    p1 = EvaluationPrediction(video_id="v1", qid="q1", result=res)
    p2 = EvaluationPrediction(video_id="v1", qid="q2", result=res)
    # Duplicate (v1, q1) in System A
    preds_a_dup = [p1, p1]
    preds_b = [p1, p2]

    with pytest.raises(ValueError, match="Duplicate prediction key"):
        paired_video_level_bootstrap(
            [q1, q2],
            preds_a_dup,
            preds_b,
            metric_fn=lambda q, r: 1.0,
        )


def test_paired_bootstrap_same_length_mismatched_keys() -> None:
    # Regression test: same length, but mismatched (video_id, qid) keys must be rejected
    q1 = QueryLabel(qid="q1", video_id="v1", query="query 1", answerable=True, gold_windows=[[0.0, 5.0]])
    q2 = QueryLabel(qid="q2", video_id="v1", query="query 2", answerable=True, gold_windows=[[10.0, 15.0]])

    res = SearchResult(abstained=False, confidence=0.5, route="visual")
    p1 = EvaluationPrediction(video_id="v1", qid="q1", result=res)
    p2 = EvaluationPrediction(video_id="v1", qid="q2", result=res)
    p3 = EvaluationPrediction(video_id="v1", qid="q3", result=res)  # Mismatched key q3

    preds_a = [p1, p2]  # keys: (v1, q1), (v1, q2)
    preds_b = [p1, p3]  # keys: (v1, q1), (v1, q3) -- same length 2, but mismatched!

    with pytest.raises(ValueError, match="keys do not match query label keys"):
        paired_video_level_bootstrap(
            [q1, q2],
            preds_a,
            preds_b,
            metric_fn=lambda q, r: 1.0,
        )


def test_unequal_size_two_video_fixture_and_cluster_resampling() -> None:
    # Fixture with unequal video cluster sizes:
    # Video 1: 1 query (hit)
    # Video 2: 3 queries (all misses)
    # If Video 1 is drawn twice: 2 queries in resample, 2 hits -> R@1 = 1.0
    # If Video 2 is drawn twice: 6 queries in resample, 0 hits -> R@1 = 0.0
    # If both drawn once: 4 queries in resample, 1 hit -> R@1 = 0.25
    q_v1 = [
        QueryLabel(qid="v1_q1", video_id="v1", query="v1 q1", answerable=True, gold_windows=[[0.0, 5.0]]),
    ]
    res_v1 = [
        SearchResult(
            abstained=False,
            confidence=0.9,
            route="visual",
            results=[CandidateWindow(start_s=0.0, end_s=5.0, score=2.0)],  # hit
        ),
    ]

    q_v2 = [
        QueryLabel(qid=f"v2_q{i}", video_id="v2", query=f"v2 q{i}", answerable=True, gold_windows=[[0.0, 5.0]])
        for i in range(1, 4)
    ]
    res_v2 = [
        SearchResult(
            abstained=False,
            confidence=0.5,
            route="visual",
            results=[CandidateWindow(start_s=20.0, end_s=25.0, score=1.0)],  # miss
        )
        for _ in range(3)
    ]

    queries = q_v1 + q_v2
    results = res_v1 + res_v2

    # Track distinct sample metric values observed during cluster bootstrap
    observed_values = set()

    def tracking_metric(qs: list[QueryLabel], rs: list[SearchResult]) -> RateResult:
        r1 = recall_at_k(qs, rs, k=1, iou_threshold=0.5)
        if r1.rate is not None:
            observed_values.add(round(r1.rate, 4))
        return r1

    b_res = video_level_bootstrap(
        queries,
        results,
        metric_fn=tracking_metric,
        n_resamples=500,
        seed=42,
    )
    assert b_res.point_estimate == 0.25  # 1 hit out of 4 queries on full dataset
    # Must observe draws where v1 is sampled twice (1.0), both sampled (0.25), and v2 sampled twice (0.0)
    assert {0.0, 0.25, 1.0}.issubset(observed_values)
    assert b_res.interval[0] is not None and b_res.interval[1] is not None
    assert b_res.interval[0] <= b_res.point_estimate <= b_res.interval[1]


def test_undefined_replicates_handling() -> None:
    # Fixture: Video 1 has only answerable queries. Video 2 has 1 unanswerable query.
    # Rejection rate is undefined if Video 2 is not drawn (i.e. Video 1 drawn twice).
    q1 = QueryLabel(qid="q1", video_id="v1", query="q1", answerable=True, gold_windows=[[0.0, 5.0]])
    q2 = QueryLabel(qid="q2", video_id="v2", query="q2", answerable=False, gold_windows=[], neg_type="near_miss")

    res1 = SearchResult(abstained=False, confidence=0.8, route="visual")
    res2 = SearchResult(abstained=True, confidence=0.2, route="visual")

    b_res = video_level_bootstrap(
        [q1, q2],
        [res1, res2],
        metric_fn=rejection_rate,
        n_resamples=1000,
        seed=999,
    )

    # Some resamples will draw Video 1 twice -> 0 unanswerables -> undefined replicate
    assert b_res.n_undefined > 0
    assert b_res.n_valid > 0
    assert b_res.n_valid + b_res.n_undefined == 1000
    assert b_res.interval != (None, None)

    # If all replicates are undefined (e.g. no unanswerables anywhere):
    b_all_undef = video_level_bootstrap(
        [q1],
        [res1],
        metric_fn=rejection_rate,
        n_resamples=100,
        seed=1,
    )
    assert b_all_undef.n_valid == 0
    assert b_all_undef.n_undefined == 100
    assert b_all_undef.interval == (None, None)
