"""Tests for pure metric functions against hand-calculated expectations."""

import pytest

from vi.eval.metrics import (
    balanced_accuracy,
    best_iou_with_gold,
    false_abstain_rate,
    false_answer_rate,
    mean_reciprocal_rank,
    recall_at_k,
    rejection_rate,
    risk_coverage_curve,
    start_time_error,
    temporal_iou,
    top_window_mean_iou,
    wilson_score_interval,
)
from vi.schema import CandidateWindow, QueryLabel, SearchResult


def test_hand_check_iou_overlapping() -> None:
    # Case: Windows [0,10) and [5,15) -> IoU = 1/3
    iou = temporal_iou((0.0, 10.0), (5.0, 15.0))
    assert abs(iou - 1.0 / 3.0) < 1e-9


def test_hand_check_iou_disjoint_abutting() -> None:
    # Case: Windows [0,10) and [10,15) -> IoU = 0
    iou = temporal_iou((0.0, 10.0), (10.0, 15.0))
    assert iou == 0.0


def test_hand_check_first_hit_at_rank_two() -> None:
    # Case: First hit is rank two at exactly IoU 0.5 -> R@1 = 0, R@5 = 1, reciprocal rank = 0.5
    query = QueryLabel(
        qid="q1",
        video_id="v1",
        query="person running",
        answerable=True,
        gold_windows=[[10.0, 20.0]],
    )
    # Rank 1 has IoU 0.2 < 0.5; Rank 2 has IoU exactly 0.5 with [10, 20]:
    # e.g. candidate [10, 15] with gold [10, 20] has intersection 5, union 10 -> IoU 0.5
    cand1 = CandidateWindow(start_s=0.0, end_s=5.0, score=2.0)  # IoU = 0.0
    cand2 = CandidateWindow(start_s=10.0, end_s=15.0, score=1.8)  # IoU = 5/10 = 0.5
    cand3 = CandidateWindow(start_s=50.0, end_s=60.0, score=1.5)  # IoU = 0.0

    res = SearchResult(
        abstained=False,
        confidence=0.8,
        route="temporal",
        results=[cand1, cand2, cand3],
    )

    r1 = recall_at_k([query], [res], k=1, iou_threshold=0.5)
    assert r1.rate == 0.0
    assert r1.numerator == 0
    assert r1.denominator == 1

    r5 = recall_at_k([query], [res], k=5, iou_threshold=0.5)
    assert r5.rate == 1.0
    assert r5.numerator == 1
    assert r5.denominator == 1

    mrr = mean_reciprocal_rank([query], [res], max_rank=5, iou_threshold=0.5)
    assert mrr == 0.5


def test_hand_check_matches_second_gold_window() -> None:
    # Case: Top candidate matches only the second gold window -> Full credit for that match
    query = QueryLabel(
        qid="q1",
        video_id="v1",
        query="action happens twice",
        answerable=True,
        gold_windows=[[5.0, 10.0], [30.0, 40.0]],
    )
    # Top candidate [30, 40] has IoU 1.0 with second window, 0.0 with first window
    cand = CandidateWindow(start_s=30.0, end_s=40.0, score=3.0)
    res = SearchResult(
        abstained=False,
        confidence=0.9,
        route="visual",
        results=[cand],
    )

    r1 = recall_at_k([query], [res], k=1, iou_threshold=0.7)
    assert r1.rate == 1.0

    m_iou = top_window_mean_iou([query], [res])
    assert m_iou == 1.0


def test_hand_check_retained_candidates_after_abstention() -> None:
    # Case: Correct top result retained after abstention -> Raw R@1 = 1, abstention-aware R@1 = 0
    query = QueryLabel(
        qid="q1",
        video_id="v1",
        query="find scene",
        answerable=True,
        gold_windows=[[10.0, 20.0]],
    )
    cand = CandidateWindow(start_s=10.0, end_s=20.0, score=2.5)  # Perfect match
    res = SearchResult(
        abstained=True,  # System abstained
        confidence=0.3,
        route="fusion",
        results=[cand],  # But candidates retained internally
    )

    raw_r1 = recall_at_k([query], [res], k=1, iou_threshold=0.5, abstention_aware=False)
    assert raw_r1.rate == 1.0

    aware_r1 = recall_at_k([query], [res], k=1, iou_threshold=0.5, abstention_aware=True)
    assert aware_r1.rate == 0.0


def test_hand_check_balanced_accuracy_scenario() -> None:
    # Case: Two answerable queries, one withheld. Two negatives, both withheld
    # Expected: Rejection = 1, false-abstain = 0.5, balanced accuracy = 0.75
    q_ans1 = QueryLabel(qid="a1", video_id="v1", query="q1", answerable=True, gold_windows=[[0.0, 5.0]])
    q_ans2 = QueryLabel(qid="a2", video_id="v1", query="q2", answerable=True, gold_windows=[[5.0, 10.0]])
    q_neg1 = QueryLabel(qid="n1", video_id="v1", query="n1", answerable=False, gold_windows=[], neg_type="near_miss")
    q_neg2 = QueryLabel(qid="n2", video_id="v1", query="n2", answerable=False, gold_windows=[], neg_type="swap")

    queries = [q_ans1, q_ans2, q_neg1, q_neg2]

    # One answerable withheld (abstained=True), one answered (abstained=False)
    res_ans1 = SearchResult(abstained=False, confidence=0.8, route="visual")
    res_ans2 = SearchResult(abstained=True, confidence=0.2, route="visual")
    # Both negatives withheld (abstained=True)
    res_neg1 = SearchResult(abstained=True, confidence=0.1, route="visual")
    res_neg2 = SearchResult(abstained=True, confidence=0.15, route="visual")

    results = [res_ans1, res_ans2, res_neg1, res_neg2]

    rej = rejection_rate(queries, results)
    assert rej.rate == 1.0
    assert rej.numerator == 2
    assert rej.denominator == 2

    fa = false_abstain_rate(queries, results)
    assert fa.rate == 0.5
    assert fa.numerator == 1
    assert fa.denominator == 2

    bacc = balanced_accuracy(queries, results)
    assert bacc == 0.75


def test_hand_check_risk_coverage_tied_confidence() -> None:
    # Case: Two equally confident queries, one correct
    # Expected: One coverage step, risk = 0.5, area = 0.5
    q1 = QueryLabel(qid="q1", video_id="v1", query="q1", answerable=True, gold_windows=[[0.0, 10.0]])
    q2 = QueryLabel(qid="q2", video_id="v1", query="q2", answerable=False, gold_windows=[], neg_type="ood")

    # Both have confidence 0.7. Q1 has correct top window (IoU 1.0 >= 0.5). Q2 is unanswerable (wrong).
    res1 = SearchResult(
        abstained=False,
        confidence=0.7,
        route="visual",
        results=[CandidateWindow(start_s=0.0, end_s=10.0, score=2.0)],
    )
    res2 = SearchResult(
        abstained=False,
        confidence=0.7,
        route="visual",
        results=[CandidateWindow(start_s=0.0, end_s=10.0, score=2.0)],
    )

    rc = risk_coverage_curve([q1, q2], [res1, res2], iou_threshold=0.5)
    assert len(rc["curve"]) == 1
    step = rc["curve"][0]
    assert step["coverage"] == 1.0
    assert step["risk"] == 0.5
    assert rc["area"] == 0.5


def test_hand_check_wilson_interval() -> None:
    # Case: 40 successes in 50 trials -> Wilson interval rounds to [0.670, 0.888]
    lower, upper = wilson_score_interval(40, 50, confidence=0.95)
    assert lower is not None and upper is not None
    assert round(lower, 3) == 0.670
    assert round(upper, 3) == 0.888


def test_empty_results_and_empty_denominators() -> None:
    q_ans = QueryLabel(qid="q1", video_id="v1", query="q", answerable=True, gold_windows=[[0.0, 10.0]])
    res_empty = SearchResult(abstained=False, confidence=0.5, route="visual", results=[])

    # Empty candidates
    assert recall_at_k([q_ans], [res_empty], k=1, iou_threshold=0.5).rate == 0.0
    assert mean_reciprocal_rank([q_ans], [res_empty], max_rank=5, iou_threshold=0.5) == 0.0
    assert top_window_mean_iou([q_ans], [res_empty]) == 0.0

    # Empty denominator for unanswerable metrics when all queries are answerable
    rej = rejection_rate([q_ans], [res_empty])
    assert rej.rate is None
    assert rej.denominator == 0

    far = false_answer_rate([q_ans], [res_empty])
    assert far.rate is None
    assert far.denominator == 0

    # Balanced accuracy with 0 negatives
    assert balanced_accuracy([q_ans], [res_empty]) is None

    # Empty list of queries
    assert wilson_score_interval(0, 0) == (None, None)
    assert recall_at_k([], [], k=1, iou_threshold=0.5).rate is None


def test_start_time_error_tie_breaking_and_summary() -> None:
    # Multiple gold windows:
    # Gold windows: [10, 20) and [50, 60)
    # Predicted top window: [12, 22) -> overlaps [10, 20) with IoU = 8/12, start diff = |12 - 10| = 2.0
    q1 = QueryLabel(qid="q1", video_id="v1", query="q1", answerable=True, gold_windows=[[10.0, 20.0], [50.0, 60.0]])
    res1 = SearchResult(
        abstained=False,
        confidence=0.8,
        route="temporal",
        results=[CandidateWindow(start_s=12.0, end_s=22.0, score=2.0)],
    )

    # Tie breaking by earliest start:
    # Two gold windows [10, 20) and [30, 40)
    # Predicted top window [10, 40)
    # IoU with [10, 20): intersection 10, union 30 -> IoU = 1/3
    # IoU with [30, 40): intersection 10, union 30 -> IoU = 1/3
    # Tied IoU! Earliest start is 10.0. Predicted start is 10.0 -> error = 0.0
    q2 = QueryLabel(qid="q2", video_id="v1", query="q2", answerable=True, gold_windows=[[10.0, 20.0], [30.0, 40.0]])
    res2 = SearchResult(
        abstained=False,
        confidence=0.9,
        route="temporal",
        results=[CandidateWindow(start_s=10.0, end_s=40.0, score=2.0)],
    )

    # Predicted top window does not overlap any gold window -> not eligible
    q3 = QueryLabel(qid="q3", video_id="v1", query="q3", answerable=True, gold_windows=[[0.0, 5.0]])
    res3 = SearchResult(
        abstained=False,
        confidence=0.5,
        route="temporal",
        results=[CandidateWindow(start_s=20.0, end_s=30.0, score=1.0)],
    )

    summary = start_time_error([q1, q2, q3], [res1, res2, res3])
    assert summary.count == 2
    # Errors are [2.0, 0.0] -> sorted: [0.0, 2.0]
    assert summary.median == 1.0
    assert summary.q25 == 0.5
    assert summary.q75 == 1.5
    assert summary.iqr == 1.0


def test_overlapping_gold_windows_scoring_takes_best_iou() -> None:
    # Query with overlapping gold windows [0, 10) and [5, 15)
    query = QueryLabel(
        qid="q_overlap",
        video_id="v1",
        query="action in overlapping intervals",
        answerable=True,
        gold_windows=[[0.0, 10.0], [5.0, 15.0]],
    )

    # Candidate 1: [0, 10) -> IoU with [0, 10) is 1.0, with [5, 15) is 1/3 -> best IoU = 1.0
    cand1 = CandidateWindow(start_s=0.0, end_s=10.0, score=2.0)
    # Candidate 2: [5, 15) -> IoU with [0, 10) is 1/3, with [5, 15) is 1.0 -> best IoU = 1.0
    cand2 = CandidateWindow(start_s=5.0, end_s=15.0, score=1.8)
    # Candidate 3: [5, 10) -> IoU with [0, 10) is 5/10 = 0.5, with [5, 15) is 5/10 = 0.5 -> best IoU = 0.5
    cand3 = CandidateWindow(start_s=5.0, end_s=10.0, score=1.5)

    res = SearchResult(
        abstained=False,
        confidence=0.9,
        route="temporal",
        results=[cand1, cand2, cand3],
    )

    # Best IoU helper takes maximum across gold windows
    assert best_iou_with_gold((0.0, 10.0), query.gold_windows) == 1.0
    assert best_iou_with_gold((5.0, 15.0), query.gold_windows) == 1.0
    assert best_iou_with_gold((5.0, 10.0), query.gold_windows) == 0.5

    # Recall and top window mean IoU credit the best match
    assert top_window_mean_iou([query], [res]) == 1.0
    assert recall_at_k([query], [res], k=1, iou_threshold=0.7).rate == 1.0


