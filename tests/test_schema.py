"""Tests for vi.schema records and validation."""

import json
import pytest
from pydantic import ValidationError

from vi.schema import (
    CandidateWindow,
    CaptionUnit,
    EvaluationPrediction,
    Evidence,
    FrameRecord,
    IndexMeta,
    QueryLabel,
    SearchResult,
    TranscriptUnit,
    VideoMetadata,
    validate_duration,
    validate_half_open_interval,
    validate_non_overlapping_intervals,
)


def test_validate_half_open_interval() -> None:
    # Valid intervals
    validate_half_open_interval(0.0, 5.0)
    validate_half_open_interval(10.5, 20.0, duration_s=25.0)

    # Negative start
    with pytest.raises(ValueError, match="start_s must be non-negative"):
        validate_half_open_interval(-1.0, 5.0)

    # start >= end
    with pytest.raises(ValueError, match="strictly less than"):
        validate_half_open_interval(5.0, 5.0)

    with pytest.raises(ValueError, match="strictly less than"):
        validate_half_open_interval(6.0, 5.0)

    # Exceeds duration
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_half_open_interval(5.0, 31.0, duration_s=30.0)

    # Non-finite
    with pytest.raises(ValueError, match="must be finite"):
        validate_half_open_interval(float("nan"), 5.0)


def test_validate_non_overlapping_intervals() -> None:
    # Disjoint intervals
    validate_non_overlapping_intervals([(0.0, 5.0), (10.0, 15.0)])
    # Abutting half-open intervals [0, 5) and [5, 10) have zero overlap
    validate_non_overlapping_intervals([(0.0, 5.0), (5.0, 10.0)])

    # Overlapping intervals
    with pytest.raises(ValueError, match="Adjacent intervals must have zero overlap"):
        validate_non_overlapping_intervals([(0.0, 6.0), (5.0, 10.0)])


def test_video_metadata() -> None:
    meta = VideoMetadata(
        video_id="v01",
        path="videos/v01.mp4",
        duration_s=60.0,
        sha256="abc123sha",
        dataset="tl",
        split="dev",
    )
    assert meta.video_id == "v01"
    assert meta.duration_s == 60.0

    with pytest.raises(ValidationError):
        VideoMetadata(
            video_id="",
            path="videos/v01.mp4",
            duration_s=60.0,
            sha256="abc",
            dataset="tl",
            split="dev",
        )

    with pytest.raises(ValidationError):
        VideoMetadata(
            video_id="v01",
            path="videos/v01.mp4",
            duration_s=-5.0,
            sha256="abc",
            dataset="tl",
            split="dev",
        )


def test_query_label_answerable_and_unanswerable() -> None:
    # Answerable: valid with one gold window
    q_ans = QueryLabel(
        qid="q01",
        video_id="v01",
        query="person walks dog",
        answerable=True,
        gold_windows=[[10.0, 20.0]],
        qtype="visual",
    )
    assert q_ans.answerable is True
    assert len(q_ans.gold_windows) == 1

    # Answerable: overlapping valid gold windows ARE allowed
    q_overlap = QueryLabel(
        qid="q02",
        video_id="v01",
        query="overlapping actions",
        answerable=True,
        gold_windows=[[5.0, 12.0], [10.0, 15.0]],
        qtype="temporal",
    )
    assert len(q_overlap.gold_windows) == 2

    # Answerable: requires at least one gold window
    with pytest.raises(ValidationError, match="at least one gold window"):
        QueryLabel(
            qid="q03",
            video_id="v01",
            query="person jumps",
            answerable=True,
            gold_windows=[],
        )

    # Answerable cannot have neg_type
    with pytest.raises(ValidationError, match="cannot have neg_type"):
        QueryLabel(
            qid="q04",
            video_id="v01",
            query="person jumps",
            answerable=True,
            gold_windows=[[5.0, 10.0]],
            neg_type="near_miss",
        )

    # Unanswerable: valid with zero gold windows and valid neg_type
    q_unans = QueryLabel(
        qid="q05",
        video_id="v01",
        query="person drives car",
        answerable=False,
        gold_windows=[],
        neg_type="near_miss",
    )
    assert q_unans.answerable is False

    # Unanswerable cannot have gold windows
    with pytest.raises(ValidationError, match="must have zero gold windows"):
        QueryLabel(
            qid="q06",
            video_id="v01",
            query="person drives car",
            answerable=False,
            gold_windows=[[1.0, 2.0]],
            neg_type="near_miss",
        )

    # Unanswerable must specify neg_type
    with pytest.raises(ValidationError, match="must specify neg_type"):
        QueryLabel(
            qid="q07",
            video_id="v01",
            query="person drives car",
            answerable=False,
            gold_windows=[],
            neg_type=None,
        )


def test_query_label_late_flag_and_serialization_round_trip() -> None:
    # Query with late=True
    q_late = QueryLabel(
        qid="tl-0099",
        video_id="tl-v01",
        query="late query added during dev",
        answerable=True,
        gold_windows=[[15.0, 25.5]],
        qtype="spoken",
        multi_event=False,
        late=True,
        author="YS",
        verifier="MJ",
    )
    assert q_late.late is True

    # JSON round trip
    json_str = q_late.model_dump_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["late"] is True

    reconstructed = QueryLabel.model_validate_json(json_str)
    assert reconstructed.late is True
    assert reconstructed.qid == q_late.qid
    assert reconstructed.gold_windows == [(15.0, 25.5)]

    # Query with default late=False
    q_default = QueryLabel(
        qid="tl-0042",
        video_id="tl-v07",
        query="standard frozen query",
        answerable=False,
        gold_windows=[],
        neg_type="near_miss",
    )
    assert q_default.late is False
    assert json.loads(q_default.model_dump_json())["late"] is False


def test_query_label_vocabulary_and_typo_validation() -> None:
    # Valid qtypes: spoken, visual, temporal, mixed
    for qtype in ["spoken", "visual", "temporal", "mixed"]:
        q = QueryLabel(
            qid="q",
            video_id="v",
            query="test",
            answerable=True,
            gold_windows=[[0.0, 5.0]],
            qtype=qtype,
        )
        assert q.qtype == qtype

    # Invalid qtype / typos
    for bad_qtype in ["visuall", "speech", "action", ""]:
        with pytest.raises(ValidationError, match="Invalid qtype"):
            QueryLabel(
                qid="q",
                video_id="v",
                query="test",
                answerable=True,
                gold_windows=[[0.0, 5.0]],
                qtype=bad_qtype,
            )

    # Valid neg_types across team set and Charades
    for neg_type in ["near_miss", "swap", "ood", "out_of_domain", "in_domain", "id"]:
        q_neg = QueryLabel(
            qid="qn",
            video_id="v",
            query="test neg",
            answerable=False,
            gold_windows=[],
            neg_type=neg_type,
        )
        assert q_neg.neg_type == neg_type

    # Invalid neg_type typos
    for bad_neg in ["near-miss", "swapp", "random", "fake"]:
        with pytest.raises(ValidationError, match="Invalid neg_type"):
            QueryLabel(
                qid="qn",
                video_id="v",
                query="test neg",
                answerable=False,
                gold_windows=[],
                neg_type=bad_neg,
            )

    # Empty neg_type
    with pytest.raises(ValidationError, match="must specify neg_type"):
        QueryLabel(
            qid="qn",
            video_id="v",
            query="test neg",
            answerable=False,
            gold_windows=[],
            neg_type="",
        )


def test_duration_aware_validation() -> None:
    video_meta = VideoMetadata(
        video_id="v01",
        path="videos/v01.mp4",
        duration_s=30.0,
        sha256="dummy_sha",
        dataset="tl",
        split="dev",
    )

    # 1. QueryLabel: ending exactly at duration vs extending past
    q_exact = QueryLabel(
        qid="q1",
        video_id="v01",
        query="action",
        answerable=True,
        gold_windows=[[10.0, 30.0]],  # exactly at duration
    )
    validate_duration(q_exact, video_meta)  # passes

    q_exceeds = QueryLabel(
        qid="q2",
        video_id="v01",
        query="action",
        answerable=True,
        gold_windows=[[10.0, 30.1]],  # extends past duration
    )
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_duration(q_exceeds, video_meta)

    # 2. CandidateWindow: ending exactly at duration vs extending past
    cand_exact = CandidateWindow(start_s=0.0, end_s=30.0, score=1.0)
    validate_duration(cand_exact, video_meta)

    cand_exceeds = CandidateWindow(start_s=0.0, end_s=30.5, score=1.0)
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_duration(cand_exceeds, video_meta)

    # 3. TranscriptUnit: ending exactly at duration vs extending past
    t_exact = TranscriptUnit(unit_id=0, start_s=0.0, end_s=30.0, text="speech")
    validate_duration(t_exact, video_meta)

    t_exceeds = TranscriptUnit(unit_id=1, start_s=0.0, end_s=30.1, text="speech")
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_duration(t_exceeds, video_meta)

    # 4. CaptionUnit: ending exactly at duration vs extending past
    c_exact = CaptionUnit(unit_id=0, start_s=0.0, end_s=30.0, text="caption")
    validate_duration(c_exact, video_meta)

    c_exceeds = CaptionUnit(unit_id=1, start_s=0.0, end_s=30.2, text="caption")
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_duration(c_exceeds, video_meta)

    # 5. FrameRecord: timestamp exactly at duration vs extending past
    f_exact = FrameRecord(frame_id=0, t_s=30.0, jpg_path="f0030.jpg")
    validate_duration(f_exact, video_meta)

    f_exceeds = FrameRecord(frame_id=1, t_s=30.01, jpg_path="f0031.jpg")
    with pytest.raises(ValueError, match="exceeds video duration"):
        validate_duration(f_exceeds, video_meta)


def test_search_result_candidate_and_abstention_rules() -> None:
    cands = [
        CandidateWindow(
            start_s=0.0,
            end_s=5.0,
            score=1.5,
            path_scores={"transcript": 0.0, "frame": 1.0, "caption": 1.5},
            evidence=Evidence(captions=["person walking"]),
        ),
        CandidateWindow(
            start_s=5.0,
            end_s=10.0,
            score=0.8,
            path_scores={"transcript": 0.5, "frame": 0.5, "caption": 0.8},
        ),
    ]

    res = SearchResult(
        abstained=False,
        confidence=0.85,
        route="visual",
        results=cands,
        timings_ms={"encode": 10.0, "search": 2.0},
    )
    assert len(res.results) == 2
    assert res.confidence == 0.85

    # Abstained result CAN retain candidates internally
    abstained_res = SearchResult(
        abstained=True,
        confidence=0.35,
        route="temporal",
        results=cands,
    )
    assert abstained_res.abstained is True
    assert len(abstained_res.results) == 2

    # Maximum 5 candidates enforced
    many_cands = [
        CandidateWindow(start_s=float(i), end_s=float(i + 1), score=1.0)
        for i in range(6)
    ]
    with pytest.raises(ValidationError, match="at most 5 candidates"):
        SearchResult(
            abstained=False,
            confidence=0.5,
            route="mixed",
            results=many_cands,
        )

    # Confidence must be in [0, 1]
    with pytest.raises(ValidationError, match="confidence must be in"):
        SearchResult(
            abstained=False,
            confidence=1.5,
            route="mixed",
            results=[],
        )


def test_evaluation_prediction() -> None:
    pred = EvaluationPrediction(
        video_id="v01",
        qid="q01",
        result=SearchResult(abstained=False, confidence=0.8, route="visual"),
    )
    assert pred.key == ("v01", "q01")
    assert pred.result.confidence == 0.8

    with pytest.raises(ValidationError):
        EvaluationPrediction(
            video_id="",
            qid="q01",
            result=SearchResult(abstained=False, confidence=0.8, route="visual"),
        )
