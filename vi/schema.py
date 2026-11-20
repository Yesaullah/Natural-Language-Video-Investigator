"""Shared record types and validation for Natural-Language Video Investigator.

Preserves the field names and validation rules defined in
plans/M1-evaluation-foundation.md and video_investigator_plan.md §§4.1, 5.1, 8.1.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

# Vocabulary constants from video_investigator_plan.md §§4.1, 4.2, 5.2
VALID_QUERY_TYPES: Set[str] = {"spoken", "visual", "temporal", "mixed"}
VALID_CATEGORIES: Set[str] = {"Speech-heavy", "Mixed", "Action-heavy"}
VALID_SPLITS: Set[str] = {"dev", "cal", "test"}
VALID_NEGATIVE_TYPES: Set[str] = {
    # Team set vocabulary (§4.1, §5.2)
    "near_miss",
    "swap",
    "ood",
    "out_of_domain",
    # Charades vocabulary (§4.2, §5.2)
    "in_domain",
    "id",
}


def validate_half_open_interval(
    start_s: float,
    end_s: float,
    duration_s: Optional[float] = None,
    allow_zero_length: bool = False,
) -> None:
    """Validate a finite half-open interval [start_s, end_s).

    Rules:
    - Finite float values.
    - 0 <= start_s < end_s (or start_s <= end_s if allow_zero_length=True).
    - If duration_s is specified, end_s <= duration_s.
    """
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise ValueError(f"Interval bounds must be finite numbers: [{start_s}, {end_s})")
    if start_s < 0.0:
        raise ValueError(f"Interval start_s must be non-negative: {start_s}")
    if allow_zero_length:
        if start_s > end_s:
            raise ValueError(f"start_s ({start_s}) must be <= end_s ({end_s})")
    else:
        if start_s >= end_s:
            raise ValueError(f"start_s ({start_s}) must be strictly less than end_s ({end_s})")
    if duration_s is not None and math.isfinite(duration_s):
        if end_s > duration_s:
            raise ValueError(f"Interval end_s ({end_s}) exceeds video duration ({duration_s})")


def validate_non_overlapping_intervals(
    intervals: Sequence[Tuple[float, float]],
    duration_s: Optional[float] = None,
) -> None:
    """Validate that half-open intervals are valid and pairwise disjoint (zero overlap).

    Adjacent intervals [s1, e1) and [s2, e2) have zero overlap iff e1 <= s2
    when sorted by start time.
    """
    if not intervals:
        return

    validated: List[Tuple[float, float]] = []
    for s, e in intervals:
        validate_half_open_interval(s, e, duration_s=duration_s, allow_zero_length=False)
        validated.append((float(s), float(e)))

    # Sort intervals by start time
    sorted_intervals = sorted(validated, key=lambda x: (x[0], x[1]))
    for i in range(len(sorted_intervals) - 1):
        curr_s, curr_e = sorted_intervals[i]
        next_s, next_e = sorted_intervals[i + 1]
        if curr_e > next_s:
            raise ValueError(
                f"Intervals [{curr_s}, {curr_e}) and [{next_s}, {next_e}) overlap. "
                "Adjacent intervals must have zero overlap."
            )


class VideoMetadata(BaseModel):
    """Video table record matching SQLite table 'video'."""

    video_id: str
    path: str
    duration_s: float
    sha256: str
    dataset: str
    split: str

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()

    @field_validator("duration_s")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"duration_s must be a finite positive number, got {v}")
        return float(v)


class TranscriptUnit(BaseModel):
    """Transcript passage record matching SQLite table 'transcript_units'."""

    unit_id: int
    start_s: float
    end_s: float
    text: str

    @model_validator(mode="after")
    def validate_bounds(self) -> "TranscriptUnit":
        if self.unit_id < 0:
            raise ValueError(f"unit_id must be non-negative: {self.unit_id}")
        validate_half_open_interval(self.start_s, self.end_s)
        return self


class FrameRecord(BaseModel):
    """Frame record matching SQLite table 'frames'."""

    frame_id: int
    t_s: float
    jpg_path: str

    @model_validator(mode="after")
    def validate_bounds(self) -> "FrameRecord":
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative: {self.frame_id}")
        if not math.isfinite(self.t_s) or self.t_s < 0.0:
            raise ValueError(f"t_s must be a finite non-negative number: {self.t_s}")
        if not self.jpg_path or not self.jpg_path.strip():
            raise ValueError("jpg_path must not be empty")
        return self


class CaptionUnit(BaseModel):
    """Caption unit record matching SQLite table 'caption_units'."""

    unit_id: int
    start_s: float
    end_s: float
    text: str

    @model_validator(mode="after")
    def validate_bounds(self) -> "CaptionUnit":
        if self.unit_id < 0:
            raise ValueError(f"unit_id must be non-negative: {self.unit_id}")
        validate_half_open_interval(self.start_s, self.end_s)
        return self


class QueryLabel(BaseModel):
    """Label for evaluation queries (§4.1)."""

    qid: str
    video_id: str
    query: str
    answerable: bool
    gold_windows: List[Tuple[float, float]] = Field(default_factory=list)
    qtype: Optional[str] = None
    multi_event: bool = False
    late: bool = False
    neg_type: Optional[str] = None
    author: Optional[str] = None
    verifier: Optional[str] = None

    @field_validator("gold_windows", mode="before")
    @classmethod
    def coerce_gold_windows(cls, v: Any) -> List[Tuple[float, float]]:
        if not v:
            return []
        windows: List[Tuple[float, float]] = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                windows.append((float(item[0]), float(item[1])))
            else:
                raise ValueError(f"Invalid gold window item: {item}")
        return windows

    @model_validator(mode="after")
    def validate_query_label(self) -> "QueryLabel":
        if not self.qid or not self.qid.strip():
            raise ValueError("qid must not be empty")
        if not self.video_id or not self.video_id.strip():
            raise ValueError("video_id must not be empty")

        # Validate qtype vocabulary
        if self.qtype is not None and self.qtype not in VALID_QUERY_TYPES:
            raise ValueError(
                f"Invalid qtype: '{self.qtype}'. Must be one of {sorted(VALID_QUERY_TYPES)}"
            )

        if self.answerable:
            if len(self.gold_windows) == 0:
                raise ValueError("Answerable query must have at least one gold window")
            # Multiple gold windows are allowed and may overlap; validate each window interval
            for s, e in self.gold_windows:
                validate_half_open_interval(s, e)
            if self.neg_type is not None:
                raise ValueError(f"Answerable query cannot have neg_type, got {self.neg_type}")
        else:
            if len(self.gold_windows) != 0:
                raise ValueError(
                    f"Unanswerable query must have zero gold windows, got {len(self.gold_windows)}"
                )
            if not self.neg_type:
                raise ValueError("Unanswerable query must specify neg_type")
            if self.neg_type not in VALID_NEGATIVE_TYPES:
                raise ValueError(
                    f"Invalid neg_type: '{self.neg_type}'. Must be one of {sorted(VALID_NEGATIVE_TYPES)}"
                )
        return self


class Evidence(BaseModel):
    """Multimodal evidence attached to a candidate window (§8.1)."""

    frames: List[str] = Field(default_factory=list)
    transcript: List[str] = Field(default_factory=list)
    captions: List[str] = Field(default_factory=list)


class CandidateWindow(BaseModel):
    """Ranked candidate window returned by search (§8.1)."""

    start_s: float
    end_s: float
    score: float
    path_scores: Dict[str, float] = Field(default_factory=dict)
    evidence: Evidence = Field(default_factory=Evidence)

    @model_validator(mode="after")
    def validate_bounds(self) -> "CandidateWindow":
        validate_half_open_interval(self.start_s, self.end_s)
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")
        return self


class SearchResult(BaseModel):
    """Standardized search outcome returned by pipeline search (§8.1)."""

    abstained: bool
    confidence: float
    route: str
    results: List[CandidateWindow] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return float(v)

    @field_validator("results")
    @classmethod
    def validate_results_count(cls, v: List[CandidateWindow]) -> List[CandidateWindow]:
        if len(v) > 5:
            raise ValueError(f"Search results must contain at most 5 candidates, got {len(v)}")
        return v


class EvaluationPrediction(BaseModel):
    """System prediction bound to video_id and qid for evaluation (§8.1, §8.3)."""

    video_id: str
    qid: str
    result: SearchResult

    @field_validator("video_id", "qid")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id and qid must not be empty")
        return v.strip()

    @property
    def key(self) -> Tuple[str, str]:
        return (self.video_id, self.qid)


class IndexMeta(BaseModel):
    """Metadata recorded in meta.json for an indexed video (§8.1)."""

    video_id: str
    models: Dict[str, str] = Field(default_factory=dict)
    revisions: Dict[str, str] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)
    config_hash: str = "synthetic"
    timings_s: Dict[str, float] = Field(default_factory=dict)
    peak_memory_mib: Dict[str, float] = Field(default_factory=dict)
    created_at: str = ""
    synthetic: bool = True


def validate_duration(
    record: Union[QueryLabel, CandidateWindow, TranscriptUnit, CaptionUnit, FrameRecord],
    video_or_duration: Union[VideoMetadata, float],
) -> None:
    """Validate that record intervals or timestamps do not extend past video duration.

    Supports:
    - QueryLabel (all gold_windows must satisfy 0 <= start < end <= duration_s)
    - CandidateWindow (0 <= start_s < end_s <= duration_s)
    - TranscriptUnit (0 <= start_s < end_s <= duration_s)
    - CaptionUnit (0 <= start_s < end_s <= duration_s)
    - FrameRecord (0 <= t_s <= duration_s)
    """
    if isinstance(video_or_duration, VideoMetadata):
        duration_s = video_or_duration.duration_s
    elif isinstance(video_or_duration, (int, float)):
        duration_s = float(video_or_duration)
    else:
        raise TypeError(f"Expected VideoMetadata or float duration, got {type(video_or_duration)}")

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration_s must be a finite positive number, got {duration_s}")

    if isinstance(record, QueryLabel):
        for s, e in record.gold_windows:
            if e > duration_s:
                raise ValueError(
                    f"Gold window [{s}, {e}) exceeds video duration ({duration_s})"
                )
    elif isinstance(record, CandidateWindow):
        if record.end_s > duration_s:
            raise ValueError(
                f"Candidate window [{record.start_s}, {record.end_s}) exceeds video duration ({duration_s})"
            )
    elif isinstance(record, (TranscriptUnit, CaptionUnit)):
        if record.end_s > duration_s:
            raise ValueError(
                f"{type(record).__name__} interval [{record.start_s}, {record.end_s}) exceeds video duration ({duration_s})"
            )
    elif isinstance(record, FrameRecord):
        if record.t_s > duration_s:
            raise ValueError(
                f"FrameRecord timestamp {record.t_s} exceeds video duration ({duration_s})"
            )
    else:
        raise TypeError(f"Unsupported record type for duration validation: {type(record)}")


class TeamVideoRecord(BaseModel):
    """Record from team inventory CSV (data/tl/videos.csv) matching §4.1 and M2."""

    video_id: str
    category: str
    source: str
    licence: str

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty or blank")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        s = v.strip()
        cat_map = {
            "speech-heavy": "Speech-heavy",
            "mixed": "Mixed",
            "action-heavy": "Action-heavy",
        }
        low = s.lower()
        if low not in cat_map:
            raise ValueError(
                f"Unknown category '{v}'. Must be one of {sorted(VALID_CATEGORIES)}"
            )
        return cat_map[low]

    @field_validator("source", "licence")
    @classmethod
    def validate_strings(cls, v: str) -> str:
        if v is None:
            raise ValueError("Field cannot be None")
        return str(v).strip()


class ManifestRecord(BaseModel):
    """Entry in a split manifest (§8.1, §8.3 M2)."""

    dataset: str
    video_id: str
    split: str
    source_split: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    licence: Optional[str] = None

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        s = v.strip().lower()
        if s == "calibration":
            s = "cal"
        if s not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{v}'. Must be one of {sorted(VALID_SPLITS)}")
        return s


class SplitManifest(BaseModel):
    """Machine-readable manifest for video split allocations (§8.3 M2)."""

    manifest_version: str = "1.0"
    dataset: str
    seed: int
    sampling_convention: str
    split_counts: Dict[str, int]
    category_counts: Optional[Dict[str, Dict[str, int]]] = None
    records: List[ManifestRecord]

