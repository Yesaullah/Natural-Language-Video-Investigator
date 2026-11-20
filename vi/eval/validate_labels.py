"""Label loading and validation utilities for M3 evaluation harness.

Provides loading labels from JSONL fixtures, loading video metadata from SQLite
indexes, alignment checks, and coverage validation.
Conforms to plans/M3-evaluation-harness.md and video_investigator_plan.md §§4.1, 8.3.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from vi.schema import QueryLabel, VideoMetadata


# Splits that may be evaluated; calibration and test are excluded from B0
_SUPPORTED_SPLITS = {"dev"}


def reject_unsupported_split(split: str) -> None:
    """Raise ValueError before opening any data if the split is not supported."""
    if split not in _SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. "
            f"Only {sorted(_SUPPORTED_SPLITS)} are supported in this run mode. "
            "Calibration and test labels belong to their assigned units (M6, Y11)."
        )


def reject_unsupported_system(system: str, supported: Sequence[str]) -> None:
    """Raise ValueError before opening any data if the system is not supported."""
    if system not in supported:
        raise ValueError(
            f"Unsupported system '{system}'. Supported systems: {sorted(supported)}."
        )


def load_labels_from_jsonl(path: Path) -> List[QueryLabel]:
    """Load and validate QueryLabel records from a JSONL file.

    Raises FileNotFoundError if the file is missing.
    Raises ValueError if any record fails schema validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    labels: List[QueryLabel] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            try:
                label = QueryLabel.model_validate(record)
            except Exception as exc:
                raise ValueError(f"Invalid label on line {lineno} of {path}: {exc}") from exc
            labels.append(label)

    return labels


def load_video_metadata_from_index(index_dir: Path) -> VideoMetadata:
    """Read video metadata from index.sqlite in read-only mode.

    Raises FileNotFoundError if the index directory or database is missing.
    """
    db_path = index_dir / "index.sqlite"
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"Index database not found: {db_path}")

    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT video_id, path, duration_s, sha256, dataset, split FROM video LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No video record found in {db_path}")
        video_id, path, duration_s, sha256, dataset, split = row
        return VideoMetadata(
            video_id=video_id,
            path=path,
            duration_s=float(duration_s),
            sha256=sha256,
            dataset=dataset,
            split=split,
        )
    finally:
        conn.close()


def check_id_consistency(
    labels: Sequence[QueryLabel],
    video_meta: Dict[str, VideoMetadata],
) -> None:
    """Validate that every label's video_id exists in the loaded index metadata.

    Raises ValueError with a clear message if any label references a missing video.
    """
    missing = {q.video_id for q in labels if q.video_id not in video_meta}
    if missing:
        raise ValueError(
            f"Labels reference video IDs not found in loaded indexes: {sorted(missing)}"
        )


def check_unique_query_keys(labels: Sequence[QueryLabel]) -> None:
    """Validate that (video_id, qid) pairs are unique.

    Raises ValueError listing all duplicates.
    """
    seen: Dict[Tuple[str, str], int] = {}
    for i, q in enumerate(labels):
        key = (q.video_id, q.qid)
        if key in seen:
            raise ValueError(
                f"Duplicate (video_id, qid) key {key} at positions {seen[key]} and {i}"
            )
        seen[key] = i


def check_gold_window_bounds(
    labels: Sequence[QueryLabel],
    video_meta: Dict[str, VideoMetadata],
) -> None:
    """Validate that all gold windows lie within their video's duration.

    Raises ValueError listing the first out-of-bounds window found.
    """
    for q in labels:
        if q.video_id not in video_meta:
            continue  # Already caught by check_id_consistency
        dur = video_meta[q.video_id].duration_s
        for s, e in q.gold_windows:
            if e > dur + 1e-9:
                raise ValueError(
                    f"Gold window [{s}, {e}) for qid='{q.qid}' exceeds "
                    f"video duration {dur}s (video_id='{q.video_id}')"
                )


def check_video_coverage(
    labels: Sequence[QueryLabel],
    expected_video_ids: Sequence[str],
) -> None:
    """Validate that all expected video IDs appear in the labels.

    Raises ValueError listing missing video IDs.
    """
    labelled_videos = {q.video_id for q in labels}
    missing = set(expected_video_ids) - labelled_videos
    if missing:
        raise ValueError(
            f"Expected video IDs not represented in labels: {sorted(missing)}"
        )


def validate_labels_and_indexes(
    labels: List[QueryLabel],
    index_dirs: Dict[str, Path],
) -> Dict[str, VideoMetadata]:
    """Full validation pipeline for labels against their indexes.

    Steps:
    1. Load VideoMetadata from each index (read-only).
    2. Check ID consistency (all label video_ids present in indexes).
    3. Check unique (video_id, qid) keys.
    4. Check gold window bounds.
    5. Check both video IDs are represented in labels.

    Returns the loaded VideoMetadata dict.
    Raises FileNotFoundError or ValueError on any failure.
    """
    video_meta: Dict[str, VideoMetadata] = {}
    for vid, index_dir in index_dirs.items():
        meta = load_video_metadata_from_index(index_dir)
        # Consistency: the directory name should match the video_id in the DB
        if meta.video_id != vid:
            raise ValueError(
                f"Index directory '{vid}' contains video_id '{meta.video_id}' – mismatch."
            )
        video_meta[vid] = meta

    check_id_consistency(labels, video_meta)
    check_unique_query_keys(labels)
    check_gold_window_bounds(labels, video_meta)
    check_video_coverage(labels, list(index_dirs.keys()))

    return video_meta


def build_gold_length_pool(labels: Sequence[QueryLabel]) -> List[float]:
    """Build the pooled list of gold-window lengths from all answerable labels.

    The brief requires: 'sample a length uniformly with replacement from the
    pooled development gold-window lengths for that dataset'.

    Raises ValueError if the pool would be empty.
    """
    pool: List[float] = []
    for q in labels:
        if q.answerable:
            for s, e in q.gold_windows:
                length = float(e - s)
                if length > 0.0:
                    pool.append(length)
    if not pool:
        raise ValueError(
            "Gold-window length pool is empty. "
            "At least one answerable query with a positive-length gold window is required."
        )
    return sorted(pool)
