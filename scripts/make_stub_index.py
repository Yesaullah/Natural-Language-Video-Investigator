"""Generate deterministic two-video synthetic index fixtures for M1.

Creates index/stub-spoken/ and index/stub-visual/ (or under a specified output directory).
Conforms to video_investigator_plan.md §8.1 and plans/M1-evaluation-foundation.md §2.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Synthetic embedding dimension for text and frame vectors
SYNTHETIC_DIM = 16


def _create_unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    """Generate deterministic float32 L2-normalized unit vectors."""
    if count == 0:
        return np.empty((0, dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size=(count, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    # Avoid zero division
    norms[norms == 0] = 1.0
    unit = raw / norms
    return unit.astype(np.float32)


def _write_synthetic_jpeg(output_path: Path, color: Tuple[int, int, int]) -> None:
    """Create a genuine, readable JPEG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(output_path, format="JPEG", quality=90)


def build_stub_index_video(
    video_dir: Path,
    video_id: str,
    duration_s: float,
    transcript_rows: List[Tuple[int, float, float, str]],
    frame_rows: List[Tuple[int, float, str, Tuple[int, int, int]]],
    caption_rows: List[Tuple[int, float, float, str]],
    seed_offset: int,
) -> None:
    """Build a complete synthetic index folder for a single video."""
    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. SQLite database
    db_path = video_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE video (
            video_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            duration_s REAL NOT NULL,
            sha256 TEXT NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE transcript_units (
            unit_id INTEGER PRIMARY KEY,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            text TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE frames (
            frame_id INTEGER PRIMARY KEY,
            t_s REAL NOT NULL,
            jpg_path TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE caption_units (
            unit_id INTEGER PRIMARY KEY,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            text TEXT NOT NULL
        )
    """)

    cur.execute(
        "INSERT INTO video VALUES (?, ?, ?, ?, ?, ?)",
        (
            video_id,
            f"synthetic/{video_id}.mp4",
            duration_s,
            f"synthetic_sha256_{video_id}",
            "synthetic",
            "synthetic",
        ),
    )

    for row in transcript_rows:
        cur.execute("INSERT INTO transcript_units VALUES (?, ?, ?, ?)", row)

    for fid, t_s, jpg_rel_path, color in frame_rows:
        cur.execute("INSERT INTO frames VALUES (?, ?, ?)", (fid, t_s, jpg_rel_path))
        # Write genuine JPEG
        full_jpg_path = video_dir / jpg_rel_path
        _write_synthetic_jpeg(full_jpg_path, color)

    for row in caption_rows:
        cur.execute("INSERT INTO caption_units VALUES (?, ?, ?, ?)", row)

    conn.commit()
    conn.close()

    # 2. NumPy arrays
    transcript_vecs = _create_unit_vectors(len(transcript_rows), SYNTHETIC_DIM, seed=100 + seed_offset)
    frames_vecs = _create_unit_vectors(len(frame_rows), SYNTHETIC_DIM, seed=200 + seed_offset)
    captions_vecs = _create_unit_vectors(len(caption_rows), SYNTHETIC_DIM, seed=300 + seed_offset)

    np.save(video_dir / "transcript.npy", transcript_vecs)
    np.save(video_dir / "frames.npy", frames_vecs)
    np.save(video_dir / "captions.npy", captions_vecs)

    # 3. meta.json
    meta = {
        "video_id": video_id,
        "synthetic": True,
        "models": {
            "whisper": "synthetic-whisper-turbo",
            "siglip": "synthetic-siglip2-so400m-patch16-384",
            "caption": "synthetic-qwen3-vl-4b-instruct",
            "text_embed": "synthetic-qwen3-embedding-0.6b",
        },
        "revisions": {
            "whisper": "synthetic-rev",
            "siglip": "synthetic-rev",
            "caption": "synthetic-rev",
            "text_embed": "synthetic-rev",
        },
        "library_versions": {
            "python": "3.13-synthetic",
            "torch": "synthetic",
            "transformers": "synthetic",
        },
        "config_hash": "synthetic-config-hash-v1",
        "timings_s": {
            "transcribe": 0.0,
            "embed_frames": 0.0,
            "caption": 0.0,
            "embed_text": 0.0,
        },
        "peak_memory_mib": {
            "transcribe": 0.0,
            "embed_frames": 0.0,
            "caption": 0.0,
            "query": 0.0,
        },
        "synthetic_dimension": SYNTHETIC_DIM,
    }
    with open(video_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def make_stub_indexes(output_base: Path | str = "index") -> Tuple[Path, Path]:
    """Generate both deterministic synthetic indexes.

    - Video 1 (stub-spoken): speech, frames and overlapping captions.
    - Video 2 (stub-visual): frames and captions with zero transcript rows
      and a transcript array of shape (0, SYNTHETIC_DIM).
    """
    base_dir = Path(output_base)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Video 1: stub-spoken (duration: 30.0s)
    spoken_dir = base_dir / "stub-spoken"
    spoken_transcripts = [
        (0, 0.0, 8.0, "Welcome to the seminar on natural language video search."),
        (1, 8.0, 18.0, "We use multimodal models to index spoken content and visual moments."),
        (2, 18.0, 28.0, "Notice how temporal IoU evaluates candidate window overlap."),
    ]
    spoken_frames = [
        (0, 0.0, "frames/f0000.jpg", (50, 100, 150)),
        (1, 5.0, "frames/f0005.jpg", (60, 110, 160)),
        (2, 10.0, "frames/f0010.jpg", (70, 120, 170)),
        (3, 15.0, "frames/f0015.jpg", (80, 130, 180)),
        (4, 20.0, "frames/f0020.jpg", (90, 140, 190)),
        (5, 25.0, "frames/f0025.jpg", (100, 150, 200)),
    ]
    # Overlapping captions (e.g. stride 4s with 6s duration)
    spoken_captions = [
        (0, 0.0, 6.0, "A presenter stands beside a projector screen presenting slides."),
        (1, 4.0, 10.0, "The presenter points to a diagram on video search architecture."),
        (2, 8.0, 14.0, "Close-up of text on the slide explaining multimodal grounding."),
        (3, 14.0, 22.0, "The speaker demonstrates query processing on a laptop."),
        (4, 20.0, 28.0, "The speaker writes notes on a digital white board."),
    ]
    build_stub_index_video(
        video_dir=spoken_dir,
        video_id="stub-spoken",
        duration_s=30.0,
        transcript_rows=spoken_transcripts,
        frame_rows=spoken_frames,
        caption_rows=spoken_captions,
        seed_offset=1,
    )

    # Video 2: stub-visual (duration: 20.0s)
    # Zero transcript rows, transcript.npy of shape (0, SYNTHETIC_DIM)
    visual_dir = base_dir / "stub-visual"
    visual_transcripts: List[Tuple[int, float, float, str]] = []
    visual_frames = [
        (0, 0.0, "frames/f0000.jpg", (200, 50, 50)),
        (1, 4.0, "frames/f0004.jpg", (210, 60, 60)),
        (2, 8.0, "frames/f0008.jpg", (220, 70, 70)),
        (3, 12.0, "frames/f0012.jpg", (230, 80, 80)),
        (4, 16.0, "frames/f0016.jpg", (240, 90, 90)),
    ]
    visual_captions = [
        (0, 0.0, 5.0, "A craftsman carves intricate patterns into a block of wood."),
        (1, 4.0, 11.0, "Wood shavings fall onto the workbench as the chisel cuts."),
        (2, 10.0, 18.0, "The craftsman sands down the smooth wooden edges."),
    ]
    build_stub_index_video(
        video_dir=visual_dir,
        video_id="stub-visual",
        duration_s=20.0,
        transcript_rows=visual_transcripts,
        frame_rows=visual_frames,
        caption_rows=visual_captions,
        seed_offset=2,
    )

    return spoken_dir, visual_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic stub indexes for M1.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="index",
        help="Base output directory for stub indexes (default: index/)",
    )
    args = parser.parse_args()
    spoken_dir, visual_dir = make_stub_indexes(args.output_dir)
    print(f"Generated synthetic indexes at:\n  {spoken_dir}\n  {visual_dir}")


if __name__ == "__main__":
    main()

