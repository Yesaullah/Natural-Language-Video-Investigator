"""Tests for deterministic two-video synthetic index generator."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.make_stub_index import SYNTHETIC_DIM, make_stub_indexes


def test_make_stub_index_end_to_end(tmp_path: Path) -> None:
    spoken_dir, visual_dir = make_stub_indexes(tmp_path)

    assert spoken_dir.exists()
    assert visual_dir.exists()

    # 1. Verify Video 1: stub-spoken
    db_spoken = spoken_dir / "index.sqlite"
    assert db_spoken.exists()

    conn = sqlite3.connect(db_spoken)
    cur = conn.cursor()

    # Verify tables exist
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"video", "transcript_units", "frames", "caption_units"}.issubset(tables)

    # Check video table
    v_rows = cur.execute("SELECT video_id, path, duration_s, dataset, split FROM video").fetchall()
    assert len(v_rows) == 1
    assert v_rows[0][0] == "stub-spoken"
    assert v_rows[0][2] == 30.0

    # Check transcript units
    t_rows = cur.execute("SELECT unit_id, start_s, end_s, text FROM transcript_units ORDER BY unit_id").fetchall()
    assert len(t_rows) == 3
    # Contiguous zero-based IDs
    assert [r[0] for r in t_rows] == [0, 1, 2]
    # Check non-overlapping half-open intervals
    for i in range(len(t_rows) - 1):
        assert t_rows[i][2] <= t_rows[i + 1][1]

    # Check frames table and image files
    f_rows = cur.execute("SELECT frame_id, t_s, jpg_path FROM frames ORDER BY frame_id").fetchall()
    assert len(f_rows) == 6
    assert [r[0] for r in f_rows] == list(range(6))
    for fid, t_s, jpg_rel in f_rows:
        img_path = spoken_dir / jpg_rel
        assert img_path.exists(), f"Image file {img_path} missing"
        with Image.open(img_path) as img:
            assert img.format == "JPEG"
            assert img.size == (64, 64)

    # Check captions table (overlapping captions)
    c_rows = cur.execute("SELECT unit_id, start_s, end_s, text FROM caption_units ORDER BY unit_id").fetchall()
    assert len(c_rows) == 5
    assert [r[0] for r in c_rows] == list(range(5))
    # Confirm captions overlap: e.g. caption 0 ends at 6.0 while caption 1 starts at 4.0
    assert c_rows[0][2] > c_rows[1][1]

    conn.close()

    # Check numpy arrays for stub-spoken
    t_arr = np.load(spoken_dir / "transcript.npy")
    f_arr = np.load(spoken_dir / "frames.npy")
    c_arr = np.load(spoken_dir / "captions.npy")

    assert t_arr.dtype == np.float32
    assert f_arr.dtype == np.float32
    assert c_arr.dtype == np.float32

    assert t_arr.shape == (len(t_rows), SYNTHETIC_DIM)
    assert f_arr.shape == (len(f_rows), SYNTHETIC_DIM)
    assert c_arr.shape == (len(c_rows), SYNTHETIC_DIM)

    # Check unit norms
    t_norms = np.linalg.norm(t_arr, axis=1)
    f_norms = np.linalg.norm(f_arr, axis=1)
    c_norms = np.linalg.norm(c_arr, axis=1)
    assert np.allclose(t_norms, 1.0, atol=1e-5)
    assert np.allclose(f_norms, 1.0, atol=1e-5)
    assert np.allclose(c_norms, 1.0, atol=1e-5)

    # Check meta.json
    with open(spoken_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["video_id"] == "stub-spoken"
    assert meta["synthetic"] is True
    assert "models" in meta
    assert "revisions" in meta
    assert "timings_s" in meta
    assert "peak_memory_mib" in meta

    # 2. Verify Video 2: stub-visual
    db_visual = visual_dir / "index.sqlite"
    assert db_visual.exists()

    conn_v = sqlite3.connect(db_visual)
    cur_v = conn_v.cursor()

    # Empty transcript table
    v_t_rows = cur_v.execute("SELECT * FROM transcript_units").fetchall()
    assert len(v_t_rows) == 0, "stub-visual must have zero transcript rows"

    # Frames and captions present
    v_f_rows = cur_v.execute("SELECT frame_id, t_s, jpg_path FROM frames ORDER BY frame_id").fetchall()
    assert len(v_f_rows) == 5
    for _, _, jpg_rel in v_f_rows:
        img_path = visual_dir / jpg_rel
        assert img_path.exists()
        with Image.open(img_path) as img:
            assert img.format == "JPEG"

    v_c_rows = cur_v.execute("SELECT * FROM caption_units").fetchall()
    assert len(v_c_rows) == 3

    conn_v.close()

    # Array for empty transcript
    v_t_arr = np.load(visual_dir / "transcript.npy")
    assert v_t_arr.dtype == np.float32
    assert v_t_arr.shape == (0, SYNTHETIC_DIM), f"Expected shape (0, {SYNTHETIC_DIM}), got {v_t_arr.shape}"

    v_f_arr = np.load(visual_dir / "frames.npy")
    assert v_f_arr.shape == (len(v_f_rows), SYNTHETIC_DIM)
    assert np.allclose(np.linalg.norm(v_f_arr, axis=1), 1.0, atol=1e-5)


def test_stub_index_repeatability(tmp_path: Path) -> None:
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    make_stub_indexes(dir1)
    make_stub_indexes(dir2)

    for vid in ["stub-spoken", "stub-visual"]:
        for arr_name in ["transcript.npy", "frames.npy", "captions.npy"]:
            arr1 = np.load(dir1 / vid / arr_name)
            arr2 = np.load(dir2 / vid / arr_name)
            np.testing.assert_array_equal(arr1, arr2)


def test_make_stub_index_cli_twice_in_same_dir(tmp_path: Path) -> None:
    # Exercise scripts/make_stub_index.py through its CLI twice in the same directory
    output_dir = tmp_path / "cli_test_indexes"

    cmd = [sys.executable, "scripts/make_stub_index.py", "--output-dir", str(output_dir)]

    res1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Generated synthetic indexes" in res1.stdout

    res2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Generated synthetic indexes" in res2.stdout

    # Check both persisted indexes
    spoken_dir = output_dir / "stub-spoken"
    visual_dir = output_dir / "stub-visual"

    assert spoken_dir.exists() and visual_dir.exists()

    # Check stub-spoken SQLite rows
    conn_s = sqlite3.connect(spoken_dir / "index.sqlite")
    cur_s = conn_s.cursor()
    assert len(cur_s.execute("SELECT * FROM video").fetchall()) == 1
    assert len(cur_s.execute("SELECT * FROM transcript_units").fetchall()) == 3
    assert len(cur_s.execute("SELECT * FROM frames").fetchall()) == 6
    assert len(cur_s.execute("SELECT * FROM caption_units").fetchall()) == 5
    conn_s.close()

    # Check stub-spoken arrays
    t_s = np.load(spoken_dir / "transcript.npy")
    f_s = np.load(spoken_dir / "frames.npy")
    c_s = np.load(spoken_dir / "captions.npy")
    assert t_s.shape == (3, SYNTHETIC_DIM) and t_s.dtype == np.float32
    assert f_s.shape == (6, SYNTHETIC_DIM) and f_s.dtype == np.float32
    assert c_s.shape == (5, SYNTHETIC_DIM) and c_s.dtype == np.float32
    assert np.allclose(np.linalg.norm(t_s, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(f_s, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(c_s, axis=1), 1.0, atol=1e-5)

    # Check stub-spoken metadata and JPEGs
    with open(spoken_dir / "meta.json", "r", encoding="utf-8") as f:
        meta_s = json.load(f)
    assert meta_s["synthetic"] is True
    for fid in range(6):
        jpg_p = spoken_dir / f"frames/f00{fid * 5:02d}.jpg"
        assert jpg_p.exists()
        with Image.open(jpg_p) as img:
            assert img.format == "JPEG"
            assert img.size == (64, 64)
            assert img.mode == "RGB"

    # Check stub-visual SQLite rows
    conn_v = sqlite3.connect(visual_dir / "index.sqlite")
    cur_v = conn_v.cursor()
    assert len(cur_v.execute("SELECT * FROM video").fetchall()) == 1
    assert len(cur_v.execute("SELECT * FROM transcript_units").fetchall()) == 0
    assert len(cur_v.execute("SELECT * FROM frames").fetchall()) == 5
    assert len(cur_v.execute("SELECT * FROM caption_units").fetchall()) == 3
    conn_v.close()

    # Check stub-visual arrays
    t_v = np.load(visual_dir / "transcript.npy")
    f_v = np.load(visual_dir / "frames.npy")
    c_v = np.load(visual_dir / "captions.npy")
    assert t_v.shape == (0, SYNTHETIC_DIM) and t_v.dtype == np.float32
    assert f_v.shape == (5, SYNTHETIC_DIM) and f_v.dtype == np.float32
    assert c_v.shape == (3, SYNTHETIC_DIM) and c_v.dtype == np.float32
    assert np.allclose(np.linalg.norm(f_v, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(c_v, axis=1), 1.0, atol=1e-5)

    # Check stub-visual metadata and JPEGs
    with open(visual_dir / "meta.json", "r", encoding="utf-8") as f:
        meta_v = json.load(f)
    assert meta_v["synthetic"] is True
    for fid in range(5):
        jpg_p = visual_dir / f"frames/f00{fid * 4:02d}.jpg"
        assert jpg_p.exists()
        with Image.open(jpg_p) as img:
            assert img.format == "JPEG"
            assert img.size == (64, 64)
            assert img.mode == "RGB"
