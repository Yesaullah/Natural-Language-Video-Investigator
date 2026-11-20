"""Acceptance tests for M2 video split manifests and Charades-STA sampling.

Conforms to:
- plans/M2-data-manifests.md
- video_investigator_plan.md §§4.1–4.3, 8.3 (unit M2)
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

from vi.manifest import (
    load_team_inventory_csv,
    parse_charades_inventory,
    sample_charades_slice,
    serialize_manifest_json,
    split_team_videos,
    verify_manifest_integrity,
    write_manifest_atomic,
    CATEGORY_ALLOCATIONS,
    CATEGORY_ORDER,
)
from vi.schema import ManifestRecord, SplitManifest, TeamVideoRecord


def _make_synthetic_charades_data() -> Tuple[List[str], List[str]]:
    """Create synthetic Charades annotations with repeated IDs and >=60 unique IDs per population."""
    # 75 unique train IDs with repeated annotations (total 200 rows) using official '##' format
    train_lines: List[str] = []
    for i in range(75):
        vid = f"charades_train_{i:04d}"
        train_lines.append(f"{vid} 0.0 4.5##person holding cup")
        if i % 2 == 0:
            train_lines.append(f"{vid} 4.0 8.0##person drinking water, looking around")
        if i % 3 == 0:
            train_lines.append(f"{vid} 8.0 12.0##person putting cup down")

    # 75 unique test IDs with repeated annotations (total 200 rows), zero overlap with train
    test_lines: List[str] = []
    for i in range(75):
        vid = f"charades_test_{i:04d}"
        test_lines.append(f"{vid} 1.0 5.0##person opening refrigerator")
        if i % 2 == 0:
            test_lines.append(f"{vid} 5.0 9.5##person taking bottle, closing door")
        if i % 3 == 0:
            test_lines.append(f"{vid} 9.5 14.0##person walking away")

    return train_lines, test_lines


def _make_synthetic_team_csv(tmp_path: Path, filename: str = "videos.csv") -> Path:
    """Create synthetic team inventory CSV with 30 distinct videos (10 per category)."""
    p = tmp_path / filename
    rows: List[dict] = []
    for i in range(10):
        rows.append({
            "video_id": f"tl-speech-{i+1:02d}",
            "category": "Speech-heavy",
            "source": f"https://example.com/speech/{i+1}",
            "licence": "CC-BY-4.0",
        })
    for i in range(10):
        rows.append({
            "video_id": f"tl-mixed-{i+1:02d}",
            "category": "Mixed",
            "source": f"https://example.com/mixed/{i+1}",
            "licence": "CC-BY-SA-4.0",
        })
    for i in range(10):
        rows.append({
            "video_id": f"tl-action-{i+1:02d}",
            "category": "Action-heavy",
            "source": f"https://example.com/action/{i+1}",
            "licence": "CC0-1.0",
        })

    with open(p, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "category", "source", "licence"])
        writer.writeheader()
        writer.writerows(rows)
    return p


class TestCharadesSampling:
    def test_parse_charades_inventory_deduplicates(self, tmp_path: Path):
        train_lines, _ = _make_synthetic_charades_data()
        p = tmp_path / "train_ann.txt"
        p.write_text("\n".join(train_lines), encoding="utf-8")

        unique_ids = parse_charades_inventory(p)
        assert len(unique_ids) == 75
        assert len(set(unique_ids)) == 75
        assert unique_ids == sorted(unique_ids)

    def test_parse_charades_comma_in_query_regression(self):
        """Two annotation rows for same ID, one with comma in query, returns exactly one ID."""
        rows = [
            "008DJ 0.0 5.4##a person is cooking, washing hands",
            "008DJ 5.4 10.0##a person is eating",
        ]
        unique_ids = parse_charades_inventory(rows)
        assert unique_ids == ["008DJ"]
        assert len(unique_ids) == 1

    def test_sample_charades_slice_quotas_and_disjointness(self):
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        manifest = sample_charades_slice(train_ids, test_ids, seed=20261114)
        assert manifest.dataset == "charades-sta"
        assert manifest.seed == 20261114
        assert manifest.split_counts == {"dev": 30, "cal": 30, "test": 60, "total": 120}

        dev_ids = {r.video_id for r in manifest.records if r.split == "dev"}
        cal_ids = {r.video_id for r in manifest.records if r.split == "cal"}
        test_ids_out = {r.video_id for r in manifest.records if r.split == "test"}

        assert len(dev_ids) == 30
        assert len(cal_ids) == 30
        assert len(test_ids_out) == 60

        # Disjointness
        assert len(dev_ids & cal_ids) == 0
        assert len(dev_ids & test_ids_out) == 0
        assert len(cal_ids & test_ids_out) == 0

        # Source split verification
        train_set = set(train_ids)
        test_set = set(test_ids)
        assert dev_ids.issubset(train_set)
        assert cal_ids.issubset(train_set)
        assert test_ids_out.issubset(test_set)

    def test_charades_reproducibility_and_input_order_invariance(self, tmp_path: Path):
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        # Run 1: Normal
        m1 = sample_charades_slice(train_ids, test_ids, seed=20261114)
        out1, h1 = write_manifest_atomic(m1, tmp_path / "m1.json")

        # Run 2: Reversed input order
        m2 = sample_charades_slice(list(reversed(train_ids)), list(reversed(test_ids)), seed=20261114)
        out2, h2 = write_manifest_atomic(m2, tmp_path / "m2.json")

        # Run 3: Arbitrary shuffle of input order
        import random
        shuf_train = list(train_ids)
        shuf_test = list(test_ids)
        random.Random(42).shuffle(shuf_train)
        random.Random(42).shuffle(shuf_test)
        m3 = sample_charades_slice(shuf_train, shuf_test, seed=20261114)
        out3, h3 = write_manifest_atomic(m3, tmp_path / "m3.json")

        assert h1 == h2 == h3
        assert out1.read_bytes() == out2.read_bytes() == out3.read_bytes()

    def test_charades_insufficient_population(self):
        train_ids = [f"vid_train_{i}" for i in range(59)]  # Need 60
        test_ids = [f"vid_test_{i}" for i in range(70)]
        with pytest.raises(ValueError, match="Insufficient unique Charades train videos"):
            sample_charades_slice(train_ids, test_ids, seed=20261114)

        train_ids_ok = [f"vid_train_{i}" for i in range(60)]
        test_ids_short = [f"vid_test_{i}" for i in range(59)]  # Need 60
        with pytest.raises(ValueError, match="Insufficient unique Charades test videos"):
            sample_charades_slice(train_ids_ok, test_ids_short, seed=20261114)

    def test_charades_source_overlap_rejected(self):
        train_ids = [f"vid_{i}" for i in range(70)]
        test_ids = [f"vid_{i}" for i in range(65, 135)]  # Overlaps 65-69
        with pytest.raises(ValueError, match="Source train and test populations overlap"):
            sample_charades_slice(train_ids, test_ids, seed=20261114)


class TestTeamVideoSplit:
    def test_load_team_inventory_validates_counts(self, tmp_path: Path):
        csv_path = _make_synthetic_team_csv(tmp_path)
        records = load_team_inventory_csv(csv_path)
        assert len(records) == 30
        assert len({r.video_id for r in records}) == 30

    def test_team_split_quotas_and_category_allocation(self, tmp_path: Path):
        csv_path = _make_synthetic_team_csv(tmp_path)
        records = load_team_inventory_csv(csv_path)

        manifest = split_team_videos(records, seed=20261114)
        assert manifest.dataset == "team"
        assert manifest.seed == 20261114
        assert manifest.split_counts == {"dev": 10, "cal": 8, "test": 12, "total": 30}

        dev_ids = {r.video_id for r in manifest.records if r.split == "dev"}
        cal_ids = {r.video_id for r in manifest.records if r.split == "cal"}
        test_ids = {r.video_id for r in manifest.records if r.split == "test"}

        # Disjointness
        assert len(dev_ids & cal_ids) == 0
        assert len(dev_ids & test_ids) == 0
        assert len(cal_ids & test_ids) == 0

        # Category allocations match Table in M2 brief
        # Speech-heavy: 4 dev, 2 cal, 4 test
        # Mixed: 3 dev, 3 cal, 4 test
        # Action-heavy: 3 dev, 3 cal, 4 test
        for cat, expected in CATEGORY_ALLOCATIONS.items():
            cat_recs = [r for r in manifest.records if r.category == cat]
            assert len(cat_recs) == 10
            assert sum(r.split == "dev" for r in cat_recs) == expected["dev"]
            assert sum(r.split == "cal" for r in cat_recs) == expected["cal"]
            assert sum(r.split == "test" for r in cat_recs) == expected["test"]

        # Preserved source and licence
        for r in manifest.records:
            assert r.source.startswith("https://example.com/")
            assert r.licence in {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}

    def test_team_reproducibility_and_input_order_invariance(self, tmp_path: Path):
        csv_path = _make_synthetic_team_csv(tmp_path)
        records1 = load_team_inventory_csv(csv_path)

        # Reverse CSV rows
        p_rev = tmp_path / "videos_rev.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = lines[0]
        data_rows = lines[1:]
        with open(p_rev, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.writelines(reversed(data_rows))

        records2 = load_team_inventory_csv(p_rev)

        m1 = split_team_videos(records1, seed=20261114)
        out1, h1 = write_manifest_atomic(m1, tmp_path / "team1.json")

        m2 = split_team_videos(records2, seed=20261114)
        out2, h2 = write_manifest_atomic(m2, tmp_path / "team2.json")

        assert h1 == h2
        assert out1.read_bytes() == out2.read_bytes()

    def test_team_duplicate_id_rejected(self, tmp_path: Path):
        p = tmp_path / "dup.csv"
        rows = [
            {"video_id": f"vid-{i}", "category": "Speech-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ] + [
            {"video_id": f"vid-{i}", "category": "Mixed", "source": "s", "licence": "l"}
            for i in range(10)  # Duplicate IDs from first category!
        ] + [
            {"video_id": f"vid-act-{i}", "category": "Action-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "category", "source", "licence"])
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="Duplicate video_id"):
            load_team_inventory_csv(p)

    def test_team_blank_id_rejected(self, tmp_path: Path):
        p = tmp_path / "blank.csv"
        rows = [
            {"video_id": "" if i == 0 else f"vid-sp-{i}", "category": "Speech-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ] + [
            {"video_id": f"vid-mx-{i}", "category": "Mixed", "source": "s", "licence": "l"}
            for i in range(10)
        ] + [
            {"video_id": f"vid-act-{i}", "category": "Action-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "category", "source", "licence"])
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="empty or blank video_id"):
            load_team_inventory_csv(p)

    def test_team_unknown_category_rejected(self, tmp_path: Path):
        p = tmp_path / "badcat.csv"
        rows = [
            {"video_id": f"vid-sp-{i}", "category": "Vlog" if i == 0 else "Speech-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ] + [
            {"video_id": f"vid-mx-{i}", "category": "Mixed", "source": "s", "licence": "l"}
            for i in range(10)
        ] + [
            {"video_id": f"vid-act-{i}", "category": "Action-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "category", "source", "licence"])
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="Unknown category 'Vlog'"):
            load_team_inventory_csv(p)

    def test_team_incorrect_counts_rejected(self, tmp_path: Path):
        p = tmp_path / "badcounts.csv"
        # 11 Speech-heavy, 9 Mixed, 10 Action-heavy (total 30)
        rows = [
            {"video_id": f"vid-sp-{i}", "category": "Speech-heavy", "source": "s", "licence": "l"}
            for i in range(11)
        ] + [
            {"video_id": f"vid-mx-{i}", "category": "Mixed", "source": "s", "licence": "l"}
            for i in range(9)
        ] + [
            {"video_id": f"vid-act-{i}", "category": "Action-heavy", "source": "s", "licence": "l"}
            for i in range(10)
        ]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "category", "source", "licence"])
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="Team inventory must have exactly 10 videos in 'Speech-heavy'"):
            load_team_inventory_csv(p)


class TestAtomicWriteAndPreservation:
    def test_failed_rerun_preserves_valid_manifest(self, tmp_path: Path):
        """Failed rerun must preserve previous valid output bytes and SHA-256."""
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        manifest_path = tmp_path / "manifest.json"
        m_valid = sample_charades_slice(train_ids, test_ids, seed=20261114)
        out, original_hash = write_manifest_atomic(m_valid, manifest_path)
        original_bytes = manifest_path.read_bytes()

        # Run with invalid data targeting the same output file
        short_train = train_ids[:20]  # Too short!
        with pytest.raises(ValueError):
            sample_charades_slice(short_train, test_ids, seed=20261114)

        # Assert manifest at manifest_path is completely preserved
        assert manifest_path.exists()
        assert manifest_path.read_bytes() == original_bytes
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == original_hash

    def test_manifest_record_removed_rejected_before_publication(self, tmp_path: Path):
        """Manifest with one record removed and unchanged count fields is rejected before publication."""
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        manifest_path = tmp_path / "valid_manifest.json"
        valid_manifest = sample_charades_slice(train_ids, test_ids, seed=20261114)
        out, original_hash = write_manifest_atomic(valid_manifest, manifest_path)
        original_bytes = manifest_path.read_bytes()

        # Remove 1 record from records, keeping declared split_counts unchanged
        corrupted_manifest = valid_manifest.model_copy(deep=True)
        corrupted_manifest.records = corrupted_manifest.records[:-1]
        assert len(corrupted_manifest.records) == 119
        assert corrupted_manifest.split_counts["total"] == 120

        with pytest.raises(ValueError, match="does not match records count|quota mismatch"):
            verify_manifest_integrity(corrupted_manifest)

        with pytest.raises(ValueError, match="does not match records count|quota mismatch"):
            write_manifest_atomic(corrupted_manifest, manifest_path)

        # File on disk remains intact and completely uncorrupted
        assert manifest_path.read_bytes() == original_bytes
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == original_hash

    def test_manifest_record_dataset_mismatch_rejected(self):
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        valid_manifest = sample_charades_slice(train_ids, test_ids, seed=20261114)
        corrupted = valid_manifest.model_copy(deep=True)
        corrupted.records[0].dataset = "other-dataset"

        with pytest.raises(ValueError, match="Record dataset 'other-dataset' does not match manifest dataset"):
            verify_manifest_integrity(corrupted)

    def test_team_manifest_record_removed_rejected_before_publication(self, tmp_path: Path):
        csv_file = _make_synthetic_team_csv(tmp_path)
        team_records = load_team_inventory_csv(csv_file)
        manifest_path = tmp_path / "team_manifest.json"
        valid_manifest = split_team_videos(team_records, seed=20261114)
        write_manifest_atomic(valid_manifest, manifest_path)
        original_bytes = manifest_path.read_bytes()

        corrupted = valid_manifest.model_copy(deep=True)
        corrupted.records = corrupted.records[:-1]
        assert len(corrupted.records) == 29
        assert corrupted.split_counts["total"] == 30

        with pytest.raises(ValueError, match="does not match records count|quota mismatch"):
            verify_manifest_integrity(corrupted)

        with pytest.raises(ValueError, match="does not match records count|quota mismatch"):
            write_manifest_atomic(corrupted, manifest_path)

        assert manifest_path.read_bytes() == original_bytes


class TestDatasetStreamIndependence:
    def test_separate_deterministic_streams(self, tmp_path: Path):
        """Running either alone gives the exact same result as running both."""
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        csv_path = _make_synthetic_team_csv(tmp_path)
        team_records = load_team_inventory_csv(csv_path)

        # Charades alone
        cs_alone = sample_charades_slice(train_ids, test_ids, seed=20261114)
        _, cs_alone_hash = write_manifest_atomic(cs_alone, tmp_path / "cs_alone.json")

        # Team alone
        team_alone = split_team_videos(team_records, seed=20261114)
        _, team_alone_hash = write_manifest_atomic(team_alone, tmp_path / "team_alone.json")

        # Both together in succession (as sample_cs_slice.py --dataset both does)
        cs_both = sample_charades_slice(train_ids, test_ids, seed=20261114)
        team_both = split_team_videos(team_records, seed=20261114)
        _, cs_both_hash = write_manifest_atomic(cs_both, tmp_path / "cs_both.json")
        _, team_both_hash = write_manifest_atomic(team_both, tmp_path / "team_both.json")

        assert cs_alone_hash == cs_both_hash
        assert team_alone_hash == team_both_hash


class TestCLIExecution:
    def test_cli_charades_sampling(self, tmp_path: Path):
        train_lines, test_lines = _make_synthetic_charades_data()
        tr_file = tmp_path / "cs_train.txt"
        te_file = tmp_path / "cs_test.txt"
        out_file = tmp_path / "cs_manifest.json"

        tr_file.write_text("\n".join(train_lines), encoding="utf-8")
        te_file.write_text("\n".join(test_lines), encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/sample_cs_slice.py",
            "--seed", "20261114",
            "--train-input", str(tr_file),
            "--test-input", str(te_file),
            "--output", str(out_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Charades-STA Slice Manifest Created" in res.stdout
        assert "VERIFIED (0 overlaps)" in res.stdout
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["split_counts"]["total"] == 120
        assert data["seed"] == 20261114

    def test_cli_team_split(self, tmp_path: Path):
        csv_file = _make_synthetic_team_csv(tmp_path)
        out_file = tmp_path / "tl_manifest.json"

        cmd = [
            sys.executable,
            "scripts/split_team_videos.py",
            "--seed", "20261114",
            "--input", str(csv_file),
            "--output", str(out_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Team Video Split Manifest Created" in res.stdout
        assert "VERIFIED (0 overlaps)" in res.stdout
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["split_counts"]["total"] == 30
        assert data["split_counts"]["dev"] == 10
        assert data["split_counts"]["cal"] == 8
        assert data["split_counts"]["test"] == 12

    def test_cli_both_in_sample_cs_slice(self, tmp_path: Path):
        train_lines, test_lines = _make_synthetic_charades_data()
        tr_file = tmp_path / "cs_train.txt"
        te_file = tmp_path / "cs_test.txt"
        cs_out = tmp_path / "cs_manifest.json"
        tr_file.write_text("\n".join(train_lines), encoding="utf-8")
        te_file.write_text("\n".join(test_lines), encoding="utf-8")

        csv_file = _make_synthetic_team_csv(tmp_path)
        tl_out = tmp_path / "tl_manifest.json"

        cmd = [
            sys.executable,
            "scripts/sample_cs_slice.py",
            "--seed", "20261114",
            "--dataset", "both",
            "--train-input", str(tr_file),
            "--test-input", str(te_file),
            "--output", str(cs_out),
            "--team-input", str(csv_file),
            "--team-output", str(tl_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Charades-STA Slice Manifest Created" in res.stdout
        assert "Team Video Split Manifest Created" in res.stdout
        assert cs_out.exists()
        assert tl_out.exists()

    def test_cli_missing_input_actionable_message(self, tmp_path: Path):
        cmd = [
            sys.executable,
            "scripts/sample_cs_slice.py",
            "--train-input", str(tmp_path / "nonexistent_train.txt"),
            "--test-input", str(tmp_path / "nonexistent_test.txt"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Missing Charades train input file" in res.stderr
        assert "Expected input format" in res.stderr

        cmd_team = [
            sys.executable,
            "scripts/split_team_videos.py",
            "--input", str(tmp_path / "nonexistent_videos.csv"),
        ]
        res_team = subprocess.run(cmd_team, capture_output=True, text=True)
        assert res_team.returncode == 1
        assert "Missing Team video inventory file" in res_team.stderr
        assert "Expected input format" in res_team.stderr

    def test_cli_both_mode_atomic_failure_preserves_both_outputs(self, tmp_path: Path):
        """Combined mode: invalid team inventory with new seed must not overwrite existing valid Charades or Team manifests."""
        # 1. Setup valid inputs
        train_lines, test_lines = _make_synthetic_charades_data()
        tr_file = tmp_path / "cs_train.txt"
        te_file = tmp_path / "cs_test.txt"
        cs_out = tmp_path / "cs_manifest.json"
        tr_file.write_text("\n".join(train_lines), encoding="utf-8")
        te_file.write_text("\n".join(test_lines), encoding="utf-8")

        valid_csv_file = _make_synthetic_team_csv(tmp_path, "valid_team.csv")
        tl_out = tmp_path / "tl_manifest.json"

        # 2. Run initial combined run with seed 20261114 to establish valid manifests
        cmd_init = [
            sys.executable,
            "scripts/sample_cs_slice.py",
            "--seed", "20261114",
            "--dataset", "both",
            "--train-input", str(tr_file),
            "--test-input", str(te_file),
            "--output", str(cs_out),
            "--team-input", str(valid_csv_file),
            "--team-output", str(tl_out),
        ]
        res_init = subprocess.run(cmd_init, capture_output=True, text=True, check=True)
        assert res_init.returncode == 0
        assert cs_out.exists()
        assert tl_out.exists()

        cs_orig_bytes = cs_out.read_bytes()
        tl_orig_bytes = tl_out.read_bytes()

        # 3. Create an invalid team inventory (only 1 row instead of 30)
        invalid_csv_file = tmp_path / "invalid_team.csv"
        invalid_csv_file.write_text(
            "video_id,category,source,licence\n"
            "bad-1,Speech-heavy,s,l\n",
            encoding="utf-8",
        )

        # 4. Rerun in combined mode with a different seed and invalid team inventory
        different_seed = "999999"
        cmd_rerun = [
            sys.executable,
            "scripts/sample_cs_slice.py",
            "--seed", different_seed,
            "--dataset", "both",
            "--train-input", str(tr_file),
            "--test-input", str(te_file),
            "--output", str(cs_out),
            "--team-input", str(invalid_csv_file),
            "--team-output", str(tl_out),
        ]
        res_rerun = subprocess.run(cmd_rerun, capture_output=True, text=True)

        # 5. Assert nonzero exit code and byte-for-byte preservation of both outputs
        assert res_rerun.returncode != 0
        assert cs_out.read_bytes() == cs_orig_bytes
        assert tl_out.read_bytes() == tl_orig_bytes


class TestLabelCustody:
    def test_manifest_preserves_test_label_custody(self, tmp_path: Path):
        """Manifest contains only video IDs and split allocations; no queries or labels."""
        train_lines, test_lines = _make_synthetic_charades_data()
        train_ids = parse_charades_inventory(train_lines)
        test_ids = parse_charades_inventory(test_lines)

        cs_manifest = sample_charades_slice(train_ids, test_ids, seed=20261114)
        cs_json = json.loads(serialize_manifest_json(cs_manifest))

        # Check records do not contain query text, gold windows, or negative labels
        for rec in cs_json["records"]:
            assert "query" not in rec
            assert "gold_windows" not in rec
            assert "neg_type" not in rec
            assert "qtype" not in rec
            assert set(rec.keys()).issubset({"dataset", "video_id", "split", "source_split"})

        csv_file = _make_synthetic_team_csv(tmp_path)
        team_records = load_team_inventory_csv(csv_file)
        team_manifest = split_team_videos(team_records, seed=20261114)
        team_json = json.loads(serialize_manifest_json(team_manifest))

        for rec in team_json["records"]:
            assert "query" not in rec
            assert "gold_windows" not in rec
            assert "neg_type" not in rec
            assert "qtype" not in rec
            assert set(rec.keys()).issubset({"dataset", "video_id", "split", "category", "source", "licence"})
