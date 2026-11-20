"""Video split manifests and deterministic sampling for M2.

Implements the CPU-only data selection unit defined in:
- plans/M2-data-manifests.md
- video_investigator_plan.md §§4.1, 4.2, 4.3, 8.3 (unit M2)

Provides:
- Deterministic seeded sampling without replacement (seed 20261114).
- Separate deterministic random streams for Charades-STA and Team datasets.
- Disjoint split allocation and validation (dev, cal, test).
- Safe atomic manifest publishing preserving prior valid outputs on error.
- Preservation of test-label custody (shared manifests contain only split assignments and metadata).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from vi.schema import ManifestRecord, SplitManifest, TeamVideoRecord, VALID_CATEGORIES, VALID_SPLITS

# Sampling convention identifier recorded in manifest metadata (§8.3 M2, brief rule 1)
SAMPLING_CONVENTION: str = "sort_eligible_ids_before_seeded_shuffle_without_replacement"

# Canonical category order and allocations (§4.1, brief rule 3)
CATEGORY_ORDER: List[str] = ["Speech-heavy", "Mixed", "Action-heavy"]

CATEGORY_ALLOCATIONS: Dict[str, Dict[str, int]] = {
    "Speech-heavy": {"dev": 4, "cal": 2, "test": 4},
    "Mixed": {"dev": 3, "cal": 3, "test": 4},
    "Action-heavy": {"dev": 3, "cal": 3, "test": 4},
}

SPLIT_ORDER: Dict[str, int] = {"dev": 0, "cal": 1, "test": 2}

CHARADES_QUOTAS: Dict[str, int] = {
    "dev": 30,
    "cal": 30,
    "test": 60,
    "total": 120,
}

TEAM_TOTAL_VIDEOS: int = 30
TEAM_PER_CATEGORY: int = 10


def compute_sha256(data: bytes) -> str:
    """Compute hexadecimal SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def parse_charades_inventory(source: Union[str, Path, Iterable[str]]) -> List[str]:
    """Parse and deduplicate Charades-STA video IDs from file or line iterable.

    Supports:
    - Official Charades-STA annotation rows: '<video_id> <start> <end>##<query>'.
      The timestamp prefix is split from the query at '##' and the video ID is extracted
      from the prefix. A comma in the query never affects row parsing.
    - Local ID-only inventories (one video_id per line).
    - CSV/TSV format where video_id is the first column or column named 'video_id'.

    Deduplicates repeated IDs originating from multiple annotations.
    """
    lines: Iterable[str]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Charades inventory file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = source

    raw_ids: Set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "##" in stripped:
            # Official Charades-STA annotation format: '<video_id> <start> <end>##<query>'
            prefix = stripped.split("##", 1)[0].strip()
            token = prefix.split()[0]
        else:
            # ID-only inventory format (one ID per line or whitespace/comma-delimited)
            token = stripped.split()[0].strip(",'\"")

        if token and token.lower() not in {"video_id", "id"}:
            raw_ids.add(token)

    return sorted(raw_ids)


def load_team_inventory_csv(path: Union[str, Path]) -> List[TeamVideoRecord]:
    """Load and validate the local CSV inventory of 30 distinct team videos (§4.1, M2).

    Expects columns: video_id, category, source, licence (or license).
    Validates:
    - File exists and is valid CSV.
    - Exactly 30 distinct video IDs (no blank or duplicate IDs).
    - Exactly 10 videos in each of the 3 canonical categories.
    - Preserves source and licence as supplied.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Team inventory CSV file not found: {p}")

    records: List[TeamVideoRecord] = []
    seen_ids: Set[str] = set()

    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Team inventory CSV '{p}' is empty or missing header row.")

        # Column name normalization
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        if "video_id" not in field_map:
            raise ValueError(
                f"Team inventory CSV missing required 'video_id' column. Found: {reader.fieldnames}"
            )
        if "category" not in field_map:
            raise ValueError(
                f"Team inventory CSV missing required 'category' column. Found: {reader.fieldnames}"
            )
        # Accept either 'licence' or 'license'
        licence_col = field_map.get("licence") or field_map.get("license")
        if not licence_col:
            raise ValueError(
                f"Team inventory CSV missing required 'licence' column. Found: {reader.fieldnames}"
            )
        source_col = field_map.get("source")
        if not source_col:
            raise ValueError(
                f"Team inventory CSV missing required 'source' column. Found: {reader.fieldnames}"
            )

        video_id_col = field_map["video_id"]
        category_col = field_map["category"]

        row_num = 1
        for row in reader:
            row_num += 1
            raw_id = row.get(video_id_col, "")
            vid = raw_id.strip() if raw_id else ""
            if not vid:
                raise ValueError(f"Team inventory row {row_num} has empty or blank video_id.")
            if vid in seen_ids:
                raise ValueError(f"Duplicate video_id '{vid}' found in team inventory (row {row_num}).")
            seen_ids.add(vid)

            raw_cat = row.get(category_col, "")
            raw_source = row.get(source_col, "")
            raw_licence = row.get(licence_col, "")

            rec = TeamVideoRecord(
                video_id=vid,
                category=raw_cat,
                source=raw_source,
                licence=raw_licence,
            )
            records.append(rec)

    if len(records) != TEAM_TOTAL_VIDEOS:
        raise ValueError(
            f"Team inventory must have exactly {TEAM_TOTAL_VIDEOS} distinct videos, found {len(records)}."
        )

    # Check category counts
    counts: Dict[str, int] = {cat: 0 for cat in CATEGORY_ORDER}
    for r in records:
        counts[r.category] += 1

    for cat in CATEGORY_ORDER:
        if counts[cat] != TEAM_PER_CATEGORY:
            raise ValueError(
                f"Team inventory must have exactly {TEAM_PER_CATEGORY} videos in '{cat}', found {counts[cat]}."
            )

    return records


def sample_charades_slice(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    seed: int = 20261114,
) -> SplitManifest:
    """Sample Charades-STA slice according to §4.2 and M2 brief rules 1, 2, 4.

    Rules:
    - Sort eligible IDs before applying seeded shuffle without replacement.
    - Check source train and test populations for overlap.
    - Check for sufficient unique populations (at least 60 train, at least 60 test).
    - Shuffles train once: 30 for dev, next 30 for cal.
    - Shuffles test: 60 for test.
    - Total: 120 unique videos, strictly disjoint.
    - Deterministic random stream seeded with `seed`.
    """
    sorted_train = sorted(set(train_ids))
    sorted_test = sorted(set(test_ids))

    # Overlap check
    overlap = set(sorted_train) & set(sorted_test)
    if overlap:
        raise ValueError(
            f"Source train and test populations overlap ({len(overlap)} overlapping IDs): {sorted(overlap)[:10]}"
        )

    # Population size checks
    if len(sorted_train) < 60:
        raise ValueError(
            f"Insufficient unique Charades train videos: need at least 60 (30 dev + 30 cal), got {len(sorted_train)}."
        )
    if len(sorted_test) < 60:
        raise ValueError(
            f"Insufficient unique Charades test videos: need at least 60 (for test split), got {len(sorted_test)}."
        )

    # Independent deterministic random stream for Charades
    rng = random.Random(seed)

    # Shuffle train once without replacement
    shuffled_train = list(sorted_train)
    rng.shuffle(shuffled_train)
    dev_ids = shuffled_train[:30]
    cal_ids = shuffled_train[30:60]

    # Shuffle test once without replacement
    shuffled_test = list(sorted_test)
    rng.shuffle(shuffled_test)
    test_ids = shuffled_test[:60]

    records: List[ManifestRecord] = []
    for vid in dev_ids:
        records.append(
            ManifestRecord(
                dataset="charades-sta",
                video_id=vid,
                split="dev",
                source_split="train",
            )
        )
    for vid in cal_ids:
        records.append(
            ManifestRecord(
                dataset="charades-sta",
                video_id=vid,
                split="cal",
                source_split="train",
            )
        )
    for vid in test_ids:
        records.append(
            ManifestRecord(
                dataset="charades-sta",
                video_id=vid,
                split="test",
                source_split="test",
            )
        )

    # Sort records stably by (split, video_id)
    records.sort(key=lambda r: (SPLIT_ORDER[r.split], r.video_id))

    manifest = SplitManifest(
        manifest_version="1.0",
        dataset="charades-sta",
        seed=seed,
        sampling_convention=SAMPLING_CONVENTION,
        split_counts={
            "dev": len(dev_ids),
            "cal": len(cal_ids),
            "test": len(test_ids),
            "total": len(records),
        },
        records=records,
    )

    verify_manifest_integrity(manifest)
    return manifest


def split_team_videos(
    records: Sequence[TeamVideoRecord],
    seed: int = 20261114,
) -> SplitManifest:
    """Split team videos according to §4.1 and M2 brief rules 1, 3, 4.

    Rules:
    - Sort eligible IDs within each category before applying seeded shuffle.
    - Category allocations:
        Speech-heavy: 4 dev, 2 cal, 4 test (10 total)
        Mixed:        3 dev, 3 cal, 4 test (10 total)
        Action-heavy: 3 dev, 3 cal, 4 test (10 total)
        Total:       10 dev, 8 cal, 12 test (30 total)
    - Preserve source and licence as supplied.
    - Deterministic random stream seeded with `seed`.
    """
    if len(records) != TEAM_TOTAL_VIDEOS:
        raise ValueError(
            f"Expected exactly {TEAM_TOTAL_VIDEOS} team records, got {len(records)}."
        )

    # Check distinct video IDs
    unique_ids = {r.video_id for r in records}
    if len(unique_ids) != TEAM_TOTAL_VIDEOS:
        raise ValueError(f"Team records contain duplicate IDs (unique count: {len(unique_ids)}).")

    # Group by category
    by_cat: Dict[str, List[TeamVideoRecord]] = {cat: [] for cat in CATEGORY_ORDER}
    for r in records:
        if r.category not in by_cat:
            raise ValueError(f"Unknown team category '{r.category}'.")
        by_cat[r.category].append(r)

    for cat in CATEGORY_ORDER:
        if len(by_cat[cat]) != TEAM_PER_CATEGORY:
            raise ValueError(
                f"Category '{cat}' must have exactly {TEAM_PER_CATEGORY} videos, got {len(by_cat[cat])}."
            )

    # Independent deterministic random stream for Team set
    rng = random.Random(seed)

    manifest_records: List[ManifestRecord] = []
    category_summary: Dict[str, Dict[str, int]] = {}

    for cat in CATEGORY_ORDER:
        # Sort eligible records by video_id before shuffle
        cat_sorted = sorted(by_cat[cat], key=lambda x: x.video_id)
        shuffled = list(cat_sorted)
        rng.shuffle(shuffled)

        alloc = CATEGORY_ALLOCATIONS[cat]
        dev_k = alloc["dev"]
        cal_k = alloc["cal"]
        test_k = alloc["test"]

        dev_slice = shuffled[:dev_k]
        cal_slice = shuffled[dev_k : dev_k + cal_k]
        test_slice = shuffled[dev_k + cal_k : dev_k + cal_k + test_k]

        category_summary[cat] = {
            "dev": len(dev_slice),
            "cal": len(cal_slice),
            "test": len(test_slice),
            "total": len(shuffled),
        }

        for r in dev_slice:
            manifest_records.append(
                ManifestRecord(
                    dataset="team",
                    video_id=r.video_id,
                    split="dev",
                    category=r.category,
                    source=r.source,
                    licence=r.licence,
                )
            )
        for r in cal_slice:
            manifest_records.append(
                ManifestRecord(
                    dataset="team",
                    video_id=r.video_id,
                    split="cal",
                    category=r.category,
                    source=r.source,
                    licence=r.licence,
                )
            )
        for r in test_slice:
            manifest_records.append(
                ManifestRecord(
                    dataset="team",
                    video_id=r.video_id,
                    split="test",
                    category=r.category,
                    source=r.source,
                    licence=r.licence,
                )
            )

    # Sort records stably by (split, video_id)
    manifest_records.sort(key=lambda r: (SPLIT_ORDER[r.split], r.video_id))

    dev_count = sum(r.split == "dev" for r in manifest_records)
    cal_count = sum(r.split == "cal" for r in manifest_records)
    test_count = sum(r.split == "test" for r in manifest_records)

    manifest = SplitManifest(
        manifest_version="1.0",
        dataset="team",
        seed=seed,
        sampling_convention=SAMPLING_CONVENTION,
        split_counts={
            "dev": dev_count,
            "cal": cal_count,
            "test": test_count,
            "total": len(manifest_records),
        },
        category_counts=category_summary,
        records=manifest_records,
    )

    verify_manifest_integrity(manifest)
    return manifest


def verify_manifest_integrity(manifest: SplitManifest) -> None:
    """Validate manifest counts, uniqueness, source split, and split disjointness.

    Recomputes split counts, total counts, and team category-by-split counts from records.
    Validates them against both declared counts and required quotas.
    Verifies that each record's dataset matches the manifest dataset.
    Raises ValueError if any check fails.
    """
    # 1. Check each record's dataset against manifest dataset
    for r in manifest.records:
        if r.dataset != manifest.dataset:
            raise ValueError(
                f"Record dataset '{r.dataset}' does not match manifest dataset '{manifest.dataset}' "
                f"for video_id '{r.video_id}'."
            )

    # 2. Recompute split and total counts from records
    recomputed_dev = sum(1 for r in manifest.records if r.split == "dev")
    recomputed_cal = sum(1 for r in manifest.records if r.split == "cal")
    recomputed_test = sum(1 for r in manifest.records if r.split == "test")
    recomputed_total = len(manifest.records)

    # 3. Check recomputed counts against declared split_counts
    declared_dev = manifest.split_counts.get("dev")
    declared_cal = manifest.split_counts.get("cal")
    declared_test = manifest.split_counts.get("test")
    declared_total = manifest.split_counts.get("total")

    if declared_dev != recomputed_dev:
        raise ValueError(
            f"Declared dev count ({declared_dev}) does not match records count ({recomputed_dev})."
        )
    if declared_cal != recomputed_cal:
        raise ValueError(
            f"Declared cal count ({declared_cal}) does not match records count ({recomputed_cal})."
        )
    if declared_test != recomputed_test:
        raise ValueError(
            f"Declared test count ({declared_test}) does not match records count ({recomputed_test})."
        )
    if declared_total != recomputed_total:
        raise ValueError(
            f"Declared total count ({declared_total}) does not match records count ({recomputed_total})."
        )

    # 4. Check split and total counts against required quotas
    if manifest.dataset == "charades-sta":
        if recomputed_dev != 30 or declared_dev != 30:
            raise ValueError(f"Charades dev quota must be 30, got records={recomputed_dev}, declared={declared_dev}.")
        if recomputed_cal != 30 or declared_cal != 30:
            raise ValueError(f"Charades cal quota must be 30, got records={recomputed_cal}, declared={declared_cal}.")
        if recomputed_test != 60 or declared_test != 60:
            raise ValueError(f"Charades test quota must be 60, got records={recomputed_test}, declared={declared_test}.")
        if recomputed_total != 120 or declared_total != 120:
            raise ValueError(f"Charades total quota must be 120, got records={recomputed_total}, declared={declared_total}.")

        # Validate source splits
        for r in manifest.records:
            if r.split in {"dev", "cal"} and r.source_split != "train":
                raise ValueError(f"Charades {r.split} video {r.video_id} has invalid source_split: {r.source_split}")
            if r.split == "test" and r.source_split != "test":
                raise ValueError(f"Charades test video {r.video_id} has invalid source_split: {r.source_split}")

    elif manifest.dataset == "team":
        if recomputed_dev != 10 or declared_dev != 10:
            raise ValueError(f"Team dev quota must be 10, got records={recomputed_dev}, declared={declared_dev}.")
        if recomputed_cal != 8 or declared_cal != 8:
            raise ValueError(f"Team cal quota must be 8, got records={recomputed_cal}, declared={declared_cal}.")
        if recomputed_test != 12 or declared_test != 12:
            raise ValueError(f"Team test quota must be 12, got records={recomputed_test}, declared={declared_test}.")
        if recomputed_total != 30 or declared_total != 30:
            raise ValueError(f"Team total quota must be 30, got records={recomputed_total}, declared={declared_total}.")

        # Recompute team category-by-split counts from records
        recomputed_cat_counts: Dict[str, Dict[str, int]] = {
            cat: {"dev": 0, "cal": 0, "test": 0, "total": 0}
            for cat in CATEGORY_ORDER
        }
        for r in manifest.records:
            if not r.category or r.category not in recomputed_cat_counts:
                raise ValueError(f"Team manifest record '{r.video_id}' has invalid or missing category: '{r.category}'.")
            if r.split not in {"dev", "cal", "test"}:
                raise ValueError(f"Team manifest record '{r.video_id}' has invalid split: '{r.split}'.")
            recomputed_cat_counts[r.category][r.split] += 1
            recomputed_cat_counts[r.category]["total"] += 1

        # Check recomputed category counts against required quotas
        for cat, expected in CATEGORY_ALLOCATIONS.items():
            rec_cat = recomputed_cat_counts[cat]
            for split_key, req_count in expected.items():
                act_count = rec_cat[split_key]
                if act_count != req_count:
                    raise ValueError(
                        f"Team category '{cat}' split '{split_key}' quota mismatch: required {req_count}, got {act_count} in records."
                    )
            if rec_cat["total"] != TEAM_PER_CATEGORY:
                raise ValueError(
                    f"Team category '{cat}' total quota mismatch: required {TEAM_PER_CATEGORY}, got {rec_cat['total']} in records."
                )

        # Check recomputed category counts against declared category_counts
        if manifest.category_counts:
            for cat, expected in CATEGORY_ALLOCATIONS.items():
                declared_cat = manifest.category_counts.get(cat)
                if not declared_cat:
                    raise ValueError(f"Declared category_counts missing entry for '{cat}'.")
                rec_cat = recomputed_cat_counts[cat]
                for split_key in ["dev", "cal", "test"]:
                    dec_count = declared_cat.get(split_key)
                    if dec_count != rec_cat[split_key]:
                        raise ValueError(
                            f"Team category '{cat}' split '{split_key}' declared count ({dec_count}) "
                            f"does not match records count ({rec_cat[split_key]})."
                        )
                dec_total = declared_cat.get("total")
                if dec_total != rec_cat["total"]:
                    raise ValueError(
                        f"Team category '{cat}' declared total ({dec_total}) "
                        f"does not match records total ({rec_cat['total']})."
                    )

    # 5. Check split disjointness
    dev_ids = {r.video_id for r in manifest.records if r.split == "dev"}
    cal_ids = {r.video_id for r in manifest.records if r.split == "cal"}
    test_ids = {r.video_id for r in manifest.records if r.split == "test"}

    dev_cal_overlap = dev_ids & cal_ids
    if dev_cal_overlap:
        raise ValueError(f"Dev and cal splits overlap on video IDs: {dev_cal_overlap}")
    dev_test_overlap = dev_ids & test_ids
    if dev_test_overlap:
        raise ValueError(f"Dev and test splits overlap on video IDs: {dev_test_overlap}")
    cal_test_overlap = cal_ids & test_ids
    if cal_test_overlap:
        raise ValueError(f"Cal and test splits overlap on video IDs: {cal_test_overlap}")

    # 6. Check total video ID uniqueness
    total_ids = [r.video_id for r in manifest.records]
    if len(total_ids) != len(set(total_ids)):
        raise ValueError(f"Duplicate video IDs found in manifest records: {len(total_ids)} vs {len(set(total_ids))}")


def serialize_manifest_json(manifest: SplitManifest) -> str:
    """Serialize manifest to deterministic, reproducible UTF-8 JSON string.

    Rules:
    - Stable key order (sort_keys=True).
    - Indent=2, ensure_ascii=False.
    - Standard LF '\n' newline.
    - No timestamps, machine paths, or non-deterministic values.
    """
    data = manifest.model_dump(exclude_none=True)
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def serialize_manifest_csv(manifest: SplitManifest) -> str:
    """Serialize manifest records to deterministic UTF-8 CSV string with LF '\n'."""
    is_team = manifest.dataset == "team"
    if is_team:
        fieldnames = ["dataset", "video_id", "split", "category", "source", "licence"]
    else:
        fieldnames = ["dataset", "video_id", "split", "source_split"]

    import io

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for r in manifest.records:
        row: Dict[str, str] = {
            "dataset": r.dataset,
            "video_id": r.video_id,
            "split": r.split,
        }
        if is_team:
            row["category"] = r.category or ""
            row["source"] = r.source or ""
            row["licence"] = r.licence or ""
        else:
            row["source_split"] = r.source_split or ""
        writer.writerow(row)

    return out.getvalue()


def write_manifest_atomic(
    manifest: SplitManifest,
    output_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str]:
    """Atomically write manifest to disk and compute SHA-256 hash.

    Preserves any previous valid output if validation or writing fails.
    Writes first to a temporary file in the destination directory,
    then atomically replaces the destination.
    """
    verify_manifest_integrity(manifest)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    json_content = serialize_manifest_json(manifest)
    json_bytes = json_content.encode("utf-8")
    sha256_hash = compute_sha256(json_bytes)

    # Write JSON to temporary file in same directory
    tmp_path = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replace
        os.replace(tmp_path, out)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if csv_path:
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        csv_content = serialize_manifest_csv(manifest)
        csv_bytes = csv_content.encode("utf-8")
        tmp_csv = csv_out.with_name(f"{csv_out.name}.tmp.{os.getpid()}")
        try:
            with open(tmp_csv, "wb") as f:
                f.write(csv_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_csv, csv_out)
        except Exception:
            if tmp_csv.exists():
                tmp_csv.unlink()
            raise

    return out, sha256_hash

