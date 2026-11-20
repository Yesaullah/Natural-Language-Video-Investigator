#!/usr/bin/env python3
"""Charades-STA video slice sampling and data manifest generator (§8.3 M2).

Deterministic sampling without replacement for Charades-STA benchmark slice (seed 20261114).
Preserves test-label custody and produces machine-readable split manifests.
Can also split team videos (--dataset team or --dataset both).

Conforms to:
- plans/M2-data-manifests.md
- video_investigator_plan.md §§4.1–4.3, 8.3 (unit M2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path if executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi.manifest import (
    load_team_inventory_csv,
    parse_charades_inventory,
    sample_charades_slice,
    split_team_videos,
    verify_manifest_integrity,
    write_manifest_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample Charades-STA slice and generate reproducible split manifests (M2)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261114,
        help="Deterministic PRNG seed (default: 20261114).",
    )
    parser.add_argument(
        "--dataset",
        choices=["cs", "team", "both"],
        default="cs",
        help="Dataset to sample/split: 'cs' (Charades-STA), 'team', or 'both' (default: cs).",
    )
    # Charades options
    parser.add_argument(
        "--train-input",
        type=str,
        default="data/cs/charades_sta_train.txt",
        help="Path to Charades-STA train annotations or ID list (default: data/cs/charades_sta_train.txt).",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default="data/cs/charades_sta_test.txt",
        help="Path to Charades-STA test annotations or ID list (default: data/cs/charades_sta_test.txt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/cs/manifest.json",
        help="Output path for Charades-STA JSON manifest (default: data/cs/manifest.json).",
    )
    parser.add_argument(
        "--cs-csv-out",
        type=str,
        default=None,
        help="Optional path to write Charades-STA manifest in CSV format.",
    )
    # Team options
    parser.add_argument(
        "--team-input",
        type=str,
        default="data/tl/videos.csv",
        help="Path to team videos inventory CSV (default: data/tl/videos.csv).",
    )
    parser.add_argument(
        "--team-output",
        type=str,
        default="data/tl/manifest.json",
        help="Output path for team JSON manifest (default: data/tl/manifest.json).",
    )
    parser.add_argument(
        "--team-csv-out",
        type=str,
        default=None,
        help="Optional path to write team manifest in CSV format.",
    )
    return parser


def run_charades_sampling(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    seed: int,
    csv_out: Path | None = None,
) -> str:
    """Run Charades-STA sampling pipeline and return manifest SHA-256."""
    if not train_path.is_file():
        sys.stderr.write(
            f"Error: Missing Charades train input file at '{train_path}'.\n"
            "Expected input format: Local text file with Charades-STA annotations "
            "('<video_id> <start> <end> <query>') or one video_id per line.\n"
            "Required population: At least 60 distinct video IDs (30 for dev, 30 for cal).\n"
            "Please provide the file or specify path via --train-input.\n"
        )
        sys.exit(1)

    if not test_path.is_file():
        sys.stderr.write(
            f"Error: Missing Charades test input file at '{test_path}'.\n"
            "Expected input format: Local text file with Charades-STA annotations "
            "('<video_id> <start> <end> <query>') or one video_id per line.\n"
            "Required population: At least 60 distinct video IDs (for test split).\n"
            "Please provide the file or specify path via --test-input.\n"
        )
        sys.exit(1)

    try:
        train_ids = parse_charades_inventory(train_path)
        test_ids = parse_charades_inventory(test_path)
        manifest = sample_charades_slice(train_ids, test_ids, seed=seed)
        out_file, sha256_hash = write_manifest_atomic(manifest, output_path, csv_path=csv_out)
    except Exception as exc:
        sys.stderr.write(f"Validation or sampling error (Charades-STA): {exc}\n")
        sys.exit(1)

    print("=== Charades-STA Slice Manifest Created ===")
    print(f"Dataset:            charades-sta")
    print(f"Seed:               {seed}")
    print(f"Sampling algorithm: {manifest.sampling_convention}")
    print(f"Dev split:          {manifest.split_counts['dev']} videos (source: train)")
    print(f"Cal split:          {manifest.split_counts['cal']} videos (source: train)")
    print(f"Test split:         {manifest.split_counts['test']} videos (source: test)")
    print(f"Total videos:       {manifest.split_counts['total']}")
    print(f"Split disjointness: VERIFIED (0 overlaps)")
    print(f"Manifest path:      {out_file}")
    print(f"SHA-256:            {sha256_hash}")
    if csv_out:
        print(f"CSV manifest path:  {csv_out}")
    return sha256_hash


def run_team_split(
    input_path: Path,
    output_path: Path,
    seed: int,
    csv_out: Path | None = None,
) -> str:
    """Run Team video split pipeline and return manifest SHA-256."""
    if not input_path.is_file():
        sys.stderr.write(
            f"Error: Missing Team video inventory file at '{input_path}'.\n"
            "Expected input format: Local CSV file with columns: video_id, category, source, licence.\n"
            "Required population: Exactly 30 distinct video IDs, with 10 in each category: "
            "'Speech-heavy', 'Mixed', 'Action-heavy'.\n"
            "Please provide the file or specify path via --team-input or --input.\n"
        )
        sys.exit(1)

    try:
        records = load_team_inventory_csv(input_path)
        manifest = split_team_videos(records, seed=seed)
        out_file, sha256_hash = write_manifest_atomic(manifest, output_path, csv_path=csv_out)
    except Exception as exc:
        sys.stderr.write(f"Validation or split error (Team): {exc}\n")
        sys.exit(1)

    print("=== Team Video Split Manifest Created ===")
    print(f"Dataset:            team")
    print(f"Seed:               {seed}")
    print(f"Sampling algorithm: {manifest.sampling_convention}")
    print(f"Dev split:          {manifest.split_counts['dev']} videos")
    print(f"Cal split:          {manifest.split_counts['cal']} videos")
    print(f"Test split:         {manifest.split_counts['test']} videos")
    print(f"Total videos:       {manifest.split_counts['total']}")
    if manifest.category_counts:
        print("Category allocation:")
        for cat, counts in manifest.category_counts.items():
            print(f"  - {cat:13s}: Dev={counts['dev']}, Cal={counts['cal']}, Test={counts['test']} (Total={counts['total']})")
    print(f"Split disjointness: VERIFIED (0 overlaps)")
    print(f"Manifest path:      {out_file}")
    print(f"SHA-256:            {sha256_hash}")
    if csv_out:
        print(f"CSV manifest path:  {csv_out}")
    return sha256_hash


def run_both(
    train_path: Path,
    test_path: Path,
    cs_out: Path,
    team_in: Path,
    team_out: Path,
    seed: int,
    cs_csv: Path | None = None,
    team_csv: Path | None = None,
) -> tuple[str, str]:
    """Run both Charades and Team pipelines atomically.

    Loads both inventories, constructs both manifests, and validates both
    before writing either output to disk.
    """
    # 1. Validate that all required input files exist
    if not train_path.is_file():
        sys.stderr.write(
            f"Error: Missing Charades train input file at '{train_path}'.\n"
            "Expected input format: Local text file with Charades-STA annotations "
            "('<video_id> <start> <end> <query>') or one video_id per line.\n"
            "Required population: At least 60 distinct video IDs (30 for dev, 30 for cal).\n"
            "Please provide the file or specify path via --train-input.\n"
        )
        sys.exit(1)

    if not test_path.is_file():
        sys.stderr.write(
            f"Error: Missing Charades test input file at '{test_path}'.\n"
            "Expected input format: Local text file with Charades-STA annotations "
            "('<video_id> <start> <end> <query>') or one video_id per line.\n"
            "Required population: At least 60 distinct video IDs (for test split).\n"
            "Please provide the file or specify path via --test-input.\n"
        )
        sys.exit(1)

    if not team_in.is_file():
        sys.stderr.write(
            f"Error: Missing Team video inventory file at '{team_in}'.\n"
            "Expected input format: Local CSV file with columns: video_id, category, source, licence.\n"
            "Required population: Exactly 30 distinct video IDs, with 10 in each category: "
            "'Speech-heavy', 'Mixed', 'Action-heavy'.\n"
            "Please provide the file or specify path via --team-input or --input.\n"
        )
        sys.exit(1)

    # 2. Load both inventories, construct both manifests, validate both before writing
    try:
        train_ids = parse_charades_inventory(train_path)
        test_ids = parse_charades_inventory(test_path)
        team_records = load_team_inventory_csv(team_in)

        cs_manifest = sample_charades_slice(train_ids, test_ids, seed=seed)
        team_manifest = split_team_videos(team_records, seed=seed)

        verify_manifest_integrity(cs_manifest)
        verify_manifest_integrity(team_manifest)
    except Exception as exc:
        sys.stderr.write(f"Validation or sampling error: {exc}\n")
        sys.exit(1)

    # 3. Write outputs only after both manifests are constructed and validated
    try:
        cs_file, cs_hash = write_manifest_atomic(cs_manifest, cs_out, csv_path=cs_csv)
        team_file, team_hash = write_manifest_atomic(team_manifest, team_out, csv_path=team_csv)
    except Exception as exc:
        sys.stderr.write(f"Error writing manifest outputs: {exc}\n")
        sys.exit(1)

    print("=== Charades-STA Slice Manifest Created ===")
    print(f"Dataset:            charades-sta")
    print(f"Seed:               {seed}")
    print(f"Sampling algorithm: {cs_manifest.sampling_convention}")
    print(f"Dev split:          {cs_manifest.split_counts['dev']} videos (source: train)")
    print(f"Cal split:          {cs_manifest.split_counts['cal']} videos (source: train)")
    print(f"Test split:         {cs_manifest.split_counts['test']} videos (source: test)")
    print(f"Total videos:       {cs_manifest.split_counts['total']}")
    print(f"Split disjointness: VERIFIED (0 overlaps)")
    print(f"Manifest path:      {cs_file}")
    print(f"SHA-256:            {cs_hash}")
    if cs_csv:
        print(f"CSV manifest path:  {cs_csv}")

    print()

    print("=== Team Video Split Manifest Created ===")
    print(f"Dataset:            team")
    print(f"Seed:               {seed}")
    print(f"Sampling algorithm: {team_manifest.sampling_convention}")
    print(f"Dev split:          {team_manifest.split_counts['dev']} videos")
    print(f"Cal split:          {team_manifest.split_counts['cal']} videos")
    print(f"Test split:         {team_manifest.split_counts['test']} videos")
    print(f"Total videos:       {team_manifest.split_counts['total']}")
    if team_manifest.category_counts:
        print("Category allocation:")
        for cat, counts in team_manifest.category_counts.items():
            print(f"  - {cat:13s}: Dev={counts['dev']}, Cal={counts['cal']}, Test={counts['test']} (Total={counts['total']})")
    print(f"Split disjointness: VERIFIED (0 overlaps)")
    print(f"Manifest path:      {team_file}")
    print(f"SHA-256:            {team_hash}")
    if team_csv:
        print(f"CSV manifest path:  {team_csv}")

    return cs_hash, team_hash


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mode = args.dataset
    train_in = Path(args.train_input)
    test_in = Path(args.test_input)
    cs_out = Path(args.output)
    cs_csv = Path(args.cs_csv_out) if args.cs_csv_out else None

    team_in = Path(args.team_input)
    team_out = Path(args.team_output)
    team_csv = Path(args.team_csv_out) if args.team_csv_out else None

    if mode == "cs":
        run_charades_sampling(train_in, test_in, cs_out, args.seed, cs_csv)
    elif mode == "team":
        run_team_split(team_in, team_out, args.seed, team_csv)
    elif mode == "both":
        run_both(
            train_in,
            test_in,
            cs_out,
            team_in,
            team_out,
            args.seed,
            cs_csv=cs_csv,
            team_csv=team_csv,
        )


if __name__ == "__main__":
    main()

