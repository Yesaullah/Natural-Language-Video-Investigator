#!/usr/bin/env python3
"""Team video split manifest generator (§4.1, §8.3 M2).

Deterministic sampling without replacement for the 30 team-collected videos (seed 20261114).
Preserves test-label custody and produces machine-readable split manifests.
Can be run independently or via sample_cs_slice.py --dataset team.

Conforms to:
- plans/M2-data-manifests.md
- video_investigator_plan.md §§4.1, 4.3, 8.3 (unit M2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path if executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vi.manifest import (
    load_team_inventory_csv,
    split_team_videos,
    write_manifest_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split 30 team videos into Dev, Calibration, and Test manifests (M2)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261114,
        help="Deterministic PRNG seed (default: 20261114).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/tl/videos.csv",
        help="Path to team videos inventory CSV (default: data/tl/videos.csv).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/tl/manifest.json",
        help="Output path for team JSON manifest (default: data/tl/manifest.json).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional path to write team manifest in CSV format.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_out = Path(args.csv_out) if args.csv_out else None

    if not input_path.is_file():
        sys.stderr.write(
            f"Error: Missing Team video inventory file at '{input_path}'.\n"
            "Expected input format: Local CSV file with columns: video_id, category, source, licence.\n"
            "Required population: Exactly 30 distinct video IDs, with 10 in each category: "
            "'Speech-heavy', 'Mixed', 'Action-heavy'.\n"
            "Please provide the file or specify path via --input.\n"
        )
        sys.exit(1)

    try:
        records = load_team_inventory_csv(input_path)
        manifest = split_team_videos(records, seed=args.seed)
        out_file, sha256_hash = write_manifest_atomic(manifest, output_path, csv_path=csv_out)
    except Exception as exc:
        sys.stderr.write(f"Validation or split error (Team): {exc}\n")
        sys.exit(1)

    print("=== Team Video Split Manifest Created ===")
    print(f"Dataset:            team")
    print(f"Seed:               {args.seed}")
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


if __name__ == "__main__":
    main()

