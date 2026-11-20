# M2 — Reproducible video selection and split manifests

Implement the CPU-only data selection unit in §8.3 of video_investigator_plan.md. Sections 4.1–4.3 define the populations and test custody. The manifest tells the team which videos belong to each split before any model runs. M3 will use it to select development videos.

## Inputs and outputs

- Team input is a local CSV inventory of 30 distinct video IDs, category, source and licence, with ten videos in each category. Preserve source and licence as supplied. Reject blank or duplicate IDs, unknown categories and incorrect totals. Shared output can contain video IDs, category and split.
- Charades inputs are the official train and test video-ID populations represented in Charades-STA annotations. Support local ID-only inventories prepared by the custodian from the respective annotation files. Deduplicate repeated IDs originating from multiple annotations before sampling. Check source train and test populations for overlap. Keep every positive query for a selected video eligible for downstream evaluation. Never sample or cap queries to reach the approximate query counts in the master plan.
- Provide the planned entry point `python scripts/sample_cs_slice.py --seed 20261114` with documented default paths and overrides for local inventories and output. Provide a team manifest entry point or mode usable independently. Write machine-readable manifests with dataset, video ID, assigned split and source split where applicable. Record the seed and deterministic sampling convention. Keep test query text, gold windows and negative labels in custodian-controlled storage. A shared video-ID manifest is sufficient.

## Sampling rules

1. Sort eligible IDs before applying a seeded shuffle without replacement. Use separate deterministic random streams for the two datasets so running either alone gives the same result as running both. Specify the algorithm in the implementation. Emit UTF-8 with stable ordering, newline conventions and no run timestamp or machine path in the hashed output.
2. For Charades shuffle the train population once, take 30 videos for dev and the next 30 for calibration. Select 60 from the official test population. Report insufficient populations before publishing outputs.
3. For the team set shuffle within each category. Use these fixed allocations to resolve the master plan's integer rounding choice while keeping category counts within one video in every split.

| Category | Dev | Calibration | Test |
|---|---:|---:|---:|
| Speech-heavy | 4 | 2 | 4 |
| Mixed | 3 | 3 | 4 |
| Action-heavy | 3 | 3 | 4 |
| Total | 10 | 8 | 12 |

4. Validate counts, uniqueness, source split and disjointness before publishing. Report SHA-256 for the actual manifest bytes. Failed validation must preserve any previous valid output. Missing real inputs should produce an actionable message identifying the expected file and format.

## Acceptance

- Exercise both entry points through the CLI in temporary directories using synthetic inventories. Include repeated Charades annotation IDs and enough unique IDs for the real quotas. Run twice with the specified seed, reverse input row order on the second run, and assert identical output bytes and SHA-256. Check identities and quotas across every split.
- Test source overlap, insufficient unique Charades videos, duplicate team IDs and incorrect team category counts. Verify that a failed rerun preserves the previous manifest. Run the existing M1 tests as part of the full suite.
- Include minimal input examples and reproducible commands. Report synthetic verification separately from real-data acceptance. Real acceptance requires the same commands on the actual inventories, verified counts, disjointness and reproducible hashes. Use synthetic inputs only inside tests. Team collection and query writing belong to S1, manual negative verification to S2, and the evaluation harness to M3.
