# M3 — Development evaluation harness and B0 chance baseline

Build the CPU evaluation entry point in master plan §§5–6 and 8.3. M3 is complete when a fresh checkout can generate M1's two stub indexes, run B0 on labelled synthetic development queries, and produce a reproducible, inspectable report. Real retrieval systems start in M4.

## Entry point and inputs

1. Support the acceptance command `python -m vi.eval.run_eval --split dev --systems B0` after `python scripts/make_stub_index.py`. Document that the default M3 run is synthetic. Explicitly mark the dataset and fixture mode inside the report. Provide overrides for the index root, output directory and seed. Any explicit real-development input mode must fail clearly on missing inputs.
2. Supply a small synthetic development label fixture covering both stub videos, answerable and unanswerable queries, and multiple gold windows. Identify fixtures explicitly. M1 stores the stub split as synthetic. Keep this mapping local to fixture mode. For real dev input, select IDs from a validated M2 manifest and load only development labels and indexes. Reject unsupported splits or systems before opening data. Keep calibration and final evaluation in their assigned units.
3. Load duration and video identity from each selected SQLite index in read-only mode. Check ID consistency, unique (video_id, qid) keys, expected video coverage and bounds using M1 schema utilities. Fail on missing indexes or labels. Align evaluation predictions by key. Never silently truncate missing pairs with zip.

## B0 definition

Use 1,000 seeded trials. Each trial returns five randomly ranked windows per query. For each window sample a length uniformly with replacement from the pooled development gold-window lengths for that dataset, cap it at the target video duration, then sample a start uniformly from the valid range. A full-duration window starts at zero. The length pool must be nonempty. Build it once. At generation time a query contributes only its key and video duration. Use deterministic ordering and stable seeds so reordering labels leaves results identical.

B0 always answers and has no ranking-confidence signal. Use one common confidence for risk–coverage, preserve random candidate order, and document these conventions. Compute metrics for each trial before averaging. A trial's five candidates are the ranked list for that trial. Keep all trials for a query together when estimating uncertainty.

## Report contract

Write strict JSON to `results/dev_B0.json` only after validation and calculation succeed. Include dataset, split, synthetic status, seeds, trial and bootstrap counts, video/query counts, answerability mix, metric definitions or unambiguous keys, and undefined values as JSON null with a reason. Reuse M1 metric functions and conventions.

- Report R@1 and R@5 at IoU 0.3/0.5/0.7, MRR, top-window mean IoU, start-time error summaries with eligible counts, raw and abstention-aware retrieval, rejection, false-answer and false-abstain rates, balanced accuracy, and the risk–coverage curve and area. Record applicable subgroup counts. With B0, raw and abstention-aware retrieval coincide. Routing scores and GPU measurements have named fields with an applicability reason. Record measured CPU baseline timing separately from reproducible score fields.
- For headline retrieval rates, average over trials and give a 95% video-cluster bootstrap interval using 10,000 fixed-seed draws. For additive metrics, cache query-level trial averages and aggregate them over each sampled set of videos with multiplicity, weighting queries equally. An unequal-size fixture must agree with a brute-force reference. Keep undefined-replicate counts. Never count 1,000 trials as 1,000 independent labelled queries.
- Wilson intervals need integer query counts. For trial-dependent rates include a separately labelled first-trial rate, hits, denominator and Wilson interval. Give the 1,000-trial mean its own field and cluster interval. For trial-invariant rates use the actual query counts directly. Undefined start-error summaries should expose the number of trials with eligible overlaps. Keep the report compact by omitting the full Monte Carlo trace.

## Acceptance checks for Flash

1. Exercise the generator and evaluation CLI in temporary directories. Run twice with the same seed and compare metric fields. Reverse fixture row order and repeat. Check that both video IDs are represented and that intervals use the documented resampling unit.
2. Use a hand-checkable full-duration fixture with one answerable and one negative query. Every prediction covers the entire video. Expect R@1 and R@5 of 1 for a full-video gold interval, rejection 0, false-answer 1, false-abstain 0, balanced accuracy 0.5 and risk–coverage area 0.5. Verify Wilson denominators and that repeated trials do not narrow intervals artificially.
3. Exercise multiple gold windows, overlong sampled lengths, unequal cluster sizes, missing or duplicate keys and out-of-bounds gold. Unsupported splits must fail before label access. A failing rerun must preserve a previous valid report.
4. Use reduced trial and resample counts for focused tests. Run the documented default command once at full counts and report elapsed time, output path, actual counts and key scores. Run the full pytest suite. Keep generated reports and synthetic indexes out of the proposed commit.
