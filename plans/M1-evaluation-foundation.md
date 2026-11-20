# M1 — Shared schema, synthetic indexes and evaluation metrics

Build the CPU-only foundation that Yesaullah's indexing code will write and Murtaza's retrieval and evaluation code will consume. M1 is complete when two synthetic indexes can be read back and known search outcomes yield the expected metrics. The source of truth is `video_investigator_plan.md`, especially §§4.1, 5.1, 6 and 8.1–8.3.

## 1. Define the shared records

Use `vi/schema.py` for the shared record types and validation. Preserve the master plan's field names for video metadata, query labels, ranked windows, evidence and SearchResult. Support multiple gold windows. Require at least one for an answerable query and zero for an unanswerable query. Store seconds from video start. Use finite half-open intervals with 0 ≤ start < end ≤ duration. Adjacent intervals have zero overlap. A frame has a single timestamp.

Preserve result order, up to five candidates, confidence in [0,1], route, per-path scores, evidence and timings. An abstained result can contain candidates internally. This lets evaluation score both retrieval alone and the decision the user experiences. Display behavior belongs to the later UI unit.

## 2. Build two synthetic indexes

Implement `python scripts/make_stub_index.py` from the repo root. Default to `index/stub-spoken/` and `index/stub-visual/`. Support an output directory argument so tests can use temporary storage. Repeated runs must reproduce the same logical records and arrays. Limit writes to the synthetic fixture paths. Generate artifacts locally and keep generated binaries out of source control.

Follow §8.1 exactly for SQLite tables and columns, the three .npy filenames and meta.json fields. Within each modality use contiguous, zero-based IDs. Every array row must match the corresponding database ID. Use deterministic float32 unit vectors with a small documented synthetic dimension and genuine, readable JPEG evidence files. Mark models, revisions and timings as synthetic in metadata so the fixtures cannot be mistaken for benchmark measurements. Synthetic video paths can be placeholders for M1.

Give the first video speech, frames and overlapping captions. Give the second frames and captions with zero transcript rows and a transcript array of shape (0, text_dimension). Include examples of answerable and unanswerable labels and constructed search results in the test fixtures. A real media decoder, model downloads and labelled dataset files are unnecessary here.

## 3. Implement the metric primitives

Use `vi/eval/metrics.py` and `vi/eval/bootstrap.py`. Implement pure functions that take records or numeric inputs. Leave CLI evaluation, running baselines, routing predictions and report generation to their later units.

- Implement temporal IoU. For each returned candidate use the greatest IoU across that query's gold windows. For answerable queries compute R@1 and R@5 at 0.3, 0.5 and 0.7, reciprocal rank through rank five at 0.5, and top-window mean IoU. Threshold equality counts as a hit. An empty result contributes zero to these metrics. For start-time error use the best-overlapping gold window for the top candidate, break ties by earliest start, and summarize eligible absolute errors with median and quartiles.
- Compute raw retrieval from retained candidates. For abstention-aware retrieval treat an abstained answerable query as having zero visible candidates. Compute rejection and false-answer rates over unanswerable queries and false-abstain rate over answerable queries. Balanced accuracy is the average of rejection rate and the share of answerable queries the system answers. Return numerator and denominator for rates. Use an explicit undefined value for an empty denominator.
- For risk–coverage, rank by descending confidence. An answer is correct when the query is answerable and its top window reaches IoU 0.5. Group tied confidence scores into one threshold. For the area use the sum of each coverage increment times the risk at its right endpoint. Record this convention in the function docstring.
- Implement the two-sided 95% Wilson interval without continuity correction. Implement percentile bootstrap intervals with 10,000 video-level resamples by default and an explicit seed. Sample whole videos with replacement, retaining every query and preserving multiplicity when a video is drawn twice. For paired differences use the same draws for both systems and align by video and query ID. Reject mismatched pairs. Skip undefined replicates, expose their count, and return an undefined interval if none are usable.

## 4. Prove the contract

Use pytest and exercise the generator through its normal entry point in a temporary directory. Reopen both SQLite databases, load all arrays and JPEGs, verify ID alignment, dtype, shape, unit norms, evidence paths and metadata, and check repeat generation. Include the following hand-checkable cases in `tests/test_metrics.py`.

| Case | Expected result |
|---|---|
| Windows [0,10) and [5,15) | IoU = 1/3 |
| Windows [0,10) and [10,15) | IoU = 0 |
| First hit is rank two at exactly IoU 0.5 | R@1 = 0, R@5 = 1, reciprocal rank = 0.5 |
| Top candidate matches only the second gold window | Full credit for that match |
| Correct top result retained after abstention | Raw R@1 = 1, abstention-aware R@1 = 0 |
| Two answerable queries, one withheld. Two negatives, both withheld | Rejection = 1, false-abstain = 0.5, balanced accuracy = 0.75 |
| Two equally confident queries, one correct | One coverage step, risk = 0.5, area = 0.5 |
| 40 successes in 50 trials | Wilson interval rounds to [0.670, 0.888] |
| Constant metric, repeated seed, identical paired systems | Zero-width interval, reproducible draws, paired difference always zero |

Add a small unequal-size two-video fixture to check cluster resampling and duplicated draws explicitly. Cover empty results, an empty denominator, malformed intervals and the empty transcript index.

## 5. Handoff

Deliver the shared records, generator, metric functions, tests and minimal reproducible CPU dependency instructions. Run the generator against temporary storage and run `python -m pytest -q` from the repo root. Report files changed, commands, actual results and any unverified checks. Keep the real dev/calibration/test data, hardware benchmarks, search pipeline, API and UI outside this unit.
