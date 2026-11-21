# B1 transcript search

Use the CPU requirements file and generate the existing synthetic indexes before
running development evaluation.

```powershell
python -m pip install -r requirements-cpu.txt
python scripts/make_stub_index.py
python -m vi.eval.run_eval --split dev --systems B1
python -m vi.eval.run_eval --split dev --systems B0,B1
python -m pytest -q
```

Reports default to `results/dev_B1.json` and `results/dev_B0,B1.json`. B0 alone
continues to write `results/dev_B0.json`. Use `--output-dir` to redirect reports.
The CLI fixture encoder hashes the input text deterministically. The synthetic
passage vectors have no semantic relationship to the transcripts.

## Real development inputs

```powershell
python -m vi.eval.run_eval --mode real --split dev --systems B0,B1 --manifest <m2-json> --label-file <dev-jsonl> --index-root <root> --b1-settings <fitted-json>
```

The manifest selects the development videos in one dataset. Supply labels for
exactly those videos. SQLite supplies video durations because the M2 manifest
has no duration field. The evaluator checks label window bounds against those
durations. B0 pools all gold-window lengths within that dataset.

When B1 is requested on real Charades development data, the evaluator validates
every selected transcript index and reports the fraction of selected videos
with at least one passage. Below 10%, B1 is inapplicable and the report includes
the reason and coverage counts. B0 still runs when requested. This case needs
no fitted settings or encoder. At exactly 10% or higher, scoring requires
fitted settings. Missing or corrupt transcript indexes fail the run.

Real reports are written under `<output-dir>/<dataset>/`. B0 alone uses
`dev_B0.json`. Applicable B1 runs use
`dev_<systems>_<variant>_<settings-id>.json`, with systems sorted and
comma-separated. The settings ID is the first 24 hex characters of the
SHA-256 digest of the validated fitted settings serialized
as UTF-8 compact JSON with sorted keys, including weights, window settings,
standardization and index provenance. Different weights within the same
variant get different files. Inapplicable B1 runs use
`dev_<systems>_inapplicable.json`.

The index writer publishes immutable SQLite and NumPy files, with the array row
for each zero-based passage ID holding a normalized, 1,024-dimensional float32
embedding of that row's exact text. Overlap is allowed. The eventual writer
should start at each Whisper segment and merge consecutive segments until
their elapsed span reaches 10 seconds, including elapsed gaps in that span
and retaining a shorter tail. Segment times and text order are preserved.
The index owner still needs to confirm the provisional writer policy and test
it on real output. The reader uses an absolute unit-norm tolerance of 0.0001.

Extend the M1 `meta.json` fields on real indexes with these entries.

- Set `synthetic` to false and `models.text_embed` to
  `Qwen/Qwen3-Embedding-0.6B`. Record the full immutable model revision in
  `revisions.text_embed` and the writer configuration hash in `config_hash`.
- Add `transcript_passage_policy` with `target_span_s` 10, `stride`
  `every_segment`, `gap` `include_elapsed_gap`, `tail` `retain_short`,
  `embedded_text` `exact_persisted_text`, `passage_prompt` null and
  `query_prompt` `query`.
- Add `transcript_file_sha256` with SHA-256 digests keyed by `index.sqlite`
  and `transcript.npy`. Close the writer and checkpoint SQLite before hashing.
  Publish the database without a WAL sidecar.

Real dense scoring loads Sentence Transformers and the locally cached model
lazily using the full recorded revision. Pre-cache that revision before the
real run. The default CPU requirements exclude model packages and weights.

## Provisional scoring and development selection

BM25 uses Lucene scoring with k1 1.5 and b 0.75. Both passages and queries use
the same Unicode alphanumeric tokenizer, case folding, and no stemming or
stopword filtering. Repeated query terms count separately. The mixed score is
the weighted sum of cosine and BM25 divided by query token count. A query
with zero tokens contributes zero lexical score. Set `dense_weight` to 0 for
lexical only, 1 for dense only, or a value between them for B1.

For each fixed scoring configuration, project raw passage scores to one-second
bins using the maximum over passages with positive overlap. Fit a single mean
and population SD across covered bins over all dev queries, including
unanswerable queries. Each covered bin contributes once per query. Use scale 1
when SD is zero and retain the recorded SD of zero. Store the fit alongside
the scoring identity and index provenance. See `RealSettings` for the JSON
contract, `SearchSettings.scoring_identity()` for the identity dictionary,
and `fit_standardization()` for estimating the constants from raw timelines.
The CPU fixture uses mean 0, SD 1, and equal lexical and dense weights.

Windows use the earliest bin of each peak plateau, keep the seed even for a
nonpositive peak, and grow over contiguous covered bins above alpha times
the peak. Alpha defaults to 0.7. The master plan's strict cutoff is the default.
Set `include_cutoff` to true to test the B1 brief's inclusive proposal. Rank by
peak score, then earlier start and end. Suppress IoU strictly above 0.5. The
uncalibrated confidence is the sigmoid of the top score. An empty transcript
abstains at confidence zero and remains in evaluation denominators.

The proposed, untried real-dev grid uses dense weights 0, 0.25, 0.5, 0.75 and 1
with the other settings fixed. Report every tried configuration and its own fit.
Select the combined B1 among 0.25, 0.5 and 0.75 by spoken raw R@1 at IoU 0.5.
Break ties by proximity to 0.5 and then lower dense weight. Report both
single-component X1 variants separately. Real tuning and the M4 gate await
the labelled team development videos and model-generated indexes.
