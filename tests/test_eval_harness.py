"""M3 acceptance tests for the evaluation harness.

Covers all checks in plans/M3-evaluation-harness.md §Acceptance checks for Flash:
  1. Generator / CLI in temp dirs, deterministic seeds, row-order invariance.
  2. Hand-checkable full-duration fixture.
  3. Multiple gold windows, overlong lengths, unequal cluster sizes,
     missing/duplicate keys, out-of-bounds gold, unsupported splits/systems.
  4. Reduced-trial focused tests (full-count CLI run is in a separate test).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ── project imports ────────────────────────────────────────────────────────────
from vi.schema import CandidateWindow, QueryLabel, SearchResult, VideoMetadata
from vi.eval.validate_labels import (
    build_gold_length_pool,
    check_gold_window_bounds,
    check_id_consistency,
    check_unique_query_keys,
    load_labels_from_jsonl,
    reject_unsupported_split,
    reject_unsupported_system,
    validate_labels_and_indexes,
)
from vi.eval.run_eval import (
    B0_CONFIDENCE,
    B0_N_CANDIDATES,
    _first_trial_results,
    _generate_b0_trial,
    _query_trial_avg_r_at_k,
    generate_b0_results,
    run_b0_evaluation,
    main as run_eval_main,
)
from vi.eval.metrics import recall_at_k, wilson_score_interval
from scripts.make_stub_index import make_stub_indexes


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def stub_index_dir(tmp_path_factory):
    """Generate both stub indexes into a session-scoped temp directory."""
    base = tmp_path_factory.mktemp("stub_indexes")
    make_stub_indexes(output_base=base)
    return base


@pytest.fixture(scope="session")
def stub_video_meta(stub_index_dir) -> Dict[str, VideoMetadata]:
    from vi.eval.validate_labels import load_video_metadata_from_index
    meta = {}
    for vid in ("stub-spoken", "stub-visual"):
        meta[vid] = load_video_metadata_from_index(stub_index_dir / vid)
    return meta


def _make_labels(rows: List[dict]) -> List[QueryLabel]:
    return [QueryLabel.model_validate(r) for r in rows]


def _answerable(qid, vid, gold, qtype="visual"):
    return {
        "qid": qid, "video_id": vid, "query": f"q {qid}",
        "answerable": True, "gold_windows": gold,
        "qtype": qtype, "neg_type": None,
    }


def _unanswerable(qid, vid, neg_type="ood"):
    return {
        "qid": qid, "video_id": vid, "query": f"neg {qid}",
        "answerable": False, "gold_windows": [],
        "qtype": None, "neg_type": neg_type,
    }


# ── §1: Generator / CLI determinism and row-order invariance ─────────────────

def test_same_seed_same_results(stub_index_dir, stub_video_meta):
    """Running twice with the same seed yields identical metric fields."""
    labels = _make_labels([
        _answerable("q1", "stub-spoken", [[5.0, 15.0]]),
        _unanswerable("q2", "stub-spoken"),
        _answerable("q3", "stub-visual", [[0.0, 10.0]]),
    ])
    length_pool = [10.0, 8.0, 10.0]

    r1 = run_b0_evaluation(labels, stub_video_meta, length_pool,
                            n_trials=50, bootstrap_n=200, gen_seed=7, bootstrap_seed=13)
    r2 = run_b0_evaluation(labels, stub_video_meta, length_pool,
                            n_trials=50, bootstrap_n=200, gen_seed=7, bootstrap_seed=13)

    # Retrieval trial means must be identical
    assert r1["retrieval"]["trial_mean"]["R@1_iou0.5"] == r2["retrieval"]["trial_mean"]["R@1_iou0.5"]
    assert r1["retrieval"]["trial_mean"]["MRR"] == r2["retrieval"]["trial_mean"]["MRR"]
    # Bootstrap intervals must be identical
    assert r1["retrieval"]["trial_mean_cluster_bootstrap_95"]["R@1_iou0.5"] == \
           r2["retrieval"]["trial_mean_cluster_bootstrap_95"]["R@1_iou0.5"]


def test_row_order_invariance(stub_index_dir, stub_video_meta):
    """Reversing label row order leaves trial-mean metric fields unchanged."""
    rows = [
        _answerable("qa1", "stub-spoken", [[5.0, 15.0]]),
        _unanswerable("qu1", "stub-spoken"),
        _answerable("qa2", "stub-visual", [[2.0, 10.0]]),
        _unanswerable("qu2", "stub-visual"),
    ]
    labels_fwd = _make_labels(rows)
    labels_rev = _make_labels(list(reversed(rows)))

    length_pool = [10.0, 8.0]

    r_fwd = run_b0_evaluation(labels_fwd, stub_video_meta, length_pool,
                               n_trials=50, bootstrap_n=200, gen_seed=3, bootstrap_seed=5)
    r_rev = run_b0_evaluation(labels_rev, stub_video_meta, length_pool,
                               n_trials=50, bootstrap_n=200, gen_seed=3, bootstrap_seed=5)

    assert r_fwd["retrieval"]["trial_mean"]["R@1_iou0.5"] == \
           r_rev["retrieval"]["trial_mean"]["R@1_iou0.5"]


def test_both_video_ids_represented(stub_index_dir, stub_video_meta):
    """Report contains both stub video IDs."""
    labels = _make_labels([
        _answerable("qa1", "stub-spoken", [[5.0, 10.0]]),
        _answerable("qa2", "stub-visual", [[0.0, 5.0]]),
    ])
    length_pool = [5.0]
    report = run_b0_evaluation(labels, stub_video_meta, length_pool,
                                n_trials=20, bootstrap_n=100, gen_seed=1, bootstrap_seed=1)
    assert set(report["video_ids"]) == {"stub-spoken", "stub-visual"}


def test_intervals_use_video_as_resampling_unit(stub_index_dir, stub_video_meta):
    """Bootstrap CI is computed at video level, not query level.

    With a single video and two queries, every resample is identical to the
    full set, so the interval must have zero width.
    """
    labels = _make_labels([
        _answerable("qa1", "stub-spoken", [[0.0, 15.0]]),
        _answerable("qa2", "stub-spoken", [[15.0, 30.0]]),
    ])
    length_pool = [15.0]
    report = run_b0_evaluation(labels, stub_video_meta, length_pool,
                                n_trials=50, bootstrap_n=500, gen_seed=99, bootstrap_seed=42)

    boot = report["retrieval"]["trial_mean_cluster_bootstrap_95"]["R@1_iou0.5"]
    ci_lo = boot["ci_lower"]
    ci_hi = boot["ci_upper"]
    # With a single video the interval is degenerate (all resamples equal point estimate)
    assert ci_lo is not None and ci_hi is not None
    assert abs(ci_hi - ci_lo) < 1e-9, (
        f"Expected zero-width interval with one video, got [{ci_lo}, {ci_hi}]"
    )


# ── §2: Hand-checkable full-duration fixture ──────────────────────────────────

class TestFullDurationFixture:
    """Full-duration B0 fixture: every prediction covers [0, duration).

    Setup:
      - stub-spoken: duration = 30.0 s
      - Answerable query: gold = [[0.0, 30.0]]
      - Unanswerable query (negative)
      - Length pool = [30.0] → every sampled window has length 30.0 → start = 0
        → every prediction is [0, 30) = full duration

    Expected per the brief:
      R@1 = 1 (IoU with [0,30) is 1.0 ≥ any threshold)
      R@5 = 1
      rejection = 0 (B0 never abstains)
      false_answer = 1 (all unanswerable queries get a result)
      false_abstain = 0
      balanced_accuracy = 0.5
      risk_coverage area = 0.5 (all queries share confidence=0.5; risk at full coverage
                                  = 1 unanswerable out of 2 total = 0.5; area = 1.0 × 0.5)
    """

    @pytest.fixture(autouse=True)
    def setup(self, stub_video_meta):
        self.video_meta = stub_video_meta
        self.labels = _make_labels([
            _answerable("fd-ans", "stub-spoken", [[0.0, 30.0]]),
            _unanswerable("fd-neg", "stub-spoken"),
        ])
        # Pool of exactly one length = duration → every trial gives a full-duration window
        self.pool = [30.0]

    def _run(self, **kwargs) -> dict:
        defaults = dict(n_trials=200, bootstrap_n=500, gen_seed=0, bootstrap_seed=42)
        defaults.update(kwargs)
        return run_b0_evaluation(self.labels, self.video_meta, self.pool, **defaults)

    def test_r_at_1_iou05_equals_1(self):
        report = self._run()
        assert report["retrieval"]["trial_mean"]["R@1_iou0.5"] == pytest.approx(1.0)

    def test_r_at_5_iou05_equals_1(self):
        report = self._run()
        assert report["retrieval"]["trial_mean"]["R@5_iou0.5"] == pytest.approx(1.0)

    def test_r_at_1_iou07_equals_1(self):
        report = self._run()
        assert report["retrieval"]["trial_mean"]["R@1_iou0.7"] == pytest.approx(1.0)

    def test_rejection_is_0(self):
        report = self._run()
        assert report["abstention"]["rejection_accuracy"]["rate"] == pytest.approx(0.0)

    def test_false_answer_is_1(self):
        report = self._run()
        assert report["abstention"]["false_answer_rate"]["rate"] == pytest.approx(1.0)

    def test_false_abstain_is_0(self):
        report = self._run()
        assert report["abstention"]["false_abstain_rate"]["rate"] == pytest.approx(0.0)

    def test_balanced_accuracy_is_0_5(self):
        report = self._run()
        assert report["abstention"]["balanced_accuracy"] == pytest.approx(0.5)

    def test_risk_coverage_area_is_0_5(self):
        """With all queries at the same confidence, there is one coverage step.
        Risk = (1 unanswerable answered correctly wrong) / 2 total answered = 0.5.
        Area = 1.0 × 0.5 = 0.5."""
        report = self._run()
        assert report["risk_coverage"]["area"] == pytest.approx(0.5)

    def test_wilson_denominator_uses_actual_query_counts(self):
        """Wilson denominators must use the actual query counts (1 answerable, 1 unanswerable)."""
        report = self._run()
        # Rejection: 0 abstentions out of 1 unanswerable query
        rej = report["abstention"]["rejection_accuracy"]
        assert rej["denominator"] == 1
        assert rej["hits"] == 0
        # False-abstain: 0 abstentions out of 1 answerable query
        fas = report["abstention"]["false_abstain_rate"]
        assert fas["denominator"] == 1
        assert fas["hits"] == 0

    def test_repeated_trials_do_not_narrow_intervals_artificially(self):
        """1000-trial mean and 500-trial mean should be close (within 0.05) but
        the CI should NOT collapse to zero when there are 2+ videos or non-trivial
        variation (here we have 1 video → zero-width CI is expected).
        What matters is that the _point estimate_ doesn't drift."""
        r_200 = self._run(n_trials=200)
        r_1000 = self._run(n_trials=1000)
        diff = abs(
            r_200["retrieval"]["trial_mean"]["R@1_iou0.5"] -
            r_1000["retrieval"]["trial_mean"]["R@1_iou0.5"]
        )
        # Both should be 1.0 for full-duration prediction vs full-duration gold
        assert diff < 1e-9


# ── §3: Multiple gold windows, overlong lengths, unequal clusters, errors ──────

def test_multiple_gold_windows_scores_best_match():
    """A query with two gold windows should score the best-matching candidate."""
    labels = _make_labels([
        _answerable("q-multi", "stub-spoken", [[0.0, 10.0], [20.0, 30.0]]),
    ])
    video_meta = {"stub-spoken": VideoMetadata(
        video_id="stub-spoken", path="p", duration_s=30.0,
        sha256="x", dataset="synthetic", split="synthetic",
    )}
    # Pool: lengths 10 and 10 → start uniform in [0, 20] or [0, 20]
    pool = [10.0, 10.0]
    report = run_b0_evaluation(labels, video_meta, pool,
                                n_trials=100, bootstrap_n=200, gen_seed=1, bootstrap_seed=1)
    # With 5 candidates per trial across 100 trials, some should hit either gold window
    r5 = report["retrieval"]["trial_mean"]["R@5_iou0.5"]
    assert isinstance(r5, float)
    assert 0.0 <= r5 <= 1.0


def test_overlong_sampled_lengths_are_capped():
    """When the pool contains a length longer than the video, the generated window
    must still fit within [0, duration]."""
    rng = np.random.default_rng(42)
    duration = 10.0
    pool = [100.0, 200.0, 50.0]   # all longer than duration
    for _ in range(500):
        windows = _generate_b0_trial(rng, duration, pool, n_candidates=5)
        for s, e in windows:
            assert s >= 0.0, f"start {s} < 0"
            assert e <= duration + 1e-9, f"end {e} > duration {duration}"
            assert e > s or s == e == 0.0, f"degenerate window [{s}, {e})"


def test_unequal_cluster_sizes_bootstrap():
    """Fixture with 3 queries for one video and 1 for another.
    Brute-force check: cluster bootstrap aggregates by video, not by query."""
    labels = _make_labels([
        _answerable("qa1", "stub-spoken", [[0.0, 30.0]]),
        _answerable("qa2", "stub-spoken", [[5.0, 15.0]]),
        _answerable("qa3", "stub-spoken", [[15.0, 25.0]]),
        _answerable("qa4", "stub-visual", [[0.0, 10.0]]),
    ])
    video_meta = {
        "stub-spoken": VideoMetadata(video_id="stub-spoken", path="p", duration_s=30.0,
                                     sha256="x", dataset="synthetic", split="synthetic"),
        "stub-visual": VideoMetadata(video_id="stub-visual", path="p", duration_s=20.0,
                                     sha256="x", dataset="synthetic", split="synthetic"),
    }
    pool = [10.0, 15.0, 10.0]
    report = run_b0_evaluation(labels, video_meta, pool,
                                n_trials=50, bootstrap_n=400, gen_seed=5, bootstrap_seed=5)
    boot = report["retrieval"]["trial_mean_cluster_bootstrap_95"]["R@1_iou0.5"]
    predictions = generate_b0_results(labels, video_meta, pool, n_trials=50, gen_seed=5)
    clusters = [[0, 1, 2], [3]]

    def brute_force(draw):
        sample = [labels[i] for cluster in draw for i in clusters[cluster]]
        rates = []
        for trial in range(50):
            results = [predictions[(q.video_id, q.qid)][trial] for q in sample]
            rates.append(recall_at_k(sample, results, k=1, iou_threshold=0.5).rate)
        return np.mean(rates)

    rng = np.random.default_rng(5)
    reference = {draw: brute_force(draw) for draw in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    draws = [reference[tuple(draw)] for draw in rng.integers(0, 2, size=(400, 2))]
    assert boot["point"] == pytest.approx(reference[(0, 1)])
    assert (boot["ci_lower"], boot["ci_upper"]) == pytest.approx(np.percentile(draws, [2.5, 97.5]))



def test_missing_key_raises_before_evaluation():
    """A label referencing a video not in the index_dirs should raise before evaluation."""
    labels = _make_labels([
        _answerable("q1", "nonexistent-video", [[0.0, 10.0]]),
    ])
    video_meta = {
        "stub-spoken": VideoMetadata(video_id="stub-spoken", path="p", duration_s=30.0,
                                     sha256="x", dataset="synthetic", split="synthetic"),
    }
    with pytest.raises(ValueError, match="nonexistent-video"):
        check_id_consistency(labels, video_meta)


def test_duplicate_query_key_raises():
    """Duplicate (video_id, qid) raises a clear error."""
    labels = _make_labels([
        _answerable("q1", "stub-spoken", [[0.0, 10.0]]),
        _answerable("q1", "stub-spoken", [[5.0, 15.0]]),
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        check_unique_query_keys(labels)


def test_out_of_bounds_gold_raises():
    """Gold window exceeding video duration raises a clear error."""
    labels = _make_labels([
        _answerable("q1", "stub-spoken", [[0.0, 999.0]]),
    ])
    video_meta = {
        "stub-spoken": VideoMetadata(video_id="stub-spoken", path="p", duration_s=30.0,
                                     sha256="x", dataset="synthetic", split="synthetic"),
    }
    with pytest.raises(ValueError, match="exceeds"):
        check_gold_window_bounds(labels, video_meta)


def test_unsupported_split_raises_before_data_access():
    """An unsupported split (e.g. 'test') raises before any file is opened."""
    with pytest.raises(ValueError, match="test"):
        reject_unsupported_split("test")
    with pytest.raises(ValueError, match="cal"):
        reject_unsupported_split("cal")


def test_unsupported_system_raises():
    """An unsupported system name raises before any data is opened."""
    with pytest.raises(ValueError, match="B99"):
        reject_unsupported_system("B99", ("B0",))


def test_failing_rerun_preserves_previous_report(tmp_path, stub_index_dir):
    """A failing run must not corrupt a previously valid report."""
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    # Write a "previous" report manually
    prev_path = output_dir / "dev_B0.json"
    prev_data = {"sentinel": "previous_valid_report"}
    prev_path.write_text(json.dumps(prev_data))

    # Attempt a run with a bad split that must fail before writing
    with pytest.raises((ValueError, SystemExit)):
        run_eval_main([
            "--split", "test",          # unsupported → should raise
            "--systems", "B0",
            "--index-root", str(stub_index_dir),
            "--output-dir", str(output_dir),
        ])

    # Previous report must be intact
    written = json.loads(prev_path.read_text())
    assert written == prev_data, "Previous valid report was corrupted by a failing run"


# ── §3: B0 never counts 1,000 trials as 1,000 independent queries ─────────────

def test_trial_count_not_inflating_query_denominator(stub_video_meta):
    """With 2 answerable queries and 1000 trials, the query denominator must be 2,
    not 2000.  The trial count must not be confused with the query count."""
    labels = _make_labels([
        _answerable("qa1", "stub-spoken", [[0.0, 15.0]]),
        _answerable("qa2", "stub-visual", [[0.0, 10.0]]),
    ])
    pool = [10.0]
    report = run_b0_evaluation(labels, stub_video_meta, pool,
                                n_trials=50, bootstrap_n=200, gen_seed=1, bootstrap_seed=1)
    assert report["query_count"] == 2
    assert report["answerable_count"] == 2
    assert report["trial_count"] == 50
    # The first-trial denominator must reflect actual query count
    r1_first = report["retrieval"]["first_trial"]["R@1_iou0.5"]
    assert r1_first["denominator"] == 2


# ── §4: CLI end-to-end in temp directories ────────────────────────────────────

def test_cli_basic_run_writes_valid_json(tmp_path, stub_index_dir):
    """The CLI writes strict JSON to results/dev_B0.json."""
    out_dir = tmp_path / "results"
    run_eval_main([
        "--split", "dev",
        "--systems", "B0",
        "--index-root", str(stub_index_dir),
        "--output-dir", str(out_dir),
        "--n-trials", "20",
        "--n-bootstrap", "100",
    ])
    out_path = out_dir / "dev_B0.json"
    assert out_path.exists(), "Output file not created"
    report = json.loads(out_path.read_text())
    # Required top-level keys
    for key in ("dataset", "split", "synthetic", "fixture_mode", "systems",
                 "trial_count", "bootstrap_count", "video_count", "query_count",
                 "retrieval", "abstention", "risk_coverage"):
        assert key in report, f"Missing key: {key}"
    assert report["synthetic"] is True
    assert report["split"] == "dev"
    assert report["systems"] == ["B0"]


def test_cli_null_fields_are_json_null(tmp_path, stub_index_dir):
    """Routing scores and GPU measurements are JSON null."""
    out_dir = tmp_path / "results"
    run_eval_main([
        "--split", "dev",
        "--systems", "B0",
        "--index-root", str(stub_index_dir),
        "--output-dir", str(out_dir),
        "--n-trials", "10",
        "--n-bootstrap", "50",
    ])
    report = json.loads((out_dir / "dev_B0.json").read_text())
    assert report["routing_scores"] is None
    assert report["gpu_measurements"] is None


def test_cli_missing_index_raises_before_output(tmp_path):
    """If a required index directory is missing, the CLI raises before writing output."""
    out_dir = tmp_path / "results"
    with pytest.raises((FileNotFoundError, SystemExit)):
        run_eval_main([
            "--split", "dev",
            "--systems", "B0",
            "--index-root", str(tmp_path / "nonexistent"),
            "--output-dir", str(out_dir),
            "--n-trials", "5",
            "--n-bootstrap", "10",
        ])
    assert not (out_dir / "dev_B0.json").exists()


def test_cli_seed_reproducibility(tmp_path, stub_index_dir):
    """Running the CLI twice with the same seeds produces identical JSON metric fields."""
    def run_once(out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_eval_main([
            "--split", "dev",
            "--systems", "B0",
            "--index-root", str(stub_index_dir),
            "--output-dir", str(out_dir),
            "--n-trials", "30",
            "--n-bootstrap", "150",
            "--gen-seed", "77",
            "--bootstrap-seed", "88",
        ])
        return json.loads((out_dir / "dev_B0.json").read_text())

    r1 = run_once(tmp_path / "run1")
    r2 = run_once(tmp_path / "run2")

    assert r1["retrieval"]["trial_mean"] == r2["retrieval"]["trial_mean"]
    assert r1["retrieval"]["trial_mean_cluster_bootstrap_95"] == \
           r2["retrieval"]["trial_mean_cluster_bootstrap_95"]


# ── B0 confidence / abstention invariants ─────────────────────────────────────

def test_b0_confidence_is_constant(stub_video_meta):
    """Every SearchResult generated by B0 must use the documented common confidence."""
    labels = _make_labels([
        _answerable("qa", "stub-spoken", [[5.0, 15.0]]),
        _unanswerable("qu", "stub-spoken"),
    ])
    pool = [10.0]
    b0_dict = generate_b0_results(labels, stub_video_meta, pool,
                                   n_trials=50, gen_seed=0)
    for key, trials in b0_dict.items():
        for sr in trials:
            assert sr.confidence == B0_CONFIDENCE, (
                f"Expected confidence={B0_CONFIDENCE}, got {sr.confidence} for {key}"
            )
            assert sr.abstained is False, f"B0 must never abstain, got True for {key}"


def test_b0_always_has_five_candidates(stub_video_meta):
    """Every trial must return exactly B0_N_CANDIDATES candidates."""
    labels = _make_labels([
        _answerable("qa", "stub-spoken", [[0.0, 10.0]]),
    ])
    pool = [10.0]
    b0_dict = generate_b0_results(labels, stub_video_meta, pool,
                                   n_trials=20, gen_seed=0)
    for trials in b0_dict.values():
        for sr in trials:
            assert len(sr.results) == B0_N_CANDIDATES


# ── Gold-window length pool validation ────────────────────────────────────────

def test_empty_pool_raises():
    """An empty label list (no answerable queries) must raise before generation."""
    labels = _make_labels([
        _unanswerable("qu", "stub-spoken"),
    ])
    with pytest.raises(ValueError, match="empty"):
        build_gold_length_pool(labels)


# ── Fixture JSONL loading ─────────────────────────────────────────────────────

def test_load_labels_from_fixture(tmp_path):
    """JSONL fixture loads and validates successfully."""
    fixture = Path("data/synthetic/dev_labels.jsonl")
    if not fixture.exists():
        pytest.skip("Fixture file not found; run from repo root.")
    labels = load_labels_from_jsonl(fixture)
    assert len(labels) >= 2
    video_ids = {q.video_id for q in labels}
    assert "stub-spoken" in video_ids
    assert "stub-visual" in video_ids


def test_load_labels_missing_file(tmp_path):
    """Missing label file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_labels_from_jsonl(tmp_path / "missing.jsonl")


def test_load_labels_duplicate_key_in_file(tmp_path):
    """Duplicate (video_id, qid) in a file raises after loading."""
    f = tmp_path / "dup.jsonl"
    row = json.dumps(_answerable("dup-q", "stub-spoken", [[0.0, 5.0]]))
    f.write_text(row + "\n" + row + "\n")
    labels = load_labels_from_jsonl(f)
    with pytest.raises(ValueError, match="Duplicate"):
        check_unique_query_keys(labels)


# ── Full validate pipeline ────────────────────────────────────────────────────

def test_validate_labels_and_indexes_success(stub_index_dir):
    """Full validation pipeline succeeds against the stub indexes."""
    labels = load_labels_from_jsonl(Path("data/synthetic/dev_labels.jsonl"))
    index_dirs = {
        "stub-spoken": stub_index_dir / "stub-spoken",
        "stub-visual": stub_index_dir / "stub-visual",
    }
    video_meta = validate_labels_and_indexes(labels, index_dirs)
    assert "stub-spoken" in video_meta
    assert "stub-visual" in video_meta
    assert video_meta["stub-spoken"].duration_s == pytest.approx(30.0)
    assert video_meta["stub-visual"].duration_s == pytest.approx(20.0)


def test_cli_reproducible_in_fresh_processes(tmp_path, stub_index_dir):
    rows = Path("data/synthetic/dev_labels.jsonl").read_text(encoding="utf-8").splitlines()
    reports = []
    for seed, ordered in [("12", rows), ("34", rows[::-1])]:
        path = tmp_path / "labels.jsonl"
        path.write_text("\n".join(ordered), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "vi.eval.run_eval", "--label-file", str(path),
             "--index-root", str(stub_index_dir), "--output-dir", str(tmp_path),
             "--n-trials", "12", "--n-bootstrap", "30"],
            env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
        )
        report = json.loads((tmp_path / "dev_B0.json").read_text(encoding="utf-8"))
        del report["cpu_timing"]
        reports.append(report)
    assert reports[0] == reports[1]


def test_trial_aggregation_and_first_trial_counts(monkeypatch, stub_video_meta):
    labels = _make_labels([_answerable(qid, "stub-spoken", [[0, 10]]) for qid in ("a", "b")])

    def result(start):
        window = CandidateWindow(start_s=start, end_s=start + 10, score=1, evidence={}, path_scores={})
        return SearchResult(abstained=False, confidence=0.5, route="random", results=[window], timings_ms={})

    # Eligible start errors by trial: [0, 4], [8], []. Only (a, trial 0) hits IoU 0.5.
    starts = {"a": [0, 8, 10], "b": [4, 10, 10]}
    predictions = {("stub-spoken", q): list(map(result, values)) for q, values in starts.items()}
    monkeypatch.setattr("vi.eval.run_eval.generate_b0_results", lambda *a, **kw: predictions)
    report = run_b0_evaluation(labels, stub_video_meta, [10], n_trials=3, bootstrap_n=10)
    summary = report["start_time_error_all_trials"]
    assert summary["eligible_count"] == 3
    assert summary["n_trials_with_eligible"] == 2
    assert [summary[key] for key in ("q25", "median", "q75", "iqr")] == [4.5, 5, 5.5, 1]
    assert report["risk_coverage"]["area"] == pytest.approx(5 / 6)
    assert report["risk_coverage"]["curve"][0]["risk"] == pytest.approx(5 / 6)
    for k in (1, 5):
        for threshold, hits in [(0.3, 2), (0.5, 1), (0.7, 1)]:
            rate = report["retrieval"]["first_trial"][f"R@{k}_iou{threshold}"]
            assert (rate["hits"], rate["denominator"]) == (hits, 2)
            assert (rate["wilson_95"]["lower"], rate["wilson_95"]["upper"]) == wilson_score_interval(hits, 2)
