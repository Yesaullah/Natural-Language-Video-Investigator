"""CLI integration and a hand-checkable paired whole-video comparison."""

import hashlib
import json
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from scripts.make_stub_index import make_stub_indexes
from vi.eval import transcript_eval
from vi.eval.run_eval import _write_report, main
from vi.eval.transcript_eval import VideoDraws, spoken_comparison
from vi.manifest import sample_charades_slice
from vi.schema import CandidateWindow, QueryLabel, SearchResult
from vi.search.dense import EncoderIdentity
from vi.search.index import MODEL_DIMENSION, MODEL_ID, RealIndexProvenance
from vi.search.settings import RealSettings, SearchSettings, Standardization

LABELS = Path("data/synthetic/dev_labels.jsonl")


def test_report_publication_under_long_output_root(tmp_path):
    root = tmp_path / "long-reports"
    relative_dir = Path("a" * 90, "b" * 90, "c" * 90, "charades-sta")
    filename = f"dev_B0,B1_combined_{'a' * 24}.json"
    output = root / relative_dir / filename
    assert len(str(output.parent)) > 260

    disk_root = root
    if sys.platform == "win32":
        absolute_root = str(root)
        if absolute_root.startswith("\\\\"):
            disk_root = Path("\\\\?\\UNC\\" + absolute_root[2:])
        else:
            disk_root = Path("\\\\?\\" + absolute_root)
    disk_output = disk_root / relative_dir / filename

    try:
        _write_report(output, {"score": 0.25})
        assert json.loads(disk_output.read_bytes()) == {"score": 0.25}
        _write_report(output, {"score": 0.75})
        assert json.loads(disk_output.read_bytes()) == {"score": 0.75}
        assert list(disk_output.parent.iterdir()) == [disk_output]
    finally:
        if disk_root.exists():
            shutil.rmtree(disk_root)


def test_paired_spoken_uses_trial_means_and_whole_video_draws():
    labels = [
        QueryLabel(video_id=video, qid=qid, query="spoken", answerable=True,
                   gold_windows=[(0, 5)], qtype=qtype)
        for video, qid, qtype in [
            ("a", "1", "spoken"), ("b", "2", "spoken"), ("b", "3", "spoken"),
            ("b", "4", "spoken"), ("c", "5", "visual"),
        ]
    ]
    hit = SearchResult(
        abstained=False, confidence=0.8, route="transcript",
        results=[CandidateWindow(start_s=0, end_s=5, score=1)],
    )
    miss = SearchResult(abstained=True, confidence=0, route="transcript")
    b0 = {("a", "1"): 0.2, ("b", "2"): 0.1, ("b", "3"): 0.4, ("b", "4"): 0.6, ("c", "5"): 0.7}
    b1 = {("a", "1"): hit, ("b", "2"): miss, ("b", "3"): hit, ("b", "4"): miss, ("c", "5"): hit}
    draws = VideoDraws(labels, 10000, 42)
    pair = spoken_comparison(labels, b0, b1, draws)
    assert pair["query_count"] == 4
    assert pair["spoken_video_count"] == 2
    assert pair["B0"]["point"] == pytest.approx(0.325)
    assert pair["B1"]["point"] == 0.5
    delta = pair["B1_minus_B0"]
    assert delta["point"] == pytest.approx(0.175)

    rng = np.random.default_rng(42)
    cluster_differences = ([0.8], [-0.1, 0.6, -0.6], [])
    brute_force = []
    for _ in range(10000):
        differences = [value for i in rng.integers(0, 3, 3) for value in cluster_differences[i]]
        if differences:
            brute_force.append(sum(differences) / len(differences))
    assert delta["n_undefined"] == 10000 - len(brute_force)
    assert [delta["ci_lower"], delta["ci_upper"]] == pytest.approx(np.percentile(brute_force, [2.5, 97.5]))
    no_spoken = [q.model_copy(update={"qtype": "visual"}) for q in labels]
    missing = spoken_comparison(no_spoken, b0, b1, draws)["B1_minus_B0"]
    assert missing["point"] is None and missing["n_undefined"] == 10000
    with pytest.raises(ValueError, match="Prediction keys"):
        spoken_comparison(labels, b0, {key: hit for key in list(b1)[:-1]}, draws)


@pytest.mark.parametrize(("selection", "expected"), [
    (["--systems", "B1,B99"], "Unsupported system"),
    (["--systems", "B1,B1"], "Duplicate"),
    (["--systems", "B0,"], "Unsupported system"),
    (["--split", "cal", "--systems", "B1"], "Unsupported split"),
    (["--mode", "real", "--systems", "B1"], "Real development requires"),
])
def test_validation_precedes_label_access(tmp_path, selection, expected):
    with pytest.raises(ValueError, match=expected):
        main(["--label-file", str(tmp_path / "missing.jsonl"), *selection])


def test_b1_cli_reproducibility_parity_and_failure_preservation(tmp_path):
    index_dir = tmp_path / "index"
    _, visual = make_stub_indexes(index_dir)
    output = tmp_path / "reports"
    reverse = tmp_path / "reversed.jsonl"
    reverse.write_text("\n".join(reversed(LABELS.read_text().splitlines())) + "\n", encoding="utf-8")
    common = [
        "--index-root", str(index_dir), "--output-dir", str(output),
        "--n-trials", "17", "--n-bootstrap", "60",
    ]

    def read(name):
        return json.loads((output / f"dev_{name}.json").read_text(encoding="utf-8"))

    def reproducible(report):
        return {key: value for key, value in report.items() if key not in {"cpu_timing", "label_sha256"}}

    main([*common, "--systems", "B0"])
    main([*common, "--systems", "B1"])
    b1 = read("B1")
    command = [
        sys.executable, "-m", "vi.eval.run_eval", *common,
        "--systems", "B1,B0", "--label-file", str(reverse),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    paired = read("B0,B1")
    assert reproducible(read("B0")) == reproducible(paired["system_reports"]["B0"])
    for key, value in reproducible(paired["system_reports"]["B1"]).items():
        assert b1[key] == value
    assert paired["query_keys"] == b1["query_keys"]
    assert len(paired["query_keys"]) == 9
    assert paired["spoken_comparison"]["query_count"] == 1
    assert b1["retrieval"]["raw"]["R@1_iou0.5"]["denominator"] == 5
    assert b1["abstention"]["false_abstain_rate"]["hits"] == 2
    assert b1["subgroups"]["mixed"]["retrieval"]["raw"]["R@1_iou0.5"]["rate"] is None

    report_path = output / "dev_B0,B1.json"
    previous = report_path.read_bytes()
    (visual / "transcript.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing transcript index artifact"):
        main([*common, "--systems", "B0,B1"])
    assert report_path.read_bytes() == previous


@pytest.fixture
def charades_cli(tmp_path):
    manifest = sample_charades_slice(
        [f"train-{i:02}" for i in range(60)], [f"test-{i:02}" for i in range(60)],
    )
    dev_ids = sorted(record.video_id for record in manifest.records if record.split == "dev")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    labels_path = tmp_path / "dev.jsonl"
    labels_path.write_text("\n".join(
        QueryLabel(video_id=vid, qid="q1", query="match", answerable=True,
                   gold_windows=[(0, 10)], qtype="spoken").model_dump_json()
        for vid in dev_ids
    ), encoding="utf-8")
    provenance = RealIndexProvenance(
        config_hash="offline-index-v1", encoder_revision="a" * 40,
        passage_policy={
            "target_span_s": 10, "stride": "every_segment", "gap": "include_elapsed_gap",
            "tail": "retain_short", "embedded_text": "exact_persisted_text",
            "passage_prompt": None, "query_prompt": "query",
        },
    )
    index_root = tmp_path / "index"
    output = tmp_path / "reports"
    command = [
        "--mode", "real", "--split", "dev", "--manifest", str(manifest_path),
        "--label-file", str(labels_path), "--index-root", str(index_root),
        "--output-dir", str(output), "--n-trials", "2", "--n-bootstrap", "10",
    ]

    def make_indexes(speech_videos):
        for position, vid in enumerate(dev_ids):
            directory = index_root / vid
            directory.mkdir(parents=True)
            database = directory / "index.sqlite"
            embeddings = directory / "transcript.npy"
            # Two passages per speech video demonstrate that coverage counts videos.
            count = 2 if position < speech_videos else 0
            with sqlite3.connect(database) as conn:
                conn.execute("CREATE TABLE video (video_id, path, duration_s, sha256, dataset, split)")
                conn.execute("INSERT INTO video VALUES (?, ?, 10.0, 'video-hash', 'charades-sta', 'dev')",
                             (vid, f"{vid}.mp4"))
                conn.execute("CREATE TABLE transcript_units (unit_id, start_s, end_s, text)")
                conn.executemany("INSERT INTO transcript_units VALUES (?, 0.0, 10.0, 'match')",
                                 [(i,) for i in range(count)])
            conn.close()
            vectors = np.zeros((count, MODEL_DIMENSION), dtype=np.float32)
            vectors[:, 0] = 1
            np.save(embeddings, vectors)
            metadata = {
                "video_id": vid, "synthetic": False, "config_hash": provenance.config_hash,
                "models": {"text_embed": MODEL_ID}, "revisions": {"text_embed": provenance.encoder_revision},
                "transcript_passage_policy": provenance.passage_policy.model_dump(),
                "transcript_file_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (database, embeddings)
                },
            }
            (directory / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    def write_settings(weight):
        search = SearchSettings(dense_weight=weight)
        settings = RealSettings(
            search=search, standardization=Standardization(mean=0.0, sd=1.0, covered_bins=30),
            index=provenance, dataset="charades-sta", fit_population="dev_query_video_pairs",
            fit_weighting="equal_covered_bins_population_sd", fitted_scoring=search.scoring_identity(),
        )
        path = tmp_path / f"fit-{weight}.json"
        path.write_text(settings.model_dump_json(), encoding="utf-8")
        return ["--b1-settings", str(path)]

    return command, output, make_indexes, write_settings, index_root / dev_ids[-1] / "transcript.npy"


@pytest.mark.parametrize(("speech_count", "systems"), [(0, "B1"), (2, "B0,B1")])
def test_real_charades_below_threshold_skips_b1(charades_cli, monkeypatch, speech_count, systems):
    command, output, make_indexes, _, _ = charades_cli
    make_indexes(speech_count)

    def unexpected_scoring(*args, **kwargs):
        pytest.fail("An inapplicable B1 run must skip search and encoder construction")

    monkeypatch.setattr(transcript_eval, "QwenQueryEncoder", unexpected_scoring)
    monkeypatch.setattr(transcript_eval, "TranscriptSearch", unexpected_scoring)
    main([*command, "--systems", systems])
    report = json.loads((output / "charades-sta" / f"dev_{systems}_inapplicable.json").read_bytes())
    assert "spoken_comparison" not in report
    if systems == "B0,B1":
        assert report["system_reports"]["B0"]["retrieval"]["trial_mean"]["R@1_iou0.5"] == 1
        b1 = report["system_reports"]["B1"]
    else:
        b1 = report
    assert b1["applicable"] is False
    assert "10%" in b1["reason"]
    assert b1["speech_coverage"] == {
        "numerator": speech_count, "denominator": 30,
        "fraction": speech_count / 30, "minimum_fraction": 0.1,
    }
    assert "retrieval" not in b1 and "abstention" not in b1


@pytest.mark.parametrize("missing", [False, True])
def test_real_charades_corrupt_or_missing_index_fails_even_without_speech(charades_cli, missing):
    command, output, make_indexes, _, embeddings = charades_cli
    make_indexes(0)
    if missing:
        embeddings.unlink()
        error, reason = FileNotFoundError, "Missing transcript index artifact"
    else:
        embeddings.write_bytes(b"invalid index")
        error, reason = ValueError, "hash mismatch"
    with pytest.raises(error, match=reason):
        main([*command, "--systems", "B0,B1"])
    assert not output.exists()


def test_real_charades_exactly_ten_percent_runs_b1(charades_cli):
    command, output, make_indexes, write_settings, _ = charades_cli
    make_indexes(3)
    args = [*command, "--systems", "B1"]
    with pytest.raises(ValueError, match="requires --b1-settings"):
        main(args)
    assert not output.exists()
    main([*args, *write_settings(0)])
    [path] = (output / "charades-sta").glob("dev_B1_lexical_only_*.json")
    report = json.loads(path.read_bytes())
    assert report["applicable"] is True
    assert report["speech_coverage"] == {
        "numerator": 3, "denominator": 30, "fraction": 0.1, "minimum_fraction": 0.1,
    }
    assert report["retrieval"]["raw"]["R@1_iou0.5"]["rate"] == 0.1


def test_real_charades_weight_sweep_retains_both_reports(charades_cli, monkeypatch):
    command, output, make_indexes, write_settings, _ = charades_cli
    make_indexes(3)

    class OfflineEncoder:
        def __init__(self, revision):
            self.identity = EncoderIdentity(MODEL_ID, revision, MODEL_DIMENSION, "query", False)

        def __call__(self, _):
            vector = np.zeros(MODEL_DIMENSION)
            vector[0] = 1
            return vector

    monkeypatch.setattr(transcript_eval, "QwenQueryEncoder", OfflineEncoder)
    for weight in (0.25, 0.75):
        main([*command, "--systems", "B1,B0", *write_settings(weight)])
    reports = {}
    for path in (output / "charades-sta").glob("dev_B0,B1_combined_*.json"):
        report = json.loads(path.read_bytes())
        settings = report["system_reports"]["B1"]["settings"]["fitted_settings"]
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        assert path.name == f"dev_B0,B1_combined_{digest}.json"
        assert report["spoken_comparison"]["query_count"] == 30
        reports[settings["search"]["dense_weight"]] = report
    assert set(reports) == {0.25, 0.75}
    main([*command, "--systems", "B0"])
    assert (output / "charades-sta" / "dev_B0.json").is_file()
