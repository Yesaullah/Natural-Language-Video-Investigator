"""Controlled CPU checks for the B1 transcript baseline."""

from pathlib import Path
import sqlite3

import numpy as np
import pytest

from scripts.make_stub_index import make_stub_indexes
from vi.search.dense import QwenQueryEncoder, cosine_scores
from vi.search.index import load_transcript_index
from vi.search.lexical import LexicalIndex, LexicalSettings, tokenize
from vi.search.pipeline import TranscriptSearch
from vi.search.settings import FIXTURE_SETTINGS, FIXTURE_STANDARDIZATION, SearchSettings
from vi.search.windows import extract_windows, project_passages
from vi.schema import TranscriptUnit


@pytest.fixture
def stub_indexes(tmp_path):
    return make_stub_indexes(tmp_path)


def test_temporal_iou_ranks_spoken_passage_first(stub_indexes):
    spoken = load_transcript_index(stub_indexes[0], "stub-spoken", synthetic=True)
    vectors = np.zeros_like(spoken.embeddings)
    vectors[:, :3] = np.eye(3)
    np.save(stub_indexes[0] / "transcript.npy", vectors)
    spoken = load_transcript_index(stub_indexes[0], "stub-spoken", synthetic=True)
    search = TranscriptSearch(spoken, FIXTURE_SETTINGS)
    encode = lambda _: vectors[2]
    result = search.search("temporal IoU", encode, FIXTURE_STANDARDIZATION)
    top = result.results[0]
    assert (top.start_s, top.end_s) == (18, 28)
    assert top.end_s <= 30
    assert top.evidence.transcript == ["Notice how temporal IoU evaluates candidate window overlap."]
    assert search.score("temporal IoU", encode).cosine.tolist() == [0, 0, 1]


def test_visual_is_valid_empty(stub_indexes):
    visual = load_transcript_index(stub_indexes[1], "stub-visual", synthetic=True)
    search = TranscriptSearch(visual, FIXTURE_SETTINGS)
    def unexpected_encoder(_):
        pytest.fail("Encoding is unnecessary for an empty transcript")
    result = search.search("temporal IoU", unexpected_encoder, FIXTURE_STANDARDIZATION)
    assert (result.results, result.confidence, result.abstained, result.route) == ([], 0, True, "transcript")


@pytest.mark.parametrize(("mutation", "message"), [
    ("missing", "Missing transcript index artifact"),
    ("ids", "contiguous and zero-based"),
    ("bounds", "exceeds video duration"),
    ("text", "usable text"),
    ("dtype", "float32"),
    ("count", "passage count"),
    ("norm", "L2-normalized"),
    ("nonfinite", "finite"),
    ("pickle", "allow_pickle=False"),
])
def test_invalid_index_is_not_empty(stub_indexes, mutation, message):
    directory = stub_indexes[0]
    path = directory / "transcript.npy"
    if mutation == "missing":
        path.unlink()
        error = FileNotFoundError
    else:
        error = ValueError
        sql = {
            "ids": "UPDATE transcript_units SET unit_id=5 WHERE unit_id=2",
            "bounds": "UPDATE transcript_units SET end_s=31 WHERE unit_id=2",
            "text": "UPDATE transcript_units SET text=' ' WHERE unit_id=2",
        }
        if mutation in sql:
            with sqlite3.connect(directory / "index.sqlite") as conn:
                conn.execute(sql[mutation])
        else:
            vectors = np.load(path)
            replacements = {
                "dtype": vectors.astype("float64"), "count": vectors[:2],
                "norm": vectors * 2, "nonfinite": vectors * np.nan,
                "pickle": vectors.astype(object),
            }
            np.save(path, replacements[mutation])
    with pytest.raises(error, match=message):
        load_transcript_index(directory, "stub-spoken", synthetic=True)


def test_bm25_repeated_absent_and_empty_terms():
    lexical = LexicalIndex(["apple apple", "apple pear", "pear pear"], LexicalSettings())
    idf = np.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = [2 * idf * 2 / (2 + 1.5), 2 * idf / (1 + 1.5), 0]
    assert lexical.score(tokenize("APPLE missing apple")) == pytest.approx(expected)
    assert lexical.score(tokenize("missing")).tolist() == [0, 0, 0]
    assert lexical.score(tokenize("!?")).tolist() == [0, 0, 0]


def test_dense_paraphrase_and_weight_disagreement(stub_indexes):
    directory = stub_indexes[0]
    vectors = np.zeros((3, 16), dtype=np.float32)
    vectors[:, 0] = [1, -1, 0]
    vectors[2, 1] = 1
    np.save(directory / "transcript.npy", vectors)
    index = load_transcript_index(directory, "stub-spoken", synthetic=True)
    def encode(_):
        return vectors[1] * 4

    for weight, expected_start in [(0, 18), (0.01, 18), (0.99, 8), (1, 8)]:
        search = TranscriptSearch(index, SearchSettings(dense_weight=weight))
        result = search.search("temporal IoU", encode, FIXTURE_STANDARDIZATION)
        assert result.results[0].start_s == expected_start
    assert cosine_scores(vectors, encode("")).tolist() == [-1, 1, 0]
    combined = TranscriptSearch(index, FIXTURE_SETTINGS)
    paraphrase = "catalog audible phrases"
    assert combined.score(paraphrase, encode).bm25.tolist() == [0, 0, 0]
    assert combined.search(paraphrase, encode, FIXTURE_STANDARDIZATION).results[0].start_s == 8
    punctuation = combined.score("!?", encode)
    assert punctuation.lexical_term_count == 0
    assert punctuation.combined.tolist() == [-0.5, 0.5, 0]
    with pytest.raises(ValueError, match="non-whitespace"):
        combined.score(" ", encode)


@pytest.mark.parametrize("vector", [[1], [[1, 0]], [0, 0], [np.nan, 1], [np.inf, 0], [1j, 0]])
def test_invalid_query_vectors(vector):
    with pytest.raises(ValueError, match="Query embed|Query encoder"):
        cosine_scores(np.eye(2), np.array(vector))


def test_missing_real_encoder_is_lazy(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    encoder = QwenQueryEncoder("97b0c614")
    with pytest.raises(RuntimeError, match="Real B1 encoder.*unavailable"):
        encoder("temporal IoU")


def test_windows_negative_gap_plateau_fractional_and_suppression():
    units = [
        TranscriptUnit(unit_id=0, start_s=0, end_s=2, text="negative"),
        TranscriptUnit(unit_id=1, start_s=1.5, end_s=2, text="weaker overlap"),
        TranscriptUnit(unit_id=2, start_s=3.5, end_s=5.3, text="positive"),
    ]
    bins = project_passages(units, np.array([-1, -2, 3]), 5.3)
    assert bins.tolist() == [-1, -1, -np.inf, 3, 3, 3]
    windows = extract_windows(bins, 5.3, FIXTURE_SETTINGS)
    assert [(w.start_s, w.end_s, w.score) for w in windows] == [(3, 5.3, 3), (0, 1, -1)]
    overlapping = extract_windows(np.array([1, 0.75, 0.85, 0.6]), 4, FIXTURE_SETTINGS)
    assert [(w.start_s, w.end_s) for w in overlapping] == [(0, 3)]
    equal_iou = extract_windows(np.array([1, 0.6, 0.8, 0.6]), 4, FIXTURE_SETTINGS)
    assert [(w.start_s, w.end_s) for w in equal_iou] == [(0, 1), (0, 4)]
    exactly_half = extract_windows(np.array([1, 0.72, 0.55, 0.75]), 4, FIXTURE_SETTINGS)
    assert [(w.start_s, w.end_s) for w in exactly_half] == [(0, 2), (0, 4)]
    settings = SearchSettings(dense_weight=0.5, alpha=0.5)
    assert extract_windows(np.array([1, 0.5]), 2, settings)[0].end_s == 1
    assert extract_windows(
        np.array([1, 0.5]), 2, settings.model_copy(update={"include_cutoff": True}),
    )[0].end_s == 2


def test_fit_weights_covered_bins_equally():
    from vi.search.settings import fit_standardization

    fit = fit_standardization([np.array([1, -np.inf, 3]), np.array([2, 2, 2])])
    assert fit.covered_bins == 5
    assert fit.mean == 2
    assert fit.sd == pytest.approx(np.sqrt(2 / 5))
    fit = fit_standardization([np.array([0.1, 0.1, 0.1])])
    assert (fit.sd, fit.scale) == (0, 1)
    with pytest.raises(ValueError, match="without covered bins"):
        fit_standardization([np.array([-np.inf])])


def test_real_index_file_binding_and_query_encoder_provenance(stub_indexes):
    import hashlib
    import json

    from vi.search.dense import EncoderIdentity
    from vi.search.index import MODEL_ID, RealIndexProvenance

    directory = stub_indexes[0]
    database = directory / "index.sqlite"
    embeddings = directory / "transcript.npy"
    metadata = directory / "meta.json"
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE video SET dataset='team', split='dev'")
    conn.close()
    vectors = np.zeros((3, 1024), dtype=np.float32)
    vectors[:, :3] = np.eye(3)
    np.save(embeddings, vectors)
    provenance = RealIndexProvenance(
        config_hash="writer-config", encoder_revision="a" * 40,
        passage_policy={
            "target_span_s": 10, "stride": "every_segment", "gap": "include_elapsed_gap",
            "tail": "retain_short", "embedded_text": "exact_persisted_text",
            "passage_prompt": None, "query_prompt": "query",
        },
    )
    raw = json.loads(metadata.read_text())
    raw.update(
        synthetic=False, config_hash=provenance.config_hash,
        transcript_passage_policy=provenance.passage_policy.model_dump(),
        transcript_file_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (database, embeddings)
        },
    )
    raw["models"]["text_embed"] = MODEL_ID
    raw["revisions"]["text_embed"] = provenance.encoder_revision
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    index = load_transcript_index(directory, "stub-spoken", synthetic=False, expected=provenance)
    search = TranscriptSearch(index, FIXTURE_SETTINGS)
    with pytest.raises(ValueError, match="encoder provenance mismatch"):
        search.score("temporal IoU", lambda _: vectors[2])
    def encode(_):
        return vectors[2]
    encode.identity = EncoderIdentity(MODEL_ID, "a" * 40, 1024, "query", False)
    assert search.score("temporal IoU", encode).cosine.tolist() == [0, 0, 1]
    wrong_config = provenance.model_copy(update={"config_hash": "other-config"})
    with pytest.raises(ValueError, match="config provenance mismatch"):
        load_transcript_index(directory, "stub-spoken", synthetic=False, expected=wrong_config)
    np.save(embeddings, vectors[::-1])
    with pytest.raises(ValueError, match="hash mismatch"):
        load_transcript_index(directory, "stub-spoken", synthetic=False, expected=provenance)
