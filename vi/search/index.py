"""Read persisted, row-aligned transcript passages and embeddings."""

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from vi.schema import IndexMeta, TranscriptUnit, VideoMetadata, validate_duration

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIMENSION = 1024
UNIT_NORM_ATOL = 1e-4


class PassagePolicy(BaseModel, frozen=True, extra="forbid"):
    """Provisional writer declaration. Search never reconstructs passages."""

    target_span_s: Literal[10]
    stride: Literal["every_segment"]
    gap: str = Field(pattern=r"\S")
    tail: str = Field(pattern=r"\S")
    embedded_text: Literal["exact_persisted_text"]
    passage_prompt: None
    query_prompt: Literal["query"]


class RealIndexProvenance(BaseModel, frozen=True, extra="forbid"):
    config_hash: str = Field(min_length=1)
    encoder_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    passage_policy: PassagePolicy


@dataclass(frozen=True)
class TranscriptIndex:
    video: VideoMetadata
    passages: tuple[TranscriptUnit, ...]
    embeddings: np.ndarray
    meta: IndexMeta
    file_hashes: dict[str, str]
    provenance: RealIndexProvenance | None


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_transcript_index(
    directory: Path, video_id: str, *, synthetic: bool,
    expected: RealIndexProvenance | None = None,
) -> TranscriptIndex:
    """Validate artifacts and recorded provenance, optionally matching a fitted real configuration."""
    if synthetic and expected is not None:
        raise ValueError("Synthetic indexes cannot have expected real provenance.")
    paths = {name: directory / name for name in ("index.sqlite", "transcript.npy", "meta.json")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing transcript index artifact: {path}")
    try:
        raw = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
        meta = IndexMeta.model_validate(raw)
        if raw.get("synthetic") is not synthetic or meta.video_id != video_id:
            raise ValueError("meta.json mode or video identity mismatch")
        for field in (meta.models.get("text_embed"), meta.revisions.get("text_embed"), raw.get("config_hash")):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("Missing text encoder or index config provenance")
        file_hashes = {name: _file_hash(path) for name, path in paths.items()}
        provenance = None
        if not synthetic:
            provenance = RealIndexProvenance(
                config_hash=raw["config_hash"], encoder_revision=meta.revisions["text_embed"],
                passage_policy=raw["transcript_passage_policy"],
            )
            if meta.models["text_embed"] != MODEL_ID or (expected is not None and provenance != expected):
                raise ValueError("Real transcript encoder or index config provenance mismatch")
            for name in ("index.sqlite", "transcript.npy"):
                if raw["transcript_file_sha256"][name] != file_hashes[name]:
                    raise ValueError(f"Transcript artifact hash mismatch: {name}")
            if paths["index.sqlite"].with_name("index.sqlite-wal").exists():
                raise ValueError("Real transcript database must be published without a WAL sidecar")
            dimension = MODEL_DIMENSION
        else:
            dimension = raw["synthetic_dimension"]
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Invalid transcript embedding dimension in metadata")
        uri = paths["index.sqlite"].resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("BEGIN")
            videos = conn.execute(
                "SELECT video_id, path, duration_s, sha256, dataset, split FROM video"
            ).fetchmany(2)
            if len(videos) != 1:
                raise ValueError("Expected exactly one SQLite video record")
            video = VideoMetadata(**dict(zip(VideoMetadata.model_fields, videos[0])))
            if videos[0][0] != video_id or not video.dataset.strip() or not video.split.strip():
                raise ValueError("SQLite video identity, dataset or split is invalid")
            if synthetic and (video.dataset, video.split) != ("synthetic", "synthetic"):
                raise ValueError("Synthetic transcript index dataset or split mismatch")
            if not synthetic and ("synthetic" in (video.dataset, video.split)):
                raise ValueError("Real transcript index has synthetic dataset or split")
            rows = conn.execute(
                "SELECT unit_id, start_s, end_s, text FROM transcript_units ORDER BY unit_id"
            ).fetchall()
        passages = []
        for position, (unit_id, start, end, content) in enumerate(rows):
            if type(unit_id) is not int or unit_id != position:
                raise ValueError("Transcript unit IDs must be contiguous and zero-based")
            if type(start) not in (int, float) or type(end) not in (int, float):
                raise ValueError(f"Invalid timestamps for transcript unit {unit_id}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Missing usable text for transcript unit {unit_id}")
            unit = TranscriptUnit(unit_id=unit_id, start_s=start, end_s=end, text=content)
            validate_duration(unit, video)
            passages.append(unit)
        vectors = np.load(paths["transcript.npy"], allow_pickle=False)
        if not isinstance(vectors, np.ndarray):
            vectors.close()
            raise ValueError("transcript.npy is not a NumPy array")
        if vectors.dtype != np.dtype("float32") or vectors.shape != (len(passages), dimension):
            raise ValueError("Transcript embeddings must be float32 with shape (passage count, dimension)")
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors.astype(np.float64), axis=1), 1, rtol=0, atol=UNIT_NORM_ATOL,
        ):
            raise ValueError("Transcript embeddings must be finite and L2-normalized")
        vectors.setflags(write=False)
    except (ValueError, KeyError, TypeError, OSError, EOFError, sqlite3.Error) as exc:
        raise ValueError(f"Invalid transcript index at {directory}: {exc}") from exc
    return TranscriptIndex(video, tuple(passages), vectors, meta, file_hashes, provenance)
