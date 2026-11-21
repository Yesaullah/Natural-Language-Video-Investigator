"""B1 evaluation and paired additive metrics over whole-video draws."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from vi.eval import metrics
from vi.eval.bootstrap import video_level_bootstrap
from vi.schema import QueryLabel, SearchResult, VideoMetadata
from vi.search.dense import QwenQueryEncoder, SyntheticEncoder
from vi.search.index import TranscriptIndex, load_transcript_index
from vi.search.pipeline import TranscriptSearch
from vi.search.settings import FIXTURE_SETTINGS, FIXTURE_STANDARDIZATION, RealSettings

QueryKey = tuple[str, str]
RECALL_SETTINGS = {
    f"R@{k}_iou{threshold}": (k, threshold)
    for k in (1, 5) for threshold in (0.3, 0.5, 0.7)
}


def key_for(label: QueryLabel) -> QueryKey:
    return label.video_id, label.qid


def check_prediction_keys(labels: list[QueryLabel], predictions: dict) -> None:
    expected = {key_for(label) for label in labels}
    if len(expected) != len(labels) or expected != predictions.keys():
        raise ValueError("Prediction keys do not match unique evaluation query keys")


class VideoDraws:
    """Keep the full video population when evaluating restricted query subsets."""

    def __init__(self, labels: list[QueryLabel], count: int, seed: int):
        self.videos = sorted({q.video_id for q in labels})
        self.count = count
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(len(self.videos), size=(count, len(self.videos)))

    def summarize(self, values: dict[QueryKey, float]) -> dict:
        counts = dict.fromkeys(self.videos, 0)
        sums = dict.fromkeys(self.videos, 0.0)
        for (video, _), value in sorted(values.items()):
            counts[video] += 1
            sums[video] += value
        denominators = np.array(list(counts.values()))[self.draws].sum(axis=1)
        numerators = np.array(list(sums.values()))[self.draws].sum(axis=1)
        valid = denominators > 0
        if valid.any():
            lower, upper = np.percentile(numerators[valid] / denominators[valid], [2.5, 97.5])
        else:
            lower, upper = None, None
        return {
            "point": sum(sums.values()) / len(values) if values else None,
            "ci_lower": lower, "ci_upper": upper, "n_resamples": self.count,
            "n_valid": int(valid.sum()), "n_undefined": int((~valid).sum()),
            **({} if values else {"reason": "No eligible queries."}),
        }


def spoken_comparison(labels, b0_top1, b1_predictions, draws: VideoDraws) -> dict:
    check_prediction_keys(labels, b1_predictions)
    if b0_top1.keys() != {key_for(q) for q in labels if q.answerable}:
        raise ValueError("B0 trial averages do not match answerable query keys")
    spoken = [q for q in labels if q.answerable and q.qtype == "spoken"]
    b0 = {key_for(q): b0_top1[key_for(q)] for q in spoken}
    b1 = {
        key_for(q): float(metrics.recall_at_k([q], [b1_predictions[key_for(q)]], 1, 0.5).numerator)
        for q in spoken
    }
    return {
        "metric": "Answerable spoken raw R@1 at IoU 0.5",
        "b0_convention": "Per-query mean hit over all B0 trials",
        "query_keys": sorted(b0),
        "query_count": len(b0), "spoken_video_count": len({video for video, _ in b0}),
        "resampling_video_count": len(draws.videos),
        "B0": draws.summarize(b0), "B1": draws.summarize(b1),
        "B1_minus_B0": draws.summarize({key: b1[key] - b0[key] for key in b0}),
    }


def query_subgroups(labels: list[QueryLabel]) -> dict[str, list[QueryLabel]]:
    groups = {
        qtype: [q for q in labels if q.answerable and q.qtype == qtype]
        for qtype in ("spoken", "visual", "temporal", "mixed")
    }
    groups["multi_event"] = [q for q in labels if q.answerable and q.multi_event]
    for kind, aliases in (
        ("near_miss", {"near_miss"}), ("swap", {"swap"}),
        ("ood", {"ood", "out_of_domain"}), ("in_domain", {"id", "in_domain"}),
    ):
        groups[kind] = [q for q in labels if not q.answerable and q.neg_type in aliases]
    return groups


def with_wilson(rate: metrics.RateResult) -> dict:
    lo, hi = metrics.wilson_score_interval(rate.numerator, rate.denominator)
    return {
        "rate": rate.rate, "hits": rate.numerator, "denominator": rate.denominator,
        "wilson_95": {"lower": lo, "upper": hi},
        **({} if rate.denominator else {"reason": "No eligible queries."}),
    }


def _retrieval(labels, predictions, draws: VideoDraws, aware: bool) -> dict:
    results = [predictions[key_for(q)] for q in labels]
    report = {}
    for name, (k, threshold) in RECALL_SETTINGS.items():
        rate = metrics.recall_at_k(labels, results, k, threshold, aware)
        values = {
            key_for(q): metrics.recall_at_k([q], [predictions[key_for(q)]], k, threshold, aware).numerator
            for q in labels if q.answerable
        }
        report[name] = {**with_wilson(rate), "cluster_bootstrap_95": draws.summarize(values)}
    for name, fn in (("MRR", metrics.mean_reciprocal_rank), ("top_window_mean_iou", metrics.top_window_mean_iou)):
        values = {
            key_for(q): fn([q], [predictions[key_for(q)]], abstention_aware=aware)
            for q in labels if q.answerable
        }
        summary = draws.summarize(values)
        report[name] = {"value": summary["point"], "count": len(values), "cluster_bootstrap_95": summary}
    start = metrics.start_time_error(labels, results, abstention_aware=aware)
    report["start_time_error"] = asdict(start)
    if not start.count:
        report["start_time_error"]["reason"] = "No top window overlapped a gold window."
    return report


def load_b1_indexes(
    directories: dict[str, Path], expected_videos: dict[str, VideoMetadata], *, synthetic: bool,
) -> tuple[dict[str, TranscriptIndex], float]:
    """Validate all selected indexes before checking speech coverage or scoring."""
    start = perf_counter()
    indexes = {}
    for vid, directory in sorted(directories.items()):
        index = load_transcript_index(directory, vid, synthetic=synthetic)
        if index.video != expected_videos[vid]:
            raise ValueError(f"Video metadata changed while loading B1 index for {vid}")
        indexes[vid] = index
    return indexes, perf_counter() - start


def charades_applicability(indexes: dict[str, TranscriptIndex]) -> dict:
    numerator = sum(bool(index.passages) for index in indexes.values())
    denominator = len(indexes)
    applicable = numerator * 10 >= denominator
    return {
        "applicable": applicable,
        "speech_coverage": {
            "numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator, "minimum_fraction": 0.1,
        },
        **({} if applicable else {"reason": "Fewer than 10% of selected Charades dev videos have speech."}),
    }


def run_b1(
    labels: list[QueryLabel], loaded_indexes: dict[str, TranscriptIndex],
    real: RealSettings | None, draws: VideoDraws, bootstrap_seed: int, *, index_load_seconds: float,
) -> tuple[dict, dict[QueryKey, SearchResult]]:
    start = perf_counter()
    settings = real.search if real else FIXTURE_SETTINGS
    fit = real.standardization if real else FIXTURE_STANDARDIZATION
    if real:
        real.validate_fit()
        for vid, index in loaded_indexes.items():
            if index.provenance != real.index:
                raise ValueError(f"Real transcript encoder or index config provenance mismatch for {vid}")
        encoder = QwenQueryEncoder(real.index.encoder_revision) if settings.dense_weight else None
    else:
        encoder = None
    searches = {}
    indexes = {}
    for vid, index in loaded_indexes.items():
        if real is None:
            dimension = index.embeddings.shape[1]
            if encoder is None:
                encoder = SyntheticEncoder(dimension)
            if encoder.identity.dimension != dimension:
                raise ValueError("Synthetic B1 indexes have inconsistent encoder dimensions")
        searches[vid] = TranscriptSearch(index, settings)
        indexes[vid] = {
            "video": index.video.model_dump(),
            "text_encoder": index.meta.models["text_embed"],
            "text_encoder_revision": index.meta.revisions["text_embed"],
            "index_config_hash": index.meta.config_hash, "artifact_sha256": index.file_hashes,
            "passage_count": len(index.passages), "embedding_dimension": index.embeddings.shape[1],
        }
    load_seconds = index_load_seconds + (perf_counter() - start)
    start = perf_counter()
    predictions = {key_for(q): searches[q.video_id].search(q.query, encoder, fit) for q in labels}
    query_seconds = perf_counter() - start
    check_prediction_keys(labels, predictions)
    results = [predictions[key_for(q)] for q in labels]
    abstention = {
        name: with_wilson(fn(labels, results))
        for name, fn in (
            ("rejection_accuracy", metrics.rejection_rate), ("false_answer_rate", metrics.false_answer_rate),
            ("false_abstain_rate", metrics.false_abstain_rate),
        )
    }
    for name, fn, eligible in (
        ("rejection_accuracy", metrics.rejection_rate, False),
        ("false_answer_rate", metrics.false_answer_rate, False),
        ("false_abstain_rate", metrics.false_abstain_rate, True),
    ):
        kept = {
            key_for(q): fn([q], [predictions[key_for(q)]]).numerator for q in labels if q.answerable == eligible
        }
        abstention[name]["cluster_bootstrap_95"] = draws.summarize(kept)

    def bootstrap(fn: Callable) -> dict:
        return asdict(video_level_bootstrap(
            labels, results, fn, n_resamples=draws.count, seed=bootstrap_seed,
        ))

    balanced = metrics.balanced_accuracy(labels, results)
    abstention["balanced_accuracy"] = {
        "value": balanced, "count": len(labels),
        "cluster_bootstrap_95": bootstrap(metrics.balanced_accuracy),
        **({} if balanced is not None else {"reason": "Both answerability classes are required."}),
    }
    risk = metrics.risk_coverage_curve(labels, results)
    answerable = sum(q.answerable for q in labels)
    risk["query_mix"] = {"answerable": answerable, "unanswerable": len(labels) - answerable}
    risk["cluster_bootstrap_95"] = bootstrap(lambda qs, rs: metrics.risk_coverage_curve(qs, rs)["area"])
    retrieval = {
        name: _retrieval(labels, predictions, draws, aware)
        for name, aware in (("raw", False), ("abstention_aware", True))
    }
    subgroups = {}
    for name, subset in query_subgroups(labels).items():
        group = {
            "query_count": len(subset), "video_count": len({q.video_id for q in subset}),
            "descriptive": True,
        }
        subset_results = [predictions[key_for(q)] for q in subset]
        if name in {"near_miss", "swap", "ood", "in_domain"}:
            group["rejection_accuracy"] = {
                **with_wilson(metrics.rejection_rate(subset, subset_results)),
                "cluster_bootstrap_95": draws.summarize({
                    key_for(q): int(predictions[key_for(q)].abstained) for q in subset
                }),
            }
        else:
            group["retrieval"] = {
                mode: _retrieval(subset, predictions, draws, aware)
                for mode, aware in (("raw", False), ("abstention_aware", True))
            }
        subgroups[name] = group

    latencies = [sum(result.timings_ms.values()) for result in results]
    return {
        "settings": {
            "variant": settings.variant,
            "search": settings.model_dump(), "scoring": settings.scoring_identity(),
            "standardization": fit.model_dump(), "standardization_source": "real_dev" if real else "fixture_constants",
            "zero_variance_scale": 1.0, "fitted_settings": real.model_dump() if real else None,
            "query_encoder": asdict(encoder.identity) if encoder else None,
            "confidence": "Uncalibrated sigmoid of the top standardized score. An empty transcript abstains at zero confidence.",
            "window_policy": {
                "seed": "Earliest bin of each local-maximum plateau, retained even when nonpositive",
                "ties": "Earliest start, then earliest end", "suppression_iou_strictly_above": 0.5,
                "limit": 5,
            },
        },
        "indexes": indexes,
        "query_keys": sorted(predictions),
        "top_scores": [
            predictions[key].results[0].score if predictions[key].results else None
            for key in sorted(predictions)
        ],
        "retrieval": retrieval, "abstention": abstention,
        "risk_coverage": risk, "subgroups": subgroups,
        "routing_scores": {"value": None, "count": 0, "reason": "B1 uses a single transcript path."},
        "gpu_measurements": {"value": None, "count": 0, "reason": "This evaluator runs on CPU."},
        "indexing_cost": {"value": None, "count": 0, "reason": "This evaluator reads prebuilt indexes."},
        "cpu_timing": {
            "load_indexes_and_bm25_s": load_seconds, "search_s": query_seconds,
            "query_latency_p50_ms": float(np.percentile(latencies, 50)),
            "query_latency_p95_ms": float(np.percentile(latencies, 95)),
            "note": "Search timing can include the first lazy model load.",
        },
    }, predictions


def b0_subgroups(labels: list[QueryLabel], details: dict, draws: VideoDraws) -> dict:
    groups = {}
    for name, subset in query_subgroups(labels).items():
        results = [details["first_trial"][key_for(q)] for q in subset]
        keys = [key_for(q) for q in subset]
        group = {
            "query_count": len(keys), "video_count": len({video for video, _ in keys}),
            "descriptive": True,
        }
        if name in {"near_miss", "swap", "ood", "in_domain"}:
            group["rejection_accuracy"] = {
                **with_wilson(metrics.rejection_rate(subset, results)),
                "cluster_bootstrap_95": draws.summarize(dict.fromkeys(keys, 0)),
            }
        else:
            group["retrieval"] = {
                "raw_equals_abstention_aware": True,
                "trial_mean_cluster_bootstrap_95": {
                    metric: draws.summarize({key: averages[key] for key in keys})
                    for metric, averages in details["trial_means"].items()
                },
                "first_trial": {
                    metric: with_wilson(metrics.recall_at_k(subset, results, k, threshold))
                    for metric, (k, threshold) in RECALL_SETTINGS.items()
                },
            }
        groups[name] = group
    return groups
