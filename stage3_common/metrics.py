"""Metric semantics for the stage3 evolution loop."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class MetricSpec:
    key: str
    description: str
    higher_is_better: bool
    include_in_prompt: bool = True
    decimals: int = 6


PRIMARY_METRIC = MetricSpec(
    key="score",
    description=(
        "Stage2 optimizer objective: "
        "(mean_l2_raw + alpha * mean_queries) / max(asr, asr_floor) "
        "+ failure_gap_penalty * mean_gap_to_threshold "
        "+ 10000 if asr == 0 (zero-ASR penalty). Lower is better."
    ),
    higher_is_better=False,
    include_in_prompt=True,
    decimals=6,
)


PROMPT_METRICS: tuple[MetricSpec, ...] = (
    PRIMARY_METRIC,
    MetricSpec("target_metric", "mean_l2_raw + alpha * mean_queries", False, True, 6),
    MetricSpec("asr", "Attack success rate", True, True, 4),
    MetricSpec("mean_l2_raw", "Mean raw L2 distortion", False, True, 4),
    MetricSpec("mean_queries", "Mean hash queries", False, True, 1),
    MetricSpec("mean_final_hash_l1", "Mean final hash distance to target", False, True, 3),
    MetricSpec("mean_gap_to_threshold", "Mean distance above the collision threshold", False, True, 3),
    MetricSpec("is_valid", "Fraction of image-pair evaluations without errors", True, True, 0),
)


def objective_value(metrics: dict[str, float]) -> float:
    value = float(metrics.get(PRIMARY_METRIC.key, inf))
    if value != value:
        return inf
    return value


def objective_sort_key(summary) -> float:
    return objective_value(summary.metrics)


def is_better_metrics(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    cand = objective_value(candidate)
    inc = objective_value(incumbent)
    return cand > inc if PRIMARY_METRIC.higher_is_better else cand < inc


def sort_summaries(summaries):
    return sorted(summaries, key=objective_sort_key, reverse=PRIMARY_METRIC.higher_is_better)
