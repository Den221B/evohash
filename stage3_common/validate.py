"""Validate evolved run_attack programs through the production evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import platform
from typing import Any, Iterable

import numpy as np

from evohash.dataset import PairSample
from evohash.optimizer import (
    OptimizerSettings,
    load_attack_config,
    load_pairs,
    resolve_parallel_runtime,
)
from evohash.parallel_eval import run_eval_jobs

from .config import EvaluationConfig
from .metrics import objective_value


@dataclass
class ProgramEvalSummary:
    program_path: str
    metrics: dict[str, float]
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: str = ""
    cell_key: tuple[int, ...] | None = None

    @property
    def score(self) -> float:
        return objective_value(self.metrics)

    def to_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "program_path": self.program_path,
            "metrics": dict(self.metrics),
            "errors": list(self.errors),
            "diagnostics": self.diagnostics,
            "cell_key": list(self.cell_key) if self.cell_key is not None else None,
        }
        if include_rows:
            data["rows"] = list(self.rows)
        return data


def load_eval_pairs(config: EvaluationConfig) -> list[PairSample]:
    cfg = config.normalized()
    return load_pairs(cfg.dataset_zip, split=cfg.split, n_pairs=cfg.max_pairs)


def validate_program(
    program_path: str | Path,
    config: EvaluationConfig,
    *,
    pairs: list[PairSample] | None = None,
    program_id: str | None = None,
) -> ProgramEvalSummary:
    """Evaluate a candidate program using the parallel image evaluation pipeline.

    Candidate files must define the same contract as normal attacks:

    ``run_attack(x_source, oracle, params=None, budget=..., seed=...)``.
    """
    cfg = config.normalized()
    if cfg.hash_id == "photodna" and platform.system() != "Windows":
        raise RuntimeError("PhotoDNA evolution is enabled only on Windows in this project pipeline.")
    pair_list = pairs if pairs is not None else load_eval_pairs(cfg)
    attack_id = program_id or Path(program_path).stem

    attack_config = _load_or_minimal_attack_config(cfg)
    settings = _optimizer_settings(cfg, attack_id=attack_id)
    runtime = resolve_parallel_runtime(settings, len(pair_list))

    jobs = []
    for index, sample in enumerate(pair_list):
        jobs.append(
            {
                "job_index": int(index),
                "hash_id": cfg.hash_id,
                "attack_id": attack_id,
                "program_path": str(Path(program_path).resolve()),
                "pair_index": int(index),
                "pair_id": getattr(sample, "pair_id", str(index)),
                "x_source": sample.x,
                "x_target": sample.y,
                "params": {},
                "seed": int(cfg.seed) + int(index),
            }
        )

    row_dicts = run_eval_jobs(
        jobs,
        config=attack_config,
        budget=int(cfg.budget),
        resize_size=cfg.resize_size,
        resize_resample=cfg.resize_resample,
        alpha=float(cfg.alpha),
        backend=runtime.backend,
        max_workers=runtime.workers,
        process_start_method=runtime.process_start_method,
        show_progress=cfg.show_progress,
    )
    metrics = summarize_row_dicts(row_dicts, config=cfg)
    errors = [
        str(row.get("error"))
        for row in row_dicts
        if row.get("status") != "ok" and row.get("error")
    ]
    diagnostics = format_feedback_diagnostics(
        row_dicts,
        metrics=metrics,
        config=cfg,
        errors=errors,
    )
    return ProgramEvalSummary(
        program_path=str(Path(program_path)),
        metrics=metrics,
        rows=row_dicts,
        errors=errors,
        diagnostics=diagnostics,
    )


def summarize_row_dicts(rows: Iterable[dict[str, Any]], *, config: EvaluationConfig) -> dict[str, float]:
    row_list = list(rows)
    if not row_list:
        return _empty_metrics()

    ok = np.array([row.get("status") == "ok" for row in row_list], dtype=np.float32)
    success = np.array([bool(row.get("success", False)) for row in row_list], dtype=np.float32)
    pixel_l2 = _finite_array((row.get("pixel_l2", 1.0) for row in row_list), default=1.0)
    pixel_l2_raw = _finite_array((row.get("pixel_l2_raw", 1e9) for row in row_list), default=1e9)
    queries = _finite_array((row.get("queries", 0.0) for row in row_list), default=0.0)
    initial = _finite_array((row.get("initial_hash_l1", 0.0) for row in row_list), default=0.0)
    final = _finite_array((row.get("final_hash_l1", 1e9) for row in row_list), default=1e9)
    best = _finite_array((row.get("best_hash_l1", 1e9) for row in row_list), default=1e9)
    threshold = _finite_array((row.get("threshold", 0.0) for row in row_list), default=0.0)
    rel = _finite_array((row.get("relative_improvement", 0.0) for row in row_list), default=0.0)
    time_sec = _finite_array((row.get("time_sec", 0.0) for row in row_list), default=0.0)
    gap = np.maximum(final - threshold, 0.0)

    asr = float(success.mean())
    mean_pixel_l2 = float(pixel_l2.mean())
    mean_l2_raw = float(pixel_l2_raw.mean())
    mean_queries = float(queries.mean())
    mean_gap = float(gap.mean())

    efficiency = float(asr / (mean_pixel_l2 + 1e-8))
    target_metric = float(mean_l2_raw + float(config.alpha) * mean_queries)
    score = float(
        target_metric / max(asr, float(config.asr_floor))
        + float(config.failure_gap_penalty) * mean_gap
        + (10_000.0 if asr == 0.0 else 0.0)
    )
    if float(ok.mean()) < 1.0:
        score = 1e9

    return {
        "is_valid": float(ok.mean()),
        "score": score,
        "optimizer_score": score,
        "efficiency": efficiency,
        "target_metric": target_metric,
        "asr": asr,
        "n_success": float(success.sum()),
        "n_pairs": float(len(row_list)),
        "mean_initial_hash_l1": float(initial.mean()),
        "mean_final_hash_l1": float(final.mean()),
        "mean_best_hash_l1": float(best.mean()),
        "mean_gap_to_threshold": mean_gap,
        "mean_relative_improvement": float(rel.mean()),
        "mean_pixel_l2": mean_pixel_l2,
        "median_pixel_l2": float(np.median(pixel_l2)),
        "mean_l2_raw": mean_l2_raw,
        "median_l2_raw": float(np.median(pixel_l2_raw)),
        "mean_queries": mean_queries,
        "median_queries": float(np.median(queries)),
        "mean_time_sec": float(time_sec.mean()),
    }


def format_feedback_diagnostics(
    rows: Iterable[dict[str, Any]],
    *,
    metrics: dict[str, float],
    config: EvaluationConfig,
    errors: Iterable[str] | None = None,
) -> str:
    """Format compact, hash-agnostic diagnostics for the next LLM mutation."""
    row_list = list(rows)
    if errors is None:
        error_list = [
            str(row.get("error"))
            for row in row_list
            if row.get("status") != "ok" and row.get("error")
        ]
    else:
        error_list = [str(error) for error in errors if str(error).strip()]

    is_valid = _metric(metrics, "is_valid", 0.0)
    score = _metric(metrics, "score", 1e9)
    asr = _metric(metrics, "asr", 0.0)
    n_pairs = int(round(_metric(metrics, "n_pairs", float(len(row_list)))))
    if n_pairs <= 0:
        n_pairs = len(row_list)
    n_success = int(round(_metric(metrics, "n_success", asr * max(n_pairs, 1))))
    mean_l2 = _metric(metrics, "mean_l2_raw", 1e9)
    mean_queries = _metric(metrics, "mean_queries", 0.0)
    median_queries = _metric(metrics, "median_queries", mean_queries)
    mean_gap = _metric(metrics, "mean_gap_to_threshold", 1e9)
    rel_improvement = _metric(metrics, "mean_relative_improvement", 0.0)

    budget = max(1, int(config.budget))
    query_fraction = mean_queries / float(budget)
    progress, mean_best_gap = _threshold_progress(row_list)

    lines = [
        "Evaluation feedback for the previous mutation:",
        (
            "- Outcome: "
            f"valid={_fmt(is_valid)}, score={_fmt(score)}, "
            f"successes={n_success}/{max(n_pairs, 0)}, asr={_pct(asr)}."
        ),
        (
            "- Distortion/query cost: "
            f"mean_l2_raw={_fmt(mean_l2)}, "
            f"mean_queries={_fmt(mean_queries)}/{budget} ({_pct(query_fraction)} of budget), "
            f"median_queries={_fmt(median_queries)}."
        ),
    ]

    if progress is None:
        lines.append(
            "- Threshold progress: no valid evaluator rows were produced; "
            "algorithmic metrics are not meaningful yet."
        )
    else:
        lines.append(
            "- Threshold progress: "
            f"{_pct(progress)} of the initial gap closed on average, "
            f"mean_final_gap={_fmt(mean_gap)}, mean_best_gap={_fmt(mean_best_gap)}, "
            f"relative_improvement={_pct(rel_improvement)}."
        )

    if error_list:
        lines.append(
            "- Runtime issues: "
            f"{len(error_list)} failing row(s). First error: {_one_line(error_list[0], limit=320)}"
        )

    lines.append(
        "- Trend: "
        + _trend_hint(
            is_valid=is_valid,
            asr=asr,
            mean_gap=mean_gap,
            rel_improvement=rel_improvement,
            query_fraction=query_fraction,
            progress=progress,
            has_errors=bool(error_list),
        )
    )
    lines.append(
        "- Next mutation focus: "
        + _next_focus(
            is_valid=is_valid,
            asr=asr,
            mean_gap=mean_gap,
            query_fraction=query_fraction,
            progress=progress,
            has_errors=bool(error_list),
        )
    )
    lines.append(
        "- Mutation hygiene: change one mechanism at a time, keep the contract, "
        "and preserve deterministic seed use."
    )
    return "\n".join(lines)


def _optimizer_settings(config: EvaluationConfig, *, attack_id: str) -> OptimizerSettings:
    return OptimizerSettings(
        hash_id=config.hash_id,
        attack_id=attack_id,
        resize_size=config.resize_size,
        resize_resample=config.resize_resample,
        budget=int(config.budget),
        seed=int(config.seed),
        alpha=float(config.alpha),
        asr_floor=float(config.asr_floor),
        failure_gap_penalty=float(config.failure_gap_penalty),
        parallel_backend=config.parallel_backend,
        parallel_workers=config.workers,
        parallel_worker_cap=int(config.worker_cap),
        parallel_photodna_worker_cap=config.photodna_worker_cap,
        parallel_process_start_method=config.process_start_method,
        parallel_show_progress=config.show_progress,
    )


def _load_or_minimal_attack_config(config: EvaluationConfig) -> dict[str, Any]:
    path = Path(config.attack_config_path)
    if path.exists():
        try:
            return load_attack_config(path)
        except Exception:
            pass
    return {
        "thresholds": {
            "phash": 12.0,
            "pdq": 92.0,
            "neuralhash": 17.0,
            "photodna": 3855.0,
        },
        "configs": {config.hash_id: {}},
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "is_valid": 0.0,
        "score": 1e9,
        "efficiency": 0.0,
        "optimizer_score": 1e9,
        "target_metric": 1e9,
        "asr": 0.0,
        "mean_pixel_l2": 1.0,
        "mean_l2_raw": 1e9,
        "mean_queries": 0.0,
        "mean_final_hash_l1": 1e9,
        "mean_gap_to_threshold": 1e9,
        "mean_relative_improvement": 0.0,
        "n_pairs": 0.0,
    }


def _threshold_progress(rows: list[dict[str, Any]]) -> tuple[float | None, float]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return None, 1e9
    initial = _finite_array((row.get("initial_hash_l1", 0.0) for row in ok_rows), default=0.0)
    final = _finite_array((row.get("final_hash_l1", 1e9) for row in ok_rows), default=1e9)
    best = _finite_array((row.get("best_hash_l1", 1e9) for row in ok_rows), default=1e9)
    threshold = _finite_array((row.get("threshold", 0.0) for row in ok_rows), default=0.0)
    denom = np.maximum(initial - threshold, 1e-6)
    progress = np.where(
        initial > threshold,
        (initial - final) / denom,
        np.where(final <= threshold, 1.0, 0.0),
    )
    progress = np.clip(progress, -1.0, 1.0)
    best_gap = np.maximum(best - threshold, 0.0)
    return float(progress.mean()), float(best_gap.mean())


def _trend_hint(
    *,
    is_valid: float,
    asr: float,
    mean_gap: float,
    rel_improvement: float,
    query_fraction: float,
    progress: float | None,
    has_errors: bool,
) -> str:
    if is_valid < 1.0 or has_errors:
        return "candidate is invalid or partially failed; algorithmic score cannot be trusted."
    if progress is None:
        return "no usable evaluator rows; fix execution before changing search logic."
    if asr <= 0.0 and progress < 0.05 and rel_improvement < 0.05:
        return "flat: hash distance barely improved before the budget ended."
    if asr < 1.0 and progress >= 0.5 and mean_gap > 0.0:
        return "converging but not reliably crossing the threshold yet."
    if asr < 1.0 and query_fraction >= 0.8:
        return "spends most of the budget before reliable collisions."
    if asr >= 1.0 and query_fraction >= 0.8:
        return "successful but expensive; there is room for earlier stopping or cheaper refinement."
    if asr >= 1.0 and mean_gap <= 0.0:
        return "threshold is being crossed; distortion and query cleanup are now the main levers."
    return "mixed signal; keep the working scaffold and make a narrow search-mechanism mutation."


def _next_focus(
    *,
    is_valid: float,
    asr: float,
    mean_gap: float,
    query_fraction: float,
    progress: float | None,
    has_errors: bool,
) -> str:
    if is_valid < 1.0 or has_errors:
        return "fix the runtime/contract failure first, then re-evaluate the same search idea."
    if progress is None:
        return "produce valid rows before optimizing the objective."
    if asr < 1.0:
        if progress >= 0.5 and mean_gap > 0.0:
            return "improve threshold crossing reliability with step adaptation, restarts, or better direction sampling."
        return "increase hash-distance progress before L2 cleanup; try a different black-box proposal family or schedule."
    if query_fraction >= 0.7:
        return "preserve the collision while adding early stop, query gating, or a shorter refinement phase."
    return "preserve successes and shrink L2 via local refinement, step decay, or rollback to the best threshold-hitting point."


def _metric(metrics: dict[str, float], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _fmt(value: float) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "nan"
    except Exception:
        return "nan"
    return f"{number:.6g}"


def _pct(value: float) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "nan%"
    except Exception:
        return "nan%"
    return f"{100.0 * number:.1f}%"


def _one_line(text: str, *, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _finite_array(values: Iterable[Any], *, default: float) -> np.ndarray:
    out = []
    for value in values:
        try:
            number = float(value)
            if not np.isfinite(number):
                number = float(default)
        except Exception:
            number = float(default)
        out.append(number)
    return np.asarray(out, dtype=np.float32)
