"""Runtime helpers used by generated GigaEvolve problem packages."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from evohash.hashes import build_hash_fn
from evohash.optimizer import (
    OptimizerSettings,
    load_attack_config,
    load_pairs,
    resolve_parallel_runtime,
)
from evohash.parallel_eval import run_eval_jobs
from evohash.stage3_common.config import EvaluationConfig
from evohash.stage3_common.validate import format_feedback_diagnostics, summarize_row_dicts


def build_context_from_env() -> dict[str, Any]:
    """Build the context dict passed to GigaEvolve entrypoint(context)."""
    hash_id = os.environ.get("EVOHASH_HASH_ID", "phash")
    dataset_zip = os.environ.get("EVOHASH_DATASET_ZIP", "dataset_v1_public.zip")
    split = os.environ.get("EVOHASH_SPLIT", "public")
    max_pairs = _env_int("EVOHASH_MAX_PAIRS", 10)
    budget = _env_int("EVOHASH_BUDGET", 10_000)
    resize_size = _env_int("EVOHASH_RESIZE_SIZE", 256)
    seed = _env_int("EVOHASH_SEED", 0)
    attack_config_path = os.environ.get(
        "EVOHASH_ATTACK_CONFIG",
        "results/attack_configs_optimized.json",
    )

    pairs = load_pairs(dataset_zip, split=split, n_pairs=max_pairs)
    return {
        "hash_id": hash_id,
        "dataset_zip": str(dataset_zip),
        "split": split,
        "max_pairs": max_pairs,
        "budget": budget,
        "resize_size": resize_size,
        "resize_resample": os.environ.get("EVOHASH_RESIZE_RESAMPLE", "bilinear"),
        "seed": seed,
        "alpha": _env_float("EVOHASH_ALPHA", 1.0 / 50.0),
        "asr_floor": _env_float("EVOHASH_ASR_FLOOR", 0.05),
        "failure_gap_penalty": _env_float("EVOHASH_FAILURE_GAP_PENALTY", 0.0),
        "parallel_backend": os.environ.get("EVOHASH_PARALLEL_BACKEND", "auto"),
        "parallel_workers": _env_int("EVOHASH_PARALLEL_WORKERS", None),
        "parallel_worker_cap": _env_int("EVOHASH_PARALLEL_WORKER_CAP", 30),
        "attack_config_path": str(attack_config_path),
        "pairs": pairs,
        "source_images": [sample.x for sample in pairs],
        "target_images": [sample.y for sample in pairs],
    }


def evaluate_entrypoint_attack(
    run_attack: Callable[..., Any],
    context: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
    attack_id: str = "candidate",
) -> dict[str, Any]:
    """Evaluate a dynamic run_attack function over context pairs."""
    cfg = _eval_config(context)
    config = _load_attack_config(cfg.attack_config_path, cfg.hash_id)
    hash_fn = build_hash_fn(cfg.hash_id, thresholds=config.get("thresholds", {}))
    attack = SimpleNamespace(
        ATTACK_ID=str(attack_id),
        DEFAULT_PARAMS=dict(params or {}),
        run_attack=run_attack,
    )
    pairs = list(context.get("pairs") or [])
    settings = OptimizerSettings(
        hash_id=cfg.hash_id,
        attack_id=str(attack_id),
        budget=int(cfg.budget),
        seed=int(cfg.seed),
        resize_size=cfg.resize_size,
        resize_resample=cfg.resize_resample,
        alpha=float(cfg.alpha),
        asr_floor=float(cfg.asr_floor),
        failure_gap_penalty=float(cfg.failure_gap_penalty),
        parallel_backend=cfg.parallel_backend,
        parallel_workers=cfg.workers,
        parallel_worker_cap=int(cfg.worker_cap),
    )
    runtime = resolve_parallel_runtime(settings, len(pairs))
    raw_code_path = os.environ.get("GIGAEVO_PROGRAM_CODE_PATH")
    code_path = Path(raw_code_path) if raw_code_path else None

    if code_path is not None and code_path.is_file():
        # File-backed path: run_attack code lives in a file on disk.
        # Delegate entirely to run_eval_jobs — supports all backends uniformly.
        rows = _evaluate_file_backed(
            code_path=code_path,
            context=context,
            cfg=cfg,
            config=config,
            runtime=runtime,
            attack_id=attack_id,
        )
        return {"rows": rows, "attack_id": str(attack_id), "eval_backend": runtime.backend}

    # In-process path: run_attack is a live callable from GigaEvolve's sandbox.
    # Local functions are not pickle-friendly, so we use a thread (or sequential)
    # backend and evaluate directly via evaluate_attack().
    backend = "thread" if runtime.backend == "process" else runtime.backend
    workers = runtime.workers

    jobs = []
    for index, sample in enumerate(pairs):
        jobs.append({
            "job_index":    int(index),
            "hash_id":      cfg.hash_id,
            "attack_id":    str(attack_id),
            "pair_index":   int(index),
            "pair_id":      getattr(sample, "pair_id", str(index)),
            "x_source":     sample.x,
            "x_target":     sample.y,
            "params":       dict(params or {}),
            "seed":         int(cfg.seed) + int(index),
            "_live_attack": attack,
        })

    # Initialise the worker context (sets _WORKER_CONTEXT used by _get_hash).
    from evohash.parallel_eval import _init_worker, _get_hash as _peval_get_hash
    _run_ctx: dict[str, Any] = {
        "config":          config,
        "budget":          int(cfg.budget),
        "resize_size":     cfg.resize_size,
        "resize_resample": cfg.resize_resample,
        "alpha":           float(cfg.alpha),
    }
    _init_worker(_run_ctx)

    def _run_one_live(job: dict[str, Any]) -> dict[str, Any]:
        from evohash.evaluator import evaluate_attack as _eval
        _attack = job["_live_attack"]
        _sample_hash = _peval_get_hash(str(job["hash_id"]))
        row = _eval(
            attack=_attack,
            hash_fn=_sample_hash,
            x_source=job["x_source"],
            x_target=job["x_target"],
            params=dict(job["params"]),
            budget=int(_run_ctx["budget"]),
            seed=int(job["seed"]),
            pair_id=job.get("pair_id"),
            resize_size=_run_ctx["resize_size"],
            resize_resample=str(_run_ctx["resize_resample"]),
        ).to_dict()
        row.pop("history", None)
        row.pop("params", None)
        row.pop("extra", None)
        row.update({
            "job_index":     int(job["job_index"]),
            "pair_index":    int(job["pair_index"]),
            "status":        "ok",
            "error":         None,
            "failure_gap":   max(0.0, float(row["final_hash_l1"]) - float(row["threshold"])),
            "target_metric": float(row["pixel_l2_raw"]) + float(_run_ctx["alpha"]) * float(row["queries"]),
        })
        return row

    job_list = list(jobs)
    if backend == "sequential" or not workers or int(workers) <= 1:
        rows = [_run_one_live(j) for j in job_list]
    else:
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
        rows = []
        with _TPE(max_workers=int(workers)) as ex:
            futs = [ex.submit(_run_one_live, j) for j in job_list]
            for f in _ac(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: int(r.get("pair_index", 0)))

    return {"rows": rows, "attack_id": str(attack_id), "eval_backend": backend}


def validate_payload(payload: Any, context: dict[str, Any]) -> dict[str, float]:
    """Validate/summarize payload returned by entrypoint(context)."""
    metrics, _feedback = _summarize_payload(payload, context)
    return metrics


def validate_payload_with_feedback(
    payload: Any,
    context: dict[str, Any],
) -> tuple[dict[str, float], str]:
    """Validate payload and return metrics plus a mutation-context artifact."""
    return _summarize_payload(payload, context)


def _summarize_payload(payload: Any, context: dict[str, Any]) -> tuple[dict[str, float], str]:
    cfg = _eval_config(context)
    if not isinstance(payload, dict) or "rows" not in payload:
        metrics = _bad_metrics()
        feedback = format_feedback_diagnostics([], metrics=metrics, config=cfg)
        feedback += "\nInvalid payload: entrypoint(context) must return a dict containing a 'rows' list."
        return metrics, feedback
    rows = payload.get("rows")
    if not isinstance(rows, list):
        metrics = _bad_metrics()
        feedback = format_feedback_diagnostics([], metrics=metrics, config=cfg)
        feedback += "\nInvalid payload: payload['rows'] must be a list of evaluator row dicts."
        return metrics, feedback
    metrics = summarize_row_dicts(rows, config=cfg)
    if metrics.get("is_valid", 0.0) < 1.0:
        metrics["score"] = 1e9
        metrics["optimizer_score"] = 1e9
    errors = [str(row.get("error")) for row in rows if row.get("status") != "ok" and row.get("error")]
    feedback = format_feedback_diagnostics(rows, metrics=metrics, config=cfg, errors=errors)
    return metrics, feedback


def _evaluate_file_backed(
    *,
    code_path: Path,
    context: dict[str, Any],
    cfg: EvaluationConfig,
    config: dict[str, Any],
    runtime: Any,
    attack_id: str,
) -> list[dict[str, Any]]:
    """Evaluate using attack code from a file — works with any backend."""
    pairs = list(context.get("pairs") or [])
    jobs = [
        {
            "job_index":    int(index),
            "hash_id":      cfg.hash_id,
            "attack_id":    str(attack_id),
            "program_path": str(code_path),
            "pair_index":   int(index),
            "pair_id":      getattr(sample, "pair_id", str(index)),
            "x_source":     sample.x,
            "x_target":     sample.y,
            "params":       {},
            "seed":         int(cfg.seed) + int(index),
        }
        for index, sample in enumerate(pairs)
    ]
    return run_eval_jobs(
        jobs,
        config=config,
        budget=int(cfg.budget),
        resize_size=cfg.resize_size,
        resize_resample=cfg.resize_resample,
        alpha=float(cfg.alpha),
        backend=runtime.backend,
        max_workers=runtime.workers,
        process_start_method=runtime.process_start_method,
        show_progress=False,
    )


def _eval_config(context: dict[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        hash_id=str(context.get("hash_id", "phash")),
        dataset_zip=str(context.get("dataset_zip", "dataset_v1_public.zip")),
        split=str(context.get("split", "public")),
        max_pairs=int(context.get("max_pairs") or len(context.get("pairs") or [])),
        budget=int(context.get("budget", 10_000)),
        seed=int(context.get("seed", 0)),
        resize_size=context.get("resize_size"),
        resize_resample=str(context.get("resize_resample", "bilinear")),
        parallel_backend=str(context.get("parallel_backend", "auto")),
        workers=context.get("parallel_workers"),
        worker_cap=int(context.get("parallel_worker_cap", 30)),
        alpha=float(context.get("alpha", 1.0 / 50.0)),
        asr_floor=float(context.get("asr_floor", 0.05)),
        failure_gap_penalty=float(context.get("failure_gap_penalty", 0.0)),
        attack_config_path=str(context.get("attack_config_path", "results/attack_configs_optimized.json")),
    )


def _load_attack_config(path: str | Path, hash_id: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if cfg_path.exists():
        return load_attack_config(cfg_path)
    return {"configs": {hash_id: {}}, "thresholds": {}}


def _bad_metrics() -> dict[str, float]:
    return {
        "score": 1e9,
        "optimizer_score": 1e9,
        "target_metric": 1e9,
        "is_valid": 0.0,
        "asr": 0.0,
        "mean_l2_raw": 1e9,
        "mean_queries": 0.0,
        "mean_final_hash_l1": 1e9,
        "mean_gap_to_threshold": 1e9,
    }


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in {None, ""}:
        return float(default)
    return float(value)
