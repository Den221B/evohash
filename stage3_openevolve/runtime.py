"""OpenEvolve evaluator entry point.

OpenEvolve calls ``evaluate(program_path)`` for every candidate produced by
the LLM.  We delegate to ``validate_program`` from ``stage3_common`` — the
*same* pipeline GigaEvolve uses, so the two frameworks score identically.

The evaluation context (hash_id, dataset, budget, ...) is read from
environment variables to keep this entry point compatible with OpenEvolve's
worker process model (each evaluator may be spawned in a subprocess).  The
notebook-first launcher (``notebook_api.run_openevolve``) sets these for the
duration of the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from evohash.stage3_common.config import EvaluationConfig
from evohash.stage3_common.validate import validate_program


# Environment variable names — must match what ``notebook_api.run_openevolve``
# (and the CLI fallback) export before invoking openevolve-run.
ENV_HASH_ID = "EVOHASH_OE_HASH_ID"
ENV_DATASET = "EVOHASH_OE_DATASET_ZIP"
ENV_SPLIT = "EVOHASH_OE_SPLIT"
ENV_MAX_PAIRS = "EVOHASH_OE_MAX_PAIRS"
ENV_BUDGET = "EVOHASH_OE_BUDGET"
ENV_SEED = "EVOHASH_OE_SEED"
ENV_RESIZE_SIZE = "EVOHASH_OE_RESIZE_SIZE"
ENV_RESIZE_RESAMPLE = "EVOHASH_OE_RESIZE_RESAMPLE"
ENV_ALPHA = "EVOHASH_OE_ALPHA"
ENV_ASR_FLOOR = "EVOHASH_OE_ASR_FLOOR"
ENV_FAILURE_GAP_PENALTY = "EVOHASH_OE_FAILURE_GAP_PENALTY"
ENV_PARALLEL_BACKEND = "EVOHASH_OE_PARALLEL_BACKEND"
ENV_PARALLEL_WORKERS = "EVOHASH_OE_PARALLEL_WORKERS"
ENV_WORKER_CAP = "EVOHASH_OE_WORKER_CAP"
ENV_ATTACK_CONFIG = "EVOHASH_OE_ATTACK_CONFIG"
ENV_TRACE_DIR = "EVOHASH_OE_TRACE_DIR"


def evaluate(program_path: str | Path) -> Any:
    """OpenEvolve evaluator entry point.

    Returns metrics plus optional artifacts, where ``score`` is the primary
    objective (lower is better — same as stage2 + stage3_gigaevolve).
    """
    cfg = _eval_config_from_env()
    summary = validate_program(program_path, cfg)
    # OpenEvolve happily accepts arbitrary float metrics; we expose everything
    # validate_program computes so the user can inspect them in the archive.
    metrics: dict[str, float] = {}
    for key, value in summary.metrics.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    # OpenEvolve ranks programs by ``combined_score`` (higher = better) and, if
    # it is absent, falls back to averaging ALL non-feature metrics (which mixes
    # score/asr/l2/queries into a meaningless number).  Our ``score`` is
    # lower-is-better, so expose ``combined_score = -score`` to align OpenEvolve's
    # maximization with our minimization. All raw metrics stay exposed for
    # inspection and as MAP-Elites feature dimensions.
    if "score" in metrics:
        metrics["combined_score"] = -float(metrics["score"])
    _write_eval_trace(program_path, metrics, summary.diagnostics)
    return _with_feedback_artifact(metrics, summary.diagnostics)


def _with_feedback_artifact(metrics: dict[str, float], diagnostics: str) -> Any:
    if not diagnostics.strip():
        return metrics
    try:
        from openevolve.evaluation_result import EvaluationResult  # type: ignore
    except Exception:
        return metrics
    return EvaluationResult(
        metrics=metrics,
        artifacts={"evohash_feedback.txt": diagnostics},
    )


def _write_eval_trace(
    program_path: str | Path,
    metrics: dict[str, float],
    diagnostics: str,
) -> None:
    """Best-effort per-evaluation trace for exact post-run champion export.

    Checkpoints are bounded snapshots, so a record-setting child can be evicted
    before the next checkpoint. Writing one independent JSON file per evaluated
    program avoids cross-process locking while preserving enough code+metric
    data to reconstruct strict record improvements later.
    """
    trace_root = os.environ.get(ENV_TRACE_DIR)
    if not trace_root:
        return
    try:
        root = Path(trace_root)
        root.mkdir(parents=True, exist_ok=True)
        path = Path(program_path)
        code = path.read_text(encoding="utf-8", errors="replace")
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        ts = time.time()
        rec = {
            "id": f"eval_{int(ts * 1_000_000)}_{os.getpid()}_{code_hash[:12]}",
            "program_path": str(path),
            "timestamp": ts,
            "pid": os.getpid(),
            "code_hash": code_hash,
            "code": code,
            "metrics": dict(metrics),
            "diagnostics": diagnostics,
        }
        name = f"{int(ts * 1_000_000):020d}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
        tmp = root / f".{name}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(root / name)
    except Exception:
        # Tracing is diagnostic/export support; never fail candidate evaluation.
        return


def _eval_config_from_env() -> EvaluationConfig:
    return EvaluationConfig(
        hash_id=os.environ.get(ENV_HASH_ID, "phash"),
        dataset_zip=os.environ.get(ENV_DATASET, "dataset_v1_public.zip"),
        split=os.environ.get(ENV_SPLIT, "public"),
        max_pairs=_env_int(ENV_MAX_PAIRS, default=5),
        budget=_env_int(ENV_BUDGET, default=10_000) or 10_000,
        seed=_env_int(ENV_SEED, default=0) or 0,
        resize_size=_env_int(ENV_RESIZE_SIZE, default=256),
        resize_resample=os.environ.get(ENV_RESIZE_RESAMPLE, "bilinear"),
        parallel_backend=os.environ.get(ENV_PARALLEL_BACKEND, "auto"),  # type: ignore[arg-type]
        workers=_env_int(ENV_PARALLEL_WORKERS, default=None),
        worker_cap=_env_int(ENV_WORKER_CAP, default=30) or 30,
        alpha=_env_float(ENV_ALPHA, default=1.0 / 50.0),
        asr_floor=_env_float(ENV_ASR_FLOOR, default=0.05),
        failure_gap_penalty=_env_float(ENV_FAILURE_GAP_PENALTY, default=0.0),
        attack_config_path=os.environ.get(
            ENV_ATTACK_CONFIG, "results/attack_configs_optimized.json"
        ),
        show_progress=False,
    )


def export_eval_env(cfg: EvaluationConfig, env: dict[str, str] | None = None) -> dict[str, str]:
    """Translate an ``EvaluationConfig`` into env vars for openevolve workers.

    Returns a *new* dict — caller decides whether to ``os.environ.update(...)``
    it (notebook in-process) or pass it to ``subprocess.Popen(env=...)``.
    """
    env = dict(env) if env is not None else {}
    env[ENV_HASH_ID] = str(cfg.hash_id)
    env[ENV_DATASET] = str(cfg.dataset_zip)
    env[ENV_SPLIT] = str(cfg.split)
    env[ENV_BUDGET] = str(int(cfg.budget))
    env[ENV_SEED] = str(int(cfg.seed))
    env[ENV_RESIZE_RESAMPLE] = str(cfg.resize_resample)
    env[ENV_PARALLEL_BACKEND] = str(cfg.parallel_backend)
    env[ENV_WORKER_CAP] = str(int(cfg.worker_cap))
    env[ENV_ALPHA] = repr(float(cfg.alpha))
    env[ENV_ASR_FLOOR] = repr(float(cfg.asr_floor))
    env[ENV_FAILURE_GAP_PENALTY] = repr(float(cfg.failure_gap_penalty))
    env[ENV_ATTACK_CONFIG] = str(cfg.attack_config_path)
    if cfg.max_pairs is not None:
        env[ENV_MAX_PAIRS] = str(int(cfg.max_pairs))
    if cfg.resize_size is not None:
        env[ENV_RESIZE_SIZE] = str(int(cfg.resize_size))
    if cfg.workers is not None:
        env[ENV_PARALLEL_WORKERS] = str(int(cfg.workers))
    return env


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
