"""Optuna-based mixed hyperparameter optimizer for EvoHash attacks.

The optimizer is intentionally thin: attack implementations stay unchanged,
while this module owns search spaces, repeated evaluation on image pairs,
budget-driven runtime caps, and notebook-friendly result tables.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from evohash.attacks import build_attack_registry
from evohash.dataset import Dataset, PairSample
from evohash.evaluator import EvalRow, evaluate_attack
from evohash.hashes.neuralhash import NeuralHashWrapper
from evohash.hashes.pdq import PDQWrapper
from evohash.hashes.phash import PHashWrapper
from evohash.hashes.photodna import PhotoDNAWrapper


META_PARAM_KEYS = {
    "profile",
    "source_units_note",
    "max_iters",
    "n_iter",
    "log_every",
}

RUNTIME_PARAM_KEYS = {
    "max_iters",
    "n_iter",
    "log_every",
}

FIXED_PARAMS_BY_HASH_ATTACK: dict[tuple[str, str], dict[str, Any]] = {
    ("pdq", "atkscopes"): {"scale": "global"},
    ("phash", "atkscopes"): {"scale": "global"},
    ("neuralhash", "atkscopes"): {"scale": "global"},
    ("photodna", "atkscopes"): {"scale": "mid"},
}

CONTINUOUS_NUMERIC_PARAMS_BY_ATTACK: dict[str, set[str]] = {
    "nes": {"sigma", "lr"},
    "prokos": {"sigma", "lr"},
    "nes_attack_v0": {"sigma", "lr"},
    "simba": {"epsilon"},
    "zo_signsgd": {"mu", "lr"},
    "atkscopes": {"a", "lr"},
}

CATEGORICAL_PARAMS_BY_ATTACK: dict[str, set[str]] = {
    "nes": {"n_samples"},
    "prokos": {"n_samples", "momentum", "grayscale_noise", "antithetic"},
    "nes_attack_v0": {"n_samples", "momentum", "grayscale_noise"},
    "simba": {"freq_dims", "stride"},
    "zo_signsgd": {"estimator", "direction_dist", "q", "eval_updated_point"},
    "atkscopes": {"max_freq", "patch_size", "beta1", "beta2"},
}


@dataclass(frozen=True)
class ParallelRuntime:
    backend: str
    workers: int | None = None
    process_start_method: str | None = None


@dataclass(frozen=True)
class OptimizerSettings:
    """Runtime and objective settings shared across trials."""

    hash_id: str = "pdq"
    attack_id: str = "atkscopes"
    resize_size: int | None = 256
    resize_resample: str = "bilinear"
    budget: int = 10_000
    seed: int = 0
    alpha: float = 1.0 / 50.0
    asr_floor: float = 0.05
    failure_gap_penalty: float = 0.0
    budget_controls_iters: bool = True
    category_choices: dict[Any, Any] = field(default_factory=dict)
    numeric_params: dict[Any, Any] = field(default_factory=dict)
    fixed_params: dict[Any, Any] = field(default_factory=dict)
    parallel_backend: str = "auto"
    parallel_workers: int | None = None
    parallel_worker_cap: int = 30
    parallel_photodna_worker_cap: int | None = None
    parallel_min_pairs: int = 2
    parallel_process_start_method: str | None = None
    parallel_show_progress: bool = False


@dataclass
class CandidateSummary:
    """Aggregated result for one candidate config."""

    score: float
    target_metric: float
    asr: float
    n_pairs: int
    successes: int
    mean_l2: float
    median_l2: float
    mean_queries: float
    median_queries: float
    mean_time_sec: float
    mean_initial_hash_l1: float
    mean_final_hash_l1: float
    mean_best_hash_l1: float
    mean_gap_to_threshold: float
    params: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AttackTuningResult:
    """Baseline and tuned result for one (hash, attack) experiment."""

    hash_id: str
    attack_id: str
    baseline: CandidateSummary
    tuned: CandidateSummary
    study: Any | None = None


@dataclass
class PhotoDNAWorkerCalibration:
    """Result of a one-time PhotoDNA process worker calibration."""

    recommended_workers: int
    max_stable_workers: int
    rows: list[dict[str, Any]]
    sequential_wall_sec: float
    n_pairs: int
    budget: int

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


def load_attack_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attack_params_from_config(config: dict[str, Any], hash_id: str, attack_id: str) -> dict[str, Any]:
    params = dict(config["configs"][hash_id][attack_id])
    return clean_public_params(params)


def clean_public_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop metadata and runtime-only knobs from a persisted attack config."""
    clean = dict(params)
    for key in META_PARAM_KEYS | RUNTIME_PARAM_KEYS:
        clean.pop(key, None)
    return clean


def load_pairs(zip_path: str | Path, *, split: str = "public", n_pairs: int | None = 3) -> list[PairSample]:
    return list(Dataset(str(zip_path), split=split, max_pairs=n_pairs))


def build_hash_fn(hash_id: str, config: dict[str, Any]):
    """Build hash wrapper for optimizer targets."""
    thresholds = config.get("thresholds", {})

    if hash_id == "pdq":
        return PDQWrapper(threshold_p=float(thresholds.get("pdq", 92.0)))
    if hash_id == "phash":
        return PHashWrapper(threshold_p=float(thresholds.get("phash", 12.0)))
    if hash_id == "neuralhash":
        return NeuralHashWrapper(threshold_p=float(thresholds.get("neuralhash", 17.0)))
    if hash_id == "photodna":
        dll_path = Path(__file__).resolve().parent / "hashes" / "dll" / "PhotoDNAx64.dll"
        kwargs: dict[str, Any] = {"threshold_p": float(thresholds.get("photodna", 3855.0))}
        if dll_path.exists():
            kwargs["dll_path"] = str(dll_path)
        return PhotoDNAWrapper(**kwargs)
    raise ValueError(f"Unknown hash_id for optimizer hash factory: {hash_id!r}")


def resolve_parallel_runtime(settings: OptimizerSettings, n_pairs: int) -> ParallelRuntime:
    """Resolve auto/explicit optimizer parallel settings for one candidate."""
    if int(n_pairs) < int(settings.parallel_min_pairs):
        return ParallelRuntime("sequential")

    requested_backend = str(settings.parallel_backend or "sequential").strip().lower()
    if requested_backend == "auto":
        backend = _auto_parallel_backend(settings.hash_id)
        start_method = None
    elif requested_backend in {"sequential", "thread", "process"}:
        backend = requested_backend
        start_method = settings.parallel_process_start_method
    else:
        raise ValueError(
            "parallel_backend must be one of: auto, sequential, thread, process; "
            f"got {settings.parallel_backend!r}"
        )

    if backend == "sequential":
        return ParallelRuntime("sequential")

    if backend == "process":
        start_method = start_method or settings.parallel_process_start_method
        if start_method is None and os.name == "posix":
            start_method = "fork"
    else:
        start_method = None

    workers = _resolve_parallel_workers(settings, n_pairs=int(n_pairs), backend=backend)
    if workers <= 1:
        return ParallelRuntime("sequential")
    return ParallelRuntime(backend=backend, workers=workers, process_start_method=start_method)


def _auto_parallel_backend(hash_id: str) -> str:
    if hash_id in {"pdq", "phash"}:
        return "process" if os.name == "posix" else "thread"
    if hash_id == "photodna":
        return "process"
    # NeuralHash can have GPU/provider-specific behavior; keep it explicit for now.
    return "sequential"


def _resolve_parallel_workers(settings: OptimizerSettings, *, n_pairs: int, backend: str) -> int:
    cpu_count = _available_cpu_count()
    pair_cap = max(1, int(n_pairs))
    cpu_cap = max(1, int(cpu_count))

    if settings.parallel_workers is not None:
        workers = int(settings.parallel_workers)
    else:
        workers = min(pair_cap, cpu_cap, max(1, int(settings.parallel_worker_cap)))

    workers = min(max(1, workers), pair_cap, cpu_cap)
    if settings.hash_id == "photodna" and backend == "process":
        workers = min(workers, _photodna_process_worker_cap(settings, n_pairs=pair_cap, cpu_count=cpu_cap))
    return max(1, int(workers))


def _photodna_process_worker_cap(
    settings: OptimizerSettings,
    *,
    n_pairs: int,
    cpu_count: int,
) -> int:
    env_cap = _env_int("EVOHASH_PHOTODNA_MAX_WORKERS")
    configured_cap = settings.parallel_photodna_worker_cap
    if env_cap is not None:
        base_cap = env_cap
    elif configured_cap is not None:
        base_cap = int(configured_cap)
    else:
        # Linux PhotoDNA uses Wine and can be memory-heavy; Windows uses the
        # native DLL, so default to all but one CPU and let benchmarking choose
        # a lower cap when a specific machine dislikes high process counts.
        base_cap = 6 if os.name == "posix" else max(1, int(cpu_count) - 1)

    if platform.system() == "Windows":
        return max(1, min(int(n_pairs), int(cpu_count), int(base_cap)))

    worker_mb = _env_int("EVOHASH_PHOTODNA_WORKER_MB")
    if worker_mb is None:
        worker_mb = 1536

    mem_cap = base_cap
    available = _available_memory_bytes()
    if available is not None and worker_mb > 0:
        usable = int(available * 0.65)
        mem_cap = max(1, usable // (int(worker_mb) * 1024 * 1024))

    return max(1, min(int(n_pairs), int(cpu_count), int(base_cap), int(mem_cap)))


def _available_cpu_count() -> int:
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            return max(1, len(affinity(0)))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


def _available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if os.name == "posix":
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except Exception:
            return None

    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            return None

    return None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def suggest_params(
    trial: Any,
    attack_id: str,
    base_params: dict[str, Any],
    *,
    hash_id: str | None = None,
    settings: OptimizerSettings | None = None,
) -> dict[str, Any]:
    """Return trial params around a known working config.

    Numeric search is local around the current config; categorical search is
    explicit and keeps hash-specific fixed knobs, such as ATKScopes scale,
    stable unless the caller changes the base config.
    """
    params = clean_public_params(base_params)
    if hash_id is not None:
        params.update(FIXED_PARAMS_BY_HASH_ATTACK.get((hash_id, attack_id), {}))
    if settings is not None:
        params.update(_fixed_overrides(settings, attack_id))

    if attack_id in {"nes", "prokos", "nes_attack_v0"}:
        if "n_samples" in params:
            base = int(params["n_samples"])
            params["n_samples"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "n_samples",
                base,
                _nearby_int_choices(base, multipliers=(0.5, 1.0, 1.5, 2.0, 3.0), minimum=4),
            )
        if "sigma" in params and _should_tune(settings, attack_id, "sigma"):
            params["sigma"] = _suggest_float_multiplier(trial, "sigma", float(params["sigma"]), 0.25, 3.0)
        if "lr" in params and _should_tune(settings, attack_id, "lr"):
            params["lr"] = _suggest_float_multiplier(trial, "lr", float(params["lr"]), 0.25, 4.0)
        if attack_id in {"prokos", "nes_attack_v0"} and "momentum" in params:
            params["momentum"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "momentum",
                float(params["momentum"]),
                _choice_union(float(params["momentum"]), [0.0, 0.25, 0.5, 0.75, 0.9]),
            )
        if attack_id in {"prokos", "nes_attack_v0"} and "grayscale_noise" in params:
            params["grayscale_noise"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "grayscale_noise",
                bool(params["grayscale_noise"]),
                _choice_union(bool(params["grayscale_noise"]), [True, False]),
            )
        if attack_id == "prokos" and "antithetic" in params:
            params["antithetic"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "antithetic",
                bool(params["antithetic"]),
                _choice_union(bool(params["antithetic"]), [True, False]),
            )

    elif attack_id == "simba":
        if "epsilon" in params and _should_tune(settings, attack_id, "epsilon"):
            params["epsilon"] = _suggest_float_multiplier(trial, "epsilon", float(params["epsilon"]), 0.25, 4.0)
        if "freq_dims" in params:
            base = int(params["freq_dims"])
            params["freq_dims"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "freq_dims",
                base,
                _choice_union(base, [1, 2, 4, 8, 12, 16, 20, 24, 32]),
            )
        if "stride" in params:
            params["stride"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "stride",
                int(params["stride"]),
                _choice_union(int(params["stride"]), [1, 2, 4, 8]),
            )

    elif attack_id == "zo_signsgd":
        if "estimator" in params:
            params["estimator"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "estimator",
                str(params["estimator"]),
                _choice_union(str(params["estimator"]), ["forward", "central", "majority"]),
            )
        if "direction_dist" in params:
            params["direction_dist"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "direction_dist",
                str(params["direction_dist"]),
                _choice_union(str(params["direction_dist"]), ["gaussian", "sphere"]),
            )
        if "q" in params:
            base = int(params["q"])
            params["q"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "q",
                base,
                _choice_union(base, [4, 8, 16, 32, 64, 128]),
            )
        if "mu" in params and _should_tune(settings, attack_id, "mu"):
            params["mu"] = _suggest_float_multiplier(trial, "mu", float(params["mu"]), 0.5, 2.5)
        if "lr" in params and _should_tune(settings, attack_id, "lr"):
            params["lr"] = _suggest_float_multiplier(trial, "lr", float(params["lr"]), 0.4, 2.5)
        if "eval_updated_point" in params:
            params["eval_updated_point"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "eval_updated_point",
                bool(params["eval_updated_point"]),
                _choice_union(bool(params["eval_updated_point"]), [True, False]),
            )

    elif attack_id == "atkscopes":
        if "a" in params and _should_tune(settings, attack_id, "a"):
            params["a"] = _suggest_float_multiplier(trial, "a", float(params["a"]), 0.4, 2.5)
        if "lr" in params and _should_tune(settings, attack_id, "lr"):
            params["lr"] = _suggest_float_multiplier(trial, "lr", float(params["lr"]), 0.4, 2.5)
        if "max_freq" in params:
            base = int(params["max_freq"])
            params["max_freq"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "max_freq",
                base,
                _choice_union(base, [4, 8, 12, 16, 24, 32]),
            )
        if params.get("scale") == "mid" and "patch_size" in params and params["patch_size"] is not None:
            base = int(params["patch_size"])
            params["patch_size"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "patch_size",
                base,
                _choice_union(base, [32, 64, 96, 128, 192, 256]),
            )
        if "beta1" in params:
            params["beta1"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "beta1",
                float(params["beta1"]),
                _choice_union(float(params["beta1"]), [0.5, 0.7, 0.9]),
            )
        if "beta2" in params:
            params["beta2"] = _suggest_categorical_param(
                trial,
                settings,
                attack_id,
                "beta2",
                float(params["beta2"]),
                _choice_union(float(params["beta2"]), [0.9, 0.99, 0.999]),
            )

    else:
        raise ValueError(f"Unknown attack_id for search space: {attack_id!r}")

    return params


def evaluate_candidate(
    *,
    pairs: Iterable[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
    params: dict[str, Any],
    job_runner: Any | None = None,
) -> CandidateSummary:
    pair_list = list(pairs)
    parallel_runtime = resolve_parallel_runtime(settings, len(pair_list))
    if parallel_runtime.backend != "sequential":
        return evaluate_candidate_parallel(
            pairs=pair_list,
            config=config,
            settings=settings,
            params=params,
            job_runner=job_runner,
            parallel_runtime=parallel_runtime,
        )

    hash_fn = build_hash_fn(settings.hash_id, config)
    attack = build_attack_registry().get(settings.attack_id)
    rows: list[EvalRow] = []
    public_params = clean_public_params(params)
    runtime_params = _runtime_params_for_budget(public_params, settings)

    for index, sample in enumerate(pair_list):
        rows.append(
            evaluate_attack(
                attack=attack,
                hash_fn=hash_fn,
                x_source=sample.x,
                x_target=sample.y,
                params=runtime_params,
                budget=settings.budget,
                seed=settings.seed + index,
                pair_id=getattr(sample, "pair_id", str(index)),
                resize_size=settings.resize_size,
                resize_resample=settings.resize_resample,
            )
        )

    return summarize_rows(rows, params=public_params, settings=settings)


def evaluate_candidate_parallel(
    *,
    pairs: Iterable[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
    params: dict[str, Any],
    job_runner: Any | None = None,
    parallel_runtime: ParallelRuntime | None = None,
) -> CandidateSummary:
    """Evaluate one optimizer candidate over image pairs in parallel.

    The candidate semantics match ``evaluate_candidate``: one hash, one attack,
    one params dict, and deterministic ``settings.seed + pair_index`` seeds.
    """
    from evohash.parallel_eval import run_eval_jobs

    pair_list = list(pairs)
    runtime = parallel_runtime or resolve_parallel_runtime(settings, len(pair_list))
    if runtime.backend == "sequential":
        return evaluate_candidate(
            pairs=pair_list,
            config=config,
            settings=OptimizerSettings(
                hash_id=settings.hash_id,
                attack_id=settings.attack_id,
                resize_size=settings.resize_size,
                resize_resample=settings.resize_resample,
                budget=settings.budget,
                seed=settings.seed,
                alpha=settings.alpha,
                asr_floor=settings.asr_floor,
                failure_gap_penalty=settings.failure_gap_penalty,
                budget_controls_iters=settings.budget_controls_iters,
                category_choices=dict(settings.category_choices),
                numeric_params=dict(settings.numeric_params),
                fixed_params=dict(settings.fixed_params),
                parallel_backend="sequential",
            ),
            params=params,
        )

    public_params = clean_public_params(params)
    runtime_params = _runtime_params_for_budget(public_params, settings)

    jobs = []
    for index, sample in enumerate(pair_list):
        jobs.append(
            {
                "job_index": int(index),
                "hash_id": settings.hash_id,
                "attack_id": settings.attack_id,
                "pair_index": int(index),
                "pair_id": getattr(sample, "pair_id", str(index)),
                "x_source": sample.x,
                "x_target": sample.y,
                "params": dict(runtime_params),
                "seed": int(settings.seed) + int(index),
            }
        )

    if job_runner is not None:
        row_dicts = job_runner.run(jobs)
    else:
        row_dicts = run_eval_jobs(
            jobs,
            config=config,
            budget=settings.budget,
            resize_size=settings.resize_size,
            resize_resample=settings.resize_resample,
            alpha=settings.alpha,
            backend=runtime.backend,
            max_workers=runtime.workers,
            process_start_method=runtime.process_start_method,
            show_progress=settings.parallel_show_progress,
        )

    errors = [row for row in row_dicts if row.get("status") != "ok"]
    if errors:
        first = errors[0]
        raise RuntimeError(
            "Parallel candidate evaluation failed for "
            f"{first.get('hash_id')}/{first.get('attack_id')} "
            f"pair={first.get('pair_id')}: {first.get('error')}"
        )

    return summarize_row_dicts(row_dicts, params=public_params, settings=settings)


def summarize_rows(rows: Iterable[EvalRow], *, params: dict[str, Any], settings: OptimizerSettings) -> CandidateSummary:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("Cannot summarize empty candidate rows")

    success = np.array([bool(row.success) for row in rows_list], dtype=np.float32)
    l2 = np.array([float(row.pixel_l2_raw) for row in rows_list], dtype=np.float32)
    queries = np.array([float(row.queries) for row in rows_list], dtype=np.float32)
    initial = np.array([float(row.initial_hash_l1) for row in rows_list], dtype=np.float32)
    final = np.array([float(row.final_hash_l1) for row in rows_list], dtype=np.float32)
    best = np.array([float(row.best_hash_l1) for row in rows_list], dtype=np.float32)
    threshold = np.array([float(row.threshold) for row in rows_list], dtype=np.float32)
    time_sec = np.array([float(row.time_sec) for row in rows_list], dtype=np.float32)
    gap = np.maximum(final - threshold, 0.0)

    asr = float(success.mean())
    mean_l2 = float(l2.mean())
    mean_queries = float(queries.mean())
    mean_gap = float(gap.mean())
    target_metric = float(mean_l2 + settings.alpha * mean_queries)
    score = (target_metric / max(asr, settings.asr_floor)) + settings.failure_gap_penalty * mean_gap

    return CandidateSummary(
        score=float(score),
        target_metric=target_metric,
        asr=asr,
        n_pairs=len(rows_list),
        successes=int(success.sum()),
        mean_l2=mean_l2,
        median_l2=float(np.median(l2)),
        mean_queries=mean_queries,
        median_queries=float(np.median(queries)),
        mean_time_sec=float(time_sec.mean()),
        mean_initial_hash_l1=float(initial.mean()),
        mean_final_hash_l1=float(final.mean()),
        mean_best_hash_l1=float(best.mean()),
        mean_gap_to_threshold=mean_gap,
        params=clean_public_params(params),
        rows=[_clean_row_dict(row) for row in rows_list],
    )


def summarize_row_dicts(
    rows: Iterable[dict[str, Any]],
    *,
    params: dict[str, Any],
    settings: OptimizerSettings,
) -> CandidateSummary:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("Cannot summarize empty candidate rows")

    success = np.array([bool(row["success"]) for row in rows_list], dtype=np.float32)
    l2 = np.array([float(row["pixel_l2_raw"]) for row in rows_list], dtype=np.float32)
    queries = np.array([float(row["queries"]) for row in rows_list], dtype=np.float32)
    initial = np.array([float(row["initial_hash_l1"]) for row in rows_list], dtype=np.float32)
    final = np.array([float(row["final_hash_l1"]) for row in rows_list], dtype=np.float32)
    best = np.array([float(row["best_hash_l1"]) for row in rows_list], dtype=np.float32)
    threshold = np.array([float(row["threshold"]) for row in rows_list], dtype=np.float32)
    time_sec = np.array([float(row["time_sec"]) for row in rows_list], dtype=np.float32)
    gap = np.maximum(final - threshold, 0.0)

    asr = float(success.mean())
    mean_l2 = float(l2.mean())
    mean_queries = float(queries.mean())
    mean_gap = float(gap.mean())
    target_metric = float(mean_l2 + settings.alpha * mean_queries)
    score = (target_metric / max(asr, settings.asr_floor)) + settings.failure_gap_penalty * mean_gap

    clean_rows = []
    for row in rows_list:
        clean = dict(row)
        clean.pop("traceback", None)
        clean_rows.append(clean)

    return CandidateSummary(
        score=float(score),
        target_metric=target_metric,
        asr=asr,
        n_pairs=len(rows_list),
        successes=int(success.sum()),
        mean_l2=mean_l2,
        median_l2=float(np.median(l2)),
        mean_queries=mean_queries,
        median_queries=float(np.median(queries)),
        mean_time_sec=float(time_sec.mean()),
        mean_initial_hash_l1=float(initial.mean()),
        mean_final_hash_l1=float(final.mean()),
        mean_best_hash_l1=float(best.mean()),
        mean_gap_to_threshold=mean_gap,
        params=clean_public_params(params),
        rows=clean_rows,
    )


def objective_factory(
    *,
    pairs: list[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
    base_params: dict[str, Any],
):
    parallel_runtime = resolve_parallel_runtime(settings, len(pairs))
    job_runner = (
        _make_parallel_job_runner(config=config, settings=settings, parallel_runtime=parallel_runtime)
        if parallel_runtime.backend != "sequential"
        else None
    )

    def objective(trial: Any) -> float:
        params = suggest_params(
            trial,
            settings.attack_id,
            base_params,
            hash_id=settings.hash_id,
            settings=settings,
        )
        summary = evaluate_candidate(
            pairs=pairs,
            config=config,
            settings=settings,
            params=params,
            job_runner=job_runner,
        )
        trial.set_user_attr("summary", _summary_attrs(summary))
        trial.set_user_attr("params_full", params)
        return summary.score

    if job_runner is not None:
        objective.close = job_runner.close  # type: ignore[attr-defined]

    return objective


def _make_parallel_job_runner(
    *,
    config: dict[str, Any],
    settings: OptimizerSettings,
    parallel_runtime: ParallelRuntime,
):
    from evohash.parallel_eval import EvalJobRunner

    context = {
        "config": config,
        "budget": int(settings.budget),
        "resize_size": settings.resize_size,
        "resize_resample": str(settings.resize_resample),
        "alpha": float(settings.alpha),
    }
    return EvalJobRunner(
        context=context,
        backend=parallel_runtime.backend,
        max_workers=parallel_runtime.workers,
        process_start_method=parallel_runtime.process_start_method,
        show_progress=settings.parallel_show_progress,
    )


def run_optuna(
    *,
    pairs: list[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
    base_params: Optional[dict[str, Any]] = None,
    n_trials: int = 20,
    study_name: str | None = None,
    storage: str | None = None,
    enqueue_base: bool = True,
):
    import optuna

    base = base_params or attack_params_from_config(config, settings.hash_id, settings.attack_id)
    sampler = optuna.samplers.TPESampler(seed=settings.seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, n_trials // 5))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )
    if enqueue_base and not study.trials:
        study.enqueue_trial(_search_space_projection(settings.attack_id, base))
    objective = objective_factory(pairs=pairs, config=config, settings=settings, base_params=base)
    try:
        study.optimize(objective, n_trials=n_trials)
    finally:
        close = getattr(objective, "close", None)
        if callable(close):
            close()
    return study


def run_attack_tuning(
    *,
    pairs: list[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
    base_params: Optional[dict[str, Any]] = None,
    n_trials: int = 20,
    study_name: str | None = None,
    storage: str | None = None,
    rerun_best: bool = False,
) -> AttackTuningResult:
    """Tune one attack and return baseline-vs-best summaries."""
    base = base_params or attack_params_from_config(config, settings.hash_id, settings.attack_id)
    study = run_optuna(
        pairs=pairs,
        config=config,
        settings=settings,
        base_params=base,
        n_trials=n_trials,
        study_name=study_name,
        storage=storage,
        enqueue_base=_can_enqueue_base(settings, base),
    )

    baseline = _baseline_from_study_or_eval(
        study=study,
        base=base,
        pairs=pairs,
        config=config,
        settings=settings,
    )
    tuned_params = dict(study.best_trial.user_attrs.get("params_full", base))
    if rerun_best:
        tuned = evaluate_candidate(
            pairs=pairs,
            config=config,
            settings=settings,
            params=tuned_params,
        )
    else:
        tuned = _summary_from_trial(study.best_trial, params=tuned_params)

    return AttackTuningResult(
        hash_id=settings.hash_id,
        attack_id=settings.attack_id,
        baseline=baseline,
        tuned=tuned,
        study=study,
    )


def run_hash_tuning_experiment(
    *,
    config_path: str | Path = "results/attack_configs.json",
    zip_path: str | Path = "notebooks/dataset_v1_public.zip",
    hash_id: str,
    attack_ids: list[str] | None = None,
    split: str = "public",
    max_pairs: int | None = 3,
    budget: int = 10_000,
    n_trials: int = 20,
    seed: int = 0,
    alpha: float = 1.0 / 50.0,
    asr_floor: float = 0.05,
    failure_gap_penalty: float = 0.0,
    category_choices: dict[Any, Any] | None = None,
    numeric_params: dict[Any, Any] | None = None,
    fixed_params: dict[Any, Any] | None = None,
    resize_size: int | None = None,
    resize_resample: str | None = None,
    storage: str | None = None,
    study_prefix: str | None = None,
    rerun_best: bool = False,
    verbose: bool = True,
    parallel_backend: str = "auto",
    parallel_workers: int | None = None,
    parallel_worker_cap: int = 30,
    parallel_photodna_worker_cap: int | None = None,
    parallel_min_pairs: int = 2,
    parallel_process_start_method: str | None = None,
    parallel_show_progress: bool = False,
) -> list[AttackTuningResult]:
    """Notebook-friendly runner for all configured attacks of one hash."""
    config = load_attack_config(config_path)
    preprocess = config.get("preprocess", {})
    resolved_resize_size = resize_size if resize_size is not None else preprocess.get("resize_size", 256)
    resolved_resize_resample = resize_resample or preprocess.get("resize_resample", "bilinear")
    resolved_attack_ids = attack_ids or list(config["configs"][hash_id].keys())
    pairs = load_pairs(zip_path, split=split, n_pairs=max_pairs)

    results: list[AttackTuningResult] = []
    for attack_id in resolved_attack_ids:
        if verbose:
            print(
                f"[{hash_id}/{attack_id}] pairs={len(pairs)} "
                f"budget={budget} trials={n_trials}",
                flush=True,
            )
        settings = OptimizerSettings(
            hash_id=hash_id,
            attack_id=attack_id,
            resize_size=resolved_resize_size,
            resize_resample=resolved_resize_resample,
            budget=budget,
            seed=seed,
            alpha=alpha,
            asr_floor=asr_floor,
            failure_gap_penalty=failure_gap_penalty,
            category_choices=dict(category_choices or {}),
            numeric_params=dict(numeric_params or {}),
            fixed_params=dict(fixed_params or {}),
            parallel_backend=parallel_backend,
            parallel_workers=parallel_workers,
            parallel_worker_cap=parallel_worker_cap,
            parallel_photodna_worker_cap=parallel_photodna_worker_cap,
            parallel_min_pairs=parallel_min_pairs,
            parallel_process_start_method=parallel_process_start_method,
            parallel_show_progress=parallel_show_progress,
        )
        runtime = resolve_parallel_runtime(settings, len(pairs))
        if verbose:
            print(
                f"  parallel={runtime.backend}"
                f"{'' if runtime.workers is None else f' workers={runtime.workers}'}"
                f"{'' if runtime.process_start_method is None else f' start={runtime.process_start_method}'}",
                flush=True,
            )
        study_name = None
        if study_prefix:
            study_name = f"{study_prefix}_{hash_id}_{attack_id}"
        results.append(
            run_attack_tuning(
                pairs=pairs,
                config=config,
                settings=settings,
                n_trials=n_trials,
                study_name=study_name,
                storage=storage,
                rerun_best=rerun_best,
            )
        )
    return results


def calibrate_photodna_process_workers(
    config: dict[str, Any] | str | Path = "results/attack_configs.json",
    *,
    attack_id: str = "nes",
    n_pairs: int = 30,
    budget: int = 100,
    seed: int = 0,
    resize_size: int | None = None,
    resize_resample: str | None = None,
    caps: Iterable[int] | None = None,
    stop_after_first_failure: bool = True,
    set_env: bool = False,
    verbose: bool = True,
) -> PhotoDNAWorkerCalibration:
    """Probe PhotoDNA process parallelism and recommend a worker cap.

    This is intended as a one-time initialization check for a machine.  It uses
    synthetic pairs and a small budget, compares every process run against a
    sequential reference, and recommends the fastest stable zero-mismatch cap.
    """
    cfg = load_attack_config(config) if isinstance(config, (str, Path)) else config
    preprocess = cfg.get("preprocess", {})
    resolved_resize_size = resize_size if resize_size is not None else preprocess.get("resize_size", 256)
    resolved_resize_resample = resize_resample or preprocess.get("resize_resample", "bilinear")
    pair_count = max(1, int(n_pairs))
    budget_value = int(budget)

    pairs = [_make_synthetic_pair(i) for i in range(pair_count)]
    params = attack_params_from_config(cfg, "photodna", attack_id)

    seq_settings = OptimizerSettings(
        hash_id="photodna",
        attack_id=attack_id,
        resize_size=resolved_resize_size,
        resize_resample=resolved_resize_resample,
        budget=budget_value,
        seed=int(seed),
        parallel_backend="sequential",
    )
    t0 = _perf_counter()
    reference = evaluate_candidate(pairs=pairs, config=cfg, settings=seq_settings, params=params)
    sequential_wall_sec = _perf_counter() - t0

    cap_list = sorted({int(c) for c in (caps or _default_calibration_caps(pair_count)) if int(c) >= 1})
    rows: list[dict[str, Any]] = [
        {
            "mode": "sequential",
            "cap": None,
            "backend": "sequential",
            "workers": 1,
            "wall_sec": sequential_wall_sec,
            "speedup": 1.0,
            "mismatches": 0,
            "stable": True,
            "error": None,
            "score": reference.score,
            "asr": reference.asr,
            "mean_queries": reference.mean_queries,
        }
    ]

    if verbose:
        print(
            f"[photodna calibration] sequential: {sequential_wall_sec:.2f}s "
            f"pairs={pair_count} budget={budget_value}",
            flush=True,
        )

    for cap in cap_list:
        settings = OptimizerSettings(
            hash_id="photodna",
            attack_id=attack_id,
            resize_size=resolved_resize_size,
            resize_resample=resolved_resize_resample,
            budget=budget_value,
            seed=int(seed),
            parallel_backend="auto",
            parallel_photodna_worker_cap=int(cap),
        )
        runtime = resolve_parallel_runtime(settings, pair_count)
        started = _perf_counter()
        row = {
            "mode": "parallel",
            "cap": int(cap),
            "backend": runtime.backend,
            "workers": runtime.workers,
            "start_method": runtime.process_start_method,
        }
        try:
            current = evaluate_candidate(pairs=pairs, config=cfg, settings=settings, params=params)
            wall_sec = _perf_counter() - started
            mismatches = _candidate_summary_mismatches(reference, current)
            row.update(
                {
                    "wall_sec": wall_sec,
                    "speedup": sequential_wall_sec / wall_sec if wall_sec else float("nan"),
                    "mismatches": len(mismatches),
                    "stable": len(mismatches) == 0,
                    "error": None,
                    "score": current.score,
                    "asr": current.asr,
                    "mean_queries": current.mean_queries,
                }
            )
            if verbose:
                status = "ok" if row["stable"] else f"mismatch={len(mismatches)}"
                print(
                    f"[photodna calibration] cap={cap} workers={runtime.workers}: "
                    f"{wall_sec:.2f}s speedup={row['speedup']:.2f} {status}",
                    flush=True,
                )
        except Exception as exc:
            wall_sec = _perf_counter() - started
            row.update(
                {
                    "wall_sec": wall_sec,
                    "speedup": float("nan"),
                    "mismatches": float("nan"),
                    "stable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "score": float("nan"),
                    "asr": float("nan"),
                    "mean_queries": float("nan"),
                }
            )
            if verbose:
                print(
                    f"[photodna calibration] cap={cap} workers={runtime.workers}: "
                    f"ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
        rows.append(row)
        if row.get("error") and stop_after_first_failure:
            break

    stable_rows = [r for r in rows if r.get("mode") == "parallel" and r.get("stable") is True]
    if stable_rows:
        best = min(stable_rows, key=lambda r: float(r["wall_sec"]))
        recommended = int(best["workers"] or best["cap"])
        max_stable = max(int(r["workers"] or r["cap"]) for r in stable_rows)
    else:
        recommended = 1
        max_stable = 1

    if set_env:
        os.environ["EVOHASH_PHOTODNA_MAX_WORKERS"] = str(recommended)

    if verbose:
        print(
            f"[photodna calibration] recommended_workers={recommended} "
            f"max_stable_workers={max_stable}"
            + (" env_set=1" if set_env else ""),
            flush=True,
        )

    return PhotoDNAWorkerCalibration(
        recommended_workers=recommended,
        max_stable_workers=max_stable,
        rows=rows,
        sequential_wall_sec=sequential_wall_sec,
        n_pairs=pair_count,
        budget=budget_value,
    )


def trials_dataframe(study: Any):
    import pandas as pd

    rows = []
    for trial in study.trials:
        summary = trial.user_attrs.get("summary", {})
        rows.append({
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            **summary,
            "params_full": trial.user_attrs.get("params_full", {}),
        })
    return pd.DataFrame(rows).sort_values("value", na_position="last").reset_index(drop=True)


def all_trials_dataframe(results: Iterable[AttackTuningResult]):
    import pandas as pd

    frames = []
    for result in results:
        if result.study is None:
            continue
        frame = trials_dataframe(result.study)
        frame.insert(0, "attack_id", result.attack_id)
        frame.insert(0, "hash_id", result.hash_id)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def tuning_plan_dataframe(
    *,
    config: dict[str, Any],
    hash_id: str,
    attack_ids: list[str] | None = None,
    category_choices: dict[Any, Any] | None = None,
    numeric_params: dict[Any, Any] | None = None,
    fixed_params: dict[Any, Any] | None = None,
):
    """Return a notebook-friendly view of what will be tuned or fixed."""
    import pandas as pd

    resolved_attack_ids = attack_ids or list(config["configs"][hash_id].keys())
    rows = []
    for attack_id in resolved_attack_ids:
        base = attack_params_from_config(config, hash_id, attack_id)
        settings = OptimizerSettings(
            hash_id=hash_id,
            attack_id=attack_id,
            category_choices=dict(category_choices or {}),
            numeric_params=dict(numeric_params or {}),
            fixed_params=dict(fixed_params or {}),
        )
        merged = dict(base)
        merged.update(FIXED_PARAMS_BY_HASH_ATTACK.get((hash_id, attack_id), {}))
        merged.update(_fixed_overrides(settings, attack_id))

        for param_name, value in sorted(merged.items()):
            fixed = _lookup_param_control(settings.fixed_params, hash_id, attack_id, param_name)
            category = _lookup_param_control(settings.category_choices, hash_id, attack_id, param_name)
            is_continuous = param_name in CONTINUOUS_NUMERIC_PARAMS_BY_ATTACK.get(attack_id, set())
            is_categorical = param_name in CATEGORICAL_PARAMS_BY_ATTACK.get(attack_id, set())

            if fixed is not _MISSING:
                mode = "fixed"
                choices = [fixed]
            elif is_continuous:
                mode = "auto_numeric" if _should_tune(settings, attack_id, param_name) else "fixed_numeric"
                choices = ""
            elif category is not _MISSING:
                choices = _resolve_category_choices(category, base=value, default_choices=[value])
                mode = _category_mode_label(category, choices=choices)
            elif is_categorical:
                mode = "category_default"
                choices = "engine default"
            else:
                mode = "fixed_base"
                choices = ""

            rows.append({
                "hash_id": hash_id,
                "attack_id": attack_id,
                "param": param_name,
                "base_value": base.get(param_name),
                "effective_value": value,
                "mode": mode,
                "choices": choices,
            })

    return pd.DataFrame(rows)


def comparison_dataframe(results: Iterable[AttackTuningResult]):
    import pandas as pd

    rows = []
    for result in results:
        base = result.baseline
        tuned = result.tuned
        rows.append({
            "hash_id": result.hash_id,
            "attack_id": result.attack_id,
            "baseline_score": base.score,
            "tuned_score": tuned.score,
            "delta_score": tuned.score - base.score,
            "pct_score": _safe_pct_change(tuned.score, base.score),
            "baseline_target_metric": base.target_metric,
            "tuned_target_metric": tuned.target_metric,
            "delta_target_metric": tuned.target_metric - base.target_metric,
            "baseline_ASR": base.asr,
            "tuned_ASR": tuned.asr,
            "delta_ASR": tuned.asr - base.asr,
            "baseline_successes": base.successes,
            "tuned_successes": tuned.successes,
            "baseline_mean_L2": base.mean_l2,
            "tuned_mean_L2": tuned.mean_l2,
            "delta_mean_L2": tuned.mean_l2 - base.mean_l2,
            "baseline_mean_queries": base.mean_queries,
            "tuned_mean_queries": tuned.mean_queries,
            "delta_mean_queries": tuned.mean_queries - base.mean_queries,
            "baseline_mean_final_hash_l1": base.mean_final_hash_l1,
            "tuned_mean_final_hash_l1": tuned.mean_final_hash_l1,
            "delta_mean_final_hash_l1": tuned.mean_final_hash_l1 - base.mean_final_hash_l1,
            "baseline_params_json": json.dumps(base.params, sort_keys=True),
            "tuned_params_json": json.dumps(tuned.params, sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values("tuned_score", na_position="last").reset_index(drop=True)


def tuned_configs_dict(results: Iterable[AttackTuningResult]) -> dict[str, dict[str, dict[str, Any]]]:
    configs: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        configs.setdefault(result.hash_id, {})[result.attack_id] = clean_public_params(result.tuned.params)
    return configs


def print_tuned_configs(results: Iterable[AttackTuningResult]) -> None:
    for result in results:
        summary = result.tuned
        print(
            f"\n[{result.hash_id}/{result.attack_id}] "
            f"score={summary.score:.6g} "
            f"target={summary.target_metric:.6g} "
            f"ASR={summary.asr:.3f} "
            f"L2={summary.mean_l2:.6g} "
            f"queries={summary.mean_queries:.1f}"
        )
        print(json.dumps(clean_public_params(summary.params), indent=2, sort_keys=True))


def _summary_attrs(summary: CandidateSummary) -> dict[str, Any]:
    return {
        "score": summary.score,
        "target_metric": summary.target_metric,
        "asr": summary.asr,
        "n_pairs": summary.n_pairs,
        "successes": summary.successes,
        "mean_l2": summary.mean_l2,
        "median_l2": summary.median_l2,
        "mean_queries": summary.mean_queries,
        "median_queries": summary.median_queries,
        "mean_time_sec": summary.mean_time_sec,
        "mean_initial_hash_l1": summary.mean_initial_hash_l1,
        "mean_final_hash_l1": summary.mean_final_hash_l1,
        "mean_best_hash_l1": summary.mean_best_hash_l1,
        "mean_gap_to_threshold": summary.mean_gap_to_threshold,
    }


def _suggest_float_multiplier(trial: Any, name: str, base: float, low_mult: float, high_mult: float) -> float:
    if base == 0.0:
        return trial.suggest_float(name, -high_mult, high_mult)
    low = min(base * low_mult, base * high_mult)
    high = max(base * low_mult, base * high_mult)
    if low > 0.0 and high > 0.0:
        return float(trial.suggest_float(name, low, high, log=True))
    return float(trial.suggest_float(name, low, high))


def _suggest_int_multiplier(trial: Any, name: str, base: int, multipliers: tuple[float, ...]) -> int:
    return int(trial.suggest_categorical(name, _nearby_int_choices(base, multipliers=multipliers, minimum=1)))


def _nearby_int_choices(base: int, *, multipliers: tuple[float, ...], minimum: int) -> list[int]:
    values = {max(int(round(base * mult)), minimum) for mult in multipliers}
    return sorted(values)


def _choice_union(base: Any, choices: Iterable[Any]) -> list[Any]:
    values: list[Any] = []
    for value in [base, *choices]:
        if value not in values:
            values.append(value)
    return values


_MISSING = object()


def _resolve_category_choices(override: Any, *, base: Any, default_choices: Iterable[Any]) -> list[Any]:
    if override is _MISSING:
        choices = list(default_choices)
    elif override is None or override is False:
        choices = [base]
    elif isinstance(override, dict):
        choices = _category_choices_from_spec(override, base=base)
    elif isinstance(override, (list, tuple, set)):
        choices = list(override)
    else:
        choices = [override]
    return _dedupe_choices(choices)


def _category_choices_from_spec(spec: dict[str, Any], *, base: Any) -> list[Any]:
    mode = str(spec.get("mode", "choices")).strip().lower()
    if mode in {"fixed", "value"}:
        return [spec.get("value", base)]
    if mode in {"choices", "choice", "list"}:
        return list(spec.get("choices", spec.get("values", [base])))
    if mode in {"around", "local"}:
        return _around_choices(spec, base=base)
    if mode in {"multipliers", "multiplier"}:
        multipliers = spec.get("multipliers", spec.get("values", [1.0]))
        values = [float(base) * float(mult) for mult in multipliers]
        if _is_int_like(base):
            values = [int(round(v)) for v in values]
        return _bounded_choices(values, spec=spec, base=base)
    raise ValueError(f"Unsupported category choice mode={mode!r}; expected choices, fixed, around, or multipliers")


def _around_choices(spec: dict[str, Any], *, base: Any) -> list[Any]:
    center = spec.get("center", spec.get("base", base))
    center_is_base = False
    if isinstance(center, str) and center.lower() in {"base", "default"}:
        center = base
        center_is_base = True
    elif center == base:
        center_is_base = True
    step = spec.get("step", 1)
    offsets = spec.get("offsets")

    if offsets is None:
        radius = spec.get("radius", 1)
        if isinstance(radius, (list, tuple)) and len(radius) == 2:
            lo_radius, hi_radius = int(radius[0]), int(radius[1])
        else:
            lo_radius = hi_radius = int(radius)
        offsets = range(-lo_radius, hi_radius + 1)

    values = [float(center) + float(offset) * float(step) for offset in offsets]
    if spec.get("include_base", center_is_base):
        values.append(base)
    if _is_int_like(base) and _is_int_like(step):
        values = [int(round(v)) for v in values]
    return _bounded_choices(values, spec=spec, base=base)


def _bounded_choices(values: Iterable[Any], *, spec: dict[str, Any], base: Any) -> list[Any]:
    minimum = spec.get("minimum", spec.get("min", 1 if _is_int_like(base) else None))
    maximum = spec.get("maximum", spec.get("max", None))
    out = []
    for value in values:
        if minimum is not None and float(value) < float(minimum):
            continue
        if maximum is not None and float(value) > float(maximum):
            continue
        out.append(value)
    return out or [base]


def _dedupe_choices(choices: Iterable[Any]) -> list[Any]:
    values: list[Any] = []
    for value in choices:
        if value not in values:
            values.append(value)
    return values


def _is_int_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _category_mode_label(override: Any, *, choices: list[Any]) -> str:
    if override is _MISSING:
        return "category_default"
    if isinstance(override, dict):
        mode = str(override.get("mode", "choices")).strip().lower()
        if len(choices) <= 1 or mode in {"fixed", "value"}:
            return "fixed_choice"
        return f"category_{mode}"
    return "fixed_choice" if len(choices) <= 1 else "category_limited"


def _suggest_categorical_param(
    trial: Any,
    settings: OptimizerSettings | None,
    attack_id: str,
    param_name: str,
    base: Any,
    default_choices: Iterable[Any],
) -> Any:
    fixed = _lookup_param_control(
        getattr(settings, "fixed_params", {}) if settings is not None else {},
        getattr(settings, "hash_id", "") if settings is not None else "",
        attack_id,
        param_name,
    )
    if fixed is not _MISSING:
        return fixed

    override = _lookup_param_control(
        getattr(settings, "category_choices", {}) if settings is not None else {},
        getattr(settings, "hash_id", "") if settings is not None else "",
        attack_id,
        param_name,
    )
    choices = _resolve_category_choices(override, base=base, default_choices=default_choices)

    if not choices:
        choices = [base]
    if len(choices) == 1:
        return choices[0]
    return trial.suggest_categorical(param_name, choices)


def _should_tune(settings: OptimizerSettings | None, attack_id: str, param_name: str) -> bool:
    if settings is None:
        return True

    fixed = _lookup_param_control(settings.fixed_params, settings.hash_id, attack_id, param_name)
    if fixed is not _MISSING:
        return False

    controls = settings.numeric_params
    if not controls:
        return True

    specific = _lookup_param_control(controls, settings.hash_id, attack_id, param_name)
    if specific is not _MISSING:
        return _truthy_tune_control(specific)

    attack_block = _lookup_attack_block(controls, settings.hash_id, attack_id)
    if attack_block is _MISSING:
        return True
    if isinstance(attack_block, dict):
        return _truthy_tune_control(attack_block.get(param_name, False))
    if isinstance(attack_block, (list, tuple, set)):
        return param_name in attack_block
    return _truthy_tune_control(attack_block)


def _fixed_overrides(settings: OptimizerSettings, attack_id: str) -> dict[str, Any]:
    block = _lookup_attack_block(settings.fixed_params, settings.hash_id, attack_id)
    if isinstance(block, dict):
        return dict(block)
    return {}


def _truthy_tune_control(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"auto", "tune", "true", "yes", "all"}
    return bool(value)


def _lookup_attack_block(mapping: dict[Any, Any], hash_id: str, attack_id: str) -> Any:
    if not mapping:
        return _MISSING

    for key in ((hash_id, attack_id), f"{hash_id}.{attack_id}", attack_id):
        if key in mapping:
            return mapping[key]

    hash_block = mapping.get(hash_id, _MISSING)
    if isinstance(hash_block, dict):
        for key in (attack_id, f"{hash_id}.{attack_id}"):
            if key in hash_block:
                return hash_block[key]

    return _MISSING


def _lookup_param_control(mapping: dict[Any, Any], hash_id: str, attack_id: str, param_name: str) -> Any:
    if not mapping:
        return _MISSING

    for key in (
        (hash_id, attack_id, param_name),
        f"{hash_id}.{attack_id}.{param_name}",
        (attack_id, param_name),
        f"{attack_id}.{param_name}",
        param_name,
    ):
        if key in mapping:
            return mapping[key]

    attack_block = _lookup_attack_block(mapping, hash_id, attack_id)
    if isinstance(attack_block, dict) and param_name in attack_block:
        return attack_block[param_name]

    return _MISSING


def _runtime_params_for_budget(params: dict[str, Any], settings: OptimizerSettings) -> dict[str, Any]:
    runtime = dict(params)
    if settings.budget_controls_iters and settings.budget is not None:
        runtime["max_iters"] = int(settings.budget)
    return runtime


def _perf_counter() -> float:
    return time.perf_counter()


def _default_calibration_caps(n_pairs: int) -> list[int]:
    limit = min(max(1, int(n_pairs)), _available_cpu_count())
    if limit <= 16:
        return list(range(1, limit + 1))

    caps = {1, 2, 4, 8, 12, 16, limit}
    caps.update(range(20, limit + 1, 4))
    return sorted(c for c in caps if 1 <= c <= limit)


def _make_synthetic_pair(seed: int, size: int = 256) -> PairSample:
    from PIL import Image, ImageDraw, ImageFilter

    def synth(local_seed: int) -> np.ndarray:
        rng = np.random.default_rng(local_seed)
        img = Image.new("RGB", (size, size), color=tuple(rng.integers(80, 200, 3).tolist()))
        draw = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(8, 18))):
            kind = rng.choice(["ellipse", "rectangle"])
            x0, y0 = rng.integers(0, size - 40, 2)
            x1 = min(size - 1, x0 + int(rng.integers(30, 120)))
            y1 = min(size - 1, y0 + int(rng.integers(30, 120)))
            color = tuple(rng.integers(0, 256, 3).tolist())
            if kind == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        return np.asarray(img, dtype=np.uint8)

    return PairSample(
        pair_id=f"synthetic_{int(seed):04d}",
        x=synth(2 * int(seed) + 1),
        y=synth(2 * int(seed) + 2),
        meta={},
    )


def _candidate_summary_mismatches(
    left: CandidateSummary,
    right: CandidateSummary,
    *,
    atol: float = 1e-6,
) -> list[dict[str, Any]]:
    fields = [
        "score",
        "target_metric",
        "asr",
        "successes",
        "mean_l2",
        "median_l2",
        "mean_queries",
        "median_queries",
        "mean_initial_hash_l1",
        "mean_final_hash_l1",
        "mean_best_hash_l1",
        "mean_gap_to_threshold",
    ]
    rows = []
    for field_name in fields:
        a = getattr(left, field_name)
        b = getattr(right, field_name)
        if isinstance(a, float):
            if not np.isclose(float(a), float(b), rtol=0.0, atol=atol, equal_nan=True):
                rows.append(
                    {
                        "field": field_name,
                        "sequential": a,
                        "parallel": b,
                        "abs_diff": abs(float(a) - float(b)),
                    }
                )
        elif a != b:
            rows.append(
                {
                    "field": field_name,
                    "sequential": a,
                    "parallel": b,
                    "abs_diff": None,
                }
            )
    return rows


def _clean_row_dict(row: EvalRow) -> dict[str, Any]:
    row_dict = row.to_dict()
    if isinstance(row_dict.get("params"), dict):
        row_dict["params"] = clean_public_params(row_dict["params"])
    return row_dict


def _summary_from_trial(trial: Any, *, params: dict[str, Any]) -> CandidateSummary:
    attrs = dict(trial.user_attrs.get("summary", {}))
    if not attrs:
        raise ValueError(f"Trial {getattr(trial, 'number', '?')} does not contain optimizer summary")
    score = float(attrs.get("score", trial.value))
    return CandidateSummary(
        score=score,
        target_metric=float(attrs["target_metric"]),
        asr=float(attrs["asr"]),
        n_pairs=int(attrs["n_pairs"]),
        successes=int(attrs["successes"]),
        mean_l2=float(attrs["mean_l2"]),
        median_l2=float(attrs.get("median_l2", attrs["mean_l2"])),
        mean_queries=float(attrs["mean_queries"]),
        median_queries=float(attrs.get("median_queries", attrs["mean_queries"])),
        mean_time_sec=float(attrs.get("mean_time_sec", 0.0)),
        mean_initial_hash_l1=float(attrs.get("mean_initial_hash_l1", 0.0)),
        mean_final_hash_l1=float(attrs["mean_final_hash_l1"]),
        mean_best_hash_l1=float(attrs.get("mean_best_hash_l1", attrs["mean_final_hash_l1"])),
        mean_gap_to_threshold=float(attrs["mean_gap_to_threshold"]),
        params=clean_public_params(dict(params)),
        rows=[],
    )


def _baseline_from_study_or_eval(
    *,
    study: Any,
    base: dict[str, Any],
    pairs: list[PairSample],
    config: dict[str, Any],
    settings: OptimizerSettings,
) -> CandidateSummary:
    base_clean = clean_public_params(base)
    if study.trials:
        first = study.trials[0]
        first_params = clean_public_params(dict(first.user_attrs.get("params_full", {})))
        if first_params == base_clean and first.user_attrs.get("summary"):
            return _summary_from_trial(first, params=base_clean)
    return evaluate_candidate(
        pairs=pairs,
        config=config,
        settings=settings,
        params=base_clean,
    )


def _can_enqueue_base(settings: OptimizerSettings, base: dict[str, Any]) -> bool:
    base_clean = clean_public_params(base)
    fixed = _fixed_overrides(settings, settings.attack_id)
    for key, value in fixed.items():
        if base_clean.get(key) != value:
            return False

    for key, value in base_clean.items():
        override = _lookup_param_control(
            settings.category_choices,
            settings.hash_id,
            settings.attack_id,
            key,
        )
        if override is _MISSING:
            continue
        allowed = _resolve_category_choices(override, base=value, default_choices=[value])
        if value not in allowed:
            return False
    return True


def _safe_pct_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return float("nan")
    return float((new - old) / abs(old))


def _search_space_projection(attack_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return only params Optuna actually suggests, for enqueueing baseline."""
    projected: dict[str, Any] = {}
    params = clean_public_params(params)
    if attack_id in {"nes", "prokos", "nes_attack_v0"}:
        for key in ("n_samples", "sigma", "lr"):
            if key in params:
                projected[key] = params[key]
        if attack_id in {"prokos", "nes_attack_v0"} and "momentum" in params:
            projected["momentum"] = params["momentum"]
        if attack_id in {"prokos", "nes_attack_v0"} and "grayscale_noise" in params:
            projected["grayscale_noise"] = params["grayscale_noise"]
        if attack_id == "prokos" and "antithetic" in params:
            projected["antithetic"] = params["antithetic"]
    elif attack_id == "simba":
        for key in ("epsilon", "freq_dims", "stride"):
            if key in params:
                projected[key] = params[key]
    elif attack_id == "zo_signsgd":
        for key in ("estimator", "direction_dist", "q", "mu", "lr", "eval_updated_point"):
            if key in params:
                projected[key] = params[key]
    elif attack_id == "atkscopes":
        for key in ("a", "lr", "max_freq", "patch_size", "beta1", "beta2"):
            if key in params and params[key] is not None:
                projected[key] = params[key]
    return projected
