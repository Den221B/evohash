"""Parallel evaluation helpers for independent EvoHash attack jobs.

The helpers here deliberately keep the existing pipeline intact: each worker
calls ``evaluate_attack(...)`` with the same inputs a sequential notebook would
use.  Parallelism is only across independent image-pair jobs.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from evohash.attacks import build_attack_registry
from evohash.evaluator import evaluate_attack
from evohash.hashes import build_hash_fn

Backend = Literal["sequential", "thread", "process"]
SeedPolicy = Literal["constant", "pair_index"]

_WORKER_CONTEXT: dict[str, Any] = {}
_WORKER_LOCAL = threading.local()

# Public constant — imported by optimizer.py to stay in sync.
META_PARAM_KEYS = {
    "profile",
    "source_units_note",
    "max_iters",
    "n_iter",
    "log_every",
}


def make_eval_jobs(
    *,
    samples: Iterable[Any],
    config: dict[str, Any],
    hash_ids: Iterable[str],
    attack_id: str,
    base_seed: int = 0,
    seed_policy: SeedPolicy = "constant",
) -> list[dict[str, Any]]:
    """Build deterministic job payloads for one attack across hashes and pairs."""
    sample_list = list(samples)
    jobs: list[dict[str, Any]] = []
    job_index = 0

    for hash_id in hash_ids:
        params = clean_public_params(config["configs"][hash_id][attack_id])
        for pair_index, sample in enumerate(sample_list):
            if seed_policy == "constant":
                seed = int(base_seed)
            elif seed_policy == "pair_index":
                seed = int(base_seed) + int(pair_index)
            else:
                raise ValueError(f"Unsupported seed_policy={seed_policy!r}")

            jobs.append(
                {
                    "job_index": job_index,
                    "hash_id": str(hash_id),
                    "attack_id": str(attack_id),
                    "pair_index": int(pair_index),
                    "pair_id": getattr(sample, "pair_id", str(pair_index)),
                    "x_source": np.asarray(sample.x),
                    "x_target": np.asarray(sample.y),
                    "params": dict(params),
                    "seed": seed,
                }
            )
            job_index += 1

    return jobs


def run_eval_jobs(
    jobs: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    budget: int,
    resize_size: int | None,
    resize_resample: str = "bilinear",
    alpha: float = 1.0 / 50.0,
    backend: Backend = "thread",
    max_workers: int | None = None,
    process_start_method: str | None = None,
    show_progress: bool = True,
    compute_lpips: bool = False,
    keep_history: bool = False,
    return_x_best: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate jobs sequentially or in a thread/process pool.

    Results are sorted by ``job_index`` so downstream grouping and comparisons
    stay stable regardless of completion order.

    Pass ``compute_lpips=True`` to compute LPIPS for every pair (requires
    ``torch`` + ``lpips``).  Defaults to False — the metric is expensive
    (~10–30 ms/pair) and not needed during search or evolution loops.

    Pass ``keep_history=True`` to retain a downsampled (~100-point) hash-distance
    history per row (used by the stage1 search convergence plots). Off by default
    to keep result rows small.
    """
    job_list = list(jobs)
    context = {
        "config": config,
        "budget": int(budget),
        "resize_size": resize_size,
        "resize_resample": str(resize_resample),
        "alpha": float(alpha),
        "compute_lpips": bool(compute_lpips),
        "keep_history": bool(keep_history),
        "return_x_best": bool(return_x_best),
        "backend": str(backend),
    }

    with EvalJobRunner(
        context=context,
        backend=backend,
        max_workers=max_workers,
        process_start_method=process_start_method,
        show_progress=show_progress,
    ) as runner:
        return runner.run(job_list)


class EvalJobRunner:
    """Reusable executor for repeated batches with the same eval context."""

    def __init__(
        self,
        *,
        context: dict[str, Any],
        backend: Backend = "thread",
        max_workers: int | None = None,
        process_start_method: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.context = dict(context)
        self.backend = backend
        self.max_workers = max_workers
        self.process_start_method = process_start_method
        self.show_progress = show_progress
        self._executor: Executor | None = None

        if self.backend not in {"sequential", "thread", "process"}:
            raise ValueError("backend must be one of: sequential, thread, process")

    def run(self, jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        job_list = list(jobs)
        if not job_list:
            return []

        if self.backend == "sequential":
            _init_worker(self.context)
            rows = [
                _run_one_job(job)
                for job in _maybe_progress(job_list, self.show_progress, "sequential")
            ]
            return sorted(rows, key=lambda row: int(row["job_index"]))

        executor = self._ensure_executor(len(job_list))
        rows: list[dict[str, Any]] = []
        futures = [executor.submit(_run_one_job, job) for job in job_list]
        try:
            iterator = as_completed(futures)
            for future in _maybe_progress(iterator, self.show_progress, self.backend, total=len(futures)):
                rows.append(future.result())
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            self._shutdown_executor(wait=False, cancel_futures=True, terminate_processes=True)
            raise
        return sorted(rows, key=lambda row: int(row["job_index"]))

    def close(self) -> None:
        self._shutdown_executor(wait=True, cancel_futures=False, terminate_processes=False)

    def _shutdown_executor(
        self,
        *,
        wait: bool,
        cancel_futures: bool,
        terminate_processes: bool,
    ) -> None:
        if self._executor is not None:
            executor = self._executor
            if terminate_processes:
                self._terminate_executor_processes(executor)
            try:
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            except TypeError:
                executor.shutdown(wait=wait)
            self._executor = None

    def _terminate_executor_processes(self, executor: Executor) -> None:
        if self.backend != "process":
            return
        processes = getattr(executor, "_processes", None)
        if not processes:
            return
        for process in list(processes.values()):
            if process.is_alive():
                process.terminate()

    def _ensure_executor(self, n_jobs: int) -> Executor:
        if self._executor is not None:
            return self._executor

        if self.max_workers is None:
            max_workers = min(n_jobs, os.cpu_count() or 1)
        else:
            max_workers = min(n_jobs, max(1, int(self.max_workers)))

        if self.backend == "thread":
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(self.context,),
            )
        elif self.backend == "process":
            executor_kwargs: dict[str, Any] = {}
            if self.process_start_method:
                executor_kwargs["mp_context"] = mp.get_context(self.process_start_method)
            self._executor = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_process_worker,
                initargs=(self.context,),
                **executor_kwargs,
            )
        else:
            raise ValueError("backend must be one of: sequential, thread, process")

        return self._executor

    def __enter__(self) -> "EvalJobRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _init_process_worker(context: dict[str, Any]) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    _init_worker(context)


def _init_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    _WORKER_LOCAL.hash_cache = {}
    _WORKER_LOCAL.attack_registry = None
    _WORKER_LOCAL.program_cache = {}


def _local_cache() -> dict[str, Any]:
    if not hasattr(_WORKER_LOCAL, "hash_cache"):
        _WORKER_LOCAL.hash_cache = {}
    if not hasattr(_WORKER_LOCAL, "attack_registry"):
        _WORKER_LOCAL.attack_registry = None
    if not hasattr(_WORKER_LOCAL, "program_cache"):
        _WORKER_LOCAL.program_cache = {}
    return _WORKER_LOCAL.__dict__


def _get_hash(hash_id: str):
    cache = _local_cache()
    hash_cache = cache["hash_cache"]
    if hash_id not in hash_cache:
        if hash_id == "neuralhash" and _WORKER_CONTEXT.get("backend") == "process":
            os.environ["NEURALHASH_PROVIDER"] = "cpu"
            os.environ.setdefault("NEURALHASH_INTRA_THREADS", "1")
            os.environ.setdefault("NEURALHASH_INTER_THREADS", "1")
        config = _WORKER_CONTEXT["config"]
        hash_cache[hash_id] = build_hash_fn(
            hash_id,
            thresholds=config.get("thresholds", {}),
        )
    return hash_cache[hash_id]


def _get_attack(attack_id: str, program_path: str | None = None):
    cache = _local_cache()
    if program_path:
        program_cache = cache["program_cache"]
        resolved = str(Path(program_path).resolve())
        if resolved not in program_cache:
            program_cache[resolved] = _load_attack_module(resolved, attack_id)
        return program_cache[resolved]

    if cache["attack_registry"] is None:
        cache["attack_registry"] = build_attack_registry()
    return cache["attack_registry"].get(attack_id)


def _downsample_history(history: Any, n: int = 100) -> list:
    """Downsample a hash-distance history to at most ~n points (for plots)."""
    if not history:
        return []
    h = list(history)
    if len(h) <= n:
        return [float(v) for v in h]
    step = max(1, len(h) // n)
    return [float(v) for v in h[::step]]


def _run_one_job(job: dict[str, Any]) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    hash_id = str(job["hash_id"])
    attack_id = str(job["attack_id"])
    program_path = job.get("program_path")
    params = dict(job["params"])

    try:
        row = evaluate_attack(
            attack=_get_attack(attack_id, str(program_path) if program_path else None),
            hash_fn=_get_hash(hash_id),
            x_source=job["x_source"],
            x_target=job["x_target"],
            params=params,
            budget=int(context["budget"]),
            seed=int(job["seed"]),
            pair_id=job.get("pair_id"),
            resize_size=context["resize_size"],
            resize_resample=str(context["resize_resample"]),
            compute_lpips=bool(context.get("compute_lpips", False)),
        )
        data = row.to_dict()
        if context.get("keep_history"):
            data["history"] = _downsample_history(data.get("history"))
        else:
            data.pop("history", None)
        if context.get("return_x_best"):
            # row.x_best survives (to_dict drops it from the copy, not the object);
            # numpy array pickles back fine through the process pool.
            data["x_best"] = np.asarray(row.x_best, dtype=np.float32)
        data.pop("params", None)
        data.pop("extra", None)
        data.update(
            {
                "job_index": int(job["job_index"]),
                "pair_index": int(job["pair_index"]),
                "status": "ok",
                "error": None,
                "program_path": str(program_path) if program_path else None,
                "params_json": json.dumps(params, sort_keys=True),
                "failure_gap": max(0.0, float(row.final_hash_l1) - float(row.threshold)),
                "target_metric": float(row.pixel_l2_raw) + float(context["alpha"]) * float(row.queries),
            }
        )
        return data
    except Exception as exc:
        return {
            "job_index": int(job["job_index"]),
            "pair_index": int(job["pair_index"]),
            "pair_id": job.get("pair_id"),
            "hash_id": hash_id,
            "attack_id": attack_id,
            "program_path": str(program_path) if program_path else None,
            "seed": int(job["seed"]),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
            "success": False,
            "queries": 0,
            "budget": int(context["budget"]),
            "time_sec": 0.0,
            "pixel_l2": np.nan,
            "pixel_l2_raw": np.nan,
            "initial_hash_l1": np.nan,
            "final_hash_l1": np.nan,
            "best_hash_l1": np.nan,
            "threshold": np.nan,
            "relative_improvement": np.nan,
            "improved": False,
            "failure_gap": np.nan,
            "target_metric": np.inf,
            "params_json": json.dumps(params, sort_keys=True),
        }


def _maybe_progress(items, show_progress: bool, desc: str, total: int | None = None):
    if not show_progress:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc, total=total)
    except Exception:
        return items


def clean_public_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop metadata and runtime-only keys from an attack config dict.

    The canonical implementation — imported by ``optimizer.py`` to stay in sync.
    """
    clean = dict(params)
    for key in META_PARAM_KEYS:
        clean.pop(key, None)
    return clean


_MODULE_LOAD_LOCK = threading.Lock()


def _load_attack_module(program_path: str, attack_id: str):
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"Evolved attack program not found: {path}")

    # Unique module name per load + a lock around exec_module. Default backend is
    # 'process' (unaffected), but with the THREAD backend two threads loading the
    # same file shared one sys.modules name and the pop-in-finally could yank a
    # module another thread was still executing -> intermittent
    # "AttributeError: 'NoneType' object has no attribute '__dict__'". This makes
    # 'thread'/'auto' safe too.
    module_name = f"_evohash_evolved_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evolved attack program: {path}")
    module = importlib.util.module_from_spec(spec)
    with _MODULE_LOAD_LOCK:
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)

    if not hasattr(module, "run_attack") or not callable(module.run_attack):
        raise AttributeError(
            f"Evolved attack {path} must define run_attack(x_source, oracle, params=None, budget=..., seed=...)"
        )
    if not hasattr(module, "ATTACK_ID"):
        module.ATTACK_ID = attack_id
    return module
