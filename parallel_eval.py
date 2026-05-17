"""Parallel evaluation helpers for independent EvoHash attack jobs.

The helpers here deliberately keep the existing pipeline intact: each worker
calls ``evaluate_attack(...)`` with the same inputs a sequential notebook would
use.  Parallelism is only across independent image-pair jobs.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from evohash.attacks import build_attack_registry
from evohash.evaluator import evaluate_attack
from evohash.hashes import NeuralHashWrapper, PDQWrapper, PHashWrapper, PhotoDNAWrapper

Backend = Literal["sequential", "thread", "process"]
SeedPolicy = Literal["constant", "pair_index"]

_WORKER_CONTEXT: dict[str, Any] = {}
_WORKER_LOCAL = threading.local()

_META_PARAM_KEYS = {
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
        params = _clean_public_params(config["configs"][hash_id][attack_id])
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
) -> list[dict[str, Any]]:
    """Evaluate jobs sequentially or in a thread/process pool.

    Results are sorted by ``job_index`` so downstream grouping and comparisons
    stay stable regardless of completion order.
    """
    job_list = list(jobs)
    context = {
        "config": config,
        "budget": int(budget),
        "resize_size": resize_size,
        "resize_resample": str(resize_resample),
        "alpha": float(alpha),
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
        iterator = as_completed(futures)
        for future in _maybe_progress(iterator, self.show_progress, self.backend, total=len(futures)):
            rows.append(future.result())
        return sorted(rows, key=lambda row: int(row["job_index"]))

    def close(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                self._executor.shutdown(wait=True)
            self._executor = None

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
                initializer=_init_worker,
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


def _init_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    _WORKER_LOCAL.hash_cache = {}
    _WORKER_LOCAL.attack_registry = None


def _local_cache() -> dict[str, Any]:
    if not hasattr(_WORKER_LOCAL, "hash_cache"):
        _WORKER_LOCAL.hash_cache = {}
    if not hasattr(_WORKER_LOCAL, "attack_registry"):
        _WORKER_LOCAL.attack_registry = None
    return _WORKER_LOCAL.__dict__


def _get_hash(hash_id: str):
    cache = _local_cache()
    hash_cache = cache["hash_cache"]
    if hash_id not in hash_cache:
        hash_cache[hash_id] = _build_hash_fn(hash_id, _WORKER_CONTEXT["config"])
    return hash_cache[hash_id]


def _get_attack(attack_id: str):
    cache = _local_cache()
    if cache["attack_registry"] is None:
        cache["attack_registry"] = build_attack_registry()
    return cache["attack_registry"].get(attack_id)


def _run_one_job(job: dict[str, Any]) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    hash_id = str(job["hash_id"])
    attack_id = str(job["attack_id"])
    params = dict(job["params"])

    try:
        row = evaluate_attack(
            attack=_get_attack(attack_id),
            hash_fn=_get_hash(hash_id),
            x_source=job["x_source"],
            x_target=job["x_target"],
            params=params,
            budget=int(context["budget"]),
            seed=int(job["seed"]),
            pair_id=job.get("pair_id"),
            resize_size=context["resize_size"],
            resize_resample=str(context["resize_resample"]),
        )
        data = row.to_dict()
        data.pop("history", None)
        data.pop("params", None)
        data.pop("extra", None)
        data.update(
            {
                "job_index": int(job["job_index"]),
                "pair_index": int(job["pair_index"]),
                "status": "ok",
                "error": None,
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


def _clean_public_params(params: dict[str, Any]) -> dict[str, Any]:
    clean = dict(params)
    for key in _META_PARAM_KEYS:
        clean.pop(key, None)
    return clean


def _build_hash_fn(hash_id: str, config: dict[str, Any]):
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

    raise ValueError(f"Unknown hash_id for parallel eval hash factory: {hash_id!r}")
