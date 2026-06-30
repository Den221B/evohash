"""Efficient benchmark stage: per-hash budgets, config-driven attacks, one pass.

* Attacks per hash come from the config (``configs[hash].keys()``): no config ->
  the attack is not run. NeuralHash therefore runs only the optimized attack;
  ``nes_attack_v0`` is excluded by default (diploma base set).
* Each hash gets its own query budget.
* A single run saves every artifact (metrics, convergence history, source/target
  and attacked images) via :class:`RunStore`, so nothing is recomputed later.
* Execution reuses the production parallel evaluator (:func:`run_eval_jobs`):
  **process** backend by default on all hashes, with a manual worker cap. The
  evaluator already loads evolved programs in workers, builds per-worker hashes,
  and (process backend) runs NeuralHash on the CPU provider — so artifacts now
  include ``x_best`` via the new ``return_x_best`` flag.
* Resume skips done attacks.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evohash.metrics import to_float32
from evohash.optimizer import (
    attack_params_from_config,
    load_attack_config,
    load_pairs,
)
from evohash.parallel_eval import run_eval_jobs
from evohash.preprocessing import ResizeSpec, apply_resize

from .artifacts import RunStore, downsample_history
from .score_evolved import discover_evolved_programs

_PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZED_CONFIG = _PKG_DIR / "results" / "attack_configs_optimized.json"
DEFAULT_THRESHOLDS = {"phash": 12.0, "pdq": 92.0, "neuralhash": 17.0, "photodna": 3855.0}


@dataclass
class BenchmarkConfig:
    hashes: tuple[str, ...] = ("phash", "pdq", "neuralhash", "photodna")
    dataset_zip: str | Path = "dataset_v1_public.zip"
    split: str = "public"
    max_pairs: int | None = 30
    budgets: dict[str, int] = field(default_factory=dict)         # per-hash override
    default_budget: int = 10_000
    seed: int = 0
    resize_size: int | None = 256
    resize_resample: str = "bilinear"
    optimized_config_path: str | Path = DEFAULT_OPTIMIZED_CONFIG
    include_evolved: bool = True
    exclude_attacks: tuple[str, ...] = ("nes_attack_v0",)
    compute_lpips: bool = True
    # Parallelism (reuses parallel_eval): process on all hashes, manual cap.
    backend: str = "process"                                       # process | thread | sequential
    workers: dict[str, int] | int | None = None                   # per-hash / global cap; None = auto
    process_start_method: str | None = None                       # e.g. "spawn"
    history_max_points: int = 500
    output_root: str | Path = "outputs/stage_bench"
    run_name: str | None = None
    skip_existing: bool = True
    thresholds: dict[str, float] | None = None
    alpha: float = 1.0 / 50.0

    def budget_for(self, hash_id: str) -> int:
        return int(self.budgets.get(hash_id, self.default_budget))

    def workers_for(self, hash_id: str) -> int | None:
        if isinstance(self.workers, dict):
            return self.workers.get(hash_id)
        return self.workers  # int or None (run_eval_jobs caps by n_jobs / cpu)


def attacks_for_hash(cfg: BenchmarkConfig, hash_id: str, config: dict[str, Any]):
    """Return ``[(name, attack_id, program_path, params, attack_type), ...]``.

    ``attack_id``/``program_path`` are what the worker needs to rebuild the attack
    (no live objects are pickled across processes).
    """
    specs: list[tuple[str, str, str | None, dict[str, Any], str]] = []
    for attack_id in (config.get("configs", {}).get(hash_id) or {}):
        if attack_id in cfg.exclude_attacks:
            continue
        specs.append((attack_id, attack_id, None,
                      attack_params_from_config(config, hash_id, attack_id), "registry"))
    if cfg.include_evolved:
        for path in discover_evolved_programs(hash_id):
            p = Path(path)
            specs.append((p.stem, p.stem, str(p.resolve()), {}, "evolved"))
    return specs


def _run_attack_over_pairs(
    *,
    name: str,
    attack_id: str,
    program_path: str | None,
    params: dict[str, Any],
    pairs: Sequence[Any],
    hash_id: str,
    run_config: dict[str, Any],
    budget: int,
    cfg: BenchmarkConfig,
    workers: int | None,
):
    jobs = []
    for i, s in enumerate(pairs):
        jobs.append({
            "job_index": i,
            "hash_id": hash_id,
            "attack_id": attack_id,
            "program_path": program_path,
            "pair_index": i,
            "pair_id": str(getattr(s, "pair_id", i)),
            "x_source": np.asarray(s.x),
            "x_target": np.asarray(s.y),
            "params": dict(params),
            "seed": int(cfg.seed) + i,
        })

    rows = run_eval_jobs(
        jobs,
        config=run_config,
        budget=budget,
        resize_size=cfg.resize_size,
        resize_resample=cfg.resize_resample,
        alpha=cfg.alpha,
        backend=cfg.backend,
        max_workers=workers,
        process_start_method=cfg.process_start_method,
        show_progress=False,
        compute_lpips=cfg.compute_lpips,
        keep_history=True,
        return_x_best=True,
    )

    out_rows, histories, adv_images = [], {}, {}
    for r in rows:
        if r.get("status") != "ok":
            raise RuntimeError(f"{hash_id}/{name} pair={r.get('pair_id')}: {r.get('error')}")
        pid = str(r["pair_id"])
        out_rows.append({
            "pair_id": pid,
            "success": bool(r["success"]),
            "initial_hash_l1": float(r["initial_hash_l1"]),
            "final_hash_l1": float(r["final_hash_l1"]),
            "best_hash_l1": float(r["best_hash_l1"]),
            "threshold": float(r["threshold"]),
            "pixel_l2": float(r["pixel_l2"]),
            "pixel_l2_raw": float(r["pixel_l2_raw"]),
            "lpips": float(r.get("lpips", float("nan"))),
            "queries": int(r["queries"]),
            "budget": int(budget),
            "time_sec": float(r["time_sec"]),
        })
        histories[pid] = downsample_history(r.get("history") or [], cfg.history_max_points)
        adv_images[pid] = to_float32(r["x_best"])
    out_rows.sort(key=lambda d: d["pair_id"])
    return out_rows, histories, adv_images


def run_benchmark(cfg: BenchmarkConfig | None = None, *, progress: bool = True) -> RunStore:
    cfg = cfg or BenchmarkConfig()
    config = load_attack_config(str(cfg.optimized_config_path))
    thresholds = cfg.thresholds or config.get("thresholds", DEFAULT_THRESHOLDS)
    run_config = {**config, "thresholds": thresholds}

    if cfg.run_name and (Path(cfg.output_root) / cfg.run_name / "manifest.json").exists():
        store = RunStore.open(Path(cfg.output_root) / cfg.run_name)
        if progress:
            print(f"resuming run: {store.root}")
    else:
        store = RunStore.create(
            {
                "dataset": str(cfg.dataset_zip),
                "split": cfg.split,
                "max_pairs": cfg.max_pairs,
                "seed": cfg.seed,
                "resize_size": cfg.resize_size,
                "default_budget": cfg.default_budget,
                "budgets": cfg.budgets,
                "exclude_attacks": list(cfg.exclude_attacks),
                "backend": cfg.backend,
                "thresholds": thresholds,
            },
            output_root=cfg.output_root,
            run_name=cfg.run_name,
        )
        if progress:
            print(f"run dir: {store.root}")

    for hash_id in cfg.hashes:
        if hash_id == "photodna" and platform.system() != "Windows":
            store.mark_skipped(hash_id, "PhotoDNA is Windows-only in this project")
            if progress:
                print(f"[{hash_id}] skipped (Windows-only)")
            continue

        specs = attacks_for_hash(cfg, hash_id, config)
        if not specs:
            store.mark_skipped(hash_id, "no config / evolved programs")
            if progress:
                print(f"[{hash_id}] skipped (no attacks)")
            continue

        pairs = load_pairs(cfg.dataset_zip, split=cfg.split, n_pairs=cfg.max_pairs)
        budget = cfg.budget_for(hash_id)
        workers = cfg.workers_for(hash_id)

        resize_spec = ResizeSpec(size=cfg.resize_size, resample=cfg.resize_resample)
        store.save_pair_images(
            hash_id,
            [(str(getattr(p, "pair_id", i)),
              apply_resize(to_float32(p.x), resize_spec),
              apply_resize(to_float32(p.y), resize_spec))
             for i, p in enumerate(pairs)],
        )
        store.record_hash(hash_id, budget=budget, attacks=[s[0] for s in specs], n_pairs=len(pairs))
        if progress:
            print(f"[{hash_id}] budget={budget} backend={cfg.backend} workers={workers} "
                  f"attacks={[s[0] for s in specs]}")

        for name, attack_id, program_path, params, atype in specs:
            if cfg.skip_existing and store.has_attack(hash_id, name):
                if progress:
                    print(f"  - {name}: skip (exists)")
                continue
            rows, histories, adv_images = _run_attack_over_pairs(
                name=name, attack_id=attack_id, program_path=program_path, params=params,
                pairs=pairs, hash_id=hash_id, run_config=run_config, budget=budget,
                cfg=cfg, workers=workers,
            )
            store.save_attack(hash_id, name, atype, rows, histories, adv_images)
            if progress:
                asr = float(np.mean([r["success"] for r in rows]))
                print(f"  - {name} ({atype}): ASR={asr:.2f} n={len(rows)}")

    store.finalize()
    if progress:
        print(f"done -> {store.root / 'summary.csv'}")
    return store


__all__ = ["BenchmarkConfig", "run_benchmark", "attacks_for_hash", "RunStore"]
