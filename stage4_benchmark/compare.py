"""Compare original/optimized registry configs and evolved attack programs.

The runner intentionally reuses the production evaluation stack:

* registry attacks go through ``optimizer.evaluate_candidate``;
* evolved ``run_attack`` files go through ``parallel_eval.run_eval_jobs``;
* both paths use ``OptimizerSettings`` for auto/process/thread resource logic.

The resulting tables are notebook-friendly and focus on the metrics requested
for reporting: L2, LPIPS, query count, and ASR.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from evohash.optimizer import (
    OptimizerSettings,
    attack_params_from_config,
    evaluate_candidate,
    load_attack_config as _load_attack_config,
    load_pairs,
    resolve_parallel_runtime,
    summarize_row_dicts,
)
from evohash.parallel_eval import run_eval_jobs


DEFAULT_HASHES = ("photodna", "pdq", "phash")
DEFAULT_ORIGINAL_CONFIG_PATH = "results/attack_configs.json"
DEFAULT_OPTIMIZED_CONFIG_PATH = "results/attack_configs_optimized.json"
DEFAULT_EVOLVED_DIR = "evolved_programs"
DEFAULT_OUTPUT_ROOT = "outputs/config_vs_evolved_benchmark"
DEFAULT_THRESHOLDS = {
    "phash": 12.0,
    "pdq": 92.0,
    "neuralhash": 17.0,
    "photodna": 3855.0,
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """Inputs for one benchmark run."""

    hash_ids: tuple[str, ...] = DEFAULT_HASHES
    dataset_zip: str | Path = "dataset_v1_public.zip"
    split: str = "public"
    max_pairs: int | None = None
    budget: int | None = None
    seed: int = 0
    resize_size: int | None = 256
    resize_resample: str = "bilinear"
    original_config_path: str | Path = DEFAULT_ORIGINAL_CONFIG_PATH
    optimized_config_path: str | Path = DEFAULT_OPTIMIZED_CONFIG_PATH
    # Backward-compatible alias: when set, it overrides optimized_config_path.
    config_path: str | Path | None = None
    evolved_dir: str | Path = DEFAULT_EVOLVED_DIR
    include_registry_configs: bool = True
    include_evolved_programs: bool = True
    compute_lpips: bool = True
    parallel_backend: str = "auto"
    workers: int | None = None
    worker_cap: int = 30
    photodna_worker_cap: int | None = None
    process_start_method: str | None = None
    show_progress: bool = True
    output_root: str | Path = DEFAULT_OUTPUT_ROOT
    run_name: str | None = None
    fail_fast: bool = False


@dataclass
class BenchmarkResult:
    """Saved outputs and in-memory rows for a benchmark run."""

    run_dir: Path
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary_csv(self) -> Path:
        return self.run_dir / "summary.csv"

    @property
    def summary_json(self) -> Path:
        return self.run_dir / "summary.json"

    def to_dataframe(self):
        return summary_dataframe(self.summary_rows)


def run_benchmark(spec: BenchmarkSpec | None = None) -> BenchmarkResult:
    """Run the comparison and save ``summary.csv`` / ``summary.json``.

    PhotoDNA is skipped with an error row on non-Windows hosts because this
    project's PhotoDNA backend is Windows-only.
    """

    spec = spec or BenchmarkSpec()
    original_cfg = _load_benchmark_config(spec.original_config_path)
    optimized_cfg = _load_benchmark_config(spec.config_path or spec.optimized_config_path)
    budget = int(
        spec.budget
        if spec.budget is not None
        else optimized_cfg.get("defaults", original_cfg.get("defaults", {})).get("budget", 10_000)
    )
    run_dir = _make_run_dir(spec)
    rows_dir = run_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    result = BenchmarkResult(run_dir=run_dir)
    _write_json(run_dir / "spec.json", _spec_to_dict(spec, budget=budget))

    for hash_id in spec.hash_ids:
        if hash_id == "photodna" and platform.system() != "Windows":
            error = _error_row(
                hash_id=hash_id,
                candidate_type="config",
                candidate="all",
                config_variant="original/optimized",
                error="PhotoDNA evaluation is Windows-only in this project.",
            )
            result.summary_rows.append(error)
            result.errors.append(error)
            continue

        pairs = load_pairs(spec.dataset_zip, split=spec.split, n_pairs=spec.max_pairs)

        if spec.include_registry_configs:
            for attack_id in _paired_config_attacks(original_cfg, optimized_cfg, hash_id):
                for config_variant, cfg in (
                    ("original", original_cfg),
                    ("optimized", optimized_cfg),
                ):
                    try:
                        row, raw_rows = _run_registry_candidate(
                            hash_id=hash_id,
                            attack_id=attack_id,
                            config_variant=config_variant,
                            pairs=pairs,
                            config=cfg,
                            spec=spec,
                            budget=budget,
                        )
                        result.summary_rows.append(row)
                        _write_json(rows_dir / _rows_filename(row), raw_rows)
                    except Exception as exc:  # noqa: BLE001 - benchmark should keep going
                        row = _error_row(
                            hash_id=hash_id,
                            candidate_type="config",
                            candidate=attack_id,
                            config_variant=config_variant,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        result.summary_rows.append(row)
                        result.errors.append(row)
                        if spec.fail_fast:
                            raise

        if spec.include_evolved_programs and hash_id in {"pdq", "phash"}:
            for program_path in discover_evolved_programs(spec.evolved_dir, hash_id=hash_id):
                candidate = program_path.stem
                try:
                    row, raw_rows = _run_evolved_candidate(
                        hash_id=hash_id,
                        program_path=program_path,
                        pairs=pairs,
                        config=optimized_cfg,
                        spec=spec,
                        budget=budget,
                    )
                    result.summary_rows.append(row)
                    _write_json(rows_dir / _rows_filename(row), raw_rows)
                except Exception as exc:  # noqa: BLE001
                    row = _error_row(
                        hash_id=hash_id,
                        candidate_type="evolved",
                        candidate=candidate,
                        config_variant="",
                        program_path=str(program_path),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    result.summary_rows.append(row)
                    result.errors.append(row)
                    if spec.fail_fast:
                        raise

    _save_summary(result)
    return result


def discover_evolved_programs(
    evolved_dir: str | Path = DEFAULT_EVOLVED_DIR,
    *,
    hash_id: str | None = None,
) -> list[Path]:
    """Find evolved programs, optionally constrained to one hash prefix."""

    root = Path(evolved_dir)
    if not root.is_absolute():
        root = _project_root() / root
    if not root.is_dir():
        return []
    pattern = f"{hash_id}_*.py" if hash_id else "*.py"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def summary_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    preferred = [
        "hash_id",
        "candidate_type",
        "config_variant",
        "candidate",
        "status",
        "asr",
        "mean_l2_raw",
        "mean_lpips",
        "mean_queries",
        "score",
        "n_success",
        "n_pairs",
        "median_l2_raw",
        "median_queries",
        "mean_time_sec",
        "backend",
        "workers",
        "program_path",
        "error",
    ]
    ordered = [col for col in preferred if col in df.columns]
    ordered.extend(col for col in df.columns if col not in ordered)
    return df[ordered]


def _run_registry_candidate(
    *,
    hash_id: str,
    attack_id: str,
    config_variant: str,
    pairs: list[Any],
    config: dict[str, Any],
    spec: BenchmarkSpec,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = _settings(spec, hash_id=hash_id, attack_id=attack_id, budget=budget)
    runtime = resolve_parallel_runtime(settings, len(pairs))
    params = attack_params_from_config(config, hash_id, attack_id)
    started = time.perf_counter()
    summary = evaluate_candidate(
        pairs=pairs,
        config=config,
        settings=settings,
        params=params,
    )
    wall_sec = time.perf_counter() - started
    row = _summary_row(
        hash_id=hash_id,
        candidate_type="config",
        config_variant=config_variant,
        candidate=attack_id,
        summary=summary,
        runtime=runtime,
        wall_sec=wall_sec,
    )
    row["params_json"] = json.dumps(summary.params, sort_keys=True)
    return row, summary.rows


def _run_evolved_candidate(
    *,
    hash_id: str,
    program_path: Path,
    pairs: list[Any],
    config: dict[str, Any],
    spec: BenchmarkSpec,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attack_id = program_path.stem
    settings = _settings(spec, hash_id=hash_id, attack_id=attack_id, budget=budget)
    runtime = resolve_parallel_runtime(settings, len(pairs))

    jobs = []
    for index, sample in enumerate(pairs):
        jobs.append(
            {
                "job_index": int(index),
                "hash_id": hash_id,
                "attack_id": attack_id,
                "program_path": str(program_path.resolve()),
                "pair_index": int(index),
                "pair_id": getattr(sample, "pair_id", str(index)),
                "x_source": sample.x,
                "x_target": sample.y,
                "params": {},
                "seed": int(spec.seed) + int(index),
            }
        )

    started = time.perf_counter()
    raw_rows = run_eval_jobs(
        jobs,
        config=config,
        budget=budget,
        resize_size=spec.resize_size,
        resize_resample=spec.resize_resample,
        alpha=settings.alpha,
        backend=runtime.backend,
        max_workers=runtime.workers,
        process_start_method=runtime.process_start_method,
        show_progress=spec.show_progress,
        compute_lpips=spec.compute_lpips,
    )
    wall_sec = time.perf_counter() - started

    errors = [row for row in raw_rows if row.get("status") != "ok"]
    if errors:
        first = errors[0]
        raise RuntimeError(f"pair={first.get('pair_id')}: {first.get('error')}")

    summary = summarize_row_dicts(raw_rows, params={}, settings=settings)
    row = _summary_row(
        hash_id=hash_id,
        candidate_type="evolved",
        config_variant="",
        candidate=attack_id,
        summary=summary,
        runtime=runtime,
        wall_sec=wall_sec,
        program_path=program_path,
    )
    return row, summary.rows


def _settings(spec: BenchmarkSpec, *, hash_id: str, attack_id: str, budget: int) -> OptimizerSettings:
    return OptimizerSettings(
        hash_id=hash_id,
        attack_id=attack_id,
        resize_size=spec.resize_size,
        resize_resample=spec.resize_resample,
        budget=budget,
        seed=int(spec.seed),
        parallel_backend=spec.parallel_backend,
        parallel_workers=spec.workers,
        parallel_worker_cap=int(spec.worker_cap),
        parallel_photodna_worker_cap=spec.photodna_worker_cap,
        parallel_process_start_method=spec.process_start_method,
        parallel_show_progress=spec.show_progress,
        compute_lpips=bool(spec.compute_lpips),
    )


def _summary_row(
    *,
    hash_id: str,
    candidate_type: str,
    config_variant: str,
    candidate: str,
    summary: Any,
    runtime: Any,
    wall_sec: float,
    program_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "hash_id": hash_id,
        "candidate_type": candidate_type,
        "config_variant": config_variant,
        "candidate": candidate,
        "status": "ok",
        "asr": float(summary.asr),
        "mean_l2_raw": float(summary.mean_l2),
        "median_l2_raw": float(summary.median_l2),
        "mean_lpips": float(summary.mean_lpips),
        "mean_queries": float(summary.mean_queries),
        "median_queries": float(summary.median_queries),
        "score": float(summary.score),
        "target_metric": float(summary.target_metric),
        "n_success": int(summary.successes),
        "n_pairs": int(summary.n_pairs),
        "mean_time_sec": float(summary.mean_time_sec),
        "wall_sec": float(wall_sec),
        "backend": runtime.backend,
        "workers": runtime.workers,
        "program_path": str(program_path) if program_path else "",
        "error": "",
    }


def _paired_config_attacks(
    original_config: dict[str, Any],
    optimized_config: dict[str, Any],
    hash_id: str,
) -> list[str]:
    original = set((original_config.get("configs", {}).get(hash_id) or {}).keys())
    optimized = set((optimized_config.get("configs", {}).get(hash_id) or {}).keys())
    return sorted(original & optimized)


def _load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load either the full attack config or a short ``best_configs.json``."""

    raw = _load_attack_config(path)
    if "configs" in raw:
        raw.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))
        raw.setdefault("defaults", {"budget": 10_000})
        return raw

    configs = {
        hash_id: attacks
        for hash_id, attacks in raw.items()
        if isinstance(attacks, dict)
    }
    attacks_seen = sorted({attack for attacks in configs.values() for attack in attacks})
    return {
        "hashes": sorted(configs),
        "allowed_attacks": attacks_seen,
        "defaults": {"budget": 10_000},
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "configs": configs,
    }


def _make_run_dir(spec: BenchmarkSpec) -> Path:
    root = Path(spec.output_root)
    if not root.is_absolute():
        root = _project_root() / root
    run_name = spec.run_name or time.strftime("benchmark_%Y%m%d_%H%M%S")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _spec_to_dict(spec: BenchmarkSpec, *, budget: int) -> dict[str, Any]:
    data = dict(spec.__dict__)
    data["hash_ids"] = list(spec.hash_ids)
    data["dataset_zip"] = str(spec.dataset_zip)
    data["original_config_path"] = str(spec.original_config_path)
    data["optimized_config_path"] = str(spec.config_path or spec.optimized_config_path)
    data["config_path"] = str(spec.config_path) if spec.config_path is not None else None
    data["evolved_dir"] = str(spec.evolved_dir)
    data["output_root"] = str(spec.output_root)
    data["resolved_budget"] = int(budget)
    return data


def _rows_filename(row: dict[str, Any]) -> str:
    variant = row.get("config_variant") or "program"
    name = f"{row['hash_id']}__{row['candidate_type']}__{variant}__{row['candidate']}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return f"{safe}.json"


def _error_row(
    *,
    hash_id: str,
    candidate_type: str,
    candidate: str,
    error: str,
    config_variant: str = "",
    program_path: str = "",
) -> dict[str, Any]:
    return {
        "hash_id": hash_id,
        "candidate_type": candidate_type,
        "config_variant": config_variant,
        "candidate": candidate,
        "status": "error",
        "asr": float("nan"),
        "mean_l2_raw": float("nan"),
        "median_l2_raw": float("nan"),
        "mean_lpips": float("nan"),
        "mean_queries": float("nan"),
        "median_queries": float("nan"),
        "score": float("nan"),
        "target_metric": float("nan"),
        "n_success": 0,
        "n_pairs": 0,
        "mean_time_sec": float("nan"),
        "wall_sec": float("nan"),
        "backend": "",
        "workers": None,
        "program_path": program_path,
        "error": error,
    }


def _save_summary(result: BenchmarkResult) -> None:
    df = summary_dataframe(result.summary_rows)
    df.to_csv(result.summary_csv, index=False)
    _write_json(result.summary_json, result.summary_rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=True), encoding="utf-8")
