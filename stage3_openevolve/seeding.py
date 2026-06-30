"""Pre-populate an OpenEvolve ``ProgramDatabase`` with EvoHash seed attacks.

OpenEvolve only takes a single ``initial_program.py``.  GigaEvolve, by
contrast, loads a whole ``initial_programs/`` directory and scores every seed
before evolution begins.  To keep the two frameworks comparable we evaluate the
*other* seed attacks here (via the **same** ``validate_program`` pipeline both
adapters use), wrap each as an OpenEvolve ``Program`` with its real metrics, and
save them into a database directory.  ``database.db_path`` in the generated
``config.yaml`` points at that directory, so the controller auto-loads the pool
on startup — the seeds are in the population from iteration 0 and the
per-mutation ``system_message`` stays clean.

This module imports ``openevolve`` lazily; static problem-package generation
does not need it, only an actual run does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evohash.stage3_common.config import EvaluationConfig
from evohash.stage3_common.validate import validate_program

from .generator import OpenEvolveProblemConfig


def prepopulate_seed_database(
    *,
    problem_dir: Path,
    config: OpenEvolveProblemConfig,
    evaluation: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate the queued seed programs and save them as an OpenEvolve DB.

    Reads the seed files written by ``generate_openevolve_problem`` into
    ``<problem_dir>/seeds/``, scores each with ``validate_program``, inserts
    them into a fresh ``ProgramDatabase`` (round-robin across islands), and
    saves to ``<problem_dir>/<config.seed_db_dirname>``.

    Returns a small report dict (easy to print in a notebook).  A no-op (empty
    report) when pre-population is disabled or there are no extra seeds.
    """
    report: dict[str, Any] = {
        "prepopulated": False,
        "db_path": None,
        "seeds": [],
        "skipped": [],
    }
    if not config.prepopulate_database:
        report["reason"] = "prepopulate_database=False"
        return report

    seeds_dir = problem_dir / "seeds"
    seed_files = sorted(seeds_dir.glob("*.py")) if seeds_dir.is_dir() else []
    if not seed_files:
        report["reason"] = "no seed files (single-attack pool or generation skipped seeds)"
        return report

    # Lazy import: only needed for an actual run.
    from openevolve.config import DatabaseConfig
    from openevolve.database import Program, ProgramDatabase

    db_path = (problem_dir / config.seed_db_dirname).resolve()
    db_config = _database_config(DatabaseConfig, config)
    db = ProgramDatabase(db_config)

    num_islands = max(1, int(config.num_islands))
    for i, seed_file in enumerate(seed_files):
        attack_id = seed_file.stem
        code = seed_file.read_text(encoding="utf-8")
        try:
            summary = validate_program(seed_file, evaluation)
            metrics = {
                key: float(value)
                for key, value in summary.metrics.items()
                if _is_float(value)
            }
            if "score" in metrics:
                # OpenEvolve maximizes combined_score.  EvoHash's objective is
                # lower-is-better, so all preloaded seeds must use the same
                # sign convention as stage3_openevolve.runtime.evaluate().
                metrics["combined_score"] = -float(metrics["score"])
        except Exception as exc:  # noqa: BLE001 - one bad seed must not abort the pool
            report["skipped"].append({"attack_id": attack_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        program = Program(
            id=f"seed_{attack_id}",
            code=code,
            generation=0,
            metrics=metrics,
            metadata={"source": "evohash_seed", "attack_id": attack_id},
        )
        db.add(program, iteration=0, target_island=i % num_islands)
        if summary.diagnostics.strip():
            db.store_artifacts(
                program.id,
                {"evohash_feedback.txt": summary.diagnostics},
            )
        report["seeds"].append({"attack_id": attack_id, "metrics": metrics, "island": i % num_islands})

    if not report["seeds"]:
        report["reason"] = "all seeds failed evaluation"
        return report

    db.save(str(db_path), iteration=0)
    report["prepopulated"] = True
    report["db_path"] = str(db_path)
    return report


def _database_config(database_config_cls: Any, config: OpenEvolveProblemConfig) -> Any:
    """Build a ``DatabaseConfig`` matching the generated config.yaml.

    We pass only fields the installed OpenEvolve actually defines so the helper
    keeps working across releases.  ``db_path`` is left unset here: we save to
    an explicit path, and an empty ``db_path`` avoids auto-loading a (possibly
    stale) directory while we build a fresh pool.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(database_config_cls)}
    desired: dict[str, Any] = {
        "population_size": int(config.population_size),
        "archive_size": int(config.archive_size),
        "num_islands": int(config.num_islands),
        "migration_interval": int(config.migration_interval),
        "migration_rate": float(config.migration_rate),
        "elite_selection_ratio": float(config.elite_selection_ratio),
        "exploration_ratio": float(config.exploration_ratio),
        "exploitation_ratio": float(config.exploitation_ratio),
        "feature_dimensions": list(config.feature_dimensions),
        # int or per-dimension dict — DatabaseConfig.feature_bins accepts both.
        "feature_bins": dict(config.feature_bins) if isinstance(config.feature_bins, dict) else int(config.feature_bins),
        "diversity_reference_size": int(config.diversity_reference_size),
        "random_seed": int(config.random_seed),
    }
    return database_config_cls(**{k: v for k, v in desired.items() if k in known})


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
