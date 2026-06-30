"""Subprocess CLI entrypoint for running EvoHash through OpenEvolve.

Invoked by the notebook launcher (``notebook_api.start_openevolve_background``) as::

    python -m evohash.stage3_openevolve.run --spec <spec.json>

Why a subprocess (not in-process)?  OpenEvolve calls ``asyncio.run()`` internally
(clashes with a Jupyter event loop), uses process-based parallel evaluation
(fork/spawn issues inside notebooks), and reconfigures logging.  A dedicated
process sidesteps all three and gives one stable entrypoint.  Mirrors
``stage3_gigaevolve/run.py``.

Outputs under ``<output_dir>``::

    runner.log                      this script + captured openevolve logs
    problem/<name>/                 generated package (initial/evaluator/config/seeds)
    logs/openevolve_*.log           openevolve internal log
    checkpoints/checkpoint_<N>/     full DB dumps (programs/, metadata.json, best_*)
    best_code.py / best_metrics.json  convenience copy of the single best
    best_programs/                  top-N + per-cell elites (code + metrics + index.json)

The run spec (JSON) has three sections — ``evaluation``, ``llm``, ``problem`` —
plus top-level ``run_name`` / ``output_dir`` / ``top_n`` / ``dotenv``.  CLI flags
override matching spec keys, so the script is also usable standalone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from evohash.stage3_common.config import EvaluationConfig, LLMConfig
from evohash.stage3_common.llm import normalize_llm_config
from evohash.stage3_openevolve.generator import (
    OpenEvolveProblemConfig,
    generate_openevolve_problem,
)
from evohash.stage3_openevolve.llm_compat import patch_openevolve_openai_llm

# Install at import time as well as in main(): multiprocessing "spawn" workers
# import the main module, and they need the same guarded LLM client.
patch_openevolve_openai_llm()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an EvoHash problem through OpenEvolve.")
    p.add_argument("--spec", default=None, help="Path to a run-spec JSON (written by the notebook).")
    # Direct overrides (used when --spec is absent, or to override spec values).
    p.add_argument("--hash", dest="hash_id", default=None,
                   choices=["phash", "pdq", "neuralhash", "photodna"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed-config", default=None)
    p.add_argument("--seed-attacks", default=None, help="Comma-separated, or 'all'.")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--llm-source", default=None, choices=["openrouter", "local", "none"])
    p.add_argument("--model", default=None, help="Model id or OpenRouter alias.")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--dotenv", default=".env")
    p.add_argument("--dry-run", action="store_true", help="Generate the package and exit.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    spec = _load_spec(args)
    _apply_cli_overrides(spec, args)

    evaluation = spec.get("evaluation") or {}
    hash_id = evaluation.get("hash_id", "phash")
    run_name = spec.get("run_name") or f"oe_{hash_id}_{int(time.time())}"
    output_dir = Path(spec.get("output_dir") or (Path("outputs/openevolve_runs") / run_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _configure_logging(output_dir)
    log.info("=" * 70)
    log.info("EvoHash x OpenEvolve run starting")
    log.info("  run_name  : %s", run_name)
    log.info("  output_dir: %s", output_dir)

    _load_dotenv(Path(spec.get("dotenv", ".env")))

    try:
        eval_cfg, llm_cfg, problem_cfg = build_configs_from_spec(
            spec, default_output_dir=str(output_dir / "problem")
        )
    except Exception as exc:  # noqa: BLE001 - surface config errors to runner.log
        log.critical("Failed to build run configuration: %s", exc, exc_info=True)
        raise

    log.info("  hash_id   : %s", eval_cfg.hash_id)
    log.info("  dataset   : %s", eval_cfg.dataset_zip)
    log.info("  max_pairs : %s  budget: %s", eval_cfg.max_pairs, eval_cfg.budget)
    log.info("  iterations: %s  islands: %s  pop: %s",
             problem_cfg.iterations, problem_cfg.num_islands, problem_cfg.population_size)
    log.info("  feature_dims: %s", list(problem_cfg.feature_dimensions))
    log.info("  llm       : %s | %s", llm_cfg.source, llm_cfg.model)

    _propagate_api_key(llm_cfg)

    problem_dir = generate_openevolve_problem(config=problem_cfg, llm=llm_cfg, evaluation=eval_cfg)
    log.info("Problem package: %s", problem_dir)

    if args.dry_run:
        log.info("Dry run requested - package generated, skipping evolution.")
        print(str(problem_dir))
        return

    # Pre-populate the ProgramDatabase with the (evaluated) extra seeds.
    from evohash.stage3_openevolve.seeding import prepopulate_seed_database

    seed_report = prepopulate_seed_database(
        problem_dir=problem_dir, config=problem_cfg, evaluation=eval_cfg
    )
    log.info("Seed prepopulation: %s", json.dumps(seed_report)[:800])

    # Export evaluation context so OpenEvolve's (parallel) evaluator workers can
    # rebuild EvaluationConfig from the environment.
    from evohash.stage3_openevolve.runtime import ENV_TRACE_DIR, export_eval_env

    trace_dir = output_dir / "evaluation_trace"
    eval_env = export_eval_env(eval_cfg)
    eval_env[ENV_TRACE_DIR] = str(trace_dir)
    os.environ.update(eval_env)
    log.info("Evaluation trace: %s", trace_dir)

    # Heavy import last, after logging is configured so its loggers attach.
    patched = patch_openevolve_openai_llm()
    log.info("OpenEvolve LLM content guard: %s", "installed" if patched else "already active")
    import openevolve

    log.info("Starting OpenEvolve (run_evolution)...")
    t_start = time.time()
    result = openevolve.run_evolution(
        initial_program=str(problem_dir / "initial_program.py"),
        evaluator=str(problem_dir / "evaluator.py"),
        config=str(problem_dir / "config.yaml"),
        iterations=int(problem_cfg.iterations),
        output_dir=str(output_dir),
        cleanup=False,
    )
    elapsed = time.time() - t_start
    log.info("Evolution finished in %.1fs", elapsed)
    result_metrics = dict(getattr(result, "metrics", None) or {})
    if "combined_score" not in result_metrics and "score" in result_metrics:
        inferred = -float(result_metrics["score"])
        log.warning(
            "Best metrics are missing combined_score; inferred combined_score=%s from score=%s. "
            "This usually means the best program came from an old prepopulated seed DB.",
            inferred,
            result_metrics["score"],
        )
    log.info("Best score (combined): %s", getattr(result, "best_score", None))
    log.info("Best metrics: %s", result_metrics or getattr(result, "metrics", None))

    _save_single_best(output_dir, result, log)

    # Primary export: the record-improvement champion chain (best seed -> each
    # program that strictly beats the running best). One folder of py+json +
    # aggregate.json for cross-program / cross-framework comparison.
    from evohash.stage3_openevolve.export import extract_champions

    agg = extract_champions(output_dir)
    champions_dir = str(Path(output_dir) / "champions")
    if agg.get("ok"):
        log.info(
            "Saved %s champions (from %s programs) to %s; total improvement (fitness)=%s",
            agg.get("n_champions"), agg.get("n_programs_total"), champions_dir,
            agg.get("total_improvement_fitness"),
        )
    else:
        log.warning("extract_champions: %s", agg.get("reason"))

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "champions_dir": champions_dir,
            "n_champions": agg.get("n_champions"),
            "best_metrics": result_metrics or getattr(result, "metrics", None),
        },
        indent=2,
        default=str,
    ))


# ---------------------------------------------------------------------------
# Spec loading + config builders
# ---------------------------------------------------------------------------

def _load_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec:
        path = Path(args.spec)
        if not path.exists():
            raise SystemExit(f"--spec file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _apply_cli_overrides(spec: dict[str, Any], args: argparse.Namespace) -> None:
    ev = spec.setdefault("evaluation", {})
    lm = spec.setdefault("llm", {})
    pr = spec.setdefault("problem", {})
    if args.hash_id is not None:
        ev["hash_id"] = args.hash_id
    if args.dataset is not None:
        ev["dataset_zip"] = args.dataset
    if args.split is not None:
        ev["split"] = args.split
    if args.max_pairs is not None:
        ev["max_pairs"] = args.max_pairs
    if args.budget is not None:
        ev["budget"] = args.budget
    if args.resize_size is not None:
        ev["resize_size"] = args.resize_size
    if args.iterations is not None:
        pr["iterations"] = args.iterations
    if args.seed_config is not None:
        ev["attack_config_path"] = args.seed_config
        pr["seed_config_path"] = args.seed_config
    if args.seed_attacks is not None:
        pr["seed_attacks"] = [s.strip() for s in args.seed_attacks.split(",") if s.strip()]
    if args.llm_source is not None:
        lm["source"] = args.llm_source
    if args.model is not None:
        lm["model"] = args.model
    if args.llm_base_url is not None:
        lm["base_url"] = args.llm_base_url
    if args.temperature is not None:
        lm["temperature"] = args.temperature
    if args.max_tokens is not None:
        lm["max_tokens"] = args.max_tokens
    if args.output_dir is not None:
        spec["output_dir"] = args.output_dir
    if args.run_name is not None:
        spec["run_name"] = args.run_name
    if args.top_n is not None:
        spec["top_n"] = args.top_n
    if args.dotenv:
        spec.setdefault("dotenv", args.dotenv)


def build_configs_from_spec(
    spec: dict[str, Any], *, default_output_dir: str = "outputs/openevolve_problems"
) -> tuple[EvaluationConfig, LLMConfig, OpenEvolveProblemConfig]:
    """Build the three config objects from a run-spec dict.

    Shared by ``main()`` (subprocess) and the notebook's dry-run preview so both
    paths derive identical configs from one spec.
    """
    eval_cfg = _build_evaluation(spec.get("evaluation") or {})
    llm_cfg = normalize_llm_config(_build_llm(spec.get("llm") or {}))
    problem_cfg = _build_problem(
        spec.get("problem") or {},
        hash_id=eval_cfg.hash_id,
        default_output_dir=default_output_dir,
    )
    return eval_cfg, llm_cfg, problem_cfg


def _filtered_kwargs(cls: type, data: dict[str, Any] | None) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in valid}


def _build_evaluation(data: dict[str, Any]) -> EvaluationConfig:
    kw = _filtered_kwargs(EvaluationConfig, data)
    if "hash_id" not in kw:
        raise SystemExit("spec.evaluation.hash_id is required")
    if "dataset_zip" not in kw:
        raise SystemExit("spec.evaluation.dataset_zip is required")
    # Resolve to absolute so OpenEvolve's worker subprocesses (which may run with
    # a different cwd) can find the dataset/seed config via EVOHASH_OE_* env.
    kw["dataset_zip"] = str(Path(kw["dataset_zip"]).expanduser().resolve())
    if "attack_config_path" in kw:
        kw["attack_config_path"] = str(Path(kw["attack_config_path"]).expanduser().resolve())
    return EvaluationConfig(**kw)


def _build_llm(data: dict[str, Any]) -> LLMConfig:
    return LLMConfig(**_filtered_kwargs(LLMConfig, data))


_TUPLE_FIELDS = {"seed_attacks", "feature_dimensions", "cascade_thresholds"}


def _build_problem(data: dict[str, Any], *, hash_id: str, default_output_dir: str) -> OpenEvolveProblemConfig:
    data = dict(data or {})
    # Notebook-friendly alias: problem_output_dir -> output_dir.
    if "problem_output_dir" in data and "output_dir" not in data:
        data["output_dir"] = data.pop("problem_output_dir")
    data.setdefault("output_dir", default_output_dir)
    data.setdefault("hash_id", hash_id)
    if "seed_config_path" in data:
        data["seed_config_path"] = str(Path(data["seed_config_path"]).expanduser().resolve())
    kw = _filtered_kwargs(OpenEvolveProblemConfig, data)
    for field_name in _TUPLE_FIELDS:
        if field_name in kw and isinstance(kw[field_name], list):
            kw[field_name] = tuple(kw[field_name])
    return OpenEvolveProblemConfig(**kw)


# ---------------------------------------------------------------------------
# Side helpers
# ---------------------------------------------------------------------------

def _configure_logging(output_dir: Path) -> logging.Logger:
    runner_log = output_dir / "runner.log"
    # OpenEvolve logs unicode (e.g. a ✅ emoji). On a Windows console defaulting to
    # cp1251/cp1252 this raises UnicodeEncodeError inside logging. Force UTF-8 on
    # the std streams so neither our handler nor openevolve's own one chokes.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(runner_log, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.handlers = [fh, ch]  # replace any inherited handlers (idempotent)
    return logging.getLogger("evohash.openevolve.run")


def _propagate_api_key(llm_cfg: LLMConfig) -> None:
    """Make the API key available to OpenEvolve's OpenAI-compatible client.

    The generated config.yaml references ``${OPENROUTER_API_KEY}`` / no key for
    local; we also set ``OPENAI_API_KEY`` because some openevolve code paths read
    it directly. A dummy is used for local endpoints that ignore the key.
    """
    if llm_cfg.source == "openrouter":
        key = llm_cfg.api_key or os.environ.get("OPENROUTER_API_KEY")
        if key:
            os.environ.setdefault("OPENROUTER_API_KEY", key)
            os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_API_KEY", "sk-local-dummy")


def _save_single_best(output_dir: Path, result: Any, log: logging.Logger) -> None:
    try:
        best_code = getattr(result, "best_code", "") or ""
        if best_code:
            (output_dir / "best_code.py").write_text(best_code, encoding="utf-8")
        metrics = dict(getattr(result, "metrics", None) or {})
        (output_dir / "best_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 - convenience copy, never fatal
        log.warning("Could not save best_code/best_metrics: %s", exc)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and _is_unset(os.environ.get(key)):
            os.environ[key] = value


def _is_unset(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip().lower() == "none"


if __name__ == "__main__":
    main()
