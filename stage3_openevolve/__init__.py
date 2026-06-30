"""OpenEvolve integration adapter for EvoHash.

OpenEvolve (``pip install openevolve``, AlphaEvolve-style local LLM evolution)
expects:

- An ``initial_program.py`` containing the function to mutate.
- An ``evaluator.py`` exposing ``def evaluate(program_path) -> dict[str, float]``.
- A ``config.yaml`` describing the LLM and evolution parameters.

This adapter generates all three from the same ``EvaluationConfig`` /
``LLMConfig`` that ``stage3_gigaevolve`` consumes, so the same seeds,
problem notes, and score formula are used by both frameworks.

Public API::

    from evohash.stage3_openevolve import (
        OpenEvolveProblemConfig,
        generate_openevolve_problem,
        run_openevolve,
    )
"""

from __future__ import annotations

# Re-export shared config types for symmetry with stage3_gigaevolve.
from evohash.stage3_common import (
    EvaluationConfig,
    LLMConfig,
    ProgramEvalSummary,
    load_eval_pairs,
    validate_program,
)

from .export import extract_best_programs, extract_champions, latest_checkpoint
from .generator import OpenEvolveProblemConfig, generate_openevolve_problem
from .notebook_api import (
    live_tail,
    preview_problem,
    read_checkpoint_metrics,
    run_openevolve,
    run_status,
    start_openevolve_background,
)
from .runtime import evaluate as evaluate_program  # noqa: F401  (test/debug entry point)
from .seeding import prepopulate_seed_database

__all__ = [
    "EvaluationConfig",
    "LLMConfig",
    "OpenEvolveProblemConfig",
    "ProgramEvalSummary",
    "evaluate_program",
    "extract_best_programs",
    "extract_champions",
    "generate_openevolve_problem",
    "latest_checkpoint",
    "live_tail",
    "load_eval_pairs",
    "prepopulate_seed_database",
    "preview_problem",
    "read_checkpoint_metrics",
    "run_openevolve",
    "run_status",
    "start_openevolve_background",
    "validate_program",
]
