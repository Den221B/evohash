"""GigaEvolve integration adapter for EvoHash.

Framework-specific glue: problem-package generation (``context.py`` /
``validate.py`` / ``metrics.yaml``), the ``entrypoint(context)`` runtime
wrapper, Redis tooling, and the notebook-first launch API.

Shared utilities (config, LLM client, validation pipeline, score formula,
seed-program helpers, problem notes) live in ``evohash.stage3_common`` and
are re-used by other adapters such as ``stage3_openevolve``.
"""

from __future__ import annotations

# Re-export shared config types for convenience so existing notebooks that
# do ``from evohash.stage3_gigaevolve import EvaluationConfig`` keep working.
from evohash.stage3_common import (
    EvaluationConfig,
    LLMConfig,
    ProgramEvalSummary,
    load_eval_pairs,
    validate_program,
)

from .generator import GigaEvolveProblemConfig, generate_problem_package

__all__ = [
    "EvaluationConfig",
    "GigaEvolveProblemConfig",
    "LLMConfig",
    "ProgramEvalSummary",
    "generate_problem_package",
    "load_eval_pairs",
    "validate_program",
]
