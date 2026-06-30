"""Shared utilities for stage3 evolution adapters (GigaEvolve, OpenEvolve, ...).

This package contains everything that is *framework-agnostic*:

- ``config``        — ``EvaluationConfig`` and ``LLMConfig`` dataclasses
- ``llm``           — OpenAI-compatible LLM client + OpenRouter model aliases
- ``metrics``       — ``PRIMARY_METRIC`` (score formula spec) + helpers
- ``validate``      — ``validate_program`` and ``summarize_row_dicts`` (the
                     production evaluation pipeline used by every adapter)
- ``programs``      — seed program generation + ``resolve_seed_attacks``
- ``problem_notes`` — per-hash research notes loaded into prompts

Framework adapters (``stage3_gigaevolve``, ``stage3_openevolve``) import from
here.  They only contain code that is genuinely specific to a single framework
(file formats, runtime contracts, CLIs).
"""

from __future__ import annotations

from .config import EvaluationConfig, LLMConfig
from .metrics import PRIMARY_METRIC, PROMPT_METRICS, objective_value
from .validate import ProgramEvalSummary, load_eval_pairs, summarize_row_dicts, validate_program
from .programs import copy_candidate, ensure_seed_programs, read_code, resolve_seed_attacks
from .problem_notes import NOTES_DIR, load_problem_notes
from .llm import (
    OPENROUTER_MODELS,
    LLMError,
    chat_completion,
    extract_python_code,
    normalize_llm_config,
)

__all__ = [
    "EvaluationConfig",
    "LLMConfig",
    "LLMError",
    "NOTES_DIR",
    "OPENROUTER_MODELS",
    "PRIMARY_METRIC",
    "PROMPT_METRICS",
    "ProgramEvalSummary",
    "chat_completion",
    "copy_candidate",
    "ensure_seed_programs",
    "extract_python_code",
    "load_eval_pairs",
    "load_problem_notes",
    "normalize_llm_config",
    "objective_value",
    "read_code",
    "resolve_seed_attacks",
    "summarize_row_dicts",
    "validate_program",
]
