"""Shared LLM model aliases for the GigaEvolve adapter."""

from __future__ import annotations

from evohash.stage3_common.llm import OPENROUTER_MODELS


DEFAULT_OPENROUTER_MODEL = "gpt_oss_120b"


def resolve_openrouter_model(model: str | None) -> str:
    """Resolve a short OpenRouter alias to a provider/model id."""
    key = (model or DEFAULT_OPENROUTER_MODEL).strip()
    if not key:
        key = DEFAULT_OPENROUTER_MODEL
    return OPENROUTER_MODELS.get(key, key)


def openrouter_aliases() -> dict[str, str]:
    """Return a copy of the known OpenRouter alias table."""
    return dict(OPENROUTER_MODELS)
