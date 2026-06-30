"""Compatibility patches for the installed OpenEvolve package.

OpenEvolve 0.2.x assumes every OpenAI-compatible chat response has
``choices[0].message.content``.  Some OpenRouter/reasoning providers can return
an error-like payload, empty choices, or a message with ``content=None``.  The
stock client then raises opaque errors such as ``'NoneType' object is not
subscriptable`` inside the evolution loop.

This patch keeps OpenEvolve's public behavior intact, but turns that failure
mode into an actionable error in ``runner.log``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any


def patch_openevolve_openai_llm() -> bool:
    """Patch ``openevolve.llm.openai.OpenAILLM._call_api`` once.

    Returns True if the patch was newly installed, False if it was already
    active or OpenEvolve is unavailable.
    """

    try:
        import openevolve.llm.openai as oe_openai  # type: ignore
    except Exception:  # noqa: BLE001 - openevolve is an optional dependency
        return False

    cls = oe_openai.OpenAILLM
    if getattr(cls, "_evohash_content_guard", False):
        return False

    async def _call_api_guarded(self: Any, params: dict[str, Any]) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        content = _extract_content_or_raise(response)

        logger = logging.getLogger("openevolve.llm.openai")
        logger.debug("API parameters: %s", _redact_params(params))
        logger.debug("API response content preview: %s", content[:500])
        return content

    cls._call_api = _call_api_guarded
    cls._evohash_content_guard = True
    return True


def _extract_content_or_raise(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(
            "LLM response has no choices. "
            f"Raw response preview: {_preview_response(response)}"
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    text = _content_to_text(content)
    if text:
        return text

    finish_reason = getattr(choice, "finish_reason", None)
    extra_keys = []
    if message is not None:
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra, dict):
            extra_keys = sorted(str(k) for k in extra)

    raise RuntimeError(
        "LLM returned empty message.content "
        f"(finish_reason={finish_reason!r}, message_extra_keys={extra_keys}). "
        "OpenEvolve needs normal text/code in message.content; reasoning-only "
        "or upstream-provider responses will fail. Use a non-reasoning instruct "
        "model, a paid/stable OpenRouter endpoint, or a local endpoint that "
        f"emits content. Raw response preview: {_preview_response(response)}"
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    chunks.append(value)
        return "\n".join(chunks).strip()
    return ""


def _preview_response(response: Any, limit: int = 1200) -> str:
    try:
        if hasattr(response, "model_dump"):
            data = response.model_dump(mode="json")
        elif hasattr(response, "dict"):
            data = response.dict()
        else:
            data = str(response)
        text = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - best-effort diagnostics
        text = repr(response)
    return text[:limit]


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(params)
    if "api_key" in redacted:
        redacted["api_key"] = "***"
    return redacted

