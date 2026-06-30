"""Small OpenAI-compatible LLM client for local endpoints and OpenRouter."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from .config import LLMConfig


OPENROUTER_MODELS: dict[str, str] = {
    "gpt_oss_120b": "openai/gpt-oss-120b:free",
    "gpt_oss_120b_free": "openai/gpt-oss-120b:free",
    "gpt_oss_120b_paid": "openai/gpt-oss-120b",
    "glm4_air": "z-ai/glm-4.5-air:free",
    "qwen35_35b": "qwen/qwen3.5-35b-a3b",
    "qwen35_122b": "qwen/qwen3.5-122b-a10b",
    "mistral_small": "mistralai/mistral-small-2603",
}


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unusable response."""


def normalize_llm_config(config: LLMConfig) -> LLMConfig:
    """Fill source-specific defaults without mutating the user's config."""
    if config.source == "none":
        return config

    if config.source == "openrouter":
        model = OPENROUTER_MODELS.get(config.model, config.model)
        api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Pass LLMConfig(api_key=...) "
                "or set the environment variable."
            )
        base_url = config.base_url
        if not base_url or base_url == "http://localhost:8000/v1":
            base_url = "https://openrouter.ai/api/v1"
        return replace(
            config,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    if config.source == "local":
        if not config.base_url:
            raise LLMError("Local LLM source requires base_url.")
        if not config.model:
            raise LLMError("Local LLM source requires model.")
        return config

    raise LLMError(f"Unsupported LLM source: {config.source!r}")


def chat_completion(
    *,
    config: LLMConfig,
    messages: list[dict[str, str]],
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint and return text."""
    cfg = normalize_llm_config(config)
    if cfg.source == "none":
        raise LLMError("LLM source is 'none'.")

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(cfg.temperature),
        "max_tokens": int(cfg.max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.source == "openrouter":
        headers.setdefault("HTTP-Referer", "https://localhost/evohash")
        headers.setdefault("X-Title", "EvoHash Stage3 Evolution")

    request = urllib.request.Request(
        _chat_url(cfg.base_url),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(cfg.timeout_s)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM HTTP {exc.code}: {text[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc

    try:
        data: dict[str, Any] = json.loads(raw)
        choices = data.get("choices") or []
        message = choices[0]["message"]
        content = message.get("content") or ""
    except Exception as exc:
        raise LLMError(f"Malformed LLM response: {raw[:1000]}") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError(f"Empty LLM response: {raw[:1000]}")
    return content


def extract_python_code(text: str) -> str:
    """Extract Python source from a model response."""
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()
    return text.strip()


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
