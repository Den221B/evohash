"""Preflight check for local and OpenRouter LLM endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import urllib.error
import urllib.request

from evohash.stage3_common.config import LLMConfig
from evohash.stage3_common.llm import LLMError, chat_completion, normalize_llm_config
from evohash.stage3_gigaevolve.models import DEFAULT_OPENROUTER_MODEL, resolve_openrouter_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check that an OpenAI-compatible LLM responds.")
    parser.add_argument("--source", choices=["openrouter", "local"], default="openrouter")
    parser.add_argument("--model", default="", help="Model id or OpenRouter alias.")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--local-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dotenv", default=".env", help="Optional .env file to load before the check.")
    parser.add_argument("--diagnostic", action="store_true", help="Also check raw HTTP /models and /chat endpoints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = check_llm(
        source=args.source,
        model=args.model,
        openrouter_base_url=args.openrouter_base_url,
        local_base_url=args.local_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        dotenv=args.dotenv,
        diagnostic=args.diagnostic,
    )
    _print(payload)
    raise SystemExit(0 if payload["ok"] else 1)


def check_llm(
    *,
    source: str = "openrouter",
    model: str = "",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    local_base_url: str = "http://localhost:8000/v1",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    dotenv: str | Path = ".env",
    diagnostic: bool = False,
) -> dict:
    """Return a notebook-friendly LLM connectivity status."""
    _load_dotenv(Path(dotenv))

    model = model.strip()
    if source == "openrouter":
        model = model or DEFAULT_OPENROUTER_MODEL
    elif not model:
        return {
            "ok": False,
            "source": source,
            "error": "Local LLM requires a model name.",
            "recommendations": ["Set LOCAL_MODEL in the notebook or pass model='...'."],
        }

    base_url = openrouter_base_url if source == "openrouter" else local_base_url
    resolved_model = resolve_openrouter_model(model) if source == "openrouter" else model
    api_key = api_key or (
        os.environ.get("OPENROUTER_API_KEY") if source == "openrouter" else os.environ.get("OPENAI_API_KEY")
    )

    diagnostics = None
    if diagnostic:
        diagnostics = _diagnose_http(
            source=source,
            base_url=base_url,
            model=resolved_model,
            api_key=api_key,
            timeout=timeout,
        )

    config = LLMConfig(
        source=source,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout,
    )
    try:
        normalized = normalize_llm_config(config)
        response = chat_completion(
            config=config,
            messages=[
                {"role": "system", "content": "Reply with exactly: EVOHASH_LLM_OK"},
                {"role": "user", "content": "ping"},
            ],
        )
    except (LLMError, Exception) as exc:
        return {
            "ok": False,
            "source": source,
            "model": resolved_model,
            "base_url": base_url,
            "api_key_seen": bool(api_key),
            "api_key_prefix": _mask_key(api_key),
            "diagnostics": diagnostics,
            "error": f"{type(exc).__name__}: {exc}",
            "recommendations": _llm_recommendations(source, api_key, diagnostics, exc),
        }

    return {
            "ok": True,
            "source": source,
            "model": normalized.model,
            "base_url": normalized.base_url,
            "api_key_seen": bool(api_key),
            "api_key_prefix": _mask_key(api_key),
            "diagnostics": diagnostics,
            "response": response.strip()[:500],
            "recommendations": _llm_recommendations(source, api_key, diagnostics, None),
    }


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


def _print(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()


def _diagnose_http(
    *,
    source: str,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
) -> dict:
    diagnostics: dict = {
        "base_url": base_url,
        "model": model,
        "api_key_seen": bool(api_key),
        "api_key_prefix": _mask_key(api_key),
    }
    headers = _headers(source=source, api_key=api_key)
    diagnostics["models"] = _http_json(
        "GET",
        _url(base_url, "models"),
        headers=headers,
        timeout=timeout,
    )
    diagnostics["chat"] = _http_json(
        "POST",
        _url(base_url, "chat/completions"),
        headers=headers,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly EVOHASH_LLM_OK"}],
            "temperature": 0,
            "max_tokens": 1024,
        },
        timeout=timeout,
    )
    return diagnostics


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    payload: dict | None = None,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(text) if text else {}
            return {
                "ok": True,
                "status": response.status,
                "url": url,
                "keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
                "sample": _small_sample(parsed),
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "url": url, "error": text[:1200]}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def _headers(*, source: str, api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if source == "openrouter":
        headers["HTTP-Referer"] = "https://localhost/evohash"
        headers["X-Title"] = "EvoHash Stage3 Evolution"
    return headers


def _url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def _mask_key(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if len(value) <= 10:
        return value[:3] + "***"
    return value[:7] + "..." + value[-4:]


def _small_sample(value: object) -> object:
    if isinstance(value, dict):
        if "data" in value and isinstance(value["data"], list):
            return {"data_len": len(value["data"]), "first": value["data"][:1]}
        if "choices" in value:
            return {"choices": value["choices"][:1]}
        return {k: value[k] for k in list(value)[:5]}
    if isinstance(value, list):
        return value[:1]
    return str(value)[:500]


def _llm_recommendations(
    source: str,
    api_key: str | None,
    diagnostics: dict | None,
    exc: BaseException | None,
) -> list[str]:
    recommendations: list[str] = []
    if source == "openrouter" and not api_key:
        recommendations.append("Set OPENROUTER_API_KEY in .env or pass api_key=...")
    if source == "local" and not api_key:
        recommendations.append("OPENAI_API_KEY is optional for many local servers; set it only if your endpoint requires auth.")
    if diagnostics:
        chat = diagnostics.get("chat") or {}
        models = diagnostics.get("models") or {}
        if not models.get("ok"):
            recommendations.append("OpenRouter /models failed: check network, base_url, proxy/VPN/firewall.")
        if chat.get("status") == 429:
            recommendations.append("OpenRouter returned 429 for chat. Free models are rate-limited; retry later, reduce diagnostic calls, or use another model/BYOK.")
        elif chat and not chat.get("ok"):
            recommendations.append("Chat endpoint failed; inspect diagnostics.chat.error.")
    if exc is not None and "OPENROUTER_API_KEY" in str(exc):
        recommendations.append("Create .env in project root with OPENROUTER_API_KEY=...")
    return recommendations


if __name__ == "__main__":
    main()
