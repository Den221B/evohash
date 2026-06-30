"""Lightweight fixup proxy between gigaevo and OpenRouter.

Some models (e.g. GPT-OSS-120b) return a bare JSON **list** when gigaevo
asks for structured output with a pydantic schema that expects an **object**.

This proxy:
  1. Forwards every request to OpenRouter unchanged.
  2. On the way back, inspects chat-completion responses that used
     ``response_format`` with a JSON schema.
  3. If the model returned ``[...]`` where an object was expected, wraps
     the list into ``{"<field>": [...]}`` using the schema to pick the
     right field name.
  4. If the response was truncated (finish_reason=length), retries with
     doubled max_tokens.
  5. If the structured output contains a ``code`` field with invalid Python
     (SyntaxError), retries the request (up to MAX_RETRIES times).
  6. Everything else passes through untouched.

Usage::

    python proxy/openrouter_proxy.py              # default :8100
    python proxy/openrouter_proxy.py --port 9000  # custom port

Then point your LLM config to ``base_url: http://localhost:8100/api/v1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from aiohttp import ClientSession, web

# ── Config ────────────────────────────────────────────────────────────────────

UPSTREAM = "https://openrouter.ai"
DEFAULT_PORT = 8100
MAX_RETRIES = 2  # up to 3 attempts total
MIN_RETRY_TEMPERATURE = 0.1  # floor for temperature on retry

log = logging.getLogger("openrouter-proxy")


# ── Schema helpers ────────────────────────────────────────────────────────────

def _find_list_field(schema: dict) -> str | None:
    """Return the name of the single root-level list field, or None.

    If the schema has exactly **one** property whose type is ``array``, that
    field is the obvious wrapper target (e.g. ``insights`` for
    ``ProgramInsights``).  If there are zero or multiple array fields we
    cannot safely guess, so we return None and skip the fix.
    """
    props = schema.get("properties", {})
    list_fields: list[str] = []
    for name, spec in props.items():
        # Direct "type": "array"
        if spec.get("type") == "array":
            list_fields.append(name)
            continue
        # anyOf / oneOf containing array (pydantic Optional[list[...]])
        for variant in spec.get("anyOf", []) + spec.get("oneOf", []):
            if variant.get("type") == "array":
                list_fields.append(name)
                break

    if len(list_fields) == 1:
        return list_fields[0]
    return None


def _extract_schema(request_body: dict) -> dict | None:
    """Pull the JSON schema from the request's ``response_format``, if any."""
    rf = request_body.get("response_format")
    if not rf or rf.get("type") != "json_schema":
        return None
    return rf.get("json_schema", {}).get("schema")


def _fix_content(content: str, schema: dict) -> str | None:
    """If *content* is a bare JSON list and the schema has one list field, wrap it.

    Returns the fixed string, or None if no fix was needed/possible.
    """
    stripped = content.strip()
    if not stripped.startswith("["):
        return None  # already an object (or not JSON)

    field = _find_list_field(schema)
    if field is None:
        return None  # can't determine wrapper field

    # Validate it's actually parseable JSON before wrapping
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None  # paranoia check

    wrapped = json.dumps({field: parsed}, ensure_ascii=False)
    log.info("Fixed bare list → wrapped into {%s: [...]}", field)
    return wrapped


# ── Response analysis ─────────────────────────────────────────────────────────

def _is_truncated(data: dict) -> bool:
    """Check if any choice has finish_reason=length (truncated output)."""
    for choice in data.get("choices", []):
        if choice.get("finish_reason") == "length":
            return True
    return False


def _has_code_syntax_error(data: dict) -> bool:
    """Check if structured output has a ``code`` field with invalid Python."""
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        code = parsed.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            compile(code, "<proxy-check>", "exec")
        except SyntaxError:
            return True
    return False


# ── Retry logic ───────────────────────────────────────────────────────────────

def _should_retry(resp_data: dict, schema: dict | None) -> str | None:
    """Return a retry reason string, or None if no retry needed."""
    if _is_truncated(resp_data):
        return "truncated"
    # Only check for code syntax errors in structured-output requests
    if schema is not None and _has_code_syntax_error(resp_data):
        return "syntax_error"
    return None


def _adjust_request_for_retry(request_body: dict, reason: str) -> dict:
    """Return a modified copy of the request body for the retry attempt."""
    body = json.loads(json.dumps(request_body))  # deep copy

    if reason == "truncated":
        old_max = body.get("max_tokens") or body.get("max_completion_tokens") or 81920
        body["max_tokens"] = old_max * 2
        # Also set max_completion_tokens for newer API format
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = old_max * 2
        log.info("Retry: doubling max_tokens %d → %d", old_max, old_max * 2)

    if reason == "syntax_error":
        temp = body.get("temperature", 0)
        if temp < MIN_RETRY_TEMPERATURE:
            body["temperature"] = MIN_RETRY_TEMPERATURE
            log.info("Retry: bumping temperature %s → %s", temp, MIN_RETRY_TEMPERATURE)

    return body


# ── Proxy handler ─────────────────────────────────────────────────────────────

async def _send_upstream(
    session: ClientSession,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
) -> tuple[int, dict, bytes]:
    """Send a request to upstream and return (status, headers, body)."""
    async with session.request(method, url, headers=headers, data=body) as resp:
        resp_body = await resp.read()
        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        }
        return resp.status, resp_headers, resp_body


async def _proxy_handler(request: web.Request) -> web.Response:
    """Forward request to OpenRouter, optionally fix the response."""
    session: ClientSession = request.app["session"]

    upstream_url = UPSTREAM + request.path_qs
    body = await request.read()

    # Build headers, forwarding auth etc.
    headers = {}
    for hdr in ("Authorization", "Content-Type", "HTTP-Referer",
                "X-Title", "Accept", "Accept-Encoding"):
        val = request.headers.get(hdr)
        if val:
            headers[hdr] = val

    # Parse request body to check for response_format (only for POST)
    schema = None
    request_body = None
    if request.method == "POST" and body:
        try:
            request_body = json.loads(body)
            schema = _extract_schema(request_body)
        except json.JSONDecodeError:
            pass

    is_chat = request.path.rstrip("/").endswith("/chat/completions")

    # Send initial request
    status, resp_headers, resp_body = await _send_upstream(
        session, request.method, upstream_url, headers, body,
    )

    # Retry loop for chat completions
    if is_chat and status == 200 and request_body is not None:
        resp_body = _log_and_fix_response(resp_body, schema)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp_data = json.loads(resp_body)
            except json.JSONDecodeError:
                break

            reason = _should_retry(resp_data, schema)
            if reason is None:
                break

            adjusted = _adjust_request_for_retry(request_body, reason)
            adjusted_body = json.dumps(adjusted, ensure_ascii=False).encode()

            log.warning(
                "Retry %d/%d reason=%s model=%s",
                attempt, MAX_RETRIES, reason, resp_data.get("model", "?"),
            )

            status, resp_headers, resp_body = await _send_upstream(
                session, request.method, upstream_url, headers, adjusted_body,
            )

            if status == 200:
                resp_body = _log_and_fix_response(resp_body, schema)
            else:
                log.warning("Retry %d returned status %d, keeping previous response", attempt, status)
                break
    elif is_chat and status == 200:
        resp_body = _log_and_fix_response(resp_body, schema)

    return web.Response(
        status=status,
        headers=resp_headers,
        body=resp_body,
    )


def _log_and_fix_response(resp_body: bytes, schema: dict | None) -> bytes:
    """Log finish_reason / usage, then fix structured-output content if needed."""
    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError:
        return resp_body

    # Log model, finish_reason, token usage for every chat completion
    model = data.get("model", "?")
    usage = data.get("usage", {})
    choices = data.get("choices", [])

    for choice in choices:
        reason = choice.get("finish_reason", "?")
        if reason == "length":
            log.warning(
                "TRUNCATED response (finish_reason=length) model=%s "
                "prompt_tokens=%s completion_tokens=%s",
                model,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
            )
        else:
            log.debug(
                "model=%s finish_reason=%s completion_tokens=%s",
                model, reason, usage.get("completion_tokens", "?"),
            )

    # Fix structured-output only if a schema was in the request
    if schema is None:
        return resp_body

    modified = False
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content")
        if not content:
            continue

        fixed = _fix_content(content, schema)
        if fixed is not None:
            msg["content"] = fixed
            modified = True

    if modified:
        return json.dumps(data, ensure_ascii=False).encode()
    return resp_body


# ── App setup ─────────────────────────────────────────────────────────────────

async def _on_startup(app: web.Application) -> None:
    app["session"] = ClientSession()
    log.info("Proxy started → upstream %s", UPSTREAM)


async def _on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    # Catch-all route — forward everything
    app.router.add_route("*", "/{path:.*}", _proxy_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter fixup proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    web.run_app(create_app(), port=args.port, print=lambda msg: log.info(msg))


if __name__ == "__main__":
    main()
