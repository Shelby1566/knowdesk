"""LLM gateway wrapping LiteLLM completion and usage extraction."""

from __future__ import annotations

import json
import os
from typing import Any

from knowdesk.service.schemas import LIVE_OPENAI_TOOLS, _messages_for_litellm

_dashscope_http_client: Any = None


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "cached_tokens": getattr(usage, "cached_tokens", None),
        }
    details = data.get("prompt_tokens_details")
    if isinstance(details, dict) and data.get("cached_tokens") is None:
        data["cached_tokens"] = details.get("cached_tokens")
    return data


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in data and data[key] is not None:
            return int(data[key])
    return None


def prefer_direct_dashscope() -> None:
    """Reach dashscope.aliyuncs.com directly. A local HTTP proxy drops its TLS handshake."""
    host = "dashscope.aliyuncs.com"
    for name in ("NO_PROXY", "no_proxy"):
        parts = [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[name] = ",".join(parts)


def _parse_completion_response(response: Any, default_model: str) -> dict[str, Any]:
    choice = response.choices[0].message
    tool_calls = []
    for call in getattr(choice, "tool_calls", None) or []:
        raw_args = call.function.arguments
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw_args}
        else:
            parsed = raw_args or {}
        tool_calls.append({"id": call.id, "name": call.function.name, "arguments": parsed})
    usage = _usage_dict(getattr(response, "usage", None))
    if usage is not None:
        usage["model"] = getattr(response, "model", None) or default_model
    return {"content": choice.content or "", "tool_calls": tool_calls, "usage": usage}


def _build_completion_kwargs(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    model = os.getenv("KNOWDESK_LLM_MODEL", "dashscope/qwen-plus")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages_for_litellm(messages),
        "tools": LIVE_OPENAI_TOOLS,
    }
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return model, kwargs


def litellm_complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call the configured chat model synchronously."""
    import httpx
    import litellm

    global _dashscope_http_client
    prefer_direct_dashscope()
    if _dashscope_http_client is None:
        _dashscope_http_client = httpx.Client(trust_env=True, follow_redirects=True, timeout=120.0)
    litellm.client_session = _dashscope_http_client
    model, kwargs = _build_completion_kwargs(messages)
    response = litellm.completion(**kwargs)
    return _parse_completion_response(response, model)


async def litellm_acomplete(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call the configured chat model asynchronously."""
    import litellm

    prefer_direct_dashscope()
    model, kwargs = _build_completion_kwargs(messages)
    response = await litellm.acompletion(**kwargs)
    return _parse_completion_response(response, model)
