"""Request models and OpenAI/DashScope function schemas for KnowDesk HTTP service."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class MessageRequest(BaseModel):
    content: str


class ApprovalRequest(BaseModel):
    decision: str = "approve"
    reason: str = ""
    modified_args: dict[str, Any] | None = None
    idempotency_key: str


class JobRequest(BaseModel):
    job_type: str = "regression_eval"
    params: dict[str, Any] = {}


def _string_param(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _function_schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Names match tau2 KnowledgeTools. Discoverable writes stay behind the dispatcher.
LIVE_OPENAI_TOOLS: list[dict[str, Any]] = [
    _function_schema(
        "KB_search",
        "Search banking policy sections.",
        {
            "query": _string_param("Policy question."),
            "top_k": {"type": "integer", "description": "Maximum sections to return."},
        },
        ["query"],
    ),
    _function_schema(
        "unlock_discoverable_agent_tool",
        "Unlock a discoverable banking tool and return its schema.",
        {"agent_tool_name": _string_param("Tool name to unlock.")},
        ["agent_tool_name"],
    ),
    _function_schema(
        "call_discoverable_agent_tool",
        "Call an unlocked discoverable banking tool.",
        {
            "agent_tool_name": _string_param("Unlocked tool name."),
            "arguments": _string_param("JSON object string of the inner arguments."),
        },
        ["agent_tool_name"],
    ),
    _function_schema(
        "give_discoverable_user_tool",
        "Hand a discoverable tool to the user.",
        {
            "discoverable_tool_name": _string_param("Tool name."),
            "arguments": _string_param("JSON object string. Defaults to '{}'."),
        },
        ["discoverable_tool_name"],
    ),
    _function_schema(
        "get_user_information_by_id",
        "Look up a customer by user id.",
        {"user_id": _string_param("Customer id.")},
        ["user_id"],
    ),
    _function_schema(
        "get_user_information_by_name",
        "Look up a customer by name.",
        {"customer_name": _string_param("Customer name.")},
        ["customer_name"],
    ),
    _function_schema(
        "get_user_information_by_email",
        "Look up a customer by email.",
        {"email": _string_param("Customer email.")},
        ["email"],
    ),
    _function_schema(
        "get_credit_card_accounts_by_user",
        "List credit-card accounts for a user.",
        {"user_id": _string_param("Customer id.")},
        ["user_id"],
    ),
    _function_schema(
        "get_credit_card_transactions_by_user",
        "List credit-card transactions for a user.",
        {"user_id": _string_param("Customer id.")},
        ["user_id"],
    ),
    _function_schema(
        "get_referrals_by_user",
        "List referrals for a user.",
        {"user_id": _string_param("Customer id.")},
        ["user_id"],
    ),
    _function_schema("get_current_time", "Return the environment clock.", {}, []),
    _function_schema(
        "list_discoverable_agent_tools",
        "List discoverable tools already called in this session.",
        {},
        [],
    ),
    _function_schema(
        "log_verification",
        "Record a completed identity verification.",
        {
            "name": _string_param("Customer name."),
            "user_id": _string_param("Customer id."),
            "address": _string_param("Address."),
            "email": _string_param("Email."),
            "phone_number": _string_param("Phone number."),
            "date_of_birth": _string_param("Date of birth."),
            "time_verified": _string_param("Verification timestamp."),
        },
        ["name", "user_id", "address", "email", "phone_number", "date_of_birth", "time_verified"],
    ),
]


def _json_argument_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps({} if value is None else value, ensure_ascii=False)


def _openai_function_call(call: dict[str, Any]) -> dict[str, Any]:
    """Copy one internal tool call into OpenAI/DashScope function form."""
    function = call.get("function")
    if call.get("type") == "function" and isinstance(function, dict):
        return {
            "id": call.get("id", ""),
            "type": "function",
            "function": {
                "name": function.get("name", ""),
                "arguments": _json_argument_string(function.get("arguments", "{}")),
            },
        }
    return {
        "id": call.get("id", ""),
        "type": "function",
        "function": {
            "name": call.get("name", ""),
            "arguments": _json_argument_string(call.get("arguments", {})),
        },
    }


def _messages_for_litellm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Outbound copy. The graph keeps the internal {id, name, arguments} shape."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            item: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "content": (
                    content
                    if isinstance(content, str)
                    else json.dumps(content if content is not None else "", ensure_ascii=False)
                ),
            }
            if message.get("name"):
                item["name"] = message["name"]
            converted.append(item)
            continue
        item = {"role": role, "content": "" if content is None else content}
        calls = message.get("tool_calls") or []
        if role == "assistant" and calls:
            item["tool_calls"] = [
                _openai_function_call(call) for call in calls if isinstance(call, dict)
            ]
        converted.append(item)
    return converted
