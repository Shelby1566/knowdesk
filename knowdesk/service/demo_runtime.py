"""Deterministic offline runtime for KNOWDESK_DEMO_MODE=1 (no external API key required)."""

from __future__ import annotations

import json
from typing import Any, Callable


def build_demo_runtime() -> dict[str, Any]:
    """Build a self-contained banking ticket workflow exercising Guard, HITL interrupt, and WAL ledger."""

    def demo_llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
        tool_names_done = [m.get("name") for m in messages if m.get("role") == "tool"]
        if "log_verification" not in tool_names_done:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "demo_tc_verify",
                        "name": "log_verification",
                        "arguments": {
                            "name": "Demo Customer",
                            "user_id": "demo_user_001",
                            "address": "100 Main St",
                            "email": "demo@knowdesk.local",
                            "phone_number": "555-0100",
                            "date_of_birth": "1990-01-01",
                            "time_verified": "2026-11-22T10:00:00Z",
                        },
                    }
                ],
            }
        if "KB_search" not in tool_names_done:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "demo_tc_kb",
                        "name": "KB_search",
                        "arguments": {"query": "credit card freeze policy", "top_k": 3},
                    }
                ],
            }
        if "call_discoverable_agent_tool" not in tool_names_done:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "demo_tc_write",
                        "name": "call_discoverable_agent_tool",
                        "arguments": {
                            "agent_tool_name": "freeze_card",
                            "arguments": json.dumps(
                                {"user_id": "demo_user_001", "card_id": "card_demo_8842", "reason": "lost_card"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        last_tool = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"), "")
        if "user_rejected" in str(last_tool):
            return {
                "content": "Understood — the supervisor rejected the card freeze request, so no write operation was executed.",
                "tool_calls": [],
            }
        return {
            "content": "Identity verified, policy checked, and your card (card_demo_8842) has been frozen after supervisor approval.",
            "tool_calls": [],
        }

    def demo_kb_search(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": json.dumps(
                [
                    {
                        "doc_id": "demo_card_freeze_policy",
                        "title": "Credit Card Security > Freeze & Lost Card",
                        "section": "Temporary & Permanent Card Freeze",
                        "body": "Verify customer identity via log_verification before calling freeze_card. High-risk card writes require supervisor confirmation.",
                    }
                ],
                ensure_ascii=False,
            ),
            "error": False,
        }

    def demo_verify(args: dict[str, Any]) -> dict[str, Any]:
        return {"verified": True, "user_id": args.get("user_id", "demo_user_001")}

    def demo_discoverable_write(args: dict[str, Any]) -> dict[str, Any]:
        inner = args.get("arguments", "{}")
        parsed = json.loads(inner) if isinstance(inner, str) else dict(inner or {})
        return {"status": "success", "tool": args.get("agent_tool_name", "freeze_card"), "applied": parsed}

    tool_funcs: dict[str, Callable[[dict[str, Any]], Any]] = {
        "KB_search": demo_kb_search,
        "log_verification": demo_verify,
        "call_discoverable_agent_tool": demo_discoverable_write,
    }
    return {"llm_fn": demo_llm, "tool_funcs": tool_funcs, "mcp_backend": None}
