"""Release-freeze checks for the live tool protocol and failed-write ledger."""

import json
import sqlite3
from pathlib import Path

from agent.graph import build_knowdesk_graph
from agent.guard import GuardEngine
from service.app import LIVE_OPENAI_TOOLS, litellm_complete
from tools.schemas import is_tool_result_success
from tools.tool_wrapper import ReliableToolExecutor


def _script(responses: list[dict]):
    index = {"n": 0}

    def llm(_messages):
        item = responses[min(index["n"], len(responses) - 1)]
        index["n"] += 1
        return item

    return llm


def test_litellm_request_uses_tools_and_openai_function_calls(monkeypatch):
    import litellm

    history = [
        {"role": "user", "content": "annual fee"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_kb", "name": "KB_search", "arguments": {"query": "annual fee"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_kb", "name": "KB_search", "content": "[]"},
    ]
    captured: dict = {}

    class _Function:
        name = "KB_search"
        arguments = '{"query": "annual fee"}'

    class _Call:
        id = "call_next"
        function = _Function()

    class _Message:
        content = "checked"
        tool_calls = [_Call()]

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None
        model = "dashscope/qwen-plus"

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    result = litellm_complete(history)

    names = [item["function"]["name"] for item in captured["tools"]]
    for required in (
        "KB_search",
        "unlock_discoverable_agent_tool",
        "call_discoverable_agent_tool",
        "give_discoverable_user_tool",
        "get_user_information_by_id",
    ):
        assert required in names
    assert captured["tools"] == LIVE_OPENAI_TOOLS

    sent_call = captured["messages"][1]["tool_calls"][0]
    assert sent_call["type"] == "function"
    assert sent_call["function"]["name"] == "KB_search"
    assert sent_call["function"]["arguments"] == '{"query": "annual fee"}'
    assert captured["messages"][2]["tool_call_id"] == "call_kb"
    assert history[1]["tool_calls"][0] == {
        "id": "call_kb",
        "name": "KB_search",
        "arguments": {"query": "annual fee"},
    }
    assert result["tool_calls"] == [
        {"id": "call_next", "name": "KB_search", "arguments": {"query": "annual fee"}}
    ]


def test_discoverable_envelope_is_forwarded_to_mcp():
    forwarded = []
    inner = []

    def mcp_call_discoverable(args):
        forwarded.append(dict(args))
        return {"content": "frozen", "error": False}

    def freeze_card(args):
        inner.append(dict(args))
        return {"frozen": True}

    guard = GuardEngine(
        enforce_unlock=False,
        enforce_verification=True,
        require_human_confirm_on_write=False,
    )
    graph = build_knowdesk_graph(
        _script([
            {
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "freeze_card",
                        "arguments": {"card_id": "card_1"},
                    },
                }],
            },
            {"content": "done", "tool_calls": []},
        ]),
        {
            "call_discoverable_agent_tool": mcp_call_discoverable,
            "freeze_card": freeze_card,
        },
        guard,
        ReliableToolExecutor(),
        max_steps=3,
    )
    graph.invoke(
        {
            "session_id": "s-mcp",
            "messages": [{"role": "user", "content": "freeze the card"}],
            "verified": True,
        },
        {"configurable": {"thread_id": "t-mcp"}},
    )
    assert forwarded == [{
        "agent_tool_name": "freeze_card",
        "arguments": json.dumps({"card_id": "card_1"}, ensure_ascii=False),
    }]
    assert inner == []

    direct = []
    direct_graph = build_knowdesk_graph(
        _script([
            {
                "content": "",
                "tool_calls": [{
                    "id": "c2",
                    "name": "freeze_card",
                    "arguments": {"card_id": "card_1"},
                }],
            },
            {"content": "done", "tool_calls": []},
        ]),
        {"freeze_card": lambda args: (direct.append(dict(args)), {"frozen": True})[1]},
        guard,
        ReliableToolExecutor(),
        max_steps=3,
    )
    direct_graph.invoke(
        {
            "session_id": "s-direct",
            "messages": [{"role": "user", "content": "freeze the card"}],
            "verified": True,
        },
        {"configurable": {"thread_id": "t-direct"}},
    )
    assert direct == [{"card_id": "card_1"}]


def test_mcp_business_error_does_not_verify_or_hit_ledger(tmp_path: Path):
    nested = json.dumps({"content": json.dumps({"error": True}), "error": False})
    assert is_tool_result_success('{"error": true}') is False
    assert is_tool_result_success('{"ok": false}') is False
    assert is_tool_result_success(nested) is False
    assert is_tool_result_success(json.dumps({"content": json.dumps([{"id": "doc"}]), "error": False})) is True
    assert is_tool_result_success(json.dumps({"content": "", "error": False})) is True
    assert is_tool_result_success('{"name": "Alice"}') is True

    db_path = tmp_path / "ledger.sqlite"
    calls = {"n": 0}

    def failed_write(_args):
        calls["n"] += 1
        return {"error": True, "message": "cannot freeze"}

    executor = ReliableToolExecutor(idempotency_db_path=db_path)
    executor.execute("sess", "freeze_card", {"card_id": "card_1"}, failed_write, is_write=True)
    executor.execute("sess", "freeze_card", {"card_id": "card_1"}, failed_write, is_write=True)
    assert calls["n"] == 2
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute("SELECT COUNT(*) FROM idempotency_ledger").fetchone()[0]
    assert stored == 0

    memory_calls = {"n": 0}

    def failed_memory(_args):
        memory_calls["n"] += 1
        return {"ok": False}

    memory = ReliableToolExecutor()
    memory.execute("sess", "freeze_card", {"card_id": "card_9"}, failed_memory, is_write=True)
    memory.execute("sess", "freeze_card", {"card_id": "card_9"}, failed_memory, is_write=True)
    assert memory_calls["n"] == 2

    guard = GuardEngine(
        enforce_unlock=False,
        enforce_verification=True,
        require_human_confirm_on_write=False,
    )
    graph = build_knowdesk_graph(
        _script([
            {
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "name": "get_user_information_by_id",
                    "arguments": {"user_id": "u1"},
                }],
            },
            {"content": "still unknown", "tool_calls": []},
        ]),
        {"get_user_information_by_id": lambda _args: {"error": True, "message": "no such user"}},
        guard,
        ReliableToolExecutor(idempotency_db_path=db_path),
        max_steps=3,
    )
    out = graph.invoke(
        {
            "session_id": "sess",
            "messages": [{"role": "user", "content": "who is u1"}],
            "verified": False,
        },
        {"configurable": {"thread_id": "t-verify"}},
    )
    assert out["verified"] is False
    with sqlite3.connect(db_path) as conn:
        stored_after = conn.execute("SELECT COUNT(*) FROM idempotency_ledger").fetchone()[0]
    assert stored_after == 0
