import pytest

try:
    from tau2.data_model.message import ToolCall
    from tau2.environment.environment import Environment
    from knowdesk.mcp.server import SessionEnvironmentManager, _get_banking_env, get_banking_tasks
except ImportError:
    ToolCall = None

if ToolCall is None:
    pytest.skip("tau2 not installed or tau2-bench not found", allow_module_level=True)

from knowdesk.mcp.client import MCPToolBackend


def test_m36_dual_path_db_hash_consistency_and_negative_case(tmp_path, monkeypatch):
    tasks = get_banking_tasks()[:10]
    db_path = tmp_path / "mcp_actions.sqlite"
    backend = MCPToolBackend(db_path=db_path)

    probe = backend.env_manager.get_or_create_env("probe_real_env")
    assert isinstance(probe, Environment)
    assert "Mock" not in type(probe).__name__

    import mcp.client.client as mcp_client_mod

    jsonrpc_used = {"count": 0}
    original = mcp_client_mod.JSONRPCDispatcher

    class _RecordingDispatcher(original):
        def __init__(self, *args, **kwargs):
            jsonrpc_used["count"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mcp_client_mod, "JSONRPCDispatcher", _RecordingDispatcher)

    for idx, task in enumerate(tasks):
        env_a = _get_banking_env()
        if task.initial_state is not None:
            env_a.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=[],
            )

        sid = f"consist_session_{idx}"
        backend.init_session(sid, initial_state=task.initial_state)

        gold_actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
        for step_i, act in enumerate(gold_actions):
            req = getattr(act, "requestor", "assistant") or "assistant"
            tc = ToolCall(id=f"a_{step_i}", name=act.name, arguments=act.arguments, requestor=req)
            resp_a = env_a.get_response(tc)
            resp_b = backend.call_tool(sid, act.name, act.arguments, requestor=req)
            assert str(getattr(resp_a, "content", "")) == resp_b["content"]

        restored = SessionEnvironmentManager(db_path=db_path)
        assert str(env_a.get_db_hash()) == restored.get_session_db_hash(sid)

    assert jsonrpc_used["count"] > 0

    found_mutating_task = False
    for task in tasks:
        env_full = _get_banking_env()
        env_empty = _get_banking_env()
        if task.initial_state is not None:
            for env in (env_full, env_empty):
                env.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                    message_history=[],
                )
        actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
        for step_i, act in enumerate(actions):
            req = getattr(act, "requestor", "assistant") or "assistant"
            env_full.get_response(ToolCall(id=f"f_{step_i}", name=act.name, arguments=act.arguments, requestor=req))
        if str(env_full.get_db_hash()) != str(env_empty.get_db_hash()):
            found_mutating_task = True
            backend.init_session("neg_test", initial_state=task.initial_state)
            assert str(env_full.get_db_hash()) != backend.get_db_hash("neg_test")
            break
    assert found_mutating_task is True
