import pytest
from unittest.mock import patch
try:
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
except ImportError:
    AssistantMessage = None

if AssistantMessage is None:
    pytest.skip("tau2 not installed or tau2-bench not found", allow_module_level=True)

from adapters.tau2_adapter import KnowDeskPlainTauAgent, KnowDeskGraphTauAgent, GUARD_TRACE_PATH

def _make_assistant_tc(call_id: str, name: str, arguments: dict) -> AssistantMessage:
    tc = ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")
    return AssistantMessage(role="assistant", content="", tool_calls=[tc])

def test_tau2_agents_internal_guard_self_correction():
    """验证 plain 与 langgraph 适配器在 τ 回合内拦截未解锁工具并引导自纠，且不污染 state.messages"""
    if GUARD_TRACE_PATH.exists():
        GUARD_TRACE_PATH.unlink()

    # 剧本：第1次违规直接调未解锁的 call_discoverable_agent_tool -> 被 Guard R03 拦截 -> 第2次自纠为先调 unlock
    bad_msg = _make_assistant_tc(
        "c1", "call_discoverable_agent_tool", {"agent_tool_name": "freeze_card", "arguments": '{"card_id": "1"}'}
    )
    good_msg = _make_assistant_tc(
        "c2", "unlock_discoverable_agent_tool", {"agent_tool_name": "freeze_card"}
    )

    for AgentCls in (KnowDeskPlainTauAgent, KnowDeskGraphTauAgent):
        seq = [bad_msg, good_msg]
        idx = 0
        def fake_generate(**kwargs):
            nonlocal idx
            res = seq[min(idx, len(seq) - 1)]
            idx += 1
            return res

        with patch("adapters.tau2_adapter.generate", side_effect=fake_generate):
            agent = AgentCls(tools=[], domain_policy="Policy", llm="dashscope/qwen3.8-flash", llm_args={})
            state = agent.get_init_state()
            user_msg = UserMessage(role="user", content="Please freeze my card.")
            out_msg, new_state = agent.generate_next_message(user_msg, state)

            # 最终返回给 τ 编排器的必须是自纠后的 unlock_discoverable_agent_tool
            assert out_msg.tool_calls[0].name == "unlock_discoverable_agent_tool"
            # τ 的 state.messages 中只包含 [user_msg, good_msg]，绝不残留被拦截的 bad_msg（保证重放判分安全）
            assert len(new_state.messages) == 2
            assert new_state.messages[-1].tool_calls[0].name == "unlock_discoverable_agent_tool"

    assert GUARD_TRACE_PATH.exists()
    events = [line for line in GUARD_TRACE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("guard_blocked" in e for e in events)
    assert any("self_corrected" in e for e in events)
