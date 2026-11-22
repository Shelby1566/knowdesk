import json
import operator
from typing import Annotated, Any, Callable, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from knowdesk.agent.guard import GuardEngine, SecurityState
from knowdesk.agent.workflow import project_messages_for_model, stage_directive
from knowdesk.tools.schemas import StructuredToolError, is_tool_result_success
from knowdesk.tools.tool_wrapper import ReliableToolExecutor, compute_idempotency_key

class AgentGraphState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    verified: bool
    unlocked_tools: list[str]
    unlocked_schemas: dict[str, Any]
    approved_keys: list[str]
    pending_tool_call: dict[str, Any] | None
    pending_tool_calls: list[dict[str, Any]]
    guard_route: str
    approval_decision: dict[str, Any] | None
    steps: int
    guard_blocks: int
    stop_reason: str
    confirm_enter_count: int

def build_knowdesk_graph(
    llm_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    tool_funcs: dict[str, Callable[[dict[str, Any]], Any]],
    guard: GuardEngine,
    executor: ReliableToolExecutor,
    checkpointer: Any = None,
    max_steps: int = 15,
    max_guard_blocks: int = 3,
):
    """
    M10: LangGraph 主状态图
    核心设计：agent -> guard (纯判断无副作用) -> human_confirm (只调用 interrupt，重跑安全) -> tool_exec (幂等执行)
    支持单轮批量 tool_calls 并发分发与顺序安全检查执行。
    """
    def agent_node(state: AgentGraphState) -> dict[str, Any]:
        steps = state.get("steps", 0) + 1
        sec = SecurityState(
            verified=bool(state.get("verified", False)),
            unlocked_tools=set(state.get("unlocked_tools", [])),
            unlocked_schemas=dict(state.get("unlocked_schemas", {})),
            approved_idempotency_keys=set(state.get("approved_keys", [])),
        )
        directive = stage_directive(state.get("messages", []), sec, available_tools=set(tool_funcs))
        visible = project_messages_for_model(state.get("messages", []), directive)
        resp = llm_fn(visible)
        tcs = resp.get("tool_calls") or []
        msg = {"role": "assistant", "content": resp.get("content", ""), "tool_calls": tcs}
        if not tcs:
            return {"messages": [msg], "steps": steps, "pending_tool_call": None, "pending_tool_calls": [], "stop_reason": "completed"}
        return {"messages": [msg], "steps": steps, "pending_tool_calls": list(tcs), "pending_tool_call": None}

    def _skip_message(call: dict[str, Any], why: str, error_type: str) -> dict[str, Any]:
        err = StructuredToolError(
            error_type=error_type,
            message=f"Skipped '{call.get('name', '')}' because an earlier call in the same batch was {why}.",
            retryable=False,
        )
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": call.get("name", ""),
            "content": err.to_tool_message(),
        }

    def guard_node(state: AgentGraphState) -> dict[str, Any]:
        queue = list(state.get("pending_tool_calls") or [])
        seated = state.get("pending_tool_call")
        if seated:
            tc = seated
        else:
            tc = queue.pop(0) if queue else {}

        t_name = tc.get("name", "")
        t_args = tc.get("arguments", {})
        tc_id = tc.get("id", "call_1")
        session_id = state.get("session_id", "default")

        sec = SecurityState(
            verified=bool(state.get("verified", False)),
            unlocked_tools=set(state.get("unlocked_tools", [])),
            unlocked_schemas=dict(state.get("unlocked_schemas", {})),
            approved_idempotency_keys=set(state.get("approved_keys", [])),
        )
        parsed = t_args
        if isinstance(t_args, str):
            try:
                parsed = json.loads(t_args) if t_args.strip() else {}
            except json.JSONDecodeError:
                parsed = t_args

        idem_key = compute_idempotency_key(
            session_id, t_name, parsed if isinstance(parsed, dict) else {"raw": str(parsed)}
        )
        decision = guard.inspect(t_name, parsed, sec, idem_key=idem_key)

        if decision.action == "block":
            blocks = state.get("guard_blocks", 0) + 1
            messages = [{
                "role": "tool", "tool_call_id": tc_id, "name": t_name,
                "content": decision.error_payload.to_tool_message() if decision.error_payload else "{}",
            }]
            messages.extend(_skip_message(rest, "blocked", "guard_blocked") for rest in queue)
            stop = "max_guard_blocks_exceeded" if blocks >= max_guard_blocks else ""
            return {
                "messages": messages,
                "guard_blocks": blocks,
                "guard_route": "end" if stop else "agent",
                "stop_reason": stop,
                "pending_tool_calls": [],
                "pending_tool_call": None,
            }

        enriched_tc = {
            "id": tc_id, "name": t_name, "arguments": parsed,
            "real_name": decision.real_tool_name, "real_args": decision.real_args, "idem_key": idem_key,
        }
        if decision.action == "require_approval":
            return {"pending_tool_call": enriched_tc, "pending_tool_calls": queue, "guard_route": "human_confirm"}
        return {"pending_tool_call": enriched_tc, "pending_tool_calls": queue, "guard_route": "tool_exec"}

    def human_confirm_node(state: AgentGraphState) -> dict[str, Any]:
        """独立人工确认节点：只调用 interrupt()，绝不执行任何工具或写库副作用！"""
        enter_cnt = state.get("confirm_enter_count", 0) + 1
        tc = state.get("pending_tool_call") or {}
        res = interrupt({
            "tool_name": tc.get("real_name", tc.get("name")),
            "arguments": tc.get("real_args", tc.get("arguments")),
            "idem_key": tc.get("idem_key"),
        })
        decision_type = (res or {}).get("decision", "reject")
        approved = list(state.get("approved_keys", []))

        if decision_type == "approve":
            if tc.get("idem_key") and tc["idem_key"] not in approved:
                approved.append(tc["idem_key"])
            return {"confirm_enter_count": enter_cnt, "approved_keys": approved, "approval_decision": {"route": "tool_exec"}}

        if decision_type == "modify":
            new_args = (res or {}).get("modified_args", {})
            t_name = tc.get("name", "")
            updated_raw = (
                {"agent_tool_name": tc.get("real_name"), "arguments": new_args}
                if t_name == "call_discoverable_agent_tool"
                else new_args
            )
            new_idem = compute_idempotency_key(state.get("session_id", "default"), t_name, updated_raw)
            if new_idem not in approved:
                approved.append(new_idem)
            new_tc = {"id": tc.get("id", "call_1"), "name": t_name, "arguments": updated_raw}
            # 改参数后必须重新回 guard 节点做安全检查！
            return {
                "confirm_enter_count": enter_cnt,
                "approved_keys": approved,
                "pending_tool_call": new_tc,
                "approval_decision": {"route": "guard"},
            }

        # reject: 拒绝执行，返回结构化反馈给 agent
        rej_err = StructuredToolError(
            error_type="user_rejected",
            message=f"Human supervisor rejected '{tc.get('real_name')}': {(res or {}).get('reason', 'No reason')}",
            retryable=False,
        )
        queue = list(state.get("pending_tool_calls") or [])
        messages = [{
            "role": "tool",
            "tool_call_id": tc.get("id", "call_1"),
            "name": tc.get("name", ""),
            "content": rej_err.to_tool_message(),
        }]
        messages.extend(_skip_message(rest, "rejected", "user_rejected") for rest in queue)
        return {
            "confirm_enter_count": enter_cnt,
            "messages": messages,
            "pending_tool_call": None,
            "pending_tool_calls": [],
            "approval_decision": {"route": "agent"},
        }

    def tool_exec_node(state: AgentGraphState) -> dict[str, Any]:
        tc = state.get("pending_tool_call") or {}
        t_name = tc.get("name", "")
        real_name = tc.get("real_name", t_name)
        real_args = tc.get("real_args", {})
        session_id = state.get("session_id", "default")

        # The outer dispatcher is what MCP's call_discoverable_agent_tool expects.
        # A directly registered inner tool, such as the unit-test freeze_card, still gets real_args.
        if t_name == "call_discoverable_agent_tool" and tool_funcs.get(t_name) is not None:
            fn = tool_funcs.get(t_name)
            args_obj = real_args if isinstance(real_args, dict) else {}
            exec_args = {
                "agent_tool_name": real_name,
                "arguments": json.dumps(args_obj, ensure_ascii=False),
            }
        else:
            fn = tool_funcs.get(t_name) or tool_funcs.get(real_name)
            exec_args = real_args if isinstance(real_args, dict) else {}
        is_write = guard.is_high_risk_write(t_name, real_name)
        if fn is None:
            out_str = json.dumps({"error_type": "not_found", "message": f"Tool '{real_name}' not found."})
        else:
            out_str = executor.execute(session_id, real_name, exec_args, fn, is_write=is_write)

        sec = SecurityState(
            verified=bool(state.get("verified", False)),
            unlocked_tools=set(state.get("unlocked_tools", [])),
            unlocked_schemas=dict(state.get("unlocked_schemas", {})),
            approved_idempotency_keys=set(state.get("approved_keys", [])),
        )
        if is_tool_result_success(out_str):
            guard.update_state_after_success(t_name, tc.get("arguments", {}), sec, result_content=out_str)

        return {
            "messages": [{"role": "tool", "tool_call_id": tc.get("id", "call_1"), "name": t_name, "content": out_str}],
            "verified": sec.verified,
            "unlocked_tools": sorted(list(sec.unlocked_tools)),
            "unlocked_schemas": dict(sec.unlocked_schemas),
            "pending_tool_call": None,
        }

    builder = StateGraph(AgentGraphState)
    builder.add_node("agent", agent_node)
    builder.add_node("guard", guard_node)
    builder.add_node("human_confirm", human_confirm_node)
    builder.add_node("tool_exec", tool_exec_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        lambda s: "guard" if (s.get("pending_tool_calls") or s.get("pending_tool_call")) else END,
        {"guard": "guard", END: END},
    )
    builder.add_conditional_edges(
        "guard",
        lambda s: s.get("guard_route", "tool_exec"),
        {"tool_exec": "tool_exec", "human_confirm": "human_confirm", "agent": "agent", "guard": "guard", "end": END},
    )
    builder.add_conditional_edges(
        "human_confirm",
        lambda s: (s.get("approval_decision") or {}).get("route", "agent"),
        {"tool_exec": "tool_exec", "guard": "guard", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "tool_exec",
        lambda s: "guard" if s.get("pending_tool_calls") else (END if s.get("steps", 0) >= max_steps else "agent"),
        {"guard": "guard", "agent": "agent", END: END},
    )
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
