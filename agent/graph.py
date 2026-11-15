import json
import operator
from typing import Annotated, Any, Callable, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from agent.guard import GuardEngine, SecurityState
from tools.schemas import StructuredToolError
from tools.tool_wrapper import ReliableToolExecutor, compute_idempotency_key

class AgentGraphState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    verified: bool
    unlocked_tools: list[str]
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
        resp = llm_fn(state.get("messages", []))
        tcs = resp.get("tool_calls") or []
        msg = {"role": "assistant", "content": resp.get("content", ""), "tool_calls": tcs}
        if not tcs:
            return {"messages": [msg], "steps": steps, "pending_tool_call": None, "pending_tool_calls": [], "stop_reason": "completed"}
        return {"messages": [msg], "steps": steps, "pending_tool_calls": list(tcs), "pending_tool_call": tcs[0]}

    def guard_node(state: AgentGraphState) -> dict[str, Any]:
        queue = list(state.get("pending_tool_calls") or [])
        if not queue and state.get("pending_tool_call"):
            queue = [state["pending_tool_call"]]
        tc = queue.pop(0) if queue else {}

        t_name = tc.get("name", "")
        t_args = tc.get("arguments", {})
        tc_id = tc.get("id", "call_1")
        session_id = state.get("session_id", "default")

        sec = SecurityState(
            verified=bool(state.get("verified", False)),
            unlocked_tools=set(state.get("unlocked_tools", [])),
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
            err_msg = {
                "role": "tool", "tool_call_id": tc_id, "name": t_name,
                "content": decision.error_payload.to_tool_message(),
            }
            stop = "max_guard_blocks_exceeded" if blocks >= max_guard_blocks else ""
            route = "end" if stop else ("guard" if queue else "agent")
            return {
                "messages": [err_msg],
                "guard_blocks": blocks,
                "guard_route": route,
                "stop_reason": stop,
                "pending_tool_calls": queue,
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
        route = "guard" if queue else "agent"
        return {
            "confirm_enter_count": enter_cnt,
            "messages": [{"role": "tool", "tool_call_id": tc.get("id", "call_1"), "name": tc.get("name", ""), "content": rej_err.to_tool_message()}],
            "pending_tool_call": None,
            "pending_tool_calls": queue,
            "approval_decision": {"route": route},
        }

    def tool_exec_node(state: AgentGraphState) -> dict[str, Any]:
        tc = state.get("pending_tool_call") or {}
        t_name = tc.get("name", "")
        real_name = tc.get("real_name", t_name)
        real_args = tc.get("real_args", {})
        session_id = state.get("session_id", "default")

        fn = tool_funcs.get(t_name) or tool_funcs.get(real_name)
        is_write = guard.is_high_risk_write(t_name, real_name)
        if fn is None:
            out_str = json.dumps({"error_type": "not_found", "message": f"Tool '{real_name}' not found."})
        else:
            out_str = executor.execute(session_id, real_name, real_args, fn, is_write=is_write)

        sec = SecurityState(
            verified=bool(state.get("verified", False)),
            unlocked_tools=set(state.get("unlocked_tools", [])),
            approved_idempotency_keys=set(state.get("approved_keys", [])),
        )
        if '"error_type":' not in out_str:
            guard.update_state_after_success(t_name, tc.get("arguments", {}), sec)

        return {
            "messages": [{"role": "tool", "tool_call_id": tc.get("id", "call_1"), "name": t_name, "content": out_str}],
            "verified": sec.verified,
            "unlocked_tools": sorted(list(sec.unlocked_tools)),
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
