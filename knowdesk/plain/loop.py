import json
from dataclasses import dataclass, field
from typing import Any, Callable

from knowdesk.agent.guard import GuardEngine, SecurityState
from knowdesk.agent.workflow import project_messages_for_model, stage_directive
from knowdesk.tools.schemas import StructuredToolError, is_tool_result_success
from knowdesk.tools.tool_wrapper import ReliableToolExecutor, compute_idempotency_key


@dataclass
class PlainAgentState:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    security: SecurityState = field(default_factory=SecurityState)
    steps: int = 0
    guard_blocks: int = 0
    stop_reason: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    pending_approval: dict[str, Any] | None = None


class PlainAgentLoop:
    """无框架手写 Agent 循环。与 LangGraph 版共享 GuardEngine 与 ReliableToolExecutor。"""

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
        tool_funcs: dict[str, Callable[[dict[str, Any]], Any]],
        guard: GuardEngine | None = None,
        executor: ReliableToolExecutor | None = None,
        max_steps: int = 15,
        max_guard_blocks: int = 3,
    ):
        self.llm_fn = llm_fn
        self.tool_funcs = tool_funcs
        self.guard = guard or GuardEngine(require_human_confirm_on_write=False)
        self.executor = executor or ReliableToolExecutor()
        self.max_steps = max_steps
        self.max_guard_blocks = max_guard_blocks

    def run_turn(self, state: PlainAgentState, user_text: str | None = None) -> PlainAgentState:
        if user_text is not None:
            state.messages.append({"role": "user", "content": user_text})

        while state.steps < self.max_steps:
            if state.pending_approval is not None:
                state.stop_reason = "awaiting_approval"
                return state
            state.steps += 1
            directive = stage_directive(state.messages, state.security, available_tools=set(self.tool_funcs))
            resp = self.llm_fn(project_messages_for_model(state.messages, directive))
            usage = resp.get("usage") or {}
            state.total_input_tokens += int(usage.get("prompt_tokens", 0))
            state.total_output_tokens += int(usage.get("completion_tokens", 0))

            tool_calls = resp.get("tool_calls") or []
            content = resp.get("content") or ""
            state.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                state.stop_reason = "completed"
                return state

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{state.steps}")
                t_name = tc.get("name", "")
                t_args = tc.get("arguments", {})
                if t_name in ("transfer_to_human_agents", "finish_conversation"):
                    state.stop_reason = "transferred_to_human"
                    return state

                parsed_args = t_args
                if isinstance(t_args, str):
                    try:
                        parsed_args = json.loads(t_args) if t_args.strip() else {}
                    except json.JSONDecodeError:
                        parsed_args = t_args

                idem_args = parsed_args if isinstance(parsed_args, dict) else {"raw": str(parsed_args)}
                idem_key = compute_idempotency_key(state.session_id, t_name, idem_args)
                decision = self.guard.inspect(t_name, parsed_args, state.security, idem_key=idem_key)

                if decision.action == "block":
                    state.guard_blocks += 1
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": t_name,
                        "content": decision.error_payload.to_tool_message() if decision.error_payload else "{}",
                    })
                    if state.guard_blocks >= self.max_guard_blocks:
                        state.stop_reason = "max_guard_blocks_exceeded"
                        return state
                    continue

                if decision.action == "require_approval":
                    state.pending_approval = {
                        "tool_call_id": tc_id,
                        "tool_name": t_name,
                        "real_tool_name": decision.real_tool_name,
                        "real_args": decision.real_args,
                        "idem_key": idem_key,
                        "raw_args": parsed_args if isinstance(parsed_args, (dict, str)) else {},
                    }
                    state.stop_reason = "awaiting_approval"
                    return state

                self._execute_decision(state, tc_id, t_name, decision, parsed_args)

        state.stop_reason = "max_steps_exceeded"
        return state

    def resume_approval(
        self,
        state: PlainAgentState,
        decision: str,
        reason: str = "",
        modified_args: dict[str, Any] | None = None,
    ) -> PlainAgentState:
        """批准、拒绝或修改挂起的写操作。修改后的参数必须重新经过 Guard。"""
        pending = state.pending_approval
        if not pending:
            state.stop_reason = "no_pending_approval"
            return state

        choice = str(decision).lower()
        if choice == "reject":
            err = StructuredToolError(
                error_type="user_rejected",
                message=f"Human supervisor rejected '{pending.get('real_tool_name')}': {reason or 'No reason'}",
                retryable=False,
            )
            state.messages.append({
                "role": "tool",
                "tool_call_id": pending["tool_call_id"],
                "name": pending["tool_name"],
                "content": err.to_tool_message(),
            })
            state.pending_approval = None
            state.stop_reason = "approval_rejected"
            return state

        if choice == "modify":
            new_args = dict(modified_args or {})
            t_name = pending["tool_name"]
            payload: dict[str, Any] = (
                {"agent_tool_name": pending["real_tool_name"], "arguments": new_args}
                if t_name == "call_discoverable_agent_tool"
                else new_args
            )
            new_idem = compute_idempotency_key(state.session_id, t_name, payload)
            inspected = self.guard.inspect(t_name, payload, state.security, idem_key=new_idem)
            if inspected.action == "block":
                state.guard_blocks += 1
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": pending["tool_call_id"],
                    "name": t_name,
                    "content": inspected.error_payload.to_tool_message() if inspected.error_payload else "{}",
                })
                state.pending_approval = None
                state.stop_reason = "approval_modify_blocked"
                return state
            if inspected.action == "require_approval":
                state.security.approved_idempotency_keys.add(new_idem)
                inspected = self.guard.inspect(t_name, payload, state.security, idem_key=new_idem)
            if inspected.action != "allow":
                state.pending_approval = None
                state.stop_reason = "approval_modify_blocked"
                return state
            self._execute_decision(state, pending["tool_call_id"], t_name, inspected, payload)
            state.pending_approval = None
            state.stop_reason = "approval_executed"
            return state

        if choice == "approve":
            state.security.approved_idempotency_keys.add(pending["idem_key"])
            inspected = self.guard.inspect(
                pending["tool_name"], pending.get("raw_args", pending["real_args"]), state.security, idem_key=pending["idem_key"]
            )
            if inspected.action != "allow":
                if inspected.error_payload is not None:
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": pending["tool_call_id"],
                        "name": pending["tool_name"],
                        "content": inspected.error_payload.to_tool_message(),
                    })
                state.pending_approval = None
                state.stop_reason = "approval_blocked"
                return state
            self._execute_decision(
                state, pending["tool_call_id"], pending["tool_name"], inspected, pending.get("raw_args", pending["real_args"])
            )
            state.pending_approval = None
            state.stop_reason = "approval_executed"
            return state

        state.stop_reason = "unknown_approval_decision"
        return state

    def _execute_decision(self, state: PlainAgentState, tc_id: str, t_name: str, decision, raw_args: Any) -> None:
        fn = self.tool_funcs.get(t_name) or self.tool_funcs.get(decision.real_tool_name)
        is_write = self.guard.is_high_risk_write(t_name, decision.real_tool_name)
        if fn is None:
            out_str = json.dumps({"error_type": "not_found", "message": f"Tool '{t_name}' not registered."})
        else:
            out_str = self.executor.execute(
                state.session_id, decision.real_tool_name, decision.real_args, fn, is_write=is_write
            )
        if is_tool_result_success(out_str):
            self.guard.update_state_after_success(
                t_name, raw_args, state.security, result_content=out_str
            )
        state.messages.append({"role": "tool", "tool_call_id": tc_id, "name": t_name, "content": out_str})
