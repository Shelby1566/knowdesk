import json
from dataclasses import dataclass, field
from typing import Any, Callable
from agent.guard import GuardEngine, SecurityState
from tools.tool_wrapper import ReliableToolExecutor, compute_idempotency_key

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

class PlainAgentLoop:
    """M9: 无框架手写 Agent 循环（与 LangGraph 版共享同一套 GuardEngine 与 ReliableToolExecutor）"""
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
            state.steps += 1
            resp = self.llm_fn(state.messages)
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

                idem_key = compute_idempotency_key(state.session_id, t_name, parsed_args if isinstance(parsed_args, dict) else {"raw": str(parsed_args)})
                decision = self.guard.inspect(t_name, parsed_args, state.security, idem_key=idem_key)

                if decision.action == "block":
                    state.guard_blocks += 1
                    state.messages.append({
                        "role": "tool", "tool_call_id": tc_id, "name": t_name,
                        "content": decision.error_payload.to_tool_message(),
                    })
                    if state.guard_blocks >= self.max_guard_blocks:
                        state.stop_reason = "max_guard_blocks_exceeded"
                        return state
                    continue

                fn = self.tool_funcs.get(t_name) or self.tool_funcs.get(decision.real_tool_name)
                is_write = self.guard.is_high_risk_write(t_name, decision.real_tool_name)
                if fn is None:
                    out_str = json.dumps({"error_type": "not_found", "message": f"Tool '{t_name}' not registered."})
                else:
                    out_str = self.executor.execute(
                        state.session_id, decision.real_tool_name, decision.real_args, fn, is_write=is_write
                    )
                    if '"error_type":' not in out_str:
                        self.guard.update_state_after_success(t_name, parsed_args, state.security)

                state.messages.append({"role": "tool", "tool_call_id": tc_id, "name": t_name, "content": out_str})

        state.stop_reason = "max_steps_exceeded"
        return state
