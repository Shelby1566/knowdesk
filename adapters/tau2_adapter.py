import json
import operator
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

import litellm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from tau2.agent.llm_agent import LLMAgent, generate
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever
from tau2.registry import registry

from agent.guard import GuardEngine, SecurityState
from kb.hybrid_retriever import KnowDeskHybridRetriever

PROMPT_V1_PATH = Path(__file__).resolve().parents[1] / "agent/prompts/v1.md"
GUARD_TRACE_PATH = Path(__file__).resolve().parents[1] / "obs/guard_events.jsonl"

def register_litellm_pricing() -> None:
    for _m, (_in_rmb, _out_rmb) in [
        ("dashscope/qwen3.8-flash", (0.8, 2.7)), ("qwen3.8-flash", (0.8, 2.7)),
        ("dashscope/deepseek-v4.1-flash-1017", (1.0, 4.0)), ("deepseek-v4.1-flash-1017", (1.0, 4.0)),
    ]:
        litellm.model_cost[_m] = {
            "max_tokens": 32768,
            "input_cost_per_token": _in_rmb / 1e6 / 7.2,
            "output_cost_per_token": _out_rmb / 1e6 / 7.2,
            "litellm_provider": "dashscope",
            "mode": "chat",
        }

def load_sop_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path is None:
        ver = os.getenv("KNOWDESK_PROMPT_VERSION", "v2")
        cand = Path(__file__).resolve().parents[1] / f"agent/prompts/{ver}.md"
        prompt_path = cand if cand.exists() else PROMPT_V1_PATH
    raw = prompt_path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return raw.strip()

def log_guard_event(event: dict[str, Any]) -> None:
    GUARD_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GUARD_TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def make_tau_tool_message(tc_id: str, content: str) -> ToolMessage:
    fields = getattr(ToolMessage, "model_fields", {})
    kwargs: dict[str, Any] = {"id": tc_id, "content": content}
    if "role" in fields:
        kwargs["role"] = "tool"
    if "requestor" in fields:
        kwargs["requestor"] = "assistant"
    return ToolMessage(**kwargs)

_ORIG_BM25_RETRIEVE = BM25Retriever.retrieve
_ORIG_BM25_RETRIEVE_BATCH = BM25Retriever.retrieve_batch

def patch_retrieval_mode(mode: str = "hybrid") -> None:
    if mode == "hybrid":
        retriever = KnowDeskHybridRetriever.get_instance()

        def _hybrid_retrieve(self, input_data: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, float]]:
            query = input_data.get(self.query_key, "")
            if not query or not str(query).strip():
                return []
            top_k = int(input_data.get("top_k", self.top_k))
            hits = retriever.search(str(query), k=top_k)
            return [(h["id"], float(h["score"])) for h in hits]

        def _hybrid_retrieve_batch(self, input_data_list: list[dict[str, Any]], state: dict[str, Any]):
            return [_hybrid_retrieve(self, item, state) for item in input_data_list]

        BM25Retriever.retrieve = _hybrid_retrieve
        BM25Retriever.retrieve_batch = _hybrid_retrieve_batch
    else:
        BM25Retriever.retrieve = _ORIG_BM25_RETRIEVE
        BM25Retriever.retrieve_batch = _ORIG_BM25_RETRIEVE_BATCH

def _update_sec_from_last_turn(state: Any, incoming_msg: Any, guard: GuardEngine, sec: SecurityState) -> None:
    """检查环境返回的 ToolMessage 是否成功，仅在非错误返回时更新核身与解锁状态，并动态捕获 Schema"""
    tool_msgs = incoming_msg.tool_messages if isinstance(incoming_msg, MultiToolMessage) else [incoming_msg]
    failed_ids: set[str] = set()
    content_by_id: dict[str, str] = {}
    for tm in tool_msgs:
        tm_id = str(getattr(tm, "id", ""))
        tm_content = str(getattr(tm, "content", ""))
        content_by_id[tm_id] = tm_content
        content_low = tm_content.lower()
        if getattr(tm, "error", False) or "error:" in content_low or "not found" in content_low:
            failed_ids.add(tm_id)

    for msg in reversed(state.messages):
        if isinstance(msg, AssistantMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tc_id = str(getattr(tc, "id", ""))
                if tc_id not in failed_ids:
                    guard.update_state_after_success(
                        tc.name, tc.arguments, sec, result_content=content_by_id.get(tc_id)
                    )
            break

class KnowDeskPlainTauAgent(LLMAgent):
    def __init__(self, tools, domain_policy: str, llm: str, llm_args: dict | None = None, **kwargs):
        register_litellm_pricing()
        tool_names = {getattr(t, "name", "") for t in (tools or [])}
        is_banking = "KB_search" in tool_names or "unlock_discoverable_agent_tool" in tool_names
        enhanced_policy = f"{domain_policy}\n\n{load_sop_prompt()}" if is_banking else domain_policy
        super().__init__(tools=tools, domain_policy=enhanced_policy, llm=llm, llm_args=llm_args)
        enable_guard = os.getenv("KNOWDESK_GUARD", "1") != "0"
        self.guard = GuardEngine(
            enforce_unlock=enable_guard and is_banking,
            enforce_verification=enable_guard,
            require_human_confirm_on_write=False,
        )
        self._sec_by_state: dict[int, SecurityState] = {}
        self.max_guard_retries = 2

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio not supported.")
        sec = self._sec_by_state.setdefault(id(state), SecurityState())

        if isinstance(message, (ToolMessage, MultiToolMessage)):
            _update_sec_from_last_turn(state, message, self.guard, sec)
            state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        else:
            state.messages.append(message)

        base_messages = list(state.system_messages) + list(state.messages)
        scratch_messages = list(base_messages)

        for attempt in range(self.max_guard_retries + 1):
            assistant_msg = generate(
                model=self.llm, tools=self.tools, messages=scratch_messages, call_name="agent_response", **self.llm_args
            )
            tcs = getattr(assistant_msg, "tool_calls", None) or []
            if not tcs:
                return assistant_msg

            blocked_decisions = []
            for tc in tcs:
                dec = self.guard.inspect(tc.name, tc.arguments, sec)
                if dec.action == "block":
                    blocked_decisions.append((tc, dec))

            if not blocked_decisions or attempt == self.max_guard_retries:
                if attempt > 0 and not blocked_decisions:
                    log_guard_event({"agent": "plain", "event": "self_corrected", "attempt": attempt, "new_tools": [t.name for t in tcs]})
                return assistant_msg

            scratch_messages.append(assistant_msg)
            for tc, dec in blocked_decisions:
                log_guard_event({
                    "agent": "plain", "event": "guard_blocked", "attempt": attempt,
                    "outer_tool": tc.name, "real_tool": dec.real_tool_name, "rule_id": dec.rule_id, "reason": dec.reason,
                })
                scratch_messages.append(make_tau_tool_message(tc.id, dec.error_payload.to_tool_message()))

        return assistant_msg

class _TurnGraphState(TypedDict, total=False):
    scratch_messages: Annotated[list[Any], operator.add]
    candidate_msg: Any
    guard_route: str
    attempt: int

class KnowDeskGraphTauAgent(KnowDeskPlainTauAgent):
    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio not supported.")
        sec = self._sec_by_state.setdefault(id(state), SecurityState())

        if isinstance(message, (ToolMessage, MultiToolMessage)):
            _update_sec_from_last_turn(state, message, self.guard, sec)
            state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        else:
            state.messages.append(message)

        base_messages = list(state.system_messages) + list(state.messages)

        def agent_node(gstate: _TurnGraphState) -> dict[str, Any]:
            att = gstate.get("attempt", 0)
            msgs = base_messages + gstate.get("scratch_messages", [])
            ans = generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response", **self.llm_args)
            return {"candidate_msg": ans, "attempt": att + 1}

        def guard_node(gstate: _TurnGraphState) -> dict[str, Any]:
            ans = gstate["candidate_msg"]
            att = gstate.get("attempt", 1)
            tcs = getattr(ans, "tool_calls", None) or []
            if not tcs:
                return {"guard_route": "done"}

            blocked = []
            for tc in tcs:
                dec = self.guard.inspect(tc.name, tc.arguments, sec)
                if dec.action == "block":
                    blocked.append((tc, dec))

            if not blocked or att > self.max_guard_retries:
                if att > 1 and not blocked:
                    log_guard_event({"agent": "langgraph", "event": "self_corrected", "attempt": att - 1, "new_tools": [t.name for t in tcs]})
                return {"guard_route": "done"}

            feedback_msgs: list[Any] = [ans]
            for tc, dec in blocked:
                log_guard_event({
                    "agent": "langgraph", "event": "guard_blocked", "attempt": att - 1,
                    "outer_tool": tc.name, "real_tool": dec.real_tool_name, "rule_id": dec.rule_id, "reason": dec.reason,
                })
                feedback_msgs.append(make_tau_tool_message(tc.id, dec.error_payload.to_tool_message()))
            return {"scratch_messages": feedback_msgs, "guard_route": "retry"}

        builder = StateGraph(_TurnGraphState)
        builder.add_node("agent", agent_node)
        builder.add_node("guard", guard_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", "guard")
        builder.add_conditional_edges("guard", lambda s: s.get("guard_route", "done"), {"retry": "agent", "done": END})
        graph = builder.compile(checkpointer=InMemorySaver())

        out = graph.invoke(
            {"scratch_messages": [], "attempt": 0},
            {"configurable": {"thread_id": f"turn_{id(state)}_{len(state.messages)}"}},
        )
        return out["candidate_msg"]

def register_knowdesk_components() -> None:
    register_litellm_pricing()
    mode = os.getenv("KNOWDESK_RETRIEVAL", "hybrid")
    patch_retrieval_mode(mode)
    existing = set(registry.get_agents()) if hasattr(registry, "get_agents") else set()
    for name, cls in [("knowdesk_plain", KnowDeskPlainTauAgent), ("knowdesk_graph", KnowDeskGraphTauAgent)]:
        if name not in existing:
            registry.register_agent_factory(cls, name)
