import json
import operator
import os
import uuid
import weakref
from pathlib import Path
from typing import Annotated, Any, TypedDict

import litellm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from tau2.agent.llm_agent import LLMAgent, generate
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage
from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever
from tau2.registry import registry

from agent.context import compress_context_watermark
from agent.guard import GuardEngine, SecurityState
from agent.workflow import stage_directive
from kb.hybrid_retriever import KnowDeskHybridRetriever
from tools.schemas import is_tool_result_success

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

SAFE_REFUSAL = (
    "Request refused by KnowDesk Guard. The proposed tool call violates policy and was not executed."
)
_SEC_CACHE_MAX = 256


def log_guard_event(event: dict[str, Any]) -> None:
    GUARD_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GUARD_TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class _SecurityCache:
    """按 session_id 或对象上的显式 key 保存核身状态，并用 weakref 回收已销毁的回合。"""

    def __init__(self) -> None:
        self._by_key: dict[str, SecurityState] = {}
        self._order: list[str] = []
        self._refs: dict[str, weakref.ref] = {}

    def _drop(self, key: str) -> None:
        self._by_key.pop(key, None)
        self._refs.pop(key, None)
        if key in self._order:
            self._order.remove(key)

    def for_state(self, state: Any) -> SecurityState:
        for key, ref in list(self._refs.items()):
            if ref() is None:
                self._drop(key)
        key = self._key_for(state)
        sec = self._by_key.get(key)
        if sec is None:
            sec = SecurityState()
            self._by_key[key] = sec
            self._order.append(key)
            try:
                self._refs[key] = weakref.ref(state, lambda _ref, bound=key: self._drop(bound))
            except TypeError:
                pass
            while len(self._order) > _SEC_CACHE_MAX:
                self._drop(self._order[0])
        return sec

    def _key_for(self, state: Any) -> str:
        sid = getattr(state, "session_id", None)
        if sid:
            return f"sid:{sid}"
        existing = getattr(state, "_knowdesk_session_key", None)
        if existing:
            return f"obj:{existing}"
        key = uuid.uuid4().hex
        try:
            object.__setattr__(state, "_knowdesk_session_key", key)
        except Exception:
            key = f"ephemeral-{uuid.uuid4().hex}"
        return f"obj:{key}"


def _fail_closed_message() -> AssistantMessage:
    return AssistantMessage(role="assistant", content=SAFE_REFUSAL, tool_calls=[])


def _thread_fragment(state: Any) -> str:
    sid = getattr(state, "session_id", None)
    if sid:
        return str(sid)
    return str(getattr(state, "_knowdesk_session_key", "turn"))


def _with_stage_directive(messages: list[Any], directive: str) -> list[Any]:
    if not directive:
        return list(messages)
    if messages and isinstance(messages[0], SystemMessage):
        first = messages[0].model_copy(update={"content": f"{messages[0].content or ''}\n\n{directive}"})
        return [first, *messages[1:]]
    return [SystemMessage(role="system", content=directive), *messages]


def _compact_llm_view(messages: list[Any]) -> list[Any]:
    """Compress only the copy sent to the model. The stored trajectory stays intact."""
    if os.getenv("KNOWDESK_COMPACTION", "0") != "1":
        return messages
    as_dicts: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            as_dicts.append(dict(message))
            continue
        role = str(getattr(message, "role", "") or "")
        as_dicts.append({
            "role": role,
            "content": getattr(message, "content", "") or "",
            "id": getattr(message, "id", None),
            "tool_call_id": getattr(message, "id", None) if role == "tool" else None,
        })
    compacted, changed = compress_context_watermark(as_dicts)
    if not changed:
        return messages
    rewritten: list[Any] = []
    for original, updated in zip(messages, compacted, strict=True):
        new_content = updated.get("content", "")
        old_content = original.get("content", "") if isinstance(original, dict) else (getattr(original, "content", "") or "")
        if new_content == old_content:
            rewritten.append(original)
            continue
        if isinstance(original, dict):
            copied = dict(original)
            copied["content"] = new_content
            rewritten.append(copied)
        else:
            rewritten.append(original.model_copy(update={"content": new_content}))
    return rewritten


def _model_turn(agent: Any, messages: list[Any]) -> Any:
    return generate(
        model=agent.llm,
        tools=agent.tools,
        messages=_compact_llm_view(messages),
        call_name="agent_response",
        **agent.llm_args,
    )


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

def install_section_retrieval(retriever) -> None:
    """混合检索仍返回文档 id，供在线文档召回使用；注入模型的正文换成命中的 H2 章节。"""
    from tau2.knowledge.pipeline import RetrievalPipeline

    if not hasattr(install_section_retrieval, "_orig_content"):
        install_section_retrieval._orig_content = RetrievalPipeline.get_document_content

    def _hybrid_retrieve(self, input_data: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, float]]:
        query = input_data.get(self.query_key, "")
        if not query or not str(query).strip():
            state["knowdesk_section_excerpts"] = {}
            return []
        top_k = int(input_data.get("top_k", getattr(self, "top_k", 10)))
        hits = retriever.search_sections(str(query), k=top_k)
        state["knowdesk_section_excerpts"] = {hit["id"]: hit["content"] for hit in hits}
        return [(hit["id"], float(hit["score"])) for hit in hits]

    def _hybrid_retrieve_batch(self, input_data_list: list[dict[str, Any]], state: dict[str, Any]):
        return [_hybrid_retrieve(self, item, state) for item in input_data_list]

    def _section_content(self, doc_id: str):
        excerpt = (self.state.get("knowdesk_section_excerpts") or {}).get(doc_id)
        if excerpt:
            return excerpt
        return install_section_retrieval._orig_content(self, doc_id)

    BM25Retriever.retrieve = _hybrid_retrieve
    BM25Retriever.retrieve_batch = _hybrid_retrieve_batch
    RetrievalPipeline.get_document_content = _section_content


def restore_builtin_retrieval() -> None:
    BM25Retriever.retrieve = _ORIG_BM25_RETRIEVE
    BM25Retriever.retrieve_batch = _ORIG_BM25_RETRIEVE_BATCH
    if hasattr(install_section_retrieval, "_orig_content"):
        from tau2.knowledge.pipeline import RetrievalPipeline
        RetrievalPipeline.get_document_content = install_section_retrieval._orig_content


def patch_retrieval_mode(mode: str = "hybrid") -> None:
    if mode == "hybrid":
        install_section_retrieval(KnowDeskHybridRetriever.get_instance())
    else:
        restore_builtin_retrieval()

def _update_sec_from_last_turn(state: Any, incoming_msg: Any, guard: GuardEngine, sec: SecurityState) -> None:
    """只对 tool_call_id 精确匹配、且结果判定成功的调用更新核身与解锁状态。"""
    tool_msgs = incoming_msg.tool_messages if isinstance(incoming_msg, MultiToolMessage) else [incoming_msg]
    success_by_id: dict[str, str] = {}
    for tm in tool_msgs:
        tm_id = str(getattr(tm, "id", "") or "")
        if not tm_id:
            continue
        tm_content = str(getattr(tm, "content", "") or "")
        if is_tool_result_success(tm_content, error_flag=bool(getattr(tm, "error", False))):
            success_by_id[tm_id] = tm_content

    for msg in reversed(state.messages):
        if isinstance(msg, AssistantMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tc_id = str(getattr(tc, "id", "") or "")
                if tc_id not in success_by_id:
                    continue
                guard.update_state_after_success(
                    tc.name, tc.arguments, sec, result_content=success_by_id[tc_id]
                )
            break

class KnowDeskPlainTauAgent(LLMAgent):
    def __init__(self, tools, domain_policy: str, llm: str, llm_args: dict | None = None, **kwargs):
        register_litellm_pricing()
        tool_names = {getattr(t, "name", "") for t in (tools or [])}
        is_banking = "KB_search" in tool_names or "unlock_discoverable_agent_tool" in tool_names
        enhanced_policy = f"{domain_policy}\n\n{load_sop_prompt()}" if is_banking else domain_policy
        super().__init__(tools=tools, domain_policy=enhanced_policy, llm=llm, llm_args=llm_args)
        self._banking = is_banking
        self._available_tools = {tool.name for tool in self.tools}
        enable_guard = os.getenv("KNOWDESK_GUARD", "1") != "0"
        self.guard = GuardEngine(
            enforce_unlock=enable_guard and is_banking,
            enforce_verification=enable_guard,
            require_human_confirm_on_write=False,
        )
        self._security = _SecurityCache()
        self.max_guard_retries = 2

    def _project(self, messages: list[Any], sec: SecurityState) -> list[Any]:
        if not self._banking:
            return list(messages)
        directive = stage_directive(messages, sec, available_tools=self._available_tools)
        return _with_stage_directive(messages, directive)

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio not supported.")
        sec = self._security.for_state(state)

        if isinstance(message, (ToolMessage, MultiToolMessage)):
            _update_sec_from_last_turn(state, message, self.guard, sec)
            state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        else:
            state.messages.append(message)

        base_messages = list(state.system_messages) + list(state.messages)
        scratch_messages = self._project(base_messages, sec)

        assistant_msg = None
        for attempt in range(self.max_guard_retries + 1):
            assistant_msg = _model_turn(self, scratch_messages)
            tcs = getattr(assistant_msg, "tool_calls", None) or []
            if not tcs:
                return assistant_msg

            blocked_decisions = []
            for tc in tcs:
                dec = self.guard.inspect(tc.name, tc.arguments, sec)
                if dec.action == "block":
                    blocked_decisions.append((tc, dec))

            if not blocked_decisions:
                if attempt > 0:
                    log_guard_event({"agent": "plain", "event": "self_corrected", "attempt": attempt, "new_tools": [t.name for t in tcs]})
                return assistant_msg

            if attempt == self.max_guard_retries:
                log_guard_event({
                    "agent": "plain",
                    "event": "guard_fail_closed",
                    "attempt": attempt,
                    "blocked_tools": [tc.name for tc, _ in blocked_decisions],
                })
                return _fail_closed_message()

            scratch_messages.append(assistant_msg)
            for tc, dec in blocked_decisions:
                log_guard_event({
                    "agent": "plain", "event": "guard_blocked", "attempt": attempt,
                    "outer_tool": tc.name, "real_tool": dec.real_tool_name, "rule_id": dec.rule_id, "reason": dec.reason,
                })
                scratch_messages.append(make_tau_tool_message(tc.id, dec.error_payload.to_tool_message()))

        return assistant_msg or _fail_closed_message()

class _TurnGraphState(TypedDict, total=False):
    scratch_messages: Annotated[list[Any], operator.add]
    candidate_msg: Any
    guard_route: str
    attempt: int

class KnowDeskGraphTauAgent(KnowDeskPlainTauAgent):
    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio not supported.")
        sec = self._security.for_state(state)

        if isinstance(message, (ToolMessage, MultiToolMessage)):
            _update_sec_from_last_turn(state, message, self.guard, sec)
            state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        else:
            state.messages.append(message)

        base_messages = list(state.system_messages) + list(state.messages)

        def agent_node(gstate: _TurnGraphState) -> dict[str, Any]:
            att = gstate.get("attempt", 0)
            msgs = self._project(base_messages + gstate.get("scratch_messages", []), sec)
            ans = _model_turn(self, msgs)
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

            if not blocked:
                if att > 1:
                    log_guard_event({"agent": "langgraph", "event": "self_corrected", "attempt": att - 1, "new_tools": [t.name for t in tcs]})
                return {"guard_route": "done"}

            if att > self.max_guard_retries:
                log_guard_event({
                    "agent": "langgraph",
                    "event": "guard_fail_closed",
                    "attempt": att - 1,
                    "blocked_tools": [tc.name for tc, _ in blocked],
                })
                return {"candidate_msg": _fail_closed_message(), "guard_route": "done"}

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
            {"configurable": {"thread_id": f"turn_{_thread_fragment(state)}_{len(state.messages)}"}},
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
