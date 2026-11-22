"""KnowDesk HTTP service — FastAPI routing and runtime wiring.

Sessions start unverified. Owner and tenant come from request headers, approval
requires a supervisor and the pending idempotency key, and per-session locking
prevents concurrent checkpoint overwrites on the same thread_id.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.types import Command

from knowdesk.agent.checkpointer import create_checkpointer
from knowdesk.agent.graph import build_knowdesk_graph
from knowdesk.agent.guard import GuardEngine
from knowdesk.kb.acl import AccessControlContext
from knowdesk.kb.hybrid_retriever import KnowDeskHybridRetriever
from knowdesk.mcp.client import MCPToolBackend
from knowdesk.obs.otel_setup import GenAITracer, current_session_id, get_ring_buffer, redact_pii
from knowdesk.service.demo_runtime import build_demo_runtime
from knowdesk.service.llm_gateway import (
    _first_int,
    _usage_dict,
    litellm_acomplete,
    litellm_complete,
    prefer_direct_dashscope,
)
from knowdesk.service.schemas import (
    LIVE_OPENAI_TOOLS,
    ApprovalRequest,
    JobRequest,
    MessageRequest,
    _messages_for_litellm,
    _openai_function_call,
)
from knowdesk.service.store import ServiceMetaStore, SessionLockRegistry
from knowdesk.tools.tool_wrapper import ReliableToolExecutor

__all__ = [
    "LIVE_OPENAI_TOOLS",
    "ApprovalRequest",
    "JobRequest",
    "MessageRequest",
    "build_demo_runtime",
    "build_live_runtime",
    "create_knowdesk_app",
    "default_db_path",
    "litellm_acomplete",
    "litellm_complete",
    "prefer_direct_dashscope",
    "request_acl",
    "_messages_for_litellm",
    "_openai_function_call",
]

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_REFUSAL = "Request refused by KnowDesk Guard. The response was blocked by output policy."

request_acl: contextvars.ContextVar[AccessControlContext | None] = contextvars.ContextVar(
    "knowdesk_request_acl", default=None
)
_request_tracer: contextvars.ContextVar[GenAITracer | None] = contextvars.ContextVar(
    "knowdesk_request_tracer", default=None
)


def default_db_path() -> str:
    return os.getenv("KNOWDESK_DB_PATH", "./data/knowdesk.sqlite")


def _prepare_db_path(db_path: str | Path | None) -> str:
    path = str(db_path or default_db_path())
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _observing_llm(inner: Callable[[list[dict[str, Any]]], dict[str, Any]]):
    def llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
        result = inner(messages)
        usage = _usage_dict(result.get("usage")) if isinstance(result, dict) else None
        prompt_tokens = _first_int(usage, "prompt_tokens") if usage else None
        completion_tokens = _first_int(usage, "completion_tokens") if usage else None
        if usage is not None and (prompt_tokens is not None or completion_tokens is not None):
            tracer = _request_tracer.get()
            if tracer is not None:
                cached = _first_int(usage, "cached_tokens")
                model_name = str(usage.get("model") or os.getenv("KNOWDESK_LLM_MODEL", "dashscope/qwen-plus"))
                tracer.record_llm_call(
                    model_name,
                    model_name,
                    input_tokens=prompt_tokens or 0,
                    output_tokens=completion_tokens or 0,
                    cached_tokens=cached or 0,
                    cost_rmb=0.0,
                )
        return result

    return llm


def build_live_runtime(db_path: str | Path | None = None) -> dict[str, Any]:
    """Wire LiteLLM, the hybrid retriever, and the persistent MCP tool backend."""
    backend = MCPToolBackend(db_path=db_path)

    def kb_search(args: dict[str, Any]) -> dict[str, Any]:
        acl = request_acl.get()
        hits = KnowDeskHybridRetriever.get_instance().search_sections(
            str(args.get("query", "")),
            k=int(args.get("top_k", 10)),
            acl=acl,
        )
        return {"content": json.dumps(hits, ensure_ascii=False), "error": False}

    def banking_tool(tool_name: str, args: dict[str, Any]) -> Any:
        session_id = current_session_id.get()
        if not session_id:
            raise RuntimeError("banking tool called without a session")
        return backend.call_tool(session_id, tool_name, args or {})

    tool_funcs: dict[str, Callable[[dict[str, Any]], Any]] = {"KB_search": kb_search}
    for name in (item["function"]["name"] for item in LIVE_OPENAI_TOOLS):
        if name == "KB_search":
            continue
        tool_funcs[name] = lambda args, name=name: banking_tool(name, args)
    return {"llm_fn": litellm_complete, "tool_funcs": tool_funcs, "mcp_backend": backend}


def _acl_from_headers(user_id: str, tenant_id: str, role: str) -> AccessControlContext:
    roles = frozenset({role}) if role else frozenset({"customer"})
    clearance = 3 if role == "supervisor" else 1
    return AccessControlContext(
        tenant_id=tenant_id or "default_tenant",
        user_id=user_id or "anonymous",
        clearance_level=clearance,
        roles=roles,
    )


def _run_registered_job(job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    if job_type == "regression_eval":
        from eval.gate import assert_frozen_integrity, load_splits

        splits = load_splits()
        assert_frozen_integrity(splits)
        return {
            "dev": len(splits["dev_ids"]),
            "test": len(splits["test_ids"]),
            "regression": len(splits["regression_ids"]),
            "params": params,
        }
    if job_type == "db_health":
        return {"ok": True, "params": params}
    raise ValueError(f"Unknown job_type: {job_type}")


def create_knowdesk_app(
    db_path: Path | str | None = None,
    llm_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    tool_funcs: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> FastAPI:
    path = _prepare_db_path(db_path)
    checkpointer = create_checkpointer(db_url_or_path=path)
    store = ServiceMetaStore(path)
    session_locks = SessionLockRegistry()
    worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="knowdesk-turn")

    if llm_fn is None or tool_funcs is None:
        runtime = build_demo_runtime() if os.getenv("KNOWDESK_DEMO_MODE") == "1" else build_live_runtime(path)
        if llm_fn is None:
            llm_fn = runtime["llm_fn"]
        if tool_funcs is None:
            tool_funcs = runtime["tool_funcs"]

    guard = GuardEngine(enforce_unlock=False, enforce_verification=True, require_human_confirm_on_write=True)
    executor = ReliableToolExecutor(idempotency_db_path=path)
    graph = build_knowdesk_graph(_observing_llm(llm_fn), tool_funcs, guard, executor, checkpointer=checkpointer)

    app = FastAPI(title="KnowDesk Banking Agent Service", version="1.0.0")
    tracers: dict[str, GenAITracer] = {}
    latencies_ms: deque[float] = deque(maxlen=2000)
    error_count = 0

    def _principal(request: Request) -> tuple[str, str, str]:
        expected = os.getenv("KNOWDESK_API_KEY")
        if expected and request.headers.get("x-api-key") != expected:
            raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Invalid API key."})
        return (
            request.headers.get("x-user-id", ""),
            request.headers.get("x-tenant-id", ""),
            request.headers.get("x-role", ""),
        )

    def _graph_exists(session_id: str) -> bool:
        state = graph.get_state({"configurable": {"thread_id": session_id}})
        return bool(state and state.values)

    def _require_session(session_id: str, user_id: str, tenant_id: str):
        row = store.get_session(session_id)
        if row is None and not _graph_exists(session_id):
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Session not found."})
        if row is None or row["owner_id"] != user_id or row["tenant_id"] != tenant_id:
            raise HTTPException(status_code=403, detail={"error_type": "forbidden", "message": "Session access denied."})
        return row

    def _finalize_reply(text: str) -> str:
        verdict = guard.inspect_output(text or "")
        if not verdict.passed:
            text = OUTPUT_REFUSAL
        return redact_pii(text)

    def _execute_job(job_id: str, job_type: str, params: dict[str, Any]) -> None:
        store.update_job(job_id, "running", None, None)
        try:
            result = _run_registered_job(job_type, params)
        except Exception as exc:
            store.update_job(job_id, "failed", None, str(exc))
            return
        store.update_job(job_id, "completed", result, None)

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        html_file = STATIC_DIR / "index.html"
        return html_file.read_text(encoding="utf-8") if html_file.exists() else "<h1>KnowDesk Service Running</h1>"

    @app.post("/sessions")
    def create_session(request: Request):
        user_id, tenant_id, _role = _principal(request)
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        store.create_session(session_id, user_id, tenant_id)
        tracers[session_id] = GenAITracer()
        graph.update_state(
            {"configurable": {"thread_id": session_id}},
            {"session_id": session_id, "messages": [], "verified": False},
        )
        return {"session_id": session_id, "verified": False, "owner_id": user_id, "tenant_id": tenant_id}

    @app.post("/sessions/{session_id}/messages")
    async def send_message(session_id: str, req: MessageRequest, request: Request):
        nonlocal error_count
        user_id, tenant_id, role = _principal(request)
        try:
            _require_session(session_id, user_id, tenant_id)
        except HTTPException:
            error_count += 1
            raise

        tracer = tracers.setdefault(session_id, GenAITracer())
        cfg = {"configurable": {"thread_id": session_id}}
        acl = _acl_from_headers(user_id, tenant_id, role)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def emit(frame: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def produce() -> None:
            session_token = current_session_id.set(session_id)
            tracer_token = _request_tracer.set(tracer)
            acl_token = request_acl.set(acl)
            t0 = time.perf_counter()
            try:
                with session_locks.for_session(session_id):
                    with tracer.start_span("knowdesk.turn", session_id=session_id):
                        final_output: dict[str, Any] = {}
                        for chunk in graph.stream(
                            {"session_id": session_id, "messages": [{"role": "user", "content": req.content}]},
                            cfg,
                            stream_mode="updates",
                        ):
                            if isinstance(chunk, dict):
                                for node_name, node_update in chunk.items():
                                    if isinstance(node_update, dict):
                                        final_output.update(node_update)
                                    node_summary = {
                                        "event": "node_transition",
                                        "node": node_name,
                                        "timestamp": time.time(),
                                        "has_tool_call": bool(node_update.get("pending_tool_call"))
                                        if isinstance(node_update, dict)
                                        else False,
                                    }
                                    emit(
                                        f"event: node_transition\ndata: {json.dumps(node_summary, ensure_ascii=False)}\n\n"
                                    )
                    state = graph.get_state(cfg)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    if state.next and "human_confirm" in state.next:
                        pending = (state.values.get("pending_tool_call") if state.values else None) or {}
                        payload = json.dumps(
                            {"event": "awaiting_approval", "pending_tool_call": pending}, ensure_ascii=False
                        )
                        emit(f"event: awaiting_approval\ndata: {payload}\n\n")
                    else:
                        messages = (state.values.get("messages") if state and state.values else None) or final_output.get(
                            "messages", []
                        )
                        reply = messages[-1].get("content", "") if messages else ""
                        payload = json.dumps({"event": "done", "reply": _finalize_reply(reply)}, ensure_ascii=False)
                        emit(f"event: done\ndata: {payload}\n\n")
            except Exception as exc:
                payload = json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False)
                emit(f"event: error\ndata: {payload}\n\n")
            finally:
                current_session_id.reset(session_token)
                _request_tracer.reset(tracer_token)
                request_acl.reset(acl_token)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker_pool.submit(produce)

        async def event_stream():
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                yield frame

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/approvals")
    def submit_approval(session_id: str, req: ApprovalRequest, request: Request):
        nonlocal error_count
        user_id, tenant_id, role = _principal(request)
        try:
            _require_session(session_id, user_id, tenant_id)
        except HTTPException:
            error_count += 1
            raise
        if role != "supervisor":
            raise HTTPException(
                status_code=403,
                detail={"error_type": "forbidden", "message": "Approval requires the supervisor role."},
            )
        cached = store.get_approval_result(session_id, req.idempotency_key)
        if cached is not None:
            return cached

        cfg = {"configurable": {"thread_id": session_id}}
        with session_locks.for_session(session_id):
            cached = store.get_approval_result(session_id, req.idempotency_key)
            if cached is not None:
                return cached
            state = graph.get_state(cfg)
            pending = (state.values.get("pending_tool_call") if state and state.values else None) or {}
            expected = pending.get("idem_key")
            if not expected or req.idempotency_key != expected:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_type": "conflict",
                        "message": "idempotency_key does not match the pending tool call.",
                    },
                )

            tracer = tracers.setdefault(session_id, GenAITracer())
            acl = _acl_from_headers(user_id, tenant_id, role)
            session_token = current_session_id.set(session_id)
            tracer_token = _request_tracer.set(tracer)
            acl_token = request_acl.set(acl)
            try:
                with tracer.start_span("knowdesk.approval_resume", decision=req.decision, session_id=session_id):
                    out = graph.invoke(
                        Command(
                            resume={
                                "decision": req.decision,
                                "reason": req.reason,
                                "modified_args": req.modified_args,
                            }
                        ),
                        cfg,
                    )
            finally:
                current_session_id.reset(session_token)
                _request_tracer.reset(tracer_token)
                request_acl.reset(acl_token)

            next_state = graph.get_state(cfg)
            if next_state and next_state.next and "human_confirm" in next_state.next:
                next_pending = (next_state.values.get("pending_tool_call") if next_state.values else None) or {}
                result = {
                    "session_id": session_id,
                    "status": "awaiting_approval",
                    "decision": req.decision,
                    "pending_tool_call": next_pending,
                    "reply": "",
                }
            else:
                messages = out.get("messages", [])
                reply = _finalize_reply(messages[-1].get("content", "") if messages else "")
                result = {"session_id": session_id, "status": "resumed", "decision": req.decision, "reply": reply}
            store.record_approval_result(session_id, req.idempotency_key, result)
            return result

    @app.get("/sessions/{session_id}/trace")
    def get_trace(session_id: str, request: Request):
        user_id, tenant_id, _role = _principal(request)
        _require_session(session_id, user_id, tenant_id)
        tracer = tracers.get(session_id, GenAITracer())
        ring_spans = get_ring_buffer().for_session(session_id)
        spans = [
            {"name": span.name, "duration_ms": round(span.duration_ms, 2), "attributes": span.attributes}
            for span in (ring_spans if ring_spans else tracer.spans)
        ]
        return {"session_id": session_id, "spans": spans, "summary": tracer.summary_metrics()}

    @app.get("/metrics")
    def get_metrics(request: Request):
        _principal(request)
        latencies = sorted(latencies_ms) if latencies_ms else [0.0]
        total_in = sum(tracer.summary_metrics()["total_input_tokens"] for tracer in tracers.values())
        total_cost = sum(tracer.summary_metrics()["total_cost_rmb"] for tracer in tracers.values())
        return {
            "sessions_count": store.count_sessions(),
            "requests_count": len(latencies_ms),
            "error_count": error_count,
            "latency_p50_ms": round(latencies[len(latencies) // 2], 2),
            "latency_p90_ms": round(latencies[int(len(latencies) * 0.9)], 2),
            "total_input_tokens": total_in,
            "total_cost_rmb": round(total_cost, 4),
        }

    @app.post("/jobs")
    def create_job(req: JobRequest, background: BackgroundTasks, request: Request):
        _principal(request)
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        payload = store.insert_job(job_id, req.job_type)
        background.add_task(_execute_job, job_id, req.job_type, req.params)
        return payload

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        _principal(request)
        payload = store.get_job(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Job not found."})
        return payload

    return app


def __getattr__(name: str):
    if name == "app":
        application = create_knowdesk_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
