import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from agent.graph import build_knowdesk_graph
from agent.guard import GuardEngine
from obs.otel_setup import GenAITracer
from tools.tool_wrapper import ReliableToolExecutor

STATIC_DIR = Path(__file__).resolve().parent / "static"

class MessageRequest(BaseModel):
    content: str

class ApprovalRequest(BaseModel):
    decision: str = "approve"
    reason: str = ""
    modified_args: dict[str, Any] | None = None
    idempotency_key: str

class JobRequest(BaseModel):
    job_type: str = "regression_eval"
    params: dict[str, Any] = {}

def create_knowdesk_app(
    db_path: Path | str = ":memory:",
    llm_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    tool_funcs: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> FastAPI:
    app = FastAPI(title="KnowDesk Banking Agent Service", version="1.0.0")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    guard = GuardEngine(enforce_unlock=False, enforce_verification=True, require_human_confirm_on_write=True)
    executor = ReliableToolExecutor()

    default_tools = tool_funcs or {
        "get_user_details": lambda a: {"user_id": a.get("user_id", "u_01"), "verified": True, "name": "Sera Chen"},
        "freeze_card": lambda a: {"status": "frozen", "card_id": a.get("card_id", "card_01")},
    }

    def default_llm(msgs: list[dict[str, Any]]) -> dict[str, Any]:
        last = msgs[-1] if msgs else {}
        if last.get("role") == "user" and "freeze" in str(last.get("content", "")).lower():
            return {"content": "", "tool_calls": [{"id": "tc_1", "name": "freeze_card", "arguments": {"card_id": "card_5320"}}]}
        return {"content": "Your request has been processed according to bank policy.", "tool_calls": []}

    active_llm = llm_fn or default_llm
    graph = build_knowdesk_graph(active_llm, default_tools, guard, executor, checkpointer=checkpointer)

    sessions_meta: dict[str, dict[str, Any]] = {}
    tracers: dict[str, GenAITracer] = {}
    approval_cache: dict[str, dict[str, Any]] = {}
    jobs_store: dict[str, dict[str, Any]] = {}
    latencies_ms: list[float] = []
    error_count = 0
    MAX_CACHE_SIZE = 1000

    def _set_lru(d: dict[str, Any], k: str, v: Any, max_size: int = MAX_CACHE_SIZE) -> None:
        if len(d) >= max_size and k not in d:
            first = next(iter(d))
            del d[first]
        d[k] = v

    def _session_exists(sid: str) -> bool:
        if sid in sessions_meta:
            return True
        cfg = {"configurable": {"thread_id": sid}}
        st = graph.get_state(cfg)
        return bool(st and st.values)

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        html_file = STATIC_DIR / "index.html"
        return html_file.read_text(encoding="utf-8") if html_file.exists() else "<h1>KnowDesk Service Running</h1>"

    @app.post("/sessions")
    def create_session(verified: bool = True):
        sid = f"sess_{uuid.uuid4().hex[:10]}"
        _set_lru(sessions_meta, sid, {"created_at": time.time(), "verified": verified})
        _set_lru(tracers, sid, GenAITracer())
        cfg = {"configurable": {"thread_id": sid}}
        graph.update_state(cfg, {"session_id": sid, "messages": [], "verified": verified})
        return {"session_id": sid, "verified": verified}

    @app.post("/sessions/{session_id}/messages")
    def send_message(session_id: str, req: MessageRequest):
        nonlocal error_count
        if not _session_exists(session_id):
            error_count += 1
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Session not found."})

        tracer = tracers.setdefault(session_id, GenAITracer())
        cfg = {"configurable": {"thread_id": session_id}}

        def event_stream():
            t0 = time.perf_counter()
            with tracer.start_span("knowdesk.turn", session_id=session_id):
                out = graph.invoke(
                    {"session_id": session_id, "messages": [{"role": "user", "content": req.content}]}, cfg
                )
                tracer.record_llm_call("qwen3.8-flash", "qwen3.8-flash", input_tokens=250, output_tokens=45, cached_tokens=120, cost_rmb=0.0003)

            st = graph.get_state(cfg)
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed)
            if len(latencies_ms) > MAX_CACHE_SIZE:
                latencies_ms.pop(0)

            if st.next and "human_confirm" in st.next:
                pending = st.values.get("pending_tool_call") or {}
                payload = json.dumps({"event": "awaiting_approval", "pending_tool_call": pending})
                yield f"event: awaiting_approval\ndata: {payload}\n\n"
            else:
                msgs = out.get("messages", [])
                reply = msgs[-1].get("content", "") if msgs else ""
                payload = json.dumps({"event": "done", "reply": reply})
                yield f"event: done\ndata: {payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/approvals")
    def submit_approval(session_id: str, req: ApprovalRequest):
        nonlocal error_count
        if not _session_exists(session_id):
            error_count += 1
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Session not found."})

        dedup_key = f"{session_id}::{req.idempotency_key}"
        if dedup_key in approval_cache:
            return approval_cache[dedup_key]

        tracer = tracers.setdefault(session_id, GenAITracer())
        cfg = {"configurable": {"thread_id": session_id}}
        with tracer.start_span("knowdesk.approval_resume", decision=req.decision):
            out = graph.invoke(
                Command(resume={"decision": req.decision, "reason": req.reason, "modified_args": req.modified_args}),
                cfg,
            )

        msgs = out.get("messages", [])
        reply = msgs[-1].get("content", "") if msgs else ""
        result = {"session_id": session_id, "status": "resumed", "decision": req.decision, "reply": reply}
        _set_lru(approval_cache, dedup_key, result)
        return result

    @app.get("/sessions/{session_id}/trace")
    def get_trace(session_id: str):
        if not _session_exists(session_id):
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Session not found."})
        tr = tracers.get(session_id, GenAITracer())
        return {
            "session_id": session_id,
            "spans": [{"name": s.name, "duration_ms": round(s.duration_ms, 2), "attributes": s.attributes} for s in tr.spans],
            "summary": tr.summary_metrics(),
        }

    @app.get("/metrics")
    def get_metrics():
        lats = sorted(latencies_ms) if latencies_ms else [0.0]
        tot_in = sum(t.summary_metrics()["total_input_tokens"] for t in tracers.values())
        tot_cost = sum(t.summary_metrics()["total_cost_rmb"] for t in tracers.values())
        return {
            "sessions_count": len(tracers),
            "requests_count": len(latencies_ms),
            "error_count": error_count,
            "latency_p50_ms": round(lats[len(lats) // 2], 2),
            "latency_p90_ms": round(lats[int(len(lats) * 0.9)], 2),
            "total_input_tokens": tot_in,
            "total_cost_rmb": round(tot_cost, 4),
        }

    @app.post("/jobs")
    def create_job(req: JobRequest):
        jid = f"job_{uuid.uuid4().hex[:8]}"
        res = {"job_id": jid, "job_type": req.job_type, "status": "completed", "result": {"pass_1": 0.40}}
        _set_lru(jobs_store, jid, res)
        return res

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs_store:
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Job not found."})
        return jobs_store[job_id]

    return app

app = create_knowdesk_app()
