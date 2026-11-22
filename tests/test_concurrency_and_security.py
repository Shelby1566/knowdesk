import asyncio
from knowdesk.agent.graph import build_knowdesk_graph
from knowdesk.agent.guard import GuardEngine, SecurityState
from knowdesk.kb.bm25 import BM25Index
from knowdesk.kb.tokenize import tokenize
from knowdesk.mcp.client import MCPToolBackend
from knowdesk.tools.tool_wrapper import ReliableToolExecutor

def test_p0_1_langgraph_parallel_tool_calls():
    """验证 LangGraph 支持单轮并发返回多个 tool_calls 并按序执行回传全部 ToolMessage"""
    called_order = []
    tools = {
        "get_user_details": lambda a: (called_order.append("user"), {"uid": "u1"})[1],
        "get_account_details": lambda a: (called_order.append("acct"), {"aid": "a1"})[1],
    }
    script = [
        {
            "content": "",
            "tool_calls": [
                {"id": "tc_1", "name": "get_user_details", "arguments": {"user_id": "u1"}},
                {"id": "tc_2", "name": "get_account_details", "arguments": {"account_id": "a1"}},
            ],
        },
        {"content": "Both queried.", "tool_calls": []},
    ]
    idx = 0
    def llm(msgs):
        nonlocal idx
        r = script[min(idx, len(script) - 1)]
        idx += 1
        return r

    guard = GuardEngine(enforce_unlock=False, enforce_verification=True, require_human_confirm_on_write=False)
    g = build_knowdesk_graph(llm, tools, guard, ReliableToolExecutor())
    out = g.invoke(
        {"session_id": "par_1", "messages": [{"role": "user", "content": "check both"}]},
        {"configurable": {"thread_id": "par_1"}},
    )
    assert called_order == ["user", "acct"]
    tool_msg_ids = [m.get("tool_call_id") for m in out["messages"] if m.get("role") == "tool"]
    assert tool_msg_ids == ["tc_1", "tc_2"]

def test_p0_2_cjk_multilingual_tokenization_and_bm25():
    """验证分词器与倒排索引 BM25 支持中文检索"""
    tokens = tokenize("信用卡年费 waiver 95元")
    assert "信" in tokens and "waiver" in tokens and "95" in tokens
    idx = BM25Index.build(["d1", "d2"], ["白金信用卡年费减免政策", "房贷提前还款违约金说明"])
    res = idx.search("信用卡年费", k=1)
    assert res[0][0] == "d1" and res[0][1] > 0

def test_p0_3_mcp_client_inside_running_async_loop(tmp_path):
    """验证在运行中的 asyncio 事件循环（如 FastAPI async 路由）内调用 MCPToolBackend 不崩溃"""
    backend = MCPToolBackend(db_path=tmp_path / "async_mcp.sqlite")
    backend.init_session("async_s1")

    async def _inside_loop():
        h1 = await backend.get_db_hash_async("async_s1")
        h2 = backend.get_db_hash("async_s1")
        assert h1 == h2 and len(h1) > 0

    asyncio.run(_inside_loop())

def test_p1_1_strict_verification_no_param_bypass():
    """验证普通工具仅携带 user_id 参数绝不会触发自动核身越权"""
    guard = GuardEngine()
    sec = SecurityState(verified=False)
    guard.update_state_after_success("some_untrusted_tool", {"user_id": "u_123"}, sec)
    assert sec.verified is False
    guard.update_state_after_success("log_verification", {"user_id": "u_123"}, sec)
    assert sec.verified is True

def test_p2_4_unexpected_tool_exception_caught_and_lru_cap():
    """验证工具抛出未知 RuntimeError 时被包装为结构化 upstream_error 而不崩溃，且 LRU 容量受控"""
    ex = ReliableToolExecutor(max_cache_size=3)
    out = ex.execute("s1", "boom_tool", {}, lambda a: (_ for _ in ()).throw(RuntimeError("db connection lost")))
    assert '"error_type":"upstream_error"' in out

    for i in range(5):
        ex.execute("s1", "w_tool", {"i": i}, lambda a: {"ok": True}, is_write=True)
    assert len(ex.idempotency_store) <= 3


def test_p0_4_executor_per_session_concurrency_and_sqlite_wal(tmp_path):
    """验证 ReliableToolExecutor 不同 session 可真正并发执行（不互相阻塞），且 SQLite 开启 WAL"""
    import sqlite3
    import threading
    import time

    db_file = tmp_path / "wal_idem.sqlite"
    ex = ReliableToolExecutor(idempotency_db_path=db_file)
    mode = sqlite3.connect(db_file).execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"

    barrier = threading.Barrier(2, timeout=2.0)

    def concurrent_write(args):
        barrier.wait()
        return {"ok": True, "who": args["who"]}

    results = {}

    def worker(sid):
        results[sid] = ex.execute(sid, "freeze_card", {"who": sid}, concurrent_write, is_write=True)

    t1 = threading.Thread(target=worker, args=("sess_a",))
    t2 = threading.Thread(target=worker, args=("sess_b",))
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert "sess_a" in results and "sess_b" in results
    assert '"ok": true' in results["sess_a"] and '"ok": true' in results["sess_b"]


def test_p0_5_approval_resume_preserves_request_acl(tmp_path):
    """验证 submit_approval 恢复图执行时保留 request_acl 上下文，不丢失租户与角色权限"""
    from fastapi.testclient import TestClient
    from knowdesk.service.app import create_knowdesk_app, request_acl

    seen_acls = []
    step = 0

    def llm(msgs):
        nonlocal step
        step += 1
        if step == 1:
            return {"content": "", "tool_calls": [{"id": "c1", "name": "log_verification", "arguments": {"user_id": "u1"}}]}
        if step == 2:
            return {"content": "", "tool_calls": [{"id": "c2", "name": "freeze_card", "arguments": {"card_id": "card_1"}}]}
        if step == 3:
            return {"content": "", "tool_calls": [{"id": "c3", "name": "KB_search", "arguments": {"query": "freeze policy"}}]}
        return {"content": "Done after approval.", "tool_calls": []}

    def kb_search_stub(args):
        seen_acls.append(request_acl.get())
        return {"content": "[]", "error": False}

    tools = {
        "log_verification": lambda a: {"ok": True},
        "freeze_card": lambda a: {"frozen": True},
        "KB_search": kb_search_stub,
    }
    app = create_knowdesk_app(db_path=tmp_path / "acl_resume.sqlite", llm_fn=llm, tool_funcs=tools)
    client = TestClient(app)

    headers = {"x-user-id": "sup_1", "x-tenant-id": "tenant_bank", "x-role": "supervisor"}
    sess = client.post("/sessions", headers=headers).json()["session_id"]
    sse_resp = client.post(f"/sessions/{sess}/messages", headers=headers, json={"content": "freeze my card"})
    assert sse_resp.status_code == 200
    assert "awaiting_approval" in sse_resp.text

    import json
    idem_key = None
    for line in sse_resp.text.splitlines():
        if line.startswith("data:") and "awaiting_approval" in line:
            payload = json.loads(line[len("data:"):].strip())
            idem_key = payload["pending_tool_call"]["idem_key"]
    assert idem_key

    resume_resp = client.post(
        f"/sessions/{sess}/approvals",
        headers=headers,
        json={"decision": "approve", "idempotency_key": idem_key},
    )
    assert resume_resp.status_code == 200
    assert len(seen_acls) == 1 and seen_acls[0] is not None
    assert seen_acls[0].tenant_id == "tenant_bank"
    assert seen_acls[0].user_id == "sup_1"
    assert "supervisor" in seen_acls[0].roles
