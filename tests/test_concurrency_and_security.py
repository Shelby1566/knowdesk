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


def test_p0_6_demo_mode_end_to_end_hitl_without_api_key(monkeypatch, tmp_path):
    """验证 KNOWDESK_DEMO_MODE=1 无需 API Key 即可跑通完整 SSE -> 人工审批 -> 幂等恢复闭环"""
    import json
    from fastapi.testclient import TestClient
    from knowdesk.service.app import create_knowdesk_app

    monkeypatch.setenv("KNOWDESK_DEMO_MODE", "1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    app = create_knowdesk_app(db_path=tmp_path / "demo_mode.sqlite")
    client = TestClient(app)
    headers = {"x-user-id": "sup_demo", "x-tenant-id": "demo_bank", "x-role": "supervisor"}
    sess = client.post("/sessions", headers=headers).json()["session_id"]

    sse_resp = client.post(f"/sessions/{sess}/messages", headers=headers, json={"content": "Please freeze my lost card"})
    assert sse_resp.status_code == 200
    assert "awaiting_approval" in sse_resp.text

    idem_key = None
    for line in sse_resp.text.splitlines():
        if line.startswith("data:") and "awaiting_approval" in line:
            idem_key = json.loads(line[len("data:"):].strip())["pending_tool_call"]["idem_key"]
    assert idem_key

    resumed = client.post(
        f"/sessions/{sess}/approvals",
        headers=headers,
        json={"decision": "approve", "idempotency_key": idem_key},
    )
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "resumed"
    assert "card_demo_8842" in body["reply"]


def test_p1_acl_prefiltering_prevents_crowding_out():
    """验证 BM25 与 FAISS 向量检索均支持 ACL 前置过滤，高权限文档不会挤占低权限用户的 Top-K 召回"""
    import numpy as np
    from knowdesk.kb.bm25 import BM25Index
    from knowdesk.kb.vectorstore import FaissVectorStore

    ids = [f"secret_{i}" for i in range(5)] + ["public_1"]
    texts = ["platinum annual fee waiver policy secret"] * 5 + ["annual fee waiver policy"]
    bm25 = BM25Index.build(ids, texts)
    # Without pre-filtering, k=1 is dominated by secret_0
    assert bm25.search("platinum annual fee waiver", k=1)[0][0].startswith("secret_")
    # With pre-filtering, low-clearance user still gets public_1 at k=1
    bm25_allowed = bm25.search("platinum annual fee waiver", k=1, allowed_ids={"public_1"})
    assert len(bm25_allowed) == 1 and bm25_allowed[0][0] == "public_1"

    store = FaissVectorStore(dim=4, index_type="flat")
    vecs = np.array(
        [[1.0, 0.0, 0.0, 0.0]] * 5 + [[0.6, 0.8, 0.0, 0.0]],
        dtype=np.float32,
    )
    store.add(ids, vecs)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert store.search(q, k=1)[0][0].startswith("secret_")
    vec_allowed = store.search(q, k=1, allowed_ids={"public_1"})
    assert len(vec_allowed) == 1 and vec_allowed[0][0] == "public_1"


def test_p0_chained_two_write_approvals_and_canonical_idem_key(tmp_path):
    """验证单轮连续两个高风险写操作时，审批恢复返回 awaiting_approval 而不断链，且幂等键统一"""
    import json
    from fastapi.testclient import TestClient
    from knowdesk.service.app import create_knowdesk_app

    executed = []
    step = 0

    def llm(msgs):
        nonlocal step
        step += 1
        if step == 1:
            return {"content": "", "tool_calls": [{"id": "c0", "name": "log_verification", "arguments": {"user_id": "u1"}}]}
        if step == 2:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "freeze_card", "arguments": {"card_id": "card_1"}},
                    {"id": "c2", "name": "freeze_card", "arguments": {"card_id": "card_2"}},
                ],
            }
        return {"content": "Both cards frozen.", "tool_calls": []}

    tools = {
        "log_verification": lambda a: {"ok": True},
        "freeze_card": lambda a: (executed.append(a["card_id"]), {"frozen": a["card_id"]})[1],
    }
    app = create_knowdesk_app(db_path=tmp_path / "chained_hitl.sqlite", llm_fn=llm, tool_funcs=tools)
    client = TestClient(app)
    headers = {"x-user-id": "sup_1", "x-tenant-id": "t1", "x-role": "supervisor"}
    sess = client.post("/sessions", headers=headers).json()["session_id"]

    sse = client.post(f"/sessions/{sess}/messages", headers=headers, json={"content": "freeze both cards"})
    assert "awaiting_approval" in sse.text
    first_key = None
    for line in sse.text.splitlines():
        if line.startswith("data:") and "awaiting_approval" in line:
            first_key = json.loads(line[len("data:"):].strip())["pending_tool_call"]["idem_key"]
    assert first_key

    # Approve first write -> must pause again for second write
    r1 = client.post(
        f"/sessions/{sess}/approvals",
        headers=headers,
        json={"decision": "approve", "idempotency_key": first_key},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "awaiting_approval"
    second_key = body1["pending_tool_call"]["idem_key"]
    assert second_key and second_key != first_key
    assert executed == ["card_1"]

    # Approve second write -> completes
    r2 = client.post(
        f"/sessions/{sess}/approvals",
        headers=headers,
        json={"decision": "approve", "idempotency_key": second_key},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "resumed"
    assert "Both cards frozen" in r2.json()["reply"]
    assert executed == ["card_1", "card_2"]


def test_p1_mcp_persistent_client_reuses_session(tmp_path):
    """验证 MCPToolBackend 复用持久 JSON-RPC 客户端会话而不是每次新建连接"""
    backend = MCPToolBackend(db_path=tmp_path / "persist_mcp.sqlite")
    backend.init_session("persist_s1")
    try:
        h1 = backend.get_db_hash("persist_s1")
        first_client = backend._active_client
        h2 = backend.get_db_hash("persist_s1")
        second_client = backend._active_client
        assert h1 == h2 and len(h1) > 0
        assert first_client is not None and first_client is second_client
    finally:
        backend.close()
