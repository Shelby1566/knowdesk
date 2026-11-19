import asyncio
from agent.graph import build_knowdesk_graph
from agent.guard import GuardEngine, SecurityState
from kb.bm25 import BM25Index
from kb.tokenize import tokenize
from mcp_client.client import MCPToolBackend
from tools.tool_wrapper import ReliableToolExecutor

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
