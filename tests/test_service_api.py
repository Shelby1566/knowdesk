from pathlib import Path
from fastapi.testclient import TestClient
from service.app import create_knowdesk_app

def test_m25_full_sse_approval_idempotency_and_sqlite_restart(tmp_path: Path):
    """
    M25 Test case: 
    ① 新建会话 -> 发消息触发 SSE awaiting_approval -> 审批恢复完成闭环
    ② 同一审批幂等键重复提交两次，底层写操作只执行 1 次
    ③ 服务重启（新建 App 实例连接同一 SQLite 文件），原 session_id 状态完整恢复并可继续审批
    ④ 不存在的 session 返回 404 与结构化错误
    """
    db_file = tmp_path / "knowdesk_ckpt.sqlite"
    write_counter = 0

    def counting_freeze(args):
        nonlocal write_counter
        write_counter += 1
        return {"frozen": args.get("card_id")}

    app1 = create_knowdesk_app(db_path=db_file, tool_funcs={"freeze_card": counting_freeze})
    client1 = TestClient(app1)

    # ④ 不存在的会话返回 404
    r_404 = client1.get("/sessions/non_existent/trace")
    assert r_404.status_code == 404

    # ① 新建会话并发送高风险冻结请求 -> 收到 SSE awaiting_approval 事件，此时写操作计数为 0
    sid = client1.post("/sessions").json()["session_id"]
    sse_resp = client1.post(f"/sessions/{sid}/messages", json={"content": "Please freeze my card"})
    assert sse_resp.status_code == 200
    assert "event: awaiting_approval" in sse_resp.text
    assert write_counter == 0

    # ③ 模拟服务进程重启：销毁 app1，用同一个 SQLite 文件创建全新 app2 实例！
    app2 = create_knowdesk_app(db_path=db_file, tool_funcs={"freeze_card": counting_freeze})
    client2 = TestClient(app2)

    # 在重启后的新服务实例上直接对旧 session_id 提交审批 -> 成功恢复并执行写操作！
    app_req = {"decision": "approve", "reason": "verified", "idempotency_key": "idem_001"}
    res1 = client2.post(f"/sessions/{sid}/approvals", json=app_req)
    assert res1.status_code == 200
    assert res1.json()["status"] == "resumed"
    assert write_counter == 1

    # ② 同一审批重复提交第二次 -> 幂等直接返回首次结果，底层写操作依然只执行了 1 次！
    res2 = client2.post(f"/sessions/{sid}/approvals", json=app_req)
    assert res2.status_code == 200
    assert write_counter == 1

    # 检查 /trace 与 /metrics
    tr = client2.get(f"/sessions/{sid}/trace").json()
    assert len(tr["spans"]) >= 1
    metrics = client2.get("/metrics").json()
    assert metrics["sessions_count"] >= 1
