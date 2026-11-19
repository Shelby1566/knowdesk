import json
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import create_knowdesk_app


def _scripted_llm():
    def llm(messages):
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        verified = any('"verified": true' in str(message.get("content", "")).lower() or '"verified":true' in str(message.get("content", "")).lower() for message in tool_messages)
        froze = any(message.get("name") == "freeze_card" for message in tool_messages)
        if not verified:
            return {
                "content": "",
                "tool_calls": [{"id": "tc_verify", "name": "get_user_details", "arguments": {"user_id": "u_01"}}],
            }
        if not froze:
            return {
                "content": "",
                "tool_calls": [{"id": "tc_freeze", "name": "freeze_card", "arguments": {"card_id": "card_5320"}}],
            }
        return {"content": "The card freeze has been submitted.", "tool_calls": []}

    return llm


def _pending_idempotency_key(sse_text: str) -> str:
    for block in sse_text.split("\n\n"):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("event") == "awaiting_approval":
                return payload["pending_tool_call"]["idem_key"]
    raise AssertionError("awaiting_approval event did not include pending_tool_call.idem_key")


def test_m25_full_sse_approval_idempotency_and_sqlite_restart(tmp_path: Path):
    """
    新建会话默认未核身。核身工具成功后，高风险写操作进入人工审批。
    审批必须携带监督员角色和挂起调用的 idem_key。同一 SQLite 上的新进程重复提交不再次写库。
    """
    db_file = tmp_path / "knowdesk_ckpt.sqlite"
    write_counter = 0

    def counting_freeze(args):
        nonlocal write_counter
        write_counter += 1
        return {"frozen": args.get("card_id")}

    tools = {
        "get_user_details": lambda args: {"user_id": args.get("user_id", "u_01"), "verified": True, "name": "Sera Chen"},
        "freeze_card": counting_freeze,
    }
    app1 = create_knowdesk_app(db_path=db_file, llm_fn=_scripted_llm(), tool_funcs=tools)
    client1 = TestClient(app1)

    missing = client1.get("/sessions/non_existent/trace")
    assert missing.status_code == 404

    created = client1.post("/sessions?verified=true")
    assert created.status_code == 200
    assert created.json()["verified"] is False
    session_id = created.json()["session_id"]

    sse_resp = client1.post(f"/sessions/{session_id}/messages", json={"content": "Please freeze my card"})
    assert sse_resp.status_code == 200
    assert "event: awaiting_approval" in sse_resp.text
    assert write_counter == 0
    idem_key = _pending_idempotency_key(sse_resp.text)

    app2 = create_knowdesk_app(db_path=db_file, llm_fn=_scripted_llm(), tool_funcs=tools)
    client2 = TestClient(app2)
    approval = {"decision": "approve", "reason": "verified", "idempotency_key": idem_key}
    headers = {"X-Role": "supervisor"}
    first = client2.post(f"/sessions/{session_id}/approvals", json=approval, headers=headers)
    assert first.status_code == 200
    assert first.json()["status"] == "resumed"
    assert write_counter == 1

    second = client2.post(f"/sessions/{session_id}/approvals", json=approval, headers=headers)
    assert second.status_code == 200
    assert second.json()["reply"] == first.json()["reply"]
    assert write_counter == 1

    trace = client2.get(f"/sessions/{session_id}/trace").json()
    assert len(trace["spans"]) >= 1
    metrics = client2.get("/metrics").json()
    assert metrics["sessions_count"] >= 1
