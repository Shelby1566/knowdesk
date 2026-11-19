import json
from pathlib import Path
from fastapi.testclient import TestClient
from agent.guard import GuardEngine, SecurityState, parse_unlocked_tool_schema
from kb.fusion import adaptive_rrf_fusion, calculate_distribution_margin
from service.app import create_knowdesk_app

def test_dim1_honest_online_replay_and_judge_calibration():
    """维度一【诚实度防线】：验证 54 条真实轨迹裁判落盘数据与消融实验指标"""
    runs_file = Path("eval/data/faithfulness_judge_runs.json")
    assert runs_file.exists(), "faithfulness_judge_runs.json 必须落盘存在"
    data = json.loads(runs_file.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    runs = data["runs"]

    assert len(runs) == 54
    assert metrics["n"] == 54
    assert metrics["p_o"] > 0.90
    assert metrics["kappa"] > 0.80
    assert "faithful" in metrics["confusion_matrix"]
    assert "unfaithful" in metrics["confusion_matrix"]

    sim_file = Path("eval/data/simulations/x2_dev10_lite_4_graph_hybrid/results.json")
    assert sim_file.exists(), "真实 10 条模拟工单轨迹 results.json 必须本地存在"
    sims = json.loads(sim_file.read_text(encoding="utf-8")).get("simulations", [])
    assert len(sims) == 10
    rewards = [s.get("reward_info", {}).get("reward", 0.0) for s in sims]
    pass_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards)
    assert abs(pass_rate - 0.40) < 1e-4

def test_dim2_dynamic_unlocked_schema_guard_missing_args():
    """维度二【机制创新】：验证动态 Schema 捕获与 R05 前置拦截缺失必填参数"""
    guard = GuardEngine(enforce_unlock=True, enforce_verification=True, require_human_confirm_on_write=False)
    sec = SecurityState(verified=True)

    unlock_response_content = """
Tool unlocked: file_credit_card_transaction_dispute_4829
Description: File a formal dispute for a credit card transaction.

Tool: file_credit_card_transaction_dispute_4829
Description: File a formal dispute for a credit card transaction.
Parameters:
  - transaction_id: string (required) - The unique identifier for the transaction
  - card_action: string (required) - Must be keep_active or cancel_and_reissue
  - card_last_4_digits: string (required) - Last 4 digits of the card
  - partial_refund_amount: number (optional) - Optional amount for partial refund
"""
    # 1. 模拟环境返回 ToolMessage，Guard 动态捕获参数 Schema
    guard.update_state_after_success(
        "unlock_discoverable_agent_tool",
        {"agent_tool_name": "file_credit_card_transaction_dispute_4829"},
        sec,
        result_content=unlock_response_content,
    )
    assert "file_credit_card_transaction_dispute_4829" in sec.unlocked_tools
    assert "file_credit_card_transaction_dispute_4829" in sec.unlocked_schemas
    schema = sec.unlocked_schemas["file_credit_card_transaction_dispute_4829"]
    assert schema.required_params == ["transaction_id", "card_action", "card_last_4_digits"]
    assert schema.optional_params == ["partial_refund_amount"]

    # 2. 模型下一步调用隐藏工具，但漏填了 card_action 与 card_last_4_digits -> R05 当场前置拦截！
    bad_call_args = {
        "agent_tool_name": "file_credit_card_transaction_dispute_4829",
        "arguments": json.dumps({"transaction_id": "txn_8848"}),
    }
    decision = guard.inspect("call_discoverable_agent_tool", bad_call_args, sec)
    assert decision.action == "block"
    assert decision.rule_id == "R05_DYNAMIC_SCHEMA_VALIDATION"
    assert "card_action" in decision.reason and "card_last_4_digits" in decision.reason

    # 3. 模型根据结构化反馈自纠，补全全部必填参数 -> R05 放行！
    good_call_args = {
        "agent_tool_name": "file_credit_card_transaction_dispute_4829",
        "arguments": json.dumps({
            "transaction_id": "txn_8848",
            "card_action": "keep_active",
            "card_last_4_digits": "1234",
        }),
    }
    decision_ok = guard.inspect("call_discoverable_agent_tool", good_call_args, sec)
    assert decision_ok.action == "allow"

def test_dim3_adaptive_margin_aware_rrf_math():
    """维度三【数学背景结合】：验证自适应信息熵/尖峰度加权 RRF (AdaRRF)"""
    # 场景 A: 精确代码/ID 查询（BM25 极度尖锐，Dense 模糊扁平）
    bm25_exact = [("doc_dispute_4829", 18.5), ("doc_other_1", 0.5), ("doc_other_2", 0.3), ("doc_other_3", 0.1)]
    dense_diffuse = [("doc_other_1", 0.82), ("doc_dispute_4829", 0.80), ("doc_other_2", 0.79), ("doc_other_3", 0.78)]

    m_bm25 = calculate_distribution_margin([s for _, s in bm25_exact])
    m_dense = calculate_distribution_margin([s for _, s in dense_diffuse])
    assert m_bm25 > m_dense * 2.0  # BM25 确信度显著超出

    fused_exact = adaptive_rrf_fusion([bm25_exact, dense_diffuse], k=3)
    # BM25 极度确信时，自适应加权使得 doc_dispute_4829 成功逆袭登顶
    assert fused_exact[0][0] == "doc_dispute_4829"

    # 场景 B: 口语化模糊查询（BM25 词频扁平低效，Dense 语义区分度极高）
    bm25_flat = [("doc_generic", 2.1), ("doc_target", 2.0), ("doc_misc", 1.9)]
    dense_sharp = [("doc_target", 0.96), ("doc_generic", 0.45), ("doc_misc", 0.20)]

    m_bm25_b = calculate_distribution_margin([s for _, s in bm25_flat])
    m_dense_b = calculate_distribution_margin([s for _, s in dense_sharp])
    assert m_dense_b > m_bm25_b

    fused_colloquial = adaptive_rrf_fusion([bm25_flat, dense_sharp], k=3)
    # Dense 确信度更高，语义目标 doc_target 稳居第 1 名
    assert fused_colloquial[0][0] == "doc_target"

def test_dim4_graph_stream_node_transitions_sse(tmp_path: Path):
    """维度四【工程深度】：验证 graph.stream 节点级实时事件流 SSE 协议"""
    db_file = tmp_path / "stream_ckpt.sqlite"

    def offline_llm(messages):
        return {"content": "Your request has been processed according to bank policy.", "tool_calls": []}

    app = create_knowdesk_app(db_path=db_file, llm_fn=offline_llm, tool_funcs={})
    client = TestClient(app)

    sid = client.post("/sessions").json()["session_id"]
    resp = client.post(f"/sessions/{sid}/messages", json={"content": "Hello KnowDesk"})
    assert resp.status_code == 200
    text = resp.text

    # 验证节点流转事件实时抛出
    assert "event: node_transition" in text
    assert '"node": "agent"' in text or '"node": "guard"' in text
    assert "event: done" in text
