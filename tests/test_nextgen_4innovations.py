import asyncio
import json
from pathlib import Path

from agent.reflection import DPOPreferenceExporter, SelfReflectionMemory
from agent.swarm import (
    ActionProposal,
    ActionSpecialistAgent,
    AuditorAgent,
    HierarchicalAgentSwarm,
    PolicySpecialistAgent,
    TriageAgent,
)
from kb.graph import BankingPolicyKnowledgeGraph
from service.prefetch import SAFE_SPECULATIVE_TOOLS, SpeculativePrefetchEngine

def test_nextgen_graphrag_constraint_engine():
    """验证 GraphRAG 确定性图谱逻辑约束求解器：先决条件达标 vs 排他条款一票否决"""
    graph = BankingPolicyKnowledgeGraph()

    # Case 1: 年消费 $35,000 达标 (>= $30,000)，无迟缴 -> 豁免成功
    facts_approved = {"annual_spend": 35000.0, "late_payment_count_12m": 0}
    res_appr = graph.evaluate_policy_constraints("diamond_elite", facts_approved)
    assert res_appr["found"] is True
    assert res_appr["can_waive"] is True
    assert "annual_fee_495" in res_appr["eligible_waivers"]
    assert "APPROVED" in res_appr["proof"]

    # Case 2: 年消费 $42,000 达标，但有 1 次迟缴记录 -> 排他条款 INVALIDATES 一票否决
    facts_denied = {"annual_spend": 42000.0, "late_payment_count_12m": 1}
    res_denied = graph.evaluate_policy_constraints("diamond_elite", facts_denied)
    assert res_denied["can_waive"] is False
    assert len(res_denied["eligible_waivers"]) == 0
    assert any(d["disqualifier_id"] == "disq_late_payment_12m" for d in res_denied["triggered_disqualifiers"])
    assert "DENIED" in res_denied["proof"]

    # Case 3: 商业优选账户，存款达标但 AML 涉案调查 -> 一票否决
    facts_fraud = {"avg_monthly_balance": 10000.0, "fraud_hold": True}
    res_fraud = graph.evaluate_policy_constraints("business_advantage", facts_fraud)
    assert res_fraud["can_waive"] is False
    assert any(d["disqualifier_id"] == "disq_fraud_investigation" for d in res_fraud["triggered_disqualifiers"])

    # Case 4: 注入检索流测试
    query = "Can I waive the annual fee for my Diamond Elite card?"
    docs = [{"id": "doc_general", "title": "Standard Card Terms", "content": "Fees apply annually.", "score": 0.8}]
    augmented = graph.augment_retrieval_with_graph(query, docs, user_facts=facts_denied)
    assert len(augmented) == 2
    top_doc = augmented[0]
    assert top_doc["source"] == "knowledge_graph_constraint_solver"
    assert top_doc["score"] == 1.0
    assert "DETERMINISTIC GRAPH CONSTRAINT AUDIT" in top_doc["content"]

def test_nextgen_speculative_prefetch_engine():
    """验证投机性并行预执行与预取引擎：意图解析、异步并行预取、缓存命中、安全隔离"""
    engine = SpeculativePrefetchEngine()

    # 1. 意图解析：卡号抽取与明细预取意图
    prompt = "Please check transactions and details for my card ending in 4589"
    intents = engine.extract_speculative_intents(prompt)
    tool_names = [i["tool_name"] for i in intents]
    assert "get_card_info" in tool_names
    assert "get_recent_transactions" in tool_names

    # 2. 异步投机预取
    session_id = "test_spec_session_01"
    keys = asyncio.run(engine.prefetch_speculative(session_id, prompt))
    assert len(keys) >= 2

    # 3. 消费命中预取
    hit, data, saved_ms = engine.claim(session_id, "get_card_info", {"last4": "4589"})
    assert hit is True
    assert data["last4"] == "4589"
    assert data["product"] == "diamond_elite"

    # 4. 再次获取应 Cache Miss（已消费）
    hit_again, _, _ = engine.claim(session_id, "get_card_info", {"last4": "4589"})
    assert hit_again is False

    # 5. 安全隔离：写操作严禁出现在投机执行白名单中
    assert "transfer_funds" not in SAFE_SPECULATIVE_TOOLS
    assert "wire_transfer" not in SAFE_SPECULATIVE_TOOLS
    assert "lock_card" not in SAFE_SPECULATIVE_TOOLS

    stats = engine.stats()
    assert stats["cache_hits"] >= 1
    assert stats["cache_misses"] >= 1

def test_nextgen_hierarchical_multi_agent_swarm():
    """验证分层多 Agent 联邦协同：分流路由、只读隔离、独立合规审核与 HITL 中断"""
    graph = BankingPolicyKnowledgeGraph()
    policy_agent = PolicySpecialistAgent(policy_graph=graph)
    auditor = AuditorAgent(max_autonomous_amount=1000.0)
    swarm = HierarchicalAgentSwarm(policy_agent=policy_agent, auditor=auditor)

    # 场景 A: 政策咨询流向 policy 专员，只读无写权限
    res_policy = swarm.dispatch_sync(
        session_id="s1",
        user_id="u1",
        tenant_id="retail",
        message="What is the policy to waive annual fee on Diamond Elite?",
        user_context={"facts": {"annual_spend": 35000, "late_payment_count_12m": 0}},
    )
    assert res_policy.route == "policy"
    assert res_policy.action_proposal is None
    assert "DETERMINISTIC POLICY PROOF" in res_policy.final_output

    # 场景 B: 未核身用户试图转账 -> Auditor 直接打回 (REJECTED)
    res_unverified = swarm.dispatch_sync(
        session_id="s2",
        user_id="u2",
        tenant_id="retail",
        message="Please transfer $500 to acc_9988",
        user_context={"verified": False},
        mock_action={"tool_name": "transfer_funds", "args": {"amount": 500.0, "recipient": "acc_9988"}},
    )
    assert res_unverified.route == "action"
    assert res_unverified.audit_decision is not None
    assert res_unverified.audit_decision.status == "REJECTED"
    assert "Identity verification required" in res_unverified.final_output

    # 场景 C: 核身用户进行超限大额转账 ($5000 > $1000) -> 触发 HITL 中断
    res_hitl = swarm.dispatch_sync(
        session_id="s3",
        user_id="u3",
        tenant_id="retail",
        message="Wire $5000 to Bob",
        user_context={"verified": True},
        mock_action={"tool_name": "transfer_funds", "args": {"amount": 5000.0, "recipient": "acc_bob"}},
    )
    assert res_hitl.interrupt_required is True
    assert res_hitl.audit_decision.status == "REQUIRES_HITL"
    assert "HUMAN APPROVAL REQUIRED" in res_hitl.final_output

    # 场景 D: 核身用户进行合规限额内转账 ($200 <= $1000) -> 审计通过并成功执行
    res_approved = swarm.dispatch_sync(
        session_id="s4",
        user_id="u4",
        tenant_id="retail",
        message="Transfer $200 to Alice",
        user_context={"verified": True},
        mock_action={"tool_name": "transfer_funds", "args": {"amount": 200.0, "recipient": "acc_alice"}},
    )
    assert res_approved.interrupt_required is False
    assert res_approved.audit_decision.status == "APPROVED"
    assert "SUCCESS" in res_approved.final_output

def test_nextgen_reflection_memory_and_dpo_pipeline(tmp_path: Path):
    """验证自适应反思记忆库与轨迹到 DPO 偏好对齐数据流水线"""
    memory = SelfReflectionMemory()

    # 1. 检索与入参缺失相关的反思卡片
    cards = memory.retrieve_relevant_reflections("I want to transfer funds without routing number", top_k=1)
    assert len(cards) >= 1
    assert cards[0].failure_category == "WRONG_TOOL_ARGUMENTS"

    # 2. 格式化 Few-Shot Prompt 注入块
    block = memory.format_fewshot_block(cards)
    assert "ADAPTIVE REFLECTION MEMORY" in block
    assert "WRONG_TOOL_ARGUMENTS" in block
    assert "INCORRECT (Anti-pattern)" in block

    # 3. 动态录入新的反思经验
    new_card = memory.record_failure(
        failure_category="TIMEOUT_LOOP",
        trigger_keywords=["status_poll", "loop"],
        critique="Agent queried balance 5 times in an infinite loop without responding.",
        remedy="Limit query retries to max 2 attempts, then return status to user.",
        example_bad="Poll status 10 times consecutively.",
        example_good="Poll twice, if still pending inform customer.",
    )
    assert new_card.card_id.startswith("refl_")
    assert len(memory.cards) == 4

    # 4. DPO 偏好对导出
    out_file = tmp_path / "dpo_test.jsonl"
    count = DPOPreferenceExporter.export_dataset(output_path=out_file)
    assert count >= 3
    assert out_file.exists()

    with out_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == count
    for item in lines:
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert "metadata" in item
        assert len(item["chosen"]) > 0
        assert len(item["rejected"]) > 0
