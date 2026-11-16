import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class TriageDecision:
    route: str  # "policy", "action", "compliance", "faq"
    confidence: float
    rationale: str

@dataclass
class ActionProposal:
    tool_name: str
    args: dict[str, Any]
    idempotency_key: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"

@dataclass
class AuditDecision:
    status: str  # "APPROVED", "REJECTED", "REQUIRES_HITL"
    risk_score: float  # 0.0 to 1.0
    rationale: str
    violations: list[str] = field(default_factory=list)

@dataclass
class SwarmDispatchResult:
    session_id: str
    user_id: str
    tenant_id: str
    route: str
    final_output: str
    action_proposal: ActionProposal | None = None
    audit_decision: AuditDecision | None = None
    interrupt_required: bool = False
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

class TriageAgent:
    """分流路由 Agent：在毫秒级内根据语义与风险度精准指派专业领域 Agent"""

    def evaluate(self, message: str) -> TriageDecision:
        msg = message.lower()
        if any(w in msg for w in ["fraud", "chargeback", "stolen", "aml", "盗刷", "拒付", "冻结", "风控"]):
            return TriageDecision(route="compliance", confidence=0.98, rationale="High-risk fraud/AML indicators detected")
        if any(w in msg for w in ["transfer", "wire", "pay", "send money", "lock", "转账", "汇款", "锁定"]):
            return TriageDecision(route="action", confidence=0.95, rationale="Transactional mutation intent detected")
        if any(w in msg for w in ["fee", "waive", "policy", "annual fee", "interest", "apr", "年费", "政策", "规则"]):
            return TriageDecision(route="policy", confidence=0.92, rationale="Policy and knowledge retrieval intent detected")
        return TriageDecision(route="faq", confidence=0.85, rationale="General inquiry or greeting")

class PolicySpecialistAgent:
    """政策检索专家 Agent：纯只读沙箱，集成 GraphRAG 确定性推理，物理隔离写操作"""

    def __init__(self, policy_graph: Any = None, retriever_fn: Callable[[str], Any] | None = None) -> None:
        self.policy_graph = policy_graph
        self.retriever_fn = retriever_fn

    def answer(self, query: str, user_facts: dict[str, Any] | None = None) -> str:
        # 1. 优先调用 GraphRAG 逻辑求解器
        graph_proof = None
        if self.policy_graph:
            product_id = self.policy_graph.detect_product(query)
            if product_id and user_facts:
                eval_res = self.policy_graph.evaluate_policy_constraints(product_id, user_facts)
                graph_proof = eval_res.get("proof")

        # 2. 模拟/实际召回政策文档
        retrieved_content = "Standard Bank Terms: Fee waivers require minimum annual spend and good standing."
        if self.retriever_fn:
            retrieved_content = str(self.retriever_fn(query))

        if graph_proof:
            return f"{graph_proof}\n\nSupporting Policy Documentation:\n{retrieved_content}"
        return f"Policy Advisory Response:\n{retrieved_content}"

class ActionSpecialistAgent:
    """核心交易执行 Agent：负责参数装配、幂等令牌生成与执行提议构建"""

    def formulate_proposal(self, tool_name: str, args: dict[str, Any], idempotency_key: str) -> ActionProposal:
        amount = float(args.get("amount", 0.0))
        if tool_name in ["transfer_funds", "wire_transfer"] and amount >= 1000.0:
            risk = "HIGH"
        elif tool_name in ["lock_card", "freeze_account"]:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return ActionProposal(
            tool_name=tool_name,
            args=args,
            idempotency_key=idempotency_key,
            risk_level=risk,
        )

class AuditorAgent:
    """独立合规与风控审计 Agent：对涉及资金与账户安全的执行提议进行双人复核校验"""

    def __init__(self, max_autonomous_amount: float = 1000.0) -> None:
        self.max_autonomous_amount = max_autonomous_amount

    def audit(
        self,
        proposal: ActionProposal,
        user_context: dict[str, Any],
    ) -> AuditDecision:
        verified = user_context.get("verified", False)
        violations = []

        # 校验 1: 未通过核身绝对禁止写操作
        if not verified:
            violations.append("SECURITY_VIOLATION: User identity is unverified.")
            return AuditDecision(
                status="REJECTED",
                risk_score=1.0,
                rationale="Identity verification required before executing transactions.",
                violations=violations,
            )

        # 校验 2: 大额资金变动必须触发 Human-in-the-Loop (HITL) 中断确认
        amount = float(proposal.args.get("amount", 0.0))
        if amount > self.max_autonomous_amount:
            violations.append(f"HITL_POLICY: Transfer amount ${amount:.2f} exceeds autonomous limit ${self.max_autonomous_amount:.2f}.")
            return AuditDecision(
                status="REQUIRES_HITL",
                risk_score=0.75,
                rationale="High-value transfer requires explicit human confirmation.",
                violations=violations,
            )

        # 校验 3: 黑名单账户校验
        recipient = str(proposal.args.get("recipient", ""))
        if "fraud" in recipient.lower() or "blocked" in recipient.lower():
            violations.append(f"AML_BLACKLIST: Recipient '{recipient}' is on fraud monitoring watchlist.")
            return AuditDecision(
                status="REJECTED",
                risk_score=0.99,
                rationale="Recipient account blocked by AML compliance.",
                violations=violations,
            )

        return AuditDecision(
            status="APPROVED",
            risk_score=0.1,
            rationale="All security and compliance rules passed.",
            violations=[],
        )

class HierarchicalAgentSwarm:
    """
    分层多 Agent 联邦协同编排器 (Hierarchical Multi-Agent Swarm Orchestrator)
    
    架构设计：
    1. Triage Agent: 接收用户请求，毫秒级快速意图与风险分类。
    2. Policy Specialist: 仅拥有只读检索与 GraphRAG 权限，杜绝提权修改数据。
    3. Action Specialist: 专注于交易参数构建与幂等校验，产生操作提议 (Proposal)。
    4. Compliance Auditor: 独立风控监督节点，对操作提议执行准入审核。对超限操作触发 HITL 流程。
    """

    def __init__(
        self,
        triage: TriageAgent | None = None,
        policy_agent: PolicySpecialistAgent | None = None,
        action_agent: ActionSpecialistAgent | None = None,
        auditor: AuditorAgent | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.triage = triage or TriageAgent()
        self.policy_agent = policy_agent or PolicySpecialistAgent()
        self.action_agent = action_agent or ActionSpecialistAgent()
        self.auditor = auditor or AuditorAgent()
        self.tool_executor = tool_executor or (lambda t, a: {"status": "SUCCESS", "tool": t, "executed_args": a})

    def dispatch_sync(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str,
        message: str,
        user_context: dict[str, Any] | None = None,
        mock_action: dict[str, Any] | None = None,
    ) -> SwarmDispatchResult:
        context = user_context or {"verified": False}
        audit_trail: list[dict[str, Any]] = []

        # 步骤 1: 分流评估
        decision = self.triage.evaluate(message)
        audit_trail.append({"step": "triage", "route": decision.route, "rationale": decision.rationale})

        # 分支 A: 政策或常见问答 -> Policy Specialist
        if decision.route in ["policy", "faq"]:
            output = self.policy_agent.answer(message, user_facts=context.get("facts"))
            audit_trail.append({"step": "policy_specialist", "status": "completed"})
            return SwarmDispatchResult(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                route=decision.route,
                final_output=output,
                audit_trail=audit_trail,
            )

        # 分支 B: 交易动作或合规审计 -> Action Specialist + Auditor
        action_data = mock_action or {
            "tool_name": "transfer_funds",
            "args": {"amount": 2500.0, "recipient": "acc_bob_9921"},
            "idempotency_key": f"tx_{session_id}_{int(asyncio.get_event_loop_policy().get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0)}",
        }

        proposal = self.action_agent.formulate_proposal(
            tool_name=action_data.get("tool_name", "transfer_funds"),
            args=action_data.get("args", {}),
            idempotency_key=action_data.get("idempotency_key", f"tx_{session_id}_auto"),
        )
        audit_trail.append({"step": "action_proposal", "proposal": proposal.__dict__})

        # 步骤 3: 独立合规审计
        audit_res = self.auditor.audit(proposal, user_context=context)
        audit_trail.append({"step": "compliance_audit", "audit_decision": audit_res.__dict__})

        if audit_res.status == "REJECTED":
            return SwarmDispatchResult(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                route=decision.route,
                action_proposal=proposal,
                audit_decision=audit_res,
                interrupt_required=False,
                final_output=f"[TRANSACTION REJECTED BY COMPLIANCE AUDITOR]: {audit_res.rationale} (Violations: {audit_res.violations})",
                audit_trail=audit_trail,
            )

        if audit_res.status == "REQUIRES_HITL":
            return SwarmDispatchResult(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                route=decision.route,
                action_proposal=proposal,
                audit_decision=audit_res,
                interrupt_required=True,
                final_output=f"[HUMAN APPROVAL REQUIRED]: {audit_res.rationale} Please confirm transaction of ${proposal.args.get('amount')} to {proposal.args.get('recipient')}.",
                audit_trail=audit_trail,
            )

        # 步骤 4: 审计通过，安全执行
        exec_res = self.tool_executor(proposal.tool_name, proposal.args)
        audit_trail.append({"step": "tool_execution", "result": exec_res})
        return SwarmDispatchResult(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            route=decision.route,
            action_proposal=proposal,
            audit_decision=audit_res,
            interrupt_required=False,
            final_output=f"[SUCCESS]: Action {proposal.tool_name} executed successfully. Result: {exec_res}",
            audit_trail=audit_trail,
        )

    async def dispatch(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str,
        message: str,
        user_context: dict[str, Any] | None = None,
        mock_action: dict[str, Any] | None = None,
    ) -> SwarmDispatchResult:
        """异步分发入口"""
        return self.dispatch_sync(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            message=message,
            user_context=user_context,
            mock_action=mock_action,
        )
