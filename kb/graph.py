import operator
from typing import Any
import networkx as nx

OPERATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

class BankingPolicyKnowledgeGraph:
    """
    GraphRAG 实体关系知识图谱与确定性逻辑约束求解器 (Deterministic Constraint Solver)
    解决纯向量检索在面对严苛先决条件与否定性排他条款 (Disqualifiers) 时的幻觉与漏判问题。
    使用 NetworkX 有向图 (DiGraph) 建模实体、先决条件与排他约束。
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._seed_default_banking_ontology()

    def _seed_default_banking_ontology(self) -> None:
        # 1. 注册核心产品 (Products)
        self.graph.add_node(
            "diamond_elite",
            type="product",
            name="Diamond Elite Card",
            base_fee=495.0,
            currency="USD",
            aliases=["diamond elite", "diamond_elite", "黑钻卡", "钻石卡"],
        )
        self.graph.add_node(
            "sapphire_reserve",
            type="product",
            name="Sapphire Reserve Card",
            base_fee=550.0,
            currency="USD",
            aliases=["sapphire reserve", "sapphire", "蓝宝石卡"],
        )
        self.graph.add_node(
            "business_advantage",
            type="product",
            name="Business Advantage Checking",
            base_fee=25.0,
            currency="USD",
            aliases=["business advantage", "商业优选账户"],
        )

        # 2. 注册费用实体 (Fees)
        self.graph.add_node("annual_fee_495", type="fee", name="Diamond Elite Annual Fee", amount=495.0)
        self.graph.add_node("annual_fee_550", type="fee", name="Sapphire Reserve Annual Fee", amount=550.0)
        self.graph.add_node("monthly_fee_25", type="fee", name="Monthly Account Maintenance Fee", amount=25.0)

        # 3. 注册豁免先决条件 (Waiver Conditions)
        self.graph.add_node(
            "cond_spend_30k",
            type="condition",
            metric="annual_spend",
            op=">=",
            threshold=30000.0,
            desc="Annual spending >= $30,000 in past 12 billing cycles",
        )
        self.graph.add_node(
            "cond_spend_50k",
            type="condition",
            metric="annual_spend",
            op=">=",
            threshold=50000.0,
            desc="Annual spending >= $50,000 in past 12 billing cycles",
        )
        self.graph.add_node(
            "cond_min_balance_5k",
            type="condition",
            metric="avg_monthly_balance",
            op=">=",
            threshold=5000.0,
            desc="Average monthly balance >= $5,000",
        )
        self.graph.add_node(
            "cond_direct_deposit_2500",
            type="condition",
            metric="monthly_direct_deposit",
            op=">=",
            threshold=2500.0,
            desc="Qualifying direct deposits totaling >= $2,500/month",
        )

        # 4. 注册严苛排他/一票否决条款 (Hard Disqualifiers)
        self.graph.add_node(
            "disq_late_payment_12m",
            type="disqualifier",
            metric="late_payment_count_12m",
            op=">",
            threshold=0,
            desc="Any late payment or delinquency recorded in past 12 months",
        )
        self.graph.add_node(
            "disq_fraud_investigation",
            type="disqualifier",
            metric="fraud_hold",
            op="==",
            threshold=True,
            desc="Account currently flagged under AML/fraud review",
        )
        self.graph.add_node(
            "disq_delinquency_gt_30d",
            type="disqualifier",
            metric="days_delinquent",
            op=">",
            threshold=30,
            desc="Past-due balance exceeding 30 calendar days",
        )

        # 5. 注册特权权益 (Benefits)
        self.graph.add_node("benefit_lounge", type="benefit", name="Unlimited Priority Pass Lounge Access")
        self.graph.add_node("benefit_travel_credit", type="benefit", name="$300 Annual Travel Statement Credit")

        # 6. 构建实体关系边 (Edges)
        # 产品 -> 费用
        self.graph.add_edge("diamond_elite", "annual_fee_495", relation="HAS_FEE")
        self.graph.add_edge("sapphire_reserve", "annual_fee_550", relation="HAS_FEE")
        self.graph.add_edge("business_advantage", "monthly_fee_25", relation="HAS_FEE")

        # 条件 -> 豁免费用
        self.graph.add_edge("cond_spend_30k", "annual_fee_495", relation="WAIVES_FEE")
        self.graph.add_edge("cond_spend_50k", "annual_fee_550", relation="WAIVES_FEE")
        self.graph.add_edge("cond_min_balance_5k", "monthly_fee_25", relation="WAIVES_FEE")
        self.graph.add_edge("cond_direct_deposit_2500", "monthly_fee_25", relation="WAIVES_FEE")

        # 排他约束 -> 否决豁免条件 (INVALIDATES)
        self.graph.add_edge("disq_late_payment_12m", "cond_spend_30k", relation="INVALIDATES")
        self.graph.add_edge("disq_fraud_investigation", "cond_spend_30k", relation="INVALIDATES")
        self.graph.add_edge("disq_delinquency_gt_30d", "cond_spend_30k", relation="INVALIDATES")

        self.graph.add_edge("disq_late_payment_12m", "cond_spend_50k", relation="INVALIDATES")
        self.graph.add_edge("disq_fraud_investigation", "cond_spend_50k", relation="INVALIDATES")

        self.graph.add_edge("disq_fraud_investigation", "cond_min_balance_5k", relation="INVALIDATES")
        self.graph.add_edge("disq_fraud_investigation", "cond_direct_deposit_2500", relation="INVALIDATES")

        # 产品 -> 权益
        self.graph.add_edge("diamond_elite", "benefit_lounge", relation="GRANTS")
        self.graph.add_edge("diamond_elite", "benefit_travel_credit", relation="GRANTS")

    def evaluate_policy_constraints(self, product_id: str, user_facts: dict[str, Any]) -> dict[str, Any]:
        """
        确定性逻辑约束求解：
        遍历产品关联的所有费用，检查豁免条件与排他约束。
        """
        if product_id not in self.graph:
            return {
                "product_id": product_id,
                "found": False,
                "can_waive": False,
                "proof": f"Product '{product_id}' not found in policy graph.",
            }

        product_data = self.graph.nodes[product_id]
        fees = [tgt for _, tgt, d in self.graph.out_edges(product_id, data=True) if d.get("relation") == "HAS_FEE"]

        eligible_waivers: list[str] = []
        denied_waivers: list[dict[str, Any]] = []
        triggered_disqualifiers: list[dict[str, Any]] = []
        audit_traces: list[str] = []

        for fee_node in fees:
            fee_info = self.graph.nodes[fee_node]
            fee_name = fee_info.get("name", fee_node)
            fee_amount = fee_info.get("amount", 0.0)

            # 寻找所有可以豁免该费用的 condition 节点
            waiver_conds = [
                src for src, _, d in self.graph.in_edges(fee_node, data=True) if d.get("relation") == "WAIVES_FEE"
            ]

            fee_waived = False
            for cond_id in waiver_conds:
                cond_info = self.graph.nodes[cond_id]
                metric = cond_info.get("metric", "")
                op_str = cond_info.get("op", "==")
                threshold = cond_info.get("threshold")
                desc = cond_info.get("desc", cond_id)
                op_func = OPERATORS.get(op_str, operator.eq)

                actual_val = user_facts.get(metric, 0.0)
                cond_satisfied = False
                try:
                    cond_satisfied = op_func(actual_val, threshold)
                except Exception:
                    cond_satisfied = False

                if not cond_satisfied:
                    denied_waivers.append({
                        "fee": fee_node,
                        "condition": cond_id,
                        "reason": f"Condition not met: {desc} (Actual: {actual_val}, Threshold: {threshold})",
                    })
                    audit_traces.append(f"  [X] Condition {cond_id} FAILED: actual {actual_val} vs required {op_str} {threshold}")
                    continue

                # 条件满足，检查是否有排他条款 (INVALIDATES)
                disqualifier_nodes = [
                    src for src, _, d in self.graph.in_edges(cond_id, data=True) if d.get("relation") == "INVALIDATES"
                ]

                is_disqualified = False
                for disq_id in disqualifier_nodes:
                    disq_info = self.graph.nodes[disq_id]
                    d_metric = disq_info.get("metric", "")
                    d_op = disq_info.get("op", "==")
                    d_threshold = disq_info.get("threshold")
                    d_desc = disq_info.get("desc", disq_id)
                    d_op_func = OPERATORS.get(d_op, operator.eq)

                    d_actual = user_facts.get(d_metric, False if isinstance(d_threshold, bool) else 0)
                    disq_triggered = False
                    try:
                        disq_triggered = d_op_func(d_actual, d_threshold)
                    except Exception:
                        disq_triggered = False

                    if disq_triggered:
                        is_disqualified = True
                        triggered_disqualifiers.append({
                            "disqualifier_id": disq_id,
                            "desc": d_desc,
                            "actual_value": d_actual,
                        })
                        audit_traces.append(f"  [!] DISQUALIFIER TRIGGERED: {d_desc} (Actual: {d_actual}). Overrules condition {cond_id}!")

                if is_disqualified:
                    denied_waivers.append({
                        "fee": fee_node,
                        "condition": cond_id,
                        "reason": f"Condition satisfied ({desc}), but INVALIDATED by disqualifier: {[d['disqualifier_id'] for d in triggered_disqualifiers]}",
                    })
                else:
                    fee_waived = True
                    eligible_waivers.append(fee_node)
                    audit_traces.append(f"  [√] Condition {cond_id} SATISFIED with no disqualifiers. Fee {fee_name} WAIVED.")
                    break

        can_waive = len(eligible_waivers) > 0
        status_text = "APPROVED" if can_waive else "DENIED"

        proof = (
            f"=== DETERMINISTIC POLICY PROOF ({status_text}) ===\n"
            f"Product: {product_data.get('name', product_id)} (Base Fee: ${product_data.get('base_fee', 0.0)})\n"
            + "\n".join(audit_traces)
        )

        return {
            "product_id": product_id,
            "found": True,
            "can_waive": can_waive,
            "eligible_waivers": eligible_waivers,
            "denied_waivers": denied_waivers,
            "triggered_disqualifiers": triggered_disqualifiers,
            "audit_traces": audit_traces,
            "proof": proof,
        }

    def detect_product(self, text: str) -> str | None:
        """从用户文本或检索结果中识别提及的银行产品 ID"""
        text_lower = text.lower()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "product":
                if node.lower() in text_lower:
                    return node
                for alias in data.get("aliases", []):
                    if alias.lower() in text_lower:
                        return node
        return None

    def augment_retrieval_with_graph(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        user_facts: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        GraphRAG 注入增强：
        识别查询或上下文中的产品，执行确定性逻辑约束求解，
        并将具备最高置信度的逻辑证据链 (Deterministic Proof) 作为合成文档注入检索流头部。
        """
        product_id = self.detect_product(query)
        if not product_id:
            for doc in retrieved_docs:
                doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"
                product_id = self.detect_product(doc_text)
                if product_id:
                    break

        if not product_id or not user_facts:
            return retrieved_docs

        constraint_eval = self.evaluate_policy_constraints(product_id, user_facts)
        synthetic_doc = {
            "id": f"graph_constraint_{product_id}",
            "title": f"GraphRAG Deterministic Policy Constraint: {product_id}",
            "content": f"[DETERMINISTIC GRAPH CONSTRAINT AUDIT]\n{constraint_eval['proof']}\n"
            "STRICT COMPLIANCE DIRECTIVE: This mathematical proof supersedes any general brochure text. "
            "Do not promise waivers or exceptions if the proof state is DENIED.",
            "source": "knowledge_graph_constraint_solver",
            "score": 1.0,
            "metadata": {
                "graph_evaluation": constraint_eval,
                "is_deterministic_constraint": True,
            },
        }

        return [synthetic_doc] + list(retrieved_docs)
