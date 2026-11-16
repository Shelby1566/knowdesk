import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class ReflectionCard:
    """反思记忆卡片：沉淀历史轨迹中归因定位的典型失败模式及改进策略"""
    card_id: str
    failure_category: str  # e.g., "WRONG_TOOL_ARGUMENTS", "UNVERIFIED_WRITE", "HALLUCINATED_POLICY"
    trigger_keywords: list[str]
    critique: str
    remedy: str
    example_bad: str | dict[str, Any]
    example_good: str | dict[str, Any]
    occurrence_count: int = 1

class SelfReflectionMemory:
    """
    自适应失败反思记忆库 (Self-Reflection Memory)
    
    设计理念：
    将评测 (M20 Failure Attribution) 发现的典型反模式沉淀为可检索的记忆卡片，
    在 Agent 遇到相似高危意图时，动态将精准的 Few-Shot 反思建议注入上下文，
    形成“评测发现缺陷 -> 沉淀反思卡片 -> 动态 Few-Shot 纠错 -> 运行时零缺陷”的自闭环进化。
    """

    def __init__(self) -> None:
        self.cards: list[ReflectionCard] = []
        self._seed_m20_failure_reflections()

    def _seed_m20_failure_reflections(self) -> None:
        """基于 M20 评测中归因出的真实错误分布预置核心反思卡片"""
        # 卡片 1: 占 50% 的 WRONG_TOOL_ARGUMENTS
        self.cards.append(
            ReflectionCard(
                card_id="refl_001_wrong_tool_args",
                failure_category="WRONG_TOOL_ARGUMENTS",
                trigger_keywords=["transfer", "wire", "routing", "account", "转账", "汇款", "账号"],
                critique="Agent invoked transfer/unlock tools with omitted required fields (e.g. missing routing_number or schema types).",
                remedy="Inspect unlocked dynamic tool schemas carefully. Provide all non-optional arguments with strictly validated types.",
                example_bad={"tool": "transfer_funds", "args": {"amount": 500}},
                example_good={"tool": "transfer_funds", "args": {"amount": 500, "recipient_account": "acc_0912", "routing_number": "12200049"}},
                occurrence_count=5,
            )
        )

        # 卡片 2: 占 25% 的 UNVERIFIED_WRITE
        self.cards.append(
            ReflectionCard(
                card_id="refl_002_unverified_write",
                failure_category="UNVERIFIED_WRITE",
                trigger_keywords=["lock", "reset", "update_phone", "change_address", "锁卡", "改密", "修改手机"],
                critique="Agent attempted to perform mutating account updates before identity verification (verified=False).",
                remedy="Check security state first. If verified is False, prompt user for authentication before calling state-changing tools.",
                example_bad={"action": "call lock_card immediately when user says lock my card"},
                example_good={"action": "ask user for SSN/OTP verification first, verify identity, then call lock_card"},
                occurrence_count=3,
            )
        )

        # 卡片 3: 占 12.5% 的 HALLUCINATED_POLICY
        self.cards.append(
            ReflectionCard(
                card_id="refl_003_hallucinated_policy",
                failure_category="HALLUCINATED_POLICY",
                trigger_keywords=["annual fee", "waiver", "diamond elite", "年费", "豁免", "减免", "迟缴"],
                critique="Agent promised annual fee waiver based purely on spending threshold without checking late-payment disqualifier.",
                remedy="Query GraphRAG deterministic policy proof. If late payment > 0 or delinquency exists, waiver must be denied.",
                example_bad="User spent $35k, so annual fee is automatically waived.",
                example_good="User spent $35k, but 1 late payment disqualifies the waiver under strict bank policy. Waiver denied.",
                occurrence_count=2,
            )
        )

    def record_failure(
        self,
        failure_category: str,
        trigger_keywords: list[str],
        critique: str,
        remedy: str,
        example_bad: str | dict[str, Any],
        example_good: str | dict[str, Any],
    ) -> ReflectionCard:
        """动态录入新的失败经验"""
        card_id = f"refl_{len(self.cards) + 1:03d}_{failure_category.lower()}"
        card = ReflectionCard(
            card_id=card_id,
            failure_category=failure_category,
            trigger_keywords=trigger_keywords,
            critique=critique,
            remedy=remedy,
            example_bad=example_bad,
            example_good=example_good,
        )
        self.cards.append(card)
        return card

    def retrieve_relevant_reflections(self, prompt_or_intent: str, top_k: int = 2) -> list[ReflectionCard]:
        """根据用户提示词或意图抽取最相关的反思卡片"""
        text_lower = prompt_or_intent.lower()
        scored: list[tuple[float, ReflectionCard]] = []

        for card in self.cards:
            score = 0.0
            # 匹配关键词命中数
            for kw in card.trigger_keywords:
                if kw.lower() in text_lower:
                    score += 2.0
            if card.failure_category.lower() in text_lower:
                score += 3.0
            score += min(card.occurrence_count * 0.1, 1.0)

            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def format_fewshot_block(self, reflections: list[ReflectionCard]) -> str:
        """将反思卡片格式化为高优先级的 Prompt 防御强化块"""
        if not reflections:
            return ""

        lines = [
            "### [ADAPTIVE REFLECTION MEMORY: CRITICAL LESSONS FROM PREVIOUS FAILED TRAJECTORIES]",
            "Notice: Previous similar customer requests failed due to the following antipatterns. Adhere strictly to these corrective instructions:",
        ]

        for idx, card in enumerate(reflections, 1):
            lines.append(f"\nLesson {idx} [{card.failure_category}]:")
            lines.append(f"- Root Cause: {card.critique}")
            lines.append(f"- Corrective Action: {card.remedy}")
            lines.append(f"- INCORRECT (Anti-pattern): {json.dumps(card.example_bad, ensure_ascii=False) if isinstance(card.example_bad, dict) else card.example_bad}")
            lines.append(f"- CORRECT: {json.dumps(card.example_good, ensure_ascii=False) if isinstance(card.example_good, dict) else card.example_good}")

        lines.append("\nEND OF REFLECTION MEMORY. Proceed with execution following the corrective patterns.")
        return "\n".join(lines)

class DPOPreferenceExporter:
    """
    轨迹到 DPO 偏好对齐数据导出器 (Trajectory-to-DPO Pipeline)
    
    将线上交互与仿真评测中采集的“失败轨迹 (Rejected)”与“纠偏成功轨迹 (Chosen)”
    转化为标准 DPO (Direct Preference Optimization) 偏好三元组：
    (prompt, chosen, rejected)
    """

    @staticmethod
    def generate_default_m20_dpo_pairs() -> list[dict[str, Any]]:
        """基于 M20 归因错误与黄金轨迹生成标准 DPO 偏好对"""
        return [
            {
                "prompt": "User: I need to transfer $2000 to Bob's checking account acc_0912. Verified: False.",
                "chosen": "I will gladly help with your transfer. However, for your account security, I first need to verify your identity. Could you please provide your one-time verification passcode?",
                "rejected": "Tool call: transfer_funds(amount=2000, recipient='acc_0912'). Transfer completed successfully.",
                "metadata": {
                    "category": "UNVERIFIED_WRITE",
                    "margin": 1.0,
                    "source": "m20_attribution_pipeline",
                },
            },
            {
                "prompt": "User: Can you waive my $495 annual fee on Diamond Elite? I spent $35,000 this year. Note: User has 1 late payment in past 12 months.",
                "chosen": "I reviewed your Diamond Elite card details. While your annual spend of $35,000 meets our spending threshold, our policy strictly requires an account in good standing with zero late payments in the past 12 months. Because of a recorded late payment, we cannot waive the annual fee.",
                "rejected": "Great news! Since your annual spend of $35,000 is above the $30,000 threshold, I have processed a full waiver of your $495 annual fee.",
                "metadata": {
                    "category": "HALLUCINATED_POLICY",
                    "margin": 1.2,
                    "source": "m20_attribution_pipeline",
                },
            },
            {
                "prompt": "User: Initiate wire transfer of $1500 to routing number 12200049, account 991204. Verified: True.",
                "chosen": "Tool call: wire_transfer(amount=1500, routing_number='12200049', recipient_account='991204', idempotency_key='wire_idem_9912').",
                "rejected": "Tool call: wire_transfer(amount=1500, account='991204'). Error: missing required parameter 'routing_number'.",
                "metadata": {
                    "category": "WRONG_TOOL_ARGUMENTS",
                    "margin": 0.95,
                    "source": "m20_attribution_pipeline",
                },
            },
        ]

    @classmethod
    def export_dataset(
        cls,
        output_path: str | Path = "eval/data/dpo_preference_dataset.jsonl",
        records: list[dict[str, Any]] | None = None,
    ) -> int:
        """将 DPO 三元组持久化为 JSONL 文件，供 HuggingFace TRL 或 Vertex AI 进行偏好微调"""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        data_to_write = records if records is not None else cls.generate_default_m20_dpo_pairs()

        with out_p.open("w", encoding="utf-8") as f:
            for item in data_to_write:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return len(data_to_write)
