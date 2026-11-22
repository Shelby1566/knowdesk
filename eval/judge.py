from collections import Counter
from typing import Callable, Sequence
from knowdesk.tools.schemas import FaithfulnessVerdict, validate_structured_output

def cohens_kappa(y_human: Sequence[str], y_judge: Sequence[str], labels: Sequence[str] = ("faithful", "unfaithful")) -> dict[str, float | dict]:
    """
    M22: 计算观察一致率 p_o、偶然一致率 p_e、Cohen's kappa 与混淆矩阵。
    公式: kappa = (p_o - p_e) / (1 - p_e)
    """
    assert len(y_human) == len(y_judge) and len(y_human) > 0
    n = len(y_human)
    po = sum(1 for h, j in zip(y_human, y_judge) if h == j) / n

    c_h = Counter(y_human)
    c_j = Counter(y_judge)
    pe = sum((c_h[lbl] / n) * (c_j[lbl] / n) for lbl in labels)

    kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else 1.0
    matrix = {h_lbl: {j_lbl: 0 for j_lbl in labels} for h_lbl in labels}
    for h, j in zip(y_human, y_judge):
        if h in matrix and j in matrix[h]:
            matrix[h][j] += 1

    return {"n": n, "p_o": po, "p_e": pe, "kappa": kappa, "confusion_matrix": matrix}

class FaithfulnessJudge:
    """只评判 Agent 回复中引用的政策细节是否与检索到的原文一致，输出强制校验 JSON Schema"""
    def __init__(self, llm_judge_fn: Callable[[str, str, int], str]):
        self.llm_judge_fn = llm_judge_fn
        self.parse_failures: list[int] = []

    def evaluate_item(self, doc_text: str, agent_reply: str, num_repeats: int = 3) -> dict:
        verdicts: list[str] = []
        last_valid = FaithfulnessVerdict(verdict="cannot_tell")
        for r in range(num_repeats):
            raw = self.llm_judge_fn(doc_text, agent_reply, r)
            parsed, _ = validate_structured_output(
                raw, FaithfulnessVerdict, fallback=FaithfulnessVerdict(verdict="cannot_tell"), failure_counter=self.parse_failures
            )
            verdicts.append(parsed.verdict)
            last_valid = parsed

        majority = Counter(verdicts).most_common(1)[0][0]
        unanimous = len(set(verdicts)) == 1
        return {
            "majority_verdict": majority,
            "unanimous": unanimous,
            "verdicts": verdicts,
            "evidence": last_valid.evidence,
            "reason": last_valid.reason,
        }
