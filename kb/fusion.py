import math
from typing import Sequence

def rrf_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 10,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF): 只看名次不看绝对分数，规避不同检索器量纲差异"""
    fused_scores: dict[str, float] = {}
    for r_list in ranked_lists:
        for rank_idx, (item_id, _) in enumerate(r_list, start=1):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank_idx)
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def calculate_distribution_margin(scores: Sequence[float], eps: float = 1e-9) -> float:
    """
    计算检索得分分布的尖峰度与确信度 Margin:
    结合 Top-1 Z-Score 标准分与相对跳跃度 ((s_1 - s_2) / (|μ| + ε)):
    - 当检索结果包含唯一强匹配时（如精确代码/卡号/专有名词），Top-1 远超均值且与 Top-2 间隙极大，Margin 显著偏高；
    - 当检索结果分布扁平（如口语化低区分度查询），相对跳跃度趋近于 0，Margin 偏低。
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 2.0
    m = float(len(scores))
    mean = sum(scores) / m
    variance = sum((s - mean) ** 2 for s in scores) / m
    std = math.sqrt(variance)

    top1 = scores[0]
    top2 = scores[1]

    z_score = max(0.0, (top1 - mean) / (std + eps)) if std > eps else 0.0
    rel_gap = max(0.0, (top1 - top2) / (abs(mean) + eps))
    margin = z_score * (1.0 + rel_gap)
    return max(0.01, min(margin, 20.0))

def adaptive_rrf_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 10,
    rrf_k: int = 60,
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """
    自适应尖峰度加权 RRF (Adaptive Margin-Aware RRF / AdaRRF)。
    与 calculate_distribution_margin 的实现一致：
    对每一路检索器 r，令 scores 为该路得分，margin_r = calculate_distribution_margin(scores)：
        无得分时 margin = 0
        只有一个得分时 margin = 2.0
        否则 μ = mean(scores)，σ = sqrt(mean((s-μ)^2))
             z = max(0, (s1 - μ) / (σ + ε))，当 σ <= ε 时 z = 0
             rel_gap = max(0, (s1 - s2) / (|μ| + ε))
             margin = clamp(z * (1 + rel_gap), 0.01, 20)
    Softmax 先减去最大 margin 做数值稳定，τ = max(1e-3, temperature)：
        ŵ_r = exp((margin_r - max_j margin_j) / τ) / Σ_j exp((margin_j - max_k margin_k) / τ)
    融合得分与代码一致：
        AdaRRF(d) = Σ_r ŵ_r / (rrf_k + rank_r(d))

    动态效果：
    - 精确代码/专业术语检索时：BM25 尖峰度极高，自适应加权主导融合；
    - 口语化语义模糊检索时：BM25 扁平低确信，Dense 向量语义区分度更高，稠密路主导融合。
    """
    valid_lists = [r for r in ranked_lists if r]
    if not valid_lists:
        return []
    if len(valid_lists) == 1:
        return rrf_fusion(valid_lists, k=k, rrf_k=rrf_k)

    margins: list[float] = []
    for r_list in valid_lists:
        scores = [float(s) for _, s in r_list]
        margins.append(calculate_distribution_margin(scores))

    tau = max(1e-3, float(temperature))
    max_m = max(margins)
    exp_weights = [math.exp((m - max_m) / tau) for m in margins]
    sum_exp = sum(exp_weights)
    norm_weights = [ew / sum_exp for ew in exp_weights]

    fused_scores: dict[str, float] = {}
    for weight, r_list in zip(norm_weights, valid_lists):
        for rank_idx, (item_id, _) in enumerate(r_list, start=1):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + weight * (1.0 / (rrf_k + rank_idx))

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def linear_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    alpha: float = 0.5,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Min-Max 归一化后的线性加权融合（作为 RRF 的消融对照组）"""
    def _minmax(items: list[tuple[str, float]]) -> dict[str, float]:
        if not items:
            return {}
        vals = [s for _, s in items]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return {i: 1.0 for i, _ in items}
        return {i: (s - lo) / (hi - lo) for i, s in items}

    norm_b = _minmax(bm25_results)
    norm_d = _minmax(dense_results)
    all_ids = set(norm_b.keys()) | set(norm_d.keys())

    combined = {
        cid: alpha * norm_b.get(cid, 0.0) + (1.0 - alpha) * norm_d.get(cid, 0.0)
        for cid in all_ids
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
