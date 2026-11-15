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
        vals = [ s for _, s in items ]
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
