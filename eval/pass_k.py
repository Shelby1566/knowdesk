import math
import random
from typing import Sequence

def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    M21: τ-bench 论文定义的 pass^k 无偏估计量: C(c, k) / C(n, k)
    物理意义：同一个任务独立运行 n 次成功 c 次，随机挑 k 次【全部成功】的概率。
    """
    if c < k or n < k or k <= 0:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)

def dataset_pass_k(trials_per_task: Sequence[tuple[int, int]], k: int) -> float:
    if not trials_per_task:
        return 0.0
    return sum(pass_at_k_unbiased(n, c, k) for n, c in trials_per_task) / len(trials_per_task)

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """计算均值及 Bootstrap (1-alpha) 置信区间 (mean, ci_low, ci_high)"""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    mean_val = sum(values) / n
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot)
    return mean_val, means[lo_idx], means[min(hi_idx, n_boot - 1)]

def paired_bootstrap_diff(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
    noise_floor: float = 0.05,
) -> dict[str, float | bool | str]:
    """M19 回归门禁配对检验：按任务配对计算 delta_i = cand_i - base_i 的 Bootstrap 95% 区间"""
    assert len(baseline_scores) == len(candidate_scores)
    diffs = [c - b for b, c in zip(baseline_scores, candidate_scores)]
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    non_regressing = (mean_diff >= -noise_floor) and (ci_hi > 0.0 or abs(mean_diff) <= noise_floor)
    statistically_improved = ci_lo > 0.0
    verdict = "IMPROVED" if statistically_improved else ("PASS_NON_REGRESSING" if non_regressing else "REGRESSED")
    return {
        "mean_diff": mean_diff,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "non_regressing": non_regressing,
        "statistically_improved": statistically_improved,
        "verdict": verdict,
    }
