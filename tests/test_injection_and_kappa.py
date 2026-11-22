from eval.judge import cohens_kappa, FaithfulnessJudge
from eval.injection import evaluate_injection_defense

def test_m22_cohens_kappa_wikipedia_exact():
    """M22 Test case 1: 维基百科标准算例 (20, 5; 10, 15) -> p_o=0.70, p_e=0.50, kappa=0.40"""
    y_h = ["faithful"] * 25 + ["unfaithful"] * 25
    y_j = (["faithful"] * 20 + ["unfaithful"] * 5) + (["faithful"] * 10 + ["unfaithful"] * 15)
    res = cohens_kappa(y_h, y_j)
    assert abs(res["p_o"] - 0.70) < 1e-9
    assert abs(res["p_e"] - 0.50) < 1e-9
    assert abs(res["kappa"] - 0.40) < 1e-9

def test_m23_injection_defense_effectiveness():
    """M23 Test: 开启 Guard 后间接提示注入与未核身越权写违规率降为 0.0"""
    res = evaluate_injection_defense()
    assert res["violation_rate_guard_off"] == 1.0
    assert res["violation_rate_guard_on_payload"] == 0.0
    assert res["violation_rate_guard_on_unverified"] == 0.0
