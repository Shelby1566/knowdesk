from pathlib import Path
import numpy as np
from kb.vectorstore import FaissVectorStore, l2_normalize
from kb.router import apply_intent_boost, evaluate_multilabel_f1

def test_l2_normalize_and_faiss_save_load(tmp_path: Path):
    """M5 Test case: L2 范数为1、自查询排第1、保存加载一致、缓存元数据变化自动失效"""
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(raw)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    store = FaissVectorStore(dim=3)
    store.add(["c1", "c2", "c3"], raw)

    # ② 用 c2 自己的向量去查，c2 必排第 1 且内积为 1.0
    res = store.search(raw[1], k=2)
    assert res[0][0] == "c2"
    assert abs(res[0][1] - 1.0) < 1e-5

    # ③ 保存再加载，结果完全一致
    meta = {"model_name": "test_m", "fingerprint": "abc"}
    store.save(tmp_path / "idx", meta=meta)
    loaded = FaissVectorStore.load(tmp_path / "idx", expected_meta=meta)
    assert loaded is not None
    assert loaded.search(raw[0], k=2) == store.search(raw[0], k=2)

    # ④ 换模型或切块版本（meta 不匹配）时，拒绝加载旧缓存
    invalid = FaissVectorStore.load(tmp_path / "idx", expected_meta={"model_name": "other", "fingerprint": "abc"})
    assert invalid is None

def test_intent_router_boost_and_f1():
    """M8 Test: 软加权与多标签 Micro/Macro F1 手算对拍"""
    y_true = [{"credit_card", "wire_transfer"}, {"checking_account"}, {"credit_card"}]
    y_pred = [{"credit_card"}, {"checking_account", "wire_transfer"}, {"credit_card"}]
    # TP: credit_card(2) + checking_account(1) = 3; FP: wire_transfer(1) = 1; FN: wire_transfer(1) = 1
    # Micro-P = 3/4 = 0.75, Micro-R = 3/4 = 0.75, Micro-F1 = 0.75
    m = evaluate_multilabel_f1(y_true, y_pred, labels=("credit_card", "checking_account", "wire_transfer"))
    assert abs(m["micro_f1"] - 0.75) < 1e-9

    ranked = [("doc_1", 0.80), ("doc_2", 0.75)]
    doc_to_line = {"doc_1": "checking_account", "doc_2": "credit_card"}
    boosted = apply_intent_boost(ranked, doc_to_line, {"credit_card"}, beta=0.2, k=2)
    assert boosted[0][0] == "doc_2"
