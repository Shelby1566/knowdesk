from knowdesk.kb.eval_retrieval import recall_at_k, reciprocal_rank
from knowdesk.kb.fusion import rrf_fusion, linear_fusion

def test_recall_and_mrr_metrics():
    gold = {"doc_a", "doc_b", "doc_c", "doc_d"}
    # 同一文档出现两次只算一次；前4个唯一文档命中 doc_b, doc_d -> Recall@4 = 2/4 = 0.5
    retrieved = ["doc_x", "doc_b", "doc_b", "doc_y", "doc_d"]
    assert abs(recall_at_k(retrieved, gold, k=4) - 0.5) < 1e-9
    # 第一个命中的 doc_b 在去重后排第 2 位 -> RR = 1/2 = 0.5
    assert abs(reciprocal_rank(retrieved, gold) - 0.5) < 1e-9
    # 完全未命中 -> 0.0
    assert recall_at_k(["doc_z"], gold, k=5) == 0.0
    assert reciprocal_rank(["doc_z"], gold) == 0.0

def test_rrf_and_linear_fusion():
    list1 = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    list2 = [("b", 0.9), ("c", 0.8), ("a", 0.1)]
    # b 在两路分别排第 2 和第 1，RRF 分数应最高
    rrf = rrf_fusion([list1, list2], k=3, rrf_k=60)
    assert rrf[0][0] == "b"
    expected_b = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert abs(rrf[0][1] - expected_b) < 1e-9

    # alpha=1.0 时线性融合名次应与第一路完全一致
    lin = linear_fusion(list1, list2, alpha=1.0, k=3)
    assert [x[0] for x in lin] == ["a", "b", "c"]
