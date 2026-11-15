import math
from rank_bm25 import BM25Okapi
from kb.tokenize import tokenize
from kb.bm25 import BM25Index

TOY_CORPUS = [
    ("doc_1", "The platinum credit card has an annual fee of 95 dollars."),
    ("doc_2", "Foreign transaction fee is waived on the platinum credit card."),
    ("doc_3", "Checking account wire transfer fee and limits."),
    ("doc_4", "How to dispute a fraudulent credit card transaction immediately."),
    ("doc_5", "Savings account interest rate and monthly maintenance fee."),
]

def test_against_rank_bm25_okapi():
    """Test case 1: 与 rank_bm25.BM25Okapi 在相同分词下逐文档对拍，误差 < 1e-6"""
    ids = [doc_id for doc_id, _ in TOY_CORPUS]
    texts = [text for _, text in TOY_CORPUS]
    tokenized_corpus = [tokenize(t) for t in texts]

    ref_bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)
    my_bm25 = BM25Index.build(ids, texts, tokenize_fn=tokenize, k1=1.5, b=0.75, epsilon=0.25)

    for q in ["platinum card fee", "wire transfer", "credit card transaction", "fee"]:
        q_tokens = tokenize(q)
        ref_scores = ref_bm25.get_scores(q_tokens)
        my_scores = my_bm25.score_all(q)
        assert len(my_scores) == len(ref_scores)
        for s_my, s_ref in zip(my_scores, ref_scores):
            assert abs(s_my - float(s_ref)) < 1e-6

def test_oov_query_returns_zeros():
    """Test case 2: 词表外的词（OOV）或空查询不报错，全为 0 分"""
    ids = [doc_id for doc_id, _ in TOY_CORPUS]
    texts = [text for _, text in TOY_CORPUS]
    my_bm25 = BM25Index.build(ids, texts, tokenize_fn=tokenize)

    assert all(s == 0.0 for s in my_bm25.score_all("cryptocurrency blockchain"))
    assert my_bm25.search("   ", k=3) == []

def test_order_invariance_and_b_zero():
    """Test case: 调换语料顺序 top-k 集合不变；b=0 时文档长度不影响同词频文档得分"""
    ids = [doc_id for doc_id, _ in TOY_CORPUS]
    texts = [text for _, text in TOY_CORPUS]

    bm25_a = BM25Index.build(ids, texts, tokenize_fn=tokenize)
    bm25_b = BM25Index.build(list(reversed(ids)), list(reversed(texts)), tokenize_fn=tokenize)

    top_a = [doc_id for doc_id, _ in bm25_a.search("platinum credit card", k=2)]
    top_b = [doc_id for doc_id, _ in bm25_b.search("platinum credit card", k=2)]
    assert set(top_a) == set(top_b)

    # ④ b=0 时，词频 f(t, d) 相同的长短文档得分应完全相等
    short_long_ids = ["short", "long"]
    short_long_texts = ["apple banana", "apple banana orange grape pear peach"]
    bm25_no_len = BM25Index.build(short_long_ids, short_long_texts, tokenize_fn=tokenize, b=0.0)
    scores = bm25_no_len.score_all("apple")
    assert abs(scores[0] - scores[1]) < 1e-9
