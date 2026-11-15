from kb.chunk import Chunk
from eval.pass_k import pass_at_k_unbiased, paired_bootstrap_diff
from agent.context import compress_context_watermark
from agent.retrieval_check import BoundedReRetriever
from agent.subagent import RetrievalSubAgent
from obs.otel_setup import GenAITracer

def test_m21_pass_k_and_m19_bootstrap():
    # M21 Test case: n=4, c=2 -> pass^2 = C(2,2)/C(4,2) = 1/6
    assert abs(pass_at_k_unbiased(4, 2, 2) - (1.0 / 6.0)) < 1e-9
    assert pass_at_k_unbiased(4, 4, 3) == 1.0
    assert pass_at_k_unbiased(4, 1, 2) == 0.0

    # M19 配对 Bootstrap 门禁
    base = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    cand = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    res = paired_bootstrap_diff(base, cand, seed=42)
    assert res["mean_diff"] == 0.4
    assert res["non_regressing"] is True

def test_m13_watermark_context_compression():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "name": "KB_search"}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "DocTitle > Fees\n" + ("long text " * 200)},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "ans"},
    ]
    # 未达高水位线时不改动
    c_no, flag_no = compress_context_watermark(msgs, high_watermark_tokens=99999, keep_recent_messages=2)
    assert flag_no is False and c_no[2]["content"] == msgs[2]["content"]

    # 触发水位线时压缩早期大段 tool 内容，且保留 tool_call_id 成对
    c_yes, flag_yes = compress_context_watermark(msgs, high_watermark_tokens=100, keep_recent_messages=2)
    assert flag_yes is True
    assert "COMPRESSED_KB_RESULT" in c_yes[2]["content"]
    assert c_yes[2]["tool_call_id"] == "tc1"

def test_m14_bounded_reretrieval_and_m15_subagent():
    # M14: 一直判不足时精确停在 max_retries=2 次并标记 needs_human_handoff
    r_Full = BoundedReRetriever(
        retrieve_fn=lambda q, k: [{"id": "d1", "title": "T", "content": "C"}],
        judge_fn=lambda q, docs: '{"sufficient": false, "missing_aspect": "limit", "rewritten_query": "q_new"}',
        max_retries=2,
    )
    out = r_Full.retrieve_with_check("init_q", k=5)
    assert out["attempts"] == 2
    assert out["needs_human_handoff"] is True

    # M15: 子 Agent 提取的 exact_excerpt 必是原文子串
    c1 = Chunk("doc1#0", "doc1", "Card > Fees", "Annual fee is $95.", "Card > Fees\nAnnual fee is $95.", "card")
    sub = RetrievalSubAgent({"doc1#0": c1}, lambda q, cs: [("doc1#0", "has fee"), ("bad_id", "none")])
    ext = sub.isolate_and_extract("fee", [c1])
    assert len(ext) == 1
    assert ext[0]["exact_excerpt"] in c1.text
    assert sub.invalid_ref_count == 1

def test_m17_otel_and_m18_cache_metrics():
    tracer = GenAITracer()
    tracer.record_llm_call("qwen3.8-flash", "qwen3.8-flash", input_tokens=1000, output_tokens=100, cached_tokens=600, cost_rmb=0.01)
    m = tracer.summary_metrics()
    assert abs(m["cache_hit_rate"] - 0.60) < 1e-9
    assert m["total_spans"] == 1
