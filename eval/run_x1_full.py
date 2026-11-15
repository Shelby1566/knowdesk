import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from kb.loader import load_documents
from kb.chunk import chunk_corpus
from kb.tokenize import tokenize, whitespace_tokenize
from kb.bm25 import BM25Index
from kb.dense import DenseRetriever
from kb.fusion import rrf_fusion, linear_fusion
from kb.rerank import LocalReranker
from kb.router import (
    map_doc_to_product_line,
    predict_product_lines_keyword,
    apply_intent_boost,
    evaluate_multilabel_f1,
)
from kb.eval_retrieval import EvalQuery, load_eval_queries, recall_at_k, reciprocal_rank

COLLOQUIAL_PATH = Path(__file__).resolve().parents[1] / "kb/colloquial_queries.json"
REPORT_PATH = Path(__file__).resolve().parents[1] / "RESULTS_X1.md"

def load_colloquial_eval_queries() -> list[EvalQuery]:
    raw = json.loads(COLLOQUIAL_PATH.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            task_id=item["task_id"],
            query=item["colloquial_query"],
            gold_doc_ids=set(item["gold_doc_ids"]),
        )
        for item in raw
    ]

def eval_ranked_lists(
    queries: list[EvalQuery],
    ranked_docs_per_q: list[list[str]],
    latencies_ms: list[float],
) -> dict[str, float]:
    n = len(queries)
    r5 = sum(recall_at_k(d, q.gold_doc_ids, 5) for q, d in zip(queries, ranked_docs_per_q)) / n
    r10 = sum(recall_at_k(d, q.gold_doc_ids, 10) for q, d in zip(queries, ranked_docs_per_q)) / n
    r20 = sum(recall_at_k(d, q.gold_doc_ids, 20) for q, d in zip(queries, ranked_docs_per_q)) / n
    mrr = sum(reciprocal_rank(d, q.gold_doc_ids) for q, d in zip(queries, ranked_docs_per_q)) / n
    lats = sorted(latencies_ms) if latencies_ms else [0.0]
    p50 = lats[len(lats) // 2]
    p90 = lats[int(len(lats) * 0.9)]
    return {"Recall@5": r5, "Recall@10": r10, "Recall@20": r20, "MRR": mrr, "P50_ms": p50, "P90_ms": p90}

def main():
    docs = load_documents()
    chunks = chunk_corpus(docs)
    orig_queries = load_eval_queries()
    coll_queries = load_colloquial_eval_queries()

    chunk_to_doc = {c.chunk_id: c.doc_id for c in chunks}
    doc_to_line = {d.doc_id: map_doc_to_product_line(d.doc_id, d.title) for d in docs}
    # 每篇文档的最佳代表文本（标题 + 前800字符）用于文档级快速重排
    doc_rep_text = {d.doc_id: f"{d.title}\n{d.content[:800]}" for d in docs}

    # 评测 M8 意图路由多标签 F1
    y_true = [{doc_to_line[did] for did in q.gold_doc_ids if did in doc_to_line} for q in orig_queries]
    y_pred_orig = [predict_product_lines_keyword(q.query) for q in orig_queries]
    y_pred_coll = [predict_product_lines_keyword(q.query) for q in coll_queries]
    m8_orig = evaluate_multilabel_f1(y_true, y_pred_orig)
    m8_coll = evaluate_multilabel_f1(y_true, y_pred_coll)
    print(f"✅ M8 产品线意图路由 (Original) -> Micro-F1: {m8_orig['micro_f1']:.4f} | Macro-F1: {m8_orig['macro_f1']:.4f}")
    print(f"✅ M8 产品线意图路由 (Colloquial) -> Micro-F1: {m8_coll['micro_f1']:.4f} | Macro-F1: {m8_coll['macro_f1']:.4f}")

    # 构建 BM25 索引
    print("🔄 构建 BM25 各配置索引...")
    idx_tau_base = BM25Index.build([d.doc_id for d in docs], [f"{d.title}\n{d.content}" for d in docs], tokenize_fn=whitespace_tokenize)
    idx_doc_clean = BM25Index.build([d.doc_id for d in docs], [f"{d.title}\n{d.content}" for d in docs], tokenize_fn=tokenize)
    idx_chunk_no_bc = BM25Index.build([c.chunk_id for c in chunks], [c.body if c.body else c.breadcrumb for c in chunks], tokenize_fn=tokenize)
    idx_chunk_bc = BM25Index.build([c.chunk_id for c in chunks], [c.text for c in chunks], tokenize_fn=tokenize)

    # 构建两个稠密向量检索器
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("🔄 加载/编码稠密模型 1: BAAI/bge-m3 ...")
    dense_bge = DenseRetriever.build_or_load("BAAI/bge-m3", chunk_ids, chunk_texts)
    bge_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    print("🔄 加载/编码稠密模型 2: Qwen/Qwen3-Embedding-0.6B ...")
    dense_qwen = DenseRetriever.build_or_load("Qwen/Qwen3-Embedding-0.6B", chunk_ids, chunk_texts)
    qwen_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"📌 显存峰值记录: bge-m3={bge_mem_mb:.1f} MB | Qwen3-Embedding-0.6B={qwen_mem_mb:.1f} MB")

    # 预先跑完所有检索结果，避免重复计算
    def run_suite(q_list: list[EvalQuery], label: str):
        print(f"\n================ 正在评测查询集: {label} (n={len(q_list)}) ================")
        rows: list[tuple[str, dict[str, float]]] = []

        def _measure(name: str, fn):
            ranked_all = []
            lats = []
            for q in q_list:
                t0 = time.perf_counter()
                res = fn(q.query)
                lats.append((time.perf_counter() - t0) * 1000.0)
                ranked_all.append([d for d, _ in res])
            m = eval_ranked_lists(q_list, ranked_all, lats)
            rows.append((name, m))
            print(f"  ✓ {name}: R@5={m['Recall@5']:.4f}, R@10={m['Recall@10']:.4f}, R@20={m['Recall@20']:.4f}, MRR={m['MRR']:.4f} (P50={m['P50_ms']:.1f}ms)")
            return ranked_all

        _measure("1. τ内置基线 (文档级 + 空格分词)", lambda q: idx_tau_base.search(q, 50))
        _measure("2. 手写BM25 (文档级 + 去标点分词)", lambda q: idx_doc_clean.search(q, 50))
        _measure("3. 手写BM25 (H2切块 - 无面包屑)", lambda q: idx_chunk_no_bc.search_docs(q, chunk_to_doc, 50))

        bm25_bc_cache: dict[str, list[tuple[str, float]]] = {}
        bge_cache: dict[str, list[tuple[str, float]]] = {}
        qwen_cache: dict[str, list[tuple[str, float]]] = {}

        def _get_bm25_bc(q: str):
            if q not in bm25_bc_cache:
                bm25_bc_cache[q] = idx_chunk_bc.search_docs(q, chunk_to_doc, 50)
            return bm25_bc_cache[q]

        def _get_bge(q: str):
            if q not in bge_cache:
                bge_cache[q] = dense_bge.search_docs(q, chunk_to_doc, 50)
            return bge_cache[q]

        def _get_qwen(q: str):
            if q not in qwen_cache:
                qwen_cache[q] = dense_qwen.search_docs(q, chunk_to_doc, 50)
            return qwen_cache[q]

        _measure("4. 手写BM25 (H2切块 + 面包屑)", _get_bm25_bc)
        _measure("5. 稠密检索: BAAI/bge-m3 (切块 + 面包屑)", _get_bge)
        _measure("6. 稠密检索: Qwen3-Embedding-0.6B (切块 + 面包屑)", _get_qwen)

        # 选召回更高的稠密路与 BM25 融合
        best_dense_fn = _get_bge if rows[-2][1]["Recall@20"] >= rows[-1][1]["Recall@20"] else _get_qwen
        best_dense_tag = "bge-m3" if best_dense_fn is _get_bge else "Qwen3-Emb"

        _measure(
            f"7. 混合检索: BM25 + {best_dense_tag} (Min-Max 线性加权 α=0.5)",
            lambda q: linear_fusion(_get_bm25_bc(q), best_dense_fn(q), alpha=0.5, k=50),
        )

        rrf_cache: dict[str, list[tuple[str, float]]] = {}
        def _get_rrf(q: str):
            if q not in rrf_cache:
                rrf_cache[q] = rrf_fusion([_get_bm25_bc(q), _get_bge(q), _get_qwen(q)], k=50, rrf_k=60)
            return rrf_cache[q]

        _measure("8. 混合检索: BM25 + 双路稠密 (RRF k=60)", _get_rrf)
        _measure(
            "9. 混合检索 + M8产品线意图软加权 (RRF + Intent Boost)",
            lambda q: apply_intent_boost(_get_rrf(q), doc_to_line, predict_product_lines_keyword(q), beta=0.15, k=50),
        )
        return rows, rrf_cache

    rows_orig, rrf_orig = run_suite(orig_queries, "原始工单查询 (Original)")
    rows_coll, rrf_coll = run_suite(coll_queries, "口语化首轮查询 (Colloquial)")

    # 释放编码模型显存，加载重排模型对 Top-30 候选做精排
    del dense_bge.model
    del dense_qwen.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for rerank_idx, rerank_model in enumerate(["BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-0.6B"], start=10):
        print(f"\n🔄 加载本地重排模型: {rerank_model} (对 RRF Top-30 候选重排)...")
        reranker = LocalReranker(rerank_model)
        short_name = rerank_model.split("/")[-1]

        for q_list, rrf_map, rows_target in [(orig_queries, rrf_orig, rows_orig), (coll_queries, rrf_coll, rows_coll)]:
            ranked_all = []
            lats = []
            for q_idx, q in enumerate(q_list, 1):
                if q_idx % 25 == 0 or q_idx == len(q_list): print(f"    -> 重排进度: {q_idx}/{len(q_list)}", flush=True)
                cands = [(did, doc_rep_text[did]) for did, _ in rrf_map[q.query][:30] if did in doc_rep_text]
                reranked, ms = reranker.rerank(q.query, cands, top_k=20)
                lats.append(ms)
                ranked_all.append([d for d, _ in reranked])
            m = eval_ranked_lists(q_list, ranked_all, lats)
            label_row = f"{rerank_idx}. RRF混合 + {short_name} 重排 (Top30->20)"
            rows_target.append((label_row, m))
            print(f"  ✓ {label_row}: R@5={m['Recall@5']:.4f}, R@10={m['Recall@10']:.4f}, R@20={m['Recall@20']:.4f}, MRR={m['MRR']:.4f} (P50={m['P50_ms']:.1f}ms)")

        del reranker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 生成 Markdown 报告
    lines = ["# X1 检索全链路消融实验结果 (n=97)\n"]
    lines.append(f"- **M8 产品线意图路由 (Original)**: Micro-F1 = `{m8_orig['micro_f1']:.4f}`, Macro-F1 = `{m8_orig['macro_f1']:.4f}`")
    lines.append(f"- **M8 产品线意图路由 (Colloquial)**: Micro-F1 = `{m8_coll['micro_f1']:.4f}`, Macro-F1 = `{m8_coll['macro_f1']:.4f}`\n")

    for title, rows in [("表 1A：原始工单查询集 (Original Queries, n=97)", rows_orig), ("表 1B：口语化首轮查询集 (Colloquial First-Turn Queries, n=97)", rows_coll)]:
        lines.append(f"## {title}\n")
        lines.append("| 检索配置 | Recall@5 | Recall@10 | Recall@20 | MRR | P50延迟(ms) | P90延迟(ms) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in rows:
            lines.append(
                f"| {name} | {m['Recall@5']:.4f} | {m['Recall@10']:.4f} | {m['Recall@20']:.4f} | {m['MRR']:.4f} | {m['P50_ms']:.1f} | {m['P90_ms']:.1f} |"
            )
        lines.append("\n")

    report_md = "\n".join(lines)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print("\n" + "=" * 70)
    print(report_md)
    print(f"✅ 完整报告已保存至: {REPORT_PATH}")

if __name__ == "__main__":
    main()
