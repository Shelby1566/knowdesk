import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.loader import load_documents
from kb.chunk import chunk_corpus
from kb.tokenize import tokenize, whitespace_tokenize
from kb.bm25 import BM25Index
from kb.eval_retrieval import load_eval_queries, evaluate_retriever

def main():
    docs = load_documents()
    chunks = chunk_corpus(docs)
    queries = load_eval_queries()
    chunk_to_doc = {c.chunk_id: c.doc_id for c in chunks}

    idx_tau_base = BM25Index.build(
        [d.doc_id for d in docs], [f"{d.title}\n{d.content}" for d in docs], tokenize_fn=whitespace_tokenize
    )
    idx_doc_clean = BM25Index.build(
        [d.doc_id for d in docs], [f"{d.title}\n{d.content}" for d in docs], tokenize_fn=tokenize
    )
    idx_chunk_no_bc = BM25Index.build(
        [c.chunk_id for c in chunks], [c.body if c.body else c.breadcrumb for c in chunks], tokenize_fn=tokenize
    )
    idx_chunk_bc = BM25Index.build(
        [c.chunk_id for c in chunks], [c.text for c in chunks], tokenize_fn=tokenize
    )

    configs = [
        ("1. τ内置基线 (文档级 + 空格分词)", lambda q, k: [d for d, _ in idx_tau_base.search(q, k)]),
        ("2. 手写BM25 (文档级 + 去标点分词)", lambda q, k: [d for d, _ in idx_doc_clean.search(q, k)]),
        ("3. 手写BM25 (H2切块 - 无面包屑)", lambda q, k: [d for d, _ in idx_chunk_no_bc.search_docs(q, chunk_to_doc, k)]),
        ("4. 手写BM25 (H2切块 + 面包屑)", lambda q, k: [d for d, _ in idx_chunk_bc.search_docs(q, chunk_to_doc, k)]),
    ]

    print(f"\n📊 X1 检索实验表（前4行，n={len(queries)} 个工单查询）：\n")
    print("| 检索配置 | Recall@5 | Recall@10 | Recall@20 | MRR |")
    print("|---|---:|---:|---:|---:|")
    for name, fn in configs:
        m = evaluate_retriever(queries, fn, ks=(5, 10, 20))
        print(f"| {name} | {m['Recall@5']:.4f} | {m['Recall@10']:.4f} | {m['Recall@20']:.4f} | {m['MRR']:.4f} |")

if __name__ == "__main__":
    main()
