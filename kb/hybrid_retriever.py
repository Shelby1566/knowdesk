from typing import Any, Callable
from agent.retrieval_check import BoundedReRetriever
from agent.subagent import RetrievalSubAgent
from kb.acl import AccessControlContext, can_access, infer_document_acl
from kb.bm25 import BM25Index
from kb.chunk import Chunk, chunk_corpus
from kb.dense import DenseRetriever
from kb.fusion import adaptive_rrf_fusion, rrf_fusion
from kb.loader import load_documents
from kb.router import apply_intent_boost, map_doc_to_product_line, predict_product_lines_keyword
from kb.sections import format_matched_section
from kb.tokenize import tokenize

class KnowDeskHybridRetriever:
    """
    整合 M1(面包屑切块) + Small-to-Big 父子块回拉 + M2(倒排BM25) + M5(Qwen3-Embedding)
    + M6(RRF) + M8(意图软加权) + M14(有界重检索) + M15(检索子Agent上下文隔离)
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "KnowDeskHybridRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.docs = load_documents()
        self.doc_map = {d.doc_id: d for d in self.docs}
        self.chunks = chunk_corpus(self.docs)
        self.chunk_map = {c.chunk_id: c for c in self.chunks}
        self.chunk_to_doc = {c.chunk_id: c.doc_id for c in self.chunks}
        self.doc_to_line = {d.doc_id: map_doc_to_product_line(d.doc_id, d.title) for d in self.docs}
        self.doc_acl = {d.doc_id: infer_document_acl(d.doc_id, d.title, d.content) for d in self.docs}

        chunk_ids = [c.chunk_id for c in self.chunks]
        chunk_texts = [c.text for c in self.chunks]
        self.bm25 = BM25Index.build(chunk_ids, chunk_texts, tokenize_fn=tokenize)
        self.dense = DenseRetriever.build_or_load("Qwen/Qwen3-Embedding-0.6B", chunk_ids, chunk_texts)

    def search_chunks(
        self, query: str, k: int = 10, fusion_mode: str = "adaptive_rrf", acl: AccessControlContext | None = None
    ) -> list[tuple[Chunk, float]]:
        """返回命中的子块 (Child Chunks) 及其 RRF / AdaRRF 分数，支持零信任 ACL 过滤"""
        b_res = self.bm25.search(query, k=50)
        q_vec = self.dense.encode_query(query)
        d_res = self.dense.store.search(q_vec, k=50)
        fuse_fn = adaptive_rrf_fusion if fusion_mode == "adaptive_rrf" else rrf_fusion
        fused = fuse_fn([b_res, d_res], k=k * 2, rrf_k=60)
        results: list[tuple[Chunk, float]] = []
        for cid, sc in fused:
            if cid in self.chunk_map:
                ch = self.chunk_map[cid]
                if acl is not None and not can_access(self.doc_acl.get(ch.doc_id), acl):
                    continue
                results.append((ch, sc))
                if len(results) >= k:
                    break
        return results

    def search_parent_sections(
        self, query: str, k: int = 10, fusion_mode: str = "adaptive_rrf", acl: AccessControlContext | None = None
    ) -> list[dict[str, Any]]:
        """
        Parent-Document Retrieval (Small-to-Big):
        先命中子块，再只返回该子块所属的父级 H2 正文 parent_section_text，以及 doc_id / title。
        不附带整篇文档，避免把未命中的章节送进上下文。
        """
        matched_chunks = self.search_chunks(query, k=k * 3, fusion_mode=fusion_mode, acl=acl)
        seen_docs: set[str] = set()
        results: list[dict[str, Any]] = []
        for ch, sc in matched_chunks:
            if ch.doc_id in seen_docs:
                continue
            seen_docs.add(ch.doc_id)
            parent_doc = self.doc_map[ch.doc_id]
            results.append({
                "id": ch.doc_id,
                "doc_id": ch.doc_id,
                "matched_chunk_id": ch.chunk_id,
                "breadcrumb": ch.breadcrumb,
                "title": parent_doc.title,
                "parent_section_text": ch.text,
                "score": round(sc, 5),
            })
            if len(results) >= k:
                break
        return results

    def search(
        self, query: str, k: int = 10, fusion_mode: str = "adaptive_rrf", acl: AccessControlContext | None = None
    ) -> list[dict[str, Any]]:
        bm25_res = self.bm25.search_docs(query, self.chunk_to_doc, k=50)
        dense_res = self.dense.search_docs(query, self.chunk_to_doc, k=50)
        fuse_fn = adaptive_rrf_fusion if fusion_mode == "adaptive_rrf" else rrf_fusion
        rrf_res = fuse_fn([bm25_res, dense_res], k=50, rrf_k=60)
        pred_lines = predict_product_lines_keyword(query)
        boosted = apply_intent_boost(rrf_res, self.doc_to_line, pred_lines, beta=0.15, k=k * 2)

        results = []
        for did, sc in boosted:
            d = self.doc_map.get(did)
            if d:
                if acl is not None and not can_access(self.doc_acl.get(did), acl):
                    continue
                results.append({"id": d.doc_id, "title": d.title, "content": d.content, "score": round(sc, 5)})
                if len(results) >= k:
                    break
        return results

    def search_sections(
        self, query: str, k: int = 10, fusion_mode: str = "adaptive_rrf", acl: AccessControlContext | None = None
    ) -> list[dict[str, Any]]:
        """给模型的检索结果：每个文档只保留得分最高的 H2 章节。"""
        matched = self.search_chunks(query, k=max(k * 3, k), fusion_mode=fusion_mode, acl=acl)
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for chunk, score in matched:
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            parent = self.doc_map[chunk.doc_id]
            results.append({
                "id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": parent.title,
                "breadcrumb": chunk.breadcrumb,
                "content": format_matched_section(chunk),
                "score": round(score, 5),
            })
            if len(results) >= k:
                break
        return results

    def search_with_bounded_check(
        self, query: str, judge_fn: Callable[[str, list[dict[str, Any]]], str], k: int = 10, max_retries: int = 2
    ) -> dict[str, Any]:
        """M14 接入主检索器：带充分性评估与查询改写的有界重检索"""
        checker = BoundedReRetriever(self.search, judge_fn=judge_fn, max_retries=max_retries)
        return checker.retrieve_with_check(query, k=k)

    def search_with_subagent_isolation(
        self, query: str, selector_fn: Callable[[str, list[Chunk]], list[tuple[str, str]]] | None = None, k: int = 5
    ) -> list[dict[str, str]]:
        """M15 接入主检索器：通过检索子 Agent 只向主 Agent 返回命中的 chunk_id 与原文精确摘录"""
        cands = [ch for ch, _ in self.search_chunks(query, k=k * 3)]
        default_sel = selector_fn or (lambda q, cs: [(c.chunk_id, "Top RRF matched policy section") for c in cs[:k]])
        sub = RetrievalSubAgent(self.chunk_map, default_sel)
        return sub.isolate_and_extract(query, cands, max_selected=k)
