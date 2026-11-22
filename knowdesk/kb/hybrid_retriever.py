from typing import Any, Callable
from knowdesk.agent.retrieval_check import BoundedReRetriever
from knowdesk.agent.subagent import RetrievalSubAgent
from knowdesk.kb.acl import AccessControlContext, can_access, infer_document_acl
from knowdesk.kb.bm25 import BM25Index
from knowdesk.kb.chunk import Chunk, chunk_corpus
from knowdesk.kb.dense import DenseRetriever
from knowdesk.kb.fusion import adaptive_rrf_fusion, linear_fusion, rrf_fusion
from knowdesk.kb.loader import load_documents
from knowdesk.kb.retrieval_config import RetrievalDefaults, load_retrieval_defaults
from knowdesk.kb.router import apply_intent_boost, map_doc_to_product_line, predict_product_lines_keyword
from knowdesk.kb.sections import format_matched_section
from knowdesk.kb.tokenize import tokenize

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
        dense_model = load_retrieval_defaults().dense_model
        self.dense = DenseRetriever.build_or_load(dense_model, chunk_ids, chunk_texts)

    def _fuse_chunks(
        self,
        bm25_hits: list[tuple[str, float]],
        dense_hits: list[tuple[str, float]],
        *,
        fuse_k: int,
        fusion_mode: str | None,
        temperature: float | None,
        alpha: float | None,
        defaults: RetrievalDefaults,
    ) -> list[tuple[str, float]]:
        mode = fusion_mode or defaults.fusion_mode
        if mode == "adaptive_rrf":
            used_temperature = defaults.temperature if temperature is None else temperature
            return adaptive_rrf_fusion(
                [bm25_hits, dense_hits], k=fuse_k, rrf_k=defaults.rrf_k, temperature=used_temperature
            )
        if mode == "rrf":
            return rrf_fusion([bm25_hits, dense_hits], k=fuse_k, rrf_k=defaults.rrf_k)
        if mode == "linear":
            used_alpha = defaults.alpha if alpha is None else alpha
            return linear_fusion(bm25_hits, dense_hits, alpha=used_alpha, k=fuse_k)
        raise RuntimeError(f"unknown fusion_mode: {mode}")

    def rank_documents(
        self,
        query: str,
        k: int = 10,
        fusion_mode: str | None = None,
        acl: AccessControlContext | None = None,
        *,
        temperature: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        use_intent_boost: bool | None = None,
    ) -> list[tuple[str, float, Chunk]]:
        """Chunk-level fusion, then one chunk per document. This is the online ranking path."""
        defaults = load_retrieval_defaults()
        if not str(query).strip():
            return []
        bm25_hits = self.bm25.search(query, k=defaults.candidate_k)
        dense_hits = self.dense.store.search(self.dense.encode_query(query), k=defaults.candidate_k)
        fused = self._fuse_chunks(
            bm25_hits,
            dense_hits,
            fuse_k=max(k * 6, defaults.candidate_k),
            fusion_mode=fusion_mode,
            temperature=temperature,
            alpha=alpha,
            defaults=defaults,
        )
        collapsed: list[tuple[str, float, Chunk]] = []
        seen: set[str] = set()
        for chunk_id, score in fused:
            chunk = self.chunk_map.get(chunk_id)
            if chunk is None or chunk.doc_id in seen:
                continue
            if acl is not None and not can_access(self.doc_acl.get(chunk.doc_id), acl):
                continue
            seen.add(chunk.doc_id)
            collapsed.append((chunk.doc_id, score, chunk))
        boost = defaults.use_intent_boost if use_intent_boost is None else use_intent_boost
        used_beta = defaults.beta if beta is None else beta
        if boost and used_beta:
            order = {doc_id: (score, chunk) for doc_id, score, chunk in collapsed}
            boosted = apply_intent_boost(
                [(doc_id, score) for doc_id, score, _ in collapsed],
                self.doc_to_line,
                predict_product_lines_keyword(query),
                beta=used_beta,
                k=len(collapsed),
            )
            collapsed = [(doc_id, score, order[doc_id][1]) for doc_id, score in boosted if doc_id in order]
        return collapsed[:k]

    def search_chunks(
        self,
        query: str,
        k: int = 10,
        fusion_mode: str | None = None,
        acl: AccessControlContext | None = None,
        *,
        temperature: float | None = None,
        alpha: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """返回命中的子块 (Child Chunks) 及其融合分数，支持零信任 ACL 过滤"""
        defaults = load_retrieval_defaults()
        bm25_hits = self.bm25.search(query, k=defaults.candidate_k)
        dense_hits = self.dense.store.search(self.dense.encode_query(query), k=defaults.candidate_k)
        fused = self._fuse_chunks(
            bm25_hits,
            dense_hits,
            fuse_k=k * 2,
            fusion_mode=fusion_mode,
            temperature=temperature,
            alpha=alpha,
            defaults=defaults,
        )
        results: list[tuple[Chunk, float]] = []
        for chunk_id, score in fused:
            chunk = self.chunk_map.get(chunk_id)
            if chunk is None:
                continue
            if acl is not None and not can_access(self.doc_acl.get(chunk.doc_id), acl):
                continue
            results.append((chunk, score))
            if len(results) >= k:
                break
        return results

    def search_parent_sections(
        self, query: str, k: int = 10, fusion_mode: str | None = None, acl: AccessControlContext | None = None
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
        self,
        query: str,
        k: int = 10,
        fusion_mode: str | None = None,
        acl: AccessControlContext | None = None,
        *,
        temperature: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        use_intent_boost: bool | None = None,
    ) -> list[dict[str, Any]]:
        ranked = self.rank_documents(
            query,
            k=k,
            fusion_mode=fusion_mode,
            acl=acl,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            use_intent_boost=use_intent_boost,
        )
        results = []
        for doc_id, score, _chunk in ranked:
            document = self.doc_map.get(doc_id)
            if document is None:
                continue
            results.append({
                "id": document.doc_id,
                "title": document.title,
                "content": document.content,
                "score": round(score, 5),
            })
        return results

    def search_sections(
        self,
        query: str,
        k: int = 10,
        fusion_mode: str | None = None,
        acl: AccessControlContext | None = None,
        *,
        temperature: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        use_intent_boost: bool | None = None,
    ) -> list[dict[str, Any]]:
        """给模型的检索结果：每个文档只保留得分最高的 H2 章节。"""
        ranked = self.rank_documents(
            query,
            k=k,
            fusion_mode=fusion_mode,
            acl=acl,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            use_intent_boost=use_intent_boost,
        )
        results: list[dict[str, Any]] = []
        for doc_id, score, chunk in ranked:
            parent = self.doc_map[doc_id]
            results.append({
                "id": doc_id,
                "chunk_id": chunk.chunk_id,
                "title": parent.title,
                "breadcrumb": chunk.breadcrumb,
                "content": format_matched_section(chunk),
                "score": round(score, 5),
            })
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
