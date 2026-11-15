from kb.loader import load_documents
from kb.chunk import chunk_corpus
from kb.tokenize import tokenize
from kb.bm25 import BM25Index
from kb.dense import DenseRetriever
from kb.fusion import rrf_fusion
from kb.router import map_doc_to_product_line, predict_product_lines_keyword, apply_intent_boost

class KnowDeskHybridRetriever:
    """整合 M1(面包屑切块) + M2(手写BM25) + M5(Qwen3-Embedding-0.6B) + M6(RRF) + M8(产品线意图软加权)"""
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
        self.chunk_to_doc = {c.chunk_id: c.doc_id for c in self.chunks}
        self.doc_to_line = {d.doc_id: map_doc_to_product_line(d.doc_id, d.title) for d in self.docs}

        chunk_ids = [c.chunk_id for c in self.chunks]
        chunk_texts = [c.text for c in self.chunks]
        self.bm25 = BM25Index.build(chunk_ids, chunk_texts, tokenize_fn=tokenize)
        self.dense = DenseRetriever.build_or_load("Qwen/Qwen3-Embedding-0.6B", chunk_ids, chunk_texts)

    def search(self, query: str, k: int = 10) -> list[dict]:
        bm25_res = self.bm25.search_docs(query, self.chunk_to_doc, k=50)
        dense_res = self.dense.search_docs(query, self.chunk_to_doc, k=50)
        rrf_res = rrf_fusion([bm25_res, dense_res], k=50, rrf_k=60)
        pred_lines = predict_product_lines_keyword(query)
        boosted = apply_intent_boost(rrf_res, self.doc_to_line, pred_lines, beta=0.15, k=k)

        results = []
        for did, sc in boosted:
            d = self.doc_map.get(did)
            if d:
                results.append({"id": d.doc_id, "title": d.title, "content": d.content, "score": round(sc, 5)})
        return results
