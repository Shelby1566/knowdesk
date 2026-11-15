import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable
from kb.tokenize import tokenize

@dataclass
class BM25Index:
    item_ids: list[str]
    doc_freqs: list[Counter[str]]
    doc_lens: list[int]
    avgdl: float
    idf: dict[str, float]
    tokenize_fn: Callable[[str], list[str]] = tokenize
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        item_ids: list[str],
        texts: list[str],
        tokenize_fn: Callable[[str], list[str]] = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> "BM25Index":
        doc_freqs: list[Counter[str]] = []
        doc_lens: list[int] = []
        df: Counter[str] = Counter()

        for text in texts:
            tokens = tokenize_fn(text)
            counts = Counter(tokens)
            doc_freqs.append(counts)
            doc_lens.append(len(tokens))
            for word in counts.keys():
                df[word] += 1

        n_docs = len(item_ids)
        avgdl = sum(doc_lens) / n_docs if n_docs > 0 else 0.0

        raw_idf: dict[str, float] = {}
        for word, freq in df.items():
            raw_idf[word] = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)

        avg_idf = sum(raw_idf.values()) / len(raw_idf) if raw_idf else 0.0
        idf: dict[str, float] = {
            word: (epsilon * avg_idf if val < 0 else val)
            for word, val in raw_idf.items()
        }

        return cls(
            item_ids=item_ids,
            doc_freqs=doc_freqs,
            doc_lens=doc_lens,
            avgdl=avgdl,
            idf=idf,
            tokenize_fn=tokenize_fn,
            k1=k1,
            b=b,
        )

    def score_all(self, query: str) -> list[float]:
        q_tokens = self.tokenize_fn(query)
        scores = [0.0] * len(self.item_ids)
        if not q_tokens or self.avgdl == 0.0:
            return scores

        for i, freq_map in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            norm = 1.0 - self.b + self.b * (dl / self.avgdl)
            s = 0.0
            for t in q_tokens:
                f = freq_map.get(t, 0)
                if f == 0:
                    continue
                w_idf = self.idf.get(t, 0.0)
                s += w_idf * (f * (self.k1 + 1.0)) / (f + self.k1 * norm)
            scores[i] = s
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.tokenize_fn(query):
            return []
        scores = self.score_all(query)
        ranked = sorted(zip(self.item_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search_docs(
        self, query: str, chunk_to_doc: dict[str, str], k: int = 10
    ) -> list[tuple[str, float]]:
        """块级检索后按 doc_id 去重（保留每个文档最高得分的块排名）"""
        if not self.tokenize_fn(query):
            return []
        scores = self.score_all(query)
        ranked_chunks = sorted(zip(self.item_ids, scores), key=lambda x: x[1], reverse=True)
        seen_docs: set[str] = set()
        doc_results: list[tuple[str, float]] = []
        for cid, sc in ranked_chunks:
            did = chunk_to_doc.get(cid, cid)
            if did not in seen_docs:
                seen_docs.add(did)
                doc_results.append((did, sc))
                if len(doc_results) >= k:
                    break
        return doc_results
