from typing import Any, Callable
from kb.chunk import Chunk

class RetrievalSubAgent:
    """
    M15: 检索子 Agent（上下文隔离）
    子 Agent 在独立上下文中阅读候选块，只返回命中的 chunk_id 与一句话理由；
    由确定性代码从内存字典提取 100% 原文片段交给主 Agent，彻底杜绝条款篡改并节省主 Agent 上下文 Token。
    """
    def __init__(self, chunk_lookup: dict[str, Chunk], selector_fn: Callable[[str, list[Chunk]], list[tuple[str, str]]]):
        self.chunk_lookup = chunk_lookup
        self.selector_fn = selector_fn
        self.invalid_ref_count = 0
        self.out_of_candidate_ref_count = 0

    def isolate_and_extract(self, query: str, candidate_chunks: list[Chunk], max_selected: int = 5) -> list[dict[str, str]]:
        selections = self.selector_fn(query, candidate_chunks)
        candidate_ids = {chunk.chunk_id for chunk in candidate_chunks}
        extracted: list[dict[str, str]] = []
        for cid, reason in selections:
            if cid not in candidate_ids:
                if cid in self.chunk_lookup:
                    self.out_of_candidate_ref_count += 1
                else:
                    self.invalid_ref_count += 1
                continue
            chunk = self.chunk_lookup.get(cid)
            if chunk is None:
                self.invalid_ref_count += 1
                continue
            extracted.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "breadcrumb": chunk.breadcrumb,
                "exact_excerpt": chunk.body if chunk.body else chunk.breadcrumb,
                "reason": reason,
            })
            if len(extracted) >= max_selected:
                break
        return extracted
