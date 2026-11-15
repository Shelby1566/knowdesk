import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from kb.loader import DEFAULT_TASKS_PATH

@dataclass(frozen=True)
class EvalQuery:
    task_id: str
    query: str
    gold_doc_ids: set[str]

def load_eval_queries(tasks_path: Path = DEFAULT_TASKS_PATH) -> list[EvalQuery]:
    raw_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    for t in raw_tasks:
        tid = str(t["id"])
        us = t.get("user_scenario", {})
        if isinstance(us, dict):
            instr = us.get("instructions", {})
            if isinstance(instr, dict):
                q_text = " ".join(
                    str(instr.get(k, ""))
                    for k in ("task_instructions", "reason_for_call", "known_info")
                    if instr.get(k)
                ).strip()
            else:
                q_text = str(instr).strip()
        else:
            q_text = str(us).strip()
        if not q_text:
            q_text = str(t.get("description", "")).strip()

        req_docs = set(str(d) for d in t.get("required_documents", []))
        if req_docs:
            queries.append(EvalQuery(task_id=tid, query=q_text, gold_doc_ids=req_docs))
    return queries

def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    # 按顺序去重后取前 k 个文档
    dedup: list[str] = []
    seen: set[str] = set()
    for d in retrieved_doc_ids:
        if d not in seen:
            seen.add(d)
            dedup.append(d)
            if len(dedup) >= k:
                break
    hits = len(set(dedup) & gold_doc_ids)
    return hits / len(gold_doc_ids)

def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: set[str]) -> float:
    seen: set[str] = set()
    rank = 0
    for d in retrieved_doc_ids:
        if d in seen:
            continue
        seen.add(d)
        rank += 1
        if d in gold_doc_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retriever(
    queries: list[EvalQuery],
    retrieve_fn: Callable[[str, int], list[str]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    max_k = max(ks)
    r_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n = len(queries)
    if n == 0:
        return {f"Recall@{k}": 0.0 for k in ks} | {"MRR": 0.0}

    for q in queries:
        ranked_docs = retrieve_fn(q.query, max_k)
        for k in ks:
            r_sums[k] += recall_at_k(ranked_docs, q.gold_doc_ids, k)
        mrr_sum += reciprocal_rank(ranked_docs, q.gold_doc_ids)

    metrics = {f"Recall@{k}": r_sums[k] / n for k in ks}
    metrics["MRR"] = mrr_sum / n
    return metrics
