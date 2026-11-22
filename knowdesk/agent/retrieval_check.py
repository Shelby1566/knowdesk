from typing import Any, Callable
from knowdesk.tools.schemas import SufficiencyOutput, validate_structured_output

class BoundedReRetriever:
    """M14: 检索充分性判断 + 有界查询改写重检索（最多 max_retries 次，超限标记需转人工）"""
    def __init__(
        self,
        retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
        judge_fn: Callable[[str, list[dict[str, Any]]], str],
        max_retries: int = 2,
    ):
        self.retrieve_fn = retrieve_fn
        self.judge_fn = judge_fn
        self.max_retries = max_retries
        self.parse_failures: list[int] = []

    def retrieve_with_check(self, initial_query: str, k: int = 10) -> dict[str, Any]:
        seen_queries: set[str] = {initial_query.strip().lower()}
        current_query = initial_query
        merged_docs: dict[str, dict[str, Any]] = {}
        attempts = 0

        while True:
            hits = self.retrieve_fn(current_query, k)
            for h in hits:
                merged_docs.setdefault(h["id"], h)

            raw_judge = self.judge_fn(current_query, list(merged_docs.values()))
            parsed, _ = validate_structured_output(
                raw_judge, SufficiencyOutput, fallback=SufficiencyOutput(sufficient=True), failure_counter=self.parse_failures
            )
            if parsed.sufficient:
                return {
                    "docs": list(merged_docs.values())[:k],
                    "attempts": attempts,
                    "sufficient": True,
                    "needs_human_handoff": False,
                }

            if attempts >= self.max_retries:
                return {
                    "docs": list(merged_docs.values())[:k],
                    "attempts": attempts,
                    "sufficient": False,
                    "needs_human_handoff": True,
                }

            attempts += 1
            next_q = (parsed.rewritten_query or f"{initial_query} {parsed.missing_aspect}").strip()
            if next_q.lower() in seen_queries:
                next_q = f"{next_q} policy details step {attempts}"
            seen_queries.add(next_q.lower())
            current_query = next_q
