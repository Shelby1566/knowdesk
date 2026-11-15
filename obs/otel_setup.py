import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GenAISpan:
    name: str
    start_time_ms: float
    end_time_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_time_ms - self.start_time_ms)

class GenAITracer:
    """M17: 遵循 OpenTelemetry GenAI Semantic Conventions (gen_ai.*) 的轻量 Span 收集器"""
    def __init__(self, record_raw_user_text: bool = False):
        self.record_raw_user_text = record_raw_user_text
        self.spans: list[GenAISpan] = []

    @contextmanager
    def start_span(self, name: str, **attrs: Any):
        span = GenAISpan(name=name, start_time_ms=time.perf_counter() * 1000.0, attributes=dict(attrs))
        try:
            yield span
        finally:
            span.end_time_ms = time.perf_counter() * 1000.0
            self.spans.append(span)

    def record_llm_call(
        self,
        request_model: str,
        response_model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cost_rmb: float = 0.0,
    ) -> GenAISpan:
        with self.start_span(
            "gen_ai.chat",
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": request_model,
                "gen_ai.response.model": response_model,
                "gen_ai.usage.input_tokens": int(input_tokens),
                "gen_ai.usage.output_tokens": int(output_tokens),
                "gen_ai.usage.cached_tokens": int(cached_tokens),
                "knowdesk.cost_rmb": float(cost_rmb),
            },
        ) as sp:
            return sp

    def summary_metrics(self) -> dict[str, float]:
        llm_spans = [s for s in self.spans if s.attributes.get("gen_ai.operation.name") == "chat"]
        tot_in = sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in llm_spans)
        tot_hit = sum(s.attributes.get("gen_ai.usage.cached_tokens", 0) for s in llm_spans)
        tot_out = sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in llm_spans)
        tot_cost = sum(s.attributes.get("knowdesk.cost_rmb", 0.0) for s in llm_spans)
        hit_rate = (tot_hit / tot_in) if tot_in > 0 else 0.0
        return {
            "total_spans": len(self.spans),
            "total_input_tokens": tot_in,
            "total_cached_tokens": tot_hit,
            "cache_hit_rate": hit_rate,
            "total_output_tokens": tot_out,
            "total_cost_rmb": tot_cost,
        }
