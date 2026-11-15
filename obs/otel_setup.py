import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_RE_CARD = re.compile(r"\b(?:\d{4}[ -]?){3}(\d{4})\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-(\d{4})\b")
_RE_PHONE = re.compile(r"\b(\d{3})[-.]?\d{3}[-.]?(\d{4})\b")
_RE_EMAIL = re.compile(r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")

def redact_pii(text: str) -> str:
    """金融级 PII 审计脱敏：屏蔽 16 位卡号、SSN、手机号与邮箱"""
    s = _RE_CARD.sub(r"****-****-****-\1", str(text))
    s = _RE_SSN.sub(r"***-**-\1", s)
    s = _RE_PHONE.sub(r"\1-***-\2", s)
    s = _RE_EMAIL.sub(r"\1***@\2", s)
    return s

def redact_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            cleaned[k] = redact_pii(v)
        else:
            cleaned[k] = v
    return cleaned

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
    """M17: 遵循 OpenTelemetry GenAI Semantic Conventions 并内置金融级 PII 脱敏的 Span 收集器"""
    def __init__(self, record_raw_user_text: bool = False):
        self.record_raw_user_text = record_raw_user_text
        self.spans: list[GenAISpan] = []

    @contextmanager
    def start_span(self, name: str, **attrs: Any):
        safe_attrs = redact_attributes(attrs)
        span = GenAISpan(name=name, start_time_ms=time.perf_counter() * 1000.0, attributes=safe_attrs)
        try:
            yield span
        finally:
            span.end_time_ms = time.perf_counter() * 1000.0
            span.attributes = redact_attributes(span.attributes)
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
