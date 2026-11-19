"""OpenTelemetry SDK tracing with recursive PII redaction before export."""

from __future__ import annotations

import contextvars
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from security.dlp import redact_text

current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "knowdesk_otel_session_id", default=None
)

_provider: TracerProvider | None = None
_ring: "SpanRingBuffer | None" = None
_memory_exporter: InMemorySpanExporter | None = None
_DEFAULT_RING = 500


def redact_pii(text: str) -> str:
    """金融级 PII 审计脱敏：屏蔽 16 位卡号、SSN、手机号与邮箱。"""
    return redact_text(text)


def redact_obj(value: Any) -> Any:
    """递归脱敏字符串，以及嵌套 dict / list / JSON 文本里的同类值。"""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return redact_pii(value)
            if isinstance(parsed, (dict, list)):
                return json.dumps(redact_obj(parsed), ensure_ascii=False)
        return redact_pii(value)
    if isinstance(value, dict):
        return {str(key): redact_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(item) for item in value)
    return value


def redact_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    return redact_obj(dict(attrs))


def _otel_safe(value: Any) -> Any:
    redacted = redact_obj(value)
    if isinstance(redacted, (dict, list)):
        return json.dumps(redacted, ensure_ascii=False)
    if isinstance(redacted, tuple):
        return tuple(_otel_safe(item) for item in redacted)
    return redacted


def _rewrite_attribute_map(attributes: Any) -> None:
    """Rewrite attributes in place. The SDK freezes them before on_end."""
    if not attributes:
        return
    frozen = bool(getattr(attributes, "_immutable", False))
    if frozen:
        attributes._immutable = False
    try:
        for key in list(attributes.keys()):
            attributes[key] = _otel_safe(attributes[key])
    finally:
        if frozen:
            attributes._immutable = True


class PIIRedactingSpanProcessor(SpanProcessor):
    """在 Span 结束、交给后续导出器之前递归脱敏属性与事件。"""

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span) -> None:
        _rewrite_attribute_map(getattr(span, "_attributes", None))
        for event in getattr(span, "_events", ()) or ():
            _rewrite_attribute_map(getattr(event, "attributes", None))
        status = getattr(span, "_status", None)
        description = getattr(status, "description", None)
        if description:
            status.description = redact_pii(str(description))

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@dataclass
class StoredSpan:
    name: str
    session_id: str
    start_time: int
    end_time: int
    attributes: dict[str, Any]

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end_time - self.start_time) / 1_000_000.0)


class SpanRingBuffer(SpanProcessor):
    """有界内存轨迹，供 /sessions/{id}/trace 查询，避免 Span 无限堆积。"""

    def __init__(self, maxlen: int = _DEFAULT_RING):
        self._spans: deque[StoredSpan] = deque(maxlen=max(1, int(maxlen)))

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span) -> None:
        attributes = {str(key): _otel_safe(value) for key, value in dict(span.attributes or {}).items()}
        session_id = str(attributes.get("session_id", "") or "")
        self._spans.append(
            StoredSpan(
                name=span.name,
                session_id=session_id,
                start_time=int(span.start_time or 0),
                end_time=int(span.end_time or 0),
                attributes=attributes,
            )
        )

    def for_session(self, session_id: str) -> list[StoredSpan]:
        return [span for span in self._spans if span.session_id == session_id]

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _traces_endpoint(endpoint: str) -> str:
    cleaned = endpoint.strip().rstrip("/")
    if cleaned.endswith("/v1/traces"):
        return cleaned
    return cleaned + "/v1/traces"


def configure_tracing(service_name: str = "knowdesk") -> TracerProvider:
    """安装脱敏处理器、有界环形缓冲、测试用内存导出器，以及可选的 OTLP 导出。"""
    global _provider, _ring, _memory_exporter
    if _provider is not None:
        return _provider

    ring_size = int(os.getenv("KNOWDESK_TRACE_RING_SIZE", str(_DEFAULT_RING)))
    provider = TracerProvider()
    ring = SpanRingBuffer(maxlen=ring_size)
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(PIIRedactingSpanProcessor())
    provider.add_span_processor(ring)
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_traces_endpoint(endpoint)))
        )

    trace.set_tracer_provider(provider)
    _provider = provider
    _ring = ring
    _memory_exporter = memory_exporter
    return provider


def get_tracer(name: str = "knowdesk"):
    configure_tracing()
    return trace.get_tracer(name)


def get_ring_buffer() -> SpanRingBuffer:
    configure_tracing()
    assert _ring is not None
    return _ring


def get_memory_exporter() -> InMemorySpanExporter:
    configure_tracing()
    assert _memory_exporter is not None
    return _memory_exporter


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
    """GenAI 语义约定门面。真正的 Span 由 OpenTelemetry SDK 创建并导出。"""

    def __init__(self, record_raw_user_text: bool = False):
        self.record_raw_user_text = record_raw_user_text
        self.spans: list[GenAISpan] = []

    @contextmanager
    def start_span(self, name: str, **attrs: Any):
        safe_attrs = redact_attributes(attrs)
        session_id = current_session_id.get()
        if session_id and "session_id" not in safe_attrs:
            safe_attrs["session_id"] = session_id
        span = GenAISpan(name=name, start_time_ms=time.perf_counter() * 1000.0, attributes=dict(safe_attrs))
        otel_attributes = {key: _otel_safe(value) for key, value in safe_attrs.items()}
        with get_tracer().start_as_current_span(name, attributes=otel_attributes):
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
        ) as recorded:
            return recorded

    def summary_metrics(self) -> dict[str, float]:
        llm_spans = [span for span in self.spans if span.attributes.get("gen_ai.operation.name") == "chat"]
        total_in = sum(span.attributes.get("gen_ai.usage.input_tokens", 0) for span in llm_spans)
        total_hit = sum(span.attributes.get("gen_ai.usage.cached_tokens", 0) for span in llm_spans)
        total_out = sum(span.attributes.get("gen_ai.usage.output_tokens", 0) for span in llm_spans)
        total_cost = sum(span.attributes.get("knowdesk.cost_rmb", 0.0) for span in llm_spans)
        hit_rate = (total_hit / total_in) if total_in > 0 else 0.0
        return {
            "total_spans": len(self.spans),
            "total_input_tokens": total_in,
            "total_cached_tokens": total_hit,
            "cache_hit_rate": hit_rate,
            "total_output_tokens": total_out,
            "total_cost_rmb": total_cost,
        }


# Imported for callers that want the SDK result enum without a second import.
__all__ = [
    "BatchSpanProcessor",
    "GenAISpan",
    "GenAITracer",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "PIIRedactingSpanProcessor",
    "SpanExportResult",
    "SpanRingBuffer",
    "configure_tracing",
    "current_session_id",
    "get_memory_exporter",
    "get_ring_buffer",
    "get_tracer",
    "redact_attributes",
    "redact_obj",
    "redact_pii",
]
