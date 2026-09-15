"""Telemetry recorder: cost, latency, tokens, cache hits, retries and security flags.

Records go to the configured :class:`~callm.storage.base.TelemetryStore` (SQLite by
default), to ``on_call`` hooks, and optionally to OpenTelemetry as spans that follow the
GenAI semantic conventions. Telemetry failures are logged and never break a call.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any

from callm.config import get_settings, get_telemetry_store
from callm.pipeline import CallState, Handler, Step
from callm.types import CallRecord, LLMResponse

logger = logging.getLogger("callm")

_LAST_CALL: ContextVar[CallRecord | None] = ContextVar("callm_last_call", default=None)


def last_call() -> CallRecord | None:
    """The telemetry record of the most recent call made in the current context."""
    return _LAST_CALL.get()


def fill_from_response(record: CallRecord, response: LLMResponse) -> None:
    record.provider = response.provider
    record.model = response.model
    if response.cached:
        # Nothing was consumed; the avoided spend is reported in ``saved_usd``.
        record.input_tokens = record.output_tokens = 0
        record.cache_read_tokens = record.cache_write_tokens = 0
        record.cost_usd = 0.0
        return
    record.input_tokens = response.usage.input_tokens
    record.output_tokens = response.usage.output_tokens
    record.cache_read_tokens = response.usage.cache_read_tokens
    record.cache_write_tokens = response.usage.cache_write_tokens
    record.cost_usd = response.cost


def emit(record: CallRecord, *, persist: bool = True, start_ns: int | None = None) -> None:
    """Publish a finished record to storage, hooks and OpenTelemetry."""
    _LAST_CALL.set(record)
    settings = get_settings()
    if persist and settings.telemetry:
        try:
            get_telemetry_store().record(record)
        except Exception:
            logger.warning("callm: failed to store telemetry record", exc_info=True)
    for hook in list(settings.on_call):
        try:
            hook(record)
        except Exception:
            logger.warning("callm: on_call hook %r raised", hook, exc_info=True)
    if settings.otel:
        _export_span(record, start_ns)


def _export_span(record: CallRecord, start_ns: int | None) -> None:
    try:
        from opentelemetry import trace
    except ImportError:
        logger.debug("callm: otel=True but opentelemetry-api is not installed")
        return
    try:
        tracer = trace.get_tracer("callm")
        end_ns = time.time_ns()
        begin = start_ns if start_ns is not None else end_ns - int(record.latency_ms * 1_000_000)
        span = tracer.start_span(
            f"chat {record.model}", start_time=begin, kind=trace.SpanKind.CLIENT
        )
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": record.provider,
            "gen_ai.request.model": record.model,
            "gen_ai.usage.input_tokens": record.input_tokens
            + record.cache_read_tokens
            + record.cache_write_tokens,
            "gen_ai.usage.output_tokens": record.output_tokens,
            "callm.function": record.function or "",
            "callm.saved_usd": record.saved_usd,
            "callm.cache_hit": record.cache_hit,
            "callm.retries": record.retries,
            "callm.validation_retries": record.validation_retries,
            "callm.status": record.status,
            "callm.injection_flagged": record.injection_flagged,
        }
        if record.cost_usd is not None:
            attributes["callm.cost_usd"] = record.cost_usd
        if record.fallback_from:
            attributes["callm.fallback_from"] = record.fallback_from
        if record.error_type:
            attributes["error.type"] = record.error_type
        if record.injection_score is not None:
            attributes["callm.injection_score"] = record.injection_score
        for entity, count in record.pii_redactions.items():
            attributes[f"callm.pii.{entity}"] = count
        for key, value in record.tags.items():
            attributes[f"callm.tag.{key}"] = str(value)
        span.set_attributes(attributes)
        if record.status != "ok":
            span.set_status(trace.Status(trace.StatusCode.ERROR, record.error_type or "error"))
        span.end(end_time=end_ns)
    except Exception:
        logger.debug("callm: failed to export OpenTelemetry span", exc_info=True)


class TelemetryMiddleware:
    name = "telemetry"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        record = state.record
        record.tags = dict(state.config.tags)
        record.streamed = state.request.stream
        start = time.perf_counter()
        start_ns = time.time_ns()
        try:
            response = yield from call_next(state)
        except BaseException as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.status = "error" if isinstance(exc, Exception) else "cancelled"
            record.error_type = type(exc).__name__
            if state.target is not None:
                record.provider, record.model = state.target.provider, state.target.model
            emit(record, persist=state.config.telemetry, start_ns=start_ns)
            raise
        record.latency_ms = (time.perf_counter() - start) * 1000
        fill_from_response(record, response)
        if not response.cached:
            response.latency_ms = record.latency_ms
        emit(record, persist=state.config.telemetry, start_ns=start_ns)
        return response


__all__ = ["TelemetryMiddleware", "emit", "fill_from_response", "last_call"]
