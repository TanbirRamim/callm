"""Storage interfaces for the response cache and telemetry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from callm.types import CallRecord

AGGREGATE_KEYS = ("provider", "model", "function")


@dataclass
class CacheEntry:
    key: str
    response: dict[str, Any]
    group: str | None = None
    embedding: list[float] | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    hits: int = 0

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (time.time() if now is None else now) >= self.expires_at

    @property
    def cost(self) -> float:
        value = self.response.get("cost")
        return float(value) if isinstance(value, (int, float)) else 0.0


class CacheStore(ABC):
    """Key-value store for cached responses, with optional embedding candidates."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Return a live entry and count a hit, or ``None``."""

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Insert or replace an entry."""

    @abstractmethod
    def candidates(self, group: str, limit: int) -> list[CacheEntry]:
        """Most recent live entries of ``group`` that have an embedding."""

    @abstractmethod
    def record_hit(self, key: str) -> None:
        """Count a (semantic) hit for ``key``."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove one entry; return whether it existed."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all entries; return how many were removed."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Backend name, entry count, hit count..."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


@dataclass
class AggregateRow:
    key: str
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    saved_usd: float = 0.0
    cache_hits: int = 0
    retries: int = 0
    fallbacks: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def add(self, record: CallRecord) -> None:
        self.calls += 1
        self.errors += record.status != "ok"
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cache_read_tokens += record.cache_read_tokens
        self.cache_write_tokens += record.cache_write_tokens
        self.cost_usd += record.cost_usd or 0.0
        self.saved_usd += record.saved_usd or 0.0
        self.cache_hits += bool(record.cache_hit)
        self.retries += record.retries
        self.fallbacks += record.fallback_from is not None
        self.total_latency_ms += record.latency_ms

    def merge(self, other: AggregateRow) -> None:
        self.calls += other.calls
        self.errors += other.errors
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_usd += other.cost_usd
        self.saved_usd += other.saved_usd
        self.cache_hits += other.cache_hits
        self.retries += other.retries
        self.fallbacks += other.fallbacks
        self.total_latency_ms += other.total_latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "saved_usd": round(self.saved_usd, 6),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class TelemetryStore(ABC):
    @abstractmethod
    def record(self, record: CallRecord) -> None:
        """Persist one call record."""

    @abstractmethod
    def query(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        function: str | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[CallRecord]:
        """Records matching the filters, newest first."""

    @abstractmethod
    def clear_records(self) -> int:
        """Delete all call records; return how many were removed."""

    def aggregate(
        self,
        by: str = "provider",
        *,
        since: float | None = None,
        until: float | None = None,
        function: str | None = None,
    ) -> list[AggregateRow]:
        if by not in AGGREGATE_KEYS:
            raise ValueError(f"by must be one of {AGGREGATE_KEYS}")
        rows: dict[str, AggregateRow] = {}
        for record in self.query(since=since, until=until, function=function):
            key = str(getattr(record, by) or "unknown")
            rows.setdefault(key, AggregateRow(key)).add(record)
        return sorted(rows.values(), key=lambda row: (-row.cost_usd, -row.calls, row.key))

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


__all__ = ["AGGREGATE_KEYS", "AggregateRow", "CacheEntry", "CacheStore", "TelemetryStore"]
