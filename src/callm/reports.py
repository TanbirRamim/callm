"""Programmatic access to telemetry and cache statistics."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from callm.config import get_cache_store, get_telemetry_store
from callm.storage.base import AggregateRow

_RELATIVE = re.compile(r"^(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>s|m|h|d|w)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str | float | datetime | None) -> float | None:
    """Accept ``"24h"``, ``"7d"``, an ISO date/datetime, a ``datetime`` or a UNIX timestamp."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().lower()
    match = _RELATIVE.match(text)
    if match:
        return time.time() - float(match.group("n")) * _UNIT_SECONDS[match.group("unit")]
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(
            f"cannot parse time {value!r}; use e.g. '24h', '7d' or '2026-01-31'"
        ) from None
    if dt.tzinfo is None:
        dt = dt.astimezone()  # interpret naive values as local time
    return dt.timestamp()


def stats(
    by: str = "provider",
    *,
    since: str | float | datetime | None = None,
    function: str | None = None,
) -> list[AggregateRow]:
    """Aggregate recorded calls, e.g. ``callm.stats(by="model", since="7d")``."""
    return get_telemetry_store().aggregate(by, since=parse_since(since), function=function)


def clear_cache() -> int:
    """Delete every cached response from the configured cache store."""
    return get_cache_store().clear()


__all__ = ["clear_cache", "parse_since", "stats"]
