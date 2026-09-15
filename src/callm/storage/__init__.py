"""Storage backends: SQLite (default), in-memory and Redis (cache only)."""

from typing import Any

from callm.storage.base import AggregateRow, CacheEntry, CacheStore, TelemetryStore
from callm.storage.memory import MemoryCacheStore, MemoryStorage, MemoryTelemetryStore
from callm.storage.sqlite import SQLiteStorage

__all__ = [
    "AggregateRow",
    "CacheEntry",
    "CacheStore",
    "MemoryCacheStore",
    "MemoryStorage",
    "MemoryTelemetryStore",
    "RedisCacheStore",
    "SQLiteStorage",
    "TelemetryStore",
]


def __getattr__(name: str) -> Any:
    if name == "RedisCacheStore":  # imported lazily: needs the optional ``redis`` package
        from callm.storage.redis import RedisCacheStore

        return RedisCacheStore
    raise AttributeError(f"module 'callm.storage' has no attribute {name!r}")
