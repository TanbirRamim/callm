"""In-process storage. Nothing is persisted."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any

from callm.storage.base import CacheEntry, CacheStore, TelemetryStore
from callm.types import CallRecord


class MemoryCacheStore(CacheStore):
    """LRU cache bounded by ``max_entries``."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        with self._cache_lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expired():
                del self._entries[key]
                return None
            entry.hits += 1
            self._entries.move_to_end(key)
            return entry

    def set(self, entry: CacheEntry) -> None:
        with self._cache_lock:
            self._entries[entry.key] = entry
            self._entries.move_to_end(entry.key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def candidates(self, group: str, limit: int) -> list[CacheEntry]:
        now = time.time()
        found: list[CacheEntry] = []
        with self._cache_lock:
            for entry in reversed(self._entries.values()):
                if entry.group == group and entry.embedding is not None and not entry.expired(now):
                    found.append(entry)
                    if len(found) >= limit:
                        break
        return found

    def record_hit(self, key: str) -> None:
        with self._cache_lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.hits += 1
                self._entries.move_to_end(key)

    def delete(self, key: str) -> bool:
        with self._cache_lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        with self._cache_lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def stats(self) -> dict[str, Any]:
        with self._cache_lock:
            return {
                "backend": "memory",
                "entries": len(self._entries),
                "hits": sum(entry.hits for entry in self._entries.values()),
            }


class MemoryTelemetryStore(TelemetryStore):
    """Keeps the most recent ``max_records`` call records."""

    def __init__(self, max_records: int = 100_000) -> None:
        self._records: deque[CallRecord] = deque(maxlen=max_records)
        self._telemetry_lock = threading.Lock()

    def record(self, record: CallRecord) -> None:
        with self._telemetry_lock:
            self._records.append(record)

    def query(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        function: str | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[CallRecord]:
        with self._telemetry_lock:
            records = list(self._records)
        result = [
            r
            for r in reversed(records)
            if (since is None or r.timestamp >= since)
            and (until is None or r.timestamp < until)
            and (function is None or r.function == function)
            and (provider is None or r.provider == provider)
        ]
        result.sort(
            key=lambda r: r.timestamp, reverse=True
        )  # stable: ties keep newest-inserted first
        return result[:limit] if limit is not None else result

    def clear_records(self) -> int:
        with self._telemetry_lock:
            count = len(self._records)
            self._records.clear()
            return count


class MemoryStorage(MemoryCacheStore, MemoryTelemetryStore):
    """Cache and telemetry kept in memory."""

    def __init__(self, max_entries: int = 10_000, max_records: int = 100_000) -> None:
        MemoryCacheStore.__init__(self, max_entries)
        MemoryTelemetryStore.__init__(self, max_records)


__all__ = ["MemoryCacheStore", "MemoryStorage", "MemoryTelemetryStore"]
