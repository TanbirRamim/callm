"""Redis cache backend for sharing cached responses across processes and hosts.

``pip install 'callm[redis]'``::

    from callm.storage.redis import RedisCacheStore
    callm.configure(cache_store=RedisCacheStore("redis://localhost:6379/0"))
"""

from __future__ import annotations

import base64
import json
import time
from array import array
from typing import Any

from callm.errors import MissingDependencyError
from callm.storage.base import CacheEntry, CacheStore


class RedisCacheStore(CacheStore):
    def __init__(
        self,
        url: str | None = "redis://localhost:6379/0",
        *,
        client: Any = None,
        prefix: str = "callm",
        max_group_size: int = 5000,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise MissingDependencyError("RedisCacheStore", "redis", "redis") from exc
            client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        self.client = client
        self.prefix = prefix
        self.max_group_size = max_group_size

    def _entry_key(self, key: str) -> str:
        return f"{self.prefix}:entry:{key}"

    def _group_key(self, group: str) -> str:
        return f"{self.prefix}:group:{group}"

    @property
    def _hits_key(self) -> str:
        return f"{self.prefix}:hits"

    @staticmethod
    def _encode(entry: CacheEntry) -> str:
        embedding = None
        if entry.embedding is not None:
            embedding = base64.b64encode(array("f", entry.embedding).tobytes()).decode("ascii")
        return json.dumps(
            {
                "key": entry.key,
                "group": entry.group,
                "response": entry.response,
                "embedding": embedding,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
            },
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _decode(raw: bytes | str) -> CacheEntry:
        data = json.loads(raw)
        embedding = None
        if data.get("embedding"):
            values = array("f")
            values.frombytes(base64.b64decode(data["embedding"]))
            embedding = values.tolist()
        return CacheEntry(
            key=data["key"],
            group=data.get("group"),
            response=data["response"],
            embedding=embedding,
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
        )

    def get(self, key: str) -> CacheEntry | None:
        raw = self.client.get(self._entry_key(key))
        if raw is None:
            return None
        entry = self._decode(raw)
        if entry.expired():
            self.delete(key)
            return None
        entry.hits = int(self.client.hincrby(self._hits_key, key, 1))
        return entry

    def set(self, entry: CacheEntry) -> None:
        pipe = self.client.pipeline()
        name = self._entry_key(entry.key)
        if entry.expires_at is not None:
            ttl_ms = max(int((entry.expires_at - time.time()) * 1000), 1)
            pipe.set(name, self._encode(entry), px=ttl_ms)
        else:
            pipe.set(name, self._encode(entry))
        if entry.group is not None and entry.embedding is not None:
            group_key = self._group_key(entry.group)
            pipe.zadd(group_key, {entry.key: entry.created_at})
            pipe.zremrangebyrank(group_key, 0, -self.max_group_size - 1)
        pipe.execute()

    def candidates(self, group: str, limit: int) -> list[CacheEntry]:
        group_key = self._group_key(group)
        keys = self.client.zrevrange(group_key, 0, max(limit - 1, 0))
        if not keys:
            return []
        names = [k.decode() if isinstance(k, bytes) else str(k) for k in keys]
        raws = self.client.mget([self._entry_key(k) for k in names])
        entries: list[CacheEntry] = []
        stale: list[str] = []
        now = time.time()
        for key, raw in zip(names, raws, strict=False):
            if raw is None:
                stale.append(key)
                continue
            entry = self._decode(raw)
            if entry.expired(now):
                stale.append(key)
                continue
            if entry.embedding is not None:
                entries.append(entry)
        if stale:
            self.client.zrem(group_key, *stale)
            self.client.hdel(self._hits_key, *stale)
        return entries

    def record_hit(self, key: str) -> None:
        self.client.hincrby(self._hits_key, key, 1)

    def delete(self, key: str) -> bool:
        raw = self.client.get(self._entry_key(key))
        pipe = self.client.pipeline()
        pipe.delete(self._entry_key(key))
        pipe.hdel(self._hits_key, key)
        if raw is not None:
            group = json.loads(raw).get("group")
            if group:
                pipe.zrem(self._group_key(group), key)
        removed = pipe.execute()[0]
        return bool(removed)

    def _scan(self, pattern: str) -> list[str]:
        return [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in self.client.scan_iter(match=pattern, count=500)
        ]

    def clear(self) -> int:
        entries = self._scan(f"{self.prefix}:entry:*")
        names = [*entries, *self._scan(f"{self.prefix}:group:*"), self._hits_key]
        for i in range(0, len(names), 500):
            self.client.delete(*names[i : i + 500])
        return len(entries)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "redis",
            "entries": len(self._scan(f"{self.prefix}:entry:*")),
            "hits": sum(int(h) for h in self.client.hvals(self._hits_key)),
        }


__all__ = ["RedisCacheStore"]
