"""SQLite storage: the zero-infrastructure default (``~/.callm/callm.db``)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from callm.storage.base import AGGREGATE_KEYS, AggregateRow, CacheEntry, CacheStore, TelemetryStore
from callm.types import CallRecord

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    grp TEXT,
    response TEXT NOT NULL,
    embedding BLOB,
    created_at REAL NOT NULL,
    expires_at REAL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS cache_group_idx ON cache (grp, created_at);
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    function TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    estimated_cost_usd REAL,
    saved_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    validation_retries INTEGER NOT NULL DEFAULT 0,
    fallback_from TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    error_type TEXT,
    streamed INTEGER NOT NULL DEFAULT 0,
    pii_redactions TEXT,
    injection_score REAL,
    injection_flagged INTEGER NOT NULL DEFAULT 0,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS calls_timestamp_idx ON calls (timestamp);
CREATE INDEX IF NOT EXISTS calls_function_idx ON calls (function, timestamp);
"""

_CALL_COLUMNS = (
    "id",
    "timestamp",
    "function",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "estimated_cost_usd",
    "saved_usd",
    "latency_ms",
    "cache_hit",
    "retries",
    "validation_retries",
    "fallback_from",
    "status",
    "error_type",
    "streamed",
    "pii_redactions",
    "injection_score",
    "injection_flagged",
    "tags",
)
_INSERT_CALL = (
    f"INSERT OR REPLACE INTO calls ({', '.join(_CALL_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _CALL_COLUMNS)})"
)


def _pack(vector: list[float] | None) -> bytes | None:
    return None if vector is None else array("f", vector).tobytes()


def _unpack(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class SQLiteStorage(CacheStore, TelemetryStore):
    """Thread-safe, fork-safe SQLite storage for cache entries and call records.

    Uses WAL journaling so several processes can share one database file.
    """

    def __init__(
        self, path: str | os.PathLike[str] = ":memory:", *, max_cache_entries: int = 100_000
    ) -> None:
        self.path = str(path)
        self.max_cache_entries = max_cache_entries
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._inherited: list[sqlite3.Connection] = []
        self._pid = os.getpid()
        self._writes = 0
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    # ----------------------------------------------------------------- connection

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn = conn
        self._pid = os.getpid()
        return conn

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn is not None and self._pid != os.getpid():
                # Never use a connection inherited across fork(); keep it referenced so
                # garbage collection does not close it underneath the parent process.
                self._inherited.append(self._conn)
                self._conn = None
            conn = self._conn if self._conn is not None else self._connect()
            yield conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ----------------------------------------------------------------- cache

    def get(self, key: str) -> CacheEntry | None:
        now = time.time()
        with self._db() as db:
            row = db.execute("SELECT * FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            if row["expires_at"] is not None and row["expires_at"] <= now:
                db.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            db.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            entry = self._entry(row)
            entry.hits += 1
            return entry

    def set(self, entry: CacheEntry) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO cache"
                " (key, grp, response, embedding, created_at, expires_at, hits)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    entry.group,
                    json.dumps(entry.response, separators=(",", ":"), default=str),
                    _pack(entry.embedding),
                    entry.created_at,
                    entry.expires_at,
                    entry.hits,
                ),
            )
            self._writes += 1
            if self._writes % 500 == 0:
                self.evict(db)

    def evict(self, db: sqlite3.Connection | None = None) -> int:
        """Drop expired entries and the oldest ones beyond ``max_cache_entries``."""
        if db is None:
            with self._db() as conn:
                return self.evict(conn)
        removed: int = db.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?", (time.time(),)
        ).rowcount
        count = db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        excess = count - self.max_cache_entries
        if excess > 0:
            removed += db.execute(
                "DELETE FROM cache WHERE key IN"
                " (SELECT key FROM cache ORDER BY created_at ASC LIMIT ?)",
                (excess,),
            ).rowcount
        return removed

    def candidates(self, group: str, limit: int) -> list[CacheEntry]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM cache WHERE grp = ? AND embedding IS NOT NULL"
                " AND (expires_at IS NULL OR expires_at > ?) ORDER BY created_at DESC LIMIT ?",
                (group, time.time(), limit),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def record_hit(self, key: str) -> None:
        with self._db() as db:
            db.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))

    def delete(self, key: str) -> bool:
        with self._db() as db:
            return bool(db.execute("DELETE FROM cache WHERE key = ?", (key,)).rowcount > 0)

    def clear(self) -> int:
        with self._db() as db:
            removed: int = db.execute("DELETE FROM cache").rowcount
            return removed

    def stats(self) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits,"
                " COALESCE(SUM(LENGTH(response)), 0) AS bytes FROM cache"
            ).fetchone()
        return {
            "backend": "sqlite",
            "path": self.path,
            "entries": row["entries"],
            "hits": row["hits"],
            "bytes": row["bytes"],
        }

    @staticmethod
    def _entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            key=row["key"],
            group=row["grp"],
            response=json.loads(row["response"]),
            embedding=_unpack(row["embedding"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            hits=row["hits"],
        )

    # ----------------------------------------------------------------- telemetry

    def record(self, record: CallRecord) -> None:
        values = (
            record.id,
            record.timestamp,
            record.function,
            record.provider,
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_tokens,
            record.cache_write_tokens,
            record.cost_usd,
            record.estimated_cost_usd,
            record.saved_usd,
            record.latency_ms,
            int(record.cache_hit),
            record.retries,
            record.validation_retries,
            record.fallback_from,
            record.status,
            record.error_type,
            int(record.streamed),
            json.dumps(record.pii_redactions) if record.pii_redactions else None,
            record.injection_score,
            int(record.injection_flagged),
            json.dumps(record.tags) if record.tags else None,
        )
        with self._db() as db:
            db.execute(_INSERT_CALL, values)

    @staticmethod
    def _where(
        since: float | None,
        until: float | None,
        function: str | None,
        provider: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp < ?")
            params.append(until)
        if function is not None:
            clauses.append("function = ?")
            params.append(function)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        function: str | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[CallRecord]:
        where, params = self._where(since, until, function, provider)
        sql = f"SELECT * FROM calls{where} ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._db() as db:
            rows = db.execute(sql, params).fetchall()
        records = []
        for row in rows:
            data = dict(row)
            data["cache_hit"] = bool(data["cache_hit"])
            data["streamed"] = bool(data["streamed"])
            data["injection_flagged"] = bool(data["injection_flagged"])
            data["pii_redactions"] = (
                json.loads(data["pii_redactions"]) if data["pii_redactions"] else {}
            )
            data["tags"] = json.loads(data["tags"]) if data["tags"] else {}
            records.append(CallRecord.from_dict(data))
        return records

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
        where, params = self._where(since, until, function)
        sql = (
            f"SELECT COALESCE({by}, 'unknown') AS key, COUNT(*) AS calls,"
            " SUM(status != 'ok') AS errors,"
            " SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
            " SUM(cache_read_tokens) AS cache_read_tokens,"
            " SUM(cache_write_tokens) AS cache_write_tokens,"
            " COALESCE(SUM(cost_usd), 0) AS cost_usd, COALESCE(SUM(saved_usd), 0) AS saved_usd,"
            " SUM(cache_hit) AS cache_hits, SUM(retries) AS retries,"
            " SUM(fallback_from IS NOT NULL) AS fallbacks, SUM(latency_ms) AS total_latency_ms"
            f" FROM calls{where} GROUP BY COALESCE({by}, 'unknown')"
            " ORDER BY cost_usd DESC, calls DESC, key ASC"
        )
        with self._db() as db:
            rows = db.execute(sql, params).fetchall()
        return [
            AggregateRow(
                key=row["key"],
                calls=row["calls"] or 0,
                errors=row["errors"] or 0,
                input_tokens=row["input_tokens"] or 0,
                output_tokens=row["output_tokens"] or 0,
                cache_read_tokens=row["cache_read_tokens"] or 0,
                cache_write_tokens=row["cache_write_tokens"] or 0,
                cost_usd=row["cost_usd"] or 0.0,
                saved_usd=row["saved_usd"] or 0.0,
                cache_hits=row["cache_hits"] or 0,
                retries=row["retries"] or 0,
                fallbacks=row["fallbacks"] or 0,
                total_latency_ms=row["total_latency_ms"] or 0.0,
            )
            for row in rows
        ]

    def clear_records(self) -> int:
        with self._db() as db:
            removed: int = db.execute("DELETE FROM calls").rowcount
            return removed

    def __repr__(self) -> str:
        return f"SQLiteStorage({self.path!r})"


__all__ = ["SQLiteStorage"]
