"""Telemetry records, hooks, OpenTelemetry and storage backends."""

from __future__ import annotations

import os
import time

import pytest

import callm
from callm.config import get_telemetry_store
from callm.storage.base import CacheEntry
from callm.storage.memory import MemoryStorage
from callm.storage.sqlite import SQLiteStorage
from callm.types import CallRecord
from helpers import user


def test_records_never_contain_prompt_text(openai_client, records):
    @callm.callm(tags={"team": "search"})
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("secret prompt")
        )

    ask()
    record = records()[0]
    assert "secret" not in str(record.to_dict())
    assert record.tags == {"team": "search"}
    assert record.latency_ms > 0
    assert callm.last_call() == record


def test_on_call_hooks_and_failing_hooks(openai_client, caplog):
    seen: list[CallRecord] = []

    def broken(record):
        raise RuntimeError("hook bug")

    callm.configure(on_call=[seen.append, broken])

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    assert ask().choices[0].message.content == "Hello from OpenAI"
    assert len(seen) == 1
    assert "hook bug" in caplog.text


def test_telemetry_can_be_disabled(openai_client, records):
    @callm.callm(telemetry=False)
    def quiet():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    quiet()
    assert records() == []
    assert callm.last_call() is not None  # still observable in-process

    callm.configure(telemetry=False)

    @callm.callm
    def also_quiet():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    also_quiet()
    assert records() == []


def test_storage_failures_do_not_break_calls(openai_client, caplog):
    class Broken(MemoryStorage):
        def record(self, record):
            raise OSError("read-only file system")

    callm.configure(storage=Broken())

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    ask()
    assert "failed to store telemetry" in caplog.text


def test_opentelemetry_spans(openai_client, openai_server):
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("callm")
    original = trace.get_tracer
    trace.get_tracer = lambda *args, **kwargs: tracer  # type: ignore[assignment]
    try:
        callm.configure(otel=True)
        openai_server.fail(429)

        @callm.callm(retry=1, block_pii=True)
        def ask():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=user("a@b.io")
            )

        ask()
    finally:
        trace.get_tracer = original  # type: ignore[assignment]
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o-mini-2024-07-18"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["callm.retries"] == 1
    assert span.attributes["callm.pii.email"] == 1
    assert span.attributes["callm.cost_usd"] > 0


def test_sqlite_is_the_default_storage(tmp_path, openai_client):
    callm.reset_settings()
    callm.configure(home=tmp_path / "home")

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    ask()
    assert (tmp_path / "home" / "callm.db").exists()
    assert isinstance(get_telemetry_store(), SQLiteStorage)
    rows = callm.stats()
    assert rows[0].key == "openai" and rows[0].calls == 1


def test_unwritable_home_falls_back_to_memory(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    callm.reset_settings()
    callm.configure(home=blocker / "sub")
    assert isinstance(get_telemetry_store(), MemoryStorage)
    assert "using in-memory storage" in caplog.text


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryStorage()
    else:
        storage = SQLiteStorage(tmp_path / "t.db")
        yield storage
        storage.close()


def _record(**overrides) -> CallRecord:
    base = {
        "function": "app.ask",
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.01,
        "latency_ms": 200.0,
    }
    base.update(overrides)
    return CallRecord(**base)


def test_telemetry_store_contract(store):
    now = time.time()
    store.record(_record(timestamp=now - 3600 * 48))
    store.record(
        _record(
            timestamp=now - 10,
            cache_hit=True,
            cost_usd=0.0,
            saved_usd=0.01,
            input_tokens=0,
            output_tokens=0,
        )
    )
    store.record(
        _record(
            timestamp=now - 5,
            provider="anthropic",
            model="claude-sonnet-5",
            cost_usd=0.02,
            retries=2,
            fallback_from="openai/gpt-4o",
        )
    )
    store.record(
        _record(
            timestamp=now - 1,
            status="error",
            error_type="RateLimitError",
            cost_usd=None,
            pii_redactions={"email": 2},
            tags={"k": "v"},
        )
    )

    everything = store.query()
    assert len(everything) == 4
    assert everything[0].status == "error"
    assert everything[0].pii_redactions == {"email": 2}
    assert everything[0].tags == {"k": "v"}
    assert len(store.query(since=now - 3600)) == 3
    assert len(store.query(provider="anthropic")) == 1
    assert len(store.query(limit=2)) == 2

    rows = {row.key: row for row in store.aggregate("provider", since=now - 3600)}
    assert rows["openai"].calls == 2
    assert rows["openai"].errors == 1
    assert rows["openai"].cache_hits == 1
    assert rows["openai"].saved_usd == pytest.approx(0.01)
    assert rows["anthropic"].cost_usd == pytest.approx(0.02)
    assert rows["anthropic"].fallbacks == 1
    assert rows["anthropic"].retries == 2
    by_model = store.aggregate("model")
    assert {row.key for row in by_model} == {"gpt-4o", "claude-sonnet-5"}
    with pytest.raises(ValueError):
        store.aggregate("bogus")
    assert store.clear_records() == 4
    assert store.query() == []


def test_cache_store_contract(store):
    now = time.time()
    store.set(CacheEntry(key="a", response={"text": "A", "cost": 0.5}, created_at=now))
    store.set(CacheEntry(key="old", response={"text": "O"}, created_at=now, expires_at=now - 1))
    store.set(
        CacheEntry(
            key="g1", response={"text": "1"}, group="g", embedding=[1.0, 0.0], created_at=now - 2
        )
    )
    store.set(
        CacheEntry(
            key="g2", response={"text": "2"}, group="g", embedding=[0.0, 1.0], created_at=now - 1
        )
    )

    hit = store.get("a")
    assert hit.response["text"] == "A" and hit.cost == 0.5 and hit.hits == 1
    assert store.get("old") is None
    assert store.get("missing") is None
    candidates = store.candidates("g", limit=10)
    assert [c.key for c in candidates] == ["g2", "g1"]
    assert candidates[0].embedding == pytest.approx([0.0, 1.0])
    assert [c.key for c in store.candidates("g", limit=1)] == ["g2"]
    store.record_hit("g1")
    assert store.stats()["entries"] >= 3
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert store.clear() >= 2
    assert store.get("g1") is None


def test_redis_cache_store_contract():
    fakeredis = pytest.importorskip("fakeredis")
    from callm.storage.redis import RedisCacheStore

    store = RedisCacheStore(client=fakeredis.FakeRedis(), prefix="test")
    test_cache_store_contract(store)
    now = time.time()
    store.set(CacheEntry(key="ttl", response={"text": "x"}, created_at=now, expires_at=now + 100))
    assert store.get("ttl") is not None
    assert store.client.pttl("test:entry:ttl") > 0


def test_redis_backend_with_decorator(openai_client, openai_server):
    fakeredis = pytest.importorskip("fakeredis")
    from callm.storage.redis import RedisCacheStore

    callm.configure(cache_store=RedisCacheStore(client=fakeredis.FakeRedis()))

    @callm.callm(cache=True)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("shared"))

    ask()
    ask()
    assert openai_server.count == 1


def test_sqlite_cache_eviction(tmp_path):
    storage = SQLiteStorage(tmp_path / "e.db", max_cache_entries=3)
    now = time.time()
    for index in range(5):
        storage.set(CacheEntry(key=str(index), response={}, created_at=now + index))
    storage.set(CacheEntry(key="expired", response={}, created_at=now, expires_at=now - 5))
    assert storage.evict() == 3
    assert storage.get("0") is None
    assert storage.get("4") is not None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_sqlite_reconnects_after_fork(tmp_path):
    storage = SQLiteStorage(tmp_path / "f.db")
    storage.record(_record())
    pid = os.fork()
    if pid == 0:  # child
        try:
            storage.record(_record())
            os._exit(0)
        except Exception:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    assert len(storage.query()) == 2
