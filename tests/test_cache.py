"""Response cache (exact and semantic)."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

import callm
from callm import CacheConfig, HashingEmbedder
from callm.middleware.cache import cache_keys
from callm.storage.memory import MemoryCacheStore
from callm.storage.sqlite import SQLiteStorage
from helpers import anthropic_body, openai_body, user


def make_ask(client, **options):
    @callm.callm(**options)
    def ask(text: str, **params):
        return client.chat.completions.create(model="gpt-4o-mini", messages=user(text), **params)

    return ask


def test_exact_cache_hit_returns_identical_native_response(openai_client, openai_server, records):
    ask = make_ask(openai_client, cache=True)

    first = ask("What is 2+2?")
    second = ask("What is 2+2?")

    assert openai_server.count == 1
    assert type(second).__name__ == "ChatCompletion"
    assert second.model_dump() == first.model_dump()
    newest, oldest = records()
    assert oldest.cache_hit is False and oldest.cost_usd > 0
    assert newest.cache_hit is True
    assert newest.cost_usd == 0
    assert newest.saved_usd == pytest.approx(oldest.cost_usd)
    assert newest.input_tokens == newest.output_tokens == 0


def test_different_prompts_params_and_models_miss(openai_client, openai_server):
    ask = make_ask(openai_client, cache=True)
    ask("What is 2+2?")
    ask("What is 2+3?")
    ask("What is 2+2?", temperature=0.9)
    assert openai_server.count == 3


def test_non_semantic_arguments_do_not_affect_the_key(openai_client, openai_server):
    ask = make_ask(openai_client, cache=True)
    ask("hello", user="u1", metadata={"trace": "1"}, timeout=10)
    ask("hello", user="u2", metadata={"trace": "2"}, timeout=20)
    assert openai_server.count == 1


def test_ttl_expiry(openai_client, openai_server, monkeypatch):
    ask = make_ask(openai_client, cache=CacheConfig(ttl=60))
    ask("hi")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 61)
    ask("hi")
    assert openai_server.count == 2


def test_refusals_are_not_cached(anthropic_client, anthropic_server):
    anthropic_server.queue(anthropic_body(text="", stop_reason="refusal"))

    @callm.callm(cache=True)
    def ask():
        return anthropic_client.messages.create(
            model="claude-opus-5", max_tokens=10, messages=user("x")
        )

    ask()
    ask()
    assert anthropic_server.count == 2


def test_cache_keys_depend_on_the_output_schema(openai_client, openai_server):
    class A(BaseModel):
        a: int

    class B(BaseModel):
        b: int

    openai_server.queue(openai_body(text='{"a": 1, "b": 2}'), openai_body(text='{"a": 1, "b": 2}'))
    make_ask(openai_client, cache=True, output_schema=A)("same prompt")
    make_ask(openai_client, cache=True, output_schema=B)("same prompt")
    assert openai_server.count == 2


def test_pii_is_redacted_before_it_reaches_the_cache_key(openai_client, openai_server):
    store = MemoryCacheStore()
    ask = make_ask(openai_client, cache=CacheConfig(store=store), block_pii=True)
    ask("email alice@example.com")
    ask("email bob@example.com")  # both become "email [EMAIL_1]"
    assert openai_server.count == 1
    (entry,) = store._entries.values()
    assert "alice" not in str(entry.response)


def test_semantic_cache_matches_near_duplicates_only(openai_client, openai_server, records):
    config = CacheConfig(semantic=True, threshold=0.9, embedder=HashingEmbedder())
    ask = make_ask(openai_client, cache=config)

    ask("How do I reset my password?")
    ask("how do i reset my password")  # near duplicate -> hit
    ask("What are your opening hours on Sunday?")  # unrelated -> miss

    assert openai_server.count == 2
    assert [r.cache_hit for r in reversed(records())] == [False, True, False]


def test_semantic_matches_never_cross_system_prompts(openai_client, openai_server):
    config = CacheConfig(semantic=True, threshold=0.5, embedder=HashingEmbedder())

    @callm.callm(cache=config)
    def ask(system: str, text: str):
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, *user(text)],
        )

    ask("You translate to French.", "good morning")
    ask("You translate to German.", "good morning")
    assert openai_server.count == 2


def test_semantic_cache_requires_an_embedder(openai_client, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sentence_transformers(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sentence_transformers)
    monkeypatch.setattr("callm.middleware.cache._default_embedder", [])
    ask = make_ask(openai_client, cache="semantic")
    with pytest.raises(callm.MissingDependencyError, match="callm-toolkit\\[cache\\]"):
        ask("hello")


def test_broken_cache_store_never_breaks_calls(openai_client, openai_server, caplog):
    class Broken(MemoryCacheStore):
        def get(self, key):
            raise RuntimeError("disk on fire")

        def set(self, entry):
            raise RuntimeError("disk on fire")

    ask = make_ask(openai_client, cache=CacheConfig(store=Broken()))
    assert ask("x").choices[0].message.content == "Hello from OpenAI"
    assert "cache read failed" in caplog.text


def test_sqlite_cache_survives_restarts(openai_client, openai_server, tmp_path):
    path = tmp_path / "cache.db"
    make_ask(openai_client, cache=CacheConfig(store=SQLiteStorage(path)))("persist me")
    make_ask(openai_client, cache=CacheConfig(store=SQLiteStorage(path)))("persist me")
    assert openai_server.count == 1


async def test_async_cache(async_openai_client, openai_server):
    @callm.callm(cache=True)
    async def ask():
        return await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("a")
        )

    await ask()
    await ask()
    assert openai_server.count == 1


def test_complete_cache_hit_rebuilds_raw(providers, openai_server):
    first = callm.complete("gpt-4o-mini", "cached?", cache=True)
    second = callm.complete("gpt-4o-mini", "cached?", cache=True)
    assert openai_server.count == 1
    assert second.cached is True and first.cached is False
    assert second.text == first.text
    assert second.raw.choices[0].message.content == "Hello from OpenAI"
    assert second.cost == 0


def test_cache_key_is_stable_and_order_independent():
    from callm.api import build_request

    a = build_request("gpt-4o-mini", "hi", temperature=0.1, max_tokens=5)
    b = build_request("gpt-4o-mini", "hi", max_tokens=5, temperature=0.1)
    config = CacheConfig()
    assert cache_keys(a, config) == cache_keys(b, config)
    c = build_request("gpt-4o-mini", "hi", max_tokens=6, temperature=0.1)
    assert cache_keys(a, config)[0] != cache_keys(c, config)[0]


def test_clear_cache(openai_client, openai_server):
    ask = make_ask(openai_client, cache=True)
    ask("x")
    assert callm.clear_cache() == 1
    ask("x")
    assert openai_server.count == 2
