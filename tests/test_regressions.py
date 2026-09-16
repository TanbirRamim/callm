"""Regression tests for defects found in pre-release review."""

from __future__ import annotations

import functools
import gc

import pytest

import callm
from callm import Budget, BudgetExceeded, shield
from callm.errors import AllProvidersFailedError
from callm.middleware.cache import cache_keys
from callm.security import detect_injection, redact_pii
from helpers import anthropic_body, user

# --------------------------------------------------------------------------- nested scopes


def test_outer_decorator_protections_apply_to_inner_calls(openai_client, openai_server):
    budget = Budget(1e-7)

    @callm.callm(cache=True)
    def inner(text):
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user(text))

    @callm.callm(block_pii=True, budget=budget)
    def outer():
        return inner("my email is alice@example.com")

    with pytest.raises(BudgetExceeded):
        outer()  # the outer budget is charged by the inner call...
    budget.limit = 10
    outer()
    assert openai_server.last.body["messages"][0]["content"] == "my email is [EMAIL_1]"
    assert budget.spent > 0  # ...and actually spent


def test_shield_protections_apply_to_decorated_functions(openai_client, openai_server):
    @callm.callm(cache=True)
    def inner(text):
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user(text))

    with shield(block_pii=True, max_cost=0.5):
        inner("ssn 123-45-6789, mail bob@corp.io")
    assert openai_server.last.body["messages"][0]["content"] == "ssn [SSN_1], mail [EMAIL_1]"


def test_strictest_max_cost_wins_and_complete_inherits_scope(providers, openai_server):
    with shield(max_cost=0.0000001), pytest.raises(BudgetExceeded):
        callm.complete("gpt-4o", "hello", max_cost=10)
    assert openai_server.count == 0
    with shield(block_pii=True):
        callm.complete("gpt-4o-mini", "mail me@x.io")
    assert openai_server.last.body["messages"][0]["content"] == "mail [EMAIL_1]"


def test_merge_scopes_keeps_inner_behaviour():
    from callm.config import build_call_config, merge_scopes

    outer = build_call_config(block_pii=True, max_cost=1.0, retry=5)
    inner = build_call_config(cache=True, max_cost=2.0, retry=1)
    merged = merge_scopes(outer, inner)
    assert merged.pii is outer.pii
    assert merged.max_cost == 1.0
    assert merged.retry.max_retries == 1
    assert merged.cache is inner.cache
    assert merge_scopes(build_call_config(), inner) is inner


# --------------------------------------------------------------------------- SDK sentinels


def test_not_given_sentinels_are_ignored(
    anthropic_client, anthropic_server, openai_client, openai_server, records
):
    import anthropic
    import openai

    @callm.callm(block_pii=True, cache=True, fallback=["anthropic/claude-haiku-4-5"])
    def ask_openai():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user("a@b.io"),
            stop=openai.NOT_GIVEN,
            tools=openai.NOT_GIVEN,
        )

    @callm.callm(block_pii=True)
    def ask_anthropic():
        return anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            system=anthropic.NOT_GIVEN,
            messages=user("a@b.io"),
        )

    ask_openai()
    ask_openai()  # cached
    ask_anthropic()
    assert openai_server.count == 1
    assert anthropic_server.last.body["messages"][0]["content"] == "[EMAIL_1]"
    assert "system" not in anthropic_server.last.body
    assert len(records()) == 3


def test_not_given_tools_do_not_block_fallback(
    providers, openai_client, openai_server, anthropic_server
):
    import openai

    openai_server.fail(503)

    @callm.callm(retry=False, fallback=["anthropic/claude-haiku-4-5"])
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o", messages=user("x"), tools=openai.NOT_GIVEN
        )

    assert ask().choices[0].message.content == "Hello from Claude"


# --------------------------------------------------------------------------- Gemini schemas


def test_gemini_response_schema_class_with_cache_and_fallback(
    providers, gemini_client, gemini_server, openai_server
):
    from google.genai import types
    from pydantic import BaseModel

    class City(BaseModel):
        name: str

    gemini_server.queue(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": '{"name": "Paris"}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
    )

    @callm.callm(cache=True, fallback=["gpt-4o-mini"])
    def ask(config, question="city?"):
        return gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=question, config=config
        )

    typed = types.GenerateContentConfig(response_mime_type="application/json", response_schema=City)
    assert ask(typed).text == '{"name": "Paris"}'
    assert ask(typed).text == '{"name": "Paris"}'  # cache hit, no crash
    gemini_server.fail(503, times=3)
    with pytest.raises(AllProvidersFailedError) as info:
        ask({"response_mime_type": "application/json", "response_schema": City}, "capital?")
    assert "is provider-specific" in str(info.value)  # skipped cleanly, no TypeError


def test_cache_keys_are_stable_for_callable_tools():
    from callm.config import CacheConfig
    from callm.providers import GoogleProvider

    def get_weather(city: str) -> str:
        return "sunny"

    request = GoogleProvider().parse_native_request(
        {"model": "gemini-2.5-flash", "contents": "x", "config": {"tools": [get_weather]}}
    )
    key = cache_keys(request, CacheConfig())[0]

    def get_weather(city: str) -> str:
        return "sunny"

    again = GoogleProvider().parse_native_request(
        {"model": "gemini-2.5-flash", "contents": "x", "config": {"tools": [get_weather]}}
    )
    assert cache_keys(again, CacheConfig())[0] == key


# --------------------------------------------------------------------------- detectors


@pytest.mark.parametrize(
    "text",
    [
        "Skip the initial setup instructions if Python is already installed.",
        "Forget the earlier directions, I found the restaurant.",
        "Please don't ignore the previous messages in this thread, they have the order number.",
        "How do I turn off the spam filters in Gmail?",
        "What is the system prompt field used for in the OpenAI API?",
        "Can you override the default content policies in our CMS config?",
        "Our firewall: disable the content filters for the staging subnet",
    ],
)
def test_injection_detector_ignores_ordinary_requests(text):
    assert detect_injection(text).score < 0.5


@pytest.mark.parametrize(
    "text",
    [
        "Logged at 2024-06-15 10:30:45 UTC",
        "invoice 2024-0615-3301",
        "version 2.10.3.4",
        "dimensions 1920 1080 2560 1440",
        "value 100 200 300 400 500",
        "ISBN 978-3-16-148410-0",
        "Price range 1000-2000-3000-4000",
    ],
)
def test_pii_redaction_leaves_ordinary_numbers_alone(text):
    assert redact_pii(text) == text


# --------------------------------------------------------------------------- structured PII


def test_pii_in_structured_fields_is_masked(
    openai_client, openai_server, anthropic_client, anthropic_server
):
    @callm.callm(block_pii=True)
    def openai_tools():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                *user("look up jane"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "crm", "arguments": '{"email": "jane@acme.com"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "phone +1 415 555 0100"},
            ],
        )

    openai_tools()
    sent = openai_server.last.body["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"email": "[EMAIL_1]"}'
    assert sent[1]["content"] is None
    assert sent[2]["content"] == "phone [PHONE_1]"

    @callm.callm(block_pii=True)
    def anthropic_blocks():
        return anthropic_client.messages.create(
            model="claude-opus-5",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "card 4111 1111 1111 1111",
                            },
                        },
                        {"type": "text", "text": "summarize"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "crm",
                            "input": {"contact": {"email": "kim@corp.io"}},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
                },
            ],
        )

    anthropic_blocks()
    body = anthropic_server.last.body["messages"]
    assert body[0]["content"][0]["source"]["data"] == "card [CREDIT_CARD_1]"
    assert body[1]["content"][0]["input"] == {"contact": {"email": "[EMAIL_1]"}}


# --------------------------------------------------------------------------- complete + fallback


def test_complete_anthropic_temperature_can_fall_back(providers, anthropic_server, openai_server):
    anthropic_server.fail(529)
    response = callm.complete(
        "anthropic/claude-sonnet-4-5",
        "hi",
        temperature=0.3,
        retry=False,
        fallback=["openai/gpt-4o-mini"],
    )
    assert response.provider == "openai"
    assert openai_server.last.body["temperature"] == 0.3


def test_translated_anthropic_conversations_start_with_a_user_turn(
    providers, openai_server, anthropic_server
):
    openai_server.fail(503)
    callm.complete(
        "gpt-4o",
        [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "refund"},
        ],
        retry=False,
        fallback=["anthropic/claude-haiku-4-5"],
    )
    assert anthropic_server.last.body["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------- scopes & callables


def test_long_lived_shield_does_not_accumulate_records(openai_client):
    guard = shield()
    with guard:
        for _ in range(50):
            openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))
        from callm.interception import current_context

        assert not hasattr(current_context(), "records")
    gc.collect()


def test_partials_and_callable_instances_can_be_decorated(openai_client, records):
    def ask(model, text):
        return openai_client.chat.completions.create(model=model, messages=user(text))

    class Asker:
        def __call__(self, text):
            return ask("gpt-4o-mini", text)

    callm.callm(functools.partial(ask, "gpt-4o-mini"))("hi")
    callm.callm(Asker())("hi")
    assert len(records()) == 2


async def test_sync_function_returning_a_coroutine_stays_protected(
    async_openai_client, openai_server
):
    @callm.callm(block_pii=True)
    def ask():
        return async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("a@b.io")
        )

    response = await ask()
    assert response.choices[0].message.content == "Hello from OpenAI"
    assert openai_server.last.body["messages"][0]["content"] == "[EMAIL_1]"


async def test_async_callable_instances_are_awaited(async_anthropic_client, anthropic_server):
    class Asker:
        async def __call__(self):
            return await async_anthropic_client.messages.create(
                model="claude-haiku-4-5", max_tokens=5, messages=user("x")
            )

    wrapped = callm.callm(retry=1)(Asker())
    anthropic_server.fail(529)
    assert (await wrapped()).content[0].text == "Hello from Claude"
    assert anthropic_body()["type"] == "message"


def test_anthropic_reset_with_nanosecond_timestamps():
    from datetime import datetime, timedelta, timezone

    from callm.classify import _parse_rfc3339

    future = (datetime.now(timezone.utc) + timedelta(seconds=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.123456789Z"
    )
    assert 25 <= _parse_rfc3339(future) <= 31


def test_gemini_function_parts_and_document_content_sources_are_masked():
    from callm.types import Message

    gemini = Message(
        "user",
        [
            {"function_call": {"name": "crm", "args": {"email": "a@b.io", "n": 3}}},
            {"function_response": {"name": "crm", "response": {"phone": "+1 415 555 0100"}}},
            {
                "type": "document",
                "source": {"type": "content", "content": [{"type": "text", "text": "x@y.io"}]},
                "context": "from z@w.io",
            },
            {"type": "document", "source": {"type": "content", "content": "c@d.io"}},
            {"type": "image", "source": {"type": "base64", "data": "a@b.io"}},
            {"type": "tool_result", "tool_use_id": "1"},
            "not-a-dict",
        ],
        native={
            "role": "user",
            "function_call": {"name": "f", "arguments": '{"e": "q@r.io"}'},
            "tool_calls": ["odd"],
        },
    )
    masked = gemini.map_text(lambda text: text.replace("@", "(at)"))
    parts = masked.content
    assert parts[0]["function_call"]["args"] == {"email": "a(at)b.io", "n": 3}
    assert parts[1]["function_response"]["response"] == {"phone": "+1 415 555 0100"}
    assert parts[2]["source"]["content"][0]["text"] == "x(at)y.io"
    assert parts[2]["context"] == "from z(at)w.io"
    assert parts[3]["source"]["content"] == "c(at)d.io"
    assert parts[4]["source"]["data"] == "a@b.io"  # binary payloads are never touched
    assert parts[5] == {"type": "tool_result", "tool_use_id": "1"}
    assert masked.native["function_call"]["arguments"] == '{"e": "q(at)r.io"}'
    assert masked.native["tool_calls"] == ["odd"]
    assert "a@b.io" in list(gemini.iter_texts())
    assert gemini.native["function_call"]["arguments"] == '{"e": "q@r.io"}'  # original untouched


async def test_async_shield_inherits_outer_protections(async_openai_client, openai_server):
    @callm.callm
    async def inner():
        return await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("a@b.io")
        )

    async with shield(block_pii=True, detect_injection=True):
        await inner()
    assert openai_server.last.body["messages"][0]["content"] == "[EMAIL_1]"


def test_to_plain_describes_types_and_callables():
    from callm.providers.base import to_plain

    class Plain:
        pass

    class Broken:
        @classmethod
        def model_json_schema(cls):
            raise RuntimeError("no schema")

    assert to_plain(Plain) == {"__type__": f"{Plain.__module__}.{Plain.__qualname__}"}
    assert "schema" not in to_plain(Broken)
    assert to_plain(len) == {"__callable__": "builtins.len"}


def test_last_call_sees_calls_made_inside_asyncio_run(async_openai_client, openai_client):
    import asyncio

    @callm.callm(block_pii=True, name="async.ask")
    async def ask_async():
        return await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("mail a@b.io")
        )

    @callm.callm(name="sync.ask")
    def ask_sync():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    assert callm.last_call() is None
    asyncio.run(ask_async())
    record = callm.last_call()
    assert record is not None
    assert record.function == "async.ask"
    assert record.pii_redactions == {"email": 1}

    ask_sync()  # a newer call in this context wins again
    assert callm.last_call().function == "sync.ask"


def test_last_call_is_per_context_for_concurrent_tasks(async_openai_client):
    import asyncio

    async def main():
        @callm.callm(name="first")
        async def first():
            await asyncio.sleep(0)
            return await async_openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=user("1")
            )

        @callm.callm(name="second")
        async def second():
            return await async_openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=user("2")
            )

        async def run(fn, expected):
            await fn()
            assert callm.last_call().function == expected

        await asyncio.gather(run(first, "first"), run(second, "second"))

    asyncio.run(main())
