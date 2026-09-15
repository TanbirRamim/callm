"""The @callm decorator intercepting real SDK clients."""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

import callm
from callm import interception
from callm.errors import ConfigurationError
from helpers import anthropic_body, gemini_body, openai_body, user


def test_openai_call_is_intercepted_and_returns_native_object(
    openai_client, openai_server, records
):
    @callm.callm
    def ask(question: str):
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user(question))

    result = ask("hi")

    assert result.choices[0].message.content == "Hello from OpenAI"
    assert type(result).__name__ == "ChatCompletion"
    assert openai_server.count == 1
    (record,) = records()
    assert record.provider == "openai"
    assert record.model == "gpt-4o-mini-2024-07-18"
    assert record.function.endswith("ask")
    assert record.input_tokens == 12
    assert record.output_tokens == 5
    assert record.cost_usd == pytest.approx((12 * 0.15 + 5 * 0.6) / 1_000_000)
    assert record.status == "ok"


def test_decorator_forms_are_equivalent(openai_client):
    def body():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    bare = callm.callm(body)
    empty = callm.callm()(body)
    configured = callm.callm(cache=True, retry=1)(body)

    for fn in (bare, empty, configured):
        assert fn().choices[0].message.content == "Hello from OpenAI"
    assert configured.callm_config.retry.max_retries == 1
    assert bare.__name__ == "body"


def test_calls_outside_a_scope_are_untouched(openai_client, openai_server, records):
    @callm.callm
    def decorated():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("a"))

    decorated()
    openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("b"))

    assert openai_server.count == 2
    assert len(records()) == 1


def test_anthropic_call_is_intercepted(anthropic_client, anthropic_server, records):
    @callm.callm(retry=2)
    def ask():
        return anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            system="Be brief.",
            messages=user("hello"),
        )

    anthropic_server.fail(529)
    message = ask()

    assert message.content[0].text == "Hello from Claude"
    assert anthropic_server.count == 2
    assert anthropic_server.last.body["system"] == "Be brief."
    (record,) = records()
    assert record.retries == 1
    assert record.cost_usd == pytest.approx((10 * 2.0 + 6 * 10.0) / 1_000_000)


def test_gemini_call_is_intercepted(gemini_client, gemini_server, records):
    @callm.callm
    def ask():
        return gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="hello",
            config={"system_instruction": "Be brief.", "max_output_tokens": 50},
        )

    response = ask()

    assert response.text == "Hello from Gemini"
    body = gemini_server.last.body
    assert body["contents"] == [{"parts": [{"text": "hello"}], "role": "user"}]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    (record,) = records()
    assert record.provider == "google"
    assert record.cost_usd == pytest.approx((8 * 0.3 + 4 * 2.5) / 1_000_000)


async def test_async_clients_are_intercepted(
    async_openai_client, async_anthropic_client, gemini_client, openai_server, records
):
    @callm.callm(retry=1)
    async def ask_all():
        a = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("x")
        )
        b = await async_anthropic_client.messages.create(
            model="claude-haiku-4-5", max_tokens=10, messages=user("x")
        )
        c = await gemini_client.aio.models.generate_content(model="gemini-2.5-flash", contents="x")
        return a, b, c

    openai_server.fail(500)
    a, b, c = await ask_all()

    assert a.choices[0].message.content == "Hello from OpenAI"
    assert b.content[0].text == "Hello from Claude"
    assert c.text == "Hello from Gemini"
    assert sorted(r.provider for r in records()) == ["anthropic", "google", "openai"]
    assert next(r for r in records() if r.provider == "openai").retries == 1


async def test_concurrent_tasks_keep_separate_scopes(async_openai_client, records):
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

    await asyncio.gather(first(), second(), first())
    assert sorted(r.function for r in records()) == ["first", "first", "second"]


def test_threads_need_context_propagation(openai_client, records):
    def call():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("t"))

    @callm.callm(name="threaded")
    def run():
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(call).result()  # plain thread: not intercepted
            ctx = contextvars.copy_context()
            pool.submit(ctx.run, call).result()  # propagated context: intercepted

    run()
    assert [r.function for r in records()] == ["threaded"]


def test_nested_decorators_attribute_calls_to_innermost(openai_client, records):
    @callm.callm(name="inner")
    def inner():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("i"))

    @callm.callm(name="outer")
    def outer():
        inner()
        return "done"

    assert outer() == "done"
    assert [r.function for r in records()] == ["inner"]


def test_methods_can_be_decorated(openai_client):
    class Bot:
        def __init__(self) -> None:
            self.client = openai_client

        @callm.callm(cache=True)
        def reply(self, text: str) -> str:
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=user(text))
            return response.choices[0].message.content

    assert Bot().reply("hey") == "Hello from OpenAI"


def test_generator_functions_are_rejected():
    with pytest.raises(TypeError, match="generator"):

        @callm.callm
        def gen():
            yield 1


def test_invalid_configuration_fails_at_decoration_time():
    with pytest.raises(ConfigurationError):
        callm.callm(cache="sometimes")
    with pytest.raises(ConfigurationError):
        callm.callm(max_cost=-1)
    with pytest.raises(ConfigurationError):
        callm.callm(fallback=["no-such-model-family"])
    with pytest.raises(ConfigurationError):
        callm.callm(retry="3")


def test_disabled_setting_is_a_kill_switch(openai_client, records):
    callm.configure(enabled=False)

    @callm.callm(block_pii=True)
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("mail me at a@b.co")
        )

    ask()
    assert records() == []


def test_raw_response_calls_pass_through(openai_client, records):
    @callm.callm
    def raw():
        return openai_client.chat.completions.with_raw_response.create(
            model="gpt-4o-mini", messages=user("raw")
        )

    response = raw()
    assert response.parse().choices[0].message.content == "Hello from OpenAI"
    assert records() == []


def test_uninstrumented_function_retries_and_records_native_responses(openai_client, records):
    attempts = []

    class Flaky(Exception):
        status_code = 503

    @callm.callm(retry=2)
    def custom():
        attempts.append(1)
        if len(attempts) < 2:
            raise Flaky("temporarily unavailable")
        # Simulate an HTTP client callm cannot see by bypassing interception.
        return callm.pipeline.call_bypassed(
            openai_client.chat.completions.create, model="gpt-4o-mini", messages=user("x")
        )

    result = custom()
    assert result.choices[0].message.content == "Hello from OpenAI"
    assert len(attempts) == 2
    (record,) = records()
    assert record.retries == 1
    assert record.output_tokens == 5


def test_uninstall_restores_original_methods(openai_client):
    import openai

    callm.callm(lambda: None)()
    assert interception.instrumented()
    interception.uninstall()
    assert not hasattr(openai.resources.chat.completions.Completions.create, "__callm_original__")
    interception.ensure_installed()
    assert hasattr(openai.resources.chat.completions.Completions.create, "__callm_original__")


def test_streaming_passes_through_with_security(openai_client, openai_server, records):
    sse = (
        'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
        '"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        "data: [DONE]\n\n"
    )
    import httpx2

    def handler(request):
        openai_server.handler(request)
        return httpx2.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    openai_client._client = httpx2.Client(transport=httpx2.MockTransport(handler))

    @callm.callm(block_pii=True, cache=True)
    def stream():
        chunks = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=user("call 415-555-0100"), stream=True
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in chunks)

    assert stream() == "Hi"
    assert openai_server.last.body["messages"][0]["content"] == "call [PHONE_1]"
    (record,) = records()
    assert record.streamed is True
    assert record.pii_redactions == {"phone": 1}


def test_bodies_helpers_are_valid_sdk_payloads():
    from anthropic.types import Message
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion

    ChatCompletion.model_validate(openai_body())
    Message.model_validate(anthropic_body())
    GenerateContentResponse.model_validate(gemini_body())
