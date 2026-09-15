"""Provider fallback chains."""

from __future__ import annotations

import pytest

import callm
from callm.errors import AllProvidersFailedError
from helpers import gemini_body, user


@pytest.fixture(autouse=True)
def _clients(providers):
    """callm's own clients for fallback targets point at the mock servers."""


def test_openai_failure_falls_back_to_anthropic_with_openai_shaped_result(
    openai_client, openai_server, anthropic_server, records
):
    openai_server.fail(503, times=3)

    @callm.callm(retry=1, fallback=["anthropic/claude-sonnet"])
    def summarize(text: str):
        return openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            temperature=0.2,
            messages=[{"role": "system", "content": "Summarize."}, *user(text)],
        )

    result = summarize("a long article")

    # The caller still receives an OpenAI ChatCompletion.
    assert type(result).__name__ == "ChatCompletion"
    assert result.choices[0].message.content == "Hello from Claude"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert openai_server.count == 2  # 1 try + 1 retry

    sent = anthropic_server.last.body
    assert sent["model"] == "claude-sonnet-5"  # alias resolved
    assert sent["system"] == "Summarize."
    assert sent["messages"] == [{"role": "user", "content": "a long article"}]
    assert sent["max_tokens"] == 200
    assert "temperature" not in sent

    (record,) = records()
    assert record.provider == "anthropic"
    assert record.fallback_from == "openai/gpt-4o"
    assert record.retries == 1


def test_anthropic_failure_falls_back_to_gemini(anthropic_client, anthropic_server, gemini_server):
    anthropic_server.fail(529)

    @callm.callm(retry=False, fallback="google/gemini-2.5-flash")
    def ask():
        return anthropic_client.messages.create(
            model="claude-opus-5", max_tokens=64, system="Be terse.", messages=user("hi")
        )

    message = ask()
    assert type(message).__name__ == "Message"
    assert message.content[0].text == "Hello from Gemini"
    body = gemini_server.last.body
    assert body["systemInstruction"]["parts"][0]["text"] == "Be terse."
    assert body["generationConfig"]["maxOutputTokens"] == 64


def test_gemini_failure_falls_back_to_openai(gemini_client, gemini_server, openai_server):
    gemini_server.fail(500)

    @callm.callm(retry=False, fallback=["gpt-4o-mini"])
    def ask():
        return gemini_client.models.generate_content(model="gemini-2.5-pro", contents="hi")

    response = ask()
    assert type(response).__name__ == "GenerateContentResponse"
    assert response.text == "Hello from OpenAI"
    assert openai_server.last.body["messages"] == [{"role": "user", "content": "hi"}]


def test_same_provider_fallback_keeps_native_arguments(openai_client, openai_server):
    openai_server.fail(429)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    @callm.callm(retry=False, fallback=["openai/gpt-4o-mini"])
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o", tools=tools, messages=user("x")
        )

    ask()
    assert [r.body["model"] for r in openai_server.requests] == ["gpt-4o", "gpt-4o-mini"]
    assert openai_server.last.body["tools"] == tools


def test_unportable_requests_skip_cross_provider_fallbacks(
    openai_client, openai_server, anthropic_server
):
    openai_server.fail(503)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    @callm.callm(retry=False, fallback=["anthropic/claude-sonnet-5", "openai/gpt-4o-mini"])
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o", tools=tools, messages=user("x")
        )

    assert ask().choices[0].message.content == "Hello from OpenAI"
    assert anthropic_server.count == 0
    assert openai_server.last.body["model"] == "gpt-4o-mini"


def test_non_transient_primary_errors_do_not_fall_back(
    openai_client, openai_server, anthropic_server
):
    import openai

    openai_server.fail(400)

    @callm.callm(fallback=["anthropic/claude-sonnet-5"])
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    with pytest.raises(openai.BadRequestError):
        ask()
    assert anthropic_server.count == 0


def test_custom_fallback_predicate(openai_client, openai_server, anthropic_server):
    openai_server.fail(404)

    @callm.callm(fallback=["anthropic/claude-sonnet-5"], fallback_on=lambda exc: True)
    def ask():
        return openai_client.chat.completions.create(model="gpt-retired", messages=user("x"))

    assert ask().choices[0].message.content == "Hello from Claude"


def test_all_providers_failing_raises_with_every_error(
    openai_client, openai_server, anthropic_server, gemini_server, records
):
    openai_server.fail(503)
    anthropic_server.fail(529)
    gemini_server.fail(500)

    @callm.callm(retry=False, fallback=["anthropic/claude-sonnet-5", "google/gemini-2.5-flash"])
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    with pytest.raises(AllProvidersFailedError) as info:
        ask()
    labels = [label for label, _ in info.value.errors]
    assert labels == ["openai/gpt-4o", "anthropic/claude-sonnet-5", "google/gemini-2.5-flash"]
    assert info.value.last_error is info.value.__cause__
    assert records()[0].status == "error"


def test_fallback_to_broken_provider_is_reported(openai_client, openai_server):
    callm.configure(clients={"ollama": object()})  # a client that cannot make calls
    openai_server.fail(503)

    @callm.callm(retry=False, fallback=["ollama/llama3"])
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    with pytest.raises(AllProvidersFailedError) as info:
        ask()
    assert info.value.errors[1][0] == "ollama/llama3"


async def test_async_fallback(async_openai_client, openai_server, gemini_server):
    openai_server.fail(503)
    gemini_server.queue(gemini_body(text="async gemini"))

    @callm.callm(retry=False, fallback=["gemini/gemini-2.5-flash"])
    async def ask():
        return await async_openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    result = await ask()
    assert result.choices[0].message.content == "async gemini"


def test_complete_with_fallback(openai_server, anthropic_server):
    openai_server.fail(503)
    response = callm.complete(
        "gpt-4o", "hello", retry=False, fallback=["anthropic/claude-haiku-4-5"], max_tokens=20
    )
    assert response.provider == "anthropic"
    assert response.text == "Hello from Claude"
    assert anthropic_server.last.body["max_tokens"] == 20
