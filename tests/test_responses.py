"""OpenAI Responses API (``client.responses.create``) interception."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

import callm
from callm.errors import BudgetExceededError
from callm.providers import AnthropicProvider, OpenAIResponsesProvider
from callm.providers.base import AttrDict
from callm.types import LLMResponse, Target, Usage
from helpers import MockServer, responses_body

adapter = OpenAIResponsesProvider()


@pytest.fixture
def responses_server() -> MockServer:
    return MockServer(responses_body)


@pytest.fixture
def client(responses_server, anthropic_server, providers):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    sdk = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(responses_server.handler)),
    )
    if not hasattr(sdk, "responses"):
        pytest.skip("this openai version has no Responses API")
    return sdk


@pytest.fixture
def aclient(responses_server):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    sdk = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(responses_server.handler)),
    )
    if not hasattr(sdk, "responses"):
        pytest.skip("this openai version has no Responses API")
    return sdk


# --------------------------------------------------------------------------- adapter


def test_round_trip_preserves_native_request():
    kwargs = {
        "model": "gpt-5-mini",
        "instructions": "Be brief.",
        "input": [
            {"role": "developer", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image?"},
                    {"type": "input_image", "image_url": "https://x/y.png"},
                ],
            },
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ],
        "max_output_tokens": 100,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "function", "name": "f", "parameters": {}}],
    }
    request = adapter.parse_native_request(kwargs)
    assert request.provider == "openai-responses"
    assert request.params == {"max_tokens": 100}
    assert [m.role for m in request.messages] == [
        "system",
        "system",
        "user",
        "function_call",
        "tool",
    ]
    assert request.messages[2].text == "what is in this image?"
    assert adapter.to_kwargs(request) == kwargs
    assert adapter.portability_issue(request) is not None


def test_string_input_stays_a_string_and_plain_requests_are_portable():
    kwargs = {"model": "gpt-5-mini", "instructions": "S", "input": "Q", "temperature": 0.2}
    request = adapter.parse_native_request(kwargs)
    assert adapter.to_kwargs(request) == kwargs
    assert adapter.portability_issue(request) is None
    assert "previous_response_id" in adapter.portability_issue(
        adapter.parse_native_request({**kwargs, "previous_response_id": "resp_1"})
    )


def test_messages_added_by_callm_turn_string_input_into_items():
    request = adapter.parse_native_request({"model": "m", "input": "Q"})
    request = request.append(
        callm.types.Message("assistant", "A"), callm.types.Message("user", "B")
    )
    assert adapter.to_kwargs(request)["input"] == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "B"},
    ]


def test_translation_from_other_providers():
    request = (
        AnthropicProvider()
        .parse_native_request(
            {
                "model": "claude-sonnet-5",
                "max_tokens": 10,
                "system": "S",
                "messages": [{"role": "user", "content": "Q"}],
                "stop_sequences": ["x"],
            }
        )
        .retarget(Target("openai-responses", "gpt-5-mini"))
    )
    assert adapter.to_kwargs(request) == {
        "model": "gpt-5-mini",
        "instructions": "S",
        "input": [{"role": "user", "content": "Q"}],
        "max_output_tokens": 10,
    }


def test_parse_response_usage_and_finish_reasons():
    request = adapter.parse_native_request({"model": "m", "input": "Q"})
    native = AttrDict.wrap(responses_body(input_tokens=100, cached_tokens=40, output_tokens=7))
    response = adapter.parse_response(native, request)
    assert response.text == "Hello from Responses"
    assert response.usage == Usage(input_tokens=60, output_tokens=7, cache_read_tokens=40)
    assert response.finish_reason == "stop"

    incomplete = AttrDict.wrap(responses_body(status="incomplete"))
    assert adapter.parse_response(incomplete, request).finish_reason == "length"
    call = {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
    tools = AttrDict.wrap(responses_body(output=[call]))
    assert adapter.parse_response(tools, request).finish_reason == "tool_calls"
    no_usage = AttrDict.wrap({**responses_body(), "usage": None})
    assert adapter.parse_response(no_usage, request).usage.reported is False


def test_build_native_is_a_real_response_object():
    pytest.importorskip("openai.types.responses")
    usage = Usage(input_tokens=3, output_tokens=2, cache_read_tokens=1)
    native = adapter.build_native(
        LLMResponse(text="hi", provider="anthropic", model="c", usage=usage)
    )
    assert type(native).__name__ == "Response"
    assert native.output_text == "hi"
    assert native.usage.input_tokens == 4
    assert native.id.startswith("resp_")


def test_build_native_without_the_sdk_still_has_output_text(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "openai.types.responses", None)
    native = adapter.build_native(LLMResponse(text="hi", provider="x", model="m"))
    assert native.output_text == "hi"
    assert native.output[0].content[0].text == "hi"


def test_text_parts_are_scanned():
    message = adapter.parse_native_request(
        {
            "model": "m",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "t"}]}],
        }
    ).messages[0]
    assert list(message.iter_texts()) == ["t"]


# --------------------------------------------------------------------------- interception


def test_create_is_tracked_with_cost(client, responses_server, records):
    @callm.callm
    def ask():
        return client.responses.create(model="gpt-4o-mini", instructions="S", input="Q")

    result = ask()
    assert type(result).__name__ == "Response"
    assert result.output_text == "Hello from Responses"
    assert responses_server.last.url.endswith("/responses")
    assert responses_server.last.body == {"model": "gpt-4o-mini", "instructions": "S", "input": "Q"}
    (record,) = records()
    assert record.provider == "openai-responses"
    assert record.input_tokens == 12 and record.output_tokens == 5
    assert record.cost_usd and record.cost_usd > 0
    assert callm.last_call().cost_usd == record.cost_usd


def test_pii_is_masked_in_string_list_and_tool_output_input(client, responses_server):
    @callm.callm(block_pii=True)
    def ask():
        return client.responses.create(
            model="gpt-4o-mini",
            instructions="Contact ops@example.com if stuck.",
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": "I am a@b.co"}]},
                {"type": "function_call_output", "call_id": "c", "output": "phone b@c.io"},
            ],
        )

    ask()
    sent = str(responses_server.last.body)
    for address in ("ops@example.com", "a@b.co", "b@c.io"):
        assert address not in sent


def test_cache_hit_returns_a_response_without_calling_openai(client, responses_server, records):
    @callm.callm(cache=True)
    def ask():
        return client.responses.create(model="gpt-4o-mini", input="What is 2+2?")

    first = ask()
    second = ask()
    assert responses_server.count == 1
    assert type(second).__name__ == "Response"
    assert second.output_text == first.output_text
    assert records()[0].cache_hit is True


def test_budget_blocks_the_call_before_sending(client, responses_server):
    @callm.callm(max_cost=0.0000001)
    def ask():
        return client.responses.create(model="gpt-4o", input="x" * 4000, max_output_tokens=4000)

    with pytest.raises(BudgetExceededError):
        ask()
    assert responses_server.count == 0


def test_output_schema_validates_responses_text(client, responses_server):
    class Answer(BaseModel):
        answer: str

    responses_server.queue(responses_body(text="not json"), responses_body(text='{"answer": "4"}'))

    @callm.callm(output_schema=Answer)
    def ask():
        return client.responses.create(model="gpt-4o-mini", input="2+2? Reply as JSON.")

    ask()
    assert responses_server.count == 2
    assert callm.last_call().validation_retries == 1


def test_fallback_to_anthropic_returns_a_response(client, responses_server, anthropic_server):
    responses_server.fail(503, times=5)

    @callm.callm(fallback=["anthropic/claude-sonnet"])
    def ask():
        return client.responses.create(
            model="gpt-4o", instructions="Summarize.", input="text", max_output_tokens=50
        )

    result = ask()
    assert type(result).__name__ == "Response"
    assert result.output_text == "Hello from Claude"
    sent = anthropic_server.last.body
    assert sent["system"] == "Summarize."
    assert sent["messages"] == [{"role": "user", "content": "text"}]
    assert sent["max_tokens"] == 50


def test_complete_can_target_the_responses_api(client, responses_server):
    callm.configure(clients={"openai": client})
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    result = callm.complete("openai-responses/gpt-4o-mini", messages)
    assert result.text == "Hello from Responses"
    assert responses_server.last.body["instructions"] == "S"


async def test_async_create(aclient, responses_server, records):
    @callm.callm
    async def ask():
        return await aclient.responses.create(model="gpt-4o-mini", input="Q")

    result = await ask()
    assert result.output_text == "Hello from Responses"
    assert records()[0].provider == "openai-responses"


def test_stream_is_returned_untouched(client, responses_server):
    @callm.callm
    def ask():
        return client.responses.create(model="gpt-4o-mini", input="Q", stream=True)

    stream = ask()
    assert type(stream).__name__ == "Stream"
    assert responses_server.last.body["stream"] is True
