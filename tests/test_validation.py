"""Structured output validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

import callm
from callm import OutputValidationError
from callm.errors import ConfigurationError
from callm.validation import candidate_json, validate_text
from helpers import anthropic_body, gemini_body, openai_body, user


class Summary(BaseModel):
    title: str
    bullets: list[str]
    sentiment: float = Field(ge=-1, le=1)


VALID = '{"title": "Rates", "bullets": ["up", "down"], "sentiment": 0.25}'


def test_returns_a_validated_model(openai_client, openai_server):
    openai_server.queue(openai_body(text=VALID))

    @callm.callm(output_schema=Summary)
    def summarize(text: str):
        return openai_client.chat.completions.create(model="gpt-4o", messages=user(text))

    result = summarize("article")
    assert result == Summary(title="Rates", bullets=["up", "down"], sentiment=0.25)


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{VALID}\n```",
        f"Sure! Here is the JSON:\n{VALID}\nLet me know if you need more.",
        f"```\n{VALID}```",
    ],
)
def test_json_is_extracted_from_prose_and_fences(text):
    assert validate_text(text, Summary).title == "Rates"


def test_invalid_output_is_re_requested_with_the_errors(openai_client, openai_server, records):
    openai_server.queue(
        openai_body(text='{"title": "Rates", "bullets": "not a list", "sentiment": 3}'),
        openai_body(text=VALID),
    )

    @callm.callm(output_schema=Summary, validation_retries=2)
    def summarize():
        return openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": "JSON only"}, *user("article")]
        )

    assert summarize().sentiment == 0.25
    assert openai_server.count == 2
    retry_messages = openai_server.last.body["messages"]
    assert [m["role"] for m in retry_messages] == ["system", "user", "assistant", "user"]
    assert "bullets" in retry_messages[-1]["content"]
    assert "sentiment" in retry_messages[-1]["content"]
    assert '"properties"' in retry_messages[-1]["content"]  # JSON schema included
    assert records()[0].validation_retries == 1


def test_validation_exhaustion_raises(anthropic_client, anthropic_server):
    anthropic_server.queue(*(anthropic_body(text="I cannot do JSON") for _ in range(3)))

    @callm.callm(output_schema=Summary, validation_retries=2)
    def summarize():
        return anthropic_client.messages.create(
            model="claude-sonnet-5", max_tokens=100, messages=user("x")
        )

    with pytest.raises(OutputValidationError) as info:
        summarize()
    assert info.value.attempts == 3
    assert info.value.raw_text == "I cannot do JSON"
    assert anthropic_server.count == 3
    # Feedback turns alternate correctly for the Messages API.
    roles = [m["role"] for m in anthropic_server.last.body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_invalid_output_is_never_cached(openai_client, openai_server):
    openai_server.queue(openai_body(text="nope"), openai_body(text=VALID), openai_body(text="nope"))

    @callm.callm(output_schema=Summary, cache=True)
    def summarize():
        return openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    summarize()
    summarize()  # served from cache: the valid answer under the original prompt
    assert openai_server.count == 2


def test_list_and_primitive_schemas(gemini_client, gemini_server):
    gemini_server.queue(gemini_body(text="[1, 2, 3]"))

    @callm.callm(output_schema=list[int])
    def numbers():
        return gemini_client.models.generate_content(model="gemini-2.5-flash", contents="numbers")

    assert numbers() == [1, 2, 3]


def test_functions_that_post_process_the_response(openai_client, openai_server):
    openai_server.queue(openai_body(text=VALID))

    @callm.callm(output_schema=Summary)
    def summarize():
        response = openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))
        return response.choices[0].message.content

    assert summarize().title == "Rates"


def test_uninstrumented_functions_are_re_run_on_invalid_output():
    outputs = iter(["garbage", {"title": "t", "bullets": [], "sentiment": 0}])

    @callm.callm(output_schema=Summary, validation_retries=1)
    def produce():
        return next(outputs)

    assert produce().title == "t"


def test_uninstrumented_validation_exhaustion():
    @callm.callm(output_schema=Summary, validation_retries=0)
    def produce():
        return {"title": 1}

    with pytest.raises(OutputValidationError, match="Summary"):
        produce()


def test_invalid_schema_is_rejected_early():
    class NotASchema:
        def __init__(self, x):
            self.x = x

    with pytest.raises(ConfigurationError):
        callm.callm(output_schema=NotASchema)
    with pytest.raises(ConfigurationError):
        callm.callm(validation_retries=-1)


async def test_async_validation(async_openai_client, openai_server):
    openai_server.queue(openai_body(text="{}"), openai_body(text=VALID))

    @callm.callm(output_schema=Summary)
    async def summarize():
        return await async_openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    assert (await summarize()).bullets == ["up", "down"]


def test_complete_with_output_schema(providers, anthropic_server):
    anthropic_server.queue(anthropic_body(text=VALID))
    response = callm.complete("anthropic/claude-sonnet-5", "summarize", output_schema=Summary)
    assert isinstance(response.parsed, Summary)


def test_candidate_json_deduplicates():
    assert candidate_json('{"a": 1}') == ['{"a": 1}']
    assert candidate_json("") == []
