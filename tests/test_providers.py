"""Provider adapters: request parsing, translation and native response synthesis."""

from __future__ import annotations

import pytest

from callm.providers import AnthropicProvider, GoogleProvider, OpenAIProvider
from callm.providers.base import AttrDict
from callm.types import LLMResponse, Message, Target, Usage
from helpers import anthropic_body, gemini_body, openai_body

openai = OpenAIProvider("openai")
anthropic = AnthropicProvider()
google = GoogleProvider()


def test_openai_round_trip_preserves_native_fields():
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "developer", "content": "rules"},
            {
                "role": "user",
                "name": "alice",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "42"},
        ],
        "max_completion_tokens": 100,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "stop": "END",
    }
    request = openai.parse_native_request(kwargs)
    assert request.params == {"max_tokens": 100, "stop": ["END"]}
    assert request.messages[0].role == "system"
    assert openai.to_kwargs(request) == kwargs
    assert openai.portability_issue(request) is not None
    text_only = openai.parse_native_request(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tools": []}
    )
    assert "tools" in openai.portability_issue(text_only)


def test_openai_translation_from_other_providers():
    request = anthropic.parse_native_request(
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "system": "S",
            "messages": [{"role": "user", "content": "Q"}],
            "stop_sequences": ["x"],
        }
    ).retarget(Target("openai", "gpt-4o"))
    assert openai.to_kwargs(request) == {
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
        "max_completion_tokens": 10,
        "stop": ["x"],
    }


def test_anthropic_system_blocks_and_mid_conversation_system_messages():
    kwargs = {
        "model": "claude-opus-5",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "cached rules", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "be brief from now on"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ],
        "thinking": {"type": "adaptive"},
    }
    request = anthropic.parse_native_request(kwargs)
    assert anthropic.to_kwargs(request) == kwargs
    redacted = request.with_messages([m.map_text(str.upper) for m in request.messages])
    out = anthropic.to_kwargs(redacted)
    assert out["system"][0]["text"] == "CACHED RULES"
    assert out["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert out["messages"][1] == {"role": "system", "content": "BE BRIEF FROM NOW ON"}
    assert anthropic.portability_issue(request) is None


def test_anthropic_translation_merges_consecutive_roles():
    request = openai.parse_native_request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            "temperature": 0.1,
        }
    ).retarget(Target("anthropic", "claude-sonnet"))
    out = anthropic.to_kwargs(request)
    assert out == {
        "model": "claude-sonnet-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "a\n\nb"}],
    }


def test_gemini_contents_forms():
    request = google.parse_native_request({"model": "gemini-2.5-flash", "contents": ["a", "b"]})
    assert len(request.messages) == 1 and request.messages[0].text == "a\nb"
    request = google.parse_native_request(
        {
            "model": "gemini-2.5-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "q"}]},
                {"role": "model", "parts": [{"text": "r"}]},
                "follow-up",
            ],
        }
    )
    assert [m.role for m in request.messages] == ["user", "assistant", "user"]
    assert google.to_kwargs(request)["contents"] == [
        {"role": "user", "parts": [{"text": "q"}]},
        {"role": "model", "parts": [{"text": "r"}]},
        "follow-up",
    ]  # unchanged requests are passed through verbatim
    changed = request.with_messages([m.map_text(str.upper) for m in request.messages])
    assert google.to_kwargs(changed)["contents"][2] == {
        "role": "user",
        "parts": [{"text": "FOLLOW-UP"}],
    }


def test_gemini_portability_checks_config():
    request = google.parse_native_request(
        {
            "model": "gemini-2.5-flash",
            "contents": "x",
            "config": {"response_schema": {"type": "object"}},
        }
    )
    assert "response_schema" in google.portability_issue(request)
    ok = google.parse_native_request(
        {"model": "gemini-2.5-flash", "contents": "x", "config": {"temperature": 0.2}}
    )
    assert google.portability_issue(ok) is None


@pytest.mark.parametrize(
    ("adapter", "body", "text", "usage"),
    [
        (
            openai,
            openai_body(prompt_tokens=100, completion_tokens=7, cached_tokens=60),
            "Hello from OpenAI",
            Usage(40, 7, 60, 0),
        ),
        (
            anthropic,
            anthropic_body(input_tokens=5, output_tokens=3, cache_read=50, cache_write=10),
            "Hello from Claude",
            Usage(5, 3, 50, 10),
        ),
    ],
)
def test_parse_response_usage(adapter, body, text, usage):
    response = adapter.parse_response(adapter.load_native(body), None)
    assert response.text == text
    assert response.usage == usage


def test_gemini_parse_response_skips_thoughts_and_counts_thinking_tokens():
    body = gemini_body()
    body["candidates"][0]["content"]["parts"].insert(0, {"text": "thinking...", "thought": True})
    body["usageMetadata"]["thoughtsTokenCount"] = 20
    native = google.load_native(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": body["candidates"][0]["content"]["parts"],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 8,
                "candidates_token_count": 4,
                "thoughts_token_count": 20,
            },
        }
    )
    response = google.parse_response(
        native, google.parse_native_request({"model": "gemini-2.5-pro", "contents": "x"})
    )
    assert response.text == "Hello from Gemini"
    assert response.usage.output_tokens == 24
    assert response.finish_reason == "STOP"
    assert response.model == "gemini-2.5-pro"


@pytest.mark.parametrize("adapter", [openai, anthropic, google])
def test_build_native_produces_real_sdk_objects(adapter):
    canonical = LLMResponse(
        text="synthesized",
        provider="other",
        model="m",
        usage=Usage(3, 4, 5, 0),
        finish_reason="length",
    )
    native = adapter.build_native(canonical)
    assert not isinstance(native, AttrDict)
    parsed = adapter.parse_response(
        native,
        None
        if adapter is not google
        else google.parse_native_request({"model": "m", "contents": "x"}),
    )
    assert parsed.text == "synthesized"
    assert parsed.usage.output_tokens == 4


def test_native_for_prefers_raw_then_dump():
    native = openai.load_native(openai_body())
    response = openai.parse_response(native, None)
    assert openai.native_for(response) is native
    response.raw = None
    assert openai.native_for(response).model_dump() == native.model_dump()


def test_message_helpers():
    message = Message(
        "user",
        [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"text": "b"}],
    )
    assert message.text == "a\nb"
    assert not message.is_text_only
    assert list(message.iter_texts()) == ["a", "b"]
    assert message.map_text(str.upper).content[2] == {"text": "B"}
