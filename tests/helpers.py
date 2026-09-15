"""Mock provider servers and response builders shared by the tests."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx2


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Captured:
    url: str
    method: str
    headers: dict[str, str]
    body: Any


class MockServer:
    """Queue of scripted replies; records every request it receives."""

    def __init__(self, default: Any) -> None:
        self.default = default
        self.replies: deque[Reply | Exception] = deque()
        self.requests: list[Captured] = []

    def queue(self, *replies: Reply | Exception | dict[str, Any]) -> MockServer:
        for reply in replies:
            self.replies.append(Reply(body=reply) if isinstance(reply, dict) else reply)
        return self

    def fail(
        self, status: int, times: int = 1, headers: dict[str, str] | None = None
    ) -> MockServer:
        for _ in range(times):
            self.replies.append(
                Reply(
                    status=status,
                    body={
                        "type": "error",
                        "error": {
                            "message": f"mock {status}",
                            "type": "mock_error",
                            "code": status,
                        },
                    },
                    headers=headers or {},
                )
            )
        return self

    def handler(self, request: Any) -> Any:
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            Captured(str(request.url), request.method, dict(request.headers), body)
        )
        reply = self.replies.popleft() if self.replies else Reply(body=self.default())
        if isinstance(reply, Exception):
            raise reply
        return httpx2.Response(reply.status, json=reply.body, headers=reply.headers)

    @property
    def last(self) -> Captured:
        return self.requests[-1]

    @property
    def count(self) -> int:
        return len(self.requests)


def openai_body(
    text: str = "Hello from OpenAI",
    model: str = "gpt-4o-mini-2024-07-18",
    prompt_tokens: int = 12,
    completion_tokens: int = 5,
    cached_tokens: int = 0,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def anthropic_body(
    text: str = "Hello from Claude",
    model: str = "claude-sonnet-5",
    input_tokens: int = 10,
    output_tokens: int = 6,
    stop_reason: str = "end_turn",
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


def gemini_body(
    text: str = "Hello from Gemini",
    model: str = "gemini-2.5-flash",
    prompt_tokens: int = 8,
    output_tokens: int = 4,
    finish: str = "STOP",
) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": prompt_tokens + output_tokens,
        },
        "modelVersion": model,
    }


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]
