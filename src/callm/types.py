"""Provider-neutral request, response and telemetry types."""

from __future__ import annotations

import copy
import dataclasses
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

#: Roles callm understands. Provider-specific roles (``tool``, ``function``) are kept verbatim.
PORTABLE_ROLES = frozenset({"system", "user", "assistant"})


def is_text_part(part: Any) -> bool:
    """True for content parts that carry plain text (OpenAI, Anthropic and Gemini shapes)."""
    return (
        isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part.get("type", "text") == "text"
        and not part.get("thought", False)
    )


@dataclass
class Message:
    """A single chat message in callm's canonical form.

    ``content`` is either a string or a list of provider-native content parts (dicts).
    ``native`` keeps the original message dict so it can be re-sent without losing
    provider-specific fields (``name``, ``tool_calls``, ``cache_control``...).
    """

    role: str
    content: str | list[dict[str, Any]]
    native: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part["text"] for part in self.content if is_text_part(part))

    @property
    def is_text_only(self) -> bool:
        if isinstance(self.content, str):
            return True
        return all(is_text_part(part) for part in self.content)

    def iter_texts(self) -> Iterator[str]:
        """Every user-visible string, for security scans.

        Covers plain text parts plus text nested in structured fields: tool results, tool
        call arguments (OpenAI ``tool_calls``, Anthropic ``tool_use.input``, Gemini
        ``function_call`` / ``function_response``) and text documents.
        """
        found: list[str] = []

        def collect(text: str) -> str:
            found.append(text)
            return text

        self.map_text(collect)
        yield from found

    def map_text(self, fn: Callable[[str], str]) -> Message:
        """Return a copy with ``fn`` applied to every segment yielded by :meth:`iter_texts`."""
        if isinstance(self.content, str):
            new_content: str | list[dict[str, Any]] = fn(self.content)
        else:
            new_content = [_map_part(part, fn) for part in self.content]
        native = self.native
        if native and (native.get("tool_calls") or native.get("function_call")):
            native = dict(native)
            if native.get("tool_calls"):
                native["tool_calls"] = [_map_tool_call(call, fn) for call in native["tool_calls"]]
            if isinstance(native.get("function_call"), dict):
                native["function_call"] = _map_tool_call({"function": native["function_call"]}, fn)[
                    "function"
                ]
        return Message(role=self.role, content=new_content, native=native)


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string inside a JSON-like structure."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    return value


def _map_tool_call(call: Any, fn: Callable[[str], str]) -> Any:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return call
    function = dict(call["function"])
    if isinstance(function.get("arguments"), str):
        function["arguments"] = fn(function["arguments"])
    return {**call, "function": function}


def _map_part(part: Any, fn: Callable[[str], str]) -> Any:
    if not isinstance(part, dict):
        return part
    if is_text_part(part):
        return {**part, "text": fn(part["text"])}
    kind = part.get("type")
    if kind == "tool_result":
        nested = part.get("content")
        if isinstance(nested, str):
            return {**part, "content": fn(nested)}
        if isinstance(nested, list):
            return {**part, "content": [_map_part(p, fn) for p in nested]}
        return part
    if kind in ("tool_use", "server_tool_use") and "input" in part:
        return {**part, "input": _map_strings(part["input"], fn)}
    if kind == "document":
        updated = dict(part)
        source = part.get("source")
        if (
            isinstance(source, dict)
            and source.get("type") == "text"
            and isinstance(source.get("data"), str)
        ):
            updated["source"] = {**source, "data": fn(source["data"])}
        elif isinstance(source, dict) and source.get("type") == "content":
            content = source.get("content")
            if isinstance(content, str):
                updated["source"] = {**source, "content": fn(content)}
            elif isinstance(content, list):
                updated["source"] = {**source, "content": [_map_part(p, fn) for p in content]}
        if isinstance(part.get("context"), str):
            updated["context"] = fn(part["context"])
        return updated
    if isinstance(part.get("function_call"), dict):  # Gemini
        call = dict(part["function_call"])
        if "args" in call:
            call["args"] = _map_strings(call["args"], fn)
        return {**part, "function_call": call}
    if isinstance(part.get("function_response"), dict):  # Gemini
        response = dict(part["function_response"])
        if "response" in response:
            response["response"] = _map_strings(response["response"], fn)
        return {**part, "function_response": response}
    return part


@dataclass(frozen=True)
class Target:
    """A provider + model pair, e.g. ``Target("anthropic", "claude-sonnet-5")``."""

    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def __str__(self) -> str:
        return self.label


@dataclass
class LLMRequest:
    """A provider-neutral LLM request.

    ``params`` holds the canonical sampling parameters callm understands
    (``max_tokens``, ``temperature``, ``top_p``, ``stop``). ``native_extra`` holds every
    other keyword argument exactly as the caller passed it to the provider SDK; it is only
    re-used when the request is sent to that same provider (``origin``).
    """

    provider: str
    model: str
    messages: list[Message]
    params: dict[str, Any] = field(default_factory=dict)
    native_extra: dict[str, Any] = field(default_factory=dict)
    origin: str | None = None
    stream: bool = False
    pristine: list[Message] | None = field(default=None, repr=False, compare=False)

    def snapshot(self) -> LLMRequest:
        """Remember the current messages so adapters can detect later modifications."""
        self.pristine = copy.deepcopy(self.messages)
        return self

    @property
    def messages_unchanged(self) -> bool:
        return self.pristine is not None and self.pristine == self.messages

    @property
    def target(self) -> Target:
        return Target(self.provider, self.model)

    def replace(self, **changes: Any) -> LLMRequest:
        new = dataclasses.replace(self, **changes)
        new.pristine = self.pristine
        return new

    def retarget(self, target: Target) -> LLMRequest:
        """Point the request at another provider/model.

        Provider-native keyword arguments only survive when the provider is unchanged.
        """
        if target.provider == self.origin:
            return self.replace(provider=target.provider, model=target.model)
        return self.replace(provider=target.provider, model=target.model, native_extra={})

    def with_messages(self, messages: list[Message]) -> LLMRequest:
        return self.replace(messages=messages)

    def append(self, *messages: Message) -> LLMRequest:
        return self.replace(messages=[*self.messages, *messages])

    @property
    def system_text(self) -> str:
        return "\n\n".join(m.text for m in self.messages if m.role == "system")

    @property
    def last_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.text
        return ""

    @property
    def all_text(self) -> str:
        return "\n".join(m.text for m in self.messages)


@dataclass
class Usage:
    """Token usage for one call.

    ``input_tokens`` counts only *uncached* input tokens; cache reads and writes are
    reported separately so they can be priced correctly.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cache_read_tokens=int(data.get("cache_read_tokens") or 0),
            cache_write_tokens=int(data.get("cache_write_tokens") or 0),
        )


@dataclass
class LLMResponse:
    """A provider-neutral LLM response.

    ``raw`` is the provider SDK's native response object (``ChatCompletion``, ``Message``,
    ``GenerateContentResponse``). ``parsed`` holds the validated ``output_schema`` instance.
    """

    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    cost: float | None = None
    latency_ms: float = 0.0
    cached: bool = False
    id: str | None = None
    raw: Any = field(default=None, repr=False, compare=False)
    native_dump: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    parsed: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "cost": self.cost,
            "id": self.id,
            "native_dump": self.native_dump,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        return cls(
            text=data.get("text") or "",
            provider=data["provider"],
            model=data["model"],
            usage=Usage.from_dict(data.get("usage")),
            finish_reason=data.get("finish_reason"),
            cost=data.get("cost"),
            id=data.get("id"),
            native_dump=data.get("native_dump"),
        )


@dataclass
class CallRecord:
    """One telemetry row. Never contains prompt or completion text."""

    function: str | None = None
    provider: str = "unknown"
    model: str = "unknown"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    saved_usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    retries: int = 0
    validation_retries: int = 0
    fallback_from: str | None = None
    status: str = "ok"
    error_type: str | None = None
    streamed: bool = False
    pii_redactions: dict[str, int] = field(default_factory=dict)
    injection_score: float | None = None
    injection_flagged: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["total_tokens"] = self.total_tokens
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallRecord:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


__all__ = [
    "PORTABLE_ROLES",
    "CallRecord",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "Target",
    "Usage",
    "is_text_part",
]
