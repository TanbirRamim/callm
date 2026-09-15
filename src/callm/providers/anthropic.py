"""Anthropic Messages API adapter."""

from __future__ import annotations

import threading
from typing import Any

from callm.errors import MissingDependencyError
from callm.pipeline import acall_bypassed, call_bypassed
from callm.providers.base import AttrDict, Provider, attr, dump_json, synthetic_id, to_plain
from callm.types import LLMRequest, LLMResponse, Message, Usage

#: Short aliases accepted in targets such as ``"anthropic/claude-sonnet"``.
MODEL_ALIASES = {
    "claude-opus": "claude-opus-5",
    "claude-sonnet": "claude-sonnet-5",
    "claude-haiku": "claude-haiku-4-5",
    "claude-fable": "claude-fable-5-1",
}

#: ``max_tokens`` is required by the Messages API; used when a translated request has none.
DEFAULT_MAX_TOKENS = 4096

_SYSTEM_MARKER = "__callm_system__"
_STOP_REASONS = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"}
)
_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "STOP": "end_turn",
    "length": "max_tokens",
    "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "SAFETY": "refusal",
}


class AnthropicProvider(Provider):
    name = "anthropic"
    canonical_param_keys = frozenset({"max_tokens", "temperature", "top_p", "stop_sequences"})

    def __init__(self) -> None:
        self._client: Any = None
        self._aclient: Any = None
        self._lock = threading.Lock()

    def owns_model(self, model: str) -> bool:
        return model.startswith("claude")

    @staticmethod
    def resolve_model(model: str) -> str:
        return MODEL_ALIASES.get(model, model)

    # ------------------------------------------------------------------ requests

    def parse_native_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        messages: list[Message] = []
        system = kwargs.get("system")
        if isinstance(system, str):
            messages.append(Message("system", system, native={_SYSTEM_MARKER: "str"}))
        elif system is not None:
            blocks = [to_plain(block) for block in system]
            messages.append(Message("system", blocks, native={_SYSTEM_MARKER: "blocks"}))

        for item in kwargs.get("messages") or []:
            native = to_plain(item)
            if not isinstance(native, dict):
                native = {"role": "user", "content": str(native)}
            content = native.get("content", "")
            if isinstance(content, list):
                content = [to_plain(block) for block in content]
            elif not isinstance(content, str):
                content = str(content)
            messages.append(Message(str(native.get("role", "user")), content, native=native))

        params: dict[str, Any] = {}
        if kwargs.get("max_tokens") is not None:
            params["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("stop_sequences"):
            params["stop"] = list(kwargs["stop_sequences"])
        raw_extra_body = kwargs.get("extra_body")
        extra_body: dict[str, Any] = raw_extra_body if isinstance(raw_extra_body, dict) else {}
        for key in ("temperature", "top_p"):
            value = kwargs.get(key, extra_body.get(key))
            if value is not None:
                params[key] = value

        extra = {k: v for k, v in kwargs.items() if k not in ("messages", "model", "system")}
        return LLMRequest(
            provider=self.name,
            model=str(kwargs.get("model", "")),
            messages=messages,
            params=params,
            native_extra=extra,
            origin=self.name,
            stream=bool(kwargs.get("stream")),
        ).snapshot()

    def to_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        if request.origin == self.name:
            return self._native_kwargs(request)
        return self._translated_kwargs(request)

    def _native_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs = {k: v for k, v in request.native_extra.items() if not k.startswith("__")}
        kwargs["model"] = self.resolve_model(request.model)
        system_parts: list[Message] = []
        body: list[dict[str, Any]] = []
        for message in request.messages:
            if (message.native or {}).get(_SYSTEM_MARKER) is not None:
                system_parts.append(message)
            elif message.native is not None:
                # Includes mid-conversation {"role": "system"} messages, kept in place.
                data = dict(message.native)
                data["content"] = message.content
                body.append(data)
            elif message.role == "system":
                system_parts.append(message)
            else:
                body.append({"role": message.role, "content": message.content})
        if len(system_parts) == 1 and (system_parts[0].native or {}).get(_SYSTEM_MARKER):
            kwargs["system"] = system_parts[0].content
        elif system_parts:
            kwargs["system"] = "\n\n".join(m.text for m in system_parts if m.text)
        kwargs["messages"] = body
        return kwargs

    def _translated_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        system = "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)
        body: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue
            role = "assistant" if message.role == "assistant" else "user"
            if body and body[-1]["role"] == role:
                body[-1]["content"] += "\n\n" + message.text
            else:
                body.append({"role": role, "content": message.text})
        if body and body[0]["role"] == "assistant":
            # The Messages API requires the conversation to start with a user turn.
            body.insert(0, {"role": "user", "content": "(continued conversation)"})
        kwargs: dict[str, Any] = {
            "model": self.resolve_model(request.model),
            "max_tokens": int(request.params.get("max_tokens") or DEFAULT_MAX_TOKENS),
            "messages": body,
        }
        if system:
            kwargs["system"] = system
        if request.params.get("stop"):
            kwargs["stop_sequences"] = list(request.params["stop"])
        # temperature / top_p are intentionally not translated: current Claude models reject
        # them and anthropic>=1.0 removed them from the SDK signature.
        return kwargs

    # ------------------------------------------------------------------ calls

    def _sdk(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'anthropic' provider", "anthropic", "anthropic"
            ) from exc
        return anthropic

    def client(self) -> Any:
        from callm.config import get_settings

        configured = get_settings().clients.get(self.name)
        if configured is not None:
            return configured
        with self._lock:
            if self._client is None:
                self._client = self._sdk().Anthropic()
            return self._client

    def async_client(self) -> Any:
        from callm.config import get_settings

        configured = get_settings().async_clients.get(self.name)
        if configured is not None:
            return configured
        with self._lock:
            if self._aclient is None:
                self._aclient = self._sdk().AsyncAnthropic()
            return self._aclient

    def call_sync(self, request: LLMRequest) -> Any:
        return call_bypassed(self.client().messages.create, **self.to_kwargs(request))

    async def call_async(self, request: LLMRequest) -> Any:
        create = self.async_client().messages.create
        return await acall_bypassed(create, **self.to_kwargs(request))

    # ------------------------------------------------------------------ responses

    def parse_response(self, native: Any, request: LLMRequest) -> LLMResponse:
        texts = [
            str(attr(block, "text", "") or "")
            for block in attr(native, "content") or []
            if attr(block, "type") == "text"
        ]
        usage_obj = attr(native, "usage")
        usage = Usage(
            input_tokens=int(attr(usage_obj, "input_tokens", 0) or 0),
            output_tokens=int(attr(usage_obj, "output_tokens", 0) or 0),
            cache_read_tokens=int(attr(usage_obj, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(attr(usage_obj, "cache_creation_input_tokens", 0) or 0),
        )
        return LLMResponse(
            text="".join(texts),
            provider=self.name,
            model=str(attr(native, "model") or request.model),
            usage=usage,
            finish_reason=attr(native, "stop_reason"),
            id=attr(native, "id"),
            raw=native,
            native_dump=dump_json(native),
        )

    def build_native(self, response: LLMResponse) -> Any:
        finish = response.finish_reason or "end_turn"
        finish = _FINISH_TO_ANTHROPIC.get(finish, finish)
        if finish not in _STOP_REASONS:
            finish = "end_turn"
        response_id = response.id or ""
        data = {
            "id": response_id if response_id.startswith("msg_") else synthetic_id("msg_"),
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": [{"type": "text", "text": response.text}],
            "stop_reason": finish,
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": response.usage.cache_read_tokens,
                "cache_creation_input_tokens": response.usage.cache_write_tokens,
            },
        }
        return self.load_native(data)

    def load_native(self, data: dict[str, Any]) -> Any:
        try:
            from anthropic.types import Message as AnthropicMessage
        except ImportError:
            return AttrDict.wrap(data)
        try:
            return AnthropicMessage.model_validate(data)
        except Exception:
            return AttrDict.wrap(data)


__all__ = ["DEFAULT_MAX_TOKENS", "MODEL_ALIASES", "AnthropicProvider"]
