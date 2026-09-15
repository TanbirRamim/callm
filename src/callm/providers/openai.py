"""OpenAI Chat Completions adapter (also used for OpenAI-compatible servers such as Ollama)."""

from __future__ import annotations

import os
import threading
from typing import Any

from callm.errors import MissingDependencyError
from callm.pipeline import acall_bypassed, call_bypassed
from callm.providers.base import AttrDict, Provider, attr, dump_json, now, synthetic_id, to_plain
from callm.types import LLMRequest, LLMResponse, Message, Usage

_FINISH_TO_OPENAI = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "STOP": "stop",
    "length": "length",
    "max_tokens": "length",
    "MAX_TOKENS": "length",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "content_filter",
    "SAFETY": "content_filter",
}


class OpenAIProvider(Provider):
    canonical_param_keys = frozenset(
        {"max_tokens", "max_completion_tokens", "temperature", "top_p", "stop"}
    )

    def __init__(
        self,
        name: str = "openai",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        max_tokens_param: str = "max_completion_tokens",
        model_prefixes: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.max_tokens_param = max_tokens_param
        self.model_prefixes = model_prefixes
        self._client: Any = None
        self._aclient: Any = None
        self._lock = threading.Lock()

    def owns_model(self, model: str) -> bool:
        return bool(self.model_prefixes) and model.startswith(self.model_prefixes)

    # ------------------------------------------------------------------ requests

    def parse_native_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        messages: list[Message] = []
        for item in kwargs.get("messages") or []:
            native = to_plain(item)
            if not isinstance(native, dict):
                native = {"role": "user", "content": str(native)}
            role = str(native.get("role", "user"))
            content = native.get("content")
            if content is None:
                content = ""
            elif isinstance(content, list):
                content = [to_plain(part) for part in content]
            elif not isinstance(content, str):
                content = str(content)
            canonical_role = "system" if role == "developer" else role
            messages.append(Message(role=canonical_role, content=content, native=native))

        params: dict[str, Any] = {}
        max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]
        stop = kwargs.get("stop")
        if stop is not None:
            params["stop"] = [stop] if isinstance(stop, str) else list(stop)

        extra = {k: v for k, v in kwargs.items() if k not in ("messages", "model")}
        return LLMRequest(
            provider=self.name,
            model=str(kwargs.get("model", "")),
            messages=messages,
            params=params,
            native_extra=extra,
            origin=self.name,
            stream=bool(kwargs.get("stream")),
        ).snapshot()

    @staticmethod
    def _serialize(message: Message, same_origin: bool) -> dict[str, Any]:
        if same_origin and message.native is not None:
            data = dict(message.native)
            data["content"] = message.content
            if message.native.get("content") is None and message.content == "":
                data["content"] = None
            return data
        role = message.role if message.role in ("system", "user", "assistant") else "user"
        content = message.content if isinstance(message.content, str) else message.text
        return {"role": role, "content": content}

    def to_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        if request.origin == self.name:
            kwargs = {k: v for k, v in request.native_extra.items() if not k.startswith("__")}
            kwargs["model"] = request.model
            kwargs["messages"] = [self._serialize(m, True) for m in request.messages]
            return kwargs
        kwargs = {
            "model": request.model,
            "messages": [self._serialize(m, False) for m in request.messages],
        }
        if request.params.get("max_tokens") is not None:
            kwargs[self.max_tokens_param] = request.params["max_tokens"]
        for key in ("temperature", "top_p"):
            if request.params.get(key) is not None:
                kwargs[key] = request.params[key]
        if request.params.get("stop"):
            kwargs["stop"] = list(request.params["stop"])[:4]
        return kwargs

    # ------------------------------------------------------------------ calls

    def _sdk(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise MissingDependencyError(f"The '{self.name}' provider", "openai", "openai") from exc
        return openai

    def _client_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.base_url:
            options["base_url"] = self.base_url
        api_key = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if api_key:
            options["api_key"] = api_key
        return options

    def client(self) -> Any:
        from callm.config import get_settings

        configured = get_settings().clients.get(self.name)
        if configured is not None:
            return configured
        with self._lock:
            if self._client is None:
                self._client = self._sdk().OpenAI(**self._client_options())
            return self._client

    def async_client(self) -> Any:
        from callm.config import get_settings

        configured = get_settings().async_clients.get(self.name)
        if configured is not None:
            return configured
        with self._lock:
            if self._aclient is None:
                self._aclient = self._sdk().AsyncOpenAI(**self._client_options())
            return self._aclient

    def call_sync(self, request: LLMRequest) -> Any:
        return call_bypassed(self.client().chat.completions.create, **self.to_kwargs(request))

    async def call_async(self, request: LLMRequest) -> Any:
        create = self.async_client().chat.completions.create
        return await acall_bypassed(create, **self.to_kwargs(request))

    # ------------------------------------------------------------------ responses

    def parse_response(self, native: Any, request: LLMRequest) -> LLMResponse:
        choices = attr(native, "choices") or []
        first = choices[0] if choices else None
        content = attr(attr(first, "message"), "content")
        if isinstance(content, list):  # some compatible servers return content parts
            text = "".join(str(attr(part, "text", "") or "") for part in content)
        else:
            text = content or ""
        usage_obj = attr(native, "usage")
        prompt = int(attr(usage_obj, "prompt_tokens", 0) or 0)
        completion = int(attr(usage_obj, "completion_tokens", 0) or 0)
        cached = int(attr(attr(usage_obj, "prompt_tokens_details"), "cached_tokens", 0) or 0)
        usage = Usage(
            input_tokens=max(prompt - cached, 0), output_tokens=completion, cache_read_tokens=cached
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(attr(native, "model") or request.model),
            usage=usage,
            finish_reason=attr(first, "finish_reason"),
            id=attr(native, "id"),
            raw=native,
            native_dump=dump_json(native),
        )

    def build_native(self, response: LLMResponse) -> Any:
        finish = _FINISH_TO_OPENAI.get(response.finish_reason or "stop", "stop")
        usage = response.usage
        prompt_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        response_id = response.id or ""
        data = {
            "id": response_id if response_id.startswith("chatcmpl") else synthetic_id("chatcmpl-"),
            "object": "chat.completion",
            "created": now(),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": finish,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": prompt_tokens + usage.output_tokens,
                "prompt_tokens_details": {"cached_tokens": usage.cache_read_tokens},
            },
        }
        return self.load_native(data)

    def load_native(self, data: dict[str, Any]) -> Any:
        try:
            from openai.types.chat import ChatCompletion
        except ImportError:
            return AttrDict.wrap(data)
        try:
            return ChatCompletion.model_validate(data)
        except Exception:
            return AttrDict.wrap(data)


__all__ = ["OpenAIProvider"]
