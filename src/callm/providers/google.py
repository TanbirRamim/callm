"""Google Gen AI (Gemini) adapter for the ``google-genai`` SDK."""

from __future__ import annotations

import threading
from typing import Any

from callm.errors import MissingDependencyError
from callm.pipeline import acall_bypassed, call_bypassed
from callm.providers.base import AttrDict, Provider, attr, dump_json, to_plain
from callm.types import LLMRequest, LLMResponse, Message, Usage

_SYSTEM_MARKER = "__callm_system__"
#: ``GenerateContentConfig`` fields that do not change the shape of the task.
_CONFIG_PORTABLE_KEYS = frozenset(
    {
        "system_instruction",
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "candidate_count",
        "seed",
        "http_options",
        "thinking_config",
        "safety_settings",
        "presence_penalty",
        "frequency_penalty",
        "labels",
    }
)
_FINISH_TO_GOOGLE = {
    "stop": "STOP",
    "end_turn": "STOP",
    "stop_sequence": "STOP",
    "length": "MAX_TOKENS",
    "max_tokens": "MAX_TOKENS",
    "content_filter": "SAFETY",
    "refusal": "SAFETY",
}


def _is_content(item: Any) -> bool:
    if isinstance(item, dict):
        return "parts" in item or "role" in item
    return hasattr(item, "parts") and hasattr(item, "role")


def _part(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"text": item}
    plain = to_plain(item)
    return plain if isinstance(plain, dict) else {"text": str(plain)}


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name else str(value)


def _config_get(config: Any, key: str) -> Any:
    return attr(config, key)


class GoogleProvider(Provider):
    name = "google"
    canonical_param_keys = frozenset({"config"})

    def __init__(self) -> None:
        self._client: Any = None
        self._lock = threading.Lock()

    def owns_model(self, model: str) -> bool:
        return model.startswith(("gemini", "models/gemini", "gemma"))

    # ------------------------------------------------------------------ requests

    def parse_native_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        messages: list[Message] = []
        config = kwargs.get("config")
        system = _config_get(config, "system_instruction")
        if system is not None:
            messages.append(self._system_message(system))

        contents = kwargs.get("contents")
        if isinstance(contents, list):
            items = contents
        elif contents is None:
            items = []
        else:
            items = [contents]
        group: list[dict[str, Any]] = []

        def flush() -> None:
            # Consecutive bare parts/strings form one user turn, as in the SDK.
            if group:
                messages.append(Message("user", list(group), native={"role": "user"}))
                group.clear()

        for item in items:
            if _is_content(item):
                flush()
                plain = to_plain(item)
                role = plain.get("role") or "user"
                parts = [_part(p) for p in plain.get("parts") or []]
                native = {k: v for k, v in plain.items() if k != "parts"}
                messages.append(Message("assistant" if role == "model" else role, parts, native))
            else:
                group.append(_part(item))
        flush()

        params: dict[str, Any] = {}
        for source, target in (
            ("max_output_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stop_sequences", "stop"),
        ):
            value = _config_get(config, source)
            if value is not None:
                params[target] = list(value) if target == "stop" else value

        extra = {k: v for k, v in kwargs.items() if k not in ("model", "contents")}
        extra["__contents__"] = contents
        return LLMRequest(
            provider=self.name,
            model=str(kwargs.get("model", "")),
            messages=messages,
            params=params,
            native_extra=extra,
            origin=self.name,
        ).snapshot()

    @staticmethod
    def _system_message(system: Any) -> Message:
        if isinstance(system, str):
            return Message("system", system, native={_SYSTEM_MARKER: "str"})
        if isinstance(system, list):
            return Message("system", [_part(p) for p in system], native={_SYSTEM_MARKER: "parts"})
        if _is_content(system):
            plain = to_plain(system)
            parts = [_part(p) for p in plain.get("parts") or []]
            return Message("system", parts, native={_SYSTEM_MARKER: "content"})
        return Message("system", [_part(system)], native={_SYSTEM_MARKER: "parts"})

    def portability_issue(self, request: LLMRequest) -> str | None:
        issue = super().portability_issue(request)
        if issue is not None:
            return issue
        config = request.native_extra.get("config")
        if config is not None:
            plain = to_plain(config) or {}
            for key, value in plain.items():
                if value not in (None, [], {}) and key not in _CONFIG_PORTABLE_KEYS:
                    return f"config field '{key}' is provider-specific"
            if (plain.get("candidate_count") or 1) != 1:
                return "candidate_count > 1 cannot be translated"
        return None

    def to_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        if request.origin == self.name:
            return self._native_kwargs(request)
        return self._translated_kwargs(request)

    def _native_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs = {k: v for k, v in request.native_extra.items() if not k.startswith("__")}
        kwargs["model"] = request.model
        if request.messages_unchanged and "__contents__" in request.native_extra:
            kwargs["contents"] = request.native_extra["__contents__"]
            return kwargs

        def is_system(message: Message) -> bool:
            return (message.native or {}).get(_SYSTEM_MARKER) is not None

        system_messages = [m for m in request.messages if is_system(m)]
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            if is_system(message):
                continue
            if message.role == "assistant":
                role = "model"
            elif message.role in ("user", "system"):
                role = "user"
            else:
                role = message.role
            if isinstance(message.content, str):
                parts = [{"text": message.content}]
            else:
                parts = list(message.content)
            native = {k: v for k, v in (message.native or {}).items() if k != "parts"}
            contents.append({**native, "role": role, "parts": parts})
        kwargs["contents"] = contents

        original_system = [m for m in (request.pristine or []) if is_system(m)]
        if system_messages != original_system:
            # A plain string is accepted in every config form (dict or pydantic object).
            new_system = "\n\n".join(m.text for m in system_messages) or None
            config = kwargs.get("config")
            if config is None:
                if new_system is not None:
                    kwargs["config"] = {"system_instruction": new_system}
            elif isinstance(config, dict):
                kwargs["config"] = {**config, "system_instruction": new_system}
            else:
                kwargs["config"] = config.model_copy(update={"system_instruction": new_system})
        return kwargs

    def _translated_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        system = "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append({"text": message.text})
            else:
                contents.append({"role": role, "parts": [{"text": message.text}]})
        config: dict[str, Any] = {}
        if system:
            config["system_instruction"] = system
        if request.params.get("max_tokens") is not None:
            config["max_output_tokens"] = int(request.params["max_tokens"])
        for key in ("temperature", "top_p"):
            if request.params.get(key) is not None:
                config[key] = request.params[key]
        if request.params.get("stop"):
            config["stop_sequences"] = list(request.params["stop"])
        kwargs: dict[str, Any] = {"model": request.model, "contents": contents}
        if config:
            kwargs["config"] = config
        return kwargs

    # ------------------------------------------------------------------ calls

    def client(self) -> Any:
        from callm.config import get_settings

        configured = get_settings().clients.get(self.name)
        if configured is not None:
            return configured
        with self._lock:
            if self._client is None:
                try:
                    from google import genai
                except ImportError as exc:
                    raise MissingDependencyError(
                        "The 'google' provider", "google-genai", "google"
                    ) from exc
                self._client = genai.Client()
            return self._client

    def call_sync(self, request: LLMRequest) -> Any:
        return call_bypassed(self.client().models.generate_content, **self.to_kwargs(request))

    async def call_async(self, request: LLMRequest) -> Any:
        from callm.config import get_settings

        aclient = get_settings().async_clients.get(self.name)
        models = aclient.models if aclient is not None else self.client().aio.models
        return await acall_bypassed(models.generate_content, **self.to_kwargs(request))

    # ------------------------------------------------------------------ responses

    def parse_response(self, native: Any, request: LLMRequest) -> LLMResponse:
        candidates = attr(native, "candidates") or []
        first = candidates[0] if candidates else None
        texts = [
            str(attr(part, "text"))
            for part in attr(attr(first, "content"), "parts") or []
            if attr(part, "text") is not None and not attr(part, "thought", False)
        ]
        meta = attr(native, "usage_metadata")
        prompt = int(attr(meta, "prompt_token_count", 0) or 0)
        cached = int(attr(meta, "cached_content_token_count", 0) or 0)
        output = int(attr(meta, "candidates_token_count", 0) or 0) + int(
            attr(meta, "thoughts_token_count", 0) or 0
        )
        usage = Usage(
            input_tokens=max(prompt - cached, 0), output_tokens=output, cache_read_tokens=cached
        )
        return LLMResponse(
            text="".join(texts),
            provider=self.name,
            model=str(attr(native, "model_version") or request.model),
            usage=usage,
            finish_reason=_enum_name(attr(first, "finish_reason")),
            id=attr(native, "response_id"),
            raw=native,
            native_dump=dump_json(native),
        )

    def build_native(self, response: LLMResponse) -> Any:
        finish = response.finish_reason or "STOP"
        finish = _FINISH_TO_GOOGLE.get(finish, finish)
        if not finish.isupper():
            finish = "STOP"
        usage = response.usage
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": response.text}]},
                    "finish_reason": finish,
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": usage.input_tokens + usage.cache_read_tokens,
                "candidates_token_count": usage.output_tokens,
                "total_token_count": usage.total_tokens,
            },
            "model_version": response.model,
        }
        if usage.cache_read_tokens:
            data["usage_metadata"]["cached_content_token_count"] = usage.cache_read_tokens
        return self.load_native(data)

    def load_native(self, data: dict[str, Any]) -> Any:
        try:
            from google.genai import types
        except ImportError:
            return AttrDict.wrap(data)
        try:
            return types.GenerateContentResponse.model_validate(data)
        except Exception:
            return AttrDict.wrap(data)


__all__ = ["GoogleProvider"]
