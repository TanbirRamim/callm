"""OpenAI Responses API adapter (``client.responses.create``).

The Responses API takes ``instructions`` plus an ``input`` that is either a string or a
list of items, and returns ``output`` items instead of ``choices``. This adapter maps
both to callm's canonical messages so PII masking, budgets, caching, validation, cost
tracking and fallback work the same way they do for Chat Completions.
"""

from __future__ import annotations

from typing import Any

from callm.pipeline import acall_bypassed, call_bypassed
from callm.providers.base import AttrDict, attr, dump_json, now, synthetic_id, to_plain
from callm.providers.openai import OpenAIProvider
from callm.types import LLMRequest, LLMResponse, Message, Usage

#: Marks the system message that came from ``instructions`` (not from an input item).
_INSTRUCTIONS = "__callm_instructions__"
#: Marks a request whose ``input`` was a plain string, so it is sent back as one.
_STRING_INPUT = "__string_input__"

_INCOMPLETE_TO_FINISH = {"max_output_tokens": "length", "content_filter": "content_filter"}


def _item_to_message(item: Any) -> Message:
    native = to_plain(item)
    if isinstance(native, str):
        return Message(role="user", content=native, native={"role": "user", "content": native})
    if not isinstance(native, dict):
        text = str(native)
        return Message(role="user", content=text, native={"role": "user", "content": text})
    kind = native.get("type", "message")
    if kind == "message":
        role = str(native.get("role", "user"))
        content = native.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            content = [to_plain(part) for part in content]
        elif not isinstance(content, str):
            content = str(content)
        canonical_role = "system" if role == "developer" else role
        return Message(role=canonical_role, content=content, native=native)
    if kind == "function_call_output" and isinstance(native.get("output"), str):
        # Tool results are user data too: expose them to PII masking and injection scans.
        return Message(role="tool", content=native["output"], native=native)
    # function_call, reasoning, file_search_call...: kept verbatim, never portable.
    return Message(role=str(kind), content="", native=native)


def _message_to_item(message: Message, same_origin: bool) -> dict[str, Any]:
    native = message.native
    if same_origin and native is not None:
        data = dict(native)
        kind = data.get("type", "message")
        if kind == "function_call_output" and isinstance(message.content, str):
            data["output"] = message.content
        elif kind == "message":
            data["content"] = message.content
        return data
    role = message.role if message.role in ("system", "user", "assistant") else "user"
    content = message.content if isinstance(message.content, str) else message.text
    return {"role": role, "content": content}


def _output_text(native: Any) -> str:
    text = attr(native, "output_text")
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for item in attr(native, "output") or []:
        if attr(item, "type") != "message":
            continue
        for part in attr(item, "content") or []:
            if attr(part, "type") == "output_text":
                parts.append(str(attr(part, "text", "") or ""))
    return "".join(parts)


class OpenAIResponsesProvider(OpenAIProvider):
    """Adapter for ``client.responses.create``; priced and reported as ``openai``."""

    canonical_param_keys = frozenset({"max_output_tokens", "temperature", "top_p"})

    def __init__(self, name: str = "openai-responses", *, pricing_name: str = "openai") -> None:
        super().__init__(name)
        self.pricing_name = pricing_name

    def owns_model(self, model: str) -> bool:
        return False  # bare model ids keep resolving to Chat Completions

    # ------------------------------------------------------------------ requests

    def parse_native_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        messages: list[Message] = []
        instructions = kwargs.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append(
                Message(role="system", content=instructions, native={_INSTRUCTIONS: True})
            )
        raw_input = kwargs.get("input")
        if isinstance(raw_input, str):
            messages.append(Message(role="user", content=raw_input, native={_STRING_INPUT: True}))
        elif raw_input is not None:
            messages.extend(_item_to_message(item) for item in raw_input)

        params: dict[str, Any] = {}
        if kwargs.get("max_output_tokens") is not None:
            params["max_tokens"] = kwargs["max_output_tokens"]
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]

        extra = {k: v for k, v in kwargs.items() if k not in ("input", "instructions", "model")}
        if isinstance(instructions, str) and not instructions:
            extra["instructions"] = instructions
        if isinstance(raw_input, str):
            extra[_STRING_INPUT] = True
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
            kwargs = {k: v for k, v in request.native_extra.items() if not k.startswith("__")}
            kwargs["model"] = request.model
            messages = list(request.messages)
            if messages and (messages[0].native or {}).get(_INSTRUCTIONS):
                kwargs["instructions"] = messages.pop(0).text
            if (
                request.native_extra.get(_STRING_INPUT)
                and len(messages) == 1
                and messages[0].role == "user"
                and isinstance(messages[0].content, str)
            ):
                kwargs["input"] = messages[0].content
            else:
                kwargs["input"] = [self._to_item(m) for m in messages]
            return kwargs

        system = "\n\n".join(m.text for m in request.messages if m.role == "system")
        kwargs = {
            "model": request.model,
            "input": [_message_to_item(m, False) for m in request.messages if m.role != "system"],
        }
        if system:
            kwargs["instructions"] = system
        if request.params.get("max_tokens") is not None:
            kwargs["max_output_tokens"] = request.params["max_tokens"]
        for key in ("temperature", "top_p"):
            if request.params.get(key) is not None:
                kwargs[key] = request.params[key]
        return kwargs  # the Responses API has no stop sequences

    @staticmethod
    def _to_item(message: Message) -> dict[str, Any]:
        native = message.native or {}
        if native.get(_INSTRUCTIONS) or native.get(_STRING_INPUT):
            # ``instructions`` moved into the input, or the string ``input`` as a list item.
            return {"role": message.role, "content": message.content}
        return _message_to_item(message, True)

    def portability_issue(self, request: LLMRequest) -> str | None:
        extra = {k: v for k, v in request.native_extra.items() if k != "instructions"}
        return super().portability_issue(request.replace(native_extra=extra))

    # ------------------------------------------------------------------ calls

    def client(self) -> Any:
        from callm.config import get_settings

        clients = get_settings().clients
        configured = clients.get(self.name) or clients.get(self.pricing_name)
        return configured if configured is not None else super().client()

    def async_client(self) -> Any:
        from callm.config import get_settings

        clients = get_settings().async_clients
        configured = clients.get(self.name) or clients.get(self.pricing_name)
        return configured if configured is not None else super().async_client()

    def call_sync(self, request: LLMRequest) -> Any:
        return call_bypassed(self.client().responses.create, **self.to_kwargs(request))

    async def call_async(self, request: LLMRequest) -> Any:
        create = self.async_client().responses.create
        return await acall_bypassed(create, **self.to_kwargs(request))

    # ------------------------------------------------------------------ responses

    def parse_response(self, native: Any, request: LLMRequest) -> LLMResponse:
        usage_obj = attr(native, "usage")
        total_input = int(attr(usage_obj, "input_tokens", 0) or 0)
        details = attr(usage_obj, "input_tokens_details")
        cached = int(attr(details, "cached_tokens", 0) or 0)
        written = int(attr(details, "cache_write_tokens", 0) or 0)
        usage = Usage(
            input_tokens=max(total_input - cached - written, 0),
            output_tokens=int(attr(usage_obj, "output_tokens", 0) or 0),
            cache_read_tokens=cached,
            cache_write_tokens=written,
            reported=usage_obj is not None,
        )
        status = attr(native, "status")
        if status == "incomplete":
            reason = attr(attr(native, "incomplete_details"), "reason")
            finish = _INCOMPLETE_TO_FINISH.get(str(reason), str(reason or "incomplete"))
        elif any(attr(item, "type") == "function_call" for item in attr(native, "output") or []):
            finish = "tool_calls"
        else:
            finish = "stop" if status in (None, "completed") else str(status)
        return LLMResponse(
            text=_output_text(native),
            provider=self.name,
            model=str(attr(native, "model") or request.model),
            usage=usage,
            finish_reason=finish,
            id=attr(native, "id"),
            raw=native,
            native_dump=dump_json(native),
        )

    def build_native(self, response: LLMResponse) -> Any:
        usage = response.usage
        input_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        finish = response.finish_reason or "stop"
        incomplete = finish in ("length", "max_tokens", "MAX_TOKENS")
        response_id = response.id or ""
        data = {
            "id": response_id if response_id.startswith("resp_") else synthetic_id("resp_"),
            "object": "response",
            "created_at": now(),
            "model": response.model,
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
            "error": None,
            "instructions": None,
            "metadata": {},
            "output": [
                {
                    "type": "message",
                    "id": synthetic_id("msg_"),
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": response.text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "cached_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                },
                "output_tokens": usage.output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + usage.output_tokens,
            },
        }
        return self.load_native(data)

    def load_native(self, data: dict[str, Any]) -> Any:
        try:
            from openai.types.responses import Response
        except ImportError:
            return AttrDict.wrap({**data, "output_text": _output_text(AttrDict.wrap(data))})
        try:
            return Response.model_validate(data)
        except Exception:
            return AttrDict.wrap({**data, "output_text": _output_text(AttrDict.wrap(data))})


__all__ = ["OpenAIResponsesProvider"]
