"""``shield`` context manager and the direct ``complete`` / ``acomplete`` API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any

from callm.config import (
    CacheConfig,
    CallConfig,
    InjectionConfig,
    PIIConfig,
    RetryConfig,
    build_call_config,
    merge_scopes,
)
from callm.errors import OutputValidationError
from callm.interception import ACTIVE, clean_kwargs, ensure_installed, enter_scope
from callm.middleware import build_handler
from callm.pipeline import CallState, run_async, run_sync
from callm.providers.anthropic import AnthropicProvider
from callm.providers.base import to_plain
from callm.providers.registry import get_provider, parse_target
from callm.types import CallRecord, LLMRequest, LLMResponse, Message, Target
from callm.validation import ValidationFailure, validate_text

if TYPE_CHECKING:
    from callm.budgets import Budget

CONFIG_FIELDS = (
    "cache",
    "retry",
    "fallback",
    "fallback_on",
    "max_cost",
    "budget",
    "block_pii",
    "detect_injection",
    "output_schema",
    "validation_retries",
    "name",
    "tags",
    "telemetry",
)


def _to_messages(messages: str | Sequence[Any]) -> list[Message]:
    if isinstance(messages, str):
        return [Message("user", messages)]
    result: list[Message] = []
    for item in messages:
        if isinstance(item, Message):
            result.append(item)
            continue
        plain = to_plain(item)
        if not isinstance(plain, dict) or "role" not in plain:
            raise TypeError(f"messages must be dicts with 'role' and 'content', got {item!r}")
        content = plain.get("content", "")
        if not isinstance(content, (str, list)):
            content = "" if content is None else str(content)
        result.append(Message(str(plain["role"]), content))
    if not result:
        raise ValueError("messages must not be empty")
    return result


def anthropic_sampling_kwargs(
    sampling: dict[str, Any], extra_body: dict[str, Any] | None
) -> dict[str, Any]:
    """Pass ``temperature``/``top_p`` the way the installed ``anthropic`` SDK accepts them.

    ``anthropic>=1.0`` removed the keyword arguments (newer Claude models reject them), but
    older models still honour them when sent in the request body via ``extra_body``.
    """
    try:
        import inspect

        from anthropic.resources.messages import Messages

        accepted = set(inspect.signature(Messages.create).parameters)
    except Exception:
        accepted = set()
    direct = {k: v for k, v in sampling.items() if k in accepted}
    body = {k: v for k, v in sampling.items() if k not in accepted}
    result: dict[str, Any] = dict(direct)
    if body or extra_body:
        result["extra_body"] = {**(extra_body or {}), **body}
    return result


def build_request(
    model: str | Target,
    messages: str | Sequence[Any],
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: str | Sequence[str] | None = None,
    **provider_kwargs: Any,
) -> LLMRequest:
    """Build a provider-native request from canonical arguments plus extra SDK kwargs."""
    provider_kwargs = clean_kwargs(provider_kwargs)
    target = parse_target(model, provider)
    adapter = get_provider(target.provider)
    params: dict[str, Any] = {}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if stop is not None:
        params["stop"] = [stop] if isinstance(stop, str) else list(stop)
    canonical = LLMRequest(
        provider=target.provider,
        model=target.model,
        messages=_to_messages(messages),
        params=params,
    )
    kwargs = adapter.to_kwargs(canonical)
    if isinstance(adapter, AnthropicProvider):
        # Explicit sampling parameters are the caller's choice on models that accept them.
        sampling: dict[str, Any] = {k: params[k] for k in ("temperature", "top_p") if k in params}
        if sampling:
            kwargs.update(anthropic_sampling_kwargs(sampling, provider_kwargs.get("extra_body")))
            provider_kwargs.pop("extra_body", None)
    if "config" in provider_kwargs and isinstance(kwargs.get("config"), dict):
        user_config = provider_kwargs.pop("config")
        if isinstance(user_config, dict):
            kwargs["config"] = {**kwargs["config"], **user_config}
        elif user_config is not None:
            kwargs["config"] = user_config.model_copy(update=kwargs["config"])
    kwargs.update(provider_kwargs)
    return adapter.parse_native_request(kwargs)


def _finalize(response: LLMResponse, config: CallConfig) -> LLMResponse:
    if response.raw is None:
        try:
            response.raw = get_provider(response.provider).native_for(response)
        except Exception:
            response.raw = None
    if config.output_schema is not None and response.parsed is None:
        try:
            response.parsed = validate_text(response.text, config.output_schema)
        except ValidationFailure as failure:
            raise OutputValidationError(
                str(failure), errors=failure.errors, raw_text=failure.text, attempts=1
            ) from None
    return response


def _state(request: LLMRequest, config: CallConfig, mode: str) -> CallState:
    record = CallRecord(
        function=config.name or "callm.complete", provider=request.provider, model=request.model
    )
    return CallState(request=request, config=config, record=record, mode=mode)


def _inherit(config: CallConfig) -> CallConfig:
    parent = ACTIVE.get()
    return merge_scopes(parent.config, config) if parent is not None else config


def run_request(config: CallConfig, request: LLMRequest) -> LLMResponse:
    """Run a canonical request through the pipeline synchronously."""
    config = _inherit(config)
    response = run_sync(build_handler(config)(_state(request, config, "sync")))
    return _finalize(response, config)


async def arun_request(config: CallConfig, request: LLMRequest) -> LLMResponse:
    """Run a canonical request through the pipeline on the event loop."""
    config = _inherit(config)
    response = await run_async(build_handler(config)(_state(request, config, "async")))
    return _finalize(response, config)


def complete(
    model: str | Target,
    messages: str | Sequence[Any],
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: str | Sequence[str] | None = None,
    cache: bool | str | CacheConfig | None = False,
    retry: bool | int | RetryConfig | None = None,
    fallback: str | Sequence[str | Target] | None = None,
    fallback_on: Callable[[BaseException], bool] | None = None,
    max_cost: float | None = None,
    budget: Budget | None = None,
    block_pii: bool | PIIConfig | None = False,
    detect_injection: bool | InjectionConfig | None = False,
    output_schema: Any = None,
    validation_retries: int = 2,
    name: str | None = None,
    tags: Mapping[str, str] | None = None,
    telemetry: bool = True,
    **provider_kwargs: Any,
) -> LLMResponse:
    """Call a model through the callm pipeline and get a provider-neutral response.

    ``model`` is ``"provider/model"`` (``"anthropic/claude-sonnet-5"``) or a model id whose
    provider can be inferred (``"gpt-4o-mini"``). ``messages`` is a string or a list of
    ``{"role", "content"}`` dicts. Extra keyword arguments go to the provider SDK unchanged.

    Example::

        response = callm.complete("gpt-4o-mini", "Say hi", cache=True, retry=3)
        print(response.text, response.cost)
    """
    config = build_call_config(
        cache=cache,
        retry=retry,
        fallback=fallback,
        fallback_on=fallback_on,
        max_cost=max_cost,
        budget=budget,
        block_pii=block_pii,
        detect_injection=detect_injection,
        output_schema=output_schema,
        validation_retries=validation_retries,
        name=name,
        tags=tags,
        telemetry=telemetry,
    )
    request = build_request(
        model,
        messages,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        **provider_kwargs,
    )
    return run_request(config, request)


async def acomplete(
    model: str | Target,
    messages: str | Sequence[Any],
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: str | Sequence[str] | None = None,
    cache: bool | str | CacheConfig | None = False,
    retry: bool | int | RetryConfig | None = None,
    fallback: str | Sequence[str | Target] | None = None,
    fallback_on: Callable[[BaseException], bool] | None = None,
    max_cost: float | None = None,
    budget: Budget | None = None,
    block_pii: bool | PIIConfig | None = False,
    detect_injection: bool | InjectionConfig | None = False,
    output_schema: Any = None,
    validation_retries: int = 2,
    name: str | None = None,
    tags: Mapping[str, str] | None = None,
    telemetry: bool = True,
    **provider_kwargs: Any,
) -> LLMResponse:
    """Async version of :func:`complete`."""
    config = build_call_config(
        cache=cache,
        retry=retry,
        fallback=fallback,
        fallback_on=fallback_on,
        max_cost=max_cost,
        budget=budget,
        block_pii=block_pii,
        detect_injection=detect_injection,
        output_schema=output_schema,
        validation_retries=validation_retries,
        name=name,
        tags=tags,
        telemetry=telemetry,
    )
    request = build_request(
        model,
        messages,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        **provider_kwargs,
    )
    return await arun_request(config, request)


class shield:
    """Apply callm protections to a block of code, or make one-off calls.

    As a context manager, every supported SDK call inside the block goes through the
    pipeline::

        with shield(block_pii=True, detect_injection=True):
            client.chat.completions.create(model="gpt-4o", messages=messages)

    The object also offers :meth:`complete` / :meth:`acomplete` for direct calls::

        with shield(block_pii=True, detect_injection=True) as s:
            response = s.complete(provider="anthropic", model="claude-sonnet-5", messages=messages)
    """

    def __init__(
        self,
        *,
        cache: bool | str | CacheConfig | None = False,
        retry: bool | int | RetryConfig | None = None,
        fallback: str | Sequence[str | Target] | None = None,
        fallback_on: Callable[[BaseException], bool] | None = None,
        max_cost: float | None = None,
        budget: Budget | None = None,
        block_pii: bool | PIIConfig | None = False,
        detect_injection: bool | InjectionConfig | None = False,
        output_schema: Any = None,
        validation_retries: int = 2,
        name: str | None = "shield",
        tags: Mapping[str, str] | None = None,
        telemetry: bool = True,
    ) -> None:
        self._options: dict[str, Any] = {
            "cache": cache,
            "retry": retry,
            "fallback": fallback,
            "fallback_on": fallback_on,
            "max_cost": max_cost,
            "budget": budget,
            "block_pii": block_pii,
            "detect_injection": detect_injection,
            "output_schema": output_schema,
            "validation_retries": validation_retries,
            "name": name,
            "tags": tags,
            "telemetry": telemetry,
        }
        self.config = build_call_config(**self._options)
        self._handler = build_handler(self.config)

    # ----------------------------------------------------------------- context manager

    def __enter__(self) -> shield:
        ensure_installed()
        ACTIVE.set(enter_scope(self.config, self._handler, self.config.name, owner=self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Restore the enclosing scope. Context variables are per task/thread, so one shield
        # object can safely be entered concurrently from several tasks.
        ctx = ACTIVE.get()
        while ctx is not None and ctx.owner is not self:
            ctx = ctx.parent
        if ctx is not None:
            ACTIVE.set(ctx.parent)

    async def __aenter__(self) -> shield:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    # ----------------------------------------------------------------- direct calls

    def _split(self, kwargs: dict[str, Any]) -> tuple[CallConfig, dict[str, Any]]:
        overrides = {key: kwargs.pop(key) for key in CONFIG_FIELDS if key in kwargs}
        config = build_call_config(**{**self._options, **overrides}) if overrides else self.config
        return config, kwargs

    def complete(
        self,
        *,
        model: str | Target,
        messages: str | Sequence[Any],
        provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Direct call with this shield's protections (same arguments as :func:`callm.complete`)."""
        config, rest = self._split(dict(kwargs))
        return run_request(config, build_request(model, messages, provider=provider, **rest))

    async def acomplete(
        self,
        *,
        model: str | Target,
        messages: str | Sequence[Any],
        provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of :meth:`complete`."""
        config, rest = self._split(dict(kwargs))
        return await arun_request(config, build_request(model, messages, provider=provider, **rest))


__all__ = ["acomplete", "arun_request", "build_request", "complete", "run_request", "shield"]
