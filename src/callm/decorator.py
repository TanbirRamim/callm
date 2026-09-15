"""The ``@callm`` decorator."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, get_args, overload

from callm import pipeline
from callm.budgets import active_budgets
from callm.classify import should_retry
from callm.config import (
    CacheConfig,
    CallConfig,
    InjectionConfig,
    PIIConfig,
    RetryConfig,
    build_call_config,
    get_settings,
)
from callm.errors import OutputValidationError
from callm.interception import ACTIVE, ExecutionContext, ensure_installed, enter_scope
from callm.middleware import build_handler
from callm.middleware.telemetry import emit
from callm.pricing import cost_for
from callm.providers.base import AttrDict
from callm.providers.registry import get_provider
from callm.types import CallRecord, LLMRequest, LLMResponse, Target
from callm.validation import ValidationFailure, schema_name, validate_text, validate_value

if TYPE_CHECKING:
    from callm.budgets import Budget

F = TypeVar("F", bound=Callable[..., Any])

_EMPTY_REQUEST = LLMRequest(provider="unknown", model="", messages=[])


def detect_provider(value: Any) -> str | None:
    """Which provider SDK produced ``value`` (a native response object), if any."""
    module = type(value).__module__ or ""
    if module.startswith("openai"):
        return "openai" if hasattr(value, "choices") else None
    if module.startswith("anthropic"):
        return "anthropic" if hasattr(value, "content") and hasattr(value, "usage") else None
    if module.startswith("google.genai"):
        return "google" if hasattr(value, "candidates") else None
    if isinstance(value, AttrDict):
        if "choices" in value:
            return "openai"
        if value.get("type") == "message":
            return "anthropic"
        if "candidates" in value:
            return "google"
    return None


def _is_instance(value: Any, schema: Any) -> bool:
    # Parameterized generics such as ``list[int]`` are not valid isinstance() targets (and
    # on Python 3.10 ``isinstance(list[int], type)`` is even True), so guard with try/except.
    if not inspect.isclass(schema) or get_args(schema):
        return False
    try:
        return isinstance(value, schema)
    except TypeError:
        return False


def parse_result(result: Any, schema: Any) -> Any:
    """Validate a decorated function's return value against ``schema``."""
    if isinstance(result, LLMResponse):
        return result.parsed if result.parsed is not None else validate_text(result.text, schema)
    if _is_instance(result, schema):
        return result
    if isinstance(result, (str, bytes)):
        text = result.decode("utf-8") if isinstance(result, bytes) else result
        return validate_text(text, schema)
    provider = detect_provider(result)
    if provider is not None:
        text = get_provider(provider).parse_response(result, _EMPTY_REQUEST).text
        return validate_text(text, schema)
    return validate_value(result, schema)


def _record_uninstrumented(
    ctx: ExecutionContext, result: Any, started: float, retries: int
) -> None:
    """Telemetry for SDK responses produced outside interception (e.g. a custom client)."""
    provider = detect_provider(result)
    if provider is None:
        return
    try:
        response = get_provider(provider).parse_response(result, _EMPTY_REQUEST)
    except Exception:
        return
    cost = cost_for(provider, response.model, response.usage) if response.model else None
    record = CallRecord(
        function=ctx.name,
        provider=provider,
        model=response.model or "unknown",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=response.usage.cache_read_tokens,
        cache_write_tokens=response.usage.cache_write_tokens,
        cost_usd=cost,
        latency_ms=(time.perf_counter() - started) * 1000,
        retries=retries,
        tags=dict(ctx.config.tags),
    )
    if cost:
        for budget in dict.fromkeys((*ctx.config.budgets, *active_budgets())):
            budget.add(cost)
    emit(record, persist=ctx.config.telemetry)


def _validation_error(
    failure: ValidationFailure, schema: Any, attempts: int
) -> OutputValidationError:
    return OutputValidationError(
        f"Return value did not match {schema_name(schema)}: {'; '.join(failure.errors[:5])}",
        errors=failure.errors,
        raw_text=failure.text,
        attempts=attempts,
    )


class _Runner:
    """Executes one decorated function under a callm scope."""

    def __init__(self, fn: Callable[..., Any], config: CallConfig) -> None:
        self.fn = fn
        self.config = config
        qualname = getattr(fn, "__qualname__", None) or type(fn).__qualname__
        module = getattr(fn, "__module__", None) or type(fn).__module__
        self.name = config.name or f"{module}.{qualname}"
        self.handler = build_handler(config)

    def _enter(self) -> tuple[ExecutionContext, Any]:
        ensure_installed()
        ctx = enter_scope(self.config, self.handler, self.name)
        return ctx, ACTIVE.set(ctx)

    def _retry_delay(self, ctx: ExecutionContext, exc: Exception, retries: int) -> float | None:
        """Retry the whole function only when no SDK call was intercepted.

        Intercepted calls are retried individually inside the pipeline, so retrying the
        function again would multiply attempts.
        """
        if ctx.calls != 0:
            return None
        retryable, info = should_retry(exc, self.config.retry)
        if not retryable or retries >= self.config.retry.max_retries:
            return None
        return self.config.retry.delay_for(retries, info.retry_after if info else None)

    def _finish(self, ctx: ExecutionContext, result: Any, started: float, retries: int) -> Any:
        if ctx.calls == 0:
            _record_uninstrumented(ctx, result, started, retries)
        if self.config.output_schema is None:
            return result
        return parse_result(result, self.config.output_schema)

    def run_sync(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not get_settings().enabled:
            return self.fn(*args, **kwargs)
        ctx, token = self._enter()
        try:
            retries = validations = 0
            while True:
                ctx.calls = 0
                started = time.perf_counter()
                try:
                    result = self.fn(*args, **kwargs)
                except Exception as exc:
                    delay = self._retry_delay(ctx, exc, retries)
                    if delay is None:
                        raise
                    retries += 1
                    pipeline.sleep_sync(delay)
                    continue
                if inspect.isawaitable(result):
                    # A plain ``def`` returned a coroutine (e.g. ``return async_client.x(...)``):
                    # finish the work inside a scope when the caller awaits it.
                    return self._resume(result)
                try:
                    return self._finish(ctx, result, started, retries)
                except ValidationFailure as failure:
                    if ctx.calls == 0 and validations < self.config.validation_retries:
                        validations += 1
                        continue
                    raise _validation_error(
                        failure, self.config.output_schema, validations + 1
                    ) from None
        finally:
            ACTIVE.reset(token)

    async def _resume(self, awaitable: Any) -> Any:
        ctx, token = self._enter()
        try:
            started = time.perf_counter()
            result = await awaitable
            try:
                return self._finish(ctx, result, started, 0)
            except ValidationFailure as failure:
                raise _validation_error(failure, self.config.output_schema, 1) from None
        finally:
            ACTIVE.reset(token)

    async def run_async(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not get_settings().enabled:
            return await self.fn(*args, **kwargs)
        ctx, token = self._enter()
        try:
            retries = validations = 0
            while True:
                ctx.calls = 0
                started = time.perf_counter()
                try:
                    result = await self.fn(*args, **kwargs)
                except Exception as exc:
                    delay = self._retry_delay(ctx, exc, retries)
                    if delay is None:
                        raise
                    retries += 1
                    await pipeline.sleep_async(delay)
                    continue
                try:
                    return self._finish(ctx, result, started, retries)
                except ValidationFailure as failure:
                    if ctx.calls == 0 and validations < self.config.validation_retries:
                        validations += 1
                        continue
                    raise _validation_error(
                        failure, self.config.output_schema, validations + 1
                    ) from None
        finally:
            ACTIVE.reset(token)


@overload
def callm(func: F, /) -> F: ...


@overload
def callm(
    func: None = None,
    /,
    *,
    cache: bool | str | CacheConfig | None = ...,
    retry: bool | int | RetryConfig | None = ...,
    fallback: str | Sequence[str | Target] | None = ...,
    fallback_on: Callable[[BaseException], bool] | None = ...,
    max_cost: float | None = ...,
    budget: Budget | None = ...,
    block_pii: bool | PIIConfig | None = ...,
    detect_injection: bool | InjectionConfig | None = ...,
    output_schema: Any = ...,
    validation_retries: int = ...,
    name: str | None = ...,
    tags: Mapping[str, str] | None = ...,
    telemetry: bool = ...,
) -> Callable[[F], F]: ...


def callm(
    func: Callable[..., Any] | None = None,
    /,
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
    name: str | None = None,
    tags: Mapping[str, str] | None = None,
    telemetry: bool = True,
) -> Any:
    """Make every LLM call inside the decorated function production-ready.

    Works as ``@callm``, ``@callm()`` or ``@callm(...)`` on sync and async functions and
    methods. Supported SDK calls made inside the function are intercepted; the function's
    code does not change.

    Args:
        cache: ``True``/``"exact"`` for an exact-match response cache, ``"semantic"`` to also
            match similar prompts, or a :class:`~callm.CacheConfig`.
        retry: Retries on rate limits, 5xx, timeouts and connection errors (default 2),
            ``False`` to disable, or a :class:`~callm.RetryConfig`.
        fallback: Models to try in order when the primary keeps failing, e.g.
            ``["anthropic/claude-sonnet-5", "google/gemini-2.5-flash"]``.
        fallback_on: Predicate deciding which primary errors trigger a fallback.
        max_cost: Refuse any single call whose estimated cost exceeds this many USD.
        budget: A shared :class:`~callm.Budget` charged by every call.
        block_pii: Mask emails, phone numbers, SSNs, card numbers, IP addresses and IBANs
            before they leave the process (or refuse with ``PIIConfig(action="block")``).
        detect_injection: Score user input for prompt injection; flag (default) or block.
        output_schema: A Pydantic model (or any Pydantic-compatible type). The function then
            returns a validated instance; invalid output is re-requested with the validation
            errors appended to the conversation.
        validation_retries: How many times to re-ask on invalid output (default 2).
        name: Function name used in telemetry (defaults to ``module.qualname``).
        tags: Extra key/value labels stored with every telemetry record.
        telemetry: ``False`` keeps this function's calls out of the telemetry store.
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

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(fn):
            raise TypeError("@callm must decorate a function")
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError(
                "@callm does not support generator functions; "
                "decorate the function that makes the LLM call instead"
            )
        runner = _Runner(fn, config)

        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            type(fn).__call__ if not inspect.isfunction(fn) else None
        ):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await runner.run_async(args, kwargs)

            async_wrapper.callm_config = config  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return runner.run_sync(args, kwargs)

        wrapper.callm_config = config  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


__all__ = ["callm", "detect_provider", "parse_result"]
