"""The middleware pipeline.

Middleware is written once as *generators* that ``yield`` effects (sleep, perform I/O) and
receive the result back. A small driver runs the generator either synchronously
(``time.sleep``, direct calls) or asynchronously (``asyncio.sleep``, ``await``). This keeps
sync and async behaviour identical without duplicating any middleware logic.

The chain is a classic chain-of-responsibility::

    Telemetry -> Security -> Cache -> Validator -> Fallback -> CostGuard -> Retry -> Transport
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from callm.config import CallConfig
    from callm.types import CallRecord, LLMRequest, LLMResponse, Target

T = TypeVar("T")


# --------------------------------------------------------------------------- effects


@dataclass(frozen=True)
class Sleep:
    seconds: float


@dataclass(frozen=True)
class Invoke:
    """Perform a (possibly blocking) operation.

    ``sync`` is used by the sync driver. The async driver awaits ``async_`` when given,
    otherwise runs ``sync`` inline, or in a worker thread when ``offload`` is true.
    """

    sync: Callable[[], Any]
    async_: Callable[[], Awaitable[Any]] | None = None
    offload: bool = False


Effect = Sleep | Invoke
Step = Generator[Effect, Any, T]


def sleep_sync(seconds: float) -> None:
    """Blocking sleep used for backoff (module-level so tests can replace it)."""
    time.sleep(seconds)


async def sleep_async(seconds: float) -> None:
    """Non-blocking sleep used for backoff (module-level so tests can replace it)."""
    await asyncio.sleep(seconds)


def run_sync(gen: Step[T]) -> T:
    """Drive a pipeline generator synchronously."""
    send_value: Any = None
    error: BaseException | None = None
    try:
        while True:
            try:
                effect = gen.throw(error) if error is not None else gen.send(send_value)
            except StopIteration as stop:
                result: T = stop.value
                return result
            send_value, error = None, None
            try:
                if isinstance(effect, Sleep):
                    if effect.seconds > 0:
                        sleep_sync(effect.seconds)
                elif isinstance(effect, Invoke):
                    send_value = effect.sync()
                else:  # pragma: no cover - defensive
                    raise TypeError(f"unknown pipeline effect: {effect!r}")
            except BaseException as exc:
                error = exc
    finally:
        gen.close()


async def run_async(gen: Step[T]) -> T:
    """Drive a pipeline generator on the running event loop."""
    send_value: Any = None
    error: BaseException | None = None
    try:
        while True:
            try:
                effect = gen.throw(error) if error is not None else gen.send(send_value)
            except StopIteration as stop:
                result: T = stop.value
                return result
            send_value, error = None, None
            try:
                if isinstance(effect, Sleep):
                    if effect.seconds > 0:
                        await sleep_async(effect.seconds)
                elif isinstance(effect, Invoke):
                    if effect.async_ is not None:
                        send_value = await effect.async_()
                    elif effect.offload:
                        send_value = await asyncio.to_thread(effect.sync)
                    else:
                        send_value = effect.sync()
                    if inspect.isawaitable(send_value):
                        send_value = await send_value
                else:  # pragma: no cover - defensive
                    raise TypeError(f"unknown pipeline effect: {effect!r}")
            except BaseException as exc:
                error = exc
    finally:
        gen.close()


# --------------------------------------------------------------------------- state


@dataclass
class NativeBinding:
    """The intercepted SDK method for the provider the user called directly."""

    provider: str
    call_sync: Callable[[dict[str, Any]], Any] | None = None
    call_async: Callable[[dict[str, Any]], Awaitable[Any]] | None = None


@dataclass
class CallState:
    request: LLMRequest
    config: CallConfig
    record: CallRecord
    mode: str = "sync"  # "sync" | "async"
    native: NativeBinding | None = None
    target: Target | None = None
    primary: Target | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary is None:
            self.primary = self.request.target


Handler = Callable[[CallState], Step["LLMResponse"]]


class Middleware(Protocol):
    name: str

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]: ...


def compose(middlewares: Sequence[Middleware], transport: Handler) -> Handler:
    """Compose middlewares (outermost first) around ``transport``."""
    handler = transport
    for middleware in reversed(middlewares):
        handler = _bind(middleware, handler)
    return handler


def _bind(middleware: Middleware, call_next: Handler) -> Handler:
    def handler(state: CallState) -> Step[LLMResponse]:
        return middleware.handle(state, call_next)

    handler.__name__ = f"{middleware.name}_handler"
    return handler


# --------------------------------------------------------------------------- interception bypass

#: True while callm itself is calling a provider SDK, so interception passes through.
BYPASS: ContextVar[bool] = ContextVar("callm_bypass", default=False)


def call_bypassed(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    token = BYPASS.set(True)
    try:
        return fn(*args, **kwargs)
    finally:
        BYPASS.reset(token)


async def acall_bypassed(fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    token = BYPASS.set(True)
    try:
        return await fn(*args, **kwargs)
    finally:
        BYPASS.reset(token)


__all__ = [
    "BYPASS",
    "CallState",
    "Effect",
    "Handler",
    "Invoke",
    "Middleware",
    "NativeBinding",
    "Sleep",
    "Step",
    "acall_bypassed",
    "call_bypassed",
    "compose",
    "run_async",
    "run_sync",
    "sleep_async",
    "sleep_sync",
]
