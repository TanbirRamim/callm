"""Transparent interception of provider SDK calls.

``@callm`` works on ordinary functions that call the official SDKs directly. While a
decorated function (or a ``with shield(...)`` block) is running, calls to

* ``openai``: ``chat.completions.create`` (sync and async clients)
* ``anthropic``: ``messages.create`` (sync and async clients)
* ``google-genai``: ``models.generate_content`` (sync and ``client.aio``)

are routed through the callm middleware pipeline. Outside of a callm scope the patched
methods call straight through to the original implementation, so importing callm never
changes the behaviour of code that does not use it.

The active scope lives in a :class:`contextvars.ContextVar`: it follows ``async`` tasks
automatically, but *not* new threads. Run thread-pool work with
``contextvars.copy_context().run`` or decorate the function that runs in the thread.
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import logging
import sys
import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from callm.config import CallConfig
from callm.pipeline import BYPASS, CallState, Handler, NativeBinding, run_async, run_sync
from callm.providers.registry import get_provider
from callm.types import CallRecord, LLMResponse

logger = logging.getLogger("callm")


@dataclass
class ExecutionContext:
    """The callm scope that intercepted SDK calls run under."""

    config: CallConfig
    handler: Handler
    name: str | None = None
    parent: ExecutionContext | None = None
    calls: int = 0
    owner: Any = None

    def mark_call(self) -> None:
        ctx: ExecutionContext | None = self
        while ctx is not None:
            ctx.calls += 1
            ctx = ctx.parent


ACTIVE: ContextVar[ExecutionContext | None] = ContextVar("callm_active_context", default=None)


def current_context() -> ExecutionContext | None:
    return ACTIVE.get()


def enter_scope(
    config: CallConfig, handler: Handler, name: str | None, owner: Any = None
) -> ExecutionContext:
    """Create a scope nested in the active one, inheriting its protections."""
    from callm.config import merge_scopes
    from callm.middleware import build_handler

    parent = ACTIVE.get()
    if parent is not None:
        merged = merge_scopes(parent.config, config)
        if merged is not config:
            config, handler = merged, build_handler(merged)
    return ExecutionContext(config=config, handler=handler, name=name, parent=parent, owner=owner)


def _is_sentinel(value: Any) -> bool:
    """``openai.NOT_GIVEN`` / ``anthropic.NOT_GIVEN`` / ``Omit`` mean "argument not passed"."""
    return type(value).__name__ in ("NotGiven", "Omit")


# --------------------------------------------------------------------------- patch targets


@dataclass(frozen=True)
class PatchTarget:
    provider: str
    module: str
    cls: str
    method: str
    is_async: bool


PATCH_TARGETS: tuple[PatchTarget, ...] = (
    PatchTarget("openai", "openai.resources.chat.completions", "Completions", "create", False),
    PatchTarget("openai", "openai.resources.chat.completions", "AsyncCompletions", "create", True),
    PatchTarget("anthropic", "anthropic.resources.messages", "Messages", "create", False),
    PatchTarget("anthropic", "anthropic.resources.messages", "AsyncMessages", "create", True),
    PatchTarget("google", "google.genai.models", "Models", "generate_content", False),
    PatchTarget("google", "google.genai.models", "AsyncModels", "generate_content", True),
)
_ROOT_PACKAGES = {"openai": "openai", "anthropic": "anthropic", "google": "google.genai"}
_RAW_RESPONSE_HEADERS = frozenset({"x-stainless-raw-response", "x-stainless-streamed-raw-response"})

_lock = threading.RLock()
_patched: dict[PatchTarget, Callable[..., Any]] = {}
_hook: _PostImportHook | None = None


def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
    """``client.x.with_raw_response.create(...)`` must bypass callm (it expects raw HTTP)."""
    headers = kwargs.get("extra_headers")
    if not headers:
        return False
    try:
        return any(str(key).lower() in _RAW_RESPONSE_HEADERS for key in headers)
    except TypeError:
        return False


def _prepare(
    ctx: ExecutionContext,
    target: PatchTarget,
    kwargs: dict[str, Any],
    mode: str,
    call_sync: Callable[[dict[str, Any]], Any] | None,
    call_async: Callable[[dict[str, Any]], Awaitable[Any]] | None,
) -> CallState | None:
    try:
        request = get_provider(target.provider).parse_native_request(clean_kwargs(kwargs))
    except Exception as exc:
        if ctx.config.pii is not None or ctx.config.injection is not None:
            raise  # never send a request we could not inspect when security is enabled
        logger.warning(
            "callm: could not interpret %s.%s arguments (%s); calling the SDK directly",
            target.cls,
            target.method,
            exc,
        )
        return None
    ctx.mark_call()
    record = CallRecord(function=ctx.name, provider=request.provider, model=request.model)
    return CallState(
        request=request,
        config=ctx.config,
        record=record,
        mode=mode,
        native=NativeBinding(target.provider, call_sync, call_async),
    )


def clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if not _is_sentinel(value)}


def _finish(target: PatchTarget, state: CallState, response: LLMResponse) -> Any:
    if state.request.stream:
        return response.raw
    return get_provider(target.provider).native_for(response)


def _make_sync_wrapper(target: PatchTarget, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ACTIVE.get()
        if ctx is None or BYPASS.get() or args or _is_raw_response_call(kwargs):
            return original(self, *args, **kwargs)
        state = _prepare(ctx, target, kwargs, "sync", lambda kw: original(self, **kw), None)
        if state is None:
            return original(self, *args, **kwargs)
        response = run_sync(ctx.handler(state))
        return _finish(target, state, response)

    wrapper.__callm_original__ = original  # type: ignore[attr-defined]
    return wrapper


def _make_async_wrapper(target: PatchTarget, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ACTIVE.get()
        if ctx is None or BYPASS.get() or args or _is_raw_response_call(kwargs):
            return await original(self, *args, **kwargs)

        async def call_async(kw: dict[str, Any]) -> Any:
            return await original(self, **kw)

        state = _prepare(ctx, target, kwargs, "async", None, call_async)
        if state is None:
            return await original(self, *args, **kwargs)
        response = await run_async(ctx.handler(state))
        return _finish(target, state, response)

    wrapper.__callm_original__ = original  # type: ignore[attr-defined]
    return wrapper


def _patch_module(module: ModuleType, provider: str) -> None:
    with _lock:
        for target in PATCH_TARGETS:
            if target in _patched or target.provider != provider:
                continue
            if target.module != module.__name__:
                continue
            cls = getattr(module, target.cls, None)
            original = getattr(cls, target.method, None) if cls is not None else None
            if cls is None or original is None:
                logger.debug("callm: %s.%s not found; skipping", target.module, target.cls)
                continue
            existing = getattr(original, "__callm_original__", None)
            if existing is not None:  # already instrumented (e.g. module reloaded)
                _patched[target] = existing
                continue
            make = _make_async_wrapper if target.is_async else _make_sync_wrapper
            setattr(cls, target.method, make(target, original))
            _patched[target] = original
            logger.debug("callm: instrumented %s.%s.%s", target.module, target.cls, target.method)


def _try_patch_loaded() -> None:
    for target in PATCH_TARGETS:
        if target in _patched or _ROOT_PACKAGES[target.provider] not in sys.modules:
            continue
        try:
            module = importlib.import_module(target.module)
        except Exception:
            logger.debug("callm: cannot import %s", target.module, exc_info=True)
            continue
        _patch_module(module, target.provider)


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Instruments SDK modules that are imported after callm started intercepting."""

    _watched = frozenset(target.module for target in PATCH_TARGETS)

    def __init__(self) -> None:
        self._local = threading.local()

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self._watched or getattr(self._local, "busy", False):
            return None
        self._local.busy = True
        spec: importlib.machinery.ModuleSpec | None = None
        try:
            for finder in list(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._local.busy = False
        loader = spec.loader if spec is not None else None
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        original_exec = loader.exec_module

        def exec_module(module: ModuleType) -> None:
            original_exec(module)
            try:
                _try_patch_loaded()
            except Exception:
                logger.debug("callm: post-import instrumentation failed", exc_info=True)

        try:
            setattr(loader, "exec_module", exec_module)  # noqa: B010
        except (AttributeError, TypeError):
            pass
        return spec


def ensure_installed() -> None:
    """Instrument already-imported SDKs and watch for later imports. Idempotent and cheap."""
    global _hook
    if _hook is not None and len(_patched) == len(PATCH_TARGETS):
        return
    with _lock:
        _try_patch_loaded()
        if _hook is None:
            _hook = _PostImportHook()
            sys.meta_path.insert(0, _hook)


def uninstall() -> None:
    """Remove all instrumentation (mainly for tests)."""
    global _hook
    with _lock:
        for target, original in list(_patched.items()):
            module = sys.modules.get(target.module)
            cls = getattr(module, target.cls, None) if module is not None else None
            if cls is not None:
                setattr(cls, target.method, original)
            del _patched[target]
        if _hook is not None:
            try:
                sys.meta_path.remove(_hook)
            except ValueError:
                pass
            _hook = None


def instrumented() -> list[str]:
    """Dotted names of the SDK methods currently instrumented."""
    return sorted(f"{t.module}.{t.cls}.{t.method}" for t in _patched)


__all__ = [
    "ACTIVE",
    "PATCH_TARGETS",
    "ExecutionContext",
    "current_context",
    "ensure_installed",
    "instrumented",
    "uninstall",
]
