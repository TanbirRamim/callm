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
import weakref
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
    #: True for SDK methods callm does not intercept yet: inside @callm they only log a
    #: one-time warning, so nobody assumes PII masking or budgets apply when they do not.
    warn_only: bool = False


PATCH_TARGETS: tuple[PatchTarget, ...] = (
    PatchTarget("openai", "openai.resources.chat.completions", "Completions", "create", False),
    PatchTarget("openai", "openai.resources.chat.completions", "AsyncCompletions", "create", True),
    PatchTarget("openai", "openai.resources.chat.completions", "Completions", "parse", False),
    PatchTarget("openai", "openai.resources.chat.completions", "AsyncCompletions", "parse", True),
    PatchTarget("openai-responses", "openai.resources.responses", "Responses", "create", False),
    PatchTarget("openai-responses", "openai.resources.responses", "AsyncResponses", "create", True),
    PatchTarget("anthropic", "anthropic.resources.messages", "Messages", "create", False),
    PatchTarget("anthropic", "anthropic.resources.messages", "AsyncMessages", "create", True),
    PatchTarget("google", "google.genai.models", "Models", "generate_content", False),
    PatchTarget("google", "google.genai.models", "AsyncModels", "generate_content", True),
    *(
        PatchTarget(provider, module, cls, method, cls.startswith("Async"), warn_only=True)
        for provider, module, classes, methods in (
            (
                "openai-responses",
                "openai.resources.responses",
                ("Responses", "AsyncResponses"),
                ("parse", "stream"),
            ),
            (
                "openai",
                "openai.resources.chat.completions",
                ("Completions", "AsyncCompletions"),
                ("stream",),
            ),
            (
                "openai",
                "openai.resources.beta.chat.completions",
                ("Completions", "AsyncCompletions"),
                ("parse", "stream"),
            ),
            (
                "anthropic",
                "anthropic.resources.messages",
                ("Messages", "AsyncMessages"),
                ("stream",),
            ),
        )
        for cls in classes
        for method in methods
    ),
)
_ROOT_PACKAGES = {
    "openai": "openai",
    "openai-responses": "openai",
    "anthropic": "anthropic",
    "google": "google.genai",
}
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


_DEFAULT_ENDPOINTS = frozenset(
    {
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
    }
)


def client_endpoint(resource: Any) -> str | None:
    """Base URL of the SDK client behind ``resource``, or None for the provider default.

    Part of the cache key, so an Azure, vLLM, Ollama or staging client never gets answers
    cached from another server that happens to use the same model name.
    """
    try:
        client = getattr(resource, "_client", None)  # openai, anthropic
        url = getattr(client, "base_url", None)
        if url is None:  # google-genai
            api_client = getattr(resource, "_api_client", None)
            url = getattr(getattr(api_client, "_http_options", None), "base_url", None)
            if getattr(api_client, "vertexai", False):
                project = getattr(api_client, "project", None)
                location = getattr(api_client, "location", None)
                return f"vertexai:{project}:{location}:{url or ''}"
    except Exception:
        return None
    if url is None:
        return None
    endpoint = str(url).rstrip("/")
    return None if endpoint in _DEFAULT_ENDPOINTS else endpoint


# Where each patched resource lives on its SDK client, to rebuild it on a client copy.
_RESOURCE_PATHS: dict[str, tuple[str, ...]] = {
    "openai": ("chat", "completions"),
    "openai-responses": ("responses",),
    "anthropic": ("messages",),
}
_no_retry_resources: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_no_retry_lock = threading.Lock()


def _without_sdk_retries(resource: Any, target: PatchTarget, config: CallConfig) -> Any:
    """The same resource on a copy of its client with the SDK's own retries switched off.

    The OpenAI and Anthropic SDKs retry twice by default. When callm retries or falls
    back itself, those hidden retries would multiply its attempts (3 x 3 = 9 requests)
    and delay fallbacks, so callm takes over. Without callm retries or fallback, the
    SDK keeps its behaviour.
    """
    if config.retry.max_retries == 0 and not config.fallback:
        return resource
    path = _RESOURCE_PATHS.get(target.provider)
    client = getattr(resource, "_client", None)
    if path is None or client is None or getattr(client, "max_retries", 0) == 0:
        return resource
    with _no_retry_lock:
        cached = _no_retry_resources.get(resource)
        if cached is not None:
            return cached
        try:
            rebuilt: Any = client.with_options(max_retries=0)
            for attribute in path:
                rebuilt = getattr(rebuilt, attribute)
        except Exception:
            logger.debug("callm: could not disable SDK retries", exc_info=True)
            return resource
        if type(rebuilt) is not type(resource):
            return resource
        _no_retry_resources[resource] = rebuilt
        return rebuilt


def _prepare(
    ctx: ExecutionContext,
    target: PatchTarget,
    kwargs: dict[str, Any],
    mode: str,
    call_sync: Callable[[dict[str, Any]], Any] | None,
    call_async: Callable[[dict[str, Any]], Awaitable[Any]] | None,
    endpoint: str | None = None,
) -> CallState | None:
    try:
        request = get_provider(target.provider).parse_native_request(clean_kwargs(kwargs))
        if endpoint:
            request.native_extra["__endpoint__"] = endpoint
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
    native = get_provider(target.provider).native_for(response)
    if target.provider == "openai" and target.method == "parse":
        native = _as_parsed_completion(native, state.request.native_extra)
    return native


def _as_parsed_completion(native: Any, extra: dict[str, Any]) -> Any:
    """The ``ParsedChatCompletion`` that ``chat.completions.parse`` promises its caller.

    A live call already returns one. A cache hit, or a response synthesized after a
    fallback, is a plain ``ChatCompletion``, so run it through the SDK's own parser
    with the caller's ``response_format`` and ``tools``.
    """
    if type(native).__name__.startswith("Parsed"):
        return native
    try:
        import openai
        from openai.lib._parsing import parse_chat_completion
    except ImportError:
        return native
    not_given = getattr(openai, "omit", None) or getattr(openai, "NOT_GIVEN", None)
    try:
        return parse_chat_completion(
            response_format=extra.get("response_format", not_given),
            input_tools=extra.get("tools", not_given),
            chat_completion=native,
        )
    except (TypeError, AttributeError):  # an AttrDict without the SDK types
        logger.debug("callm: could not rebuild a ParsedChatCompletion", exc_info=True)
        return native


_warned_methods: set[str] = set()
_SUPPORTED_ALTERNATIVE = {
    "anthropic": "messages.create",
    "openai-responses": "responses.create",
}


def _unprotected_features(config: CallConfig) -> str:
    features = []
    if config.pii is not None:
        features.append("PII masking")
    if config.injection is not None:
        features.append("injection checks")
    if config.max_cost is not None or config.budgets:
        features.append("budgets")
    if config.cache is not None:
        features.append("caching")
    if config.output_schema is not None:
        features.append("output validation")
    if config.fallback:
        features.append("fallback")
    if config.retry.max_retries:
        features.append("callm retries")
    features.append("cost tracking and telemetry")
    return ", ".join(features)


def _make_warning_wrapper(target: PatchTarget, original: Callable[..., Any]) -> Callable[..., Any]:
    """Pass-through wrapper for SDK methods callm does not intercept yet.

    Plain (non-async) on purpose: async originals return their coroutine or stream manager
    unchanged, so behaviour and return types stay exactly those of the SDK.
    """
    name = f"{target.provider} {target.cls}.{target.method}"
    label = (
        "responses." + target.method
        if "responses" in target.module
        else ("messages." if target.provider == "anthropic" else "chat.completions.")
        + target.method
    )

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ACTIVE.get()
        if ctx is not None and not BYPASS.get() and name not in _warned_methods:
            _warned_methods.add(name)
            logger.warning(
                "callm: %s (%s) is not intercepted by callm yet, so inside %s it runs without %s. "
                "Use %s for now: https://tanbirramim.github.io/callm/faq/",
                label,
                target.provider,
                ctx.name or "@callm",
                _unprotected_features(ctx.config),
                _SUPPORTED_ALTERNATIVE.get(target.provider, "chat.completions.create"),
            )
        return original(self, *args, **kwargs)

    wrapper.__callm_original__ = original  # type: ignore[attr-defined]
    return wrapper


def _make_sync_wrapper(target: PatchTarget, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = ACTIVE.get()
        if ctx is None or BYPASS.get() or args or _is_raw_response_call(kwargs):
            return original(self, *args, **kwargs)
        sdk = _without_sdk_retries(self, target, ctx.config)
        state = _prepare(
            ctx,
            target,
            kwargs,
            "sync",
            lambda kw: original(sdk, **kw),
            None,
            client_endpoint(self),
        )
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

        sdk = _without_sdk_retries(self, target, ctx.config)

        async def call_async(kw: dict[str, Any]) -> Any:
            return await original(sdk, **kw)

        state = _prepare(ctx, target, kwargs, "async", None, call_async, client_endpoint(self))
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
            if target.warn_only:
                make = _make_warning_wrapper
            else:
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
