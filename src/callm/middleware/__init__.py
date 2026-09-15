"""The callm middleware stack.

Order (outermost first)::

    1. Telemetry   - records cost, latency, tokens and flags for every call
    2. Security    - injection detection and PII redaction (before caching or sending)
    3. Cache       - returns a cached response on a hit
    4. Validator   - parses into output_schema, re-asks with the errors on failure
    5. Fallback    - moves to the next provider/model when one keeps failing
    6. Cost guard  - refuses calls that would exceed max_cost or a budget
    7. Retry       - exponential backoff on 429 / 5xx / timeouts
    8. Transport   - the actual provider API call
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from callm.middleware.cache import CacheMiddleware
from callm.middleware.cost import CostGuardMiddleware
from callm.middleware.fallback import FallbackMiddleware
from callm.middleware.retry import RetryMiddleware
from callm.middleware.security import SecurityMiddleware
from callm.middleware.telemetry import TelemetryMiddleware
from callm.middleware.validator import ValidatorMiddleware
from callm.pipeline import (
    Handler,
    Invoke,
    Middleware,
    Step,
    acall_bypassed,
    call_bypassed,
    compose,
)
from callm.pricing import cost_for
from callm.providers.registry import get_provider
from callm.types import LLMResponse

if TYPE_CHECKING:
    from callm.config import CallConfig
    from callm.pipeline import CallState


def _async_only() -> Any:  # pragma: no cover - guarded by construction
    raise RuntimeError("callm: an async-only provider call was driven synchronously")


def transport(state: CallState) -> Step[LLMResponse]:
    """Send the request to ``state.target`` and return a canonical response."""
    target = state.target or state.request.target
    request = state.request if target == state.request.target else state.request.retarget(target)
    provider = get_provider(target.provider)
    native = state.native

    if native is not None and native.provider == target.provider:
        # Same provider as the intercepted SDK call: reuse the caller's own client.
        kwargs = provider.to_kwargs(request)
        call_sync, call_async = native.call_sync, native.call_async
        effect = Invoke(
            sync=(lambda: call_bypassed(call_sync, kwargs)) if call_sync else _async_only,
            async_=(lambda: acall_bypassed(call_async, kwargs)) if call_async else None,
        )
    elif state.mode == "async":
        effect = Invoke(sync=_async_only, async_=lambda: provider.call_async(request))
    else:
        effect = Invoke(sync=lambda: provider.call_sync(request))

    started = time.perf_counter()
    native_response = yield effect
    elapsed_ms = (time.perf_counter() - started) * 1000
    if request.stream:
        return LLMResponse(
            text="",
            provider=target.provider,
            model=target.model,
            raw=native_response,
            latency_ms=elapsed_ms,
        )
    response = provider.parse_response(native_response, request)
    response.latency_ms = elapsed_ms
    response.cost = cost_for(
        target.provider, response.model or target.model, response.usage, warn=False
    )
    if response.cost is None:
        response.cost = cost_for(target.provider, target.model, response.usage)
    return response


MIDDLEWARE_ORDER: tuple[type[Middleware], ...] = (
    TelemetryMiddleware,
    SecurityMiddleware,
    CacheMiddleware,
    ValidatorMiddleware,
    FallbackMiddleware,
    CostGuardMiddleware,
    RetryMiddleware,
)


def build_handler(config: CallConfig) -> Handler:
    """Compose the middleware stack, skipping layers the config does not use."""
    layers: list[Middleware] = [TelemetryMiddleware()]
    if config.pii is not None or config.injection is not None:
        layers.append(SecurityMiddleware())
    if config.cache is not None:
        layers.append(CacheMiddleware())
    if config.output_schema is not None:
        layers.append(ValidatorMiddleware())
    layers.append(FallbackMiddleware())
    layers.append(CostGuardMiddleware())
    if config.retry.max_retries > 0:
        layers.append(RetryMiddleware())
    return compose(layers, transport)


__all__ = [
    "MIDDLEWARE_ORDER",
    "CacheMiddleware",
    "CostGuardMiddleware",
    "FallbackMiddleware",
    "RetryMiddleware",
    "SecurityMiddleware",
    "TelemetryMiddleware",
    "ValidatorMiddleware",
    "build_handler",
    "transport",
]
