"""Provider fallback chain."""

from __future__ import annotations

import logging

from callm.classify import classify
from callm.errors import AllProvidersFailedError, ProviderNotAvailableError, is_control_error
from callm.pipeline import CallState, Handler, Step
from callm.providers.registry import get_provider
from callm.types import LLMRequest, LLMResponse, Target

logger = logging.getLogger("callm")


def default_fallback_on(exc: BaseException) -> bool:
    """Fail over on rate limits, overloads, 5xx, timeouts and connection errors."""
    return classify(exc).is_transient


def fallback_blocker(request: LLMRequest, target: Target) -> str | None:
    """Why ``request`` cannot be sent to ``target``, or ``None`` if it can."""
    try:
        get_provider(target.provider)
    except ProviderNotAvailableError as exc:
        return str(exc)
    if request.origin is None or target.provider == request.origin:
        return None  # same provider: native arguments are re-used as-is
    return get_provider(request.origin).portability_issue(request)


class FallbackMiddleware:
    name = "fallback"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        assert state.primary is not None
        primary = state.primary
        targets = [primary, *(t for t in state.config.fallback if t != primary)]
        if len(targets) == 1 or state.request.stream:
            state.target = primary
            return (yield from call_next(state))

        should_fallback = state.config.fallback_on or default_fallback_on
        failures: list[tuple[str, BaseException]] = []

        for index, target in enumerate(targets):
            if index > 0:
                blocker = fallback_blocker(state.request, target)
                if blocker is not None:
                    logger.warning("callm: skipping fallback %s: %s", target, blocker)
                    failures.append((target.label, ProviderNotAvailableError(blocker)))
                    continue
            state.target = target
            try:
                response = yield from call_next(state)
            except Exception as exc:
                if is_control_error(exc) or (index == 0 and not should_fallback(exc)):
                    raise
                failures.append((target.label, exc))
                if index + 1 < len(targets):
                    logger.warning(
                        "callm: %s failed (%s: %s); trying the next fallback",
                        target,
                        type(exc).__name__,
                        exc,
                    )
                continue
            if index > 0:
                state.record.fallback_from = primary.label
            return response

        raise AllProvidersFailedError(failures) from failures[-1][1]


__all__ = ["FallbackMiddleware", "default_fallback_on", "fallback_blocker"]
