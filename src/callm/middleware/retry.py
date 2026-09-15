"""Retry engine: exponential backoff with jitter, honouring provider ``retry-after`` hints."""

from __future__ import annotations

import logging

from callm.classify import should_retry
from callm.pipeline import CallState, Handler, Sleep, Step
from callm.types import LLMResponse

logger = logging.getLogger("callm")


class RetryMiddleware:
    name = "retry"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        config = state.config.retry
        attempt = 0
        while True:
            try:
                return (yield from call_next(state))
            except Exception as exc:
                retryable, info = should_retry(exc, config)
                if not retryable or attempt >= config.max_retries:
                    raise
                retry_after = info.retry_after if info is not None else None
                delay = config.delay_for(attempt, retry_after)
                target = state.target or state.request.target
                if delay is None:
                    logger.warning(
                        "callm: %s asked to wait %.0fs (> max_retry_after=%.0fs); not retrying",
                        target,
                        retry_after or 0.0,
                        config.max_retry_after,
                    )
                    raise
                attempt += 1
                state.record.retries += 1
                logger.info(
                    "callm: %s failed (%s%s); retry %d/%d in %.2fs",
                    target,
                    type(exc).__name__,
                    f" {info.status}" if info is not None and info.status else "",
                    attempt,
                    config.max_retries,
                    delay,
                )
                yield Sleep(delay)


__all__ = ["RetryMiddleware"]
