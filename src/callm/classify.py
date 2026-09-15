"""Classify provider exceptions: HTTP status, retryability and ``retry-after`` hints.

Works with the OpenAI, Anthropic and Google Gen AI SDKs, raw ``httpx``/``httpx2``/``requests``
errors and builtin network exceptions, without importing any of them.
"""

from __future__ import annotations

import email.utils
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from callm.config import DEFAULT_RETRY_STATUSES, RetryConfig
from callm.errors import is_control_error

_CONNECTION_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "RemoteProtocolError",
        "ReadError",
        "WriteError",
        "NetworkError",
        "ProtocolError",
    }
)


@dataclass(frozen=True)
class ErrorInfo:
    status: int | None
    retry_after: float | None
    is_timeout: bool
    is_connection: bool

    @property
    def is_transient(self) -> bool:
        """Rate limits, overloads, 5xx, timeouts and connection failures."""
        return (
            self.is_timeout
            or self.is_connection
            or (self.status is not None and self.status in DEFAULT_RETRY_STATUSES)
        )


def _status_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
        return value
    return None


def get_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "status"):
        status = _status_value(getattr(exc, attr, None))
        if status is not None:
            return status
    status = _status_value(getattr(exc, "code", None))  # google.genai.errors.APIError
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    for attr in ("status_code", "status"):
        status = _status_value(getattr(response, attr, None))
        if status is not None:
            return status
    return None


def get_headers(exc: BaseException) -> Mapping[str, str]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except Exception:
        return {}


_DURATION_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")


def parse_duration(value: str) -> float | None:
    """Parse ``"1.5"``, ``"250ms"``, ``"6m0s"`` or ``"30s"`` into seconds."""
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    matched = False
    position = 0
    for match in _DURATION_RE.finditer(text):
        if match.start() != position:
            return None
        matched = True
        position = match.end()
        number = float(match.group("value"))
        unit = match.group("unit")
        total += {"ms": number / 1000, "s": number, "m": number * 60, "h": number * 3600}[unit]
    if not matched or position != len(text):
        return None
    return total


def _parse_http_date(value: str) -> float | None:
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:  # pragma: no cover - older Pythons
        return None
    return max(parsed.timestamp() - time.time(), 0.0)


def _parse_rfc3339(value: str) -> float | None:
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.strip().replace("Z", "+00:00"))
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return max(parsed.timestamp() - time.time(), 0.0)


def _rate_limit_reset(headers: Mapping[str, str]) -> float | None:
    """Time until the *exhausted* rate limit resets.

    Providers report a reset time for every limit; only the ones whose ``remaining``
    counter reached zero explain the 429. When none is visibly exhausted, the soonest
    reset is used.
    """
    exhausted: list[float] = []
    every: list[float] = []
    for key, value in headers.items():
        if key.startswith("x-ratelimit-reset-"):  # OpenAI: durations like "6m0s"
            parsed = parse_duration(value)
            remaining_key = "x-ratelimit-remaining-" + key.removeprefix("x-ratelimit-reset-")
        elif key.startswith("anthropic-ratelimit-") and key.endswith("-reset"):  # RFC 3339
            parsed = _parse_rfc3339(value)
            remaining_key = key.removesuffix("-reset") + "-remaining"
        else:
            continue
        if parsed is None:
            continue
        every.append(parsed)
        if headers.get(remaining_key, "").strip() == "0":
            exhausted.append(parsed)
    if exhausted:
        return max(exhausted)
    if every:
        return min(every)
    return None


def _google_retry_delay(exc: BaseException) -> float | None:
    details: Any = getattr(exc, "details", None)
    if details is None:
        body = getattr(exc, "response_json", None)
        if isinstance(body, (str, bytes)):
            try:
                body = json.loads(body)
            except ValueError:
                body = None
        details = body
    if not isinstance(details, Mapping):
        return None
    error = details.get("error", details)
    items = error.get("details") if isinstance(error, Mapping) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, Mapping) and str(item.get("@type", "")).endswith("RetryInfo"):
            delay = item.get("retryDelay")
            if isinstance(delay, str):
                return parse_duration(delay)
    return None


def get_retry_after(exc: BaseException) -> float | None:
    """Server-provided wait time in seconds, if any.

    Understands ``retry-after-ms`` (OpenAI), ``retry-after`` (seconds or HTTP date,
    all providers), ``x-ratelimit-reset-*`` durations (OpenAI) and
    ``anthropic-ratelimit-*-reset`` timestamps (Anthropic) on 429 responses, and the
    ``RetryInfo.retryDelay`` detail in Google API error bodies.
    """
    headers = get_headers(exc)

    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(float(ms) / 1000.0, 0.0)
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        seconds = parse_duration(retry_after)
        if seconds is None:
            seconds = _parse_http_date(retry_after)
        if seconds is not None:
            return max(seconds, 0.0)

    if get_status(exc) == 429:
        reset = _rate_limit_reset(headers)
        if reset is not None:
            return reset

    return _google_retry_delay(exc)


def classify(exc: BaseException) -> ErrorInfo:
    names = [cls.__name__ for cls in type(exc).__mro__]
    status = get_status(exc)
    is_timeout = isinstance(exc, TimeoutError) or any("Timeout" in name for name in names)
    is_connection = isinstance(exc, ConnectionError) or any(
        name in _CONNECTION_ERROR_NAMES for name in names
    )
    if status is not None:
        # A real HTTP status wins over generic connection-error base classes.
        is_connection = False
    return ErrorInfo(
        status=status,
        retry_after=get_retry_after(exc),
        is_timeout=is_timeout,
        is_connection=is_connection,
    )


def should_retry(exc: BaseException, config: RetryConfig) -> tuple[bool, ErrorInfo | None]:
    if is_control_error(exc) or not isinstance(exc, Exception):
        return False, None
    info = classify(exc)
    if info.status is not None:
        return info.status in config.retry_on_status, info
    if info.is_timeout:
        return config.retry_on_timeout, info
    if info.is_connection:
        return config.retry_on_connection_error, info
    return False, info


__all__ = [
    "ErrorInfo",
    "classify",
    "get_headers",
    "get_retry_after",
    "get_status",
    "parse_duration",
    "should_retry",
]
