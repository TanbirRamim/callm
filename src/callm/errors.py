"""Exceptions raised by callm.

Every exception callm raises on its own behalf derives from :class:`CallmError`.
Errors raised by provider SDKs (``openai.RateLimitError`` and friends) are passed
through unchanged unless a fallback chain was exhausted, in which case they are
collected on :class:`AllProvidersFailedError`.
"""

from __future__ import annotations

from collections.abc import Sequence


class CallmError(Exception):
    """Base class for all callm errors."""


class ConfigurationError(CallmError, ValueError):
    """Invalid configuration was passed to callm."""


class MissingDependencyError(CallmError, ImportError):
    """An optional dependency needed for a feature is not installed."""

    def __init__(self, feature: str, package: str, extra: str) -> None:
        self.feature = feature
        self.package = package
        self.extra = extra
        super().__init__(
            f"{feature} requires the '{package}' package. "
            f"Install it with: pip install 'callm-toolkit[{extra}]'"
        )


class ProviderNotAvailableError(CallmError):
    """A provider cannot serve the request (unknown provider, SDK missing, incompatible request)."""


class BudgetExceeded(CallmError):
    """A call was refused because it would exceed a cost limit.

    Raised *before* the request is sent to the provider.
    """

    def __init__(
        self,
        message: str,
        *,
        scope: str,
        limit: float,
        estimated: float,
        spent: float = 0.0,
    ) -> None:
        self.scope = scope
        self.limit = limit
        self.estimated = estimated
        self.spent = spent
        super().__init__(message)


#: PEP 8 style alias.
BudgetExceededError = BudgetExceeded


class SecurityError(CallmError):
    """Base class for input security violations."""


class PIIDetectedError(SecurityError):
    """PII was found in a request and the PII policy is ``action="block"``."""

    def __init__(self, entities: Sequence[str]) -> None:
        self.entities = sorted(set(entities))
        super().__init__(f"Request blocked: PII detected ({', '.join(self.entities)})")


class PromptInjectionError(SecurityError):
    """A likely prompt injection was found and the policy is ``action="block"``."""

    def __init__(self, score: float, matches: Sequence[str]) -> None:
        self.score = score
        self.matches = list(matches)
        detail = f": {', '.join(self.matches)}" if self.matches else ""
        super().__init__(f"Request blocked: prompt injection suspected (score={score:.2f}){detail}")


class OutputValidationError(CallmError):
    """The model output could not be validated against ``output_schema``."""

    def __init__(
        self, message: str, *, errors: Sequence[str], raw_text: str, attempts: int
    ) -> None:
        self.errors = list(errors)
        self.raw_text = raw_text
        self.attempts = attempts
        super().__init__(message)


class AllProvidersFailedError(CallmError):
    """The primary provider and every fallback failed.

    ``errors`` is a list of ``(target, exception)`` pairs in the order they were tried.
    """

    def __init__(self, errors: Sequence[tuple[str, BaseException]]) -> None:
        self.errors: list[tuple[str, BaseException]] = list(errors)
        lines = [f"  - {target}: {type(exc).__name__}: {exc}" for target, exc in self.errors]
        super().__init__("All providers failed:\n" + "\n".join(lines))

    @property
    def last_error(self) -> BaseException | None:
        return self.errors[-1][1] if self.errors else None


def is_control_error(exc: BaseException) -> bool:
    """Errors that represent a policy decision and must never be retried or failed over."""
    return isinstance(
        exc,
        (
            BudgetExceeded,
            SecurityError,
            ConfigurationError,
            OutputValidationError,
            MissingDependencyError,
        ),
    )


__all__ = [
    "AllProvidersFailedError",
    "BudgetExceeded",
    "BudgetExceededError",
    "CallmError",
    "ConfigurationError",
    "MissingDependencyError",
    "OutputValidationError",
    "PIIDetectedError",
    "PromptInjectionError",
    "ProviderNotAvailableError",
    "SecurityError",
    "is_control_error",
]
