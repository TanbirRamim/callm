"""Global settings and per-call configuration."""

from __future__ import annotations

import logging
import os
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from callm.errors import ConfigurationError

if TYPE_CHECKING:
    from callm.budgets import Budget
    from callm.embeddings import Embedder
    from callm.storage.base import CacheStore, TelemetryStore
    from callm.types import CallRecord, Target

logger = logging.getLogger("callm")

#: HTTP statuses that are retried by default. 529 is Anthropic's "overloaded".
DEFAULT_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


# --------------------------------------------------------------------------- per-feature


@dataclass(frozen=True)
class RetryConfig:
    """Exponential backoff with full jitter.

    ``max_retries`` is the number of *additional* attempts after the first one, so
    ``RetryConfig(max_retries=3)`` makes at most four requests.
    """

    max_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True
    retry_on_status: frozenset[int] = DEFAULT_RETRY_STATUSES
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    respect_retry_after: bool = True
    #: A ``retry-after`` longer than this aborts retrying (and moves on to a fallback).
    max_retry_after: float = 60.0

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ConfigurationError("retry: max_retries must be an int >= 0")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ConfigurationError("retry: delays must be >= 0")
        object.__setattr__(self, "retry_on_status", frozenset(self.retry_on_status))

    def backoff(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (0-based), ignoring ``retry-after``."""
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return random.uniform(0, ceiling) if self.jitter else ceiling

    def delay_for(self, attempt: int, retry_after: float | None) -> float | None:
        """Delay to sleep, or ``None`` when the server asked us to wait too long."""
        if retry_after is not None and self.respect_retry_after:
            if retry_after > self.max_retry_after:
                return None
            return max(retry_after, 0.0)
        return self.backoff(attempt)


@dataclass(frozen=True)
class CacheConfig:
    """Response cache settings.

    By default the cache is an *exact-match* cache: two requests hit the same entry only
    when provider, model, every message and every generation parameter are identical.

    ``semantic=True`` additionally matches requests whose final user message is
    semantically similar (cosine similarity >= ``threshold``) to a cached one, provided
    everything else (system prompt, earlier turns, parameters) is identical. Semantic
    matching needs an embedder (``pip install 'callm[cache]'`` or pass ``embedder=``).
    """

    ttl: float | None = None
    semantic: bool = False
    threshold: float = 0.95
    embedder: Embedder | None = None
    store: CacheStore | None = None
    namespace: str = "default"
    #: Maximum number of candidates compared per semantic lookup.
    max_candidates: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ConfigurationError("cache: threshold must be in (0, 1]")
        if self.ttl is not None and self.ttl <= 0:
            raise ConfigurationError("cache: ttl must be positive (seconds) or None")
        if self.max_candidates < 1:
            raise ConfigurationError("cache: max_candidates must be >= 1")


PIIEntity = Literal["email", "phone", "ssn", "credit_card", "ip_address", "iban", "person"]
DEFAULT_PII_ENTITIES: tuple[str, ...] = (
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ip_address",
    "iban",
)
KNOWN_PII_ENTITIES = frozenset({*DEFAULT_PII_ENTITIES, "person"})


@dataclass(frozen=True)
class PIIConfig:
    """PII redaction settings.

    ``action="mask"`` replaces each finding with a numbered placeholder such as
    ``[EMAIL_1]`` before the request leaves the process. ``action="block"`` raises
    :class:`~callm.errors.PIIDetectedError` instead.

    ``ner=True`` (or including ``"person"`` in ``entities``) adds person-name detection
    with spaCy (``pip install 'callm[security]'`` and
    ``python -m spacy download en_core_web_sm``).
    """

    entities: tuple[str, ...] = DEFAULT_PII_ENTITIES
    action: Literal["mask", "block"] = "mask"
    ner: bool = False
    ner_model: str = "en_core_web_sm"
    roles: tuple[str, ...] = ("system", "user", "assistant", "tool")

    def __post_init__(self) -> None:
        if self.action not in ("mask", "block"):
            raise ConfigurationError("block_pii: action must be 'mask' or 'block'")
        unknown = set(self.entities) - KNOWN_PII_ENTITIES
        if unknown:
            raise ConfigurationError(f"block_pii: unknown entities {sorted(unknown)}")
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "roles", tuple(self.roles))


@dataclass(frozen=True)
class InjectionConfig:
    """Prompt injection detection settings.

    The default detector is a fast, dependency-free heuristic scorer. Set ``classifier``
    to a Hugging Face text-classification model id (for example
    ``"protectai/deberta-v3-base-prompt-injection-v2"``) to also run a local ML
    classifier (requires ``transformers``).

    ``action="flag"`` logs a warning and records the score in telemetry;
    ``action="block"`` raises :class:`~callm.errors.PromptInjectionError`.
    """

    threshold: float = 0.5
    action: Literal["flag", "block"] = "flag"
    roles: tuple[str, ...] = ("user", "tool")
    classifier: str | None = None
    classifier_label: str = "INJECTION"

    def __post_init__(self) -> None:
        if self.action not in ("flag", "block"):
            raise ConfigurationError("detect_injection: action must be 'flag' or 'block'")
        if not 0.0 < self.threshold <= 1.0:
            raise ConfigurationError("detect_injection: threshold must be in (0, 1]")
        object.__setattr__(self, "roles", tuple(self.roles))


# --------------------------------------------------------------------------- call config


@dataclass(frozen=True)
class CallConfig:
    """Fully resolved configuration for one decorated function / shield / complete call."""

    cache: CacheConfig | None = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    fallback: tuple[Target, ...] = ()
    fallback_on: Callable[[BaseException], bool] | None = None
    max_cost: float | None = None
    budget: Budget | None = None
    pii: PIIConfig | None = None
    injection: InjectionConfig | None = None
    output_schema: Any = None
    validation_retries: int = 2
    name: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    telemetry: bool = True
    #: Budgets inherited from enclosing callm scopes (see :func:`merge_scopes`).
    scope_budgets: tuple[Budget, ...] = ()

    @property
    def budgets(self) -> tuple[Budget, ...]:
        """Every budget this call is charged to (own budget first, no duplicates)."""
        own = (self.budget,) if self.budget is not None else ()
        return tuple(dict.fromkeys((*own, *self.scope_budgets)))


def merge_scopes(outer: CallConfig, inner: CallConfig) -> CallConfig:
    """Configuration for a callm scope nested inside another one.

    Behaviour (cache, retries, fallback, validation, telemetry name) comes from the innermost
    scope, but protections accumulate: PII masking and injection detection enabled by any
    enclosing scope stay enabled, the strictest ``max_cost`` applies, and every enclosing
    budget is charged.
    """
    changes: dict[str, Any] = {}
    if inner.pii is None and outer.pii is not None:
        changes["pii"] = outer.pii
    if inner.injection is None and outer.injection is not None:
        changes["injection"] = outer.injection
    if outer.max_cost is not None and (inner.max_cost is None or outer.max_cost < inner.max_cost):
        changes["max_cost"] = outer.max_cost
    budgets = tuple(b for b in outer.budgets if b not in inner.budgets)
    if budgets:
        changes["scope_budgets"] = (*inner.scope_budgets, *budgets)
    if not changes:
        return inner
    import dataclasses

    return dataclasses.replace(inner, **changes)


def build_call_config(
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
) -> CallConfig:
    """Validate user-facing options and turn them into a :class:`CallConfig`."""
    from callm.budgets import Budget
    from callm.providers.registry import parse_target

    # cache
    cache_cfg: CacheConfig | None
    if cache is None or cache is False:
        cache_cfg = None
    elif cache is True or cache == "exact":
        cache_cfg = CacheConfig()
    elif cache == "semantic":
        cache_cfg = CacheConfig(semantic=True)
    elif isinstance(cache, CacheConfig):
        cache_cfg = cache
    else:
        raise ConfigurationError(
            f"cache must be True, False, 'exact', 'semantic' or a CacheConfig, got {cache!r}"
        )

    # retry
    retry_cfg: RetryConfig
    if retry is None or retry is True:
        retry_cfg = RetryConfig(max_retries=get_settings().default_retries)
    elif retry is False:
        retry_cfg = RetryConfig(max_retries=0)
    elif isinstance(retry, int):
        retry_cfg = RetryConfig(max_retries=retry)
    elif isinstance(retry, RetryConfig):
        retry_cfg = retry
    else:
        raise ConfigurationError(f"retry must be an int, bool or RetryConfig, got {retry!r}")

    # fallback
    if fallback is None:
        fallback_targets: tuple[Target, ...] = ()
    else:
        items = [fallback] if isinstance(fallback, str) else list(fallback)
        fallback_targets = tuple(parse_target(item) for item in items)
    if fallback_on is not None and not callable(fallback_on):
        raise ConfigurationError("fallback_on must be a callable(exception) -> bool")

    # cost
    if max_cost is not None:
        if isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)) or max_cost <= 0:
            raise ConfigurationError("max_cost must be a positive number of US dollars")
        max_cost = float(max_cost)
    if budget is not None and not isinstance(budget, Budget):
        raise ConfigurationError("budget must be a callm.Budget instance")

    # security
    pii_cfg: PIIConfig | None
    if block_pii is None or block_pii is False:
        pii_cfg = None
    elif block_pii is True:
        pii_cfg = PIIConfig()
    elif isinstance(block_pii, PIIConfig):
        pii_cfg = block_pii
    else:
        raise ConfigurationError("block_pii must be a bool or PIIConfig")

    inj_cfg: InjectionConfig | None
    if detect_injection is None or detect_injection is False:
        inj_cfg = None
    elif detect_injection is True:
        inj_cfg = InjectionConfig()
    elif isinstance(detect_injection, InjectionConfig):
        inj_cfg = detect_injection
    else:
        raise ConfigurationError("detect_injection must be a bool or InjectionConfig")

    # validation
    if isinstance(validation_retries, bool) or not isinstance(validation_retries, int):
        raise ConfigurationError("validation_retries must be an int >= 0")
    if validation_retries < 0:
        raise ConfigurationError("validation_retries must be >= 0")
    if output_schema is not None:
        from callm.validation import check_schema

        check_schema(output_schema)

    return CallConfig(
        cache=cache_cfg,
        retry=retry_cfg,
        fallback=fallback_targets,
        fallback_on=fallback_on,
        max_cost=max_cost,
        budget=budget,
        pii=pii_cfg,
        injection=inj_cfg,
        output_schema=output_schema,
        validation_retries=validation_retries,
        name=name,
        tags=dict(tags or {}),
        telemetry=bool(telemetry),
    )


# --------------------------------------------------------------------------- global settings


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def default_home() -> Path:
    env = os.environ.get("CALLM_HOME")
    return Path(env).expanduser() if env else Path.home() / ".callm"


@dataclass
class Settings:
    """Process-wide settings. Change them with :func:`callm.configure`."""

    home: Path = field(default_factory=default_home)
    #: ``"sqlite"`` (default, stored in ``home/callm.db``), ``"memory"``, or a storage object
    #: implementing both :class:`CacheStore` and :class:`TelemetryStore`.
    storage: Any = field(default_factory=lambda: os.environ.get("CALLM_STORAGE", "sqlite"))
    #: Overrides the cache backend only (e.g. a shared ``RedisCacheStore``).
    cache_store: CacheStore | None = None
    telemetry: bool = field(default_factory=lambda: _env_flag("CALLM_TELEMETRY", True))
    otel: bool = field(default_factory=lambda: _env_flag("CALLM_OTEL", False))
    enabled: bool = field(default_factory=lambda: not _env_flag("CALLM_DISABLED", False))
    default_retries: int = 2
    #: Output tokens assumed by the cost guard when a request does not set ``max_tokens``.
    assumed_output_tokens: int = 1024
    #: Provider SDK clients used for fallback targets and ``callm.complete``.
    clients: dict[str, Any] = field(default_factory=dict)
    async_clients: dict[str, Any] = field(default_factory=dict)
    on_call: list[Callable[[CallRecord], None]] = field(default_factory=list)


_settings = Settings()
_settings_lock = threading.RLock()
_stores: dict[str, Any] = {}

_CONFIGURABLE = frozenset(
    {
        "home",
        "storage",
        "cache_store",
        "telemetry",
        "otel",
        "enabled",
        "default_retries",
        "assumed_output_tokens",
        "clients",
        "async_clients",
        "on_call",
    }
)


def get_settings() -> Settings:
    return _settings


def configure(**options: Any) -> Settings:
    """Change global settings.

    Example::

        callm.configure(home="/var/lib/myapp/callm", telemetry=True, default_retries=3)
        callm.configure(storage="memory")                   # nothing touches disk
        callm.configure(cache_store=RedisCacheStore("redis://localhost:6379/0"))
        callm.configure(clients={"anthropic": Anthropic(api_key=...)})
    """
    unknown = set(options) - _CONFIGURABLE
    if unknown:
        raise ConfigurationError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    with _settings_lock:
        if "home" in options:
            options["home"] = Path(options["home"]).expanduser()
        if "default_retries" in options:
            value = options["default_retries"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError("default_retries must be an int >= 0")
        storage = options.get("storage")
        if isinstance(storage, str) and storage not in ("sqlite", "memory"):
            raise ConfigurationError("storage must be 'sqlite', 'memory' or a storage object")
        if "on_call" in options:
            hooks = options["on_call"]
            options["on_call"] = [hooks] if callable(hooks) else list(hooks or [])
        for key in ("clients", "async_clients"):
            if key in options:
                options[key] = dict(options[key] or {})
        for key, value in options.items():
            setattr(_settings, key, value)
        if {"home", "storage", "cache_store"} & set(options):
            _close_stores()
        if "home" in options:
            from callm.pricing import reload_pricing

            reload_pricing()
    return _settings


def reset_settings() -> None:
    """Restore default settings (mainly useful in tests)."""
    global _settings
    with _settings_lock:
        _close_stores()
        _settings = Settings()


def _close_stores() -> None:
    for store in _stores.values():
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("error while closing store", exc_info=True)
    _stores.clear()


def _default_storage() -> Any:
    with _settings_lock:
        store = _stores.get("default")
        if store is not None:
            return store
        storage = _settings.storage
        if storage == "memory":
            from callm.storage.memory import MemoryStorage

            store = MemoryStorage()
        elif storage == "sqlite":
            from callm.storage.sqlite import SQLiteStorage

            try:
                store = SQLiteStorage(_settings.home / "callm.db")
            except Exception as exc:  # unwritable home, read-only FS, ...
                from callm.storage.memory import MemoryStorage

                logger.warning(
                    "callm: could not open SQLite storage in %s (%s); using in-memory storage",
                    _settings.home,
                    exc,
                )
                store = MemoryStorage()
        else:
            store = storage
        _stores["default"] = store
        return store


def get_cache_store() -> CacheStore:
    if _settings.cache_store is not None:
        return _settings.cache_store
    store: CacheStore = _default_storage()
    return store


def get_telemetry_store() -> TelemetryStore:
    store: TelemetryStore = _default_storage()
    return store


__all__ = [
    "DEFAULT_RETRY_STATUSES",
    "CacheConfig",
    "CallConfig",
    "InjectionConfig",
    "PIIConfig",
    "RetryConfig",
    "Settings",
    "build_call_config",
    "configure",
    "get_cache_store",
    "get_settings",
    "get_telemetry_store",
    "merge_scopes",
    "reset_settings",
]
