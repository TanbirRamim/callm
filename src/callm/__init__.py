"""callm - the production toolkit for LLM calls.

Wrap any function that calls OpenAI, Anthropic or Gemini and get caching, retries,
provider fallback, cost tracking and budgets, PII redaction, prompt injection detection,
structured output validation and telemetry - with zero required dependencies::

    from callm import callm

    @callm(cache=True, retry=3, fallback=["anthropic/claude-sonnet-5"], max_cost=0.25)
    def summarize(text: str):
        return openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": text}]
        )
"""

import logging as _logging

from callm.__about__ import __version__
from callm.api import acomplete, complete, shield
from callm.budgets import Budget, budget, budget_for
from callm.config import (
    CacheConfig,
    InjectionConfig,
    PIIConfig,
    RetryConfig,
    Settings,
    configure,
    get_settings,
    reset_settings,
)
from callm.decorator import callm
from callm.embeddings import HashingEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from callm.errors import (
    AllProvidersFailedError,
    BudgetExceeded,
    BudgetExceededError,
    CallmError,
    ConfigurationError,
    MissingDependencyError,
    OutputValidationError,
    PIIDetectedError,
    PromptInjectionError,
    ProviderNotAvailableError,
    SecurityError,
)
from callm.middleware.telemetry import last_call
from callm.pricing import ModelPrice, get_price, set_price
from callm.providers import OpenAIProvider, Provider, register_provider
from callm.reports import clear_cache, stats
from callm.security import detect_injection, redact_pii
from callm.types import CallRecord, LLMRequest, LLMResponse, Message, Target, Usage

_logging.getLogger("callm").addHandler(_logging.NullHandler())

__all__ = [
    "AllProvidersFailedError",
    "Budget",
    "BudgetExceeded",
    "BudgetExceededError",
    "CacheConfig",
    "CallRecord",
    "CallmError",
    "ConfigurationError",
    "HashingEmbedder",
    "InjectionConfig",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "MissingDependencyError",
    "ModelPrice",
    "OpenAIEmbedder",
    "OpenAIProvider",
    "OutputValidationError",
    "PIIConfig",
    "PIIDetectedError",
    "PromptInjectionError",
    "Provider",
    "ProviderNotAvailableError",
    "RetryConfig",
    "SecurityError",
    "SentenceTransformerEmbedder",
    "Settings",
    "Target",
    "Usage",
    "__version__",
    "acomplete",
    "budget",
    "budget_for",
    "callm",
    "clear_cache",
    "complete",
    "configure",
    "detect_injection",
    "get_price",
    "get_settings",
    "last_call",
    "redact_pii",
    "register_provider",
    "reset_settings",
    "set_price",
    "shield",
    "stats",
]
