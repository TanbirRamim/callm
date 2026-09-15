"""Provider registry and ``"provider/model"`` target parsing."""

from __future__ import annotations

import threading

from callm.errors import ConfigurationError, ProviderNotAvailableError
from callm.providers.anthropic import AnthropicProvider
from callm.providers.base import Provider
from callm.providers.google import GoogleProvider
from callm.providers.openai import OpenAIProvider
from callm.types import Target

_lock = threading.Lock()
_providers: dict[str, Provider] = {}
_ALIASES = {"gemini": "google", "google-genai": "google", "claude": "anthropic"}
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _install_defaults() -> None:
    _providers.update(
        {
            "openai": OpenAIProvider("openai", model_prefixes=_OPENAI_PREFIXES),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
            "ollama": OpenAIProvider(
                "ollama",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                max_tokens_param="max_tokens",
            ),
        }
    )


_install_defaults()


def register_provider(name: str, provider: Provider) -> None:
    """Register a provider adapter, for example an OpenAI-compatible endpoint::

        callm.register_provider(
            "groq",
            OpenAIProvider("groq", base_url="https://api.groq.com/openai/v1",
                           api_key_env="GROQ_API_KEY", max_tokens_param="max_tokens"),
        )

    It can then be used in targets such as ``fallback=["groq/llama-3.3-70b-versatile"]``.
    """
    if not name or "/" in name:
        raise ConfigurationError("provider names must be non-empty and must not contain '/'")
    if not isinstance(provider, Provider):
        raise ConfigurationError("provider must be a callm.Provider instance")
    with _lock:
        _providers[name] = provider


def get_provider(name: str) -> Provider:
    key = _ALIASES.get(name, name)
    try:
        return _providers[key]
    except KeyError:
        raise ProviderNotAvailableError(
            f"Unknown provider '{name}'. Known providers: {', '.join(sorted(_providers))}"
        ) from None


def provider_names() -> list[str]:
    return sorted(_providers)


def infer_provider(model: str) -> str | None:
    for name, provider in _providers.items():
        if provider.owns_model(model):
            return name
    return None


def _resolve_alias(provider: str, model: str) -> str:
    if provider == "anthropic":
        return AnthropicProvider.resolve_model(model)
    return model


def parse_target(spec: str | Target, provider: str | None = None) -> Target:
    """Parse ``"anthropic/claude-sonnet-5"``, ``"gpt-4o-mini"`` or a :class:`Target`."""
    if isinstance(spec, Target):
        return spec
    if not isinstance(spec, str) or not spec.strip():
        raise ConfigurationError(f"invalid model target {spec!r}")
    spec = spec.strip()
    if provider is not None:
        name = _ALIASES.get(provider, provider)
        get_provider(name)
        model = spec[len(name) + 1 :] if spec.startswith(name + "/") else spec
        return Target(name, _resolve_alias(name, model))
    head, sep, tail = spec.partition("/")
    if sep:
        name = _ALIASES.get(head, head)
        if name in _providers and tail:
            return Target(name, _resolve_alias(name, tail))
    inferred = infer_provider(spec)
    if inferred is None:
        raise ConfigurationError(
            f"Cannot infer the provider for model '{spec}'. Use 'provider/model', e.g. "
            f"'openai/{spec}'. Known providers: {', '.join(sorted(_providers))}"
        )
    return Target(inferred, _resolve_alias(inferred, spec))


def reset_providers() -> None:
    """Restore the built-in providers (mainly for tests)."""
    with _lock:
        _providers.clear()
        _install_defaults()


__all__ = [
    "get_provider",
    "infer_provider",
    "parse_target",
    "provider_names",
    "register_provider",
    "reset_providers",
]
