"""Token pricing and cost calculation.

Prices are USD per one million tokens. Lookup order:

1. prices set at runtime with :func:`set_price`
2. ``$CALLM_HOME/pricing.json`` written by ``callm pricing update``
3. the table bundled with this release (``callm/data/pricing.json``)

Bundled prices are a snapshot; provider prices change. Run ``callm pricing update`` to
refresh them from the community-maintained LiteLLM price list, or call
:func:`set_price` for negotiated or self-hosted rates.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import threading
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from callm.types import LLMRequest, Usage

logger = logging.getLogger("callm")

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

_LITELLM_PROVIDERS = {"openai": "openai", "anthropic": "anthropic", "gemini": "google"}
_EXCLUDED_MARKERS = (
    "ft:",
    "audio",
    "realtime",
    "search",
    "tts",
    "transcribe",
    "image",
    "computer-use",
    "embedding",
    "moderation",
    "lyria",
    "robotics",
    "live",
    "container",
    "codex",
)


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def cost(self, usage: Usage) -> float:
        cache_read = self.cache_read if self.cache_read is not None else self.input
        cache_write = self.cache_write if self.cache_write is not None else self.input
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, float]:
        data = {"input": self.input, "output": self.output}
        if self.cache_read is not None:
            data["cache_read"] = self.cache_read
        if self.cache_write is not None:
            data["cache_write"] = self.cache_write
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPrice:
        return cls(
            input=float(data["input"]),
            output=float(data["output"]),
            cache_read=None if data.get("cache_read") is None else float(data["cache_read"]),
            cache_write=None if data.get("cache_write") is None else float(data["cache_write"]),
        )


_lock = threading.RLock()
_overrides: dict[str, dict[str, ModelPrice]] = {}
_loaded: dict[str, dict[str, ModelPrice]] | None = None
_warned: set[str] = set()


def normalize_model(model: str) -> str:
    name = model.strip().lower()
    for prefix in ("models/", "gemini/", "openai/", "anthropic/", "google/"):
        name = name.removeprefix(prefix)
    return name.split("@", 1)[0]  # Vertex AI style "claude-opus-4-5@20251101"


def _parse_table(data: dict[str, Any]) -> dict[str, dict[str, ModelPrice]]:
    table: dict[str, dict[str, ModelPrice]] = {}
    for provider, models in (data.get("models") or {}).items():
        table[provider] = {
            normalize_model(name): ModelPrice.from_dict(price) for name, price in models.items()
        }
    return table


def user_pricing_path() -> Path:
    from callm.config import get_settings

    return get_settings().home / "pricing.json"


def _bundled() -> dict[str, Any]:
    text = resources.files("callm.data").joinpath("pricing.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def _load() -> dict[str, dict[str, ModelPrice]]:
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        table = _parse_table(_bundled())
        path = user_pricing_path()
        if path.is_file():
            try:
                user = _parse_table(json.loads(path.read_text(encoding="utf-8")))
                for provider, models in user.items():
                    table.setdefault(provider, {}).update(models)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("callm: ignoring unreadable pricing file %s (%s)", path, exc)
        _loaded = table
        return table


def reload_pricing() -> None:
    """Forget cached price tables (re-read on next lookup)."""
    global _loaded
    with _lock:
        _loaded = None
        _warned.clear()


def _lookup(table: dict[str, ModelPrice], name: str) -> ModelPrice | None:
    if name in table:
        return table[name]
    # Longest known prefix on a "-" boundary: "gpt-4o-2024-08-06" -> "gpt-4o",
    # "claude-haiku-4-5-20251001" -> "claude-haiku-4-5".
    best: str | None = None
    for known in table:
        if name.startswith(known + "-") and (best is None or len(known) > len(best)):
            best = known
    return table[best] if best is not None else None


def _provider_key(provider: str) -> str:
    return "google" if provider == "gemini" else provider


def get_price(provider: str, model: str) -> ModelPrice | None:
    """Price for ``provider``/``model`` or ``None`` when unknown."""
    provider = _provider_key(provider)
    name = normalize_model(model)
    with _lock:
        price = _lookup(_overrides.get(provider, {}), name) or _lookup(
            _overrides.get("*", {}), name
        )
    if price is not None:
        return price
    return _lookup(_load().get(provider, {}), name)


def set_price(
    model: str,
    *,
    input: float,
    output: float,
    cache_read: float | None = None,
    cache_write: float | None = None,
    provider: str = "*",
) -> None:
    """Register a price (USD per 1M tokens), e.g. for fine-tunes or self-hosted models.

    ``provider="*"`` applies to every provider serving a model with that name.
    """
    for value in (input, output, cache_read, cache_write):
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("prices must be non-negative numbers (USD per 1M tokens)")
    with _lock:
        _overrides.setdefault(_provider_key(provider), {})[normalize_model(model)] = ModelPrice(
            input=float(input),
            output=float(output),
            cache_read=cache_read,
            cache_write=cache_write,
        )
        _warned.clear()


def clear_price_overrides() -> None:
    with _lock:
        _overrides.clear()


def cost_for(provider: str, model: str, usage: Usage, *, warn: bool = True) -> float | None:
    """Actual cost of a completed call, or ``None`` when the model's price is unknown."""
    price = get_price(provider, model)
    if price is None:
        key = f"{provider}/{model}"
        if warn and key not in _warned:
            _warned.add(key)
            logger.warning(
                "callm: no price known for %s; cost will not be tracked. "
                "Run `callm pricing update` or call callm.set_price().",
                key,
            )
        return None
    return price.cost(usage)


# --------------------------------------------------------------------------- estimation

_IMAGE_TOKENS_ESTIMATE = 1500
_encoders: dict[str, Any] = {}


def estimate_tokens(text: str, model: str | None = None, provider: str | None = None) -> int:
    """Estimate a token count. Uses ``tiktoken`` for OpenAI models when it is installed."""
    if not text:
        return 0
    if provider in (None, "openai") and model is not None:
        encoder = _tiktoken_encoder(model)
        if encoder is not None:
            return len(encoder.encode(text, disallowed_special=()))
    # Roughly 4 characters per token for English; divide by 3.5 to stay conservative.
    return math.ceil(len(text) / 3.5)


def _tiktoken_encoder(model: str) -> Any:
    if model in _encoders:
        return _encoders[model]
    try:
        import tiktoken
    except ImportError:
        _encoders[model] = None
        return None
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            encoder = tiktoken.get_encoding("o200k_base")
        except Exception:
            encoder = None
    except Exception:
        encoder = None
    _encoders[model] = encoder
    return encoder


def estimate_input_tokens(request: LLMRequest) -> int:
    tokens = 0
    for message in request.messages:
        tokens += 4  # per-message framing overhead
        tokens += estimate_tokens(message.text, request.model, request.provider)
        if isinstance(message.content, list):
            tokens += _IMAGE_TOKENS_ESTIMATE * sum(
                1 for part in message.content if not (isinstance(part, dict) and "text" in part)
            )
    for key in ("tools", "functions", "response_format"):
        if key in request.native_extra:
            tokens += estimate_tokens(json.dumps(request.native_extra[key], default=str))
    return tokens


def estimate_cost(request: LLMRequest, assumed_output_tokens: int) -> tuple[float | None, int, int]:
    """Worst-case cost estimate before sending ``request``.

    Returns ``(cost_or_None, input_tokens, output_tokens)``. When the request sets
    ``max_tokens`` that value is used for output, otherwise ``assumed_output_tokens``.
    """
    input_tokens = estimate_input_tokens(request)
    max_tokens = request.params.get("max_tokens")
    output_tokens = int(max_tokens) if max_tokens else assumed_output_tokens
    price = get_price(request.provider, request.model)
    if price is None:
        return None, input_tokens, output_tokens
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return price.cost(usage), input_tokens, output_tokens


# --------------------------------------------------------------------------- updating


def convert_litellm(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Convert LiteLLM's price list into callm's ``{"provider": {"model": price}}`` form."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, info in data.items():
        if not isinstance(info, dict):
            continue
        provider = _LITELLM_PROVIDERS.get(str(info.get("litellm_provider")))
        if provider is None or info.get("mode") != "chat":
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _EXCLUDED_MARKERS):
            continue
        inp, outp = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if not isinstance(inp, (int, float)) or not isinstance(outp, (int, float)):
            continue
        price: dict[str, float] = {"input": round(inp * 1e6, 6), "output": round(outp * 1e6, 6)}
        cache_read = info.get("cache_read_input_token_cost")
        cache_write = info.get("cache_creation_input_token_cost")
        if isinstance(cache_read, (int, float)):
            price["cache_read"] = round(cache_read * 1e6, 6)
        if isinstance(cache_write, (int, float)):
            price["cache_write"] = round(cache_write * 1e6, 6)
        out.setdefault(provider, {})[normalize_model(name)] = price
    return {provider: dict(sorted(models.items())) for provider, models in sorted(out.items())}


def update_pricing(
    url: str = LITELLM_PRICES_URL, dest: Path | None = None, timeout: float = 30
) -> int:
    """Download current prices into ``$CALLM_HOME/pricing.json``. Returns the model count."""
    if not url.startswith(("https://", "http://", "file://")):
        raise ValueError("pricing URL must use https, http or file")
    request = urllib.request.Request(url, headers={"User-Agent": "callm-pricing-update"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    models = convert_litellm(raw)
    count = sum(len(v) for v in models.values())
    if count == 0:
        raise ValueError("the downloaded price list contained no usable models")
    payload = {
        "source": url,
        "fetched": datetime.date.today().isoformat(),
        "unit": "USD per 1M tokens",
        "models": models,
    }
    path = dest or user_pricing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    reload_pricing()
    return count


def pricing_metadata() -> dict[str, Any]:
    bundled = _bundled()
    meta: dict[str, Any] = {
        "bundled_fetched": bundled.get("fetched"),
        "bundled_models": sum(len(v) for v in bundled.get("models", {}).values()),
        "source": bundled.get("source"),
    }
    path = user_pricing_path()
    if path.is_file():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
            meta["user_file"] = str(path)
            meta["user_fetched"] = user.get("fetched")
        except (OSError, ValueError):
            meta["user_file"] = f"{path} (unreadable)"
    return meta


__all__ = [
    "LITELLM_PRICES_URL",
    "ModelPrice",
    "clear_price_overrides",
    "convert_litellm",
    "cost_for",
    "estimate_cost",
    "estimate_input_tokens",
    "estimate_tokens",
    "get_price",
    "normalize_model",
    "pricing_metadata",
    "reload_pricing",
    "set_price",
    "update_pricing",
    "user_pricing_path",
]
