"""Provider adapter interface.

An adapter translates between a provider SDK's native call shape and callm's canonical
:class:`~callm.types.LLMRequest` / :class:`~callm.types.LLMResponse`, and knows how to send a
canonical request with the official SDK.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from callm.types import PORTABLE_ROLES, LLMRequest, LLMResponse


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an SDK object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_plain(obj: Any) -> Any:
    """Convert pydantic SDK objects (recursively) into plain dicts/lists."""
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, type):
        # e.g. ``response_schema=MyModel``: describe the class, don't call unbound methods.
        described: dict[str, Any] = {"__type__": f"{obj.__module__}.{obj.__qualname__}"}
        schema = getattr(obj, "model_json_schema", None)
        if callable(schema):
            try:
                described["schema"] = schema()
            except Exception:
                pass
        return described
    if callable(obj) and not hasattr(obj, "model_dump"):
        # Python functions passed as tools: a stable, address-free description.
        name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
        return {"__callable__": f"{getattr(obj, '__module__', '')}.{name}"}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    if isinstance(obj, Mapping):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def dump_json(obj: Any) -> dict[str, Any] | None:
    """JSON-safe dump of a native response (for the cache), or ``None``."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump(mode="json", exclude_none=True)
        except Exception:
            return None
        return result if isinstance(result, dict) else None
    if isinstance(obj, dict):
        return obj
    return None


class AttrDict(dict[str, Any]):
    """A dict that also allows attribute access; used when an SDK type is unavailable."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value

    def model_dump(self, **_: Any) -> dict[str, Any]:
        result: dict[str, Any] = _unwrap(self)
        return result


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def synthetic_id(prefix: str) -> str:
    return f"{prefix}callm{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())


#: Native keyword arguments that can be dropped when a request moves to another provider
#: without changing what the model is asked to do.
BENIGN_KWARGS = frozenset(
    {
        "metadata",
        "user",
        "store",
        "service_tier",
        "seed",
        "timeout",
        "extra_headers",
        "extra_query",
        "stream_options",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
        "reasoning_effort",
        "verbosity",
        "thinking",
        "cache_control",
        "inference_geo",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "user_profile_id",
        "workspace_id",
        "top_k",
        "stream",
    }
)


#: ``extra_body`` keys that are plain sampling parameters (translated via ``params``).
SAMPLING_KEYS = frozenset({"temperature", "top_p"})


class Provider(ABC):
    """Adapter for one LLM provider."""

    name: str = "base"
    #: Native arguments that map to canonical ``params`` (and therefore translate).
    canonical_param_keys: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ requests

    @abstractmethod
    def parse_native_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        """Turn SDK keyword arguments into a canonical request (``origin`` = this provider)."""

    @abstractmethod
    def to_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """SDK keyword arguments for ``request``.

        When ``request.origin == self.name`` provider-native arguments are preserved;
        otherwise the request is translated from its canonical form.
        """

    def portability_issue(self, request: LLMRequest) -> str | None:
        """Why ``request`` (native to this provider) cannot move to another provider."""
        for message in request.messages:
            if message.role not in PORTABLE_ROLES:
                return f"message role '{message.role}' is provider-specific"
            if not message.is_text_only:
                return "non-text content (images, files, tool calls) cannot be translated"
            native = message.native or {}
            for key in ("tool_calls", "function_call", "tool_call_id"):
                if native.get(key):
                    return f"message field '{key}' is provider-specific"
        for key, value in request.native_extra.items():
            if key.startswith("__") or value is None:
                continue
            if key == "n":
                if value != 1:
                    return "n > 1 cannot be translated"
                continue
            if key == "extra_body" and isinstance(value, dict) and set(value) <= SAMPLING_KEYS:
                continue
            if key not in BENIGN_KWARGS and key not in self.canonical_param_keys:
                return f"argument '{key}' is provider-specific"
        return None

    # ------------------------------------------------------------------ calls

    @abstractmethod
    def call_sync(self, request: LLMRequest) -> Any:
        """Send ``request`` with the provider's sync SDK client."""

    @abstractmethod
    async def call_async(self, request: LLMRequest) -> Any:
        """Send ``request`` with the provider's async SDK client."""

    # ------------------------------------------------------------------ responses

    @abstractmethod
    def parse_response(self, native: Any, request: LLMRequest) -> LLMResponse:
        """Canonical response from a native SDK response."""

    @abstractmethod
    def build_native(self, response: LLMResponse) -> Any:
        """Synthesize this provider's native response object from a canonical response."""

    def load_native(self, data: dict[str, Any]) -> Any:
        """Rebuild a native response object from :func:`dump_json` output."""
        return AttrDict.wrap(data)

    def native_for(self, response: LLMResponse) -> Any:
        """The native object to hand back to code that called this provider's SDK."""
        if response.provider == self.name:
            if response.raw is not None:
                return response.raw
            if response.native_dump is not None:
                try:
                    return self.load_native(response.native_dump)
                except Exception:
                    pass
        return self.build_native(response)

    def owns_model(self, model: str) -> bool:
        """Whether a bare model id (without ``provider/``) belongs to this provider."""
        return False


__all__ = [
    "BENIGN_KWARGS",
    "AttrDict",
    "Provider",
    "attr",
    "dump_json",
    "now",
    "synthetic_id",
    "to_plain",
]
