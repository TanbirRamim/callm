"""Response cache layer: exact-match by default, optional semantic matching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from callm.config import CacheConfig, get_cache_store
from callm.embeddings import Embedder, cosine_similarity, default_embedder
from callm.errors import MissingDependencyError
from callm.pipeline import CallState, Handler, Invoke, Step
from callm.providers.base import to_plain
from callm.storage.base import CacheEntry, CacheStore
from callm.types import LLMRequest, LLMResponse
from callm.validation import schema_fingerprint

logger = logging.getLogger("callm")

T = TypeVar("T")

CACHE_FORMAT_VERSION = 1

#: Arguments that never change what the model generates.
NON_SEMANTIC_KWARGS = frozenset(
    {
        "timeout",
        "extra_headers",
        "extra_query",
        "metadata",
        "user",
        "store",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
        "stream_options",
        "user_profile_id",
        "workspace_id",
        "stream",
    }
)
#: Finish reasons whose responses are never cached.
UNCACHEABLE_FINISH = frozenset(
    {
        "refusal",
        "content_filter",
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
    }
)
_default_embedder: list[Embedder] = []


def _json_default(value: Any) -> Any:
    plain = to_plain(value)
    if plain is not value:
        return plain
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, (set, frozenset)):
        return sorted(map(str, value))
    return repr(value)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _message_material(request: LLMRequest, *, mask_last_user: bool) -> list[Any]:
    last_user = max((i for i, m in enumerate(request.messages) if m.role == "user"), default=-1)
    material: list[Any] = []
    for index, message in enumerate(request.messages):
        native = {k: v for k, v in (message.native or {}).items() if k != "content"}
        content: Any = "<<query>>" if mask_last_user and index == last_user else message.content
        material.append({"role": message.role, "content": content, "native": native})
    return material


def cache_keys(
    request: LLMRequest, config: CacheConfig, schema: Any = None
) -> tuple[str, str | None, str]:
    """Return ``(exact_key, semantic_group, query_text)`` for ``request``.

    The semantic group covers everything except the final user message, so semantic
    matches never cross system prompts, conversation history, parameters or schemas.
    """
    extra = {
        key: value
        for key, value in request.native_extra.items()
        if key not in NON_SEMANTIC_KWARGS and not key.startswith("__")
    }
    base = {
        "v": CACHE_FORMAT_VERSION,
        "namespace": config.namespace,
        "provider": request.provider,
        "model": request.model,
        "params": request.params,
        "extra": extra,
        "schema": schema_fingerprint(schema),
    }
    exact = _digest({**base, "messages": _message_material(request, mask_last_user=False)})
    last = request.messages[-1] if request.messages else None
    if last is None or last.role != "user" or not last.is_text_only:
        return exact, None, ""
    group = _digest({**base, "messages": _message_material(request, mask_last_user=True)})
    return exact, group, last.text


def is_cacheable(request: LLMRequest) -> bool:
    if request.stream:
        return False
    if request.native_extra.get("n") not in (None, 1):
        return False
    config = request.native_extra.get("config")
    if isinstance(config, dict):
        count = config.get("candidate_count")
    else:
        count = getattr(config, "candidate_count", None)
    return count in (None, 1)


def _embedder_for(config: CacheConfig) -> Embedder:
    if config.embedder is not None:
        return config.embedder
    if not _default_embedder:
        _default_embedder.append(default_embedder())
    return _default_embedder[0]


def _safe(fn: Callable[[], T], action: str) -> T | None:
    try:
        return fn()
    except Exception:
        logger.warning("callm: cache %s failed; continuing without cache", action, exc_info=True)
        return None


class CacheMiddleware:
    name = "cache"

    def handle(self, state: CallState, call_next: Handler) -> Step[LLMResponse]:
        config = state.config.cache
        request = state.request
        if config is None or not is_cacheable(request):
            return (yield from call_next(state))

        store: CacheStore = config.store or get_cache_store()
        exact_key, group, query = cache_keys(request, config, state.config.output_schema)
        embedding: list[float] | None = None

        entry = _safe(lambda: store.get(exact_key), "read")
        if entry is None and config.semantic and group is not None and query.strip():
            embedder = _embedder_for(config)
            try:
                embedding = yield Invoke(sync=lambda: embedder.embed([query])[0], offload=True)
            except MissingDependencyError:
                raise
            except Exception:
                logger.warning("callm: embedding failed; semantic lookup skipped", exc_info=True)
                embedding = None
            if embedding is not None:
                entry = self._semantic_match(store, group, embedding, config)

        if entry is not None:
            response = LLMResponse.from_dict(entry.response)
            response.cached = True
            response.cost = 0.0
            state.record.cache_hit = True
            state.record.saved_usd = entry.cost
            return response

        response = yield from call_next(state)

        if response.finish_reason not in UNCACHEABLE_FINISH:
            created = time.time()
            new_entry = CacheEntry(
                key=exact_key,
                response=response.to_dict(),
                group=group if embedding is not None else None,
                embedding=embedding,
                created_at=created,
                expires_at=created + config.ttl if config.ttl else None,
            )
            _safe(lambda: store.set(new_entry), "write")
        return response

    @staticmethod
    def _semantic_match(
        store: CacheStore, group: str, embedding: list[float], config: CacheConfig
    ) -> CacheEntry | None:
        candidates = _safe(lambda: store.candidates(group, config.max_candidates), "read") or []
        best: CacheEntry | None = None
        best_score = -1.0
        for candidate in candidates:
            if candidate.embedding is None:
                continue
            score = cosine_similarity(embedding, candidate.embedding)
            if score > best_score:
                best, best_score = candidate, score
        if best is None or best_score < config.threshold:
            return None
        key = best.key
        _safe(lambda: store.record_hit(key), "write")
        return best


__all__ = ["NON_SEMANTIC_KWARGS", "CacheMiddleware", "cache_keys", "is_cacheable"]
