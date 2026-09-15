"""Text embedders for the semantic cache."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from callm.errors import MissingDependencyError


@runtime_checkable
class Embedder(Protocol):
    """Anything with ``embed(texts) -> vectors`` can back the semantic cache."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SentenceTransformerEmbedder:
    """Local embeddings with ``sentence-transformers`` (``pip install 'callm-toolkit[cache]'``)."""

    _models: dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model
        self.device = device

    def _model(self) -> Any:
        key = f"{self.model_name}@{self.device}"
        with self._lock:
            model = self._models.get(key)
            if model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise MissingDependencyError(
                        "Semantic caching", "sentence-transformers", "cache"
                    ) from exc
                model = SentenceTransformer(self.model_name, device=self.device)
                self._models[key] = model
            return model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model().encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]


class OpenAIEmbedder:
    """Embeddings from the OpenAI API (``text-embedding-3-small`` by default)."""

    def __init__(self, model: str = "text-embedding-3-small", client: Any = None) -> None:
        self.model = model
        self._client = client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from callm.pipeline import call_bypassed

        client = self._client
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise MissingDependencyError("OpenAIEmbedder", "openai", "openai") from exc
            client = self._client = openai.OpenAI()
        response = call_bypassed(client.embeddings.create, model=self.model, input=list(texts))
        return [normalize_vector(item.embedding) for item in response.data]


class HashingEmbedder:
    """Dependency-free *lexical* embedder (hashed word and character n-grams).

    It captures surface similarity (typos, casing, word order) but not meaning. Useful for
    tests, offline use, or near-duplicate detection. Prefer a real embedding model for
    semantic caching in production.
    """

    def __init__(self, dim: int = 1024, char_ngram: int = 3) -> None:
        self.dim = dim
        self.char_ngram = char_ngram

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                index, sign = self._index("w:" + word)
                vector[index] += sign
                padded = f"#{word}#"
                for i in range(max(len(padded) - self.char_ngram + 1, 1)):
                    index, sign = self._index("c:" + padded[i : i + self.char_ngram])
                    vector[index] += 0.5 * sign
            vectors.append(normalize_vector(vector))
        return vectors


def default_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
    "default_embedder",
    "normalize_vector",
]
