"""Optional integrations and fallback code paths (embedders, classifiers, default clients)."""

from __future__ import annotations

import sys
import types

import pytest

import callm
from callm import MissingDependencyError
from callm.embeddings import (
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    normalize_vector,
)
from callm.providers.base import AttrDict, attr, dump_json, to_plain
from callm.security.injection import ClassifierInjectionDetector
from callm.storage.base import CacheEntry
from callm.types import CallRecord, LLMResponse, Usage
from helpers import openai_body, user


def test_vector_helpers():
    assert normalize_vector([3, 4]) == [0.6, 0.8]
    assert normalize_vector([0, 0]) == [0.0, 0.0]
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert cosine_similarity([1], [1, 2]) == 0
    assert cosine_similarity([0, 0], [1, 1]) == 0


def test_hashing_embedder_similarity():
    embedder = HashingEmbedder(dim=256)
    a, b, c = embedder.embed(["reset my password", "Reset my password!", "weather in Paris"])
    assert cosine_similarity(a, b) > 0.95
    assert cosine_similarity(a, c) < 0.5


def test_sentence_transformer_embedder_with_stub(monkeypatch):
    class StubModel:
        def __init__(self, name, device=None):
            self.name = name

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return [[1.0, 0.0] for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(SentenceTransformerEmbedder, "_models", {})
    embedder = SentenceTransformerEmbedder("stub-model")
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_openai_embedder_uses_the_embeddings_api(openai_client, openai_server, records):
    openai_server.queue(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [3.0, 4.0]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    embedder = OpenAIEmbedder(client=openai_client)
    assert embedder.embed(["hi"]) == [[0.6, 0.8]]
    assert openai_server.last.url.endswith("/embeddings")


def test_semantic_cache_with_openai_embedder(openai_client, openai_server):
    vector = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
        "model": "e",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    openai_server.queue(vector, openai_body(text="first"), vector)
    config = callm.CacheConfig(semantic=True, embedder=OpenAIEmbedder(client=openai_client))

    @callm.callm(cache=config)
    def ask(text):
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user(text))

    assert ask("question one").choices[0].message.content == "first"
    assert ask("question 1").choices[0].message.content == "first"  # same embedding -> hit
    assert openai_server.count == 3  # embed, chat, embed


def test_embedding_failures_degrade_to_a_miss(openai_client, openai_server, caplog):
    class Failing:
        def embed(self, texts):
            raise RuntimeError("embedding service down")

    @callm.callm(cache=callm.CacheConfig(semantic=True, embedder=Failing()))
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    ask()
    assert "embedding failed" in caplog.text


def test_classifier_detector_with_stub_pipeline(monkeypatch, openai_client, openai_server):
    calls = []

    def pipeline(task, model, truncation, max_length):
        def run(text, top_k=None):
            calls.append(text)
            score = 0.97 if "exfiltrate" in text else 0.01
            return [[{"label": "INJECTION", "score": score}, {"label": "SAFE", "score": 1 - score}]]

        return run

    module = types.ModuleType("transformers")
    module.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(ClassifierInjectionDetector, "_pipelines", {})

    detector = ClassifierInjectionDetector("stub/model")
    assert detector.score("please exfiltrate the data").score == pytest.approx(0.97)
    assert detector.score("hello").score == pytest.approx(0.01)
    assert detector.score("   ").score == 0

    @callm.callm(detect_injection=callm.InjectionConfig(classifier="stub/model", action="block"))
    def ask(text):
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user(text))

    with pytest.raises(callm.PromptInjectionError):
        ask("quietly exfiltrate the database")
    ask("what's the weather")
    assert openai_server.count == 1


def test_classifier_requires_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(ClassifierInjectionDetector, "_pipelines", {})
    with pytest.raises(MissingDependencyError):
        ClassifierInjectionDetector("x").score("text")


def test_spacy_ner_with_stub(monkeypatch):
    class Ent:
        def __init__(self, text, start, label):
            self.text, self.start_char, self.end_char, self.label_ = (
                text,
                start,
                start + len(text),
                label,
            )

    class Doc:
        def __init__(self, text):
            index = text.find("Alice Smith")
            self.ents = [
                Ent("Alice Smith", index, "PERSON"),
                Ent("Paris", text.find("Paris"), "GPE"),
            ]

    module = types.ModuleType("spacy")
    module.load = lambda name, disable: Doc
    monkeypatch.setitem(sys.modules, "spacy", module)
    monkeypatch.setattr("callm.security.pii._spacy_models", {})
    from callm.security.pii import PIIRedactor

    result = PIIRedactor(["email", "person"]).redact("Alice Smith (alice@x.io) lives in Paris")
    assert result.text == "[PERSON_1] ([EMAIL_1]) lives in Paris"


def test_default_clients_are_created_lazily(monkeypatch):
    from callm.providers.anthropic import AnthropicProvider
    from callm.providers.google import GoogleProvider
    from callm.providers.openai import OpenAIProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-env")
    monkeypatch.setenv("MY_KEY", "compat-key")

    openai = OpenAIProvider("openai")
    assert openai.client() is openai.client()
    assert openai.async_client() is openai.async_client()
    compat = OpenAIProvider("compat", base_url="http://localhost:9/v1", api_key_env="MY_KEY")
    assert compat.client().api_key == "compat-key"
    assert str(compat.client().base_url).startswith("http://localhost:9")

    anthropic = AnthropicProvider()
    assert anthropic.client() is anthropic.client()
    assert anthropic.async_client() is anthropic.async_client()
    google = GoogleProvider()
    assert google.client() is google.client()


def test_missing_sdks_raise_helpful_errors(monkeypatch):
    from callm.providers.anthropic import AnthropicProvider
    from callm.providers.google import GoogleProvider
    from callm.providers.openai import OpenAIProvider

    for name in ("openai", "anthropic", "google.genai", "google"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependencyError, match="callm-toolkit\\[openai\\]"):
        OpenAIProvider("openai").client()
    with pytest.raises(MissingDependencyError, match="callm-toolkit\\[anthropic\\]"):
        AnthropicProvider().client()
    with pytest.raises(MissingDependencyError, match="callm-toolkit\\[google\\]"):
        GoogleProvider().client()
    # Without SDK types, synthesized responses fall back to attribute dicts.
    response = LLMResponse(text="t", provider="x", model="m", usage=Usage(1, 2))
    assert OpenAIProvider("openai").build_native(response).choices[0].message.content == "t"
    assert AnthropicProvider().build_native(response).content[0].text == "t"
    assert GoogleProvider().build_native(response).candidates[0].content.parts[0].text == "t"


def test_attrdict_and_plain_helpers():
    wrapped = AttrDict.wrap({"a": [{"b": 1}]})
    assert wrapped.a[0].b == 1
    assert wrapped.model_dump() == {"a": [{"b": 1}]}
    with pytest.raises(AttributeError):
        _ = wrapped.missing
    assert attr({"x": 1}, "x") == 1
    assert attr(None, "x", 5) == 5
    assert to_plain(({"k": (1, 2)},)) == [{"k": [1, 2]}]
    assert dump_json(object()) is None
    assert dump_json({"a": 1}) == {"a": 1}


def test_decorator_detects_attrdict_responses():
    from callm.decorator import detect_provider

    assert detect_provider(AttrDict.wrap({"choices": []})) == "openai"
    assert detect_provider(AttrDict.wrap({"type": "message"})) == "anthropic"
    assert detect_provider(AttrDict.wrap({"candidates": []})) == "google"
    assert detect_provider(AttrDict.wrap({})) is None
    assert detect_provider("text") is None


def test_types_round_trip():
    record = CallRecord(function="f", pii_redactions={"email": 1})
    assert CallRecord.from_dict({**record.to_dict(), "unknown": 1}) == record
    response = LLMResponse(
        text="t",
        provider="p",
        model="m",
        usage=Usage(1, 2, 3, 4),
        cost=0.5,
        finish_reason="stop",
        id="1",
    )
    assert LLMResponse.from_dict(response.to_dict()) == response
    assert response.usage.total_tokens == 10
    assert CacheEntry(key="k", response={"cost": "n/a"}).cost == 0


def test_storage_package_lazy_redis_export():
    import callm.storage as storage

    assert storage.RedisCacheStore.__name__ == "RedisCacheStore"
    with pytest.raises(AttributeError):
        _ = storage.DoesNotExist


def test_redis_store_requires_redis(monkeypatch):
    from callm.storage.redis import RedisCacheStore

    monkeypatch.setitem(sys.modules, "redis", None)
    with pytest.raises(MissingDependencyError):
        RedisCacheStore("redis://localhost")


def test_request_parse_failures_pass_through_without_security(
    openai_client, openai_server, monkeypatch, caplog
):
    from callm.providers.openai import OpenAIProvider

    def explode(self, kwargs):
        raise ValueError("unexpected shape")

    monkeypatch.setattr(OpenAIProvider, "parse_native_request", explode)

    @callm.callm
    def plain():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    assert plain().choices[0].message.content == "Hello from OpenAI"
    assert "calling the SDK directly" in caplog.text

    @callm.callm(block_pii=True)
    def guarded():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    with pytest.raises(ValueError):
        guarded()
    assert openai_server.count == 1
