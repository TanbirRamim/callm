"""Shared fixtures: isolated settings and mock provider servers behind real SDK clients.

The provider SDKs are exercised for real; only the HTTP transport is replaced, so request
serialization, response parsing and error classes are the genuine SDK behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import callm
from callm import budgets, pipeline, pricing
from callm.config import get_telemetry_store, reset_settings
from callm.providers.registry import reset_providers

httpx2 = pytest.importorskip("httpx2")

from helpers import MockServer, anthropic_body, gemini_body, openai_body  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_callm(tmp_path, monkeypatch) -> Iterator[None]:
    """Every test gets its own callm home and in-memory storage."""
    monkeypatch.setenv("CALLM_HOME", str(tmp_path / "callm-home"))
    for name in ("CALLM_TELEMETRY", "CALLM_DISABLED", "CALLM_STORAGE", "CALLM_OTEL"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    callm.configure(storage="memory")
    pricing.clear_price_overrides()
    pricing.reload_pricing()
    budgets.clear_budgets()
    reset_providers()
    yield
    reset_settings()
    pricing.clear_price_overrides()
    pricing.reload_pricing()
    budgets.clear_budgets()
    reset_providers()


@pytest.fixture(autouse=True)
def slept(monkeypatch) -> list[float]:
    """Record backoff delays instead of sleeping."""
    delays: list[float] = []

    def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    async def fake_async_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(pipeline, "sleep_sync", fake_sleep)
    monkeypatch.setattr(pipeline, "sleep_async", fake_async_sleep)
    return delays


@pytest.fixture
def openai_server() -> MockServer:
    return MockServer(openai_body)


@pytest.fixture
def anthropic_server() -> MockServer:
    return MockServer(anthropic_body)


@pytest.fixture
def gemini_server() -> MockServer:
    return MockServer(gemini_body)


@pytest.fixture
def openai_client(openai_server):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(openai_server.handler)),
    )


@pytest.fixture
def async_openai_client(openai_server):
    openai = pytest.importorskip("openai")
    return openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(openai_server.handler)),
    )


@pytest.fixture
def anthropic_client(anthropic_server):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(anthropic_server.handler)),
    )


@pytest.fixture
def async_anthropic_client(anthropic_server):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(anthropic_server.handler)),
    )


@pytest.fixture
def gemini_client(gemini_server):
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    return genai.Client(
        api_key="gemini-test",
        http_options=types.HttpOptions(
            httpx_client=httpx2.Client(transport=httpx2.MockTransport(gemini_server.handler)),
            httpx_async_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(gemini_server.handler)
            ),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@pytest.fixture
def providers(
    openai_client, async_openai_client, anthropic_client, async_anthropic_client, gemini_client
):
    """Route callm's own provider clients (fallback targets, ``complete``) to the mocks."""
    callm.configure(
        clients={"openai": openai_client, "anthropic": anthropic_client, "google": gemini_client},
        async_clients={
            "openai": async_openai_client,
            "anthropic": async_anthropic_client,
            "google": gemini_client.aio,
        },
    )


@pytest.fixture
def records():
    """Callable returning the telemetry records captured so far (newest first)."""
    return lambda: get_telemetry_store().query()
