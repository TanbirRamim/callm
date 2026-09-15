"""Retry engine and error classification."""

from __future__ import annotations

import email.utils
import time
from datetime import datetime, timedelta, timezone

import httpx2
import pytest

import callm
from callm.classify import classify, get_retry_after, parse_duration, should_retry
from callm.config import RetryConfig
from callm.errors import BudgetExceeded, ConfigurationError
from helpers import user


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}


class StatusError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = FakeResponse(status, headers)


def test_retries_rate_limits_then_succeeds(openai_client, openai_server, records, slept):
    openai_server.fail(429, times=2, headers={"retry-after-ms": "250"})

    @callm.callm(retry=3)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    assert ask().choices[0].message.content == "Hello from OpenAI"
    assert openai_server.count == 3
    assert slept == [0.25, 0.25]
    assert records()[0].retries == 2


def test_gives_up_after_max_retries(openai_client, openai_server, records, slept):
    import openai

    openai_server.fail(503, times=5)

    @callm.callm(retry=2)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    with pytest.raises(openai.InternalServerError):
        ask()
    assert openai_server.count == 3
    (record,) = records()
    assert record.status == "error"
    assert record.error_type == "InternalServerError"
    assert record.retries == 2
    assert len(slept) == 2


def test_client_errors_are_not_retried(openai_client, openai_server, slept):
    import openai

    openai_server.fail(400)

    @callm.callm(retry=5)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    with pytest.raises(openai.BadRequestError):
        ask()
    assert openai_server.count == 1
    assert slept == []


def test_retry_false_disables_retries(openai_client, openai_server):
    import openai

    openai_server.fail(429)

    @callm.callm(retry=False)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    with pytest.raises(openai.RateLimitError):
        ask()
    assert openai_server.count == 1


def test_retry_after_longer_than_limit_stops_retrying(openai_client, openai_server, slept):
    import openai

    openai_server.fail(429, headers={"retry-after": "3600"})

    @callm.callm(retry=RetryConfig(max_retries=3, max_retry_after=30))
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    with pytest.raises(openai.RateLimitError):
        ask()
    assert openai_server.count == 1
    assert slept == []


def test_connection_errors_are_retried(openai_client, openai_server, records):
    openai_server.queue(httpx2.ConnectError("boom"))

    @callm.callm(retry=1)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    assert ask().choices[0].message.content == "Hello from OpenAI"
    assert records()[0].retries == 1


async def test_async_retry(async_anthropic_client, anthropic_server, slept):
    anthropic_server.fail(529)

    @callm.callm(retry=RetryConfig(max_retries=1, base_delay=2, jitter=False))
    async def ask():
        return await async_anthropic_client.messages.create(
            model="claude-haiku-4-5", max_tokens=5, messages=user("x")
        )

    assert (await ask()).content[0].text == "Hello from Claude"
    assert slept == [2]


def test_gemini_errors_are_retried(gemini_client, gemini_server, records):
    gemini_server.fail(503)

    @callm.callm(retry=1)
    def ask():
        return gemini_client.models.generate_content(model="gemini-2.5-flash", contents="x")

    assert ask().text == "Hello from Gemini"
    assert gemini_server.count == 2
    assert records()[0].retries == 1


def test_backoff_is_exponential_and_capped():
    config = RetryConfig(base_delay=1, max_delay=5, jitter=False)
    assert [config.backoff(i) for i in range(5)] == [1, 2, 4, 5, 5]
    jittered = RetryConfig(base_delay=1, max_delay=5)
    assert all(0 <= jittered.backoff(3) <= 5 for _ in range(50))


def test_retry_config_validation():
    with pytest.raises(ConfigurationError):
        RetryConfig(max_retries=-1)
    with pytest.raises(ConfigurationError):
        RetryConfig(base_delay=-0.1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5", 1.5),
        ("250ms", 0.25),
        ("6m0s", 360.0),
        ("1h2m", 3720.0),
        ("30s", 30.0),
        ("", None),
        ("soon", None),
        ("5x", None),
    ],
)
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


def test_retry_after_header_variants():
    assert get_retry_after(StatusError(429, {"retry-after-ms": "1500"})) == 1.5
    assert get_retry_after(StatusError(429, {"Retry-After": "7"})) == 7
    future = email.utils.format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert 25 <= get_retry_after(StatusError(503, {"retry-after": future})) <= 31  # clock slack
    assert get_retry_after(StatusError(500)) is None


def test_openai_rate_limit_reset_uses_the_exhausted_limit():
    headers = {
        "x-ratelimit-remaining-requests": "10",
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "6m0s",
    }
    assert get_retry_after(StatusError(429, headers)) == 360
    headers["x-ratelimit-remaining-tokens"] = "50"
    assert get_retry_after(StatusError(429, headers)) == 1


def test_anthropic_rate_limit_reset_timestamps():
    reset = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat().replace("+00:00", "Z")
    headers = {
        "anthropic-ratelimit-requests-remaining": "0",
        "anthropic-ratelimit-requests-reset": reset,
    }
    assert 15 <= get_retry_after(StatusError(429, headers)) <= 21  # clock slack


def test_google_retry_info_detail():
    class GoogleError(Exception):
        code = 429
        details = {
            "error": {
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}
                ]
            }
        }

    assert get_retry_after(GoogleError()) == 12


def test_classify_and_should_retry():
    config = RetryConfig()
    assert should_retry(StatusError(429), config)[0]
    assert should_retry(StatusError(529), config)[0]
    assert not should_retry(StatusError(401), config)[0]
    assert should_retry(TimeoutError(), config)[0]
    assert should_retry(ConnectionResetError(), config)[0]
    assert not should_retry(ValueError("bug"), config)[0]
    assert not should_retry(BudgetExceeded("x", scope="call", limit=1, estimated=2), config)[0]
    assert not should_retry(KeyboardInterrupt(), config)[0]
    info = classify(StatusError(503))
    assert info.status == 503 and info.is_transient
    assert not classify(StatusError(404)).is_transient
    custom = RetryConfig(retry_on_status=frozenset({418}))
    assert should_retry(StatusError(418), custom)[0]
    assert not should_retry(StatusError(429), custom)[0]


def test_real_sdk_exceptions_are_classified(
    openai_client, openai_server, anthropic_client, anthropic_server
):
    import anthropic
    import openai

    openai_server.fail(429, headers={"retry-after": "2"})
    with pytest.raises(openai.RateLimitError) as oai:
        openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))
    assert classify(oai.value).status == 429
    assert classify(oai.value).retry_after == 2

    anthropic_server.fail(529)
    with pytest.raises(anthropic.APIStatusError) as ant:
        anthropic_client.messages.create(model="claude-haiku-4-5", max_tokens=1, messages=user("x"))
    assert classify(ant.value).status == 529
    assert classify(ant.value).is_transient
    time.sleep(0)
