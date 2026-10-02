"""OpenAI structured outputs: ``client.chat.completions.parse``."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

import callm
from callm.errors import BudgetExceededError
from helpers import openai_body


class Invoice(BaseModel):
    vendor: str
    total: float


INVOICE_JSON = json.dumps({"vendor": "ACME", "total": 12.5})


@pytest.fixture(autouse=True)
def _needs_parse(openai_client):
    if not hasattr(openai_client.chat.completions, "parse"):
        pytest.skip("this openai version has no chat.completions.parse")


def ask(client, text="Extract the invoice: ACME, 12.50 EUR"):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        response_format=Invoice,
    )


def test_parse_is_tracked_and_returns_the_parsed_object(openai_client, openai_server, records):
    openai_server.queue(openai_body(text=INVOICE_JSON))

    result = callm.callm(lambda: ask(openai_client))()

    assert type(result).__name__.startswith("ParsedChatCompletion")
    assert result.choices[0].message.parsed == Invoice(vendor="ACME", total=12.5)
    sent = openai_server.last.body
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "Invoice"
    (record,) = records()
    assert record.provider == "openai"
    assert record.cost_usd and record.cost_usd > 0


def test_cache_hit_still_returns_a_parsed_completion(openai_client, openai_server, records):
    openai_server.queue(openai_body(text=INVOICE_JSON))
    cached = callm.callm(cache=True)(lambda: ask(openai_client))

    first = cached()
    second = cached()

    assert openai_server.count == 1
    assert type(second).__name__.startswith("ParsedChatCompletion")
    assert second.choices[0].message.parsed == first.choices[0].message.parsed
    assert records()[0].cache_hit is True


def test_different_response_formats_do_not_share_a_cache_entry(openai_client, openai_server):
    class Receipt(BaseModel):
        vendor: str
        total: float

    openai_server.queue(openai_body(text=INVOICE_JSON), openai_body(text=INVOICE_JSON))

    @callm.callm(cache=True)
    def both():
        a = ask(openai_client)
        b = openai_client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Extract the invoice: ACME, 12.50 EUR"}],
            response_format=Receipt,
        )
        return a, b

    a, b = both()
    assert openai_server.count == 2
    assert isinstance(b.choices[0].message.parsed, Receipt)


def test_pii_is_masked_in_parse_requests(openai_client, openai_server):
    openai_server.queue(openai_body(text=INVOICE_JSON))

    callm.callm(block_pii=True)(lambda: ask(openai_client, "Invoice from ops@acme.example"))()

    assert "ops@acme.example" not in json.dumps(openai_server.last.body)


def test_budget_blocks_parse_before_sending(openai_client, openai_server):
    @callm.callm(max_cost=0.0000001)
    def expensive():
        return openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x" * 4000}],
            response_format=Invoice,
            max_completion_tokens=4000,
        )

    with pytest.raises(BudgetExceededError):
        expensive()
    assert openai_server.count == 0


async def test_async_parse(async_openai_client, openai_server, records):
    openai_server.queue(openai_body(text=INVOICE_JSON))

    @callm.callm
    async def run():
        return await async_openai_client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Extract the invoice"}],
            response_format=Invoice,
        )

    result = await run()
    assert result.choices[0].message.parsed == Invoice(vendor="ACME", total=12.5)
    assert records()[0].provider == "openai"
