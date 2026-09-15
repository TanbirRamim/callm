"""Pricing, cost tracking, max_cost and budgets."""

from __future__ import annotations

import json
import threading

import pytest

import callm
from callm import Budget, BudgetExceeded, pricing
from callm.api import build_request
from callm.errors import ConfigurationError
from callm.types import Usage
from helpers import openai_body, user


@pytest.mark.parametrize(
    ("provider", "model", "expected_input"),
    [
        ("openai", "gpt-4o", 2.5),
        ("openai", "gpt-4o-2024-08-06", 2.5),
        ("openai", "gpt-4o-mini-2024-07-18", 0.15),
        ("anthropic", "claude-sonnet-5", 2.0),
        ("anthropic", "claude-haiku-4-5-20251001", 1.0),
        ("anthropic", "claude-opus-4-5@20251101", 5.0),
        ("google", "gemini-2.5-flash", 0.3),
        ("gemini", "models/gemini-2.5-pro", 1.25),
    ],
)
def test_bundled_prices(provider, model, expected_input):
    price = pricing.get_price(provider, model)
    assert price is not None
    assert price.input == expected_input


def test_unknown_models_have_no_price(caplog):
    assert pricing.get_price("openai", "my-private-finetune") is None
    assert pricing.cost_for("openai", "my-private-finetune", Usage(1, 1)) is None
    assert "no price known" in caplog.text


def test_cost_includes_cached_tokens():
    price = pricing.ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
    usage = Usage(
        input_tokens=1000, output_tokens=500, cache_read_tokens=10_000, cache_write_tokens=2000
    )
    expected = (1000 * 3 + 500 * 15 + 10_000 * 0.3 + 2000 * 3.75) / 1_000_000
    assert price.cost(usage) == pytest.approx(expected)
    no_cache_price = pricing.ModelPrice(input=1.0, output=2.0)
    assert no_cache_price.cost(Usage(cache_read_tokens=1_000_000)) == pytest.approx(1.0)


def test_set_price_overrides(openai_client, openai_server, records):
    callm.set_price("gpt-4o-mini", input=1000.0, output=0.0)
    assert callm.get_price("openai", "gpt-4o-mini-2024-07-18").input == 1000.0

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("x"))

    ask()
    assert records()[0].cost_usd == pytest.approx(12 * 1000 / 1_000_000)
    with pytest.raises(ValueError):
        callm.set_price("x", input=-1, output=1)


def test_openai_cached_prompt_tokens_are_priced_separately(openai_client, openai_server, records):
    openai_server.queue(
        openai_body(model="gpt-4o", prompt_tokens=1000, completion_tokens=100, cached_tokens=800)
    )

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o", messages=user("x"))

    ask()
    record = records()[0]
    assert record.input_tokens == 200
    assert record.cache_read_tokens == 800
    assert record.cost_usd == pytest.approx((200 * 2.5 + 800 * 1.25 + 100 * 10) / 1_000_000)


def test_update_pricing_from_litellm_format(tmp_path):
    source = tmp_path / "prices.json"
    source.write_text(
        json.dumps(
            {
                "sample_spec": {"note": "ignored"},
                "my-model": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 2e-6,
                },
                "gpt-image-1": {
                    "litellm_provider": "openai",
                    "mode": "image_generation",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 1e-6,
                },
                "gemini/gemini-9": {
                    "litellm_provider": "gemini",
                    "mode": "chat",
                    "input_cost_per_token": 5e-7,
                    "output_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 5e-8,
                },
            }
        )
    )
    count = pricing.update_pricing(source.as_uri())
    assert count == 2
    assert pricing.get_price("openai", "my-model").output == 2.0
    assert pricing.get_price("google", "gemini-9").cache_read == 0.05
    assert pricing.get_price("openai", "gpt-4o") is not None  # bundled prices still present
    assert pricing.pricing_metadata()["user_fetched"]


def test_update_pricing_rejects_bad_sources(tmp_path):
    with pytest.raises(ValueError):
        pricing.update_pricing("ftp://example.com/prices.json")
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="no usable models"):
        pricing.update_pricing(empty.as_uri())


def test_estimates_use_max_tokens_when_set():
    request = build_request("gpt-4o", "x" * 350, max_tokens=1000)
    cost, input_tokens, output_tokens = pricing.estimate_cost(request, assumed_output_tokens=10)
    assert output_tokens == 1000
    assert input_tokens >= 100
    assert cost == pytest.approx((input_tokens * 2.5 + 1000 * 10) / 1_000_000)
    _, _, assumed = pricing.estimate_cost(build_request("gpt-4o", "x"), assumed_output_tokens=77)
    assert assumed == 77


def test_max_cost_blocks_before_sending(openai_client, openai_server, records):
    @callm.callm(max_cost=0.01)
    def expensive():
        return openai_client.chat.completions.create(
            model="gpt-4o", max_tokens=100_000, messages=user("write a novel")
        )

    with pytest.raises(BudgetExceeded) as info:
        expensive()
    assert openai_server.count == 0
    assert info.value.scope == "call"
    assert info.value.estimated > 0.01
    assert records()[0].status == "error"


def test_max_cost_allows_cheap_calls(openai_client, openai_server):
    @callm.callm(max_cost=0.25)
    def cheap():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=100, messages=user("hi")
        )

    cheap()
    assert openai_server.count == 1


def test_budget_accumulates_and_blocks(openai_client, openai_server):
    callm.set_price("gpt-4o-mini", input=1_000_000, output=0)  # $1 per input token
    budget = Budget(40.0, name="team")

    @callm.callm(budget=budget, retry=False)
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1, messages=user("hi")
        )

    ask()  # actual cost: 12 input tokens -> $12
    assert budget.spent == pytest.approx(12)
    assert budget.reserved == 0
    ask()
    ask()
    assert budget.spent == pytest.approx(36)
    with pytest.raises(BudgetExceeded) as info:
        ask()
    assert info.value.scope == "budget:team"
    assert openai_server.count == 3


def test_failed_calls_release_their_reservation(openai_client, openai_server):
    import openai

    budget = Budget(1.0)
    openai_server.fail(400)

    @callm.callm(budget=budget)
    def ask():
        return openai_client.chat.completions.create(model="gpt-4o-mini", messages=user("hi"))

    with pytest.raises(openai.BadRequestError):
        ask()
    assert budget.spent == 0
    assert budget.reserved == 0


def test_context_budgets_and_budget_for(openai_client, openai_server):
    callm.set_price("gpt-4o-mini", input=100_000, output=0)  # 12 tokens -> $1.20

    @callm.callm
    def ask():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1, messages=user("hi")
        )

    with callm.budget(2.0) as session:
        ask()
        ask()  # the estimate still fits: actual spend may overshoot by one call
        with pytest.raises(BudgetExceeded):
            ask()  # already over the limit: refused before sending
    assert session.spent == pytest.approx(2.4)
    assert openai_server.count == 2
    ask()  # outside the scope: no limit

    user_budget = callm.budget_for("user:42", limit=1.5)
    assert callm.budget_for("user:42") is user_budget
    with user_budget:
        ask()
    assert user_budget.spent == pytest.approx(1.2)
    with pytest.raises(ConfigurationError):
        callm.budget_for("user:unknown")


def test_budget_reservations_are_thread_safe():
    budget = Budget(10.0)
    successes = []

    def worker():
        try:
            reservation = budget.reserve(1.0)
        except BudgetExceeded:
            return
        successes.append(1)
        reservation.commit(1.0)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 10
    assert budget.spent == pytest.approx(10.0)
    assert budget.remaining == 0


def test_budget_validation_and_repr():
    with pytest.raises(ConfigurationError):
        Budget(-1)
    with pytest.raises(ConfigurationError):
        Budget(True)  # type: ignore[arg-type]
    budget = Budget(1, name="x")
    budget.add(0.25)
    assert "0.250000" in repr(budget)
    budget.reset()
    assert budget.spent == 0


async def test_async_budget_scope(async_openai_client, openai_server):
    callm.set_price("gpt-4o-mini", input=100_000, output=0)

    @callm.callm
    async def ask():
        return await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1, messages=user("x")
        )

    async with callm.budget(0.1) as scope:
        with pytest.raises(BudgetExceeded):
            await ask()  # the estimate alone exceeds the limit
    assert scope.spent == 0
    assert openai_server.count == 0
