"""The ``callm`` command line interface."""

from __future__ import annotations

import io
import json
import time

import pytest

from callm import cli
from callm.config import get_cache_store, get_telemetry_store
from callm.storage.base import AggregateRow, CacheEntry
from callm.types import CallRecord


@pytest.fixture
def populated():
    store = get_telemetry_store()
    now = time.time()
    for _ in range(3):
        store.record(
            CallRecord(
                function="app.summarize",
                provider="openai",
                model="gpt-4o",
                input_tokens=400_000,
                output_tokens=0,
                cost_usd=4.77,
                timestamp=now,
            )
        )
    store.record(
        CallRecord(
            function="app.summarize",
            provider="openai",
            model="gpt-4o",
            cache_hit=True,
            cost_usd=0.0,
            saved_usd=2.8,
            timestamp=now,
        )
    )
    store.record(
        CallRecord(
            function="app.chat",
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=380_000,
            cost_usd=4.87,
            retries=1,
            fallback_from="openai/gpt-4o",
            pii_redactions={"email": 1},
            timestamp=now,
        )
    )
    store.record(
        CallRecord(
            function="app.chat",
            provider="anthropic",
            model="claude-sonnet-5",
            status="error",
            error_type="RateLimitError",
            timestamp=now - 86400 * 10,
        )
    )
    return store


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_stats_table(populated, capsys):
    code, out, _ = run(capsys, "stats")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == [
        "Provider",
        "Calls",
        "Tokens",
        "Cost",
        "Cache",
        "Hits",
        "Saved",
        "Errors",
        "Latency",
    ]
    assert lines[2].startswith("openai")
    assert (
        "1.2M" in lines[2]
        and "$14.31" in lines[2]
        and "1 (25%)" in lines[2]
        and "$2.80" in lines[2]
    )
    assert lines[-1].startswith("Total")
    assert "6" in lines[-1]


def test_stats_json_and_filters(populated, capsys):
    code, out, _ = run(capsys, "stats", "--json", "--by", "function", "--since", "7d")
    payload = json.loads(out)
    assert code == 0
    assert {row["key"] for row in payload["rows"]} == {"app.summarize", "app.chat"}
    assert payload["total"]["calls"] == 5
    code, out, _ = run(capsys, "stats", "--json", "--function", "app.chat")
    assert json.loads(out)["total"]["calls"] == 2


def test_stats_empty(capsys):
    code, out, _ = run(capsys, "stats")
    assert code == 0 and "No calls recorded" in out


def test_calls_listing(populated, capsys):
    code, out, _ = run(capsys, "calls", "--limit", "2")
    assert code == 0
    assert len(out.strip().splitlines()) == 2
    code, out, _ = run(capsys, "calls", "--json", "--function", "app.chat")
    records = json.loads(out)
    assert records[0]["fallback_from"] == "openai/gpt-4o"
    code, out, _ = run(capsys, "calls", "--since", "1h")
    assert "fallback-from=openai/gpt-4o" in out and "pii=email:1" in out and "cache-hit" in out


def test_cache_commands(capsys):
    get_cache_store().set(CacheEntry(key="k", response={"text": "x"}))
    code, out, _ = run(capsys, "cache", "stats", "--json")
    assert json.loads(out)["entries"] == 1
    code, out, _ = run(capsys, "cache", "clear")
    assert "Removed 1" in out
    code, out, _ = run(capsys, "cache", "stats")
    assert "entries: 0" in out


def test_pricing_commands(capsys, tmp_path):
    code, out, _ = run(capsys, "pricing", "show", "claude-sonnet-5")
    assert code == 0 and json.loads(out)["input"] == 2.0
    code, _, err = run(capsys, "pricing", "show", "openai/not-a-model")
    assert code == 1 and "No price" in err
    code, _, err = run(capsys, "pricing", "show")
    assert code == 2
    code, out, _ = run(capsys, "pricing", "info")
    assert json.loads(out)["bundled_models"] > 50
    source = tmp_path / "p.json"
    source.write_text(
        json.dumps(
            {
                "x-model": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 1e-6,
                }
            }
        )
    )
    code, out, _ = run(capsys, "pricing", "update", "--url", source.as_uri())
    assert code == 0 and "1 models" in out


def test_telemetry_clear_info_version_and_errors(populated, capsys):
    code, out, _ = run(capsys, "telemetry", "clear")
    assert "Removed 6" in out
    code, out, _ = run(capsys, "info")
    info = json.loads(out)
    assert info["optional_packages"]["openai"] is True
    assert "openai: Completions.create" in info["instrumented_methods"]
    code, out, _ = run(capsys)
    assert "usage" in out
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    code, _, err = run(capsys, "stats", "--since", "yesterday-ish")
    assert code == 1 and "cannot parse" in err


def test_home_option(tmp_path, capsys):
    code, out, _ = run(capsys, "--home", str(tmp_path / "other"), "info")
    assert json.loads(out)["home"] == str(tmp_path / "other")


def test_formatting_helpers():
    assert cli.format_tokens(999) == "999"
    assert cli.format_tokens(1_500) == "1.5K"
    assert cli.format_tokens(380_000) == "380K"
    assert cli.format_tokens(1_580_000) == "1.58M"
    assert cli.format_money(19.19) == "$19.19"
    assert cli.format_money(0.0012) == "$0.0012"
    buffer = io.StringIO()
    cli.render_table([AggregateRow("openai", calls=1, cost_usd=1.0)], "provider", out=buffer)
    assert "Total" not in buffer.getvalue()
