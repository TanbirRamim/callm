"""Pipeline drivers, configuration and packaging."""

from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import callm
from callm.config import CacheConfig, build_call_config, configure, get_settings
from callm.errors import ConfigurationError
from callm.pipeline import Invoke, Sleep, run_async, run_sync
from callm.reports import parse_since


def _gen(log):
    log.append("start")
    try:
        value = yield Invoke(sync=lambda: 21, async_=None)
        yield Sleep(0.5)
        try:
            yield Invoke(sync=lambda: 1 / 0)
        except ZeroDivisionError:
            log.append("caught")
        return value * 2
    finally:
        log.append("closed")


def test_sync_driver(slept):
    log: list[str] = []
    assert run_sync(_gen(log)) == 42
    assert log == ["start", "caught", "closed"]
    assert slept == [0.5]


async def test_async_driver_awaits_and_offloads(slept):
    async def fetch():
        await asyncio.sleep(0)
        return "async"

    def gen():
        a = yield Invoke(sync=lambda: "sync", async_=fetch)
        b = yield Invoke(sync=lambda: "thread", offload=True)
        return a, b

    assert await run_async(gen()) == ("async", "thread")
    log: list[str] = []
    assert await run_async(_gen(log)) == 42


def test_driver_propagates_uncaught_errors():
    def gen():
        yield Invoke(sync=lambda: (_ for _ in ()).throw(KeyError("x")))

    with pytest.raises(KeyError):
        run_sync(gen())


def test_build_call_config_normalization():
    config = build_call_config(
        cache=True, retry=4, fallback="claude-haiku", max_cost=1, tags={"a": "b"}
    )
    assert config.cache == CacheConfig()
    assert config.retry.max_retries == 4
    assert config.fallback[0].model == "claude-haiku-4-5"
    assert config.max_cost == 1.0
    assert build_call_config(cache="semantic").cache.semantic is True
    assert build_call_config().retry.max_retries == get_settings().default_retries


def test_configure_validation(tmp_path):
    with pytest.raises(ConfigurationError, match="Unknown setting"):
        configure(colour="blue")
    with pytest.raises(ConfigurationError):
        configure(storage="postgres")
    with pytest.raises(ConfigurationError):
        configure(default_retries=-1)
    configure(default_retries=5, home=str(tmp_path), on_call=print)
    assert get_settings().default_retries == 5
    assert get_settings().home == tmp_path
    assert get_settings().on_call == [print]
    with pytest.raises(ConfigurationError):
        CacheConfig(threshold=1.5)
    with pytest.raises(ConfigurationError):
        CacheConfig(ttl=0)


def test_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLM_HOME", str(tmp_path / "envhome"))
    monkeypatch.setenv("CALLM_TELEMETRY", "0")
    monkeypatch.setenv("CALLM_DISABLED", "true")
    monkeypatch.setenv("CALLM_STORAGE", "memory")
    callm.reset_settings()
    settings = get_settings()
    assert settings.home == tmp_path / "envhome"
    assert settings.telemetry is False
    assert settings.enabled is False
    assert settings.storage == "memory"


def test_parse_since():
    import time
    from datetime import datetime, timezone

    assert abs(parse_since("2h") - (time.time() - 7200)) < 2
    assert parse_since(123.0) == 123.0
    assert parse_since(None) is None
    assert (
        parse_since(datetime(2026, 1, 1, tzinfo=timezone.utc))
        == datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    )
    assert parse_since("2026-01-31") > 0
    with pytest.raises(ValueError):
        parse_since("last tuesday")


def test_public_api_is_complete():
    for name in callm.__all__:
        assert hasattr(callm, name), name


def test_import_has_no_side_effects_and_needs_no_dependencies():
    """``import callm`` must not import SDKs or third-party packages."""
    code = (
        "import sys, callm; "
        "heavy = [m for m in ('openai', 'anthropic', 'google.genai', 'pydantic', 'httpx', 'rich') "
        "if m in sys.modules]; "
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=_env()
    )
    assert result.stdout.strip() == "[]"


def _env() -> dict[str, str]:
    src = str(Path(callm.__file__).parents[1])
    return {**os.environ, "PYTHONPATH": src, "CALLM_STORAGE": "memory"}


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "callm", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert callm.__version__ in result.stdout


def test_sdk_imported_after_callm_is_instrumented():
    """The post-import hook patches SDKs imported lazily inside decorated functions."""
    code = """
import sys, callm, json
from callm import interception
@callm.callm
def lazy():
    import anthropic  # first import happens inside the scope
    return interception.instrumented()
print(json.dumps(lazy()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_env()
    )
    assert result.returncode == 0, result.stderr
    assert "anthropic.resources.messages.Messages.create" in result.stdout
    importlib.invalidate_caches()
