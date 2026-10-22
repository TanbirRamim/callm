"""Examples that need an API key explain how to set one instead of raising a traceback."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import callm

EXAMPLES = Path(__file__).parents[1] / "examples"


def _env_without_keys() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OPENAI_", "ANTHROPIC_"))
    }
    # If a check ever regresses, nothing can reach a real (paid) endpoint.
    unreachable = "http://127.0.0.1:9"
    env.update(OPENAI_BASE_URL=unreachable, ANTHROPIC_BASE_URL=unreachable)
    env.update(PYTHONPATH=str(Path(callm.__file__).parents[1]), CALLM_STORAGE="memory")
    return env


@pytest.mark.parametrize(
    ("script", "key"),
    [
        ("quickstart.py", "OPENAI_API_KEY"),
        ("async_budgets.py", "OPENAI_API_KEY"),
        ("structured_output.py", "ANTHROPIC_API_KEY"),
        ("ticket_triage.py", "OPENAI_API_KEY"),
    ],
)
def test_example_without_api_key_explains_setup(script, key):
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    pytest.importorskip("pydantic")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        env=_env_without_keys(),
        timeout=60,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert f"Set {key}" in result.stderr
    assert "examples/offline_demo.py" in result.stderr
