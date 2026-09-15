"""Regenerate ``src/callm/data/pricing.json`` from LiteLLM's public price list.

Usage::

    python scripts/update_bundled_pricing.py              # download the latest list
    python scripts/update_bundled_pricing.py prices.json  # convert a local copy
"""

from __future__ import annotations

import datetime
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from callm.pricing import LITELLM_PRICES_URL, convert_litellm  # noqa: E402


def main() -> int:
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(LITELLM_PRICES_URL, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
    models = convert_litellm(raw)
    payload = {
        "source": LITELLM_PRICES_URL,
        "fetched": datetime.date.today().isoformat(),
        "unit": "USD per 1M tokens",
        "models": models,
    }
    dest = ROOT / "src" / "callm" / "data" / "pricing.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = ", ".join(f"{provider}={len(items)}" for provider, items in models.items())
    print(f"wrote {dest.relative_to(ROOT)} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
