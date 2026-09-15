"""``callm`` command line interface.

Commands::

    callm stats [--since 7d] [--by provider|model|function] [--function NAME] [--json]
    callm calls [--limit 20] [--since 24h] [--function NAME] [--json]
    callm cache stats|clear [--json]
    callm pricing show MODEL | update [--url URL] | info
    callm telemetry clear
    callm info
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from callm.__about__ import __version__
from callm.storage.base import AggregateRow


def format_tokens(value: int) -> str:
    """``1_234_567 -> "1.23M"``, ``380_000 -> "380K"``, ``1_500 -> "1.5K"``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}K"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return str(value)


def format_money(value: float) -> str:
    if value and abs(value) < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def total_row(rows: Sequence[AggregateRow]) -> AggregateRow:
    total = AggregateRow("Total")
    for row in rows:
        total.merge(row)
    return total


def _cells(row: AggregateRow) -> list[str]:
    return [
        row.key,
        f"{row.calls:,}",
        format_tokens(row.total_tokens),
        format_money(row.cost_usd),
        f"{row.cache_hits:,} ({row.cache_hit_rate:.0%})",
        format_money(row.saved_usd),
        f"{row.errors:,}",
        f"{row.avg_latency_ms:,.0f}ms",
    ]


def render_table(rows: Sequence[AggregateRow], by: str, out: TextIO | None = None) -> None:
    """Print the stats table (uses ``rich`` on a terminal when it is installed)."""
    stream = out or sys.stdout
    headers = [
        by.capitalize(),
        "Calls",
        "Tokens",
        "Cost",
        "Cache Hits",
        "Saved",
        "Errors",
        "Latency",
    ]
    body = [_cells(row) for row in rows]
    footer = _cells(total_row(rows))
    show_total = len(rows) > 1

    if out is None and sys.stdout.isatty() and importlib.util.find_spec("rich") is not None:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_footer=show_total, box=None, header_style="bold", pad_edge=False)
        for index, header in enumerate(headers):
            table.add_column(
                header,
                justify="left" if index == 0 else "right",
                footer=footer[index] if show_total else "",
                footer_style="bold",
            )
        for cells in body:
            table.add_row(*cells)
        Console().print(table)
        return

    widths = [len(h) for h in headers]
    for cells in [*body, footer]:
        widths = [max(w, len(c)) for w, c in zip(widths, cells, strict=True)]

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "   ".join(parts).rstrip()

    rule = "─" * len(line(headers))
    print(line(headers), file=stream)
    print(rule, file=stream)
    for cells in body:
        print(line(cells), file=stream)
    if show_total:
        print(rule, file=stream)
        print(line(footer), file=stream)


def _cmd_stats(args: argparse.Namespace) -> int:
    from callm.reports import stats

    rows = stats(args.by, since=args.since, function=args.function)
    if args.json:
        payload = {
            "by": args.by,
            "rows": [row.to_dict() for row in rows],
            "total": total_row(rows).to_dict(),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if not rows:
        print("No calls recorded yet. Decorate a function with @callm and make a call.")
        return 0
    render_table(rows, args.by)
    return 0


def _describe(record: Any) -> str:
    when = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    flags = []
    if record.cache_hit:
        flags.append("cache-hit")
    if record.retries:
        flags.append(f"retries={record.retries}")
    if record.fallback_from:
        flags.append(f"fallback-from={record.fallback_from}")
    if record.validation_retries:
        flags.append(f"revalidated={record.validation_retries}")
    if record.pii_redactions:
        flags.append(
            "pii=" + ",".join(f"{k}:{v}" for k, v in sorted(record.pii_redactions.items()))
        )
    if record.injection_flagged:
        flags.append(f"injection={record.injection_score or 0:.2f}")
    cost = "-" if record.cost_usd is None else format_money(record.cost_usd)
    status = record.status if record.status == "ok" else f"{record.status}:{record.error_type}"
    parts = [
        when,
        f"{record.provider}/{record.model}",
        f"{format_tokens(record.total_tokens)} tok",
        cost,
        f"{record.latency_ms:.0f}ms",
        status,
        record.function or "",
        " ".join(flags),
    ]
    return "  ".join(part for part in parts if part)


def _cmd_calls(args: argparse.Namespace) -> int:
    from callm.config import get_telemetry_store
    from callm.reports import parse_since

    records = get_telemetry_store().query(
        since=parse_since(args.since), function=args.function, limit=args.limit
    )
    if args.json:
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return 0
    if not records:
        print("No calls recorded.")
        return 0
    for record in records:
        print(_describe(record))
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    from callm.config import get_cache_store

    store = get_cache_store()
    if args.action == "clear":
        print(f"Removed {store.clear():,} cached response(s).")
        return 0
    info = store.stats()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


def _cmd_pricing(args: argparse.Namespace) -> int:
    from callm import pricing
    from callm.providers.registry import parse_target

    if args.action == "update":
        count = pricing.update_pricing(args.url) if args.url else pricing.update_pricing()
        print(f"Updated prices for {count} models in {pricing.user_pricing_path()}")
        return 0
    if args.action == "info":
        print(json.dumps(pricing.pricing_metadata(), indent=2))
        return 0
    if not args.model:
        print("usage: callm pricing show MODEL", file=sys.stderr)
        return 2
    target = parse_target(args.model)
    price = pricing.get_price(target.provider, target.model)
    if price is None:
        print(f"No price known for {target}. Try `callm pricing update`.", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"model": target.label, "unit": "USD per 1M tokens", **price.to_dict()}, indent=2
        )
    )
    return 0


def _cmd_telemetry(args: argparse.Namespace) -> int:
    from callm.config import get_telemetry_store

    print(f"Removed {get_telemetry_store().clear_records():,} call record(s).")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from callm.config import get_settings
    from callm.interception import PATCH_TARGETS
    from callm.providers.registry import provider_names

    settings = get_settings()
    optional = {}
    for module in (
        "openai",
        "anthropic",
        "google.genai",
        "pydantic",
        "sentence_transformers",
        "spacy",
        "redis",
        "opentelemetry",
        "tiktoken",
        "rich",
    ):
        try:
            optional[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            optional[module] = False
    storage = (
        settings.storage if isinstance(settings.storage, str) else type(settings.storage).__name__
    )
    info = {
        "version": __version__,
        "python": sys.version.split()[0],
        "home": str(settings.home),
        "storage": storage,
        "telemetry": settings.telemetry,
        "otel": settings.otel,
        "enabled": settings.enabled,
        "providers": provider_names(),
        "instrumented_methods": sorted(
            {f"{t.provider}: {t.cls}.{t.method}" for t in PATCH_TARGETS}
        ),
        "optional_packages": optional,
    }
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="callm", description="callm - the production toolkit for LLM calls"
    )
    parser.add_argument("--version", action="version", version=f"callm {__version__}")
    parser.add_argument(
        "--home", type=Path, help="callm data directory (default: $CALLM_HOME or ~/.callm)"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("stats", help="cost, token and cache summary")
    p.add_argument("--by", choices=["provider", "model", "function"], default="provider")
    p.add_argument("--since", help="only calls newer than this, e.g. 24h, 7d or 2026-01-31")
    p.add_argument("--function", help="only calls made by this function")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(handler=_cmd_stats)

    p = sub.add_parser("calls", help="list recent calls")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--since")
    p.add_argument("--function")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_calls)

    p = sub.add_parser("cache", help="inspect or clear the response cache")
    p.add_argument("action", choices=["stats", "clear"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_cache)

    p = sub.add_parser("pricing", help="show or update model prices")
    p.add_argument("action", choices=["show", "update", "info"])
    p.add_argument("model", nargs="?")
    p.add_argument("--url", help="price list URL in LiteLLM format")
    p.set_defaults(handler=_cmd_pricing)

    p = sub.add_parser("telemetry", help="manage recorded calls")
    p.add_argument("action", choices=["clear"])
    p.set_defaults(handler=_cmd_telemetry)

    p = sub.add_parser("info", help="show environment and configuration")
    p.set_defaults(handler=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home is not None:
        from callm.config import configure

        configure(home=args.home)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        code: int = handler(args)
        return code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"callm: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
