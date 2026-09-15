"""Reproducible, offline benchmarks for callm.

Runs the real OpenAI SDK against an in-process fake server (no network, no API key) and
measures three things:

1. overhead  - latency callm adds per call (the fake server itself answers instantly)
2. caching   - spend with and without the cache on repetitive and non-repetitive traffic
3. failures  - request success rate when the provider fails transiently

    pip install "callm-toolkit[openai,validation]"
    python benchmarks/run.py            # prints a Markdown report
    python benchmarks/run.py --quick    # smaller sample sizes
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import tempfile
import time

os.environ["CALLM_STORAGE"] = "memory"
os.environ.setdefault("CALLM_HOME", tempfile.mkdtemp(prefix="callm-bench-"))

import httpx2
import openai

import callm
from callm import pipeline, pricing
from callm.types import Usage

MODEL = "gpt-4o-mini"
PROMPT_TOKENS, COMPLETION_TOKENS = 850, 180  # a typical support-bot turn


def completion_body(text: str = "ok") -> dict:
    return {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }


def make_client(handler) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="bench",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def ok_handler(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=completion_body())


# --------------------------------------------------------------------------- 1. overhead


def bench_overhead(samples: int) -> list[tuple[str, float]]:
    client = make_client(ok_handler)
    messages = [
        {"role": "system", "content": "You are a helpful support assistant for Acme Cloud."},
        {
            "role": "user",
            "content": "Hi, I'm jane@example.com. Why was I charged twice this month?",
        },
    ]

    def raw() -> object:
        return client.chat.completions.create(model=MODEL, messages=messages)

    variants = {
        "SDK call without callm": raw,
        "@callm() defaults, in-memory telemetry": callm.callm()(raw),
        "@callm() defaults, SQLite telemetry (default storage)": "sqlite",
        "+ block_pii + detect_injection": callm.callm(block_pii=True, detect_injection=True)(raw),
        "+ max_cost + budget": callm.callm(
            block_pii=True, detect_injection=True, max_cost=1.0, budget=callm.Budget(1e9)
        )(raw),
    }
    results = []
    for name, fn in variants.items():
        if fn == "sqlite":
            from callm.storage.sqlite import SQLiteStorage

            callm.configure(storage=SQLiteStorage(os.path.join(tempfile.mkdtemp(), "bench.db")))
            fn = callm.callm()(raw)
        else:
            callm.configure(storage="memory")
        for _ in range(20):  # warm up
            fn()
        timings = []
        for _ in range(samples):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000)
        results.append((name, statistics.median(timings)))
    callm.configure(storage="memory")
    return results


# --------------------------------------------------------------------------- 2. caching


def bench_cache(requests: int, distinct: int, *, skewed: bool, seed: int = 7) -> dict:
    """Traffic over ``distinct`` possible prompts.

    ``skewed=True`` draws Zipf-like (a few questions asked very often, as in support or FAQ
    traffic); ``skewed=False`` draws uniformly (little natural repetition).
    """
    rng = random.Random(seed)
    weights = [1 / (rank + 1) for rank in range(distinct)] if skewed else None
    questions = [f"Support question #{rank}: how do I fix this?" for rank in range(distinct)]
    workload = rng.choices(questions, weights=weights, k=requests)
    callm.clear_cache()

    calls = {"count": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["count"] += 1
        return httpx2.Response(200, json=completion_body())

    client = make_client(handler)

    @callm.callm(cache=True, name="bench.cached")
    def cached(question: str) -> object:
        return client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": question}]
        )

    for question in workload:
        cached(question)

    price = pricing.get_price("openai", MODEL)
    assert price is not None
    per_call = price.cost(Usage(input_tokens=PROMPT_TOKENS, output_tokens=COMPLETION_TOKENS))
    return {
        "requests": requests,
        "distinct": len(set(workload)),
        "provider_calls": calls["count"],
        "cost_without_cache": per_call * requests,
        "cost_with_cache": per_call * calls["count"],
    }


# --------------------------------------------------------------------------- 3. failures


def bench_failures(requests: int, failure_rate: float, seed: int = 11) -> list[tuple[str, float]]:
    rng = random.Random(seed)

    def flaky(request: httpx2.Request) -> httpx2.Response:
        if rng.random() < failure_rate:
            return httpx2.Response(503, json={"error": {"message": "overloaded"}})
        return httpx2.Response(200, json=completion_body())

    primary = make_client(flaky)
    backup = make_client(ok_handler)
    callm.configure(clients={"openai": backup})  # fallback target uses a healthy deployment

    def raw() -> object:
        return primary.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

    variants = {
        "SDK call without callm": raw,
        "@callm(retry=2)": callm.callm(retry=2)(raw),
        "@callm(retry=2, fallback=['openai/gpt-4o-mini'])": callm.callm(
            retry=2, fallback=["openai/gpt-4o-mini"]
        )(raw),
    }
    results = []
    for name, fn in variants.items():
        ok = 0
        for _ in range(requests):
            try:
                fn()
                ok += 1
            except Exception:
                pass
        results.append((name, ok / requests))
    return results


# --------------------------------------------------------------------------- report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    pipeline.sleep_sync = lambda seconds: None  # measure logic, not backoff waiting
    samples, cache_requests, failure_requests = (
        (200, 1000, 1000) if args.quick else (1000, 5000, 5000)
    )

    overhead = bench_overhead(samples)
    caches = [
        (
            "Support/FAQ traffic (Zipf over 500 questions)",
            bench_cache(cache_requests, distinct=500, skewed=True),
        ),
        (
            "Mostly unique prompts (uniform over 100,000)",
            bench_cache(cache_requests, distinct=100_000, skewed=False),
        ),
    ]
    failures = bench_failures(failure_requests, failure_rate=0.2)

    if args.json:
        print(
            json.dumps({"overhead_ms": overhead, "cache": caches, "failures": failures}, indent=2)
        )
        return

    baseline = overhead[0][1]
    print(
        f"callm {callm.__version__} · Python {platform.python_version()} · {platform.system()} {platform.machine()}\n"
    )
    print("### Overhead per call (median)\n")
    print("| Configuration | Median latency | Added by callm |")
    print("|---|---:|---:|")
    for name, value in overhead:
        added = "—" if name.startswith("SDK") else f"+{value - baseline:.3f} ms"
        print(f"| {name} | {value:.3f} ms | {added} |")

    print("\n### Exact-match cache (gpt-4o-mini, 850 input + 180 output tokens per call)\n")
    print("| Workload | Requests | Provider calls | Cost without cache | Cost with cache | Saved |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, cache in caches:
        saved = 1 - cache["cost_with_cache"] / cache["cost_without_cache"]
        print(
            f"| {label} | {cache['requests']:,} | {cache['provider_calls']:,} | "
            f"${cache['cost_without_cache']:.2f} | ${cache['cost_with_cache']:.2f} | {saved:.0%} |"
        )

    print("\n### Success rate with 20% transient provider failures\n")
    print("| Configuration | Successful requests |")
    print("|---|---:|")
    for name, rate in failures:
        print(f"| {name} | {rate:.1%} |")


if __name__ == "__main__":
    main()
