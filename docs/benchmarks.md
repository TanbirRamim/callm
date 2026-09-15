# Benchmarks

All numbers come from [`benchmarks/run.py`](https://github.com/TanbirRamim/callm/blob/main/benchmarks/run.py),
which runs the real OpenAI SDK against an in-process fake server. Nothing touches the network,
so the results isolate what callm itself does. Reproduce them with:

```bash
git clone https://github.com/TanbirRamim/callm && cd callm
uv sync
uv run python benchmarks/run.py
```

Results below: callm 0.1.0, Python 3.12, macOS on Apple silicon.

## Overhead per call

The fake server answers instantly, so any difference from the plain SDK call is callm's own work
(parsing the request, running the middleware, recording telemetry).

| Configuration | Median latency | Added by callm |
|---|---:|---:|
| SDK call without callm | 0.328 ms | — |
| `@callm()` defaults, in-memory telemetry | 0.372 ms | +0.043 ms |
| `@callm()` defaults, SQLite telemetry (default storage) | 0.390 ms | +0.062 ms |
| + `block_pii` + `detect_injection` | 0.424 ms | +0.095 ms |
| + `max_cost` + `budget` | 0.428 ms | +0.100 ms |

For comparison, a real LLM API call typically takes hundreds of milliseconds to several seconds,
so callm's overhead is a fraction of a percent of end-to-end latency.

## Cache savings

Each call is priced as `gpt-4o-mini` with 850 input and 180 output tokens. Savings depend
entirely on how often identical requests repeat, so two workloads are shown:

| Workload | Requests | Provider calls | Cost without cache | Cost with cache | Saved |
|---|---:|---:|---:|---:|---:|
| Support/FAQ traffic (Zipf over 500 questions) | 5,000 | 461 | $1.18 | $0.11 | 91% |
| Mostly unique prompts (uniform over 100,000) | 5,000 | 4,891 | $1.18 | $1.15 | 2% |

- The first row models traffic where a few questions are asked constantly: support bots,
  documentation assistants, classification of recurring inputs.
- The second row models prompts that rarely repeat, such as free-form chat with long history.
  There the exact-match cache helps little; don't enable caching expecting savings there.
- Semantic caching (`cache="semantic"`) can raise hit rates on paraphrased questions at the risk of
  returning an answer to a slightly different question. See [Caching](guides/caching.md).

## Reliability under provider failures

Every request to the primary deployment fails with HTTP 503 with probability 20%, independently.
The fallback target is a healthy deployment.

| Configuration | Successful requests |
|---|---:|
| SDK call without callm | 79.9% |
| `@callm(retry=2)` | 99.4% |
| `@callm(retry=2, fallback=["openai/gpt-4o-mini"])` | 100.0% |

Three attempts each failing with probability 0.2 give an expected success rate of
1 − 0.2³ = 99.2%, which matches the measurement. Real outages are often correlated (a provider is
down for minutes rather than failing requests at random); that is the case fallback to another
deployment or provider covers.

## Caveats

- These are synthetic workloads designed to show mechanisms, not predictions for your traffic.
  Measure with your own telemetry: `callm stats` reports cache hit rate and savings per function.
- Backoff sleeps are disabled in the failure benchmark; with real backoff, retried requests take
  longer to succeed.
- Overhead was measured on one machine; absolute numbers vary, the order of magnitude does not.
