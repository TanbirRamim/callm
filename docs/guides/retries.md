# Retries

```python
@callm(retry=3)                 # up to 3 retries (4 attempts)
@callm(retry=False)             # no retries
@callm(retry=RetryConfig(...))  # full control
```

`@callm()` without options retries twice (change the default with
`callm.configure(default_retries=...)`).

## What is retried

| Condition | Retried by default |
|---|---|
| HTTP 408, 409, 425, 429, 500, 502, 503, 504, 529 (Anthropic "overloaded") | yes |
| Timeouts (`TimeoutError`, SDK `APITimeoutError`, httpx timeouts) | yes |
| Connection errors (`APIConnectionError`, `ConnectionError`, httpx connect/read errors) | yes |
| Other 4xx (bad request, auth, not found) | no |
| callm policy errors (`BudgetExceeded`, `PIIDetectedError`, ...) | never |

## Backoff

Delays use exponential backoff with *full jitter*: retry *n* sleeps a random duration between 0
and `min(max_delay, base_delay * 2**n)`. Jitter spreads retries from many workers so they do
not hit the provider in synchronized waves.

When the provider says how long to wait, callm waits exactly that long instead:

| Hint | Provider |
|---|---|
| `retry-after-ms` header | OpenAI |
| `retry-after` header (seconds or HTTP date) | all |
| `x-ratelimit-reset-requests` / `-tokens` for the exhausted limit | OpenAI (429) |
| `anthropic-ratelimit-*-reset` for the exhausted limit | Anthropic (429) |
| `RetryInfo.retryDelay` in the error body | Google |

If the server asks for longer than `max_retry_after` (60 s by default), callm stops retrying
that target immediately — a [fallback](fallback.md) usually serves the user faster.

## RetryConfig

```python
from callm import RetryConfig

RetryConfig(
    max_retries=2,
    base_delay=0.5,          # seconds
    max_delay=30.0,
    jitter=True,
    retry_on_status=frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529}),
    retry_on_timeout=True,
    retry_on_connection_error=True,
    respect_retry_after=True,
    max_retry_after=60.0,
)
```

## SDK built-in retries

The OpenAI and Anthropic SDKs retry twice on their own by default. Those retries happen inside
a single callm attempt. For predictable behaviour and accurate retry counts in telemetry, create
clients with `max_retries=0` and let callm handle retries:

```python
client = openai.OpenAI(max_retries=0)
```

## Functions without an intercepted call

If a decorated function raises a retryable error *without* making an intercepted SDK call (for
example it uses its own HTTP client), callm retries the whole function with the same policy.
Functions that did make intercepted calls are never re-run, because those calls are already
retried individually.
