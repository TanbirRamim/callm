# Configuration

Per-call behaviour is configured on `@callm(...)`, `shield(...)` or `complete(...)`. Process-wide
settings use `callm.configure`:

```python
import callm

callm.configure(
    home="~/.callm",               # data directory: callm.db, pricing.json
    storage="sqlite",              # "sqlite" | "memory" | storage object
    cache_store=None,              # separate cache backend, e.g. RedisCacheStore(...)
    telemetry=True,                # persist call records
    otel=False,                    # OpenTelemetry spans
    enabled=True,                  # False: decorated functions run untouched
    default_retries=2,             # retries for @callm() without retry=
    assumed_output_tokens=1024,    # cost estimate when max_tokens is not set
    clients={},                    # provider name -> SDK client for fallbacks / complete()
    async_clients={},
    on_call=[],                    # callables receiving every CallRecord
)
```

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CALLM_HOME` | `~/.callm` | data directory |
| `CALLM_STORAGE` | `sqlite` | `sqlite` or `memory` |
| `CALLM_TELEMETRY` | `1` | `0` disables persisting call records |
| `CALLM_OTEL` | `0` | `1` enables OpenTelemetry spans |
| `CALLM_DISABLED` | `0` | `1` turns callm into a no-op (kill switch) |

Provider SDKs read their own variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY` / `GEMINI_API_KEY`.

## Tests and CI

```python
# conftest.py
import callm

def pytest_configure():
    callm.configure(storage="memory", telemetry=False)
```

Or set `CALLM_DISABLED=1` to run your test suite without callm in the loop.

## Logging

callm logs to the `callm` logger (with a `NullHandler` attached by default). Warnings include
skipped fallbacks, flagged injections, cache or telemetry failures and unknown prices; retries
are logged at `INFO`.

```python
import logging
logging.getLogger("callm").setLevel(logging.INFO)
```
