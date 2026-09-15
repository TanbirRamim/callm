# Telemetry & CLI

Every intercepted call produces a `CallRecord`:

| Field | Meaning |
|---|---|
| `function` | `module.qualname` of the decorated function, or `name=` |
| `provider`, `model` | who actually answered (after fallback) |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` | token usage (0 on cache hits) |
| `cost_usd`, `estimated_cost_usd`, `saved_usd` | actual cost, pre-call estimate, savings from cache hits |
| `latency_ms` | wall-clock duration including retries |
| `cache_hit`, `retries`, `validation_retries`, `fallback_from` | what the pipeline did |
| `status`, `error_type` | `ok`, `error` or `cancelled` |
| `pii_redactions`, `injection_score`, `injection_flagged` | security results |
| `streamed`, `tags` | streaming flag, your labels |

Records **never contain prompt or completion text**.

## Where records go

- the telemetry store — SQLite at `~/.callm/callm.db` by default (disable with
  `callm.configure(telemetry=False)`, `CALLM_TELEMETRY=0`, or `@callm(telemetry=False)`)
- `on_call` hooks — `callm.configure(on_call=[send_to_metrics])`
- OpenTelemetry — `callm.configure(otel=True)` with `pip install "callm-toolkit[otel]"`
- `callm.last_call()` — the most recent record in the current context

Telemetry failures are logged and never break a call.

## OpenTelemetry

With `otel=True`, callm emits one client span per call on the globally configured tracer
provider, using the GenAI semantic conventions plus callm attributes:

```text
span  chat gpt-4o-2024-08-06
  gen_ai.operation.name   chat
  gen_ai.system           openai
  gen_ai.request.model    gpt-4o-2024-08-06
  gen_ai.usage.input_tokens / output_tokens
  callm.cost_usd  callm.saved_usd  callm.cache_hit  callm.retries  callm.fallback_from
  callm.pii.email  callm.injection_score  callm.tag.<name>
```

Configure an exporter as usual (OTLP to Datadog, Grafana, Honeycomb, ...) and the spans appear
next to your application traces.

## CLI

```console
$ callm stats
Provider    Calls   Tokens     Cost   Cache Hits    Saved   Errors   Latency
────────────────────────────────────────────────────────────────────────────
openai        847     1.2M   $14.32    312 (37%)    $8.41        3     820ms
anthropic     203     380K    $4.87     89 (44%)    $3.82        0     910ms
────────────────────────────────────────────────────────────────────────────
Total       1,050    1.58M   $19.19    401 (38%)   $12.23        3     838ms
```

| Command | Description |
|---|---|
| `callm stats [--by provider\|model\|function] [--since 7d] [--function NAME] [--json]` | aggregated costs, tokens, cache hits, savings |
| `callm calls [--limit 20] [--since 24h] [--function NAME] [--json]` | recent calls with retries, fallbacks and flags |
| `callm cache stats\|clear [--json]` | inspect or clear the response cache |
| `callm pricing show MODEL` / `update [--url URL]` / `info` | prices |
| `callm telemetry clear` | delete call records |
| `callm info` | version, storage, installed extras |
| `callm --home PATH ...` | use another data directory |

`--since` accepts relative durations (`90m`, `24h`, `7d`, `2w`) and ISO dates.

## Programmatic access

```python
import callm
from callm.config import get_telemetry_store

callm.stats(by="function", since="24h")             # list[AggregateRow]
get_telemetry_store().query(function="app.chat", limit=50)
```
