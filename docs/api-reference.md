# API reference

Everything below is importable from the top-level `callm` package unless noted.

## `@callm`

```python
callm(
    func=None, /, *,
    cache: bool | "exact" | "semantic" | CacheConfig = False,
    retry: bool | int | RetryConfig | None = None,       # None -> default_retries (2)
    fallback: str | Sequence[str | Target] | None = None,
    fallback_on: Callable[[BaseException], bool] | None = None,
    max_cost: float | None = None,                        # USD per call
    budget: Budget | None = None,
    block_pii: bool | PIIConfig = False,
    detect_injection: bool | InjectionConfig = False,
    output_schema: Any = None,                            # any Pydantic-compatible type
    validation_retries: int = 2,
    name: str | None = None,                              # telemetry name
    tags: Mapping[str, str] | None = None,                # telemetry labels
    telemetry: bool = True,
)
```

Usable as `@callm`, `@callm()` or `@callm(...)` on functions, methods and `async def` functions.
Generator functions are rejected. Invalid options raise `ConfigurationError` at decoration time.
The wrapped function exposes its resolved configuration as `fn.callm_config`.

**Returns:** whatever the function returns, or the validated `output_schema` instance.

## `shield`

```python
shield(*, cache=False, retry=None, fallback=None, fallback_on=None, max_cost=None, budget=None,
       block_pii=False, detect_injection=False, output_schema=None, validation_retries=2,
       name="shield", tags=None, telemetry=True)
```

Context manager (`with` / `async with`) applying the options to intercepted SDK calls in the
block. Methods:

- `complete(*, model, messages, provider=None, **kwargs) -> LLMResponse`
- `async acomplete(*, model, messages, provider=None, **kwargs) -> LLMResponse`

`kwargs` accept the same options as `callm.complete` and override the shield's options.

## `complete` / `acomplete`

```python
complete(model, messages, *, provider=None, max_tokens=None, temperature=None, top_p=None,
         stop=None, <all @callm options>, **provider_kwargs) -> LLMResponse
```

- `model`: `"provider/model"`, an inferable model id, or a `Target`.
- `messages`: a string or a list of `{"role": ..., "content": ...}` dicts (or `Message`).
- `provider_kwargs`: passed to the SDK unchanged.

## Configuration objects

### `RetryConfig`

`max_retries=2, base_delay=0.5, max_delay=30.0, jitter=True,
retry_on_status=frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529}), retry_on_timeout=True,
retry_on_connection_error=True, respect_retry_after=True, max_retry_after=60.0`

### `CacheConfig`

`ttl=None, semantic=False, threshold=0.95, embedder=None, store=None, namespace="default",
max_candidates=2000`

### `PIIConfig`

`entities=("email", "phone", "ssn", "credit_card", "ip_address", "iban"), action="mask",
ner=False, ner_model="en_core_web_sm", roles=("system", "user", "assistant", "tool")`

### `InjectionConfig`

`threshold=0.5, action="flag", roles=("user", "tool"), classifier=None,
classifier_label="INJECTION"`

## Budgets

| API | Description |
|---|---|
| `Budget(limit, name=None)` | thread-safe USD limit; `.spent`, `.reserved`, `.remaining`, `.limit` (settable), `.add(cost)`, `.reset()`; usable with `with` / `async with` |
| `budget(limit, name=None)` | create a `Budget` for a `with` block |
| `budget_for(key, limit=None)` | process-wide budget registered under `key` |

## Results and records

### `LLMResponse`

`text, provider, model, usage: Usage, finish_reason, cost, latency_ms, cached, id, raw, parsed`

### `Usage`

`input_tokens` (uncached), `output_tokens`, `cache_read_tokens`, `cache_write_tokens`,
`total_tokens`

### `CallRecord`

See [Telemetry](guides/telemetry.md). `callm.last_call()` returns the latest record in the
current context.

### `Target`

`Target(provider, model)`, `.label` → `"provider/model"`

## Settings and helpers

| API | Description |
|---|---|
| `configure(**settings)` | change global settings (see [Configuration](guides/configuration.md)) |
| `get_settings()` / `reset_settings()` | inspect / restore defaults |
| `stats(by="provider", since=None, function=None)` | aggregated telemetry rows |
| `clear_cache()` | delete all cached responses |
| `get_price(provider, model)` / `set_price(model, *, input, output, cache_read=None, cache_write=None, provider="*")` | prices in USD per 1M tokens |
| `register_provider(name, provider)` | add a provider adapter |
| `OpenAIProvider(name, *, base_url, api_key, api_key_env, max_tokens_param)` | adapter for OpenAI-compatible APIs |
| `redact_pii(text, entities=None, *, ner=False)` | mask PII in a string |
| `detect_injection(text)` | heuristic `InjectionResult(score, matches)` |
| `HashingEmbedder`, `SentenceTransformerEmbedder`, `OpenAIEmbedder` | embedders for semantic caching |

## Storage (`callm.storage`)

| Class | Description |
|---|---|
| `SQLiteStorage(path)` | cache + telemetry in one SQLite file (default backend) |
| `MemoryStorage()` / `MemoryCacheStore()` / `MemoryTelemetryStore()` | in-process |
| `RedisCacheStore(url, *, client=None, prefix="callm")` | shared cache (`callm.storage.redis`) |
| `CacheStore`, `TelemetryStore` | abstract base classes for custom backends |

## Errors

| Exception | Raised when |
|---|---|
| `CallmError` | base class of all callm errors |
| `ConfigurationError` | invalid options (also a `ValueError`) |
| `MissingDependencyError` | an optional package is needed (also an `ImportError`) |
| `BudgetExceeded` (`BudgetExceededError`) | `max_cost` or a budget would be exceeded — `.scope`, `.limit`, `.estimated`, `.spent` |
| `PIIDetectedError` | PII found with `action="block"` — `.entities` |
| `PromptInjectionError` | injection score over threshold with `action="block"` — `.score`, `.matches` |
| `OutputValidationError` | output never matched the schema — `.errors`, `.raw_text`, `.attempts` |
| `AllProvidersFailedError` | primary and all fallbacks failed — `.errors`, `.last_error` |
| `ProviderNotAvailableError` | unknown provider or untranslatable request |

Provider SDK exceptions (for example `openai.RateLimitError`) propagate unchanged when no
fallback is configured.
