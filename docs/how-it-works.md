# How it works

## Interception

When a function decorated with `@callm` (or a `with shield(...)` block) runs, callm sets a
*scope* in a `contextvars.ContextVar`. The following SDK methods are instrumented the first time
callm is used — including SDKs imported later, through an import hook:

| SDK | Methods |
|---|---|
| `openai` | `chat.completions.create` (sync and async clients) |
| `anthropic` | `messages.create` (sync and async clients) |
| `google-genai` | `models.generate_content` (sync and `client.aio`) |

Outside a callm scope the instrumented methods call the original implementation directly, so
importing callm never changes code that doesn't use it. Calls made with `with_raw_response`
are always passed through untouched.

Scopes nest. The innermost scope decides caching, retries, fallback and validation, while
protections from enclosing scopes (PII masking, injection detection, the strictest `max_cost` and
all budgets) always apply.

Inside a scope, the SDK keyword arguments are parsed into a provider-neutral `LLMRequest`, sent
through the middleware stack, and the final response is converted back into the SDK's own
response type (`ChatCompletion`, `Message`, `GenerateContentResponse`) — even when it came from
the cache or from a different provider.

!!! note "Threads"
    Context variables follow `asyncio` tasks automatically but not threads. When you hand work
    to a thread pool from inside a decorated function, submit
    `contextvars.copy_context().run(fn)` or decorate the function that runs in the thread.

## The middleware stack

```text
Telemetry → Security → Cache → Validator → Fallback → Cost guard → Retry → Transport
```

| Layer | Responsibility | Enabled when |
|---|---|---|
| Telemetry | Times the call and records tokens, cost, savings, retries, fallbacks and flags | always |
| Security | Scores injection risk, masks PII — before anything is cached or sent | `detect_injection` / `block_pii` |
| Cache | Returns a stored response on a hit; stores successful responses | `cache` |
| Validator | Parses into `output_schema`; on failure appends the errors and asks again | `output_schema` |
| Fallback | Moves to the next target when the current one keeps failing | `fallback` |
| Cost guard | Refuses calls whose estimate exceeds `max_cost` or a budget; charges actual cost | always (limits optional) |
| Retry | Backs off and retries transient errors | `retry > 0` |
| Transport | Calls the SDK — the caller's own client for the original provider | always |

The order is deliberate:

- **Security runs before the cache**, so cache keys and stored entries never contain the
  redacted PII.
- **The validator sits inside the cache**, so only responses that passed validation are cached.
- **The cost guard sits inside the fallback layer**, so every target is checked with its own
  price.
- **Retries are innermost**, so each fallback target gets its own retries.

## One implementation for sync and async

Each middleware is a generator that `yield`s *effects* — "sleep for 2 seconds", "call this
function" — and receives the result. A synchronous driver executes effects with `time.sleep`
and direct calls; an asynchronous driver uses `asyncio.sleep` and `await`. The same middleware
code therefore serves `def` and `async def` functions with identical semantics.

## Translation between providers

Requests keep their provider-native keyword arguments while they stay with the same provider
(for example a fallback from `gpt-4o` to `gpt-4o-mini`). When a fallback crosses providers,
the canonical parts are translated: system prompts, user/assistant turns, `max_tokens`,
`temperature`/`top_p` (where the target accepts them) and stop sequences.

Anything that cannot be translated faithfully — tools, images, files, response formats,
`n > 1` — makes callm skip cross-provider fallbacks for that request, so a fallback never
silently changes what you asked the model to do.

## Storage

- **SQLite** (default): `~/.callm/callm.db`, WAL mode, safe across threads, processes and
  `fork()`.
- **Memory**: `callm.configure(storage="memory")`.
- **Redis** for the cache: `callm.configure(cache_store=RedisCacheStore(url))`.
- Your own: implement `CacheStore` / `TelemetryStore` from `callm.storage`.
