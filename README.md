<div align="center">

<img src="docs/assets/logo.svg" alt="callm" width="96" height="96">

# callm

**The production toolkit for LLM calls.**

Caching, retries, provider fallback, cost tracking, budgets, PII redaction, prompt-injection
detection, structured output validation and telemetry — in one decorator, on top of the SDKs
you already use. No proxy. No database server. Zero required dependencies.

[![CI](https://github.com/TanbirRamim/callm/actions/workflows/ci.yml/badge.svg)](https://github.com/TanbirRamim/callm/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Documentation](https://tanbirramim.github.io/callm) ·
[Quickstart](#quickstart) ·
[Features](#features) ·
[How it works](#how-it-works) ·
[CLI](#the-callm-cli)

</div>

---

```python
from callm import callm

@callm(cache=True, retry=3, fallback=["anthropic/claude-sonnet-5"], max_cost=0.25,
       block_pii=True, detect_injection=True, output_schema=Summary)
def summarize(text: str) -> Summary:
    return openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": text}]
    )
```

`summarize()` still calls OpenAI with your code, your client and your API key — but now every
call is cached, retried on rate limits, failed over to Claude if OpenAI is down, refused if it
would cost more than 25¢, stripped of emails and phone numbers before it leaves your process,
scanned for prompt injection, validated into a `Summary` object, and recorded in a local cost
dashboard.

## Why callm

Every team shipping LLM features writes the same production checklist: retry on 429s, cache
repeated prompts, fall back when a provider has an outage, track spend, keep PII out of prompts,
catch injection attempts, and make the model return valid JSON. That usually means stitching
together `tenacity`, a cache, a PII library, an output-parsing library and a pile of glue code —
or deploying a proxy service.

callm is a library: install it, add a decorator, ship.

- **No rewrite.** Keep calling `openai`, `anthropic` or `google-genai` directly. callm intercepts
  the SDK call inside decorated functions and hands you back the SDK's own response type.
- **No infrastructure.** State lives in a local SQLite file (or memory, or Redis if you want a
  shared cache).
- **No required dependencies.** The core is standard library only; features that need extra
  packages are optional extras.
- **Sync and async.** Same behaviour for `def` and `async def`, including concurrent tasks.

## Quickstart

```bash
pip install "callm[openai,validation] @ git+https://github.com/TanbirRamim/callm"   # or [all]
```

> callm is not on PyPI yet; until the first release, install it from GitHub as shown. Once
> published, `pip install "callm[openai,validation]"` will do the same.

```python
import openai
from callm import callm

client = openai.OpenAI()

@callm()  # zero-config: retries + cost tracking
def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

print(ask("What is the capital of France?"))
```

```console
$ callm stats
Provider   Calls   Tokens    Cost   Cache Hits   Saved   Errors   Latency
──────────────────────────────────────────────────────────────────────────
openai         1       31   $0.0000     0 (0%)   $0.00        0     412ms
```

### Try it without an API key

`examples/offline_demo.py` runs the real OpenAI and Anthropic SDKs against a scripted fake
server and walks through a rate-limit retry, PII masking, schema validation, a cache hit, a
fallback to Claude during an outage, a blocked expensive call and a flagged injection:

```bash
pip install "callm[openai,anthropic,validation] @ git+https://github.com/TanbirRamim/callm"
export CALLM_HOME=/tmp/callm-demo      # keep demo data out of ~/.callm
python examples/offline_demo.py
callm stats
```

## Features

| | Feature | What you get |
|---|---|---|
| 🔄 | **Smart retries** | Exponential backoff with full jitter on 408/409/429/5xx/529, timeouts and connection errors. Honours `retry-after`, `retry-after-ms`, OpenAI `x-ratelimit-reset-*`, Anthropic `anthropic-ratelimit-*-reset` and Gemini `RetryInfo`. |
| 🗄️ | **Response cache** | Exact-match by default; opt-in semantic matching with sentence-transformers, OpenAI embeddings or your own embedder. SQLite, memory or Redis. TTLs. Refusals and invalid output are never cached. |
| 🔀 | **Provider fallback** | Ordered chains across OpenAI, Anthropic, Gemini, Ollama and any OpenAI-compatible endpoint. Requests are translated between providers, and your code still receives the response type of the SDK it called. |
| 💰 | **Cost tracking & budgets** | Per-call cost from a bundled price table (refreshable with `callm pricing update`), including prompt-cache read/write pricing. `max_cost` per call, shared `Budget`s per function, session or user — enforced *before* the request is sent. |
| 🛡️ | **Input security** | PII redaction (emails, phones, SSNs, Luhn-checked cards, IPs, checksum-validated IBANs, optional spaCy names) with stable placeholders. Prompt-injection scoring with a fast heuristic detector and an optional local ML classifier; flag or block. |
| ✅ | **Structured output** | Pass any Pydantic type as `output_schema`. Invalid output is re-requested with the validation errors appended, and the function returns the validated object. |
| 📊 | **Telemetry** | Every call records provider, model, tokens, cost, savings, latency, retries, fallbacks and security flags — never prompt text. `callm stats`, `callm calls`, JSON export, `on_call` hooks and OpenTelemetry spans. |

## Examples

### Structured output with automatic repair

```python
from pydantic import BaseModel
from callm import callm

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

@callm(output_schema=Invoice, validation_retries=2)
def extract(text: str):
    return anthropic_client.messages.create(
        model="claude-sonnet-5", max_tokens=1024,
        messages=[{"role": "user", "content": f"Extract the invoice as JSON:\n{text}"}],
    )

invoice = extract(raw_text)   # -> Invoice(vendor=..., total=..., currency=...)
```

### Fallback across providers

```python
@callm(retry=2, fallback=["anthropic/claude-sonnet-5", "google/gemini-2.5-flash", "ollama/llama3.1"])
def answer(question: str):
    return openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )
```

If OpenAI keeps returning 429/5xx, callm retries, then sends the same conversation to Claude,
then Gemini, then a local model — and `answer()` still returns an OpenAI `ChatCompletion`.
Requests that use provider-specific features (tools, images, response formats) only fall back
to models of the same provider, so a fallback never silently changes what you asked for.

### Budgets per user

```python
import callm

@callm.callm(max_cost=0.05)
def chat(messages): ...

def handle(user_id: str, messages):
    with callm.budget_for(f"user:{user_id}", limit=2.00):
        return chat(messages)   # raises callm.BudgetExceeded once the user spent $2
```

### Security without a decorator

```python
from callm import shield

with shield(block_pii=True, detect_injection=True) as s:
    # Any supported SDK call inside the block is protected...
    openai_client.chat.completions.create(model="gpt-4o-mini", messages=user_messages)
    # ...and you can make provider-neutral calls directly.
    response = s.complete(provider="anthropic", model="claude-sonnet-5", messages=user_messages)
    print(response.text, response.cost)
```

### Direct, provider-neutral calls

```python
response = callm.complete("gemini/gemini-2.5-flash", "Summarize: ...", max_tokens=200, cache=True)
response.text, response.usage.total_tokens, response.cost, response.raw
```

More in the [cookbook](docs/cookbook): a support chatbot, RAG answers with citations and data
extraction.

## How it works

```text
your function                         callm middleware stack
─────────────                         ──────────────────────
@callm(...)                    ┌────────────────────────────────────┐
def summarize():   ─────────►  │ 1. Telemetry     cost, latency, tokens
    client.chat.completions    │ 2. Security      injection scan, PII masking
        .create(...)           │ 3. Cache         exact / semantic lookup
                               │ 4. Validator     parse + re-ask on invalid output
                               │ 5. Fallback      next provider when one keeps failing
                               │ 6. Cost guard    max_cost and budgets, before sending
                               │ 7. Retry         backoff honouring retry-after
                               │ 8. Transport ──► the real SDK call
                               └────────────────────────────────────┘
```

1. `@callm` sets a scope (a `ContextVar`) while your function runs.
2. The official SDK methods (`chat.completions.create`, `messages.create`,
   `models.generate_content`, sync and async) are instrumented on first use. Outside a callm
   scope they call straight through, so importing callm never changes other code.
3. Inside a scope, the SDK call is converted into a provider-neutral request and sent through
   the middleware chain. Each layer is independent and only enabled when configured.
4. The response is converted back into the SDK's native type before it is returned to you.

The middleware is written once as generators and executed by a sync or an async driver, so
`def` and `async def` functions behave identically.

## Configuration

```python
import callm

callm.configure(
    home="~/.callm",          # SQLite database and price overrides
    storage="sqlite",         # "sqlite" | "memory" | a storage object
    telemetry=True,
    otel=False,               # emit OpenTelemetry spans
    default_retries=2,
    on_call=[print],          # called with every CallRecord
)
```

| Environment variable | Effect |
|---|---|
| `CALLM_HOME` | Data directory (default `~/.callm`) |
| `CALLM_STORAGE` | `sqlite` or `memory` |
| `CALLM_TELEMETRY=0` | Do not persist call records |
| `CALLM_OTEL=1` | Export OpenTelemetry spans |
| `CALLM_DISABLED=1` | Kill switch: decorated functions run untouched |

Every option of `@callm` is documented in the [API reference](docs/api-reference.md).

## The `callm` CLI

```console
$ callm stats --since 7d                 # cost, tokens, cache hits and savings by provider
$ callm stats --by function --json       # machine-readable, per function
$ callm calls --limit 20                 # recent calls with retries, fallbacks, flags
$ callm cache stats | clear
$ callm pricing show claude-sonnet-5     # USD per 1M tokens
$ callm pricing update                   # refresh prices from the LiteLLM price list
$ callm info                             # environment and installed extras
```

## Installation extras

| Extra | Installs | Needed for |
|---|---|---|
| `openai` / `anthropic` / `google` | provider SDKs | calling those providers |
| `validation` | `pydantic>=2` | `output_schema` |
| `cache` | `sentence-transformers` | semantic caching with local embeddings |
| `security` | `spacy` | person-name redaction (`PIIConfig(ner=True)`) |
| `redis` | `redis` | shared cache across hosts |
| `otel` | `opentelemetry-api` | OpenTelemetry spans |
| `tokens` | `tiktoken` | exact OpenAI token estimates for the cost guard |
| `cli` | `rich` | prettier `callm stats` tables |
| `all` | everything above | |

## Design decisions and limits

- **The cache is exact-match unless you opt into semantic matching.** A semantic cache with a
  similarity threshold would happily return the answer for *"Summarize https://a.example"* when
  asked about *"https://b.example"*. Use `cache="semantic"` (or `CacheConfig(semantic=True)`) for
  FAQ-style traffic; semantic matches never cross different system prompts, histories,
  parameters or schemas.
- **Injection detection flags by default.** Heuristics have false positives, so the default
  logs a warning and records the score; use `InjectionConfig(action="block")` to refuse. No
  detector catches every attack — keep treating model output as untrusted.
- **Budgets use estimates before a call and actual cost after it.** Set `max_tokens` for a
  tight worst-case estimate; without it the cost guard assumes 1,024 output tokens.
- **Streaming calls** get security, retries on connection setup, budgets and telemetry, but are
  not cached or validated.
- **Threads:** the callm scope follows `asyncio` tasks automatically. Work handed to a thread
  pool needs `contextvars.copy_context().run(...)`, or decorate the function running in the
  thread.
- **Prices** are a bundled snapshot. Run `callm pricing update` or `callm.set_price(...)` for
  current or negotiated rates.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The test suite runs
entirely offline against the real provider SDKs with mocked HTTP transports:

```bash
uv sync
uv run pytest
```

## License

[MIT](LICENSE)
