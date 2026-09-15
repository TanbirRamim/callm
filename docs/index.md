# callm

**The production toolkit for LLM calls.** Caching, retries, provider fallback, cost tracking,
budgets, PII redaction, prompt-injection detection, structured output validation and telemetry —
in one decorator, on top of the SDKs you already use.

```python
from callm import callm

@callm(
    cache=True,                          # response cache (SQLite by default)
    retry=3,                             # exponential backoff on 429/5xx
    fallback=["anthropic/claude-sonnet-5"],  # auto-switch on failure
    max_cost=0.25,                       # hard budget per call
    block_pii=True,                      # mask emails, phones, SSNs, cards
    detect_injection=True,               # flag prompt injection attempts
    output_schema=MyResponse,            # Pydantic validation + auto-retry
)
def summarize(text: str) -> MyResponse:
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}],
    )
```

## What makes callm different

<div class="grid cards" markdown>

- **Keep your code.** callm intercepts the official `openai`, `anthropic` and `google-genai`
  SDK calls inside decorated functions and returns the SDK's own response objects.
- **No infrastructure.** No proxy to deploy, no database server, no SaaS account. State lives in
  a local SQLite file, in memory, or in Redis when you want a shared cache.
- **Zero required dependencies.** The core is pure standard library. Optional extras add SDKs,
  Pydantic, embeddings, spaCy, Redis and OpenTelemetry.
- **Sync and async.** Identical behaviour for `def` and `async def`, including under
  `asyncio.gather`.

</div>

## Features

| Feature | Summary | Guide |
|---|---|---|
| Retries | Backoff with jitter; honours every provider's rate-limit headers | [Retries](guides/retries.md) |
| Caching | Exact-match by default, semantic on request; SQLite, memory or Redis | [Caching](guides/caching.md) |
| Fallback | Ordered provider chains with request translation | [Fallback](guides/fallback.md) |
| Cost | Per-call cost, `max_cost`, shared budgets per function, session or user | [Cost](guides/cost.md) |
| Security | PII masking and prompt-injection scoring before requests leave your process | [Security](guides/security.md) |
| Validation | Pydantic output schemas with automatic repair | [Structured output](guides/validation.md) |
| Telemetry | Local call records, CLI dashboard, hooks, OpenTelemetry | [Telemetry](guides/telemetry.md) |

## Install

```bash
pip install "callm[openai] @ git+https://github.com/TanbirRamim/callm"   # plus the providers you use
pip install "callm[all] @ git+https://github.com/TanbirRamim/callm"      # everything
```

Continue with the [quickstart](quickstart.md).
