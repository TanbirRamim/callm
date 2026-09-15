# Quickstart

## 1. Install

```bash
pip install "callm-toolkit[openai,validation]"
```

!!! note
    The package is published on PyPI as `callm-toolkit`. You import it as `callm`, and the
    command-line tool is `callm`.

Pick the extras you need: `openai`, `anthropic`, `google`, `validation` (Pydantic), `cache`
(local embeddings for semantic caching), `security` (spaCy name detection), `redis`, `otel`,
`tokens` (tiktoken), `cli` (rich tables) — or `all`.

## 2. Decorate a function that calls an LLM

```python
import openai
from callm import callm

client = openai.OpenAI()

@callm()
def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

print(ask("Give me one fun fact about octopuses."))
```

With no options, `@callm()` already retries rate limits and server errors (twice) and records
the cost of every call.

## 3. Turn on what you need

```python
from pydantic import BaseModel

class Fact(BaseModel):
    animal: str
    fact: str
    source_hint: str

@callm(
    cache=True,
    retry=3,
    fallback=["anthropic/claude-haiku-4-5"],
    max_cost=0.02,
    block_pii=True,
    detect_injection=True,
    output_schema=Fact,
)
def fun_fact(animal: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with JSON: animal, fact, source_hint."},
            {"role": "user", "content": f"A fun fact about {animal}"},
        ],
    )

fact = fun_fact("octopus")      # -> Fact(animal='octopus', fact='...', source_hint='...')
fact = fun_fact("octopus")      # -> served from the cache, $0
```

## 4. Look at the numbers

```console
$ callm stats
Provider   Calls   Tokens     Cost   Cache Hits    Saved   Errors   Latency
───────────────────────────────────────────────────────────────────────────
openai         2      212   $0.0001     1 (50%)  $0.0001        0     305ms
```

```python
import callm

record = callm.last_call()         # telemetry of the most recent call in this context
record.cost_usd, record.cache_hit, record.retries, record.pii_redactions
```

## 5. Async works the same way

```python
client = openai.AsyncOpenAI()

@callm(cache=True, retry=3)
async def ask(question: str):
    return await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
```

## Try everything without an API key

```bash
curl -O https://raw.githubusercontent.com/TanbirRamim/callm/main/examples/offline_demo.py
CALLM_HOME=/tmp/callm-demo python offline_demo.py
CALLM_HOME=/tmp/callm-demo callm stats
```

The script drives the real OpenAI and Anthropic SDKs against a scripted fake server, so you can
watch a retry, a cache hit, a fallback, a blocked over-budget call and a flagged injection
happen, then inspect the recorded costs.

## Next steps

- [How it works](how-it-works.md) — interception and the middleware stack
- [Configuration](guides/configuration.md) — storage location, telemetry, environment variables
- [Cookbook](cookbook/index.md) — complete, realistic examples
