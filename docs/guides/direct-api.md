# shield & complete

Not everything fits a decorator. callm offers two more entry points with the same options.

## `shield` — protect a block of code

```python
from callm import shield

with shield(block_pii=True, detect_injection=True, name="support-bot"):
    reply = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    summary = anthropic_client.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=messages)
```

Every supported SDK call inside the block goes through the pipeline. Shields work with
`async with`, and one shield object can be used concurrently by several tasks.

## Nesting

Scopes nest: a decorated function called inside a `shield` (or inside another decorated
function) uses its own cache, retry, fallback and validation settings, but **protections
accumulate** — PII masking and injection detection enabled by any enclosing scope stay on, the
strictest `max_cost` applies, and every enclosing budget is charged. `callm.complete(...)` called
inside a scope inherits those protections too.

## `complete` — provider-neutral calls

```python
import callm

response = callm.complete(
    "anthropic/claude-sonnet-5",
    [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain retries with jitter in one sentence."},
    ],
    max_tokens=200,
    cache=True,
    fallback=["gpt-4o-mini"],
)

response.text            # "..."
response.provider        # "anthropic"
response.model           # "claude-sonnet-5"
response.usage           # Usage(input_tokens=..., output_tokens=..., ...)
response.cost            # 0.00123
response.cached          # False
response.raw             # the anthropic.types.Message
response.parsed          # the validated object when output_schema is set
```

- `messages` may be a string (a single user message) or a list of `{"role", "content"}` dicts.
- Canonical parameters: `max_tokens`, `temperature`, `top_p`, `stop`.
- Any other keyword argument is passed to the provider SDK unchanged (`tools=...`,
  `response_format=...`, `thinking=...`, `config=...`).
- `await callm.acomplete(...)` is the async version.

A shield exposes the same call with its protections applied, and accepts per-call overrides:

```python
guard = shield(block_pii=True, detect_injection=True)
response = guard.complete(provider="anthropic", model="claude-sonnet-5", messages=user_messages)
response = await guard.acomplete(model="gpt-4o-mini", messages="hi", cache=True)
```
