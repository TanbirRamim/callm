# Cost tracking & budgets

Every call is priced from its actual token usage — including prompt-cache reads and writes —
and recorded in telemetry. Limits are enforced **before** a request is sent.

## Per-call limit

```python
@callm(max_cost=0.25)
def summarize(text): ...
```

If the estimated cost exceeds `max_cost`, `callm.BudgetExceeded` is raised and nothing is sent.
The estimate uses the input size and the request's `max_tokens` (or 1,024 output tokens when
`max_tokens` is not set — configurable with `callm.configure(assumed_output_tokens=...)`).
For a strict worst-case bound, always set `max_tokens`.

## Shared budgets

```python
from callm import Budget

team_budget = Budget(50.00, name="search-team")

@callm(budget=team_budget)
def search(query): ...

team_budget.spent, team_budget.remaining
```

Budgets are thread-safe. Each call *reserves* its estimate, then settles with the actual cost,
so concurrent calls cannot race past the limit. A call is refused once
`spent + reserved + estimate > limit`.

### Per session

```python
with callm.budget(0.50) as session:
    plan = planner(task)
    for step in plan:
        executor(step)
print(session.spent)
```

Budgets entered with `with` apply to every callm call in that context (including nested calls
and `asyncio` tasks started inside it). `async with` works too.

### Per user or tenant

```python
def handle_request(user_id: str, question: str):
    with callm.budget_for(f"user:{user_id}", limit=2.00):
        return answer(question)
```

`budget_for` returns the same process-wide `Budget` for the same key. Budgets are in-memory;
reset them on your own schedule (for example daily) with `budget.reset()`.

## Prices

callm ships a snapshot of current prices for OpenAI, Anthropic and Gemini chat models
(generated from the community-maintained [LiteLLM price list](https://github.com/BerriAI/litellm)).

```console
$ callm pricing show claude-sonnet-5
{"model": "anthropic/claude-sonnet-5", "unit": "USD per 1M tokens", "input": 2.0, "output": 10.0, ...}
$ callm pricing update        # download current prices into ~/.callm/pricing.json
```

Dated or regional model names resolve to their base price (`gpt-4o-2024-08-06` → `gpt-4o`,
`claude-opus-4-5@20251101` → `claude-opus-4-5`). Register custom prices for fine-tunes,
negotiated rates or self-hosted models:

```python
callm.set_price("ft:gpt-4o-mini:acme::abc123", input=0.30, output=1.20)
callm.set_price("llama3.1", input=0.0, output=0.0, provider="ollama")
```

When a model has no known price, its cost is recorded as unknown (a warning is logged once) and
cost limits cannot be enforced for it.

## Reports

```console
$ callm stats --since 30d
$ callm stats --by function
$ callm stats --by model --json
```

```python
for row in callm.stats(by="model", since="7d"):
    print(row.key, row.calls, row.cost_usd, row.saved_usd, row.cache_hit_rate)
```
