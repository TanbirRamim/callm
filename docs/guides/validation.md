# Structured output

```python
from pydantic import BaseModel, Field
from callm import callm

class Summary(BaseModel):
    title: str
    bullets: list[str]
    sentiment: float = Field(ge=-1, le=1)

@callm(output_schema=Summary, validation_retries=2)
def summarize_article(url: str):
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize {url} as JSON with title, bullets, sentiment."}],
    )

result = summarize_article("https://example.com/article")   # a validated Summary
```

Requires `pip install "callm-toolkit[validation]"` (Pydantic v2). `output_schema` accepts anything
Pydantic can validate: models, dataclasses, `TypedDict`s, `list[Model]`, `dict[str, int]`...

## What happens

1. The model's text is extracted (OpenAI message content, Anthropic text blocks, Gemini text
   parts — thinking content is ignored).
2. callm tries to parse it as JSON: the whole text, then fenced code blocks, then the outermost
   `{...}` / `[...]` span. Prose around the JSON is fine.
3. If validation fails, callm appends the model's answer and a message with the validation
   errors and the JSON Schema to the conversation, and asks again — up to
   `validation_retries` times.
4. If it still fails, `OutputValidationError` is raised with `.errors`, `.raw_text` and
   `.attempts`.

Retried requests keep all your original arguments, so provider features such as OpenAI's
`response_format` or Anthropic structured outputs keep working alongside callm validation.

## Return values

The decorated function returns the validated object. It works whether your function returns the
SDK response or post-processes it:

```python
@callm(output_schema=Summary)
def summarize(text):
    response = client.chat.completions.create(...)
    return response.choices[0].message.content      # a string is validated too
```

If a function makes more than one LLM call, every call inside it is validated against the
schema. Split multi-step flows into separately decorated functions.

## Caching

Only responses that passed validation are cached, and the schema is part of the cache key, so
changing a schema never serves stale shapes.

## Without an intercepted call

For functions that produce data some other way (a custom client, a local model), callm validates
the return value and re-runs the function up to `validation_retries` times if it is invalid.
