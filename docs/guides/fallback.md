# Provider fallback

```python
@callm(retry=2, fallback=["anthropic/claude-sonnet-5", "google/gemini-2.5-flash", "ollama/llama3.1"])
def answer(question):
    return openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )
```

callm tries the primary model (with its retries), then each fallback in order (each with its
own retries). The first success wins, and your function still receives the response type of the
SDK it called — here an OpenAI `ChatCompletion`, even when Claude or Gemini answered.

## Targets

| Spec | Meaning |
|---|---|
| `"anthropic/claude-sonnet-5"` | provider / model |
| `"gpt-4o-mini"`, `"claude-haiku-4-5"`, `"gemini-2.5-flash"` | provider inferred from the model name |
| `"anthropic/claude-sonnet"` | alias for the current model of that family (`claude-opus`, `claude-sonnet`, `claude-haiku`, `claude-fable`) |
| `"ollama/llama3.1"` | local Ollama server (`http://localhost:11434/v1`) |
| `Target("groq", "llama-3.3-70b")` | explicit target object |

Built-in providers: `openai`, `anthropic`, `google` (alias `gemini`) and `ollama`. Register any
OpenAI-compatible endpoint:

```python
import callm

callm.register_provider(
    "groq",
    callm.OpenAIProvider(
        "groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        max_tokens_param="max_tokens",
    ),
)
```

## Credentials and clients

Fallback targets use the official SDK clients created from the usual environment variables
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` / `GEMINI_API_KEY`). To control
timeouts, base URLs or credentials, pass your own clients:

```python
callm.configure(
    clients={"anthropic": anthropic.Anthropic(timeout=30, max_retries=0)},
    async_clients={"anthropic": anthropic.AsyncAnthropic(timeout=30, max_retries=0)},
)
```

When the fallback uses the *same* provider as the original call, callm reuses your client.

## When fallback happens

By default callm fails over on transient errors: rate limits, overloads, 5xx, timeouts and
connection failures. Client errors such as 400 or 401 are raised immediately, because another
provider would not fix a malformed request or a wrong key. Customize the decision:

```python
@callm(fallback=["gpt-4o-mini"], fallback_on=lambda exc: getattr(exc, "status_code", None) != 400)
```

Once in fallback mode, any error from a fallback target (including a missing SDK or API key)
moves on to the next target. If everything fails, `AllProvidersFailedError` is raised with every
`(target, exception)` pair in `.errors`; the last exception is chained as `__cause__`.

## What translates

| Translated across providers | Stays with its provider |
|---|---|
| system prompts, user/assistant turns | tools / function calling |
| `max_tokens`, `stop` sequences | images, audio, files |
| `temperature`, `top_p` (not sent to Anthropic, whose current models reject them) | response formats / JSON schemas |
| | `n > 1`, provider-specific config |

Requests with non-translatable features skip cross-provider fallbacks (a warning is logged) and
only use fallbacks of the same provider, which receive the original arguments unchanged.
Anthropic translations use `max_tokens=4096` when the request did not set one.

## Telemetry

A successful fallback is recorded with the provider and model that answered and
`fallback_from="openai/gpt-4o"`; `callm stats` counts fallbacks per provider.
