# FAQ & limitations

## Does callm change my SDK calls outside decorated functions?

No. The instrumented SDK methods check for an active callm scope first and call the original
implementation directly when there is none. `CALLM_DISABLED=1` turns callm off entirely.

## Is monkeypatching safe?

callm wraps ten public SDK methods (the `create` / `generate_content` calls and OpenAI's `chat.completions.parse`, sync and async), keeps the originals, and only acts inside a scope.
Observability tools such as OpenTelemetry instrumentations use the same technique. If an SDK
changes its internals in a way callm cannot parse, the call is passed through unchanged with a
warning — unless PII or injection protection is enabled, in which case callm refuses to send a
request it could not inspect.

## Which calls are not intercepted?

- Streaming helpers that bypass `create` internally: OpenAI's `client.responses.stream` and
  `client.chat.completions.stream`, `client.responses.parse`, the older
  `client.beta.chat.completions.parse/stream`, and Anthropic's `client.messages.stream(...)`.
  Inside a callm function these log a one-time warning naming the protections that do not
  apply (PII masking, budgets, caching...), so this never happens silently. Use `create`
  (or `callm.complete(...)`) for now. `client.responses.create` and
  `client.chat.completions.parse` are fully intercepted.
- `with_raw_response` calls (intentionally passed through).
- Calls in threads started without copying the context.
- Clients callm has no adapter for. Their responses are still recorded in telemetry when the
  decorated function returns a recognised SDK response object.

## Does callm work with the OpenAI Responses API?

Yes. `client.responses.create(...)` (sync and async) gets the same treatment as
`chat.completions.create`: `instructions`, string `input` and input items are masked and
scanned (including `function_call_output` tool results), budgets and `max_cost` apply, results
are cached, validated and costed, and fallbacks to Anthropic or Gemini still return a
`Response` with `output_text`. Calls are recorded under the provider name `openai-responses`
and priced with OpenAI's prices. Requests that depend on server-side state or tools
(`previous_response_id`, `conversation`, `tools`, `reasoning`...) never fall back to another
provider, and `callm.complete("openai-responses/gpt-5-mini", ...)` sends a request through
the Responses API directly.

## Does callm work with structured outputs (`chat.completions.parse`)?

Yes. `client.chat.completions.parse(..., response_format=MyModel)` gets the same treatment as
`create`: PII masking, budgets, caching, retries, cost and telemetry. You still get a
`ParsedChatCompletion` with `.choices[0].message.parsed`, including on a cache hit: callm
stores the completion and re-runs the SDK's own parser with your `response_format`. The cache
key includes the response format's JSON schema, so two different models never share an entry.
Fallbacks stay on OpenAI models, because `response_format` has no equivalent on other providers.

## What happens with `NOT_GIVEN` arguments?

The SDKs' `NOT_GIVEN` / `Omit` sentinels mean "argument not passed" and are dropped before callm
inspects the request, so patterns like `system=system or NOT_GIVEN` work normally.

## Does callm support streaming?

Streaming `create(..., stream=True)` calls get PII redaction, injection detection, budget
reservations, retries on connection setup and telemetry. They are not cached, validated or
failed over, and token usage is not recorded.

## How accurate are cost estimates?

Actual costs use the provider's reported token usage and are exact for known prices. *Estimates*
(for `max_cost` and budgets) count input tokens with `tiktoken` for OpenAI models when installed,
otherwise with a conservative character heuristic, and use `max_tokens` for output. A budget can
be overshot by at most the estimation error of the calls in flight; set `max_tokens` for tight
bounds.

## Why is the default cache exact-match?

Because a semantic cache can return a wrong answer for a question that merely *looks* similar.
See [Caching](guides/caching.md#semantic-matching).

## Can I use callm with LangChain, LlamaIndex or other frameworks?

Yes, when they call the official OpenAI, Anthropic or Google SDK clients underneath: decorate
the function that invokes the chain, or wrap it in `with shield(...)`.

## How do I test code that uses callm?

The callm test suite runs the real SDKs against mocked HTTP transports — you can do the same
(see `tests/helpers.py` in the repository). Configure `callm.configure(storage="memory")` in
tests, or disable callm with `CALLM_DISABLED=1`.

## Where is my data stored?

In `~/.callm/callm.db` (cache entries and call records) and `~/.callm/pricing.json` (if you ran
`callm pricing update`). Cache entries contain responses; call records contain no prompt or
completion text.
