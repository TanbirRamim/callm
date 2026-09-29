# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-29

### Added

- OpenAI Responses API support: `client.responses.create(...)`, sync and async, is intercepted
  like `chat.completions.create`. `instructions`, string and item `input` and tool outputs get
  PII masking and injection checks; budgets, caching, output validation, retries, cost and
  telemetry apply; fallbacks to other providers return a real `Response` object. The adapter is
  also available as the `openai-responses` provider for `callm.complete` and fallback chains.
- Inside a callm function, SDK helpers callm does not intercept yet (`responses.parse/stream`,
  `chat.completions.parse/stream`, `beta.chat.completions.parse/stream`, Anthropic
  `messages.stream`) now log a one-time warning naming the protections that do not apply,
  instead of silently skipping PII masking, budgets and telemetry.

## [0.1.2] - 2026-09-27

### Fixed

- The SDK's own retries no longer multiply callm's. With a default OpenAI or Anthropic client,
  `@callm(retry=2)` could send nine requests on a persistent 429/5xx and wait on hidden
  backoff before a fallback. When callm retries or falls back it now calls the SDK with
  `max_retries=0` (on a copy; your client is unchanged).
- A response without a `usage` field was recorded as free, so budgets and `max_cost` never
  filled with OpenAI-compatible servers and proxies that omit it. The pre-call estimate is
  charged instead.
- Tool calls, tool results and Gemini function parts were estimated as 1,500-token images,
  so short agent loops were refused by `max_cost`. Only images, audio, documents and files
  use the media estimate now.
- Validation re-asks were charged to budgets but only the last attempt reached telemetry and
  `callm stats`; a call whose attempts all failed recorded nothing. Usage and cost are now
  summed across attempts.
- Validation errors (and the repair prompt) described a nested list instead of the object
  the model returned, for example `(root): Input should be an object` instead of
  `needs_human: Field required`.
- The exact cache now keys on the client's endpoint, so an Azure, vLLM, local or staging
  client never gets answers cached from another server with the same model name.
- Rewriting a cache entry (concurrent misses on one prompt) reset its hit count.
- `examples/offline_demo.py` failed on openai < 3 / anthropic < 1 (`No module named httpx2`).
- PII detection: non-ASCII emails, lowercase IBANs and IPv6 addresses are now masked, and SSH
  remotes such as `git@github.com:org/repo.git` are no longer masked as email addresses.

### Documentation

- The README and docs home examples now run as written, and the no-key demo comes first in
  the README quickstart.
- The quickstart no longer shadows the decorator with `import callm` in step 4.

## [0.1.1] - 2026-09-16

### Fixed

- `callm.last_call()` returned nothing (or a stale record) for calls made inside
  `asyncio.run(...)` or another thread; it now falls back to the most recent record overall.

### Added

- A runnable ticket triage example that works with any OpenAI-compatible endpoint.
- Reproducible offline benchmarks (`benchmarks/run.py`) and a benchmarks page in the docs.
- Demo GIF, before/after example and a comparison with related tools in the README.
- Nested scopes, SDK `NOT_GIVEN` sentinels and structured-field PII masking, all covered by
  regression tests (first reported during pre-release review).

## [0.1.0] - 2026-09-15

Initial release.

### Added

- `@callm` decorator for sync and async functions and methods, intercepting OpenAI
  (`chat.completions.create`), Anthropic (`messages.create`) and Google Gen AI
  (`models.generate_content`) SDK calls and returning the SDK's native response types.
- `shield` context manager and provider-neutral `complete` / `acomplete`.
- Retry engine with exponential backoff and jitter that honours `retry-after`,
  `retry-after-ms`, OpenAI and Anthropic rate-limit reset headers and Gemini `RetryInfo`.
- Response cache: exact-match by default, optional semantic matching (sentence-transformers,
  OpenAI embeddings, or a custom embedder); SQLite, in-memory and Redis backends; TTLs.
- Provider fallback chains with request translation between OpenAI, Anthropic, Gemini, Ollama
  and OpenAI-compatible providers; provider-specific requests only fall back within their
  provider.
- Cost tracking from a bundled price table (`callm pricing update` refreshes it), including
  prompt-cache read/write pricing; `max_cost` per call; thread-safe `Budget`s per function,
  session (`callm.budget`) or key (`callm.budget_for`).
- PII redaction for emails, phone numbers, SSNs, payment cards (Luhn), IPv4 addresses and IBANs
  (checksum), plus optional spaCy person names; mask or block.
- Heuristic prompt-injection detection (instruction overrides, prompt extraction, role
  hijacking, chat-template tokens, hidden unicode, multilingual variants) with an optional
  Hugging Face classifier; flag or block.
- Structured output validation for any Pydantic type with automatic re-asking on invalid output.
- Telemetry records (no prompt text), `on_call` hooks, OpenTelemetry spans, `callm.stats()` and
  `callm.last_call()`.
- Nested scopes: protections (PII masking, injection detection, the strictest `max_cost`, all
  budgets) from enclosing decorators and shields apply to inner calls.
- `callm` CLI: `stats`, `calls`, `cache`, `pricing`, `telemetry`, `info`.

[Unreleased]: https://github.com/TanbirRamim/callm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/TanbirRamim/callm/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/TanbirRamim/callm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/TanbirRamim/callm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TanbirRamim/callm/releases/tag/v0.1.0
