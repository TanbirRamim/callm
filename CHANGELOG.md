# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Reproducible offline benchmarks (`benchmarks/run.py`) and a benchmarks page in the docs.
- Demo GIF, before/after example and a comparison with related tools in the README.

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

[Unreleased]: https://github.com/TanbirRamim/callm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TanbirRamim/callm/releases/tag/v0.1.0
