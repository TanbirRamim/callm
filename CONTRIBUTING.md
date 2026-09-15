# Contributing to callm

Thanks for helping make LLM calls boring and reliable. This guide covers everything you need to
get a change merged.

## Ground rules

- **Zero required dependencies.** The core package imports only the standard library. Anything
  that needs a third-party package goes behind an optional extra and a lazy import that raises
  `MissingDependencyError` with the install hint.
- **Never break the caller.** Telemetry, cache and hook failures are logged, never raised.
  Policy decisions (budgets, security, validation) raise `CallmError` subclasses.
- **Sync and async parity.** Middleware is written once as a generator; don't add
  sync-only or async-only behaviour.
- **No prompt text in telemetry.** Call records contain numbers and flags only.

## Development setup

```bash
git clone https://github.com/TanbirRamim/callm && cd callm
uv sync                      # creates .venv with dev dependencies
uv run pre-commit install    # optional: lint on commit
```

## Checks

```bash
uv run pytest                          # full suite, offline, ~3 seconds
uv run pytest --cov=callm              # coverage (CI requires 90%+)
uv run ruff check src tests scripts examples benchmarks
uv run ruff format src tests scripts examples benchmarks
uv run python benchmarks/run.py         # offline benchmarks
uv run mypy                            # strict mode
uv run --only-group docs mkdocs serve  # docs at http://127.0.0.1:8000
```

Tests use the **real** OpenAI, Anthropic and Google SDKs with mocked HTTP transports
(`tests/helpers.py`), so they exercise genuine request serialization, response parsing and
error classes without network access or API keys. Please follow that pattern: prefer a mock
server reply over mocking callm internals.

## Project layout

```text
src/callm/
  decorator.py        @callm
  api.py              shield, complete, acomplete
  interception.py     SDK instrumentation and the active scope
  pipeline.py         effects, sync/async drivers, chain composition
  middleware/         telemetry, security, cache, validator, fallback, cost, retry
  providers/          OpenAI, Anthropic, Gemini adapters and the registry
  storage/            SQLite, memory and Redis backends
  security/           PII redaction and injection detection
  pricing.py          price table and cost estimates (data/pricing.json)
  cli.py              the `callm` command
```

## Common changes

**Adding a provider.** Subclass `callm.providers.base.Provider` (see `openai.py` for a compact
example): parse native kwargs into an `LLMRequest`, translate canonical requests back, call the
official SDK, parse responses and synthesize native responses. If the provider speaks the
OpenAI protocol, `OpenAIProvider(name, base_url=..., api_key_env=...)` is usually enough.
Add interception targets to `interception.PATCH_TARGETS` when the SDK should be intercepted.

**Updating prices.** Run `python scripts/update_bundled_pricing.py` and commit
`src/callm/data/pricing.json`.

**Improving injection detection.** Add patterns to `security/injection.py` together with
positive *and* negative examples in `tests/test_security.py`. False positives matter as much
as misses.

## Pull requests

1. Open an issue first for large changes so we can agree on the design.
2. Keep PRs focused; include tests and docs for behaviour changes.
3. Add an entry under "Unreleased" in `CHANGELOG.md`.
4. Make sure CI is green.

## Releasing (maintainers)

1. Bump `src/callm/__about__.py` and move the changelog entries under the new version.
2. Tag `vX.Y.Z` and publish a GitHub release — the `Publish` workflow builds, tests and
   uploads to PyPI with trusted publishing.
3. Optionally dry-run first: run the `Publish` workflow manually with `target=testpypi`.

## Code of conduct

Be kind, assume good intent, and keep discussions about the work. Harassment of any kind is not
tolerated; report concerns to the maintainers privately.
