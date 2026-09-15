# Cookbook

Complete examples for common production patterns. Each one runs against real providers once you
set the relevant API keys; the repository's `examples/` folder has runnable versions, including
`examples/offline_demo.py`, which needs no API key at all.

| Recipe | Shows |
|---|---|
| [Support chatbot](chatbot.md) | per-user budgets, PII masking, injection blocking, fallback, async |
| [RAG with citations](rag.md) | structured answers with source ids, caching retrieval-augmented prompts, indirect-injection scanning of retrieved documents |
| [Data extraction](data-extraction.md) | Pydantic schemas, automatic repair, batch processing with a spend cap, cost reports |

## Runnable examples

| File | Needs a key? | What it shows |
|---|---|---|
| [`examples/offline_demo.py`](https://github.com/TanbirRamim/callm/blob/main/examples/offline_demo.py) | no | a guided tour: retry, cache hit, fallback, blocked call, flagged injection |
| [`examples/ticket_triage.py`](https://github.com/TanbirRamim/callm/blob/main/examples/ticket_triage.py) | yes | a complete mini-project: support-ticket triage with validation, caching, PII masking and a cost report (works with OpenAI, OpenRouter, Ollama or any OpenAI-compatible endpoint) |
| [`examples/quickstart.py`](https://github.com/TanbirRamim/callm/blob/main/examples/quickstart.py) | yes | the smallest useful program |
| [`examples/structured_output.py`](https://github.com/TanbirRamim/callm/blob/main/examples/structured_output.py) | yes | schemas with automatic repair and a fallback provider |
| [`examples/async_budgets.py`](https://github.com/TanbirRamim/callm/blob/main/examples/async_budgets.py) | yes | concurrent async calls under a shared session budget |
