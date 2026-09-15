# Cookbook

Complete examples for common production patterns. Each one runs against real providers once you
set the relevant API keys; the repository's `examples/` folder has runnable versions, including
`examples/offline_demo.py`, which needs no API key at all.

| Recipe | Shows |
|---|---|
| [Support chatbot](chatbot.md) | per-user budgets, PII masking, injection blocking, fallback, async |
| [RAG with citations](rag.md) | structured answers with source ids, caching retrieval-augmented prompts, indirect-injection scanning of retrieved documents |
| [Data extraction](data-extraction.md) | Pydantic schemas, automatic repair, batch processing with a spend cap, cost reports |
