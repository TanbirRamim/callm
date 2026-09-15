# Caching

```python
@callm(cache=True)                        # exact-match cache
@callm(cache="semantic")                  # + semantic matching (needs callm-toolkit[cache])
@callm(cache=CacheConfig(ttl=3600, semantic=True, threshold=0.93))
```

A cache hit returns instantly, costs nothing, and is recorded in telemetry with the amount it
saved. Callers receive the same native SDK object they would get from a live call.

## Exact matching

Two requests share a cache entry only when all of these are identical:

- provider and model
- every message (after [PII masking](security.md), so raw PII never reaches cache keys)
- generation parameters (`temperature`, `max_tokens`, tools, response format, ...)
- the `output_schema`, if any
- the cache `namespace`

Arguments that don't affect generation — `timeout`, `extra_headers`, `metadata`, `user`,
`store`, `service_tier`, `prompt_cache_key` and similar — are ignored.

What is never cached: streaming calls, requests with `n > 1` (or Gemini `candidate_count > 1`),
refusals and content-filtered responses, and output that failed `output_schema` validation.

## Semantic matching

Semantic caching also returns a stored answer when the **final user message** is similar
enough (cosine similarity ≥ `threshold`) to a cached one — and everything else (system prompt,
earlier conversation turns, parameters, schema) is identical.

```python
from callm import CacheConfig, OpenAIEmbedder, SentenceTransformerEmbedder

CacheConfig(semantic=True)                                    # all-MiniLM-L6-v2, local
CacheConfig(semantic=True, embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"))
CacheConfig(semantic=True, embedder=OpenAIEmbedder("text-embedding-3-small"))
CacheConfig(semantic=True, embedder=my_embedder)              # anything with .embed(texts)
```

!!! warning "Choose semantic caching deliberately"
    Similar is not the same. *"Summarize https://a.example/post-1"* and
    *"Summarize https://a.example/post-2"* are nearly identical to an embedding model, yet need
    different answers. That is why `cache=True` is exact-match. Use semantic matching for
    FAQ-style traffic (support questions, documentation lookups), keep thresholds high
    (0.93–0.97), and put variable data in the system prompt or earlier turns where it is part of
    the exact key.

`HashingEmbedder` is a dependency-free *lexical* embedder — handy for tests and near-duplicate
detection (casing, punctuation, typos), but it does not understand meaning.

## CacheConfig

| Field | Default | Meaning |
|---|---|---|
| `ttl` | `None` | Seconds until an entry expires (`None` = never) |
| `semantic` | `False` | Enable semantic matching |
| `threshold` | `0.95` | Minimum cosine similarity for a semantic hit |
| `embedder` | `None` | Embedder for semantic matching (default: sentence-transformers) |
| `store` | `None` | Cache backend for this function (default: the global store) |
| `namespace` | `"default"` | Separate caches that share a backend |
| `max_candidates` | `2000` | Most recent entries compared per semantic lookup |

## Backends

```python
import callm
from callm.storage import MemoryCacheStore, SQLiteStorage
from callm.storage.redis import RedisCacheStore

callm.configure(storage="sqlite")                      # default: ~/.callm/callm.db
callm.configure(storage="memory")                      # process-local
callm.configure(cache_store=RedisCacheStore("redis://cache:6379/0"))  # shared by all workers
```

Cache failures (disk full, Redis down) are logged and treated as misses — they never fail the
call.

## Managing the cache

```console
$ callm cache stats
$ callm cache clear
```

```python
callm.clear_cache()
```
