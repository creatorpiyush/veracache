# Architecture: veracache

Status: partly built. The lookup pipeline without guards, both embedders, and the memory store exist. Everything else here is design.

## 1. Overview

veracache sits between an application and an LLM. For each request it tries to prove that a stored response is a correct answer. If it cannot, the request goes upstream.

```
        Node / Python app      OpenAI SDK client      `veracache eval`
              │                       │                     │
        veracache-node/py       veracache-proxy       veracache-eval
              │                       │                     │
              └───────────┬───────────┴─────────────────────┘
                          ▼
                   veracache (facade)
                          │  composes from config
                          ▼
                   veracache-core ── lookup pipeline, traits, types
                     ▲    ▲    ▲
         ┌───────────┘    │    └───────────┐
   veracache-embed  veracache-store  veracache-guard
```

Two rules shape the design:

- **Wrong answers are worse than misses.** Anything that cannot be verified is rejected.
- **The cache is optional to the caller.** A cache failure becomes a miss, never a failed request.

## 2. Crates and dependency direction

Dependencies point one way, toward `veracache-core`.

| Crate | Owns | Depends on |
|---|---|---|
| `veracache-core` | `Embedder`, `VectorStore`, `Guard`, `Analyzer`, `Reranker` traits; `Analyzed`, `CacheEntry`, `Namespace`; normalization; pipeline | none |
| `veracache-embed` | fastembed and HTTP embedders, LRU | core |
| `veracache-store` | memory, sqlite, redis, pgvector stores; conformance suite | core |
| `veracache-guard` | English analyzer, seven guards | core |
| `veracache` | `Cache` API, config, default composition | core, embed, store, guard |
| `veracache-eval` | Replay, sweep, recommendation, reports | veracache |
| `veracache-proxy` | axum server, SSE tee and replay, admin endpoints | veracache |
| `veracache-cli` | `veracache` binary: `serve`, `eval`, `replay`, `stats`, `inspect` | proxy, eval |
| `veracache-node`, `veracache-py` | Bindings | veracache |

The pipeline in core knows only traits, so every test can use fakes. Core has no network or ONNX dependency.

## 3. Core contracts

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;            // part of the namespace
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, entry: CacheEntry) -> Result<()>;
    async fn search(&self, ns: &Namespace, vector: &[f32], k: usize) -> Result<Vec<Candidate>>;
    async fn get_exact(&self, ns: &Namespace, key_hash: &[u8; 32]) -> Result<Option<CacheEntry>>;
    async fn delete(&self, filter: DeleteFilter) -> Result<u64>;
    async fn record_feedback(&self, id: EntryId, fb: Feedback) -> Result<()>;
}

pub trait Analyzer: Send + Sync {
    fn analyze(&self, text: &str) -> Result<Analyzed>;
}

pub trait Guard: Send + Sync {
    fn name(&self) -> &'static str;
    /// Ok(()) means no objection. Err(reason) rejects this candidate.
    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason>;
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn score(&self, query: &str, candidate: &str) -> Result<f32>;
}
```

`search` takes a required `Namespace`, and no store method can search without one. That makes cross-namespace search unrepresentable at the API.

`Analyzed` holds the prompt text and extracted features: numbers, dates, entities, negations, direction words, and quoted strings. It carries a feature version. Entries analyzed by an older version fail closed, then repopulate as misses.

## 4. Data model

```
CacheEntry {
  id, namespace, key_hash,
  prompt_text?, analyzed_features?, embedding,
  response, response_meta { model, finish_reason, token counts },
  created_at, ttl, tags[], version,
  stats { hits, false_hit_reports, last_hit_at },
  threshold_override?          // learned per entry
}
```

`prompt_text` and `analyzed_features` are absent in no-text mode. Guards cannot run without them, so that mode is exact-only unless the operator opts into unguarded semantic matching. `response` keeps the original JSON bytes, so a hit returns exactly what upstream sent, and streaming replay derives chunks from it.

## 5. Read path

```
request
  │
  ├─ uncacheable? ──────────────────────────────────► BYPASS, call upstream
  ▼
normalize, build namespace
  │
  ├─ exact lookup (blake3) ── found, not expired ───► HIT
  ▼
analyze ── error ─────────────────────────────────► MISS
  ▼
embed ── error ───────────────────────────────────► MISS
  ▼
ANN top-k within namespace, drop expired
  ▼
threshold filter
  ▼
for each candidate, best score first:
  all guards pass, and rerank passes if configured? ► HIT
  ▼
none survive ─────────────────────────────────────► MISS
```

Only the exact path skips guards, because the text is identical after normalization. That is why exact hashing preserves case: "Turkey" and "turkey" must not collide.

Guards run in full on the survivors so the trace can report every reject reason. They are synchronous and cheap, since features are precomputed.

## 6. Write path

On a miss, the caller gets the upstream response first. The store happens after, if the response is cacheable: `finish_reason` is `stop`, there is no error, and it is not a tool call unless tool-call caching is enabled.

```
upstream response
  ▼
cacheable? ── no ──► done
  ▼
embed (served by the embedder's LRU)
  ▼
store.upsert(entry) ── error ──► count it, done
```

`get` and `set` are separate calls that share no state, so `set` asks for the embedding again. The embedder's LRU, keyed by normalized text, returns the vector computed during lookup, which keeps the public API simple. Concurrent identical misses may each call upstream and overwrite the same entry, which wastes cost but serves no wrong answer, so it is accepted for v0.1.

## 7. Namespace

```
namespace = blake3(format_version
                   || lp(tenant) || lp(model) || lp(embedder_id)
                   || lp(system_hash) || lp(tools_hash)
                   || lp(response_format) || lp(context_fingerprint)
                   || lp(version_tag))
```

`lp` is a length prefix, which stops adjacent fields from colliding. Tools and response schemas are canonicalized before hashing. `embedder_id` in the namespace means changing models starts a fresh cache instead of comparing incompatible vectors.

## 8. Stores

All backends share one contract and one conformance suite.

| Backend | Vectors | Index | Notes |
|---|---|---|---|
| `memory` | In process | Brute force under ~5k per namespace, HNSW above | Default; optional snapshot |
| `sqlite` | Blobs | In-memory ANN rebuilt on start | sqlite-vec evaluated later |
| `redis` | RediSearch | RediSearch vector index | Feature-gated |
| `pgvector` | Postgres | pgvector | Feature-gated |

Each namespace has its own index, so search cost does not grow with other tenants' data. TTL is checked at read time on every backend; background cleanup only reclaims space.

## 9. Proxy

```
client ──► axum handler ──► parse ──► Cache::lookup
                                        │
                     HIT ◄──────────────┤
                      │                 └─► MISS ──► reqwest ──► upstream
        stream? replay as SSE : JSON                    │
                                     stream? tee to client + buffer : buffer
                                                        │
                                  clean [DONE], size ok ► Cache::store
```

- **Passthrough**: unknown routes forward unchanged.
- **Client disconnect** on a streaming miss stores nothing, because no clean `[DONE]` arrived.
- **Size limit**: past `max_entry_bytes` the buffer drops and streaming continues.
- **Tenant resolution** runs before cache access, from the configured source. The default is a hash of the caller's API key. A client's tenant header is trusted only behind a trusted gateway.
- **Startup validation** rejects unsafe configs instead of warning: multi-tenant mode with no tenant source, or admin endpoints off loopback with no token.

## 10. Eval

`veracache-eval` drives the real `Cache` through a replay with a store and clock it controls, so measured behavior is shipped behavior.

Each sweep point replays the log over a fresh store, because a hit changes what gets stored next. Embeddings are computed once and reused across replays.

```
log ──► embed once ──► replay per threshold ──► decision records ──► correctness ──► report
                                                                         ▲
                                                 labels, then judge, then heuristics
```

Golden embeddings are precomputed per embedder id and committed, so CI needs no model.

## 11. Observability

Each lookup opens a `tracing` span with its candidates, scores, thresholds, reject reasons, and final decision. The same structured decision record feeds logs, `veracache inspect`, and eval, so there is one definition of what happened.

## 12. Failure behavior

| Failure | Behavior |
|---|---|
| Analyzer error, or feature version unknown | Reject candidate, or miss |
| Embedder down | Miss; upstream serves |
| Store read error | Miss; upstream serves |
| Store write error | Response returned; counted |
| Upstream error | Passed to the client; not cached |
| Stream cut before `[DONE]` | Not cached |
| Unsafe proxy config | Refuse to start |

## 13. Extension points

| To add | Implement | Registered in |
|---|---|---|
| Embedder | `Embedder` | `veracache-embed`, config |
| Store | `VectorStore`, pass conformance | `veracache-store`, feature flag |
| Guard | `Guard`, add golden pairs | `veracache-guard`, config |
| Language | `Analyzer` | `veracache-guard`, config |
| Reranker | `Reranker` | `veracache`, config |
