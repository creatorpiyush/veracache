# veracache

**A semantic cache for LLMs where a hit must be proven safe, not just similar.**

> **Status: early development.** The core cache works as a Rust library, but it is not published and **has no guards yet, so it can still return false hits. Do not use it for real answers.** Guards are the next milestone.

## The problem

Most semantic caches embed a prompt, find its nearest neighbour, and return the cached answer if cosine similarity passes one global threshold. These prompts are all very similar and need different answers:

| Prompt A | Prompt B | Differs by |
|---|---|---|
| Capital of France? | Capital of Germany? | Entity |
| Revenue in Q2 | Revenue in Q3 | Number |
| Recipe with nuts | Recipe without nuts | Negation |
| Flights to Delhi | Flights from Delhi | Direction |
| Weather today? | Weather tomorrow? | Date |

When a cache gets these wrong, nothing errors. The user just gets the wrong answer.

## The approach

Similarity finds candidates. A guard pipeline decides.

```
request → exact match? → embed → nearest neighbours → threshold
        → guards: number, date, negation, direction, entity, quote, length
        → all pass: HIT    any veto: MISS, ask the LLM
```

If veracache cannot show a hit is safe, it treats it as a miss. A miss costs a little money. A false hit costs trust.

## Works today

- **Exact and semantic lookup** in a Rust library. The guard step above is not built yet.
- **Isolation**: a hit never crosses tenant, model, embedder, system prompt, tool schema, or conversation history.
- **Local embedding**: runs bge-small-en-v1.5 in-process, offline once the model is downloaded.
- **Remote embedding**: any OpenAI-compatible `/embeddings` API.
- **In-memory store**: TTL, delete by tag or version, and an HNSW index for large namespaces.
- **Fails open**: if the store or embedder fails, the request is a miss, not an error.

## Planned

- **Guarded matching**, with a recorded reason for every rejected candidate.
- **Eval CLI**: replay your own logs to get hit rate, false-hit rate, and a recommended threshold before you trust it.
- **OpenAI-compatible proxy**: a single binary that supports streaming. Point an existing client at it.
- **Node and Python packages**, built on the same Rust core.
- **More storage**: SQLite, Redis, pgvector.

## Try it

You need Rust 1.92 or newer and a C++ compiler.

```sh
git clone https://github.com/creatorpiyush/veracache.git
cd veracache
cargo run -p veracache --example paraphrase
```

The first run downloads the embedding model to `.fastembed_cache`. Output:

```
MISS  What is the capital of France?
HIT   What is the capital of France? -> Paris (exact, 1.000)
HIT   What's the capital city of France? -> Paris (semantic, 0.961)
MISS  How do I bake sourdough bread?
```

See [paraphrase.rs](crates/veracache/examples/paraphrase.rs) for the API.

### Platforms

| Target | Local embedder | Remote embedder |
|---|---|---|
| Linux x64 and arm64 (gnu) | Yes | Yes |
| macOS arm64 | Yes | Yes |
| Windows x64 | Yes | Yes |
| Linux x64 and arm64 (musl) | No | Yes |
| macOS x64 | No | Yes |

ONNX Runtime has no prebuilt binaries for the last two rows. Build those targets with `--no-default-features` and use the remote embedder. That build is expected to work, but the full library has not been tested on them yet.

## Results

No false-hit numbers yet. They will be published here with the command that reproduces them, once guards exist.

Lookup latency at 100k entries, from `cargo bench -p veracache --bench lookup`:

| Bench | Mean |
|---|---|
| Exact hit | 1.2µs |
| Semantic hit | 68µs |

These are early numbers from an Apple M4 Pro, with synthetic vectors, a fake embedder, and no guards. Expect semantic lookup to be slower with guards on.

## Roadmap

| Phase | Delivers | Status |
|---|---|---|
| 0 | Workspace, CI, project skeleton | Done |
| 1 | Core cache: exact and semantic lookup, embedders, memory store | Done |
| 2 | Guard pipeline and golden test suite | Next |
| 3 | Eval CLI | Planned |
| 4 | OpenAI-compatible proxy | Planned |
| 5 | SQLite, Redis, and pgvector backends | Planned |
| 6 | Node and Python packages | Planned |
| 7 | Adaptive per-entry thresholds | Planned |
| 8 | v0.1 launch | Planned |

## Documentation

- [ARCH.md](ARCH.md): system design
- [CONTRIBUTING.md](CONTRIBUTING.md): how to contribute

## License

[MIT](LICENSE)
