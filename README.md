# veracache

**A semantic cache for LLMs where a hit must be proven safe, not just similar.**

> **Status: design stage. No code exists yet, and nothing here can be installed.** Everything below describes the planned v0.1.

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

## Planned features

- **Guarded matching**, with a recorded reason for every rejected candidate.
- **Eval CLI**: replay your own logs to get hit rate, false-hit rate, and a recommended threshold before you trust it.
- **OpenAI-compatible proxy**: a single binary that supports streaming. Point an existing client at it.
- **Node and Python packages**, built on the same Rust core.
- **Local first**: runs offline with a local ONNX embedder. Remote embedding APIs are optional.
- **Tenant isolation**: a hit never crosses tenant, model, system prompt, or tool schema.
- **Pluggable storage**: memory, SQLite, Redis, pgvector.

## Planned usage

These examples are design targets, not working code.

Measure on your own logs:

```sh
veracache eval --input prompts.jsonl --sweep 0.80:0.98:0.01
```

Run the proxy:

```sh
veracache serve --config veracache.toml
```

Use it from Node:

```ts
import { SemanticCache } from "veracache";

const cache = await SemanticCache.create({ store: "memory", embedder: "local", guards: "default" });

const hit = await cache.get({ model, messages, tenant: userId });
if (!hit) {
  const res = await llm(model, messages);
  await cache.set({ model, messages, tenant: userId }, res);
}
```

## Results

None yet. False-hit numbers will be published here with the command that reproduces them, starting with Phase 2.

## Roadmap

| Phase | Delivers |
|---|---|
| 0 | Workspace, CI, project skeleton |
| 1 | Core cache: exact and semantic lookup, local embedder |
| 2 | Guard pipeline and golden test suite |
| 3 | Eval CLI |
| 4 | OpenAI-compatible proxy |
| 5 | SQLite, Redis, and pgvector backends |
| 6 | Node and Python packages |
| 7 | Adaptive per-entry thresholds |
| 8 | v0.1 launch |

## Documentation

- [ARCH.md](ARCH.md): system design

## License

[MIT](LICENSE)
