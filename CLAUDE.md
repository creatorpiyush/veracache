# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early development: exact and semantic lookup, both embedders, and the memory store work. Guards, eval, the proxy, other stores, and bindings are not built. The design is in `ARCH.md`.

## What this is

A Rust semantic cache for LLMs where a hit must be proven safe, not just similar. Similarity only finds candidates; a guard pipeline decides. False hits (the user silently gets the wrong answer) are the core bug this project exists to prevent.

## Working rules

- Before adding a new dependency, explain why and ask.
- Check the latest stable crate versions when adding them; don't trust versions from memory.
- Use `#![forbid(unsafe_code)]` in every crate except the bindings crates (`veracache-node`, `veracache-py`).
- Use `#![warn(missing_docs)]` on public crates. Every public item gets a doc comment.
- Put traits at boundaries (embedder, store, guard, analyzer, reranker) so backends are pluggable and testable with fakes.
- Write the failing test first for anything in the guard layer.

## Commands

```sh
cargo test                                   # all tests
cargo test -p veracache-guard                # one crate
cargo test -p veracache-guard <test_name>    # one test
cargo clippy --all-targets -- -D warnings    # lint (must be clean)
cargo fmt --check                            # format check (must pass)
```

## Architecture

Cargo workspace under `crates/`: `veracache-core` (types, traits, lookup pipeline, config) sits at the center. `-embed`, `-store`, and `-guard` implement its traits. `veracache` is the facade that composes them from config. `-eval`, `-proxy` (axum, OpenAI-compatible, a library), `-cli` (the only binary), and later `-node` (napi-rs) and `-py` (PyO3) sit on top.

Lookup flow: cacheability check, normalize, build namespace, exact lookup (blake3), analyze, embed, ANN top-k, threshold filter, guards (all must pass), optional rerank, then best surviving candidate is a hit. On a miss, call upstream and store only clean responses. Every decision emits a `tracing` trace, which powers `veracache inspect` and the eval report.

Invariants that span crates:

- **Namespace isolation**: a hit is only possible within one namespace, a hash of tenant, model, embedder id, system prompt, tools schema, response format, context fingerprint, and version tag. Never search across namespaces. The proxy must refuse to start in shared mode without a tenant source.
- **`Analyzed`**: features (numbers, dates, entities, negations, direction words, quotes) are extracted once so guards don't re-parse. Keep it pluggable for language-specific analyzers later.
- **Multi-turn context**: by default only single-user-turn conversations are cached. Never match a context-dependent follow-up against a standalone question.
- **Upstream**: never cache upstream errors. On streaming, store only after a clean `[DONE]`. Redact `Authorization` headers in logs.

`eval/datasets/golden.jsonl` drives the guard tests. The false-hit rate on this suite must not regress.
