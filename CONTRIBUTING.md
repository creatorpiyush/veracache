# Contributing to veracache

veracache is at an early stage. The design is in [ARCH.md](ARCH.md).

## Before you open a pull request

All of these must pass:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Rules

- Guard changes start with a failing test.
- Every false hit you find gets a new pair in the golden suite.
- Never weaken a guard, or delete a golden pair, to make a test or metric pass.
- Do not add `unsafe` code. Only the bindings crates may use it.
- Every public item needs a doc comment.
- Discuss a new dependency before adding it.

## Commits and branches

Use [Conventional Commits](https://www.conventionalcommits.org/). Name branches `type/short-description`, for example `fix/date-guard-relative-words`.

## License

Contributions are licensed under [MIT](LICENSE).
