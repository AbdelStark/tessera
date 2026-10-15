# Tessera

Tessera proves that a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls every piece of hardware and software, hides weights, activations, prompts and outputs from the verifier, and is designed to cost a small multiple of the inference it certifies. An independent verifier rebuilds every constraint from a public claim list and shares no code with the prover.

**Status:** pre-release. Milestone v0.1 (primitives and the unit-cost gate) is in progress. Nothing here is usable for audits yet, and every performance figure in the specification is a projection until it is measured.

## Specification

- [`SPEC.md`](SPEC.md): executive summary, reading order, release plan and risk register.
- [`docs/spec/`](docs/spec/): overview, architecture, public API, data model, error model, observability, security, testing, performance, release policy and glossary.
- [`docs/rfcs/`](docs/rfcs/): the twenty accepted design decisions (RFC-0001 to RFC-0020).
- [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md): the issue tracker generated from the corpus.

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Prover workspace: one crate per component of [01 Architecture](docs/spec/01-architecture.md#2-components) |
| `xtask/` | Repository automation, run as `cargo xtask <command>` |
| `docs/` | Specification, RFCs and roadmap |

The layout and the verifier-independence rules it enforces are fixed by [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md).

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it on first use. The minimum supported Rust version is 1.85.

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo xtask help
```

No crate needs a GPU or the CUDA toolkit to build or test.
