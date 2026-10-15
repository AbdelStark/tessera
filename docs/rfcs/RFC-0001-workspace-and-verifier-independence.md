# RFC-0001: Workspace Layout, Language Boundaries, and Verifier Independence

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1

## Summary

Tessera is built as two Rust workspaces, a Python package and a CUDA kernel library. The prover workspace (`crates/`) holds everything the prover runs. The verifier workspace (`verifier/`) holds an independent implementation of everything the verifier runs and may not depend on any prover crate. Python and Rust exchange only canonical artifacts on disk. CUDA is reachable only through one crate. A CI gate enforces the isolation mechanically. This RFC locks the layout because the verifier's independence is a security property (SP-6), not a code-organization preference.

## Motivation

The source makes the verifier the whole trusted base: "It recompiles every constraint from the public claim list, shares no code with the prover, and is the target of formal verification" (source §6, principle 12; §9, problem 9). A shared bug in field arithmetic, encoding or table generation would let an invalid proof pass both the prover's self-checks and the verifier. Independence turns such bugs into detectable disagreements.

Scenarios this layout must support:

- A contributor changes `Fp` multiplication in the prover. The verifier's arithmetic is untouched, and the conformance vectors catch any divergence.
- An inspectorate builds `tessera-verify` from source and audits only `verifier/` plus an allowlist of third-party crates.
- An ML engineer quantizes a checkpoint on a machine with no Rust toolchain.

## Goals

- The verifier's build graph contains no prover crate, checked on every PR.
- The verifier's third-party dependencies are a short, reviewed allowlist.
- Every prover crate builds and tests without a GPU or CUDA toolkit.
- The Python front-end installs and tests without Rust.
- Tests can compare prover and verifier behavior directly.

## Non-Goals

- Language diversity for the verifier (a non-Rust verifier). See Alternatives.
- Mechanized proofs of the verifier (non-goal NG-7 in [00](../spec/00-overview.md#6-non-goals-for-v1)).
- Packaging and publication policy; see [09](../spec/09-release-and-versioning.md).

## Proposed Design

### Repository tree

```text
Cargo.toml                 # prover workspace: members = crates/*, xtask
rust-toolchain.toml        # exact stable toolchain pin for reproducible builds
deny.toml                  # license and advisory policy, prover workspace
crates/
  tessera-codec/ tessera-field/ tessera-hash/ tessera-poly/ tessera-sumcheck/
  tessera-logup/ tessera-tables/ tessera-pcs/ tessera-ir/ tessera-compiler/
  tessera-gadgets/ tessera-witness/ tessera-scoring/ tessera-anchor/
  tessera-protocol/ tessera-prover/ tessera-gpu/ tessera-bench/ tessera-cli/
  tessera-e2e/             # end-to-end and tamper tests (publish = false)
verifier/
  Cargo.toml               # separate workspace: members = crates/*
  Cargo.lock               # separate lockfile
  rust-toolchain.toml      # same pin as root, checked equal by CI
  deny.toml                # stricter policy, no license exceptions
  allowlist.toml           # permitted third-party packages with reasons
  crates/ tv-field/ tv-codec/ tv-hash/ tv-tables/ tv-core/ tv-transport/ tv-cli/
python/
  pyproject.toml uv.lock
  tessera_quant/ tests/
spec/
  vectors/                 # shared conformance data: codec, field, transcript, tables, models, tamper
  schemas/                 # JSON Schemas for reports
xtask/                     # repository automation: check-isolation, gen-vectors, release helpers
docs/
```

### Responsibilities by workspace

| Workspace | Contains | Must not contain |
|---|---|---|
| `crates/` | Witness generation, commitments, provers, compiler, GPU runtime, CLI | — |
| `verifier/` | Codec reader, field, hash and Merkle checks, transcript replay, table regeneration, claim-list recompilation and range re-derivation, stage, gadget, PCS, anchor and scoring checks, CLI | Any dependency on `crates/`, any `include!`/`#[path]` reaching outside `verifier/`, any symlink, any `unsafe` code |
| `python/` | Checkpoint import, calibration, quantization, `IntegerModel` export, integer reference, fidelity evaluation | Rust extensions, network access at import time |

### Language boundaries

- **Python to Rust.** Only through TCE artifacts (`IntegerModel`, token files, logits files). The Python package carries its own TCE writer and reader for the schemas it touches; conformance vectors check it against both Rust readers (DEP-4 in [01](../spec/01-architecture.md#3-dependency-rules)).
- **Rust to CUDA.** Only `tessera-gpu` contains `.cu` sources and FFI. Its `build.rs` compiles kernels with `nvcc` into fatbins for `sm_90` and `sm_100` when the `gpu` feature is on. Without the feature the crate builds as an empty stub. With the feature but without CUDA ≥ 12.8, the build fails with a message naming the missing toolkit.
- **Unsafe code.** Every crate declares `#![forbid(unsafe_code)]` except `tessera-gpu` (FFI and device memory) and `tessera-field` (optional SIMD backends, each `unsafe` block justified by a `// SAFETY:` comment and covered by tests against the portable backend). The verifier forbids `unsafe` in every crate.

### Cargo features

| Feature | Crates | Effect | Allowed in release builds |
|---|---|---|---|
| `gpu` | `tessera-gpu`, `tessera-prover`, `tessera-bench`, `tessera-cli` | Enables CUDA backend | Yes |
| `test-rng` | `tessera-prover`, `tessera-pcs`, `tessera-sumcheck` | Seeded masking randomness | No: `compile_error!` without `debug_assertions` |
| `test-params` | `tv-core`, `tessera-protocol` | Accepts the `test-small` parameter set | No |
| `unsafe-debug-dump` | `tessera-witness`, `tessera-prover` | Dumps secret tensors for debugging | No |
| `otel` | `tessera-cli` | OTLP trace export | Yes |

### Enforcement

`cargo xtask check-isolation` runs in CI on every PR and fails on any violation:

1. Runs `cargo metadata --format-version 1 --locked` inside `verifier/` and rejects any package whose manifest path lies outside `verifier/` and inside the repository, and any package named `tessera-*`.
2. Rejects any external package (registry or git source) not listed in `verifier/allowlist.toml`. Each allowlist entry has a name, an allowed major version and a reason.
3. Scans `verifier/` sources for `include!`, `include_str!`, `include_bytes!` and `#[path = ...]` whose argument resolves outside `verifier/`, and for symlinks anywhere under `verifier/`.
4. Checks that `verifier/rust-toolchain.toml` and the root `rust-toolchain.toml` pin the same toolchain.
5. Checks that the root workspace uses verifier crates only as `dev-dependencies` (DEP-2).

Initial allowlist (each with its reason recorded in the file): `blake3` 1.x (hash, RFC-0004), `sha2` 0.10.x (recorder digests, RFC-0015), `ed25519-dalek` 2.x (manifest signatures, RFC-0015), `rayon` 1.x (parallel column checks), `thiserror` 2.x (error types), `clap` 4.x (CLI only, `tv-cli`), `serde` 1.x and `serde_json` 1.x (verdict report output only, `tv-cli`), `rustls` 0.23.x (live TLS transport, `tv-transport` only, so verification logic never links a TLS stack; [RFC-0014](RFC-0014-interaction-and-soundness.md#live-transport)), plus their transitive dependencies as resolved in `verifier/Cargo.lock`, each audited with `cargo vet`.

### Shared artifacts that are not code

The specification documents and `spec/vectors/` are shared. Vectors are data files (TCE bytes, hex, JSON) generated by `cargo xtask gen-vectors`, which is prover-side code. A vector change therefore requires the verifier to agree independently: a PR that regenerates vectors must pass the verifier's vector tests unchanged, or it must change verifier code in the same PR with a second reviewer (see [09 §6](../spec/09-release-and-versioning.md#6-contribution-workflow)).

### Failure modes

| Failure | Response |
|---|---|
| A contributor adds `tessera-field` to `tv-core` | `check-isolation` fails with the offending edge |
| A transitive dependency of an allowlisted crate changes | `cargo vet` fails until audited |
| A vector regeneration hides a shared bug | Impossible to merge without the verifier passing the same vectors through its own code |
| CUDA toolkit absent on a contributor machine | Default build (no `gpu` feature) succeeds; GPU tests are skipped and run on the GPU runner |

## Alternatives Considered

1. **One workspace with the verifier as ordinary crates.** Considered for simpler builds and a single lockfile. Rejected: Cargo unifies features and versions across a workspace, so isolation would depend on discipline; one lockfile couples dependency upgrades of the trusted base to prover churn.
2. **A separate repository for the verifier.** Considered for the strongest isolation and separate release permissions. Rejected for v1 because the specification, vectors and verifier change together during development, and cross-repository CI doubles the coordination cost. The separate workspace keeps the option open (OQ-0001-1).
3. **A verifier in a different language** (for example Lean 4, OCaml or Go) for implementation diversity and an easier path to mechanized proofs. Rejected for v1: verification hashes gigabytes of columns and re-encodes 2^22-element rows, which needs a fast systems language, and splitting expertise across languages slows the critical path. Rust with no `unsafe` keeps paths open to Rust-to-proof-assistant extraction tools.
4. **PyO3 bindings between Python and Rust.** Rejected: it couples two build systems for one boundary crossing (export of `D`), and it would let the Python side bypass the specified artifact format.
5. **A C++ host prover.** Rejected: the CUDA kernels are separate sources in any case, and one host language for prover and verifier eases review, even though the codebases stay separate.

## Drawbacks

- Field arithmetic, the codec, hashing wrappers, table generators and gadget checks exist twice. Estimated duplication: 6,000–10,000 lines.
- Contributors who change a tier-A behavior must change two implementations and the vectors.
- Two lockfiles and two `deny.toml` files to maintain.

## Migration / Rollout

Greenfield. The bootstrap issue creates both workspaces, the Python package skeleton, `xtask check-isolation`, and the CI job before any functional code lands, so no crate ever exists outside the rule.

## Testing Strategy

- `check-isolation` has its own tests with fixture workspaces: (a) a verifier crate depending on a path crate outside `verifier/` fails; (b) a verifier crate with `include_str!("../../../crates/x.rs")` fails; (c) a symlink under `verifier/` fails; (d) a non-allowlisted registry crate fails; (e) a clean fixture passes.
- CI asserts that `cargo build -p tessera-gpu` without the `gpu` feature succeeds on a runner without CUDA.
- CI asserts that a release build with `test-rng`, `test-params` or `unsafe-debug-dump` fails to compile.

## Open Questions

- **OQ-0001-1.** Should the verifier move to its own repository at v1.0 to separate release permissions and audits? Owner: @AbdelStark. Resolve during v1.0 release planning, based on the external review's recommendation ([06 §9](../spec/06-security.md#9-security-gates)).

## References

- Source §6 principle 12; §7 "Anchor, sampling, verifier"; §9 problem 9.
- [01 Architecture §3](../spec/01-architecture.md#3-dependency-rules), [06 Security SP-6](../spec/06-security.md#4-security-properties).
- RFC-0019 (verifier design).
