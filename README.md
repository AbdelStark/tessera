# Tessera

Tessera proves that a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls every piece of hardware and software, hides weights, activations, prompts and outputs from the verifier, and is designed to cost a small multiple of the inference it certifies. An independent verifier rebuilds every constraint from a public claim list and shares no code with the prover.

**Status:** pre-release. Milestone v0.1 (primitives and the unit-cost gate) is in progress. Nothing here is usable for audits yet, and every performance figure in the specification is a projection until it is measured.

The reference CPU path now exposes [`tessera_prover::Prover`](docs/spec/02-public-api.md#31-prover)
for one exact-greedy request through NI and Live sessions. Tiny-model acceptance
tests use `test-small/1`, which certifies nothing. The current protocol/1 path
provides no zero knowledge. The [independent verifier CLI](docs/verifier/reference-cli.md)
checks NI proofs against caller-owned statement and weight pins. Prover CLI
orchestration and the pass planner remain separate milestones.

Optional metrics and collector export are described in the [telemetry guide](docs/telemetry.md).
The prover protocol library exposes [Live directory and optional mTLS transports](docs/live-transports.md),
with host-owned sockets and identity configuration. The CLI's Live interface remains directory-based.

## Specification

- Build or download the [navigable documentation artifact](docs/documentation.md), including both Rust API references.
- [`SPEC.md`](SPEC.md): executive summary, reading order, release plan and risk register.
- [`docs/spec/`](docs/spec/): overview, architecture, public API, data model, error model, observability, security, testing, performance, release policy and glossary.
- [`docs/rfcs/`](docs/rfcs/): design decisions and draft proposals; each document records its status.
- [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md): the issue tracker generated from the corpus.

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Prover workspace: one crate per component of [01 Architecture](docs/spec/01-architecture.md#2-components) |
| `verifier/` | Verifier workspace: an independent implementation of every check, with its own lockfile and a third-party allowlist ([RFC-0019](docs/rfcs/RFC-0019-verifier.md)) |
| `xtask/` | Repository automation, run as `cargo xtask <command>` |
| `docs/` | Specification, RFCs and roadmap |

The layout and the verifier-independence rules it enforces are fixed by [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md). The verifier never depends on prover code; only shared data (the specification and test vectors) connects the two workspaces.

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it on first use. The minimum supported Rust version is 1.89.

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo xtask lint                # unsafe-code policy, SPDX headers, licenses, Markdown links, report schemas
cargo xtask check-isolation     # the verifier shares no code with the prover (RFC-0001)
cargo xtask check-test-features # test-only features fail release builds (RFC-0001)
cargo xtask check-mask-rank     # every mask row's revealed points are independent (R10, #488)
cargo xtask check-branchless    # the field compiles branch-free in x86_64 loops (INV-ARCH-2)

# The verifier is a separate workspace:
cd verifier && cargo build --locked && cargo test --locked
```

No crate needs a GPU or the CUDA toolkit to build or test. The CUDA backend is the `gpu` feature of `tessera-gpu`; it needs the CUDA toolkit 12.8 or later (`nvcc` on `PATH` or `CUDA_HOME` set) and builds kernels for `sm_90` (H100, H200) and `sm_100` (B200). Without CUDA, `TESSERA_GPU_SKIP_KERNELS=1 cargo clippy -p tessera-gpu --features gpu` type-checks the host code.

## Continuous integration

During the fast-build phase, [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/pr-policy.yml`](.github/workflows/pr-policy.yml) are the workflows that run automatically on pull requests (`ci.yml` also runs on `main`). `ci.yml` runs on commit lifecycle events (`opened`, `synchronize`, `reopened`), while `pr-policy.yml` also runs on `labeled` and `unlabeled` so label edits rerun only the policy gate; these five checks block merging:

| Job | What it runs |
|---|---|
| `checks` | rustfmt, clippy `-D warnings`, repository lint, research KB, verifier isolation, test-only feature, mask-rank and branchless-field gates |
| `pr-policy` | PR changelog entry (waived by `no-changelog`) and DCO sign-off; the only check rerun on label events |
| `test` | Parallel Nextest runs on separate prover/verifier runners; runtime library units, primitive conformance/adversarial tests, prover components, tiny-model integration and the complete independent verifier, plus test-parameter unit and claim validation coverage |
| `python` | `uv sync --locked --extra cpu`, Ruff, strict Pyright and pytest on Python 3.13 ([python/README.md](python/README.md)) |
| `constraint-digest` | Independent constraint-system digest conformance and the tiny servable-model case |

In the core pipeline, documentation-only changes (root Markdown, `docs/`, `research/`) run routing tests and repository/PR policies without rebuilding unchanged code. Unknown paths and manual dispatch run the complete bootstrap pipeline. CI omits debug symbols while retaining overflow checks and debug assertions.

The [documentation workflow](.github/workflows/docs.yml) builds the Markdown book and both Rust API references, compiles tier-B Rust examples, checks rendered links and uploads a private artifact on every PR and main push. See the [build procedure](docs/documentation.md).

Fuzz, soundness calibration, microbenchmarks and PR-policy also remain available with manual dispatch from the Actions page. [The manual quality workflow](.github/workflows/quality.yml) retains dependency audits, MSRV, macOS builds, GPU host checks, rustdoc and both complete workspace suites. Exhaustive vector-oracle regeneration, infrastructure self-tests, standalone compiler/witness stress tests, benchmark/report tests and the remaining doctests run there; the constraint-digest example runs once in its dedicated automatic job. They do not run automatically during fast backlog development. The normal checks still protect verifier independence, secret handling, protocol tests, Python packaging and the existing tiny-model conformance path. To reproduce a failure, run the job's commands locally.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) describes the workflow, sign-off, changelog, review and RFC rules. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

Coding agents start from [AGENTS.md](AGENTS.md): ground truth, commands, the invariants a change must not break, and the project skills under `.claude/skills/`.

## License

Tessera is dual-licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT), at your option. Every Rust, Python and CUDA source file starts with the header `SPDX-License-Identifier: Apache-2.0 OR MIT`, and every Cargo and Python manifest declares the same expression; `cargo xtask lint` checks both.

Contributions are inbound = outbound: unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in Tessera is dual-licensed as above, without additional terms or conditions. The Developer Certificate of Origin sign-off on each commit affirms this ([09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow)).

Model weights are never redistributed. Test vectors use randomly generated weights, and users supply checkpoints under their own licenses ([09 §7](docs/spec/09-release-and-versioning.md#7-licensing)).

The [exact-mode CLI suite](docs/testing/exact-cli.md) runs the CPU compile → commit → prove → independent verify path in NI and directory Live modes. Its synthetic test parameters certify nothing.
