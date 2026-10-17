# Tessera

Tessera proves that a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls every piece of hardware and software, hides weights, activations, prompts and outputs from the verifier, and is designed to cost a small multiple of the inference it certifies. An independent verifier rebuilds every constraint from a public claim list and shares no code with the prover.

**Status:** pre-release. Milestone v0.1 (primitives and the unit-cost gate) is in progress. Nothing here is usable for audits yet, and every performance figure in the specification is a projection until it is measured.

## Specification

- [`SPEC.md`](SPEC.md): executive summary, reading order, release plan and risk register.
- [`docs/spec/`](docs/spec/): overview, architecture, public API, data model, error model, observability, security, testing, performance, release policy and glossary.
- [`docs/rfcs/`](docs/rfcs/): the accepted design decisions, RFC-0001 to RFC-0020, and RFC-0022 to RFC-0035.
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

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it on first use. The minimum supported Rust version is 1.88.

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo xtask lint                # unsafe-code policy, SPDX headers, licenses, Markdown links, report schemas
cargo xtask check-isolation     # the verifier shares no code with the prover (RFC-0001)
cargo xtask check-test-features # test-only features fail release builds (RFC-0001)
cargo xtask check-branchless    # the field compiles branch-free in x86_64 loops (INV-ARCH-2)

# The verifier is a separate workspace:
cd verifier && cargo build --locked && cargo test --locked
```

No crate needs a GPU or the CUDA toolkit to build or test. The CUDA backend is the `gpu` feature of `tessera-gpu`; it needs the CUDA toolkit 12.8 or later (`nvcc` on `PATH` or `CUDA_HOME` set) and builds kernels for `sm_90` (H100, H200) and `sm_100` (B200). Without CUDA, `TESSERA_GPU_SKIP_KERNELS=1 cargo clippy -p tessera-gpu --features gpu` type-checks the host code.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request and on `main`. Each job blocks merging:

| Job | What it runs |
|---|---|
| `checks` | rustfmt, clippy `-D warnings` and rustdoc `-D warnings` in both workspaces; `cargo xtask lint`; `cargo xtask check-isolation`; `cargo xtask check-test-features`; `cargo xtask check-branchless` |
| `test` | `cargo test --workspace --locked` in both workspaces |
| `dependencies` | `cargo xtask check-deps`: `cargo deny check` (licenses, advisories, bans, sources) in both workspaces, `cargo vet --locked` in the verifier, and a self-test showing the policies reject an MPL-2.0 crate and a crate under a RustSec advisory; cargo-deny 0.20.2 and cargo-vet 0.10.2 |
| `msrv` | `cargo +1.88.0 check --workspace --all-targets --locked` in both workspaces |
| `python` | `uv sync --locked --extra cpu`, `ruff`, `pyright` (strict) and `pytest` in `python/`, on Python 3.11, 3.12 and 3.13 ([python/README.md](python/README.md)) |
| `pr-policy` | `cargo xtask check-pr`: every change to a release train adds a changelog entry (`CHANGELOG.md` for `crates/*/src` and `python/tessera_quant/`, `verifier/CHANGELOG.md` for `verifier/`, either for `spec/`) unless the PR is labeled `no-changelog`; every commit is signed off (`git commit -s`, [09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow)) |

To run the dependency checks locally, install cargo-deny 0.20.2 and cargo-vet 0.10.2. `pr-policy` runs in its own workflow so that adding or removing the label re-runs only that check; locally, `cargo xtask check-pr --base origin/main` checks the current branch. A build-only job on macOS aarch64, a tier-2 platform ([09 §8](docs/spec/09-release-and-versioning.md#8-supported-platforms)), runs on `main` only. To reproduce a failure, run the job's commands locally; they are the commands listed above.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) describes the workflow, sign-off, changelog, review and RFC rules. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

Coding agents start from [AGENTS.md](AGENTS.md): ground truth, commands, the invariants a change must not break, and the project skills under `.claude/skills/`.

## License

Tessera is dual-licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT), at your option. Every Rust, Python and CUDA source file starts with the header `SPDX-License-Identifier: Apache-2.0 OR MIT`, and every Cargo and Python manifest declares the same expression; `cargo xtask lint` checks both.

Contributions are inbound = outbound: unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in Tessera is dual-licensed as above, without additional terms or conditions. The Developer Certificate of Origin sign-off on each commit affirms this ([09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow)).

Model weights are never redistributed. Test vectors use randomly generated weights, and users supply checkpoints under their own licenses ([09 §7](docs/spec/09-release-and-versioning.md#7-licensing)).
