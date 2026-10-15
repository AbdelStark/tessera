# Changelog

All notable changes to the prover release train are recorded here: the crates under `crates/`, the `tessera` command-line tool and the `tessera-quant` Python package. The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [09 Release and versioning](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Added

- Prover workspace with the twenty crates of RFC-0001 as skeletons, edition 2024, minimum supported Rust 1.85 and an exact toolchain pin (1.95.0) ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `cargo xtask` dispatcher, and a test that enforces the unsafe-code policy of RFC-0001: every crate forbids unsafe code except `tessera-field` and `tessera-gpu`, which deny it crate-wide ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `deny.toml` license, advisory, ban and source policy for the prover workspace ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `LICENSE-APACHE` and `LICENSE-MIT`, and `cargo xtask lint`, which checks the unsafe-code policy, the SPDX header of every Rust, Python and CUDA source file, and the dual license in every Cargo and Python manifest ([#26](https://github.com/AbdelStark/tessera/issues/26)).
- `cargo xtask lint` checks the verifier crates too: each must forbid unsafe code, with no exemptions ([#28](https://github.com/AbdelStark/tessera/issues/28)).
- `cargo xtask check-isolation`, the verifier-independence gate of RFC-0001 'Enforcement'. It implements the RFC's five checks: no `tessera-*` or out-of-tree path package in the verifier graph; every third-party package from crates.io, allowlisted at a compatible version and scoped to its `direct-dependents`; no file reference or symlink reaching outside `verifier/`; equal toolchain pins; verifier crates only as prover dev-dependencies in the resolved graph. It adds hardening against bypasses found in review: every target of a verifier path package must compile a source inside `verifier/` (`[lib] path` can point anywhere), no build scripts or procedural macros in the verifier, include and `path` arguments must be plain literals, and no `[patch]`/`[replace]`, cargo configuration or alias rewrite that could redirect the verifier's dependencies or neuter the gate ([#36](https://github.com/AbdelStark/tessera/issues/36)).
