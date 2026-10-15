# Changelog

All notable changes to the prover release train are recorded here: the crates under `crates/`, the `tessera` command-line tool and the `tessera-quant` Python package. The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [09 Release and versioning](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Added

- Prover workspace with the twenty crates of RFC-0001 as skeletons, edition 2024, minimum supported Rust 1.85 and an exact toolchain pin (1.95.0) ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `cargo xtask` dispatcher, and a test that enforces the unsafe-code policy of RFC-0001: every crate forbids unsafe code except `tessera-field` and `tessera-gpu`, which deny it crate-wide ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `deny.toml` license, advisory, ban and source policy for the prover workspace ([#25](https://github.com/AbdelStark/tessera/issues/25)).
