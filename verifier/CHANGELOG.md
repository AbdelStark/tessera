# Changelog

All notable changes to the verifier release train are recorded here: the crates under `verifier/` and the `tessera-verify` command-line tool. The verifier is versioned independently of the prover (tags `verify-vX.Y.Z`); the two trains are compatible through protocol versions only ([09 §1](../docs/spec/09-release-and-versioning.md#1-what-is-versioned)). The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Verifier workspace with the seven `tv-*` crates of RFC-0001 as skeletons, its own lockfile, the same toolchain pin as the prover workspace, a stricter `cargo-deny` policy and `allowlist.toml`, the third-party packages permitted in the trusted base ([#28](https://github.com/AbdelStark/tessera/issues/28)).
- Every verifier crate forbids unsafe code; `tv-core` and `tv-codec` also deny `unwrap`, `expect`, `panic!` and unchecked indexing outside tests (RFC-0019) ([#28](https://github.com/AbdelStark/tessera/issues/28)).
- `tv-core` declares the `test-params` feature, which will accept the `test-small` parameter set; a release build with it fails to compile, so test parameters can never certify a statement (02 §5, RFC-0001) ([#37](https://github.com/AbdelStark/tessera/issues/37)).
