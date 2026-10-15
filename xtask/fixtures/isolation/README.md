# Isolation-gate fixture

A miniature repository for the tests of `cargo xtask check-isolation` (`xtask/src/isolation/tests.rs`). The tests copy it to a temporary directory, add `rust-toolchain.toml` pins equal to the repository's, apply one violation at a time, and check that the gate reports exactly that violation.

- `crates/`: a prover workspace. `tessera-probe` uses the verifier crate `tv-a` as a dev-dependency, which DEP-2 allows. `shared-util` is a prover-side crate that verifier crates must not reach.
- `verifier/`: a verifier workspace with an allowlist that admits `blake3` 1.x and `clap` 4.x (`clap` for `tv-cli` only). `tv-cli` uses the test-support crate `tv-testkit` as a dev-dependency, which check 7 allows.
- `vendor/`: stand-ins for crates.io packages. The tests serve crates.io from this directory through cargo `--config` source replacement, so they never touch the network. None of these is the real crate.

This directory is not a member of the repository's workspace.
