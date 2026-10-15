# Dependency-policy fixture

Self-test inputs for `cargo xtask check-deps` (`xtask/src/deps.rs`). The command copies this directory to a temporary location and checks that the repository's `cargo-deny` policies reject what they must:

- `licensed/` depends on `mpl-licensed`, which declares `MPL-2.0`. The verifier policy (`verifier/deny.toml`) must reject it; the prover policy (`deny.toml`) must reject it until an exception with a `# Reason:` comment is added.
- `vulnerable/` depends on `smallvec` 1.6.0, which RustSec advisory RUSTSEC-2021-0003 covers. The advisory check must reject it.

`vendor/` holds stand-ins for these crates.io packages, served through the source replacement in `.cargo/config.toml`, so the checks need no crate downloads. Neither stand-in is the real crate. This directory is not a member of the repository's workspace.
