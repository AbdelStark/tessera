---
name: verifier-independence
description: How to write code under verifier/ (the tv-* crates) so it shares no bug with the prover. Use when implementing or testing a verifier crate, or adding a dependency to one.
---

# Verifier independence

The verifier is the whole trusted base of an audit (RFC-0019) and must agree with the prover only because both follow the specification (SP-6). Two implementations that share a misreading of the spec are one implementation.

1. **Specify first.** Write down, from the RFC and data-model sections alone, what the component computes and rejects. Done when every error code and reject reason it can return is listed with its trigger.
2. **Implement from that specification**, not from the prover's code. Where the spec admits several algorithms, choose a different one from the prover's (another reduction, evaluation order or traversal), so that a structural bug cannot sit in both. Done when the crate compiles under the verifier's lints: no `unsafe`, and in `tv-core` and `tv-codec` no `unwrap`, `expect`, `panic!` or unchecked indexing.
3. **Test against independent data**: official test vectors, and vectors in `spec/vectors/` produced by an independent oracle such as a Python big-integer script. Read them, and draw seeded random cases, with `tv-testkit` as a dev-dependency: test crates such as `serde_json` or `proptest` would join the audited dependency graph. Differential tests that run prover code against the verifier live in prover crates, which may use verifier crates as dev-dependencies (DEP-2), as `crates/tessera-hash/tests/verifier.rs` does. Done when every vector passes and every rejection has a test that triggers it.
4. **Compare** with the prover's implementation only after step 3 is green, to understand a disagreement, never to copy.

## Dependencies

Every third-party package in the verifier's graph, transitive ones included, is listed in `verifier/allowlist.toml` with its major version and reason, and passes `cargo vet` as `safe-to-deploy` with no exemption (#50). Before proposing a crate, measure what it costs to audit: add it in a scratch copy and run `cargo vet` there; the lines no imported audit covers are what the second reviewer would have to accept. `blake3` and `sha2` failed that test, about 248,000 unaudited lines across 17 crates, so `tv-hash` implements both hash functions itself (RFC-0019 'Dependencies'). When a change adds a crate: add an entry for it and for each new transitive package, then run `cargo xtask check-isolation` and `cargo xtask check-deps`. Prefer writing a small routine over adding a crate; each crate is audit surface for every party that runs the verifier.

## Review

Changes under `verifier/` need a second reviewer (09 §6). State in the PR whether one has reviewed it.
