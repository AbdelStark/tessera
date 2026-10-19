# Constraint-system diagnostic framing

`generate.py` writes `framing.json` with an independent Python tree encoder and the
existing independent Python BLAKE3 reference. It imports no production Rust code.
Run `python3 spec/vectors/constraint-digest/generate.py` or `cargo xtask gen-vectors`.

The twelve cases pin the diagnostic format described in
[the conformance guide](../../../docs/testing/constraint-digest.md): canonical bytes,
CI-only derive-key context and digest. They cover empty containers, scalar type
distinctions, integer boundaries, signed-i128 endpoints as strings, pointer escaping
and Unicode byte lengths. These bytes are outside the proof protocol.

`crates/tessera-e2e/examples/constraint_digest/tests.rs` and
`verifier/crates/tv-testkit/tests/constraint_digest.rs` consume the cases independently.
