# Log manifest vectors

`generate.py` writes `manifests.json` from [RFC-0015 'Log format' and 'Signatures'](../../../docs/rfcs/RFC-0015-transcript-anchor.md#log-format) and [RFC-0060](../../../docs/rfcs/RFC-0060-log-manifests.md), in standard-library Python, importing nothing from either Rust workspace. Run `python3 -I spec/vectors/manifest/generate.py`; `cargo xtask gen-vectors` runs it too.

- **Ed25519** is RFC 8032's reference arithmetic (Section 5.1), checked against its test vectors 1 to 3 before anything is written.
- **SHA-256 and SHA-512** are `hashlib`'s.
- **The envelope digest** that is a manifest's identity is the BLAKE3 of `spec/vectors/hash/generate.py`.
- **The entries** are encoded by [`anchor/`](../anchor/README.md)'s oracle.
- **RFC 9162's tree and audit paths** are written here.

`manifests.json` holds one log of three manifests, signed with one recorder seed. The first period's session is still open at the second period's start: the second period begins with its `SessionCarry` (RFC-0062) and closes it. The third period is empty, so its root is `SHA-256("")`. Each manifest gives:
- its entries' TCE encodings and leaves, its root, and every entry's inclusion proof;
- its period and first `seq`, and `prev_manifest`, the previous manifest's identity or zero for the first;
- its signed message, signature, TCE encoding, envelope and identity.

`crates/tessera-anchor/tests/manifest.rs` rebuilds the chain from the seed and checks every value.
