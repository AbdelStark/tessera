# RFC-0060: Log Manifests, Their Chain and Inclusion Proofs

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0015](RFC-0015-transcript-anchor.md#log-format) 'Log format': the digest that names a manifest, `prev_manifest`, the tree over a period's entries, and inclusion proofs;
  - [RFC-0015](RFC-0015-transcript-anchor.md#signatures) 'Signatures': the envelope's lengths, Ed25519's verification rule, and the key identifier's check;
  - [RFC-0043](RFC-0043-rejection-payloads.md) 'Non-Goals' and 'Anchor': the inclusion proofs' encoding, and the rejection of a manifest's own checks.
- Scope: what a manifest is and what makes one valid. The recorder that writes manifests is [#139](https://github.com/AbdelStark/tessera/issues/139)'s, and where the verifier requires a chain of them is [#150](https://github.com/AbdelStark/tessera/issues/150)'s and [#498](https://github.com/AbdelStark/tessera/issues/498)'s.
- Origin: [#138](https://github.com/AbdelStark/tessera/issues/138)

## Summary

RFC-0015 gives a recorder's `LogManifest` and its `SignatureEnvelope`. It leaves open:
- **Naming.** Which digest names a manifest, in a statement's `manifest_digest` and the next manifest's `prev_manifest`.
- **The chain.** What a chain of manifests must satisfy, and what a log's first manifest holds.
- **The tree.** Which tree a period's entries form, empty periods included.
- **Signatures.** How Ed25519 signatures are verified, and how long the envelope's key and signature are.

This RFC fixes them:
- **A manifest's identity** is its envelope digest. That is the digest of its `.tlm` artifact (schema `0x0402`, RFC-0003), which names it in a statement and in the next manifest.
- **The chain.** Consecutive manifests of a log share its identifier and link by identity. Their entries and their periods continue each other without gap or overlap. A log's first manifest links to 32 zero bytes, from entry 0.
- **The tree** is RFC 9162's over the period's entries in `seq` order, the empty tree for an empty period. Inclusion proofs are RFC 9162's.
- **Ed25519** is RFC 8032's, verified strictly, with a 32-byte key and a 64-byte signature. The key's SHA-256 is the manifest's `recorder_key_id`. Scheme 1 and every unknown scheme are refused.

## Motivation

[#138](https://github.com/AbdelStark/tessera/issues/138) implements manifests: their Merkle roots, signatures, inclusion proofs and chain. Its acceptance criteria include "chain break detected", and RFC-0015 never says what the chain must satisfy. Its other gaps would let two implementations disagree:
- **The manifest's name.** A statement's `manifest_digest` and a manifest's `prev_manifest` are 32 bytes each, with no hash named for either.
- **Verification rules.** Ed25519 implementations accept different signature sets. They differ on non-canonical `S`, on small-order keys and on the cofactored equation, so a verifier and a monitor could disagree on one manifest.
- **An empty period** has no stated root.

## Goals

- One identity per manifest, for the statement and the chain alike.
- Every check a holder of consecutive manifests can run, each with one outcome.
- One set of valid signatures, which the prover's and the verifier's implementations agree on.

## Non-Goals

- **The recorder** (#139) and its key management.
- **Which manifests a verifier requires.** RFC-0015 check 1 stays a check on one manifest, and #150 and #498 decide where a chain is required.
- **ML-DSA-65.** It is OQ-SEC-1's, and scheme 1 stays reserved.

## Proposed Design

### The manifest's identity

`identity(m)` is the envelope digest of `m` sealed as schema `0x0402`, version 1 (RFC-0003 'Envelope'). That is BLAKE3 in derive-key mode, with context `"tessera/v1/envelope"`, over the envelope's header and `TCE(m)`, the signature included. It is the last 32 bytes of `m`'s `.tlm` file.

A statement's `manifest_digest` is `identity(m)` for the manifest its batch is anchored to (03 INV-DM-7), and so is RFC-0016's `manifest_digest` in the sample seed.

### The tree over a period's entries

- **The entries.** A manifest covers the entries with `seq` from `first_seq` to `first_seq + entry_count − 1`, each exactly once, and no other, so `first_seq + entry_count` does not overflow. A recorder numbers its entries contiguously from 0, across periods.
- **Their times.** A period is the half-open interval `[period_start_ns, period_end_ns)` of the recorder's clock, and the recorder logs each entry, `t_ns` included, in the period that holds `t_ns`. A verifier does not check entry times: they are the recorder's (A-ANCHOR-2).
- **The leaves.** The entry with `seq = s` is the leaf at index `s − first_seq`, `SHA-256(0x00 ‖ TCE(entry))`.
- **The root.** `entries_root` is RFC 9162's Merkle tree hash over the leaves in index order (§2.1.1), interior nodes `SHA-256(0x01 ‖ left ‖ right)`. A period without entries has `entry_count = 0` and the empty tree's root, `SHA-256("")`.
- **Inclusion proofs.** They are RFC 9162's audit paths, generated as §2.1.3.1 and verified as §2.1.3.2.
  - **The encoding.** On the wire an `InclusionProof` is the path's digests, from the leaf upward: `Vec<[u8; 32]>` in TCE, with at most 64 digests, the path of a tree of `2^64` leaves. A longer one is refused before any hashing. It carries neither an index nor a size, unlike RFC 9162's own `InclusionProofDataV2`. This closes RFC-0043's deferral of the encoding.
  - **What fixes the proof's shape.** The leaf index comes from the entry, `seq − first_seq`, and the tree size and root from the signed manifest. Never from the proof: one path can verify against one root for several pairs of index and size. Leaf 0 of four leaves, for instance, verifies under the same root as leaf 0 of three.
  - **Out of range.** An entry whose `seq` lies outside the manifest's range is not in it.

### Signatures

`SignatureEnvelope { scheme: u16, public_key: Bytes, signature: Bytes }`, TCE-encoded as RFC-0003 gives (`Bytes` is a `u64` length and its bytes).

- **Scheme 0, Ed25519.** RFC 8032's pure Ed25519, without a context.
  - **The lengths.** `public_key` is exactly 32 bytes, the encoded point `A`; `signature` is exactly 64, `R ‖ S`.
  - **Strict verification.** This is `ed25519-dalek` 2's `verify_strict` (2.2.0, without `legacy_compatibility`), the crate RFC-0019's allowlist names for the verifier. #150 locks the same version and features in the verifier's lockfile. The signature verifies when:
    - `S < L`;
    - `R` and `A` decode to points, and neither is of small order. `curve25519-dalek` 4 decodes as RFC 8032 §5.1.3 does, except that it also accepts the non-canonical encodings `y ∈ [p, 2^255)`, which that section refuses;
    - `R`'s 32 bytes equal the encoding of `[S]B − [k]A`, with `k = SHA-512(R ‖ A ‖ message)` reduced modulo `L`. That is the cofactorless equation, and it leaves `R` only its canonical encoding.

    `k` hashes the key's 32 bytes as they are. A key is compared with the pinned keys byte for byte before it is decoded, so only a pinned encoding is ever decoded.
- **Scheme 1** (ML-DSA-65) is reserved and refused by v1 (OQ-SEC-1).
- **Any other scheme** is refused.
- **The key's identifier.** `recorder_key_id = SHA-256(public_key)`, which a verifier checks before the signature. The key must be one the verifier pins (RFC-0015 'Signatures').
- **The signed message.** `"tessera-manifest/v1" ‖ TCE(m without signature)`. The ASCII label has no terminator, and the second part is the encoding of every field but `signature`, in order: 19 + 32 + 16 + 8 + 8 + 8 + 8 + 32 + 32 = 163 bytes.

### The chain

For consecutive manifests `a` then `b` of one log:
1. `b.log_id = a.log_id`;
2. `b.prev_manifest = identity(a)`;
3. `b.first_seq = a.first_seq + a.entry_count`, with no overflow;
4. `b.period_start_ns = a.period_end_ns`.

The first manifest of a log has `prev_manifest` = 32 zero bytes and `first_seq = 0`.

Every manifest also passes its own checks, in this order:
1. a valid signature under a pinned key, as 'Signatures' orders its checks;
2. `period_start_ns ≤ period_end_ns`;
3. `first_seq + entry_count` without overflow;
4. for `entry_count = 0`, `entries_root = SHA-256("")`.

Keys may differ from one manifest to the next: a key rotation is a pinning of both keys.

A chain check over a sequence of manifests reports the first break:
- **First,** the genesis, on the first manifest.
- **Then, for each manifest in the sequence's order,** its own checks, followed by its link to the one before, in the order above.

## Security Argument

- **One name per manifest.** The envelope digest covers the whole manifest, its signature and its schema included. Two manifests share an identity only through a BLAKE3 collision.
- **A chain cannot fork silently.** A second manifest for a period has another identity, so its successor links to one of the two. A monitor holding both sees the fork: two signed manifests by the log's keys with one `prev_manifest`.
- **A chain cannot drop or repeat entries.** Contiguous `seq` ranges leave no entry outside some manifest and none in two. The manifests' periods tile the recorder's time without gap or overlap. That an entry's `t_ns` lies in its own manifest's period is the recorder's rule, not a check.
- **One set of valid signatures.** Under strict verification, a signature is valid for one message and key exactly when the shared crate says so. No implementation accepts a malleated signature, with `S` replaced by `S + L`, that another refuses.

## Alternatives Considered

1. **`prev_manifest` as SHA-256 of the previous manifest's TCE**, to keep the recorder to one hash function. The recorder computes envelope digests anyway, to write `.tlm` artifacts, and two identities per manifest would invite a mismatch between the statement and the chain. Rejected.
2. **Cofactored verification (ZIP-215)**, for batch verification. One manifest per statement needs no batching, and the shared crate's strict rule gives one acceptance set. Rejected.
3. **Leaving the empty period undefined** and requiring `entry_count ≥ 1`. A quiet period would then break the chain's continuity. Rejected.

## Drawbacks

- **Two hash functions in the recorder.** It needs BLAKE3 for envelopes, beside SHA-256 for the log.
- **Pinned keys are global.** A pinned key signs for any log and any period, so a retired key keeps its power while it stays pinned. Binding keys to logs or times is the recorder's trust assumption (A-ANCHOR-2) and out of scope.
- **Proofs cost a subtree each.** The reference `inclusion_proof` recomputes the path's sibling subtrees: `k` proofs over `n` entries cost about `k·(n − 1)` hashes. The recorder (#139) should keep the tree's levels.

## Migration / Rollout

- **#138.** `tessera-anchor`'s `LogManifest`, `SignatureEnvelope`, the tree, inclusion proofs, signing and verification, and the chain check.
- **#139** writes manifests with them. **#150** verifies a statement's manifest, and **#498** decides where a chain is checked.

## Testing Strategy

- **Inclusion proofs.**
  - The 98 RFC 6962 inclusion vectors of `spec/vectors/rfc6962/`, valid and invalid.
  - Proofs generated for random positions in trees of random sizes, verified by the prover's code and by the verifier's independent `tv-hash` (DEP-2).
  - A manifest's entry check: each entry at its own `seq`, and the refusals of a `seq` before or after the range, a proof of 65 digests, and another entry's proof.
- **Ed25519.**
  - RFC 8032's test vectors: in the oracle, and the Rust public key of test 1.
  - Each strict rule alone, refused: `S + L`; a small-order `R` under the honest key; a small-order key. The last two satisfy the cofactorless equation, which a lax verifier accepts.
  - A pinned key that is not a point, refused.
  - A wrong key identifier, an unpinned key, scheme 1, an unknown scheme, and wrong key and signature lengths, each refused.
- **A manifest's other checks.** An overflowing range and a wrong empty root, each refused.
- **The chain.** An honest chain of three passes. Each break is reported alone:
  - a wrong link;
  - another log identifier;
  - a gap or overlap of entries;
  - a gap of time;
  - a first manifest that is not the genesis;
  - a reordered pair.
- **Shared vectors.** An independent standard-library Python oracle writes three manifests of one log:
  - their entries, leaves and roots;
  - each manifest's signed message, signature and envelope;
  - each identity and the chain's links.

  It uses the BLAKE3 of `spec/vectors/hash/`, `hashlib`'s SHA-256 and SHA-512, and RFC 8032's reference arithmetic.

## Open Questions

None.

## References

- [RFC-0015](RFC-0015-transcript-anchor.md) 'Log format', 'Signatures', 'Verifier checks'; [RFC-0059](RFC-0059-frame-headers-and-nonces.md).
- [RFC-0003](RFC-0003-canonical-encoding.md) 'Envelope'; [RFC-0016](RFC-0016-audit-sampling.md) 'Sample derivation'; [RFC-0019](RFC-0019-verifier.md) 'Dependencies'.
- RFC 9162 (Certificate Transparency 2.0) §2.1; RFC 6962; RFC 8032 (EdDSA).
