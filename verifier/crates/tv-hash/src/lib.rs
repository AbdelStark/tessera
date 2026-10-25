// SPDX-License-Identifier: Apache-2.0 OR MIT

//! BLAKE3 domains, chained column digests, Merkle multiproofs, transcript replay and proof-of-work
//! checks, the SHA-256 Merkle tree of recorder logs and the strict Ed25519 verification of their
//! manifests (RFC-0004, RFC-0015, RFC-0060), written from the specifications without reference to
//! the prover's hashing crate, so that the two agree only because both follow the specification
//! (SP-6, RFC-0001).
//!
//! The crate takes no third-party code (RFC-0019 'Dependencies'): BLAKE3 is written from its
//! specification (O'Connor, Aumasson, Neves and Wilcox-O'Hearn, 2020), SHA-256 and SHA-512 from
//! FIPS 180-4 and Ed25519 from RFC 8032 with RFC-0060's acceptance set, in safe Rust. Longer
//! BLAKE3 inputs are hashed several chunks at a time, in a layout the compiler turns into vector
//! instructions.
//!
//! | Item | What it computes or checks |
//! |---|---|
//! | [`hash`], [`DomainHasher`], [`Domain`] | `H(domain, bytes)` for the 19 domains of RFC-0004 |
//! | [`column::column_digest`] | a column's chained digest, its Merkle leaf |
//! | [`merkle::node`], [`merkle::verify_multiproof`] | a commitment tree's nodes and canonical multiproofs |
//! | [`transcript::Transcript`] | the Fiat–Shamir transcript, its challenges and proof-of-work checks |
//! | [`sha256::sha256`], [`sha256::Sha256`] | SHA-256 |
//! | [`sha512::sha512`], [`sha512::Sha512`] | SHA-512 |
//! | [`ed25519::verify_strict`] | a recorder manifest's Ed25519 signature, strictly |
//! | [`rfc6962::leaf_hash`], [`rfc6962::node_hash`], [`rfc6962::verify_inclusion`] | the recorder log's tree and inclusion proofs |
//! | [`rfc6962::TreeHash`] | the root of a period's log, RFC 9162's tree hash over its leaves as they arrive |
//!
//! # Specification
//!
//! What this crate computes and rejects, written down from RFC-0004 ('Hash function and
//! domains', 'Column digests', 'Merkle trees', 'Transcript', 'Failure modes'), RFC-0014
//! ('Parameter sets', 'Grinding'), RFC-0015 ('Log format') and RFC 9162 §2.1 before the code. The
//! tests check every item against `spec/vectors/` and with inputs that violate exactly one check.
//!
//! Every input yields a value or a typed error: no input panics or loops without bound
//! (RFC-0019 'Robustness'). Only two functions allocate, each exactly its result:
//! `challenge_indices` its `count` indices, a number from the verifier's parameter set, and
//! `SessionContext::to_tce` the context's encoding.
//!
//! ## Hashing
//!
//! - `H(domain, x) = BLAKE3.derive_key(context = "tessera/v1/" + domain, key_material = x)`, 32
//!   bytes, for the domains `envelope`, `column-init`, `column-step`, `merkle-node`,
//!   `column-merge`, `weights-root`, `transcript-init`, `transcript-absorb`,
//!   `transcript-squeeze`, `transcript-ratchet`, `pow`, `coin-commit`, `coin-toss`, `table`,
//!   `audit-sample`, `audit-select`, `claim-list` and `statement`. Each context key is derived
//!   once, by the compiler. Inputs of every length hash; nothing fails.
//! - SHA-256 of FIPS 180-4, for messages shorter than `2^61` bytes, its domain; nothing fails.
//! - SHA-512 of FIPS 180-4, for messages shorter than `2^64` bytes; nothing fails.
//!
//! ## Column digests
//!
//! [`column::column_digest`]`(commitment_id, j, B, symbols)` is `d_j[ceil(m / B)]` with
//! `d_j[0] = H("column-init", commitment_id ‖ u64(j))` and
//! `d_j[i+1] = H("column-step", d_j[i] ‖ the 8-byte encodings of rows iB .. min((i+1)B, m) − 1)`,
//! `m = symbols.len()`, possibly 0. No check: `B ≥ 1` by type (`NonZeroU32`), symbols are
//! canonical by type (`Fp`), and `m` is checked by the caller against the public layout.
//!
//! ## Merkle multiproofs
//!
//! [`merkle::verify_multiproof`]`(root, n, leaves, siblings)`, `n` from public data and never from
//! the proof, rejects with [`merkle::MultiproofError`], first failure in this order:
//!
//! 1. `n` is not a power of two, zero included: `TreeSize`.
//! 2. No leaf is opened: `NoLeaves`.
//! 3. Leaf by leaf: an index not above the previous one (unsorted or duplicate):
//!    `Unordered { position }`; then an index `≥ n`: `OutOfRange { position }`.
//! 4. Fewer digests than the canonical multiproof of these indices holds:
//!    `MissingSiblings { expected, found }`; more: `ExtraSiblings { expected, found }`.
//! 5. The root recomputed with each sibling at its canonical place (level by level from the
//!    leaves, left to right within a level) differs from `root`: `RootMismatch`. This catches a
//!    wrong leaf digest, a wrong sibling and misordered siblings.
//!
//! Checks 4 and 5 judge the proof. Checks 1 to 3 judge the verifier's own inputs, `n` and the
//! indices, sorted and deduplicated from the transcript's draws: their errors mean a fault of the
//! verifier, never of a proof.
//!
//! ## Transcript
//!
//! [`transcript::Transcript`] replays RFC-0004's transcript:
//!
//! - [`transcript::Ascii::new`]: a byte outside `0x20..=0x7E` is `NotAscii { position }`, at the
//!   first such byte. Labels, the protocol name and the parameter set are `Ascii`, built by the
//!   verifier, so this is a fault of the verifier, never of a proof.
//! - `TCE(SessionContext) = Ascii(protocol) ‖ u16(mode: Live 0, NonInteractive 1) ‖
//!   Ascii(params) ‖ statement_digest ‖ session_id`; `state = H("transcript-init", ·)`.
//! - `absorb`, `squeeze`: as RFC-0004 writes them, labels as TCE `Ascii`; nothing fails.
//! - `challenge_fp`: the first word below `P` of the first squeeze that has one;
//!   `challenge_fp2`: the first two words of the first squeeze where both are below `P`. More
//!   than [`transcript::MAX_CHALLENGE_SQUEEZES`] squeezes is `Exhausted`, which happens with
//!   probability below `2^-496` and is `Inconclusive(ERR-INTERNAL)`, never a verdict on the proof
//!   (RFC-0004 'Failure modes').
//! - `challenge_indices(label, log_n, count)`: `log_n > 64` is `LogN`, and memory that cannot be
//!   reserved for `count` indices is `Memory`, both before any squeeze.
//! - `check_grind(label, bits, nonce)`: `H("pow", state ‖ Ascii(label) ‖ u64(nonce))` with fewer
//!   than `bits` leading zero bits (the most significant bit of byte 0 first) is
//!   [`transcript::ProofOfWorkFailed`] (`Reject(ProofOfWork)`), and the transcript, taken by
//!   value, is gone: nothing can be squeezed after a failed proof of work. A valid nonce is
//!   absorbed under the same label, whether or not it is the smallest.
//!
//! ## RFC 6962 trees
//!
//! [`rfc6962::leaf_hash`] is `SHA-256(0x00 ‖ entry)` and [`rfc6962::node_hash`]
//! `SHA-256(0x01 ‖ left ‖ right)`. [`rfc6962::TreeHash`] computes RFC 9162's tree hash (§2.1.1)
//! of the leaves pushed into it, `SHA-256("")` for none, in constant memory (RFC-0060 'The tree
//! over a period's entries'); a push past `2^64 − 1` leaves is `TreeFull`, and the tree is
//! unchanged. [`rfc6962::verify_inclusion`] follows RFC 9162 §2.1.3.2 for
//! trees of any size, rejecting with [`rfc6962::InclusionError`]: `leaf_index ≥ tree_size` (an
//! empty tree included) is `IndexBeyondSize`; proof digests left when the path reaches the root,
//! `ProofTooLong`; a proof that ends below the root, `ProofTooShort`; another root,
//! `RootMismatch`. The tree size and the root come together from the signed manifest, and the
//! leaf index from the entry: a proof can verify against one root for several pairs of a leaf
//! index and a size whose paths have the same shape.
//!
//! ## Ed25519
//!
//! [`ed25519::verify_strict`]`(A, M, R ‖ S)` accepts RFC-0060's set and rejects with
//! [`ed25519::Ed25519Error`], first failure in this order: `S ≥ L` (`ScalarOutOfRange`); `R`, then
//! `A`, not a point (`RNotAPoint`, `KeyNotAPoint`), decoding by RFC 8032 §5.1.3 with `y ∈ [p,
//! 2^255)` reduced instead of refused; `R`, then `A`, of small order (`RSmallOrder`,
//! `KeySmallOrder`); `R`'s bytes not the encoding of `[S]B − [k]A`, `k = SHA-512(R ‖ A ‖ M) mod L`
//! (`Equation`). The [module](ed25519) cites RFC-0060 for each rule.

#![forbid(unsafe_code)]
// Proofs come from a hostile prover and must be rejected with typed errors, never by panicking
// (RFC-0019 'Robustness'). Size arithmetic is checked or explicitly wrapping, and conversions
// are explicit.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::as_conversions
    )
)]

mod blake3;
pub mod column;
mod domain;
pub mod ed25519;
pub mod merkle;
pub mod rfc6962;
pub mod sha256;
pub mod sha512;
pub mod transcript;

pub use domain::{Domain, DomainHasher, hash};

/// A 32-byte digest: the output of every hash of this crate.
pub type Digest = [u8; 32];
