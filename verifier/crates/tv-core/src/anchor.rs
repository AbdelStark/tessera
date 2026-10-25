// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transcript anchor's log checks: the recorder's manifest and its signature, the period's
//! verified log, the window of a period and the one before it, the period's candidates, the
//! sampler pins, and the anchor checks of a batch's requests (RFC-0015 'Verifier checks' 1 and 2,
//! RFC-0019 'Check order' steps 1 and 4, with RFC-0059, RFC-0060, RFC-0061, RFC-0062, RFC-0063 and
//! RFC-0043 'Anchor').
//!
//! They are written from those documents without reference to the prover's code, so that the
//! two agree only because both follow the specification (SP-6, RFC-0001). Where the
//! specification admits several algorithms this module takes its own: the log's root is RFC
//! 9162's tree hash computed as the entries stream past ([`tv_hash::rfc6962::TreeHash`]), never
//! from a list of leaves; and the rules read a per-session index of the batch's sessions, built
//! in one pass over each log.
//!
//! | Item | What it checks |
//! |---|---|
//! | [`DigestLogEntry::decode`] | one log entry, schema `0x0401`: its tag (0 to 4), its fields, a direction tag of 0 or 1 |
//! | [`LogManifest::open`] | a sealed manifest, `.tlm`, schema `0x0402`, and its identity, the envelope's digest |
//! | [`SealedManifest::check`] | a manifest's own checks: its signature under a pinned key, its period, its range, its empty root |
//! | [`VerifiedLog::check`] | a period's log, `.tdl`, decoded whole and checked against its manifest: count, `seq`, root |
//! | [`Window::bind`] | the previous period's log, bound to the period's through the chain of manifests |
//! | [`Window::candidates`] | the period's candidates, ended and unterminated, with their lengths |
//! | [`SamplerPins::load`] | the verifier's pinned sampler keys and policy, at load |
//! | [`check_batch`] | the anchor checks of every request of a batch, in batch order |
//!
//! # Specification
//!
//! What this module computes and rejects, in order, written down from the RFCs before the code.
//! The tests check every item against `spec/vectors/manifest/`, `spec/vectors/anchor/` and
//! `spec/vectors/audit/`, and with inputs that break exactly one check.
//!
//! ## Decoding
//!
//! The manifest and the log are the verifier's own inputs, delivered by the recorder (A-ANCHOR-2),
//! so a decoding failure is `Inconclusive(ERR-CODEC-*)`, never a rejection (RFC-0043
//! 'Malformed'). Decoding follows RFC-0003 through `tv-codec`, with its codes and offsets.
//!
//! - **An entry** is a `u16` tag, then its variant's fields: `SessionOpen` (0) `seq, t_ns,
//!   session_id: [u8; 16], key_commitment: [u8; 32]`, 66 bytes; `Frame` (1) `seq, t_ns,
//!   session_id, request_id: u64, direction, frame_index: u32, length: u32, frame_seq: u32,
//!   sha256: [u8; 32]`, 88 bytes; `SessionClose` (2) `seq, t_ns, session_id`, 34 bytes;
//!   `SessionCarry` (3) as an open, 66 bytes; `RequestEnd` (4) `seq, t_ns, session_id,
//!   request_id`, 42 bytes. A tag of 5 or more is `ERR-CODEC-NONCANONICAL` at the tag; so is a
//!   direction tag other than 0 (`In`) or 1 (`Out`), a TCE enum whose high byte must be 0. A
//!   frame's `length` is not checked here: a `length` that is not `22 + 4n` is a check of the
//!   request's frames, `Frames` (RFC-0059 'Verifier check 2, amended').
//! - **A period's log** is its entries' encodings concatenated, with no envelope and no length
//!   prefix (RFC-0062 'On disk'). It is checked against `max_total_bytes` first, then cut entry by
//!   entry until it ends; an entry cut short is `ERR-CODEC-TRUNCATED` at its field. Offsets count
//!   from the stream's start.
//! - **A manifest** is a `.tlm` envelope, schema `0x0402` version 1, opened in RFC-0003's order
//!   with the BLAKE3 `envelope` digest. Its payload is `recorder_key_id: [u8; 32], log_id:
//!   [u8; 16], period_start_ns, period_end_ns, first_seq, entry_count: u64, entries_root: [u8;
//!   32], prev_manifest: [u8; 32]`, then `SignatureEnvelope { scheme: u16, public_key: Bytes,
//!   signature: Bytes }`, a struct of its own. Any `scheme` and any lengths decode. Its identity is
//!   the envelope's digest, the file's last 32 bytes (RFC-0060 'The manifest's identity').
//!
//! ## The manifest's own checks
//!
//! In RFC-0060's order ('The chain', 'Signatures'), each failure `Reject(Anchor { request_id:
//! None, kind: ManifestSignature })` (RFC-0015 check 1, RFC-0043 'Anchor'):
//!
//! 1. the signature: `scheme` is 0, Ed25519 (scheme 1 is reserved, others unknown); `public_key`
//!    is 32 bytes and `signature` 64; `recorder_key_id = SHA-256(public_key)`; the key's bytes
//!    equal a pinned key's, compared before anything decodes the key; and
//!    [`tv_hash::ed25519::verify_strict`] accepts `signature` over `"tessera-manifest/v1" ‖
//!    TCE(m without signature)`, 163 bytes: RFC-0060's strict set, which refuses `S ≥ L`, a
//!    point that does not decode, a small-order `R` or key, and the cofactored equation alone;
//! 2. `period_start_ns ≤ period_end_ns`;
//! 3. `first_seq + entry_count` does not overflow;
//! 4. `entry_count = 0` implies `entries_root = SHA-256("")`.
//!
//! ## The period's log
//!
//! [`VerifiedLog::check`] runs once per manifest, after its own checks and before any entry is
//! used (RFC-0062 'The check', 'When'). After the stream decodes whole, in this order, each
//! failure `Inconclusive(ERR-CODEC-DIGEST)` (04 §3):
//!
//! 1. it holds exactly `entry_count` entries, else at the stream's end;
//! 2. the `i`-th entry's `seq` is `first_seq + i`, else at the first wrong entry's start;
//! 3. RFC 9162's tree hash over the leaves `SHA-256(0x00 ‖ TCE(entry))` is `entries_root`, else
//!    at the stream's end.
//!
//! A [`VerifiedLog`] holds the entries and the manifest they were checked against, and nothing
//! else constructs one, so every later use reads the checked copy.
//!
//! ## The window
//!
//! A period `P` and `P − 1`, the last period of non-zero length before it, reached through
//! `prev_manifest` and any zero-length periods between (RFC-0063 'The periods'). [`Window::bind`]
//! follows the chain back from `P`: each zero-length manifest between has no entries and precedes
//! the one after it, by RFC-0060's four links (`log_id`, `prev_manifest`, `first_seq`, period
//! continuity); `P − 1`'s manifest has non-zero length and precedes the earliest reached; with no
//! `P − 1`, that earliest manifest is a log's first (`prev_manifest` zero, `first_seq` 0). Any
//! other input is `Inconclusive(ERR-USAGE)` (RFC-0063 'The previous period's log', RFC-0019
//! step 1).
//!
//! ## Candidates
//!
//! [`Window::candidates`]: every `(session_id, request_id)` with a `RequestEnd` in `P` (ended),
//! and every one with a frame in `P − 1` and no `RequestEnd` in `P − 1` or `P` (unterminated),
//! sorted, each with the token count of all its frames in `P − 1` and `P`; a frame whose `length`
//! is not `22 + 4n` makes the count `2^64 − 1` (RFC-0063, RFC-0016 'Sample derivation' step 4). A
//! period of zero length has none.
//!
//! ## Sampler pins
//!
//! [`SamplerPins::load`] runs RFC-0061's load checks, each failure `Inconclusive(ERR-USAGE)`:
//! the policy (`window_ns ≠ 0`, `2·max_grace_ns ≤ window_ns`, `max_class_bits ≤ 64`), then each
//! pin in list order (`W ≠ 0`, `2·G ≤ W`, `b ≤ 64`, `from_ns ≥ received_ns`, `from_ns + W + G`
//! without overflow, `from_ns ≥` the previous `from_ns + W`, `W ≥ 2·G` of the previous pin). A pin
//! whose `G` or `b` exceeds the policy's is absent for the containment.
//!
//! ## A batch's anchor checks
//!
//! [`check_batch`], for a statement whose window and pins are bound: `P`'s identity is
//! `statement.manifest_digest`, else `Inconclusive(ERR-USAGE)`; then, request by request in
//! `statement.batch` order, its session's rules, then its own frames, then its sampler key, the
//! first failure `Reject(Anchor { request_id: Some(r), kind })` with the kinds of RFC-0043:
//! `Session`, `Frames` and `SamplerKey`. [`check_batch`] lists every rule in its order.

mod entry;
mod log;
mod manifest;
mod pins;
mod rules;

pub use entry::{DigestLogEntry, Direction, FrameEntry};
pub use log::{Candidate, VerifiedLog, Window, log_failure};
pub use manifest::{
    CheckedManifest, ED25519, LogManifest, ManifestError, RecorderKeys, SIGNED_LABEL, SIGNED_LEN,
    SealedManifest, SignatureEnvelope,
};
pub use pins::{PinError, PinRule, PinnedSamplerKey, SamplerPins, SamplerPolicy};
pub use rules::check_batch;

use crate::verdict::{ErrorCode, RejectReason, Verdict, VerifierError};

/// How an anchor check ends when it does not pass: a rejection of the proof, or the verifier's
/// inability to judge it (04 §1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The proof's anchor fails: `Reject(Anchor { .. })`.
    Rejected(RejectReason),
    /// The verifier's own inputs or resources fail: `Inconclusive`.
    Inconclusive(VerifierError),
}

impl From<Failure> for Verdict {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Rejected(reason) => Self::Reject(reason),
            Failure::Inconclusive(error) => Self::Inconclusive(error),
        }
    }
}

impl From<ManifestError> for Failure {
    fn from(error: ManifestError) -> Self {
        Self::Rejected(error.reason())
    }
}

impl From<PinError> for Failure {
    fn from(error: PinError) -> Self {
        error.failure()
    }
}

/// `Inconclusive(ERR-USAGE)`: the verifier's own inputs disagree.
pub(crate) const fn usage() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Usage,
    })
}
