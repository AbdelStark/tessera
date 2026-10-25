// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit's accounting: the policy, the verifier's coin toss, the sample derived from the
//! period's verified logs, batching, the ledger and its checks, and the audit report (RFC-0016
//! 'Policy', 'Sample derivation', 'Batching', 'Verifier checks' and 'Audit report'; RFC-0019
//! 'Check order' step 4 and 'Ledger'; RFC-0063 'Which requests a period draws from'; RFC-0067,
//! the ledger's encoding, its file and the verifier's own sample).
//!
//! They are written from those documents without reference to the prover's code, so that the
//! two agree only because both follow the specification (SP-6, RFC-0001). Where the
//! specification admits several algorithms this module takes its own: a candidate's bucket is
//! found by binary search over the buckets' bounds; a policy file's threshold is one division of
//! a 128-bit integer; the ledger's record of a sample is rebuilt by merging the old and new
//! indices; and first-fit decreasing finds a batch's room through a tree of maxima. Nothing here is floating point: the planning figures are [`audit_plan`]'s
//! (INV-DM-6).
//!
//! [`audit_plan`]: crate::audit_plan
//!
//! | Item | What it computes or checks |
//! |---|---|
//! | [`RateBucket::threshold_of`] | a policy file's decimal rate as its exact threshold over `2^64` |
//! | [`AuditPolicy::new`] | a policy's rules: buckets strictly increasing, `max_batch_tokens ≥ max(1, 2 · min_batch_tokens)` |
//! | [`CoinTossTranscript::check`] | `c_V = H("coin-commit", s_V)` |
//! | [`AuditToss`] | the verifier's share: committed after the policy, revealed after both logs are verified, the sample recorded in the ledger as the verifier's own |
//! | [`AuditSample::derive`] | the sample of a window's period: seed, candidates, selection |
//! | [`AuditSample::open`] | a `.tas` artifact, schema `0x0403`, and its identity |
//! | [`pack`] | first-fit decreasing batches of a sample's requests |
//! | [`VerifierLedger::check_accounting`] | the verifier's own sample, then RFC-0016 checks 1 and 2 for a proof's statement |
//! | [`VerifierLedger::record_accepted`] | an accepted proof's requests and their `Û` |
//! | [`VerifierLedger::open`], [`LedgerFile`], [`read_ledger`] | a `.tvl` artifact, schema `0x0701`, and its file: one regular file of one name, locked, written atomically with its directory synced |
//! | [`AuditReport::from_ledger`] | the report at the deadline, `tessera.audit-report/1` |
//!
//! # Specification
//!
//! What this module computes and rejects, in order, written down from the RFCs before the code.
//! The tests check every item against `spec/vectors/audit/` and with inputs that break exactly
//! one rule.
//!
//! ## The policy
//!
//! `AuditPolicy { period_ns: u64, buckets: Vec<RateBucket>, min_batch_tokens: u32,
//! max_batch_tokens: u32, proof_deadline_ns: u64, params: ParamSetId }`, `RateBucket {
//! max_tokens: u32, threshold: u64 }`, TCE in field order, the name an `Ascii`. Beyond RFC-0003's
//! checks, the decoder refuses a bucket whose `max_tokens` is not above the previous bucket's,
//! `ERR-CODEC-NONCANONICAL` at the bucket's first byte, and a `max_batch_tokens` of 0 or below
//! `2 · min_batch_tokens`, `ERR-CODEC-LIMIT` at `max_batch_tokens`. A policy file's rate is `0`,
//! `1`, or `0.` or `1.` followed by one to 19 digits, at most 1; its threshold is `⌊rate ·
//! 2^64⌋`, capped at `2^64 − 1`. [`AuditPolicy::new`] applies the same two rules, as
//! [`PolicyError`]s; a policy file that breaks one is `ERR-USAGE`.
//!
//! ## The sample
//!
//! 1. **The logs.** The period's manifest passes its own checks, its signature first, and its
//!    log and the previous period's are verified and bound into a [`Window`]
//!    ([`crate::anchor`]). Nothing below runs on anything else.
//! 2. **The coin toss.** [`AuditToss::commit`] draws `s_V`, 32 fresh bytes, once the policy is
//!    fixed, and gives `c_V = H("coin-commit", s_V)`; [`AuditToss::reveal`] runs only over a
//!    window of the toss's manifest, `Inconclusive(ERR-USAGE)` otherwise, and records the
//!    sample it derives in the ledger's `derived`, the file written, before it returns the
//!    sample: the only way a sample becomes the verifier's own (RFC-0067 'The verifier's own
//!    sample'). The transcript
//!    `CoinTossTranscript { c_V, s_P, s_V }` is 96 bytes; a reveal that does not open `c_V` is
//!    `Inconclusive(ERR-PROTO-COINTOSS)`.
//! 3. **The seed.** `sample_seed = H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`, the
//!    manifest digest the period's identity.
//! 4. **The selection.** Each candidate of [`Window::candidates`], with its length `L`, is
//!    selected iff `u64_le(H("audit-select", sample_seed ‖ session_id ‖ u64(request_id))[0..8])`
//!    is below the threshold of the first bucket whose `max_tokens` is at least `L`, or `L` is
//!    past every bucket. A period of zero length has no candidates, and a frame of impossible
//!    length makes `L = 2^64 − 1`.
//! 5. **The artifact.** `AuditSample { manifest_digest, policy, coin_toss, selected:
//!    Vec<RequestKey> }`, schema `0x0403` version 1, `RequestKey { session_id: [u8; 16],
//!    request_id: u64 }`. `selected` strictly increases by `session_id` as bytes, then
//!    `request_id` as a number, not as its little-endian bytes (RFC-0067 'Request order'): a key
//!    not above the one before it is `ERR-CODEC-NONCANONICAL` at its first byte. Every list of
//!    requests here sorts so. The sample's identity is its envelope's digest.
//!
//! ## Batching
//!
//! [`pack`]: within each group, decreasing `L`, ties in `RequestKey` order; each request into
//! the first batch, in opening order, that it fits under `max_batch_tokens`, else a new one; a
//! request longer than `max_batch_tokens` alone. The first batch with room is found through a
//! tree of maxima over the batches' room, so `n` requests cost `O(n log n)`. A request listed
//! twice is `Inconclusive(ERR-USAGE)`.
//!
//! ## The accounting checks
//!
//! For a proof's statement, after its fields before `manifest_digest` (RFC-0039 'The checks',
//! RFC-0043 'The audit sample's checks', RFC-0067 'The accounting'), first the sample: one not in
//! the ledger's `derived` is `Inconclusive(ERR-USAGE)`, before any check of the statement, since
//! a sample whose coin toss opens could still be the prover's. Then:
//!
//! 1. `statement.manifest_digest` is the sample's, else `Reject(StatementMismatch { field:
//!    ManifestDigest })`;
//! 2. in batch order, each request is in `selected`, was in no earlier accepted batch of the
//!    sample (the ledger's `proven`), and appears once in the batch, else
//!    `Reject(StatementMismatch { field: Batch })`.
//!
//! RFC-0015's anchor checks follow ([`crate::anchor::check_batch`]). After `Accept`, the batch's
//! requests join the ledger's record of the sample with their `Û` (RFC-0019 step 13), the checks
//! repeated first, so that no request is recorded twice (INV-AUD-3).
//!
//! ## The ledger
//!
//! `VerifierLedger`, schema `0x0701` version 1, `.tvl`, TCE in field order: `weight_roots:
//! Vec<(Digest, Vec<(u32, Digest)>)>`, `successors: Vec<(Digest, Digest)>`, `used_blinding_rows:
//! Vec<(Digest, u32, u8, BitSet)>`, `proven: Vec<(Digest, BitSet, Vec<FixedBits>)>`. A `BitSet`
//! is `Bytes`, index `i` at bit `i mod 8` of byte `⌊i / 8⌋`, whose last byte is not zero, else
//! `ERR-CODEC-NONCANONICAL` at that byte, as the set is read. `derived: Vec<Digest>`, before
//! `proven`, names the samples the verifier derived, strictly increasing (else
//! `ERR-CODEC-NONCANONICAL` at the first digest out of order). In `proven`, one record per
//! derived sample with an accepted request, by strictly increasing sample identity (else
//! `ERR-CODEC-NONCANONICAL` at the record's first byte, as for a sample not in `derived`), each
//! with a non-empty set (else at its length) and one `Û` per index in increasing order (else at
//! the list's length). The weight fields are #158's and decode without further rules.
//!
//! The file ([`LedgerFile`], [`read_ledger`]) is a regular file, not a symbolic link and of one
//! hard link, else `ERR-USAGE`, checked without opening it, so a FIFO never blocks; its
//! directory is resolved once at open. A write goes to a synced temporary file renamed over the
//! old one, then syncs the directory, a failure of which is `ERR-RES-IO`, under a lock file that
//! keeps a second writer out.
//!
//! ## The report
//!
//! [`AuditReport::from_ledger`], at the deadline: the sample is in the ledger's `derived`, else
//! `Inconclusive(ERR-USAGE)`; its coin toss opens, else `Inconclusive(ERR-PROTO-COINTOSS)`; each selected request, in order, is `Accepted { uhat }`
//! when the ledger records it for the sample's identity, else `Unproven(DeadlineMissed)`; a
//! record naming an index past `selected` is `Inconclusive(ERR-USAGE)`. `sample` is the seed,
//! and the summary counts the outcomes, `max_uhat` the largest accepted `Û`.

mod batch;
mod file;
mod ledger;
mod policy;
mod report;
mod sample;
mod toss;

pub use batch::{Batch, BatchRequest, pack};
pub use file::{LedgerFile, read_ledger};
pub use ledger::{BitSet, BlindingRows, ProvenSample, VerifierLedger, WeightRoots};
pub use policy::{AuditPolicy, PolicyError, RateBucket};
pub use report::{AuditReport, AuditSummary, RequestOutcome, UnprovenReason};
pub use sample::{AuditSample, CoinTossTranscript, RequestKey, select, selection_value};
pub use toss::AuditToss;

use crate::anchor::Failure;
#[cfg(doc)]
use crate::anchor::Window;
use crate::verdict::{ErrorCode, VerifierError};
use tv_codec::Expected;
use tv_hash::{Digest, Domain, hash};

/// `payload` in RFC-0003's envelope: magic, envelope version 1, the schema and its version,
/// flags 0, the payload's length, the payload, then `H("envelope", everything before)`.
pub(crate) fn seal(schema: Expected, payload: &[u8]) -> Result<Vec<u8>, VerifierError> {
    let len = payload.len().checked_add(52).ok_or(VerifierError {
        code: ErrorCode::Memory,
    })?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| VerifierError {
        code: ErrorCode::Memory,
    })?;
    out.extend_from_slice(&tv_codec::MAGIC);
    out.extend_from_slice(&tv_codec::ENVELOPE_VERSION.to_le_bytes());
    out.extend_from_slice(&schema.schema_id.to_le_bytes());
    out.extend_from_slice(&schema.version.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(
        &u64::try_from(payload.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    out.extend_from_slice(payload);
    let digest = hash(Domain::Envelope, &out);
    out.extend_from_slice(&digest);
    Ok(out)
}

/// The digest a sealed artifact ends with, recomputed over its header and payload.
pub(crate) fn envelope_digest(sealed: &[u8]) -> Digest {
    let body = sealed.len().saturating_sub(32);
    hash(Domain::Envelope, sealed.get(..body).unwrap_or_default())
}

/// `Inconclusive(ERR-RES-MEMORY)`.
pub(crate) const fn memory() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Memory,
    })
}
