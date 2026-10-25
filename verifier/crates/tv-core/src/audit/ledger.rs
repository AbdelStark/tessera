// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's ledger (RFC-0019 'Ledger', RFC-0067, schema `0x0701`, `.tvl`): its encoding,
//! the samples the verifier derived, the audit's accounting checks (RFC-0016 'Verifier checks':
//! the parameter set, then checks 1 and 2) and the record of accepted requests. Its file is
//! [`super::file`]'s.

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, decode, size};
use tv_hash::{Digest, Domain, hash};

use super::policy::put_len;
use super::sample::{AuditSample, RequestKey};
use super::seal;
use crate::anchor::{Failure, usage};
use crate::claims::ParamSetId;
use crate::protocol::{FixedBits, PublicStatement};
use crate::validation::StatementField;
use crate::verdict::{ErrorCode as Code, RejectReason, VerifierError};

// ----- BitSet ----------------------------------------------------------------------------------

/// A set of indices (RFC-0019 'Ledger'). TCE has no set type (RFC-0040 notes it of RFC-0012's
/// sketch), so the ledger encodes one as `Bytes`: index `i` is bit `i mod 8`, least significant
/// first, of byte `⌊i / 8⌋`, and the last byte is not zero. Each set thus has one encoding, the
/// empty set the empty `Bytes`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BitSet {
    bytes: Vec<u8>,
}

impl BitSet {
    /// The empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `index` is in the set.
    pub fn contains(&self, index: u64) -> bool {
        let (byte, bit) = split(index);
        usize::try_from(byte)
            .ok()
            .and_then(|b| self.bytes.get(b))
            .is_some_and(|&b| b & bit != 0)
    }

    /// Adds `index`.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when the set cannot grow to hold it.
    pub fn insert(&mut self, index: u64) -> Result<(), VerifierError> {
        let (byte, bit) = split(index);
        let byte = usize::try_from(byte).map_err(|_| VerifierError { code: Code::Memory })?;
        if byte >= self.bytes.len() {
            let grow = byte
                .checked_add(1)
                .and_then(|n| n.checked_sub(self.bytes.len()))
                .ok_or(VerifierError { code: Code::Memory })?;
            self.bytes
                .try_reserve(grow)
                .map_err(|_| VerifierError { code: Code::Memory })?;
            self.bytes.resize(byte.saturating_add(1), 0);
        }
        if let Some(b) = self.bytes.get_mut(byte) {
            *b |= bit;
        }
        Ok(())
    }

    /// The number of indices in the set.
    pub fn len(&self) -> u64 {
        self.bytes
            .iter()
            .fold(0u64, |n, b| n.saturating_add(u64::from(b.count_ones())))
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The indices, increasing.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.bytes.iter().enumerate().flat_map(|(i, &b)| {
            let base = u64::try_from(i).unwrap_or(u64::MAX).saturating_mul(8);
            (0u32..8)
                .filter(move |&bit| b & mask(bit) != 0)
                .map(move |bit| base.saturating_add(u64::from(bit)))
        })
    }

    /// The encoding's bytes, after the length.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// `Bytes`, whose last byte, if any, is not zero: one that is is `ERR-CODEC-NONCANONICAL` at
    /// that byte.
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        let bytes = c.bytes()?;
        if bytes.last() == Some(&0) {
            return Err(DecodeError::new(
                ErrorCode::Noncanonical,
                c.offset().saturating_sub(1),
            ));
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(bytes.len())
            .map_err(|_| DecodeError::new(ErrorCode::Memory, c.offset()))?;
        owned.extend_from_slice(bytes);
        Ok(Self { bytes: owned })
    }

    fn encoded_len(&self) -> Option<u64> {
        size::SEQUENCE.checked_add(u64::try_from(self.bytes.len()).ok()?)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        put_len(out, self.bytes.len());
        out.extend_from_slice(&self.bytes);
    }
}

/// Index `i`'s byte, `⌊i / 8⌋`, and the mask of its bit, `i mod 8`.
fn split(index: u64) -> (u64, u8) {
    let byte = index.checked_div(8).unwrap_or(0);
    let bit = u32::try_from(index.checked_rem(8).unwrap_or(0)).unwrap_or(0);
    (byte, mask(bit))
}

/// The mask of bit `bit` of a byte, least significant first; 0 past the byte.
fn mask(bit: u32) -> u8 {
    1u8.checked_shl(bit).unwrap_or(0)
}

// ----- The ledger's records --------------------------------------------------------------------

/// A sample's accepted requests: the indices into its `selected` of the requests an accepted
/// proof covered, and each one's `Û`, in increasing index order (RFC-0019 'Ledger' `proven`,
/// with the `Û` that RFC-0016's audit report gives an accepted request).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvenSample {
    sample: Digest,
    accepted: BitSet,
    uhat: Vec<FixedBits>,
}

impl ProvenSample {
    /// The smallest encoding: the digest and two empty sequences.
    const MIN_LEN: u64 = size::sum(&[size::DIGEST, size::SEQUENCE, size::SEQUENCE]);

    /// The sample's identity, its envelope's digest.
    pub fn sample(&self) -> &Digest {
        &self.sample
    }

    /// The indices into the sample's `selected` of the accepted requests.
    pub fn accepted(&self) -> &BitSet {
        &self.accepted
    }

    /// Each accepted request's `Û`, in increasing index order.
    pub fn uhat(&self) -> &[FixedBits] {
        &self.uhat
    }

    /// The accepted requests with their `Û`, by increasing index.
    pub fn iter(&self) -> impl Iterator<Item = (u64, FixedBits)> + '_ {
        self.accepted.iter().zip(self.uhat.iter().copied())
    }

    fn decode(
        c: &mut Cursor<'_>,
        previous: Option<&Digest>,
        derived: &[Digest],
    ) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let at = c.offset();
            let sample = c.digest()?;
            // Both rules read the sample alone, and both refuse at the record's first byte.
            if previous.is_some_and(|p| &sample <= p) || derived.binary_search(&sample).is_err() {
                return Err(DecodeError::new(ErrorCode::Noncanonical, at));
            }
            let set_at = c.offset();
            let accepted = BitSet::decode(c)?;
            if accepted.is_empty() {
                // A record exists only once a request is accepted.
                return Err(DecodeError::new(ErrorCode::Noncanonical, set_at));
            }
            let uhat_at = c.offset();
            let uhat = c.vec(size::U64, |c| c.nested(|c| c.u64().map(FixedBits)))?;
            if u64::try_from(uhat.len()).ok() != Some(accepted.len()) {
                return Err(DecodeError::new(ErrorCode::Noncanonical, uhat_at));
            }
            Ok(Self {
                sample,
                accepted,
                uhat,
            })
        })
    }

    fn encoded_len(&self) -> Option<u64> {
        let uhat = u64::try_from(self.uhat.len())
            .ok()?
            .checked_mul(size::U64)?;
        size::sum(&[size::DIGEST, size::SEQUENCE])
            .checked_add(self.accepted.encoded_len()?)?
            .checked_add(uhat)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.sample);
        self.accepted.encode(out);
        put_len(out, self.uhat.len());
        for u in &self.uhat {
            out.extend_from_slice(&u.0.to_le_bytes());
        }
    }
}

/// A claim list's weight roots by epoch (RFC-0019 'Ledger' `weight_roots`). Their rules are
/// #158's; this version keeps them as decoded.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightRoots {
    /// The claim list's digest.
    pub claim_list: Digest,
    /// Its `(epoch, root)` pairs.
    pub epochs: Vec<(u32, Digest)>,
}

/// One matrix's used blinding rows in an epoch (RFC-0019 'Ledger' `used_blinding_rows`). Their
/// rules are #158's; this version keeps them as decoded.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlindingRows {
    /// The weight root.
    pub root: Digest,
    /// The epoch.
    pub epoch: u32,
    /// The matrix.
    pub matrix: u8,
    /// The rows used.
    pub rows: BitSet,
}

// ----- The ledger ------------------------------------------------------------------------------

/// The verifier's persistent state across sessions (RFC-0019 'Ledger', RFC-0067 'The encoding',
/// schema `0x0701`, version 1, `.tvl`, private to the verifier), TCE in field order:
///
/// ```text
/// weight_roots:       Vec<(Digest, Vec<(u32, Digest)>)>
/// successors:         Vec<(Digest, Digest)>
/// used_blinding_rows: Vec<(Digest, u32, u8, BitSet)>
/// derived:            Vec<(Digest, Digest)>           // (manifest, its one sample), by manifest
/// proven:             Vec<(Digest /*sample*/, BitSet /*selected index*/, Vec<FixedBits> /*Û*/)>
/// ```
///
/// `derived` names, for each manifest, the one sample this verifier derived itself for it, by
/// revealing its own toss ([`AuditToss::reveal`](super::AuditToss::reveal)): only those are its
/// samples, and a period has one (RFC-0067 'The verifier's own sample'). `proven` is the audit's accounting: one record per derived sample
/// with an accepted request, by increasing sample identity. The first three fields are the
/// weight epochs' (#158), kept as they decode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierLedger {
    weight_roots: Vec<WeightRoots>,
    successors: Vec<(Digest, Digest)>,
    used_blinding_rows: Vec<BlindingRows>,
    derived: Vec<(Digest, Digest)>,
    proven: Vec<ProvenSample>,
}

impl VerifierLedger {
    /// The envelope's schema: `0x0701`, version 1 (03 §5).
    pub const SCHEMA: Expected = Expected {
        schema_id: 0x0701,
        version: 1,
    };

    /// An empty ledger, a verifier's first.
    pub fn new() -> Self {
        Self::default()
    }

    /// The weight roots by claim list.
    pub fn weight_roots(&self) -> &[WeightRoots] {
        &self.weight_roots
    }

    /// Each root's successor.
    pub fn successors(&self) -> &[(Digest, Digest)] {
        &self.successors
    }

    /// The used blinding rows.
    pub fn used_blinding_rows(&self) -> &[BlindingRows] {
        &self.used_blinding_rows
    }

    /// The samples this verifier derived, `(manifest, sample identity)`, by increasing manifest.
    pub fn derived(&self) -> &[(Digest, Digest)] {
        &self.derived
    }

    /// The identity of the one sample this verifier derived for `manifest`, if any.
    pub fn derived_for(&self, manifest: &Digest) -> Option<&Digest> {
        self.derived
            .binary_search_by(|(m, _)| m.cmp(manifest))
            .ok()
            .and_then(|i| self.derived.get(i))
            .map(|(_, sample)| sample)
    }

    /// Whether `sample` is the one this verifier derived for its manifest.
    pub fn is_derived(&self, sample: &AuditSample) -> bool {
        self.derived_for(sample.manifest_digest()) == Some(sample.identity())
    }

    /// Adds `sample`, which the verifier derived, to `derived` under its manifest (RFC-0067 'One
    /// sample per manifest'): nothing when it is there already, and `Inconclusive(ERR-PROTO-ORDER)`
    /// when the manifest holds another sample, which a second share or a second toss over one
    /// manifest would give. Only the reveal of the verifier's own toss calls it
    /// ([`super::file::LedgerFile`]), so that no other sample becomes the verifier's.
    pub(crate) fn add_derived(&mut self, sample: &AuditSample) -> Result<(), Failure> {
        let (manifest, identity) = (*sample.manifest_digest(), *sample.identity());
        match self.derived.binary_search_by(|(m, _)| m.cmp(&manifest)) {
            Ok(at) => match self.derived.get(at) {
                Some((_, held)) if held == &identity => Ok(()),
                _ => Err(Failure::Inconclusive(VerifierError {
                    code: Code::ProtocolOrder,
                })),
            },
            Err(at) => {
                self.derived.try_reserve(1).map_err(|_| memory())?;
                self.derived.insert(at, (manifest, identity));
                Ok(())
            }
        }
    }

    /// The samples with an accepted request, by increasing identity.
    pub fn proven_samples(&self) -> &[ProvenSample] {
        &self.proven
    }

    /// The record of the sample whose identity is `sample`, `None` when no request of it has
    /// been accepted.
    pub fn proven(&self, sample: &Digest) -> Option<&ProvenSample> {
        self.proven
            .binary_search_by(|p| p.sample.cmp(sample))
            .ok()
            .and_then(|i| self.proven.get(i))
    }

    /// RFC-0016 'Verifier checks' for a proof of `sample` whose claim list names the parameter
    /// set `params` and whose statement is `statement`, against this ledger (RFC-0043 'The audit
    /// sample's checks', RFC-0067 'The accounting'). First, before any check of the proof, the
    /// sample must be the verifier's own, in `derived`, else `Inconclusive(ERR-USAGE)`: a sample
    /// whose coin toss merely opens could be the prover's (RFC-0067 'The verifier's own
    /// sample'). Then, in order, the first that fails giving the result:
    ///
    /// 0. `params` is the parameter set the sample's policy names ([`AuditPolicy::params`]),
    ///    compared byte for byte, else `Reject(Params { name })`, naming `params`, the claim
    ///    list's (RFC-0016 'The parameter set', RFC-0067 'The accounting' step 0, RFC-0043
    ///    'Parameter sets'). It is RFC-0019 'Check order' step 2 for a proof of an audit sample,
    ///    so it comes before the statement's fields: the policy fixes the soundness every proof
    ///    of the audit gives, and a proof under another set, such as `live-64/1` under a policy
    ///    of `ni-100/1`, must not count toward it. Whether this verifier allows the set at all is
    ///    the step's other half, [`ParamSet::accept`], which claim-list validation runs
    ///    ([`validate`]);
    /// 1. `statement.manifest_digest` is the sample's, else `StatementMismatch { field:
    ///    ManifestDigest }`;
    /// 2. then each request of `statement.batch` is in `selected`, is not in an earlier accepted
    ///    batch (this ledger's record of the sample), and is not named twice by the batch, else
    ///    `StatementMismatch { field: Batch }`, a replayed request included.
    ///
    /// Checks 1 and 2 run in RFC-0039's field order. Every failure of check 2 gives the same
    /// reason, so its checks need no order among themselves: each request is looked up by
    /// binary search, and repeats are found by sorting a copy of the indices, so a batch of `n`
    /// requests costs `O(n log n)` whatever the prover lists. Returns each request's index into
    /// `selected`, in batch order.
    ///
    /// [`AuditPolicy::params`]: super::AuditPolicy::params
    /// [`ParamSet::accept`]: crate::params::ParamSet::accept
    /// [`validate`]: crate::validation::validate
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-USAGE)` for a sample the verifier did not derive, `Failure::Rejected`
    /// with the reasons above, and `Inconclusive(ERR-RES-MEMORY)` when the indices, or the
    /// rejection's copy of the name, cannot be reserved.
    pub fn check_accounting(
        &self,
        sample: &AuditSample,
        params: &ParamSetId,
        statement: &PublicStatement,
    ) -> Result<Vec<u64>, Failure> {
        if !self.is_derived(sample) {
            return Err(usage());
        }
        // RFC-0016 'The parameter set' (RFC-0019 step 2): the policy's set and no other, before
        // any field of the statement.
        if params != sample.policy().params() {
            let name = params.try_clone().ok_or(memory())?;
            return Err(Failure::Rejected(RejectReason::Params { name }));
        }
        if &statement.manifest_digest != sample.manifest_digest() {
            return Err(rejected(StatementField::ManifestDigest));
        }
        let record = self.proven(sample.identity());
        let requests = &statement.batch.requests;
        let mut indices: Vec<u64> = Vec::new();
        indices
            .try_reserve_exact(requests.len())
            .map_err(|_| memory())?;
        for request in requests {
            let key = RequestKey {
                session_id: request.session_id,
                request_id: request.request_id,
            };
            let index = sample
                .index_of(&key)
                .and_then(|i| u64::try_from(i).ok())
                .ok_or(rejected(StatementField::Batch))?;
            if record.is_some_and(|r| r.accepted.contains(index)) {
                return Err(rejected(StatementField::Batch));
            }
            indices.push(index);
        }
        let mut sorted: Vec<u64> = Vec::new();
        sorted
            .try_reserve_exact(indices.len())
            .map_err(|_| memory())?;
        sorted.extend_from_slice(&indices);
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| matches!(w, [a, b] if a == b)) {
            return Err(rejected(StatementField::Batch));
        }
        Ok(indices)
    }

    /// Records an accepted proof (RFC-0019 step 13) of `sample`, whose claim list names the
    /// parameter set `params`: its requests join the sample's record with their `Û`, `uhat` in
    /// `statement.batch` order. The accounting checks ([`check_accounting`]) run again first, in
    /// their order: the sample must be the verifier's own, then `params` the policy's, so that no
    /// proof of another parameter set is recorded (RFC-0016 'The parameter set', RFC-0067 'The
    /// accounting'), then the statement's fields, so that a request is never recorded twice
    /// (INV-AUD-3). Nothing changes unless every step passes, and a batch of no request changes
    /// nothing.
    ///
    /// [`check_accounting`]: Self::check_accounting
    ///
    /// # Errors
    ///
    /// The accounting checks' failure, `Reject(Params { name })` and the statement's rejections
    /// first; `Inconclusive(ERR-USAGE)` for a `uhat` of another length than the batch;
    /// `Inconclusive(ERR-RES-MEMORY)` when memory cannot be reserved.
    pub fn record_accepted(
        &mut self,
        sample: &AuditSample,
        params: &ParamSetId,
        statement: &PublicStatement,
        uhat: &[FixedBits],
    ) -> Result<(), Failure> {
        let indices = self.check_accounting(sample, params, statement)?;
        if indices.len() != uhat.len() {
            return Err(usage());
        }
        if indices.is_empty() {
            // A batch of no request accepts nothing, and a record exists only once one is.
            return Ok(());
        }
        let mut added: Vec<(u64, FixedBits)> = Vec::new();
        added
            .try_reserve_exact(indices.len())
            .map_err(|_| memory())?;
        added.extend(indices.into_iter().zip(uhat.iter().copied()));
        added.sort_unstable_by_key(|&(i, _)| i);
        let identity = *sample.identity();
        let position = self.proven.binary_search_by(|p| p.sample.cmp(&identity));
        let before = match position {
            Ok(i) => self.proven.get(i),
            Err(_) => None,
        };
        // Merge the record's pairs and the new ones by index; the checks above keep them apart.
        let old: Vec<(u64, FixedBits)> = before.map(|r| r.iter().collect()).unwrap_or_default();
        let mut accepted = BitSet::new();
        let mut merged: Vec<FixedBits> = Vec::new();
        merged
            .try_reserve_exact(old.len().saturating_add(added.len()))
            .map_err(|_| memory())?;
        let (mut a, mut b) = (old.iter().peekable(), added.iter().peekable());
        loop {
            let next = match (a.peek(), b.peek()) {
                (Some(x), Some(y)) if x.0 < y.0 => a.next(),
                (Some(_), Some(_)) => b.next(),
                (Some(_), None) => a.next(),
                (None, Some(_)) => b.next(),
                (None, None) => break,
            };
            if let Some(&(index, u)) = next {
                accepted.insert(index).map_err(Failure::Inconclusive)?;
                merged.push(u);
            }
        }
        let record = ProvenSample {
            sample: identity,
            accepted,
            uhat: merged,
        };
        match position {
            Ok(i) => {
                if let Some(slot) = self.proven.get_mut(i) {
                    *slot = record;
                }
            }
            Err(i) => {
                self.proven.try_reserve(1).map_err(|_| memory())?;
                self.proven.insert(i, record);
            }
        }
        Ok(())
    }

    /// Opens a `.tvl` artifact: RFC-0003's envelope, schema `0x0701` version 1, then the
    /// payload ([`decode_payload`](Self::decode_payload)'s rules, offsets from the envelope's
    /// start).
    ///
    /// # Errors
    ///
    /// The first failure, `ERR-CODEC-*` at its offset, or `ERR-RES-MEMORY`.
    pub fn open(tvl: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        Envelope::open(tvl, Self::SCHEMA, limits, |b| hash(Domain::Envelope, b))?
            .decode_payload(|c, _| Self::fields(c))
    }

    /// Decodes a payload alone, its fields in order (RFC-0067 'The encoding'). Beyond RFC-0003's
    /// checks, each `ERR-CODEC-NONCANONICAL`:
    ///
    /// - a `BitSet` whose last byte is zero, at that byte, as the set is read;
    /// - in `derived`, an entry whose manifest is not above the one before it, at that entry;
    /// - in `proven`, a record whose sample is not above the one before it, or is not in
    ///   `derived`, at the record's first byte; an empty set, at its length; and a `Û` list of
    ///   another length than the set, at the list's length.
    ///
    /// # Errors
    ///
    /// The first failure, `ERR-CODEC-*` at its offset, or `ERR-RES-MEMORY`.
    pub fn decode_payload(payload: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        decode(payload, limits, Self::fields)
    }

    fn fields(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let weight_roots = c.vec(40, |c| {
                c.nested(|c| {
                    Ok(WeightRoots {
                        claim_list: c.digest()?,
                        epochs: c.vec(36, |c| c.nested(|c| Ok((c.u32()?, c.digest()?))))?,
                    })
                })
            })?;
            let successors = c.vec(64, |c| c.nested(|c| Ok((c.digest()?, c.digest()?))))?;
            let used_blinding_rows = c.vec(45, |c| {
                c.nested(|c| {
                    Ok(BlindingRows {
                        root: c.digest()?,
                        epoch: c.u32()?,
                        matrix: c.u8()?,
                        rows: BitSet::decode(c)?,
                    })
                })
            })?;
            let mut last: Option<Digest> = None;
            let derived = c.vec(DERIVED_LEN, |c| {
                c.nested(|c| {
                    let at = c.offset();
                    let manifest = c.digest()?;
                    if last.is_some_and(|l| manifest <= l) {
                        return Err(DecodeError::new(ErrorCode::Noncanonical, at));
                    }
                    last = Some(manifest);
                    Ok((manifest, c.digest()?))
                })
            })?;
            // The derived samples, sorted, for the records' rule.
            let at = c.offset();
            let mut samples: Vec<Digest> = Vec::new();
            samples
                .try_reserve_exact(derived.len())
                .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
            samples.extend(derived.iter().map(|(_, sample)| *sample));
            samples.sort_unstable();
            let mut previous: Option<Digest> = None;
            let proven = c.vec(ProvenSample::MIN_LEN, |c| {
                let record = ProvenSample::decode(c, previous.as_ref(), &samples)?;
                previous = Some(record.sample);
                Ok(record)
            })?;
            Ok(Self {
                weight_roots,
                successors,
                used_blinding_rows,
                derived,
                proven,
            })
        })
    }

    /// The payload's TCE encoding.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn encode_payload(&self) -> Result<Vec<u8>, VerifierError> {
        let len = self
            .encoded_len()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(VerifierError { code: Code::Memory })?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| VerifierError { code: Code::Memory })?;
        put_len(&mut out, self.weight_roots.len());
        for w in &self.weight_roots {
            out.extend_from_slice(&w.claim_list);
            put_len(&mut out, w.epochs.len());
            for (epoch, root) in &w.epochs {
                out.extend_from_slice(&epoch.to_le_bytes());
                out.extend_from_slice(root);
            }
        }
        put_len(&mut out, self.successors.len());
        for (root, next) in &self.successors {
            out.extend_from_slice(root);
            out.extend_from_slice(next);
        }
        put_len(&mut out, self.used_blinding_rows.len());
        for r in &self.used_blinding_rows {
            out.extend_from_slice(&r.root);
            out.extend_from_slice(&r.epoch.to_le_bytes());
            out.push(r.matrix);
            r.rows.encode(&mut out);
        }
        put_len(&mut out, self.derived.len());
        for (manifest, sample) in &self.derived {
            out.extend_from_slice(manifest);
            out.extend_from_slice(sample);
        }
        put_len(&mut out, self.proven.len());
        for p in &self.proven {
            p.encode(&mut out);
        }
        Ok(out)
    }

    fn encoded_len(&self) -> Option<u64> {
        let mut n = size::SEQUENCE.checked_mul(5)?;
        for w in &self.weight_roots {
            let epochs = u64::try_from(w.epochs.len()).ok()?.checked_mul(36)?;
            n = n.checked_add(40)?.checked_add(epochs)?;
        }
        n = n.checked_add(u64::try_from(self.successors.len()).ok()?.checked_mul(64)?)?;
        for r in &self.used_blinding_rows {
            n = n.checked_add(37)?.checked_add(r.rows.encoded_len()?)?;
        }
        n = n.checked_add(
            u64::try_from(self.derived.len())
                .ok()?
                .checked_mul(DERIVED_LEN)?,
        )?;
        for p in &self.proven {
            n = n.checked_add(p.encoded_len()?)?;
        }
        Some(n)
    }

    /// The `.tvl` artifact: the payload in RFC-0003's envelope, schema `0x0701` version 1.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn seal(&self) -> Result<Vec<u8>, VerifierError> {
        seal(Self::SCHEMA, &self.encode_payload()?)
    }
}

const fn rejected(field: StatementField) -> Failure {
    Failure::Rejected(RejectReason::StatementMismatch { field })
}

/// A `derived` entry's size: two digests.
const DERIVED_LEN: u64 = 64;

const fn memory() -> Failure {
    Failure::Inconclusive(VerifierError { code: Code::Memory })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::protocol::{BatchDescriptor, Decoding, PredictorSpec, RequestRef};
    use tv_codec::DecodeLimits;

    /// A ledger whose `derived` holds `samples`, and nothing else.
    pub(crate) fn ledger_of(samples: &[&AuditSample]) -> VerifierLedger {
        let mut ledger = VerifierLedger::new();
        for s in samples {
            ledger.add_derived(s).unwrap();
        }
        ledger
    }

    /// A sample over the manifest `[manifest; 32]` selecting `keys`, decoded from a payload
    /// built here, its policy selecting everything and its toss opening; `seed` varies the toss.
    pub(crate) fn sample_over(manifest: u8, keys: &[RequestKey], seed: u8) -> AuditSample {
        let mut payload = vec![manifest; 32];
        payload.extend(86_400u64.to_le_bytes());
        payload.extend(1u64.to_le_bytes());
        payload.extend(u32::MAX.to_le_bytes());
        payload.extend(u64::MAX.to_le_bytes());
        payload.extend(16_384u32.to_le_bytes());
        payload.extend(262_144u32.to_le_bytes());
        payload.extend(604_800u64.to_le_bytes());
        payload.extend(9u64.to_le_bytes());
        payload.extend(b"live-64/1");
        let share = [seed; 32];
        payload.extend(hash(Domain::CoinCommit, &share));
        payload.extend([4; 32]);
        payload.extend(share);
        payload.extend(u64::try_from(keys.len()).unwrap().to_le_bytes());
        for k in keys {
            payload.extend(k.session_id);
            payload.extend(k.request_id.to_le_bytes());
        }
        AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap()
    }

    /// A sample of session 1's requests 1 to 3 over the manifest `[seed; 32]`: each seed is
    /// another period, with its own sample.
    pub(crate) fn derived_sample(seed: u8) -> AuditSample {
        let keys: Vec<RequestKey> = (1..=3)
            .map(|r| RequestKey {
                session_id: [1; 16],
                request_id: r,
            })
            .collect();
        sample_over(seed, &keys, seed)
    }

    /// The statement of a batch of the sample's selected requests at `indices`.
    pub(crate) fn statement_of(sample: &AuditSample, indices: &[usize]) -> PublicStatement {
        PublicStatement {
            claim_list_digest: [0; 32],
            weights_root: [0; 32],
            weights_epoch: 0,
            manifest_digest: *sample.manifest_digest(),
            batch: BatchDescriptor {
                requests: indices
                    .iter()
                    .map(|&i| {
                        let k = sample.selected()[i];
                        RequestRef {
                            request_id: k.request_id,
                            session_id: k.session_id,
                            prompt_len: 1,
                            output_len: 1,
                        }
                    })
                    .collect(),
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        }
    }

    #[test]
    fn derived_holds_one_sample_per_manifest() {
        let (a, b, c) = (derived_sample(5), derived_sample(1), derived_sample(9));
        let mut ledger = ledger_of(&[&a, &b, &c]);
        let manifests: Vec<Digest> = ledger.derived().iter().map(|(m, _)| *m).collect();
        assert_eq!(manifests, [[1; 32], [5; 32], [9; 32]]);
        assert!(ledger.is_derived(&a) && ledger.is_derived(&c));
        // The same sample again changes nothing.
        let before = ledger.clone();
        ledger.add_derived(&a).unwrap();
        assert_eq!(ledger, before);
        // Another sample of a manifest the ledger holds: refused, the ledger as it was.
        let other = sample_over(5, &[], 77);
        assert_ne!(other.identity(), a.identity());
        assert_eq!(
            ledger.add_derived(&other),
            Err(Failure::Inconclusive(VerifierError {
                code: Code::ProtocolOrder
            }))
        );
        assert_eq!(ledger, before);
        assert!(!ledger.is_derived(&other));
        // A sample of a manifest the ledger does not hold is not derived; nor is one whose identity
        // a hand-made ledger lists under another manifest.
        assert!(!ledger.is_derived(&sample_over(8, &[], 5)));
        let mut payload = vec![0; 24];
        payload.extend(1u64.to_le_bytes());
        payload.extend([3; 32]);
        payload.extend(a.identity());
        payload.extend(0u64.to_le_bytes());
        let misplaced = VerifierLedger::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap();
        assert!(!misplaced.is_derived(&a));
        let bytes = ledger.encode_payload().unwrap();
        assert_eq!(
            VerifierLedger::decode_payload(&bytes, &DecodeLimits::MESSAGE).unwrap(),
            ledger
        );
    }

    #[test]
    fn a_sample_the_verifier_did_not_derive_is_refused_before_the_statement() {
        let sample = derived_sample(1);
        let mut ledger = ledger_of(&[&derived_sample(2)]);
        let live = sample.policy().params();
        let other = ParamSetId::new("ni-100/1").unwrap();
        // The statement may be broken too, in its manifest, and the parameter set another than
        // the policy's: the sample's check comes first.
        let mut wrong = statement_of(&sample, &[0]);
        wrong.manifest_digest = [0; 32];
        for s in [statement_of(&sample, &[0]), wrong] {
            for params in [live, &other] {
                assert_eq!(ledger.check_accounting(&sample, params, &s), Err(usage()));
                assert_eq!(
                    ledger.record_accepted(&sample, params, &s, &[FixedBits(0)]),
                    Err(usage())
                );
            }
        }
        ledger.add_derived(&sample).unwrap();
        assert_eq!(
            ledger.check_accounting(&sample, live, &statement_of(&sample, &[2, 0])),
            Ok(vec![2, 0])
        );
    }
}
