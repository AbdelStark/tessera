// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's ledger (RFC-0019 'Ledger', schema `0x0701`, `.tvl`): its encoding, the audit's
//! accounting checks (RFC-0016 'Verifier checks' 1 and 2), the record of accepted requests, and
//! the file, written atomically.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, decode, size};
use tv_hash::{Digest, Domain, hash};

use super::policy::put_len;
use super::sample::{AuditSample, RequestKey};
use super::seal;
use crate::anchor::{Failure, usage};
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

    fn decode(c: &mut Cursor<'_>, previous: Option<&Digest>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let at = c.offset();
            let sample = c.digest()?;
            if previous.is_some_and(|p| &sample <= p) {
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

/// The verifier's persistent state across sessions (RFC-0019 'Ledger', schema `0x0701`, version
/// 1, `.tvl`, private to the verifier), TCE in field order:
///
/// ```text
/// weight_roots:       Vec<(Digest, Vec<(u32, Digest)>)>
/// successors:         Vec<(Digest, Digest)>
/// used_blinding_rows: Vec<(Digest, u32, u8, BitSet)>
/// proven:             Vec<(Digest /*sample*/, BitSet /*selected index*/, Vec<FixedBits> /*Û*/)>
/// ```
///
/// `proven` is the audit's accounting: one record per sample with an accepted request, by
/// increasing sample identity. The first three fields are the weight epochs' (#158), kept as they
/// decode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierLedger {
    weight_roots: Vec<WeightRoots>,
    successors: Vec<(Digest, Digest)>,
    used_blinding_rows: Vec<BlindingRows>,
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

    /// RFC-0016 'Verifier checks' 1 and 2, for a proof's statement against the sample and this
    /// ledger, in RFC-0039's field order (RFC-0043 'The audit sample's checks'):
    ///
    /// 1. `statement.manifest_digest` is the sample's, else `StatementMismatch { field:
    ///    ManifestDigest }`;
    /// 2. then each request of `statement.batch` is in `selected`, is not in an earlier accepted
    ///    batch (this ledger's record of the sample), and is not named twice by the batch, else
    ///    `StatementMismatch { field: Batch }`, a replayed request included.
    ///
    /// Every failure of check 2 gives the same reason, so the checks need no order among
    /// themselves: each request is looked up by binary search, and repeats are found by sorting
    /// a copy of the indices, so a batch of `n` requests costs `O(n log n)` whatever the prover
    /// lists. Returns each request's index into `selected`, in batch order.
    ///
    /// # Errors
    ///
    /// `Failure::Rejected` with the reason above, and `Inconclusive(ERR-RES-MEMORY)` when the
    /// indices cannot be reserved.
    pub fn check_accounting(
        &self,
        sample: &AuditSample,
        statement: &PublicStatement,
    ) -> Result<Vec<u64>, Failure> {
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

    /// Records an accepted proof (RFC-0019 step 13): its requests join the sample's record with
    /// their `Û`, `uhat` in `statement.batch` order. The accounting checks run again first, so a
    /// request is never recorded twice (INV-AUD-3). Nothing changes unless every step passes,
    /// and a batch of no request changes nothing.
    ///
    /// # Errors
    ///
    /// The accounting checks' failure, a rejection first; `Inconclusive(ERR-USAGE)` for a
    /// `uhat` of another length than the batch; `Inconclusive(ERR-RES-MEMORY)` when memory
    /// cannot be reserved.
    pub fn record_accepted(
        &mut self,
        sample: &AuditSample,
        statement: &PublicStatement,
        uhat: &[FixedBits],
    ) -> Result<(), Failure> {
        let indices = self.check_accounting(sample, statement)?;
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

    /// Decodes a payload alone, its fields in order. Beyond RFC-0003's checks, a `BitSet` whose
    /// last byte is zero is `ERR-CODEC-NONCANONICAL` at that byte; and in `proven`, a sample not
    /// above the one before it is `ERR-CODEC-NONCANONICAL` at its first byte, an empty set at its
    /// length, and a `Û` list of another length than the set at the list's length.
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
            let mut previous: Option<Digest> = None;
            let proven = c.vec(ProvenSample::MIN_LEN, |c| {
                let record = ProvenSample::decode(c, previous.as_ref())?;
                previous = Some(record.sample);
                Ok(record)
            })?;
            Ok(Self {
                weight_roots,
                successors,
                used_blinding_rows,
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
        put_len(&mut out, self.proven.len());
        for p in &self.proven {
            p.encode(&mut out);
        }
        Ok(out)
    }

    fn encoded_len(&self) -> Option<u64> {
        let mut n = size::SEQUENCE.checked_mul(4)?;
        for w in &self.weight_roots {
            let epochs = u64::try_from(w.epochs.len()).ok()?.checked_mul(36)?;
            n = n.checked_add(40)?.checked_add(epochs)?;
        }
        n = n.checked_add(u64::try_from(self.successors.len()).ok()?.checked_mul(64)?)?;
        for r in &self.used_blinding_rows {
            n = n.checked_add(37)?.checked_add(r.rows.encoded_len()?)?;
        }
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

const fn memory() -> Failure {
    Failure::Inconclusive(VerifierError { code: Code::Memory })
}

const fn io() -> VerifierError {
    VerifierError { code: Code::Io }
}

// ----- The file --------------------------------------------------------------------------------

/// Reads the ledger at `path`, a snapshot for reading only: the file as its last atomic write
/// left it, decoded under `limits`. It takes no lock, since a write replaces the file whole.
///
/// # Errors
///
/// `ERR-RES-IO` for a file that cannot be read or is not a regular file, `ERR-CODEC-LIMIT` for a
/// file larger than `limits.max_total_bytes`, and the decoder's errors.
pub fn read_ledger(path: &Path, limits: &DecodeLimits) -> Result<VerifierLedger, VerifierError> {
    let bytes = read_bounded(path, limits.max_total_bytes)?;
    VerifierLedger::open(&bytes, limits).map_err(|e| VerifierError {
        code: e.code().into(),
    })
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, VerifierError> {
    let mut file = File::open(path).map_err(|_| io())?;
    let meta = file.metadata().map_err(|_| io())?;
    if !meta.is_file() {
        return Err(io());
    }
    if meta.len() > limit {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(meta.len()).map_err(|_| io())?)
        .map_err(|_| VerifierError { code: Code::Memory })?;
    // One byte past the limit shows a file that grew after its size was read.
    Read::by_ref(&mut file)
        .take(limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|_| io())?;
    if u64::try_from(out.len()).map_or(true, |n| n > limit) {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    Ok(out)
}

/// The verifier's ledger file, held for writing: a lock file beside it, `<name>.lock`, created
/// exclusively, keeps a second writer out until this value is dropped, so that two sessions
/// cannot both accept one request from the same file (INV-AUD-3).
///
/// Every update is written atomically ([`record_accepted`](Self::record_accepted)): the new
/// ledger goes to a temporary file beside it, which is synced and then renamed over the old one,
/// so the file holds the old ledger or the new one, never part of either, whenever the writer
/// stops.
///
/// A lock file left by a writer that stopped without dropping its value, a crash, refuses later
/// writers with `ERR-RES-IO` until the operator, having checked that no verifier runs, removes it.
#[derive(Debug)]
pub struct LedgerFile {
    path: PathBuf,
    lock: PathBuf,
    ledger: VerifierLedger,
}

impl LedgerFile {
    /// Creates a ledger file at `path`, holding an empty ledger, and holds it.
    ///
    /// # Errors
    ///
    /// `ERR-USAGE` when a file exists at `path`, whose ledger would be lost; `ERR-RES-IO` when
    /// the lock is held or a file cannot be written; `ERR-RES-MEMORY`.
    pub fn create(path: &Path) -> Result<Self, VerifierError> {
        let lock = Lock::take(path)?;
        let ledger = VerifierLedger::new();
        let bytes = ledger.seal()?;
        write_new(path, &bytes)?;
        Ok(Self {
            path: path.to_path_buf(),
            lock: lock.release(),
            ledger,
        })
    }

    /// Opens the ledger file at `path` and holds it.
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when the lock is held or the file cannot be read, and the errors of
    /// [`read_ledger`].
    pub fn open(path: &Path, limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let lock = Lock::take(path)?;
        let ledger = read_ledger(path, limits)?;
        Ok(Self {
            path: path.to_path_buf(),
            lock: lock.release(),
            ledger,
        })
    }

    /// The ledger as last written.
    pub fn ledger(&self) -> &VerifierLedger {
        &self.ledger
    }

    /// Records an accepted proof ([`VerifierLedger::record_accepted`]) and writes the ledger
    /// atomically. The ledger in memory changes only once the file holds the new one, so a
    /// failed write leaves both as they were, and the caller's verdict is not `Accept`.
    ///
    /// # Errors
    ///
    /// Those of [`VerifierLedger::record_accepted`], and `Inconclusive(ERR-RES-IO)` when the
    /// file cannot be written.
    pub fn record_accepted(
        &mut self,
        sample: &AuditSample,
        statement: &PublicStatement,
        uhat: &[FixedBits],
    ) -> Result<(), Failure> {
        let mut next = self.ledger.clone();
        next.record_accepted(sample, statement, uhat)?;
        let bytes = next.seal().map_err(Failure::Inconclusive)?;
        write_atomic(&self.path, &bytes, &|_| false).map_err(Failure::Inconclusive)?;
        self.ledger = next;
        Ok(())
    }
}

impl Drop for LedgerFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock);
    }
}

/// The lock file of a ledger at `path` while it is being taken: removed if dropped armed.
struct Lock {
    path: PathBuf,
    armed: bool,
}

impl Lock {
    fn take(ledger: &Path) -> Result<Self, VerifierError> {
        let mut name = ledger.file_name().ok_or(io())?.to_os_string();
        name.push(".lock");
        let path = ledger.with_file_name(name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| io())?;
        let lock = Self { path, armed: true };
        // The holder's process, for the operator; the lock is the file's existence.
        let _ = writeln!(file, "{}", std::process::id());
        Ok(lock)
    }

    fn release(mut self) -> PathBuf {
        self.armed = false;
        core::mem::take(&mut self.path)
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// The steps of an atomic write, at which a test stops it as a crash would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// The temporary file exists, empty.
    Created,
    /// Half the bytes are written.
    Partial,
    /// Every byte is written, not yet synced.
    Written,
    /// The temporary file is synced, not yet renamed.
    Synced,
}

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A temporary file beside `path`: removed when dropped armed.
struct Temporary {
    path: PathBuf,
    armed: bool,
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Creates a fresh temporary file in `path`'s directory, readable by its owner only.
fn temporary(path: &Path) -> Result<(File, Temporary), VerifierError> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path.file_name().ok_or(io())?.to_string_lossy();
    for _ in 0..64 {
        let temp = dir.join(format!(
            ".{name}.{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temp) {
            Ok(file) => {
                return Ok((
                    file,
                    Temporary {
                        path: temp,
                        armed: true,
                    },
                ));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(io()),
        }
    }
    Err(io())
}

/// Writes `bytes` over `path` atomically: a temporary file in the same directory, written,
/// synced, then renamed over `path`, and the directory synced, as far as the host allows.
/// `stop` returns true at the step where a test stops the write as a crash would, leaving the
/// temporary file where it is.
fn write_atomic(
    path: &Path,
    bytes: &[u8],
    stop: &dyn Fn(Step) -> bool,
) -> Result<(), VerifierError> {
    let (mut file, mut temp) = temporary(path)?;
    let crash = |temp: &mut Temporary| {
        temp.armed = false;
        Err(io())
    };
    if stop(Step::Created) {
        return crash(&mut temp);
    }
    let half = bytes.len().checked_div(2).unwrap_or(0);
    let (first, second) = bytes.split_at_checked(half).ok_or(io())?;
    file.write_all(first).map_err(|_| io())?;
    if stop(Step::Partial) {
        return crash(&mut temp);
    }
    file.write_all(second).map_err(|_| io())?;
    if stop(Step::Written) {
        return crash(&mut temp);
    }
    file.sync_all().map_err(|_| io())?;
    drop(file);
    if stop(Step::Synced) {
        return crash(&mut temp);
    }
    fs::rename(&temp.path, path).map_err(|_| io())?;
    temp.armed = false;
    sync_directory(path);
    Ok(())
}

/// Writes `bytes` to `path`, which must not exist: a synced temporary file, hard-linked to
/// `path`, which fails if anything is there, then removed.
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), VerifierError> {
    let (mut file, temp) = temporary(path)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| io())?;
    drop(file);
    fs::hard_link(&temp.path, path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            VerifierError { code: Code::Usage }
        } else {
            io()
        }
    })?;
    drop(temp);
    sync_directory(path);
    Ok(())
}

/// Syncs `path`'s directory, so that a rename survives a power loss, where the host supports it;
/// a failure leaves the rename done, which is what atomicity needs.
fn sync_directory(path: &Path) {
    #[cfg(unix)]
    {
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tessera-ledger-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        dir
    }

    fn leftovers(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".tmp"))
            .collect()
    }

    fn ledger_with(sample: u8) -> VerifierLedger {
        let mut ledger = VerifierLedger::new();
        let mut accepted = BitSet::new();
        accepted.insert(3).unwrap();
        ledger.proven.push(ProvenSample {
            sample: [sample; 32],
            accepted,
            uhat: vec![FixedBits(7)],
        });
        ledger
    }

    #[test]
    fn an_interrupted_write_leaves_the_old_ledger_at_every_step() {
        for step in [Step::Created, Step::Partial, Step::Written, Step::Synced] {
            let dir = scratch("interrupt");
            let path = dir.join("ledger.tvl");
            let old = ledger_with(1);
            let old_bytes = old.seal().unwrap();
            write_atomic(&path, &old_bytes, &|_| false).unwrap();
            let new_bytes = ledger_with(2).seal().unwrap();
            assert_eq!(
                write_atomic(&path, &new_bytes, &|s| s == step),
                Err(io()),
                "{step:?}"
            );
            // The file is the old ledger, byte for byte, and still reads as it.
            assert_eq!(fs::read(&path).unwrap(), old_bytes, "{step:?}");
            assert_eq!(
                read_ledger(&path, &DecodeLimits::MESSAGE).unwrap(),
                old,
                "{step:?}"
            );
            // The crash left its temporary file, which no reader takes for the ledger.
            assert_eq!(leftovers(&dir).len(), 1, "{step:?}");
            // A later write succeeds beside it.
            write_atomic(&path, &new_bytes, &|_| false).unwrap();
            assert_eq!(fs::read(&path).unwrap(), new_bytes);
            for leftover in leftovers(&dir) {
                fs::remove_file(leftover).unwrap();
            }
            fs::remove_file(&path).unwrap();
            fs::remove_dir(&dir).unwrap();
        }
    }

    #[test]
    fn a_failed_write_removes_its_temporary_file() {
        let dir = scratch("failed");
        // A directory where the ledger should be: the rename fails, and nothing is left.
        let path = dir.join("ledger.tvl");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("keep"), b"x").unwrap();
        let bytes = ledger_with(1).seal().unwrap();
        assert_eq!(write_atomic(&path, &bytes, &|_| false), Err(io()));
        assert!(leftovers(&dir).is_empty());
        fs::remove_file(path.join("keep")).unwrap();
        fs::remove_dir(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }
}
