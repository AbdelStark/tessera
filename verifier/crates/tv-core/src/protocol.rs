// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent protocol/1 messages (RFC-0036), decoded under RFC-0003 limits.
//!
//! Decoding establishes syntax only. The driver must check protocol version, session,
//! supported predictor, schedule-derived lengths and every cryptographic obligation in
//! RFC-0019 order. In particular, stage/split/group counts are not codec predicates.
//!
//! Each composite takes its own depth, including `StageId` and `FixedBits`. Matrix ids
//! are checked in 0..10 and strictly increasing before reading an entry's remaining
//! fields. Existing independent protocol vectors pin every proper-prefix offset.
//! Encoding writes decoded values, never a retained copy of the incoming bytes.

use core::fmt;
use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, Rational, size};
use tv_field::{Fp, Fp2};
use tv_hash::{Digest, Domain, DomainHasher};

use crate::opening::CombinedRow;
use crate::stages::StageProof;
use crate::sumcheck::SumcheckProof;

/// The verifier's commitment to its fresh seed share, schema `0x0501`.
#[derive(Clone, PartialEq, Eq)]
pub struct SeedCommit {
    /// Session selected by the verifier.
    pub session_id: [u8; 16],
    /// `H("coin-commit", s_V)`.
    pub commitment: Digest,
}
/// The prover's seed share accompanying Round1, schema `0x0502`.
#[derive(Clone, PartialEq, Eq)]
pub struct ProverSeedShare {
    /// The same session as Round1.
    pub session_id: [u8; 16],
    /// The fresh 32-byte prover share.
    pub share: [u8; 32],
}
/// The verifier's opening of its seed commitment, schema `0x0503`.
#[derive(Clone, PartialEq, Eq)]
pub struct SeedReveal {
    /// The same session as SeedCommit.
    pub session_id: [u8; 16],
    /// The verifier share whose commitment the prover checks.
    pub share: [u8; 32],
}
/// Decoding selected by the statement (RFC-0017).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoding {
    /// Lowest-index maximum.
    Greedy,
    /// Inverse-CDF decoding at public rational temperature `t`.
    Temperature {
        /// Canonical rational; support/range is checked after decoding.
        t: Rational,
    },
}
/// The committed predictor; syntax does not imply that this driver supports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictorSpec {
    /// Token-for-token equality, with zero unexplained information.
    Exact {
        /// Greedy or temperature decoding.
        decoding: Decoding,
    },
    /// The bounded-mode mixture (RFC-0017).
    Mixture {
        /// Greedy or temperature decoding.
        decoding: Decoding,
        /// `epsilon = 2^-eps_log2`; its range is a later check.
        eps_log2: u8,
    },
}
/// A request's public identity and sequence lengths (03 section 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestRef {
    /// Request id in batch order.
    pub request_id: u64,
    /// Recorder session id.
    pub session_id: [u8; 16],
    /// Number of prompt positions.
    pub prompt_len: u32,
    /// Number of recorded output positions.
    pub output_len: u32,
}
/// Requests in the public statement's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchDescriptor {
    /// The batch's requests.
    pub requests: Vec<RequestRef>,
}
/// The public statement, whose TCE digest binds the transcript (03 section 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicStatement {
    /// Digest of the verifier's own claim list.
    pub claim_list_digest: Digest,
    /// Trusted root of the public weight artifact.
    pub weights_root: Digest,
    /// Its epoch.
    pub weights_epoch: u32,
    /// Digest of the recorder manifest.
    pub manifest_digest: Digest,
    /// Requests and lengths.
    pub batch: BatchDescriptor,
    /// The predictor being certified.
    pub predictor: PredictorSpec,
}
/// Round1's advice root and descriptor digest (RFC-0031).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceCommitment {
    /// Merkle root of the segment-merged column digests.
    pub root: Digest,
    /// Independently derived commitment id.
    pub layout_digest: Digest,
}
/// Commitments fixed before the seed, schema `0x0510`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round1 {
    /// Raw protocol number; checked by the session, not by this decoder.
    pub protocol_version: u16,
    /// Proof session id.
    pub session_id: [u8; 16],
    /// Claimed public statement, to compare with trusted inputs.
    pub statement: PublicStatement,
    /// Advice commitment; weights are supplied separately (RFC-0040).
    pub advice: AdviceCommitment,
    /// The leaf stage's grouping, absorbed with the rest before any challenge (RFC-0054).
    pub leaf_groups: LeafGroups,
}
/// The most group bits RFC-0054 lets a leaf stage take: `g_T + g_L ≤ 14`.
pub const MAX_LEAF_GROUP_BITS: u8 = 14;
/// A leaf stage's grouping (RFC-0054 'Messages'): `2^token_bits` token subsets by
/// `2^layer_bits` layer subsets; zero for a leaf stage in one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafGroups {
    /// `g_T`, the token index bits a group fixes.
    pub token_bits: u8,
    /// `g_L`, the layer index bits a group fixes.
    pub layer_bits: u8,
}
impl LeafGroups {
    /// No grouping: the leaf stage in one batch, RFC-0010's.
    pub const NONE: Self = Self {
        token_bits: 0,
        layer_bits: 0,
    };
}
/// LogUp roots and final value evaluations, in schedule order.
#[derive(Clone, PartialEq, Eq)]
pub struct LogUpProof {
    /// Four values per instance: `n_a, d_a, n_t, d_t`.
    pub roots: Vec<Fp2>,
    /// Value-column evaluations of the instances that have values.
    pub values: Vec<Fp2>,
}
/// Leaf levels, the groups and their merge (RFC-0054), then stage0's split values.
#[derive(Clone, PartialEq, Eq)]
pub struct TreeProof {
    /// Fraction/product tree levels in root-to-leaf order; a grouped stage's table batch.
    pub levels: Vec<SumcheckProof>,
    /// One per group in group order; none in an ungrouped stage.
    pub groups: Vec<GroupProof>,
    /// The table trees' roots of a grouped stage; none in an ungrouped one.
    pub table_roots: Vec<Fp2>,
    /// The merge of a grouped stage; no rounds and no evaluations in an ungrouped one.
    pub merge: SumcheckProof,
    /// The leaf stage's split values.
    pub splits: Vec<Fp2>,
}
/// One token group's roots, levels, values and splits (RFC-0054 'The leaf stage with groups').
#[derive(Clone, PartialEq, Eq)]
pub struct GroupProof {
    /// The group's lookup-tree roots `(n, d)`, two per tree, in schedule order.
    pub roots: Vec<Fp2>,
    /// The group's levels, from its roots.
    pub levels: Vec<SumcheckProof>,
    /// Its lookup trees' value evaluations.
    pub values: Vec<Fp2>,
    /// Its split values.
    pub splits: Vec<Fp2>,
}
/// Combined rows for one opened weight matrix (RFC-0052).
#[derive(Clone, PartialEq, Eq)]
pub struct WeightCombinedRows {
    /// Matrix declaration index, 0..9.
    pub matrix: u8,
    /// Original claims in pool order, one row each.
    pub evaluations: Vec<CombinedRow>,
    /// The seed-time proximity combination.
    pub proximity: CombinedRow,
}
/// Fixed-point bits, in units of `2^-8` bits; a one-field TCE struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedBits(pub u64);
/// Claimed per-request unexplained-information bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoringClaims {
    /// Bounds in statement batch order; exact mode requires zero.
    pub requests: Vec<FixedBits>,
}
/// Tree, stage and opening messages before column queries, schema `0x0511`.
#[derive(Clone, PartialEq, Eq)]
pub struct Round2Response {
    /// Lookup roots and value evaluations.
    pub logup: LogUpProof,
    /// Leaf levels and split values.
    pub trees: TreeProof,
    /// Stage `k` at index `k-1` (a later schedule check).
    pub stages: Vec<StageProof>,
    /// Advice group combinations, then the proximity row.
    pub advice_combined: Vec<CombinedRow>,
    /// Opened matrices in strictly increasing matrix-id order.
    pub weight_combined: Vec<WeightCombinedRows>,
    /// Per-request claimed bounds.
    pub scoring: ScoringClaims,
}
/// Draws for one weight matrix, preserving repeats and draw order.
#[derive(Clone, PartialEq, Eq)]
pub struct MatrixQueries {
    /// Matrix declaration index.
    pub matrix: u8,
    /// Drawn column indices; range/count are checked by the session.
    pub indices: Vec<u32>,
}
/// Live verifier draws fixed after all combined rows, schema `0x0512`.
#[derive(Clone, PartialEq, Eq)]
pub struct ColumnQueries {
    /// Advice draws with replacement.
    pub advice: Vec<u32>,
    /// Weight draws in strictly increasing matrix-id order.
    pub weights: Vec<MatrixQueries>,
}
/// Canonical multiproof siblings; canonicality is checked against the actual queries.
#[derive(Clone, PartialEq, Eq)]
pub struct MultiProof {
    /// Siblings in RFC-0004 order.
    pub siblings: Vec<Digest>,
}
/// Answer for one matrix, in increasing distinct-query order.
#[derive(Clone, PartialEq, Eq)]
pub struct MatrixColumns {
    /// Matrix declaration index.
    pub matrix: u8,
    /// Each queried column's actual committed rows.
    pub columns: Vec<Vec<Fp>>,
    /// The matrix's canonical multiproof.
    pub proof: MultiProof,
}
/// Authenticated-column answers, schema `0x0513`.
#[derive(Clone, PartialEq, Eq)]
pub struct Round3Response {
    /// Advice columns in increasing distinct-query order.
    pub advice_columns: Vec<Vec<Fp>>,
    /// Multiproof against Round1's advice root.
    pub advice_proof: MultiProof,
    /// Weight answers in strictly increasing matrix-id order.
    pub weight_columns: Vec<MatrixColumns>,
}
/// Non-interactive proof, with payloads rather than inner envelopes, schema `0x0520`.
#[derive(Clone, PartialEq, Eq)]
#[must_use = "decoding establishes syntax only; run every verifier check before acceptance"]
pub struct NiProof {
    /// Commitments fixed before the seed.
    pub round1: Round1,
    /// Seed proof-of-work nonce.
    pub seed_nonce: u64,
    /// All messages fixed before query generation.
    pub round2: Round2Response,
    /// Query proof-of-work nonce.
    pub query_nonce: u64,
    /// Answers to transcript-derived queries.
    pub round3: Round3Response,
}

macro_rules! counts {
    ($ty:ident, $($field:ident),+) => {
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($ty))$(.field(stringify!($field), &self.$field.len()))+.finish_non_exhaustive()
            }
        }
    };
}
counts!(LogUpProof, roots, values);
counts!(TreeProof, levels, groups, table_roots, splits);
counts!(GroupProof, roots, levels, values, splits);
counts!(WeightCombinedRows, evaluations);
counts!(Round2Response, stages, advice_combined, weight_combined);
counts!(MatrixQueries, indices);
counts!(ColumnQueries, advice, weights);
counts!(MultiProof, siblings);
counts!(MatrixColumns, columns);
counts!(Round3Response, advice_columns, weight_columns);
macro_rules! opaque {
    ($($ty:ident),+) => {$(
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(concat!(stringify!($ty), " { .. }"))
            }
        }
    )+};
}
opaque!(SeedCommit, ProverSeedShare, SeedReveal, NiProof);

fn fp2s(c: &mut Cursor<'_>) -> Result<Vec<Fp2>, DecodeError> {
    c.vec(size::FP2, Cursor::fp2)
}
fn combined(c: &mut Cursor<'_>) -> Result<CombinedRow, DecodeError> {
    c.nested(|c| {
        Ok(CombinedRow {
            u: fp2s(c)?,
            mask: fp2s(c)?,
        })
    })
}
fn sumcheck(c: &mut Cursor<'_>) -> Result<SumcheckProof, DecodeError> {
    c.nested(|c| {
        Ok(SumcheckProof {
            rounds: c.vec(size::SEQUENCE, fp2s)?,
            final_evals: fp2s(c)?,
        })
    })
}
fn group(c: &mut Cursor<'_>) -> Result<GroupProof, DecodeError> {
    c.nested(|c| {
        Ok(GroupProof {
            roots: fp2s(c)?,
            levels: c.vec(16, sumcheck)?,
            values: fp2s(c)?,
            splits: fp2s(c)?,
        })
    })
}
fn stage(c: &mut Cursor<'_>) -> Result<StageProof, DecodeError> {
    c.nested(|c| {
        Ok(StageProof {
            stage: c.nested(|c| c.u16())?,
            sumcheck: sumcheck(c)?,
            splits: fp2s(c)?,
        })
    })
}
fn columns(c: &mut Cursor<'_>) -> Result<Vec<Vec<Fp>>, DecodeError> {
    c.vec(size::SEQUENCE, |c| c.vec(size::FP, Cursor::fp))
}
fn proof(c: &mut Cursor<'_>) -> Result<MultiProof, DecodeError> {
    c.nested(|c| {
        Ok(MultiProof {
            siblings: c.vec(size::DIGEST, Cursor::digest)?,
        })
    })
}
fn matrices<T>(
    c: &mut Cursor<'_>,
    minimum: u64,
    mut fields: impl FnMut(&mut Cursor<'_>, u8) -> Result<T, DecodeError>,
) -> Result<Vec<T>, DecodeError> {
    let mut previous = None;
    c.vec(minimum, |c| {
        c.nested(|c| {
            let at = c.offset();
            let id = c.u8()?;
            if id > 9 || previous.is_some_and(|last| id <= last) {
                return Err(DecodeError::new(ErrorCode::Noncanonical, at));
            }
            previous = Some(id);
            fields(c, id)
        })
    })
}
impl Decoding {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.enumeration(2, |c, tag| match tag {
            0 => Ok(Self::Greedy),
            _ => Ok(Self::Temperature { t: c.rational()? }),
        })
    }
}
impl PredictorSpec {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.enumeration(2, |c, tag| {
            let decoding = Decoding::decode(c)?;
            Ok(match tag {
                0 => Self::Exact { decoding },
                _ => Self::Mixture {
                    decoding,
                    eps_log2: c.u8()?,
                },
            })
        })
    }
}
impl PublicStatement {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                claim_list_digest: c.digest()?,
                weights_root: c.digest()?,
                weights_epoch: c.u32()?,
                manifest_digest: c.digest()?,
                batch: c.nested(|c| {
                    Ok(BatchDescriptor {
                        requests: c.vec(32, |c| {
                            c.nested(|c| {
                                Ok(RequestRef {
                                    request_id: c.u64()?,
                                    session_id: c.byte_array()?,
                                    prompt_len: c.u32()?,
                                    output_len: c.u32()?,
                                })
                            })
                        })?,
                    })
                })?,
                predictor: PredictorSpec::decode(c)?,
            })
        })
    }
    /// Canonical statement bytes, reconstructed from its fields.
    ///
    /// # Errors
    /// `Memory` if the encoded length overflows or its reservation fails.
    pub fn to_tce(&self) -> Result<Vec<u8>, ErrorCode> {
        encode(self)
    }

    /// `H("statement", TCE(self))`, written directly into the independent hasher.
    pub fn digest(&self) -> Digest {
        let mut out = DomainHasher::new(Domain::Statement);
        self.write(&mut out);
        out.finalize()
    }
}
impl SeedCommit {
    /// Decode fields only; session/commitment checks belong to the driver.
    ///
    /// # Errors
    /// RFC-0003 truncation, depth and memory errors at their exact offsets.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                session_id: c.byte_array()?,
                commitment: c.digest()?,
            })
        })
    }
}
impl ProverSeedShare {
    /// Decode fields only, under the cursor's limits.
    ///
    /// # Errors
    /// RFC-0003 truncation and depth errors at their exact offsets.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                session_id: c.byte_array()?,
                share: c.byte_array()?,
            })
        })
    }
}
impl SeedReveal {
    /// Decode fields only; opening the commitment is a later session check.
    ///
    /// # Errors
    /// RFC-0003 truncation and depth errors at their exact offsets.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                session_id: c.byte_array()?,
                share: c.byte_array()?,
            })
        })
    }
}
impl Round1 {
    /// Decode every field, without judging the protocol number or public statement.
    ///
    /// # Errors
    /// RFC-0003 malformed tags, rationals, truncation and limits, at their exact offsets.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                protocol_version: c.u16()?,
                session_id: c.byte_array()?,
                statement: PublicStatement::decode(c)?,
                advice: c.nested(|c| {
                    Ok(AdviceCommitment {
                        root: c.digest()?,
                        layout_digest: c.digest()?,
                    })
                })?,
                leaf_groups: leaf_groups(c)?,
            })
        })
    }
}
/// RFC-0054's grouping: more than 14 group bits is `ERR-CODEC-LIMIT` at the struct's first byte.
fn leaf_groups(c: &mut Cursor<'_>) -> Result<LeafGroups, DecodeError> {
    let at = c.offset();
    let groups = c.nested(|c| {
        Ok(LeafGroups {
            token_bits: c.u8()?,
            layer_bits: c.u8()?,
        })
    })?;
    let bits = u16::from(groups.token_bits).saturating_add(u16::from(groups.layer_bits));
    if bits > u16::from(MAX_LEAF_GROUP_BITS) {
        return Err(DecodeError::new(ErrorCode::Limit, at));
    }
    Ok(groups)
}
impl Round2Response {
    /// Decode syntax, including canonical matrix ids, without schedule-dependent checks.
    ///
    /// # Errors
    /// RFC-0003 field, truncation and limit errors; a bad matrix id is Noncanonical at its byte.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                logup: c.nested(|c| {
                    Ok(LogUpProof {
                        roots: fp2s(c)?,
                        values: fp2s(c)?,
                    })
                })?,
                trees: c.nested(|c| {
                    Ok(TreeProof {
                        levels: c.vec(16, sumcheck)?,
                        groups: c.vec(32, group)?,
                        table_roots: fp2s(c)?,
                        merge: sumcheck(c)?,
                        splits: fp2s(c)?,
                    })
                })?,
                stages: c.vec(26, stage)?,
                advice_combined: c.vec(16, combined)?,
                weight_combined: matrices(c, 25, |c, matrix| {
                    Ok(WeightCombinedRows {
                        matrix,
                        evaluations: c.vec(16, combined)?,
                        proximity: combined(c)?,
                    })
                })?,
                scoring: c.nested(|c| {
                    Ok(ScoringClaims {
                        requests: c.vec(size::U64, |c| c.nested(|c| c.u64().map(FixedBits)))?,
                    })
                })?,
            })
        })
    }
}
impl ColumnQueries {
    /// Decode draw order/repeats and canonical matrix ids; counts/ranges are later checks.
    ///
    /// # Errors
    /// RFC-0003 truncation and limits; a bad matrix id is Noncanonical at its byte.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                advice: c.vec(size::U32, Cursor::u32)?,
                weights: matrices(c, 9, |c, matrix| {
                    Ok(MatrixQueries {
                        matrix,
                        indices: c.vec(size::U32, Cursor::u32)?,
                    })
                })?,
            })
        })
    }
}
impl Round3Response {
    /// Decode columns and proofs without judging their query-derived shapes or authentication.
    ///
    /// # Errors
    /// RFC-0003 field, truncation and limit errors; a bad matrix id is Noncanonical at its byte.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                advice_columns: columns(c)?,
                advice_proof: proof(c)?,
                weight_columns: matrices(c, 17, |c, matrix| {
                    Ok(MatrixColumns {
                        matrix,
                        columns: columns(c)?,
                        proof: proof(c)?,
                    })
                })?,
            })
        })
    }
}
impl NiProof {
    /// Decode nested payloads, without inner envelopes or semantic verification.
    ///
    /// # Errors
    /// The first RFC-0003 failure, including a nested message's error, at the original offset.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            Ok(Self {
                round1: Round1::decode(c)?,
                seed_nonce: c.u64()?,
                round2: Round2Response::decode(c)?,
                query_nonce: c.u64()?,
                round3: Round3Response::decode(c)?,
            })
        })
    }
}

fn decode_time(deadline: Option<std::time::Instant>, offset: u64) -> Result<(), DecodeError> {
    if deadline.is_some_and(|until| std::time::Instant::now() >= until) {
        return Err(DecodeError::new(ErrorCode::Time, offset));
    }
    Ok(())
}

fn envelope_digest(
    bytes: &[u8],
    mut check: impl FnMut(u64) -> Result<(), DecodeError>,
) -> Result<Digest, DecodeError> {
    let mut digest = DomainHasher::new(Domain::Envelope);
    let mut offset = 0u64;
    for block in bytes.chunks(65536) {
        check(offset)?;
        digest.update(block);
        offset = offset.saturating_add(u64::try_from(block.len()).unwrap_or(u64::MAX));
    }
    let digest = digest.finalize();
    check(offset)?;
    Ok(digest)
}

macro_rules! message {
    ($ty:ident, $id:literal) => {
        impl $ty {
            /// Its protocol/1 envelope schema; protocol/2 must never decode as this layout.
            pub const SCHEMA: Expected = Expected {
                schema_id: $id,
                version: 1,
            };
            /// Check the envelope and digest, then decode its payload under the same limits.
            ///
            /// # Errors
            /// The first RFC-0003 envelope/payload failure, with envelope-relative offset.
            pub fn open(bytes: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
                Self::open_with_deadline(bytes, limits, None)
            }
            pub(crate) fn open_with_deadline(
                bytes: &[u8],
                limits: &DecodeLimits,
                deadline: Option<std::time::Instant>,
            ) -> Result<Self, DecodeError> {
                decode_time(deadline, 0)?;
                Envelope::open_checked_digest(bytes, Self::SCHEMA, limits, |bytes| {
                    envelope_digest(bytes, |offset| decode_time(deadline, offset))
                })?
                .decode_payload(|c, _| {
                    if let Some(deadline) = deadline {
                        c.set_deadline(deadline)?;
                    }
                    Self::decode(c)
                })
            }
            /// Reconstruct its TCE payload from values, without an envelope.
            ///
            /// # Errors
            /// `Memory` if the encoded length overflows or its reservation fails.
            pub fn to_tce(&self) -> Result<Vec<u8>, ErrorCode> {
                encode(self)
            }
        }
    };
}
message!(SeedCommit, 0x0501);
message!(ProverSeedShare, 0x0502);
message!(SeedReveal, 0x0503);
message!(Round1, 0x0510);
message!(Round2Response, 0x0511);
message!(ColumnQueries, 0x0512);
message!(Round3Response, 0x0513);
message!(NiProof, 0x0520);

/// A message the prover sends in a live session, which [`crate::driver::open_live`] opens from
/// its sealed bytes and RFC-0043 'Malformed' names when it fails.
pub trait ProverMessage: Sized {
    /// Its envelope schema.
    const SCHEMA: Expected;
    /// Its envelope and payload decoded, as the type's own `open` does.
    ///
    /// # Errors
    /// The first RFC-0003 envelope or payload failure.
    fn open_sealed(bytes: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError>;
    /// Open under an absolute cooperative deadline; `None` retains ordinary decoding.
    ///
    /// # Errors
    /// The first framing/payload fault, or operational `Time` on expiry.
    fn open_sealed_with_deadline(
        bytes: &[u8],
        limits: &DecodeLimits,
        deadline: Option<std::time::Instant>,
    ) -> Result<Self, DecodeError>;
}
macro_rules! prover_message {
    ($($ty:ident),+) => {$(
        impl ProverMessage for $ty {
            const SCHEMA: Expected = $ty::SCHEMA;
            fn open_sealed(bytes: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
                $ty::open(bytes, limits)
            }
            fn open_sealed_with_deadline(bytes: &[u8], limits: &DecodeLimits, deadline: Option<std::time::Instant>) -> Result<Self, DecodeError> {
                $ty::open_with_deadline(bytes, limits, deadline)
            }
        }
    )+};
}
prover_message!(ProverSeedShare, Round1, Round2Response, Round3Response);

pub(crate) trait Sink {
    fn put(&mut self, bytes: &[u8]);
}
impl Sink for Vec<u8> {
    fn put(&mut self, b: &[u8]) {
        self.extend_from_slice(b);
    }
}
impl Sink for DomainHasher {
    fn put(&mut self, b: &[u8]) {
        self.update(b);
    }
}
struct Count(Option<usize>);
impl Sink for Count {
    fn put(&mut self, b: &[u8]) {
        self.0 = self.0.and_then(|n| n.checked_add(b.len()));
    }
}
pub(crate) trait Tce {
    fn write(&self, out: &mut impl Sink);
}
pub(crate) fn encode(value: &(impl Tce + ?Sized)) -> Result<Vec<u8>, ErrorCode> {
    let mut count = Count(Some(0));
    value.write(&mut count);
    let len = count.0.ok_or(ErrorCode::Memory)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| ErrorCode::Memory)?;
    value.write(&mut out);
    Ok(out)
}
macro_rules! integer {
    ($($ty:ty),+) => {$(impl Tce for $ty {
        fn write(&self, out: &mut impl Sink) { out.put(&self.to_le_bytes()); }
    })+};
}
integer!(u8, u16, u32, u64);
impl<const N: usize> Tce for [u8; N] {
    fn write(&self, out: &mut impl Sink) {
        out.put(self);
    }
}
impl<T: Tce> Tce for [T] {
    fn write(&self, out: &mut impl Sink) {
        u64::try_from(self.len()).unwrap_or(u64::MAX).write(out);
        for item in self {
            item.write(out);
        }
    }
}
impl<T: Tce> Tce for Vec<T> {
    fn write(&self, out: &mut impl Sink) {
        self.as_slice().write(out);
    }
}
pub(crate) fn encoded_len(value: &(impl Tce + ?Sized)) -> Result<u64, ErrorCode> {
    let mut count = Count(Some(0));
    value.write(&mut count);
    u64::try_from(count.0.ok_or(ErrorCode::Memory)?).map_err(|_| ErrorCode::Memory)
}
impl Tce for Fp {
    fn write(&self, out: &mut impl Sink) {
        self.to_canonical().write(out);
    }
}
impl Tce for Fp2 {
    fn write(&self, out: &mut impl Sink) {
        self.c0.write(out);
        self.c1.write(out);
    }
}
impl Tce for Rational {
    fn write(&self, out: &mut impl Sink) {
        out.put(&self.num().to_le_bytes());
        self.den().write(out);
    }
}
impl Tce for FixedBits {
    fn write(&self, out: &mut impl Sink) {
        self.0.write(out);
    }
}
impl Tce for Decoding {
    fn write(&self, out: &mut impl Sink) {
        match self {
            Self::Greedy => 0u16.write(out),
            Self::Temperature { t } => {
                1u16.write(out);
                t.write(out);
            }
        }
    }
}
impl Tce for PredictorSpec {
    fn write(&self, out: &mut impl Sink) {
        match self {
            Self::Exact { decoding } => {
                0u16.write(out);
                decoding.write(out);
            }
            Self::Mixture { decoding, eps_log2 } => {
                1u16.write(out);
                decoding.write(out);
                eps_log2.write(out);
            }
        }
    }
}
macro_rules! fields {
    ($ty:ty, $($field:ident),+) => {impl Tce for $ty {
        fn write(&self, out: &mut impl Sink) { $(self.$field.write(out);)+ }
    }};
}
fields!(SeedCommit, session_id, commitment);
fields!(ProverSeedShare, session_id, share);
fields!(SeedReveal, session_id, share);
fields!(RequestRef, request_id, session_id, prompt_len, output_len);
fields!(BatchDescriptor, requests);
fields!(
    PublicStatement,
    claim_list_digest,
    weights_root,
    weights_epoch,
    manifest_digest,
    batch,
    predictor
);
fields!(AdviceCommitment, root, layout_digest);
fields!(LeafGroups, token_bits, layer_bits);
fields!(
    Round1,
    protocol_version,
    session_id,
    statement,
    advice,
    leaf_groups
);
fields!(LogUpProof, roots, values);
fields!(TreeProof, levels, groups, table_roots, merge, splits);
fields!(GroupProof, roots, levels, values, splits);
fields!(SumcheckProof, rounds, final_evals);
fields!(StageProof, stage, sumcheck, splits);
fields!(CombinedRow, u, mask);
fields!(WeightCombinedRows, matrix, evaluations, proximity);
fields!(ScoringClaims, requests);
fields!(
    Round2Response,
    logup,
    trees,
    stages,
    advice_combined,
    weight_combined,
    scoring
);
fields!(MatrixQueries, matrix, indices);
fields!(ColumnQueries, advice, weights);
fields!(MultiProof, siblings);
fields!(MatrixColumns, matrix, columns, proof);
fields!(Round3Response, advice_columns, advice_proof, weight_columns);
fields!(NiProof, round1, seed_nonce, round2, query_nonce, round3);

#[cfg(test)]
mod deadline_tests {
    use super::*;
    #[test]
    fn envelope_hash_checks_between_blocks_and_before_completion() {
        let bytes = vec![7; 3 * 65536];
        assert_eq!(
            envelope_digest(&bytes, |_| Ok(())).unwrap(),
            tv_hash::hash(Domain::Envelope, &bytes)
        );
        for stop in 1..=4 {
            let mut polls = 0;
            let error = envelope_digest(&bytes, |offset| {
                polls += 1;
                if polls == stop {
                    Err(DecodeError::new(ErrorCode::Time, offset))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
            assert_eq!(error, DecodeError::new(ErrorCode::Time, (stop - 1) * 65536));
        }
        assert_eq!(
            NiProof::open_with_deadline(
                &[],
                &DecodeLimits::MESSAGE,
                Some(std::time::Instant::now())
            )
            .unwrap_err()
            .code(),
            ErrorCode::Time
        );
    }
}
