// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Round2Response` and its parts (RFC-0036 '`Round2Response`', RFC-0037 'Messages and
//! transcript').
//!
//! The order of the fields is not that of the transcript: the verifier reads the leaf stage as
//! `logup.roots`, `trees.levels`, `logup.values`, then `trees.splits`. Round messages,
//! evaluations, split values and combined rows are protocol-class (05 §3), so the debug forms
//! of these types show lengths only. The scoring claims are public.

use core::fmt;

use tessera_codec::{Decode, DecodeError, Encode, Reader, Sink};
use tessera_field::GoldilocksExt2;
use tessera_ir::StageId;
use tessera_sumcheck::SumcheckProof;

use crate::matrix::{ByMatrix, decode_by_matrix, read_matrix};

/// Step 3: the prover's messages from the seed to the openings (schema `0x0511`).
///
/// The schedule and the statement fix every length the decoder cannot check alone: one stage
/// per reduction stage, `stages[i].stage = i + 1`, the matrices of `weight_combined`, one
/// scoring claim per request and each stage's split values. The verifier checks them after
/// RFC-0019 step 6 (`Reject(Malformed)`).
#[derive(Clone, Debug, PartialEq, Eq, Encode)]
pub struct Round2Response {
    /// The LogUp roots and value evaluations of the leaf stage.
    pub logup: LogUpProof,
    /// The leaf stage's levels and split values.
    pub trees: TreeProof,
    /// `stages[i]` is reduction stage `i + 1`.
    pub stages: Vec<StageProof>,
    /// RFC-0011's `G + 1` combined rows: the groups in group order, then the proximity row.
    pub advice_combined: Vec<CombinedRow>,
    /// One entry per weight matrix the proof opens, `matrix` strictly increasing.
    pub weight_combined: Vec<WeightCombinedRows>,
    /// The claimed `Û_r` per request.
    pub scoring: ScoringClaims,
}

/// `weight_combined` is decoded by matrix id, which strictly increases.
impl Decode for Round2Response {
    const MIN_SIZE: usize = LogUpProof::MIN_SIZE
        + TreeProof::MIN_SIZE
        + <Vec<StageProof>>::MIN_SIZE
        + <Vec<CombinedRow>>::MIN_SIZE
        // weight_combined's length.
        + u64::MIN_SIZE
        + ScoringClaims::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let value = Self {
            logup: Decode::decode_from(reader)?,
            trees: Decode::decode_from(reader)?,
            stages: Decode::decode_from(reader)?,
            advice_combined: Decode::decode_from(reader)?,
            weight_combined: decode_by_matrix(reader)?,
            scoring: Decode::decode_from(reader)?,
        };
        reader.leave();
        Ok(value)
    }
}

/// The LogUp part of the leaf stage (RFC-0008 'Fraction tree'), every instance in schedule
/// order. Both lists are absorbed even when empty.
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub struct LogUpProof {
    /// `n_a, d_a, n_t, d_t` of every instance, 4 per instance, under `logup/roots`.
    pub roots: Vec<GoldilocksExt2>,
    /// `valũe(ρ_a)` of every instance whose table has values, under `logup/values`.
    pub values: Vec<GoldilocksExt2>,
}

/// Lengths only: the roots and values are protocol-class.
impl fmt::Debug for LogUpProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogUpProof")
            .field("roots", &self.roots.len())
            .field("values", &self.values.len())
            .finish()
    }
}

/// The leaf stage's levels (RFC-0010 'Tree levels'), its token groups and their merge
/// (RFC-0054 'Messages'), then the values of its split claims, stage 0's (RFC-0037 'Splits').
///
/// An ungrouped stage, `Round1.leaf_groups` zero, has no groups, no table roots and a merge of
/// no rounds and no evaluations, and nothing is absorbed for them.
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub struct TreeProof {
    /// `levels[ℓ]` is level `ℓ` from the roots: the one sumcheck its fraction trees and product
    /// forests share. In a grouped stage, the table batch's levels.
    pub levels: Vec<SumcheckProof<GoldilocksExt2>>,
    /// One entry per token group, in group order; empty when the stage is ungrouped.
    pub groups: Vec<GroupProof>,
    /// The table trees' roots, under `tables/logup/roots`; empty when the stage is ungrouped.
    pub table_roots: Vec<GoldilocksExt2>,
    /// The merge's rounds and evaluations, under `merge/round/{j}` and `merge/evals`; empty when
    /// the stage is ungrouped.
    pub merge: SumcheckProof<GoldilocksExt2>,
    /// The leaf stage's split values, under `stage/0/split`: in a grouped stage, RFC-0037's atom
    /// splits of the merged claims.
    pub splits: Vec<GoldilocksExt2>,
}

impl TreeProof {
    /// An ungrouped leaf stage's proof: its levels and its split values.
    pub fn ungrouped(
        levels: Vec<SumcheckProof<GoldilocksExt2>>,
        splits: Vec<GoldilocksExt2>,
    ) -> Self {
        Self {
            levels,
            groups: Vec::new(),
            table_roots: Vec::new(),
            merge: SumcheckProof {
                rounds: Vec::new(),
                final_evals: Vec::new(),
            },
            splits,
        }
    }
}

/// The levels' own forms and counts: tree messages are protocol-class.
impl fmt::Debug for TreeProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeProof")
            .field("levels", &self.levels)
            .field("groups", &self.groups)
            .field("table_roots", &self.table_roots.len())
            .field("merge", &self.merge)
            .field("splits", &self.splits.len())
            .finish()
    }
}

/// One token group's part of a grouped leaf stage (RFC-0054 'The leaf stage with groups').
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub struct GroupProof {
    /// The roots `(n, d)` of the group's lookup trees, in schedule order, under
    /// `group/{h}/logup/roots`.
    pub roots: Vec<GoldilocksExt2>,
    /// The group's levels, under `group/{h}/tree/{ℓ}/…`.
    pub levels: Vec<SumcheckProof<GoldilocksExt2>>,
    /// Its lookup trees' value evaluations, under `group/{h}/logup/values`.
    pub values: Vec<GoldilocksExt2>,
    /// Its split values, under `group/{h}/split`.
    pub splits: Vec<GoldilocksExt2>,
}

/// Counts and the levels' own forms: group messages are protocol-class.
impl fmt::Debug for GroupProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupProof")
            .field("roots", &self.roots.len())
            .field("levels", &self.levels)
            .field("values", &self.values.len())
            .field("splits", &self.splits.len())
            .finish()
    }
}

/// A reduction stage's messages (RFC-0010 'Sumcheck messages'), then the values of the claims
/// it splits (RFC-0037 'Splits').
///
/// Its TCE encoding is the struct's, `stage` a struct of one `u16` (03 §3): one level of depth
/// of its own, as RFC-0003 'Limits' counts it.
#[derive(Clone, PartialEq, Eq)]
pub struct StageProof {
    /// The stage, from 1: the leaf stage is stage 0.
    pub stage: StageId,
    /// Its rounds and final evaluations.
    pub sumcheck: SumcheckProof<GoldilocksExt2>,
    /// The values of its split claims' parts, claim by claim, under `stage/{k}/split`.
    pub splits: Vec<GoldilocksExt2>,
}

/// The stage, the sumcheck's own form and a count: stage messages are protocol-class.
impl fmt::Debug for StageProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageProof")
            .field("stage", &self.stage.0)
            .field("sumcheck", &self.sumcheck)
            .field("splits", &self.splits.len())
            .finish()
    }
}

/// The fields in order, `StageId(u16)` as its one field.
impl Encode for StageProof {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.stage.0.encode_to(out);
        self.sumcheck.encode_to(out);
        self.splits.encode_to(out);
    }
}

impl Decode for StageProof {
    const MIN_SIZE: usize =
        u16::MIN_SIZE + <SumcheckProof<GoldilocksExt2>>::MIN_SIZE + <Vec<GoldilocksExt2>>::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        // `StageId` is a struct: it takes a level before its bytes are read.
        reader.enter()?;
        let stage = StageId(u16::decode_from(reader)?);
        reader.leave();
        let sumcheck = Decode::decode_from(reader)?;
        let splits = Decode::decode_from(reader)?;
        reader.leave();
        Ok(Self {
            stage,
            sumcheck,
            splits,
        })
    }
}

/// A combined row (RFC-0011 'Opening protocol' step 5; RFC-0012 'Opening by the folding pass').
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub struct CombinedRow {
    /// `u`: `k` values for the advice, `k_T` for matrix `T`.
    pub u: Vec<GoldilocksExt2>,
    /// The combined mask polynomial's coefficients, constant first: `P_A` for the advice, none
    /// for a weight matrix in `tessera-protocol/1`.
    pub mask: Vec<GoldilocksExt2>,
}

/// Lengths only: combined rows are protocol-class.
impl fmt::Debug for CombinedRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CombinedRow")
            .field("u", &self.u.len())
            .field("mask", &self.mask.len())
            .finish()
    }
}

/// One opened weight matrix's combined rows (RFC-0012 'Opening by the folding pass',
/// RFC-0052 '`WeightCombinedRows`').
///
/// It has no `Decode` of its own: an entry decodes only inside its message, which checks that
/// the ids strictly increase (RFC-0036 'Matrix ids').
#[derive(Clone, Debug, PartialEq, Eq, Encode)]
pub struct WeightCombinedRows {
    /// The matrix's id ([`WeightMatrix`](crate::WeightMatrix)).
    pub matrix: u8,
    /// The evaluation combined rows: one for `Embed`..`LmHead`, and `m ≥ 1` in pool order for
    /// `ChannelParams` (RFC-0052).
    pub evaluations: Vec<CombinedRow>,
    /// The proximity combined row.
    pub proximity: CombinedRow,
}

/// Decoded only inside `Round2Response`, whose order it must keep.
impl ByMatrix for WeightCombinedRows {
    const MIN_SIZE: usize = u8::MIN_SIZE + <Vec<CombinedRow>>::MIN_SIZE + CombinedRow::MIN_SIZE;

    fn decode_after(reader: &mut Reader<'_>, previous: Option<u8>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let matrix = read_matrix(reader, previous)?;
        let evaluations = Decode::decode_from(reader)?;
        let proximity = Decode::decode_from(reader)?;
        reader.leave();
        Ok(Self {
            matrix,
            evaluations,
            proximity,
        })
    }

    fn matrix(&self) -> u8 {
        self.matrix
    }
}

/// The scoring claims (RFC-0017 'In-circuit scoring', step 5): public.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ScoringClaims {
    /// One claimed `Û_r` per request, in `statement.batch` order: 0 in exact mode.
    pub requests: Vec<FixedBits>,
}

/// A quantity of bits in units of `2^{−8}` bits: the scoring claims, and the values of
/// `Reject(UBoundExceeded)` (04 §2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode)]
pub struct FixedBits(pub u64);
