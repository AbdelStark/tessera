// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage checks: the leaf stage's splits and every reduction stage of a
//! [`Schedule`], RFC-0019 'Check order' step 8 for
//! `tessera-protocol/1` (RFC-0010 'Batching within a stage', 'Sumcheck messages'; RFC-0037
//! 'Splits', 'The layer point', 'Stages and their instances'; RFC-0035 'Zero-check points').
//! Zero-knowledge terms are `tessera-protocol/2`'s (#159), and the leaf stage's trees and the
//! gadgets' leaf checks (#124) give the leaf claims these checks start from.
//!
//! # What is checked, in order
//!
//! [`verify`]`(transcript, schedule, rope, leaf, leaf_splits, stages)` starts from the transcript
//! as the leaf stage leaves it, after `logup/values`, and from the leaf claims `leaf`: for each
//! leaf claim of the schedule, in order, its point and its values, one per real layer for a
//! template site and one otherwise. In this order, the first failure being the error:
//!
//! 1. **The verifier's inputs.** Each leaf claim has a point of its tensor's variables and as many
//!    values as it should, else [`StageError::Leaf`], a fault of the verifier.
//! 2. **Shapes** (RFC-0036 'Shapes', which the decoder of `Round2Response` checks first): as many
//!    stage proofs as reduction stages, else [`StageError::StageCount`]; stage proof `i` names
//!    stage `i + 1`, else [`StageError::StageId`]; each stage, the leaf stage first, sends as many
//!    split values as its split claims, else [`StageError::SplitCount`]. All three are
//!    `Reject(Malformed)`.
//! 3. **The leaf stage's splits.** `stage/0/split` absorbs `leaf_splits`. Site by site, layer by
//!    layer, claim by claim, each split claim's values must give its value: a virtual part's
//!    value weighted by its atom's coefficient at that layer, a committed group's counted once
//!    (RFC-0037 'Splits', as amended). A mismatch is `Reject(Sumcheck { stage: 0, round: 0 })`.
//! 4. **Each reduction stage `k`**, in order:
//!    1. stage 1 squeezes the layer point `ρ_1`, `λ` coordinates under `stage/1/layer/{c}`;
//!    2. a stage with a zero-check squeezes its zero-check point under `stage/{k}/zero/{c}`;
//!    3. each instance's claim: a leaf claim's per-layer values stacked as
//!       `Σ_ℓ eq(ρ_1, ℓ)·v_ℓ`, a padded layer's value being 0; the value a part or a ready
//!       claim of stage `k − 1` gives; 0 for a zero-check;
//!    4. the batched sumcheck of [`sumcheck::verify_batch`] under
//!       `stage/{k}`, of the stage's degree, each instance's final value computed with its public
//!       factors (the table of [`schedule`](crate::schedule)): `eq` of the claim's point against
//!       the instance's, `w̃(r'_ℓ) = eq(ρ, r'_ℓ)` for a stacked instance (the sum over the whole
//!       cube of `2^λ` layers), the heads' `eq` of `MatMulAct`, `Rotate`'s kernel `Ẽ_r(r')`, the
//!       constant of an equality zero-check and `eq(z_i, r')`. A failure is
//!       `Reject(Sumcheck { stage: k, round })` with the round the sumcheck reports;
//!    5. `stage/{k}/split` absorbs the stage's split values, and each split claim, instance by
//!       instance and evaluation by evaluation, must add up as in step 3:
//!       `Reject(Sumcheck { stage: k, round: n_k })` otherwise, `n_k` the stage's rounds.
//!
//! It returns the pool ([`Verified`]): every claim on committed tensors, as a combination at one
//! point (RFC-0037 'The claim pool'). A stacked kind's term has a weight for each real layer `ℓ`:
//! `eq(ρ, ℓ)` times its coefficient there, or the one layer `c_L` reads. The layer input's term
//! is split into its slices, `c_0` with the weight of layer 0 and `X_last` with each next
//! layer's weight. Nothing here checks the pool, so the result is `#[must_use]`.
//!
//! # Robustness
//!
//! Every length the proof gives is compared with the schedule before anything is computed from it,
//! the sumchecks' by [`sumcheck::verify_batch`]; nothing panics,
//! indexes unchecked or computes sizes unchecked (RFC-0019 'Robustness'). The work is linear in the
//! schedule's instances and the proof's length, plus `n·d_head/2` constant pairs for each `Rotate`
//! instance.

use tv_field::{Fp, Fp2};
use tv_hash::transcript::Transcript;
use tv_tables::rope::Rope;

use crate::fiat_shamir::{self, FiatShamir, Step};
use crate::lowering::TensorId;
use crate::schedule::{
    Atom, Claim, Coefficient, Commitment, Instance, Layers, LeafClaim, Part, Point, Relation,
    Route, Schedule, Source, TensorRef,
};
use crate::selectors::rotate_kernel;
use crate::sumcheck::{self, Prefix, SumcheckError, SumcheckProof};

/// A reduction stage's messages (RFC-0036 `StageProof`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StageProof {
    /// The stage's index `k`, from 1.
    pub stage: u16,
    /// Its sumcheck's rounds and final evaluations.
    pub sumcheck: SumcheckProof,
    /// The values of the claims it splits.
    pub splits: Vec<Fp2>,
}

/// One leaf claim as the leaf stage leaves it: its point and its values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeafValues {
    /// The point, its lowest coordinates first, of the claimed tensor's variables.
    pub point: Vec<Fp2>,
    /// The value at each real layer for a template site's claim, else the one value.
    pub values: Vec<Fp2>,
}

/// A tensor of a pool claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolTensor {
    /// A kind stacked over the layers, named by layer 0's tensor.
    Stacked(TensorId),
    /// A tensor outside the layers.
    Single(TensorId),
}

/// One term of a pool claim: a tensor and its weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolTerm {
    /// The tensor.
    pub tensor: PoolTensor,
    /// For a stacked kind, its weight at each real layer; for a tensor outside the layers, its
    /// one coefficient.
    pub weights: Vec<Fp2>,
}

/// A claim of the pool: `Σ_t Σ_ℓ w_t(ℓ)·P̃_{t,ℓ}(point) = value`, every tensor of one commitment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolClaim {
    /// The stage that left it, 0 for the leaf stage.
    pub stage: u16,
    /// The commitment its tensors belong to.
    pub commitment: Commitment,
    /// The point, its lowest coordinates first, without the layer index.
    pub point: Vec<Fp2>,
    /// The terms.
    pub terms: Vec<PoolTerm>,
    /// The claimed value.
    pub value: Fp2,
}

/// Stages that passed every check of [`verify`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "the stages leave claims on committed tensors, which the openings must check"]
pub struct Verified {
    /// The claim pool, in the order the stages left its claims.
    pub pool: Vec<PoolClaim>,
}

/// The payload of `Reject(Sumcheck { stage, round })` (04 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SumcheckRejection {
    /// The stage, 0 for the leaf stage.
    pub stage: u16,
    /// The round, counted from 0 as the labels count; the stage's rounds for its final check and
    /// its splits.
    pub round: u32,
}

/// Why stages were rejected or could not be checked: the first check that failed, in the order of
/// the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageError {
    /// The proof has `found` stage proofs where the schedule has `expected` reduction stages:
    /// `Reject(Malformed)`.
    StageCount {
        /// The schedule's reduction stages.
        expected: usize,
        /// The stage proofs.
        found: usize,
    },
    /// Stage proof `index` names stage `found`, not `index + 1`: `Reject(Malformed)`.
    StageId {
        /// The stage proof's position.
        index: usize,
        /// The stage it names.
        found: u16,
    },
    /// Stage `stage` sends `found` split values where its split claims send `expected`:
    /// `Reject(Malformed)`.
    SplitCount {
        /// The stage, 0 for the leaf stage.
        stage: u16,
        /// The values its split claims send.
        expected: usize,
        /// The values it sends.
        found: usize,
    },
    /// Stage `stage`'s sumcheck failed, or a challenge of the stage could not be drawn: see
    /// [`SumcheckError`] for whose fault.
    Sumcheck {
        /// The stage.
        stage: u16,
        /// The failure.
        error: SumcheckError,
    },
    /// A claim stage `stage` splits is not given by its values: `Reject(Sumcheck { stage, round
    /// })`, `round` the stage's rounds.
    Split {
        /// The stage, 0 for the leaf stage.
        stage: u16,
        /// The stage's rounds, 0 for the leaf stage.
        round: u32,
    },
    /// Leaf claim `claim` does not fit the schedule: a point or values of another length. A fault
    /// of the verifier, which derived them.
    Leaf {
        /// The leaf claim.
        claim: usize,
    },
    /// Memory for the checks could not be reserved.
    Memory,
    /// A step the schedule makes infallible failed, such as a public factor it names that cannot
    /// be evaluated: a bug of the verifier.
    Internal,
}

impl StageError {
    /// The payload of `Reject(Sumcheck { stage, round })` for a sumcheck or a split that failed,
    /// or `None` for another error.
    pub fn rejection(&self) -> Option<SumcheckRejection> {
        match *self {
            StageError::Sumcheck { stage, error } => error
                .rejected_round()
                .map(|round| SumcheckRejection { stage, round }),
            StageError::Split { stage, round } => Some(SumcheckRejection { stage, round }),
            _ => None,
        }
    }

    /// Whether it is `Reject(Malformed)`: a shape of the proof that the schedule fixes.
    pub fn is_malformed(&self) -> bool {
        matches!(
            self,
            StageError::StageCount { .. }
                | StageError::StageId { .. }
                | StageError::SplitCount { .. }
        )
    }
}

impl core::fmt::Display for StageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StageError::StageCount { expected, found } => {
                write!(
                    f,
                    "{found} stage proofs where the schedule has {expected} stages"
                )
            }
            StageError::StageId { index, found } => {
                write!(f, "stage proof {index} names stage {found}")
            }
            StageError::SplitCount {
                stage,
                expected,
                found,
            } => write!(
                f,
                "stage {stage} sends {found} split values where its split claims send {expected}"
            ),
            StageError::Sumcheck { stage, error } => write!(f, "stage {stage}: {error}"),
            StageError::Split { stage, .. } => {
                write!(f, "a claim stage {stage} splits is not given by its values")
            }
            StageError::Leaf { claim } => write!(f, "leaf claim {claim} does not fit the schedule"),
            StageError::Memory => f.write_str("memory for the stage checks could not be reserved"),
            StageError::Internal => f.write_str("an internal step of the stage checks failed"),
        }
    }
}

impl std::error::Error for StageError {}

/// Checks the leaf stage's splits and every reduction stage of `schedule`, in the order of the
/// module documentation, and returns the claim pool.
///
/// `rope` is the claim list's `Rope` table, regenerated, whose constants `Rotate`'s kernel reads.
/// `leaf` holds the leaf claims' points and values in the schedule's order, `leaf_splits` the leaf
/// stage's split values (`TreeProof.splits`), and `stages` the reduction stages' messages.
pub fn verify(
    transcript: &mut Transcript,
    schedule: &Schedule,
    rope: &Rope,
    leaf: &[LeafValues],
    leaf_splits: &[Fp2],
    stages: &[StageProof],
) -> Result<Verified, StageError> {
    verify_with(transcript, schedule, rope, leaf, leaf_splits, stages)
}

/// What an instance's claim gets from the stage before.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Incoming {
    Missing,
    Value(Fp2),
    PerLayer(Vec<Fp2>),
}

/// A pool claim of the leaf stage, its per-layer values gathered before `ρ_1` stacks them.
struct LeafGroup {
    claim: usize,
    atoms: Vec<usize>,
    values: Vec<Fp2>,
}

/// The transcript's coordinates as the stages draw them.
#[derive(Default)]
struct Sources {
    leaf: Vec<Vec<Fp2>>,
    layer: Vec<Fp2>,
    rounds: Vec<Vec<Fp2>>,
    zero: Vec<Vec<Fp2>>,
}

impl Sources {
    fn resolve(&self, point: &Point) -> Result<Vec<Fp2>, StageError> {
        let mut out = Vec::new();
        reserve(&mut out, point.len())?;
        for run in point.runs() {
            let source: &[Fp2] = match run.source {
                Source::Leaf(c) => self.leaf.get(index(c)?).ok_or(StageError::Internal)?,
                Source::Layer => &self.layer,
                Source::Rounds(k) => stage_entry(&self.rounds, k)?,
                Source::Zero(k) => stage_entry(&self.zero, k)?,
            };
            let end = run.start.checked_add(run.len).ok_or(StageError::Internal)?;
            out.extend_from_slice(source.get(run.start..end).ok_or(StageError::Internal)?);
        }
        Ok(out)
    }
}

/// The entry of stage `k`, from 1.
fn stage_entry(entries: &[Vec<Fp2>], k: u16) -> Result<&[Fp2], StageError> {
    let i = usize::from(k).checked_sub(1).ok_or(StageError::Internal)?;
    entries
        .get(i)
        .map(Vec::as_slice)
        .ok_or(StageError::Internal)
}

/// An instance's points, resolved before its stage's sumcheck.
struct Resolved {
    claim: Vec<Fp2>,
    layers: Option<Vec<Fp2>>,
    zero: Vec<Fp2>,
}

/// [`verify`] on any transcript.
pub(crate) fn verify_with(
    transcript: &mut impl FiatShamir,
    schedule: &Schedule,
    rope: &Rope,
    leaf: &[LeafValues],
    leaf_splits: &[Fp2],
    stages: &[StageProof],
) -> Result<Verified, StageError> {
    // 1. and 2. The verifier's inputs, then the proof's shapes.
    check_inputs(schedule, leaf)?;
    check_shapes(schedule, leaf_splits, stages)?;
    let layers = index(schedule.layers())?;

    let mut sources = Sources::default();
    reserve(&mut sources.leaf, leaf.len())?;
    for values in leaf {
        sources.leaf.push(values.point.clone());
    }
    let mut pool = Vec::new();

    // 3. The leaf stage's splits.
    fiat_shamir::absorb(transcript, Prefix::Stage(0), Step::Split, leaf_splits)
        .map_err(|error| StageError::Sumcheck { stage: 0, error })?;
    let first = schedule.stages().first().ok_or(StageError::Internal)?;
    let mut incoming = vec_of(Incoming::Missing, first.instances.len())?;
    let groups = leaf_stage(schedule, &sources, leaf, leaf_splits, &mut incoming)?;

    // 4. The reduction stages.
    for (stage, proof) in schedule.stages().iter().zip(stages) {
        let k = stage.index;
        let prefix = Prefix::Stage(u32::from(k));
        let fail = |error| StageError::Sumcheck { stage: k, error };
        // 4.1 and 4.2: the layer point, then the zero-check point.
        if k == 1 {
            sources.layer = squeeze(transcript, k, stage.layer_vars, Step::Layer)?;
            let eq_layers = layer_eq(&sources.layer, layers)?;
            for group in groups.iter() {
                push(
                    &mut pool,
                    leaf_pool_claim(schedule, &sources, &eq_layers, group)?,
                )?;
            }
        }
        let zero = squeeze(transcript, k, stage.zero_vars, Step::Zero)?;
        push(&mut sources.zero, zero)?;

        // 4.3: the instances' claims, one arriving for each.
        if incoming.len() != stage.instances.len() {
            return Err(StageError::Internal);
        }
        let eq_layers = layer_eq(&sources.layer, layers)?;
        let mut batch = Vec::new();
        let mut resolved = Vec::new();
        reserve(&mut batch, stage.instances.len())?;
        reserve(&mut resolved, stage.instances.len())?;
        for (instance, arriving) in stage.instances.iter().zip(&incoming) {
            let claim = match (&instance.relation, arriving) {
                (
                    Relation::Equality { .. }
                    | Relation::Booleanity
                    | Relation::Vocabulary { .. }
                    | Relation::OneHotCount { .. }
                    | Relation::OneHotIndex { .. },
                    Incoming::Missing,
                ) => Fp2::ZERO,
                (_, Incoming::Value(v)) => *v,
                (_, Incoming::PerLayer(values)) => stacked(&eq_layers, values)?,
                _ => return Err(StageError::Internal),
            };
            batch.push(sumcheck::Instance {
                claim,
                vars: instance.vars,
                evals: instance.evals.len(),
            });
            resolved.push(Resolved {
                claim: sources.resolve(&instance.point)?,
                layers: match &instance.layers {
                    Layers::Point(p) => Some(sources.resolve(p)?),
                    Layers::None => None,
                    Layers::One(_) => return Err(StageError::Internal),
                },
                zero: sources.resolve(&instance.zero)?,
            });
        }
        // 4.4: the sumcheck.
        let lambda = schedule.layer_vars();
        let passed = sumcheck::verify_batch_with(
            transcript,
            prefix,
            stage.degree,
            &batch,
            &proof.sumcheck,
            |i, point, evals| {
                let instance = stage.instances.get(i)?;
                let resolved = resolved.get(i)?;
                final_value(instance, resolved, lambda, rope, point, evals)
            },
        )
        .map_err(fail)?;
        push(&mut sources.rounds, passed.point)?;

        // 4.5: the splits, then the claims the evaluations leave.
        fiat_shamir::absorb(transcript, prefix, Step::Split, &proof.splits).map_err(fail)?;
        let next_len = schedule
            .stages()
            .get(usize::from(k))
            .map_or(0, |next| next.instances.len());
        let mut next = vec_of(Incoming::Missing, next_len)?;
        let round = u32::try_from(stage.rounds).map_err(|_| StageError::Internal)?;
        let mut splits = proof.splits.iter();
        let mut evals = proof.sumcheck.final_evals.iter();
        for instance in &stage.instances {
            for claim in &instance.evals {
                let value = *evals.next().ok_or(StageError::Internal)?;
                follow(
                    schedule,
                    &sources,
                    claim,
                    value,
                    &mut splits,
                    &mut next,
                    &mut pool,
                    k,
                    round,
                )?;
            }
        }
        incoming = next;
    }
    Ok(Verified { pool })
}

/// Step 1: each leaf claim's point and values fit the schedule.
fn check_inputs(schedule: &Schedule, leaf: &[LeafValues]) -> Result<(), StageError> {
    if leaf.len() != schedule.leaf().len() {
        return Err(StageError::Leaf {
            claim: leaf.len().min(schedule.leaf().len()),
        });
    }
    for (c, (claim, values)) in schedule.leaf().iter().zip(leaf).enumerate() {
        let per_layer = if claim.stacked {
            index(schedule.layers())?
        } else {
            1
        };
        if values.point.len() != claim.claim.vars || values.values.len() != per_layer {
            return Err(StageError::Leaf { claim: c });
        }
    }
    Ok(())
}

/// Step 2: the shapes RFC-0036 fixes.
fn check_shapes(
    schedule: &Schedule,
    leaf_splits: &[Fp2],
    stages: &[StageProof],
) -> Result<(), StageError> {
    if stages.len() != schedule.stages().len() {
        return Err(StageError::StageCount {
            expected: schedule.stages().len(),
            found: stages.len(),
        });
    }
    for (i, proof) in stages.iter().enumerate() {
        let expected = i.checked_add(1).ok_or(StageError::Internal)?;
        if usize::from(proof.stage) != expected {
            return Err(StageError::StageId {
                index: i,
                found: proof.stage,
            });
        }
    }
    if leaf_splits.len() != schedule.leaf_splits() {
        return Err(StageError::SplitCount {
            stage: 0,
            expected: schedule.leaf_splits(),
            found: leaf_splits.len(),
        });
    }
    for (stage, proof) in schedule.stages().iter().zip(stages) {
        if proof.splits.len() != stage.splits {
            return Err(StageError::SplitCount {
                stage: stage.index,
                expected: stage.splits,
                found: proof.splits.len(),
            });
        }
    }
    Ok(())
}

/// Step 3: the leaf stage's split claims, site by site, layer by layer, claim by claim; the
/// virtual atoms' per-layer values go to stage 1's instances, and the committed groups' are
/// returned for the pool.
fn leaf_stage(
    schedule: &Schedule,
    sources: &Sources,
    leaf: &[LeafValues],
    leaf_splits: &[Fp2],
    incoming: &mut [Incoming],
) -> Result<Vec<LeafGroup>, StageError> {
    let layers = index(schedule.layers())?;
    let claims = schedule.leaf();
    let mut groups: Vec<LeafGroup> = Vec::new();
    // The pool claims of each leaf claim, in order: its whole self, or its groups.
    let mut first_group = Vec::new();
    reserve(&mut first_group, claims.len())?;
    for (c, claim) in claims.iter().enumerate() {
        first_group.push(groups.len());
        let per_layer = if claim.stacked { layers } else { 1 };
        let members: Vec<Vec<usize>> = match &claim.claim.route {
            Route::Pool => vec![(0..claim.claim.atoms.len()).collect()],
            Route::Split(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    Part::Group { atoms } => Some(atoms.clone()),
                    Part::Virtual { .. } => None,
                })
                .collect(),
            Route::Instance(_) => Vec::new(),
        };
        for atoms in members {
            push(
                &mut groups,
                LeafGroup {
                    claim: c,
                    atoms,
                    values: vec_of(Fp2::ZERO, per_layer)?,
                },
            )?;
        }
    }
    let mut splits = leaf_splits.iter();
    let mut start = 0usize;
    while start < claims.len() {
        // The claims of one site are consecutive.
        let site = claims.get(start).ok_or(StageError::Internal)?;
        let end = claims
            .iter()
            .enumerate()
            .skip(start)
            .find(|(_, c)| c.site != site.site || c.input.is_none() != site.input.is_none())
            .map_or(claims.len(), |(i, _)| i);
        let per_layer = if site.stacked { layers } else { 1 };
        for layer in 0..per_layer {
            for c in start..end {
                let claim: &LeafClaim = claims.get(c).ok_or(StageError::Internal)?;
                let value = *leaf
                    .get(c)
                    .and_then(|l| l.values.get(layer))
                    .ok_or(StageError::Internal)?;
                let group = *first_group.get(c).ok_or(StageError::Internal)?;
                leaf_claim(
                    sources,
                    claim,
                    layer,
                    per_layer,
                    value,
                    &mut splits,
                    incoming,
                    &mut groups,
                    group,
                )?;
            }
        }
        start = end;
    }
    Ok(groups)
}

/// One leaf claim at one layer: its value goes to its instance or its pool group, or its split
/// values are checked and go to theirs.
#[allow(clippy::too_many_arguments)]
fn leaf_claim<'s>(
    sources: &Sources,
    claim: &LeafClaim,
    layer: usize,
    per_layer: usize,
    value: Fp2,
    splits: &mut impl Iterator<Item = &'s Fp2>,
    incoming: &mut [Incoming],
    groups: &mut [LeafGroup],
    first_group: usize,
) -> Result<(), StageError> {
    let set = |incoming: &mut [Incoming], instance: usize, v: Fp2| -> Result<(), StageError> {
        let slot = incoming.get_mut(instance).ok_or(StageError::Internal)?;
        if per_layer == 1 && !claim.stacked {
            *slot = Incoming::Value(v);
            return Ok(());
        }
        if *slot == Incoming::Missing {
            *slot = Incoming::PerLayer(vec_of(Fp2::ZERO, per_layer)?);
        }
        match slot {
            Incoming::PerLayer(values) => {
                *values.get_mut(layer).ok_or(StageError::Internal)? = v;
                Ok(())
            }
            _ => Err(StageError::Internal),
        }
    };
    match &claim.claim.route {
        Route::Instance(instance) => set(incoming, *instance, value),
        Route::Pool => {
            let group = groups.get_mut(first_group).ok_or(StageError::Internal)?;
            *group.values.get_mut(layer).ok_or(StageError::Internal)? = value;
            Ok(())
        }
        Route::Split(parts) => {
            let mut sum = Fp2::ZERO;
            let mut group = first_group;
            for part in parts {
                let sent = *splits.next().ok_or(StageError::Internal)?;
                match part {
                    Part::Virtual { atom, instance } => {
                        let atom = claim.claim.atoms.get(*atom).ok_or(StageError::Internal)?;
                        sum += coefficient(sources, &atom.coefficient, layer)? * sent;
                        set(incoming, *instance, sent)?;
                    }
                    Part::Group { .. } => {
                        sum += sent;
                        let entry = groups.get_mut(group).ok_or(StageError::Internal)?;
                        *entry.values.get_mut(layer).ok_or(StageError::Internal)? = sent;
                        group = group.checked_add(1).ok_or(StageError::Internal)?;
                    }
                }
            }
            if sum == value {
                Ok(())
            } else {
                Err(StageError::Split { stage: 0, round: 0 })
            }
        }
    }
}

/// A leaf pool claim, its per-layer values stacked at `ρ_1`.
fn leaf_pool_claim(
    schedule: &Schedule,
    sources: &Sources,
    eq_layers: &[Fp2],
    group: &LeafGroup,
) -> Result<PoolClaim, StageError> {
    let leaf = schedule
        .leaf()
        .get(group.claim)
        .ok_or(StageError::Internal)?;
    let value = if leaf.stacked {
        stacked(eq_layers, &group.values)?
    } else {
        *group.values.first().ok_or(StageError::Internal)?
    };
    let claim = &leaf.claim;
    pool_claim(
        schedule,
        sources,
        claim,
        &group.atoms,
        value,
        0,
        Some(eq_layers),
    )
}

/// A pool claim on the atoms `members` of `claim`, of value `value`. For a leaf claim of a
/// template site, `stacking` holds `eq(ρ_1, ℓ)` for each real layer, and each atom's coefficient
/// is its own at each layer.
fn pool_claim(
    schedule: &Schedule,
    sources: &Sources,
    claim: &Claim,
    members: &[usize],
    value: Fp2,
    stage: u16,
    stacking: Option<&[Fp2]>,
) -> Result<PoolClaim, StageError> {
    let layers = index(schedule.layers())?;
    let lead = members
        .first()
        .and_then(|&m| claim.atoms.get(m))
        .ok_or(StageError::Internal)?;
    let commitment = match lead.class {
        crate::schedule::Class::Committed(commitment) => commitment,
        crate::schedule::Class::Virtual { .. } => return Err(StageError::Internal),
    };
    let point = sources.resolve(&lead.point)?;
    let mut terms = Vec::new();
    for &m in members {
        let atom: &Atom = claim.atoms.get(m).ok_or(StageError::Internal)?;
        let weights: Option<Vec<Fp2>> = match (&atom.layers, stacking) {
            (Layers::None, _) => None,
            // A leaf claim's per-layer coefficients, stacked at ρ_1.
            (Layers::Point(_), Some(eq_layers)) => {
                let mut weights = vec_of(Fp2::ZERO, layers)?;
                for (l, w) in weights.iter_mut().enumerate() {
                    let eq = *eq_layers.get(l).ok_or(StageError::Internal)?;
                    *w = eq * coefficient(sources, &atom.coefficient, l)?;
                }
                Some(weights)
            }
            (Layers::Point(p), None) => {
                let rho = sources.resolve(p)?;
                let eq_layers = layer_eq(&rho, layers)?;
                let c = coefficient(sources, &atom.coefficient, 0)?;
                Some(eq_layers.iter().map(|&e| e * c).collect())
            }
            (Layers::One(l), _) => {
                let mut weights = vec_of(Fp2::ZERO, layers)?;
                *weights.get_mut(index(*l)?).ok_or(StageError::Internal)? =
                    coefficient(sources, &atom.coefficient, 0)?;
                Some(weights)
            }
        };
        match (atom.tensor, weights) {
            (TensorRef::Single(t), None) => push(
                &mut terms,
                PoolTerm {
                    tensor: PoolTensor::Single(t),
                    weights: vec![coefficient(sources, &atom.coefficient, 0)?],
                },
            )?,
            (TensorRef::Stacked(t), Some(weights)) => push(
                &mut terms,
                PoolTerm {
                    tensor: PoolTensor::Stacked(t),
                    weights,
                },
            )?,
            // The layer input's slices: c_0 at layer 0, X_last[ℓ − 1] at layer ℓ.
            (TensorRef::LayerInput, Some(weights)) => {
                let first = *weights.first().ok_or(StageError::Internal)?;
                push(
                    &mut terms,
                    PoolTerm {
                        tensor: PoolTensor::Single(schedule.first_checkpoint()),
                        weights: vec![first],
                    },
                )?;
                let mut shifted = vec_of(Fp2::ZERO, layers)?;
                for (l, w) in shifted.iter_mut().enumerate() {
                    if let Some(&next) = weights.get(l.saturating_add(1)) {
                        *w = next;
                    }
                }
                push(
                    &mut terms,
                    PoolTerm {
                        tensor: PoolTensor::Stacked(schedule.last_output()),
                        weights: shifted,
                    },
                )?;
            }
            _ => return Err(StageError::Internal),
        }
    }
    Ok(PoolClaim {
        stage,
        commitment,
        point,
        terms,
        value,
    })
}

/// Step 4.5 for one claim an evaluation leaves: its value goes to the next stage's instance or to
/// the pool, or its split values are checked.
#[allow(clippy::too_many_arguments)]
fn follow<'s>(
    schedule: &Schedule,
    sources: &Sources,
    claim: &Claim,
    value: Fp2,
    splits: &mut impl Iterator<Item = &'s Fp2>,
    next: &mut [Incoming],
    pool: &mut Vec<PoolClaim>,
    stage: u16,
    round: u32,
) -> Result<(), StageError> {
    match &claim.route {
        Route::Instance(instance) => {
            *next.get_mut(*instance).ok_or(StageError::Internal)? = Incoming::Value(value);
        }
        Route::Pool => {
            let members: Vec<usize> = (0..claim.atoms.len()).collect();
            push(
                pool,
                pool_claim(schedule, sources, claim, &members, value, stage, None)?,
            )?;
        }
        Route::Split(parts) => {
            let mut sum = Fp2::ZERO;
            let mut groups = Vec::new();
            for part in parts {
                let sent = *splits.next().ok_or(StageError::Internal)?;
                match part {
                    Part::Virtual { atom, instance } => {
                        let atom = claim.atoms.get(*atom).ok_or(StageError::Internal)?;
                        sum += coefficient(sources, &atom.coefficient, 0)? * sent;
                        *next.get_mut(*instance).ok_or(StageError::Internal)? =
                            Incoming::Value(sent);
                    }
                    Part::Group { atoms } => {
                        sum += sent;
                        push(&mut groups, (atoms, sent))?;
                    }
                }
            }
            if sum != value {
                return Err(StageError::Split { stage, round });
            }
            for (atoms, sent) in groups {
                push(
                    pool,
                    pool_claim(schedule, sources, claim, atoms, sent, stage, None)?,
                )?;
            }
        }
    }
    Ok(())
}

/// An instance's relation at its point `point`, from its evaluations, with the public factors of
/// the module documentation; `None` when the schedule names a factor that cannot be evaluated.
fn final_value(
    instance: &Instance,
    resolved: &Resolved,
    lambda: usize,
    rope: &Rope,
    point: &[Fp2],
    evals: &[Fp2],
) -> Option<Fp2> {
    let layer_vars = if instance.stacked { lambda } else { 0 };
    let own_len = point.len().checked_sub(layer_vars)?;
    let (own, layers) = point.split_at_checked(own_len)?;
    // w̃(r'_ℓ) = eq(ρ, r'_ℓ) for the claim's layer point ρ, over the whole cube.
    let layer_factor = match &resolved.layers {
        Some(rho) => sumcheck::eq(rho, layers)?,
        None if layers.is_empty() => Fp2::ONE,
        None => return None,
    };
    let r = resolved.claim.as_slice();
    match (&instance.relation, evals) {
        (Relation::MatMulWeight { .. }, &[w, x]) => Some(layer_factor * w * x),
        (
            Relation::MatMulAct {
                k_vars,
                m_vars,
                n_vars,
                lead_vars,
                ..
            },
            &[a, b],
        ) => {
            let heads_from = n_vars.checked_add(*m_vars)?;
            let r_h = r.get(heads_from..)?;
            let b_end = k_vars.checked_add(*lead_vars)?;
            let r_b = own.get(*k_vars..b_end)?;
            Some(sumcheck::eq(r_h, r_b)? * layer_factor * a * b)
        }
        (Relation::Elementwise { .. }, &[a, b]) => {
            Some(sumcheck::eq(r, own)? * layer_factor * a * b)
        }
        (Relation::SumSquares { col_vars }, &[x]) => {
            let rows = own.get(*col_vars..)?;
            Some(sumcheck::eq(r, rows)? * layer_factor * x * x)
        }
        (
            Relation::Rotate {
                positions,
                middle_vars,
            },
            &[x],
        ) => Some(rotate_kernel(rope, *positions, *middle_vars, r, own)? * layer_factor * x),
        (Relation::Booleanity, &[h]) => {
            Some(sumcheck::eq(&resolved.zero, point)? * h * (Fp2::ONE - h))
        }
        (Relation::Vocabulary { vocab, vocab_vars }, &[h]) => Some(
            sumcheck::eq(&resolved.zero, point)?
                * (Fp2::ONE - crate::selectors::lt(point.get(..*vocab_vars)?, u64::from(*vocab))?)
                * h,
        ),
        (
            Relation::OneHotCount {
                vocab_vars,
                positions,
            },
            &[h],
        ) => {
            let rows = point.get(*vocab_vars..)?;
            let zero_rows = resolved.zero.get(*vocab_vars..)?;
            let scale = Fp2::from(
                Fp::reduce(2)
                    .pow(u64::try_from(*vocab_vars).ok()?)
                    .inverse()?,
            );
            Some(
                sumcheck::eq(zero_rows, rows)?
                    * (h - crate::selectors::lt(rows, u64::from(*positions))? * scale),
            )
        }
        (Relation::OneHotIndex { vocab_vars }, &[h, token]) => {
            let rows = point.get(*vocab_vars..)?;
            let zero_rows = resolved.zero.get(*vocab_vars..)?;
            let (mut index, mut size) = (Fp2::ZERO, Fp2::ONE);
            for &r in point.get(..*vocab_vars)? {
                index += size * r;
                size += size;
            }
            Some(sumcheck::eq(zero_rows, rows)? * (index * h - token * size.inverse()?))
        }
        (Relation::Equality { constant }, &[y, x]) => {
            Some(sumcheck::eq(&resolved.zero, point)? * (y - Fp2::from(*constant) * x))
        }
        _ => None,
    }
}

/// A coefficient's value at layer `layer`: each term's constant there, or its one constant, times
/// its `LT` factors.
fn coefficient(
    sources: &Sources,
    coefficient: &Coefficient,
    layer: usize,
) -> Result<Fp2, StageError> {
    let mut sum = Fp2::ZERO;
    for term in coefficient.terms() {
        let constant = if term.constant.len() == 1 {
            term.constant.first()
        } else {
            term.constant.get(layer)
        }
        .ok_or(StageError::Internal)?;
        let mut product = Fp2::from(*constant);
        for (point, bound) in &term.lts {
            let at = sources.resolve(point)?;
            product *= crate::selectors::lt(&at, *bound).ok_or(StageError::Internal)?;
        }
        sum += product;
    }
    Ok(sum)
}

/// `Σ_ℓ eq(ρ_1, ℓ)·v_ℓ` over the real layers; a padded layer's value is 0.
fn stacked(eq_layers: &[Fp2], values: &[Fp2]) -> Result<Fp2, StageError> {
    if values.len() > eq_layers.len() {
        return Err(StageError::Internal);
    }
    Ok(eq_layers
        .iter()
        .zip(values)
        .fold(Fp2::ZERO, |sum, (&e, &v)| sum + e * v))
}

/// `eq(ρ, ℓ)` for each real layer `ℓ < layers`, `ρ` over `λ` coordinates.
fn layer_eq(rho: &[Fp2], layers: usize) -> Result<Vec<Fp2>, StageError> {
    let mut out = Vec::new();
    reserve(&mut out, layers)?;
    for l in 0..layers {
        let mut product = Fp2::ONE;
        for (i, &r) in rho.iter().enumerate() {
            let set = u32::try_from(i)
                .ok()
                .and_then(|i| l.checked_shr(i))
                .is_some_and(|s| s & 1 == 1);
            product *= if set { r } else { Fp2::ONE - r };
        }
        out.push(product);
    }
    Ok(out)
}

/// `count` challenges of stage `stage` under the labels `step(0)`, `step(1)`, ….
fn squeeze(
    transcript: &mut impl FiatShamir,
    stage: u16,
    count: usize,
    step: fn(usize) -> Step,
) -> Result<Vec<Fp2>, StageError> {
    let mut out = Vec::new();
    reserve(&mut out, count)?;
    for c in 0..count {
        let challenge =
            fiat_shamir::challenge(transcript, Prefix::Stage(u32::from(stage)), step(c))
                .map_err(|error| StageError::Sumcheck { stage, error })?;
        out.push(challenge);
    }
    Ok(out)
}

fn index(i: u32) -> Result<usize, StageError> {
    usize::try_from(i).map_err(|_| StageError::Internal)
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), StageError> {
    values
        .try_reserve(additional)
        .map_err(|_| StageError::Memory)
}

fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), StageError> {
    reserve(values, 1)?;
    values.push(value);
    Ok(())
}

fn vec_of<T: Clone>(value: T, len: usize) -> Result<Vec<T>, StageError> {
    let mut out = Vec::new();
    reserve(&mut out, len)?;
    out.resize(len, value);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use crate::fiat_shamir::tce;
    use tv_hash::transcript::TranscriptError;

    /// A golden model's schedule at `n` tokens, and its RoPE constants.
    fn schedule(name: &str, n: u32) -> (Schedule, Rope) {
        let file = tv_testkit::read_vectors("claims/claim-lists.json");
        let cases = file
            .get("cases")
            .and_then(tv_testkit::Json::as_array)
            .unwrap();
        let case = cases
            .iter()
            .find(|c| c.get("name").and_then(tv_testkit::Json::as_str) == Some(name))
            .unwrap();
        let bytes = tv_testkit::hex(case.get("hex").unwrap().as_str().unwrap());
        let claims =
            crate::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        let public = crate::validation::check_public(&claims).unwrap();
        let graph = crate::lowering::lower(&public, n).unwrap();
        let kind = graph.tables()[0].kind.clone();
        let rope = tv_tables::Table::generate(tv_tables::TableId(0), kind)
            .unwrap()
            .rope()
            .unwrap()
            .clone();
        (crate::schedule::derive(&graph).unwrap(), rope)
    }

    /// A proof of zeros of the right shapes: every claim, round polynomial, evaluation and split
    /// value is 0. Nonhomogeneous row selectors require the padded-row corner
    /// for the label-order control; exhaustion tests stop before the final check.
    fn zeros(schedule: &Schedule) -> (Vec<LeafValues>, Vec<Fp2>, Vec<StageProof>) {
        let leaf = schedule
            .leaf()
            .iter()
            .map(|c| LeafValues {
                point: vec![Fp2::ZERO; c.claim.vars],
                values: vec![
                    Fp2::ZERO;
                    if c.stacked {
                        schedule.layers() as usize
                    } else {
                        1
                    }
                ],
            })
            .collect();
        let stages = schedule
            .stages()
            .iter()
            .map(|s| StageProof {
                stage: s.index,
                sumcheck: SumcheckProof {
                    rounds: vec![vec![Fp2::ZERO; s.degree]; s.rounds],
                    final_evals: vec![Fp2::ZERO; s.instances.iter().map(|i| i.evals.len()).sum()],
                },
                splits: vec![Fp2::ZERO; s.splits],
            })
            .collect();
        (leaf, vec![Fp2::ZERO; schedule.leaf_splits()], stages)
    }

    /// RFC-0014 'Transcript schedule' row 6, as RFC-0037 'Messages and transcript' amends it:
    /// stage 0's split values; then each stage's layer point (stage 1), zero-check point, `γ`,
    /// rounds, evaluations and split values, in order.
    #[test]
    fn the_stages_run_the_transcript_schedule_in_order() {
        for name in ["tiny-2-servable", "tiny-4-servable"] {
            let (schedule, rope) = schedule(name, 5);
            let (leaf, leaf_splits, stages) = zeros(&schedule);
            // n=5 has public padding at row7. All-one coordinates select that row,
            // where the zero control satisfies the row selector. Only label order is tested.
            let mut stub = Stub::constant(Fp2::ONE);
            let verified =
                verify_with(&mut stub, &schedule, &rope, &leaf, &leaf_splits, &stages).unwrap();
            assert!(!verified.pool.is_empty());
            let mut expected = vec!["+stage/0/split".to_owned()];
            for stage in schedule.stages() {
                let k = stage.index;
                for c in 0..stage.layer_vars {
                    expected.push(format!("?stage/{k}/layer/{c}"));
                }
                for c in 0..stage.zero_vars {
                    expected.push(format!("?stage/{k}/zero/{c}"));
                }
                expected.push(format!("?stage/{k}/gamma"));
                for j in 0..stage.rounds {
                    expected.push(format!("+stage/{k}/round/{j}"));
                    expected.push(format!("?stage/{k}/r/{j}"));
                }
                expected.push(format!("+stage/{k}/evals"));
                expected.push(format!("+stage/{k}/split"));
            }
            assert_eq!(stub.script(), expected, "{name}");
            assert_eq!(schedule.stages()[0].layer_vars, schedule.layer_vars());
            assert!(schedule.stages()[0].zero_vars > 0);
            // The split values are absorbed as TCE Vec<Fp2>.
            assert_eq!(
                stub.ops[0],
                Op::Absorb("stage/0/split".into(), tce(&leaf_splits))
            );
        }
    }

    /// Each challenge the stages draw can find no value, as RFC-0004's bound on squeezes allows:
    /// the layer point's, the zero-check point's, `γ` and a round's are each the transcript's
    /// error, a fault of the verifier, and nothing follows the failed one.
    #[test]
    fn an_exhausted_challenge_stops_the_stages() {
        let (schedule, rope) = schedule("tiny-2-servable", 4);
        let (leaf, leaf_splits, stages) = zeros(&schedule);
        let stage1 = &schedule.stages()[0];
        let layer = 0;
        let zero = stage1.layer_vars;
        let gamma = zero + stage1.zero_vars;
        for fail_at in [layer, zero, gamma, gamma + 1] {
            let mut stub = Stub::failing_at(fail_at);
            let error =
                verify_with(&mut stub, &schedule, &rope, &leaf, &leaf_splits, &stages).unwrap_err();
            assert_eq!(
                error,
                StageError::Sumcheck {
                    stage: 1,
                    error: SumcheckError::Transcript(TranscriptError::Exhausted),
                },
                "challenge {fail_at}"
            );
            assert_eq!(error.rejection(), None);
            assert!(matches!(stub.ops.last(), Some(Op::Challenge(_))));
        }
    }

    /// RFC-0037: a reduction split's committed group already includes its coefficients.
    #[test]
    fn a_reduction_split_must_equal_its_incoming_claim() {
        let (schedule, _) = schedule("tiny-4-servable", 5);
        let (leaf, _, _) = zeros(&schedule);
        let claim = schedule
            .leaf()
            .iter()
            .map(|leaf| &leaf.claim)
            .find(|claim| matches!(&claim.route, Route::Split(parts) if parts.iter().any(|part| matches!(part, Part::Group { .. }))))
            .unwrap();
        let Route::Split(parts) = &claim.route else {
            unreachable!()
        };
        let group = parts
            .iter()
            .position(|part| matches!(part, Part::Group { .. }))
            .unwrap();
        let sources = Sources {
            leaf: leaf.into_iter().map(|leaf| leaf.point).collect(),
            layer: vec![Fp2::ZERO; schedule.layer_vars()],
            ..Sources::default()
        };
        for sent in [Fp2::ONE, -Fp2::ONE] {
            let mut splits = vec![Fp2::ZERO; parts.len()];
            splits[group] = sent;
            let check = |value| {
                let mut next = vec![Incoming::Missing; schedule.stages()[0].instances.len()];
                let mut pool = Vec::new();
                follow(
                    &schedule,
                    &sources,
                    claim,
                    value,
                    &mut splits.iter(),
                    &mut next,
                    &mut pool,
                    2,
                    9,
                )
                .map(|()| pool)
            };
            let pool = check(sent).expect("the committed group counts with weight one");
            assert_eq!(pool.len(), 1);
            assert_eq!(pool[0].value, sent);
            assert_eq!(
                check(Fp2::ZERO),
                Err(StageError::Split { stage: 2, round: 9 })
            );
        }
    }

    #[test]
    fn errors_say_whose_fault_they_are() {
        let rejections = [
            StageError::Split { stage: 0, round: 0 },
            StageError::Split { stage: 2, round: 9 },
            StageError::Sumcheck {
                stage: 1,
                error: SumcheckError::FinalMismatch { round: 12 },
            },
        ];
        for error in rejections {
            assert!(error.rejection().is_some(), "{error}");
            assert!(!error.is_malformed(), "{error}");
        }
        let malformed = [
            StageError::StageCount {
                expected: 2,
                found: 3,
            },
            StageError::StageId { index: 0, found: 2 },
            StageError::SplitCount {
                stage: 0,
                expected: 4,
                found: 3,
            },
        ];
        for error in malformed {
            assert!(
                error.is_malformed() && error.rejection().is_none(),
                "{error}"
            );
        }
        let verifier = [
            StageError::Leaf { claim: 3 },
            StageError::Memory,
            StageError::Internal,
            StageError::Sumcheck {
                stage: 1,
                error: SumcheckError::Relation { instance: 2 },
            },
        ];
        for error in verifier {
            assert!(
                !error.is_malformed() && error.rejection().is_none(),
                "{error}"
            );
        }
    }
}
