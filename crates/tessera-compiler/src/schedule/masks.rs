// SPDX-License-Identifier: Apache-2.0 OR MIT

//! R10's rank check on the compiled schedule (`docs/zk/simulator.md` §3 Lemma 2 and §9 R10, #488).
//!
//! In `tessera-protocol/2` every revealed evaluation of a masked tensor `V` is
//! `V'(p) = Ṽ(p) + Z(p)·R̃_V(p_col)` (RFC-0013 'Masked extensions'), so it reads `V`'s mask row
//! `R_V` through one linear functional. The values are jointly uniform when, on every mask row,
//! those functionals are linearly independent (Lemma 2). Lemma 2's `e²/|Fp2|` bound holds for
//! points of independent uniform coordinates; the compiled schedule shares coordinates between
//! points, so this module checks the condition on the schedule itself, the challenges as
//! indeterminates.
//!
//! # What a functional is
//!
//! A mask row is a row of `k = 2^{log2 k}` entries. A value at `p` on `V`'s row is the vector
//! `eq(p_col_c ‖ 0^{log2 k − c})` over the row's `log2 k` coordinates, `p_col_c` being `V`'s low
//! `c` column variables in layout order ([`ColumnRule`]):
//! - **Committed advice tensors.** The mask's columns are the data claim's column sub-point,
//!   which [`ModelAdviceLayout::claim`] gives (RFC-0011 rule 3, RFC-0047's order, RFC-0034's for
//!   attention tiles), and every coordinate past its inputs is 0: `c = min(n_P, log2 k)` of the
//!   placed tensor's variables (RFC-0013 'Committed tensors', #554). A stacked kind's layer
//!   variables, a tile's index bits and the placed tensor's variables past `log2 k` are row bits,
//!   never mask variables ([`CommittedColumns::DataColumn`]). [`CommittedColumns::TensorOrder`],
//!   the reading #554 rejected (the tensor's variables, then its layer variables, up to
//!   `log2 k`), stays as a control: a value that differs from another only in a layer or tile
//!   coordinate is refused under the rule and passes under the control.
//! - **Virtual tensors** have no placement: their variables are their relation's, lowest first,
//!   then the layer variables.
//! - **A fold** (RFC-0013 'Virtual tensors', #538) reveals `u = R̃_row(r'')`, the whole row's
//!   extension at `r''`, the stage's last `log2 k` challenges: the functional `eq(r'')`.
//! - **Weights.** A plain matrix's claim at `p ‖ ρ` has column part its first `log2 k_T`
//!   coordinates (RFC-0012 'Layout', [`WeightLayout::matrix`]); a `ChannelParams` claim's is the
//!   first `log2 k_CP` coordinates of its point on `Ã_CP` ([`WeightLayout::channel_point`],
//!   RFC-0052, RFC-0065 'The masks').
//!
//! # Which row a functional reads ([`MaskRow`])
//!
//! - A committed advice tensor in its own variables: a [`Kind::Tensor`], a stacked kind's layer
//!   read at one layer when `λ ≥ 1` (RFC-0069), or a stacked kind of one layer (`λ = 0`). Two
//!   kinds that name one placed tensor this way share its row: the epilogue's read of `c_L` and
//!   the prologue's `c_0` are such tensors. A stacked kind with `λ ≥ 1` has a row of its own,
//!   even where one of its layers is another kind's tensor (`Kind::LayerInput`'s layer 0 is
//!   `c_0`, RFC-0069 'Motivation'). Its `Z` runs over its `λ` layer variables too, but its mask
//!   does not: the row's columns are its data claim's column sub-point (#554).
//! - A virtual kind's row, which a linear node's output never has: the schedule substitutes it by
//!   its atoms (`docs/zk/simulator.md` 'Virtual tensors').
//! - A weight matrix's `R_W` (RFC-0013); each `ChannelParams` claim's `R_i` (RFC-0065).
//!
//! # Coordinates
//!
//! - **Rounds.** A reduction stage with a fold has `n = max(n_max, log2 k)` rounds in
//!   `tessera-protocol/2` (RFC-0013 'The stage's rounds'), every instance front-padded (RFC-0010):
//!   compiled round `j` of a stage of `n_max` rounds is round `j + n − n_max`, and rounds below
//!   `n − n_max` are fresh. The schedule's points list the lowest variable first, and an instance
//!   of `n_i` variables takes the stage's last `n_i` challenges, its lowest variable the earliest
//!   (`relations.rs`, `challenges = rounds − vars..rounds`).
//! - **The leaf point.** Every leaf claim lies at `P[0..n]` of one point `P` shared by every site
//!   (RFC-0045 'The leaf point'; RFC-0068 'P in `tessera-protocol/2`'), as the prover sets
//!   [`Coord::Leaf`]: one indeterminate per index, whatever the site.
//! - **The layer point** is RFC-0068's `ρ'_y`, the zero-check point stage 1's.
//!
//! # The check ([`check`])
//!
//! Values with the same row and full point, layer weights included, are one value, a public
//! relation, and merge; so do identical folds, never one with a value. A committed stacked value
//! whose per-layer constants differ between layers weighs its layers by `eq(ρ, ℓ)·c_ℓ` (RFC-0037
//! 'Claims and atoms'), so it merges only with values of its own coefficient
//! ([`Functional::scale`]). A functional whose length is not its row's width is refused
//! ([`Finding::Width`]), never sketched. Each indeterminate takes a deterministic pseudo-random [`GoldilocksExt2`] value, `½`,
//! `0` and `1` themselves. Each functional is sketched against `m + 8` random product vectors,
//! `⟨eq(p), ⊗_j (a_j, b_j)⟩ = Π_j ((1 − p_j)·a_j + p_j·b_j)`, and the sketches are eliminated over
//! `Fp2`. Full rank proves the functionals generically independent; a deficiency is reported as
//! the dependent set ([`Finding::Dependent`]), and is a true dependency but with probability
//! about `m·log2 k / |Fp2|`.
//!
//! **The pair `β_T`** (RFC-0065 'Zero knowledge'). For each weight matrix, the blinding claims
//! `q_i = ⟨β_T, eq(column_point_i)⟩` and every stage-revealed value of a type tensor at another
//! point are functionals of `β_T`. A value with a secret term must lie outside the span of the
//! others ([`Finding::Spanned`]); one with `S ≡ 0`, decided symbolically (the claim's row point
//! and `Z(y) ≡ Z(x_i)`), joins the span and is never refused. The compiled schedule leaves one
//! claim per type tensor (RFC-0052, INV-SCH-5); a further point on one would be such a value. No
//! compiled schedule has one today, so on them the pass sees blinding claims alone: it guards
//! future schedules, and the synthetic tests exercise it.
//!
//! **Tensors of fewer than two variables** cannot carry RFC-0013's mask, whose `Z` spares the
//! lowest variable ([`Finding::Unmaskable`]). `tessera-protocol/2` pads the token axis to four rows
//! at least, laid out as a padded axis, so a sequence of `n` positions is checked at
//! [`protocol2_length`]`(n)`; a tensor that still has fewer than two variables there is refused.
//!
//! # What this check does not cover
//!
//! It reads the compiled [`StageSchedule`], which does not hold every revealed point of
//! `tessera-protocol/2`. RFC-0013 'Committed tensors' 'Hiding' states R10's gap as one list, these
//! ten items in this order, and makes it block #154 and #159; each is to be added as its
//! geometry is compiled:
//! 1. the leaf stage's values of gadget outputs and auxiliary columns at their leaf read points
//!    (RFC-0068 step 5's `v_{s,t}` on value forms); the schedule's leaf sites are gadget inputs
//!    only;
//! 2. the tensors a site reads over some of its block's axes (`π_t`, RFC-0045 'The split');
//! 3. RFC-0068's column masks and their values at the pair points and `ρ'_{w_i}`, and the layer
//!    sumcheck's own folds;
//! 4. `SoftmaxRow`'s reciprocal at its row point, its maximum, `Exp` at its score-leaf point and
//!    the `Exp` row sum at key coordinates `½` (RFC-0037 'The claim pool', RFC-0030), which the
//!    prover assembles outside the schedule;
//! 5. the native relations' raw evaluations, on graph tensors and on auxiliary columns at the
//!    natives' own points: the root's bytes and input limbs in each coefficient product,
//!    `SoftmaxRow`'s maximum in its key, forest and padding relations, and the greedy certificate's
//!    reads of the one-hot tokens at their carry points, whose `tessera-protocol/2` form RFC-0053
//!    disclaims;
//! 6. the multiplicities' claims;
//! 7. the tree levels' masks, the identity sumcheck's root levels (RFC-0032) and the `SoftmaxRow`
//!    leaf mask;
//! 8. the masked value of a committed group whose layer weights carry per-layer constants
//!    (RFC-0037 'Claims and atoms'), which no RFC specifies in `tessera-protocol/2` yet (#556): on
//!    the advice the check keeps such a value apart from every other ([`Functional::scale`]) and
//!    reads it through the stacked kind's mask row; on a weight matrix it refuses it
//!    ([`Finding::ScaledWeight`]), since a matrix's pair would drop it as its first claim's value;
//! 9. batches of several requests (RFC-0071, `R ≥ 2`): a claim list is checked for one request;
//! 10. the anchor's circuits (RFC-0066, RFC-0072), whose schedules this check does not lower:
//!     their committed bits and bytes, once they join a batch's graph (#146, #526).
//!
//! Outside R10 altogether, other lemmas cover the masking polynomials' `G(r)` and RFC-0070's
//! batching (`M`, `S`, `r*`; Lemma 4a), and the combined rows, the advice groups' blinding rows
//! and the proximity rows (Lemmas 5 to 7).

use std::collections::HashMap;
use std::fmt;

use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_ir::{ClaimList, CommitmentClass, TensorId};
use tessera_pcs::{WeightLayout, WeightMatrix, WeightShapes};

use std::sync::Arc;

use super::{
    Atom, Coord, Factor, Kind, Layers, NodeRef, Request, Source, StageSchedule,
    derive_composed_schedule, model_natives,
};
use crate::{
    AdviceSource, CompileError, Lowering, ModelAdviceLayout, PublicModel, Selection, analyze,
    lower, select,
};

type Fp2 = GoldilocksExt2;

/// The seed of the gate's pseudo-random assignment and sketches.
pub const SEED: u64 = 0x5231_305f_6d61_736b;

/// The shortest length the check lowers at: three positions, whose token axis pads to four rows
/// and is laid out as a padded axis, as `tessera-protocol/2` lays out every shorter sequence's.
pub const MIN_POSITIONS: u32 = 3;

/// The test vectors a row's sketch takes beyond its functionals' count.
const EXTRA_TESTS: usize = 8;

fn internal(rule: &'static str) -> CompileError {
    CompileError::Schedule { node: None, rule }
}

/// How many of a tensor's variables its mask row takes, `c` (RFC-0013 'Masked extensions').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColumnRule {
    /// `c = min(n_V, log2 k)` for every tensor: RFC-0013 before #488 amended it. It lets a fold
    /// read a virtual tensor's row at the point of a value, and serves as the regression control.
    Row,
    /// RFC-0013 as amended by #488: `c = min(n_V, log2 k − 1)` for a virtual tensor whose mask a
    /// reduction stage folds; for a committed tensor, its data column's `min(n_P, log2 k)` under
    /// [`CommittedColumns::RULE`] (#554).
    BelowRow,
}

/// Which of a committed advice tensor's variables its mask's column takes: #554's two readings,
/// RFC-0013 'Committed tensors', which #554 made the rule, and RFC-0068's `n_V`, which it
/// rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommittedColumns {
    /// The rejected reading, kept as a control: the tensor's low `c = min(n_V, log2 k)` variables
    /// in layout order, the data claim's column inputs, then the variables the column sub-point
    /// leaves out (a tile's index bits), then the layer variables.
    TensorOrder,
    /// RFC-0013's rule (#554): the data claim's own column inputs alone, so the mask's column
    /// part is the data's, `eq(p_col ‖ 0)`, and every coordinate past its inputs is 0.
    DataColumn,
}

impl CommittedColumns {
    /// The reading RFC-0013 'Committed tensors' specifies, which R10's gate checks (#554).
    pub const RULE: Self = Self::DataColumn;
}

/// An indeterminate: a challenge of `tessera-protocol/2` that the check treats as a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    /// Coordinate `i` of the leaf point `P`, which every site shares (RFC-0045 'The leaf point').
    Leaf(u32),
    /// Coordinate `i` of stage 1's layer point, RFC-0068's `ρ'_y`.
    Layer(u32),
    /// Coordinate `i` of stage 1's zero-check point.
    Zero(u32),
    /// Round `round` of stage `stage`, in `tessera-protocol/2`'s rounds (RFC-0013 'The stage's
    /// rounds').
    Round {
        /// The stage.
        stage: u16,
        /// The round, from 0.
        round: u32,
    },
}

/// A coordinate of a functional's point: an indeterminate or a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Src {
    /// An indeterminate.
    Var(Var),
    /// The constant 0: a padding bit, or a column position above a tensor's `c`.
    Zero,
    /// The constant 1: a type or base bit.
    One,
    /// The constant `½` of a sum over an axis (RFC-0010 'Relations').
    Half,
}

/// The mask row a functional reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaskRow {
    /// A committed advice tensor in its own variables: a [`Kind::Tensor`], a stacked kind's layer
    /// read at one layer (RFC-0069), or the one layer of a stacked kind when `λ = 0`.
    Tensor(TensorId),
    /// A stacked committed advice kind, over its `λ ≥ 1` layer variables too.
    Stacked(Kind),
    /// A virtual kind, the output of a sumcheck relation, whose fold its proving stage runs.
    Virtual(Kind),
    /// A weight matrix's `R_W` (RFC-0013): `Embed` to `LmHead`.
    Weight(WeightMatrix),
    /// The `ChannelParams` claim on a type tensor, its own `R_i` (RFC-0065 'The masks').
    Channel(Kind),
}

/// Where a functional comes from in the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Atom `atom` of pool claim `claim`, a part of a claim of stage `stage` (RFC-0037 'The claim
    /// pool').
    Pool {
        /// The stage that leaves it.
        stage: u16,
        /// The pool claim, in [`StageSchedule::pool`].
        claim: usize,
        /// The atom.
        atom: usize,
    },
    /// A virtual atom of claim `claim` of stage `stage`, which instance `instance` of the next
    /// stage proves.
    Virtual {
        /// The stage that reveals the value.
        stage: u16,
        /// The claim, in that stage's claims.
        claim: usize,
        /// The instance of stage `stage + 1` that proves it.
        instance: usize,
    },
    /// The fold of instance `instance` of stage `stage` (RFC-0013 'Virtual tensors').
    Fold {
        /// The stage.
        stage: u16,
        /// The instance.
        instance: usize,
    },
}

/// What a functional of a mask row reveals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// A masked value `V'(p)`: its mask term `Z(p)·⟨R_V, eq(p_col)⟩`.
    Value,
    /// A fold's final value `u = ⟨R_V, eq(r'')⟩` (RFC-0013 'Virtual tensors').
    Fold,
}

/// A linear functional `⟨R, eq(point)⟩` of one mask row.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Functional {
    /// What it reveals.
    pub role: Role,
    /// The point, over the row's coordinates, the lowest first.
    pub point: Vec<Src>,
    /// The value's identity: its full point, layer coordinates included, or a fold's `r''`.
    /// Functionals of one role, one key and one scale are one value.
    pub key: Vec<Src>,
    /// The coefficient of a committed stacked value whose per-layer constants differ between
    /// layers, `Factor::Constant` of a template position that is not uniform: its layers weigh
    /// `eq(ρ, ℓ)·c_ℓ` (RFC-0037 'Claims and atoms'), so it is another value than an unscaled one
    /// at its point. `None` for every other functional.
    pub scale: Option<Vec<Vec<Factor>>>,
    /// Where it comes from: several once merged.
    pub origins: Vec<Origin>,
}

/// One mask row and the functionals the schedule reveals of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSystem {
    /// The row.
    pub row: MaskRow,
    /// The tensor that stands for it: layer 0's for a stacked kind, the weight or type tensor's.
    pub tensor: Option<TensorId>,
    /// `n_V`, the masked tensor's variables, layer variables included.
    pub vars: u32,
    /// The row's coordinates, `log2 k`.
    pub width: u32,
    /// The functionals, in schedule order.
    pub functionals: Vec<Functional>,
}

/// What a functional of a weight matrix's pair `β_T` reveals (RFC-0065 'Zero knowledge').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PairRole {
    /// A claim's blinding claim `q_i = ⟨β_T, eq(column_point_i)⟩`.
    Blinding,
    /// A stage-revealed value of a type tensor at another point `y`, with its functional
    /// `⟨β_T, e_y⟩`; `secret` unless `S_{i,y} ≡ 0`.
    Value {
        /// Whether the value carries a secret term, `S_{i,y} ≢ 0`.
        secret: bool,
    },
}

/// A functional of a matrix's pair `β_T`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairFunctional {
    /// What it reveals.
    pub role: PairRole,
    /// The column point, `log2 k_T` coordinates.
    pub point: Vec<Src>,
    /// The value's full point on the type tensor, layer coordinates included.
    pub key: Vec<Src>,
    /// The type tensor whose claim or value it is.
    pub tensor: Option<TensorId>,
    /// Where it comes from.
    pub origins: Vec<Origin>,
}

/// A weight matrix's pair `β_T` and its functionals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairSystem {
    /// The matrix.
    pub matrix: WeightMatrix,
    /// `log2 k_T`.
    pub width: u32,
    /// The functionals, in pool order.
    pub functionals: Vec<PairFunctional>,
}

/// Every mask row's functionals and every matrix pair's, as the schedule reveals them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaskSystem {
    /// The mask rows, in the order the schedule first reaches them.
    pub rows: Vec<RowSystem>,
    /// The weight matrices' pairs, in the order the pool first reaches them.
    pub pairs: Vec<PairSystem>,
    /// The weight claims whose coefficient is not a public scalar uniform over the layers, which
    /// the check refuses ([`Finding::ScaledWeight`]) and leaves out of the rows and pairs.
    pub scaled_weights: Vec<ScaledWeight>,
}

/// A weight claim whose coefficient carries a per-layer constant that differs between layers
/// (`Factor::Constant` of a template position that is not uniform): its layers weigh
/// `eq(ρ, ℓ)·c_ℓ`, a value no RFC masks in `tessera-protocol/2` (#556).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaledWeight {
    /// The matrix.
    pub matrix: WeightMatrix,
    /// The weight or type tensor, layer 0's.
    pub tensor: Option<TensorId>,
    /// The template position whose constants differ between layers.
    pub position: u32,
    /// The claim.
    pub origin: Origin,
}

/// A refusal of R10.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    /// Functionals of one mask row that are linearly dependent: the last of them lies in the span
    /// of the others (Lemma 2).
    Dependent {
        /// The row.
        row: MaskRow,
        /// The tensor that stands for it.
        tensor: Option<TensorId>,
        /// The dependent functionals, the one in the span of the others last.
        functionals: Vec<Functional>,
    },
    /// A value with a secret term whose functional of `β_T` lies in the span of the matrix's
    /// other functionals (RFC-0065 'Zero knowledge').
    Spanned {
        /// The matrix.
        matrix: WeightMatrix,
        /// The value.
        value: PairFunctional,
        /// The functionals whose span holds it.
        span: Vec<PairFunctional>,
    },
    /// A weight claim whose coefficient is not a public scalar uniform over the layers (#556).
    ScaledWeight(ScaledWeight),
    /// A functional whose point has another length than its row's width: a malformed system,
    /// which the check refuses rather than sketch a truncated functional.
    Width {
        /// The row, or `None` for a matrix's pair.
        row: Option<MaskRow>,
        /// The matrix, for a pair.
        matrix: Option<WeightMatrix>,
        /// The row's width.
        width: u32,
        /// The functional's coordinates.
        found: usize,
    },
    /// A revealed tensor of fewer than two variables, which RFC-0013's mask cannot hide: its `Z`
    /// spares the lowest variable, so it would not vanish on the hypercube.
    Unmaskable {
        /// The row.
        row: MaskRow,
        /// The tensor that stands for it.
        tensor: Option<TensorId>,
        /// Its variables.
        vars: u32,
        /// Where it is revealed.
        origins: Vec<Origin>,
    },
}

impl fmt::Display for MaskRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(t) => write!(f, "advice tensor t{}", t.0),
            Self::Stacked(kind) => write!(f, "stacked advice kind {kind:?}"),
            Self::Virtual(kind) => write!(f, "virtual kind {kind:?}"),
            Self::Weight(matrix) => write!(f, "weight matrix {matrix:?}"),
            Self::Channel(kind) => write!(f, "ChannelParams claim on {kind:?}"),
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Pool { stage, claim, atom } => {
                write!(f, "pool claim {claim} atom {atom} (stage {stage})")
            }
            Self::Virtual {
                stage,
                claim,
                instance,
            } => write!(
                f,
                "stage {stage} claim {claim} (stage {} instance {instance})",
                stage + 1
            ),
            Self::Fold { stage, instance } => {
                write!(f, "fold of stage {stage} instance {instance}")
            }
        }
    }
}

fn point(p: &[Src]) -> String {
    let coords: Vec<String> = p
        .iter()
        .map(|s| match s {
            Src::Var(Var::Leaf(i)) => format!("P{i}"),
            Src::Var(Var::Layer(i)) => format!("Y{i}"),
            Src::Var(Var::Zero(i)) => format!("Z{i}"),
            Src::Var(Var::Round { stage, round }) => format!("R{stage}.{round}"),
            Src::Zero => "0".to_owned(),
            Src::One => "1".to_owned(),
            Src::Half => "½".to_owned(),
        })
        .collect();
    format!("[{}]", coords.join(" "))
}

fn origins(list: &[Origin]) -> String {
    let parts: Vec<String> = list.iter().map(ToString::to_string).collect();
    parts.join(", ")
}

fn tensor(t: Option<TensorId>) -> String {
    t.map_or_else(String::new, |t| format!(" (t{})", t.0))
}

/// One line per finding, naming the row, its tensor, the functionals and where the schedule
/// reveals them, with the rule broken.
impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dependent {
                row,
                tensor: t,
                functionals,
            } => {
                write!(
                    f,
                    "{row}{}: {} revealed functionals are linearly dependent \
                     (R10, Lemma 2, RFC-0013 'Masked extensions'):",
                    tensor(*t),
                    functionals.len()
                )?;
                for functional in functionals {
                    let role = match functional.role {
                        Role::Value => "value",
                        Role::Fold => "fold",
                    };
                    write!(
                        f,
                        " {role} at {} from {};",
                        point(&functional.point),
                        origins(&functional.origins)
                    )?;
                }
                Ok(())
            }
            Self::Spanned {
                matrix,
                value,
                span,
            } => {
                write!(
                    f,
                    "weight matrix {matrix:?}: a value with a secret term on{} at {} from {} lies \
                     in the span of {} other functionals of beta_T (R10, RFC-0065 'Zero \
                     knowledge'):",
                    tensor(value.tensor),
                    point(&value.point),
                    origins(&value.origins),
                    span.len()
                )?;
                for other in span {
                    write!(
                        f,
                        " {} from {};",
                        point(&other.point),
                        origins(&other.origins)
                    )?;
                }
                Ok(())
            }
            Self::ScaledWeight(ScaledWeight {
                matrix,
                tensor: t,
                position,
                origin,
            }) => write!(
                f,
                "weight matrix {matrix:?}{}: {origin} weighs its layers by the constants of \
                 position {position}, which differ between layers; no RFC masks such a weight \
                 value in tessera-protocol/2 (R10, #556)",
                tensor(*t)
            ),
            Self::Width {
                row,
                matrix,
                width,
                found,
            } => {
                match (row, matrix) {
                    (Some(row), _) => write!(f, "{row}")?,
                    (None, Some(matrix)) => write!(f, "weight matrix {matrix:?}'s pair beta_T")?,
                    (None, None) => f.write_str("a row")?,
                }
                write!(
                    f,
                    ": a functional of {found} coordinates on a row of {width}, a malformed \
                     system (R10)"
                )
            }
            Self::Unmaskable {
                row,
                tensor: t,
                vars,
                origins: list,
            } => write!(
                f,
                "{row}{}: revealed with {vars} variable(s), below the two RFC-0013's mask needs \
                 (its Z spares the lowest variable), at {}",
                tensor(*t),
                origins(list)
            ),
        }
    }
}

/// `tessera-protocol/2`'s rounds of each stage: its compiled rounds, shifted by the leading rounds
/// a fold adds (RFC-0013 'The stage's rounds').
struct Rounds {
    /// Per stage: (protocol/2 rounds, shift of the compiled rounds).
    stages: Vec<(u32, u32)>,
}

impl Rounds {
    fn new(schedule: &StageSchedule, log2_k: u32) -> Self {
        let stages = schedule
            .stages
            .iter()
            .map(|stage| {
                let folds = stage
                    .instances
                    .iter()
                    .any(|i| matches!(i.source, Source::Part { .. }));
                let rounds = if folds {
                    stage.rounds.max(log2_k)
                } else {
                    stage.rounds
                };
                (rounds, rounds - stage.rounds)
            })
            .collect();
        Self { stages }
    }

    fn src(&self, coord: Coord) -> Result<Src, CompileError> {
        Ok(match coord {
            Coord::Leaf { index, .. } => Src::Var(Var::Leaf(index)),
            Coord::Layer(i) => Src::Var(Var::Layer(i)),
            Coord::Zero(i) => Src::Var(Var::Zero(i)),
            Coord::Round { stage, round } => {
                let &(rounds, shift) = self
                    .stages
                    .get(usize::from(stage))
                    .ok_or(internal("a round coordinate names a stage of the schedule"))?;
                let round = round + shift;
                if round >= rounds {
                    return Err(internal("a round coordinate names a round of its stage"));
                }
                Src::Var(Var::Round { stage, round })
            }
            Coord::Half => Src::Half,
        })
    }

    fn point(&self, coords: &[Coord]) -> Result<Vec<Src>, CompileError> {
        coords.iter().map(|&c| self.src(c)).collect()
    }
}

/// A position of a layout function's output: one of its inputs, or a constant bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Input(usize),
    Zero,
    One,
}

/// Distinct probe values, none 0 or 1, for a layout function's `n` inputs.
fn probes(n: usize) -> Vec<Fp2> {
    (0..n as u64)
        .map(|j| Fp2::new(Goldilocks::from_u64_reduced(j + 2), Goldilocks::ONE))
        .collect()
}

/// Reads a layout function's output on [`probes`] back as slots. The layout code moves
/// coordinates and inserts constant bits only (RFC-0011 rule 3, RFC-0052); anything else is
/// refused.
fn slots(out: &[Fp2], n: usize) -> Result<Vec<Slot>, CompileError> {
    let probe = probes(n);
    out.iter()
        .map(|v| {
            if *v == Fp2::ZERO {
                Ok(Slot::Zero)
            } else if *v == Fp2::ONE {
                Ok(Slot::One)
            } else {
                probe
                    .iter()
                    .position(|p| p == v)
                    .map(Slot::Input)
                    .ok_or(internal(
                        "a layout point moves its claim's coordinates and adds constant bits",
                    ))
            }
        })
        .collect()
}

fn apply(slots: &[Slot], input: &[Src]) -> Vec<Src> {
    slots
        .iter()
        .map(|slot| match *slot {
            Slot::Input(j) => input[j],
            Slot::Zero => Src::Zero,
            Slot::One => Src::One,
        })
        .collect()
}

/// `Z(x) = Π_{j ≥ 1} x_j(1 − x_j)` up to its constant, as a multiset of the coordinates it
/// multiplies, or `None` when a Boolean coordinate makes it vanish.
fn vanishing(x: &[Src]) -> Option<Vec<Src>> {
    let mut factors: Vec<Src> = x.iter().skip(1).copied().collect();
    if factors.iter().any(|s| matches!(s, Src::Zero | Src::One)) {
        return None;
    }
    factors.sort_unstable();
    Some(factors)
}

/// The claim a matrix's row holds first, against which later values are read (RFC-0065).
struct FirstClaim {
    key: Vec<Src>,
    row_point: Vec<Src>,
    z: Option<Vec<Src>>,
}

/// Builds a [`MaskSystem`].
struct Builder<'a> {
    lowering: &'a Lowering,
    schedule: &'a StageSchedule,
    advice: &'a ModelAdviceLayout<'a>,
    weights: &'a WeightLayout,
    rule: ColumnRule,
    committed: CommittedColumns,
    log2_k: u32,
    rounds: Rounds,
    system: MaskSystem,
    rows: HashMap<MaskRow, usize>,
    pairs: HashMap<WeightMatrix, usize>,
    claims: HashMap<MaskRow, FirstClaim>,
    columns: HashMap<TensorId, Vec<usize>>,
}

impl<'a> Builder<'a> {
    fn row(&mut self, row: MaskRow, tensor: Option<TensorId>, vars: u32, width: u32) -> usize {
        let rows = &mut self.system.rows;
        *self.rows.entry(row).or_insert_with(|| {
            rows.push(RowSystem {
                row,
                tensor,
                vars,
                width,
                functionals: Vec::new(),
            });
            rows.len() - 1
        })
    }

    fn push(
        &mut self,
        row: MaskRow,
        tensor: Option<TensorId>,
        vars: u32,
        width: u32,
        functional: Functional,
    ) -> Result<(), CompileError> {
        let index = self.row(row, tensor, vars, width);
        let system = &mut self.system.rows[index];
        if system.vars != vars || system.width != width || functional.point.len() != width as usize
        {
            return Err(internal(
                "a mask row's functionals have its variables and its width",
            ));
        }
        system.functionals.push(functional);
        Ok(())
    }

    fn pair(&mut self, matrix: WeightMatrix, width: u32, functional: PairFunctional) {
        let pairs = &mut self.system.pairs;
        let index = *self.pairs.entry(matrix).or_insert_with(|| {
            pairs.push(PairSystem {
                matrix,
                width,
                functionals: Vec::new(),
            });
            pairs.len() - 1
        });
        pairs[index].functionals.push(functional);
    }

    fn tensor(&self, kind: Kind, layer: u32) -> Result<TensorId, CompileError> {
        self.schedule
            .layers
            .tensor(&self.lowering.graph, kind, layer)
            .ok_or(internal("a pool atom's kind has its layer's tensor"))
    }

    /// The natural positions of `tensor`'s data column sub-point, in layout order: every image of
    /// a claim on it has one ([`ModelAdviceLayout::claim`]), and its padding bits are zero.
    fn column_order(&mut self, tensor: TensorId, vars: usize) -> Result<Vec<usize>, CompileError> {
        if let Some(order) = self.columns.get(&tensor) {
            return Ok(order.clone());
        }
        let probe = probes(vars);
        let log2_k = self.log2_k as usize;
        let mut order: Option<Vec<Slot>> = None;
        for image in self.advice.claim(AdviceSource::Tensor(tensor), &probe)? {
            let image = image?;
            let column = image
                .point
                .get(..log2_k)
                .ok_or(internal("an advice point has a column sub-point"))?;
            let found = slots(column, vars)?;
            match &order {
                None => order = Some(found),
                Some(first) if *first == found => {}
                Some(_) => {
                    return Err(internal(
                        "every image of an advice claim has one column sub-point",
                    ));
                }
            }
        }
        let order = order.ok_or(internal("an advice claim has an image"))?;
        let mut positions = Vec::new();
        let mut padding = false;
        for slot in order {
            match slot {
                Slot::Input(j) if !padding && !positions.contains(&j) => positions.push(j),
                Slot::Zero => padding = true,
                _ => {
                    return Err(internal(
                        "a column sub-point holds a tensor's coordinates, then zero padding",
                    ));
                }
            }
        }
        self.columns.insert(tensor, positions.clone());
        Ok(positions)
    }

    /// `c`, the column variables a tensor of `vars` variables gives its mask row.
    fn columns_of(&self, vars: u32, virtual_folded: bool) -> u32 {
        match (self.rule, virtual_folded) {
            (ColumnRule::BelowRow, true) => vars.min(self.log2_k.saturating_sub(1)),
            _ => vars.min(self.log2_k),
        }
    }

    /// `eq(p_col_c ‖ 0^{log2 k − c})` for `ordered`, a tensor's variables in layout order.
    fn row_point(&self, ordered: &[Src], c: u32) -> Vec<Src> {
        let mut p: Vec<Src> = ordered[..c as usize].to_vec();
        p.resize(self.log2_k as usize, Src::Zero);
        p
    }

    fn advice_atom(&mut self, atom: &Atom, origin: Origin) -> Result<(), CompileError> {
        let lambda = self.schedule.layers.vars;
        let r = self.rounds.point(&atom.point)?;
        let mut scale = None;
        let (row, data, rho) = match (atom.kind, &atom.layers) {
            (Kind::Tensor(t), Layers::None) => (MaskRow::Tensor(t), t, Vec::new()),
            (kind, &Layers::One(layer)) if !matches!(kind, Kind::Tensor(_)) => {
                let t = self.tensor(kind, layer)?;
                (MaskRow::Tensor(t), t, Vec::new())
            }
            (kind, Layers::Eq(rho)) if !matches!(kind, Kind::Tensor(_)) => {
                let t = self.tensor(kind, 0)?;
                if lambda == 0 {
                    if !rho.is_empty() {
                        return Err(internal("a kind of one layer has no layer coordinate"));
                    }
                    (MaskRow::Tensor(t), t, Vec::new())
                } else {
                    // Every layer's tensor has layer 0's shape, so one column order serves; the
                    // last layer's is checked against it.
                    let last = self.tensor(kind, self.schedule.layers.count - 1)?;
                    if self.column_order(last, r.len())? != self.column_order(t, r.len())? {
                        return Err(internal(
                            "a stacked kind's layers share one column sub-point",
                        ));
                    }
                    let scaled = atom.coefficient.iter().flatten().any(|factor| {
                        matches!(factor, Factor::Constant(NodeRef::Position(p))
                            if !self.schedule.layers.uniform(*p))
                    });
                    if scaled {
                        scale = Some(atom.coefficient.clone());
                    }
                    (MaskRow::Stacked(kind), t, self.rounds.point(rho)?)
                }
            }
            _ => {
                return Err(internal(
                    "an advice pool atom weighs its layers by eq or one",
                ));
            }
        };
        let order = self.column_order(data, r.len())?;
        let mut ordered: Vec<Src> = order.iter().map(|&j| r[j]).collect();
        ordered.extend((0..r.len()).filter(|j| !order.contains(j)).map(|j| r[j]));
        ordered.extend(&rho);
        let vars = u32::try_from(ordered.len()).unwrap_or(u32::MAX);
        let c = match self.committed {
            CommittedColumns::TensorOrder => self.columns_of(vars, false),
            CommittedColumns::DataColumn => {
                self.columns_of(u32::try_from(order.len()).unwrap_or(u32::MAX), false)
            }
        };
        let mut key = r;
        key.extend(&rho);
        let functional = Functional {
            role: Role::Value,
            point: self.row_point(&ordered, c),
            key,
            scale,
            origins: vec![origin],
        };
        self.push(row, Some(data), vars, self.log2_k, functional)
    }

    fn weight_atom(&mut self, atom: &Atom, origin: Origin) -> Result<(), CompileError> {
        let data = self.tensor(atom.kind, 0)?;
        // A coefficient must be a public scalar uniform over the layers: a per-layer constant
        // that differs between layers makes another value than the matrix's at the point, which
        // the pair would drop as the first claim's (#556). Fail closed.
        let scaled = atom
            .coefficient
            .iter()
            .flatten()
            .find_map(|factor| match factor {
                Factor::Constant(NodeRef::Position(p)) if !self.schedule.layers.uniform(*p) => {
                    Some(*p)
                }
                _ => None,
            });
        if let Some(position) = scaled {
            let matrix = self
                .schedule
                .weight_type(&self.lowering.graph, atom.kind)
                .map_or(WeightMatrix::ChannelParams, |w| w.matrix());
            self.system.scaled_weights.push(ScaledWeight {
                matrix,
                tensor: Some(data),
                position,
                origin,
            });
            return Ok(());
        }
        let r = self.rounds.point(&atom.point)?;
        let rho = match &atom.layers {
            Layers::Eq(rho) => self.rounds.point(rho)?,
            Layers::None => Vec::new(),
            _ => return Err(internal("a weight pool atom weighs its layers by eq")),
        };
        let graph = &self.lowering.graph;
        let mut x = r.clone();
        x.extend(&rho);
        let vars = u32::try_from(x.len()).unwrap_or(u32::MAX);
        let (matrix, row, embedded) =
            if let Some((weight, _)) = graph.weights.iter().find(|(_, t)| *t == data) {
                let matrix = weight.ty.matrix();
                let layout = self.weights.matrix(matrix);
                if x.len() != (layout.log2_k() + layout.row_vars()) as usize {
                    return Err(internal("a weight claim has its matrix's variables"));
                }
                (matrix, MaskRow::Weight(matrix), x.clone())
            } else if let Some((weight, _)) = self
                .lowering
                .classes
                .iter()
                .chain(&self.lowering.multipliers)
                .find(|(_, t)| *t == data)
            {
                // The type tensor's point `r ‖ ρ` on probes, read back from `Ã_CP`'s point.
                let probe = probes(x.len());
                let (channel, layer) = probe.split_at(r.len());
                let full = self
                    .weights
                    .channel_point(weight.ty.matrix(), channel, layer)
                    .map_err(|_| internal("a ChannelParams claim has its point on Ã_CP"))?;
                (
                    WeightMatrix::ChannelParams,
                    MaskRow::Channel(atom.kind),
                    apply(&slots(&full, x.len())?, &x),
                )
            } else {
                return Err(internal(
                    "a weight pool atom is a weight or a ChannelParams tensor",
                ));
            };
        let width = self.weights.matrix(matrix).log2_k();
        let (column, row_point) = embedded.split_at(width as usize);
        let functional = Functional {
            role: Role::Value,
            point: column.to_vec(),
            key: x.clone(),
            scale: None,
            origins: vec![origin],
        };
        self.push(row, Some(data), vars, width, functional)?;
        // The row's first claim is its claim; a later point on it is a value of the claim's
        // type tensor at another point, `S ≡ 0` exactly when it keeps the claim's row point and
        // `Z` (RFC-0065 'Zero knowledge').
        let role = match self.claims.get(&row) {
            None => {
                self.claims.insert(
                    row,
                    FirstClaim {
                        key: x.clone(),
                        row_point: row_point.to_vec(),
                        z: vanishing(&x),
                    },
                );
                PairRole::Blinding
            }
            Some(first) if first.key == x => return Ok(()),
            Some(first) => PairRole::Value {
                secret: !(first.row_point == row_point && first.z == vanishing(&x)),
            },
        };
        self.pair(
            matrix,
            width,
            PairFunctional {
                role,
                point: column.to_vec(),
                key: x,
                tensor: Some(data),
                origins: vec![origin],
            },
        );
        Ok(())
    }

    fn virtual_value(
        &mut self,
        kind: Kind,
        coords: &[Coord],
        layers: &Layers,
        origin: Origin,
    ) -> Result<(), CompileError> {
        let mut x = self.rounds.point(coords)?;
        match layers {
            Layers::Eq(rho) => x.extend(self.rounds.point(rho)?),
            Layers::None => {}
            _ => return Err(internal("a relation proves a claim on every layer or none")),
        }
        let vars = u32::try_from(x.len()).unwrap_or(u32::MAX);
        let c = self.columns_of(vars, true);
        let functional = Functional {
            role: Role::Value,
            point: self.row_point(&x, c),
            key: x,
            scale: None,
            origins: vec![origin],
        };
        let tensor = self.schedule.layers.tensor(&self.lowering.graph, kind, 0);
        self.push(
            MaskRow::Virtual(kind),
            tensor,
            vars,
            self.log2_k,
            functional,
        )
    }
}

/// The weights' layout a claim list and its lowering give (RFC-0040 'The matrices'), as the
/// prover builds it.
///
/// # Errors
///
/// [`CompileError::Graph`] for an architecture without weight shapes, [`CompileError::Model`]
/// when the layout refuses them.
pub fn weight_layout(
    claims: &ClaimList,
    lowering: &Lowering,
) -> Result<WeightLayout, CompileError> {
    let channels = lowering
        .classes
        .iter()
        .chain(&lowering.multipliers)
        .map(|(w, _)| (w.ty.matrix(), w.layer))
        .collect();
    WeightLayout::new(
        claims.layout.weights,
        WeightShapes {
            layers: claims.arch.n_layers,
            shapes: claims.arch.weight_shapes()?,
            channels,
        },
    )
    .map_err(|_| CompileError::Model {
        rule: "the claim list's weights have a layout (RFC-0040)",
    })
}

/// The functionals `schedule` reveals of every mask row and every matrix pair, under `rule` and
/// the reading `committed` of committed tensors' mask columns.
///
/// `advice` and `weights` are `lowering`'s layouts for the claim list the schedule is proven
/// under: the column points come from them.
///
/// # Errors
///
/// [`CompileError::Schedule`] for a schedule the layouts do not place, or whose layout points do
/// more than move coordinates and add constant bits.
pub fn extract(
    lowering: &Lowering,
    schedule: &StageSchedule,
    advice: &ModelAdviceLayout<'_>,
    weights: &WeightLayout,
    rule: ColumnRule,
    committed: CommittedColumns,
) -> Result<MaskSystem, CompileError> {
    let log2_k = advice.layout().params().log2_k;
    let mut builder = Builder {
        lowering,
        schedule,
        advice,
        weights,
        rule,
        committed,
        log2_k,
        rounds: Rounds::new(schedule, log2_k),
        system: MaskSystem::default(),
        rows: HashMap::new(),
        pairs: HashMap::new(),
        claims: HashMap::new(),
        columns: HashMap::new(),
    };
    for (c, claim) in schedule.pool.iter().enumerate() {
        for (a, atom) in claim.atoms.iter().enumerate() {
            let origin = Origin::Pool {
                stage: claim.stage,
                claim: c,
                atom: a,
            };
            match claim.commitment {
                CommitmentClass::Advice => builder.advice_atom(atom, origin)?,
                CommitmentClass::Weight => builder.weight_atom(atom, origin)?,
            }
        }
    }
    for stage in schedule.stages.iter().skip(1) {
        let (rounds, _) = builder.rounds.stages[usize::from(stage.index)];
        let fold: Vec<Src> = (rounds - log2_k..rounds)
            .map(|round| {
                Src::Var(Var::Round {
                    stage: stage.index,
                    round,
                })
            })
            .collect();
        for (i, instance) in stage.instances.iter().enumerate() {
            let Source::Part { claim, .. } = instance.source else {
                continue;
            };
            let value = Origin::Virtual {
                stage: stage.index - 1,
                claim,
                instance: i,
            };
            builder.virtual_value(instance.kind, &instance.point, &instance.layers, value)?;
            let row = MaskRow::Virtual(instance.kind);
            let index = builder.rows[&row];
            let vars = builder.system.rows[index].vars;
            let tensor = builder.system.rows[index].tensor;
            builder.push(
                row,
                tensor,
                vars,
                log2_k,
                Functional {
                    role: Role::Fold,
                    point: fold.clone(),
                    key: fold.clone(),
                    scale: None,
                    origins: vec![Origin::Fold {
                        stage: stage.index,
                        instance: i,
                    }],
                },
            )?;
        }
    }
    Ok(builder.system)
}

/// A deterministic pseudo-random stream (SplitMix64).
struct Stream(u64);

impl Stream {
    fn new(seed: u64, tag: u64) -> Self {
        Self(mix(seed ^ mix(tag)))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    fn fp2(&mut self) -> Fp2 {
        let c0 = Goldilocks::from_u64_reduced(self.next());
        let c1 = Goldilocks::from_u64_reduced(self.next());
        Fp2::new(c0, c1)
    }
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The pseudo-random value of each indeterminate, `½`, 0 and 1 themselves.
struct Assignment {
    seed: u64,
    half: Fp2,
}

impl Assignment {
    fn new(seed: u64) -> Self {
        let two = Fp2::ONE + Fp2::ONE;
        Self {
            seed,
            half: two.inverse().unwrap_or(Fp2::ZERO),
        }
    }

    fn value(&self, src: Src) -> Fp2 {
        let tag = |kind: u64, a: u64, b: u64| (kind << 60) ^ (a << 32) ^ b;
        match src {
            Src::Zero => Fp2::ZERO,
            Src::One => Fp2::ONE,
            Src::Half => self.half,
            Src::Var(var) => {
                let tag = match var {
                    Var::Leaf(i) => tag(1, 0, u64::from(i)),
                    Var::Layer(i) => tag(2, 0, u64::from(i)),
                    Var::Zero(i) => tag(3, 0, u64::from(i)),
                    Var::Round { stage, round } => tag(4, u64::from(stage), u64::from(round)),
                };
                Stream::new(self.seed, tag).fp2()
            }
        }
    }
}

/// `m` points' sketches against `tests`: entry `j` of point `p`'s is
/// `⟨eq(p), ⊗_t (a_{jt}, b_{jt})⟩ = Π_t ((1 − p_t)·a_{jt} + p_t·b_{jt})`.
fn sketch(points: &[Vec<Fp2>], tests: &[Vec<(Fp2, Fp2)>]) -> Vec<Vec<Fp2>> {
    points
        .iter()
        .map(|p| {
            tests
                .iter()
                .map(|test| {
                    p.iter().zip(test).fold(Fp2::ONE, |acc, (&x, &(a, b))| {
                        acc * ((Fp2::ONE - x) * a + x * b)
                    })
                })
                .collect()
        })
        .collect()
}

/// Gaussian elimination over `Fp2` of `rows`, in order. For each row in the span of the rows
/// before it, the support of the combination, that row last.
fn dependencies(rows: &[Vec<Fp2>]) -> Vec<Vec<usize>> {
    let mut basis: Vec<(usize, Vec<Fp2>, Vec<Fp2>)> = Vec::new();
    let mut found = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let (residue, combination) = reduce(&basis, row, i, rows.len());
        match residue.iter().position(|x| *x != Fp2::ZERO) {
            Some(pivot) => {
                let scale = residue[pivot].inverse().unwrap_or(Fp2::ZERO);
                let residue = residue.iter().map(|&x| x * scale).collect();
                let combination = combination.iter().map(|&x| x * scale).collect();
                basis.push((pivot, residue, combination));
            }
            None => {
                let support = (0..rows.len())
                    .filter(|&j| j != i && combination[j] != Fp2::ZERO)
                    .chain([i])
                    .collect();
                found.push(support);
            }
        }
    }
    found
}

/// `row` reduced by `basis`, with its combination of the original rows: row `index` of `count`.
fn reduce(
    basis: &[(usize, Vec<Fp2>, Vec<Fp2>)],
    row: &[Fp2],
    index: usize,
    count: usize,
) -> (Vec<Fp2>, Vec<Fp2>) {
    let mut residue = row.to_vec();
    let mut combination = vec![Fp2::ZERO; count];
    combination[index] = Fp2::ONE;
    for (pivot, vector, comb) in basis {
        let factor = residue[*pivot];
        if factor != Fp2::ZERO {
            for (r, &v) in residue.iter_mut().zip(vector) {
                *r -= factor * v;
            }
            for (c, &v) in combination.iter_mut().zip(comb) {
                *c -= factor * v;
            }
        }
    }
    (residue, combination)
}

/// Random product test vectors: `count` of them over `width` coordinates.
fn tests(stream: &mut Stream, count: usize, width: u32) -> Vec<Vec<(Fp2, Fp2)>> {
    (0..count)
        .map(|_| (0..width).map(|_| (stream.fp2(), stream.fp2())).collect())
        .collect()
}

/// Merges functionals of one role and one key, a public relation: the first keeps its place
/// and gathers the others' origins.
fn merge(functionals: &[Functional]) -> Vec<Functional> {
    let mut merged: Vec<Functional> = Vec::new();
    for functional in functionals {
        match merged.iter_mut().find(|m| {
            m.role == functional.role && m.key == functional.key && m.scale == functional.scale
        }) {
            Some(m) => m.origins.extend(&functional.origins),
            None => merged.push(functional.clone()),
        }
    }
    merged
}

/// Merges values of one type tensor at one point, which are one value. Blinding claims never
/// merge: equal ones are a public relation already (RFC-0065 'Shared column points'), and a
/// repeated vector leaves every span as it is.
fn merge_pair(functionals: &[PairFunctional]) -> Vec<PairFunctional> {
    let mut merged: Vec<PairFunctional> = Vec::new();
    for functional in functionals {
        let same = |m: &PairFunctional| {
            matches!(
                (m.role, functional.role),
                (PairRole::Value { .. }, PairRole::Value { .. })
            ) && m.tensor == functional.tensor
                && m.key == functional.key
        };
        match merged.iter_mut().find(|m| same(m)) {
            Some(m) => m.origins.extend(&functional.origins),
            None => merged.push(functional.clone()),
        }
    }
    merged
}

/// R10 on `system`, with the indeterminates' values and the sketches drawn from `seed`: every
/// dependent set of a mask row, every secret value of a matrix pair in the span of the pair's
/// other functionals, and every revealed tensor of fewer than two variables.
pub fn check(system: &MaskSystem, seed: u64) -> Vec<Finding> {
    let assignment = Assignment::new(seed);
    let mut stream = Stream::new(seed, u64::MAX);
    let numeric =
        |p: &[Src]| -> Vec<Fp2> { p.iter().map(|&s| assignment.value(s)).collect::<Vec<_>>() };
    let mut findings: Vec<Finding> = system
        .scaled_weights
        .iter()
        .cloned()
        .map(Finding::ScaledWeight)
        .collect();
    for row in &system.rows {
        if let Some(bad) = row
            .functionals
            .iter()
            .find(|f| f.point.len() != row.width as usize)
        {
            findings.push(Finding::Width {
                row: Some(row.row),
                matrix: None,
                width: row.width,
                found: bad.point.len(),
            });
            continue;
        }
        let functionals = merge(&row.functionals);
        if row.vars < 2 && functionals.iter().any(|f| f.role == Role::Value) {
            findings.push(Finding::Unmaskable {
                row: row.row,
                tensor: row.tensor,
                vars: row.vars,
                origins: functionals
                    .iter()
                    .flat_map(|f| f.origins.iter().copied())
                    .collect(),
            });
        }
        let points: Vec<Vec<Fp2>> = functionals.iter().map(|f| numeric(&f.point)).collect();
        let tests = tests(&mut stream, points.len() + EXTRA_TESTS, row.width);
        for support in dependencies(&sketch(&points, &tests)) {
            findings.push(Finding::Dependent {
                row: row.row,
                tensor: row.tensor,
                functionals: support.iter().map(|&j| functionals[j].clone()).collect(),
            });
        }
    }
    for pair in &system.pairs {
        if let Some(bad) = pair
            .functionals
            .iter()
            .find(|f| f.point.len() != pair.width as usize)
        {
            findings.push(Finding::Width {
                row: None,
                matrix: Some(pair.matrix),
                width: pair.width,
                found: bad.point.len(),
            });
            continue;
        }
        let functionals = merge_pair(&pair.functionals);
        let points: Vec<Vec<Fp2>> = functionals.iter().map(|f| numeric(&f.point)).collect();
        let tests = tests(&mut stream, points.len() + EXTRA_TESTS, pair.width);
        let sketches = sketch(&points, &tests);
        for (i, functional) in functionals.iter().enumerate() {
            if functional.role != (PairRole::Value { secret: true }) {
                continue;
            }
            // The others first, then the value: a dependency ending at the value is its span.
            let mut rows: Vec<Vec<Fp2>> = sketches
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, s)| s.clone())
                .collect();
            let others: Vec<usize> = (0..functionals.len()).filter(|&j| j != i).collect();
            rows.push(sketches[i].clone());
            let last = rows.len() - 1;
            if let Some(support) = dependencies(&rows)
                .into_iter()
                .find(|s| s.last() == Some(&last))
            {
                findings.push(Finding::Spanned {
                    matrix: pair.matrix,
                    value: functional.clone(),
                    span: support[..support.len() - 1]
                        .iter()
                        .map(|&j| functionals[others[j]].clone())
                        .collect(),
                });
            }
        }
    }
    findings
}

/// The length at which R10 checks a sequence of `n` positions: the token-axis layout
/// `tessera-protocol/2` gives it.
///
/// A masked tensor has two variables at least (RFC-0013 'Masked extensions'), so in
/// `tessera-protocol/2` the token axis pads to `2^max(⌈log2 n⌉, 2)` rows, and so does a batch's
/// slot (RFC-0071 'The slots'); RFC-0035's padded rows fill the rows past `n`. An axis padded to
/// four rows is laid out as a padded axis (RFC-0047 'The order of a tensor's variables'), so
/// `n = 1`, 2 and 3 all take the layout of `n = 3`, a 3-row axis padded to 4. From 3 on, a
/// lowering at `n` has `tessera-protocol/2`'s shape and order. The real `n` matters: at 3 the
/// token axis is padded and at 4 it is not, so RFC-0047 orders a tensor's variables, and with
/// them its column sub-point, differently at the two lengths.
pub fn protocol2_length(n: u32) -> u32 {
    n.max(MIN_POSITIONS)
}

/// R10 on one claim list at one length and request, under one reading of committed tensors'
/// mask columns.
#[derive(Clone, Debug)]
pub struct MaskReport {
    /// The sequence's positions.
    pub n: u32,
    /// The length its schedule was derived at: [`protocol2_length`]`(n)` for
    /// [`check_claim_list`], `n` itself for [`check_at`].
    pub length: u32,
    /// The reading of committed tensors' mask columns.
    pub committed: CommittedColumns,
    /// The lowering the schedule was derived from, which names the findings' tensors.
    pub lowering: Arc<Lowering>,
    /// The functionals, before merging.
    pub system: MaskSystem,
    /// What [`check`] refuses.
    pub findings: Vec<Finding>,
}

impl MaskReport {
    /// The mask rows, the functionals and the folds the system holds, and its pairs' functionals,
    /// before merging.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let rows = self.system.rows.len();
        let functionals = self.system.rows.iter().map(|r| r.functionals.len()).sum();
        let folds = self
            .system
            .rows
            .iter()
            .flat_map(|r| &r.functionals)
            .filter(|f| f.role == Role::Fold)
            .count();
        let pairs = self.system.pairs.iter().map(|p| p.functionals.len()).sum();
        (rows, functionals, folds, pairs)
    }
}

/// R10 on `claims` for a sequence of `n` positions and `request`, as `tessera-protocol/2` lays it
/// out: [`check_at`] at [`protocol2_length`]`(n)`.
///
/// # Errors
///
/// Those of [`check_at`]; among them [`CompileError::Usage`] for a model whose `max_context` is
/// below the padded length.
pub fn check_claim_list(
    claims: &ClaimList,
    n: u32,
    request: Request,
    rule: ColumnRule,
    committed: CommittedColumns,
) -> Result<MaskReport, CompileError> {
    let report = check_at(claims, protocol2_length(n), request, rule, committed)?;
    Ok(MaskReport { n, ..report })
}

/// R10 on the schedule of `claims` lowered at exactly `length` positions, for `request`, under
/// one reading of committed tensors' mask columns: [`check_readings`] for that reading alone.
///
/// # Errors
///
/// Those of [`check_readings`].
pub fn check_at(
    claims: &ClaimList,
    length: u32,
    request: Request,
    rule: ColumnRule,
    committed: CommittedColumns,
) -> Result<MaskReport, CompileError> {
    check_readings(claims, length, request, rule, &[committed])?
        .pop()
        .ok_or(internal("a check has its reading's report"))
}

/// R10 on the schedule of `claims` lowered at exactly `length` positions, for `request`, under
/// each reading of `readings`: the composed schedule the prover proves (`derive_composed_schedule`
/// with [`model_natives`]), its layouts, [`extract`] under `rule` and each reading, and [`check`]
/// with [`SEED`]. The readings share one lowering. It does not pad the token axis, so below
/// [`MIN_POSITIONS`] it checks a layout `tessera-protocol/2` never gives: the controls' entry
/// point; [`check_claim_list`] is the protocol's.
///
/// # Errors
///
/// The lowering's, the range analysis', the selection's, the schedule's and the layouts' errors,
/// and [`CompileError::Usage`] when the selection's tables are not the claim list's.
pub fn check_readings(
    claims: &ClaimList,
    length: u32,
    request: Request,
    rule: ColumnRule,
    readings: &[CommittedColumns],
) -> Result<Vec<MaskReport>, CompileError> {
    let lowering = lower(PublicModel::of_claims(claims), length)?;
    let ranges = analyze(&lowering.graph, &lowering.tables)?;
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &claims.gadget_policy,
    )?;
    let natives = model_natives(&lowering, request)?;
    let schedule = derive_composed_schedule(&lowering, &natives.deferred, &natives.native)?;
    check_schedule(
        claims, lowering, &selection, &schedule, length, rule, readings,
    )
}

/// R10 on `schedule`, a schedule of `lowering`, which is `claims` lowered at `length` positions
/// with `selection` its gadgets, under `rule` and each reading of `readings`: the layouts,
/// [`extract`] and [`check`] with [`SEED`]. [`check_readings`] runs it on the composed schedule;
/// a gate's self-test runs it on a schedule it plants a violation in.
///
/// # Errors
///
/// The layouts' errors, and [`extract`]'s for a schedule the layouts do not place.
pub fn check_schedule(
    claims: &ClaimList,
    lowering: Lowering,
    selection: &Selection,
    schedule: &StageSchedule,
    length: u32,
    rule: ColumnRule,
    readings: &[CommittedColumns],
) -> Result<Vec<MaskReport>, CompileError> {
    let advice = ModelAdviceLayout::new(&lowering, selection, claims)?;
    let weights = weight_layout(claims, &lowering)?;
    let systems = readings
        .iter()
        .map(|&committed| {
            extract(&lowering, schedule, &advice, &weights, rule, committed)
                .map(|system| (committed, system))
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(advice);
    let lowering = Arc::new(lowering);
    Ok(systems
        .into_iter()
        .map(|(committed, system)| MaskReport {
            n: length,
            length,
            committed,
            lowering: Arc::clone(&lowering),
            findings: check(&system, SEED),
            system,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        Assignment, EXTRA_TESTS, Fp2, Src, Stream, Var, dependencies, merge, sketch, tests,
    };
    use super::{Functional, Origin, Role};
    use tessera_field::Field;

    fn x(i: u32) -> Src {
        Src::Var(Var::Zero(i))
    }

    /// The dependent sets of the functionals at `points`, each over `width` coordinates.
    fn dependent(points: &[Vec<Src>], width: u32) -> Vec<Vec<usize>> {
        let assignment = Assignment::new(7);
        let mut stream = Stream::new(7, 0);
        let numeric: Vec<Vec<Fp2>> = points
            .iter()
            .map(|p| p.iter().map(|&s| assignment.value(s)).collect())
            .collect();
        let tests = tests(&mut stream, points.len() + EXTRA_TESTS, width);
        dependencies(&sketch(&numeric, &tests))
    }

    /// The sketch is `⟨eq(p), ⊗_t (a_t, b_t)⟩`, checked against the 2^w-entry vectors.
    #[test]
    fn the_sketch_is_the_inner_product_with_a_product_vector() {
        let assignment = Assignment::new(3);
        let p: Vec<Fp2> = (0..3).map(|i| assignment.value(x(i))).collect();
        let mut stream = Stream::new(3, 1);
        let test = tests(&mut stream, 1, 3);
        let mut expected = Fp2::ZERO;
        for index in 0..8usize {
            let mut eq = Fp2::ONE;
            let mut g = Fp2::ONE;
            for (t, &pt) in p.iter().enumerate() {
                let (a, b) = test[0][t];
                if index >> t & 1 == 1 {
                    eq *= pt;
                    g *= b;
                } else {
                    eq *= Fp2::ONE - pt;
                    g *= a;
                }
            }
            expected += eq * g;
        }
        assert_eq!(sketch(&[p], &test), vec![vec![expected]]);
    }

    /// Generic points are independent; a repeated point and the pair-bit line are not: `eq(λ, r)`
    /// is `(1 − λ)·eq(0, r) + λ·eq(1, r)`.
    #[test]
    fn structured_points_are_dependent() {
        let generic = [vec![x(0), x(1)], vec![x(2), x(3)], vec![x(1), x(0)]];
        assert_eq!(dependent(&generic, 2), Vec::<Vec<usize>>::new());
        let repeated = [vec![x(0), x(1)], vec![x(2), x(3)], vec![x(0), x(1)]];
        assert_eq!(dependent(&repeated, 2), vec![vec![0, 2]]);
        let line = [
            vec![Src::Zero, x(1)],
            vec![x(2), x(3)],
            vec![Src::One, x(1)],
            vec![x(0), x(1)],
        ];
        assert_eq!(dependent(&line, 2), vec![vec![0, 2, 3]]);
        // At ½ the line point is the two pair points' mean.
        let half = [
            vec![Src::Zero, x(1)],
            vec![Src::One, x(1)],
            vec![Src::Half, x(1)],
        ];
        assert_eq!(dependent(&half, 2), vec![vec![0, 1, 2]]);
        // A row of 2^w entries holds at most 2^w independent functionals.
        let all: Vec<Vec<Src>> = (0..5).map(|i| vec![x(2 * i), x(2 * i + 1)]).collect();
        assert_eq!(dependent(&all, 2), vec![vec![0, 1, 2, 3, 4]]);
    }

    /// Values merge on their key; a fold never merges with a value at its point.
    #[test]
    fn a_fold_never_merges_with_a_value() {
        let functional = |role, key: Vec<Src>| Functional {
            role,
            point: key.clone(),
            key,
            scale: None,
            origins: vec![Origin::Fold {
                stage: 1,
                instance: 0,
            }],
        };
        let p = vec![x(0), x(1)];
        let merged = merge(&[
            functional(Role::Value, p.clone()),
            functional(Role::Fold, p.clone()),
            functional(Role::Value, p.clone()),
            functional(Role::Fold, p),
        ]);
        let roles: Vec<(Role, usize)> = merged.iter().map(|f| (f.role, f.origins.len())).collect();
        assert_eq!(roles, [(Role::Value, 2), (Role::Fold, 2)]);
    }
}
