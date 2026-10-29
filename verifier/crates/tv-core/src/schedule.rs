// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedule (RFC-0010 'Stages', 'Batching within a stage'; RFC-0037; RFC-0035
//! 'Zero-check points'), derived from the verifier's own lowered graph and from nothing else
//! (INV-SCH-1): [`derive()`] lists the leaf stage's claims, then each reduction stage's instances in
//! order, with their relations, variable counts and degrees, the point and layer weights of the
//! claim each receives, and where each claim their evaluations leave goes: to an instance of the
//! next stage, to the claim pool, or split into one value per part. The stage checks of
//! [`stages`](crate::stages) walk it.
//!
//! The derivation is written from the RFCs without reference to the prover's. Its points are
//! named by runs of coordinates of the transcript's sources, not one coordinate at a time, and
//! it walks a claim's linear maps with an explicit stack.
//!
//! # Layers and kinds (RFC-0037 'Layers and kinds')
//!
//! The checkpoints `c_0, …, c_L` bound the layers: layer `ℓ` holds the nodes after `c_ℓ`'s
//! producer up to and including `c_{ℓ+1}`'s, the prologue the nodes up to `c_0`'s producer, and
//! the epilogue the nodes after `c_L`'s. `λ = ⌈log2 L⌉`. The schedule refuses, as
//! [`ScheduleError::Template`] naming the first node that breaks it, a graph whose layers are not
//! one template:
//!
//! 1. `L ≥ 1`, each checkpoint has a producer, the producers' ids increase, and every layer has
//!    the same number of nodes `M`; node `p` of layer `ℓ` is position `p` of the template.
//! 2. At each position, every layer's node has the operation of layer 0's, up to its layer's own
//!    weights, constants, shift classes and tables: `MatMulWeight` names the same type at its own
//!    layer; `ConstMul` and `TableMap` may differ; `Requant` keeps `lo` and `hi`; `SoftmaxRow`
//!    keeps `σ`; every other operation is equal. It has as many inputs and outputs, of the same
//!    shapes and kinds, a lookup output's table aside.
//! 3. Each input of a node of layer `ℓ` is an earlier output of its own layer, the same position's
//!    and output's in every layer; its layer's checkpoint `c_ℓ`; the shared `positions`; or a
//!    graph input of its own layer, a weight, class or multiplier vector of the same type in every
//!    layer.
//! 4. A prologue node reads graph inputs and earlier prologue outputs. An epilogue node reads
//!    graph inputs, earlier epilogue outputs and `c_L`.
//! 5. Every checkpoint is committed advice.
//!
//! A tensor's kind is then its node's position and output index, stacked over the layers, the
//! layer index highest; the layer input `X_in[ℓ] = c_ℓ` is a kind of its own, whose slices are
//! `c_0` and `X_last[ℓ − 1]`, `X_last` the kind of the layer's last output; a per-layer graph
//! input is stacked by its type; prologue and epilogue tensors and the shared inputs have no layer
//! index. Kinds are named by layer 0's tensor ([`TensorRef`]).
//!
//! # Points, coefficients and claims
//!
//! A tensor of shape `[d_1, …, d_m]` is padded per dimension: axis `a` takes `⌈log2 d_a⌉`
//! variables, the last axis the lowest (RFC-0010 'Multilinear extensions', RFC-0011 rule 1). A
//! [`Point`] lists a tensor's coordinates by [`Run`]s of the transcript's [`Source`]s: a leaf
//! claim's point, the layer point `ρ_1`, a stage's round challenges and its zero-check point.
//!
//! A claim names a tensor, a point and [`Layers`]: for a stacked kind, `eq(ρ, ·)` at a point `ρ`
//! of `λ` coordinates, or the one layer `ℓ` a tensor outside the layers reads (`c_L`, read by the
//! epilogue). Through a linear node whose output is virtual, a claim becomes terms on the node's
//! inputs (RFC-0010 'Relations', RFC-0037 'Head layouts', 'Claims and atoms'):
//!
//! - `Reshape`: the same coordinates, when in each group of axes that merge into one, every axis
//!   but the outermost has a length that is a power of two (axes of length 1 aside); any other
//!   `Reshape` is refused ([`ScheduleError::Unsupported`]).
//! - `Transpose { perm }`: output axis `i` is input axis `perm[i]`, so input axis `a` takes the
//!   coordinates of the output axis `i` with `perm[i] = a`.
//! - `Expand { axis, len }`: the inserted axis's coordinates `r_a` are dropped, with the
//!   coefficient `LT(r_a, len)`, which is 1 when `len` is a power of two.
//! - `ConstMul { c }`: the coefficient times the constant. At a template position each layer has
//!   its own: a stacked committed advice atom takes each layer's, and any other atom takes them
//!   only when every layer's is equal (else [`ScheduleError::Unsupported`], RFC-0037 'Claims and
//!   atoms', RFC-0068).
//! - `Add`: both inputs, the first first.
//!
//! The walk is depth first, inputs in their node's order; it stops at a tensor that is not the
//! virtual output of a linear node, an **atom**: virtual (a sumcheck relation's output), committed
//! (advice, a weight matrix, or a class or multiplier vector of `ChannelParams`) or public. Atoms
//! on one tensor at one point with the same layers merge, their coefficients adding, and keep the
//! first one's place. A public atom is refused ([`ScheduleError::Unsupported`]): no claim of the
//! lowering reaches one.
//!
//! # Splits (RFC-0037 'Splits', as amended)
//!
//! A claim is **ready** when it is one virtual atom of coefficient 1, which becomes an instance of
//! the next stage ([`Route::Instance`]), or committed atoms of one commitment at one point, which
//! go to the pool as one claim ([`Route::Pool`]). Otherwise its stage splits it
//! ([`Route::Split`]): one value for each virtual atom, its tensor's evaluation, weighted by the
//! atom's coefficient in the check; and one for each group of committed atoms of one commitment at
//! one point, the group's combination `Σ_t a_t·P̃_t(r)`, coefficients included, counted with
//! weight 1, placed where the group's first atom is. The values come in atom order, claim by
//! claim: the leaf stage's by site, then by layer, then by the site's inputs; a reduction
//! stage's by instance, then by evaluation.
//!
//! # The leaf stage (RFC-0037 'The layer point'; RFC-0045)
//!
//! Its sites are the gadget nodes (`OneHot`, `Requant`, `RsqrtRow`, `SoftmaxRow`, `TableMap`) in
//! node order, a template position once for all its layers, then the logits, the LM head's
//! output. It leaves one claim on each input of each site, in the node's input order, at a point
//! of the site, one per real layer for a template position; and one on the logits. Its points are
//! the leaf stage's interface ([`Source::Leaf`]). The gadgets' own relations (RFC-0009,
//! RFC-0030) and RFC-0035's padding zero-checks join the schedule with the gadgets (#122, #123,
//! #126). A template site's claims are split per layer, each layer's coefficients its own.
//!
//! # Stages and their instances (RFC-0037 'Stages and their instances')
//!
//! Stage `k ≥ 1` holds one instance per virtual atom that stage `k − 1` leaves, in order: the
//! atom's producing relation, at the atom's point, stacked over the layers when the atom is a
//! stacked kind. Stage 1 stacks the leaf stage's per-layer claims at the layer point `ρ_1`, and
//! holds after them the four OneHot relations and equality zero-checks of committed linear outputs, in node order:
//! `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))` at the last coordinates of its zero-check point. The schedule
//! ends at the first stage with no instance.
//!
//! A stacked instance has its own variables, then `λ` layer variables; it takes its claim's
//! `eq(ρ, ·)` as a public factor over them. Instance `i` of `n_i` variables takes the stage's last
//! `n_i` challenges `r'`; its evaluations are those of the polynomials its relation's identity
//! names, in that order (RFC-0010 'Sumcheck messages', as amended), each a claim:
//!
//! | Relation | Variables | Degree | Evaluations, at |
//! |---|---|---|---|
//! | `MatMulWeight`, `Ỹ(r_o ‖ r_t) = Σ_i W̃(r_o, i)·X̃(r_t, i)` | `in`, layers | 3 stacked, else 2 | `W̃(r'_i ‖ r_o)`, `X̃(r'_i ‖ r_t)` |
//! | `MatMulAct`, `Z̃(r_c ‖ r_a ‖ r_h) = Σ_{j,b} eq(r_h, b)·Ã(j ‖ r_a ‖ b)·B̃` | `k`, leading axes, layers | 3 when any of the last two, else 2 | `Ã(r'_j ‖ r_a ‖ r'_b)`, `B̃(r_c ‖ r'_j ‖ r'_b)` or, transposed, `B̃(r'_j ‖ r_c ‖ r'_b)` |
//! | `Elementwise` (`Mul`), `Ỹ(r) = Σ_x eq(r, x)·Ã(x)·B̃(x)` | the output's, layers | 3 | `Ã(r')`, `B̃` at the coordinates of `b`'s axes: the highest for `Trailing`, the lowest for `Leading` |
//! | `SumSquares` (last axis), `Ũ(r) = Σ_x eq(r, x_row)·X̃(x)^2` | the input's, layers | 3 | `X̃(r')` |
//! | `Rotate`, `Ỹ(r) = Σ_x E_r(x)·X̃(x)` | the input's, layers | 2 | `X̃(r')` |
//! | `ZeroCheck` (OneHot vocabulary), `0 = Σ eq(z,x)·(1-LT(v,vocab))·H(x)` | the indicator's | 3 | `H(r')` |
//! | `ZeroCheck` (OneHot count), `0 = Σ eq(z_row,t)·(H(v,t)-psi(t)/V_pad)` | the indicator's | 2 | `H(r')` |
//! | `ZeroCheck` (OneHot index), `0 = Σ eq(z_row,t)·(index(v)·H(v,t)-T(t)/V_pad)` | the indicator's | 2 | `H(r')`, `T(r'_row)` |
//! | `ZeroCheck` (OneHot booleanity), `0 = Σ_x eq(z,x)·H(x)·(1-H(x))` | the indicator's | 3 | `H(r')` |
//! | `ZeroCheck` (equality, `ConstMul`), `0 = Σ_x eq(z, x)·(Ỹ(x) − c·X̃(x))` | the output's | 2 | `Ỹ(r')`, `X̃(r')` |
//!
//! Stacked, every evaluation's claim has the layer weights `eq(r'_ℓ, ·)`. A stage's degree is
//! the largest among its instances, its rounds `n_max` the most variables, and its zero-check
//! point has as many coordinates as its largest `ZeroCheck`. The stage count is one more than the
//! number of reduction stages, which does not depend on `L` or `n` (INV-SCH-4).
//!
//! # Operations
//!
//! The schedule proves the Llama lowering's operations. A graph that holds another, one the
//! anchor circuits add (`Sub`, `ConstAdd`, `Sum`, `Slice`, `GatherPublic`, `Bitwise`,
//! `BitDecompose`, `BitFn`, `AssertEqual`), a `ConstMul` by a constant per channel or a
//! `TableMap` into one of their tables, is [`ScheduleError::Unsupported`] at the first such node,
//! before its layers are read, until #150 derives their relations (RFC-0072).
//!
//! # Errors
//!
//! Every [`ScheduleError`] is a fault of the verifier, `Inconclusive(ERR-INTERNAL)`, or
//! `Inconclusive(ERR-RES-MEMORY)` for [`ScheduleError::Memory`]: the graph is the verifier's own
//! lowering of a validated claim list, which RFC-0038 makes one template.

use tv_field::Fp;

use crate::claims::WeightType;
use crate::lowering::{Broadcast, Graph, Kind, Node, NodeId, Op, Role, TensorId};

pub mod degree;
pub use degree::NativeKind;

/// Where a run of coordinates comes from. The stage checks fill each source as the stages run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// The point of leaf claim `c`: the leaf stage's interface.
    Leaf(u32),
    /// The layer point `ρ_1`, squeezed under `stage/1/layer/{c}`.
    Layer,
    /// Stage `k`'s round challenges, in round order.
    Rounds(u16),
    /// Stage `k`'s zero-check point, squeezed under `stage/{k}/zero/{c}`.
    Zero(u16),
}

/// The coordinates `start` to `start + len − 1` of a source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Run {
    /// The source.
    pub source: Source,
    /// Its first coordinate taken.
    pub start: usize,
    /// How many coordinates.
    pub len: usize,
}

/// A point: runs of coordinates, its lowest coordinates first. Two points are the same point
/// when they name the same coordinates, which their runs, merged where contiguous, show.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Point {
    runs: Vec<Run>,
}

impl Point {
    /// The coordinates `start .. start + len` of `source`.
    fn of(source: Source, start: usize, len: usize) -> Result<Self, ScheduleError> {
        let mut point = Self::default();
        point.push(Run { source, start, len })?;
        Ok(point)
    }

    /// The runs, lowest coordinates first.
    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// The number of coordinates.
    pub fn len(&self) -> usize {
        self.runs
            .iter()
            .fold(0usize, |total, run| total.saturating_add(run.len))
    }

    /// Whether the point has no coordinate.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    fn try_clone(&self) -> Result<Self, ScheduleError> {
        Ok(Self {
            runs: copy(&self.runs)?,
        })
    }

    /// Appends a run above the others, merged with the last when contiguous.
    fn push(&mut self, run: Run) -> Result<(), ScheduleError> {
        if run.len == 0 {
            return Ok(());
        }
        if let Some(last) = self.runs.last_mut()
            && last.source == run.source
            && last.start.checked_add(last.len) == Some(run.start)
        {
            last.len = last
                .len
                .checked_add(run.len)
                .ok_or(ScheduleError::Internal)?;
            return Ok(());
        }
        push(&mut self.runs, run)
    }

    /// Coordinates `from .. from + len`.
    fn slice(&self, from: usize, len: usize) -> Result<Self, ScheduleError> {
        let end = from.checked_add(len).ok_or(ScheduleError::Internal)?;
        if end > self.len() {
            return Err(ScheduleError::Internal);
        }
        let mut out = Self::default();
        let mut at = 0usize;
        for run in &self.runs {
            let run_end = at.checked_add(run.len).ok_or(ScheduleError::Internal)?;
            let lo = from.max(at);
            let hi = end.min(run_end);
            if lo < hi {
                let offset = lo.checked_sub(at).ok_or(ScheduleError::Internal)?;
                out.push(Run {
                    source: run.source,
                    start: run
                        .start
                        .checked_add(offset)
                        .ok_or(ScheduleError::Internal)?,
                    len: hi.checked_sub(lo).ok_or(ScheduleError::Internal)?,
                })?;
            }
            at = run_end;
        }
        Ok(out)
    }

    /// The concatenation of `parts`, the first lowest.
    fn concat(parts: &[&Point]) -> Result<Self, ScheduleError> {
        let mut out = Self::default();
        for part in parts {
            for &run in &part.runs {
                out.push(run)?;
            }
        }
        Ok(out)
    }

    /// The point with the coordinates of the axes `axes`, each a `(from, len)` block of this
    /// point, in that order, the first lowest.
    fn blocks(&self, axes: &[(usize, usize)]) -> Result<Self, ScheduleError> {
        let mut out = Self::default();
        for &(from, len) in axes {
            for run in self.slice(from, len)?.runs {
                out.push(run)?;
            }
        }
        Ok(out)
    }
}

/// The layer weights of a claim on a stacked kind.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Layers {
    /// No layer index: a tensor outside the layers.
    None,
    /// `eq(ρ, ·)` over the cube of `2^λ` layers, at the point `ρ` of `λ` coordinates.
    Point(Point),
    /// The one layer `ℓ`: `c_L`, read by the epilogue, is layer `L − 1`'s last output.
    One(u32),
}

impl Layers {
    fn try_clone(&self) -> Result<Self, ScheduleError> {
        Ok(match self {
            Self::None => Self::None,
            Self::Point(point) => Self::Point(point.try_clone()?),
            Self::One(layer) => Self::One(*layer),
        })
    }
}

/// A tensor of the schedule: a kind stacked over the layers, the layer input, or a tensor outside
/// the layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorRef {
    /// A kind stacked over the layers, named by layer 0's tensor: a template position's output,
    /// or a per-layer weight, class or multiplier vector.
    Stacked(TensorId),
    /// The layer input `X_in[ℓ] = c_ℓ`, whose slices are `c_0` and `X_last[ℓ − 1]`.
    LayerInput,
    /// A tensor outside the layers: a prologue or epilogue tensor, or a shared graph input.
    Single(TensorId),
}

/// The commitment a committed tensor belongs to (RFC-0011, RFC-0012, RFC-0040).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Commitment {
    /// The advice commitment: advice, lookup outputs, the tokens and the checkpoints.
    Advice,
    /// A weight matrix of the weight commitment.
    Weight(WeightType),
    /// `ChannelParams`: the class and multiplier vectors.
    ChannelParams,
}

/// What an atom is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// A sumcheck relation's output, produced by `node`.
    Virtual {
        /// The producing node: layer 0's for a stacked kind.
        node: NodeId,
    },
    /// A committed tensor.
    Committed(Commitment),
}

/// A coefficient: a sum of terms, each a constant times `LT` factors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Coefficient {
    terms: Vec<Term>,
}

/// One term of a coefficient.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Term {
    /// The constant: one value, or one per real layer for a stacked committed advice atom.
    pub constant: Vec<Fp>,
    /// `LT(point, bound)` factors, from `Expand`s of a length that is not a power of two.
    pub lts: Vec<(Point, u64)>,
}

impl Term {
    fn try_clone(&self) -> Result<Self, ScheduleError> {
        let mut lts = Vec::new();
        lts.try_reserve_exact(self.lts.len())
            .map_err(|_| ScheduleError::Memory)?;
        for (point, bound) in &self.lts {
            lts.push((point.try_clone()?, *bound));
        }
        Ok(Self {
            constant: copy(&self.constant)?,
            lts,
        })
    }
}

impl Coefficient {
    /// The terms.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Whether it is the constant 1, in every layer.
    pub fn is_one(&self) -> bool {
        match self.terms.as_slice() {
            [term] => term.lts.is_empty() && term.constant.iter().all(|&c| c == Fp::ONE),
            _ => false,
        }
    }
}

/// An atom of a claim: a coefficient on a tensor at a point with layer weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// The tensor.
    pub tensor: TensorRef,
    /// Its point: the tensor's coordinates, without the layer index.
    pub point: Point,
    /// Its layer weights.
    pub layers: Layers,
    /// Its coefficient in the claim.
    pub coefficient: Coefficient,
    /// Virtual or committed.
    pub class: Class,
}

/// How a claim continues.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// One virtual atom of coefficient 1: instance `0`'s claim in the next stage.
    Instance(usize),
    /// Committed atoms of one commitment at one point: one claim in the pool.
    Pool,
    /// Split into one value per part, in atom order.
    Split(Vec<Part>),
}

/// A part of a split claim.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// A virtual atom: its value is its tensor's evaluation, weighted by its coefficient in the
    /// check, and becomes the claim of the next stage's instance `instance`.
    Virtual {
        /// The atom's index in the claim.
        atom: usize,
        /// The next stage's instance.
        instance: usize,
    },
    /// Committed atoms of one commitment at one point: the value is their combination,
    /// coefficients included, counted with weight 1, and becomes one claim in the pool.
    Group {
        /// The atoms' indices in the claim, in atom order.
        atoms: Vec<usize>,
    },
}

/// A claim: the tensor it names, its atoms and its route.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Claim {
    /// The tensor claimed.
    pub tensor: TensorRef,
    /// The tensor's variables, without the layer index: its point's length.
    pub vars: usize,
    /// The atoms, in atom order.
    pub atoms: Vec<Atom>,
    /// How it continues.
    pub route: Route,
}

impl Claim {
    /// The values a split of it sends: one per part, 0 for a ready claim.
    pub fn parts(&self) -> usize {
        match &self.route {
            Route::Split(parts) => parts.len(),
            Route::Instance(_) | Route::Pool => 0,
        }
    }
}

/// A graph leaf source supplied by a native relation instead: `(site, Some(input))` for a
/// gadget input; `(LM-head node, None)` for the logits consumed by the greedy certificate.
pub type Deferred = (NodeId, Option<usize>);

/// One original graph source returned by a native relation. Coordinates are a contiguous
/// projection of that instance's natural final point, before its highest layer coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeOutput {
    /// Actual graph tensor in layer zero, or outside the layer template.
    pub tensor: TensorId,
    /// First local coordinate of the native final point.
    pub from: usize,
    /// Source coordinates, excluding the layer axis.
    pub vars: usize,
}

/// A verifier-owned native stage-one instance, never decoded from a proof.
/// Its callback must implement the normative relation and retain every non-graph debt.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Native {
    /// Unique verifier-owned callback identity.
    pub tag: u32,
    /// Semantic relation, fixed by the native plan (RFC-0013 'Degree / The keys').
    pub kind: NativeKind,
    /// Owning first-layer gadget node, or a node outside the template.
    pub node: NodeId,
    /// Total natural variables, INCLUDING the layer axis for a template site.
    pub vars: usize,
    /// Degree of the complete polynomial, including public equality/selectors.
    pub degree: usize,
    /// Raw final evaluations sent by the prover; distinct from transformed graph outputs.
    pub evaluations: usize,
    /// Whether this is a stage-one zero-check, ordered by owning node after virtual instances.
    pub zero: bool,
    /// Transformed original graph source claims, in callback output order.
    pub outputs: Vec<NativeOutput>,
}

/// A relation kind (RFC-0010 'Relations').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationKind {
    /// `MatMulWeight`.
    MatMulWeight,
    /// `MatMulAct`.
    MatMulAct,
    /// `Elementwise`.
    Elementwise,
    /// `Rotate`.
    Rotate,
    /// `SumSquares`.
    SumSquares,
    /// `ZeroCheck`.
    ZeroCheck,
}

/// An instance's relation, with what its final check needs to know of its shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    /// A normative native relation supplied by the verifier-owned composition callback.
    Native {
        /// Unique verifier-owned identity, never selected by proof bytes.
        tag: u32,
        /// Semantic relation from the verifier-owned native plan, never inferred from arity.
        kind: NativeKind,
        /// Raw proof evaluations, before transformation to outgoing graph claims.
        evaluations: usize,
        /// Whether its incoming claim must be zero.
        zero: bool,
    },
    /// `Ỹ(r_o ‖ r_t) = Σ_i W̃(r_o, i)·X̃(r_t, i)`.
    MatMulWeight {
        /// The summed axis's variables, `in`.
        in_vars: usize,
        /// The output axis's, the claim's lowest.
        out_vars: usize,
    },
    /// `Z̃(r_c ‖ r_a ‖ r_h) = Σ_{j,b} eq(r_h, b)·Ã(j ‖ r_a ‖ b)·B̃`, `b` over the leading axes.
    MatMulAct {
        /// Whether the right operand is transposed.
        transpose_rhs: bool,
        /// The summed axis's variables, `k`.
        k_vars: usize,
        /// The output's row axis's variables, `m`.
        m_vars: usize,
        /// The output's column axis's variables, `n`, the claim's lowest.
        n_vars: usize,
        /// The leading axes' variables.
        lead_vars: usize,
    },
    /// `Ỹ(r) = Σ_x eq(r, x)·Ã(x)·B̃(x)`.
    Elementwise {
        /// How `B` broadcasts.
        broadcast: Broadcast,
        /// The first of the output's coordinates that `B` reads.
        b_from: usize,
        /// How many it reads.
        b_vars: usize,
    },
    /// `Ũ(r) = Σ_x eq(r, x_row)·X̃(x)^2`, summing the last axis.
    SumSquares {
        /// The summed axis's variables, the lowest.
        col_vars: usize,
    },
    /// `Ỹ(r) = Σ_x E_r(x)·X̃(x)`, RoPE's kernel.
    Rotate {
        /// The sequence's positions, the input's first axis.
        positions: u32,
        /// The variables of the axes between the positions and the last.
        middle_vars: usize,
        /// Multi-request slot geometry when `R ≥ 2` (RFC-0071 'The public selectors').
        slots: Option<crate::lowering::Slots>,
    },
    /// The original OneHot zero-check: `0 = Σ_x eq(z,x)·H(x)·(1-H(x))`.
    Booleanity,
    /// Zero-check for indicator entries outside the real vocabulary.
    Vocabulary {
        /// Real vocabulary length.
        vocab: u32,
        /// Lowest coordinates, the padded vocabulary axis.
        vocab_vars: usize,
    },
    /// Row count zero-check, with its public selector divided by the padded vocabulary.
    OneHotCount {
        /// Lowest coordinates, the padded vocabulary axis.
        vocab_vars: usize,
        /// Real token rows (`N`).
        positions: u32,
        /// Multi-request slot geometry when `R ≥ 2` (RFC-0071 'The public selectors').
        slots: Option<crate::lowering::Slots>,
    },
    /// Row index zero-check against the original token source.
    OneHotIndex {
        /// Lowest coordinates, the padded vocabulary axis.
        vocab_vars: usize,
    },
    /// `0 = Σ_x eq(z,x)·(Y(x)-c·X(x))` for a committed ConstMul output.
    Equality {
        /// The constant `c`.
        constant: Fp,
    },
}

impl Relation {
    /// Its kind.
    pub fn kind(&self) -> RelationKind {
        match self {
            Relation::Native { zero, .. } => {
                if *zero {
                    RelationKind::ZeroCheck
                } else {
                    RelationKind::Elementwise
                }
            }
            Relation::MatMulWeight { .. } => RelationKind::MatMulWeight,
            Relation::MatMulAct { .. } => RelationKind::MatMulAct,
            Relation::Elementwise { .. } => RelationKind::Elementwise,
            Relation::SumSquares { .. } => RelationKind::SumSquares,
            Relation::Rotate { .. } => RelationKind::Rotate,
            Relation::Equality { .. }
            | Relation::Booleanity
            | Relation::Vocabulary { .. }
            | Relation::OneHotCount { .. }
            | Relation::OneHotIndex { .. } => RelationKind::ZeroCheck,
        }
    }
}

/// One instance of a reduction stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Instance {
    /// Its relation.
    pub relation: Relation,
    /// The node whose relation it proves: layer 0's for a stacked instance.
    pub node: NodeId,
    /// The tensor its claim names: the node's output.
    pub tensor: TensorRef,
    /// Whether it is stacked over the layers.
    pub stacked: bool,
    /// Its variables `n_i`: its own, then `λ` when stacked.
    pub vars: usize,
    /// Its degree.
    pub degree: usize,
    /// Its claim's point on the tensor, without the layer index; empty for a zero-check.
    pub point: Point,
    /// Its claim's layer weights.
    pub layers: Layers,
    /// For a zero-check, its point `z`: the last `n_i` coordinates of the stage's zero-check
    /// point; empty otherwise.
    pub zero: Point,
    /// The claims its final evaluations leave, in the order its relation's identity names them.
    pub evals: Vec<Claim>,
}
impl Instance {
    /// Raw final proof evaluations consumed by this instance. A native callback may transform
    /// them into a different number of original graph claims in `evals`.
    pub fn input_evaluations(&self) -> usize {
        match self.relation {
            Relation::Native { evaluations, .. } => evaluations,
            _ => self.evals.len(),
        }
    }
}

/// A reduction stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stage {
    /// Its index `k`, from 1.
    pub index: u16,
    /// Its degree: the largest of its instances'.
    pub degree: usize,
    /// Its rounds `n_max`: the most variables of its instances.
    pub rounds: usize,
    /// The layer point's coordinates it squeezes: `λ` in stage 1, else 0.
    pub layer_vars: usize,
    /// Its zero-check point's coordinates, 0 when it holds no zero-check.
    pub zero_vars: usize,
    /// Its instances, in order.
    pub instances: Vec<Instance>,
    /// The values its split claims send.
    pub splits: usize,
}

/// A claim of the leaf stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeafClaim {
    /// The gadget node, layer 0's for a template position, or the LM head's matmul for the
    /// logits.
    pub site: NodeId,
    /// The gadget input the claim names, or `None` for the logits.
    pub input: Option<usize>,
    /// Whether it holds one value per layer: a template position's site.
    pub stacked: bool,
    /// The claim.
    pub claim: Claim,
}

/// The stage schedule of one lowered graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schedule {
    layers: u32,
    layer_vars: usize,
    leaf: Vec<LeafClaim>,
    leaf_splits: usize,
    stages: Vec<Stage>,
    first_checkpoint: TensorId,
    last_output: TensorId,
}

impl Schedule {
    /// `L`, the model's layers.
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// `λ = ⌈log2 L⌉`, the layer variables of a stacked kind.
    pub fn layer_vars(&self) -> usize {
        self.layer_vars
    }

    /// The leaf stage's claims, in order.
    pub fn leaf(&self) -> &[LeafClaim] {
        &self.leaf
    }

    /// The values the leaf stage's split claims send under `stage/0/split`.
    pub fn leaf_splits(&self) -> usize {
        self.leaf_splits
    }

    /// The reduction stages, stage 1 first.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// The number of stages, the leaf stage included.
    pub fn stage_count(&self) -> usize {
        self.stages.len().saturating_add(1)
    }

    /// `c_0`, the layer input's first slice.
    pub fn first_checkpoint(&self) -> TensorId {
        self.first_checkpoint
    }

    /// Layer 0's last output, which names `X_last`, the layer input's other slices.
    pub fn last_output(&self) -> TensorId {
        self.last_output
    }
}

/// Why no schedule could be derived. Each is a fault of the verifier, whose graph comes from its
/// own lowering: see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleError {
    /// The graph's layers are not one template: `node` is the first that breaks a rule of
    /// 'Layers and kinds'.
    Template {
        /// The node.
        node: NodeId,
    },
    /// A claim reaches what the schedule does not prove: a `Reshape` that is no identity, a
    /// per-layer constant on a virtual, weight or `ChannelParams` atom, a public atom, or a
    /// relation of another shape, at `node`; or `node` is the first of an operation the module
    /// documentation's 'Operations' names.
    Unsupported {
        /// The node.
        node: NodeId,
    },
    /// Memory for the schedule could not be reserved.
    Memory,
    /// A step the graph's checks make infallible failed: a bug of this crate.
    Internal,
}

impl core::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScheduleError::Template { node } => {
                write!(
                    f,
                    "the graph's layers are not one template at node {}",
                    node.0
                )
            }
            ScheduleError::Unsupported { node } => {
                write!(
                    f,
                    "the schedule does not prove the claim at node {}",
                    node.0
                )
            }
            ScheduleError::Memory => f.write_str("memory for the schedule could not be reserved"),
            ScheduleError::Internal => f.write_str("an internal step of the schedule failed"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Derives the stage schedule of `graph`, in the order of the module documentation.
pub fn derive(graph: &Graph) -> Result<Schedule, ScheduleError> {
    let layout = Layout::of(graph)?;
    let deriving = Deriving {
        graph,
        layout: &layout,
        native: &[],
        deferred: &[],
    };
    deriving.schedule()
}

/// Compose independently derived native stage-one relations with graph reductions.
/// `deferred` names gadget inputs or logits supplied by native outputs instead of the graph leaf stage.
/// Every such input must be replaced by an output of a native relation at that SAME site.
/// All descriptors/callbacks are trusted verifier-owned metadata: this is a component schedule,
/// not an acceptance predicate or a proof-supplied alternative model definition.
///
/// # Errors
/// Existing graph schedule errors; `Internal` for duplicate native tags/deferred inputs, unknown
/// sites, invalid projections, wrong template representatives or absent replacement outputs.
/// Native reserves use the same fallible `Memory` paths as graph derivation.
pub fn derive_composed(
    graph: &Graph,
    deferred: &[Deferred],
    native: &[Native],
) -> Result<Schedule, ScheduleError> {
    let layout = Layout::of(graph)?;
    for (i, spec) in native.iter().enumerate() {
        if native
            .get(..i)
            .ok_or(ScheduleError::Internal)?
            .iter()
            .any(|n| n.tag == spec.tag)
            || spec.vars > 63
            || !(1..=crate::sumcheck::Protocol::One.max_degree(crate::sumcheck::Prefix::Stage(1)))
                .contains(&spec.degree)
            || spec.evaluations == 0
        {
            return Err(ScheduleError::Internal);
        }
        let node = node_of(graph, spec.node)?;
        if !(is_gadget(node.op()) || is_logits(node))
            || layout.position(spec.node).is_some_and(|(l, _)| l != 0)
        {
            return Err(ScheduleError::Internal);
        }
        let lambda = if layout.position(spec.node).is_some() {
            layout.layer_vars
        } else {
            0
        };
        let own = spec
            .vars
            .checked_sub(lambda)
            .ok_or(ScheduleError::Internal)?;
        for output in &spec.outputs {
            if output
                .from
                .checked_add(output.vars)
                .ok_or(ScheduleError::Internal)?
                > own
            {
                return Err(ScheduleError::Internal);
            }
        }
    }
    for (i, &(node, input)) in deferred.iter().enumerate() {
        if deferred
            .get(..i)
            .ok_or(ScheduleError::Internal)?
            .contains(&(node, input))
        {
            return Err(ScheduleError::Internal);
        }
        let definition = node_of(graph, node)?;
        if !(is_gadget(definition.op()) || is_logits(definition))
            || layout.position(node).is_some_and(|(l, _)| l != 0)
        {
            return Err(ScheduleError::Internal);
        }
        let tensor = match input {
            Some(index) => *definition
                .inputs()
                .get(index)
                .ok_or(ScheduleError::Internal)?,
            None if is_logits(definition) => *definition
                .outputs()
                .first()
                .ok_or(ScheduleError::Internal)?,
            None => return Err(ScheduleError::Internal),
        };
        if !native
            .iter()
            .any(|n| n.node == node && n.outputs.iter().any(|o| o.tensor == tensor))
        {
            return Err(ScheduleError::Internal);
        }
    }
    Deriving {
        graph,
        layout: &layout,
        native,
        deferred,
    }
    .schedule()
}

fn is_logits(node: &Node) -> bool {
    matches!(node.op(),Op::MatMulWeight{weight} if weight.ty==WeightType::LmHead)
}

/// Validate the common layer template without deriving unrelated graph reductions.
pub(crate) fn template_layers(graph: &Graph) -> Result<u32, ScheduleError> {
    Ok(Layout::of(graph)?.layers)
}

// ----- Layers and the template --------------------------------------------------------------

/// Where a tensor lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// Outside the layers.
    Outside,
    /// In layer `layer`, as the kind layer 0's tensor `rep` names.
    Layer { layer: u32, rep: TensorId },
}

/// The graph's layers, checked as one template.
pub(crate) struct Layout {
    layers: u32,
    layer_vars: usize,
    /// Node ids of layer 0's first node, and of the first node after the last layer.
    template_start: u32,
    epilogue_start: u32,
    /// Nodes per layer.
    len: u32,
    /// Each tensor's place, by id.
    places: Vec<Place>,
    /// Each graph input's role, by tensor id.
    roles: Vec<Option<Role>>,
    first_checkpoint: TensorId,
    last_output: TensorId,
}

impl Layout {
    pub(crate) fn of(graph: &Graph) -> Result<Self, ScheduleError> {
        // 'Operations': the anchor circuits' nodes, before anything else is read.
        if let Some(node) = graph.first_non_llama() {
            return Err(ScheduleError::Unsupported { node });
        }
        let checkpoints = graph.checkpoints();
        let layers = checkpoints
            .len()
            .checked_sub(1)
            .filter(|&l| l >= 1)
            .ok_or(ScheduleError::Template { node: NodeId(0) })?;
        let layers = u32::try_from(layers).map_err(|_| ScheduleError::Internal)?;
        // 1. The producers bound the layers, each of M nodes.
        let mut producers = Vec::new();
        reserve(&mut producers, checkpoints.len())?;
        for &c in checkpoints {
            let producer = graph
                .tensor(c)
                .and_then(|t| t.producer)
                .ok_or(ScheduleError::Template { node: NodeId(0) })?;
            producers.push(producer.0);
        }
        let first = *producers.first().ok_or(ScheduleError::Internal)?;
        let second = *producers.get(1).ok_or(ScheduleError::Internal)?;
        let len = second
            .checked_sub(first)
            .filter(|&m| m >= 1)
            .ok_or(ScheduleError::Template {
                node: NodeId(second),
            })?;
        for pair in producers.windows(2) {
            if let [a, b] = pair
                && b.checked_sub(*a) != Some(len)
            {
                return Err(ScheduleError::Template { node: NodeId(*b) });
            }
        }
        let template_start = first.checked_add(1).ok_or(ScheduleError::Internal)?;
        let last_producer = *producers.last().ok_or(ScheduleError::Internal)?;
        let epilogue_start = last_producer
            .checked_add(1)
            .ok_or(ScheduleError::Internal)?;
        let layer_vars = ceil_log2(u64::from(layers))?;

        let mut roles = Vec::new();
        reserve(&mut roles, graph.tensors().len())?;
        roles.resize(graph.tensors().len(), None);
        for input in graph.inputs() {
            *roles
                .get_mut(index(input.tensor.0)?)
                .ok_or(ScheduleError::Internal)? = Some(input.role);
        }
        let mut places = Vec::new();
        reserve(&mut places, graph.tensors().len())?;
        places.resize(graph.tensors().len(), Place::Outside);

        let first_checkpoint = *checkpoints.first().ok_or(ScheduleError::Internal)?;
        let last_output = *checkpoints.get(1).ok_or(ScheduleError::Internal)?;
        let mut layout = Self {
            layers,
            layer_vars,
            template_start,
            epilogue_start,
            len,
            places,
            roles,
            first_checkpoint,
            last_output,
        };
        layout.check_template(graph)?;
        Ok(layout)
    }

    /// Node `position` of layer `layer`.
    fn node_id(&self, layer: u32, position: u32) -> Result<NodeId, ScheduleError> {
        let offset = layer
            .checked_mul(self.len)
            .and_then(|o| o.checked_add(position))
            .ok_or(ScheduleError::Internal)?;
        Ok(NodeId(
            self.template_start
                .checked_add(offset)
                .ok_or(ScheduleError::Internal)?,
        ))
    }

    /// The layer and position of a node inside the layers, or `None` outside them.
    fn position(&self, node: NodeId) -> Option<(u32, u32)> {
        let offset = node.0.checked_sub(self.template_start)?;
        let layer = offset.checked_div(self.len)?;
        let position = offset.checked_rem(self.len)?;
        (layer < self.layers).then_some((layer, position))
    }

    fn place(&self, tensor: TensorId) -> Result<Place, ScheduleError> {
        self.places
            .get(index(tensor.0)?)
            .copied()
            .ok_or(ScheduleError::Internal)
    }

    /// Actual tensors of a checked layer-zero representative, in real-layer order.
    pub(crate) fn family(&self, rep: TensorId) -> Result<Vec<TensorId>, ScheduleError> {
        if self.place(rep)? != (Place::Layer { layer: 0, rep }) {
            return Err(ScheduleError::Internal);
        }
        let mut found = Vec::new();
        reserve(&mut found, index(self.layers)?)?;
        found.resize(index(self.layers)?, None);
        for (id, place) in self.places.iter().enumerate() {
            if let Place::Layer { layer, rep: r } = *place
                && r == rep
            {
                let slot = found
                    .get_mut(index(layer)?)
                    .ok_or(ScheduleError::Internal)?;
                if slot
                    .replace(TensorId(
                        u32::try_from(id).map_err(|_| ScheduleError::Internal)?,
                    ))
                    .is_some()
                {
                    return Err(ScheduleError::Internal);
                }
            }
        }
        let mut family = Vec::new();
        reserve(&mut family, found.len())?;
        for tensor in found {
            family.push(tensor.ok_or(ScheduleError::Internal)?);
        }
        Ok(family)
    }

    fn role(&self, tensor: TensorId) -> Result<Option<Role>, ScheduleError> {
        self.roles
            .get(index(tensor.0)?)
            .copied()
            .ok_or(ScheduleError::Internal)
    }

    /// Whether a graph input of a node of layer `layer`, whose layer-0 counterpart is `rep`, is
    /// one rule 3 admits: the shared `positions`, or a weight, class or multiplier vector of the
    /// node's own layer, of the type layer 0's names. A per-layer input is recorded as its kind.
    fn graph_input(
        &mut self,
        input: TensorId,
        rep: TensorId,
        layer: u32,
    ) -> Result<bool, ScheduleError> {
        let same = match (self.role(input)?, self.role(rep)?) {
            (Some(Role::Positions), _) => return Ok(input == rep),
            (Some(Role::Weight(w)), Some(Role::Weight(r)))
            | (Some(Role::Classes(w)), Some(Role::Classes(r)))
            | (Some(Role::Multipliers(w)), Some(Role::Multipliers(r))) => {
                w.ty == r.ty && r.layer == 0 && w.layer == layer
            }
            _ => false,
        };
        if same {
            *self
                .places
                .get_mut(index(input.0)?)
                .ok_or(ScheduleError::Internal)? = Place::Layer { layer, rep };
        }
        Ok(same)
    }

    /// Rules 2 to 5 of 'Layers and kinds', recording each tensor's place.
    fn check_template(&mut self, graph: &Graph) -> Result<(), ScheduleError> {
        // Each layer's outputs are its own, as layer 0's names them.
        for layer in 0..self.layers {
            for position in 0..self.len {
                let id = self.node_id(layer, position)?;
                let node = node_of(graph, id)?;
                let rep = node_of(graph, self.node_id(0, position)?)?;
                if node.outputs().len() != rep.outputs().len() {
                    return Err(ScheduleError::Template { node: id });
                }
                for (&out, &rep_out) in node.outputs().iter().zip(rep.outputs()) {
                    *self
                        .places
                        .get_mut(index(out.0)?)
                        .ok_or(ScheduleError::Internal)? = Place::Layer {
                        layer,
                        rep: rep_out,
                    };
                }
            }
        }
        // Rule 2 and 3, node by node.
        for layer in 0..self.layers {
            let checkpoint = *graph
                .checkpoints()
                .get(index(layer)?)
                .ok_or(ScheduleError::Internal)?;
            for position in 0..self.len {
                let id = self.node_id(layer, position)?;
                let fault = ScheduleError::Template { node: id };
                let node = node_of(graph, id)?;
                let rep = node_of(graph, self.node_id(0, position)?)?;
                if !same_op(node.op(), rep.op(), layer) || node.inputs().len() != rep.inputs().len()
                {
                    return Err(fault);
                }
                for (&out, &rep_out) in node.outputs().iter().zip(rep.outputs()) {
                    let (a, b) = (tensor_of(graph, out)?, tensor_of(graph, rep_out)?);
                    if a.shape != b.shape || kind_class(a.kind) != kind_class(b.kind) {
                        return Err(fault);
                    }
                }
                for (&input, &rep_input) in node.inputs().iter().zip(rep.inputs()) {
                    let ok = if input == checkpoint {
                        rep_input == self.first_checkpoint
                    } else if let Some(producer) = tensor_of(graph, input)?.producer {
                        // An earlier output of its own layer, the same in every layer.
                        let earlier = self
                            .position(producer)
                            .is_some_and(|(l, p)| l == layer && p < position);
                        earlier
                            && matches!(self.place(input)?, Place::Layer { layer: own, rep } if own == layer && rep == rep_input)
                    } else {
                        self.graph_input(input, rep_input, layer)?
                    };
                    if !ok {
                        return Err(fault);
                    }
                }
            }
        }
        // Rule 4: the prologue and the epilogue.
        let last_checkpoint = *graph.checkpoints().last().ok_or(ScheduleError::Internal)?;
        for (i, node) in graph.nodes().iter().enumerate() {
            let id = NodeId(u32::try_from(i).map_err(|_| ScheduleError::Internal)?);
            let in_prologue = id.0 < self.template_start;
            let in_epilogue = id.0 >= self.epilogue_start;
            if !in_prologue && !in_epilogue {
                continue;
            }
            for &input in node.inputs() {
                let producer = tensor_of(graph, input)?.producer;
                let ok = match producer {
                    None => true,
                    Some(p) if in_prologue => p.0 < self.template_start,
                    Some(p) => p.0 >= self.epilogue_start || input == last_checkpoint,
                };
                if !ok {
                    return Err(ScheduleError::Template { node: id });
                }
            }
        }
        // Rule 5: every checkpoint is committed advice.
        for &c in graph.checkpoints() {
            let t = tensor_of(graph, c)?;
            if !matches!(t.kind, Kind::Advice | Kind::Lookup { .. }) {
                return Err(ScheduleError::Template {
                    node: t.producer.unwrap_or(NodeId(0)),
                });
            }
        }
        Ok(())
    }
}

/// Whether `op`, at a node of layer `layer`, is layer 0's `rep` up to the layer's own weights,
/// constants, shift classes and tables.
fn same_op(op: &Op, rep: &Op, layer: u32) -> bool {
    match (op, rep) {
        (Op::MatMulWeight { weight }, Op::MatMulWeight { weight: w }) => {
            weight.ty == w.ty && weight.layer == layer && w.layer == 0
        }
        (Op::ConstMul { .. }, Op::ConstMul { .. }) | (Op::TableMap { .. }, Op::TableMap { .. }) => {
            true
        }
        (Op::Requant { lo, hi, .. }, Op::Requant { lo: l, hi: h, .. }) => lo == l && hi == h,
        (Op::SoftmaxRow { sigma, .. }, Op::SoftmaxRow { sigma: s, .. }) => sigma == s,
        // 'Operations': `Layout::of` refuses these first; no template holds them.
        (
            Op::Sub
            | Op::ConstAdd { .. }
            | Op::Sum { .. }
            | Op::Slice { .. }
            | Op::GatherPublic { .. }
            | Op::Bitwise { .. }
            | Op::BitDecompose { .. }
            | Op::BitFn { .. }
            | Op::AssertEqual,
            _,
        ) => false,
        _ => op == rep,
    }
}

/// A tensor kind's class: a lookup output is one class whatever its table.
fn kind_class(kind: Kind) -> u8 {
    match kind {
        Kind::Advice => 0,
        Kind::Weight => 1,
        Kind::Public => 2,
        Kind::Virtual => 3,
        Kind::Lookup { .. } => 4,
    }
}

// ----- The derivation -------------------------------------------------------------------------

/// A virtual atom waiting for its instance in the next stage.
struct Pending {
    tensor: TensorRef,
    point: Point,
    layers: Layers,
    node: NodeId,
    native: Option<usize>,
}

/// A work item of a claim's walk: a coefficient on a tensor at a point.
struct Item {
    tensor: TensorRef,
    point: Point,
    layers: Layers,
    term: Term,
    /// Outermost layer-0 template `ConstMul` on this path whose constant differs across layers.
    non_uniform_const: Option<NodeId>,
}

/// An instance's relation and size, before its evaluations' claims are derived.
struct Shaped {
    relation: Relation,
    atom: Pending,
    stacked: bool,
    vars: usize,
    degree: usize,
    zero: bool,
}

struct Deriving<'g, 'l> {
    graph: &'g Graph,
    layout: &'l Layout,
    native: &'l [Native],
    deferred: &'l [Deferred],
}

impl Deriving<'_, '_> {
    fn schedule(&self) -> Result<Schedule, ScheduleError> {
        let layout = self.layout;
        // The leaf stage.
        let mut pending = Vec::new();
        let leaf = self.leaf(&mut pending)?;
        let mut leaf_splits = 0usize;
        for claim in &leaf {
            let per_layer = if claim.stacked { layout.layers } else { 1 };
            let values = claim
                .claim
                .parts()
                .checked_mul(index(per_layer)?)
                .ok_or(ScheduleError::Internal)?;
            leaf_splits = leaf_splits
                .checked_add(values)
                .ok_or(ScheduleError::Internal)?;
        }
        // The reduction stages, until one has no instance. Each stage consumes a relation of a
        // chain, so there are fewer stages than nodes.
        let mut stages = Vec::new();
        let mut index_k: u16 = 1;
        loop {
            let zero_checks = if index_k == 1 {
                self.zero_checks()?
            } else {
                Vec::new()
            };
            if pending.is_empty() && zero_checks.is_empty() {
                break;
            }
            if stages.len() > self.graph.nodes().len() {
                return Err(ScheduleError::Internal);
            }
            let (stage, next) = self.stage(index_k, pending, zero_checks)?;
            push(&mut stages, stage)?;
            pending = next;
            index_k = index_k.checked_add(1).ok_or(ScheduleError::Internal)?;
        }
        Ok(Schedule {
            layers: layout.layers,
            layer_vars: layout.layer_vars,
            leaf,
            leaf_splits,
            stages,
            first_checkpoint: layout.first_checkpoint,
            last_output: layout.last_output,
        })
    }

    /// The leaf stage's claims: each gadget input of each site in node order, a template position
    /// once, then the logits.
    fn leaf(&self, pending: &mut Vec<Pending>) -> Result<Vec<LeafClaim>, ScheduleError> {
        let layout = self.layout;
        let mut leaf = Vec::new();
        let stacking = Layers::Point(Point::of(Source::Layer, 0, layout.layer_vars)?);
        for (i, node) in self.graph.nodes().iter().enumerate() {
            let id = NodeId(u32::try_from(i).map_err(|_| ScheduleError::Internal)?);
            if !is_gadget(node.op()) {
                continue;
            }
            let template = match layout.position(id) {
                Some((0, _)) => true,
                Some(_) => continue,
                None => false,
            };
            for (input_index, &input) in node.inputs().iter().enumerate() {
                if self.deferred.contains(&(id, Some(input_index))) {
                    continue;
                }
                let c = u32::try_from(leaf.len()).map_err(|_| ScheduleError::Internal)?;
                let (tensor, layers) = if template {
                    (self.in_template(input)?, stacking.try_clone()?)
                } else {
                    self.outside(input)?
                };
                let vars = self.vars(tensor)?;
                let point = Point::of(Source::Leaf(c), 0, vars)?;
                let claim = self.claim(tensor, point, layers, pending)?;
                push(
                    &mut leaf,
                    LeafClaim {
                        site: id,
                        input: Some(input_index),
                        stacked: template,
                        claim,
                    },
                )?;
            }
            self.native_pending(id, pending)?;
        }
        // The logits: the LM head's output.
        let mut logits = None;
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if let Op::MatMulWeight { weight } = node.op()
                && weight.ty == WeightType::LmHead
            {
                let id = NodeId(u32::try_from(i).map_err(|_| ScheduleError::Internal)?);
                if logits.is_some() || id.0 < layout.epilogue_start {
                    return Err(ScheduleError::Template { node: id });
                }
                logits = Some((id, *node.outputs().first().ok_or(ScheduleError::Internal)?));
            }
        }
        let (site, output) = logits.ok_or(ScheduleError::Internal)?;
        if !self.deferred.contains(&(site, None)) {
            let c = u32::try_from(leaf.len()).map_err(|_| ScheduleError::Internal)?;
            let tensor = TensorRef::Single(output);
            let point = Point::of(Source::Leaf(c), 0, self.vars(tensor)?)?;
            let claim = self.claim(tensor, point, Layers::None, pending)?;
            push(
                &mut leaf,
                LeafClaim {
                    site,
                    input: None,
                    stacked: false,
                    claim,
                },
            )?;
        }
        self.native_pending(site, pending)?;
        Ok(leaf)
    }

    fn native_pending(
        &self,
        node: NodeId,
        pending: &mut Vec<Pending>,
    ) -> Result<(), ScheduleError> {
        for (index, _) in self
            .native
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node == node && !n.zero)
        {
            let shaped = self.native_shape(index)?;
            push(pending, shaped.atom)?;
        }
        Ok(())
    }

    /// Stage 1's checks in node order: the four original OneHot relations, then committed
    /// linear outputs. The lowering's only equality check is `c_0`'s ConstMul
    /// (RFC-0037 'The committed first checkpoint'); any other is refused.
    fn zero_checks(&self) -> Result<Vec<Shaped>, ScheduleError> {
        let mut checks = Vec::new();
        for (i, node) in self.graph.nodes().iter().enumerate() {
            let id = NodeId(u32::try_from(i).map_err(|_| ScheduleError::Internal)?);
            if matches!(node.op(), Op::OneHot { .. }) {
                let &[out] = node.outputs() else {
                    return Err(ScheduleError::Internal);
                };
                if self.layout.position(id).is_some() {
                    return Err(ScheduleError::Unsupported { node: id });
                }
                let output = TensorRef::Single(out);
                let shape = dims(self.graph, out)?;
                let &[positions, vocab] = shape.as_slice() else {
                    return Err(ScheduleError::Internal);
                };
                let vocab_vars = axis_vars(vocab)?;
                let token_replaced = self.deferred.contains(&(id, Some(0)));
                if token_replaced {
                    let &[token] = node.inputs() else {
                        return Err(ScheduleError::Internal);
                    };
                    let row_vars = axis_vars(positions)?;
                    let vars = vocab_vars
                        .checked_add(row_vars)
                        .ok_or(ScheduleError::Internal)?;
                    if !self.native.iter().any(|n| {
                        let [h, t] = n.outputs.as_slice() else {
                            return false;
                        };
                        n.node == id
                            && n.zero
                            && n.degree == 2
                            && n.vars == vars
                            && n.evaluations == 2
                            && h.tensor == out
                            && h.from == 0
                            && h.vars == vars
                            && t.tensor == token
                            && t.from == vocab_vars
                            && t.vars == row_vars
                    }) {
                        return Err(ScheduleError::Internal);
                    }
                }
                let slots = slotted(self.graph)?;
                if let Some(ref s) = slots
                    && positions != s.token_len()
                {
                    return Err(ScheduleError::Internal);
                }
                for relation in [
                    Relation::Booleanity,
                    Relation::Vocabulary { vocab, vocab_vars },
                    Relation::OneHotCount {
                        vocab_vars,
                        positions,
                        slots,
                    },
                    Relation::OneHotIndex { vocab_vars },
                ] {
                    if token_replaced && matches!(relation, Relation::OneHotIndex { .. }) {
                        continue;
                    }
                    let degree = if matches!(
                        relation,
                        Relation::OneHotCount { .. } | Relation::OneHotIndex { .. }
                    ) {
                        2
                    } else {
                        3
                    };
                    push(
                        &mut checks,
                        Shaped {
                            relation,
                            atom: Pending {
                                tensor: output,
                                point: Point::default(),
                                layers: Layers::None,
                                node: id,
                                native: None,
                            },
                            stacked: false,
                            vars: self.vars(output)?,
                            degree,
                            zero: true,
                        },
                    )?;
                }
                continue;
            }
            if !is_linear(node.op()) && !is_relation(node.op()) {
                continue;
            }
            for &out in node.outputs() {
                if tensor_of(self.graph, out)?.kind == Kind::Virtual {
                    continue;
                }
                let (Op::ConstMul { c }, [_], None) =
                    (node.op(), node.inputs(), self.layout.position(id))
                else {
                    return Err(ScheduleError::Unsupported { node: id });
                };
                let output = TensorRef::Single(out);
                push(
                    &mut checks,
                    Shaped {
                        relation: Relation::Equality {
                            constant: self.constant(*c)?,
                        },
                        atom: Pending {
                            tensor: output,
                            point: Point::default(),
                            layers: Layers::None,
                            node: id,
                            native: None,
                        },
                        stacked: false,
                        vars: self.vars(output)?,
                        degree: 2,
                        zero: true,
                    },
                )?;
            }
        }
        for (index, spec) in self.native.iter().enumerate().filter(|(_, n)| n.zero) {
            let shaped = self.native_shape(index)?;
            let at = checks.partition_point(|s| s.atom.node.0 <= spec.node.0);
            reserve(&mut checks, 1)?;
            checks.insert(at, shaped);
        }
        Ok(checks)
    }

    fn native_shape(&self, index: usize) -> Result<Shaped, ScheduleError> {
        let spec = self.native.get(index).ok_or(ScheduleError::Internal)?;
        let definition = node_of(self.graph, spec.node)?;
        let stacked = self.layout.position(spec.node).is_some();
        let tensor = *definition
            .outputs()
            .first()
            .ok_or(ScheduleError::Internal)?;
        let (tensor, layers) = if stacked {
            (
                self.in_template(tensor)?,
                Layers::Point(Point::of(Source::Layer, 0, self.layout.layer_vars)?),
            )
        } else {
            self.outside(tensor)?
        };
        Ok(Shaped {
            relation: Relation::Native {
                tag: spec.tag,
                kind: spec.kind,
                evaluations: spec.evaluations,
                zero: spec.zero,
            },
            atom: Pending {
                tensor,
                point: Point::default(),
                layers,
                node: spec.node,
                native: Some(index),
            },
            stacked,
            vars: spec.vars,
            degree: spec.degree,
            zero: spec.zero,
        })
    }

    /// Stage `k` from the virtual atoms stage `k − 1` left and, in stage 1, the zero-checks; and
    /// the virtual atoms it leaves.
    fn stage(
        &self,
        k: u16,
        pending: Vec<Pending>,
        zero_checks: Vec<Shaped>,
    ) -> Result<(Stage, Vec<Pending>), ScheduleError> {
        let lambda = self.layout.layer_vars;
        // The instances' relations and sizes, in order.
        let mut shapes = Vec::new();
        reserve(&mut shapes, pending.len().saturating_add(zero_checks.len()))?;
        for atom in pending {
            if let Some(index) = atom.native {
                shapes.push(self.native_shape(index)?);
                continue;
            }
            let (relation, own, degree) = self.relation(&atom)?;
            let stacked = matches!(atom.layers, Layers::Point(_));
            let layer_vars = if stacked { lambda } else { 0 };
            let vars = own.checked_add(layer_vars).ok_or(ScheduleError::Internal)?;
            // A matmul is of degree 3 in its stacking variables (RFC-0010 'Relations').
            let degree = match relation {
                Relation::MatMulWeight { .. } if layer_vars > 0 => 3,
                Relation::MatMulAct { lead_vars, .. } if lead_vars > 0 || layer_vars > 0 => 3,
                _ => degree,
            };
            shapes.push(Shaped {
                relation,
                atom,
                stacked,
                vars,
                degree,
                zero: false,
            });
        }
        shapes.extend(zero_checks);
        let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
        let degree = shapes.iter().map(|s| s.degree).max().unwrap_or(1);
        let zero_vars = shapes
            .iter()
            .filter(|s| s.zero)
            .map(|s| s.vars)
            .max()
            .unwrap_or(0);

        // Each instance's evaluations, at its last n_i challenges.
        let mut next = Vec::new();
        let mut instances = Vec::new();
        reserve(&mut instances, shapes.len())?;
        let mut splits = 0usize;
        for shaped in shapes {
            let Shaped {
                relation,
                atom,
                stacked,
                vars,
                degree,
                zero,
            } = shaped;
            let start = rounds.checked_sub(vars).ok_or(ScheduleError::Internal)?;
            let at = Point::of(Source::Rounds(k), start, vars)?;
            let own_vars = if stacked {
                vars.checked_sub(lambda).ok_or(ScheduleError::Internal)?
            } else {
                vars
            };
            let own = at.slice(0, own_vars)?;
            let layers = if stacked {
                Layers::Point(at.slice(own_vars, lambda)?)
            } else {
                Layers::None
            };
            let zero_point = if zero {
                let from = zero_vars.checked_sub(vars).ok_or(ScheduleError::Internal)?;
                Point::of(Source::Zero(k), from, vars)?
            } else {
                Point::default()
            };
            let targets = if let Some(index) = atom.native {
                let spec = self.native.get(index).ok_or(ScheduleError::Internal)?;
                let mut out = Vec::new();
                reserve(&mut out, spec.outputs.len())?;
                for output in &spec.outputs {
                    out.push((output.tensor, own.slice(output.from, output.vars)?));
                }
                out
            } else {
                self.evaluations(&relation, &atom, &own)?
            };
            let mut evals = Vec::new();
            reserve(&mut evals, targets.len())?;
            for (input, point) in targets {
                let (tensor, input_layers) = if stacked {
                    (self.in_template(input)?, layers.try_clone()?)
                } else {
                    self.outside(input)?
                };
                if point.len() != self.vars(tensor)? {
                    return Err(ScheduleError::Internal);
                }
                let claim = self.claim(tensor, point, input_layers, &mut next)?;
                splits = splits
                    .checked_add(claim.parts())
                    .ok_or(ScheduleError::Internal)?;
                evals.push(claim);
            }
            instances.push(Instance {
                relation,
                node: atom.node,
                tensor: atom.tensor,
                stacked,
                vars,
                degree,
                point: atom.point,
                layers: atom.layers,
                zero: zero_point,
                evals,
            });
        }
        let degree = degree::stage_degree(crate::sumcheck::Protocol::One, k, &[degree])?;
        let stage = Stage {
            index: k,
            degree,
            rounds,
            layer_vars: if k == 1 { lambda } else { 0 },
            zero_vars,
            instances,
            splits,
        };
        Ok((stage, next))
    }

    /// The relation of a virtual atom's producer, its own variables and its degree before
    /// stacking.
    fn relation(&self, atom: &Pending) -> Result<(Relation, usize, usize), ScheduleError> {
        let unsupported = ScheduleError::Unsupported { node: atom.node };
        let node = node_of(self.graph, atom.node)?;
        let output = *node.outputs().first().ok_or(ScheduleError::Internal)?;
        let out_dims = dims(self.graph, output)?;
        match (node.op(), node.inputs()) {
            (Op::MatMulWeight { .. }, &[x, w]) => {
                let w_dims = dims(self.graph, w)?;
                let x_dims = dims(self.graph, x)?;
                let (&[out, inner], Some(&x_inner)) = (w_dims.as_slice(), x_dims.last()) else {
                    return Err(unsupported);
                };
                if x_inner != inner || out_dims.last() != Some(&out) {
                    return Err(unsupported);
                }
                let in_vars = axis_vars(inner)?;
                Ok((
                    Relation::MatMulWeight {
                        in_vars,
                        out_vars: axis_vars(out)?,
                    },
                    in_vars,
                    2,
                ))
            }
            (Op::MatMulAct { transpose_rhs }, &[a, b]) => {
                let a_dims = dims(self.graph, a)?;
                let b_dims = dims(self.graph, b)?;
                let rank = a_dims.len();
                let lead = rank.checked_sub(2).ok_or(unsupported)?;
                let m = *a_dims.get(lead).ok_or(unsupported)?;
                let k = *a_dims.get(lead.saturating_add(1)).ok_or(unsupported)?;
                let n = *out_dims.last().ok_or(unsupported)?;
                if let Some(s) = self.graph.slots().filter(|s| s.lengths().len() >= 2) {
                    let b_lead_ok = b_dims.len() == rank
                        && b_dims.get(..lead) == a_dims.get(..lead)
                        && out_dims.len() == rank
                        && out_dims.get(..lead) == a_dims.get(..lead)
                        && out_dims.get(lead) == Some(&s.token_len());
                    let shape_ok = if *transpose_rhs {
                        m == s.token_len()
                            && n == s.n_max()
                            && b_dims.get(lead) == Some(&s.token_len())
                            && b_dims.get(lead.saturating_add(1)) == Some(&k)
                    } else {
                        m == s.token_len()
                            && k == s.n_max()
                            && b_dims.get(lead) == Some(&s.token_len())
                            && b_dims.get(lead.saturating_add(1)) == Some(&n)
                    };
                    if !b_lead_ok || !shape_ok {
                        return Err(unsupported);
                    }
                    let slot_vars = index(s.slot_bits())?;
                    let lead_vars = index(s.request_bits())?
                        .checked_add(sum_vars(a_dims.get(..lead).ok_or(unsupported)?)?)
                        .ok_or(ScheduleError::Internal)?;
                    let (k_vars, n_vars) = if *transpose_rhs {
                        (axis_vars(k)?, slot_vars)
                    } else {
                        (slot_vars, axis_vars(n)?)
                    };
                    Ok((
                        Relation::MatMulAct {
                            transpose_rhs: *transpose_rhs,
                            k_vars,
                            m_vars: slot_vars,
                            n_vars,
                            lead_vars,
                        },
                        k_vars
                            .checked_add(lead_vars)
                            .ok_or(ScheduleError::Internal)?,
                        2,
                    ))
                } else {
                    let lead_vars = sum_vars(a_dims.get(..lead).ok_or(unsupported)?)?;
                    let k_vars = axis_vars(k)?;
                    Ok((
                        Relation::MatMulAct {
                            transpose_rhs: *transpose_rhs,
                            k_vars,
                            m_vars: axis_vars(m)?,
                            n_vars: axis_vars(n)?,
                            lead_vars,
                        },
                        k_vars
                            .checked_add(lead_vars)
                            .ok_or(ScheduleError::Internal)?,
                        2,
                    ))
                }
            }
            (Op::Mul { broadcast }, &[a, b]) => {
                let a_dims = dims(self.graph, a)?;
                let b_dims = dims(self.graph, b)?;
                let total = sum_vars(&a_dims)?;
                let b_vars = sum_vars(&b_dims)?;
                let b_from = match broadcast {
                    // b's axes are a's leading ones, the highest coordinates.
                    Broadcast::Trailing => {
                        if a_dims.get(..b_dims.len()) != Some(b_dims.as_slice()) {
                            return Err(unsupported);
                        }
                        total.checked_sub(b_vars).ok_or(unsupported)?
                    }
                    // b is a's last axis, the lowest coordinates.
                    Broadcast::Leading => {
                        if b_dims.as_slice()
                            != a_dims.last().map(std::slice::from_ref).unwrap_or(&[])
                        {
                            return Err(unsupported);
                        }
                        0
                    }
                };
                Ok((
                    Relation::Elementwise {
                        broadcast: *broadcast,
                        b_from,
                        b_vars,
                    },
                    total,
                    3,
                ))
            }
            (Op::SumSquares { axis }, &[x]) => {
                let x_dims = dims(self.graph, x)?;
                if usize::from(*axis).checked_add(1) != Some(x_dims.len()) {
                    return Err(unsupported);
                }
                let col = *x_dims.last().ok_or(unsupported)?;
                Ok((
                    Relation::SumSquares {
                        col_vars: axis_vars(col)?,
                    },
                    sum_vars(&x_dims)?,
                    3,
                ))
            }
            (Op::Rotate { .. }, &[x, _]) => {
                let x_dims = dims(self.graph, x)?;
                let (Some(&positions), Some(&width)) = (x_dims.first(), x_dims.last()) else {
                    return Err(unsupported);
                };
                // The pairs differ in one bit: d_head / 2 is a power of two.
                let half = width.checked_div(2).filter(|_| width.is_multiple_of(2));
                if x_dims.len() < 2 || !half.is_some_and(u32::is_power_of_two) {
                    return Err(unsupported);
                }
                let middle = x_dims
                    .get(1..x_dims.len().saturating_sub(1))
                    .ok_or(unsupported)?;
                let slots = slotted(self.graph)?;
                if let Some(ref s) = slots
                    && positions != s.token_len()
                {
                    return Err(unsupported);
                }
                Ok((
                    Relation::Rotate {
                        positions,
                        middle_vars: sum_vars(middle)?,
                        slots,
                    },
                    sum_vars(&x_dims)?,
                    2,
                ))
            }
            _ => Err(unsupported),
        }
    }

    /// The tensors an instance's evaluations name, in its relation's order, and their points
    /// from the instance's own coordinates `own` and its claim's point.
    fn evaluations(
        &self,
        relation: &Relation,
        atom: &Pending,
        own: &Point,
    ) -> Result<Vec<(TensorId, Point)>, ScheduleError> {
        let node = node_of(self.graph, atom.node)?;
        let r = &atom.point;
        let out = match (relation, node.inputs()) {
            (Relation::MatMulWeight { in_vars, out_vars }, &[x, w]) => {
                let r_o = r.slice(0, *out_vars)?;
                let r_t = r.slice(
                    *out_vars,
                    r.len()
                        .checked_sub(*out_vars)
                        .ok_or(ScheduleError::Internal)?,
                )?;
                let r_i = own.slice(0, *in_vars)?;
                vector([
                    (w, Point::concat(&[&r_i, &r_o])?),
                    (x, Point::concat(&[&r_i, &r_t])?),
                ])?
            }
            (
                Relation::MatMulAct {
                    transpose_rhs,
                    k_vars,
                    m_vars,
                    n_vars,
                    lead_vars,
                },
                &[a, b],
            ) => {
                let r_c = r.slice(0, *n_vars)?;
                let r_a = r.slice(*n_vars, *m_vars)?;
                let r_j = own.slice(0, *k_vars)?;
                let r_b = own.slice(*k_vars, *lead_vars)?;
                let b_point = if *transpose_rhs {
                    Point::concat(&[&r_j, &r_c, &r_b])?
                } else {
                    Point::concat(&[&r_c, &r_j, &r_b])?
                };
                vector([(a, Point::concat(&[&r_j, &r_a, &r_b])?), (b, b_point)])?
            }
            (Relation::Elementwise { b_from, b_vars, .. }, &[a, b]) => {
                vector([(a, own.try_clone()?), (b, own.slice(*b_from, *b_vars)?)])?
            }
            (Relation::SumSquares { .. }, &[x]) | (Relation::Rotate { .. }, &[x, _]) => {
                vector([(x, own.try_clone()?)])?
            }
            (
                Relation::Booleanity | Relation::Vocabulary { .. } | Relation::OneHotCount { .. },
                [_],
            ) => {
                let TensorRef::Single(output) = atom.tensor else {
                    return Err(ScheduleError::Internal);
                };
                vector([(output, own.try_clone()?)])?
            }
            (Relation::OneHotIndex { vocab_vars }, [token]) => {
                let TensorRef::Single(output) = atom.tensor else {
                    return Err(ScheduleError::Internal);
                };
                vector([
                    (output, own.try_clone()?),
                    (
                        *token,
                        own.slice(
                            *vocab_vars,
                            own.len()
                                .checked_sub(*vocab_vars)
                                .ok_or(ScheduleError::Internal)?,
                        )?,
                    ),
                ])?
            }
            // The output Ỹ first, then f's input (RFC-0010 'Sumcheck messages', as amended).
            (Relation::Equality { .. }, &[x]) => {
                let TensorRef::Single(output) = atom.tensor else {
                    return Err(ScheduleError::Internal);
                };
                vector([(output, own.try_clone()?), (x, own.try_clone()?)])?
            }
            _ => return Err(ScheduleError::Internal),
        };
        Ok(out)
    }

    /// A claim on `tensor` at `point` with `layers`: its atoms and its route, the virtual atoms it
    /// leaves appended to `pending`.
    fn claim(
        &self,
        tensor: TensorRef,
        point: Point,
        layers: Layers,
        pending: &mut Vec<Pending>,
    ) -> Result<Claim, ScheduleError> {
        let vars = point.len();
        let atoms = self.atoms(tensor, point, layers)?;
        let route = self.route(&atoms, pending)?;
        Ok(Claim {
            tensor,
            vars,
            atoms,
            route,
        })
    }

    /// The atoms of a claim, in atom order, merged.
    fn atoms(
        &self,
        tensor: TensorRef,
        point: Point,
        layers: Layers,
    ) -> Result<Vec<Atom>, ScheduleError> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut stack = vector([Item {
            tensor,
            point,
            layers,
            term: Term {
                constant: vector([Fp::ONE])?,
                lts: Vec::new(),
            },
            non_uniform_const: None,
        }])?;
        // Each step either expands a node, whose inputs come after it, or emits an atom: the
        // walk visits each node of the claim's linear chains once per path.
        let mut steps = 0usize;
        while let Some(item) = stack.pop() {
            steps = steps.checked_add(1).ok_or(ScheduleError::Internal)?;
            if steps
                > self
                    .graph
                    .nodes()
                    .len()
                    .saturating_mul(4)
                    .saturating_add(16)
            {
                return Err(ScheduleError::Internal);
            }
            if let Some(expanded) = self.expand(&item)? {
                // Inputs in their node's order: the first is popped first.
                for next in expanded.into_iter().rev() {
                    push(&mut stack, next)?;
                }
                continue;
            }
            let class = self.class(item.tensor)?;
            if let Some(node) = item.non_uniform_const {
                // RFC-0037 'Claims and atoms', RFC-0068 step 7, RFC-0013 'Committed tensors' and
                // 'Weights' (#556): a per-layer `ConstMul` factor is permitted only on a stacked
                // committed advice atom; any path reaching a virtual, weight or `ChannelParams`
                // atom is refused at the outer non-uniform `ConstMul`.
                let supported_advice = matches!(class, Class::Committed(Commitment::Advice))
                    && matches!(item.layers, Layers::Point(_))
                    && matches!(item.tensor, TensorRef::LayerInput | TensorRef::Stacked(_));
                if !supported_advice {
                    return Err(ScheduleError::Unsupported { node });
                }
            }
            let merged = atoms.iter_mut().find(|a| {
                a.tensor == item.tensor && a.point == item.point && a.layers == item.layers
            });
            match merged {
                Some(atom) => push(&mut atom.coefficient.terms, item.term)?,
                None => push(
                    &mut atoms,
                    Atom {
                        tensor: item.tensor,
                        point: item.point,
                        layers: item.layers,
                        coefficient: Coefficient {
                            terms: vector([item.term])?,
                        },
                        class,
                    },
                )?,
            }
        }
        if atoms.is_empty() {
            return Err(ScheduleError::Internal);
        }
        Ok(atoms)
    }

    /// The terms on a linear node's inputs, when `item`'s tensor is the virtual output of one;
    /// `None` for an atom.
    fn expand(&self, item: &Item) -> Result<Option<Vec<Item>>, ScheduleError> {
        let (id, template) = match item.tensor {
            TensorRef::LayerInput => return Ok(None),
            TensorRef::Stacked(t) => (t, true),
            TensorRef::Single(t) => (t, false),
        };
        let t = tensor_of(self.graph, id)?;
        let Some(producer) = t.producer.filter(|_| t.kind == Kind::Virtual) else {
            return Ok(None);
        };
        let node = node_of(self.graph, producer)?;
        if !is_linear(node.op()) {
            return Ok(None);
        }
        let unsupported = ScheduleError::Unsupported { node: producer };
        if template && !matches!(item.layers, Layers::Point(_)) {
            return Err(unsupported);
        }
        let input_ref = |input: TensorId| -> Result<(TensorRef, Layers), ScheduleError> {
            if template {
                Ok((self.in_template(input)?, item.layers.try_clone()?))
            } else {
                self.outside(input)
            }
        };
        let out_dims = t.shape.dims();
        let items = match (node.op(), node.inputs()) {
            (Op::Reshape, &[x]) => {
                if !reshape_is_identity(&dims(self.graph, x)?, out_dims)? {
                    return Err(unsupported);
                }
                let (tensor, layers) = input_ref(x)?;
                vector([Item {
                    tensor,
                    point: item.point.try_clone()?,
                    layers,
                    term: item.term.try_clone()?,
                    non_uniform_const: item.non_uniform_const,
                }])?
            }
            (Op::Transpose { perm }, &[x]) => {
                let blocks = axis_blocks(out_dims)?;
                let x_rank = dims(self.graph, x)?.len();
                if perm.len() != x_rank || blocks.len() != x_rank {
                    return Err(unsupported);
                }
                // Input axis a, from the last (lowest) to the first, takes output axis i's block,
                // perm[i] = a.
                let mut axes = Vec::new();
                reserve(&mut axes, x_rank)?;
                for a in (0..x_rank).rev() {
                    let i = perm
                        .iter()
                        .position(|&p| usize::from(p) == a)
                        .ok_or(unsupported)?;
                    axes.push(*blocks.get(i).ok_or(unsupported)?);
                }
                let (tensor, layers) = input_ref(x)?;
                vector([Item {
                    tensor,
                    point: item.point.blocks(&axes)?,
                    layers,
                    term: item.term.try_clone()?,
                    non_uniform_const: item.non_uniform_const,
                }])?
            }
            (Op::Expand { axis, len }, &[x]) => {
                let blocks = axis_blocks(out_dims)?;
                let axis = usize::from(*axis);
                let dropped = *blocks.get(axis).ok_or(unsupported)?;
                let mut kept = Vec::new();
                reserve(&mut kept, blocks.len())?;
                for a in (0..blocks.len()).rev().filter(|&a| a != axis) {
                    kept.push(*blocks.get(a).ok_or(unsupported)?);
                }
                let mut term = item.term.try_clone()?;
                let cube = 1u64
                    .checked_shl(u32::try_from(dropped.1).map_err(|_| ScheduleError::Internal)?)
                    .ok_or(ScheduleError::Internal)?;
                if u64::from(*len) != cube {
                    push(
                        &mut term.lts,
                        (item.point.slice(dropped.0, dropped.1)?, u64::from(*len)),
                    )?;
                }
                let (tensor, layers) = input_ref(x)?;
                vector([Item {
                    tensor,
                    point: item.point.blocks(&kept)?,
                    layers,
                    term,
                    non_uniform_const: item.non_uniform_const,
                }])?
            }
            (Op::ConstMul { c }, &[x]) => {
                let (constant, non_uniform_const) = if template {
                    let (_, position) = self.layout.position(producer).ok_or(unsupported)?;
                    let mut values = Vec::new();
                    reserve(&mut values, index(self.layout.layers)?)?;
                    for layer in 0..self.layout.layers {
                        let node = node_of(self.graph, self.layout.node_id(layer, position)?)?;
                        let Op::ConstMul { c } = node.op() else {
                            return Err(ScheduleError::Internal);
                        };
                        values.push(self.constant(*c)?);
                    }
                    let first = *values.first().ok_or(ScheduleError::Internal)?;
                    if values.iter().all(|&v| v == first) {
                        (vector([first])?, item.non_uniform_const)
                    } else {
                        (values, item.non_uniform_const.or(Some(producer)))
                    }
                } else {
                    (vector([self.constant(*c)?])?, item.non_uniform_const)
                };
                let mut term = item.term.try_clone()?;
                term.constant = times(&term.constant, &constant)?;
                let (tensor, layers) = input_ref(x)?;
                vector([Item {
                    tensor,
                    point: item.point.try_clone()?,
                    layers,
                    term,
                    non_uniform_const,
                }])?
            }
            (Op::Add, &[a, b]) => {
                let (ta, la) = input_ref(a)?;
                let (tb, lb) = input_ref(b)?;
                vector([
                    Item {
                        tensor: ta,
                        point: item.point.try_clone()?,
                        layers: la,
                        term: item.term.try_clone()?,
                        non_uniform_const: item.non_uniform_const,
                    },
                    Item {
                        tensor: tb,
                        point: item.point.try_clone()?,
                        layers: lb,
                        term: item.term.try_clone()?,
                        non_uniform_const: item.non_uniform_const,
                    },
                ])?
            }
            _ => return Err(unsupported),
        };
        Ok(Some(items))
    }

    /// A claim's route from its atoms; its virtual atoms join `pending`.
    fn route(&self, atoms: &[Atom], pending: &mut Vec<Pending>) -> Result<Route, ScheduleError> {
        let take = |atom: &Atom, pending: &mut Vec<Pending>| -> Result<usize, ScheduleError> {
            let Class::Virtual { node } = atom.class else {
                return Err(ScheduleError::Internal);
            };
            if matches!(atom.layers, Layers::One(_)) {
                return Err(ScheduleError::Unsupported { node });
            }
            let at = pending.len();
            push(
                pending,
                Pending {
                    tensor: atom.tensor,
                    point: atom.point.try_clone()?,
                    layers: atom.layers.try_clone()?,
                    node,
                    native: None,
                },
            )?;
            Ok(at)
        };
        if let [atom] = atoms
            && matches!(atom.class, Class::Virtual { .. })
            && atom.coefficient.is_one()
        {
            return Ok(Route::Instance(take(atom, pending)?));
        }
        let first = atoms.first().ok_or(ScheduleError::Internal)?;
        let one_group = atoms.iter().all(|a| {
            matches!((a.class, first.class), (Class::Committed(x), Class::Committed(y)) if x == y)
                && a.point == first.point
        });
        if one_group {
            return Ok(Route::Pool);
        }
        let mut parts: Vec<Part> = Vec::new();
        for (i, atom) in atoms.iter().enumerate() {
            match atom.class {
                Class::Virtual { .. } => {
                    let instance = take(atom, pending)?;
                    push(&mut parts, Part::Virtual { atom: i, instance })?;
                }
                Class::Committed(commitment) => {
                    let group = parts.iter_mut().find_map(|part| match part {
                        Part::Group { atoms: members } => {
                            let lead = members.first().and_then(|&m| atoms.get(m))?;
                            (lead.class == Class::Committed(commitment) && lead.point == atom.point)
                                .then_some(members)
                        }
                        Part::Virtual { .. } => None,
                    });
                    match group {
                        Some(members) => push(members, i)?,
                        None => push(
                            &mut parts,
                            Part::Group {
                                atoms: vector([i])?,
                            },
                        )?,
                    }
                }
            }
        }
        Ok(Route::Split(parts))
    }

    /// What a tensor of the schedule is: virtual with its producer, or committed with its
    /// commitment; a public tensor is refused.
    fn class(&self, tensor_ref: TensorRef) -> Result<Class, ScheduleError> {
        let id = match tensor_ref {
            TensorRef::LayerInput => return Ok(Class::Committed(Commitment::Advice)),
            TensorRef::Stacked(t) | TensorRef::Single(t) => t,
        };
        let t = tensor_of(self.graph, id)?;
        match t.kind {
            Kind::Advice | Kind::Lookup { .. } => Ok(Class::Committed(Commitment::Advice)),
            Kind::Weight => match self.layout.role(id)? {
                Some(Role::Weight(w)) => Ok(Class::Committed(Commitment::Weight(w.ty))),
                Some(Role::Classes(_) | Role::Multipliers(_)) => {
                    Ok(Class::Committed(Commitment::ChannelParams))
                }
                _ => Err(ScheduleError::Internal),
            },
            Kind::Virtual => {
                let producer = t.producer.ok_or(ScheduleError::Internal)?;
                if is_relation(node_of(self.graph, producer)?.op()) {
                    Ok(Class::Virtual { node: producer })
                } else {
                    Err(ScheduleError::Unsupported { node: producer })
                }
            }
            Kind::Public => Err(ScheduleError::Unsupported {
                node: t.producer.unwrap_or(NodeId(0)),
            }),
        }
    }

    /// A template node's input, as the kind layer 0 names: the layer input for `c_0`, a stacked
    /// kind for layer 0's outputs and parameters, `Single` for a shared input.
    fn in_template(&self, input: TensorId) -> Result<TensorRef, ScheduleError> {
        if input == self.layout.first_checkpoint {
            return Ok(TensorRef::LayerInput);
        }
        match self.layout.place(input)? {
            Place::Layer { layer: 0, rep } => Ok(TensorRef::Stacked(rep)),
            Place::Layer { .. } => Err(ScheduleError::Internal),
            Place::Outside => Ok(TensorRef::Single(input)),
        }
    }

    /// A tensor read outside the layers: itself, or the one layer's slice of a stacked kind
    /// (`c_L`).
    fn outside(&self, input: TensorId) -> Result<(TensorRef, Layers), ScheduleError> {
        match self.layout.place(input)? {
            Place::Outside => Ok((TensorRef::Single(input), Layers::None)),
            Place::Layer { layer, rep } => Ok((TensorRef::Stacked(rep), Layers::One(layer))),
        }
    }

    /// A tensor's variables, without the layer index.
    fn vars(&self, tensor_ref: TensorRef) -> Result<usize, ScheduleError> {
        let id = match tensor_ref {
            TensorRef::LayerInput => self.layout.first_checkpoint,
            TensorRef::Stacked(t) | TensorRef::Single(t) => t,
        };
        let d = dims(self.graph, id)?;
        match (self.graph.attention_key_bits(id), d.split_last()) {
            (Some(kb), Some((_, prefix))) => sum_vars(prefix)?
                .checked_add(index(kb)?)
                .ok_or(ScheduleError::Internal),
            _ => sum_vars(&d),
        }
    }

    /// Constant `c` of the graph, of one value, in the field. 'Operations' refuses a constant
    /// per channel before the derivation starts.
    fn constant(&self, c: u32) -> Result<Fp, ScheduleError> {
        match self.graph.constant(c) {
            Some(&[value]) => Ok(signed(value)),
            _ => Err(ScheduleError::Internal),
        }
    }
}

/// What the schedule makes of an operation: a gadget, a site of the leaf stage; linear, its
/// claims mapped to its inputs' (RFC-0010 'Relations', `Linear`; RFC-0037 'Head layouts'); or a
/// sumcheck relation that proves its output. The anchor circuits' operations are none of these
/// yet: `Layout::of` refuses them first ('Operations').
#[derive(Clone, Copy, PartialEq, Eq)]
enum Treatment {
    Gadget,
    Linear,
    Relation,
    Refused,
}

/// The treatment of `op`, with an arm for every operation.
fn treatment(op: &Op) -> Treatment {
    match op {
        Op::OneHot { .. }
        | Op::Requant { .. }
        | Op::RsqrtRow { .. }
        | Op::SoftmaxRow { .. }
        | Op::TableMap { .. } => Treatment::Gadget,
        Op::Reshape | Op::Transpose { .. } | Op::Expand { .. } | Op::ConstMul { .. } | Op::Add => {
            Treatment::Linear
        }
        Op::MatMulWeight { .. }
        | Op::MatMulAct { .. }
        | Op::Mul { .. }
        | Op::SumSquares { .. }
        | Op::Rotate { .. } => Treatment::Relation,
        Op::Sub
        | Op::ConstAdd { .. }
        | Op::Sum { .. }
        | Op::Slice { .. }
        | Op::GatherPublic { .. }
        | Op::Bitwise { .. }
        | Op::BitDecompose { .. }
        | Op::BitFn { .. }
        | Op::AssertEqual => Treatment::Refused,
    }
}

/// Whether an operation is gadget-backed: a site of the leaf stage.
fn is_gadget(op: &Op) -> bool {
    treatment(op) == Treatment::Gadget
}

/// Whether an operation is linear in its inputs, its claims mapped to its inputs'.
fn is_linear(op: &Op) -> bool {
    treatment(op) == Treatment::Linear
}

/// Whether an operation's output is proven by a sumcheck relation.
fn is_relation(op: &Op) -> bool {
    treatment(op) == Treatment::Relation
}

/// Whether a `Reshape` from `from` to `to` maps coordinates as the identity: in each group of
/// axes that merge into one, every axis but the outermost has a length that is a power of two,
/// axes of length 1 aside.
fn reshape_is_identity(from: &[u32], to: &[u32]) -> Result<bool, ScheduleError> {
    let mut a = Vec::new();
    reserve(&mut a, from.len())?;
    a.extend(from.iter().filter(|&&d| d != 1).map(|&d| u64::from(d)));
    let mut b = Vec::new();
    reserve(&mut b, to.len())?;
    b.extend(to.iter().filter(|&&d| d != 1).map(|&d| u64::from(d)));
    let (mut i, mut j) = (a.len(), b.len());
    while i > 0 && j > 0 {
        let (i_end, j_end) = (i, j);
        i = i.saturating_sub(1);
        j = j.saturating_sub(1);
        let (Some(&first_a), Some(&first_b)) = (a.get(i), b.get(j)) else {
            return Ok(false);
        };
        let (mut pa, mut pb) = (first_a, first_b);
        while pa != pb {
            if pa < pb {
                let Some(next) = i.checked_sub(1).and_then(|k| a.get(k)) else {
                    return Ok(false);
                };
                let Some(product) = pa.checked_mul(*next) else {
                    return Ok(false);
                };
                pa = product;
                i = i.saturating_sub(1);
            } else {
                let Some(next) = j.checked_sub(1).and_then(|k| b.get(k)) else {
                    return Ok(false);
                };
                let Some(product) = pb.checked_mul(*next) else {
                    return Ok(false);
                };
                pb = product;
                j = j.saturating_sub(1);
            }
        }
        // Every axis of the group but its outermost is a power of two.
        let inner_a = a.get(i.saturating_add(1)..i_end).unwrap_or(&[]);
        let inner_b = b.get(j.saturating_add(1)..j_end).unwrap_or(&[]);
        if !inner_a.iter().chain(inner_b).all(|d| d.is_power_of_two()) {
            return Ok(false);
        }
    }
    Ok(i == 0 && j == 0)
}

/// The `(first coordinate, variables)` block of each axis of a tensor, outermost first; the last
/// axis takes the lowest coordinates.
fn axis_blocks(dims: &[u32]) -> Result<Vec<(usize, usize)>, ScheduleError> {
    let mut blocks = Vec::new();
    reserve(&mut blocks, dims.len())?;
    let mut from = 0usize;
    for &d in dims.iter().rev() {
        let vars = axis_vars(d)?;
        blocks.push((from, vars));
        from = from.checked_add(vars).ok_or(ScheduleError::Internal)?;
    }
    blocks.reverse();
    Ok(blocks)
}

/// `⌈log2 d⌉`, an axis's variables.
fn axis_vars(d: u32) -> Result<usize, ScheduleError> {
    ceil_log2(u64::from(d))
}

/// The variables of a tensor of axes `dims`.
fn sum_vars(dims: &[u32]) -> Result<usize, ScheduleError> {
    let mut total = 0usize;
    for &d in dims {
        total = total
            .checked_add(axis_vars(d)?)
            .ok_or(ScheduleError::Internal)?;
    }
    Ok(total)
}

/// `⌈log2 x⌉` for `x ≥ 1`.
fn ceil_log2(x: u64) -> Result<usize, ScheduleError> {
    let below = x.checked_sub(1).ok_or(ScheduleError::Internal)?;
    let bits = u64::BITS
        .checked_sub(below.leading_zeros())
        .ok_or(ScheduleError::Internal)?;
    usize::try_from(bits).map_err(|_| ScheduleError::Internal)
}

/// A constant's per-layer product: one value broadcasts over the layers.
fn times(a: &[Fp], b: &[Fp]) -> Result<Vec<Fp>, ScheduleError> {
    let len = a.len().max(b.len());
    let mut out = Vec::new();
    reserve(&mut out, len)?;
    for l in 0..len {
        let x = if a.len() == 1 { a.first() } else { a.get(l) };
        let y = if b.len() == 1 { b.first() } else { b.get(l) };
        match (x, y) {
            (Some(&x), Some(&y)) => out.push(x * y),
            _ => return Err(ScheduleError::Internal),
        }
    }
    Ok(out)
}

/// A signed constant in the field, centered (RFC-0002 'Signed integers').
fn signed(v: i64) -> Fp {
    let magnitude = Fp::reduce(v.unsigned_abs());
    if v < 0 { -magnitude } else { magnitude }
}

fn node_of(graph: &Graph, id: NodeId) -> Result<&Node, ScheduleError> {
    graph.node(id).ok_or(ScheduleError::Internal)
}

fn slotted(graph: &Graph) -> Result<Option<crate::lowering::Slots>, ScheduleError> {
    match graph.slots().filter(|s| s.lengths().len() >= 2) {
        Some(s) => s.try_clone().map(Some).map_err(|e| match e {
            crate::lowering::LowerError::Memory => ScheduleError::Memory,
            _ => ScheduleError::Internal,
        }),
        None => Ok(None),
    }
}

fn tensor_of(graph: &Graph, id: TensorId) -> Result<&crate::lowering::Tensor, ScheduleError> {
    graph.tensor(id).ok_or(ScheduleError::Internal)
}

fn dims(graph: &Graph, id: TensorId) -> Result<Vec<u32>, ScheduleError> {
    copy(tensor_of(graph, id)?.shape.dims())
}

fn index(i: u32) -> Result<usize, ScheduleError> {
    usize::try_from(i).map_err(|_| ScheduleError::Internal)
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), ScheduleError> {
    values
        .try_reserve(additional)
        .map_err(|_| ScheduleError::Memory)
}

fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), ScheduleError> {
    reserve(values, 1)?;
    values.push(value);
    Ok(())
}

fn vector<T, const N: usize>(values: [T; N]) -> Result<Vec<T>, ScheduleError> {
    let mut out = Vec::new();
    out.try_reserve_exact(N)
        .map_err(|_| ScheduleError::Memory)?;
    out.extend(values);
    Ok(out)
}

fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>, ScheduleError> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len())
        .map_err(|_| ScheduleError::Memory)?;
    out.extend_from_slice(values);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reshape_is_an_identity_when_inner_axes_are_powers_of_two() {
        assert!(reshape_is_identity(&[5, 4, 16], &[5, 64]).unwrap());
        assert!(reshape_is_identity(&[5, 64], &[5, 4, 16]).unwrap());
        assert!(reshape_is_identity(&[2, 2, 5, 16], &[4, 5, 16]).unwrap());
        assert!(reshape_is_identity(&[3, 4, 16], &[3, 64]).unwrap());
        // The outermost axis of a merge group may be any length.
        assert!(reshape_is_identity(&[5, 3, 16], &[5, 48]).unwrap());
        assert!(reshape_is_identity(&[1, 6, 1], &[6]).unwrap());
        // An inner axis that is not a power of two moves coordinates.
        assert!(!reshape_is_identity(&[5, 4, 6], &[5, 24]).unwrap());
        assert!(!reshape_is_identity(&[6, 5], &[30]).unwrap());
        assert!(!reshape_is_identity(&[4, 6], &[6, 4]).unwrap());
        assert!(!reshape_is_identity(&[4, 6], &[25]).unwrap());
    }

    #[test]
    fn points_merge_contiguous_runs_and_slice_across_them() {
        let a = Point::of(Source::Rounds(2), 3, 4).unwrap();
        let b = Point::of(Source::Rounds(2), 7, 2).unwrap();
        let c = Point::of(Source::Leaf(1), 0, 3).unwrap();
        let joined = Point::concat(&[&a, &b, &c]).unwrap();
        assert_eq!(
            joined.runs(),
            [
                Run {
                    source: Source::Rounds(2),
                    start: 3,
                    len: 6
                },
                Run {
                    source: Source::Leaf(1),
                    start: 0,
                    len: 3
                }
            ]
        );
        assert_eq!(joined.len(), 9);
        let middle = joined.slice(5, 3).unwrap();
        assert_eq!(
            middle,
            Point::concat(&[
                &Point::of(Source::Rounds(2), 8, 1).unwrap(),
                &Point::of(Source::Leaf(1), 0, 2).unwrap()
            ])
            .unwrap()
        );
        assert_eq!(joined.slice(8, 2), Err(ScheduleError::Internal));
        assert_eq!(
            joined.blocks(&[(6, 3), (0, 6)]).unwrap(),
            Point::concat(&[&c, &a, &b]).unwrap()
        );
    }

    #[test]
    fn point_run_length_overflow_is_an_internal_error() {
        let mut point = Point::of(Source::Rounds(1), 0, usize::MAX).unwrap();
        assert_eq!(
            point.push(Run {
                source: Source::Rounds(1),
                start: usize::MAX,
                len: 1,
            }),
            Err(ScheduleError::Internal)
        );
        assert_eq!(point.len(), usize::MAX);
    }

    /// A golden model's graph, lowered at `n` tokens.
    fn graph(name: &str, n: u32) -> Graph {
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
        crate::lowering::lower(&public, n).unwrap()
    }

    /// Layer `layer`'s first node, and the number of nodes per layer.
    fn layer_start(graph: &Graph, layer: usize) -> (u32, u32) {
        let producer = |c: TensorId| graph.tensor(c).unwrap().producer.unwrap().0;
        let checkpoints = graph.checkpoints();
        let start = producer(checkpoints[layer]) + 1;
        let len = producer(checkpoints[1]) - producer(checkpoints[0]);
        (start, len)
    }

    /// The position in the layer of its first node whose operation `pick` accepts, after `from`.
    fn position(graph: &Graph, from: u32, pick: impl Fn(&Op) -> bool) -> u32 {
        let (start, len) = layer_start(graph, 0);
        (from..len)
            .find(|&p| pick(graph.node(NodeId(start + p)).unwrap().op()))
            .unwrap()
    }

    fn at(graph: &Graph, layer: usize, position: u32) -> NodeId {
        NodeId(layer_start(graph, layer).0 + position)
    }

    #[test]
    fn the_golden_models_derive() {
        for name in [
            "tiny-2-servable",
            "tiny-2-minimal",
            "tiny-2-affine",
            "tiny-4-servable",
        ] {
            derive(&graph(name, 5)).unwrap();
        }
    }

    /// Rule 2: an operation that differs in one layer.
    #[test]
    fn a_layer_whose_operation_differs_is_refused() {
        let mut g = graph("tiny-2-servable", 4);
        let p = position(&g, 0, |op| matches!(op, Op::Requant { .. }));
        let node = at(&g, 1, p);
        let Op::Requant { shift, lo, hi } = *g.node(node).unwrap().op() else {
            panic!()
        };
        g.set_op(
            node,
            Op::Requant {
                shift,
                lo: lo - 1,
                hi,
            },
        );
        assert_eq!(derive(&g), Err(ScheduleError::Template { node }));
        // A weight of another layer than the node's.
        let mut g = graph("tiny-2-servable", 4);
        let p = position(&g, 0, |op| matches!(op, Op::MatMulWeight { .. }));
        let node = at(&g, 1, p);
        let Op::MatMulWeight { weight } = *g.node(node).unwrap().op() else {
            panic!()
        };
        let weight = crate::lowering::WeightRef { layer: 0, ..weight };
        g.set_op(node, Op::MatMulWeight { weight });
        assert_eq!(derive(&g), Err(ScheduleError::Template { node }));
    }

    /// Rule 3: a node of layer 1 that reads layer 0's tensor, or the prologue's.
    #[test]
    fn a_layer_reading_outside_itself_is_refused() {
        let mut g = graph("tiny-2-servable", 4);
        let node = at(&g, 1, 1);
        let other = g.node(at(&g, 0, 0)).unwrap().outputs()[0];
        g.set_input(node, 0, other);
        assert_eq!(derive(&g), Err(ScheduleError::Template { node }));
        // The first checkpoint read by layer 1 for its own.
        let mut g = graph("tiny-2-servable", 4);
        let node = at(&g, 1, 0);
        let c0 = g.checkpoints()[0];
        g.set_input(node, 0, c0);
        assert_eq!(derive(&g), Err(ScheduleError::Template { node }));
    }

    /// Rule 4: the epilogue reads `c_L` and nothing else of the layers.
    #[test]
    fn the_epilogue_reading_inside_a_layer_is_refused() {
        let mut g = graph("tiny-2-servable", 4);
        let (start, len) = layer_start(&g, 1);
        let epilogue = NodeId(start + len);
        let inside = g.node(at(&g, 1, 2)).unwrap().outputs()[0];
        g.set_input(epilogue, 0, inside);
        assert_eq!(derive(&g), Err(ScheduleError::Template { node: epilogue }));
    }

    /// Rule 5: every checkpoint is committed.
    #[test]
    fn a_checkpoint_that_is_not_committed_is_refused() {
        let mut g = graph("tiny-2-servable", 4);
        let checkpoints = g.checkpoints().to_vec();
        for &c in &checkpoints[1..] {
            g.set_kind(c, Kind::Virtual);
        }
        let producer = g.tensor(checkpoints[1]).unwrap().producer.unwrap();
        assert_eq!(derive(&g), Err(ScheduleError::Template { node: producer }));
    }

    /// RFC-0037 'Head layouts': a `Reshape` that moves coordinates, in every layer.
    #[test]
    fn a_reshape_that_moves_coordinates_is_unsupported() {
        let mut g = graph("tiny-2-servable", 5);
        let p = position(&g, 0, |op| matches!(op, Op::Rotate { .. }));
        for layer in 0..2 {
            let rotated = g.node(at(&g, layer, p)).unwrap().outputs()[0];
            g.set_shape(rotated, &[5, 6, 10]);
        }
        let reshape = at(&g, 0, p + 1);
        assert_eq!(g.node(reshape).unwrap().op(), &Op::Reshape);
        assert_eq!(
            derive(&g),
            Err(ScheduleError::Unsupported { node: reshape })
        );
    }

    #[test]
    fn six_and_three_heads_derive_and_non_power_of_two_head_widths_reject() {
        let file = tv_testkit::read_vectors("claims/claim-lists.json");
        let case = file
            .get("cases")
            .and_then(tv_testkit::Json::as_array)
            .unwrap()
            .iter()
            .find(|c| c.get("name").and_then(tv_testkit::Json::as_str) == Some("tiny-2-servable"))
            .unwrap();
        let bytes = tv_testkit::hex(case.get("hex").unwrap().as_str().unwrap());
        let base =
            crate::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();

        let mut six = base.clone();
        six.arch.n_heads = 6;
        six.arch.n_kv_heads = 3;
        six.arch.d_model = 96;
        let public = crate::validation::check_public(&six).unwrap();
        for n in [1, 3, 5] {
            let lowered = crate::lowering::lower(&public, n).unwrap();
            derive(&lowered).unwrap();
        }

        for (heads, kv_heads, d_head, bad_node) in [
            (2, 1, 6, NodeId(12)),
            (6, 3, 48, NodeId(12)),
            (1, 1, 6, NodeId(11)),
        ] {
            let mut claims = base.clone();
            claims.arch.n_heads = heads;
            claims.arch.n_kv_heads = kv_heads;
            claims.arch.d_head = d_head;
            claims.arch.d_model = heads * d_head;
            let public = crate::validation::check_public(&claims).unwrap();
            let lowered = crate::lowering::lower(&public, 3).unwrap();
            assert_eq!(
                derive(&lowered),
                Err(ScheduleError::Unsupported { node: bad_node })
            );
        }
    }

    /// RFC-0037 'Claims and atoms' and RFC-0013 (#556): a reduction stage's stacked claim through
    /// a `ConstMul` whose constant differs between layers is refused when the path reaches a
    /// virtual atom (even if a second `ConstMul` cancels the per-layer difference), while a
    /// uniform constant across layers succeeds.
    #[test]
    fn a_per_layer_constant_in_a_stacked_claim_is_unsupported() {
        let base = graph("tiny-2-servable", 4);
        let o_mm = position(
            &base,
            0,
            |op| matches!(op, Op::MatMulWeight { weight } if weight.ty == WeightType::O),
        );
        let c = position(&base, 0, |op| matches!(op, Op::ConstMul { .. }));
        let req = position(&base, c, |op| matches!(op, Op::Requant { .. }));
        let s = position(&base, c, |op| matches!(op, Op::SumSquares { .. }));
        let rewired = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &constant) in constants.iter().enumerate() {
                let acc = g.node(at(&g, layer, o_mm)).unwrap().outputs()[0];
                let sum = g.node(at(&g, layer, c)).unwrap().inputs()[0];
                let doubled = g.node(at(&g, layer, c)).unwrap().outputs()[0];
                // Bypass `c` at stage 0 so only stage 1's `SumSquares` reaches `c` -> `acc`.
                g.set_input(at(&g, layer, req), 0, sum);
                g.set_input(at(&g, layer, c), 0, acc);
                g.set_input(at(&g, layer, s), 0, doubled);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, constant);
            }
            g
        };
        // One constant in every layer: a uniform scalar factor on the virtual atom.
        derive(&rewired([2, 2])).unwrap();
        let constant = at(&base, 0, c);
        assert_eq!(
            derive(&rewired([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );

        // Two non-uniform `ConstMul`s whose product `[2, 3] * [3, 2] = [6, 6]` cancels in `Fp`
        // still cross a non-uniform template `ConstMul` on the path to a virtual atom and are
        // refused at the outer `ConstMul` (whereas `[2, 2] * [3, 3] = [6, 6]` succeeds).
        let c2 = position(&base, c + 1, |op| matches!(op, Op::ConstMul { .. }));
        let mlp_add = position(&base, req + 1, |op| matches!(op, Op::Add));
        let chained = |pairs: [(i64, i64); 2]| {
            let mut g = base.clone();
            for (layer, (inner_k, outer_k)) in pairs.into_iter().enumerate() {
                let acc = g.node(at(&g, layer, o_mm)).unwrap().outputs()[0];
                let inner_out = g.node(at(&g, layer, c)).unwrap().outputs()[0];
                let outer_out = g.node(at(&g, layer, req)).unwrap().outputs()[0];
                let checkpoint = g.checkpoints()[layer];
                let Op::ConstMul { c: inner_idx } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                let Op::ConstMul { c: outer_idx } = *g.node(at(&g, layer, c2)).unwrap().op() else {
                    panic!()
                };
                g.set_input(at(&g, layer, c), 0, acc);
                g.set_constant(inner_idx, inner_k);
                g.set_op(at(&g, layer, req), Op::ConstMul { c: outer_idx });
                g.set_kind(outer_out, Kind::Virtual);
                g.set_input(at(&g, layer, req), 0, inner_out);
                g.set_constant(outer_idx, outer_k);
                g.set_input(at(&g, layer, mlp_add), 0, checkpoint);
            }
            g
        };
        derive(&chained([(2, 3), (2, 3)])).unwrap();
        let outer = at(&base, 0, req);
        assert_eq!(
            derive(&chained([(2, 3), (3, 2)])),
            Err(ScheduleError::Unsupported { node: outer })
        );
    }

    /// RFC-0068 step 7 and RFC-0037 'Claims and atoms' (#556): a stage-0 (leaf-stage) claim whose
    /// linear substitution path crosses a non-uniform per-layer `ConstMul` and reaches a virtual
    /// atom is refused (`ScheduleError::Unsupported`), while a uniform per-layer `ConstMul`
    /// succeeds.
    #[test]
    fn a_per_layer_constant_on_a_leaf_virtual_atom_is_unsupported() {
        let base = graph("tiny-2-servable", 4);
        let o_mm = position(
            &base,
            0,
            |op| matches!(op, Op::MatMulWeight { weight } if weight.ty == WeightType::O),
        );
        let add_pos = position(&base, o_mm, |op| matches!(op, Op::Add));
        let c = position(&base, 0, |op| matches!(op, Op::ConstMul { .. }));
        let constant = at(&base, 0, c);

        // Case 1: direct leaf site -> `ConstMul` -> virtual `MatMulWeight(O)`.
        let direct = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let acc = g.node(at(&g, layer, o_mm)).unwrap().outputs()[0];
                g.set_input(at(&g, layer, c), 0, acc);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
            }
            g
        };
        let ok_schedule = derive(&direct([2, 2])).unwrap();
        let req_site = at(
            &base,
            0,
            position(&base, c, |op| matches!(op, Op::Requant { .. })),
        );
        let leaf_claim = ok_schedule
            .leaf()
            .iter()
            .find(|l| l.site == req_site && l.input == Some(0))
            .unwrap();
        assert_eq!(leaf_claim.claim.atoms.len(), 1);
        assert!(matches!(
            leaf_claim.claim.atoms[0].class,
            Class::Virtual { .. }
        ));
        assert_eq!(
            leaf_claim.claim.atoms[0].coefficient.terms()[0].constant,
            vec![Fp::reduce(2)]
        );
        assert_eq!(
            derive(&direct([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );

        // Case 2: split leaf site -> `ConstMul` -> `Add(X_in, MatMulWeight(O))` where the first
        // branch is committed advice (permitted) and the second is virtual (forbidden).
        let split = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let acc = g.node(at(&g, layer, o_mm)).unwrap().outputs()[0];
                g.set_input(at(&g, layer, add_pos), 1, acc);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
            }
            g
        };
        derive(&split([2, 2])).unwrap();
        assert_eq!(
            derive(&split([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );

        // Case 3: `Add(MatMulWeight(O), update)` (LHS branch virtual, RHS branch committed advice).
        let split_lhs = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let acc = g.node(at(&g, layer, o_mm)).unwrap().outputs()[0];
                g.set_input(at(&g, layer, add_pos), 0, acc);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
            }
            g
        };
        derive(&split_lhs([2, 2])).unwrap();
        assert_eq!(
            derive(&split_lhs([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );

        // Case 4: `ConstMul -> Reshape -> Transpose -> Elementwise (Virtual)` (`AttnOut` reshape).
        let attn_reshape = position(&base, o_mm - 3, |op| matches!(op, Op::Reshape));
        let via_reshape_transpose = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let o = g.node(at(&g, layer, attn_reshape)).unwrap().outputs()[0];
                g.set_input(at(&g, layer, c), 0, o);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
            }
            g
        };
        derive(&via_reshape_transpose([2, 2])).unwrap();
        assert_eq!(
            derive(&via_reshape_transpose([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );

        // Case 5: `ConstMul -> Expand -> Transpose -> Reshape -> MatMulWeight(K)` (`KV` heads).
        let k_mm = position(
            &base,
            0,
            |op| matches!(op, Op::MatMulWeight { weight } if weight.ty == WeightType::K),
        );
        let exp_pos = position(&base, k_mm, |op| matches!(op, Op::Expand { .. }));
        let k_split = position(&base, exp_pos - 2, |op| matches!(op, Op::Reshape));
        let req_pos = position(&base, c, |op| matches!(op, Op::Requant { .. }));
        let s_pos = position(&base, req_pos, |op| matches!(op, Op::SumSquares { .. }));
        let el_pos = position(&base, s_pos, |op| matches!(op, Op::Mul { .. }));
        let mlp_add = position(&base, req_pos + 1, |op| matches!(op, Op::Add));
        let via_expand = |constants: [i64; 2]| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let k_virt = g.node(at(&g, layer, k_mm)).unwrap().outputs()[0];
                let expanded = g.node(at(&g, layer, exp_pos)).unwrap().outputs()[0];
                let exp_dims = g.tensor(expanded).unwrap().shape.dims().to_vec();
                let c_out = g.node(at(&g, layer, c)).unwrap().outputs()[0];
                let req_out = g.node(at(&g, layer, req_pos)).unwrap().outputs()[0];
                let checkpoint = g.checkpoints()[layer];
                g.set_input(at(&g, layer, k_split), 0, k_virt);
                g.set_input(at(&g, layer, c), 0, expanded);
                g.set_shape(c_out, &exp_dims);
                g.set_shape(req_out, &exp_dims);
                g.set_input(at(&g, layer, s_pos), 0, checkpoint);
                g.set_input(at(&g, layer, el_pos), 0, checkpoint);
                g.set_input(at(&g, layer, mlp_add), 0, checkpoint);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
            }
            g
        };
        derive(&via_expand([2, 2])).unwrap();
        assert_eq!(
            derive(&via_expand([2, 3])),
            Err(ScheduleError::Unsupported { node: constant })
        );
    }

    /// RFC-0037 'The claim pool' and RFC-0013 'Weights' (#556): a stacked `Weight` or
    /// `ChannelParams` (`Classes` or `Multipliers`) atom reached through a non-uniform per-layer
    /// `ConstMul` is refused at both stage 0 and stage 1, while a uniform per-layer `ConstMul`
    /// succeeds.
    #[test]
    fn a_per_layer_constant_on_a_weight_or_channel_params_atom_is_unsupported() {
        // 1. Stacked `Commitment::Weight(WeightType::O)` at stage 0 and stage 1.
        let base = graph("tiny-2-servable", 4);
        let o_mm = position(
            &base,
            0,
            |op| matches!(op, Op::MatMulWeight { weight } if weight.ty == WeightType::O),
        );
        let c = position(&base, 0, |op| matches!(op, Op::ConstMul { .. }));
        let req = position(&base, c, |op| matches!(op, Op::Requant { .. }));
        let s = position(&base, c, |op| matches!(op, Op::SumSquares { .. }));
        let constant = at(&base, 0, c);
        let weight_case = |constants: [i64; 2], stage_one: bool| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let w = g.node(at(&g, layer, o_mm)).unwrap().inputs()[1];
                let w_dims = g.tensor(w).unwrap().shape.dims().to_vec();
                let sum = g.node(at(&g, layer, c)).unwrap().inputs()[0];
                let doubled = g.node(at(&g, layer, c)).unwrap().outputs()[0];
                g.set_shape(doubled, &w_dims);
                g.set_input(at(&g, layer, c), 0, w);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
                if stage_one {
                    g.set_input(at(&g, layer, req), 0, sum);
                    g.set_input(at(&g, layer, s), 0, doubled);
                }
            }
            g
        };
        for stage_one in [false, true] {
            derive(&weight_case([2, 2], stage_one)).unwrap();
            assert_eq!(
                derive(&weight_case([2, 3], stage_one)),
                Err(ScheduleError::Unsupported { node: constant }),
                "Weight(O), stage_one = {stage_one}"
            );
        }

        // 2. Stacked `Commitment::ChannelParams` (`Role::Classes`) at stage 0 and stage 1.
        let (_, layer_len) = layer_start(&base, 0);
        let (cls_pos, cls_idx) = (0..layer_len)
            .find_map(|p| {
                base.node(at(&base, 0, p))
                    .unwrap()
                    .inputs()
                    .iter()
                    .enumerate()
                    .find(|&(_, &t)| {
                        base.inputs()
                            .iter()
                            .any(|i| i.tensor == t && matches!(i.role, Role::Classes(_)))
                    })
                    .map(|(idx, _)| (p, idx))
            })
            .unwrap();
        let classes_case = |constants: [i64; 2], stage_one: bool| -> Graph {
            let mut g = base.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let cls = g.node(at(&g, layer, cls_pos)).unwrap().inputs()[cls_idx];
                let cls_dims = g.tensor(cls).unwrap().shape.dims().to_vec();
                let sum = g.node(at(&g, layer, c)).unwrap().inputs()[0];
                let doubled = g.node(at(&g, layer, c)).unwrap().outputs()[0];
                g.set_shape(doubled, &cls_dims);
                g.set_input(at(&g, layer, c), 0, cls);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
                if stage_one {
                    g.set_input(at(&g, layer, req), 0, sum);
                    g.set_op(at(&g, layer, s), Op::SumSquares { axis: 0 });
                    g.set_input(at(&g, layer, s), 0, doubled);
                }
            }
            g
        };
        for stage_one in [false, true] {
            derive(&classes_case([2, 2], stage_one)).unwrap();
            assert_eq!(
                derive(&classes_case([2, 3], stage_one)),
                Err(ScheduleError::Unsupported { node: constant }),
                "ChannelParams(Classes), stage_one = {stage_one}"
            );
        }

        // 3. Stacked `Commitment::ChannelParams` (`Role::Multipliers`) at stage 0 and stage 1.
        let aff = graph("tiny-2-affine", 4);
        let mul_pos = position(&aff, 0, |op| {
            matches!(
                op,
                Op::Mul {
                    broadcast: Broadcast::Leading
                }
            )
        });
        let aff_c = position(&aff, 0, |op| matches!(op, Op::ConstMul { .. }));
        let aff_add = position(&aff, aff_c, |op| matches!(op, Op::Add));
        let aff_req = position(&aff, aff_c, |op| matches!(op, Op::Requant { .. }));
        let aff_s = position(&aff, aff_c, |op| matches!(op, Op::SumSquares { .. }));
        let aff_constant = at(&aff, 0, aff_c);
        let multipliers_case = |constants: [i64; 2], stage_one: bool| -> Graph {
            let mut g = aff.clone();
            for (layer, &k) in constants.iter().enumerate() {
                let m = g.node(at(&g, layer, mul_pos)).unwrap().inputs()[1];
                let m_dims = g.tensor(m).unwrap().shape.dims().to_vec();
                let checkpoint = g.checkpoints()[layer];
                let scaled = g.node(at(&g, layer, aff_c)).unwrap().outputs()[0];
                g.set_shape(scaled, &m_dims);
                g.set_input(at(&g, layer, aff_c), 0, m);
                // Keep `aff_add` well-shaped (`[n, d]`) by reading `checkpoint` directly.
                g.set_input(at(&g, layer, aff_add), 1, checkpoint);
                let Op::ConstMul { c: index } = *g.node(at(&g, layer, aff_c)).unwrap().op() else {
                    panic!()
                };
                g.set_constant(index, k);
                if stage_one {
                    g.set_op(at(&g, layer, aff_s), Op::SumSquares { axis: 0 });
                    g.set_input(at(&g, layer, aff_s), 0, scaled);
                } else {
                    g.set_input(at(&g, layer, aff_req), 0, scaled);
                }
            }
            g
        };
        for stage_one in [false, true] {
            derive(&multipliers_case([2, 2], stage_one)).unwrap();
            assert_eq!(
                derive(&multipliers_case([2, 3], stage_one)),
                Err(ScheduleError::Unsupported { node: aff_constant }),
                "ChannelParams(Multipliers), stage_one = {stage_one}"
            );
        }
    }

    /// RFC-0037 'Claims and atoms' and 'The claim pool' (#556): a stacked committed advice atom
    /// reached through a non-uniform per-layer `ConstMul` is permitted at stage 0 and at reduction
    /// stages `k >= 1`, and carries each layer's constant on its coefficient terms.
    #[test]
    fn a_per_layer_constant_on_committed_advice_is_supported_across_stages() {
        // 1. `tiny-2-servable` (`TwoSteps`): both `LayerInput` (`res`) and `Stacked` advice
        // (`update`) reached through `ConstMul([2, 3])` at stage 0 (`Requant`) and stage 1
        // (`SumSquares`), each leaving `Route::Pool`.
        let base = graph("tiny-2-servable", 4);
        let c = position(&base, 0, |op| matches!(op, Op::ConstMul { .. }));
        let req = position(&base, c, |op| matches!(op, Op::Requant { .. }));
        let s = position(&base, c, |op| matches!(op, Op::SumSquares { .. }));
        let constants = [2i64, 3];
        let expected_fp: Vec<Fp> = constants.iter().map(|&v| signed(v)).collect();

        let mut g = base.clone();
        for (layer, &k) in constants.iter().enumerate() {
            let doubled = g.node(at(&g, layer, c)).unwrap().outputs()[0];
            g.set_input(at(&g, layer, s), 0, doubled);
            let Op::ConstMul { c: index } = *g.node(at(&g, layer, c)).unwrap().op() else {
                panic!()
            };
            g.set_constant(index, k);
        }
        let schedule = derive(&g).expect("committed advice supports per-layer constants");

        // Stage 0: `AttnResidual`'s `Requant` leaf claim has `Route::Pool` and per-layer constants.
        let req_site = at(&g, 0, req);
        let leaf_claim = schedule
            .leaf()
            .iter()
            .find(|l| l.site == req_site && l.input == Some(0))
            .unwrap();
        assert_eq!(leaf_claim.claim.route, Route::Pool);
        assert_eq!(leaf_claim.claim.atoms.len(), 2);
        assert_eq!(leaf_claim.claim.atoms[0].tensor, TensorRef::LayerInput);
        assert!(matches!(
            leaf_claim.claim.atoms[1].tensor,
            TensorRef::Stacked(_)
        ));
        for atom in &leaf_claim.claim.atoms {
            assert_eq!(atom.class, Class::Committed(Commitment::Advice));
            assert_eq!(atom.coefficient.terms()[0].constant, expected_fp);
        }

        // Stage 1: `MlpNorm`'s `SumSquares` evaluation claim has `Route::Pool` and the same
        // per-layer constants on both stacked advice atoms.
        let s_node = at(&g, 0, s);
        let sum_squares = schedule.stages()[0]
            .instances
            .iter()
            .find(|inst| inst.node == s_node)
            .unwrap();
        let stage1_claim = &sum_squares.evals[0];
        assert_eq!(stage1_claim.route, Route::Pool);
        assert_eq!(stage1_claim.atoms.len(), 2);
        assert_eq!(stage1_claim.atoms[0].tensor, TensorRef::LayerInput);
        for atom in &stage1_claim.atoms {
            assert_eq!(atom.class, Class::Committed(Commitment::Advice));
            assert_eq!(atom.coefficient.terms()[0].constant, expected_fp);
        }

        // 2. 4-layer `tiny-4-servable` (`FusedKey`) & `tiny-2-affine`: stage-1 split claim
        // `Add(z, ConstMul(k_l, res))` where `z` is virtual (uniform `[1]`) and `res` is `X_in`
        // (advice, per-layer constants `[2, 3, 5, 7]` and `[-3, 5]`).
        let four = graph("tiny-4-servable", 4);
        let four_c = position(&four, 0, |op| matches!(op, Op::ConstMul { .. }));
        let four_add = position(&four, four_c, |op| matches!(op, Op::Add));
        let four_s = position(&four, four_c, |op| matches!(op, Op::SumSquares { .. }));
        let four_constants = [2i64, 3, 5, 7];
        let mut four_g = four.clone();
        for (layer, &k) in four_constants.iter().enumerate() {
            let sum = four_g.node(at(&four_g, layer, four_add)).unwrap().outputs()[0];
            four_g.set_input(at(&four_g, layer, four_s), 0, sum);
            let Op::ConstMul { c: index } = *four_g.node(at(&four_g, layer, four_c)).unwrap().op()
            else {
                panic!()
            };
            four_g.set_constant(index, k);
        }
        let four_schedule = derive(&four_g).expect("4-layer split advice succeeds");
        let four_inst = four_schedule.stages()[0]
            .instances
            .iter()
            .find(|inst| inst.node == at(&four_g, 0, four_s))
            .unwrap();
        let four_claim = &four_inst.evals[0];
        assert!(matches!(four_claim.route, Route::Split(_)));
        assert_eq!(
            four_claim.atoms[1].coefficient.terms()[0].constant,
            four_constants
                .iter()
                .map(|&v| signed(v))
                .collect::<Vec<_>>()
        );

        let aff = graph("tiny-2-affine", 4);
        let aff_c = position(&aff, 0, |op| matches!(op, Op::ConstMul { .. }));
        let aff_add = position(&aff, aff_c, |op| matches!(op, Op::Add));
        let aff_s = position(&aff, aff_c, |op| matches!(op, Op::SumSquares { .. }));
        let mut split_g = aff.clone();
        for (layer, k) in [-3i64, 5].into_iter().enumerate() {
            let sum = split_g
                .node(at(&split_g, layer, aff_add))
                .unwrap()
                .outputs()[0];
            split_g.set_input(at(&split_g, layer, aff_s), 0, sum);
            let Op::ConstMul { c: index } = *split_g.node(at(&split_g, layer, aff_c)).unwrap().op()
            else {
                panic!()
            };
            split_g.set_constant(index, k);
        }
        let split_schedule = derive(&split_g).expect("split virtual + per-layer advice succeeds");
        let split_inst = split_schedule.stages()[0]
            .instances
            .iter()
            .find(|inst| inst.node == at(&split_g, 0, aff_s))
            .unwrap();
        let split_claim = &split_inst.evals[0];
        assert!(matches!(split_claim.route, Route::Split(_)));
        assert!(matches!(split_claim.atoms[0].class, Class::Virtual { .. }));
        assert_eq!(
            split_claim.atoms[0].coefficient.terms()[0].constant,
            vec![Fp::ONE]
        );
        assert_eq!(
            split_claim.atoms[1].class,
            Class::Committed(Commitment::Advice)
        );
        assert_eq!(split_claim.atoms[1].tensor, TensorRef::LayerInput);
        assert_eq!(
            split_claim.atoms[1].coefficient.terms()[0].constant,
            vec![signed(-3), signed(5)]
        );
    }

    /// A claim on a public tensor names no value the stages can prove.
    #[test]
    fn a_public_atom_is_unsupported() {
        let mut g = graph("tiny-2-servable", 4);
        let p = position(&g, 0, |op| matches!(op, Op::RsqrtRow { .. }));
        let positions = g
            .inputs()
            .iter()
            .find(|i| i.role == Role::Positions)
            .unwrap()
            .tensor;
        for layer in 0..2 {
            g.set_input(at(&g, layer, p), 0, positions);
        }
        assert_eq!(
            derive(&g),
            Err(ScheduleError::Unsupported { node: NodeId(0) })
        );
    }

    /// Relations of shapes the stages do not prove: `SumSquares` over another axis than the last.
    #[test]
    fn a_relation_of_another_shape_is_unsupported() {
        let mut g = graph("tiny-2-servable", 4);
        let p = position(&g, 0, |op| matches!(op, Op::SumSquares { .. }));
        for layer in 0..2 {
            g.set_op(at(&g, layer, p), Op::SumSquares { axis: 0 });
        }
        assert_eq!(
            derive(&g),
            Err(ScheduleError::Unsupported { node: at(&g, 0, p) })
        );
    }

    /// RFC-0037 'The committed first checkpoint': a committed linear output inside the layers
    /// would need a stacked equality zero-check, which the lowering never needs.
    #[test]
    fn a_committed_linear_output_inside_the_layers_is_unsupported() {
        let mut g = graph("tiny-2-servable", 4);
        let c = position(&g, 0, |op| matches!(op, Op::ConstMul { .. }));
        for layer in 0..2 {
            let out = g.node(at(&g, layer, c)).unwrap().outputs()[0];
            g.set_kind(out, Kind::Advice);
        }
        assert_eq!(
            derive(&g),
            Err(ScheduleError::Unsupported { node: at(&g, 0, c) })
        );
    }

    /// The logits are the LM head's one output, in the epilogue.
    #[test]
    fn a_second_lm_head_is_refused() {
        let mut g = graph("tiny-2-servable", 4);
        let (start, len) = layer_start(&g, 1);
        let lm_head = NodeId(g.nodes().len() as u32 - 1);
        let weight = crate::lowering::WeightRef {
            ty: WeightType::LmHead,
            layer: 0,
        };
        g.set_op(NodeId(start + len + 1), Op::MatMulWeight { weight });
        assert_eq!(derive(&g), Err(ScheduleError::Template { node: lm_head }));
    }

    #[test]
    fn axis_blocks_give_the_last_axis_the_lowest_coordinates() {
        assert_eq!(
            axis_blocks(&[5, 4, 16]).unwrap(),
            vec![(6, 3), (4, 2), (0, 4)]
        );
        assert_eq!(axis_blocks(&[1, 3]).unwrap(), vec![(2, 0), (0, 2)]);
    }
}
