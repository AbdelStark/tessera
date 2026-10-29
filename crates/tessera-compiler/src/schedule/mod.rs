// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedule (RFC-0010 'Stages', RFC-0037): the claims each stage leaves, how each splits,
//! the relation instances that prove them, and what reaches the claim pool, derived from the
//! lowered graph alone (INV-SCH-1).
//!
//! [`derive_schedule`] reads a [`Lowering`], and [`derive_graph_schedule`] a graph with its tables,
//! checkpoints and logits, such as the anchor's circuits, which have none (RFC-0072 'Graphs without
//! layers'). Both take the same five steps:
//!
//! 1. **Layers.** The checkpoints `c_0, …, c_L` bound the layers. Every layer must be one template,
//!    node for node ([`LayerMap`]), so that the outputs at one position form one kind stacked over
//!    the layers, with `λ = ⌈log2 L⌉` layer variables, the highest (RFC-0037 'Layers and kinds').
//!    A graph with no checkpoint has no layers, `L = λ = 0`.
//! 2. **The graph's refusals.** Before any claim, RFC-0072's rules that read the graph alone: a
//!    committed output stage 1 does not prove, and a sum over an axis that is not a power of two
//!    whose inputs are not 0 at its padded positions, downstream of a lookup non-zero at padding
//!    ([`nonzero_at_padding`]) or of a committed graph input nothing pins, are refused.
//! 3. **The leaf stage.** It leaves one claim per input of each gadget but `BitDecompose` and
//!    `AssertEqual` ([`has_leaf_site`]), and, when the graph has logits, one on them that the
//!    scoring certificates read (RFC-0017), each at a point of its site. A site in the layers leaves
//!    one claim per layer, all at that point ('The layer point'). A claim on a public input, such
//!    as AES's counter blocks, is that one public term, with no part: the leaf layout sends no
//!    value for it (RFC-0045 'The split'). This is the leaf stage's interface to the rest of the
//!    schedule. The gadgets' own relations, such as `SoftmaxRow`'s masked keys and the padding
//!    zero-checks of RFC-0035, enter with the gadgets (#118).
//! 4. **Claims.** A claim maps through the linear nodes onto atoms: virtual, the output of a
//!    sumcheck relation, `Slice`, `GatherPublic` and `BitFn` among them, or committed ('Claims and
//!    atoms'). What reaches a public tensor ([`public_tensors`]), and a `ConstAdd`'s constant, is
//!    a [`PublicTerm`], which the verifier evaluates and moves to the claim's value (RFC-0072
//!    'Public tensors', 'Linear operations that are not coordinate maps'). Within the claim, two
//!    or more virtual atoms that select rows of one tensor along one axis, at one point with one
//!    set of layer weights, are one atom of coefficient 1 ([`Atom::combined`], RFC-0072 'Combined
//!    selections'). A claim that is then one virtual atom of coefficient 1, or committed atoms of
//!    one commitment at one point, is ready. Any other is split: the stage sends one value per
//!    part ('Splits'). A claim of public terms alone has no part.
//! 5. **Stages.** Stage `k` proves one instance per virtual part that stage `k − 1` leaves, in
//!    order, stacked over the layers where its kind is: a selection's over the selected axis
//!    alone, against its public kernel (RFC-0072 'The selection relation'), and a `BitFn`'s as
//!    RFC-0010's `Elementwise` (RFC-0066 '`BitFn`'). Stage 1 then adds the equality zero-check
//!    of each committed output of a linear or degree-2 node ('The committed first checkpoint'),
//!    and after them RFC-0066's zero-checks of each `BitDecompose` and `AssertEqual`: its
//!    committed input's padding, its bits' booleanity and padding, its recomposition summed over
//!    the bit axis, and `AssertEqual`'s equality (RFC-0072 'The bit zero-checks'). Each
//!    instance's evaluations are the stage's claims, which go to the next stage or the pool.
//!
//! Every point is symbolic: each [`Coord`] names the challenge, or the constant `½`, that sets it.
//! So the schedule fixes which atoms share a point, and so how claims merge, split and group, before
//! any challenge is drawn. The prover evaluates it at the proof's challenges; the independent
//! verifier derives the same schedule from RFC-0037 itself (#114).

mod claims;
mod degree;
mod layers;
pub mod masks;
mod natives;
mod padding;
mod public;
mod relations;
mod tokens;

use core::ops::Range;

use tessera_ir::{
    BoolFn, Broadcast, CommitmentClass, Graph, NodeId, NodeKind, Op, OpClass, TensorId, WeightType,
};
use tessera_tables::TableSpec;

use crate::{CompileError, Lowering};

pub use degree::{
    Constant, DegreeKey, DegreeRelation, DegreeRow, EqualityOp, FOLD_DEGREE, InstanceDegree,
    StageDegree, degree_row, protocol2_degree, protocol2_stages, stage_degree,
};
pub use layers::LayerMap;
pub use natives::{ModelNatives, NativeKind, Request, model_natives, model_natives_batch};
pub use padding::{nonzero_at_padding, reads_padding, tensors_nonzero_at_padding};
pub use public::{producers, public_tensors};
pub use tokens::token_axes;

/// A coordinate of a point, by the challenge or constant that sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coord {
    /// Coordinate `index` of the point at which the leaf stage leaves its claim of leaf site
    /// `site` ([`StageSchedule::sites`]), the lowest first: in every layer, the same point.
    Leaf {
        /// The site.
        site: u32,
        /// The coordinate.
        index: u32,
    },
    /// Coordinate `index` of stage 1's layer point `ρ_1`, squeezed under `stage/1/layer/{index}`.
    Layer(u32),
    /// Coordinate `index` of stage 1's zero-check point, squeezed under `stage/1/zero/{index}`.
    Zero(u32),
    /// The challenge of round `round` of stage `stage`, squeezed under `stage/{stage}/r/{round}`.
    Round {
        /// The stage.
        stage: u16,
        /// The round, from 0.
        round: u32,
    },
    /// The constant `½` at which a sum over an axis places its input (RFC-0010 'Relations').
    Half,
}

/// What a claim or an atom is on (RFC-0037 'Layers and kinds').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Output `output` of the node at template position `position`, stacked over the layers.
    Layer {
        /// The position in the layer, from 0.
        position: u32,
        /// The output.
        output: u32,
    },
    /// The layer input `X_in[ℓ] = c_ℓ`, stacked over the layers: committed in every layer.
    LayerInput,
    /// Input `input` of the node at template position `position`, a graph input of its own in
    /// every layer: a weight, a channel-class vector or multipliers, stacked over the layers.
    LayerParameter {
        /// The position of the node that reads it.
        position: u32,
        /// The input.
        input: u32,
    },
    /// A tensor outside the layers: the prologue's and the epilogue's, and the graph inputs that
    /// every layer shares, such as the positions.
    Tensor(TensorId),
}

/// How an atom or a claim weighs the layers of its kind: `Σ_ℓ w(ℓ)·X̃_{κ,ℓ}(r)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Layers {
    /// A kind without layers.
    None,
    /// The leaf stage's claim of a site in the layers is one claim per layer, all at one point.
    /// An atom of it weighs its claim's layer alone.
    Each,
    /// One layer alone, `w(ℓ) = [ℓ = layer]`: the epilogue's claims on the last checkpoint.
    One(u32),
    /// `w = eq(ρ, ·)` at the layer point `ρ`, of `λ` coordinates.
    Eq(Vec<Coord>),
}

/// A node's relation, in every layer for a template position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeRef {
    /// A node outside the layers.
    Node(NodeId),
    /// The node at a template position, in each layer.
    Position(u32),
}

/// A factor of an atom's coefficient.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Factor {
    /// `−1`, the subtrahend's.
    Minus,
    /// `2^j`, of a sum over an axis of `j` variables, placed at `½` (RFC-0010 'Relations').
    Pow2(u32),
    /// The constant of a `ConstMul` by one constant. At a template position each layer has its
    /// own, which multiplies that layer's weight.
    Constant(NodeRef),
    /// `LT(r_a, len)` at the coordinates `r_a` that an `Expand` of `len` positions drops, when
    /// `len` is not a power of two (RFC-0037 'Head layouts').
    Lt {
        /// The dropped coordinates, the lowest first.
        coords: Vec<Coord>,
        /// The inserted axis's length.
        len: u32,
    },
}

/// An atom's class (RFC-0037 'Claims and atoms').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomClass {
    /// The output of a sumcheck relation, which a later stage proves.
    Virtual,
    /// Committed in the advice (lookup outputs among it) or the weight commitment.
    Committed(CommitmentClass),
}

/// A term of a claim after substitution: `Σ_ℓ w(ℓ)·c(ℓ)·X̃_{κ,ℓ}(r)`, or a combined selection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// The kind `κ`: a combined selection's first member's.
    pub kind: Kind,
    /// The point `r` on the kind's own variables, its layer variables aside.
    pub point: Vec<Coord>,
    /// The layer weights `w`.
    pub layers: Layers,
    /// The coefficient `c`: a sum of products, one for each atom merged into this one, in order.
    /// It is 1 when it is one empty product, as a combined selection's is.
    pub coefficient: Vec<Vec<Factor>>,
    /// Virtual or committed.
    pub class: AtomClass,
    /// For a combined selection (RFC-0072 'Combined selections', RFC-0037 'Claims and atoms'), its
    /// members, two or more, in the order substitution reached them: virtual atoms of one claim,
    /// outputs of `Slice` or `GatherPublic` that select rows of one input tensor along one axis,
    /// at this atom's point with its layer weights. The atom is then the one selection
    /// `Σ_t a_t·ỹ_t(r)`, its coefficient 1 and its kind the first member's, and its relation's
    /// kernel is `K = Σ_t a_t·S_t`: its split value is the whole combination, weighed by 1, and
    /// its members' own values are never sent. Empty for every other atom, a single selection's
    /// among them, whose coefficient weighs its split value.
    pub combined: Vec<Member>,
}

impl Atom {
    /// Whether the coefficient is the constant 1: one empty product.
    pub fn is_unit(&self) -> bool {
        matches!(self.coefficient.as_slice(), [product] if product.is_empty())
    }
}

/// A member of a combined selection ([`Atom::combined`]): the output of one `Slice` or
/// `GatherPublic` and its coefficient `a_t` in the combination, a sum of products as an
/// [`Atom`]'s, at the combination's point and layer weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Member {
    /// The selection's output.
    pub kind: Kind,
    /// Its coefficient `a_t`: a sum of products, one for each atom merged into it, in order.
    pub coefficient: Vec<Vec<Factor>>,
}

/// What a [`PublicTerm`] evaluates: a value both parties compute from public data, the public
/// graph inputs and the graph's constants (RFC-0072 'Public tensors').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PublicSource {
    /// The extension of a public tensor ([`public_tensors`]) over its padded shape: the verifier
    /// computes the tensor from the public graph inputs and constants through the nodes that are
    /// not gadget-backed, in `O(size)`.
    Tensor(TensorId),
    /// The shift `ĉ` of a `ConstAdd` node, the multilinear extension of `ψ·c` over its output's
    /// shape `(n_a)`, the constant at real positions only (RFC-0072 'Padded positions'). With the
    /// selector `ψ̃(r) = Π_a LT(r_a, n_a)`, it is `c·ψ̃(r)` for a constant of one value, a scalar
    /// whatever the last axis's length, and `c̃(r_c)·Π_{a≠last} LT(r_a, n_a)` for one per channel
    /// of the last axis, `c̃` the constant's extension, 0 at padded channels ('Linear operations
    /// that are not coordinate maps'). `c̃(r_c)·ψ̃(r)` agrees with it on the cube only.
    Shift(NodeId),
}

/// A public term of a claim, `c·s(r)` for a [`PublicSource`] `s` at the point `r` with the
/// coefficient `c`: the verifier evaluates it and moves it to the claim's value (RFC-0037 'Claims
/// and atoms', 'Splits'). It reaches no instance and no pool, and the proof sends nothing for it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PublicTerm {
    /// What it evaluates.
    pub source: PublicSource,
    /// The point `r`, on the source's tensor's variables: a `ConstAdd`'s output's for a shift.
    pub point: Vec<Coord>,
    /// The coefficient `c`, as an [`Atom`]'s: a sum of products, one per term merged into this
    /// one. A public term is outside the layers, so it has no layer weights.
    pub coefficient: Vec<Vec<Factor>>,
}

/// Where a part of a claim goes (RFC-0037 'Splits').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// A virtual atom, which the next stage proves as its instance `instance`.
    Instance {
        /// The atom, in the claim's atoms.
        atom: usize,
        /// The instance, in the next stage's.
        instance: usize,
    },
    /// Committed atoms of one commitment at one point, which reach the pool as its claim `claim`.
    Pool {
        /// The atoms, in the claim's atoms, in order.
        atoms: Vec<usize>,
        /// The pool claim, in [`StageSchedule::pool`].
        claim: usize,
    },
}

/// A claim a stage leaves: the value of a kind's extension at a point, weighed over its layers,
/// with its atoms and where they go.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Claim {
    /// The kind whose extension it states.
    pub kind: Kind,
    /// The point, on the kind's own variables.
    pub point: Vec<Coord>,
    /// The layer weights.
    pub layers: Layers,
    /// Its atoms, in the order substitution reaches them, depth first, inputs in their node's
    /// order. Atoms on one kind at one point with the same layer weights merge, in the first one's
    /// place; then selections of one tensor along one axis at one point with one set of layer
    /// weights combine, in the first one's place ([`Atom::combined`]).
    pub atoms: Vec<Atom>,
    /// Its public terms, in the order substitution reaches them: the public tensors it reaches
    /// and the shifts of the `ConstAdd`s it maps through. Terms of one source at one point merge,
    /// in the first one's place. The verifier subtracts their sum `p` from the claim's value `v`:
    /// a ready claim's instance or pool claim is `v − p`, and a split's values must give `v − p`
    /// (RFC-0037 'Splits'). A claim with no atom has no part, and its check is `v = p`.
    pub public: Vec<PublicTerm>,
    /// Whether the stage splits it, sending one value per part under `stage/{k}/split`.
    pub split: bool,
    /// Its parts, in atom order: one when the claim is ready, none when it has no atom.
    pub parts: Vec<Part>,
}

/// A relation an instance proves (RFC-0010 'Relations', RFC-0037 'Rotate').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    /// A native stage relation, whose tables and source transformations the CPU backend supplies.
    Native {
        /// Verifier-independent stable identity in the compiler's native plan.
        tag: u32,
        /// What it proves, the native plan's kind for its tag ([`Native::kind`]): RFC-0013
        /// 'Degree' keys a native's row by it, never by its descriptor.
        kind: NativeKind,
        /// Degree including the equality/selectors.
        degree: usize,
        /// Raw proof evaluations before graph-source transformation.
        evaluations: usize,
        /// Whether its target is the mandatory zero-check value zero.
        zero: bool,
    },
    /// `Y = X·Wᵀ`; its evaluations are `W`'s, then `X`'s.
    MatMulWeight,
    /// `Z = A·B` over the heads; its evaluations are `A`'s, then `B`'s, or `K`'s when `B` enters
    /// transposed.
    MatMulAct {
        /// Whether the right operand is stored as `K`, `[heads, c, j]`.
        transposed: bool,
    },
    /// `Y = A·B`, `B` broadcast; its evaluations are `A`'s, then `B`'s.
    Mul(Broadcast),
    /// The sum of squares over the last axis; one evaluation.
    SumSquares,
    /// RoPE as the kernel product `Ỹ(r) = Σ_x E_r(x)·X̃(x)`; one evaluation, the positions and
    /// the frequencies being public.
    Rotate,
    /// A `ConstMul` by one constant per channel, public; one evaluation.
    ScaleChannels,
    /// RFC-0072's selection relation of a `Slice` or a `GatherPublic` of `x` along its selected
    /// axis `a`, `n_a` positions: `ỹ(r_lo ‖ r_a ‖ r_hi) = Σ_k S̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)` with
    /// `S(j, k) = [j < len]·[idx_j = k]` public, `idx_j` the `GatherPublic`'s row `j` or the
    /// `Slice`'s `start + j`. A sumcheck over `k` alone, the instance's `⌈log2 n_a⌉` variables,
    /// of degree 2, `S̃(r_a, ·)` its one public factor; for a combined selection
    /// ([`Atom::combined`]) the kernel `K = Σ_t a_t·S_t` in its place. The instance's point is
    /// the atom's on `y`, its kind `y`'s (the first member's). One evaluation, a claim on `x` at
    /// `r_lo ‖ r'_k ‖ r_hi`, `r'_k` the stage's last `⌈log2 n_a⌉` challenges.
    Selection,
    /// `BitFn { f }` as RFC-0010's `Elementwise`, `Z̃(r) = Σ_x eq(r, x)·f(I_1(x), …)`, with `f`
    /// RFC-0066's multilinear polynomial of the Boolean function: degree 3, or 4 for `Xor3` and
    /// `Maj`. Its evaluations are its inputs' that are not public, in order, each at the
    /// instance's point.
    BitFn(BoolFn),
    /// The equality zero-check `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))` of a node's committed output `Y`;
    /// its evaluations are `Y`'s, then those of `f`'s inputs.
    Equality,
    /// The booleanity zero-check `0 = Σ_x eq(z, x)·H̃(x)·(1 − H̃(x))` of a committed output `H`:
    /// the original `OneHot`'s indicator, or a `BitDecompose`'s `b` or `h` (RFC-0066, RFC-0072
    /// 'The bit zero-checks'), the instance's kind; one evaluation.
    Booleanity,
    /// The original indicator is zero outside the real vocabulary; one H evaluation.
    OneHotVocabulary,
    /// Zero-check that each indicator row sums to its public row selector; one H evaluation.
    OneHotCount,
    /// Zero-check that its weighted sum is the original token; H then token evaluations.
    OneHotIndex,
    /// RFC-0066's padding check of the committed graph input `u` that a `BitDecompose` reads,
    /// directly or moved, when `u` has padded positions: `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·ũ(x)`, `ψ` 1
    /// on `u`'s real positions (RFC-0072 'The bit zero-checks', 'Real positions'). The instance's
    /// kind is `u`'s; one evaluation, `ũ`'s.
    InputPadding,
    /// RFC-0066's bit-padding check of a `BitDecompose`'s output `b` of `low` bits, or `h` of
    /// `high`, when that count `n` is not a power of two: `0 = Σ_x eq(z, x)·(1 − LT(x_bit, n))·
    /// b̃(x)` over the bit axis's coordinates `x_bit`, the lowest. The instance's kind is the
    /// output's; one evaluation.
    BitPadding,
    /// RFC-0072's recomposition of a `BitDecompose { low, high }` of `x`, one zero-check over `y`
    /// summed over the bit axis's `k = max(k_b, k_h)` variables, the lowest of its `n_y + k`:
    /// `0 = Σ_{(i, y)} eq(z_y, y)·[2^{−k}·x̃(y) − W_b(i)·b̂(i ‖ y) − 2^{low}·W_h(i)·ĥ(i ‖ y)]`.
    /// The outputs are zero-extended on their high bit coordinates, and `eq` takes the highest
    /// `n_y` coordinates of the instance's point (RFC-0035 'Zero-check points'). Its evaluations
    /// are `x̃(r'_y)` unless `x` is public, then `b̃` and `h̃` at the high `k_b` and `k_h` of
    /// `r'_i` with `r'_y`. The instance's kind is the output whose bit axis has the `k`
    /// variables: `b`, or `h` when `k_h > k_b`.
    Recomposition,
    /// `AssertEqual`'s zero-check `0 = Σ_x eq(z, x)·ψ̃(x)·(ã(x) − b̃(x))`, `ψ` 1 on the tensors'
    /// real positions (RFC-0066, RFC-0072 'Real positions'); its evaluations are `ã`'s, then
    /// `b̃`'s, each unless public, none when both are. The instance's kind is `a`'s.
    AssertEqual,
}

/// What an instance proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// A fixed native target supplied by its own component before stage gamma.
    Native(u32),
    /// The value of a virtual part: part `part` of claim `claim` of the stage before.
    Part {
        /// The claim, in the stage before's claims.
        claim: usize,
        /// The part.
        part: usize,
    },
    /// A zero-check, whose claim is 0.
    ZeroCheck,
}

/// A relation instance of a stage, stacked over the layers when its kind is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Instance {
    /// What it proves.
    pub source: Source,
    /// Its relation.
    pub relation: Relation,
    /// The node whose relation it is.
    pub node: NodeRef,
    /// The kind of the node's output it states; for a bit zero-check the tensor its
    /// [`Relation`] names (RFC-0072 'The bit zero-checks').
    pub kind: Kind,
    /// Its claim's point on that output: the virtual part's, or for a zero-check its coordinates
    /// of the zero-check point, the last `n_i`. A recomposition's `eq` takes the highest `n_y`
    /// of them (RFC-0035 'Zero-check points').
    pub point: Vec<Coord>,
    /// Its claim's layer weights: `eq` at the part's layer point, or for a stacked zero-check at
    /// the zero-check point's highest coordinates.
    pub layers: Layers,
    /// Its variables `n_i`, the layer variables the highest.
    pub vars: u32,
    /// The point its sumcheck ends at: the stage's last `n_i` challenges.
    pub challenges: Vec<Coord>,
    /// The claims its evaluations leave, in its relation's order, in the stage's claims.
    pub claims: Range<usize>,
}

/// A stage of the schedule (RFC-0010 'Stages').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stage {
    /// Its index: 0 for the leaf stage.
    pub index: u16,
    /// Its instances, in order: those of the virtual parts the stage before leaves, then in stage
    /// 1 the zero-checks in node order, and after them the bit zero-checks of each `BitDecompose`
    /// and `AssertEqual` in node order (RFC-0072 'The bit zero-checks'). The leaf stage has none
    /// here: its fraction trees and product trees are RFC-0008's and RFC-0009's.
    pub instances: Vec<Instance>,
    /// Its rounds `n_max`, the most variables among its instances: 0 for the leaf stage.
    pub rounds: u32,
    /// The coordinates of its zero-check point, the most variables among its zero-checks.
    pub zero_vars: u32,
    /// The claims it leaves, in order: the leaf stage's one per site, a reduction stage's those of
    /// each instance's evaluations in instance order.
    pub claims: Vec<Claim>,
}

/// A claim of the pool: a combination `Σ_t a_t·P̃_t(r) = v` of committed atoms of one commitment
/// at one point (RFC-0037 'The claim pool').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolClaim {
    /// The commitment that holds every atom.
    pub commitment: CommitmentClass,
    /// The point `r`.
    pub point: Vec<Coord>,
    /// The atoms. A leaf stage's per-layer claims arrive stacked at the layer point `ρ_1`.
    pub atoms: Vec<Atom>,
    /// The stage whose claim it is a part of.
    pub stage: u16,
    /// The claim, in that stage's claims.
    pub claim: usize,
}

/// Where the leaf stage leaves a claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    /// Input `input` of a gadget's node.
    Gadget {
        /// The node.
        node: NodeRef,
        /// The input.
        input: u32,
    },
    /// The logits, which the scoring certificates read (RFC-0017), when the graph has them.
    Logits,
}

/// A source provided by a native relation rather than the graph leaf stage. `None` names the
/// logits at the LM-head node; `Some(i)` names gadget input `i`.
pub type Deferred = (NodeId, Option<usize>);

/// One original graph source returned by a native component.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeOutput {
    /// Original first-layer tensor, or an outside tensor.
    pub tensor: TensorId,
    /// Its local coordinate projection within the native final point, before layer coordinates.
    pub from: u32,
    /// Number of local source coordinates.
    pub vars: u32,
}

/// Compiler-owned native metadata for one stage-one component. No proof chooses this plan.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Native {
    /// Backend component identity.
    pub tag: u32,
    /// What it proves: the relation of RFC-0050, RFC-0030 or RFC-0053 the native plan names for
    /// the tag ([`ModelNatives::kinds`]). The schedule's instance carries it
    /// ([`Relation::Native`]), so that RFC-0013 'Degree' keys the instance from the schedule
    /// alone: `Key`, `Forest` and `Reciprocal` share a descriptor of degree 3 and two
    /// evaluations, and their rows differ.
    pub kind: NativeKind,
    /// Owning first-layer gadget or LM-head node.
    pub node: NodeId,
    /// Total variables, with any layer coordinates highest.
    pub vars: u32,
    /// Full polynomial degree.
    pub degree: usize,
    /// Raw proof evaluations.
    pub evaluations: usize,
    /// Mandatory zero-check.
    pub zero: bool,
    /// Transformed outgoing graph sources in component order.
    pub outputs: Vec<NativeOutput>,
}

/// The stage schedule of a lowered graph (RFC-0010 'Stages', RFC-0037).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StageSchedule {
    /// The layers and their template.
    pub layers: LayerMap,
    /// The leaf stage's sites, in node order, then the logits when the graph has them: the leaf
    /// stage's claim `s` is site `s`'s.
    pub sites: Vec<Site>,
    /// The stages, from the leaf stage.
    pub stages: Vec<Stage>,
    /// The claim pool, in the order the stages reach it.
    pub pool: Vec<PoolClaim>,
}

impl StageSchedule {
    /// The number of stages, the leaf stage among them, which the commitment report states
    /// ([`stage_count`](crate::stage_count)).
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// The atom that an instance of stage `stage` proves when its source is `source`: the virtual
    /// atom of part `part` of claim `claim` of the stage before, a combined selection's members
    /// among it ([`Atom::combined`]). `None` for a zero-check, a native, or a claim or part the
    /// schedule does not have.
    pub fn atom_of(&self, stage: u16, source: Source) -> Option<&Atom> {
        let Source::Part { claim, part } = source else {
            return None;
        };
        let claim = self
            .stages
            .get(usize::from(stage).checked_sub(1)?)?
            .claims
            .get(claim)?;
        match claim.parts.get(part)? {
            Part::Instance { atom, .. } => claim.atoms.get(*atom),
            Part::Pool { .. } => None,
        }
    }

    /// The weight type of `kind`, when it is a weight matrix of `graph`.
    pub fn weight_type(&self, graph: &Graph, kind: Kind) -> Option<WeightType> {
        let tensor = match kind {
            Kind::Tensor(tensor) => tensor,
            Kind::LayerParameter { .. } => self.layers.tensor(graph, kind, 0)?,
            Kind::Layer { .. } | Kind::LayerInput => return None,
        };
        graph
            .weights
            .iter()
            .find(|&&(_, t)| t == tensor)
            .map(|&(weight, _)| weight.ty)
    }
}

/// The stage schedule of `lowering`, derived from its graph and checkpoints alone (INV-SCH-1), with
/// a claim on its logits.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for a graph the lowering does not emit: layers that
/// are not one template ([`LayerMap`]), or one of the graph's refusals ([`derive_graph_schedule`]).
pub fn derive_schedule(lowering: &Lowering) -> Result<StageSchedule, CompileError> {
    derive_composed_schedule(lowering, &[], &[])
}

/// The stage schedule of `graph`, whose tables are `tables`, its layers bounded by `checkpoints`
/// and its leaf stage leaving a claim on `logits` when it has some (INV-SCH-1). The anchor's
/// circuits have no checkpoint and no logits: no layers, every node outside them, and no scoring
/// site (RFC-0072 'Graphs without layers'). [`derive_schedule`] is this function on a lowering's
/// graph, tables, checkpoints and logits.
///
/// **Precondition.** The schedule proves `graph`'s relations as RFC-0010 states them over the
/// field, so `graph` must have passed RFC-0006 range analysis ([`analyze`](crate::analyze)):
/// every gadget input a true integer inside its gadget's domain, and every committed advice graph
/// input proven in its declared range and pinned at its padded positions. This function checks
/// the graph's rules ([`Graph::validate`]) but runs no range analysis; its summing rule treats an
/// unpinned committed input as non-zero at padding rather than rely on it (RFC-0072 'Summing
/// relations').
///
/// # Errors
///
/// [`CompileError::Graph`] for a graph that breaks a rule of RFC-0006 'Operation set'. Then
/// [`CompileError::Schedule`] (`ERR-INTERNAL`):
///
/// - layers that are not one template, a single checkpoint, or a template holding an operation
///   whose stacked form RFC-0072 leaves unspecified, a lookup non-zero at padding or a public
///   tensor ([`LayerMap`]);
/// - a committed output of a node other than a gadget, `ConstMul`, `ConstAdd`, `Add`, `Sub` or
///   `Mul`, which stage 1 does not prove (RFC-0072 'A committed output');
/// - a `Sum`, `SumSquares` or `MatMulAct` over an axis whose length is not a power of two none of
///   whose summed inputs is 0 at its padded positions (RFC-0072 'Summing relations');
/// - a gadget other than a lookup (`Requant`, `TableMap`, `Bitwise`), `BitDecompose` or
///   `AssertEqual`, such as a `OneHot`, `SoftmaxRow`, `RsqrtRow`, `Argmax` or `InverseCdf`, that
///   reads a tensor with padded positions which a lookup non-zero at padding reaches
///   ([`tensors_nonzero_at_padding`]): RFC-0035's row relations take a padded row's input as 0,
///   so no witness passes them (RFC-0072 'Padded positions');
/// - logits that no node produces;
/// - a matmul with one public operand, which its relations do not read as `Public`;
/// - a graph with layers whose schedule has no stage 1, which draws the layer point.
///
/// A claim that reaches a public tensor takes it as a [`PublicTerm`], and so does one through a
/// `ConstAdd` its shift; a relation's public operand leaves no claim (RFC-0072 'Public tensors'
/// 'As an operand'). A leaf site that reads a public tensor, such as AES's counter blocks, leaves
/// a claim of that one public term and no part, since the leaf layout sends no value for it
/// (RFC-0045 'The split'). A claim through a `Slice` or a `GatherPublic` is a selection's virtual
/// atom, several of which may combine ([`Atom::combined`]), and one through a `BitFn` a virtual
/// atom of its `Elementwise` relation.
///
/// It runs no range analysis. The preconditions that rest on ranges are RFC-0006's range
/// analysis's ([`analyze`](crate::analyze)) and gadget selection's, which a verifier re-derives
/// from public data before it derives the schedule (SP-7): a `BitFn`'s inputs are bits that an
/// operation proves, a `BitDecompose`'s input lies in its bits' range, and `AssertEqual`'s inputs
/// are true integers. A schedule derived from a graph that range analysis refuses proves the
/// relations as written, not the operations.
pub fn derive_graph_schedule(
    graph: &Graph,
    tables: &[TableSpec],
    checkpoints: &[TensorId],
    logits: Option<TensorId>,
) -> Result<StageSchedule, CompileError> {
    graph.validate()?;
    derive(graph, tables, checkpoints, logits, &[], &[])
}

/// Compose model-owned native components with the existing graph derivation. Each deferred
/// source must be replaced by a native output at the same site. This is public protocol geometry,
/// not a proof or an acceptance verdict.
///
/// # Errors
/// Existing graph schedule errors; `CompileError::Schedule` for inconsistent own descriptors,
/// duplicate identities/deferred sources, bad projections or missing graph replacements.
pub fn derive_composed_schedule(
    lowering: &Lowering,
    deferred: &[Deferred],
    native: &[Native],
) -> Result<StageSchedule, CompileError> {
    derive(
        &lowering.graph,
        &lowering.tables,
        &lowering.checkpoints,
        Some(lowering.logits),
        deferred,
        native,
    )
}

/// The schedule of `graph` with its natives: [`derive_graph_schedule`] and
/// [`derive_composed_schedule`] alike.
fn derive(
    graph: &Graph,
    tables: &[TableSpec],
    checkpoints: &[TensorId],
    logits: Option<TensorId>,
    deferred: &[Deferred],
    native: &[Native],
) -> Result<StageSchedule, CompileError> {
    let layers = LayerMap::of(graph, tables, checkpoints)?;
    let producers = producers(graph);
    check_graph(graph, tables, &producers)?;
    let lm = match logits {
        Some(logits) => Some(
            producers
                .get(logits.0 as usize)
                .copied()
                .flatten()
                .ok_or(CompileError::Schedule {
                    node: None,
                    rule: "the lowered graph has its logits producer",
                })?
                .0,
        ),
        None => None,
    };
    let site_node = |node: NodeId| {
        graph.nodes.get(node.0 as usize).filter(|n| {
            n.op.class() == OpClass::Gadget
                || (Some(n.id) == lm
                    && matches!(n.op,Op::MatMulWeight{weight} if weight.ty==WeightType::LmHead))
        })
    };
    for (i, n) in native.iter().enumerate() {
        if native[..i].iter().any(|other| other.tag == n.tag)
            || n.vars > 63
            || !(1..=7).contains(&n.degree)
            || n.evaluations == 0
            || site_node(n.node).is_none()
            || layers.position(n.node).is_some_and(|(l, _)| l != 0)
        {
            return refuse(
                Some(n.node),
                "a native descriptor has unique identity and legal model geometry",
            );
        }
        let lambda = if layers.position(n.node).is_some() {
            layers.vars
        } else {
            0
        };
        let own = n.vars.checked_sub(lambda).ok_or(CompileError::Schedule {
            node: Some(n.node),
            rule: "native layer coordinates fit its variables",
        })?;
        for out in &n.outputs {
            if out.from.checked_add(out.vars).is_none_or(|end| end > own)
                || graph.tensors.get(out.tensor.0 as usize).is_none()
                || tensor_vars(graph, out.tensor) != out.vars
            {
                return refuse(
                    Some(n.node),
                    "a native graph projection fits its original source domain",
                );
            }
        }
    }
    for (i, &(node, input)) in deferred.iter().enumerate() {
        let Some(definition) = site_node(node) else {
            return refuse(Some(node), "a deferred source belongs to a model site");
        };
        let tensor = match input {
            Some(i) => definition.inputs.get(i),
            None if Some(node) == lm => definition.outputs.first(),
            None => None,
        };
        if matches!(definition.op, Op::OneHot { .. }) && input == Some(0) {
            let token = definition.inputs[0];
            let h = definition.outputs[0];
            let total = blocks(&graph.tensors[h.0 as usize].shape).iter().sum();
            let time = blocks(&graph.tensors[token.0 as usize].shape).iter().sum();
            let vocabulary = total - time;
            if !native.iter().any(|n| {
                n.node == node
                    && n.zero
                    && n.degree == 2
                    && n.vars == total
                    && n.evaluations == 2
                    && n.outputs
                        == [
                            NativeOutput {
                                tensor: h,
                                from: 0,
                                vars: total,
                            },
                            NativeOutput {
                                tensor: token,
                                from: vocabulary,
                                vars: time,
                            },
                        ]
            }) {
                return refuse(
                    Some(node),
                    "the deferred OneHot index has its exact native token binding",
                );
            }
        }
        if deferred[..i].contains(&(node, input))
            || layers.position(node).is_some_and(|(l, _)| l != 0)
            || !native
                .iter()
                .any(|n| n.node == node && n.outputs.iter().any(|o| Some(&o.tensor) == tensor))
        {
            return refuse(
                Some(node),
                "each deferred graph leaf is replaced at that same native site",
            );
        }
    }
    let mut derivation = claims::Derivation::new(graph, &layers, producers);
    let mut sites = leaf_sites(graph, &layers, lm.is_some());
    sites.retain(|&s| {
        let key = match s {
            Site::Gadget { node, input } => (
                match node {
                    NodeRef::Node(n) => n,
                    NodeRef::Position(p) => layers.node(p, 0),
                },
                Some(input as usize),
            ),
            Site::Logits => match lm {
                Some(lm) => (lm, None),
                None => return true,
            },
        };
        !deferred.contains(&key)
    });
    let mut leaf = Stage {
        index: 0,
        instances: Vec::new(),
        rounds: 0,
        zero_vars: 0,
        claims: Vec::new(),
    };
    for (s, &site) in (0u32..).zip(&sites) {
        let (kind, weights) = match site {
            Site::Gadget {
                node: NodeRef::Position(position),
                input,
            } => {
                let kind = layers.input(position, input);
                if matches!(kind, Kind::Tensor(_)) {
                    return refuse(
                        Some(layers.node(position, 0)),
                        "a gadget in a layer reads a tensor that every layer shares",
                    );
                }
                (kind, Layers::Each)
            }
            Site::Gadget {
                node: NodeRef::Node(node),
                input,
            } => layers.outside(input_of(graph, node, input)),
            Site::Logits => layers.outside(logits.ok_or(CompileError::Schedule {
                node: None,
                rule: "a logits site reads the graph's logits",
            })?),
        };
        // A site that reads a public tensor, such as AES's counter blocks, leaves a claim of one
        // public term and no part: the leaf layout sends no value for it, and the verifier
        // evaluates it at the block's point (RFC-0045 'The split', RFC-0072 'Public tensors' 'In
        // the leaf stage').
        let vars = derivation.vars(kind);
        let point = (0..vars)
            .map(|index| Coord::Leaf { site: s, index })
            .collect();
        let claim = derivation.claim(0, leaf.claims.len(), kind, point, weights)?;
        leaf.claims.push(claim);
    }
    let mut stages = vec![leaf];
    loop {
        let previous = stages.last().map_or(&[][..], |s| &s.claims[..]);
        let index = u16::try_from(stages.len()).map_err(|_| CompileError::Schedule {
            node: None,
            rule: "a schedule of at most 65,535 stages",
        })?;
        let stage = relations::stage(&mut derivation, index, previous, &sites, native, deferred)?;
        if stage.instances.is_empty() {
            break;
        }
        // Native slots interleave with graph atoms; update the previous parts' actual indices.
        for (actual, instance) in stage.instances.iter().enumerate() {
            if let Source::Part { claim, part } = instance.source {
                let previous = stages.last_mut().ok_or(CompileError::Schedule {
                    node: None,
                    rule: "a reduction stage has its previous claims",
                })?;
                let Part::Instance { instance, .. } = &mut previous.claims[claim].parts[part]
                else {
                    return refuse(None, "an instance proves its previous virtual atom");
                };
                *instance = actual;
            }
        }
        stages.push(stage);
    }
    // Stage 1 draws the layer point `ρ_1` at which the leaf stage's per-layer claims stack
    // (RFC-0037 'The layer point'). A graph with no virtual leaf part and no zero-check has no
    // stage 1 (RFC-0072 'No reduction stage'), which only a graph without layers may lack.
    if layers.count > 0 && stages.len() == 1 {
        return refuse(
            None,
            "a graph with layers has a stage 1, which draws its layer point",
        );
    }
    let pool = derivation.into_pool();
    Ok(StageSchedule {
        layers,
        sites,
        stages,
        pool,
    })
}

/// Whether the leaf stage gives a node of `op` a site, one claim per input (RFC-0045 'Sites',
/// RFC-0072 'Gadgets and leaf sites'): every gadget but `BitDecompose` and `AssertEqual`, whose
/// claims all come from stage 1's zero-checks. `OneHot` keeps its site. The prover's leaf layout
/// lays out the same sites.
pub fn has_leaf_site(op: &Op) -> bool {
    op.class() == OpClass::Gadget && !matches!(op, Op::BitDecompose { .. } | Op::AssertEqual)
}

/// The leaf stage's sites: each input of each node with a site ([`has_leaf_site`]) in node order,
/// a template position's once, then the logits when the graph has them.
fn leaf_sites(graph: &Graph, layers: &LayerMap, logits: bool) -> Vec<Site> {
    let mut sites = Vec::new();
    for node in &graph.nodes {
        if !has_leaf_site(&node.op) {
            continue;
        }
        let node_ref = match layers.position(node.id) {
            Some((0, position)) => NodeRef::Position(position),
            Some(_) => continue,
            None => NodeRef::Node(node.id),
        };
        for input in 0..node.inputs.len() {
            sites.push(Site::Gadget {
                node: node_ref,
                input: u32::try_from(input).unwrap_or(u32::MAX),
            });
        }
    }
    if logits {
        sites.push(Site::Logits);
    }
    sites
}

/// RFC-0072's refusals that read the graph alone, before any claim (`ERR-INTERNAL`), in this
/// order:
///
/// - **A matmul with one public operand** ('Public tensors' 'As an operand'): its relations read
///   no operand as `Public`, whether or not a claim reaches it. With both operands public, its
///   output is public, and no instance arises.
/// - **A committed output** of a node that is not a gadget is proven by stage 1's equality
///   zero-check only for a pointwise `ConstMul`, `ConstAdd`, `Add`, `Sub` or `Mul` ('Linear
///   operations that are not coordinate maps' 'A committed output'): any other's, a selection's,
///   `Sum`'s, `SumSquares`'s, `Rotate`'s, `Transpose`'s, `Reshape`'s or `Expand`'s, is refused,
///   although the IR admits the declaration, whether or not a claim reaches it.
/// - **Summing relations** over a padded axis ('Summing relations').
/// - **A gadget's input non-zero at padding** ('Padded positions'): a gadget other than a lookup,
///   `BitDecompose` or `AssertEqual` that reads a tensor with padded positions that a lookup
///   non-zero at padding reaches.
fn check_graph(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
) -> Result<(), CompileError> {
    let public = public_tensors(graph);
    let is_public = |t: &TensorId| public.get(t.0 as usize).copied().unwrap_or(false);
    for node in &graph.nodes {
        // A matmul's relations read their operands as witness tables, never as `Public` (RFC-0072
        // 'Public tensors' 'As an operand'). With both operands public its output is public and
        // no instance arises; with one, the relation would need a public read it does not have.
        let matmul = matches!(node.op, Op::MatMulWeight { .. } | Op::MatMulAct { .. });
        if matmul && node.inputs.iter().any(is_public) && !node.inputs.iter().all(is_public) {
            return refuse(
                Some(node.id),
                "a matmul's operands are not public, unless both are",
            );
        }
        let committed = node.outputs.iter().any(|t| {
            graph
                .tensors
                .get(t.0 as usize)
                .is_some_and(|spec| matches!(spec.kind, NodeKind::Committed { .. }))
        });
        let pointwise = matches!(
            node.op,
            Op::ConstMul { .. } | Op::ConstAdd { .. } | Op::Add | Op::Sub | Op::Mul { .. }
        );
        if committed && node.op.class() != OpClass::Gadget && !pointwise {
            return refuse(
                Some(node.id),
                "a committed output is of a ConstMul, ConstAdd, Add, Sub or Mul",
            );
        }
    }
    if graph.slots.is_some() {
        let slotted = padding::slotted_tensors(graph, producers);
        for node in &graph.nodes {
            if matches!(node.op, Op::ConstAdd { .. } | Op::AssertEqual)
                && node
                    .inputs
                    .iter()
                    .any(|t| slotted.get(t.0 as usize).copied().unwrap_or(false))
            {
                return refuse(
                    Some(node.id),
                    "no ConstAdd or AssertEqual lies downstream of a slotted batch's token axis",
                );
            }
        }
    }
    padding::check_sums(graph, tables, producers)?;
    padding::check_gadgets(graph, tables, producers)
}

/// Input `input` of `node`.
fn input_of(graph: &Graph, node: NodeId, input: u32) -> TensorId {
    graph.nodes[node.0 as usize].inputs[input as usize]
}

/// The class of an atom on a tensor of kind `kind`, or `None` for a virtual or public one.
fn committed(kind: NodeKind) -> Option<CommitmentClass> {
    match kind {
        NodeKind::Committed { commitment } => Some(commitment),
        NodeKind::LookupOutput { .. } => Some(CommitmentClass::Advice),
        NodeKind::Virtual | NodeKind::Public => None,
    }
}

/// `⌈log2 len⌉`: the variables of an axis of `len` positions, padded to a power of two.
fn axis_vars(len: u32) -> u32 {
    u32::BITS - len.saturating_sub(1).leading_zeros()
}

/// The variables of each axis of `shape`, the last axis first: the blocks of a point, lowest
/// first (RFC-0011 rule 1).
fn blocks(shape: &[u32]) -> Vec<u32> {
    shape.iter().rev().map(|&len| axis_vars(len)).collect()
}

/// Whether `tensor` is an attention score or exponential tensor whose last axis is a slotted
/// batch's key axis of `S = 2^b` padded columns (RFC-0071 'The slots', 'Lowering').
pub(super) fn is_slot_key_tensor(
    graph: &Graph,
    slots: &tessera_ir::BatchSlots,
    tensor: TensorId,
) -> bool {
    let n = slots.token_len();
    let has_token_rows = |t: TensorId| {
        graph
            .tensors
            .get(t.0 as usize)
            .is_some_and(|spec| spec.shape.len() == 3 && spec.shape[1] == n)
    };
    if !has_token_rows(tensor) {
        return false;
    }
    graph.nodes.iter().any(|node| match node.op {
        Op::MatMulAct {
            transpose_rhs: true,
        } => {
            node.outputs.first() == Some(&tensor)
                && node.inputs.get(1).copied().is_some_and(has_token_rows)
        }
        Op::MatMulAct {
            transpose_rhs: false,
        } => {
            node.inputs.first() == Some(&tensor)
                && node.inputs.get(1).copied().is_some_and(has_token_rows)
        }
        Op::SoftmaxRow { .. } => {
            node.inputs.first() == Some(&tensor) || node.outputs.first() == Some(&tensor)
        }
        _ => false,
    })
}

/// The coordinate blocks of `tensor` in `graph`, lowest axis first, accounting for `S = 2^b` on a
/// slotted batch's key axis when `b > ⌈log2 n_max⌉` in `tessera-protocol/2`.
pub(super) fn tensor_blocks(graph: &Graph, tensor: TensorId) -> Vec<u32> {
    let mut b = blocks(&graph.tensors[tensor.0 as usize].shape);
    if let Some(slots) = &graph.slots
        && is_slot_key_tensor(graph, slots, tensor)
        && let Some(key) = b.first_mut()
    {
        *key = (*key).max(slots.slot_bits());
    }
    b
}

/// The variable count of `tensor` in `graph`, its layer variables aside.
pub(super) fn tensor_vars(graph: &Graph, tensor: TensorId) -> u32 {
    tensor_blocks(graph, tensor).iter().sum()
}

/// A refusal.
fn refuse<T>(node: Option<NodeId>, rule: &'static str) -> Result<T, CompileError> {
    Err(CompileError::Schedule { node, rule })
}
