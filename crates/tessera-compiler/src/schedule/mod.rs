// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedule (RFC-0010 'Stages', RFC-0037): the claims each stage leaves, how each splits,
//! the relation instances that prove them, and what reaches the claim pool, derived from the
//! lowered graph alone (INV-SCH-1).
//!
//! [`derive_schedule`] reads a [`Lowering`] in four steps:
//!
//! 1. **Layers.** The checkpoints `c_0, …, c_L` bound the layers. Every layer must be one template,
//!    node for node ([`LayerMap`]), so that the outputs at one position form one kind stacked over
//!    the layers, with `λ = ⌈log2 L⌉` layer variables, the highest (RFC-0037 'Layers and kinds').
//! 2. **The leaf stage.** It leaves one claim per input of each gadget, and one on the logits that
//!    the scoring certificates read (RFC-0017), each at a point of its site. A site in the layers
//!    leaves one claim per layer, all at that point ('The layer point'). This is the leaf stage's
//!    interface to the rest of the schedule. The gadgets' own relations, such as `SoftmaxRow`'s
//!    masked keys and the padding zero-checks of RFC-0035, enter with the gadgets (#118).
//! 3. **Claims.** A claim maps through the linear nodes onto atoms: virtual, the output of a
//!    sumcheck relation, or committed ('Claims and atoms'). A claim that is one virtual atom of
//!    coefficient 1, or committed atoms of one commitment at one point, is ready. Any other is
//!    split: the stage sends one value per part ('Splits').
//! 4. **Stages.** Stage `k` proves one instance per virtual part that stage `k − 1` leaves, in
//!    order, stacked over the layers where its kind is. Stage 1 then adds the equality zero-check
//!    of each committed output of a linear or degree-2 node ('The committed first checkpoint').
//!    Each instance's evaluations are the stage's claims, which go to the next stage or the pool.
//!
//! Every point is symbolic: each [`Coord`] names the challenge, or the constant `½`, that sets it.
//! So the schedule fixes which atoms share a point, and so how claims merge, split and group, before
//! any challenge is drawn. The prover evaluates it at the proof's challenges; the independent
//! verifier derives the same schedule from RFC-0037 itself (#114).

mod claims;
mod layers;
mod relations;

use core::ops::Range;

use tessera_ir::{Broadcast, CommitmentClass, Graph, NodeId, NodeKind, Op, TensorId, WeightType};

use crate::{CompileError, Lowering};

pub use layers::LayerMap;

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

/// A term of a claim after substitution: `Σ_ℓ w(ℓ)·c(ℓ)·X̃_{κ,ℓ}(r)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// The kind `κ`.
    pub kind: Kind,
    /// The point `r` on the kind's own variables, its layer variables aside.
    pub point: Vec<Coord>,
    /// The layer weights `w`.
    pub layers: Layers,
    /// The coefficient `c`: a sum of products, one for each atom merged into this one, in order.
    /// It is 1 when it is one empty product.
    pub coefficient: Vec<Vec<Factor>>,
    /// Virtual or committed.
    pub class: AtomClass,
}

impl Atom {
    /// Whether the coefficient is the constant 1: one empty product.
    pub fn is_unit(&self) -> bool {
        matches!(self.coefficient.as_slice(), [product] if product.is_empty())
    }
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
    /// place.
    pub atoms: Vec<Atom>,
    /// Whether the stage splits it, sending one value per part under `stage/{k}/split`.
    pub split: bool,
    /// Its parts, in atom order: one when the claim is ready.
    pub parts: Vec<Part>,
}

/// A relation an instance proves (RFC-0010 'Relations', RFC-0037 'Rotate').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    /// A native stage relation, whose tables and source transformations the CPU backend supplies.
    Native {
        /// Verifier-independent stable identity in the compiler's native plan.
        tag: u32,
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
    /// The equality zero-check `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))` of a node's committed output `Y`;
    /// its evaluations are `Y`'s, then those of `f`'s inputs.
    Equality,
    /// The original OneHot's `0 = sum_x eq(z,x) H(x)(1-H(x))`; one evaluation.
    Booleanity,
    /// The original indicator is zero outside the real vocabulary; one H evaluation.
    OneHotVocabulary,
    /// Zero-check that each indicator row sums to its public row selector; one H evaluation.
    OneHotCount,
    /// Zero-check that its weighted sum is the original token; H then token evaluations.
    OneHotIndex,
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
    /// The kind of the node's output it states.
    pub kind: Kind,
    /// Its claim's point on that output: the virtual part's, or for a zero-check its coordinates
    /// of the zero-check point.
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
    /// 1 the zero-checks in node order. The leaf stage has none here: its fraction trees and
    /// product trees are RFC-0008's and RFC-0009's.
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
    /// The logits, which the scoring certificates read (RFC-0017).
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
    /// The leaf stage's sites, in node order, then the logits: the leaf stage's claim `s` is site
    /// `s`'s.
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

/// The stage schedule of `lowering`, derived from its graph and checkpoints alone (INV-SCH-1).
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for a graph the lowering does not emit: layers that
/// are not one template ([`LayerMap`]), or a claim that reaches an operation RFC-0037 maps no claim
/// through, such as `ConstAdd`, `Slice` or `GatherPublic`.
pub fn derive_schedule(lowering: &Lowering) -> Result<StageSchedule, CompileError> {
    derive_composed_schedule(lowering, &[], &[])
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
    let layers = LayerMap::new(lowering)?;
    let graph = &lowering.graph;
    let lm = graph
        .nodes
        .iter()
        .find(|n| n.outputs.contains(&lowering.logits))
        .ok_or(CompileError::Schedule {
            node: None,
            rule: "the lowered graph has its logits producer",
        })?
        .id;
    let site_node = |node: NodeId| {
        graph.nodes.get(node.0 as usize).filter(|n| {
            n.op.class() == tessera_ir::OpClass::Gadget
                || (n.id == lm
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
                || graph
                    .tensors
                    .get(out.tensor.0 as usize)
                    .is_none_or(|t| blocks(&t.shape).iter().sum::<u32>() != out.vars)
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
            None if node == lm => definition.outputs.first(),
            None => None,
        };
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
    let mut derivation = claims::Derivation::new(&lowering.graph, &layers);
    let mut sites = leaf_sites(&lowering.graph, &layers);
    sites.retain(|&s| {
        let key = match s {
            Site::Gadget { node, input } => (
                match node {
                    NodeRef::Node(n) => n,
                    NodeRef::Position(p) => layers.node(p, 0),
                },
                Some(input as usize),
            ),
            Site::Logits => (lm, None),
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
            } => layers.outside(input_of(&lowering.graph, node, input)),
            Site::Logits => layers.outside(lowering.logits),
        };
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
        let stage = relations::stage(&mut derivation, index, previous, &sites, native)?;
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
    let pool = derivation.into_pool();
    Ok(StageSchedule {
        layers,
        sites,
        stages,
        pool,
    })
}

/// The leaf stage's sites: each input of each gadget in node order, a template position's once,
/// then the logits.
fn leaf_sites(graph: &Graph, layers: &LayerMap) -> Vec<Site> {
    let mut sites = Vec::new();
    let mut gadget = |node: NodeRef, op: &Op, inputs: usize| {
        if op.class() == tessera_ir::OpClass::Gadget {
            for input in 0..inputs {
                sites.push(Site::Gadget {
                    node,
                    input: u32::try_from(input).unwrap_or(u32::MAX),
                });
            }
        }
    };
    for node in &graph.nodes {
        let id = node.id.0;
        if id < layers.first {
            gadget(NodeRef::Node(node.id), &node.op, node.inputs.len());
        } else if let Some(position) = id.checked_sub(layers.first).filter(|&p| p < layers.width) {
            gadget(NodeRef::Position(position), &node.op, node.inputs.len());
        } else if id >= layers.end() {
            gadget(NodeRef::Node(node.id), &node.op, node.inputs.len());
        }
    }
    sites.push(Site::Logits);
    sites
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
    len.next_power_of_two().trailing_zeros()
}

/// The variables of each axis of `shape`, the last axis first: the blocks of a point, lowest
/// first (RFC-0011 rule 1).
fn blocks(shape: &[u32]) -> Vec<u32> {
    shape.iter().rev().map(|&len| axis_vars(len)).collect()
}

/// A refusal.
fn refuse<T>(node: Option<NodeId>, rule: &'static str) -> Result<T, CompileError> {
    Err(CompileError::Schedule { node, rule })
}
