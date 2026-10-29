// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-protocol/2`'s round degree of a reduction stage (RFC-0013 'Masked extensions'
//! 'Degree', #565).
//!
//! In `tessera-protocol/2` a relation reads each committed or virtual tensor through its masked
//! extension `V' = Ṽ + Z·R̃_V`, and `Z = Π_{i ≥ 2} x_i(1 − x_i)` adds 2 to that read's degree in
//! the round that binds its last variable. RFC-0013's table gives each relation's degree `d`,
//! keyed by what the schedule decides ([`DegreeKey`]):
//!
//! - the relation, or a native relation's kind, which the schedule's native instance carries
//!   ([`Relation::Native`]);
//! - the `BoolFn`, and an equality zero-check's operation;
//! - for an `Equality.ConstMul`, whether its constant is a scalar, per channel or per layer
//!   ([`Constant`]);
//! - which operands are public (RFC-0072 'Public tensors'), and for the matmuls, whether the
//!   instance is stacked.
//!
//! [`degree_row`] is the table. [`protocol2_degree`] keys a schedule's instance and reads its row,
//! and [`protocol2_stages`] gives each reduction stage's rounds, folds and degree, the largest of
//! its instances' and its folds'. [`stage_degree`] checks a stage's degree against its protocol's
//! bound, `d ≤ 10` in `tessera-protocol/2` and `d ≤ 7` in `tessera-protocol/1`
//! ([`Protocol::max_degree`]). An instance that matches no row and a stage above its bound are
//! refused, `ERR-INTERNAL`.

use core::fmt;

use tessera_ir::{BoolFn, Graph, Node, Op, TensorId};
use tessera_sumcheck::{Prefix, Protocol};

use super::{
    Instance, Kind, Layers, NativeKind, NodeRef, Relation, Source, StageSchedule, public_tensors,
    refuse,
};
use crate::CompileError;

/// The degree of a fold (RFC-0013 'Virtual tensors'): `eq` times the mask row, read unmasked.
pub const FOLD_DEGREE: usize = 2;

/// An equality zero-check's operation (RFC-0037 'The committed first checkpoint').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EqualityOp {
    /// `ConstMul`.
    ConstMul,
    /// `ConstAdd`.
    ConstAdd,
    /// `Add`.
    Add,
    /// `Sub`.
    Sub,
    /// `Mul`.
    Mul,
}

/// What RFC-0013's table names in its `relation` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DegreeRelation {
    /// `MatMulWeight`.
    MatMulWeight,
    /// `MatMulAct`.
    MatMulAct,
    /// `Mul`, the relation of a virtual product.
    Mul,
    /// `SumSquares`.
    SumSquares,
    /// `Rotate`.
    Rotate,
    /// `Selection`, of a `Slice` or a `GatherPublic`.
    Selection,
    /// `ScaleChannels`.
    ScaleChannels,
    /// A `BitFn`: `Xor`, `And`, `Ch`, `Xor3` or `Maj`.
    BitFn(BoolFn),
    /// An equality zero-check: `Equality.ConstMul` and the others.
    Equality(EqualityOp),
    /// `Booleanity`.
    Booleanity,
    /// `OneHotVocabulary`.
    OneHotVocabulary,
    /// `OneHotCount`.
    OneHotCount,
    /// `OneHotIndex`.
    OneHotIndex,
    /// `InputPadding`.
    InputPadding,
    /// `BitPadding`.
    BitPadding,
    /// `Recomposition`.
    Recomposition,
    /// `AssertEqual`.
    AssertEqual,
    /// A native relation, by the native plan's kind.
    Native(NativeKind),
    /// A fold (RFC-0013 'Virtual tensors'), which is no instance of the schedule.
    Fold,
}

impl DegreeRelation {
    /// The table's string for the relation, which `spec/vectors/schedule/degrees.json` records.
    pub fn name(self) -> &'static str {
        match self {
            Self::MatMulWeight => "MatMulWeight",
            Self::MatMulAct => "MatMulAct",
            Self::Mul => "Mul",
            Self::SumSquares => "SumSquares",
            Self::Rotate => "Rotate",
            Self::Selection => "Selection",
            Self::ScaleChannels => "ScaleChannels",
            Self::BitFn(f) => match f {
                BoolFn::Xor => "Xor",
                BoolFn::Xor3 => "Xor3",
                BoolFn::And => "And",
                BoolFn::Ch => "Ch",
                BoolFn::Maj => "Maj",
            },
            Self::Equality(op) => match op {
                EqualityOp::ConstMul => "Equality.ConstMul",
                EqualityOp::ConstAdd => "Equality.ConstAdd",
                EqualityOp::Add => "Equality.Add",
                EqualityOp::Sub => "Equality.Sub",
                EqualityOp::Mul => "Equality.Mul",
            },
            Self::Booleanity => "Booleanity",
            Self::OneHotVocabulary => "OneHotVocabulary",
            Self::OneHotCount => "OneHotCount",
            Self::OneHotIndex => "OneHotIndex",
            Self::InputPadding => "InputPadding",
            Self::BitPadding => "BitPadding",
            Self::Recomposition => "Recomposition",
            Self::AssertEqual => "AssertEqual",
            Self::Native(kind) => match kind {
                NativeKind::Product => "Product",
                NativeKind::RootPad => "RootPad",
                NativeKind::ResidualPad => "ResidualPad",
                NativeKind::Key => "Key",
                NativeKind::Forest => "Forest",
                NativeKind::Reciprocal => "Reciprocal",
                NativeKind::MaxPad => "MaxPad",
                NativeKind::ReciprocalPad => "ReciprocalPad",
                NativeKind::Tokens => "Tokens",
                NativeKind::GreedyKey => "GreedyKey",
                NativeKind::GreedySelected => "GreedySelected",
                NativeKind::Observe => "Observe",
            },
            Self::Fold => "Fold",
        }
    }
}

impl fmt::Display for DegreeRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The class of an `Equality.ConstMul`'s constant (RFC-0013 'Degree' 'Per layer').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    /// One value: outside the layers, or at a template position of a graph with `λ = 0`.
    Scalar,
    /// One value per channel of the last axis, outside the layers (RFC-0072 'Not inside a
    /// layer').
    Channel,
    /// At a template position of a graph with `λ ≥ 1`, whatever the layers' values.
    Layer,
}

impl Constant {
    /// The vector's string: `scalar`, `channel` or `layer`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Channel => "channel",
            Self::Layer => "layer",
        }
    }
}

/// What RFC-0013's table keys an instance by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DegreeKey {
    /// The relation, or a native's kind.
    pub relation: DegreeRelation,
    /// Stacked: layer weights over `λ ≥ 1` coordinates, or a `MatMulAct` whose operands' leading
    /// axes hold two heads or more ('Degree' 'Stacked'). Only the matmuls' rows read it.
    pub stacked: bool,
    /// One Boolean per operand, in the order the relation's final message would list them if
    /// none were public ('Degree' 'Operands'): empty for a native and a fold.
    pub public: Vec<bool>,
    /// An `Equality.ConstMul`'s constant class; `None` for every other relation.
    pub constant: Option<Constant>,
}

/// A row of RFC-0013's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DegreeRow {
    /// `d_1`, the factors of the term that sets `d`: one `eq`, the public factors and the reads.
    /// A fold's, which the table leaves blank, is the rule's 2: `eq` and its mask row.
    pub d_1: usize,
    /// `s`, that term's masked reads: a fold's, blank in the table, is 0.
    pub s: usize,
    /// `d = d_1 + 2s`, the round messages' length in a stage where it is the largest.
    pub d: usize,
    /// The largest degree the row's tiny instance reaches in a round ('Testing Strategy'): `d`
    /// on a row the table marks tight, else the degree its "Tight" cell gives.
    pub reach: usize,
}

impl DegreeRow {
    /// A row by the rule, `d = d_1 + 2s`, reaching `reach`.
    const fn rule(d_1: usize, s: usize, reach: usize) -> Self {
        Self {
            d_1,
            s,
            d: d_1 + 2 * s,
            reach,
        }
    }

    /// A row by the rule that its tiny instance reaches: the table's "yes".
    const fn tight(d_1: usize, s: usize) -> Self {
        Self::rule(d_1, s, d_1 + 2 * s)
    }

    /// Whether the table marks the row tight.
    pub fn is_tight(&self) -> bool {
        self.reach == self.d
    }
}

/// A refusal of the degree's derivation: `ERR-INTERNAL`.
fn no_row<T>() -> Result<T, CompileError> {
    refuse(
        None,
        "a reduction stage's instance matches a row of RFC-0013's degree table",
    )
}

/// RFC-0013 'Masked extensions' 'Degree''s table: the row of `key`, `tessera-protocol/2`'s
/// degree of a reduction stage's instance, from stage 1 on.
///
/// | `relation` | Relation and public operands | `d_1` | `s` | `d` | Tight |
/// |---|---|---|---|---|---|
/// | `MatMulWeight`, `MatMulAct` | no public operand | 2, or 3 stacked | 2 | 6, or 7 stacked | yes |
/// | `Mul` | none public; one operand public | 3 | 2; 1 | 7; 5 | yes |
/// | `SumSquares` | | 3 | 2 | 7 | yes |
/// | `Rotate`, `Selection` | the kernel as `eq` | 2 | 1 | 4 | yes |
/// | `ScaleChannels` | | 3 | 1 | 5 | yes |
/// | `Xor`, `And`, `Ch` | none public | 3 | 2 | 7 | yes |
/// | `Xor`, `And` | one input public | 3 | 1 | 5 | yes |
/// | `Ch` | `e`, or any two inputs, public | 3 | 1 | 5 | yes |
/// | `Ch` | `f` or `g` alone public | 3 | 2 | 7 | yes |
/// | `Xor3`, `Maj` | none; one; two inputs public | 4 | 3; 2; 1 | 10; 8; 6 | yes |
/// | `Equality.ConstMul` (scalar), `.ConstAdd`, `.Add`, `.Sub` | whatever inputs are public | 2 | 1 | 4 | yes |
/// | `Equality.ConstMul` | per channel or per layer; its input public | 3; 2 | 1 | 5; 4 | yes |
/// | `Equality.Mul` | none; one; both inputs public | 3; 3; 2 | 2; 1; 1 | 7; 5; 4 | yes |
/// | `Booleanity` | | 3 | 2 | 7 | yes |
/// | `OneHotVocabulary`, `InputPadding`, `BitPadding` | | 3 | 1 | 5 | yes |
/// | `OneHotIndex` | its token public or not | 3 | 1 | 5 | no: 4 |
/// | `OneHotCount` | | 2 | 1 | 4 | yes |
/// | `AssertEqual` | one side public or none; both | 3 | 1; 0 | 5; 3 | yes |
/// | `Recomposition` | `x` public or not | 3 | 1 | 5 | no: 4 |
/// | `Product` | | 4 | 3 | 10 | yes |
/// | `RootPad`, `ResidualPad`, `Key`, `Forest`, `MaxPad`, `ReciprocalPad` | | 3 | 1 | 5 | yes |
/// | `Reciprocal` | | 3 | 2 | 7 | yes |
/// | `GreedySelected` | provisional | 3 | 2 | 7 | no: 5 |
/// | `Tokens`, `GreedyKey` | provisional | 3 | 1 | 5 | no: 4 |
/// | `Observe` | provisional, a zero summand | 1 | 1 | 3 | no: 0 |
/// | `Fold` | | — | — | 2 | yes |
///
/// A row that names no public operand holds only when no operand is public, and a row that
/// names a pattern holds for that pattern. A key's [`DegreeKey::public`] holds one entry per
/// operand of its relation, and only an `Equality.ConstMul` has a [`Constant`]. Only the
/// matmuls read [`DegreeKey::stacked`]: every other relation's stacking weight is part of its
/// `eq` factor.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for a key that matches no row: a relation with
/// another number of operands, a public operand where its row names none, such as a matmul's
/// (RFC-0072 'Public tensors' 'As an operand'), every input of a `BitFn` public, a constant
/// class on a relation other than `Equality.ConstMul` or none on it, or a public operand of a
/// native or a fold.
pub fn degree_row(key: &DegreeKey) -> Result<DegreeRow, CompileError> {
    use DegreeRelation as R;
    let public = key.public.as_slice();
    let count = public.iter().filter(|&&p| p).count();
    let operands = |n: usize| public.len() == n;
    let constant = key.constant;
    if constant.is_some() != (key.relation == R::Equality(EqualityOp::ConstMul)) {
        return no_row();
    }
    let row = match key.relation {
        R::MatMulWeight | R::MatMulAct if operands(2) && count == 0 => {
            if key.stacked {
                DegreeRow::tight(3, 2)
            } else {
                DegreeRow::tight(2, 2)
            }
        }
        R::Mul if operands(2) && count == 0 => DegreeRow::tight(3, 2),
        R::Mul if operands(2) && count == 1 => DegreeRow::tight(3, 1),
        R::SumSquares if operands(1) && count == 0 => DegreeRow::tight(3, 2),
        R::Rotate | R::Selection if operands(1) && count == 0 => DegreeRow::tight(2, 1),
        R::ScaleChannels if operands(1) && count == 0 => DegreeRow::tight(3, 1),
        R::BitFn(BoolFn::Xor | BoolFn::And) if operands(2) => match count {
            0 => DegreeRow::tight(3, 2),
            1 => DegreeRow::tight(3, 1),
            _ => return no_row(),
        },
        // `Ch(e, f, g) = ef + g − eg`: `e`, or two inputs, public leave one read per term; `f`
        // or `g` alone public leaves `e·g` or `e·f`, two reads.
        R::BitFn(BoolFn::Ch) if operands(3) => match (public[0], count) {
            (_, 0) => DegreeRow::tight(3, 2),
            (true, 1) | (_, 2) => DegreeRow::tight(3, 1),
            (false, 1) => DegreeRow::tight(3, 2),
            _ => return no_row(),
        },
        R::BitFn(BoolFn::Xor3 | BoolFn::Maj) if operands(3) && count < 3 => {
            DegreeRow::tight(4, 3 - count)
        }
        // Operands: the committed output `Y`, never public, then `f`'s inputs.
        R::Equality(op) if public.first() == Some(&false) => {
            let inputs = public.len() - 1;
            match (op, inputs, constant) {
                (EqualityOp::ConstMul, 1, Some(Constant::Scalar))
                | (EqualityOp::ConstAdd, 1, None)
                | (EqualityOp::Add | EqualityOp::Sub, 2, None) => DegreeRow::tight(2, 1),
                (EqualityOp::ConstMul, 1, Some(Constant::Channel | Constant::Layer)) => {
                    if count == 0 {
                        DegreeRow::tight(3, 1)
                    } else {
                        DegreeRow::tight(2, 1)
                    }
                }
                (EqualityOp::Mul, 2, None) => match count {
                    0 => DegreeRow::tight(3, 2),
                    1 => DegreeRow::tight(3, 1),
                    _ => DegreeRow::tight(2, 1),
                },
                _ => return no_row(),
            }
        }
        R::Booleanity if operands(1) && count == 0 => DegreeRow::tight(3, 2),
        R::OneHotVocabulary | R::InputPadding | R::BitPadding if operands(1) && count == 0 => {
            DegreeRow::tight(3, 1)
        }
        // `H`, never public, then the token, public or not.
        R::OneHotIndex if operands(2) && !public[0] => DegreeRow::rule(3, 1, 4),
        R::OneHotCount if operands(1) && count == 0 => DegreeRow::tight(2, 1),
        R::AssertEqual if operands(2) => match count {
            2 => DegreeRow::tight(3, 0),
            _ => DegreeRow::tight(3, 1),
        },
        // `x`, public or not, then the bits `b` and `h`, committed.
        R::Recomposition if (operands(2) || operands(3)) && !public[1..].contains(&true) => {
            DegreeRow::rule(3, 1, 4)
        }
        R::Native(kind) if public.is_empty() => match kind {
            NativeKind::Product => DegreeRow::tight(4, 3),
            NativeKind::RootPad
            | NativeKind::ResidualPad
            | NativeKind::Key
            | NativeKind::Forest
            | NativeKind::MaxPad
            | NativeKind::ReciprocalPad => DegreeRow::tight(3, 1),
            NativeKind::Reciprocal => DegreeRow::tight(3, 2),
            NativeKind::GreedySelected => DegreeRow::rule(3, 2, 5),
            NativeKind::Tokens | NativeKind::GreedyKey => DegreeRow::rule(3, 1, 4),
            // A zero summand, by convention RFC-0053's degree 1 and its one evaluation.
            NativeKind::Observe => DegreeRow::rule(1, 1, 0),
        },
        R::Fold if public.is_empty() => DegreeRow::tight(2, 0),
        _ => return no_row(),
    };
    Ok(row)
}

/// A stage's degree, the largest of `degrees`, checked against `protocol`'s bound at stage
/// `stage` ([`Protocol::max_degree`]): `d ≤ 10` at a reduction stage of `tessera-protocol/2`,
/// `d ≤ 7` in `tessera-protocol/1` (RFC-0013 'Degree' 'The invariant'). No relation of the table
/// exceeds 10, so no input reaches the refusal today; it is the compiler's check, beside the
/// sumcheck engine's own.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for a stage above its bound, or of no degree.
pub fn stage_degree(
    protocol: Protocol,
    stage: u16,
    degrees: impl IntoIterator<Item = usize>,
) -> Result<usize, CompileError> {
    let d = degrees.into_iter().max().unwrap_or(0);
    if d == 0 {
        return refuse(None, "a reduction stage has an instance");
    }
    if d > protocol.max_degree(Prefix::Stage(u32::from(stage))) {
        return refuse(
            None,
            "a stage's degree is within its protocol's bound: 10 at a reduction stage of \
             tessera-protocol/2, 7 in tessera-protocol/1",
        );
    }
    Ok(d)
}

/// An instance's key and row.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceDegree {
    /// Its key.
    pub key: DegreeKey,
    /// Its row.
    pub row: DegreeRow,
}

/// A reduction stage's `tessera-protocol/2` degree.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StageDegree {
    /// The stage's index, from 1.
    pub index: u16,
    /// `tessera-protocol/2`'s rounds: `max(max_i n_i, log2 k)` when the stage has a fold
    /// (RFC-0013 'Virtual tensors' 'The stage's rounds'), `max_i n_i` otherwise.
    pub rounds: u32,
    /// Its folds: one per instance whose claim is a virtual part the stage before left (RFC-0037
    /// 'Stages and their instances'), none for a native or a zero-check.
    pub folds: usize,
    /// `d`, the largest of its instances' and its folds' ([`FOLD_DEGREE`]).
    pub d: usize,
    /// Each instance's key and row, in schedule order.
    pub instances: Vec<InstanceDegree>,
}

/// The keys of a schedule's instances: the graph, its public tensors, and its layers.
struct Keys<'a> {
    graph: &'a Graph,
    schedule: &'a StageSchedule,
    public: Vec<bool>,
}

impl<'a> Keys<'a> {
    fn new(graph: &'a Graph, schedule: &'a StageSchedule) -> Self {
        Self {
            graph,
            schedule,
            public: public_tensors(graph),
        }
    }

    /// Whether a claim on `kind` is public (RFC-0072 'Public tensors'): a tensor outside the
    /// layers that the verifier computes.
    fn public(&self, kind: Kind) -> bool {
        match kind {
            Kind::Tensor(t) => self.public.get(t.0 as usize).copied().unwrap_or(false),
            _ => false,
        }
    }

    /// The node an instance's relation is, layer 0's for a template position.
    fn node(&self, node: NodeRef) -> Result<&'a Node, CompileError> {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(position) => self.schedule.layers.node(position, 0),
        };
        self.graph
            .nodes
            .get(id.0 as usize)
            .ok_or(CompileError::Schedule {
                node: Some(id),
                rule: "an instance's node is the graph's",
            })
    }

    /// The kind of input `input` of `node`, as the schedule reads it.
    fn input(&self, node: NodeRef, graph_node: &Node, input: usize) -> Result<Kind, CompileError> {
        let tensor = |i: usize| {
            graph_node
                .inputs
                .get(i)
                .copied()
                .ok_or(CompileError::Schedule {
                    node: Some(graph_node.id),
                    rule: "an operand is an input of its node",
                })
        };
        Ok(match node {
            NodeRef::Position(position) => self
                .schedule
                .layers
                .input(position, u32::try_from(input).unwrap_or(u32::MAX)),
            NodeRef::Node(_) => self.schedule.layers.outside(tensor(input)?).0,
        })
    }

    /// Whether each input of `node` in `inputs` is public.
    fn inputs(
        &self,
        node: NodeRef,
        graph_node: &Node,
        inputs: &[usize],
    ) -> Result<Vec<bool>, CompileError> {
        inputs
            .iter()
            .map(|&i| Ok(self.public(self.input(node, graph_node, i)?)))
            .collect()
    }

    /// Whether a tensor outside the layers is public.
    fn tensor(&self, tensor: TensorId) -> bool {
        self.public(self.schedule.layers.outside(tensor).0)
    }

    fn key(&self, instance: &Instance) -> Result<DegreeKey, CompileError> {
        use DegreeRelation as R;
        let layered = matches!(&instance.layers, Layers::Eq(rho) if !rho.is_empty());
        let mut stacked = layered;
        let mut constant = None;
        if let Relation::Native { kind, .. } = instance.relation {
            return Ok(DegreeKey {
                relation: R::Native(kind),
                stacked,
                public: Vec::new(),
                constant,
            });
        }
        let node = self.node(instance.node)?;
        let at = instance.node;
        let output = self.public(instance.kind);
        let (relation, public) = match instance.relation {
            Relation::Native { .. } => return no_row(),
            // Its evaluations are `W`'s, then `X`'s: the node's inputs 1 and 0.
            Relation::MatMulWeight => (R::MatMulWeight, self.inputs(at, node, &[1, 0])?),
            Relation::MatMulAct { .. } => {
                // Operands whose leading axes, before their last two, hold two heads or more, or
                // a slotted batch whose `MatMulAct` stacks over the `R ≥ 2` request slots.
                let first = node
                    .inputs
                    .first()
                    .and_then(|t| self.graph.tensors.get(t.0 as usize));
                let heads: u64 = first
                    .map(|t| {
                        let lead = t.shape.len().saturating_sub(2);
                        t.shape[..lead].iter().map(|&n| u64::from(n)).product()
                    })
                    .unwrap_or(1);
                let slotted = self.graph.slots.as_ref().is_some_and(|slots| {
                    first.is_some_and(|t| t.shape.len() == 3 && t.shape[1] == slots.token_len())
                });
                stacked |= heads >= 2 || slotted;
                (R::MatMulAct, self.inputs(at, node, &[0, 1])?)
            }
            Relation::Mul(_) => (R::Mul, self.inputs(at, node, &[0, 1])?),
            Relation::SumSquares => (R::SumSquares, self.inputs(at, node, &[0])?),
            // `X` alone: its positions, public, build the kernel.
            Relation::Rotate => (R::Rotate, self.inputs(at, node, &[0])?),
            Relation::ScaleChannels => (R::ScaleChannels, self.inputs(at, node, &[0])?),
            Relation::Selection => (R::Selection, self.inputs(at, node, &[0])?),
            Relation::BitFn(f) => {
                let inputs: Vec<usize> = (0..node.inputs.len()).collect();
                (R::BitFn(f), self.inputs(at, node, &inputs)?)
            }
            Relation::Equality => {
                let (op, inputs): (EqualityOp, &[usize]) = match node.op {
                    Op::ConstMul { c } => {
                        constant = Some(match at {
                            NodeRef::Position(_) if self.schedule.layers.vars >= 1 => {
                                Constant::Layer
                            }
                            NodeRef::Position(_) => Constant::Scalar,
                            NodeRef::Node(_) => {
                                match self.graph.constants.get(c.0 as usize).map(Vec::len) {
                                    Some(1) => Constant::Scalar,
                                    Some(_) => Constant::Channel,
                                    None => return no_row(),
                                }
                            }
                        });
                        (EqualityOp::ConstMul, &[0])
                    }
                    Op::ConstAdd { .. } => (EqualityOp::ConstAdd, &[0]),
                    Op::Add => (EqualityOp::Add, &[0, 1]),
                    Op::Sub => (EqualityOp::Sub, &[0, 1]),
                    Op::Mul { .. } => (EqualityOp::Mul, &[0, 1]),
                    _ => return no_row(),
                };
                let public = [vec![output], self.inputs(at, node, inputs)?].concat();
                (R::Equality(op), public)
            }
            Relation::Booleanity => (R::Booleanity, vec![output]),
            Relation::OneHotVocabulary => (R::OneHotVocabulary, vec![output]),
            Relation::OneHotCount => (R::OneHotCount, vec![output]),
            Relation::OneHotIndex => (
                R::OneHotIndex,
                [vec![output], self.inputs(at, node, &[0])?].concat(),
            ),
            Relation::InputPadding => (R::InputPadding, vec![output]),
            Relation::BitPadding => (R::BitPadding, vec![output]),
            // `x`, `b`, then `h` when the decomposition has high bits.
            Relation::Recomposition => {
                let bits = node.outputs.iter().map(|&t| self.tensor(t));
                let public = self.inputs(at, node, &[0])?.into_iter().chain(bits);
                (R::Recomposition, public.collect())
            }
            Relation::AssertEqual => (R::AssertEqual, self.inputs(at, node, &[0, 1])?),
        };
        Ok(DegreeKey {
            relation,
            stacked,
            public,
            constant,
        })
    }

    fn instance(&self, instance: &Instance) -> Result<InstanceDegree, CompileError> {
        let key = self.key(instance)?;
        let row = degree_row(&key)?;
        Ok(InstanceDegree { key, row })
    }
}

/// `tessera-protocol/2`'s degree of `instance`, an instance of a reduction stage of `schedule`,
/// a schedule of `graph`: its key, read from the schedule and the graph, and its row of RFC-0013
/// 'Masked extensions' 'Degree''s table ([`degree_row`]), whose `d` is the instance's degree.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for an instance that matches no row, or whose
/// node is not `graph`'s.
pub fn protocol2_degree(
    graph: &Graph,
    schedule: &StageSchedule,
    instance: &Instance,
) -> Result<InstanceDegree, CompileError> {
    Keys::new(graph, schedule).instance(instance)
}

/// Each reduction stage of `schedule`, a schedule of `graph`, in `tessera-protocol/2`, its advice
/// rows of `2^{log2_k}` entries: its rounds, its folds, its instances' rows and its degree `d`,
/// the largest of theirs and of its folds', within the bound ([`stage_degree`]). RFC-0013
/// 'Masked extensions' 'Degree' gives the rows; `spec/vectors/schedule/degrees.json` records
/// them.
///
/// The fold count is the current schedules': one per instance of a virtual part. A combined
/// selection's part has several members, each with its own mask row, and its count is #145's.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) for an instance that matches no row, or a stage
/// above `tessera-protocol/2`'s bound.
pub fn protocol2_stages(
    graph: &Graph,
    schedule: &StageSchedule,
    log2_k: u32,
) -> Result<Vec<StageDegree>, CompileError> {
    let keys = Keys::new(graph, schedule);
    schedule
        .stages
        .iter()
        .skip(1)
        .map(|stage| {
            let instances = stage
                .instances
                .iter()
                .map(|i| keys.instance(i))
                .collect::<Result<Vec<_>, _>>()?;
            let folds = stage
                .instances
                .iter()
                .filter(|i| matches!(i.source, Source::Part { .. }))
                .count();
            let rounds = if folds > 0 {
                stage.rounds.max(log2_k)
            } else {
                stage.rounds
            };
            let degrees = instances.iter().map(|i| i.row.d);
            let fold = (folds > 0).then_some(FOLD_DEGREE);
            let d = stage_degree(Protocol::Two, stage.index, degrees.chain(fold))?;
            Ok(StageDegree {
                index: stage.index,
                rounds,
                folds,
                d,
                instances,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(relation: DegreeRelation, public: &[bool]) -> DegreeKey {
        DegreeKey {
            relation,
            stacked: false,
            public: public.to_vec(),
            constant: None,
        }
    }

    fn refused(key: &DegreeKey) -> bool {
        matches!(
            degree_row(key),
            Err(CompileError::Schedule { node: None, .. })
        )
    }

    /// RFC-0013 'Testing Strategy' 'The no-row refusal': a `MatMulAct` with one public operand
    /// and an `Xor3` whose three inputs are public match no row, `ERR-INTERNAL`.
    #[test]
    fn an_instance_that_matches_no_row_is_refused() {
        for public in [[true, false], [false, true]] {
            let matmul = key(DegreeRelation::MatMulAct, &public);
            assert!(refused(&matmul), "{matmul:?}");
            let stacked = DegreeKey {
                stacked: true,
                ..matmul
            };
            assert!(refused(&stacked), "{stacked:?}");
        }
        let xor3 = key(DegreeRelation::BitFn(BoolFn::Xor3), &[true; 3]);
        assert!(refused(&xor3), "{xor3:?}");
        assert_eq!(
            degree_row(&xor3).unwrap_err().code(),
            tessera_codec::error::ErrorCode::Internal
        );
    }

    /// Keys no derivation gives: the wrong number of operands, a constant on another relation or
    /// none on a `ConstMul`'s zero-check, a public output, a public operand of a native.
    #[test]
    fn keys_outside_the_table_are_refused() {
        use DegreeRelation as R;
        let maj = R::BitFn(BoolFn::Maj);
        let const_mul = R::Equality(EqualityOp::ConstMul);
        let cases = [
            key(R::Mul, &[true, true]),
            key(R::Mul, &[false]),
            key(maj, &[false, false]),
            key(R::BitFn(BoolFn::Ch), &[true, true, true]),
            key(R::BitFn(BoolFn::Xor), &[true, true]),
            key(R::SumSquares, &[true]),
            key(R::Rotate, &[true]),
            key(const_mul, &[false, false]),
            DegreeKey {
                constant: Some(Constant::Scalar),
                ..key(R::Mul, &[false, false])
            },
            DegreeKey {
                constant: Some(Constant::Scalar),
                ..key(const_mul, &[true, false])
            },
            key(R::Equality(EqualityOp::Add), &[false, false]),
            key(R::OneHotIndex, &[true, false]),
            key(R::Recomposition, &[false, true, false]),
            key(R::Native(NativeKind::Product), &[false]),
            key(R::Fold, &[false]),
        ];
        for case in cases {
            assert!(refused(&case), "{case:?}");
        }
    }

    /// RFC-0013 'Testing Strategy' 'The bound, at the stage level': a fabricated instance of
    /// degree 11 is refused in `tessera-protocol/2`, one of degree 8 in `tessera-protocol/1`;
    /// 10 and 7 pass.
    #[test]
    fn a_stage_above_its_protocols_bound_is_refused() {
        let internal = |r: Result<usize, CompileError>| {
            matches!(r, Err(CompileError::Schedule { node: None, .. }))
        };
        assert!(internal(stage_degree(Protocol::Two, 1, [7, 11, 4])));
        assert!(internal(stage_degree(Protocol::One, 1, [8])));
        assert!(internal(stage_degree(Protocol::One, 2, [7, 8])));
        assert!(internal(stage_degree(Protocol::Two, 1, [])));
        assert_eq!(stage_degree(Protocol::Two, 1, [7, 10, 2]).ok(), Some(10));
        assert_eq!(
            stage_degree(Protocol::Two, 3, [7, FOLD_DEGREE]).ok(),
            Some(7)
        );
        assert_eq!(stage_degree(Protocol::One, 1, [7, 3]).ok(), Some(7));
    }

    /// No row of the table exceeds 10 ('The invariant'), and every row keeps the rule.
    #[test]
    fn every_row_keeps_the_rule_and_the_bound() {
        use DegreeRelation as R;
        let mut relations = vec![
            R::MatMulWeight,
            R::MatMulAct,
            R::Mul,
            R::SumSquares,
            R::Rotate,
            R::Selection,
            R::ScaleChannels,
            R::Booleanity,
            R::OneHotVocabulary,
            R::OneHotCount,
            R::OneHotIndex,
            R::InputPadding,
            R::BitPadding,
            R::Recomposition,
            R::AssertEqual,
            R::Fold,
        ];
        relations.extend(
            [
                BoolFn::Xor,
                BoolFn::Xor3,
                BoolFn::And,
                BoolFn::Ch,
                BoolFn::Maj,
            ]
            .map(R::BitFn),
        );
        relations.extend(
            [
                EqualityOp::ConstMul,
                EqualityOp::ConstAdd,
                EqualityOp::Add,
                EqualityOp::Sub,
                EqualityOp::Mul,
            ]
            .map(R::Equality),
        );
        relations.extend(
            [
                NativeKind::Product,
                NativeKind::RootPad,
                NativeKind::ResidualPad,
                NativeKind::Key,
                NativeKind::Forest,
                NativeKind::Reciprocal,
                NativeKind::MaxPad,
                NativeKind::ReciprocalPad,
                NativeKind::Tokens,
                NativeKind::GreedyKey,
                NativeKind::GreedySelected,
                NativeKind::Observe,
            ]
            .map(R::Native),
        );
        let mut rows = 0;
        for relation in relations {
            for operands in 0..=3usize {
                for mask in 0..1u32 << operands {
                    let public: Vec<bool> = (0..operands).map(|i| mask >> i & 1 == 1).collect();
                    let constants = [
                        None,
                        Some(Constant::Scalar),
                        Some(Constant::Channel),
                        Some(Constant::Layer),
                    ];
                    for constant in constants {
                        for stacked in [false, true] {
                            let key = DegreeKey {
                                relation,
                                stacked,
                                public: public.clone(),
                                constant,
                            };
                            if let Ok(row) = degree_row(&key) {
                                assert_eq!(row.d, row.d_1 + 2 * row.s, "{key:?}");
                                assert!(row.reach <= row.d && row.d <= 10, "{key:?}");
                                rows += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(rows > 0);
        assert_eq!(
            R::Equality(EqualityOp::ConstMul).name(),
            "Equality.ConstMul"
        );
        assert_eq!(R::BitFn(BoolFn::Xor3).to_string(), "Xor3");
    }
}
