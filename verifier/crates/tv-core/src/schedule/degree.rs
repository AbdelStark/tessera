// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0013 'Masked extensions / Degree', independently keyed from the verifier's schedule.
//! A row is a protocol degree, not an optimization based on a particular mask's columns.
//! Graph keying covers the current independent Llama IR; the table also covers the anchor
//! relations whose graph derivation is separate work in #150 and #159.

use super::{Instance, Layers, Relation, Schedule, ScheduleError};
use crate::lowering::{Graph, Kind, TensorId};
use crate::sumcheck::{Prefix, Protocol};

/// The semantic relation of a native callback, fixed by the verifier's own model plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeKind {
    /// RFC-0050 coefficient products.
    Product,
    /// Root output padding.
    RootPad,
    /// Root residual padding.
    ResidualPad,
    /// Softmax lookup keys.
    Key,
    /// Softmax fraction forest.
    Forest,
    /// Softmax reciprocal.
    Reciprocal,
    /// Softmax maximum padding.
    MaxPad,
    /// Softmax reciprocal padding.
    ReciprocalPad,
    /// Token to indicator binding.
    Tokens,
    /// Greedy key binding.
    GreedyKey,
    /// Greedy selected logits.
    GreedySelected,
    /// Observed logits (zero summand).
    Observe,
}
impl NativeKind {
    /// The normative table's relation string.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Product => "Product",
            Self::RootPad => "RootPad",
            Self::ResidualPad => "ResidualPad",
            Self::Key => "Key",
            Self::Forest => "Forest",
            Self::Reciprocal => "Reciprocal",
            Self::MaxPad => "MaxPad",
            Self::ReciprocalPad => "ReciprocalPad",
            Self::Tokens => "Tokens",
            Self::GreedyKey => "GreedyKey",
            Self::GreedySelected => "GreedySelected",
            Self::Observe => "Observe",
        }
    }
}

/// Constant classification of an equality zero-check (RFC-0013 'Per layer').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constant {
    /// One constant outside the template, or a template with no layer variable.
    Scalar,
    /// One value per channel outside the template.
    Channel,
    /// A template position with at least one layer variable, even if all values coincide.
    Layer,
}

/// The public metadata selecting one row of RFC-0013's table.
#[derive(Clone, Copy, Debug)]
pub struct Key<'a> {
    /// Normative relation name; unknown names are refused.
    pub relation: &'a str,
    /// At least one stacking variable, including a matmul's leading head axis.
    pub stacked: bool,
    /// Whether each operand (in final-evaluation order) is public.
    pub public: &'a [bool],
    /// Constant class, present only for `Equality.ConstMul`.
    pub constant: Option<Constant>,
}

/// The table's factor count and protocol degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// `d_1`, the factors of the term setting the degree.
    pub base: usize,
    /// `s`, that term's masked reads, counted with multiplicity.
    pub masked: usize,
    /// `d_1 + 2s`, the round message length.
    pub degree: usize,
    /// The exact maximum degree attained by the prescribed tiny instance.
    pub reach: usize,
}
impl Row {
    fn new(base: usize, masked: usize) -> Self {
        // Only table constants reach this helper: base <= 4 and masked <= 3.
        let degree = base.saturating_add(masked.saturating_mul(2));
        Self {
            base,
            masked,
            degree,
            reach: degree,
        }
    }
    fn loose(base: usize, masked: usize, reach: usize) -> Self {
        Self {
            reach,
            ..Self::new(base, masked)
        }
    }
}

/// Select a normative degree row. Unknown relations, operand counts, public patterns or constant
/// classes are `ScheduleError::Internal` (`ERR-INTERNAL`), never a fallback degree.
/// This includes the table's rows not yet derived by the graph IR (#150 / #159).
pub fn degree_row(key: Key<'_>) -> Result<Row, ScheduleError> {
    let p = key.public;
    let n = p.len();
    let count = p.iter().filter(|&&v| v).count();
    if (key.relation == "Equality.ConstMul") != key.constant.is_some() {
        return Err(ScheduleError::Internal);
    }
    let row = match (key.relation, n, count) {
        ("MatMulWeight" | "MatMulAct", 2, 0) => Row::new(if key.stacked { 3 } else { 2 }, 2),
        ("Mul", 2, 0) | ("SumSquares", 1, 0) => Row::new(3, 2),
        ("Mul", 2, 1) => Row::new(3, 1),
        ("Rotate" | "Selection", 1, 0) => Row::new(2, 1),
        ("ScaleChannels", 1, 0) => Row::new(3, 1),
        ("Xor" | "And", 2, 0) | ("Ch", 3, 0) => Row::new(3, 2),
        ("Xor" | "And", 2, 1) | ("Ch", 3, 2) => Row::new(3, 1),
        ("Ch", 3, 1) => Row::new(3, if p.first() == Some(&true) { 1 } else { 2 }),
        ("Xor3" | "Maj", 3, 0..=2) => Row::new(4, 3usize.saturating_sub(count)),
        ("Equality.ConstMul", 2, _) if p.first() == Some(&false) => match (key.constant, count) {
            (Some(Constant::Scalar), _) | (_, 1) => Row::new(2, 1),
            (Some(Constant::Channel | Constant::Layer), 0) => Row::new(3, 1),
            _ => return Err(ScheduleError::Internal),
        },
        ("Equality.ConstAdd", 2, _) | ("Equality.Add" | "Equality.Sub", 3, _)
            if p.first() == Some(&false) =>
        {
            Row::new(2, 1)
        }
        ("Equality.Mul", 3, _) if p.first() == Some(&false) => match count {
            0 => Row::new(3, 2),
            1 => Row::new(3, 1),
            _ => Row::new(2, 1),
        },
        ("Booleanity", 1, 0) => Row::new(3, 2),
        ("OneHotVocabulary" | "InputPadding" | "BitPadding", 1, 0) => Row::new(3, 1),
        ("OneHotCount", 1, 0) => Row::new(2, 1),
        ("OneHotIndex", 2, _) if p.first() == Some(&false) => Row::loose(3, 1, 4),
        ("Recomposition", 2..=3, _)
            if !p.get(1..).ok_or(ScheduleError::Internal)?.contains(&true) =>
        {
            Row::loose(3, 1, 4)
        }
        ("AssertEqual", 2, _) => Row::new(3, usize::from(count < 2)),
        ("Product", 0, 0) => Row::new(4, 3),
        ("RootPad" | "ResidualPad" | "Key" | "Forest" | "MaxPad" | "ReciprocalPad", 0, 0) => {
            Row::new(3, 1)
        }
        ("Reciprocal", 0, 0) => Row::new(3, 2),
        ("GreedySelected", 0, 0) => Row::loose(3, 2, 5),
        ("Tokens" | "GreedyKey", 0, 0) => Row::loose(3, 1, 4),
        ("Observe", 0, 0) => Row::loose(1, 1, 0),
        ("Fold", 0, 0) => Row::new(2, 0),
        _ => return Err(ScheduleError::Internal),
    };
    Ok(row)
}

/// Maximum of a stage's independently derived degrees, checked against its protocol's bound.
/// Empty stages and out-of-bound degrees are `ERR-INTERNAL` (RFC-0013 'The invariant').
pub fn stage_degree(
    protocol: Protocol,
    stage: u16,
    degrees: &[usize],
) -> Result<usize, ScheduleError> {
    let degree = degrees
        .iter()
        .copied()
        .max()
        .ok_or(ScheduleError::Internal)?;
    if degree == 0 || degree > protocol.max_degree(Prefix::Stage(u32::from(stage))) {
        return Err(ScheduleError::Internal);
    }
    Ok(degree)
}

/// One schedule instance's key and normative degree, derived from verifier-owned public data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceDegree {
    /// Normative relation string.
    pub relation: &'static str,
    /// At least one layer or matmul head stacking variable.
    pub stacked: bool,
    /// Public operand pattern, in final-evaluation order.
    pub public: Vec<bool>,
    /// Equality constant class, if any.
    pub constant: Option<Constant>,
    /// Normative degree row.
    pub row: Row,
}

fn public(graph: &Graph, tensor: TensorId) -> Result<bool, ScheduleError> {
    Ok(graph.tensor(tensor).ok_or(ScheduleError::Internal)?.kind == Kind::Public)
}

/// Derive a protocol/2 row from the independent graph and schedule. No proof supplies the key;
/// native kinds come from the independent model plan. Unsupported metadata is `ERR-INTERNAL`.
pub fn protocol2_instance(
    graph: &Graph,
    schedule: &Schedule,
    instance: &Instance,
) -> Result<InstanceDegree, ScheduleError> {
    let mut stacked = matches!(&instance.layers, Layers::Point(p) if !p.is_empty());
    let mut operands = Vec::new();
    operands
        .try_reserve_exact(3)
        .map_err(|_| ScheduleError::Memory)?;
    let mut constant = None;
    let relation = if let Relation::Native { kind, .. } = instance.relation {
        kind.name()
    } else {
        let node = graph.node(instance.node).ok_or(ScheduleError::Internal)?;
        let inputs = node.inputs();
        let output = public(
            graph,
            *node.outputs().first().ok_or(ScheduleError::Internal)?,
        )?;
        let mut input = |i| -> Result<(), ScheduleError> {
            operands.push(public(
                graph,
                *inputs.get(i).ok_or(ScheduleError::Internal)?,
            )?);
            Ok(())
        };
        match &instance.relation {
            Relation::MatMulWeight { .. } => {
                input(1)?;
                input(0)?;
                "MatMulWeight"
            }
            Relation::MatMulAct { lead_vars, .. } => {
                stacked |= *lead_vars > 0;
                input(0)?;
                input(1)?;
                "MatMulAct"
            }
            Relation::Elementwise { .. } => {
                input(0)?;
                input(1)?;
                "Mul"
            }
            Relation::SumSquares { .. } => {
                input(0)?;
                "SumSquares"
            }
            Relation::Rotate { .. } => {
                input(0)?;
                "Rotate"
            }
            Relation::Booleanity => {
                operands.push(output);
                "Booleanity"
            }
            Relation::Vocabulary { .. } => {
                operands.push(output);
                "OneHotVocabulary"
            }
            Relation::OneHotCount { .. } => {
                operands.push(output);
                "OneHotCount"
            }
            Relation::OneHotIndex { .. } => {
                let token = public(graph, *inputs.first().ok_or(ScheduleError::Internal)?)?;
                operands.extend([output, token]);
                "OneHotIndex"
            }
            Relation::Equality { .. } => {
                let x = public(graph, *inputs.first().ok_or(ScheduleError::Internal)?)?;
                operands.extend([output, x]);
                constant = Some(if instance.stacked && schedule.layer_vars() > 0 {
                    Constant::Layer
                } else {
                    Constant::Scalar
                });
                "Equality.ConstMul"
            }
            Relation::Native { .. } => return Err(ScheduleError::Internal),
        }
    };
    let row = degree_row(Key {
        relation,
        stacked,
        public: &operands,
        constant,
    })?;
    Ok(InstanceDegree {
        relation,
        stacked,
        public: operands,
        constant,
        row,
    })
}

/// One reduction stage's protocol/2 degree and folds (RFC-0013 'Testing Strategy / The vector').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageDegree {
    /// Stage index, starting at one.
    pub index: u16,
    /// Round count, including the mask row width when a fold is present.
    pub rounds: usize,
    /// Virtual-part fold instances; native relations and zero-checks have none.
    pub folds: usize,
    /// Maximum normative degree, including degree-two folds.
    pub degree: usize,
    /// Per-instance keys and rows, in independent schedule order.
    pub instances: Vec<InstanceDegree>,
}

/// Derive every protocol/2 reduction-stage degree without trusting proof metadata. Allocations
/// follow the already bounded schedule; memory failure is `Memory`, bad keys `Internal`.
pub fn protocol2_stages(
    graph: &Graph,
    schedule: &Schedule,
    log2_k: usize,
) -> Result<Vec<StageDegree>, ScheduleError> {
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(schedule.stages().len())
        .map_err(|_| ScheduleError::Memory)?;
    for stage in schedule.stages() {
        let mut instances = Vec::new();
        instances
            .try_reserve_exact(stage.instances.len())
            .map_err(|_| ScheduleError::Memory)?;
        let mut folds = 0usize;
        let mut degree = 0;
        for instance in &stage.instances {
            let own = protocol2_instance(graph, schedule, instance)?;
            degree = degree.max(own.row.degree);
            if !matches!(instance.relation, Relation::Native { .. })
                && instance.relation.kind() != super::RelationKind::ZeroCheck
            {
                folds = folds.checked_add(1).ok_or(ScheduleError::Internal)?;
            }
            instances.push(own);
        }
        let degree = stage_degree(
            Protocol::Two,
            stage.index,
            &[degree, if folds > 0 { 2 } else { 0 }],
        )?;
        stages.push(StageDegree {
            index: stage.index,
            rounds: if folds > 0 {
                stage.rounds.max(log2_k)
            } else {
                stage.rounds
            },
            folds,
            degree,
            instances,
        });
    }
    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::{self, Role};
    use crate::{claims::ClaimList, validation};
    use tv_codec::DecodeLimits;
    use tv_testkit::{Json, hex, read_vectors};

    #[test]
    fn public_inputs_are_classified_from_the_independent_graph() {
        let json = read_vectors("claims/claim-lists.json");
        let first = json
            .get("cases")
            .and_then(Json::as_array)
            .unwrap()
            .first()
            .unwrap();
        let bytes = hex(first.get("hex").and_then(Json::as_str).unwrap());
        let claims = ClaimList::open(&bytes, &DecodeLimits::MESSAGE).unwrap();
        let graph = lowering::lower(&validation::check_public(&claims).unwrap(), 4).unwrap();
        let mut public_count = 0;
        for input in graph.inputs() {
            let expected = input.role == Role::Positions;
            assert_eq!(public(&graph, input.tensor), Ok(expected));
            public_count += usize::from(expected);
        }
        assert_eq!(public_count, 1);
        assert_eq!(
            public(&graph, TensorId(u32::MAX)),
            Err(ScheduleError::Internal)
        );
    }
}
