// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Padded positions (RFC-0072 'Padded positions'): which lookups are non-zero at padding, the
//! sums over a padded axis the schedule refuses ('Summing relations'), and the gadgets that may
//! not read a tensor non-zero at its padded positions.
//!
//! A padded position holds what its operation computes from its inputs' padded positions, and
//! every operation maps zeros there to zeros but a lookup whose function does not. So a tensor is
//! 0 at its padded positions unless a lookup non-zero at padding lies upstream of it, or it is an
//! `Expand` or a selection, which are 0 past their length along their axis whatever their input.
//!
//! A committed advice graph input's padded positions are computed by nothing: they are the
//! prover's. RFC-0006 'Committed graph inputs' admits a graph only when an operation pins them,
//! a `OneHot` reading the input directly or a `BitDecompose` reading it directly or moved, and
//! [`crate::analyze`] refuses any other. The schedule does not rely on that analysis having run:
//! an input that neither pins counts here as a lookup non-zero at padding does. On a graph that
//! range analysis admits, this changes no verdict.

use std::collections::HashMap;

use tessera_ir::{CommitmentClass, Graph, Node, NodeId, NodeKind, Op, TensorId};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

use super::refuse;
use crate::CompileError;
use crate::range::committed_source;

/// Whether `node` is a lookup node non-zero at padding (RFC-0072 'Padded positions'): a
/// `Requant`, `TableMap` or `Bitwise` whose function maps zero inputs, in any of its classes, to a
/// value other than 0. It is decided from public data:
///
/// - **`Requant`**: RFC-0005's `rq(0, r_c, lo, hi) = clamp(⌊2^{r_c − 1} / 2^{r_c}⌋, lo, hi)` for
///   each class's shift `r_c`, which is 0 exactly when `lo ≤ 0 ≤ hi`, as at every Llama site;
/// - **`TableMap` and `Bitwise`**: the table's value at the key that packs input 0 with each class
///   (RFC-0008 'Table kinds', `pack(c, 0)`; `0 + 2^8·0` for `Bitwise8`). `AesSbox(0) = 0x63` is
///   non-zero; `Gf256Mul`, `Bitwise8` and SiLU map 0 to 0.
///
/// Every other node is none: the gadgets' outputs that are not lookups are 0 on padded rows
/// (RFC-0035), and the exponentials of `SoftmaxRow` look up the masked key, which maps to 0.
///
/// # Errors
///
/// [`CompileError::Schedule`] (`ERR-INTERNAL`) when a lookup's table is not among `tables`, is
/// not a lookup table, or its kind's parameters do not generate it.
pub fn nonzero_at_padding(
    graph: &Graph,
    tables: &[TableSpec],
    node: &Node,
) -> Result<bool, CompileError> {
    match &node.op {
        Op::Requant { shift, lo, hi } => {
            let Some(shifts) = graph.shift_classes.get(shift.0 as usize) else {
                return refuse(
                    Some(node.id),
                    "a requantization's shift classes are the graph's",
                );
            };
            Ok(shifts.iter().any(|&r| rq_of_zero(r, *lo, *hi) != 0))
        }
        Op::TableMap { table } | Op::Bitwise { table } => table_nonzero(tables, node.id, *table),
        _ => Ok(false),
    }
}

/// RFC-0005's `rq(0, r, lo, hi) = clamp(⌊(0 + 2^{r−1}) / 2^r⌋, lo, hi)`, `1 ≤ r ≤ 63`.
fn rq_of_zero(r: u8, lo: i64, hi: i64) -> i128 {
    let r = u32::from(r.clamp(1, 63));
    ((1i128 << (r - 1)) >> r).clamp(i128::from(lo), i128::from(hi))
}

/// Whether table `table` maps the zero key to a non-zero value in some class.
fn table_nonzero(tables: &[TableSpec], node: NodeId, table: TableId) -> Result<bool, CompileError> {
    let Some(spec) = tables.iter().find(|spec| spec.id == table) else {
        return refuse(Some(node), "a lookup's table is the graph's");
    };
    let classes = match &spec.kind {
        TableKind::Requant { r_by_class, .. } | TableKind::Silu { r_by_class, .. } => {
            u8::try_from(r_by_class.len()).unwrap_or(u8::MAX)
        }
        _ => 1,
    };
    for class in 0..classes {
        if zero_value(spec, class, node)? != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The value table `spec` gives the key that packs input 0 with class `class` (RFC-0008 'Table
/// kinds'). The kinds whose rows are closed forms, or whose digests hash a few hundred values,
/// come from `tessera_tables`' generator itself. The others' definitions give it: building them
/// generates `Exp`'s, `Exp2Frac`'s and `Silu`'s dense cores, up to `2^24` values, and hashes
/// `Bitwise8`'s `2^16` rows into its digest. So `0 ⊕ 0 = 0 ∧ 0 = 0` for `Bitwise8`;
/// `round(2^σ·e^0) = 2^σ` for `Exp`; `⌊2^{out_bits}·2^0⌋` for `Exp2Frac`; and `silu(0) = 0` for
/// `Silu`, whatever its class's shift and scales.
fn zero_value(spec: &TableSpec, class: u8, node: NodeId) -> Result<i64, CompileError> {
    match &spec.kind {
        TableKind::Range { .. }
        | TableKind::Requant { .. }
        | TableKind::Pow2 { .. }
        | TableKind::AesSbox
        | TableKind::Gf256Mul { .. } => GeneratedTable::new(spec.id, spec.kind.clone())
            .ok()
            .and_then(|table| table.lookup(0, class))
            .ok_or(CompileError::Schedule {
                node: Some(node),
                rule: "a lookup's table generates its value at the zero key",
            }),
        TableKind::Bitwise8 { .. } | TableKind::Silu { .. } => Ok(0),
        TableKind::Exp { sigma, .. } => Ok(1i64 << (*sigma).min(62)),
        TableKind::Exp2Frac { out_bits, .. } => Ok(1i64 << (*out_bits).min(62)),
        TableKind::Rope { .. } => refuse(Some(node), "a lookup's table is a lookup table"),
    }
}

/// Whether each committed advice graph input of `graph` is pinned at its padded positions, by
/// tensor id (RFC-0006 'Committed graph inputs', RFC-0072 'Padded positions'): read directly by a
/// `OneHot`, whose row selector pins its own input's padded rows, or by a `BitDecompose`, directly
/// or moved by `Reshape`, `Slice`, `Transpose`, `GatherPublic` or `Expand`, whose input padding
/// check pins the whole input's. Range analysis takes the same walk.
fn pinned(graph: &Graph, producers: &[Option<(NodeId, u32)>]) -> Vec<bool> {
    let producer = |t: TensorId| {
        producers
            .get(t.0 as usize)
            .copied()
            .flatten()
            .and_then(|(node, _)| graph.nodes.get(node.0 as usize))
    };
    let mut pinned = vec![false; graph.tensors.len()];
    for node in &graph.nodes {
        let Some(&input) = node.inputs.first() else {
            continue;
        };
        let source = match node.op {
            Op::BitDecompose { .. } => committed_source(graph, producer, input),
            Op::OneHot { .. } => producer(input).is_none().then_some(input),
            _ => None,
        };
        if let Some(slot) = source.and_then(|t| pinned.get_mut(t.0 as usize)) {
            *slot = true;
        }
    }
    pinned
}

/// Whether a lookup node non-zero at padding ([`nonzero_at_padding`]) lies upstream of each tensor
/// of `graph`, its own producer included, by tensor id (RFC-0072 'Padded positions'): the
/// tensors whose padded positions may hold a value other than 0. Every other tensor is 0 at its
/// padded positions, so its padded table is its values padded per dimension with zeros.
///
/// A tensor marked here holds, at each padded position, what its operation computes from its
/// inputs' padded positions: the prover's witness and stage tables compute it there
/// (`tessera_prover::stages::PaddedWitness`). A selection's or an `Expand`'s padded rows along its
/// axis are 0 whatever its input, but its other padded positions are its input's, so it is
/// marked with its input. No public tensor is marked: a lookup is a gadget, which no public
/// tensor has upstream ([`public_tensors`](super::public_tensors)). Committed graph inputs are
/// not marked: the honest prover pads them with zeros, which RFC-0006's checks pin.
///
/// # Errors
///
/// As [`nonzero_at_padding`].
pub fn tensors_nonzero_at_padding(
    graph: &Graph,
    tables: &[TableSpec],
) -> Result<Vec<bool>, CompileError> {
    reach(graph, tables, &super::producers(graph), false)
}

/// Whether a lookup node non-zero at padding, or a committed advice graph input that nothing pins
/// ([`pinned`]), lies upstream of each tensor of `graph`, its own producer or itself included, by
/// tensor id. Each table is evaluated once.
fn upstream(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
) -> Result<Vec<bool>, CompileError> {
    reach(graph, tables, producers, true)
}

/// Whether a lookup node non-zero at padding lies upstream of each tensor of `graph`, its own
/// producer included, by tensor id, and, when `unpinned`, a committed advice graph input that
/// nothing pins ([`pinned`]) too. Each table is evaluated once.
fn reach(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
    unpinned: bool,
) -> Result<Vec<bool>, CompileError> {
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    let pinned = pinned(graph, producers);
    let mut reached: Vec<bool> = graph
        .tensors
        .iter()
        .zip(producers)
        .zip(&pinned)
        .map(|((spec, producer), &pinned)| {
            unpinned && producer.is_none() && spec.kind == advice && !pinned
        })
        .collect();
    let mut by_table: HashMap<TableId, bool> = HashMap::new();
    for node in &graph.nodes {
        let own = match node.op {
            Op::TableMap { table } | Op::Bitwise { table } => match by_table.get(&table) {
                Some(&nonzero) => nonzero,
                None => {
                    let nonzero = table_nonzero(tables, node.id, table)?;
                    by_table.insert(table, nonzero);
                    nonzero
                }
            },
            _ => nonzero_at_padding(graph, tables, node)?,
        };
        let reach = own
            || node
                .inputs
                .iter()
                .any(|t| reached.get(t.0 as usize).copied().unwrap_or(false));
        for &t in &node.outputs {
            if let Some(slot) = reached.get_mut(t.0 as usize) {
                *slot = reach;
            }
        }
    }
    Ok(reached)
}

/// Refuses each summing relation over an axis whose length is not a power of two none of whose
/// summed inputs is 0 at the axis's padded positions (RFC-0072 'Summing relations'). `Sum` and
/// `SumSquares` sum their input over their axis; `MatMulAct` sums both inputs over its inner axis,
/// and one clean input suffices, since a product is 0 where either factor is. An input is clean
/// when neither a lookup non-zero at padding nor a committed advice graph input that nothing pins
/// lies upstream of it ([`upstream`]), or when it is an `Expand` or a selection along that axis.
/// `MatMulWeight` sums against weights that are 0 there, and is never refused; a weight, and a
/// channel-class vector or multipliers committed with the weights, takes the domain its
/// publisher vouches for (RFC-0048).
pub(super) fn check_sums(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
) -> Result<(), CompileError> {
    let reached = upstream(graph, tables, producers)?;
    let shape = |t: TensorId| {
        graph
            .tensors
            .get(t.0 as usize)
            .map_or(&[][..], |spec| spec.shape.as_slice())
    };
    // Whether input `t` is 0 at the padded positions of its axis `axis`.
    let clean = |t: TensorId, axis: usize| {
        if !reached.get(t.0 as usize).copied().unwrap_or(false) {
            return true;
        }
        let producer = producers
            .get(t.0 as usize)
            .copied()
            .flatten()
            .and_then(|(node, _)| graph.nodes.get(node.0 as usize));
        match producer.map(|node| &node.op) {
            Some(Op::Expand { axis: a, .. } | Op::Slice { axis: a, .. }) => usize::from(*a) == axis,
            Some(Op::GatherPublic { .. }) => axis == 0,
            _ => false,
        }
    };
    for node in &graph.nodes {
        // The inputs the relation sums, each with the axis it sums.
        let summed: Vec<(TensorId, usize)> = match node.op {
            Op::Sum { axis } | Op::SumSquares { axis } => match node.inputs.first() {
                Some(&x) => vec![(x, usize::from(axis))],
                None => continue,
            },
            // `A`'s inner axis is its last; `B`'s its second-to-last, or its last when it enters
            // transposed, as `K` does in `QKᵀ`.
            Op::MatMulAct { transpose_rhs } => match node.inputs[..] {
                [a, b] => {
                    let right = if transpose_rhs { 1 } else { 2 };
                    match (
                        shape(a).len().checked_sub(1),
                        shape(b).len().checked_sub(right),
                    ) {
                        (Some(left), Some(right)) => vec![(a, left), (b, right)],
                        _ => continue,
                    }
                }
                _ => continue,
            },
            _ => continue,
        };
        let Some(&(x, axis)) = summed.first() else {
            continue;
        };
        let padded = shape(x).get(axis).is_some_and(|len| !len.is_power_of_two());
        if padded && !summed.iter().any(|&(t, axis)| clean(t, axis)) {
            return refuse(
                Some(node.id),
                "a Sum, SumSquares or MatMulAct over an axis whose length is not a power of two \
                 sums an input that is 0 at its padded positions",
            );
        }
    }
    Ok(())
}

/// The rule of [`check_gadgets`], which the prover's padded witness states alike
/// (`tessera_prover::stages::PaddedWitness`).
pub(super) const GADGET_INPUT: &str = "no gadget but a lookup, BitDecompose or AssertEqual reads a \
                                       tensor non-zero at its padded positions: RFC-0035's row \
                                       relations take a padded row's input as 0";

/// Whether a gadget's padded positions are specified for an input non-zero there (RFC-0072
/// 'Padded positions'): a lookup's are the lookup of what its key packs there, `BitDecompose`'s
/// the bits of its input there, and `AssertEqual` weighs its sides by `ψ`, 0 there. Every other
/// gadget, `OneHot`, `SoftmaxRow`, `RsqrtRow`, `Argmax`, `InverseCdf` and any gadget added later,
/// is not: RFC-0035's row relations take a padded row's input as 0.
pub fn reads_padding(op: &Op) -> bool {
    matches!(
        op,
        Op::Requant { .. }
            | Op::TableMap { .. }
            | Op::Bitwise { .. }
            | Op::BitDecompose { .. }
            | Op::AssertEqual
    )
}

/// Refuses a gadget whose padded positions are not specified for an input non-zero there
/// ([`reads_padding`]) that reads a tensor with padded positions which a lookup non-zero at
/// padding reaches ([`tensors_nonzero_at_padding`], RFC-0072 'Padded positions'). Decided from
/// the graph and the tables alone, before any claim.
pub(super) fn check_gadgets(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
) -> Result<(), CompileError> {
    let reached = reach(graph, tables, producers, false)?;
    let padded = |t: &TensorId| {
        reached.get(t.0 as usize).copied().unwrap_or(false)
            && graph
                .tensors
                .get(t.0 as usize)
                .is_some_and(|spec| spec.shape.iter().any(|len| !len.is_power_of_two()))
    };
    for node in &graph.nodes {
        let gadget = node.op.class() == tessera_ir::OpClass::Gadget;
        if gadget && !reads_padding(&node.op) && node.inputs.iter().any(padded) {
            return refuse(Some(node.id), GADGET_INPUT);
        }
    }
    Ok(())
}
