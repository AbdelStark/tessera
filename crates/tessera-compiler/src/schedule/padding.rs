// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Padded positions (RFC-0072 'Padded positions'): which lookups are non-zero at padding, and the
//! sums over a padded axis the schedule refuses ('Summing relations').
//!
//! A padded position holds what its operation computes from its inputs' padded positions, and
//! every operation maps zeros there to zeros but a lookup whose function does not. So a tensor is
//! 0 at its padded positions unless a lookup non-zero at padding lies upstream of it, or it is an
//! `Expand` or a selection, which are 0 past their length along their axis whatever their input.

use std::collections::HashMap;

use tessera_ir::{Graph, Node, NodeId, Op};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

use super::refuse;
use crate::CompileError;

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
/// kinds'). The kinds whose rows are closed forms or a few hundred values come from
/// `tessera_tables`' generator itself. The others' definitions give it, since generating them
/// builds up to `2^16` or `2^24` values: `0 ⊕ 0 = 0 ∧ 0 = 0` for `Bitwise8`; `round(2^σ·e^0) = 2^σ`
/// for `Exp`; `⌊2^{out_bits}·2^0⌋` for `Exp2Frac`; and `silu(0) = 0` for `Silu`, whatever its
/// class's shift and scales.
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

/// Whether a lookup node non-zero at padding lies upstream of each tensor of `graph`, its own
/// producer included, by tensor id. Each table is evaluated once.
fn upstream(graph: &Graph, tables: &[TableSpec]) -> Result<Vec<bool>, CompileError> {
    let mut reached = vec![false; graph.tensors.len()];
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
/// when no lookup non-zero at padding lies upstream of it, or when it is an `Expand` or a selection
/// along that axis. `MatMulWeight` sums against weights that are 0 there, and is never refused.
pub(super) fn check_sums(
    graph: &Graph,
    tables: &[TableSpec],
    producers: &[Option<(NodeId, u32)>],
) -> Result<(), CompileError> {
    let reached = upstream(graph, tables)?;
    let shape = |t: tessera_ir::TensorId| {
        graph
            .tensors
            .get(t.0 as usize)
            .map_or(&[][..], |spec| spec.shape.as_slice())
    };
    // Whether input `t` is 0 at the padded positions of its axis `axis`.
    let clean = |t: tessera_ir::TensorId, axis: usize| {
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
        let summed: Vec<(tessera_ir::TensorId, usize)> = match node.op {
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
