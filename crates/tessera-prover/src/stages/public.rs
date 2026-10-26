// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A claim's public terms (RFC-0072 'Public tensors', 'Linear operations that are not coordinate
//! maps'; RFC-0037 'Claims and atoms', 'Splits'): values both parties compute from public data,
//! which move to the claim's value, and the public tables a relation reads its constants from.

use tessera_compiler::schedule::PublicSource;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::{Graph, NodeId, Op, TensorId};
use tessera_poly::{evaluate_base, less_than};

use super::tables::{axis_vars, pad};
use super::{StageError, WitnessSource};

/// The value of `source` at `point` (RFC-0072 'Public tensors', 'Linear operations that are not
/// coordinate maps'), which both parties compute from public data, in `O(size)`:
///
/// - **A public tensor**'s extension `t̃(point)`, the tensor padded per dimension with zeros, its
///   values those `values` holds. Its padded positions are 0: no lookup lies upstream of a public
///   tensor, so none is non-zero at padding ('Padded positions').
/// - **A `ConstAdd`'s shift** `ĉ(point)`, the multilinear extension of `ψ·c` over the node's output
///   shape `(n_a)`, the constant at real positions only: `c·Π_a LT(r_a, n_a)` for a constant of
///   one value, a scalar whatever the last axis's length, and `c̃(r_c)·Π_{a≠last} LT(r_a, n_a)` for
///   one per channel of the last axis, with `c̃` the constant's extension, 0 at padded channels.
///   The last axis's selector does not appear there: `c̃(r_c)·ψ̃(r)` agrees with `ĉ` on the cube
///   only, and is not multilinear when the last axis is not a power of two. Each `LT` costs
///   `O(log n_a)` and `c̃(r_c)` `O(channels)`.
///
/// The point lists the lowest variable first: the last axis's coordinates first (RFC-0010
/// 'Multilinear extensions'). The prover passes its witness, which holds every tensor; a verifier
/// passes the public tensors it computes itself from the public graph inputs and constants
/// through the nodes that are not gadget-backed. Either way the values are public-class (05 §3).
///
/// # Errors
///
/// [`StageError::Internal`] when `values` lacks the tensor, the point does not have the tensor's
/// variables, or a shift names a node that is not a `ConstAdd` of the graph.
pub fn public_value<W: WitnessSource + ?Sized>(
    graph: &Graph,
    source: PublicSource,
    values: &W,
    point: &[GoldilocksExt2],
) -> Result<GoldilocksExt2, StageError> {
    match source {
        PublicSource::Tensor(tensor) => {
            let table = public_table(graph, values, tensor)?;
            evaluate_base::<GoldilocksExt2>(&table, point).map_err(|_| StageError::Internal {
                rule: "a public term's point has its tensor's coordinates",
            })
        }
        PublicSource::Shift(node) => shift(graph, node, point),
    }
}

/// The table of the public tensor `tensor`, its values those `values` holds, padded per dimension
/// with zeros: what [`public_value`] evaluates, and what a relation reads a public operand from.
/// Public-class data, which the verifier computes too, so it lives in a plain table.
pub(super) fn public_table<W: WitnessSource + ?Sized>(
    graph: &Graph,
    values: &W,
    tensor: TensorId,
) -> Result<Vec<Goldilocks>, StageError> {
    let shape = &graph
        .tensors
        .get(tensor.0 as usize)
        .ok_or(StageError::Internal {
            rule: "a public tensor is the graph's",
        })?
        .shape;
    let values = values.values(tensor).ok_or(StageError::Internal {
        rule: "the public values hold every public tensor a claim or an operand reaches",
    })?;
    let mut table = public_zeros(1usize << vars_of(shape))?;
    pad(values, shape, &mut table)?;
    Ok(table)
}

/// `ĉ(point)` of the `ConstAdd` `node` ([`public_value`]).
fn shift(
    graph: &Graph,
    node: NodeId,
    point: &[GoldilocksExt2],
) -> Result<GoldilocksExt2, StageError> {
    let (constant, shape) = const_add(graph, node)?;
    let per_channel = constant.len() > 1;
    let mut value = GoldilocksExt2::ONE;
    let mut at = 0;
    // The blocks of the point, the last axis's first.
    for (axis, &len) in shape.iter().enumerate().rev() {
        let vars = axis_vars(len) as usize;
        let r = point.get(at..at + vars).ok_or(StageError::Internal {
            rule: "a shift's point has its ConstAdd's coordinates",
        })?;
        at += vars;
        value *= if per_channel && axis + 1 == shape.len() {
            let table = channel_table(constant, vars)?;
            evaluate_base::<GoldilocksExt2>(&table, r).map_err(|_| StageError::Internal {
                rule: "a shift's point has its ConstAdd's coordinates",
            })?
        } else {
            less_than(r, u64::from(len)).map_err(|_| StageError::Internal {
                rule: "an axis's length fits its variables",
            })?
        };
    }
    if at != point.len() {
        return Err(StageError::Internal {
            rule: "a shift's point has its ConstAdd's coordinates",
        });
    }
    if !per_channel {
        value *= GoldilocksExt2::from(Goldilocks::from_i64(constant[0]));
    }
    Ok(value)
}

/// The constant of the `ConstAdd` `node` and its output's shape.
pub(super) fn const_add(graph: &Graph, node: NodeId) -> Result<(&[i64], &[u32]), StageError> {
    let node = graph
        .nodes
        .get(node.0 as usize)
        .ok_or(StageError::Internal {
            rule: "a shift's node is the graph's",
        })?;
    let Op::ConstAdd { c } = node.op else {
        return Err(StageError::Internal {
            rule: "a shift is a ConstAdd's",
        });
    };
    constant_and_shape(graph, node, c)
}

/// A `ConstMul`'s or `ConstAdd`'s constant `c` and `node`'s output shape: a scalar, or one value
/// per channel of the last axis.
pub(super) fn constant_and_shape<'g>(
    graph: &'g Graph,
    node: &tessera_ir::Node,
    c: tessera_ir::ConstRef,
) -> Result<(&'g [i64], &'g [u32]), StageError> {
    let constant =
        graph
            .constants
            .get(c.0 as usize)
            .map(Vec::as_slice)
            .ok_or(StageError::Internal {
                rule: "a constant the graph holds",
            })?;
    let shape = node
        .outputs
        .first()
        .and_then(|t| graph.tensors.get(t.0 as usize))
        .map(|spec| spec.shape.as_slice())
        .ok_or(StageError::Internal {
            rule: "a ConstAdd or ConstMul has its output",
        })?;
    let channels = shape.last().map_or(1, |&len| len as usize);
    if constant.is_empty() || (constant.len() > 1 && constant.len() != channels) {
        return Err(StageError::Internal {
            rule: "a constant is a scalar or one value per channel",
        });
    }
    Ok((constant, shape))
}

/// A constant per channel over its `vars` channel variables, 0 at padded channels.
pub(super) fn channel_table(constant: &[i64], vars: usize) -> Result<Vec<Goldilocks>, StageError> {
    let mut table = public_zeros(1usize << vars)?;
    if constant.len() > table.len() {
        return Err(StageError::Internal {
            rule: "a constant per channel fits its axis",
        });
    }
    for (slot, &value) in table.iter_mut().zip(constant) {
        *slot = Goldilocks::from_i64(value);
    }
    Ok(table)
}

/// A public table of `len` zeros.
pub(super) fn public_zeros(len: usize) -> Result<Vec<Goldilocks>, StageError> {
    let mut table = Vec::new();
    table
        .try_reserve_exact(len)
        .map_err(|_| StageError::Memory)?;
    table.resize(len, Goldilocks::ZERO);
    Ok(table)
}

/// The variables of a tensor of `shape`, padded per dimension.
fn vars_of(shape: &[u32]) -> u32 {
    shape.iter().map(|&len| axis_vars(len)).sum()
}
