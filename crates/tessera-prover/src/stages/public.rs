// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A claim's public terms (RFC-0072 'Public tensors', 'Linear operations that are not coordinate
//! maps'; RFC-0037 'Claims and atoms', 'Splits'): values both parties compute from public data,
//! which move to the claim's value, and the public tables a relation reads its constants from.

use tessera_compiler::schedule::{PublicSource, public_tensors};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::{Graph, NodeId, Op, TensorId};
use tessera_poly::{evaluate_base, less_than};

use super::tables::{axis_vars, pad};
use super::{StageError, WitnessSource};

/// Which tensors of a graph are public, by tensor id (RFC-0072 'Public tensors'): the mask
/// [`public_value`] checks a tensor source against before it reads any value, so that no
/// secret-class tensor is copied into a public table (05 §3 OBS-6). Computed once per graph, in
/// one pass over its nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicTensors {
    mask: Vec<bool>,
}

impl PublicTensors {
    /// The public tensors of `graph`, as
    /// [`public_tensors`](tessera_compiler::schedule::public_tensors) closes them.
    pub fn of(graph: &Graph) -> Self {
        Self {
            mask: public_tensors(graph),
        }
    }

    /// Whether `tensor` is public.
    pub fn contains(&self, tensor: TensorId) -> bool {
        self.mask.get(tensor.0 as usize).copied().unwrap_or(false)
    }

    /// Whether the mask is `graph`'s, by its tensor count: a cheap guard against a mask of another
    /// graph.
    fn fits(&self, graph: &Graph) -> bool {
        self.mask.len() == graph.tensors.len()
    }
}

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
/// through the nodes that are not gadget-backed. Either way only public-class values are read
/// (05 §3): a tensor source that `public`, the graph's [`PublicTensors`], does not mark is refused
/// before any value is read.
///
/// # Errors
///
/// [`StageError::Internal`] when the tensor is not public in `graph` or `public` is not `graph`'s
/// mask, `values` lacks the tensor, the point does not have the tensor's variables, or a shift
/// names a node that is not a `ConstAdd` of the graph.
pub fn public_value<W: WitnessSource + ?Sized>(
    graph: &Graph,
    public: &PublicTensors,
    source: PublicSource,
    values: &W,
    point: &[GoldilocksExt2],
) -> Result<GoldilocksExt2, StageError> {
    match source {
        PublicSource::Tensor(tensor) => {
            let table = public_table(graph, public, values, tensor)?;
            evaluate_base::<GoldilocksExt2>(&table, point).map_err(|_| StageError::Internal {
                rule: "a public term's point has its tensor's coordinates",
            })
        }
        PublicSource::Shift(node) => shift(graph, node, point),
    }
}

/// The table of the public tensor `tensor`, its values those `values` holds, padded per dimension
/// with zeros: what [`public_value`] evaluates, and what a relation reads a public operand from.
/// Public-class data, which the verifier computes too, so it lives in a plain table; a tensor that
/// `public` does not mark is refused before its values are read.
pub(super) fn public_table<W: WitnessSource + ?Sized>(
    graph: &Graph,
    public: &PublicTensors,
    values: &W,
    tensor: TensorId,
) -> Result<Vec<Goldilocks>, StageError> {
    if !public.fits(graph) {
        return Err(StageError::Internal {
            rule: "a public mask is its graph's",
        });
    }
    if !public.contains(tensor) {
        return Err(StageError::Internal {
            rule: "a public table is of a public tensor",
        });
    }
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

/// The selector `ψ` of a tensor of `shape` over its padded shape, or `1 − ψ` when `outside`: 1 at
/// its real positions and 0 at its padded ones, the reverse for `1 − ψ` (RFC-0072 'Real
/// positions'). Its extension is `ψ̃(r) = Π_a LT(r_a, n_a)`: `AssertEqual`'s selector, and the
/// input padding check's `1 − ψ̃`.
pub(super) fn selector(shape: &[u32], outside: bool) -> Result<Vec<Goldilocks>, StageError> {
    let count: usize = shape.iter().map(|&len| len as usize).product();
    let mut table = public_zeros(1usize << vars_of(shape))?;
    pad(&vec![1; count], shape, &mut table)?;
    if outside {
        for slot in &mut table {
            *slot = Goldilocks::ONE - *slot;
        }
    }
    Ok(table)
}

/// `1 − LT(x_bit, bits)` over a tensor of `shape` whose last axis is a bit axis of `bits` bits:
/// 1 at its padded slots, 0 at its real ones, at every row (RFC-0066 'Padding'). The bit-padding
/// check's public factor.
pub(super) fn bit_slots(shape: &[u32], bits: u8) -> Result<Vec<Goldilocks>, StageError> {
    if shape.last() != Some(&u32::from(bits)) {
        return Err(StageError::Internal {
            rule: "a bit axis is its output's last",
        });
    }
    let width = 1usize << axis_vars(u32::from(bits));
    let mut table = public_zeros(1usize << vars_of(shape))?;
    for (index, slot) in table.iter_mut().enumerate() {
        if index % width >= usize::from(bits) {
            *slot = Goldilocks::ONE;
        }
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
