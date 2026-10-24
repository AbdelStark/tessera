// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Each tensor's token axis (RFC-0054 'Groups', 'The token axis'), from the graph alone.
//!
//! # Specification
//!
//! [`TokenAxes::derive`] gives every tensor of a lowered graph at most one token axis, input to
//! output, from the operations and shapes alone: lengths never decide it, since a head or channel
//! axis can be as long as the sequence. With `x` a node's first input and `a` its token axis:
//!
//! 1. **Inputs.** `Tokens` and `Positions` have it at axis 0; every other input (weights,
//!    classes, multipliers) has none.
//! 2. **Kept.** `ConstMul`, `Rotate`, `Requant`, `TableMap`, `RsqrtRow` and `OneHot` keep `a`,
//!    which for `OneHot` is its tokens' axis 0 (the vocabulary axis is appended).
//!    - `Add` keeps `a` and needs its second input's to be `a`.
//!    - `Mul { Trailing }` keeps `a` and needs its operand's to be `a`; `Mul { Leading }` needs
//!      its operand to have none.
//!    - `MatMulWeight` keeps `a` and needs it not to be `x`'s last axis, which the weight
//!      contracts.
//! 3. **Moved.** `Transpose { perm }` puts it at the output axis `i` with `perm[i] = a`.
//!    `Expand { axis }` moves it to `a + 1` when `a ≥ axis`. `SumSquares { axis }` moves it to
//!    `a − 1` when `a > axis`, and may not remove it.
//! 4. **Regrouped.** `Reshape` keeps the token axis whole: its output axis is the first whose
//!    leading axes hold as many elements as `x`'s before `a`, and which is as long as `x`'s `a`.
//!    The other axes, before or after it, may be regrouped (RFC-0044's `rope` and `heads` do
//!    both).
//! 5. **Rows.** `MatMulAct` and `SoftmaxRow` keep `a`, which must be `x`'s second-to-last axis,
//!    the queries' rows; the second operand of `MatMulAct`, the keys, is never read. Both of
//!    `SoftmaxRow`'s outputs keep it: in `R`, `x` without its last axis, it is the last axis.
//! 6. A tensor with no token axis gives outputs with none, except through the rules above, which
//!    never create one: only the two inputs of rule 1 do.
//!
//! RFC-0054's other rules name operations this lowering never emits (`Sub`, `ConstAdd`, `Sum`,
//! `Slice`, `GatherPublic`, `Argmax`, `InverseCdf`, `Bitwise`): [`Op`] has no variant for them.
//!
//! **Refusals**, each [`AxisError::Refused`] at its node, `Inconclusive(ERR-INTERNAL)`:
//! `Add` with inputs whose axes differ; `Mul` whose operand breaks its broadcast's rule;
//! `MatMulWeight` contracting it; `SumSquares` removing it; `Reshape` with element counts that
//! differ or no axis that keeps it whole; `MatMulAct` or `SoftmaxRow` whose `x` has it elsewhere
//! than at its second-to-last axis; and any output whose derived axis lies outside it or is not
//! as long as `x`'s. A node that reads a tensor no earlier step has derived, or an input given
//! twice, is [`AxisError::Internal`]; failed reservations [`AxisError::Memory`].

use core::fmt;

use crate::lowering::{Broadcast, Graph, NodeId, Op, Role, TensorId};

/// Every tensor's token axis, by tensor id (RFC-0054 'The token axis').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenAxes {
    axes: Vec<Option<u8>>,
}

/// Why a graph's token axes could not be derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisError {
    /// `Inconclusive(ERR-INTERNAL)`: the node breaks a rule of RFC-0054 'The token axis'.
    Refused {
        /// The node refused.
        node: NodeId,
    },
    /// `Inconclusive(ERR-RES-MEMORY)`.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: a graph out of topological order or with dangling ids.
    Internal,
}

impl fmt::Display for AxisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused { node } => write!(f, "node {} breaks a token-axis rule", node.0),
            Self::Memory => f.write_str("memory for the token axes could not be reserved"),
            Self::Internal => f.write_str("the graph's tensors are not in derivation order"),
        }
    }
}
impl std::error::Error for AxisError {}

impl TokenAxes {
    /// Derive every tensor's token axis by the rules of the module documentation, the graph's
    /// inputs first, then its nodes in order.
    ///
    /// # Errors
    /// [`AxisError::Refused`] at the first node that breaks a rule; `Internal` for a graph whose
    /// nodes read tensors not yet derived; `Memory` on failed reservation.
    pub fn derive(graph: &Graph) -> Result<Self, AxisError> {
        let count = graph.tensors().len();
        // `None` until derived: a node may read only tensors derived before it.
        let mut known: Vec<Option<Option<u8>>> = Vec::new();
        known
            .try_reserve_exact(count)
            .map_err(|_| AxisError::Memory)?;
        known.resize(count, None);
        for input in graph.inputs() {
            let axis = match input.role {
                Role::Tokens | Role::Positions => {
                    let rank = graph
                        .tensor(input.tensor)
                        .ok_or(AxisError::Internal)?
                        .shape
                        .dims()
                        .len();
                    if rank != 1 {
                        return Err(AxisError::Internal);
                    }
                    Some(0)
                }
                Role::Weight(_) | Role::Classes(_) | Role::Multipliers(_) => None,
            };
            let slot = known
                .get_mut(index(input.tensor)?)
                .ok_or(AxisError::Internal)?;
            if slot.replace(axis).is_some() {
                return Err(AxisError::Internal);
            }
        }
        for (position, node) in graph.nodes().iter().enumerate() {
            let id = NodeId(u32::try_from(position).map_err(|_| AxisError::Internal)?);
            let refused = AxisError::Refused { node: id };
            let axis_of = |tensor: TensorId| -> Result<Option<u8>, AxisError> {
                known
                    .get(index(tensor)?)
                    .copied()
                    .flatten()
                    .ok_or(AxisError::Internal)
            };
            let dims = |tensor: TensorId| -> Result<&[u32], AxisError> {
                Ok(graph
                    .tensor(tensor)
                    .ok_or(AxisError::Internal)?
                    .shape
                    .dims())
            };
            let x = *node.inputs().first().ok_or(AxisError::Internal)?;
            let a = axis_of(x)?;
            let x_dims = dims(x)?;
            let second = || -> Result<Option<u8>, AxisError> {
                axis_of(*node.inputs().get(1).ok_or(AxisError::Internal)?)
            };
            // The second-to-last axis of `x`, the rows of `MatMulAct` and `SoftmaxRow`.
            let rows = || {
                x_dims
                    .len()
                    .checked_sub(2)
                    .and_then(|r| u8::try_from(r).ok())
            };
            let out = match *node.op() {
                Op::OneHot { .. }
                | Op::ConstMul { .. }
                | Op::Rotate { .. }
                | Op::Requant { .. }
                | Op::TableMap { .. }
                | Op::RsqrtRow { .. } => a,
                Op::Add => {
                    if second()? != a {
                        return Err(refused);
                    }
                    a
                }
                Op::Mul { broadcast } => {
                    let b = second()?;
                    let fits = match broadcast {
                        Broadcast::Trailing => b == a,
                        Broadcast::Leading => b.is_none(),
                    };
                    if !fits {
                        return Err(refused);
                    }
                    a
                }
                Op::MatMulWeight { .. } => {
                    let last = x_dims
                        .len()
                        .checked_sub(1)
                        .and_then(|r| u8::try_from(r).ok());
                    if a.is_some() && a == last {
                        return Err(refused);
                    }
                    a
                }
                Op::Transpose { perm } => match a {
                    Some(a) => Some(
                        perm.iter()
                            .position(|&p| p == a)
                            .and_then(|i| u8::try_from(i).ok())
                            .ok_or(refused)?,
                    ),
                    None => None,
                },
                Op::Expand { axis, .. } => match a {
                    Some(a) if a >= axis => Some(a.checked_add(1).ok_or(refused)?),
                    other => other,
                },
                Op::SumSquares { axis } => match a {
                    Some(a) if a == axis => return Err(refused),
                    Some(a) if a > axis => Some(a.checked_sub(1).ok_or(refused)?),
                    other => other,
                },
                Op::Reshape => match a {
                    Some(a) => {
                        let output = *node.outputs().first().ok_or(AxisError::Internal)?;
                        Some(regrouped(x_dims, dims(output)?, a).ok_or(refused)?)
                    }
                    None => None,
                },
                Op::MatMulAct { .. } | Op::SoftmaxRow { .. } => {
                    if a.is_some() && a != rows() {
                        return Err(refused);
                    }
                    a
                }
            };
            for &output in node.outputs() {
                if let (Some(o), Some(a)) = (out, a) {
                    let length = dims(output)?.get(usize::from(o));
                    if length.is_none() || length != x_dims.get(usize::from(a)) {
                        return Err(refused);
                    }
                }
                let slot = known.get_mut(index(output)?).ok_or(AxisError::Internal)?;
                if slot.replace(out).is_some() {
                    return Err(AxisError::Internal);
                }
            }
        }
        let mut axes = Vec::new();
        axes.try_reserve_exact(count)
            .map_err(|_| AxisError::Memory)?;
        for axis in known {
            axes.push(axis.ok_or(AxisError::Internal)?);
        }
        Ok(Self { axes })
    }

    /// The token axis of `tensor`: `Some(None)` when it has none, `None` for an id outside the
    /// graph.
    pub fn get(&self, tensor: TensorId) -> Option<Option<u8>> {
        self.axes.get(usize::try_from(tensor.0).ok()?).copied()
    }

    /// Every tensor's token axis, by tensor id.
    pub fn as_slice(&self) -> &[Option<u8>] {
        &self.axes
    }
}

fn index(tensor: TensorId) -> Result<usize, AxisError> {
    usize::try_from(tensor.0).map_err(|_| AxisError::Internal)
}

/// `Reshape`'s token axis: the first output axis whose leading axes hold as many elements as
/// `input`'s before `axis` and which is as long; `None` when there is none or the element counts
/// differ (RFC-0006: `Reshape` keeps them).
fn regrouped(input: &[u32], output: &[u32], axis: u8) -> Option<u8> {
    let product = |dims: &[u32]| {
        dims.iter()
            .try_fold(1u64, |p, &d| p.checked_mul(u64::from(d)))
    };
    if product(input)? != product(output)? {
        return None;
    }
    let axis = usize::from(axis);
    let before = product(input.get(..axis)?)?;
    let length = *input.get(axis)?;
    let mut leading = 1u64;
    for (j, &d) in output.iter().enumerate() {
        if leading == before && d == length {
            return u8::try_from(j).ok();
        }
        leading = leading.checked_mul(u64::from(d))?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The first node, from `from`, whose operation `pick` accepts.
    fn find(graph: &Graph, from: u32, pick: impl Fn(&Op) -> bool) -> NodeId {
        let at = graph
            .nodes()
            .iter()
            .enumerate()
            .skip(from as usize)
            .find(|(_, n)| pick(n.op()))
            .unwrap()
            .0;
        NodeId(at as u32)
    }
    fn node(graph: &Graph, id: NodeId) -> crate::lowering::Node {
        *graph.node(id).unwrap()
    }
    fn refused(graph: &Graph, id: NodeId) {
        assert_eq!(
            TokenAxes::derive(graph),
            Err(AxisError::Refused { node: id }),
            "node {id:?}"
        );
    }

    #[test]
    fn the_lowering_satisfies_every_rule() {
        for model in [
            "tiny-2-servable",
            "tiny-2-minimal",
            "tiny-2-affine",
            "tiny-4-servable",
        ] {
            for n in [1, 4, 5, 16, 64] {
                let g = graph(model, n);
                let axes = TokenAxes::derive(&g).unwrap();
                assert_eq!(axes.as_slice().len(), g.tensors().len());
            }
        }
    }

    #[test]
    fn each_rule_refuses_the_node_that_breaks_it() {
        let base = graph("tiny-2-servable", 4);
        let tokens = base.inputs()[0].tensor;
        let weight = base.weights()[0].1;
        // `Add` whose inputs disagree: the residual's sum reads a weight, which has none.
        let add = find(&base, 0, |op| matches!(op, Op::Add));
        let mut g = base.clone();
        g.set_input(add, 1, weight);
        refused(&g, add);
        // `Mul { Trailing }` whose operand's axis is another: the attention's `Mul(PV, R)`, both
        // at axis 1, reads the first norm's root `[n]`, at axis 0, in place of `R`.
        let root = find(&base, 0, |op| matches!(op, Op::RsqrtRow { .. }));
        let rho = node(&base, root).outputs()[0];
        let softmax = find(&base, 0, |op| matches!(op, Op::SoftmaxRow { .. }));
        let scaled = find(&base, softmax.0, |op| matches!(op, Op::Mul { .. }));
        let trailing = Op::Mul {
            broadcast: Broadcast::Trailing,
        };
        assert_eq!(*node(&base, scaled).op(), trailing);
        let mut g = base.clone();
        g.set_input(scaled, 1, rho);
        refused(&g, scaled);
        // `Mul { Leading }` whose operand has one: the norm's root `[n]` broadcast by channel.
        let norm = find(&base, 0, |op| matches!(op, Op::Mul { .. }));
        assert_eq!(*node(&base, norm).op(), trailing);
        let mut g = base.clone();
        g.set_op(
            norm,
            Op::Mul {
                broadcast: Broadcast::Leading,
            },
        );
        refused(&g, norm);
        // `MatMulWeight` contracting it: the embedding reads the tokens `[n]`, their token
        // axis their last.
        let embed = find(&base, 0, |op| matches!(op, Op::MatMulWeight { .. }));
        let mut g = base.clone();
        g.set_input(embed, 0, tokens);
        refused(&g, embed);
        // `SumSquares` removing it.
        let squares = find(&base, 0, |op| matches!(op, Op::SumSquares { .. }));
        let mut g = base.clone();
        g.set_op(squares, Op::SumSquares { axis: 0 });
        refused(&g, squares);
        // `Reshape` that merges the token axis into another: RoPE's `[n, k, dh]` as `[n·k, dh]`.
        let reshape = find(&base, 0, |op| matches!(op, Op::Reshape));
        let split = node(&base, reshape).outputs()[0];
        let dims = base.tensor(split).unwrap().shape.dims().to_vec();
        assert_eq!(dims.len(), 3);
        let mut g = base.clone();
        g.set_shape(split, &[dims[0] * dims[1], dims[2]]);
        refused(&g, reshape);
        // `Reshape` that changes the element count.
        let mut g = base.clone();
        g.set_shape(split, &[dims[0], dims[1], dims[2] * 2]);
        refused(&g, reshape);
        // `MatMulAct` and `SoftmaxRow` whose rows are not the token axis: they read RoPE's
        // `[n, h, dh]`, whose token axis is 0, not 1.
        let scores = find(&base, 0, |op| matches!(op, Op::MatMulAct { .. }));
        let mut g = base.clone();
        g.set_input(scores, 0, split);
        refused(&g, scores);
        let mut g = base.clone();
        g.set_input(softmax, 0, split);
        refused(&g, softmax);
        // An output whose token axis is not as long as `x`'s: a `Requant` that doubles it.
        let requant = find(&base, 0, |op| matches!(op, Op::Requant { .. }));
        let out = node(&base, requant).outputs()[0];
        let dims = base.tensor(out).unwrap().shape.dims().to_vec();
        let mut g = base.clone();
        g.set_shape(out, &[dims[0] * 2, dims[1]]);
        refused(&g, requant);
        // The unedited graph passes, so each refusal above is its edit's.
        let axes = TokenAxes::derive(&base).unwrap();
        assert_eq!(axes.get(rho), Some(Some(0)));
        assert_eq!(axes.get(node(&base, softmax).outputs()[1]), Some(Some(1)));
        assert_eq!(axes.get(weight), Some(None));
        assert_eq!(axes.get(TensorId(u32::MAX)), None);
    }

    #[test]
    fn moved_axes_shift_only_past_the_axis_inserted_or_removed() {
        let base = graph("tiny-2-servable", 4);
        // `Expand` after the token axis leaves it: K's `[k, n, dh]` expanded at its end keeps
        // the queries at 1, which the GQA `Reshape` after it then cannot find as `[h, n, dh]`'s.
        let expand = find(&base, 0, |op| matches!(op, Op::Expand { .. }));
        let out = node(&base, expand).outputs()[0];
        let dims = base.tensor(out).unwrap().shape.dims().to_vec();
        let mut g = base.clone();
        g.set_op(
            expand,
            Op::Expand {
                axis: 3,
                len: dims[1],
            },
        );
        g.set_shape(out, &[dims[0], dims[2], dims[3], dims[1]]);
        refused(&g, NodeId(expand.0 + 1));
        // `SumSquares` before the token axis moves it down: the MLP norm summing the scores
        // `[h, n, n]` over their heads leaves the queries at 0 of `[n, n]`.
        let scores = find(&base, 0, |op| matches!(op, Op::MatMulAct { .. }));
        let scored = node(&base, scores).outputs()[0];
        let squares = find(&base, scores.0, |op| matches!(op, Op::SumSquares { .. }));
        let summed = node(&base, squares).outputs()[0];
        let mut g = base.clone();
        g.set_input(squares, 0, scored);
        g.set_op(squares, Op::SumSquares { axis: 0 });
        g.set_shape(summed, &[4, 4]);
        assert_eq!(TokenAxes::derive(&g).unwrap().get(summed), Some(Some(0)));
        // Summing the keys instead keeps it at 1, which the root after it, `[n]`, cannot hold:
        // at 0 it would, and at 2 the sum itself would be refused.
        g.set_op(squares, Op::SumSquares { axis: 2 });
        let root = find(&base, squares.0, |op| matches!(op, Op::RsqrtRow { .. }));
        refused(&g, root);
    }

    #[test]
    fn a_node_reading_an_underived_tensor_is_internal() {
        let base = graph("tiny-2-servable", 4);
        let first = find(&base, 0, |op| matches!(op, Op::OneHot { .. }));
        let last = TensorId(base.tensors().len() as u32 - 1);
        let mut g = base.clone();
        g.set_input(first, 0, last);
        assert_eq!(TokenAxes::derive(&g), Err(AxisError::Internal));
    }
}
