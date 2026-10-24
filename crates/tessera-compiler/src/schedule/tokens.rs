// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The token axis of every tensor (RFC-0054 'Groups', 'The token axis'): the axis a token group
//! splits, derived from the graph alone, input to output, so that the prover's and the verifier's
//! lowerings agree (RFC-0044). Lengths cannot tell it: in a small model a head or channel axis can
//! be as long as the sequence.

use tessera_ir::{Broadcast, Op, TensorId};

use crate::{CompileError, Lowering};

/// Each tensor's token axis, by tensor id: `Some(axis)`, counted from the outermost, or `None`
/// for a tensor that no token indexes, such as a weight, a constant vector or a channel's class.
///
/// - **Inputs.** The tokens and the positions, `[n]`, have it at axis 0; every other input has
///   none.
/// - **Kept.** `Add`, `Sub`, `ConstMul`, `ConstAdd`, `Rotate`, `Requant`, `TableMap`, `RsqrtRow`,
///   `MatMulWeight` and `Mul` keep their first input's, and `OneHot` its tokens'. `Add` and `Sub`
///   need their inputs' to agree, `Mul` its `Trailing` operand's when it has one, and
///   `MatMulWeight` it not to be the last axis, the one the weights contract.
/// - **Moved.** `Transpose` moves it with its permutation, `Expand` past its new axis, and
///   `Sum`, `SumSquares` and `Slice` past an axis they remove or cut, which may not be it.
///   `Reshape` keeps it whole, the axes before it and those after it each regrouping among
///   themselves: the token axis is then the first output axis whose leading axes hold as many
///   elements as the input's before it, and which is as long.
/// - **Rows.** `MatMulAct` and `SoftmaxRow` keep their first input's, which, when it has one,
///   must be the second-to-last axis, the queries' rows: the scores' token axis is the query
///   axis, and the keys, `MatMulAct`'s second operand, are never split.
///
/// # Errors
///
/// [`CompileError::Schedule`], naming the node, for a graph these rules do not fit:
/// `GatherPublic` over the token axis, `Argmax`, `InverseCdf` and `Bitwise`, which RFC-0006's
/// Llama lowering does not emit, and any operation whose inputs break its rule above.
pub fn token_axes(lowering: &Lowering) -> Result<Vec<Option<u8>>, CompileError> {
    let graph = &lowering.graph;
    let mut axes: Vec<Option<u8>> = vec![None; graph.tensors.len()];
    for input in [lowering.tokens, lowering.positions] {
        let slot = axes
            .get_mut(input.0 as usize)
            .ok_or(CompileError::Schedule {
                node: None,
                rule: "the tokens and the positions are tensors of the graph",
            })?;
        *slot = Some(0);
    }
    for node in &graph.nodes {
        let bad = |rule| CompileError::Schedule {
            node: Some(node.id),
            rule,
        };
        let of = |t: TensorId| axes.get(t.0 as usize).copied().flatten();
        let rank = |t: TensorId| graph.tensors.get(t.0 as usize).map_or(0, |s| s.shape.len());
        let first = node
            .inputs
            .first()
            .copied()
            .ok_or(bad("an operation has an input"))?;
        let x = of(first);
        let derived: Vec<Option<u8>> = match &node.op {
            Op::Add | Op::Sub => {
                let y = node.inputs.get(1).copied().map(of);
                if y.is_some_and(|y| y != x) {
                    return Err(bad("a sum's inputs have one token axis"));
                }
                vec![x]
            }
            Op::ConstMul { .. }
            | Op::ConstAdd { .. }
            | Op::Rotate { .. }
            | Op::Requant { .. }
            | Op::TableMap { .. }
            | Op::RsqrtRow { .. } => vec![x],
            Op::MatMulWeight { .. } => {
                if x.is_some_and(|a| usize::from(a) + 1 == rank(first)) {
                    return Err(bad("weights contract no token axis"));
                }
                vec![x]
            }
            Op::Mul { broadcast } => {
                let b = node.inputs.get(1).copied().and_then(of);
                match broadcast {
                    Broadcast::Trailing if b.is_some() && b != x => {
                        return Err(bad("a row broadcast shares its row's token axis"));
                    }
                    Broadcast::Leading if b.is_some() => {
                        return Err(bad("a channel broadcast has no token axis"));
                    }
                    _ => vec![x],
                }
            }
            Op::OneHot { .. } => vec![x.map(|_| 0)],
            Op::Transpose { perm } => vec![match x {
                Some(a) => Some(
                    perm.iter()
                        .position(|&p| p == a)
                        .and_then(|i| u8::try_from(i).ok())
                        .ok_or(bad("a permutation moves every axis"))?,
                ),
                None => None,
            }],
            Op::Expand { axis, .. } => vec![x.map(|a| if *axis <= a { a + 1 } else { a })],
            Op::Sum { axis } | Op::SumSquares { axis } => vec![match x {
                Some(a) if a == *axis => return Err(bad("a sum runs over no token axis")),
                Some(a) if *axis < a => Some(a - 1),
                other => other,
            }],
            Op::Slice { axis, .. } => {
                if x == Some(*axis) {
                    return Err(bad("a slice cuts no token axis"));
                }
                vec![x]
            }
            Op::Reshape => {
                let shape = |t: TensorId| {
                    graph
                        .tensors
                        .get(t.0 as usize)
                        .map_or(&[][..], |s| s.shape.as_slice())
                };
                let output = node.outputs.first().copied().map_or(&[][..], shape);
                vec![match x {
                    Some(a) => {
                        // The axes before it regrouped, it untouched: the output axis whose
                        // leading axes hold as many elements, and which is as long.
                        let input = shape(first);
                        let before = |dims: &[u32], k: usize| {
                            dims[..k].iter().map(|&d| u64::from(d)).product::<u64>()
                        };
                        let a = usize::from(a);
                        let at = (0..output.len()).find(|&k| {
                            output[k] == input[a] && before(output, k) == before(input, a)
                        });
                        Some(
                            at.and_then(|k| u8::try_from(k).ok())
                                .ok_or(bad("a reshape keeps the token axis whole"))?,
                        )
                    }
                    None => None,
                }]
            }
            Op::MatMulAct { .. } | Op::SoftmaxRow { .. } => {
                if x.is_some_and(|a| usize::from(a) + 2 != rank(first)) {
                    return Err(bad(
                        "the token axis of a product of activations is its rows'",
                    ));
                }
                vec![x; node.outputs.len()]
            }
            Op::GatherPublic { .. } => {
                if x == Some(0) {
                    return Err(bad("a gather picks no token's rows"));
                }
                vec![x]
            }
            Op::Argmax | Op::InverseCdf { .. } | Op::Bitwise { .. } => {
                return Err(bad("the Llama lowering emits no such operation"));
            }
        };
        for (&output, axis) in node.outputs.iter().zip(derived.iter().cycle()) {
            if let Some(slot) = axes.get_mut(output.0 as usize) {
                *slot = *axis;
            }
        }
    }
    Ok(axes)
}
