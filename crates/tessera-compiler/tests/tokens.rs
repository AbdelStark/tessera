// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 'The token axis': every tensor's token axis, derived from the graph alone.
//!
//! - **The lowerings.** In both profiles and at several lengths, the activations, the checkpoints,
//!   the one-hot vectors and the logits have it at axis 0, the scores and their rows at the query
//!   axis, and weights, classes and multipliers none; every token axis is as long as the sequence.
//! - **Lengths do not decide.** A head axis or a head's width as long as the sequence leaves it
//!   where the structure puts it.
//! - **Refusals.** A graph the rules do not fit is refused at the node that breaks them.

mod common;

use common::llama;
use tessera_compiler::schedule::token_axes;
use tessera_compiler::{CompileError, Lowering, PublicModel, lower};
use tessera_ir::{Op, Profile, TensorId};

/// `(d_model, heads, kv_heads, d_ff)`; every head is 128 wide.
type Shape = (u32, u32, u32, u32);

const SMALL: Shape = (256, 2, 1, 768);
/// Llama-3.1-405B's widths.
const B405: Shape = (16_384, 128, 8, 53_248);

fn lowering(profile: Profile, (d, heads, kv, ff): Shape, n: u32) -> Lowering {
    let (arch, quant, silu) = llama(profile, 2, d, heads, kv, ff);
    let model = PublicModel {
        arch: &arch,
        profile,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    lower(model, n).expect("the shape lowers")
}

fn axis(axes: &[Option<u8>], t: TensorId) -> Option<u8> {
    axes[t.0 as usize]
}

#[test]
fn every_tensor_has_its_structural_token_axis() {
    let cases = [
        (SMALL, 1),
        (SMALL, 3),
        (SMALL, 16),
        (SMALL, 129),
        (B405, 2),
        (B405, 16),
    ];
    for profile in [Profile::Minimal, Profile::Servable] {
        for (shape, n) in cases {
            let l = lowering(profile, shape, n);
            let axes = token_axes(&l).expect("the rules fit the lowering");
            let graph = &l.graph;
            for t in [l.tokens, l.positions, l.one_hot, l.logits] {
                assert_eq!(axis(&axes, t), Some(0), "{profile:?}, n = {n}");
            }
            for &t in &l.checkpoints {
                assert_eq!(axis(&axes, t), Some(0), "a checkpoint");
            }
            for &(_, t) in graph.weights.iter().chain(&l.classes).chain(&l.multipliers) {
                assert_eq!(axis(&axes, t), None, "weights, classes and multipliers");
            }
            let mut scores = 0;
            for node in &graph.nodes {
                if let Op::SoftmaxRow { .. } | Op::MatMulAct { .. } = node.op {
                    // The scores, their exponentials and rows, and the attention output: the
                    // query axis, second-to-last of the left operand.
                    assert_eq!(axis(&axes, node.inputs[0]), Some(1));
                    for &out in &node.outputs {
                        assert_eq!(axis(&axes, out), Some(1), "{:?}", node.op);
                    }
                    scores += 1;
                }
            }
            assert!(scores > 0);
            // Every token axis is as long as the sequence.
            for (spec, a) in graph.tensors.iter().zip(&axes) {
                if let Some(a) = a {
                    assert_eq!(spec.shape[usize::from(*a)], n, "tensor {}", spec.id.0);
                }
            }
        }
    }
}

#[test]
fn an_axis_as_long_as_the_sequence_does_not_move_it() {
    // Two heads at n = 2, and 128-wide heads at n = 128: the scores' head, query and key axes, or
    // a head's width, are as long as the sequence.
    for n in [2, 128] {
        let l = lowering(Profile::Servable, SMALL, n);
        let axes = token_axes(&l).expect("the rules fit the lowering");
        for node in &l.graph.nodes {
            if let Op::SoftmaxRow { .. } = node.op {
                let scores = &l.graph.tensors[node.inputs[0].0 as usize].shape;
                assert!(scores.iter().filter(|&&len| len == n).count() >= 2);
                assert_eq!(axis(&axes, node.inputs[0]), Some(1), "n = {n}");
            }
        }
    }
}

#[test]
fn a_graph_the_rules_do_not_fit_is_refused_at_its_node() {
    let base = lowering(Profile::Minimal, SMALL, 4);
    let find = |pred: &dyn Fn(&Op) -> bool| {
        base.graph
            .nodes
            .iter()
            .position(|node| pred(&node.op))
            .expect("the lowering has the node")
    };
    let refused = |l: &Lowering, at: usize, rule: &'static str| {
        assert_eq!(
            token_axes(l),
            Err(CompileError::Schedule {
                node: Some(l.graph.nodes[at].id),
                rule,
            })
        );
    };
    // A sum of squares over the positions.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::SumSquares { .. }));
    l.graph.nodes[at].op = Op::SumSquares { axis: 0 };
    refused(&l, at, "a sum runs over no token axis");
    // A reshape that splits the positions.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::Reshape));
    let out = l.graph.nodes[at].outputs[0].0 as usize;
    l.graph.tensors[out].shape.reverse();
    refused(&l, at, "a reshape keeps the token axis whole");
    // A transpose that moves the queries' axis last: the scores' product refuses it.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::Transpose { perm } if perm.len() == 3));
    l.graph.nodes[at].op = Op::Transpose {
        perm: vec![1, 2, 0],
    };
    match token_axes(&l) {
        Err(CompileError::Schedule {
            node: Some(id),
            rule,
        }) => {
            assert_eq!(
                rule,
                "the token axis of a product of activations is its rows'"
            );
            assert!(matches!(
                l.graph.nodes[id.0 as usize].op,
                Op::MatMulAct { .. }
            ));
        }
        other => panic!("{other:?}"),
    }
    // An operation the Llama lowering does not emit.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::TableMap { .. }));
    l.graph.nodes[at].op = Op::Argmax;
    refused(&l, at, "the Llama lowering emits no such operation");
    // A sum of a token-indexed tensor and a weight.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::Add));
    l.graph.nodes[at].inputs[1] = l.graph.weights[0].1;
    refused(&l, at, "a sum's inputs have one token axis");
    // A weight product over a softmax row's queries, its token axis last: the first weight
    // product after the softmax, so that the rows are derived before it.
    let mut l = base.clone();
    let softmax = find(&|op| matches!(op, Op::SoftmaxRow { .. }));
    let rows = l.graph.nodes[softmax].outputs[1];
    let at = softmax
        + l.graph.nodes[softmax..]
            .iter()
            .position(|node| matches!(node.op, Op::MatMulWeight { .. }))
            .expect("a weight product after the softmax");
    l.graph.nodes[at].inputs[0] = rows;
    refused(&l, at, "weights contract no token axis");
    // A row broadcast of another token axis: the softmax's reciprocal broadcast over its scores,
    // whose token axis is their second, from the positions, whose token axis is their first.
    let mut l = base.clone();
    let at = softmax
        + l.graph.nodes[softmax..]
            .iter()
            .position(|node| {
                matches!(
                    node.op,
                    Op::Mul {
                        broadcast: tessera_ir::Broadcast::Trailing
                    }
                )
            })
            .expect("a row broadcast after the softmax");
    let a = axis(
        &token_axes(&base).expect("axes"),
        l.graph.nodes[at].inputs[0],
    );
    assert_eq!(a, Some(1), "the broadcast runs over the scores");
    l.graph.nodes[at].inputs[1] = l.positions;
    refused(&l, at, "a row broadcast shares its row's token axis");
    // A slice, and a gather, of the positions.
    for (op, rule) in [
        (
            Op::Slice {
                axis: 0,
                start: 0,
                len: 1,
            },
            "a slice cuts no token axis",
        ),
        (
            Op::GatherPublic {
                index: tessera_ir::ConstRef(0),
            },
            "a gather picks no token's rows",
        ),
    ] {
        let mut l = base.clone();
        let at = find(&|op| matches!(op, Op::TableMap { .. }));
        l.graph.nodes[at].op = op;
        refused(&l, at, rule);
    }
    // A row broadcast from another axis: the norm's root over the channels.
    let mut l = base.clone();
    let at = find(&|op| matches!(op, Op::Mul { .. }));
    l.graph.nodes[at].op = Op::Mul {
        broadcast: tessera_ir::Broadcast::Leading,
    };
    let channel = l.graph.nodes[at].inputs[1];
    if axis(&token_axes(&base).expect("axes"), channel).is_some() {
        refused(&l, at, "a channel broadcast has no token axis");
    }
}
