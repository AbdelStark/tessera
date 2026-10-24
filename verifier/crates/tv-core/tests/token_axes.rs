// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 'The token axis' on the conformance models' lowerings, against an element-level
//! oracle: every element's token followed through RFC-0006's operations, independently of the
//! axis rules. The refusals, which need edited graphs, are `leaf::groups::axes`'s unit tests.
mod claim_support;
mod group_support;
use group_support::{candidates, model, tags};
use tv_core::leaf::groups::TokenAxes;
use tv_core::lowering::{Graph, Kind, NodeId, Op, Role, TensorId};

const MODELS: [&str; 4] = [
    "tiny-2-servable",
    "tiny-2-minimal",
    "tiny-2-affine",
    "tiny-4-servable",
];

#[test]
fn every_tensor_has_the_axis_its_elements_follow_even_when_other_axes_are_as_long() {
    for name in MODELS {
        // 4 is tiny-2's head count, 16 its head width, 64 its width and context; 5 and 1 are
        // ragged and degenerate.
        for n in [1, 4, 5, 16, 64] {
            let (graph, _) = model(name, None, n);
            let axes = TokenAxes::derive(&graph).unwrap();
            let tags = tags(&graph);
            assert_eq!(axes.as_slice().len(), graph.tensors().len());
            let mut ambiguous_by_length = 0;
            for (id, &axis) in axes.as_slice().iter().enumerate() {
                let tensor = TensorId(id as u32);
                let found = candidates(&graph, &tags, tensor);
                let dims = graph.tensor(tensor).unwrap().shape.dims();
                match axis {
                    None => assert!(found.is_empty(), "{name} n={n} {tensor:?}: {found:?}"),
                    Some(a) if n > 1 => assert_eq!(found, [a], "{name} n={n} {tensor:?}"),
                    // One token: every axis of length 1 indexes it.
                    Some(a) => assert!(found.contains(&a), "{name} n={n} {tensor:?}"),
                }
                if axis.is_some() && dims.iter().filter(|&&d| d == n).count() >= 2 {
                    ambiguous_by_length += 1;
                }
            }
            if name.starts_with("tiny-2") && [4, 16, 64].contains(&n) {
                assert!(
                    ambiguous_by_length > 0,
                    "{name} n={n}: some tensor has another axis as long as the sequence"
                );
            }
        }
    }
}

fn nth(graph: &Graph, pick: fn(&Op) -> bool, nth: usize) -> NodeId {
    let at = graph
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| pick(node.op()))
        .nth(nth)
        .unwrap()
        .0;
    NodeId(at as u32)
}

#[test]
fn the_attention_axes_are_the_queries_rows_wherever_the_lengths_coincide() {
    // n = 4 = n_heads, so `[h, n, n]`, `[h, n]` and `[n, h, dh]` cannot be told by lengths.
    let (graph, _) = model("tiny-2-servable", None, 4);
    let axes = TokenAxes::derive(&graph).unwrap();
    let axis = |t: TensorId| axes.get(t).unwrap();
    let out = |node: NodeId, i: usize| graph.node(node).unwrap().outputs()[i];
    let dims = |t: TensorId| graph.tensor(t).unwrap().shape.dims().to_vec();
    for input in graph.inputs() {
        let expected = match input.role {
            Role::Tokens | Role::Positions => Some(0),
            _ => None,
        };
        assert_eq!(axis(input.tensor), expected, "{:?}", input.role);
    }
    let scores = out(nth(&graph, |op| matches!(op, Op::MatMulAct { .. }), 0), 0);
    assert_eq!((dims(scores), axis(scores)), (vec![4, 4, 4], Some(1)));
    let softmax = nth(&graph, |op| matches!(op, Op::SoftmaxRow { .. }), 0);
    assert_eq!(axis(out(softmax, 0)), Some(1));
    assert_eq!(
        (dims(out(softmax, 1)), axis(out(softmax, 1))),
        (vec![4, 4], Some(1))
    );
    assert!(matches!(
        graph.tensor(out(softmax, 1)).unwrap().kind,
        Kind::Advice
    ));
    // K and V: `Expand` past the head axis, then `Reshape` regroups the axes before it.
    let expand = nth(&graph, |op| matches!(op, Op::Expand { .. }), 0);
    assert_eq!(
        (dims(out(expand, 0)), axis(out(expand, 0))),
        (vec![2, 2, 4, 16], Some(2))
    );
    let regrouped = out(NodeId(expand.0 + 1), 0);
    assert_eq!(
        (dims(regrouped), axis(regrouped)),
        (vec![4, 4, 16], Some(1))
    );
    // The attention output, transposed back to the queries first, then flattened.
    let back = nth(&graph, |op| matches!(op, Op::Transpose { .. }), 3);
    assert_eq!(
        (dims(out(back, 0)), axis(out(back, 0))),
        (vec![4, 4, 16], Some(0))
    );
    assert_eq!(axis(out(NodeId(back.0 + 1), 0)), Some(0));
}
