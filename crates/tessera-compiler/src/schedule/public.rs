// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the schedule reads of every tensor before any claim: its producer, and whether the
//! verifier knows it (RFC-0072 'Public tensors').

use tessera_ir::{Graph, NodeId, NodeKind, OpClass};

/// Each tensor's producing node and the output it is, by tensor id: `None` for a graph input.
/// One pass over the nodes, so that finding a producer costs `O(1)` on a graph of thousands of
/// nodes, as a SHA-256 block's 6,032 are.
pub fn producers(graph: &Graph) -> Vec<Option<(NodeId, u32)>> {
    let mut producers = vec![None; graph.tensors.len()];
    for node in &graph.nodes {
        for (output, &t) in (0u32..).zip(&node.outputs) {
            if let Some(slot) = producers.get_mut(t.0 as usize) {
                *slot = Some((node.id, output));
            }
        }
    }
    producers
}

/// Whether each tensor of `graph` is public, by tensor id (RFC-0072 'Public tensors'): the
/// verifier computes it from public data, so a claim on it reaches no instance and no pool.
///
/// - A graph input declared public is public.
/// - So is the output of a node that is not gadget-backed (`OpClass::Gadget`) and has at least
///   one tensor input, every one public: linear maps, selections, `Mul` and `BitFn` propagate
///   publicness, such as a `GatherPublic` of SHA-256's `SHR` masks or a `Slice` of its digest.
/// - A tensor declared committed is committed, whatever its inputs; a lookup's output, and every
///   other gadget's, is never public.
///
/// Constants are not tensors: a `ConstMul` or `ConstAdd` of a public tensor is public.
pub fn public_tensors(graph: &Graph) -> Vec<bool> {
    let produced = producers(graph);
    let mut public: Vec<bool> = graph
        .tensors
        .iter()
        .zip(&produced)
        .map(|(spec, producer)| producer.is_none() && spec.kind == NodeKind::Public)
        .collect();
    for node in &graph.nodes {
        let derived = node.op.class() != OpClass::Gadget
            && !node.inputs.is_empty()
            && node
                .inputs
                .iter()
                .all(|t| public.get(t.0 as usize).copied().unwrap_or(false));
        if !derived {
            continue;
        }
        for &t in &node.outputs {
            let declared = graph
                .tensors
                .get(t.0 as usize)
                .is_some_and(|spec| matches!(spec.kind, NodeKind::Committed { .. }));
            if let Some(slot) = public.get_mut(t.0 as usize) {
                *slot = !declared;
            }
        }
    }
    public
}
