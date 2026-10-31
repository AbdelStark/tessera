// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a regeneration reads of a lowering, from public data alone: each tensor's producer and
//! checkpoint, and each attention node's neighbours (RFC-0006 'Llama lowering', RFC-0037 'Head
//! layouts').
//!
//! - **Producers and checkpoints.** A tile's tensors come from the nodes between its layer's
//!   checkpoint and its own node: a replay walks back from the node through the producers, and
//!   stops at the checkpoints `c_0 … c_L` and at the graph's inputs (RFC-0018 'Witness engine').
//! - **Attention cores.** A layer's attention is the product `QK^T` of its query and key heads, its
//!   `SoftmaxRow`, the product `PV` of the exponentials and the value heads, and each row of that
//!   product times the row's reciprocal. Every one of these maps the row `(h, q)` of its first
//!   input to the same row of its output, so each row, and each attention tile's rows, compute on
//!   their own: the scores and exponentials, `[n_heads, n, n]`, never need to be whole.
//!   [`Structure::new`] recognizes a core only where each of its tensors has exactly the reader
//!   the core gives it, so nothing else ever needs them whole either.

use core::ops::Range;

use tessera_compiler::Lowering;
use tessera_ir::{ArchSpec, Broadcast, Graph, Node, Op, TableId, TensorId};

use crate::{Engine, WitnessError};

/// A layer's attention as the lowering builds it: `scores = QK^T`, `(e, R) = SoftmaxRow(scores)`,
/// `pv = PV` and `o = pv·R`, row by row over `(head, query)`, every tensor `[heads, n, ·]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Core {
    /// The node of `QK^T`.
    pub(crate) scores_node: u32,
    /// The `SoftmaxRow`: the layer's attention node (RFC-0041 'Tiles').
    pub(crate) softmax: u32,
    /// The node of `PV`.
    pub(crate) pv_node: u32,
    /// The node of `pv·R`.
    pub(crate) mul_node: u32,
    /// The query, key and value heads, `[heads, n, d_head]`.
    pub(crate) q: TensorId,
    pub(crate) k: TensorId,
    pub(crate) v: TensorId,
    /// The scores and exponentials `[heads, n, n]`, the reciprocals `[heads, n]`, `PV` and the
    /// output `[heads, n, d_head]`.
    pub(crate) scores: TensorId,
    pub(crate) e: TensorId,
    pub(crate) r: TensorId,
    pub(crate) pv: TensorId,
    pub(crate) o: TensorId,
    /// `heads`, `n` (token rows), `n_key` (score/exponential columns) and `d_head`.
    pub(crate) heads: usize,
    pub(crate) n: usize,
    pub(crate) n_key: usize,
    pub(crate) d_head: usize,
    /// The `Exp` table and the exponentials' precision `σ`.
    pub(crate) table: TableId,
    pub(crate) sigma: u8,
}

/// What a regeneration reads of an engine's lowering: public data alone.
pub(crate) struct Structure<'a> {
    pub(crate) engine: &'a Engine<'a>,
    pub(crate) lowering: &'a Lowering,
    /// Each tensor's producing node, or `None` for a graph input.
    producer: Vec<Option<u32>>,
    /// Each tensor's index among the checkpoints `c_0 … c_L`.
    checkpoint: Vec<Option<usize>>,
    /// The attention cores, in node order.
    pub(crate) cores: Vec<Core>,
    /// The query heads, those of one KV head (RFC-0041 'Tiles'), and the KV heads.
    heads: u32,
    group: u32,
    kv_heads: u32,
}

// RFC-0018: one exact-reserved reader slot per public input edge, including duplicates.
fn reader_lists(graph: &Graph) -> Result<Vec<Vec<u32>>, WitnessError> {
    let mut counts = vec![0usize; graph.tensors.len()];
    for node in &graph.nodes {
        for &tensor in &node.inputs {
            if let Some(count) = counts.get_mut(tensor.0 as usize) {
                *count = count
                    .checked_add(1)
                    .ok_or(WitnessError::Memory { tensor })?;
            }
        }
    }
    let mut readers: Vec<Vec<u32>> = counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    drop(counts);
    for node in &graph.nodes {
        for &tensor in &node.inputs {
            if let Some(list) = readers.get_mut(tensor.0 as usize) {
                list.push(node.id.0);
            }
        }
    }
    Ok(readers)
}

impl<'a> Structure<'a> {
    pub(crate) fn heap_bytes(&self) -> Option<usize> {
        self.producer
            .capacity()
            .checked_mul(size_of::<Option<u32>>())?
            .checked_add(
                self.checkpoint
                    .capacity()
                    .checked_mul(size_of::<Option<usize>>())?,
            )?
            .checked_add(self.cores.capacity().checked_mul(size_of::<Core>())?)
    }

    /// The structure of `engine`'s lowering, of a model of architecture `arch`.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Usage`] for an engine made without a lowering; [`WitnessError::Model`] for
    /// an architecture whose KV heads do not divide its query heads.
    pub(crate) fn new(engine: &'a Engine<'a>, arch: &ArchSpec) -> Result<Self, WitnessError> {
        let lowering = engine.require_lowering()?;
        if arch.n_kv_heads == 0 || !arch.n_heads.is_multiple_of(arch.n_kv_heads) {
            return Err(WitnessError::Model {
                rule: "n_kv_heads divides n_heads",
            });
        }
        let graph = &lowering.graph;
        let tensors = graph.tensors.len();
        let mut producer = vec![None; tensors];
        let readers = reader_lists(graph)?;
        for node in &graph.nodes {
            for &tensor in &node.outputs {
                if let Some(slot) = producer.get_mut(tensor.0 as usize) {
                    *slot = Some(node.id.0);
                }
            }
        }
        let mut checkpoint = vec![None; tensors];
        for (k, &t) in lowering.checkpoints.iter().enumerate() {
            if let Some(slot) = checkpoint.get_mut(t.0 as usize) {
                *slot = Some(k);
            }
        }
        let found = || {
            graph
                .nodes
                .iter()
                .filter_map(|node| core(graph, &producer, &readers, node))
        };
        let mut cores = Vec::with_capacity(found().count());
        cores.extend(found());
        Ok(Self {
            engine,
            lowering,
            producer,
            checkpoint,
            cores,
            heads: arch.n_heads,
            group: arch.n_heads / arch.n_kv_heads,
            kv_heads: arch.n_kv_heads,
        })
    }

    pub(crate) fn graph(&self) -> &'a Graph {
        &self.lowering.graph
    }

    /// The node that produces `tensor`, or `None` for a graph input.
    pub(crate) fn producer(&self, tensor: TensorId) -> Option<&'a Node> {
        let id = self.producer.get(tensor.0 as usize).copied().flatten()?;
        self.lowering.graph.nodes.get(id as usize)
    }

    /// `tensor`'s index among the checkpoints.
    pub(crate) fn checkpoint(&self, tensor: TensorId) -> Option<usize> {
        self.checkpoint.get(tensor.0 as usize).copied().flatten()
    }

    /// Whether a replay computes `tensor`: a node's output that is no checkpoint. A checkpoint
    /// comes from the store, and a graph input from the model and the tokens.
    pub(crate) fn computed(&self, tensor: TensorId) -> bool {
        self.producer(tensor).is_some() && self.checkpoint(tensor).is_none()
    }

    /// The core whose attention node is `node`.
    pub(crate) fn core_of(&self, node: u32) -> Option<usize> {
        self.cores.iter().position(|c| c.softmax == node)
    }

    /// `n_heads`, the query heads.
    pub(crate) fn heads(&self) -> usize {
        self.heads as usize
    }

    /// The query heads of KV head `kv_head`: those `h` with `⌊h / (n_heads / n_kv_heads)⌋ =
    /// kv_head` (RFC-0041 'Tiles'), or `None` past the KV heads.
    pub(crate) fn query_heads(&self, kv_head: u32) -> Option<Range<usize>> {
        (kv_head < self.kv_heads).then(|| {
            let group = self.group as usize;
            kv_head as usize * group..(kv_head as usize + 1) * group
        })
    }

    /// `n`, the sequence's positions.
    pub(crate) fn positions(&self) -> usize {
        self.lowering
            .graph
            .tensors
            .get(self.lowering.positions.0 as usize)
            .and_then(|t| t.shape.first())
            .map_or(0, |&n| n as usize)
    }
}

/// The attention core whose `SoftmaxRow` is `node`, if the graph around it is the lowering's and
/// each of its tensors has the one reader the core gives it.
fn core(
    graph: &Graph,
    producer: &[Option<u32>],
    readers: &[Vec<u32>],
    node: &Node,
) -> Option<Core> {
    let Op::SoftmaxRow { exp_table, sigma } = node.op else {
        return None;
    };
    let (&[scores], &[e, r]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
        return None;
    };
    let shape = |t: TensorId| graph.tensors.get(t.0 as usize).map(|s| s.shape.as_slice());
    let only = |t: TensorId| match readers.get(t.0 as usize).map(Vec::as_slice) {
        Some(&[reader]) => graph.nodes.get(reader as usize),
        _ => None,
    };
    let product = graph
        .nodes
        .get(producer.get(scores.0 as usize).copied()?? as usize)?;
    let pv_node = only(e)?;
    let mul_node = only(r)?;
    let (&[q, k], &[pv_scores]) = (product.inputs.as_slice(), product.outputs.as_slice()) else {
        return None;
    };
    let (&[pv_e, v], &[pv]) = (pv_node.inputs.as_slice(), pv_node.outputs.as_slice()) else {
        return None;
    };
    let (&[mul_pv, mul_r], &[o]) = (mul_node.inputs.as_slice(), mul_node.outputs.as_slice()) else {
        return None;
    };
    let fits = product.op
        == (Op::MatMulAct {
            transpose_rhs: true,
        })
        && pv_node.op
            == (Op::MatMulAct {
                transpose_rhs: false,
            })
        && mul_node.op
            == (Op::Mul {
                broadcast: Broadcast::Trailing,
            })
        && pv_scores == scores
        && pv_e == e
        && mul_pv == pv
        && mul_r == r
        && only(scores).map(|n| n.id) == Some(node.id)
        && only(pv).map(|n| n.id) == Some(mul_node.id);
    if !fits {
        return None;
    }
    let [heads, n, d_head] = *shape(q)? else {
        return None;
    };
    let n_key = graph
        .slots
        .as_ref()
        .filter(|s| n == s.token_len())
        .map_or(n, tessera_ir::BatchSlots::n_max);
    let rows = [heads, n, d_head];
    let shapes_fit = shape(k)? == rows
        && shape(v)? == rows
        && shape(pv)? == rows
        && shape(o)? == rows
        && shape(scores)? == [heads, n, n_key]
        && shape(e)? == [heads, n, n_key]
        && shape(r)? == [heads, n];
    shapes_fit.then_some(Core {
        scores_node: product.id.0,
        softmax: node.id.0,
        pv_node: pv_node.id.0,
        mul_node: mul_node.id.0,
        q,
        k,
        v,
        scores,
        e,
        r,
        pv,
        o,
        heads: heads as usize,
        n: n as usize,
        n_key: n_key as usize,
        d_head: d_head as usize,
        table: exp_table,
        sigma,
    })
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use crate::test_support::fixture;

    #[test]
    fn reader_and_core_metadata_reserve_exact_public_cardinalities() {
        let (model, lowering) = fixture(3);
        let graph = &lowering.graph;
        let readers = reader_lists(graph).unwrap();
        assert_eq!(readers.capacity(), graph.tensors.len());
        for (id, list) in readers.iter().enumerate() {
            let expected = graph
                .nodes
                .iter()
                .flat_map(|node| {
                    node.inputs
                        .iter()
                        .filter(move |t| t.0 as usize == id)
                        .map(move |_| node.id.0)
                })
                .collect::<Vec<_>>();
            assert_eq!(list, &expected);
            assert_eq!(list.capacity(), expected.len());
        }
        let engine = Engine::new(&lowering).unwrap();
        let mut structure = None;
        let info = allocation_counter::measure(|| {
            structure = Some(Structure::new(&engine, &model.arch).unwrap())
        });
        let structure = structure.unwrap();
        assert_eq!(structure.producer.capacity(), graph.tensors.len());
        assert_eq!(structure.checkpoint.capacity(), graph.tensors.len());
        assert_eq!(structure.cores.capacity(), structure.cores.len());
        let expected = graph.tensors.len()
            * (size_of::<Option<u32>>() + size_of::<Option<usize>>())
            + structure.cores.len() * size_of::<Core>();
        assert_eq!(usize::try_from(info.bytes_current).unwrap(), expected);
    }
}
