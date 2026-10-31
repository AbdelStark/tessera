// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Replays of `D` from the checkpoints (RFC-0018 'Witness engine'): the tensors a node reads,
//! computed from its layer's checkpoint, never from the layers before it.
//!
//! - **The cone.** [`Replay::compute`] walks back from the tensors it is asked for, through their
//!   producers, to the checkpoints and the graph's inputs, and runs those nodes in node order
//!   through [`Engine::node`], the reference engine's own operations, so every value is the first
//!   pass's.
//! - **Memory.** A tensor lives until the last node of the cone that reads it: then its buffer
//!   drops, and clears its memory (OBS-6). A checkpoint is read out of the store, and a channel
//!   vector out of the model, for each node that reads it, and dropped after it. The weights stay
//!   the model's INT8 matrices, lent to the matmuls. A layer's attention core runs row by row
//!   ([`Replay::attention`]): its `[n_heads, n, n]` scores and exponentials are never whole, and
//!   an attention tile computes its own rows alone ([`Replay::attention_rows`]).
//! - **Checks.** Debug builds check every value against its tensor's analyzed range, as a run
//!   does (INV-DM-2).
//!
//! The tensors are secret-class (05 §3): each lives in a `SecretBuf`, and errors name tensors and
//! rules only. The tokens leave their `Secret` in the engine, which reads them for a replay.

use core::ops::Range;

use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::Lowering;
use tessera_ir::{IntegerModel, Node, TensorId};

use crate::engine::{Input, Operand, act_row, check, graph_inputs, softmax_row, zeros};
use crate::structure::{Core, Structure};
use crate::{Checkpoints, WitnessError};

/// The tensors a replay holds, by id: the computed ones that a node still to run, or its caller,
/// reads.
pub(crate) struct Live {
    values: Vec<Option<SecretBuf<i64>>>,
}

impl Live {
    fn new(tensors: usize) -> Self {
        Self {
            values: (0..tensors).map(|_| None).collect(),
        }
    }

    /// The values of `tensor`, if the replay holds them.
    pub(crate) fn get(&self, tensor: TensorId) -> Option<&[i64]> {
        self.values.get(tensor.0 as usize)?.as_deref()
    }

    /// Transfer a requested tensor without retaining or copying the replay's allocation.
    pub(crate) fn take(&mut self, tensor: TensorId) -> Option<SecretBuf<i64>> {
        self.values.get_mut(tensor.0 as usize)?.take()
    }

    fn require(&self, tensor: TensorId) -> Result<&[i64], WitnessError> {
        self.get(tensor).ok_or(WitnessError::Internal {
            rule: "a node's inputs are computed before it",
        })
    }

    /// One reader of `tensor` has run: when it was the last, the buffer drops.
    fn release(&mut self, tensor: TensorId, remaining: &mut [u32]) {
        let Some(count) = remaining.get_mut(tensor.0 as usize) else {
            return;
        };
        if *count == 0 {
            return;
        }
        *count -= 1;
        if *count == 0
            && let Some(slot) = self.values.get_mut(tensor.0 as usize)
        {
            *slot = None;
        }
    }
}

/// Where each input of a node comes from: a tensor the replay holds, a leaf read for the node, or
/// a weight the model lends.
pub(crate) struct Operands<'m> {
    sources: Vec<Source<'m>>,
}

enum Source<'m> {
    Live(TensorId),
    Leaf(SecretBuf<i64>),
    Weight(&'m [i8]),
}

// RFC-0018: exact-reserved source and borrowed-operand descriptor arrays.
pub(crate) fn operand_metadata_bytes(count: u64) -> Option<u64> {
    usize::try_from(count).ok()?;
    let sources = count.checked_mul(size_of::<Source<'_>>() as u64)?;
    let operands = count.checked_mul(size_of::<Operand<'_>>() as u64)?;
    if sources > isize::MAX as u64 || operands > isize::MAX as u64 {
        return None;
    }
    sources.checked_add(operands)
}

impl<'m> Operands<'m> {
    /// The node's inputs, in its order.
    pub(crate) fn all<'s>(&'s self, live: &'s Live) -> Result<Vec<Operand<'s>>, WitnessError>
    where
        'm: 's,
    {
        let mut operands = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            operands.push(match source {
                Source::Live(t) => Operand::Values(live.require(*t)?),
                Source::Leaf(values) => Operand::Values(values),
                Source::Weight(entries) => Operand::Weight(entries),
            });
        }
        Ok(operands)
    }
}

/// The rows of an attention tile: the scores, exponentials and reciprocals of its query heads and
/// held queries, row `(h, q)` after row, every key in each.
pub(crate) struct Rows {
    pub(crate) scores: SecretBuf<i64>,
    pub(crate) e: SecretBuf<i64>,
    pub(crate) r: SecretBuf<i64>,
}

/// A replay of a lowering's sequence: its structure, the model and the tokens.
pub(crate) struct Replay<'a> {
    pub(crate) structure: Structure<'a>,
    model: &'a IntegerModel,
    tokens: &'a Secret<Vec<u32>>,
}

impl<'a> Replay<'a> {
    /// A replay of `model` on `tokens`, after checking every graph input as a run does, in a run's
    /// order: the tokens, the positions, the weights, the channel classes and multipliers.
    ///
    /// # Errors
    ///
    /// As [`Engine::run`](crate::Engine::run) for the tokens and the model.
    pub(crate) fn new(
        structure: Structure<'a>,
        model: &'a IntegerModel,
        tokens: &'a Secret<Vec<u32>>,
    ) -> Result<Self, WitnessError> {
        let lowering = structure.lowering;
        for tensor in graph_inputs(lowering) {
            structure
                .engine
                .graph_input(lowering, model, tokens, tensor)?;
        }
        Ok(Self {
            structure,
            model,
            tokens,
        })
    }

    pub(crate) fn lowering(&self) -> &'a Lowering {
        self.structure.lowering
    }

    /// The tokens, as the token tensor's values.
    pub(crate) fn tokens(&self) -> Result<SecretBuf<i64>, WitnessError> {
        self.structure
            .engine
            .tokens(self.lowering(), self.model, self.tokens)
    }

    /// A tensor no replay computes: a checkpoint, read out of `store` as `i64`, or a graph input.
    pub(crate) fn leaf(
        &self,
        store: &Checkpoints,
        tensor: TensorId,
    ) -> Result<Input<'a>, WitnessError> {
        if let Some(k) = self.structure.checkpoint(tensor) {
            let residual = store.residual(k).ok_or(WitnessError::Internal {
                rule: "a replay reads only the checkpoints before its tensors",
            })?;
            let mut values = zeros(tensor, residual.len())?;
            for (slot, &v) in values.iter_mut().zip(residual) {
                *slot = i64::from(v);
            }
            return Ok(Input::Values(values));
        }
        self.structure
            .engine
            .graph_input(self.lowering(), self.model, self.tokens, tensor)
    }

    /// The inputs of `node`: the tensors `live` holds, and its leaves, read for it.
    pub(crate) fn operands(
        &self,
        store: &Checkpoints,
        node: &Node,
        live: &Live,
    ) -> Result<Operands<'a>, WitnessError> {
        let mut sources = Vec::with_capacity(node.inputs.len());
        for &t in &node.inputs {
            let source = if live.get(t).is_some() {
                Source::Live(t)
            } else {
                if self.structure.computed(t) {
                    return Err(WitnessError::Internal {
                        rule: "a node's inputs are computed before it",
                    });
                }
                match self.leaf(store, t)? {
                    Input::Values(values) => Source::Leaf(values),
                    Input::Weight(entries) => Source::Weight(entries),
                }
            };
            sources.push(source);
        }
        Ok(Operands { sources })
    }

    /// The outputs of `node` on `inputs`, each checked against its analyzed range in debug builds
    /// (INV-DM-2).
    pub(crate) fn run(
        &self,
        node: &Node,
        inputs: &[Operand<'_>],
    ) -> Result<Vec<SecretBuf<i64>>, WitnessError> {
        let graph = self.structure.graph();
        let outputs = self.structure.engine.node(node, inputs)?;
        for (&t, values) in node.outputs.iter().zip(&outputs) {
            check(t, values, graph.tensors[t.0 as usize].range)?;
        }
        Ok(outputs)
    }

    // RFC-0006: Engine::for_graph validates dense topological node order. Each
    // computed input's producer is earlier, so one reverse scan closes the cone
    // without a DFS worklist. Checkpoints remain leaves (INV-ARCH-3).
    fn needed_nodes(&self, wanted: &[TensorId]) -> Vec<bool> {
        let graph = self.structure.graph();
        let mut needed = vec![false; graph.nodes.len()];
        let mark = |tensor, needed: &mut [bool]| {
            if self.structure.computed(tensor)
                && let Some(node) = self.structure.producer(tensor)
            {
                needed[node.id.0 as usize] = true;
            }
        };
        for &tensor in wanted {
            mark(tensor, &mut needed);
        }
        for node in graph.nodes.iter().rev() {
            if needed[node.id.0 as usize] {
                for &tensor in &node.inputs {
                    mark(tensor, &mut needed);
                }
            }
        }
        needed
    }

    /// Computes `wanted` from the checkpoints in `store`, and returns a [`Live`] that holds them
    /// and nothing else: the computed tensors among them, since a checkpoint or a graph input is
    /// read where it is used.
    ///
    /// The nodes run are those `wanted` needs, back to the checkpoints and the graph's inputs, in
    /// node order. An attention core whose output they need runs as one, row by row, unless a
    /// wanted tensor is one of its others.
    ///
    /// # Errors
    ///
    /// As [`Engine::run`](crate::Engine::run): a node's refusal or a value outside its range.
    pub(crate) fn compute(
        &self,
        store: &Checkpoints,
        wanted: &[TensorId],
    ) -> Result<Live, WitnessError> {
        let structure = &self.structure;
        let graph = structure.graph();
        let needed = self.needed_nodes(wanted);
        // A core runs as one, at its last node, when its output is needed and no wanted tensor is
        // one of its others: by then its query, key and value heads are all computed.
        let mut unit: Vec<Option<&Core>> = vec![None; graph.nodes.len()];
        let mut inside = vec![false; graph.nodes.len()];
        for core in &structure.cores {
            let others = [core.scores, core.e, core.r, core.pv];
            if needed[core.mul_node as usize] && !wanted.iter().any(|t| others.contains(t)) {
                unit[core.mul_node as usize] = Some(core);
                for node in [core.scores_node, core.softmax, core.pv_node] {
                    inside[node as usize] = true;
                }
            }
        }
        let reads = |id: usize| -> Vec<TensorId> {
            match unit[id] {
                Some(core) => vec![core.q, core.k, core.v],
                None => graph.nodes[id].inputs.clone(),
            }
        };
        let mut remaining = vec![0u32; graph.tensors.len()];
        for id in (0..graph.nodes.len()).filter(|&id| needed[id] && !inside[id]) {
            for t in reads(id) {
                if structure.computed(t) {
                    remaining[t.0 as usize] += 1;
                }
            }
        }
        for &t in wanted {
            if structure.computed(t) {
                remaining[t.0 as usize] += 1;
            }
        }
        let mut live = Live::new(graph.tensors.len());
        for id in (0..graph.nodes.len()).filter(|&id| needed[id] && !inside[id]) {
            if let Some(core) = unit[id] {
                let o = self.attention(core, &live)?;
                for t in reads(id) {
                    live.release(t, &mut remaining);
                }
                if remaining[core.o.0 as usize] > 0 {
                    live.values[core.o.0 as usize] = Some(o);
                }
                continue;
            }
            let node = &graph.nodes[id];
            let outputs = {
                let operands = self.operands(store, node, &live)?;
                self.run(node, &operands.all(&live)?)?
            };
            for &t in &node.inputs {
                live.release(t, &mut remaining);
            }
            for (&t, values) in node.outputs.iter().zip(outputs) {
                if remaining[t.0 as usize] > 0 {
                    live.values[t.0 as usize] = Some(values);
                }
            }
        }
        Ok(live)
    }

    /// The output `o` of `core`, every row `(h, q)` computed on its own: `q`'s scores against
    /// every key, its exponentials and reciprocal, its product with the value heads, and that
    /// product times the reciprocal. Only one row's scores, exponentials and product are held at a
    /// time, beside `o`.
    fn attention(&self, core: &Core, live: &Live) -> Result<SecretBuf<i64>, WitnessError> {
        let (q, k, v) = (
            live.require(core.q)?,
            live.require(core.k)?,
            live.require(core.v)?,
        );
        let (heads, n, n_key, d_head) = (core.heads, core.n, core.n_key, core.d_head);
        let graph = self.structure.graph();
        let slotted = graph.slots.as_ref().filter(|s| n == s.token_len() as usize);
        let range = |t: TensorId| graph.tensors[t.0 as usize].range;
        let table = self.structure.engine.lookup(core.table)?;
        let mut o = zeros(core.o, heads * n * d_head)?;
        let mut scores = zeros(core.scores, n_key)?;
        let mut e = zeros(core.e, n_key)?;
        let mut pv = zeros(core.pv, d_head)?;
        for head in 0..heads {
            let keys = &k[head * n * d_head..(head + 1) * n * d_head];
            let values = &v[head * n * d_head..(head + 1) * n * d_head];
            for query in 0..n {
                if slotted.is_some_and(|s| !s.is_real_row(query)) {
                    continue;
                }
                let (slot_base, q_pos) = match slotted {
                    Some(slots) => {
                        let s = slots.slot_len as usize;
                        ((query / s) * s, query % s)
                    }
                    None => (0, query),
                };
                let slot_keys = &keys[slot_base * d_head..(slot_base + n_key) * d_head];
                let slot_values = &values[slot_base * d_head..(slot_base + n_key) * d_head];
                let row = head * n + query;
                act_row(
                    &q[row * d_head..(row + 1) * d_head],
                    slot_keys,
                    (d_head, n_key),
                    true,
                    &mut scores,
                );
                e.fill(0);
                let r = softmax_row(table, core.sigma, &scores, q_pos, &mut e)?;
                act_row(&e, slot_values, (n_key, d_head), false, &mut pv);
                check(core.scores, &scores, range(core.scores))?;
                check(core.e, &e, range(core.e))?;
                check(core.r, &[r], range(core.r))?;
                check(core.pv, &pv, range(core.pv))?;
                for (slot, &x) in o[row * d_head..(row + 1) * d_head]
                    .iter_mut()
                    .zip(pv.iter())
                {
                    *slot = x * r;
                }
            }
        }
        check(core.o, &o, range(core.o))?;
        Ok(o)
    }

    /// The rows of the attention tile of `core` at the query heads `heads` and the queries
    /// `queries`, from the query and key heads `live` holds.
    pub(crate) fn attention_rows(
        &self,
        core: &Core,
        live: &Live,
        heads: Range<usize>,
        queries: Range<usize>,
    ) -> Result<Rows, WitnessError> {
        let (q, k) = (live.require(core.q)?, live.require(core.k)?);
        let (n, n_key, d_head) = (core.n, core.n_key, core.d_head);
        let graph = self.structure.graph();
        let slotted = graph.slots.as_ref().filter(|s| n == s.token_len() as usize);
        let range = |t: TensorId| graph.tensors[t.0 as usize].range;
        let table = self.structure.engine.lookup(core.table)?;
        let rows = heads.len() * queries.len();
        let mut scores = zeros(core.scores, rows * n_key)?;
        let mut e = zeros(core.e, rows * n_key)?;
        let mut r = zeros(core.r, rows)?;
        let mut x = 0;
        for head in heads {
            let keys = &k[head * n * d_head..(head + 1) * n * d_head];
            for query in queries.clone() {
                let (slot_base, q_pos, is_real) = match slotted {
                    Some(slots) => {
                        let s = slots.slot_len as usize;
                        ((query / s) * s, query % s, slots.is_real_row(query))
                    }
                    None => (0, query, true),
                };
                let slot_keys = &keys[slot_base * d_head..(slot_base + n_key) * d_head];
                let row = head * n + query;
                let held = x * n_key..(x + 1) * n_key;
                act_row(
                    &q[row * d_head..(row + 1) * d_head],
                    slot_keys,
                    (d_head, n_key),
                    true,
                    &mut scores[held.clone()],
                );
                if is_real {
                    r[x] = softmax_row(
                        table,
                        core.sigma,
                        &scores[held.clone()],
                        q_pos,
                        &mut e[held],
                    )?;
                }
                x += 1;
            }
        }
        check(core.scores, &scores, range(core.scores))?;
        check(core.e, &e, range(core.e))?;
        check(core.r, &r, range(core.r))?;
        Ok(Rows { scores, e, r })
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use crate::Engine;
    use crate::test_support::fixture;

    #[test]
    fn reverse_cones_match_independent_set_frontiers_and_stop_at_checkpoints() {
        use std::collections::BTreeSet;
        let (model, lowering) = fixture(3);
        let engine = Engine::new(&lowering).unwrap();
        let tokens = Secret::new(vec![0, 1, 2]);
        let replay = Replay::new(
            Structure::new(&engine, &model.arch).unwrap(),
            &model,
            &tokens,
        )
        .unwrap();
        let graph = &lowering.graph;
        let mut cases = graph
            .tensors
            .iter()
            .map(|tensor| vec![tensor.id])
            .collect::<Vec<_>>();
        cases.extend(
            graph
                .tensors
                .iter()
                .map(|tensor| vec![tensor.id, tensor.id]),
        );
        cases.extend([
            vec![],
            vec![TensorId(u32::MAX)],
            graph.tensors.iter().rev().map(|tensor| tensor.id).collect(),
        ]);
        for wanted in cases {
            let mut pending = wanted
                .iter()
                .map(|tensor| tensor.0)
                .collect::<BTreeSet<_>>();
            let mut visited = BTreeSet::new();
            let mut expected = vec![false; graph.nodes.len()];
            while let Some(id) = pending.pop_first() {
                let tensor = TensorId(id);
                if !visited.insert(id) || !replay.structure.computed(tensor) {
                    continue;
                }
                if let Some(node) = replay.structure.producer(tensor) {
                    expected[node.id.0 as usize] = true;
                    pending.extend(node.inputs.iter().map(|tensor| tensor.0));
                }
            }
            assert_eq!(replay.needed_nodes(&wanted), expected);
        }
    }

    #[test]
    fn replay_operand_arrays_match_the_public_arity_without_growth() {
        let (model, lowering) = fixture(3);
        let engine = Engine::new(&lowering)
            .unwrap()
            .threads(core::num::NonZeroUsize::MIN);
        let tokens = Secret::new(vec![0, 1, 2]);
        let store = Checkpoints::new(&engine, &model, &tokens).unwrap();
        let replay = Replay::new(
            Structure::new(&engine, &model.arch).unwrap(),
            &model,
            &tokens,
        )
        .unwrap();
        let wanted = lowering
            .graph
            .tensors
            .iter()
            .filter(|tensor| {
                !lowering
                    .graph
                    .weights
                    .iter()
                    .any(|&(_, id)| id == tensor.id)
            })
            .map(|tensor| tensor.id)
            .collect::<Vec<_>>();
        let live = replay.compute(&store, &wanted).unwrap();
        for node in &lowering.graph.nodes {
            let mut operands = None;
            let built = allocation_counter::measure(|| {
                operands = Some(replay.operands(&store, node, &live).unwrap())
            });
            let operands = operands.unwrap();
            assert_eq!(operands.sources.capacity(), node.inputs.len());
            let leaves = operands
                .sources
                .iter()
                .map(|source| match source {
                    Source::Leaf(values) => values.capacity() * size_of::<i64>(),
                    _ => 0,
                })
                .sum::<usize>();
            let expected = node.inputs.len() * size_of::<Source<'_>>() + leaves;
            assert_eq!(usize::try_from(built.bytes_current).unwrap(), expected);
            // The positions reader owns a temporary public Vec while constructing
            // its SecretBuf copy; all other leaf readers allocate only that copy.
            let positions = if node.inputs.contains(&lowering.positions) {
                crate::engine::elements(
                    &lowering.graph.tensors[lowering.positions.0 as usize].shape,
                ) * size_of::<i64>()
            } else {
                0
            };
            assert!(usize::try_from(built.bytes_max).unwrap() <= expected + positions);
            let mut borrowed = None;
            let borrows =
                allocation_counter::measure(|| borrowed = Some(operands.all(&live).unwrap()));
            assert_eq!(borrowed.unwrap().capacity(), node.inputs.len());
            assert_eq!(
                usize::try_from(borrows.bytes_current).unwrap(),
                node.inputs.len() * size_of::<Operand<'_>>()
            );
        }
    }
}
