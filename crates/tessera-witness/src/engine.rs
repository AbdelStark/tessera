// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU reference engine: every tensor of a lowered graph, computed exactly (RFC-0018 'Witness
//! engine').
//!
//! The engine runs a graph's nodes in their topological order, each by the semantics of RFC-0006
//! 'Operation set', on values held as `i64`: [`Engine::evaluate`] from any graph's input values,
//! and [`Engine::run`] from a model's weights and a sequence's tokens, for a [`Lowering`]. Range analysis bounds every tensor inside
//! the centered field range, so `i64` holds every value, and each sum or product stays within its
//! tensor's analyzed bound, so no accumulation overflows. Gadgets compute through their tables,
//! regenerated from their specifications, and the requantization, root and reciprocal through
//! exact integer arithmetic.
//!
//! Values are secret-class (05 §3): the prompt's tokens arrive in a [`Secret`], which this file
//! reads (`secret-access.toml`), and every tensor lives in a [`SecretBuf`] that clears its memory.
//! Nothing here formats a value; errors name tensors, tables and rules.
//!
//! The matrix products split their output rows among threads. Each value is one thread's exact
//! sum in a fixed order, so the witness does not depend on the thread count.

use core::num::NonZeroUsize;
use std::thread;

use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::Lowering;
use tessera_ir::{
    Broadcast, Graph, IntRange, IntegerModel, LogitsFile, Node, Op, TableId, TensorId,
};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableKind, TableSpec};

use crate::WitnessError;

/// A table a node reads: a lookup table, or RoPE's constants.
enum Loaded {
    Lookup(GeneratedTable),
    Rope(RopeTable),
}

/// A graph ready to run: its tables regenerated from their specifications.
pub struct Engine<'a> {
    graph: &'a Graph,
    /// The lowering the graph comes from, which [`Engine::run`] binds a model and tokens to.
    lowering: Option<&'a Lowering>,
    tables: Vec<Option<Loaded>>,
    threads: NonZeroUsize,
}

impl core::fmt::Debug for Engine<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Engine")
            .field("tensors", &self.graph.tensors.len())
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl<'a> Engine<'a> {
    /// The engine of `lowering`, which regenerates every table it names, on as many threads as
    /// the machine offers.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Table`] when a table does not regenerate from its specification.
    pub fn new(lowering: &'a Lowering) -> Result<Self, WitnessError> {
        let mut engine = Self::for_graph(&lowering.graph, &lowering.tables)?;
        engine.lowering = Some(lowering);
        Ok(engine)
    }

    /// The engine of any `graph` whose nodes read the tables `tables` specifies.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Table`] when a table does not regenerate from its specification.
    pub fn for_graph(graph: &'a Graph, tables: &[TableSpec]) -> Result<Self, WitnessError> {
        let count = tables
            .iter()
            .map(|t| t.id.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut loaded_tables: Vec<Option<Loaded>> = (0..count).map(|_| None).collect();
        for spec in tables {
            let fail = |error| WitnessError::Table {
                table: spec.id,
                error,
            };
            let loaded = if matches!(spec.kind, TableKind::Rope { .. }) {
                Loaded::Rope(RopeTable::from_spec(spec).map_err(fail)?)
            } else {
                Loaded::Lookup(GeneratedTable::from_spec(spec).map_err(fail)?)
            };
            loaded_tables[spec.id.0 as usize] = Some(loaded);
        }
        let threads = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Ok(Self {
            graph,
            lowering: None,
            tables: loaded_tables,
            threads,
        })
    }

    /// The same engine on `threads` threads.
    #[must_use]
    pub fn threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// `D` on `tokens`: every tensor of the lowering, from the model's weights and channel
    /// classes. The lowering's graph is for `tokens.len()` positions.
    ///
    /// # Errors
    ///
    /// - [`WitnessError::Usage`] for an engine made without a lowering, a sequence of another
    ///   length or a token outside the vocabulary;
    /// - [`WitnessError::Model`] for a model whose weights or classes the lowering cannot take;
    /// - [`WitnessError::Range`], in debug builds, for a value outside its tensor's range;
    /// - [`WitnessError::Memory`] when a tensor cannot be allocated;
    /// - [`WitnessError::Internal`] for a key outside its table.
    pub fn run(
        &self,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
    ) -> Result<Witness, WitnessError> {
        let lowering = self.lowering.ok_or(WitnessError::Usage {
            rule: "a model runs on the engine of its lowering",
        })?;
        let mut values: Vec<Option<SecretBuf<i64>>> =
            (0..self.graph.tensors.len()).map(|_| None).collect();
        self.inputs(lowering, model, tokens, &mut values)?;
        let mut witness = self.execute(values)?;
        witness.logits = Some((lowering.logits, model.arch.vocab));
        Ok(witness)
    }

    /// Every tensor of the graph from its inputs' values: each tensor no node produces, given
    /// once, with its shape's number of values inside its declared range.
    ///
    /// # Errors
    ///
    /// - [`WitnessError::Usage`] for an input missing, given twice, of another size or outside its
    ///   range, or a tensor that is not an input;
    /// - as [`Engine::run`] otherwise.
    pub fn evaluate(
        &self,
        inputs: Vec<(TensorId, SecretBuf<i64>)>,
    ) -> Result<Witness, WitnessError> {
        let graph = self.graph;
        let mut values: Vec<Option<SecretBuf<i64>>> =
            (0..graph.tensors.len()).map(|_| None).collect();
        let produced: Vec<bool> = {
            let mut produced = vec![false; graph.tensors.len()];
            for node in &graph.nodes {
                for &t in &node.outputs {
                    produced[t.0 as usize] = true;
                }
            }
            produced
        };
        for (tensor, buffer) in inputs {
            let index = tensor.0 as usize;
            let spec = graph.tensors.get(index).ok_or(WitnessError::Usage {
                rule: "an input is a tensor of the graph",
            })?;
            if produced[index] {
                return Err(WitnessError::Usage {
                    rule: "an input is a tensor no node produces",
                });
            }
            if values[index].is_some() {
                return Err(WitnessError::Usage {
                    rule: "each input is given once",
                });
            }
            if buffer.len() != elements(&spec.shape) {
                return Err(WitnessError::Usage {
                    rule: "an input holds its shape's number of values",
                });
            }
            let range = spec.range;
            if buffer
                .iter()
                .any(|&v| i128::from(v) < range.lo || i128::from(v) > range.hi)
            {
                return Err(WitnessError::Usage {
                    rule: "every input value lies in its tensor's range",
                });
            }
            values[index] = Some(buffer);
        }
        if (0..graph.tensors.len()).any(|i| !produced[i] && values[i].is_none()) {
            return Err(WitnessError::Usage {
                rule: "every tensor no node produces is given",
            });
        }
        self.execute(values)
    }

    /// Runs every node on `values`, which hold the graph's inputs.
    fn execute(&self, mut values: Vec<Option<SecretBuf<i64>>>) -> Result<Witness, WitnessError> {
        let graph = self.graph;
        for node in &graph.nodes {
            let outputs = self.node(node, &values)?;
            for (&id, buffer) in node.outputs.iter().zip(outputs) {
                check(id, &buffer, graph.tensors[id.0 as usize].range)?;
                values[id.0 as usize] = Some(buffer);
            }
        }
        let tensors = values
            .into_iter()
            .map(|v| {
                v.ok_or(WitnessError::Internal {
                    rule: "every tensor is an input or a node's output",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Witness {
            tensors,
            logits: None,
        })
    }

    /// The tokens, the positions, every weight and every channel-class vector.
    fn inputs(
        &self,
        lowering: &Lowering,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
        values: &mut [Option<SecretBuf<i64>>],
    ) -> Result<(), WitnessError> {
        let graph = &lowering.graph;
        let n = elements(&graph.tensors[lowering.positions.0 as usize].shape);
        // The prompt is secret-class: read here, into the token tensor's buffer.
        #[allow(clippy::disallowed_methods)]
        let ids = tokens.expose_secret();
        if ids.len() != n {
            return Err(WitnessError::Usage {
                rule: "the sequence has as many tokens as the lowering's positions",
            });
        }
        if ids.iter().any(|&t| t >= model.arch.vocab) {
            return Err(WitnessError::Usage {
                rule: "every token lies inside the vocabulary",
            });
        }
        let mut token_values = zeros(lowering.tokens, n)?;
        for (slot, &t) in token_values.iter_mut().zip(ids) {
            *slot = i64::from(t);
        }
        values[lowering.tokens.0 as usize] = Some(token_values);
        let mut positions = zeros(lowering.positions, n)?;
        for (p, slot) in positions.iter_mut().enumerate() {
            *slot = p as i64;
        }
        values[lowering.positions.0 as usize] = Some(positions);
        for &(weight, tensor) in &graph.weights {
            let w = model
                .weights
                .iter()
                .find(|w| (w.ty, w.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds every weight the lowering reads",
                })?;
            let len = elements(&graph.tensors[tensor.0 as usize].shape);
            if w.data.len() != len {
                return Err(WitnessError::Model {
                    rule: "each weight has the lowering's shape",
                });
            }
            let mut buffer = zeros(tensor, len)?;
            for (slot, &x) in buffer.iter_mut().zip(&w.data) {
                *slot = i64::from(x);
            }
            values[tensor.0 as usize] = Some(buffer);
        }
        for &(weight, tensor) in &lowering.classes {
            let params = model
                .quant
                .channels
                .iter()
                .find(|c| (c.ty, c.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds the channel classes of every classed site",
                })?;
            let len = elements(&graph.tensors[tensor.0 as usize].shape);
            if params.class.len() != len {
                return Err(WitnessError::Model {
                    rule: "a channel-class vector holds one class per output channel",
                });
            }
            let mut buffer = zeros(tensor, len)?;
            for (slot, &c) in buffer.iter_mut().zip(&params.class) {
                *slot = i64::from(c);
            }
            values[tensor.0 as usize] = Some(buffer);
        }
        // RFC-0025: an affine site's multipliers, one per output channel of the weight before it.
        for &(weight, tensor) in &lowering.multipliers {
            let params = model
                .quant
                .channels
                .iter()
                .find(|c| (c.ty, c.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds the multipliers of every affine site",
                })?;
            let len = elements(&graph.tensors[tensor.0 as usize].shape);
            if params.multiplier.len() != len {
                return Err(WitnessError::Model {
                    rule: "a multiplier vector holds one multiplier per output channel",
                });
            }
            let mut buffer = zeros(tensor, len)?;
            for (slot, &m) in buffer.iter_mut().zip(&params.multiplier) {
                *slot = i64::from(m);
            }
            values[tensor.0 as usize] = Some(buffer);
        }
        Ok(())
    }

    fn table(&self, id: TableId) -> Result<&Loaded, WitnessError> {
        self.tables
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(WitnessError::Internal {
                rule: "a node names a table of the lowering",
            })
    }

    fn lookup(&self, id: TableId) -> Result<&GeneratedTable, WitnessError> {
        match self.table(id)? {
            Loaded::Lookup(t) => Ok(t),
            Loaded::Rope(_) => Err(WitnessError::Internal {
                rule: "a lookup reads a lookup table",
            }),
        }
    }

    /// The outputs of `node`, from the values of its inputs.
    fn node(
        &self,
        node: &Node,
        values: &[Option<SecretBuf<i64>>],
    ) -> Result<Vec<SecretBuf<i64>>, WitnessError> {
        let graph = self.graph;
        let input = |k: usize| -> Result<&[i64], WitnessError> {
            node.inputs
                .get(k)
                .and_then(|t| values[t.0 as usize].as_deref())
                .ok_or(WitnessError::Internal {
                    rule: "a node's inputs are computed before it",
                })
        };
        let shape = |k: usize| graph.tensors[node.inputs[k].0 as usize].shape.as_slice();
        let out = |k: usize| node.outputs[k];
        let len = |k: usize| elements(&graph.tensors[node.outputs[k].0 as usize].shape);
        let output = match &node.op {
            Op::MatMulWeight { .. } => {
                let (x, w) = (input(0)?, input(1)?);
                let ws = shape(1);
                let (rows, cols) = (ws[0] as usize, ws[1] as usize);
                let n = x.len() / cols;
                let mut y = zeros(out(0), n * rows)?;
                self.parallel(&mut y, n, rows, |r, row| {
                    let xr = &x[r * cols..(r + 1) * cols];
                    for (o, slot) in row.iter_mut().enumerate() {
                        *slot = dot(&w[o * cols..(o + 1) * cols], xr);
                    }
                });
                vec![y]
            }
            Op::Add | Op::Sub => {
                let (a, b) = (input(0)?, input(1)?);
                let sub = matches!(node.op, Op::Sub);
                let mut y = zeros(out(0), a.len())?;
                for ((slot, &p), &q) in y.iter_mut().zip(a).zip(b) {
                    *slot = if sub { p - q } else { p + q };
                }
                vec![y]
            }
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                let x = input(0)?;
                let constant = graph
                    .constants
                    .get(c.0 as usize)
                    .ok_or(WitnessError::Internal {
                        rule: "a constant the graph holds",
                    })?;
                let channels = last(shape(0));
                let mul = matches!(node.op, Op::ConstMul { .. });
                let mut y = zeros(out(0), x.len())?;
                for (i, (slot, &v)) in y.iter_mut().zip(x).enumerate() {
                    let k = if constant.len() == 1 {
                        constant[0]
                    } else {
                        constant[i % channels]
                    };
                    *slot = if mul { v * k } else { v + k };
                }
                vec![y]
            }
            Op::Rotate { rope } => {
                let Loaded::Rope(table) = self.table(*rope)? else {
                    return Err(WitnessError::Internal {
                        rule: "a rotation reads a RoPE table",
                    });
                };
                let (x, pos) = (input(0)?, input(1)?);
                let width = last(shape(0));
                let half = width / 2;
                let per_row = x.len() / shape(0)[0] as usize;
                let mut y = zeros(out(0), x.len())?;
                for (r, row) in y.chunks_mut(per_row).enumerate() {
                    let position = u32::try_from(pos[r]).map_err(|_| WitnessError::Internal {
                        rule: "a position is a u32",
                    })?;
                    let xr = &x[r * per_row..(r + 1) * per_row];
                    for (yv, xv) in row.chunks_mut(width).zip(xr.chunks(width)) {
                        for i in 0..half {
                            let (c, s) = table.constants().at(position, i as u32).ok_or(
                                WitnessError::Internal {
                                    rule: "a rotation's position and frequency are in its table",
                                },
                            )?;
                            let (a, b) = (xv[i], xv[i + half]);
                            yv[i] = a * c - b * s;
                            yv[i + half] = a * s + b * c;
                        }
                    }
                }
                vec![y]
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                let x = input(0)?;
                let (outer, length, inner) = split(shape(0), usize::from(*axis));
                let square = matches!(node.op, Op::SumSquares { .. });
                let mut y = zeros(out(0), outer * inner)?;
                for o in 0..outer {
                    for i in 0..inner {
                        let mut total = 0i64;
                        for k in 0..length {
                            let v = x[(o * length + k) * inner + i];
                            total += if square { v * v } else { v };
                        }
                        y[o * inner + i] = total;
                    }
                }
                vec![y]
            }
            Op::Reshape => {
                let x = input(0)?;
                let mut y = zeros(out(0), x.len())?;
                y.copy_from_slice(x);
                vec![y]
            }
            Op::Slice {
                axis,
                start,
                len: span,
            } => {
                let x = input(0)?;
                let (outer, length, inner) = split(shape(0), usize::from(*axis));
                let (start, span) = (*start as usize, *span as usize);
                let mut y = zeros(out(0), outer * span * inner)?;
                for o in 0..outer {
                    let from = (o * length + start) * inner;
                    y[o * span * inner..(o + 1) * span * inner]
                        .copy_from_slice(&x[from..from + span * inner]);
                }
                vec![y]
            }
            Op::GatherPublic { index } => {
                let x = input(0)?;
                let rows = graph
                    .constants
                    .get(index.0 as usize)
                    .ok_or(WitnessError::Internal {
                        rule: "an index the graph holds",
                    })?;
                let per_row = x.len() / shape(0)[0] as usize;
                let mut y = zeros(out(0), rows.len() * per_row)?;
                for (k, &r) in rows.iter().enumerate() {
                    let r = usize::try_from(r).map_err(|_| WitnessError::Internal {
                        rule: "a gathered row exists",
                    })?;
                    y[k * per_row..(k + 1) * per_row]
                        .copy_from_slice(&x[r * per_row..(r + 1) * per_row]);
                }
                vec![y]
            }
            Op::MatMulAct { transpose_rhs } => {
                let (a, b) = (input(0)?, input(1)?);
                let (sa, sb) = (shape(0), shape(1));
                let (m, k) = (sa[sa.len() - 2] as usize, last(sa));
                let n = if *transpose_rhs {
                    sb[sb.len() - 2] as usize
                } else {
                    last(sb)
                };
                let batch = a.len() / (m * k);
                let mut y = zeros(out(0), batch * m * n)?;
                let transpose = *transpose_rhs;
                self.parallel(&mut y, batch * m, n, |r, row| {
                    let (g, i) = (r / m, r % m);
                    let ar = &a[(g * m + i) * k..(g * m + i + 1) * k];
                    let bg = &b[g * k * n..(g + 1) * k * n];
                    for (j, slot) in row.iter_mut().enumerate() {
                        *slot = if transpose {
                            dot(ar, &bg[j * k..(j + 1) * k])
                        } else {
                            ar.iter().enumerate().map(|(t, &v)| v * bg[t * n + j]).sum()
                        };
                    }
                });
                vec![y]
            }
            Op::Mul { broadcast } => {
                let (a, b) = (input(0)?, input(1)?);
                let mut y = zeros(out(0), a.len())?;
                match broadcast {
                    // One value of `b` per row: `a`'s elements `[i · ratio, (i + 1) · ratio)`.
                    Broadcast::Trailing => {
                        let ratio = a.len() / b.len();
                        for (i, (slot, &v)) in y.iter_mut().zip(a).enumerate() {
                            *slot = v * b[i / ratio];
                        }
                    }
                    // One value of `b` per channel of `a`'s last axis.
                    Broadcast::Leading => {
                        for (slot, (&v, &m)) in y.iter_mut().zip(a.iter().zip(b.iter().cycle())) {
                            *slot = v * m;
                        }
                    }
                }
                vec![y]
            }
            Op::OneHot { vocab } => {
                let t = input(0)?;
                let vocab = *vocab as usize;
                let mut y = zeros(out(0), t.len() * vocab)?;
                for (i, &token) in t.iter().enumerate() {
                    let token = usize::try_from(token).ok().filter(|&v| v < vocab).ok_or(
                        WitnessError::Internal {
                            rule: "a token indexes its one-hot vector",
                        },
                    )?;
                    y[i * vocab + token] = 1;
                }
                vec![y]
            }
            Op::Requant { shift, lo, hi } => {
                let x = input(0)?;
                let shifts =
                    graph
                        .shift_classes
                        .get(shift.0 as usize)
                        .ok_or(WitnessError::Internal {
                            rule: "a shift class set the graph holds",
                        })?;
                let classes = if shifts.len() > 1 {
                    Some(input(1)?)
                } else {
                    None
                };
                let channels = last(shape(0));
                let mut y = zeros(out(0), x.len())?;
                for (i, (slot, &v)) in y.iter_mut().zip(x).enumerate() {
                    let r = shifts[class(classes, i % channels, shifts.len())?];
                    *slot = rq(v, r, *lo, *hi);
                }
                vec![y]
            }
            Op::TableMap { table } => {
                let x = input(0)?;
                let t = self.lookup(*table)?;
                let classes = if node.inputs.len() > 1 {
                    Some(input(1)?)
                } else {
                    None
                };
                let channels = last(shape(0));
                let count = match &t.spec().kind {
                    TableKind::Silu { r_by_class, .. } | TableKind::Requant { r_by_class, .. } => {
                        r_by_class.len()
                    }
                    _ => 1,
                };
                let mut y = zeros(out(0), x.len())?;
                for (i, (slot, &v)) in y.iter_mut().zip(x).enumerate() {
                    let c = class(classes, i % channels, count)?;
                    *slot = t.lookup(v, c as u8).ok_or(WitnessError::Internal {
                        rule: "a table holds every key its lookup reads",
                    })?;
                }
                vec![y]
            }
            Op::SoftmaxRow { exp_table, sigma } => {
                let s = input(0)?;
                let t = self.lookup(*exp_table)?;
                let n = last(shape(0));
                let rows = s.len() / n;
                let mut e = zeros(out(0), s.len())?;
                let mut reciprocal = zeros(out(1), rows)?;
                for r in 0..rows {
                    // Row q of its sequence sees keys 0 to q (RFC-0006 'Operation set').
                    let q = r % n;
                    let scores = &s[r * n..r * n + q + 1];
                    let top = scores.iter().copied().max().unwrap_or(0);
                    let mut total = 0i64;
                    for (i, &x) in scores.iter().enumerate() {
                        let v = t.lookup(top - x, 0).ok_or(WitnessError::Internal {
                            rule: "the exponential table holds every spread",
                        })?;
                        e[r * n + i] = v;
                        total += v;
                    }
                    if total <= 0 {
                        return Err(WitnessError::Internal {
                            rule: "a row's maximum keeps its exponential",
                        });
                    }
                    reciprocal[r] = (1i64 << (2 * u32::from(*sigma))) / total;
                }
                vec![e, reciprocal]
            }
            Op::RsqrtRow {
                kappa,
                eps_int,
                dim,
            } => {
                let u = input(0)?;
                let bound = u128::from(*dim) << (2 * u32::from(*kappa));
                let mut rho = zeros(out(0), len(0))?;
                for (slot, &v) in rho.iter_mut().zip(u) {
                    let total = u128::try_from(v)
                        .ok()
                        .and_then(|v| v.checked_add(u128::from(*eps_int)))
                        .filter(|&t| t > 0)
                        .ok_or(WitnessError::Internal {
                            rule: "a sum of squares and ε is positive",
                        })?;
                    *slot = i64::try_from((bound / total).isqrt()).map_err(|_| {
                        WitnessError::Internal {
                            rule: "a root fits 64 bits",
                        }
                    })?;
                }
                vec![rho]
            }
            Op::Bitwise { table } => {
                let (a, b) = (input(0)?, input(1)?);
                let t = self.lookup(*table)?;
                let mut y = zeros(out(0), a.len())?;
                for ((slot, &p), &q) in y.iter_mut().zip(a).zip(b) {
                    // RFC-0008: the key of a Bitwise8 lookup is a + 2^8 · b.
                    *slot = t.lookup(p + (q << 8), 0).ok_or(WitnessError::Internal {
                        rule: "a table holds every key its lookup reads",
                    })?;
                }
                vec![y]
            }
            // The scoring certificates commit nothing a later node reads (RFC-0017).
            Op::Argmax | Op::InverseCdf { .. } => Vec::new(),
        };
        Ok(output)
    }

    /// Fills `out`, `rows` rows of `cols` values, by `row(r, values)`, spreading the rows over
    /// the engine's threads in contiguous runs.
    fn parallel(
        &self,
        out: &mut [i64],
        rows: usize,
        cols: usize,
        row: impl Fn(usize, &mut [i64]) + Sync,
    ) {
        if rows == 0 || cols == 0 {
            return;
        }
        let threads = self.threads.get().min(rows);
        let per = rows.div_ceil(threads);
        if threads == 1 {
            for (r, values) in out.chunks_mut(cols).enumerate() {
                row(r, values);
            }
            return;
        }
        thread::scope(|scope| {
            for (t, block) in out.chunks_mut(per * cols).enumerate() {
                let row = &row;
                scope.spawn(move || {
                    for (k, values) in block.chunks_mut(cols).enumerate() {
                        row(t * per + k, values);
                    }
                });
            }
        });
    }
}

/// Every tensor of a graph's run, each in a buffer that clears its memory.
pub struct Witness {
    tensors: Vec<SecretBuf<i64>>,
    /// The logits' tensor and the vocabulary, for a model's run.
    logits: Option<(TensorId, u32)>,
}

impl core::fmt::Debug for Witness {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Witness")
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl Witness {
    /// The values of `tensor`, flat and row-major, lent to the code that computes on them.
    pub fn tensor(&self, tensor: TensorId) -> Option<&SecretBuf<i64>> {
        self.tensors.get(tensor.0 as usize)
    }

    /// A model's logits, one row of `vocab` values per position: the secret-class artifact
    /// `0x0602` (RFC-0003 'Auxiliary schemas'). `None` for a run of a graph without a model.
    pub fn logits(&self) -> Option<LogitsFile> {
        let (tensor, vocab) = self.logits?;
        let positions = self.tensors[tensor.0 as usize]
            .chunks((vocab as usize).max(1))
            .map(<[i64]>::to_vec)
            .collect();
        Some(LogitsFile { vocab, positions })
    }

    /// Every tensor's values, by id: a dump of secret-class data for differential tests, which a
    /// release build cannot enable (OBS-4).
    #[cfg(feature = "unsafe-debug-dump")]
    pub fn dump(&self) -> Vec<Vec<i64>> {
        self.tensors.iter().map(|t| t.to_vec()).collect()
    }
}

fn elements(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

fn last(shape: &[u32]) -> usize {
    shape.last().map_or(1, |&d| d as usize)
}

/// A shape around `axis`: the product of the axes before it, its length, and the product of the
/// axes after it.
fn split(shape: &[u32], axis: usize) -> (usize, usize, usize) {
    let outer = elements(&shape[..axis]);
    let inner = elements(&shape[axis + 1..]);
    (outer, shape[axis] as usize, inner)
}

fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(&p, &q)| p * q).sum()
}

/// RFC-0005's `rq(x, r, lo, hi) = clamp(⌊(x + 2^{r−1}) / 2^r⌋, lo, hi)`, through 128 bits.
fn rq(x: i64, r: u8, lo: i64, hi: i64) -> i64 {
    let shifted = (i128::from(x) + (1i128 << (r - 1))) >> r;
    // Clamped into [lo, hi], an i64 range.
    shifted.clamp(i128::from(lo), i128::from(hi)) as i64
}

/// The class of channel `channel`: its class vector's entry, or 0 for one class.
fn class(classes: Option<&[i64]>, channel: usize, count: usize) -> Result<usize, WitnessError> {
    let Some(classes) = classes else {
        return Ok(0);
    };
    usize::try_from(classes[channel])
        .ok()
        .filter(|&c| c < count)
        .ok_or(WitnessError::Model {
            rule: "a channel's class indexes its site's shift classes",
        })
}

/// A buffer of `len` zeros for `tensor`.
fn zeros(tensor: TensorId, len: usize) -> Result<SecretBuf<i64>, WitnessError> {
    let mut buffer =
        SecretBuf::try_with_capacity(len).map_err(|_| WitnessError::Memory { tensor })?;
    buffer.resize(len, 0);
    Ok(buffer)
}

/// INV-DM-2: every value lies in its tensor's analyzed range. Debug builds check it.
fn check(tensor: TensorId, values: &[i64], range: IntRange) -> Result<(), WitnessError> {
    if cfg!(debug_assertions)
        && values
            .iter()
            .any(|&v| i128::from(v) < range.lo || i128::from(v) > range.hi)
    {
        return Err(WitnessError::Range { tensor });
    }
    Ok(())
}
