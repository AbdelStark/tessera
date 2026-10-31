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
//!
//! The regeneration of `crate::regenerate` runs the same nodes through [`Engine::node`], reading
//! its graph inputs one at a time through [`Engine::graph_input`]: the weights stay the model's
//! INT8 matrices, which an [`Operand::Weight`] lends, and the tokens leave their [`Secret`] here
//! alone. A layer's attention runs row by row there, through the kernels [`act_row`] and
//! [`softmax_row`] the whole-tensor operations use, so both compute every value alike.

use core::num::NonZeroUsize;
use std::thread;

use tessera_codec::{Secret, SecretBuf, Zeroize};
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

/// An input of a node as the node reads it: values held as `i64`, or a weight matrix the model
/// lends, INT8 and row-major, which only a weight matmul reads.
#[derive(Clone, Copy)]
pub(crate) enum Operand<'v> {
    /// The tensor's values, flat and row-major.
    Values(&'v [i64]),
    /// A weight's entries, the model's own.
    Weight(&'v [i8]),
}

/// A graph input of a lowering: a weight the model lends, or values read into a buffer.
pub(crate) enum Input<'m> {
    /// A weight matrix, the model's entries.
    Weight(&'m [i8]),
    /// The tokens, the positions, or a channel-class or multiplier vector.
    Values(SecretBuf<i64>),
}

/// Sealed graph input tensors (`(TensorId, SecretBuf<i64>)`) whose buffers clear on drop.
///
/// Key and token readers outside `tessera-witness` seal graph inputs inside `Secret<SecretGraphInputs>`
/// via [`SecretGraphInputs::new`] so raw keys, salts, and tokens never leave their [`Secret`] to
/// unlisted callers (05 §3 OBS-1); [`Engine::evaluate_secret`] unpacks them inside this reader.
pub struct SecretGraphInputs(Vec<(TensorId, SecretBuf<i64>)>);

impl SecretGraphInputs {
    /// Seals pre-sized `(TensorId, SecretBuf<i64>)` graph inputs inside a [`Secret`].
    pub fn new(inputs: Vec<(TensorId, SecretBuf<i64>)>) -> Secret<Self> {
        Secret::new(Self(inputs))
    }
}

impl Zeroize for SecretGraphInputs {
    fn zeroize(&mut self) {
        self.0.clear();
    }
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
    /// - [`WitnessError::Usage`] for a graph that breaks RFC-0006 'Operation set', which a
    ///   [`tessera_ir::GraphBuilder`] would have refused: the evaluation relies on its shapes;
    /// - [`WitnessError::Table`] when a table does not regenerate from its specification.
    pub fn for_graph(graph: &'a Graph, tables: &[TableSpec]) -> Result<Self, WitnessError> {
        graph.validate().map_err(|_| WitnessError::Usage {
            rule: "the graph follows RFC-0006 'Operation set'",
        })?;
        let loaded_tables = Self::load_tables(tables)?;
        let threads = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Ok(Self {
            graph,
            lowering: None,
            tables: loaded_tables,
            threads,
        })
    }

    // Keep table loading separate from graph validation and platform thread discovery:
    // its buffer bound covers this constructor, not OS/runtime allocations.
    fn load_tables(tables: &[TableSpec]) -> Result<Vec<Option<Loaded>>, WitnessError> {
        // Check registry addressability without pre-validating later specifications:
        // constructors must retain their original first-error order.
        let count = tables
            .iter()
            .map(|t| u64::from(t.id.0) + 1)
            .max()
            .unwrap_or(0);
        let count = usize::try_from(count).map_err(|_| WitnessError::MemoryTable {
            table: tables
                .iter()
                .max_by_key(|t| t.id.0)
                .map_or(TableId(0), |t| t.id),
        })?;
        let mut loaded_tables: Vec<Option<Loaded>> = Vec::new();
        loaded_tables
            .try_reserve_exact(count)
            .map_err(|_| WitnessError::MemoryTable {
                table: tables
                    .iter()
                    .max_by_key(|t| t.id.0)
                    .map_or(TableId(0), |t| t.id),
            })?;
        loaded_tables.resize_with(count, || None);
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
        Ok(loaded_tables)
    }

    /// Plan the table loader's owned buffers from public specifications, without allocation.
    /// This charges registry gaps, cloned class vectors, lookup cores, RoPE frequencies
    /// and digest scratch. Graph validation, thread runtime and allocator overhead are
    /// separate; invalid specifications still require the constructor's validation.
    ///
    /// # Errors
    /// Invalid lookup parameters, or overflowing/unaddressable buffer geometry.
    pub fn table_memory(tables: &[TableSpec]) -> Result<crate::TableMemory, WitnessError> {
        crate::TableMemory::new(tables, size_of::<Option<Loaded>>() as u64)
    }

    /// Checked owned-buffer bounds for canonical tokens, checkpoints and tensor-cache
    /// construction, derived from this engine's validated public lowering without allocation.
    /// Borrowed engine/model/graph data, thread-runtime/stack storage, tile/gadget/native
    /// work and allocator overhead remain separately charged (RFC-0018).
    ///
    /// # Errors
    /// `Usage` without a lowering or for missing tensor ids, and `Memory` on an
    /// overflowing or unaddressable buffer count. This does not admit a CPU route.
    pub fn checkpoint_memory(&self) -> Result<crate::CheckpointMemory, WitnessError> {
        crate::CheckpointMemory::new(self.require_lowering()?)
    }

    /// The same engine on `threads` threads.
    #[must_use]
    pub fn threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// The lowering, or the refusal of a model's run on an engine without one.
    pub(crate) fn require_lowering(&self) -> Result<&'a Lowering, WitnessError> {
        self.lowering.ok_or(WitnessError::Usage {
            rule: "a model runs on the engine of its lowering",
        })
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
        self.run_tokens(model, tokens, false)
    }

    /// Evaluate concatenated request sequences, inserting zero slot gaps inside the reader
    /// (RFC-0071 'The tokens'). Singleton inputs match [`Self::run`].
    ///
    /// # Errors
    /// As [`Self::run`], with `Usage` when the token count differs from the sum of real lengths.
    pub fn run_batch(
        &self,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
    ) -> Result<Witness, WitnessError> {
        self.run_tokens(model, tokens, true)
    }

    /// Retain a clearing copy of the canonical token tensor for later checkpoint replays.
    /// Input sequences concatenate real requests, as in [`Self::run_batch`]; slot gaps are
    /// inserted as zero tokens. The copy borrows nothing from the caller's secret buffer.
    ///
    /// # Errors
    /// As [`Self::run_batch`] for token count, range and allocation failures. This does not
    /// execute the model or validate a recorded continuation's greedy certificate.
    #[allow(clippy::disallowed_methods)]
    pub fn canonical_batch_tokens(
        &self,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
    ) -> Result<Secret<Vec<u32>>, WitnessError> {
        let lowering = self.require_lowering()?;
        let input = self.token_input(lowering, model, tokens, true)?;
        let mut copy = Secret::new(Vec::new());
        copy.expose_secret_mut()
            .try_reserve_exact(input.len())
            .map_err(|_| WitnessError::Memory {
                tensor: lowering.tokens,
            })?;
        for &value in input.iter() {
            copy.expose_secret_mut()
                .push(u32::try_from(value).map_err(|_| WitnessError::Internal {
                    rule: "canonical tokens are 32-bit identifiers",
                })?);
        }
        Ok(copy)
    }

    fn run_tokens(
        &self,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
        concatenated: bool,
    ) -> Result<Witness, WitnessError> {
        let lowering = self.require_lowering()?;
        let mut values: Vec<Option<SecretBuf<i64>>> =
            (0..self.graph.tensors.len()).map(|_| None).collect();
        for tensor in graph_inputs(lowering) {
            let input = if concatenated && tensor == lowering.tokens {
                Input::Values(self.token_input(lowering, model, tokens, true)?)
            } else {
                self.graph_input(lowering, model, tokens, tensor)?
            };
            let buffer = match input {
                Input::Values(buffer) => buffer,
                // The witness holds every tensor, the weights as `i64` among them.
                Input::Weight(entries) => {
                    let mut buffer = zeros(tensor, entries.len())?;
                    for (slot, &x) in buffer.iter_mut().zip(entries) {
                        *slot = i64::from(x);
                    }
                    buffer
                }
            };
            values[tensor.0 as usize] = Some(buffer);
        }
        let mut witness = self.execute(values)?;
        witness.logits = Some((lowering.logits, model.arch.vocab));
        Ok(witness)
    }

    /// Regenerate every model tensor from the original witness's token input. Weights and
    /// all other model inputs are read anew, and every node is executed again (RFC-0055 P3/P3b).
    /// The transient token copy is a clearing `Secret`, never a plain returned value.
    ///
    /// # Errors
    /// As [`Self::run`], or `Internal` if the original witness lacks its canonical token input.
    #[allow(clippy::disallowed_methods)]
    pub fn regenerate(
        &self,
        model: &IntegerModel,
        original: &Witness,
    ) -> Result<Witness, WitnessError> {
        let lowering = self.require_lowering()?;
        let input = original
            .tensor(lowering.tokens)
            .ok_or(WitnessError::Internal {
                rule: "regeneration has the original token input",
            })?;
        let mut tokens = Secret::new(Vec::new());
        tokens
            .expose_secret_mut()
            .try_reserve_exact(input.len())
            .map_err(|_| WitnessError::Memory {
                tensor: lowering.tokens,
            })?;
        for &value in input.iter() {
            tokens
                .expose_secret_mut()
                .push(u32::try_from(value).map_err(|_| WitnessError::Internal {
                    rule: "a regenerated token is a canonical 32-bit identifier",
                })?);
        }
        self.run(model, &tokens)
    }

    /// Evaluates the graph on sealed [`SecretGraphInputs`], keeping raw keys, salts and tokens
    /// inside their [`Secret`] across crate boundaries (05 §3 OBS-1).
    ///
    /// # Errors
    /// Same as [`Engine::evaluate`].
    pub fn evaluate_secret(
        &self,
        mut inputs: Secret<SecretGraphInputs>,
    ) -> Result<Witness, WitnessError> {
        #[allow(clippy::disallowed_methods)]
        let pairs = core::mem::take(&mut inputs.expose_secret_mut().0);
        self.evaluate(pairs)
    }

    /// Every tensor of the graph from its inputs' values: each tensor no node produces, given
    /// once, with its shape's number of values inside its declared range.
    ///
    /// # Errors
    ///
    /// - [`WitnessError::Usage`] for an input missing, given twice, of another size or outside its
    ///   range, or a tensor that is not an input; and for an integer outside its `BitDecompose`'s
    ///   width or a `BitFn` input that is not a bit, which range analysis rules out (RFC-0066);
    /// - [`WitnessError::Unequal`] for an `AssertEqual` whose two tensors differ;
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
            let outputs = {
                let operands = node
                    .inputs
                    .iter()
                    .map(|t| {
                        values
                            .get(t.0 as usize)
                            .and_then(Option::as_deref)
                            .map(Operand::Values)
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or(WitnessError::Internal {
                        rule: "a node's inputs are computed before it",
                    })?;
                self.node(node, &operands)?
            };
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

    /// The prompt's tokens, read out of their [`Secret`] into the token tensor's buffer after
    /// checking that there are as many as the lowering's positions, each inside the vocabulary.
    /// The one place the crate reads them (`secret-access.toml`).
    pub(crate) fn tokens(
        &self,
        lowering: &Lowering,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
    ) -> Result<SecretBuf<i64>, WitnessError> {
        self.token_input(lowering, model, tokens, false)
    }

    fn token_input(
        &self,
        lowering: &Lowering,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
        concatenated: bool,
    ) -> Result<SecretBuf<i64>, WitnessError> {
        let n = elements(&lowering.graph.tensors[lowering.positions.0 as usize].shape);
        // The prompt is secret-class: read here, into the token tensor's buffer.
        #[allow(clippy::disallowed_methods)]
        let ids = tokens.expose_secret();
        if concatenated && let Some(slots) = &lowering.graph.slots {
            let real = slots
                .lengths
                .iter()
                .try_fold(0usize, |sum, &len| sum.checked_add(len as usize))
                .ok_or(WitnessError::Memory {
                    tensor: lowering.tokens,
                })?;
            if ids.len() != real || ids.iter().any(|&t| t >= model.arch.vocab) {
                return Err(WitnessError::Usage {
                    rule: "concatenated batch tokens match real request lengths and vocabulary",
                });
            }
            let mut values = zeros(lowering.tokens, n)?;
            let mut start = 0;
            for (r, &len) in slots.lengths.iter().enumerate() {
                let len = len as usize;
                for (dst, &token) in values
                    [r * slots.slot_len as usize..r * slots.slot_len as usize + len]
                    .iter_mut()
                    .zip(&ids[start..start + len])
                {
                    *dst = i64::from(token);
                }
                start += len;
            }
            return Ok(values);
        }
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
        if let Some(slots) = &lowering.graph.slots
            && (0..n).any(|tau| !slots.is_real_row(tau) && ids[tau] != 0)
        {
            return Err(WitnessError::Usage {
                rule: "every padded slot position holds token 0 (RFC-0071)",
            });
        }
        let mut values = zeros(lowering.tokens, n)?;
        for (slot, &t) in values.iter_mut().zip(ids) {
            *slot = i64::from(t);
        }
        Ok(values)
    }

    /// The value of `tensor`, a graph input of `lowering`, for `model` on `tokens`: the tokens,
    /// the positions `0` to `n − 1`, a weight the model lends as it holds it, or one of its
    /// channel-class or multiplier vectors (RFC-0025) as `i64`.
    ///
    /// # Errors
    ///
    /// As [`Engine::run`] for the tokens and the model; [`WitnessError::Internal`] for a tensor
    /// that is none of these.
    pub(crate) fn graph_input<'m>(
        &self,
        lowering: &Lowering,
        model: &'m IntegerModel,
        tokens: &Secret<Vec<u32>>,
        tensor: TensorId,
    ) -> Result<Input<'m>, WitnessError> {
        let graph = &lowering.graph;
        let len = graph
            .tensors
            .get(tensor.0 as usize)
            .map(|spec| elements(&spec.shape))
            .ok_or(WitnessError::Internal {
                rule: "a graph input is a tensor of the graph",
            })?;
        if tensor == lowering.tokens {
            return self.tokens(lowering, model, tokens).map(Input::Values);
        }
        if tensor == lowering.positions {
            let vals = tessera_compiler::positions_values(lowering);
            if vals.len() != len {
                return Err(WitnessError::Internal {
                    rule: "positions_values matches the lowering's positions shape",
                });
            }
            let mut positions = zeros(tensor, len)?;
            positions.copy_from_slice(&vals);
            return Ok(Input::Values(positions));
        }
        if let Some(&(weight, _)) = graph.weights.iter().find(|&&(_, t)| t == tensor) {
            let w = model
                .weights
                .iter()
                .find(|w| (w.ty, w.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds every weight the lowering reads",
                })?;
            if w.data.len() != len {
                return Err(WitnessError::Model {
                    rule: "each weight has the lowering's shape",
                });
            }
            return Ok(Input::Weight(&w.data));
        }
        if let Some(&(weight, _)) = lowering.classes.iter().find(|&&(_, t)| t == tensor) {
            let params = model
                .quant
                .channels
                .iter()
                .find(|c| (c.ty, c.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds the channel classes of every classed site",
                })?;
            if params.class.len() != len {
                return Err(WitnessError::Model {
                    rule: "a channel-class vector holds one class per output channel",
                });
            }
            let mut buffer = zeros(tensor, len)?;
            for (slot, &c) in buffer.iter_mut().zip(&params.class) {
                *slot = i64::from(c);
            }
            return Ok(Input::Values(buffer));
        }
        // RFC-0025: an affine site's multipliers, one per output channel of the weight before it.
        if let Some(&(weight, _)) = lowering.multipliers.iter().find(|&&(_, t)| t == tensor) {
            let params = model
                .quant
                .channels
                .iter()
                .find(|c| (c.ty, c.layer) == (weight.ty, weight.layer))
                .ok_or(WitnessError::Model {
                    rule: "the model holds the multipliers of every affine site",
                })?;
            if params.multiplier.len() != len {
                return Err(WitnessError::Model {
                    rule: "a multiplier vector holds one multiplier per output channel",
                });
            }
            let mut buffer = zeros(tensor, len)?;
            for (slot, &m) in buffer.iter_mut().zip(&params.multiplier) {
                *slot = i64::from(m);
            }
            return Ok(Input::Values(buffer));
        }
        Err(WitnessError::Internal {
            rule: "a graph input is the tokens, the positions, a weight or a channel vector",
        })
    }

    fn table(&self, id: TableId) -> Result<&Loaded, WitnessError> {
        self.tables
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(WitnessError::Internal {
                rule: "a node names a table of the lowering",
            })
    }

    /// The lookup table `id`, regenerated from its specification.
    pub(crate) fn lookup(&self, id: TableId) -> Result<&GeneratedTable, WitnessError> {
        match self.table(id)? {
            Loaded::Lookup(t) => Ok(t),
            Loaded::Rope(_) => Err(WitnessError::Internal {
                rule: "a lookup reads a lookup table",
            }),
        }
    }

    /// `W x` over the last axis of `x` for a weight `w` of `rows × cols`, held as any integers.
    fn matmul<W: Copy + Into<i64> + Sync>(
        &self,
        x: &[i64],
        w: &[W],
        (rows, cols): (usize, usize),
        out: TensorId,
    ) -> Result<SecretBuf<i64>, WitnessError> {
        let n = x.len() / cols;
        let mut y = zeros(out, n * rows)?;
        self.parallel(&mut y, n, rows, |r, row| {
            let xr = &x[r * cols..(r + 1) * cols];
            for (o, slot) in row.iter_mut().enumerate() {
                *slot = dot(&w[o * cols..(o + 1) * cols], xr);
            }
        });
        Ok(y)
    }

    /// The outputs of `node`, from its inputs `inputs`, in its inputs' order.
    pub(crate) fn node(
        &self,
        node: &Node,
        inputs: &[Operand<'_>],
    ) -> Result<Vec<SecretBuf<i64>>, WitnessError> {
        let graph = self.graph;
        let missing = WitnessError::Internal {
            rule: "a node's inputs are computed before it",
        };
        let input = |k: usize| -> Result<&[i64], WitnessError> {
            match inputs.get(k) {
                Some(Operand::Values(values)) => Ok(*values),
                Some(Operand::Weight(_)) => Err(WitnessError::Internal {
                    rule: "only a weight matmul reads a weight's entries",
                }),
                None => Err(missing),
            }
        };
        let shape = |k: usize| graph.tensors[node.inputs[k].0 as usize].shape.as_slice();
        let out = |k: usize| node.outputs[k];
        let len = |k: usize| elements(&graph.tensors[node.outputs[k].0 as usize].shape);
        let output = match &node.op {
            Op::MatMulWeight { .. } => {
                let x = input(0)?;
                let ws = shape(1);
                let dims = (ws[0] as usize, ws[1] as usize);
                let y = match inputs.get(1) {
                    Some(Operand::Values(w)) => self.matmul(x, w, dims, out(0))?,
                    Some(Operand::Weight(w)) => self.matmul(x, w, dims, out(0))?,
                    None => return Err(missing),
                };
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
                let slotted = graph
                    .slots
                    .as_ref()
                    .filter(|s| shape(0).first() == Some(&s.token_len()));
                let mut y = zeros(out(0), x.len())?;
                for (r, row) in y.chunks_mut(per_row).enumerate() {
                    if slotted.is_some_and(|s| !s.is_real_row(r)) {
                        continue;
                    }
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
            Op::Transpose { perm } => {
                let x = input(0)?;
                let dims: Vec<usize> = shape(0).iter().map(|&d| d as usize).collect();
                // The input's stride of each axis, row-major, and the output's axes in order.
                let mut strides = vec![1usize; dims.len()];
                for a in (0..dims.len().saturating_sub(1)).rev() {
                    strides[a] = strides[a + 1] * dims[a + 1];
                }
                let axes: Vec<(usize, usize)> = perm
                    .iter()
                    .map(|&a| (dims[usize::from(a)], strides[usize::from(a)]))
                    .collect();
                let mut y = zeros(out(0), x.len())?;
                // Walk the output in row-major order, the input's position following it.
                let mut index = vec![0usize; axes.len()];
                let mut from = 0usize;
                for value in y.iter_mut() {
                    *value = x[from];
                    for k in (0..axes.len()).rev() {
                        let (length, stride) = axes[k];
                        index[k] += 1;
                        from += stride;
                        if index[k] < length {
                            break;
                        }
                        from -= length * stride;
                        index[k] = 0;
                    }
                }
                vec![y]
            }
            Op::Expand { axis, len: copies } => {
                let x = input(0)?;
                let dims = shape(0);
                let axis = usize::from(*axis);
                let inner: usize = dims[axis..].iter().map(|&d| d as usize).product();
                let copies = *copies as usize;
                let mut y = zeros(out(0), x.len() * copies)?;
                for (o, row) in x.chunks_exact(inner.max(1)).enumerate() {
                    for j in 0..copies {
                        let at = (o * copies + j) * inner;
                        y[at..at + inner].copy_from_slice(row);
                    }
                }
                vec![y]
            }
            Op::MatMulAct { transpose_rhs } => {
                let (a, b) = (input(0)?, input(1)?);
                let (sa, sb) = (shape(0), shape(1));
                let (m, k) = (sa[sa.len() - 2] as usize, last(sa));
                let n = last(&graph.tensors[out(0).0 as usize].shape);
                let transpose = *transpose_rhs;
                let b_rows = sb[sb.len() - 2] as usize;
                let slotted = graph.slots.as_ref().filter(|s| {
                    let tok = s.token_len() as usize;
                    m == tok
                        && b_rows == tok
                        && ((transpose && n != b_rows) || (!transpose && k != b_rows))
                });
                let batch = a.len() / (m * k);
                let mut y = zeros(out(0), batch * m * n)?;
                if let Some(slots) = slotted {
                    let slot_len = slots.slot_len as usize;
                    self.parallel(&mut y, batch * m, n, |r, row| {
                        let (g, i) = (r / m, r % m);
                        let ar = &a[(g * m + i) * k..(g * m + i + 1) * k];
                        let slot_base = (i / slot_len) * slot_len;
                        let bg = if transpose {
                            &b[(g * m + slot_base) * k..(g * m + slot_base + n) * k]
                        } else {
                            &b[(g * m + slot_base) * n..(g * m + slot_base + k) * n]
                        };
                        act_row(ar, bg, (k, n), transpose, row);
                    });
                } else {
                    self.parallel(&mut y, batch * m, n, |r, row| {
                        let (g, i) = (r / m, r % m);
                        let ar = &a[(g * m + i) * k..(g * m + i + 1) * k];
                        let bg = &b[g * k * n..(g + 1) * k * n];
                        act_row(ar, bg, (k, n), transpose, row);
                    });
                }
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
                let slotted = graph.slots.as_ref().filter(|s| shape(0) == [s.token_len()]);
                let mut y = zeros(out(0), t.len() * vocab)?;
                for (i, &token) in t.iter().enumerate() {
                    if slotted.is_some_and(|s| !s.is_real_row(i)) {
                        if token != 0 {
                            return Err(WitnessError::Internal {
                                rule: "a padded slot row holds token 0",
                            });
                        }
                        continue;
                    }
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
                let in_shape = shape(0);
                let n = last(in_shape);
                let seq_rows = in_shape
                    .get(in_shape.len().saturating_sub(2))
                    .copied()
                    .unwrap_or(1) as usize;
                let slotted = graph
                    .slots
                    .as_ref()
                    .filter(|sl| seq_rows == sl.token_len() as usize && n == sl.n_max() as usize);
                let rows = s.len() / n;
                let mut e = zeros(out(0), s.len())?;
                let mut reciprocal = zeros(out(1), rows)?;
                for (r, slot) in reciprocal.iter_mut().enumerate() {
                    let q_row = r % seq_rows;
                    if let Some(slots) = slotted {
                        if !slots.is_real_row(q_row) {
                            continue;
                        }
                        let row = r * n..(r + 1) * n;
                        let q_pos = q_row % (slots.slot_len as usize);
                        *slot = softmax_row(t, *sigma, &s[row.clone()], q_pos, &mut e[row])?;
                    } else {
                        // Row q of its sequence sees keys 0 to q (RFC-0006 'Operation set').
                        let row = r * n..(r + 1) * n;
                        *slot = softmax_row(t, *sigma, &s[row.clone()], q_row, &mut e[row])?;
                    }
                }
                vec![e, reciprocal]
            }
            Op::RsqrtRow {
                kappa,
                eps_int,
                dim,
            } => {
                let u = input(0)?;
                let slotted = graph
                    .slots
                    .as_ref()
                    .filter(|s| shape(0).first() == Some(&s.token_len()));
                let per_tok = slotted.map_or(1, |s| (u.len() / s.token_len() as usize).max(1));
                let bound = u128::from(*dim) << (2 * u32::from(*kappa));
                let mut rho = zeros(out(0), len(0))?;
                for (i, (slot, &v)) in rho.iter_mut().zip(u).enumerate() {
                    if slotted.is_some_and(|s| !s.is_real_row(i / per_tok)) {
                        continue;
                    }
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
            Op::BitDecompose { low, high } => {
                let x = input(0)?;
                let (low, high) = (usize::from(*low), usize::from(*high));
                let top = 1i64 << (low + high);
                if x.iter().any(|&v| !(0..top).contains(&v)) {
                    // Only a malformed input can fail, as a committed byte of value 256 would.
                    return Err(WitnessError::Usage {
                        rule: "a decomposed integer lies in [0, 2^{low + high})",
                    });
                }
                let mut b = zeros(out(0), x.len() * low)?;
                for (bits, &v) in b.chunks_exact_mut(low).zip(x) {
                    for (i, bit) in bits.iter_mut().enumerate() {
                        *bit = (v >> i) & 1;
                    }
                }
                let mut outputs = Vec::with_capacity(1 + usize::from(high > 0));
                outputs.push(b);
                if high > 0 {
                    let mut h = zeros(out(1), x.len() * high)?;
                    for (bits, &v) in h.chunks_exact_mut(high).zip(x) {
                        for (j, bit) in bits.iter_mut().enumerate() {
                            *bit = (v >> (low + j)) & 1;
                        }
                    }
                    outputs.push(h);
                }
                outputs
            }
            Op::BitFn { f } => {
                let mut ins = Vec::with_capacity(f.arity());
                for index in 0..f.arity() {
                    ins.push(input(index)?);
                }
                // Range analysis puts them in [0, 1]; an unanalyzed graph could overflow `eval`.
                if ins.iter().any(|v| v.iter().any(|&b| b != 0 && b != 1)) {
                    return Err(WitnessError::Usage {
                        rule: "a Boolean function's inputs are bits",
                    });
                }
                let mut y = zeros(out(0), ins[0].len())?;
                let mut at = [0i64; 3];
                for (k, slot) in y.iter_mut().enumerate() {
                    for (a, values) in at.iter_mut().zip(&ins) {
                        *a = values[k];
                    }
                    *slot = f.eval(&at[..f.arity()]);
                }
                vec![y]
            }
            Op::AssertEqual => {
                if input(0)? != input(1)? {
                    return Err(WitnessError::Unequal { node: node.id });
                }
                Vec::new()
            }
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

    /// The number of tensors it holds: all of its graph's.
    pub(crate) fn len(&self) -> usize {
        self.tensors.len()
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

/// The graph inputs of `lowering` in the order a run reads them, which fixes the order of its
/// refusals: the tokens, the positions, the weights in declaration order, then the channel-class
/// and multiplier vectors.
pub(crate) fn graph_inputs(lowering: &Lowering) -> impl Iterator<Item = TensorId> + '_ {
    [lowering.tokens, lowering.positions]
        .into_iter()
        .chain(lowering.graph.weights.iter().map(|&(_, t)| t))
        .chain(lowering.classes.iter().map(|&(_, t)| t))
        .chain(lowering.multipliers.iter().map(|&(_, t)| t))
}

pub(crate) fn elements(shape: &[u32]) -> usize {
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

/// `Σ a_i · b_i` in order, `a` held as any integers: range analysis bounds every partial sum.
fn dot<W: Copy + Into<i64>>(a: &[W], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(&p, &q)| p.into() * q).sum()
}

/// One output row of `MatMulAct`: row `ar` of a batch's left operand, of `k` values, by the
/// batch's right operand `bg`, into the `n` values of `row`. With `transpose`, `bg` is `[n, k]`
/// and enters transposed, as `K` does in `QK^T`; otherwise it is `[k, n]`, as `V` in `PV`.
pub(crate) fn act_row(
    ar: &[i64],
    bg: &[i64],
    (k, n): (usize, usize),
    transpose: bool,
    row: &mut [i64],
) {
    for (j, slot) in row.iter_mut().enumerate() {
        *slot = if transpose {
            dot(ar, &bg[j * k..(j + 1) * k])
        } else {
            ar.iter().enumerate().map(|(t, &v)| v * bg[t * n + j]).sum()
        };
    }
}

/// The maximum of query `q`'s valid scores, `scores[0..=q]`: the max certificate's `c_q`
/// (RFC-0009 'SoftmaxRow'). `q` is below the row's length.
pub(crate) fn row_max(scores: &[i64], q: usize) -> i64 {
    scores[..=q].iter().copied().max().unwrap_or(0)
}

/// One row of a causal `SoftmaxRow`, the scores `scores` of query `q` against every key: the
/// exponentials `Exp[c_q − x_qi]` of the keys `i ≤ q` into `e`, whose other entries stay as given,
/// zero, and the reciprocal `⌊2^{2σ} / Σ_i e_qi⌋` (RFC-0005, RFC-0009).
///
/// # Errors
///
/// [`WitnessError::Internal`] for a spread outside the table, or a row whose maximum does not keep
/// its exponential.
pub(crate) fn softmax_row(
    table: &GeneratedTable,
    sigma: u8,
    scores: &[i64],
    q: usize,
    e: &mut [i64],
) -> Result<i64, WitnessError> {
    let top = row_max(scores, q);
    let mut total = 0i64;
    for (slot, &x) in e.iter_mut().zip(&scores[..=q]) {
        let v = table.lookup(top - x, 0).ok_or(WitnessError::Internal {
            rule: "the exponential table holds every spread",
        })?;
        *slot = v;
        total += v;
    }
    if total <= 0 {
        return Err(WitnessError::Internal {
            rule: "a row's maximum keeps its exponential",
        });
    }
    Ok((1i64 << (2 * u32::from(sigma))) / total)
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
pub(crate) fn zeros(tensor: TensorId, len: usize) -> Result<SecretBuf<i64>, WitnessError> {
    let mut buffer =
        SecretBuf::try_with_capacity(len).map_err(|_| WitnessError::Memory { tensor })?;
    buffer.resize(len, 0);
    Ok(buffer)
}

/// INV-DM-2: every value lies in its tensor's analyzed range. Debug builds check it.
pub(crate) fn check(tensor: TensorId, values: &[i64], range: IntRange) -> Result<(), WitnessError> {
    if cfg!(debug_assertions)
        && values
            .iter()
            .any(|&v| i128::from(v) < range.lo || i128::from(v) > range.hi)
    {
        return Err(WitnessError::Range { tensor });
    }
    Ok(())
}

#[cfg(test)]
mod window_regeneration_tests {
    use super::*;
    use tessera_codec::{DecodeLimits, envelope, schema};

    #[test]
    fn regeneration_reruns_nodes_and_reloads_model_inputs_instead_of_copying_a_witness() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("models.json")).unwrap()).unwrap();
        let entry = index["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == "tiny-2-servable")
            .unwrap();
        let model: IntegerModel = envelope::open(
            &std::fs::read(root.join(entry["file"].as_str().unwrap())).unwrap(),
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        )
        .unwrap();
        let lowering =
            tessera_compiler::lower(tessera_compiler::PublicModel::of(&model), 3).unwrap();
        let engine = Engine::new(&lowering).unwrap();
        let tokens = Secret::new(vec![1, 2, 3]);
        let expected = engine.run(&model, &tokens).unwrap();
        let mut poisoned = engine.run(&model, &tokens).unwrap();
        for (i, values) in poisoned.tensors.iter_mut().enumerate() {
            if i != lowering.tokens.0 as usize {
                values.fill(i64::MAX);
            }
        }
        let regenerated = engine.regenerate(&model, &poisoned).unwrap();
        assert_eq!(regenerated.tensors.len(), expected.tensors.len());
        for (i, (actual, expected)) in regenerated
            .tensors
            .iter()
            .zip(&expected.tensors)
            .enumerate()
        {
            assert!(
                actual[..] == expected[..],
                "regenerated tensor {i} must be recomputed"
            );
        }
    }
}

#[cfg(test)]
mod table_memory_tests {
    use super::*;
    use tessera_codec::Rational;

    fn rope() -> TableKind {
        TableKind::Rope {
            theta: 500_000,
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max: 8192,
            bits: 16,
            max_context: 3,
            d_head: 6,
        }
    }
    fn spec(id: u32, kind: TableKind) -> TableSpec {
        if matches!(kind, TableKind::Rope { .. }) {
            RopeTable::new(TableId(id), kind).unwrap().spec().clone()
        } else {
            GeneratedTable::new(TableId(id), kind)
                .unwrap()
                .spec()
                .clone()
        }
    }

    #[test]
    fn table_loader_bounds_cover_gaps_classes_rope_digest_and_replacement() {
        let range = spec(0, TableKind::Range { bits: 32 });
        let exp = spec(
            0,
            TableKind::Exp {
                key_bits: 8,
                sigma: 4,
                scale: Rational::new(1, 1).unwrap(),
            },
        );
        let exp_values = GeneratedTable::from_spec(&exp)
            .unwrap()
            .storage_bytes()
            .unwrap()
            - size_of::<GeneratedTable>();
        let silu = spec(
            3,
            TableKind::Silu {
                b: 8,
                r_by_class: vec![1, 3, 6],
                in_exp: -10,
                out_exp: -4,
            },
        );
        let rq = spec(
            7,
            TableKind::Requant {
                b: 8,
                r_by_class: vec![1, 3],
                lo: -128,
                hi: 127,
            },
        );
        // The caller's spare class capacity must not be cloned or charged as loaded storage.
        let mut rq = rq;
        if let TableKind::Requant { r_by_class, .. } = &mut rq.kind {
            r_by_class.reserve_exact(100);
        }
        let rope = spec(1, rope());
        let mut old = exp.clone();
        old.id = TableId(1);
        old = spec(1, old.kind);
        for (tables, retained, exact, extra) in [
            (vec![], 0, true, 0),
            (
                vec![spec(
                    0,
                    TableKind::Exp2Frac {
                        frac_bits: 8,
                        out_bits: 8,
                    },
                )],
                size_of::<Option<Loaded>>() + 256 * 8,
                true,
                0,
            ),
            (
                vec![spec(0, TableKind::Pow2 { max_exp: 63 })],
                size_of::<Option<Loaded>>(),
                true,
                0,
            ),
            (
                vec![spec(0, TableKind::AesSbox)],
                size_of::<Option<Loaded>>(),
                true,
                0,
            ),
            (
                vec![spec(0, TableKind::Gf256Mul { c: 3 })],
                size_of::<Option<Loaded>>(),
                true,
                0,
            ),
            (
                vec![spec(
                    0,
                    TableKind::Bitwise8 {
                        op: tessera_tables::BitOp::Xor,
                    },
                )],
                size_of::<Option<Loaded>>(),
                true,
                0,
            ),
            (vec![range], size_of::<Option<Loaded>>(), true, 0),
            (vec![exp], size_of::<Option<Loaded>>() + exp_values, true, 8),
            (
                vec![rope.clone()],
                2 * size_of::<Option<Loaded>>() + 3 * 16,
                true,
                0,
            ),
            (
                vec![silu, rq, rope.clone()],
                8 * size_of::<Option<Loaded>>() + (128 + 32 + 4) * 8 + 3 + 2 + 3 * 16,
                true,
                0,
            ),
            (
                vec![old, rope],
                2 * size_of::<Option<Loaded>>() + exp_values + 3 * 16,
                false,
                8,
            ),
        ] {
            let mut plan = None;
            let planning =
                allocation_counter::measure(|| plan = Some(Engine::table_memory(&tables).unwrap()));
            assert_eq!(planning.count_total, 0);
            let plan = plan.unwrap();
            assert_eq!(plan.retained_bytes(), retained as u64);
            assert_eq!(plan.peak_bytes(), (retained + extra) as u64);
            let mut loaded = None;
            let measured = allocation_counter::measure(|| {
                loaded = Some(Engine::load_tables(&tables).unwrap())
            });
            let loaded = loaded.unwrap();
            assert_eq!(loaded.capacity(), loaded.len());
            assert!(u64::try_from(measured.bytes_current).unwrap() <= plan.retained_bytes());
            assert!(
                measured.bytes_max <= plan.peak_bytes(),
                "{measured:?} {plan:?}"
            );
            if exact {
                assert_eq!(
                    u64::try_from(measured.bytes_current).unwrap(),
                    plan.retained_bytes()
                );
            }
            // Single Exp pins the digest piece while its complete core is live.
            if tables.len() == 1 && extra == 8 {
                assert_eq!(measured.bytes_max, plan.peak_bytes());
            }
        }
    }

    #[test]
    fn table_loading_keeps_the_first_constructor_error_before_later_parameter_errors() {
        let mut first = spec(0, TableKind::Range { bits: 8 });
        first.digest[0] ^= 1;
        let later = TableSpec {
            id: TableId(1),
            kind: TableKind::Range { bits: 0 },
            log2_rows: 0,
            digest: [0; 32],
        };
        assert!(matches!(
            Engine::load_tables(&[first, later]),
            Err(WitnessError::Table {
                table: TableId(0),
                error: tessera_tables::TableError::Digest
            })
        ));
    }

    #[test]
    fn large_table_geometry_and_invalid_lookup_are_planned_without_allocating() {
        let mut large_rope = rope();
        if let TableKind::Rope {
            d_head,
            max_context,
            ..
        } = &mut large_rope
        {
            *d_head = u16::MAX - 1;
            *max_context = 1;
        }
        let tables = [TableSpec {
            id: TableId(9),
            kind: large_rope,
            log2_rows: 24,
            digest: [0; 32],
        }];
        let measured = allocation_counter::measure(|| {
            let plan = Engine::table_memory(&tables).unwrap();
            assert_eq!(
                plan.retained_bytes(),
                (10 * size_of::<Option<Loaded>>() + usize::from(u16::MAX / 2) * 16) as u64
            );
            assert_eq!(plan.peak_bytes(), plan.retained_bytes());
        });
        assert_eq!(measured.count_total, 0);
        let bad = [TableSpec {
            id: TableId(2),
            kind: TableKind::Range { bits: 0 },
            log2_rows: 0,
            digest: [0; 32],
        }];
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                Engine::table_memory(&bad),
                Err(WitnessError::Table {
                    table: TableId(2),
                    error: tessera_tables::TableError::Parameter {
                        kind: "Range",
                        rule: "1 ≤ bits ≤ 32"
                    }
                })
            )
        });
        assert_eq!(measured.count_total, 0);
    }
}
