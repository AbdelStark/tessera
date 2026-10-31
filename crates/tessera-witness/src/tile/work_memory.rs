// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conservative request traffic for any one resident/cached-source tile (RFC-0018).
use super::*;
use crate::Engine;
use tessera_compiler::{AuxColumn, RuntimePlan, Share};
use tessera_gadgets::{Requant, RsqrtRow, RsqrtTerm, SoftmaxRow, Term};
use tessera_ir::{Node, Op};

struct Work {
    bytes: u64,
    context: TensorId,
}
impl Work {
    fn new(context: TensorId) -> Self {
        Self { bytes: 0, context }
    }
    fn add(&mut self, bytes: u64) -> Result<(), WitnessError> {
        self.bytes = crate::memory::add(self.bytes, bytes, self.context)?;
        Ok(())
    }
    fn array<T>(&mut self, count: u64) -> Result<(), WitnessError> {
        self.add(crate::memory::array::<T>(count, self.context)?)
    }
    fn growing<T>(&mut self, count: u64) -> Result<(), WitnessError> {
        self.add(crate::memory::growing::<T>(count, self.context)?)
    }
    fn repeat(&mut self, n: u64, bytes: u64) -> Result<(), WitnessError> {
        self.add(crate::memory::mul(n, bytes, self.context)?)
    }
    fn gadget(&mut self, bytes: Option<u64>) -> Result<(), WitnessError> {
        self.add(bytes.ok_or(WitnessError::Memory {
            tensor: self.context,
        })?)
    }
}

fn spec(
    lowering: &tessera_compiler::Lowering,
    tensor: TensorId,
) -> Result<&tessera_ir::TensorSpec, WitnessError> {
    lowering
        .graph
        .tensors
        .get(tensor.0 as usize)
        .ok_or(WitnessError::Memory { tensor })
}
fn cells(
    lowering: &tessera_compiler::Lowering,
    tensor: TensorId,
    padded: bool,
) -> Result<u64, WitnessError> {
    spec(lowering, tensor)?
        .shape
        .iter()
        .try_fold(1u64, |n, &d| {
            let d = if padded {
                u64::from(d).next_power_of_two()
            } else {
                u64::from(d)
            };
            crate::memory::mul(n, d, tensor)
        })
}
fn first(node: &Node) -> Result<TensorId, WitnessError> {
    node.inputs.first().copied().ok_or(WitnessError::Internal {
        rule: "a model tile operation has its input",
    })
}

impl Engine<'_> {
    /// Allocation-free upper bound for non-count allocation requests of any one
    /// `Resident::new`/`from_tensors` source tile, including returned columns,
    /// target/share validation, scalar gadget work and table-padding recounts.
    /// It sums all public graph-node work: a table tile can visit every contributing
    /// node, at most once, with attention handled one row at a time. This deliberately
    /// also bounds an individual node/attention tile, including empty attention cuts.
    ///
    /// Add the canonical `MultiplicityPlan::payload_bytes` for a table tile.
    /// Borrowed model/tensors/templates/SequenceTile, source construction, arbitrary
    /// TensorSource callback allocations, replay execution, encoding, stack/runtime
    /// and allocator overhead are separate. Assumes a valid canonical lowering and
    /// source gadget parameters; it authenticates neither and admits no CPU route.
    /// Runtime certificates require [`Self::runtime_source_tile_work_memory`].
    ///
    /// # Errors
    /// `Usage` without a model lowering; `Internal` for missing operation inputs;
    /// `Memory` for missing tensor geometry or overflowing/unaddressable requests.
    pub fn source_tile_work_memory(&self) -> Result<u64, WitnessError> {
        let lowering = self.require_lowering()?;
        let context = lowering.tokens;
        let mut work = Work::new(context);
        work.array::<i64>(cells(lowering, context, false)?)?;
        work.growing::<SecretBuf<i64>>(1)?;
        work.growing::<Share>(1)?;
        work.array::<Range<u32>>(spec(lowering, context)?.shape.len() as u64)?;
        for node in &lowering.graph.nodes {
            let outputs = node.outputs.len() as u64;
            work.growing::<Operand<'_>>(node.inputs.len() as u64)?;
            work.growing::<SecretBuf<i64>>(outputs)?;
            work.growing::<Option<SecretBuf<i64>>>(outputs)?;
            work.growing::<SecretBuf<i64>>(outputs)?;
            work.array::<Option<usize>>(outputs)?;
            // A table visit creates its requested shares, then target validation
            // creates another expected set. Both sets own their coordinate ranges.
            for _ in 0..2 {
                work.growing::<Share>(outputs)?;
                for &tensor in &node.outputs {
                    work.array::<Range<u32>>(spec(lowering, tensor)?.shape.len() as u64)?;
                }
            }
            for &tensor in &node.outputs {
                work.array::<i64>(cells(lowering, tensor, false)?)?;
            }
            match node.op {
                Op::Requant { .. }
                | Op::TableMap { .. }
                | Op::Bitwise { .. }
                | Op::RsqrtRow { .. } => {
                    let input = first(node)?;
                    let rows = cells(lowering, input, false)?;
                    let (lookups, auxiliary) = match node.op {
                        Op::Requant { .. } => (5, 3),
                        Op::RsqrtRow { .. } => (189, 186),
                        _ => (1, 0),
                    };
                    work.growing::<Lookup>(lookups)?;
                    work.growing::<(AuxColumn, SecretBuf<i64>)>(auxiliary)?;
                    work.repeat(lookups, crate::memory::array::<u32>(rows, context)?)?;
                    if matches!(node.op, Op::Requant { .. } | Op::TableMap { .. })
                        && node.inputs.len() > 1
                    {
                        work.array::<u8>(rows)?;
                    }
                    if matches!(node.op, Op::Requant { .. }) {
                        work.gadget(Requant::witness_memory(rows))?;
                        let terms = crate::memory::growing::<(Term, Option<&[i64]>)>(4, context)?;
                        work.repeat(lookups, terms)?;
                    }
                    if matches!(node.op, Op::RsqrtRow { .. }) {
                        work.array::<i64>(rows)?;
                        work.array::<bool>(rows)?;
                        work.gadget(RsqrtRow::witness_memory(rows))?;
                        let terms =
                            crate::memory::growing::<(RsqrtTerm, Option<&[i64]>)>(4, context)?;
                        work.repeat(lookups, terms)?;
                    }
                    if cells(lowering, input, true)? != rows
                        && matches!(node.op, Op::Requant { .. } | Op::TableMap { .. })
                    {
                        let calls = if let Some(&classes) = node.inputs.get(1) {
                            work.array::<i64>(cells(lowering, classes, false)?)?;
                            let width = spec(lowering, input)?
                                .shape
                                .last()
                                .copied()
                                .ok_or(WitnessError::Memory { tensor: input })?;
                            u64::from(width) + 1
                        } else {
                            1
                        };
                        if matches!(node.op, Op::Requant { .. }) {
                            work.repeat(
                                calls,
                                Requant::witness_memory(1)
                                    .ok_or(WitnessError::Memory { tensor: context })?,
                            )?;
                        }
                    }
                }
                Op::SoftmaxRow { .. } => {
                    let input = first(node)?;
                    // Attention/count callbacks own a score cut as well as e/r.
                    work.array::<i64>(cells(lowering, input, false)?)?;
                    let reciprocal = node
                        .outputs
                        .get(1)
                        .copied()
                        .ok_or(WitnessError::Memory { tensor: input })?;
                    let rows = cells(lowering, reciprocal, false)?;
                    let keys = u64::from(
                        *spec(lowering, input)?
                            .shape
                            .last()
                            .ok_or(WitnessError::Memory { tensor: input })?,
                    );
                    let padded = keys
                        .checked_next_power_of_two()
                        .ok_or(WitnessError::Memory { tensor: input })?
                        .max(
                            lowering
                                .graph
                                .slots
                                .as_ref()
                                .map_or(0, |s| u64::from(s.slot_len)),
                        );
                    let mut row = Work::new(context);
                    row.gadget(SoftmaxRow::witness_memory(padded))?;
                    row.gadget(SoftmaxRow::evaluation_memory(padded))?;
                    row.array::<i64>(padded)?;
                    row.array::<i64>(97)?;
                    row.array::<u32>(crate::memory::add(keys, 99, context)?)?;
                    row.growing::<SecretBuf<i64>>(2)?;
                    row.growing::<(AuxColumn, SecretBuf<i64>)>(97)?;
                    row.growing::<Lookup>(100)?;
                    // R separate row callbacks, or one whole cut; the extra event
                    // also covers descriptor allocation when the cut has zero rows.
                    work.repeat(crate::memory::add(rows, 1, context)?, row.bytes)?;
                }
                _ => {}
            }
        }
        Ok(work.bytes)
    }

    /// [`Self::source_tile_work_memory`] plus the runtime maximum/key-column work.
    /// Counts all request key columns at their complete public padded lengths.
    /// Count backing and the other exclusions of that method remain separate.
    ///
    /// # Errors
    /// As [`Self::source_tile_work_memory`], or `Memory` on runtime size overflow.
    pub fn runtime_source_tile_work_memory(
        &self,
        runtime: &RuntimePlan,
    ) -> Result<u64, WitnessError> {
        let context = self.require_lowering()?.tokens;
        let mut work = Work {
            bytes: self.source_tile_work_memory()?,
            context,
        };
        if let Some(certificate) = runtime.certificate() {
            work.array::<i64>(u64::from(certificate.positions))?;
            let columns = runtime.key_columns().len() as u64;
            work.growing::<SecretBuf<u32>>(columns)?;
            work.growing::<Lookup>(columns)?;
            for column in runtime.key_columns() {
                let entries = 1u64
                    .checked_shl(column.variables)
                    .ok_or(WitnessError::Memory { tensor: context })?;
                work.array::<u32>(entries)?;
            }
        }
        Ok(work.bytes)
    }
}
