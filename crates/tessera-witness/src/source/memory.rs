// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public source construction and validation charges, separate from tile work (RFC-0018).
use super::*;

/// Requested-byte upper bound for source metadata construction. Each planner
/// documents its covered constructor work; runtime setup and transferred groups
/// are included only when requested explicitly. Sums of allocation requests bound
/// peak and retained heap. Borrowed inputs, replay execution, returned tiles and
/// allocator/runtime overhead remain separate. This is not an admission check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceMetadataMemory {
    bytes: u64,
}
impl SourceMetadataMemory {
    /// Upper bound on all structure/gadget metadata allocation requests.
    pub fn bytes(self) -> u64 {
        self.bytes
    }
}
impl Engine<'_> {
    /// Derive source structure/gadget construction bytes from public lowering and
    /// selection, without allocating or reading model/token/witness values.
    /// Covers `Resident::new`/`from_tensors` metadata, excluding borrower-specific
    /// validation work; a caller-supplied TensorSource may itself allocate on reads.
    /// Runtime/regenerator/grouping and tile work require the separate charges above.
    ///
    /// # Errors
    /// `Usage` without a model lowering; `Internal`/`Gadget` for inconsistent public
    /// metadata, `Memory` for unaddressable or overflowing requested counts.
    pub fn source_metadata_memory(
        &self,
        selection: &Selection,
    ) -> Result<SourceMetadataMemory, WitnessError> {
        let lowering = self.require_lowering()?;
        let structure = Structure::construction_bytes(lowering)?;
        let gadgets = Gadgets::construction_bytes(&lowering.graph, selection, lowering.tokens)?;
        Ok(SourceMetadataMemory {
            bytes: crate::memory::add(structure, gadgets, lowering.tokens)?,
        })
    }
}

impl SourceMetadataMemory {
    /// Include one transfer of a caller-owned grouped-column vector and the
    /// temporary sorted table-id copy. `capacity` includes spare slots; `len`
    /// counts initialized entries. Previous source allocations remain charged.
    /// Returns `None` if `len > capacity` or any byte count is unaddressable.
    /// Repeated grouping calls need a separate charge for each transfer.
    pub fn with_grouped_columns(self, len: u64, capacity: u64) -> Option<Self> {
        let array = |n: u64, size: usize| {
            let bytes = n.checked_mul(size as u64)?;
            (usize::try_from(n).is_ok() && bytes <= isize::MAX as u64).then_some(bytes)
        };
        if len > capacity {
            return None;
        }
        Some(Self {
            bytes: self
                .bytes
                .checked_add(array(capacity, size_of::<(TableId, u64)>())?)?
                .checked_add(array(len, size_of::<TableId>())?)?,
        })
    }
}

impl Engine<'_> {
    /// Source metadata constructor bound with exact-greedy runtime setup:
    /// [`Self::source_metadata_memory`] plus public range validation, read-vector
    /// growth and copied key-column descriptors. No values are read or allocated.
    /// Regenerator graph-input validation, transferred groups, borrowed inputs,
    /// returned tiles and allocator/runtime overhead remain separate charges.
    ///
    /// # Errors
    /// As [`Self::source_metadata_memory`]; `Memory` for an unaddressable analysis
    /// or metadata allocation. Does not authenticate a runtime plan or admit a route.
    pub fn runtime_source_metadata_memory(
        &self,
        runtime: &RuntimePlan,
    ) -> Result<SourceMetadataMemory, WitnessError> {
        use crate::memory::{add, array, growing};
        let mut bound = self.source_metadata_memory(runtime.selection())?;
        if runtime.certificate().is_some() {
            let lowering = self.require_lowering()?;
            let context = lowering.tokens;
            let analysis = tessera_compiler::analysis_memory_bytes(&lowering.graph)
                .ok_or(WitnessError::Memory { tensor: context })?;
            let selection = runtime.selection();
            // Both initial read streams can contribute before deduplication.
            let mut reads = 1u64;
            for node in &lowering.graph.nodes {
                reads = add(
                    reads,
                    selection.reads(node.id).unwrap_or(&[]).len() as u64,
                    context,
                )?;
            }
            for &(node, _) in &selection.gadgets {
                reads = add(
                    reads,
                    selection.reads(node).unwrap_or(&[]).len() as u64,
                    context,
                )?;
            }
            for bytes in [
                analysis,
                growing::<TableId>(reads, context)?,
                array::<tessera_compiler::GreedyKeyColumn>(
                    runtime.key_columns().len() as u64,
                    context,
                )?,
            ] {
                bound.bytes = add(bound.bytes, bytes, context)?;
            }
        }
        Ok(bound)
    }

    /// Allocation-request upper bound for `Regenerator::new`/`runtime` graph-input
    /// and checkpoint-shape validation, additional to source metadata. Model weights
    /// stay borrowed INT8. Tokens/channel vectors get one clearing `i64` copy;
    /// positions also have a public `i64` construction vector. Checkpoint-shape
    /// validation creates temporary id and residual-buffer descriptor vectors.
    /// Repeated inputs are charged each time. Assumes canonical lowering shapes.
    /// No model/token values are read and no allocation is made. This excludes replay execution, checkpoints/cache and tiles.
    ///
    /// # Errors
    /// `Usage` without a model lowering; `Memory` for missing tensor references
    /// or overflowing/unaddressable buffers. This does not validate model inputs.
    pub fn source_replay_validation_memory(&self) -> Result<u64, WitnessError> {
        use crate::memory::{add, array, mul};
        let lowering = self.require_lowering()?;
        let context = lowering.tokens;
        // Checkpoints::fits constructs an empty store to validate shapes.
        let nc = lowering.checkpoints.len() as u64;
        let mut bytes = add(
            array::<TensorId>(nc, context)?,
            array::<SecretBuf<i32>>(nc, context)?,
            context,
        )?;
        for tensor in crate::engine::graph_inputs(lowering) {
            let spec = lowering
                .graph
                .tensors
                .get(tensor.0 as usize)
                .ok_or(WitnessError::Memory { tensor })?;
            let count = spec
                .shape
                .iter()
                .try_fold(1u64, |n, &d| mul(n, u64::from(d), tensor))?;
            // Match graph_input's precedence, including malformed overlapping roles.
            let copies = if tensor == lowering.tokens {
                1
            } else if tensor == lowering.positions {
                2
            } else if lowering.graph.weights.iter().any(|&(_, t)| t == tensor) {
                0
            } else {
                1
            };
            bytes = add(
                bytes,
                mul(array::<i64>(count, tensor)?, copies, context)?,
                context,
            )?;
        }
        Ok(bytes)
    }
}
