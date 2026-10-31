// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public source metadata construction, separate from replay/runtime/tile work (RFC-0018).
use super::*;

/// Requested-byte upper bound for the structure and gadget metadata constructed by
/// a resident source. It sums allocation requests, bounding peak and retained heap.
/// Borrowed inputs, runtime range validation/key columns, grouped-column transfers,
/// regenerator input validation, returned tiles and allocator/runtime overhead are
/// separate. This neither authenticates metadata nor admits a source route.
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
