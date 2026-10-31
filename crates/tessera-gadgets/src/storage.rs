// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requested owned template storage, excluding inline descriptors (RFC-0018).

use tessera_tables::GeneratedTable;

pub(crate) mod requests;

pub(crate) fn array<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

pub(crate) fn table(table: &GeneratedTable) -> Option<usize> {
    table
        .storage_bytes()?
        .checked_sub(size_of::<GeneratedTable>())
}

/// Requested allocation-byte bounds for public gadget construction and Goldilocks
/// constraint metadata. Summing all requests also bounds peak and retained heap.
/// Each gadget's planner states its input preconditions and exclusions; scalar
/// witness generation, borrowed data and whole proof phases are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplateMemory {
    pub(crate) template: u64,
    pub(crate) constraints: u64,
}
impl TemplateMemory {
    /// Upper bound on requested allocator bytes during template construction.
    pub fn template_bytes(self) -> u64 {
        self.template
    }
    /// Upper bound on requested allocator bytes during full Goldilocks constraint
    /// construction, including relations a source may subsequently discard.
    pub fn constraints_bytes(self) -> u64 {
        self.constraints
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn requested_template_arrays_refuse_product_overflow() {
        assert_eq!(super::array::<u64>(usize::MAX / 8 + 1), None);
        assert_eq!(super::array::<u64>(0), Some(0));
    }
}
