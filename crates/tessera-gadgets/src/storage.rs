// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requested owned template storage, excluding inline descriptors (RFC-0018).

use tessera_tables::GeneratedTable;

pub(crate) fn array<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

pub(crate) fn table(table: &GeneratedTable) -> Option<usize> {
    table
        .storage_bytes()?
        .checked_sub(size_of::<GeneratedTable>())
}

#[cfg(test)]
mod tests {
    #[test]
    fn requested_template_arrays_refuse_product_overflow() {
        assert_eq!(super::array::<u64>(usize::MAX / 8 + 1), None);
        assert_eq!(super::array::<u64>(0), Some(0));
    }
}
