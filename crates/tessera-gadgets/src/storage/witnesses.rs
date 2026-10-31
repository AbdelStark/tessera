// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public request bounds for model gadget witnesses and Softmax evaluation.
use super::requests::{array, growing, sum, times};
use crate::{
    Committed, CompareColumn, CompareWitness, Group, Requant, RsqrtColumn, RsqrtRelation, RsqrtRow,
    SoftmaxColumn, SoftmaxRow,
};
use tessera_codec::SecretBuf;
use tessera_field::Goldilocks as F;
use tessera_tables::TableId;

impl Requant {
    /// Allocation-free upper bound on every requested byte of `witness` for
    /// `rows` inputs, including returned columns and descriptors, over all valid
    /// Requant forms/classes. Inputs and the existing template are borrowed.
    /// Excludes lookup evaluation, stack/runtime and allocator overhead; does not
    /// check input values or admit a source route. `None` means size overflow or
    /// unaddressable buffers. Consuming `Witness::into_tensors` allocates nothing.
    pub fn witness_memory(rows: u64) -> Option<u64> {
        sum([
            times(4, array::<i64>(rows)),
            array::<(Committed, SecretBuf<i64>)>(4),
        ])
    }
}

impl RsqrtRow {
    /// Allocation-free requested-byte upper bound for `witness` and its later
    /// `RsqrtWitness::into_tensors` transfer, for every valid root parameter tuple.
    /// Includes temporary coefficient polynomials, comparison constraints and
    /// clearing columns; inputs/template are borrowed. Excludes lookup evaluation,
    /// stack/runtime and allocator overhead. Does not validate values or admit a
    /// source. `None` means size overflow or unaddressable buffers.
    pub fn witness_memory(rows: u64) -> Option<u64> {
        const M: u64 = (2 * 32 + 55u64).div_ceil(8);
        // Root + input low limbs + root low bytes <=7 columns, plus a safe-input
        // copy. Each bracket has product/constant columns and 6m comparison columns.
        let columns = 8 + 16 * M;
        // constraints() expands the same two bracket products and comparison
        // constraints as witness(), plus their mapped relations/lookups. Its full
        // request bound therefore also covers witness's public polynomial work.
        let polynomials = Self::memory()?.constraints_bytes();
        sum([
            Some(polynomials),
            times(columns, array::<i64>(rows)),
            growing::<(RsqrtColumn, SecretBuf<i64>)>(7 + 12 * M),
            growing::<CompareWitness>(2),
            times(2, array::<RsqrtRelation<F>>(M)),
            times(4, array::<SecretBuf<i64>>(M)),
            times(4, array::<&[i64]>(M)),
            times(2, array::<(CompareColumn, SecretBuf<i64>)>(6 * M)),
        ])
    }
}

impl SoftmaxRow {
    /// Allocation-free upper bound on all `witness` allocation requests for a
    /// padded row of `positions`. Covers all valid sigma/limb choices (<=49 limbs),
    /// both temporary slack witnesses and their copied returned columns. Template
    /// and scores are borrowed. Excludes evaluate/lookup counting, stack/runtime
    /// and allocator overhead. Does not check values or admit a route. Returns
    /// `None` for a non-power-of-two row or overflowing/unaddressable sizes.
    pub fn witness_memory(positions: u64) -> Option<u64> {
        if !positions.is_power_of_two() {
            return None;
        }
        const LOW: u64 = 48;
        sum([
            array::<i64>(positions),
            array::<i64>(2 + 4 * LOW),
            growing::<(SoftmaxColumn, SecretBuf<i64>)>(3 + 2 * LOW),
            times(2, array::<SecretBuf<i64>>(LOW)),
            times(2, array::<(Committed, SecretBuf<i64>)>(LOW)),
        ])
    }

    /// Allocation-free requested-byte upper bound for Goldilocks `evaluate` on a
    /// `positions` padded row. Includes full constraints, clearing lookup columns,
    /// grouped concatenation/growth, product leaves and padding checks. At most
    /// four tables occur: Exp, reciprocal Range and two NonNeg limb widths.
    /// The witness/template/scores are borrowed; excludes multiplicity counting,
    /// stack/runtime and allocator overhead. Does not verify obligations or admit
    /// a route. `None` means a non-power-of-two row or unaddressable/overflowing sizes.
    pub fn evaluation_memory(positions: u64) -> Option<u64> {
        if !positions.is_power_of_two() {
            return None;
        }
        const LOOKUPS: u64 = 100;
        const GROUPS: u64 = 4;
        // Only Exp has position keys/values. Reciprocal and both slacks are
        // scalar: <=99 Range keys. Exp is the only valued table, so its columns
        // are moved into one group without concatenation or reallocation.
        let entries = positions.checked_mul(2)?.checked_add(LOOKUPS - 1)?;
        // Scalar Range keys grow in <=3 groups, including replaced buffers.
        let concatenated = (LOOKUPS - 1)
            .checked_add(4 * (GROUPS - 1))?
            .checked_mul(4)?;
        let indices = LOOKUPS.checked_add(4 * GROUPS)?.checked_mul(4)?;
        sum([
            Self::constraint_memory(),
            times(2, array::<F>(positions)),
            array::<F>(entries),
            array::<F>(concatenated),
            array::<F>(2),
            array::<(TableId, SecretBuf<F>, Option<SecretBuf<F>>)>(LOOKUPS),
            growing::<Group<F>>(GROUPS),
            times(2, array::<usize>(indices)),
        ])
    }
}
