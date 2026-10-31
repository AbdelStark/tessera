// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exp-core construction and full Softmax metadata requests (RFC-0018/0030/0050).

use super::{SoftmaxColumn, SoftmaxLookup, SoftmaxRelation, SoftmaxRow, SoftmaxTerm};
use crate::storage::requests::{array, growing, nonneg, sum};
use crate::{GadgetError, TemplateMemory};
use tessera_field::Goldilocks as F;
use tessera_tables::{GeneratedTable, TableError, TableId, TableKind, TableSpec};

impl SoftmaxRow {
    pub(crate) fn constraint_memory() -> Option<u64> {
        const LIMBS: u64 = 31 + 18;
        // Seven relation vectors have 18 monomials and 23 literal factors. The
        // first two lookups have three affine vectors; both slack mappings add
        // 2*(L-1)+2*L coefficient/term pairs, including both virtual top limbs.
        sum([
            growing::<SoftmaxColumn>(3 + 2 * (LIMBS - 1)),
            growing::<SoftmaxLookup<F>>(2 + 2 * LIMBS),
            array::<SoftmaxRelation<F>>(7),
            array::<(F, Vec<SoftmaxTerm>)>(18),
            array::<SoftmaxTerm>(23),
            array::<(F, SoftmaxTerm)>(4 * LIMBS + 1),
            nonneg(LIMBS),
        ])
    }

    /// Plan requested template and Goldilocks constraint bytes without allocation
    /// or core generation. The Exp payload follows its public parameters; metadata
    /// covers every valid limb/context choice (at most 49 NonNeg limbs).
    /// This authenticates neither specs/digests nor constructor preconditions and
    /// excludes `.tables()`, scalar witnesses, borrowed specs, inline `Self`, stack
    /// and allocator/runtime overhead. It is not a source/CPU admission decision.
    ///
    /// # Errors
    /// `Table` for missing/non-Exp metadata; `Spec` for invalid Exp geometry or
    /// requested arrays outside the target address space.
    pub fn memory(exp: TableId, tables: &[TableSpec]) -> Result<TemplateMemory, GadgetError> {
        let spec = tables
            .iter()
            .find(|spec| spec.id == exp)
            .ok_or(GadgetError::Table {
                table: exp,
                rule: "SoftmaxRow's Exp specification exists",
            })?;
        if !matches!(spec.kind, TableKind::Exp { .. }) {
            return Err(GadgetError::Table {
                table: exp,
                rule: "SoftmaxRow reads Exp",
            });
        }
        let fail = |error| GadgetError::Spec { table: exp, error };
        let core = GeneratedTable::core_storage_len(&spec.kind).map_err(fail)?;
        // Exact core allocation and the digest's one-u64 zero-tail piece. Range
        // templates compute rows and digest directly, without heap payload.
        let template = array::<i64>(core)
            .and_then(|n| n.checked_add(size_of::<u64>() as u64))
            .ok_or_else(|| fail(TableError::Memory))?;
        let constraints = Self::constraint_memory().ok_or_else(|| fail(TableError::Memory))?;
        Ok(TemplateMemory {
            template,
            constraints,
        })
    }
}
