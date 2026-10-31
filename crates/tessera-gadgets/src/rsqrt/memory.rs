// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public upper bounds on allocation traffic for root metadata (RFC-0018/0050).

use crate::storage::requests::{array, growing, nonneg, sum, times};
use crate::{
    Affine, CompareColumn, Lookup, RsqrtColumn, RsqrtRelation, RsqrtRow, RsqrtTerm, TemplateMemory,
    Term,
};
use tessera_field::Goldilocks as F;
use tessera_tables::GeneratedTable;

fn comparison(limbs: u64) -> Option<u64> {
    let committed = 6 * limbs;
    let equations = 3 * limbs + 3;
    sum([
        growing::<CompareColumn>(committed),
        growing::<Affine<F, CompareColumn>>(equations),
        growing::<Lookup<F, CompareColumn>>(committed),
        times(committed, array::<(F, CompareColumn)>(1)),
        times(equations, growing::<(F, CompareColumn)>(5)),
    ])
}

impl RsqrtRow {
    /// Allocation-free public bound for root-template construction and complete
    /// Goldilocks constraint metadata. It covers every valid parameter tuple: the
    /// constructor requires `N/e < 2^64`, so roots have <=32 bits, <=4 bytes and
    /// <=15 product limbs (RFC-0022/0023/0050). It authenticates no supplied specs
    /// and covers neither scalar witnesses nor a complete source/CPU phase.
    /// Returns `None` if a requested count exceeds the target address space.
    pub fn memory() -> Option<TemplateMemory> {
        const LIMBS: u64 = (2 * 32 + 55u64).div_ceil(8);
        // Each comparison owns two bound copies, two growing carry arrays,
        // <=32 Range tables, and the temporary needed-width array.
        let one_template = sum([
            times(2, array::<u64>(LIMBS)),
            times(2, growing::<u8>(LIMBS)),
            growing::<u8>(2 + 2 * LIMBS),
            growing::<GeneratedTable>(32),
        ])?;
        let template = sum([times(3, array::<u64>(LIMBS)), times(2, Some(one_template))])?;

        let committed = 7 + 12 * LIMBS;
        let lookups = 9 + 12 * LIMBS;
        let equations = 6 * LIMBS + 7;
        // Only one normalization side per bracket expands rho^2*u. Each (a,b,k)
        // occurs at exactly one coefficient: two brackets *4*4 root bytes *4 u limbs.
        let products = 2 * 4 * 4 * 4;
        let digit = sum([nonneg(4), array::<(F, Term)>(4), array::<(F, RsqrtTerm)>(4)])?;
        let fallback = sum([array::<Affine<F, RsqrtTerm>>(1), array::<(F, RsqrtTerm)>(1)])?;
        // An equation has its initial constant and <=5 comparison terms; each
        // input product replaces one such term. Allow the minimum Vec capacity
        // separately per equation. Product result vectors similarly allow it per
        // coefficient, including coefficients whose expansion is empty.
        let term_arrays = array::<(F, Vec<Affine<F, RsqrtTerm>>)>(
            4 * (10 * equations + 2 * products + 8 * LIMBS),
        );
        let constraints = sum([
            growing::<RsqrtColumn>(committed),
            growing::<RsqrtRelation<F>>(equations),
            growing::<Lookup<F, RsqrtTerm>>(lookups),
            times(lookups, array::<(F, RsqrtTerm)>(4)),
            times(2, nonneg(4)),
            times(2, comparison(LIMBS)),
            term_arrays,
            times(products, array::<Affine<F, RsqrtTerm>>(3)),
            times(3 * products, Some(digit)),
            times(5 * equations, Some(fallback)),
            array::<Affine<F, RsqrtTerm>>(2),
            times(2, array::<(F, RsqrtTerm)>(1)),
        ])?;
        Some(TemplateMemory {
            template,
            constraints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocation_counter::measure;

    fn growth<T: Clone>(value: T) {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 31, 65, 129] {
            let mut output = None;
            let measured = measure(|| {
                let mut items = Vec::new();
                for _ in 0..n {
                    items.push(value.clone());
                }
                output = Some(items);
            });
            assert!(
                measured.bytes_total <= growing::<T>(n).unwrap(),
                "{n}: {measured:?}"
            );
            drop(output);
        }
    }

    #[test]
    fn pinned_vec_allocation_traffic_fits_each_element_size_regime() {
        growth(0u8);
        growth(0u64);
        growth([0u64; 160]);
        growth(());
    }

    #[test]
    fn root_metadata_count_arithmetic_refuses_address_and_product_overflow() {
        let measured = measure(|| {
            assert_eq!(array::<u64>(isize::MAX as u64 / 8 + 1), None);
            assert_eq!(array::<u64>(u64::MAX / 8 + 1), None);
            assert_eq!(growing::<u8>(u64::MAX / 4 + 1), None);
            assert_eq!(growing::<()>(u64::MAX), Some(0));
        });
        assert_eq!(measured.count_total, 0);
    }
}
