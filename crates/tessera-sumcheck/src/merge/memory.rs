// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public requested-allocation geometry of RFC-0054's two-pass merge.

use core::ops::Range;

use super::{Class, MAX_MERGE_VARS, MergePolynomial};
use crate::WindowError;
use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

/// Checked storage for `MergeTables::new`/`add_group` and `prove_merge_two_pass`.
///
/// Borrowed polynomial/claim/Merge inputs, caller-owned blocks and regenerated
/// source allocations, transcript/encoding callbacks, stack, allocator and process
/// overhead are additional. In particular, a regeneration callback may retain
/// spare capacity or arbitrary scratch: this model does not bound that callback.
/// It does not admit a CPU route or authenticate the supplied public geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeMemory {
    tables: u64,
    construction: u64,
    proving: u64,
    output: u64,
}

impl MergeMemory {
    /// Retained group-table storage, including cloned shapes and class metadata.
    pub fn table_bytes(self) -> u64 {
        self.tables
    }

    /// Group-table construction peak, including all retained tables. Caller
    /// blocks, claim inputs and their construction are additional.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }

    /// Additional two-pass engine peak beside its borrowed Merge/group tables
    /// and caller-owned regenerated source/callback work. Includes returned output.
    pub fn proving_peak_bytes(self) -> u64 {
        self.proving
    }

    /// Returned proof, point and evaluation-index allocations, included above.
    pub fn output_bytes(self) -> u64 {
        self.output
    }

    /// Bound the two-pass engine from polynomial shapes and, for each group, the
    /// polynomial index of every claim in order. Charge one distinct point class
    /// per claim; no transcript point, coincidence, value or secret is inspected.
    /// This calculation allocates nothing and preserves repeated claims.
    ///
    /// # Errors
    /// `WindowError::Shape` for empty/invalid polynomial geometry, an unknown
    /// claimed polynomial or an illegal cut. `WindowError::Allocation` for
    /// overflowing or unaddressable requested storage.
    pub fn for_two_pass<F: PolyField>(
        polynomials: &[MergePolynomial],
        claim_polynomials: &[Vec<usize>],
        cut: usize,
    ) -> Result<Self, WindowError> {
        if polynomials.is_empty()
            || polynomials.iter().any(|p| {
                p.vars > MAX_MERGE_VARS
                    || p.vars >= usize::BITS as usize
                    || p.group_vars.windows(2).any(|w| w[0] >= w[1])
                    || p.group_vars.last().is_some_and(|&v| v >= p.vars)
            })
        {
            return Err(WindowError::Shape("invalid merge polynomial geometry"));
        }
        if cut > super::largest_table_rounds(polynomials) {
            return Err(WindowError::Shape("merge table cut exceeds free variables"));
        }
        let rounds = polynomials.iter().map(|p| p.vars).max().unwrap_or(0);
        let count = polynomials.len() as u64;
        let groups = claim_polynomials.len() as u64;
        let mut tables = add(
            array::<MergePolynomial>(count)?,
            array::<Option<(F, usize)>>(groups)?,
        )?;
        for p in polynomials {
            tables = add(tables, array::<usize>(p.group_vars.len() as u64)?)?;
        }
        let (mut claims, mut constructor_scratch) = (0, 0);
        for group in claim_polynomials {
            let mut combined = 0;
            for &index in group {
                let p = polynomials
                    .get(index)
                    .ok_or(WindowError::Shape("unknown merge claim polynomial"))?;
                let own = cut.saturating_sub(rounds - p.vars);
                let mut class = add(
                    array::<F>(p.free_vars() as u64)?,
                    array::<SecretBuf<F>>(own as u64)?,
                )?;
                for t in 0..own {
                    class = add(class, array::<F>(power(t + 1)?)?)?;
                }
                tables = add(tables, class)?;
                combined = add(combined, array::<F>(power(p.free_vars())?)?)?;
            }
            let n = group.len() as u64;
            // All full-sized Q buffers coexist before any class is contracted.
            let descriptors = add(
                grow::<(&MergePolynomial, &[F], u64)>(n)?,
                grow::<SecretBuf<F>>(n)?,
            )?;
            constructor_scratch = constructor_scratch.max(add(combined, descriptors)?);
            claims = add(claims, n)?;
        }
        // Geometric class-vector growth includes simultaneous old/new storage.
        tables = add(tables, grow::<Class<F>>(claims)?)?;
        let construction = add(tables, constructor_scratch)?;
        let output = add(
            add(
                array::<Vec<F>>(rounds as u64)?,
                array::<F>(mul(2, rounds as u64)?)?,
            )?,
            add(
                array::<F>(rounds as u64)?,
                add(
                    array::<F>(count)?,
                    add(array::<usize>(count)?, array::<Range<usize>>(count)?)?,
                )?,
            )?,
        )?;
        let fixed = add(
            output,
            add(
                32, // Labels::new's reused merge label; vars <= 40.
                add(
                    array::<F>(groups)?,
                    add(
                        // The second sums vector shadows the first; charge both.
                        array::<F>(mul(2, count)?)?,
                        array::<(SecretBuf<F>, Vec<F>, usize)>(count)?,
                    )?,
                )?,
            )?,
        )?;
        let mut retained = 0;
        let mut peak = if cut == 0 {
            0
        } else {
            array::<F>(power(cut - 1)?)? // table-round eq_low scratch
        };
        for p in polynomials {
            let full = power(p.vars)?;
            let bound = cut.saturating_sub(rounds - p.vars);
            let folded = power(p.vars - bound)?;
            // Prior compact states coexist with one base permutation, full
            // values/weights, equality scratch and any compact replacement copy.
            let mut scratch = add(
                array::<F::Base>(full)?,
                add(
                    array::<F>(mul(2, full)?)?,
                    add(
                        array::<F>(power(p.free_vars())?)?,
                        grow::<usize>(p.vars as u64)?,
                    )?,
                )?,
            )?;
            if bound != 0 {
                scratch = add(scratch, array::<F>(mul(2, folded)?)?)?;
            }
            peak = peak.max(add(retained, scratch)?);
            retained = add(retained, array::<F>(mul(2, folded)?)?)?;
        }
        let proving = add(fixed, peak.max(retained))?;
        Ok(Self {
            tables,
            construction,
            proving,
            output,
        })
    }
}

fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or(WindowError::Allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or(WindowError::Allocation)
}
fn power(bits: usize) -> Result<u64, WindowError> {
    1u64.checked_shl(bits as u32).ok_or(WindowError::Allocation)
}
fn array<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn grow<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    // Rust1.95 Vec geometric old/new overlap, including minimum capacity.
    let minimum = if size_of::<T>() == 1 { 8 } else { 4 };
    array::<T>(mul(4, n.max(minimum))?)
}
