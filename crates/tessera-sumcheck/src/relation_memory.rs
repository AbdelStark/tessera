// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical relation-table construction and instance metadata (RFC-0010/0018).
use crate::{ElementwiseShape, Input, MatMulActShape, Table, Term, WindowError};
use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

/// Relation-table builder requests only. Original borrowed operands/point/kernel,
/// original relation objects retained separately, outer relation/instance
/// containers, claims, sumcheck state, callbacks and
/// stack/runtime/allocator overhead remain additional. No CPU route is admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationMemory {
    construction: u64,
    retained: u64,
    instance: u64,
}
impl RelationMemory {
    /// Canonical table constructor peak beside borrowed operands.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }
    /// Owned tables and public metadata retained after construction.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }
    /// Additional canonical instance metadata, borrowing the retained tables.
    pub fn instance_bytes(self) -> u64 {
        self.instance
    }

    /// MatMulAct::tables; MatMulWeight uses (stack=layers,row=tokens,col=outputs).
    /// Both possible F contractions are charged even when an operand is borrowed.
    /// Arbitrary spare capacity in supplied precontracted buffers remains additional.
    /// # Errors
    /// Unaddressable/overflowing public geometry.
    pub fn matmul<F: PolyField>(shape: MatMulActShape) -> Result<Self, WindowError> {
        let MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        } = shape;
        check_bits(&[stack_vars, row_vars, inner_vars])?;
        check_bits(&[stack_vars, col_vars, inner_vars])?;
        let vars = sum(&[stack_vars, inner_vars])?;
        let field = array::<F>(power(vars)?)?;
        let eq = if stack_vars == 0 { 0 } else { field };
        let retained = add(
            add(mul(2, field)?, eq)?,
            array::<F>(sum(&[row_vars, col_vars])? as u64)?,
        )?;
        let scratch = array::<F>(add(
            add(power(row_vars)?, power(col_vars)?)?,
            power(stack_vars)?,
        )?)?;
        Ok(Self {
            construction: add(retained, scratch)?,
            retained,
            instance: array::<Table<'static, F>>(3)?,
        })
    }
    /// Elementwise::new/tables and instance, before any product specialization.
    /// `expanded` counts Partner/Broadcast inputs; terms give every factor count.
    /// Original inputs and their payloads are not included.
    /// # Errors
    /// Invalid shape/counts or unaddressable/overflowing public geometry.
    pub fn elementwise<F: PolyField>(
        shape: ElementwiseShape,
        inputs: usize,
        expanded: usize,
        factors: &[usize],
    ) -> Result<Self, WindowError> {
        if shape.summed_vars > shape.vars || inputs == 0 || expanded > inputs || factors.is_empty()
        {
            return Err(bad());
        }
        check_bits(&[shape.vars])?;
        let target = shape.vars - shape.summed_vars;
        let eq = if target == 0 {
            0
        } else {
            array::<F>(power(shape.vars)?)?
        };
        let own_terms = terms::<F>(factors, 0)?;
        let relation = add(array::<Input<'static, F>>(inputs as u64)?, own_terms)?;
        let derived = add(
            array::<Option<SecretBuf<F::Base>>>(inputs as u64)?,
            mul(expanded as u64, array::<F::Base>(power(shape.vars)?)?)?,
        )?;
        let retained = add(relation, add(eq, derived)?)?;
        let scratch = if target == 0 {
            0
        } else {
            array::<F>(power(target)?)?
        };
        let tables = inputs
            .checked_add(usize::from(target > 0))
            .ok_or(WindowError::Allocation)?;
        let instance = add(
            array::<Table<'static, F>>(tables as u64)?,
            terms::<F>(factors, usize::from(target > 0))?,
        )?;
        Ok(Self {
            construction: add(
                retained,
                add(
                    add(growing::<Input<'static, F>>(inputs as u64)?, own_terms)?,
                    scratch,
                )?,
            )?,
            retained,
            instance,
        })
    }
    /// Rotate::tables and instance, including the public kernel and split EQ.
    /// # Errors
    /// Invalid or unaddressable/overflowing public geometry.
    pub fn rotate<F: PolyField>(vars: usize) -> Result<Self, WindowError> {
        if vars == 0 {
            return Err(bad());
        }
        check_bits(&[vars])?;
        let retained = array::<F>(power(vars)?)?;
        let scratch = add(
            array::<F>(power(vars - 1)?)?,
            growing::<F>((vars - 1) as u64)?,
        )?;
        Ok(Self {
            construction: add(retained, scratch)?,
            retained,
            instance: array::<Table<'static, F>>(2)?,
        })
    }
    /// Selection::tables and instance, preserving every selected member. Original
    /// member indices are borrowed; a canonical stage's Selected view array is charged.
    /// # Errors
    /// Unaddressable/overflowing public geometry.
    pub fn selection<F: PolyField>(
        low: usize,
        axis: usize,
        high: usize,
        rows: usize,
        members: usize,
    ) -> Result<Self, WindowError> {
        check_bits(&[low, axis, high])?;
        check_bits(&[low, rows, high])?;
        let retained = add(
            mul(2, array::<F>(power(axis)?)?)?,
            array::<F>(sum(&[low, high])? as u64)?,
        )?;
        // eq(rows), low-variable contraction, and both other EQ tables can overlap.
        let scratch = array::<F>(add(
            add(power(rows)?, power(sum(&[axis, high])?)?)?,
            add(power(low)?, power(high)?)?,
        )?)?;
        let views = growing::<crate::Selected<'static, F>>(members as u64)?;
        Ok(Self {
            construction: add(retained, add(scratch, views)?)?,
            retained,
            instance: array::<Table<'static, F>>(2)?,
        })
    }
    /// Recomposition::tables and instance; public-x clone overlap is always charged.
    /// # Errors
    /// Invalid widths or unaddressable/overflowing public geometry.
    pub fn recomposition<F: PolyField>(
        low: u8,
        high: u8,
        rows: usize,
    ) -> Result<Self, WindowError> {
        if low == 0 || u16::from(low) + u16::from(high) > 62 {
            return Err(bad());
        }
        let bit_vars = |n: u8| u32::from(n).next_power_of_two().trailing_zeros() as usize;
        let k = bit_vars(low).max(bit_vars(high));
        check_bits(&[rows, k])?;
        let n = power(sum(&[rows, k])?)?;
        let outputs = 1 + u64::from(high > 0);
        let field = array::<F>(n)?;
        let base = array::<F::Base>(n)?;
        let retained = add(mul(1 + outputs, field)?, mul(1 + 2 * outputs, base)?)?;
        let scratch = add(base, array::<F>(power(rows)?)?)?;
        let count = 5 + 3 * u64::from(high > 0);
        let instance = add(
            growing::<Table<'static, F>>(count)?,
            terms::<F>(if high > 0 { &[2, 3, 3] } else { &[2, 3] }, 0)?,
        )?;
        Ok(Self {
            construction: add(retained, scratch)?,
            retained,
            instance,
        })
    }
}
fn bad() -> WindowError {
    WindowError::Shape("invalid canonical relation memory geometry")
}
fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or(WindowError::Allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or(WindowError::Allocation)
}
fn sum(parts: &[usize]) -> Result<usize, WindowError> {
    parts.iter().try_fold(0usize, |n, &v| {
        n.checked_add(v).ok_or(WindowError::Allocation)
    })
}
fn check_bits(parts: &[usize]) -> Result<(), WindowError> {
    if sum(parts)? >= usize::BITS as usize {
        Err(bad())
    } else {
        Ok(())
    }
}
fn power(bits: usize) -> Result<u64, WindowError> {
    if bits >= usize::BITS as usize {
        return Err(bad());
    }
    1u64.checked_shl(bits as u32).ok_or(WindowError::Allocation)
}
fn array<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn growing<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, n.max(minimum))?)
}
fn terms<F: PolyField>(factors: &[usize], shift: usize) -> Result<u64, WindowError> {
    let mut out = growing::<Term<F>>(factors.len() as u64)?;
    for &n in factors {
        out = add(
            out,
            growing::<usize>(n.checked_add(shift).ok_or(WindowError::Allocation)? as u64)?,
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_and_dimension_arithmetic_never_wraps() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<u64>(isize::MAX as u64 / 8 + 1),
                Err(WindowError::Allocation)
            );
            assert_eq!(add(u64::MAX, 1), Err(WindowError::Allocation));
            assert_eq!(mul(u64::MAX, 2), Err(WindowError::Allocation));
            assert_eq!(sum(&[usize::MAX, 1]), Err(WindowError::Allocation));
            assert_eq!(
                growing::<u8>(u64::MAX / 4 + 1),
                Err(WindowError::Allocation)
            );
            assert_eq!(power(usize::MAX), Err(bad()));
        });
        assert_eq!(measured.count_total, 0);
    }
}
