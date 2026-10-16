// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `eq` tables and `eq` of two points (RFC-0010 'Multilinear extensions'):
//! `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`.

use tessera_field::Field;

use crate::PolyError;

/// The table of `eq(point, x)` over the hypercube: entry `x` is the product of `r_i` over the
/// bits `x_i = 1` of `x` and of `1 − r_i` over the others, `x_i` being bit `i − 1`, so
/// `f̃(r) = Σ_x f[x]·table[x]`.
///
/// Built by doubling: after the coordinates `r_1, …, r_j` the table holds the `2^j` products of
/// the first `j` factors, and `r_{j+1}` splits each entry `e` into `e·(1 − r_{j+1})` in place and
/// `e·r_{j+1}` in the upper half, as `x_{j+1}` is the next bit. One multiplication per entry,
/// `2^n − 1` in all, and one allocation of exactly `2^n` entries. A point with no coordinate gives
/// `[1]`.
///
/// # Errors
///
/// [`PolyError::TooLarge`] when `2^n` entries cannot be allocated, `n` the number of coordinates.
pub fn eq_table<F: Field>(point: &[F]) -> Result<Vec<F>, PolyError> {
    let vars = point.len();
    let too_large = PolyError::TooLarge { vars };
    let len = u32::try_from(vars)
        .ok()
        .and_then(|v| 1usize.checked_shl(v))
        .ok_or(too_large)?;
    let mut table = Vec::new();
    table.try_reserve_exact(len).map_err(|_| too_large)?;
    table.push(F::ONE);
    for &r in point {
        let half = table.len();
        // Within the reserved capacity: no reallocation.
        table.resize(2 * half, F::ZERO);
        let (low, high) = table.split_at_mut(half);
        for (zero, one) in low.iter_mut().zip(high) {
            *one = *zero * r;
            *zero -= *one;
        }
    }
    Ok(table)
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))` for two points of the same length: the
/// multilinear extension of the equality predicate, evaluated at two arbitrary points
/// (RFC-0010 'Multilinear extensions'). Two multiplications per coordinate.
///
/// # Errors
///
/// [`PolyError::PointLength`] when the points differ in length (`expected` is `a`'s).
pub fn eq<F: Field>(a: &[F], b: &[F]) -> Result<F, PolyError> {
    same_length(a, b)?;
    Ok(a.iter()
        .zip(b)
        .fold(F::ONE, |product, (&a, &b)| product * eq_factor(a, b)))
}

/// `a·b + (1 − a)·(1 − b) = 2ab − a − b + 1`: one multiplication.
#[inline]
fn eq_factor<F: Field>(a: F, b: F) -> F {
    (a * b).double() - a - b + F::ONE
}

/// Checks that two points have the same number of coordinates.
pub(crate) fn same_length<F>(a: &[F], b: &[F]) -> Result<(), PolyError> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(PolyError::PointLength {
            expected: a.len(),
            found: b.len(),
        })
    }
}
