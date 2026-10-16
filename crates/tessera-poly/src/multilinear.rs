// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dense multilinear tables: binding the lowest variable and evaluation at a point (RFC-0010
//! 'Multilinear extensions').
//!
//! A table of `2^n` values is the multilinear `f̃(x_1, …, x_n)` equal to `f[Σ_i x_i·2^{i−1}]` on
//! the hypercube, so `x_1` is the least significant bit of the index, and a point lists its
//! coordinates from `x_1` up.

use tessera_field::Field;

use crate::{PolyError, PolyField, eq_table};

/// The low variables an evaluation sums out per block with one `eq` table of `2^BLOCK_VARS`
/// entries, small enough to stay in the first-level cache.
const BLOCK_VARS: usize = 10;

/// The number of variables `n` of a table of `len = 2^n` entries.
///
/// # Errors
///
/// [`PolyError::TableLength`] when `len` is not a power of two (zero included).
pub fn num_vars(len: usize) -> Result<usize, PolyError> {
    if len.is_power_of_two() {
        Ok(len.trailing_zeros() as usize)
    } else {
        Err(PolyError::TableLength { len })
    }
}

/// The number of variables of a table that has at least one.
fn bindable(len: usize) -> Result<usize, PolyError> {
    match num_vars(len)? {
        0 => Err(PolyError::NoVariable),
        vars => Ok(vars),
    }
}

/// Checks that a point has one coordinate per variable.
fn check_point(vars: usize, point: usize) -> Result<(), PolyError> {
    if point == vars {
        Ok(())
    } else {
        Err(PolyError::PointLength {
            expected: vars,
            found: point,
        })
    }
}

/// Binds the lowest variable `x_1` of a table over the prime subfield to `r`, into a new table
/// over `F`: entry `k` of the result is `table[2k] + r·(table[2k+1] − table[2k])`, so the result is
/// the table of `f̃(r, x_2, …, x_n)`.
///
/// This is the first round of a sumcheck over committed values: an `Fp` table bound with an `Fp2`
/// challenge becomes an `Fp2` table. Each pair costs one base subtraction, two base
/// multiplications ([`PolyField::mul_by_base`]) and one base addition. Later rounds bind in place
/// with [`bind_lowest_in_place`].
///
/// # Errors
///
/// [`PolyError::TableLength`] when the length is not a power of two, [`PolyError::NoVariable`]
/// for a table of one entry, and [`PolyError::TooLarge`] when the result cannot be allocated.
pub fn bind_lowest<F: PolyField>(table: &[F::Base], r: F) -> Result<Vec<F>, PolyError> {
    let vars = bindable(table.len())?;
    let mut bound = Vec::new();
    bound
        .try_reserve_exact(table.len() / 2)
        .map_err(|_| PolyError::TooLarge { vars: vars - 1 })?;
    bound.extend(
        table
            .chunks_exact(2)
            .map(|pair| r.mul_by_base(pair[1] - pair[0]).add_base(pair[0])),
    );
    Ok(bound)
}

/// Binds the lowest variable `x_1` of `table` to `r` in place: entry `k` becomes
/// `table[2k] + r·(table[2k+1] − table[2k])` and the table halves, keeping its allocation.
///
/// One pass over the table, one multiplication per pair, no allocation: the form a sumcheck
/// prover calls in every round after the first (RFC-0010 'Multilinear extensions': round `j` binds
/// `x_j`). On error the table is unchanged.
///
/// # Errors
///
/// [`PolyError::TableLength`] when the length is not a power of two, and
/// [`PolyError::NoVariable`] for a table of one entry.
pub fn bind_lowest_in_place<F: Field>(table: &mut Vec<F>, r: F) -> Result<(), PolyError> {
    bindable(table.len())?;
    fold_in_place(table, r);
    table.truncate(table.len() / 2);
    Ok(())
}

/// Writes `t[k] = t[2k] + r·(t[2k+1] − t[2k])` for every `k < t.len() / 2`.
///
/// Ascending `k` is safe in place. After `k = 0`, the outputs are written in ranges `[s, 2s)`
/// for `s = 1, 2, 4, …`: the inputs of a range, `[2s, 4s)`, lie past every output written so far,
/// and its outputs were the inputs of the range before, already read. The split borrow makes each
/// range a plain zip, with no index arithmetic and no bounds checks.
fn fold_in_place<F: Field>(t: &mut [F], r: F) {
    let half = t.len() / 2;
    if half == 0 {
        return;
    }
    t[0] = t[0] + r * (t[1] - t[0]);
    let mut start = 1;
    while start < half {
        let end = (2 * start).min(half);
        let (outputs, inputs) = t.split_at_mut(2 * start);
        for (out, pair) in outputs[start..end].iter_mut().zip(inputs.chunks_exact(2)) {
            *out = pair[0] + r * (pair[1] - pair[0]);
        }
        start = end;
    }
}

/// The value `f̃(point)` of the table over `F`.
///
/// `f̃(r) = Σ_x f[x]·eq(r, x)`. Writing `x = x_low + 2^b·x_high` for the `b` lowest variables
/// splits `eq(r, x) = eq(r_low, x_low)·eq(r_high, x_high)`, so the inner products of each block of
/// `2^b` entries with one small `eq` table give the table of `f̃(r_low, x_high)`, which successive
/// binding then evaluates at `r_high`. That costs about one multiplication per entry, without the
/// writes of binding the full table, and measures faster than successive binding
/// (`benches/throughput.rs`).
///
/// # Errors
///
/// [`PolyError::TableLength`] when the length is not a power of two, and
/// [`PolyError::PointLength`] when the point does not have one coordinate per variable.
pub fn evaluate<F: Field>(table: &[F], point: &[F]) -> Result<F, PolyError> {
    check_point(num_vars(table.len())?, point.len())?;
    evaluate_blocked(table, point, |eq, value| eq * value)
}

/// The value `f̃(point)` of a table over the prime subfield at a point over `F`: an `Fp` table at
/// an `Fp2` point, as committed values are opened.
///
/// The same method as [`evaluate`], with products by base elements
/// ([`PolyField::mul_by_base`]: two base multiplications in a quadratic extension).
///
/// # Errors
///
/// As [`evaluate`].
pub fn evaluate_base<F: PolyField>(table: &[F::Base], point: &[F]) -> Result<F, PolyError> {
    check_point(num_vars(table.len())?, point.len())?;
    evaluate_blocked(table, point, |eq: F, value| eq.mul_by_base(value))
}

/// `Σ_x table[x]·eq(point, x)` for a power-of-two table with one coordinate per variable, with
/// `mul(eq, value)` the product of an `eq` entry by a table entry.
fn evaluate_blocked<T: Copy, F: Field>(
    table: &[T],
    point: &[F],
    mul: impl Fn(F, T) -> F,
) -> Result<F, PolyError> {
    let low = point.len().min(BLOCK_VARS);
    let eq_low = eq_table(&point[..low])?;
    // The table of f̃(r_low, x_high): one inner product per block of 2^low entries.
    let mut partial = Vec::new();
    partial
        .try_reserve_exact(table.len() >> low)
        .map_err(|_| PolyError::TooLarge {
            vars: point.len() - low,
        })?;
    partial.extend(
        table
            .chunks_exact(eq_low.len())
            .map(|block| inner_product(block, &eq_low, &mul)),
    );
    for &r in &point[low..] {
        fold_in_place(&mut partial, r);
        partial.truncate(partial.len() / 2);
    }
    Ok(partial[0])
}

/// `Σ_i mul(eq[i], values[i])` over equal-length slices, with four accumulators so that the
/// additions of consecutive products do not wait on each other.
fn inner_product<T: Copy, F: Field>(values: &[T], eq: &[F], mul: &impl Fn(F, T) -> F) -> F {
    let mut sums = [F::ZERO; 4];
    let mut values = values.chunks_exact(4);
    let mut eq = eq.chunks_exact(4);
    for (v, e) in (&mut values).zip(&mut eq) {
        sums[0] += mul(e[0], v[0]);
        sums[1] += mul(e[1], v[1]);
        sums[2] += mul(e[2], v[2]);
        sums[3] += mul(e[3], v[3]);
    }
    for (&v, &e) in values.remainder().iter().zip(eq.remainder()) {
        sums[0] += mul(e, v);
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}
