// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The structured evaluation of a table's multilinear extension (RFC-0008 'Structured
//! evaluation'): `t̃(ρ) = kẽy(ρ) + β·valũe(ρ)` over the table tree's `2^n` rows, `n` the
//! table's `log2_rows`, in time set by the table's parameters rather than its up to `2^32` rows.
//!
//! The conventions are RFC-0010's (`tessera-poly`): row `j` is the point whose coordinate `i` is
//! bit `i` of `j`, the lowest first, and `eq(ρ, j) = Π_i (j_i ? ρ_i : 1 − ρ_i)`. Rows `T` to
//! `2^n − 1` repeat row 0, so with `t_j` the rows below `T`,
//!
//! ```text
//! t̃(ρ) = Σ_{j<T} eq(ρ, j)·t_j + t_0·(1 − LT(ρ, T)),   LT(ρ, a) = Σ_{h<a} eq(ρ, h).
//! ```
//!
//! - **Keys.** Row `j`'s key is `j`, an affine function of its bits, and its sum over a prefix
//!   of rows has a closed form, [`prefix_affine`]: `O(n)`.
//! - **`Range`**: no values.
//! - **`Requant`**, class by class. `x` is the row's offset key `k + 2^{b−1}` and `r` the class's
//!   shift. Before the clamp, `⌊(k + 2^{r−1})/2^r⌋` is `(x >> r) + x_{r−1} − 2^{b−1−r}` for
//!   `r < b`, an affine function of `x`'s bits, and 0 for `r ≥ b`. It never decreases with `x`,
//!   so the clamp holds `lo` on a prefix of the class, `hi` on a suffix, and the affine function
//!   between. Each part is a difference of two prefix sums: `O(n)` per class, whatever `b`, `lo`
//!   and `hi`.
//! - **`Silu`**: a class's value depends on `h = x >> r` only, so its `r` low variables sum out,
//!   `Σ_{x_low} eq(ρ_low, x_low) = 1`, and the class's `2^{b−r}` core values are evaluated at
//!   the high coordinates: `O(2^{b−r})`, at most `2^24` over the classes.
//! - **`Exp`**: its core, the keys below its zero tail `z ≤ 2^24`; 0 above.
//! - **`Exp2Frac`, `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul`**: at most `2^16` rows, enumerated.
//!
//! A dense core is cut as RFC-0008 cuts a piecewise function: the runs of equal values at either
//! end, such as `Silu`'s saturation at 0 and 127, cost a difference of two `LT`s, and the values
//! between are cut into aligned blocks, each an inner product with an `eq` table built by doubling
//! ([`eq_table`]) and folded at the block's high coordinates ([`evaluate`]). No step divides, so
//! every point is handled, those with a coordinate 0 or 1 included, where updating `eq` along a
//! Gray code would divide by zero.
//! Table parameters are public: the control flow depends on them and on lengths only.

use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_poly::{PolyError, PolyField, eq_table, evaluate, less_than};

use crate::table::GeneratedTable;
use crate::{Table, TableKind};

/// The low variables whose `eq` table a dense block's inner products share: `2^12` entries.
const BLOCK_VARS: usize = 12;

/// `t̃(ρ) = kẽy(ρ) + β·valũe(ρ)` for `table` over its `2^{log2_rows}` rows (module
/// documentation).
pub(crate) fn evaluate_table<F: PolyField<Base = Goldilocks>>(
    table: &GeneratedTable,
    rho: &[F],
    beta: F,
) -> Result<F, PolyError> {
    let vars = usize::from(table.log2_rows());
    if rho.len() != vars {
        return Err(PolyError::PointLength {
            expected: vars,
            found: rho.len(),
        });
    }
    let rows = table.row_count();
    let powers: Vec<F> = (0..vars).map(|i| integer(1i64 << i)).collect();
    let key = prefix_affine(rho, &powers, F::ZERO, rows)?;
    let padding = F::ONE - less_than(rho, rows)?;
    let value = values(table, rho)? + padding * integer(table.value(0));
    Ok(key + beta * value)
}

/// `Σ_{j<T} eq(ρ, j)·value_j`: the value column over the table's `T` rows, without padding.
fn values<F: PolyField<Base = Goldilocks>>(
    table: &GeneratedTable,
    rho: &[F],
) -> Result<F, PolyError> {
    match &table.spec().kind {
        // A `GeneratedTable` never holds `Rope`, which `new` refuses.
        TableKind::Range { .. } | TableKind::Rope { .. } => Ok(F::ZERO),
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => by_class(rho, *b, r_by_class.len(), |class, low| {
            requant_class(low, *b, r_by_class[class], *lo, *hi)
        }),
        TableKind::Silu { b, r_by_class, .. } => {
            let mut start = 0;
            by_class(rho, *b, r_by_class.len(), |class, low| {
                let r = usize::from(r_by_class[class]);
                let count = 1usize << (usize::from(*b) - r);
                let core = &table.core()[start..start + count];
                start += count;
                dense(core, &low[r..])
            })
        }
        TableKind::Exp { .. } | TableKind::Exp2Frac { .. } => dense(table.core(), rho),
        TableKind::Pow2 { .. }
        | TableKind::Bitwise8 { .. }
        | TableKind::AesSbox
        | TableKind::Gf256Mul { .. } => {
            let column: Vec<i64> = (0..table.row_count()).map(|j| table.value(j)).collect();
            dense(&column, rho)
        }
    }
}

/// `Σ_{c<classes} eq(ρ_high, c)·f(c, ρ_low)` for a table whose row `j` is `c·2^b + x`: the
/// class in the high bits, from bit `b`, above the offset key `x` in the `b` low ones.
fn by_class<F: Field>(
    rho: &[F],
    b: u8,
    classes: usize,
    mut f: impl FnMut(usize, &[F]) -> Result<F, PolyError>,
) -> Result<F, PolyError> {
    let (low, high) = rho.split_at(usize::from(b));
    let mut sum = F::ZERO;
    for (class, index) in (0..classes).zip(0u64..) {
        sum += eq_at(high, index) * f(class, low)?;
    }
    Ok(sum)
}

/// `Σ_{x<2^b} eq(ρ_low, x)·clamp(g(x), lo, hi)` for one class of shift `r`, where
/// `g(x) = ⌊(x + 2^{r−1} − 2^{b−1})/2^r⌋` is `rq` of the key `k = x − 2^{b−1}` before its clamp
/// (RFC-0005 'Integer semantics').
///
/// `g` never decreases, so `g(x) ≥ v` exactly from `x = v·2^r − 2^{r−1} + 2^{b−1}` on: the clamp
/// holds `lo` below `a`, the first `x` with `g(x) ≥ lo`, and `hi` from `c`, the first with
/// `g(x) ≥ hi + 1`, both cut to `[0, 2^b]`, and `a ≤ c` as `lo ≤ hi`. Between them the value
/// is `g`, an affine function of `x`'s bits:
///
/// - for `r < b`, `x = q·2^r + s` with `s < 2^r` gives `⌊(x + 2^{r−1})/2^r⌋ = q + [s ≥ 2^{r−1}]`,
///   so `g(x) = Σ_{i≥r} 2^{i−r}·x_i + x_{r−1} − 2^{b−1−r}`, the offset `2^{b−1}` being a multiple
///   of `2^r`;
/// - for `r ≥ b`, `x + 2^{r−1} − 2^{b−1}` lies in `[0, 2^r)` and `g(x) = 0`.
///
/// So the sum is `lo·LT(ρ, a) + G(c) − G(a) + hi·(1 − LT(ρ, c))`, with `G(t)` the sum of `eq·g`
/// below `t`. Every intermediate fits an `i128`: `|v·2^r| < 2^{32+63}`.
fn requant_class<F: PolyField<Base = Goldilocks>>(
    low: &[F],
    b: u8,
    r: u8,
    lo: i32,
    hi: i32,
) -> Result<F, PolyError> {
    let cube = 1i128 << b;
    let shift = 1i128 << r;
    let offset = (1i128 << (r - 1)) - (1i128 << (b - 1));
    let first_at_least = |v: i64| {
        let x = (i128::from(v) * shift - offset).clamp(0, cube);
        // Within `[0, 2^b]`, `b ≤ 32`.
        u64::try_from(x).unwrap_or(0)
    };
    let (a, c) = (first_at_least(lo.into()), first_at_least(i64::from(hi) + 1));
    let mut weights = vec![F::ZERO; low.len()];
    let mut constant = F::ZERO;
    if r < b {
        let r = usize::from(r);
        for (i, weight) in weights.iter_mut().enumerate().skip(r) {
            *weight = integer(1i64 << (i - r));
        }
        weights[r - 1] = F::ONE;
        constant = integer(-(1i64 << (usize::from(b) - 1 - r)));
    }
    let g = |t| prefix_affine(low, &weights, constant, t);
    let below = less_than(low, a)?;
    let above = F::ONE - less_than(low, c)?;
    let (lo, hi): (F, F) = (integer(i64::from(lo)), integer(i64::from(hi)));
    Ok(lo * below + (g(c)? - g(a)?) + hi * above)
}

/// `Σ_{h<a} eq(r, h)·(w0 + Σ_i w_i·h_i)` over the `n`-bit integers `h`, for `0 ≤ a ≤ 2^n`:
/// the sum of an affine function of the bits over a prefix of the cube, with about four
/// multiplications per coordinate. `w` has one weight per coordinate of `r`.
///
/// `h < a` exactly when, at the highest bit `k` where they differ, `a_k = 1` and `h_k = 0`. Fix
/// the bits of `h` above `k` to `a`'s and bit `k` to 0, and sum over the free bits below: their
/// `eq` factors sum to 1, and each `h_i` sums to `r_i`. So
///
/// ```text
/// Σ_{h<a} = Σ_{k: a_k = 1} (Π_{i>k} ê_i)·(1 − r_k)·(w0 + Σ_{i>k} w_i·a_i + Σ_{i<k} w_i·r_i)
/// ```
///
/// with `ê_i = r_i` when `a_i = 1` and `1 − r_i` otherwise, the `eq` factors of the fixed bits.
/// For `a = 2^n` the sum covers the cube: `w0 + Σ_i w_i·r_i`.
///
/// # Errors
///
/// [`PolyError::BoundTooLarge`] for `a > 2^n`.
fn prefix_affine<F: Field>(r: &[F], w: &[F], w0: F, a: u64) -> Result<F, PolyError> {
    let vars = r.len();
    let mut low = r
        .iter()
        .zip(w)
        .fold(F::ZERO, |sum, (&ri, &wi)| sum + wi * ri);
    if vars < 64 {
        let cube = 1u64 << vars;
        if a > cube {
            return Err(PolyError::BoundTooLarge { bound: a, vars });
        }
        if a == cube {
            return Ok(w0 + low);
        }
    }
    let mut sum = F::ZERO;
    // Π_{i>k} ê_i and w0 + Σ_{i>k} w_i·a_i, as `k` comes down.
    let mut fixed = F::ONE;
    let mut high = w0;
    for k in (0..vars).rev() {
        // Now Σ_{i<k} w_i·r_i.
        low -= w[k] * r[k];
        if k < 64 && (a >> k) & 1 == 1 {
            sum += fixed * (F::ONE - r[k]) * (high + low);
            fixed *= r[k];
            high += w[k];
        } else {
            fixed *= F::ONE - r[k];
        }
    }
    Ok(sum)
}

/// `eq(r, x) = Π_i (x_i ? r_i : 1 − r_i)` for an integer `x` below `2^{|r|}`.
fn eq_at<F: Field>(r: &[F], x: u64) -> F {
    r.iter().enumerate().fold(F::ONE, |product, (i, &ri)| {
        let bit = i < 64 && (x >> i) & 1 == 1;
        product * if bit { ri } else { F::ONE - ri }
    })
}

/// `Σ_{h<len} eq(r, h)·values[h]` for `len = values.len() ≤ 2^n`, values true integers in the
/// centered range, as RFC-0008 cuts a piecewise function: the runs of equal values at either end
/// are a constant times a difference of `LT`s, and only the values between are enumerated.
fn dense<F: PolyField<Base = Goldilocks>>(values: &[i64], r: &[F]) -> Result<F, PolyError> {
    let (Some(&first), Some(&last)) = (values.first(), values.last()) else {
        return Ok(F::ZERO);
    };
    let len = values.len();
    let lt = |a: usize| less_than(r, u64::try_from(a).unwrap_or(u64::MAX));
    let below = values.iter().position(|&v| v != first).unwrap_or(len);
    if below == len {
        return Ok(integer::<F>(first) * lt(len)?);
    }
    // A value other than `first` exists, so the run of `last` ends after `below`.
    let above = len - values.iter().rev().position(|&v| v != last).unwrap_or(0);
    let (first, last): (F, F) = (integer(first), integer(last));
    let ends = first * lt(below)? + last * (lt(len)? - lt(above)?);
    Ok(ends + range(&values[below..above], below, r)?)
}

/// `Σ_{h<len} eq(r, start + h)·values[h]`: the range `[start, start + len)` cut into aligned
/// blocks, each the largest that starts where the last ended and fits, at most two per bit. A
/// block of `2^k` rows shares their bits from `k` up, so it is `eq(r_high, start >> k)` times its
/// values evaluated at the `k` low coordinates. A table has at most 32 variables, so every shift
/// stays under 64.
fn range<F: PolyField<Base = Goldilocks>>(
    values: &[i64],
    start: usize,
    r: &[F],
) -> Result<F, PolyError> {
    let end = start + values.len();
    let mut at = start;
    let mut sum = F::ZERO;
    while at < end {
        let fits = (end - at).ilog2();
        let k = usize::try_from(at.trailing_zeros().min(fits)).unwrap_or(0);
        let block = &values[at - start..at - start + (1 << k)];
        let (low, high) = r.split_at(k);
        let prefix = u64::try_from(at >> k).unwrap_or(u64::MAX);
        sum += eq_at(high, prefix) * power_of_two_block(block, low)?;
        at += 1 << k;
    }
    Ok(sum)
}

/// `Σ_x eq(r, x)·values[x]` for `2^{|r|}` values: inner products of `2^12`-value chunks with the
/// `eq` table of the low coordinates, then the resulting table evaluated at the rest.
fn power_of_two_block<F: PolyField<Base = Goldilocks>>(
    values: &[i64],
    r: &[F],
) -> Result<F, PolyError> {
    let (low, high) = r.split_at(r.len().min(BLOCK_VARS));
    let eq_low = eq_table(low)?;
    let partial: Vec<F> = values
        .chunks_exact(eq_low.len())
        .map(|chunk| inner_product(chunk, &eq_low))
        .collect();
    evaluate(&partial, high)
}

/// `Σ_i eq[i]·values[i]` over equal-length slices, with four accumulators so that the additions
/// of consecutive products do not wait on each other.
fn inner_product<F: PolyField<Base = Goldilocks>>(values: &[i64], eq: &[F]) -> F {
    let term = |e: &F, &v: &i64| e.mul_by_base(Goldilocks::from_i64(v));
    let mut sums = [F::ZERO; 4];
    let (mut values4, mut eq4) = (values.chunks_exact(4), eq.chunks_exact(4));
    for (v, e) in (&mut values4).zip(&mut eq4) {
        sums[0] += term(&e[0], &v[0]);
        sums[1] += term(&e[1], &v[1]);
        sums[2] += term(&e[2], &v[2]);
        sums[3] += term(&e[3], &v[3]);
    }
    for (v, e) in values4.remainder().iter().zip(eq4.remainder()) {
        sums[0] += term(e, v);
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

/// A true integer as an element of `F`, in the centered representation.
fn integer<F: PolyField<Base = Goldilocks>>(x: i64) -> F {
    F::from(Goldilocks::from_i64(x))
}
