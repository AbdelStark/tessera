// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The pieces of structured evaluation (RFC-0008 'Structured evaluation'), over a point `ρ` that
//! lists its coordinates lowest variable first: `eq` at one index, `LT` with the prefix sum of
//! an affine function of the index bits, and the extension of a dense prefix.
//!
//! None of them divides, so a coordinate 0 or 1 is no special case.

use core::mem;

use tv_field::{Fp, Fp2};

use crate::TableError;

/// The variables a dense core folds a chunk at a time: chunks of `2^10` values.
const CHUNK_VARS: usize = 10;

/// `eq(ρ, index) = Π_i (ρ_i·j_i + (1 − ρ_i)·(1 − j_i))` with `j_i` bit `i` of `index`; bits at
/// and above `|ρ|` are not read.
pub(crate) fn eq_at(rho: &[Fp2], index: u64) -> Fp2 {
    let mut rest = index;
    let mut product = Fp2::ONE;
    for &r in rho {
        product *= if rest & 1 == 1 { r } else { Fp2::ONE - r };
        rest >>= 1;
    }
    product
}

/// Over the `n = |ρ|` variables of a cube and for `a ≤ 2^n`: `LT(ρ, a) = Σ_{h<a} eq(ρ, h)`
/// and `Σ_{h<a} eq(ρ, h)·(w_0 + Σ_i w_i·h_i)`, the prefix sum of an affine function of the
/// bits of `h`, with `w_i = weights[i]`. `None` when `a > 2^n` or `weights` is shorter than `ρ`.
///
/// One pass from the lowest bit up, in `O(n)`. After bit `k`, `lt` is `LT` of the bound `a mod
/// 2^{k+1}` over the first `k + 1` variables, `low` the matching prefix sum of `Σ_{i≤k} w_i·h_i`,
/// and `all` the sum of that function over the whole subcube, `Σ_{i≤k} w_i·ρ_i`. The `h` below
/// `a mod 2^{k+1}` either have `h_k = 0` and any lower bits, when `a_k = 1`, or `h_k = a_k` and
/// lower bits below `a mod 2^k`:
///
/// - `a_k = 1`: `lt ← (1 − ρ_k) + ρ_k·lt` and `low ← (1 − ρ_k)·all + ρ_k·(low + w_k·lt)`;
/// - `a_k = 0`: `lt ← (1 − ρ_k)·lt` and `low ← (1 − ρ_k)·low`.
///
/// `a = 2^n` is the whole cube: `LT = 1` and the sum is `w_0 + Σ_i w_i·ρ_i`.
pub(crate) fn prefix(rho: &[Fp2], a: u64, w0: Fp, weights: &[Fp]) -> Option<(Fp2, Fp2)> {
    if weights.len() < rho.len() {
        return None;
    }
    let (mut lt, mut low, mut all) = (Fp2::ZERO, Fp2::ZERO, Fp2::ZERO);
    let mut rest = a;
    for (&r, &w) in rho.iter().zip(weights) {
        let below = Fp2::ONE - r;
        if rest & 1 == 1 {
            low = below * all + r * (low + lt * w);
            lt = below + r * lt;
        } else {
            low = below * low;
            lt = below * lt;
        }
        all += r * w;
        rest >>= 1;
    }
    match rest {
        // `a < 2^n`.
        0 => Some((lt, lt * w0 + low)),
        // `a = 2^n`, the whole cube.
        1 if a.trailing_zeros() == width(rho) => Some((Fp2::ONE, all + Fp2::from(w0))),
        _ => None,
    }
}

/// `|ρ|` as a `u32`, saturated: a point never has `2^32` coordinates.
fn width(rho: &[Fp2]) -> u32 {
    u32::try_from(rho.len()).unwrap_or(u32::MAX)
}

/// `Σ_{h < len} eq(ρ, h)·v_h`: the extension, over the `n = |ρ|` variables, of the vector that
/// holds `values` (`len` of them, embedded by `embed`) and zeros after them, `len ≤ 2^n`; a
/// longer vector is `Internal`, with `2^n` computed without wrapping.
///
/// The vector is folded one variable at a time, the lowest first, each pair `(f_{2u}, f_{2u+1})`
/// becoming `f_{2u} + ρ_i·(f_{2u+1} − f_{2u})`, and a value past the end counting as 0. Chunks
/// of `2^10` values are folded over the low variables first, each to one value, and those values
/// over the others: `O(len)` products with `O(len / 2^10)` values in memory beside the core.
pub(crate) fn dense<T: Copy>(
    values: &[T],
    rho: &[Fp2],
    embed: impl Fn(T) -> Fp,
) -> Result<Fp2, TableError> {
    // The cube holds `2^n` values. `1 << n` is exact for `n < usize::BITS`, and for a wider
    // cube there is no shift: it holds more values than any slice.
    let capacity = 1usize.checked_shl(width(rho));
    if capacity.is_some_and(|capacity| values.len() > capacity) {
        return Err(TableError::Internal);
    }
    let (inner, outer) = rho.split_at(rho.len().min(CHUNK_VARS));
    let chunk = 1usize
        .checked_shl(width(inner))
        .ok_or(TableError::Internal)?;
    let chunks = values.len().div_ceil(chunk);
    let mut partial = reserve(chunks)?;
    let mut spare = reserve(chunks.div_ceil(2))?;
    let half = chunk.div_ceil(2);
    let mut scratch = reserve(half)?;
    let mut scratch_spare = reserve(half.div_ceil(2))?;
    for block in values.chunks(chunk) {
        partial.push(fold_block(
            block,
            inner,
            &embed,
            &mut scratch,
            &mut scratch_spare,
        ));
    }
    for &r in outer {
        fold(&mut partial, r, &mut spare);
    }
    Ok(partial.first().copied().unwrap_or(Fp2::ZERO))
}

/// An empty vector with room for `len` values, or `Memory`.
fn reserve(len: usize) -> Result<Vec<Fp2>, TableError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| TableError::Memory)?;
    Ok(values)
}

/// The extension of one chunk over the `|inner|` variables, at most `2^|inner|` values.
fn fold_block<T: Copy>(
    block: &[T],
    inner: &[Fp2],
    embed: &impl Fn(T) -> Fp,
    scratch: &mut Vec<Fp2>,
    spare: &mut Vec<Fp2>,
) -> Fp2 {
    let Some((&first, rest)) = inner.split_first() else {
        // No variable: a chunk of one value.
        return block
            .first()
            .map_or(Fp2::ZERO, |&value| Fp2::from(embed(value)));
    };
    scratch.clear();
    scratch.extend(block.chunks(2).map(|pair| match *pair {
        [low, high] => {
            let low = embed(low);
            Fp2::from(low) + first * (embed(high) - low)
        }
        [low] => {
            let low = embed(low);
            Fp2::from(low) - first * low
        }
        _ => Fp2::ZERO,
    }));
    for &r in rest {
        fold(scratch, r, spare);
    }
    scratch.first().copied().unwrap_or(Fp2::ZERO)
}

/// Binds the lowest variable of `values` to `r`, halving it (rounded up); `spare` is scratch
/// space that ends up with the old values.
fn fold(values: &mut Vec<Fp2>, r: Fp2, spare: &mut Vec<Fp2>) {
    spare.clear();
    spare.extend(values.chunks(2).map(|pair| match *pair {
        [low, high] => low + r * (high - low),
        [low] => low - r * low,
        _ => Fp2::ZERO,
    }));
    mem::swap(values, spare);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(seed: u64, n: usize) -> Vec<Fp2> {
        (0..n as u64)
            .map(|i| {
                let a = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i * 7919);
                Fp2::new(Fp::reduce(a), Fp::reduce(a.rotate_left(17) ^ 0xABCD))
            })
            .collect()
    }

    /// `Σ_{h<a} eq(ρ, h)·f(h)` by enumeration.
    fn brute(rho: &[Fp2], a: u64, f: impl Fn(u64) -> Fp) -> Fp2 {
        (0..a).fold(Fp2::ZERO, |sum, h| sum + eq_at(rho, h) * f(h))
    }

    #[test]
    fn prefix_sums_match_enumeration_for_every_bound() {
        for n in 0..=6usize {
            let rho = point(n as u64 + 1, n);
            let weights: Vec<Fp> = (0..n as u64).map(|i| Fp::reduce(3 * i + 2)).collect();
            let w0 = Fp::reduce(11);
            let affine = |h: u64| {
                (0..n).fold(w0, |sum, i| {
                    if (h >> i) & 1 == 1 {
                        sum + weights[i]
                    } else {
                        sum
                    }
                })
            };
            for a in 0..=(1u64 << n) {
                let (lt, sum) = prefix(&rho, a, w0, &weights).unwrap();
                assert_eq!(lt, brute(&rho, a, |_| Fp::ONE), "LT, n = {n}, a = {a}");
                assert_eq!(sum, brute(&rho, a, affine), "sum, n = {n}, a = {a}");
            }
            assert_eq!(prefix(&rho, (1 << n) + 1, w0, &weights), None);
        }
    }

    #[test]
    fn dense_prefixes_match_enumeration_across_chunks() {
        for n in [0usize, 1, 3, 10, 11, 13] {
            let rho = point(100 + n as u64, n);
            for len in [0u64, 1, 2, 3, (1 << n) / 2 + 1, 1 << n] {
                let len = len.min(1 << n);
                let values: Vec<u64> = (0..len).map(|h| h * h + 5).collect();
                let expected = brute(&rho, len, |h| Fp::reduce(h * h + 5));
                assert_eq!(
                    dense(&values, &rho, Fp::reduce).unwrap(),
                    expected,
                    "n = {n}, len = {len}"
                );
            }
        }
        let too_long = [1u64; 3];
        assert_eq!(
            dense(&too_long, &point(1, 1), Fp::reduce),
            Err(TableError::Internal)
        );
    }

    /// The room of a cube, `2^{|ρ|}`, is computed without wrapping: at `usize::BITS` variables
    /// or more it exceeds every length, and a short vector is still admitted. On a 32-bit target
    /// this is a table's 32 variables.
    #[test]
    fn dense_admits_short_vectors_on_cubes_wider_than_a_usize() {
        let values = [3u64, 5, 7];
        let width = usize::BITS as usize;
        for n in [width - 1, width, width + 1, width + 6] {
            let rho = point(7 + n as u64, n);
            assert_eq!(
                dense(&values, &rho, Fp::reduce),
                Ok(brute(&rho, 3, |h| Fp::reduce(values[h as usize]))),
                "n = {n}"
            );
        }
    }
}
