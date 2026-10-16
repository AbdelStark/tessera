// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public predicates whose multilinear extensions the prover and the verifier each evaluate
//! in `O(n)` (RFC-0010 'Multilinear extensions'): less-than against an integer bound
//! (RFC-0008 'Structured evaluation'), less-than between two positions, and the causal selector
//! of attention (RFC-0009 'SoftmaxRow').
//!
//! Each closed form is a recursion from the lowest bit up that compares integers from the most
//! significant bit down. It equals its defining sum of `eq` products, which is multilinear (a sum
//! of multilinear terms) and equals the predicate on the hypercube; the rustdoc of each function
//! derives the recursion from the sum.

use tessera_field::Field;

use crate::PolyError;
use crate::eq::same_length;

/// `LT(r, a) = Σ_{h < a} eq(r, h)` over the `n`-bit integers `h`, for `0 ≤ a ≤ 2^n`: the
/// multilinear extension of the indicator of `[0, a)`, evaluated at `r` with one multiplication
/// per coordinate. `LT(r, 0) = 0` and `LT(r, 2^n) = 1`.
///
/// Let `T_j` be the same sum over the `j` lowest variables, with `h` ranging over
/// `[0, a mod 2^j)`, and split `h` by its top bit `h_j`. The `eq` factors of the free lower bits
/// sum to 1, so:
///
/// - when the bit of `a` at `x_j` (bit `j − 1`) is 1, every `h` with `h_j = 0` lies below, and
///   the ones with `h_j = 1` compare by their lower bits: `T_j = (1 − r_j) + r_j·T_{j−1}`;
/// - when it is 0, only `h_j = 0` can lie below, by its lower bits: `T_j = (1 − r_j)·T_{j−1}`.
///
/// With `T_0 = 0` this gives `T_n = LT(r, a)` for `a < 2^n`; `a = 2^n` covers the whole cube,
/// whose `eq` weights sum to 1. Each step is affine in `r_j`, and `T_{j−1}` does not involve
/// `r_j`, so the closed form is multilinear; at a Boolean `r = x`, `eq(x, h) = [x = h]`, so it
/// equals `[x < a]`. Coordinates beyond the 64th see zero bits of `a`. `a` is public: the
/// branches on its bits leak nothing.
///
/// # Errors
///
/// [`PolyError::BoundTooLarge`] when `a > 2^n`.
pub fn less_than<F: Field>(r: &[F], a: u64) -> Result<F, PolyError> {
    let vars = r.len();
    if vars < 64 {
        let cube = 1u64 << vars;
        if a > cube {
            return Err(PolyError::BoundTooLarge { bound: a, vars });
        }
        if a == cube {
            return Ok(F::ONE);
        }
    }
    let mut below = F::ZERO;
    for (j, &rj) in r.iter().enumerate() {
        let bit = j < 64 && (a >> j) & 1 == 1;
        below = if bit {
            F::ONE - rj * (F::ONE - below)
        } else {
            below - rj * below
        };
    }
    Ok(below)
}

/// `LT2(x, y) = Σ_{x' < y'} eq(x, x')·eq(y, y')` over pairs of `n`-bit integers: the multilinear
/// extension, in `2n` variables, of the predicate `x' < y'`, evaluated with two multiplications
/// per coordinate.
///
/// From the lowest bit up, `L_0 = 0` and
///
/// ```text
/// L_j = (1 − x_j)·y_j + (x_j·y_j + (1 − x_j)·(1 − y_j))·L_{j−1}
/// ```
///
/// Split the pairs by their top bits `(x'_j, y'_j)`: `x'_j = 0 < 1 = y'_j` puts `x'` below `y'`
/// whatever the lower bits, whose `eq` factors sum to 1, and equal top bits defer to the lower
/// bits. Both coefficients have degree at most one in `x_j` and in `y_j`, and `L_{j−1}` involves
/// neither, so `L_n` is multilinear. On the hypercube the coefficients are the indicators
/// `[x'_j < y'_j]` and `[x'_j = y'_j]`, so `L_n` is the comparison from the top bit down,
/// `[x' < y']`. A multilinear polynomial is determined by its values on the hypercube, so `L_n` is
/// the extension of the predicate: the defining sum.
///
/// # Errors
///
/// [`PolyError::PointLength`] when the points differ in length.
pub fn less_than_points<F: Field>(x: &[F], y: &[F]) -> Result<F, PolyError> {
    same_length(x, y)?;
    Ok(compare(x, y, F::ZERO))
}

/// The causal selector `C(q, i) = Σ_{i' ≤ q'} eq(q, q')·eq(i, i')` over pairs of `n`-bit
/// positions: the multilinear extension of the causal mask, under which key position `i'` is
/// visible from query position `q'` when `i' ≤ q'` (RFC-0009 'SoftmaxRow'). Two multiplications
/// per coordinate.
///
/// `C(q, i) = LT2(i, q) + eq(i, q)`. It is the recursion of [`less_than_points`] with `x = i`,
/// `y = q` and `C_0 = 1` in place of `L_0 = 0`: the empty comparison is an equality, and
/// `0 ≤ 0`. The recursion is affine in its starting value with slope `Π_j eq(i_j, q_j) =
/// eq(i, q)`, which gives the identity, and the argument of [`less_than_points`] applies
/// unchanged: the closed form is multilinear and equals `[i' ≤ q']` on the hypercube.
///
/// # Errors
///
/// [`PolyError::PointLength`] when the points differ in length (`expected` is `q`'s).
pub fn causal_selector<F: Field>(q: &[F], i: &[F]) -> Result<F, PolyError> {
    same_length(q, i)?;
    Ok(compare(i, q, F::ONE))
}

/// `LT2(x, y) + equal·eq(x, y)`: the recursion of [`less_than_points`] started at `equal`.
fn compare<F: Field>(x: &[F], y: &[F], equal: F) -> F {
    x.iter().zip(y).fold(equal, |lower, (&xj, &yj)| {
        let both = xj * yj;
        // (1 − x_j)·y_j: x'_j = 0 and y'_j = 1, below whatever the lower bits.
        let below = yj - both;
        // x_j·y_j + (1 − x_j)·(1 − y_j): equal top bits, ordered by the lower bits.
        let same = both.double() - xj - yj + F::ONE;
        below + same * lower
    })
}
