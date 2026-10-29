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

/// `eq(r, index) = Π_j (r_j·b_j + (1 − r_j)·(1 − b_j))` at a public integer vertex `index`,
/// evaluated without allocating.
fn eq_index<F: Field>(r: &[F], index: usize) -> F {
    r.iter().enumerate().fold(F::ONE, |acc, (j, &rj)| {
        let bit = j < usize::BITS as usize && ((index >> j) & 1) == 1;
        acc * if bit { rj } else { F::ONE - rj }
    })
}

/// The number of request-slot coordinates `⌈log2 R⌉` above the `b` intra-slot coordinates
/// (RFC-0071 'The slots'): `0` when `R ≤ 1`.
fn request_vars(requests: usize) -> usize {
    if requests <= 1 {
        0
    } else {
        (requests - 1).ilog2() as usize + 1
    }
}

/// `C_a(q, i) = Σ_{0 ≤ i' ≤ q' < a} eq(q, q')·eq(i, i')` over pairs of `n`-bit positions, for
/// `0 ≤ a ≤ 2^n`: the multilinear extension of the causal prefix restricted to real queries
/// `q' < a` within a slot (RFC-0071 'The selectors and the causal prefix').
///
/// Scan bits from the most significant (`n − 1`) down to `0`, tracking the four feasible prefix
/// orderings `(cmp(i_hi, q_hi), cmp(q_hi, a_hi))` in `{=, <} × {=, <}`. Each step multiplies
/// previous states by bilinear weights of degree at most one in `i_j` and in `q_j`, so the result
/// is multilinear and equals `[i' ≤ q' ∧ q' < a]` on `{0, 1}^{2n}`. `a` is public (INV-ARCH-2).
///
/// # Errors
///
/// [`PolyError::PointLength`] when `q` and `i` differ in length; [`PolyError::BoundTooLarge`] when
/// `a > 2^n`.
pub fn causal_selector_bounded<F: Field>(q: &[F], i: &[F], bound: u64) -> Result<F, PolyError> {
    same_length(q, i)?;
    let vars = q.len();
    if vars < 64 {
        let cube = 1u64 << vars;
        if bound > cube {
            return Err(PolyError::BoundTooLarge { bound, vars });
        }
        if bound == cube {
            return causal_selector(q, i);
        }
    }
    if bound == 0 {
        return Ok(F::ZERO);
    }
    let (mut eq_lt, mut lt_lt, mut eq_eq, mut lt_eq) = (F::ZERO, F::ZERO, F::ONE, F::ZERO);
    for bit_idx in (0..vars).rev() {
        let kc = i[bit_idx];
        let qp = q[bit_idx];
        let not_kc = F::ONE - kc;
        let not_qp = F::ONE - qp;
        let w00 = not_kc * not_qp;
        let w01 = not_kc * qp;
        let w11 = kc * qp;
        let bound_bit = bit_idx < 64 && ((bound >> bit_idx) & 1) == 1;
        let mut next_lt_lt = lt_lt + eq_lt * w01;
        let mut next_eq_lt = eq_lt * (w00 + w11);
        let (next_lt_eq, next_eq_eq) = if bound_bit {
            next_lt_lt += lt_eq * not_qp;
            next_eq_lt += eq_eq * w00;
            (lt_eq * qp + eq_eq * w01, eq_eq * w11)
        } else {
            (lt_eq * not_qp, eq_eq * w00)
        };
        eq_lt = next_eq_lt;
        lt_lt = next_lt_lt;
        eq_eq = next_eq_eq;
        lt_eq = next_lt_eq;
    }
    Ok(eq_lt + lt_lt)
}

/// The batched slot row selector `ψ̃(r_p, r_β) = Σ_{r < R} eq(r_β, r)·LT(r_p, n_r)` over a token
/// axis of `R` slots of width `S = 2^{slot_bits}` (RFC-0071 'The selectors and the causal
/// prefix').
///
/// `r_token` lists the `slot_bits` intra-slot position coordinates lowest, followed by
/// `⌈log2 R⌉` request-slot coordinates (`0` coordinates when `R ≤ 1`, where this reduces to
/// [`less_than`]).
///
/// # Errors
///
/// [`PolyError::PointLength`] when `r_token.len() != slot_bits + ⌈log2 R⌉`;
/// [`PolyError::BoundTooLarge`] when any `n_r > 2^{slot_bits}`.
pub fn slot_selector<F: Field>(
    r_token: &[F],
    slot_bits: usize,
    lengths: &[u64],
) -> Result<F, PolyError> {
    let req_vars = request_vars(lengths.len());
    let expected = slot_bits
        .checked_add(req_vars)
        .ok_or(PolyError::PointLength {
            expected: usize::MAX,
            found: r_token.len(),
        })?;
    if r_token.len() != expected {
        return Err(PolyError::PointLength {
            expected,
            found: r_token.len(),
        });
    }
    match lengths {
        [] => Ok(F::ZERO),
        [only] => less_than(r_token, *only),
        many => {
            let (r_p, r_beta) = r_token.split_at(slot_bits);
            let mut sum = F::ZERO;
            for (r, &n_r) in many.iter().enumerate() {
                sum += eq_index(r_beta, r) * less_than(r_p, n_r)?;
            }
            Ok(sum)
        }
    }
}

/// The batched intra-slot causal selector
/// `C̃((r_p, r_β), r_c) = Σ_{r < R} eq(r_β, r)·C_{n_r}(r_p, r_c)` (RFC-0071 'The selectors and the
/// causal prefix'): compares a query's intra-slot position `r_p` against a key's `r_c` within one
/// slot `r_β`, and vanishes on padded query rows (`p ≥ n_r` or `β ≥ R`).
///
/// # Errors
///
/// [`PolyError::PointLength`] when `r_token.len() != r_key.len() + ⌈log2 R⌉`;
/// [`PolyError::BoundTooLarge`] when any `n_r > 2^{r_key.len()}`.
pub fn batch_causal_selector<F: Field>(
    r_token: &[F],
    r_key: &[F],
    lengths: &[u64],
) -> Result<F, PolyError> {
    let slot_bits = r_key.len();
    let req_vars = request_vars(lengths.len());
    let expected = slot_bits
        .checked_add(req_vars)
        .ok_or(PolyError::PointLength {
            expected: usize::MAX,
            found: r_token.len(),
        })?;
    if r_token.len() != expected {
        return Err(PolyError::PointLength {
            expected,
            found: r_token.len(),
        });
    }
    match lengths {
        [] => Ok(F::ZERO),
        [only] => causal_selector_bounded(r_token, r_key, *only),
        many => {
            let (r_p, r_beta) = r_token.split_at(slot_bits);
            let mut sum = F::ZERO;
            for (r, &n_r) in many.iter().enumerate() {
                sum += eq_index(r_beta, r) * causal_selector_bounded(r_p, r_key, n_r)?;
            }
            Ok(sum)
        }
    }
}
