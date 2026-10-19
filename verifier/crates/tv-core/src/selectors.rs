// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public factors the stage checks compute themselves (RFC-0010 'Multilinear extensions',
//! 'Sumcheck messages'): the comparison `LT`, the causal selector `C̃` of RFC-0030, the row
//! selector `ψ̃` of RFC-0035, the layer weights `w̃` of a stacked instance (RFC-0037 'Stages and
//! their instances'), and the extension of `Rotate`'s kernel (RFC-0037 'Rotate').
//!
//! Every point lists its coordinates lowest variable first: coordinate `i` sets bit `i` of an
//! index, and `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`. Each function returns `None`
//! for inputs that name no value, such as a bound past its cube, which a caller's schedule never
//! passes: the caller treats it as a fault of its own.
//!
//! The closed forms here run from the highest coordinate down, a different traversal from the
//! one `tv-tables` takes for `LT` in its structured evaluation, from the lowest coordinate up.
//! None of them divides, so a coordinate 0 or 1 is no special case.
//!
//! - **`LT(r, a) = Σ_{h<a} eq(r, h)`** for `0 ≤ a ≤ 2^n` ([`lt`]). An `h < a` agrees with `a` on
//!   the bits above the highest bit `k` where they differ, has `h_k = 0` where `a_k = 1`, and any
//!   bits below `k`, over which `eq` sums to 1. So `LT` is `Σ_{k: a_k = 1} (1 − r_k)·Π_{j>k}
//!   eq(r_j, a_j)`, and `a = 2^n`, the whole cube, gives 1.
//! - **`C̃(q, i) = Σ_{i' ≤ q' < S} eq(q, q')·eq(i, i')`** over pairs of `m`-bit positions, for a
//!   sequence of `S ≤ 2^m` positions ([`causal`]). From the highest bit down, a pair of prefixes
//!   carries two comparisons, `q'` against `S` and `i'` against `q'`, each still equal or already
//!   less; a prefix already greater in either drops out. The sum is over the pairs whose `q'` ends
//!   less than `S` and whose `i'` ends less than or equal to `q'`.
//! - **`ψ̃ = LT(r_ℓ, n_layers)·LT(r_h, n_heads)·LT(r_t, n)`**, the row selector of RFC-0035, with
//!   the head and layer factors only for rows indexed by them ([`row_selector`]).
//! - **`w̃(ρ') = Σ_ℓ eq(ρ', ℓ)·w(ℓ)`** over the whole cube of `2^λ` layers ([`layer_weights`]):
//!   a padded layer's weight, such as `eq(ρ, ℓ)`, need not be 0 (RFC-0037, as amended).
//! - **The padding zero-check** of RFC-0035, `0 = Σ_x eq(z, x)·(1 − ψ(x))·w̃(x)` for a committed
//!   value `w` that a padded row leaves free ([`padding_zero_check`]): its final value
//!   `eq(z_i, r')·(1 − ψ̃(r'))·w̃(r')` at the instance's point `r'`, from the one evaluation
//!   `w̃(r')` it sends. It is of degree 3. The gadgets that need it (#122, #123, #126) put it in
//!   stage 1 (RFC-0037 'Stages and their instances').
//! - **`Ẽ_r(r')`** for `Rotate` on a tensor `[n, …, d_head]` of `n` positions, its pair bit `b`
//!   the highest of the last axis, `d_head/2 = 2^b` ([`rotate_kernel`]):
//!   `eq(r_mid, r'_mid)·(eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩)`, with
//!   `w(p, f) = eq(r_pos, p)·eq(r'_pos, p)·eq(r_f, f)·eq(r'_f, f)` over the `n` positions `p` and
//!   the `d_head/2` frequencies `f`, `C` and `S` RoPE's constants at position `p` and frequency
//!   `f`. RoPE's constants depend on neither the head nor the layer, so the middle axes'
//!   coordinates contribute `eq(r_mid, r'_mid)`, and a padded row's constants are 0: the inner
//!   products run over `n·d_head/2` entries.

use tv_field::{Fp, Fp2};
use tv_tables::rope::Rope;

/// `LT(point, bound) = Σ_{h < bound} eq(point, h)`, for `bound ≤ 2^n` with `n = |point|`, or
/// `None` for a larger bound.
pub fn lt(point: &[Fp2], bound: u64) -> Option<Fp2> {
    let vars = u32::try_from(point.len()).ok()?;
    match 1u64.checked_shl(vars) {
        Some(cube) if bound > cube => return None,
        Some(cube) if bound == cube => return Some(Fp2::ONE),
        // Past 63 coordinates every u64 bound is below the cube.
        _ => {}
    }
    let mut sum = Fp2::ZERO;
    // Π_{j > k} eq(r_j, a_j): the weight of the prefixes that agree with `bound` above bit k.
    let mut agree = Fp2::ONE;
    for (k, &r) in point.iter().enumerate().rev() {
        if bit(bound, k) {
            sum += agree * (Fp2::ONE - r);
            agree *= r;
        } else {
            agree *= Fp2::ONE - r;
        }
    }
    Some(sum)
}

/// `C̃(query, key)` for a sequence of `length` positions: `Σ_{i' ≤ q' < length}
/// eq(query, q')·eq(key, i')`, over positions of `m = |query| = |key|` bits; `None` when the
/// points differ in length or `length > 2^m`.
pub fn causal(query: &[Fp2], key: &[Fp2], length: u64) -> Option<Fp2> {
    if query.len() != key.len() {
        return None;
    }
    let vars = u32::try_from(query.len()).ok()?;
    // Every q' of the cube lies below a length of 2^m: the comparison with the length starts
    // decided.
    let whole = match 1u64.checked_shl(vars) {
        Some(cube) if length > cube => return None,
        Some(cube) => length == cube,
        None => false,
    };
    // The weight of the prefix pairs in each state: [q' vs length][i' vs q'], 0 for still equal
    // and 1 for already less.
    let mut states = [[Fp2::ZERO; 2]; 2];
    let start = usize::from(whole);
    if let Some(cell) = states.get_mut(start).and_then(|row| row.first_mut()) {
        *cell = Fp2::ONE;
    }
    for (k, (&q, &i)) in query.iter().zip(key).enumerate().rev() {
        let s = bit(length, k);
        let mut next = [[Fp2::ZERO; 2]; 2];
        for (q_less, row) in states.iter().enumerate() {
            for (i_less, &weight) in row.iter().enumerate() {
                if weight == Fp2::ZERO {
                    continue;
                }
                for q_bit in [false, true] {
                    // q' against the length: once less, always less; while equal, a bit below
                    // the length's makes it less, and a bit above drops the pair.
                    let q_state = match (q_less == 1, q_bit, s) {
                        (true, _, _) => 1,
                        (false, false, true) => 1,
                        (false, b, s) if b == s => 0,
                        _ => continue,
                    };
                    let q_factor = if q_bit { q } else { Fp2::ONE - q };
                    for i_bit in [false, true] {
                        // i' against q': the same rule, against this pair's q' bit.
                        let i_state = match (i_less == 1, i_bit, q_bit) {
                            (true, _, _) => 1,
                            (false, false, true) => 1,
                            (false, b, a) if b == a => 0,
                            _ => continue,
                        };
                        let i_factor = if i_bit { i } else { Fp2::ONE - i };
                        let cell = next.get_mut(q_state)?.get_mut(i_state)?;
                        *cell += weight * q_factor * i_factor;
                    }
                }
            }
        }
        states = next;
    }
    // q' < length, and i' ≤ q' whether equal or less.
    let [_, [equal, less]] = states;
    Some(equal + less)
}

/// `ψ̃` of RFC-0035 'The row selector' at a row point: `LT(rows, length)`, times
/// `LT(heads, n_heads)` for per-head rows and `LT(layers, n_layers)` for rows stacked over
/// layers; `None` when a bound passes its cube.
pub fn row_selector(
    rows: &[Fp2],
    length: u64,
    heads: Option<(&[Fp2], u64)>,
    layers: Option<(&[Fp2], u64)>,
) -> Option<Fp2> {
    let mut value = lt(rows, length)?;
    for (point, bound) in heads.into_iter().chain(layers) {
        value *= lt(point, bound)?;
    }
    Some(value)
}

/// The rows of a gadget's committed values, as its row selector reads them (RFC-0035 'The row
/// selector'): the tokens' coordinates lowest, then the heads', then the layers', each with its
/// number of variables and of real rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rows {
    /// The tokens' variables and the sequence's length `n`.
    pub tokens: (usize, u64),
    /// The heads' variables and `n_heads`, for per-head rows.
    pub heads: Option<(usize, u64)>,
    /// The layers' variables and `n_layers`, for rows stacked over the layers.
    pub layers: Option<(usize, u64)>,
}

/// The final value of RFC-0035's padding zero-check of a committed value `w` at the instance's
/// point `point`, `r'`: `eq(zero, r')·(1 − ψ̃(r'))·w̃(r')`, `zero` being the instance's last
/// coordinates of its stage's zero-check point and `w` its one evaluation. `None` when the points'
/// lengths do not fit `rows`.
pub fn padding_zero_check(zero: &[Fp2], point: &[Fp2], rows: &Rows, w: Fp2) -> Option<Fp2> {
    let (tokens, rest) = point.split_at_checked(rows.tokens.0)?;
    let (heads, rest) = rest.split_at_checked(rows.heads.map_or(0, |h| h.0))?;
    let (layers, rest) = rest.split_at_checked(rows.layers.map_or(0, |l| l.0))?;
    if !rest.is_empty() {
        return None;
    }
    let psi = row_selector(
        tokens,
        rows.tokens.1,
        rows.heads.map(|(_, count)| (heads, count)),
        rows.layers.map(|(_, count)| (layers, count)),
    )?;
    Some(eq_points(zero, point)? * (Fp2::ONE - psi) * w)
}

/// `w̃(point) = Σ_ℓ eq(point, ℓ)·w(ℓ)`: the extension of the layer weights `weights`, one per
/// layer of the whole cube of `2^λ`, `λ = |point|`, at `point`; `None` for another number of
/// weights.
///
/// The weights are folded one variable at a time, the lowest first: each pair `(w_{2u},
/// w_{2u+1})` becomes `w_{2u} + ρ'_i·(w_{2u+1} − w_{2u})`.
pub fn layer_weights(weights: &[Fp2], point: &[Fp2]) -> Option<Fp2> {
    let cube = 1usize.checked_shl(u32::try_from(point.len()).ok()?)?;
    if weights.len() != cube {
        return None;
    }
    let mut folded = weights.to_vec();
    for &r in point {
        let half = folded.len().checked_div(2)?;
        for u in 0..half {
            let low = *folded.get(u.checked_mul(2)?)?;
            let high = *folded.get(u.checked_mul(2)?.checked_add(1)?)?;
            *folded.get_mut(u)? = low + r * (high - low);
        }
        folded.truncate(half);
    }
    folded.first().copied()
}

/// `Ẽ_r(r')`, the extension at `point` (`r'`) of `Rotate`'s kernel built from the claim's point
/// `claim` (`r`), for a tensor `[positions, …, d_head]` whose middle axes take `middle_vars`
/// coordinates, with `d_head = 2·|rope.frequencies()|` (RFC-0037 'Rotate').
///
/// `None` when the points differ in length or are shorter than `b + 1 + middle_vars`, when
/// `d_head/2` is not a power of two, or when the positions pass the table's or the position
/// coordinates' range.
pub fn rotate_kernel(
    rope: &Rope,
    positions: u32,
    middle_vars: usize,
    claim: &[Fp2],
    point: &[Fp2],
) -> Option<Fp2> {
    if claim.len() != point.len() || positions > rope.positions() {
        return None;
    }
    let half = rope.frequencies().len();
    if !half.is_power_of_two() {
        return None;
    }
    // The pair bit b, d_head/2 = 2^b, is the highest of the last axis's b + 1 coordinates.
    let b = usize::try_from(half.trailing_zeros()).ok()?;
    let low = b.checked_add(1)?.checked_add(middle_vars)?;
    let position_vars = claim.len().checked_sub(low)?;
    let position_cube = 1u64.checked_shl(u32::try_from(position_vars).ok()?)?;
    if u64::from(positions) > position_cube {
        return None;
    }
    let (claim_f, claim_rest) = claim.split_at_checked(b)?;
    let (point_f, point_rest) = point.split_at_checked(b)?;
    let (&claim_b, claim_rest) = claim_rest.split_first()?;
    let (&point_b, point_rest) = point_rest.split_first()?;
    let (claim_mid, claim_pos) = claim_rest.split_at_checked(middle_vars)?;
    let (point_mid, point_pos) = point_rest.split_at_checked(middle_vars)?;

    let frequency_weights = joint_eq(claim_f, point_f)?;
    let position_weights = joint_eq(claim_pos, point_pos)?;
    let (mut cosines, mut sines) = (Fp2::ZERO, Fp2::ZERO);
    for position in 0..positions {
        let weight = *position_weights.get(usize::try_from(position).ok()?)?;
        let (mut c_sum, mut s_sum) = (Fp2::ZERO, Fp2::ZERO);
        for (frequency, &f_weight) in frequency_weights.iter().enumerate() {
            let (c, s) = rope.pair(position, frequency)?;
            c_sum += f_weight * signed(c);
            s_sum += f_weight * signed(s);
        }
        cosines += weight * c_sum;
        sines += weight * s_sum;
    }
    let pair_eq = claim_b * point_b + (Fp2::ONE - claim_b) * (Fp2::ONE - point_b);
    Some(eq_points(claim_mid, point_mid)? * (pair_eq * cosines + (claim_b - point_b) * sines))
}

/// `eq(a, ·)·eq(b, ·)` over the cube of `|a| = |b|` variables, entry `x` for each index `x`:
/// built a coordinate at a time, each doubling the table with the factors `(1 − a_i)(1 − b_i)`
/// for bit `i` clear and `a_i·b_i` for it set.
fn joint_eq(a: &[Fp2], b: &[Fp2]) -> Option<Vec<Fp2>> {
    if a.len() != b.len() {
        return None;
    }
    let size = 1usize.checked_shl(u32::try_from(a.len()).ok()?)?;
    let mut table = Vec::new();
    table.try_reserve_exact(size).ok()?;
    table.push(Fp2::ONE);
    for (&x, &y) in a.iter().zip(b) {
        let clear = (Fp2::ONE - x) * (Fp2::ONE - y);
        let set = x * y;
        let filled = table.len();
        for u in 0..filled {
            let value = *table.get(u)?;
            table.push(value * set);
            *table.get_mut(u)? = value * clear;
        }
    }
    Some(table)
}

/// `eq(a, b)`, or `None` for points of different lengths.
fn eq_points(a: &[Fp2], b: &[Fp2]) -> Option<Fp2> {
    crate::sumcheck::eq(a, b)
}

/// Bit `k` of `x`, false past bit 63.
fn bit(x: u64, k: usize) -> bool {
    u32::try_from(k)
        .ok()
        .and_then(|k| x.checked_shr(k))
        .is_some_and(|shifted| shifted & 1 == 1)
}

/// A signed integer in the field, centered: `v` for `v ≥ 0`, `P − |v|` below (RFC-0002 'Signed
/// integers'), for `|v| < P`.
pub(crate) fn signed(v: i64) -> Fp2 {
    let magnitude = Fp2::from(Fp::reduce(v.unsigned_abs()));
    if v < 0 { -magnitude } else { magnitude }
}
