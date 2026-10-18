// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The calculator's arithmetic (RFC-0014 'Soundness calculator'), integers only: LOG2, a lower
//! bound on `log2(a/b)` in fixed point that rounds down at every step, and the exact comparisons
//! that decide each row. No floating point enters it, so no platform's `log2` can move a bit.
//!
//! A row's error is one of two forms:
//!
//! - **field**, `ε = c/q` for an integer numerator `c` and `q = |Fp2| = p^2 < 2^128`;
//! - **queries**, `ε = (B/A)^τ` with `A = 2n`, `B = n + k + P`: `((1 + ρ')/2)^τ` for
//!   `ρ' = (k + P)/n`.
//!
//! A proof of work of `g` bits adds `g`, and since a probability is at most 1, an error above 1
//! counts as 1: the row's bits are `λ = g + max(0, −log2 ε)`.

use core::cmp::Ordering;

use tessera_field::Goldilocks;

use crate::FixedBits;

/// LOG2's working precision: a fixed-point value has `W` fractional bits, so that its square
/// fits a `u128`.
pub(crate) const W: u32 = 62;

/// The fractional bits of a field-form row's LOG2: FixedBits' own, `2^-8` bits.
pub(crate) const FIELD_FRACTION: u32 = 8;

/// The fractional bits of the per-query LOG2, before it is multiplied by `τ`: `τ ≤ 2^16` then
/// loses at most `2^-24` bits more than rounding to FixedBits does.
pub(crate) const QUERY_FRACTION: u32 = 48;

/// FixedBits per bit.
pub(crate) const UNIT: u64 = 256;

/// `q = |Fp2| = p^2`, `p = 2^64 − 2^32 + 1` (RFC-0002): below `2^128`, so a numerator of `2^k`
/// gives a hair less than `128 − k` bits.
pub(crate) const Q: u128 = (Goldilocks::ORDER as u128) * (Goldilocks::ORDER as u128);

/// A natural number: 64-bit limbs, the least significant first, with no zero limb on top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Nat(Vec<u64>);

impl Nat {
    /// `x`.
    pub(crate) fn from_u128(x: u128) -> Self {
        Self(vec![x as u64, (x >> 64) as u64]).trimmed()
    }

    fn trimmed(mut self) -> Self {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
        self
    }

    /// `self · 2^shift`.
    pub(crate) fn shl(&self, shift: u64) -> Self {
        if self.0.is_empty() {
            return self.clone();
        }
        let (limbs, bits) = ((shift / 64) as usize, (shift % 64) as u32);
        let mut out = vec![0u64; limbs];
        out.reserve(self.0.len() + 1);
        let mut carry = 0u64;
        for &limb in &self.0 {
            out.push((limb << bits) | carry);
            // A shift by 64 − 0 would overflow: no bit crosses limbs when `bits` is 0.
            carry = if bits == 0 { 0 } else { limb >> (64 - bits) };
        }
        out.push(carry);
        Self(out).trimmed()
    }

    /// `self − other`, for `self ≥ other`.
    fn sub(&self, other: &Self) -> Self {
        debug_assert!(*self >= *other);
        let mut out = self.0.clone();
        let mut borrow = false;
        for (i, limb) in out.iter_mut().enumerate() {
            let subtrahend = other.0.get(i).copied().unwrap_or(0);
            let (d1, b1) = limb.overflowing_sub(subtrahend);
            let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
            *limb = d2;
            borrow = b1 || b2;
        }
        debug_assert!(!borrow);
        Self(out).trimmed()
    }

    /// `self · other`, schoolbook.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        if self.0.is_empty() || other.0.is_empty() {
            return Self(Vec::new());
        }
        let mut out = vec![0u64; self.0.len() + other.0.len()];
        for (i, &x) in self.0.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in other.0.iter().enumerate() {
                // x·y + out + carry < 2^128: (2^64 − 1)^2 + 2·(2^64 − 1) = 2^128 − 1.
                let t = u128::from(x) * u128::from(y) + u128::from(out[i + j]) + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
            out[i + other.0.len()] = carry as u64;
        }
        Self(out).trimmed()
    }

    /// `self^exponent`, by squaring.
    pub(crate) fn pow(&self, exponent: u32) -> Self {
        let mut result = Self::from_u128(1);
        for bit in (0..u32::BITS - exponent.leading_zeros()).rev() {
            result = result.mul(&result);
            if exponent >> bit & 1 == 1 {
                result = result.mul(self);
            }
        }
        result
    }
}

impl Ord for Nat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for Nat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// LOG2(a, b, F) of RFC-0014 'Soundness calculator': an integer `L ≤ 2^F·log2(a/b)`, for
/// `a ≥ b ≥ 1`, with `L > 2^F·log2(a/b) − 1 − 4.33·2^{F − W}`.
///
/// 1. `e`, the largest integer with `b·2^e ≤ a`.
/// 2. `y = ⌊a·2^W / (b·2^e)⌋`, in `[2^W, 2^{W+1})`: `a/b/2^e` in fixed point.
/// 3. `L = e`; `F` times: `y = ⌊y^2/2^W⌋`, `L = 2L`, and if `y ≥ 2^{W+1}`, `y = ⌊y/2⌋` and
///    `L = L + 1`.
///
/// Every step rounds down, so `L + log2(y/2^W)/2^i`, after step `i`, never exceeds
/// `2^i·log2(a/b)`; each rounding costs at most `2^{-W}/ln 2` of `log2 y`, which later steps scale
/// by `2^{-i}`.
pub(crate) fn log2(a: u128, b: u128, fraction: u32) -> u128 {
    debug_assert!(a >= b && b >= 1);
    // a ≥ b, so a has at least as many bits, and b·2^e has a's bit count: no overflow.
    let mut e = b.leading_zeros() - a.leading_zeros();
    if b << e > a {
        e -= 1;
    }
    let mut y = normalized(a, b, e);
    let mut result = u128::from(e);
    for _ in 0..fraction {
        // y < 2^{W+1}: y^2 < 2^{2W+2} = 2^126.
        y = (y * y) >> W;
        result <<= 1;
        if y >> (W + 1) != 0 {
            y >>= 1;
            result |= 1;
        }
    }
    result
}

/// `⌊a·2^W / (b·2^e)⌋ ∈ [2^W, 2^{W+1})`, for `b·2^e ≤ a < b·2^{e+1}`, by restoring division over
/// natural numbers: `a·2^W` holds up to 190 bits.
fn normalized(a: u128, b: u128, e: u32) -> u128 {
    let mut remainder = Nat::from_u128(a).shl(u64::from(W));
    let divisor = Nat::from_u128(b).shl(u64::from(e));
    let mut quotient = 0u128;
    for bit in (0..=W).rev() {
        let step = divisor.shl(u64::from(bit));
        if remainder >= step {
            remainder = remainder.sub(&step);
            quotient |= 1 << bit;
        }
    }
    debug_assert!(quotient >> W == 1);
    quotient
}

/// The FixedBits of a field-form row: `None` for no term (`c = 0`), `256·g` when `c ≥ q`, and
/// `LOG2(q, c, 8) + 256·g` otherwise.
pub(crate) fn field_bits(c: u128, g: u32) -> Option<FixedBits> {
    let grinding = UNIT * u64::from(g);
    match c {
        0 => None,
        c if c >= Q => Some(FixedBits(grinding)),
        // At most 127·256 + 255.
        c => Some(FixedBits(log2(Q, c, FIELD_FRACTION) as u64 + grinding)),
    }
}

/// Whether a field-form row reaches `t` bits, exactly: `g + max(0, log2(q/c)) ≥ t`, which holds
/// when `t ≤ g` or `c·2^{t − g} ≤ q`, that is `c ≤ ⌊q / 2^{t − g}⌋`.
pub(crate) fn field_pass(c: u128, g: u32, t: u32) -> bool {
    if c == 0 || t <= g {
        return true;
    }
    let shift = t - g;
    shift < 128 && c <= Q >> shift
}

/// The per-query FixedBits and the row's of `(B/A)^τ` under `g` bits: `⌊L/2^{40}⌋` and
/// `⌊τ·L/2^{40}⌋ + 256·g` with `L = LOG2(A, B, 48)`, and 0 and `256·g` when `B ≥ A`.
pub(crate) fn query_bits(a: u128, b: u128, tau: u32, g: u32) -> (FixedBits, FixedBits) {
    let grinding = UNIT * u64::from(g);
    if b >= a {
        return (FixedBits(0), FixedBits(grinding));
    }
    let shift = QUERY_FRACTION - FIELD_FRACTION;
    // A < 2B: L < 2^48, and τ·L < 2^64.
    let l = log2(a, b, QUERY_FRACTION);
    let row = (u128::from(tau) * l) >> shift;
    (
        FixedBits((l >> shift) as u64),
        FixedBits(row as u64 + grinding),
    )
}

/// Whether a queries row reaches `t` bits, exactly: `g + max(0, τ·log2(A/B)) ≥ t`, which holds
/// when `t ≤ g` or `A^τ ≥ 2^{t − g}·B^τ`.
///
/// LOG2 decides first. `L = LOG2(A, B, 48)` lies within `2^48·log2(A/B) − 2 < L ≤
/// 2^48·log2(A/B)`, so `τ·L ≥ 2^48·T` proves the row passes and `τ·(L + 2) ≤ 2^48·T` that it
/// fails. Only a row within `τ·2^{-47}` bits of its target compares the powers.
pub(crate) fn queries_pass(a: u128, b: u128, tau: u32, g: u32, t: u32) -> bool {
    if t <= g {
        return true;
    }
    let target = t - g;
    if b >= a {
        return false;
    }
    let l = log2(a, b, QUERY_FRACTION);
    let scaled = u128::from(target) << QUERY_FRACTION;
    let tau128 = u128::from(tau);
    if tau128 * l >= scaled {
        true
    } else if tau128 * (l + 2) <= scaled {
        false
    } else {
        powers_pass(a, b, tau, target)
    }
}

/// `A^τ ≥ 2^T·B^τ`, in natural numbers.
fn powers_pass(a: u128, b: u128, tau: u32, target: u32) -> bool {
    Nat::from_u128(a).pow(tau) >= Nat::from_u128(b).pow(tau).shl(u64::from(target))
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    fn nat(x: u128) -> Nat {
        Nat::from_u128(x)
    }

    /// The smallest `x` with `x^k ≥ 2^m`, by bisection over natural numbers.
    fn root_ceil(k: u32, m: u64) -> u128 {
        boundary(1, k, m)
    }

    /// The smallest `a ≥ b` with `a^k ≥ 2^m·b^k`: `k·log2(a/b)` reaches `m` at `a`, and `a − 1`
    /// lies just below it, by about `1/b`.
    fn boundary(b: u128, k: u32, m: u64) -> u128 {
        let goal = nat(b).pow(k).shl(m);
        let (mut lo, mut hi) = (b, b << (m / u64::from(k) + 1));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if nat(mid).pow(k) >= goal {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    proptest! {
        /// On values that fit a `u128`, the natural numbers' arithmetic is the machine's.
        #[test]
        fn natural_numbers_agree_with_u128(x in any::<u64>(), y in any::<u64>(), s in 0u64..64) {
            let (x, y) = (u128::from(x), u128::from(y));
            prop_assert_eq!(nat(x).mul(&nat(y)), nat(x * y));
            prop_assert_eq!(nat(x).shl(s), nat(x << s));
            prop_assert_eq!(nat(x).cmp(&nat(y)), x.cmp(&y));
            let (big, small) = (x.max(y), x.min(y));
            prop_assert_eq!(nat(big).sub(&nat(small)), nat(big - small));
        }

        /// Subtraction across limbs is the machine's on 128 bits.
        #[test]
        fn subtraction_agrees_with_u128(x in any::<u128>(), y in any::<u128>()) {
            let (big, small) = (x.max(y), x.min(y));
            prop_assert_eq!(nat(big).sub(&nat(small)), nat(big - small));
        }

        /// Across limbs: `(x·2^s)·(y·2^t) = x·y·2^{s+t}`, and a power is repeated products.
        #[test]
        fn shifts_and_powers_compose(x in 1u128.., y in 1u128.., s in 0u64..300, t in 0u64..300, k in 0u32..9) {
            prop_assert_eq!(
                nat(x).shl(s).mul(&nat(y).shl(t)),
                nat(x).mul(&nat(y)).shl(s + t)
            );
            let mut product = nat(1);
            for _ in 0..k {
                product = product.mul(&nat(x));
            }
            prop_assert_eq!(nat(x).pow(k), product);
        }

        /// LOG2 is a lower bound with at most one unit of loss at `F = 8`: `2^L·b^256 ≤ a^256 <
        /// 2^{L+2}·b^256`, exactly, for `a, b < 2^64`.
        #[test]
        fn log2_is_a_lower_bound_within_a_unit(a in 1u64.., b in 1u64..) {
            let (a, b) = (u128::from(a.max(b)), u128::from(a.min(b)));
            let l = log2(a, b, 8);
            let (top, bottom) = (nat(a).pow(256), nat(b).pow(256));
            let l = u64::try_from(l).expect("small");
            prop_assert!(bottom.shl(l) <= top);
            prop_assert!(bottom.shl(l + 2) > top);
        }

        /// LOG2 never reaches a boundary it lies below, at the inputs where that is hardest: `a −
        /// 1` for the smallest `a` with `log2(a/b) ≥ k/2^F`, about `2^{-62}` below it for `b`
        /// near `2^62`. Rounding any step up instead of down crosses some of these. At `a` it
        /// loses at most a unit.
        #[test]
        fn log2_stays_below_the_boundaries_it_lies_under(
            b in (1u128 << 40)..(1 << 62), f in 1u32..6, k in 1u64..64,
        ) {
            let k = k % ((2 << f) - 1) + 1;
            let a = boundary(b, 1 << f, k);
            let below = log2(a - 1, b, f);
            prop_assert!(below < u128::from(k), "{} reaches {k}", below);
            prop_assert!(log2(a, b, f) + 1 >= u128::from(k));
        }

        /// The queries' decision is the comparison of powers, wherever LOG2 settles it.
        #[test]
        fn the_queries_decision_is_the_powers(
            n in 1u64..(1 << 32), k in 1u64..(1 << 32), mask in 0u64..(1 << 20),
            tau in 1u32..300, g in 0u32..40, t in 0u32..140,
        ) {
            let (a, b) = (2 * u128::from(n), u128::from(n) + u128::from(k) + u128::from(mask));
            let exact = t <= g || powers_pass(a, b, tau, t - g);
            prop_assert_eq!(queries_pass(a, b, tau, g, t), exact);
        }

        /// The field form's decision is `c·2^{t−g} ≤ q`, which this checks in natural numbers.
        #[test]
        fn the_field_decision_is_the_product(c in 1u128.., g in 0u32..64, t in 0u32..200) {
            let exact = t <= g || nat(c).shl(u64::from(t - g)) <= nat(Q);
            prop_assert_eq!(field_pass(c, g, t), exact);
        }
    }

    /// `q = p^2`, the order of RFC-0002's quadratic extension, `2^128 − 2^97 + 3·2^64 − 2^33 + 1`.
    #[test]
    fn q_is_the_order_of_fp2() {
        assert_eq!(Q, u128::MAX - (1 << 97) + (3 << 64) - (1 << 33) + 2);
    }

    /// A queries row within `τ·2^{-47}` bits of its target leaves LOG2 undecided, and the powers
    /// decide it, both ways. With `τ = 3`, `T = 1` and `B = 2^60`, the row passes exactly when
    /// `A^3 ≥ 2^181`: at `A = ⌈2^{181/3}⌉` it does, and at `A − 1` it does not, though LOG2 cannot
    /// tell the two apart.
    #[test]
    fn a_near_tie_is_decided_by_the_powers() {
        let b = 1u128 << 60;
        let a = root_ceil(3, 181);
        for (a, passes) in [(a, true), (a - 1, false)] {
            let l = log2(a, b, QUERY_FRACTION);
            let scaled = 1u128 << QUERY_FRACTION;
            assert!(
                3 * l < scaled && 3 * (l + 2) > scaled,
                "LOG2 leaves it open: {l}"
            );
            assert_eq!(powers_pass(a, b, 3, 1), passes);
            assert_eq!(queries_pass(a, b, 3, 0, 1), passes);
            assert_eq!(
                queries_pass(a, b, 3, 1, 2),
                passes,
                "with a bit of grinding"
            );
        }
    }

    /// LOG2 may lose a unit at a queries row's boundary: the row passes, and the fast path, which
    /// fails a row only when `τ·(L + 2)` stays below the target, leaves it to the powers. With
    /// `τ = 4`, `T = 1` and `B` above `2^60`, the smallest `A` with `A^4 ≥ 2·B^4` sometimes has
    /// `L = 2^46 − 1`, one unit short of `2^48·log2(A/B) ≥ 2^46`.
    #[test]
    fn a_lost_unit_at_the_boundary_is_left_to_the_powers() {
        let mut found = 0;
        for i in 0..64 {
            let b = (1u128 << 60) + i;
            let a = boundary(b, 4, 1);
            let l = log2(a, b, QUERY_FRACTION);
            if 4 * l + 4 <= 1 << QUERY_FRACTION {
                found += 1;
                assert!(powers_pass(a, b, 4, 1));
                assert!(queries_pass(a, b, 4, 0, 1), "B = 2^60 + {i}");
                assert!(!queries_pass(a - 1, b, 4, 0, 1), "B = 2^60 + {i}");
            }
        }
        assert!(found > 0, "no lost unit in 64 boundaries");
    }

    /// Powers that are equal pass: `A^τ = 2^T·B^τ` holds at `A = 2B`, `T = τ`.
    #[test]
    fn equal_powers_pass() {
        assert!(powers_pass(4, 2, 7, 7));
        assert!(!powers_pass(4, 2, 7, 8));
        assert!(powers_pass(6, 3, 1, 1) && !powers_pass(5, 3, 1, 1));
    }

    /// A borrow crosses every limb it must: `(2^128 + 5) − 6 = 2^128 − 1`.
    #[test]
    fn a_borrow_crosses_every_limb() {
        let top = Nat(vec![5, 0, 1]);
        assert_eq!(top.sub(&nat(6)), nat(u128::MAX));
        assert_eq!(Nat(vec![0, 0, 1]).sub(&nat(1)), nat(u128::MAX));
    }

    /// The edges of each form: no term, an error of at least 1, which counts as 1, and grinding
    /// that reaches the target alone.
    #[test]
    fn the_edges_of_each_form() {
        assert_eq!(field_bits(0, 20), None);
        assert!(field_pass(0, 0, 1000));
        assert_eq!(field_bits(Q, 3), Some(FixedBits(3 * 256)));
        assert_eq!(field_bits(u128::MAX, 0), Some(FixedBits(0)));
        assert!(!field_pass(Q, 0, 1) && field_pass(Q, 1, 1));
        // c = 1: 128 − 6.7·10^-10 bits.
        assert_eq!(field_bits(1, 0), Some(FixedBits(128 * 256 - 1)));
        assert!(field_pass(1, 0, 127) && !field_pass(1, 0, 128));
        assert!(!field_pass(1, 0, 1000));
        // ρ' ≥ 1: no query gives a bit, and only grinding counts.
        assert_eq!(query_bits(8, 8, 3, 5), (FixedBits(0), FixedBits(5 * 256)));
        assert!(queries_pass(8, 9, 1, 5, 5) && !queries_pass(8, 9, 1, 5, 6));
        // A = 2B: one bit per query, exactly.
        assert_eq!(log2(2, 1, 48), 1 << 48);
        assert_eq!(query_bits(4, 2, 7, 0).1, FixedBits(7 * 256));
        assert!(queries_pass(4, 2, 7, 0, 7) && !queries_pass(4, 2, 7, 0, 8));
    }

    fn vectors() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/vectors/soundness/soundness.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("the vectors")).expect("JSON")
    }

    /// LOG2 gives the oracle's result at its edges and at random 128-bit inputs.
    #[test]
    fn log2_is_the_oracles() {
        let vectors = vectors();
        let precision = &vectors["log2"];
        assert_eq!(precision["precision"], W);
        assert_eq!(precision["field_fraction"], FIELD_FRACTION);
        assert_eq!(precision["query_fraction"], QUERY_FRACTION);
        let cases = vectors["log2_cases"].as_array().expect("cases");
        assert!(cases.len() > 30);
        for case in cases {
            let number = |key: &str| -> u128 {
                case[key]
                    .as_str()
                    .expect("a decimal")
                    .parse()
                    .expect("a u128")
            };
            let fraction = u32::try_from(case["fraction"].as_u64().expect("F")).expect("F");
            assert_eq!(
                log2(number("a"), number("b"), fraction),
                number("result"),
                "{case}"
            );
        }
        assert_eq!(vectors["field"]["q"], Q.to_string());
    }
}
