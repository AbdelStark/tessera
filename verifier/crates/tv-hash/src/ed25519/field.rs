// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The field `GF(p)`, `p = 2^255 − 19`, of RFC 8032 §5.1.
//!
//! An element is four little-endian 64-bit limbs holding its canonical value, `0 ≤ value < p`,
//! after every operation: equality is equality of limbs, and the encoding and the parity are read
//! off the limbs directly. Products are schoolbook products into eight limbs, reduced with
//! `2^256 ≡ 38 (mod p)` and then `2^255 ≡ 19`. Inversion and square roots are exponentiations by
//! fixed exponents. Nothing here is constant-time: every input of the verifier is public.

/// `p = 2^255 − 19`.
const P: [u64; 4] = [
    0xffff_ffff_ffff_ffed,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x7fff_ffff_ffff_ffff,
];

/// `p − 2`, the exponent of inversion (Fermat).
const P_MINUS_2: [u64; 4] = [
    0xffff_ffff_ffff_ffeb,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x7fff_ffff_ffff_ffff,
];

/// `(p + 3) / 8`, the exponent of the candidate square root of RFC 8032 §5.1.3 step 3.
const P_PLUS_3_OVER_8: [u64; 4] = [
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x0fff_ffff_ffff_ffff,
];

/// The low 64 bits of `t`.
#[inline(always)]
#[allow(clippy::as_conversions)] // The truncation is the point.
const fn lo(t: u128) -> u64 {
    t as u64
}

/// The high 64 bits of `t`.
#[inline(always)]
const fn hi(t: u128) -> u64 {
    lo(t >> 64)
}

/// `a + b + carry` as a limb and a carry (0 or 1, for a carry of 0 or 1).
#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = u128::from(a)
        .wrapping_add(u128::from(b))
        .wrapping_add(u128::from(carry));
    (lo(t), hi(t))
}

/// `a − b − borrow` as a limb and a borrow (0 or 1, for a borrow of 0 or 1).
#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(borrow);
    (d, u64::from(b1 | b2))
}

/// `acc + a·b + carry` as a limb and a carry limb. It cannot overflow 128 bits:
/// `(2^64 − 1)^2 + 2·(2^64 − 1) = 2^128 − 1`.
#[inline(always)]
fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = u128::from(a)
        .wrapping_mul(u128::from(b))
        .wrapping_add(u128::from(acc))
        .wrapping_add(u128::from(carry));
    (lo(t), hi(t))
}

/// `v − m` and whether it borrowed, over four limbs.
fn sub_limbs(v: &[u64; 4], m: &[u64; 4]) -> ([u64; 4], bool) {
    let (r0, b) = sbb(v[0], m[0], 0);
    let (r1, b) = sbb(v[1], m[1], b);
    let (r2, b) = sbb(v[2], m[2], b);
    let (r3, b) = sbb(v[3], m[3], b);
    ([r0, r1, r2, r3], b == 1)
}

/// `v mod p` for any `v < 2^256`: fold bit 255 as `19`, which leaves a value below
/// `2^255 + 19 < 2p`, then subtract `p` once if it is not below `p`.
fn reduce(v: [u64; 4]) -> [u64; 4] {
    let top = v[3] >> 63;
    let (r0, c) = adc(v[0], top.wrapping_mul(19), 0);
    let (r1, c) = adc(v[1], 0, c);
    let (r2, c) = adc(v[2], 0, c);
    let (r3, _) = adc(v[3] & (u64::MAX >> 1), 0, c);
    let r = [r0, r1, r2, r3];
    match sub_limbs(&r, &P) {
        (below, false) => below,
        (_, true) => r,
    }
}

/// An element of `GF(p)`, canonical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Fe([u64; 4]);

impl Fe {
    pub(super) const ZERO: Fe = Fe([0; 4]);
    pub(super) const ONE: Fe = Fe([1, 0, 0, 0]);

    /// The element whose canonical value is given as limbs below `p`; for constants.
    pub(super) const fn from_canonical_limbs(limbs: [u64; 4]) -> Fe {
        Fe(limbs)
    }

    /// The 255-bit little-endian integer of `bytes`, bit 255 ignored, reduced modulo `p`. Values
    /// in `[p, 2^255)` are reduced, not refused: that is RFC-0060's decoding rule, and the
    /// caller that needs canonical input compares encodings.
    pub(super) fn from_bytes_mod_p(bytes: &[u8; 32]) -> Fe {
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut le = [0u8; 8];
            le.copy_from_slice(chunk);
            *limb = u64::from_le_bytes(le);
        }
        limbs[3] &= u64::MAX >> 1;
        Fe(reduce(limbs))
    }

    /// The canonical 32-byte little-endian encoding; bit 255 is 0.
    pub(super) fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (chunk, limb) in out.chunks_exact_mut(8).zip(self.0) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// Whether the canonical value is odd: RFC 8032's "negative" `x`.
    pub(super) fn is_odd(self) -> bool {
        self.0[0] & 1 == 1
    }

    pub(super) fn is_zero(self) -> bool {
        self == Fe::ZERO
    }

    /// `self + other`. Both are below `p`, so the sum is below `2^256` and one reduction does.
    pub(super) fn plus(self, other: Fe) -> Fe {
        let (a, b) = (self.0, other.0);
        let (r0, c) = adc(a[0], b[0], 0);
        let (r1, c) = adc(a[1], b[1], c);
        let (r2, c) = adc(a[2], b[2], c);
        let (r3, _) = adc(a[3], b[3], c);
        Fe(reduce([r0, r1, r2, r3]))
    }

    /// `self − other`: the difference of the limbs, plus `p` when it borrowed.
    pub(super) fn minus(self, other: Fe) -> Fe {
        match sub_limbs(&self.0, &other.0) {
            (r, false) => Fe(r),
            (r, true) => {
                // `self − other + 2^256` with `self < other`; adding `p` and dropping the carry
                // out of 2^256 leaves `self − other + p`, which is in `[0, p)`.
                let (r0, c) = adc(r[0], P[0], 0);
                let (r1, c) = adc(r[1], P[1], c);
                let (r2, c) = adc(r[2], P[2], c);
                let (r3, _) = adc(r[3], P[3], c);
                Fe([r0, r1, r2, r3])
            }
        }
    }

    /// `−self`.
    pub(super) fn negate(self) -> Fe {
        Fe::ZERO.minus(self)
    }

    /// `self · other`: the eight-limb schoolbook product, row by row, then reduced.
    pub(super) fn times(self, other: Fe) -> Fe {
        let [a0, a1, a2, a3] = self.0;
        let [b0, b1, b2, b3] = other.0;
        let (t0, c) = mac(0, a0, b0, 0);
        let (t1, c) = mac(0, a0, b1, c);
        let (t2, c) = mac(0, a0, b2, c);
        let (t3, t4) = mac(0, a0, b3, c);

        let (t1, c) = mac(t1, a1, b0, 0);
        let (t2, c) = mac(t2, a1, b1, c);
        let (t3, c) = mac(t3, a1, b2, c);
        let (t4, t5) = mac(t4, a1, b3, c);

        let (t2, c) = mac(t2, a2, b0, 0);
        let (t3, c) = mac(t3, a2, b1, c);
        let (t4, c) = mac(t4, a2, b2, c);
        let (t5, t6) = mac(t5, a2, b3, c);

        let (t3, c) = mac(t3, a3, b0, 0);
        let (t4, c) = mac(t4, a3, b1, c);
        let (t5, c) = mac(t5, a3, b2, c);
        let (t6, t7) = mac(t6, a3, b3, c);

        Fe(reduce_wide([t0, t1, t2, t3, t4, t5, t6, t7]))
    }

    pub(super) fn square(self) -> Fe {
        self.times(self)
    }

    /// `self^e`, square and multiply from the top bit of `e` down.
    fn pow(self, e: &[u64; 4]) -> Fe {
        let mut acc = Fe::ONE;
        for limb in e.iter().rev() {
            for bit in (0..64u32).rev() {
                acc = acc.square();
                if limb.wrapping_shr(bit) & 1 == 1 {
                    acc = acc.times(self);
                }
            }
        }
        acc
    }

    /// `self^(p − 2)`: the inverse of a non-zero element, and 0 for 0.
    pub(super) fn invert(self) -> Fe {
        self.pow(&P_MINUS_2)
    }

    /// `self^((p + 3) / 8)`: a square root of `self` or of `−self` when `self` is a square
    /// (RFC 8032 §5.1.3 step 3).
    pub(super) fn sqrt_candidate(self) -> Fe {
        self.pow(&P_PLUS_3_OVER_8)
    }
}

/// `t mod p` for an eight-limb `t < 2^512`. With `t = lo + 2^256·hi` and `2^256 ≡ 38`,
/// `lo + 38·hi < 39·2^256` is four limbs and a fifth `r4 ≤ 38`; folding `r4` as `38·r4 < 1482`
/// can carry out of `2^256` only when the four limbs end below 1482, so folding that carry as 38
/// carries no further; [`reduce`] finishes.
fn reduce_wide(t: [u64; 8]) -> [u64; 4] {
    let [l0, l1, l2, l3, h0, h1, h2, h3] = t;
    let (r0, c) = mac(l0, h0, 38, 0);
    let (r1, c) = mac(l1, h1, 38, c);
    let (r2, c) = mac(l2, h2, 38, c);
    let (r3, r4) = mac(l3, h3, 38, c);

    let (r0, c) = mac(r0, r4, 38, 0);
    let (r1, c) = adc(r1, 0, c);
    let (r2, c) = adc(r2, 0, c);
    let (r3, c) = adc(r3, 0, c);

    let (r0, c) = adc(r0, c.wrapping_mul(38), 0);
    let (r1, c) = adc(r1, 0, c);
    let (r2, c) = adc(r2, 0, c);
    let (r3, _) = adc(r3, 0, c);
    reduce([r0, r1, r2, r3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small element.
    fn small(n: u64) -> Fe {
        Fe([n, 0, 0, 0])
    }

    fn from_bytes(bytes: [u8; 32]) -> Fe {
        Fe::from_bytes_mod_p(&bytes)
    }

    #[test]
    fn p_reduces_to_zero_and_the_values_above_it_to_their_remainders() {
        let mut p = [0xffu8; 32];
        p[0] = 0xed;
        p[31] = 0x7f;
        assert_eq!(from_bytes(p), Fe::ZERO);
        for t in 1..19u8 {
            let mut v = p;
            v[0] = 0xed + t;
            assert_eq!(from_bytes(v), small(u64::from(t)));
        }
        // Bit 255 is not part of the value.
        let mut top = [0u8; 32];
        top[31] = 0x80;
        assert_eq!(from_bytes(top), Fe::ZERO);
    }

    #[test]
    fn minus_one_squares_to_one_and_inverts_to_itself() {
        let minus_one = Fe::ZERO.minus(Fe::ONE);
        assert_eq!(minus_one.0, [P[0] - 1, P[1], P[2], P[3]]);
        assert_eq!(minus_one.square(), Fe::ONE);
        assert_eq!(minus_one.invert(), minus_one);
        assert_eq!(minus_one.plus(Fe::ONE), Fe::ZERO);
    }

    #[test]
    fn products_agree_with_repeated_addition_and_inverses_invert() {
        // Elements near p and near 2^255 / 2^256 boundaries, where carries and folds happen.
        let minus_one = Fe::ZERO.minus(Fe::ONE);
        let samples = [
            Fe::ONE,
            small(2),
            small(19),
            small(38),
            minus_one,
            minus_one.minus(small(18)),
            Fe([u64::MAX, u64::MAX, u64::MAX, 0x3fff_ffff_ffff_ffff]),
            Fe([0, 0, 0, 0x4000_0000_0000_0000]),
            Fe([
                0x0123_4567_89ab_cdef,
                0xfedc_ba98_7654_3210,
                0x0f0f_0f0f_0f0f_0f0f,
                0x7000_0000_0000_0001,
            ]),
        ];
        for a in samples {
            // a·5 = a + a + a + a + a.
            let five = a.plus(a).plus(a).plus(a).plus(a);
            assert_eq!(a.times(small(5)), five);
            // (a + 1)^2 = a^2 + 2a + 1.
            let lhs = a.plus(Fe::ONE).square();
            let rhs = a.square().plus(a).plus(a).plus(Fe::ONE);
            assert_eq!(lhs, rhs);
            assert_eq!(a.times(a.invert()), Fe::ONE);
            assert_eq!(a.negate().plus(a), Fe::ZERO);
            for b in samples {
                assert_eq!(a.times(b), b.times(a));
                assert_eq!(a.minus(b).plus(b), a);
            }
        }
        assert_eq!(Fe::ZERO.invert(), Fe::ZERO);
    }

    #[test]
    fn encodings_round_trip() {
        let a = Fe([
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
            7,
            0x7000_0000_0000_0001,
        ]);
        assert_eq!(Fe::from_bytes_mod_p(&a.to_bytes()), a);
        assert_eq!(small(1).to_bytes()[0], 1);
        assert!(small(3).is_odd());
        assert!(!small(4).is_odd());
    }
}
