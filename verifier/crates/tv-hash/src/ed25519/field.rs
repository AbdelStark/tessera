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

    /// Operands `a` and `b`, as limbs, and the value modulo `p` of their product (or sum) that the
    /// reduction reaches through one of its rare carries.
    type KnownAnswer = ([u64; 4], [u64; 4], [u64; 4]);

    /// The second fold carries out of `2^256` (`r + 38·r4 ≥ 2^256`), and the first out of limb 0.
    const FOLD_2: [KnownAnswer; 2] = [
        (
            [
                0xc509_df61_7060_50d3,
                0x0d1a_9540_14e1_e1cf,
                0xdf82_b45f_4523_a8c6,
                0x5973_4b17_b8b6_3865,
            ],
            [
                0xc434_388f_abf9_a75f,
                0x25fa_90d0_0be4_26a7,
                0x92cc_2086_4180_3686,
                0x4dc3_7cdf_e5f7_8374,
            ],
            [0x53, 0, 0, 0],
        ),
        (
            [
                0xd520_7b80_4e15_da6d,
                0x31cf_f0da_65c9_8991,
                0xeb12_a7a1_a295_e206,
                0x7ead_93ac_1a69_62b4,
            ],
            [
                0x1e5e_70db_f3b5_d7f8,
                0x20e5_75bb_e4c4_1429,
                0x138d_6773_c6b6_4043,
                0x35af_abd4_6c9f_c5fe,
            ],
            [0x68, 0, 0, 0],
        ),
    ];

    /// The first fold carries out of limb 0 (`r0 + 38·r4 ≥ 2^64`), and the second does not carry.
    const FOLD_1: [KnownAnswer; 2] = [
        (
            [
                0x461e_6e5d_c44b_7891,
                0x321b_cf2f_cb10_fc7b,
                0xd59c_588d_11eb_9840,
                0x242a_5412_a14b_648a,
            ],
            [
                0x0945_7640_7b76_f954,
                0xd7ee_455d_1189_0da1,
                0xb206_9e6f_0ce5_4513,
                0x207c_6b6d_1cb3_db35,
            ],
            [
                0x35,
                0x505a_58a2_0c9f_553e,
                0x5185_a623_fe02_d715,
                0x0faf_ada7_a48d_3e14,
            ],
        ),
        (
            [
                0x70a5_386f_2356_78c7,
                0x7216_396d_c97a_a7a2,
                0xda52_b0b8_a47b_073e,
                0x5432_5030_f204_6521,
            ],
            [
                0x5c82_547c_7db0_8f53,
                0x25ea_da62_eec1_b780,
                0xb438_0c2b_e9fe_9e91,
                0x40fa_a3ce_df26_2bb3,
            ],
            [
                0x54,
                0x48be_8cdd_2569_f9b6,
                0x81a7_5383_6b47_bc47,
                0x7343_2f3c_ea08_8b2d,
            ],
        ),
    ];

    /// `reduce`, after the folds, takes bit 255 as 19 and carries out of limb 0.
    const REDUCE_LIMB_0: [KnownAnswer; 2] = [
        (
            [
                0x3d0e_6f3a_2db5_1443,
                0x8171_7dc5_d615_8963,
                0x5240_c23b_0721_5372,
                0x231a_4be3_8658_a56b,
            ],
            [
                0xac6b_9ca3_aa40_c31f,
                0xba81_cbda_1d78_e270,
                0xa49b_faf7_13bf_7a86,
                0x051e_8b66_f8e5_bb5b,
            ],
            [
                0x0e,
                0x6a92_d0c2_fed5_84d6,
                0x183a_5db7_0809_abfd,
                0x6a63_e836_58aa_61d7,
            ],
        ),
        (
            [
                0xb147_47d4_7f99_cb99,
                0xf903_a090_b80c_24a2,
                0xf9ac_cccf_b93a_a199,
                0x69aa_6320_4fc2_bf31,
            ],
            [
                0xcbff_c7e3_f31b_adaf,
                0x5ea1_9aea_99ab_9d7b,
                0x45e0_dd0a_97bb_41b0,
                0x04dc_d2e2_f2b1_c793,
            ],
            [
                0x12,
                0xfd9a_b9bf_de91_4127,
                0xcdef_811b_5f91_4fb2,
                0x6371_c88b_59dc_b5af,
            ],
        ),
    ];

    /// Sums of `a` and `b` at or above `2^255` whose limb 0 is at least `2^64 − 19`: `reduce`'s
    /// fold of bit 255 as 19 carries out of limb 0.
    const SUM_LIMB_0: [KnownAnswer; 2] = [
        (
            [
                0x4f4f_306c_9da5_b926,
                0x8564_5643_6247_d6f4,
                0xc0cc_dba7_3e88_3667,
                0x5353_7885_15e7_869d,
            ],
            [
                0xb0b0_cf93_625a_46d7,
                0xfca4_d78a_07c8_45b4,
                0xe22e_4d48_85c2_b7fd,
                0x3d3c_f60e_c263_4828,
            ],
            [
                0x10,
                0x8209_2dcd_6a10_1ca9,
                0xa2fb_28ef_c44a_ee65,
                0x1090_6e93_d84a_cec6,
            ],
        ),
        (
            [
                0xf3f9_64b0_b96e_8a07,
                0xbaf8_1a24_e509_b16a,
                0x1b6d_7df2_8742_162a,
                0x5a89_a066_a375_bd8c,
            ],
            [
                0x0c06_9b4f_4691_75ef,
                0xfb38_782d_df9e_1560,
                0x416d_66de_6139_2c5d,
                0x436c_abba_4ad5_53f7,
            ],
            [
                0x09,
                0xb630_9252_c4a7_c6cb,
                0x5cda_e4d0_e87b_4288,
                0x1df6_4c20_ee4b_1183,
            ],
        ),
    ];

    /// Every carry of the reduction, against values modulo `p` from a big-integer model (Python's
    /// `a * b % p` and `(a + b) % p`). The operands are crafted, not drawn: with
    /// `a·b = hi·2^256 + lo`, `hi` is solved for so that `lo + 38·hi` lands where the carry
    /// happens. Random operands meet these carries with probability below `2^-55` (the first
    /// fold), `2^-247` (the second) and `2^-60` (`reduce`'s limb 0): no random test reaches them.
    #[test]
    fn every_carry_of_the_reduction_against_known_answers() {
        for (a, b, expected) in FOLD_2.into_iter().chain(FOLD_1).chain(REDUCE_LIMB_0) {
            assert_eq!(Fe(a).times(Fe(b)), Fe(expected), "{a:x?} · {b:x?}");
            assert_eq!(Fe(b).times(Fe(a)), Fe(expected), "{b:x?} · {a:x?}");
        }
        for (a, b, expected) in SUM_LIMB_0 {
            assert_eq!(Fe(a).plus(Fe(b)), Fe(expected), "{a:x?} + {b:x?}");
            assert_eq!(Fe(b).plus(Fe(a)), Fe(expected), "{b:x?} + {a:x?}");
        }
        // The top of each function's domain: 2^256 − 1 = 2p + 37 carries through every limb of
        // `reduce`, and 2^512 − 1 ≡ 38^2 − 1 = 1443 takes both folds' carries in `reduce_wide`.
        assert_eq!(reduce([u64::MAX; 4]), [37, 0, 0, 0]);
        assert_eq!(reduce_wide([u64::MAX; 8]), [1443, 0, 0, 0]);
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
