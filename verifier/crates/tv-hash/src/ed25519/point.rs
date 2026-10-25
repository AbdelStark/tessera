// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The twisted Edwards curve `−x^2 + y^2 = 1 + d·x^2·y^2` over `GF(p)` of RFC 8032 §5.1, its
//! encoding (§5.1.2), its decoding (§5.1.3, with RFC-0060's acceptance of `y ∈ [p, 2^255)`) and
//! its group law.
//!
//! Points are kept in projective coordinates `(X : Y : Z)`, `x = X/Z`, `y = Y/Z`, `Z ≠ 0`, and
//! added by one unified formula, Bernstein, Birkner, Joye, Lange and Peters's "add-2008-bbjlp"
//! ("Twisted Edwards curves", AFRICACRYPT 2008, §6) with `a = −1`. That formula is complete on
//! this curve, because `a = −1` is a square and `d` is not modulo `p`: it adds any two points,
//! equal, opposite or the identity included, so doubling is an addition of a point to itself and
//! no case is special. Multiplication by a scalar is double-and-add from the top bit, in
//! variable time: the inputs are public.

use super::field::Fe;

/// `d = −121665/121666`.
const D: Fe = Fe::from_canonical_limbs([
    0x75eb_4dca_1359_78a3,
    0x0070_0a4d_4141_d8ab,
    0x8cc7_4079_7779_e898,
    0x5203_6cee_2b6f_fe73,
]);

/// `2^((p − 1)/4)`, a square root of `−1`.
const SQRT_M1: Fe = Fe::from_canonical_limbs([
    0xc4ee_1b27_4a0e_a0b0,
    0x2f43_1806_ad2f_e478,
    0x2b4d_0099_3dfb_d7a7,
    0x2b83_2480_4fc1_df0b,
]);

/// The base point `B`'s `x`, the even root for `y = 4/5`.
const BASE_X: Fe = Fe::from_canonical_limbs([
    0xc956_2d60_8f25_d51a,
    0x692c_c760_9525_a7b2,
    0xc0a4_e231_fdd6_dc5c,
    0x2169_36d3_cd6e_53fe,
]);

/// The base point `B`'s `y = 4/5`.
const BASE_Y: Fe = Fe::from_canonical_limbs([
    0x6666_6666_6666_6658,
    0x6666_6666_6666_6666,
    0x6666_6666_6666_6666,
    0x6666_6666_6666_6666,
]);

/// A point of the curve in projective coordinates.
#[derive(Clone, Copy)]
pub(super) struct Point {
    x: Fe,
    y: Fe,
    z: Fe,
}

impl Point {
    /// The neutral element `(0, 1)`.
    pub(super) const IDENTITY: Point = Point {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
    };

    /// The base point `B` of RFC 8032 §5.1.
    pub(super) const BASE: Point = Point {
        x: BASE_X,
        y: BASE_Y,
        z: Fe::ONE,
    };

    /// Decodes 32 bytes by RFC 8032 §5.1.3 with one exception, RFC-0060's: a `y` in
    /// `[p, 2^255)` is reduced modulo `p` instead of refused.
    ///
    /// 1. `y` is the 255 low bits, little-endian, modulo `p`; `x_0` is bit 255.
    /// 2. `x^2 = (y^2 − 1) / (d·y^2 + 1)`; the denominator is never 0, since `−1/d` is not a
    ///    square.
    /// 3. `x = (x^2)^((p + 3)/8)`; if its square is `−x^2`, `x` is multiplied by `√−1`; if its
    ///    square is then not `x^2`, `y` is on no point: `None`.
    /// 4. `x = 0` with `x_0 = 1` is `None`. If `x mod 2 ≠ x_0`, `x` becomes `p − x`.
    pub(super) fn decode(bytes: &[u8; 32]) -> Option<Point> {
        let sign = bytes[31] >> 7 == 1;
        let y = Fe::from_bytes_mod_p(bytes);
        let yy = y.square();
        let x2 = yy.minus(Fe::ONE).times(D.times(yy).plus(Fe::ONE).invert());
        let mut x = x2.sqrt_candidate();
        if x.square() != x2 {
            x = x.times(SQRT_M1);
            if x.square() != x2 {
                return None;
            }
        }
        if x.is_zero() && sign {
            return None;
        }
        if x.is_odd() != sign {
            x = x.negate();
        }
        Some(Point { x, y, z: Fe::ONE })
    }

    /// The canonical encoding of RFC 8032 §5.1.2: `y` in 255 little-endian bits, below `p`, and
    /// bit 255 the parity of `x`.
    pub(super) fn encode(&self) -> [u8; 32] {
        let z_inv = self.z.invert();
        let x = self.x.times(z_inv);
        let mut out = self.y.times(z_inv).to_bytes();
        if x.is_odd() {
            out[31] |= 0x80;
        }
        out
    }

    /// `self + other` by "add-2008-bbjlp" with `a = −1`:
    /// `A = Z1·Z2`, `B = A^2`, `C = X1·X2`, `D = Y1·Y2`, `E = d·C·D`, `F = B − E`, `G = B + E`,
    /// `X3 = A·F·((X1 + Y1)·(X2 + Y2) − C − D)`, `Y3 = A·G·(D + C)`, `Z3 = F·G`.
    pub(super) fn plus(&self, other: &Point) -> Point {
        let a = self.z.times(other.z);
        let b = a.square();
        let c = self.x.times(other.x);
        let d = self.y.times(other.y);
        let e = D.times(c).times(d);
        let f = b.minus(e);
        let g = b.plus(e);
        let cross = self
            .x
            .plus(self.y)
            .times(other.x.plus(other.y))
            .minus(c)
            .minus(d);
        Point {
            x: a.times(f).times(cross),
            y: a.times(g).times(d.plus(c)),
            z: f.times(g),
        }
    }

    /// `−self = (−x, y)`.
    pub(super) fn negate(&self) -> Point {
        Point {
            x: self.x.negate(),
            y: self.y,
            z: self.z,
        }
    }

    /// `[n]self` for the 256-bit little-endian integer `n`, exactly: `n` is not reduced modulo
    /// the group order, so the result is right for points of any order.
    pub(super) fn multiple(&self, n: &[u8; 32]) -> Point {
        let mut acc = Point::IDENTITY;
        for byte in n.iter().rev() {
            for bit in (0..8u32).rev() {
                acc = acc.plus(&acc);
                if byte.wrapping_shr(bit) & 1 == 1 {
                    acc = acc.plus(self);
                }
            }
        }
        acc
    }

    /// Whether this is the identity: `X = 0` and `Y = Z`.
    pub(super) fn is_identity(&self) -> bool {
        self.x.is_zero() && self.y == self.z
    }

    /// Whether the point is of small order, a divisor of the cofactor 8: whether `[8]P` is the
    /// identity.
    pub(super) fn is_small_order(&self) -> bool {
        let p2 = self.plus(self);
        let p4 = p2.plus(&p2);
        p4.plus(&p4).is_identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `L = 2^252 + 27742317777372353535851937790883648493`, little-endian.
    const L: [u8; 32] = [
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde,
        0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
    ];

    fn small(n: u64) -> Fe {
        Fe::from_canonical_limbs([n, 0, 0, 0])
    }

    fn on_curve(p: &Point) -> bool {
        // −X^2·Z^2 + Y^2·Z^2 = Z^4 + d·X^2·Y^2.
        let (xx, yy, zz) = (p.x.square(), p.y.square(), p.z.square());
        yy.minus(xx).times(zz) == zz.square().plus(D.times(xx).times(yy))
    }

    fn equal(p: &Point, q: &Point) -> bool {
        p.x.times(q.z) == q.x.times(p.z) && p.y.times(q.z) == q.y.times(p.z)
    }

    #[test]
    fn the_constants_are_rfc_8032s() {
        // d·121666 = −121665.
        assert_eq!(D.times(small(121_666)), small(121_665).negate());
        assert_eq!(SQRT_M1.square(), Fe::ONE.negate());
        // y = 4/5, x even, on the curve.
        assert_eq!(BASE_Y.times(small(5)), small(4));
        assert!(!BASE_X.is_odd());
        assert!(on_curve(&Point::BASE));
    }

    #[test]
    fn the_base_point_has_order_l_and_its_standard_encoding() {
        let mut encoding = [0x66u8; 32];
        encoding[0] = 0x58;
        assert_eq!(Point::BASE.encode(), encoding);
        let decoded = Point::decode(&encoding).unwrap();
        assert!(equal(&decoded, &Point::BASE));
        assert!(Point::BASE.multiple(&L).is_identity());
        assert!(!Point::BASE.is_small_order());
        let mut l_minus_1 = L;
        l_minus_1[0] -= 1;
        assert!(equal(
            &Point::BASE.multiple(&l_minus_1),
            &Point::BASE.negate()
        ));
    }

    #[test]
    fn the_group_law_is_consistent() {
        let mut three = [0u8; 32];
        three[0] = 3;
        let b2 = Point::BASE.plus(&Point::BASE);
        let b3 = b2.plus(&Point::BASE);
        assert!(on_curve(&b2) && on_curve(&b3));
        assert!(equal(&Point::BASE.multiple(&three), &b3));
        assert!(equal(&b3.plus(&Point::BASE.negate()), &b2));
        assert!(Point::BASE.plus(&Point::BASE.negate()).is_identity());
        assert!(equal(&Point::IDENTITY.plus(&b3), &b3));
        let decoded = Point::decode(&b3.encode()).unwrap();
        assert!(equal(&decoded, &b3));
    }
}
