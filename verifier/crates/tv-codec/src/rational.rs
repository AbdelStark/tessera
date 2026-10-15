// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Rational` (RFC-0003 'Primitive types'), the exact stand-in for a float (INV-DM-6).

/// `num / den` in lowest terms with a positive denominator, so that each rational number has
/// exactly one encoding: zero is `0/1`, and `-3/7` is never `3/-7` or `-6/14`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: u64,
}

impl Rational {
    /// `num / den`, or `None` unless `den > 0` and `gcd(|num|, den) = 1`.
    pub const fn new(num: i64, den: u64) -> Option<Self> {
        if den != 0 && gcd(num.unsigned_abs(), den) == 1 {
            Some(Self { num, den })
        } else {
            None
        }
    }

    /// The numerator.
    pub const fn num(self) -> i64 {
        self.num
    }

    /// The denominator, at least 1.
    pub const fn den(self) -> u64 {
        self.den
    }
}

/// Euclid's algorithm: `gcd(a, b) = gcd(b, a mod b)` until the remainder is 0, and
/// `gcd(a, 0) = a`. `checked_rem` is `None` exactly when `b = 0`.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while let Some(remainder) = a.checked_rem(b) {
        a = b;
        b = remainder;
    }
    a
}
