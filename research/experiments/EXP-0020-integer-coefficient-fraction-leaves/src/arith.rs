// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Fixed-width arithmetic: all array lengths and reduction cadences are public.
use crate::{BenchField, Field, Fp, PrimeField};
use tessera_codec::Zeroize;
pub const B: usize = 1024;
#[derive(Clone, Copy, Default)]
pub struct Acc(pub [u64; 3]);
impl Zeroize for Acc {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}
impl Acc {
    pub fn add(&mut self, x: u128) {
        let (a, c) = self.0[0].overflowing_add(x as u64);
        let (b, d) = self.0[1].overflowing_add((x >> 64) as u64);
        let (b, e) = b.overflowing_add(u64::from(c));
        self.0 = [
            a,
            b,
            self.0[2]
                .wrapping_add(u64::from(d))
                .wrapping_add(u64::from(e)),
        ];
    }
    pub fn add_words(&mut self, x: [u64; 3]) {
        let (a, c) = self.0[0].overflowing_add(x[0]);
        let (b, d) = self.0[1].overflowing_add(x[1]);
        let (b, e) = b.overflowing_add(u64::from(c));
        self.0 = [
            a,
            b,
            self.0[2]
                .wrapping_add(x[2])
                .wrapping_add(u64::from(d))
                .wrapping_add(u64::from(e)),
        ];
    }
    pub fn unsigned(self) -> Fp {
        let [a, b, c] = self.0;
        let r = Fp::from_u64_reduced;
        r(a) + r(b) * r((1 << 32) - 1) - r(c) * r(1 << 32)
    }
    pub fn signed(self) -> Fp {
        self.unsigned() - Fp::from_u64_reduced(self.0[2] >> 63)
    }
    pub fn weighted(&mut self, c: i128, w: u64) {
        let mask = (c >> 127) as u128;
        let mag = ((c as u128) ^ mask).wrapping_sub(mask);
        let a = (mag as u64 as u128) * u128::from(w);
        let b = (mag >> 64) * u128::from(w);
        let (mid, carry) = (a >> 64).overflowing_add(b as u64 as u128);
        // mid addition is at most 65 bits, never overflows u128.
        debug_assert!(!carry);
        let words = [a as u64, mid as u64, ((b >> 64) + (mid >> 64)) as u64];
        let neg = 0u64.wrapping_sub((c < 0) as u64);
        let mut out = [0; 3];
        let mut borrow = 0u64;
        for j in 0..3 {
            let y = words[j] ^ neg;
            let (z, b1) = y.overflowing_add(if j == 0 { neg & 1 } else { borrow });
            out[j] = z;
            borrow = u64::from(b1);
        }
        self.add_words(out);
    }
}
#[derive(Clone, Copy, Default)]
pub struct ExtAcc {
    pub a: Acc,
    pub b: Acc,
    pub c: Acc,
}
impl Zeroize for ExtAcc {
    fn zeroize(&mut self) {
        self.a.zeroize();
        self.b.zeroize();
        self.c.zeroize();
    }
}
impl ExtAcc {
    pub fn product<F: BenchField>(&mut self, x: F, y: F) {
        F::note(12, 4);
        F::note(13, 4);
        let [a, b] = x.words();
        let [c, d] = y.words();
        self.a.add(u128::from(a) * u128::from(c));
        self.b.add(u128::from(a) * u128::from(d));
        self.b.add(u128::from(b) * u128::from(c));
        self.c.add(u128::from(b) * u128::from(d));
    }
    pub fn reduce<F: BenchField>(self) -> F {
        F::note(14, 3);
        F::from_words([
            self.a.unsigned() + self.c.unsigned() * Fp::from_u64_reduced(7),
            self.b.unsigned(),
        ])
    }
}
pub fn fraction<F, N, const ONE: bool>(
    n: Option<(&[N], &[N])>,
    de: &[F],
    dd: &[F],
    lambda: F,
    lo: &[F],
    hi: &[F],
    num: impl Fn(F, N) -> F,
) -> [F; 3]
where
    F: BenchField,
    N: Copy + core::ops::Add<Output = N> + core::ops::Sub<Output = N>,
{
    let mut sums = [F::ZERO; 3];
    let group = 2 * lo.len();
    let scaled: tessera_codec::SecretBuf<F> = lo.iter().map(|&w| w * lambda).collect();
    for (h, &wh) in hi.iter().enumerate() {
        for start in (0..lo.len()).step_by(B) {
            let mut acc = [ExtAcc::default(); 3];
            for i in start..lo.len().min(start + B) {
                let p = h * group + 2 * i;
                let w = lo[i];
                let wl = scaled[i];
                for z in [0, 2, 1] {
                    if z == 1 && !ONE {
                        continue;
                    }
                    let line = |a: F, b: F| match z {
                        0 => a,
                        1 => b,
                        _ => b + b - a,
                    };
                    let d0 = line(de[p], de[p + 1]);
                    let d1 = line(dd[p], dd[p + 1]);
                    let (a, b) = match n {
                        None => (w + wl * d0, w),
                        Some((ne, no)) => {
                            let line_n = |a: N, b: N| match z {
                                0 => a,
                                1 => b,
                                _ => b + b - a,
                            };
                            (
                                num(w, line_n(ne[p], ne[p + 1])) + wl * d0,
                                num(w, line_n(no[p], no[p + 1])),
                            )
                        }
                    };
                    acc[z].product(a, d1);
                    acc[z].product(b, d0);
                }
            }
            for z in 0..3 {
                if z != 1 || ONE {
                    sums[z] += wh * acc[z].reduce::<F>();
                }
            }
            acc.zeroize();
        }
    }
    sums
}
