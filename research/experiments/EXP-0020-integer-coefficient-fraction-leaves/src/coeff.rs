// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Challenge-monomial grids. Domains/dispatch/cadence depend only on public metadata.
use crate::arith::{Acc, B};
use crate::{BenchField, Field, Fp, PrimeField};
use tessera_codec::{SecretBuf, Zeroize};
pub const RESET: usize = 1024;
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub k: u128,
    pub v: u128,
    pub m: u128,
}
thread_local! {pub static DOMAIN:core::cell::Cell<Option<Domain>>=const{core::cell::Cell::new(None)};}
pub fn coordinate_ok(x: u128) -> bool {
    x <= i64::MAX as u128
}
pub fn coefficient_ok(t: u128) -> bool {
    t < 1u128 << 127
}
pub fn bucket_ok(t: u128, cadence: usize) -> bool {
    if !coefficient_ok(t) || !cadence.is_power_of_two() {
        return false;
    }
    let mut x = Acc::default();
    x.weighted(t as i128, Fp::MODULUS - 1);
    let sh = cadence.trailing_zeros();
    sh < 64 && x.0[2] >> (63 - sh) == 0
}
pub fn bound(d: Domain, v: usize, cadence: usize) -> Option<u128> {
    let a = 3u128.checked_pow(v as u32)?;
    let a2 = a.checked_mul(a)?;
    if !coordinate_ok(a.checked_mul(d.k.max(d.v).max(d.m))?) {
        return None;
    }
    let mul = |xs: &[u128]| xs.iter().try_fold(1u128, |a, &b| a.checked_mul(b));
    let t = [
        Some(1),
        mul(&[2, a, d.k]),
        mul(&[2, a, d.v]),
        mul(&[a2, d.k, d.k]),
        mul(&[2, a2, d.k, d.v]),
        mul(&[a2, d.v, d.v]),
        mul(&[2, a, d.m]),
        mul(&[2, a2, d.m, d.k]),
        mul(&[2, a2, d.m, d.v]),
    ]
    .into_iter()
    .collect::<Option<Vec<_>>>()?
    .into_iter()
    .max()?;
    if !bucket_ok(t, cadence) {
        return None;
    }
    Some(t)
}
pub struct Raw<F: BenchField> {
    pub k: SecretBuf<i64>,
    pub v: SecretBuf<i64>,
    pub m: SecretBuf<i64>,
    pub alpha: F,
    pub beta: F,
    pub has_values: bool,
}
impl<F: BenchField> Raw<F> {
    #[allow(clippy::too_many_arguments)] // Explicit public input columns and padding tuple.
    pub fn new(
        keys: &[Fp],
        values: Option<&[Fp]>,
        neg_m: Option<&[Fp]>,
        count: usize,
        alpha: F,
        beta: F,
        pad_key: Fp,
        pad_value: Option<Fp>,
    ) -> Self {
        let center = |x: Fp| {
            let u = x.to_canonical_u64();
            (i128::from(u)
                - i128::from(Fp::MODULUS) * i128::from((u > (Fp::MODULUS - 1) / 2) as u8))
                as i64
        };
        let mut k = SecretBuf::with_capacity(count);
        let mut v = SecretBuf::with_capacity(if values.is_some() { count } else { 0 });
        let mut m = SecretBuf::with_capacity(if neg_m.is_some() { count } else { 0 });
        for i in 0..count {
            let present = i < keys.len();
            k.push(if present { keys[i] } else { pad_key }.to_canonical_u64() as i64);
            if let Some(vs) = values {
                v.push(center(if present {
                    vs[i]
                } else {
                    pad_value.unwrap_or(Fp::ZERO)
                }));
            }
            if let Some(ms) = neg_m {
                m.push(-center(ms[i]));
            }
        }
        F::note(
            20,
            (count * (1 + usize::from(values.is_some()) + usize::from(neg_m.is_some()))) as u64,
        );
        Self {
            k,
            v,
            m,
            alpha,
            beta,
            has_values: values.is_some(),
        }
    }
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn table(&self) -> bool {
        !self.m.is_empty()
    }
    pub fn denominator(&self, i: usize) -> F {
        let k = Fp::from_i64(self.k[i]);
        let v = if self.has_values {
            Fp::from_i64(self.v[i])
        } else {
            Fp::ZERO
        };
        self.alpha - self.beta.mul_by_base(v).add_base(k)
    }
    pub fn value(&self, i: usize) -> (F, F) {
        let n = if self.table() {
            F::from(-Fp::from_i64(self.m[i]))
        } else {
            F::ONE
        };
        (n, self.denominator(i))
    }
    pub fn cross(&self, i: usize) -> (F, F) {
        let d0 = self.denominator(i);
        let d1 = self.denominator(i + 1);
        let n = if self.table() {
            d1.mul_by_base(-Fp::from_i64(self.m[i])) + d0.mul_by_base(-Fp::from_i64(self.m[i + 1]))
        } else {
            d0 + d1
        };
        (n, d0 * d1)
    }
    pub fn bind(&self, cs: &[F]) -> Vec<SecretBuf<F>> {
        F::note(
            18,
            (self.len() * (1 + usize::from(self.has_values) + usize::from(self.table()))) as u64,
        );
        let weights = crate::contract::eq_weights(cs);
        let mut out: Vec<SecretBuf<F>> = (0..4)
            .map(|_| SecretBuf::with_capacity((self.len() / 2) >> cs.len()))
            .collect();
        for parity in 0..2 {
            for base in (0..self.len() / 2).step_by(weights.len()) {
                let (mut k, mut v, mut m) = (F::ZERO, F::ZERO, F::ZERO);
                for (j, &w) in weights.iter().enumerate() {
                    let i = 2 * (base + j) + parity;
                    k += w.mul_by_base(Fp::from_i64(self.k[i]));
                    if self.has_values {
                        v += w.mul_by_base(Fp::from_i64(self.v[i]));
                    }
                    if self.table() {
                        m += w.mul_by_base(Fp::from_i64(self.m[i]));
                    }
                }
                if self.table() {
                    out[parity].push(-m);
                }
                out[parity + 2].push(self.alpha - k - self.beta * v);
            }
        }
        out
    }
}
fn grid(input: &[i64], v: usize, out: &mut [i64; 27]) {
    out[..input.len()].copy_from_slice(input);
    let mut stride = 1;
    for k in 0..v {
        for g in (0..1usize << (v - k - 1)).rev() {
            for j in (0..stride).rev() {
                let a = out[g * 2 * stride + j];
                let b = out[g * 2 * stride + stride + j];
                out[g * 3 * stride + j] = a;
                out[g * 3 * stride + stride + j] = b;
                out[g * 3 * stride + 2 * stride + j] = 2 * b - a;
            }
        }
        stride *= 3;
    }
}
pub fn channels(k0: i64, k1: i64, v0: i64, v1: i64, m0: i64, m1: i64) -> [i128; 8] {
    let (k0, k1, v0, v1, m0, m1) = (
        i128::from(k0),
        i128::from(k1),
        i128::from(v0),
        i128::from(v1),
        i128::from(m0),
        i128::from(m1),
    );
    [
        k0 + k1,
        v0 + v1,
        k0 * k1,
        k0 * v1 + v0 * k1,
        v0 * v1,
        m0 + m1,
        m0 * k1 + m1 * k0,
        m0 * v1 + m1 * v0,
    ]
}
pub struct Window<F: BenchField> {
    pub grid: SecretBuf<F>,
    pub depth: usize,
    pub challenges: SecretBuf<F>,
}
impl<F: BenchField> Window<F> {
    #[allow(clippy::needless_range_loop)] // Fixed parallel coordinate arrays, not one iterable row.
    pub fn new(raw: &Raw<F>, point: &[F], lambda: F, v: usize) -> Self {
        let cells = 3usize.pow(v as u32);
        let block = 1 << v;
        let rest = &point[v..];
        let lo = rest.len().div_ceil(2);
        let el = crate::contract::eq_weights(&rest[..lo]);
        let eh = crate::contract::eq_weights(&rest[lo..]);
        let active: &[usize] = match (raw.has_values, raw.table()) {
            (true, false) => &[0, 1, 2, 3, 4],
            (true, true) => &[0, 1, 2, 3, 4, 5, 6, 7],
            (false, false) => &[0, 2],
            (false, true) => &[0, 2, 5, 6],
        };
        let mut sums = SecretBuf::filled([F::ZERO; 8], cells);
        let mut input = [[0i64; 8]; 6];
        let mut g = [[0i64; 27]; 6];
        for (h, &wh) in eh.iter().enumerate() {
            for start in (0..el.len()).step_by(RESET) {
                let mut acc = SecretBuf::filled([[Acc::default(); 2]; 8], cells);
                for (l, &wl) in el
                    .iter()
                    .enumerate()
                    .take((start + RESET).min(el.len()))
                    .skip(start)
                {
                    let base = (h * el.len() + l) * block;
                    for j in 0..block {
                        let i = 2 * (base + j);
                        input[0][j] = raw.k[i];
                        input[1][j] = raw.k[i + 1];
                        if raw.has_values {
                            input[2][j] = raw.v[i];
                            input[3][j] = raw.v[i + 1];
                        }
                        if raw.table() {
                            input[4][j] = raw.m[i];
                            input[5][j] = raw.m[i + 1];
                        }
                    }
                    F::note(
                        17,
                        ((2 + 2 * usize::from(raw.has_values) + 2 * usize::from(raw.table()))
                            * (0..v)
                                .map(|k| 3usize.pow(k as u32) * (1usize << (v - k - 1)))
                                .sum::<usize>()) as u64,
                    );
                    for c in 0..6 {
                        if c < 2 || (c < 4 && raw.has_values) || (c >= 4 && raw.table()) {
                            grid(&input[c][..block], v, &mut g[c]);
                        }
                    }
                    let ws = wl.words();
                    for z in 0..cells {
                        let mut c = [0i128; 8];
                        let k0 = i128::from(g[0][z]);
                        let k1 = i128::from(g[1][z]);
                        let v0 = i128::from(g[2][z]);
                        let v1 = i128::from(g[3][z]);
                        let m0 = i128::from(g[4][z]);
                        let m1 = i128::from(g[5][z]);
                        c[0] = k0 + k1;
                        c[2] = k0 * k1;
                        F::note(15, 1);
                        F::note(16, 1);
                        if raw.has_values {
                            c[1] = v0 + v1;
                            c[3] = k0 * v1 + v0 * k1;
                            c[4] = v0 * v1;
                            F::note(15, 3);
                            F::note(16, 2);
                        }
                        if raw.table() {
                            c[5] = m0 + m1;
                            c[6] = m0 * k1 + m1 * k0;
                            F::note(15, 2);
                            F::note(16, 2);
                            if raw.has_values {
                                c[7] = m0 * v1 + m1 * v0;
                                F::note(15, 2);
                                F::note(16, 1);
                            }
                        }
                        for &j in active {
                            for (a, &w) in ws.iter().enumerate() {
                                F::note(9, 2);
                                F::note(10, 1);
                                acc[z][j][a].weighted(c[j], w);
                            }
                        }
                    }
                }
                for z in 0..cells {
                    for &j in active {
                        F::note(11, 2);
                        sums[z][j] += wh * F::from_words(acc[z][j].map(|x| x.signed()));
                    }
                }
            }
        }
        input.zeroize();
        g.zeroize();
        let a = raw.alpha;
        let b = raw.beta;
        let la = lambda * a;
        let constant = la * a + if raw.table() { F::ZERO } else { a + a };
        let mut weights = [F::ZERO; 8];
        weights[0] = -la - if raw.table() { F::ZERO } else { F::ONE };
        weights[2] = lambda;
        if raw.has_values {
            weights[1] = -la * b - if raw.table() { F::ZERO } else { b };
            weights[3] = lambda * b;
            weights[4] = weights[3] * b;
        }
        if raw.table() {
            weights[5] = -a;
            weights[6] = F::ONE;
            if raw.has_values {
                weights[7] = b;
            }
        }
        let grid = sums
            .iter()
            .map(|s| {
                let mut q = constant;
                for &j in active {
                    if j == 6 {
                        q += s[j];
                    } else {
                        q += weights[j] * s[j];
                    }
                }
                q
            })
            .collect();
        Self {
            grid,
            depth: v,
            challenges: SecretBuf::with_capacity(v),
        }
    }
    pub fn sums(&self, point: &[F]) -> [F; 3] {
        let remaining = self.depth - self.challenges.len() - 1;
        let w = crate::contract::eq_weights(&point[..remaining]);
        let mut out = [F::ZERO; 3];
        for (i, &weight) in w.iter().enumerate() {
            let (mut q, mut stride, mut offset) = (i, 3, 0);
            for _ in 0..remaining {
                offset += (q & 1) * stride;
                q >>= 1;
                stride *= 3;
            }
            for (z, x) in out.iter_mut().enumerate() {
                *x += weight * self.grid[offset + z];
            }
        }
        out
    }
    pub fn bind(&mut self, c: F, half: F) {
        let mut next = SecretBuf::with_capacity(self.grid.len() / 3);
        for a in self.grid.chunks_exact(3) {
            next.push(
                a[0] + c * (a[1] - a[0]) + c * (c - F::ONE) * half * (a[2] - a[1].double() + a[0]),
            );
        }
        self.grid = next;
        self.challenges.push(c);
    }
}
