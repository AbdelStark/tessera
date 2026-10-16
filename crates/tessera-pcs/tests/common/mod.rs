// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: a seeded generator, and the definitions the tests compare the crate
//! against, written from RFC-0002 'Constants' and RFC-0011 'Encoding' without the crate's
//! algorithms: roots of unity from `7^((P − 1)/m)`, evaluation by Horner's rule, the DFT as a
//! sum, and Lagrange interpolation.

#![allow(dead_code)] // each test binary uses part of it

use tessera_field::{Field, Goldilocks, PrimeField, batch_inverse};

pub type Fp = Goldilocks;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A uniform element: words at or above `P` are rejected.
    pub fn fp(&mut self) -> Fp {
        loop {
            if let Some(x) = Fp::from_canonical_u64(self.next_u64()) {
                return x;
            }
        }
    }

    pub fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    /// A uniform non-zero element.
    pub fn nonzero(&mut self) -> Fp {
        loop {
            let x = self.fp();
            if x != Fp::ZERO {
                return x;
            }
        }
    }

    /// The first `len` entries of a uniform permutation of `0..n` (Fisher–Yates).
    pub fn subset(&mut self, n: usize, len: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..len {
            let j = i + self.below(n - i);
            all.swap(i, j);
        }
        all.truncate(len);
        all
    }
}

pub fn fp(x: u64) -> Fp {
    Fp::from_u64_reduced(x)
}

/// `7`, the generator of `Fp^*` and the codeword's coset shift.
pub fn seven() -> Fp {
    fp(7)
}

/// `ω_m = 7^((P − 1)/m)` for a power of two `m`, from its definition.
pub fn root(m: usize) -> Fp {
    assert!(m.is_power_of_two() && m as u64 <= 1 << 32);
    seven().pow((Fp::MODULUS - 1) / m as u64)
}

/// `Σ_i coefficients[i]·x^i` by Horner's rule.
pub fn horner(coefficients: &[Fp], x: Fp) -> Fp {
    coefficients
        .iter()
        .rev()
        .fold(Fp::ZERO, |acc, &c| acc * x + c)
}

/// The values of `coefficients` at `shift·ω_n^j` for `j < n`, one evaluation per point.
pub fn dft(coefficients: &[Fp], shift: Fp) -> Vec<Fp> {
    let n = coefficients.len();
    let w = root(n);
    let mut x = shift;
    (0..n)
        .map(|_| {
            let y = horner(coefficients, x);
            x *= w;
            y
        })
        .collect()
}

/// The inverse of [`dft`] by its formula: `a_i = (1/n)·shift^{−i}·Σ_j y_j·ω_n^{−ij}`.
pub fn inverse_dft(values: &[Fp], shift: Fp) -> Vec<Fp> {
    let n = values.len();
    let n_inv = fp(n as u64).inverse().expect("n is a unit");
    let w_inv = root(n).inverse().expect("a root is a unit");
    let shift_inv = shift.inverse().expect("a non-zero shift");
    let mut out = Vec::with_capacity(n);
    let mut w_i = Fp::ONE; // ω^{−i}
    let mut s_i = n_inv; // shift^{−i}/n
    for _ in 0..n {
        // Σ_j y_j·(ω^{−i})^j, by Horner in ω^{−i}.
        out.push(horner(values, w_i) * s_i);
        w_i *= w_inv;
        s_i *= shift_inv;
    }
    out
}

/// The codeword's evaluation points `x_j = 7·ω_n^j`, `j < n`.
pub fn codeword_points(n: usize) -> Vec<Fp> {
    let w = root(n);
    let mut x = seven();
    (0..n)
        .map(|_| {
            let p = x;
            x *= w;
            p
        })
        .collect()
}

/// The polynomial of degree below `nodes.len()` through `(x_s, y_s)`, in barycentric Lagrange
/// form, evaluated at each of `at` (none of which is a node).
pub fn interpolate_at(nodes: &[(Fp, Fp)], at: &[Fp]) -> Vec<Fp> {
    // w_s = 1 / Π_{s' ≠ s} (x_s − x_s').
    let mut weights: Vec<Fp> = nodes
        .iter()
        .enumerate()
        .map(|(s, &(xs, _))| {
            nodes
                .iter()
                .enumerate()
                .filter(|&(t, _)| t != s)
                .fold(Fp::ONE, |acc, (_, &(xt, _))| acc * (xs - xt))
        })
        .collect();
    batch_inverse(&mut weights).expect("distinct nodes");
    at.iter()
        .map(|&x| {
            // L(x) = ℓ(x)·Σ_s w_s·y_s/(x − x_s), ℓ(x) = Π_s (x − x_s).
            let mut differences: Vec<Fp> = nodes.iter().map(|&(xs, _)| x - xs).collect();
            let ell = differences.iter().fold(Fp::ONE, |acc, &d| acc * d);
            batch_inverse(&mut differences).expect("x is not a node");
            let sum = nodes
                .iter()
                .zip(&weights)
                .zip(&differences)
                .fold(Fp::ZERO, |acc, ((&(_, ys), &ws), &d)| acc + ws * ys * d);
            ell * sum
        })
        .collect()
}

/// The degree test of RFC-0011 'Testing Strategy': the polynomial of degree below `degree`
/// through the symbols at the positions `subset` (exactly `degree` of them) predicts every other
/// symbol of `codeword`.
pub fn passes_degree_test(codeword: &[Fp], subset: &[usize], degree: usize) -> bool {
    assert_eq!(
        subset.len(),
        degree,
        "a degree-d test interpolates d points"
    );
    let points = codeword_points(codeword.len());
    let nodes: Vec<(Fp, Fp)> = subset.iter().map(|&j| (points[j], codeword[j])).collect();
    let mut inside = vec![false; codeword.len()];
    for &j in subset {
        inside[j] = true;
    }
    let others: Vec<usize> = (0..codeword.len()).filter(|&j| !inside[j]).collect();
    let at: Vec<Fp> = others.iter().map(|&j| points[j]).collect();
    let predicted = interpolate_at(&nodes, &at);
    others.iter().zip(predicted).all(|(&j, y)| codeword[j] == y)
}

/// A mask source that replays `coefficients` and counts its draws; it panics when asked for
/// more than it holds.
pub struct Replay {
    coefficients: Vec<Fp>,
    pub drawn: usize,
}

impl Replay {
    pub fn new(coefficients: Vec<Fp>) -> Self {
        Self {
            coefficients,
            drawn: 0,
        }
    }
}

impl tessera_pcs::MaskSource for Replay {
    fn draw(&mut self) -> Fp {
        let x = self.coefficients[self.drawn];
        self.drawn += 1;
        x
    }
}
