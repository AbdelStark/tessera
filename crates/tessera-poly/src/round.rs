// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Univariate round polynomials in the evaluation form of RFC-0010 'Sumcheck messages'.

use core::fmt;
use core::marker::PhantomData;

use tessera_field::{Field, PrimeField};

use crate::{PolyError, PolyField};

/// The largest round-polynomial degree of any sumcheck: 10, the bound of a reduction stage in
/// `tessera-protocol/2`, whose masked reads raise a round's degree (RFC-0013 'Masked extensions'
/// 'Degree', RFC-0014 'Soundness budget'). A [`RoundPoly`] holds up to this many evaluations.
///
/// It is not every sumcheck's bound: tree levels, and every sumcheck of `tessera-protocol/1`,
/// stay at 7. A sumcheck's own bound is its protocol's and its kind's
/// (`tessera_sumcheck::Protocol::max_degree`), and a verifier compares
/// [`RoundPoly::degree`] with the degree its stage allows.
pub const MAX_DEGREE: usize = 10;

/// A sumcheck round polynomial `p` of degree `d`, as a round message carries it: its
/// evaluations at `0, 2, 3, …, d`, `d` values for `1 ≤ d ≤ MAX_DEGREE` (RFC-0010 'Sumcheck
/// messages').
///
/// `p(1)` is not sent. The receiver holds the round's claim `p(0) + p(1)` and derives
/// `p(1) = claim − p(0)`, so [`evaluate`](Self::evaluate) takes the claim with the challenge and
/// returns `p(r)`, the claim of the next round. The number of evaluations sets the degree, so a
/// verifier compares [`degree`](Self::degree) with the degree its relation allows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RoundPoly<F> {
    /// `p(0), p(2), …, p(d)`, then zeros up to `MAX_DEGREE` entries.
    evaluations: [F; MAX_DEGREE],
    degree: usize,
}

impl<F: Field> fmt::Debug for RoundPoly<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoundPoly")
            .field("degree", &self.degree)
            .field("evaluations", &self.evaluations())
            .finish()
    }
}

impl<F: Field> RoundPoly<F> {
    /// The round polynomial of degree `d = evaluations.len()` whose values at `0, 2, …, d` are
    /// `evaluations`.
    ///
    /// # Errors
    ///
    /// [`PolyError::Degree`] unless `1 ≤ d ≤ MAX_DEGREE`.
    pub fn new(evaluations: &[F]) -> Result<Self, PolyError> {
        let degree = evaluations.len();
        if !(1..=MAX_DEGREE).contains(&degree) {
            return Err(PolyError::Degree { degree });
        }
        let mut sent = [F::ZERO; MAX_DEGREE];
        sent[..degree].copy_from_slice(evaluations);
        Ok(Self {
            evaluations: sent,
            degree,
        })
    }

    /// The degree `d`.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The evaluations as sent: `p(0), p(2), …, p(d)`.
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations[..self.degree]
    }
}

impl<F: PolyField> RoundPoly<F> {
    /// `p(r)`, given the round's claim `p(0) + p(1)`.
    ///
    /// With `p(1) = claim − p(0)`, `p` is known at the nodes `0, 1, …, d`, and Lagrange
    /// interpolation over them gives
    ///
    /// ```text
    /// p(r) = Σ_k p(k)·w_k·Π_{m ≠ k} (r − m),    w_k = 1 / Π_{m ≠ k} (k − m) = (−1)^{d−k} / (k!·(d − k)!)
    /// ```
    ///
    /// The weights are integers of the prime subfield computed at compile time from its modulus,
    /// and the products `Π_{m ≠ k} (r − m)` come from prefix and suffix products, so the
    /// evaluation is exact, has no division, and is correct at the nodes too: `4d + 3`
    /// multiplications and `d + 1` products by a base constant.
    pub fn evaluate(&self, claim: F, r: F) -> F {
        let d = self.degree;
        let weights = &Weights::<F::Base>::TABLE[d];
        let p0 = self.evaluations[0];
        let at = |k: usize| match k {
            0 => p0,
            1 => claim - p0,
            k => self.evaluations[k - 1],
        };
        // diff[m] = r − m and prefix[k] = Π_{m < k} (r − m).
        let mut diff = [F::ZERO; MAX_DEGREE + 1];
        let mut prefix = [F::ONE; MAX_DEGREE + 1];
        let mut node = r;
        for m in 0..=d {
            diff[m] = node;
            node -= F::ONE;
            if m < d {
                prefix[m + 1] = prefix[m] * diff[m];
            }
        }
        // From the top node down, suffix = Π_{m > k} (r − m).
        let mut suffix = F::ONE;
        let mut value = F::ZERO;
        for k in (0..=d).rev() {
            let basis = (prefix[k] * suffix).mul_by_base(F::Base::from_u64_reduced(weights[k]));
            value += at(k) * basis;
            suffix *= diff[k];
        }
        value
    }
}

/// The interpolation weights of a prime field, as canonical integers: `TABLE[d][k] = w_k` for the
/// nodes `0, …, d`.
struct Weights<B>(PhantomData<B>);

impl<B: PrimeField> Weights<B> {
    const TABLE: [[u64; MAX_DEGREE + 1]; MAX_DEGREE + 1] = weights(B::MODULUS);
}

/// `(a·b) mod p`.
const fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    (a as u128 * b as u128 % p as u128) as u64
}

/// `a^e mod p`, by square-and-multiply.
const fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut acc = 1 % p;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, a, p);
        }
        a = mul_mod(a, a, p);
        e >>= 1;
    }
    acc
}

/// `table[d][k] = (−1)^{d−k} / (k!·(d − k)!) mod p` for `0 ≤ k ≤ d ≤ MAX_DEGREE`, inverses by
/// Fermat's little theorem (`p` is prime). Row 0 is unused.
const fn weights(p: u64) -> [[u64; MAX_DEGREE + 1]; MAX_DEGREE + 1] {
    // A characteristic of at most MAX_DEGREE has no interpolation over 0..=MAX_DEGREE: the
    // nodes collide. A compile-time error, raised when such a field is used.
    assert!(
        p > MAX_DEGREE as u64,
        "round polynomials need a field of characteristic above MAX_DEGREE"
    );
    let mut inverse_factorial = [0u64; MAX_DEGREE + 1];
    let mut factorial = 1u64;
    let mut k = 0;
    while k <= MAX_DEGREE {
        if k > 0 {
            factorial = mul_mod(factorial, k as u64, p);
        }
        inverse_factorial[k] = pow_mod(factorial, p - 2, p);
        k += 1;
    }
    let mut table = [[0u64; MAX_DEGREE + 1]; MAX_DEGREE + 1];
    let mut d = 1;
    while d <= MAX_DEGREE {
        let mut k = 0;
        while k <= d {
            let w = mul_mod(inverse_factorial[k], inverse_factorial[d - k], p);
            table[d][k] = if (d - k) % 2 == 0 { w } else { p - w };
            k += 1;
        }
        d += 1;
    }
    table
}
