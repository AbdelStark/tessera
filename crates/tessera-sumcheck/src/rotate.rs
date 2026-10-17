// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Rotate` (RFC-0037 'Rotate'): RoPE's rotation of each pair as one product against a public
//! kernel, which leaves one claim on its input.
//!
//! RoPE pairs the entries of `X` whose flattened indices differ in bit `b`, and multiplies each
//! pair, low member first, by `M = [[C, −S], [S, C]]`: `(lo, hi) ↦ (lo·C − hi·S, lo·S + hi·C)`
//! (RFC-0005 'RoPE'). `C` and `S` depend on the pair's other bits `x_{−b}` only. The extension of
//! `M` in its row bit `y` and column bit `c` is `C·eq(y, c) + S·(y − c)`, so a claim on the
//! output at `r` is the sum
//!
//! ```text
//! Ỹ(r) = Σ_x E_r(x)·X̃(x),        E_r(x) = eq(r_{−b}, x_{−b})·(C(x_{−b})·eq(r_b, x_b) + S(x_{−b})·(r_b − x_b))
//! ```
//!
//! of degree 2. `E_r` is a public factor, which the verifier evaluates at the instance's point
//! `r'` as `eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩`, with `w = eq(r_{−b}, ·)·eq(r'_{−b}, ·)`.
//! The sumcheck ends with one evaluation, a claim on `X̃` at `r'`.

use core::fmt;

use tessera_codec::Secret;
use tessera_poly::PolyField;

use crate::contract::eq_weights;
use crate::{Claim, Instance, SumcheckError, Table};

/// `Rotate` (RFC-0037 'Rotate'): RoPE on the pairs of `X` that differ in bit `b`, as the product
/// `Ỹ(r) = Σ_x E_r(x)·X̃(x)` with its public kernel `E_r`, of degree 2.
///
/// It borrows `X`, of `2^n` entries over the prime subfield, and the pairs' constants `C` and `S`,
/// public, `2^{n−1}` entries each indexed by a pair's other bits: entry `k` belongs to the pair
/// whose index, with bit `b` removed, is `k`. The schedule lays RoPE's `(C, S)` out this way from
/// the positions and frequencies, broadcast over heads.
///
/// It is proven in the steps of the matmul and elementwise relations:
/// [`tables`](Self::tables) builds the kernel from the input claim's point,
/// [`prove_batch`](crate::prove_batch) proves [`RotateTables::instance`] among the stage's
/// instances, and [`RotateTables::claim`] gives the claim on `X`.
///
/// `X` is secret-class (05 §3): the debug form shows the shape only.
pub struct Rotate<'a, F: PolyField> {
    x: &'a [F::Base],
    cos: &'a [F::Base],
    sin: &'a [F::Base],
    bit: usize,
    vars: usize,
}

impl<F: PolyField> Clone for Rotate<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Rotate<'_, F> {}

impl<'a, F: PolyField> Rotate<'a, F> {
    /// RoPE on `x`, of `2^vars` entries, pairing the entries whose indices differ in bit `bit`,
    /// with the pairs' constants `cos` and `sin`, of `2^{vars−1}` entries each.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::RelationShape`] when `2^vars` is not a count of a `usize`;
    /// [`SumcheckError::PartnerBit`] when `bit` is not one of the `vars` variables; then
    /// [`SumcheckError::InputLength`] for the first of `x` (input 0), `cos` (1) and `sin` (2)
    /// whose length is not its `2^vars` or `2^{vars−1}`.
    pub fn new(
        x: &'a [F::Base],
        cos: &'a [F::Base],
        sin: &'a [F::Base],
        bit: usize,
        vars: usize,
    ) -> Result<Self, SumcheckError> {
        if vars >= usize::BITS as usize {
            return Err(SumcheckError::RelationShape {
                vars,
                summed_vars: 0,
            });
        }
        if bit >= vars {
            return Err(SumcheckError::PartnerBit {
                input: 0,
                bit,
                vars,
            });
        }
        let half = 1usize << (vars - 1);
        for (input, len, expected) in [
            (0, x.len(), 2 * half),
            (1, cos.len(), half),
            (2, sin.len(), half),
        ] {
            if len != expected {
                return Err(SumcheckError::InputLength { input, len, vars });
            }
        }
        Ok(Self {
            x,
            cos,
            sin,
            bit,
            vars,
        })
    }

    /// [`new`](Self::new) on a secret-class operand, in its [`Secret`]: the relation reads it in
    /// its sumcheck (05 §3 OBS-1).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    #[allow(clippy::disallowed_methods)]
    pub fn secret(
        x: &'a Secret<Vec<F::Base>>,
        cos: &'a [F::Base],
        sin: &'a [F::Base],
        bit: usize,
        vars: usize,
    ) -> Result<Self, SumcheckError> {
        Self::new(x.expose_secret(), cos, sin, bit, vars)
    }

    /// The degree of its sumcheck, 2: the kernel times `X`.
    pub fn degree(&self) -> usize {
        2
    }

    /// The variables of its instance, `n`.
    pub fn vars(&self) -> usize {
        self.vars
    }

    /// The bit the pairs differ in, `b`.
    pub fn bit(&self) -> usize {
        self.bit
    }

    /// The instance's tables for a claim on the output at `point`, its `n` coordinates, the
    /// lowest first: the kernel `E_r` at every entry, a public factor over `F`, and `X` as it is.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have `n` coordinates, before
    /// anything is computed.
    pub fn tables(&self, point: &[F]) -> Result<RotateTables<'a, F>, SumcheckError> {
        if point.len() != self.vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.vars,
            });
        }
        let r_b = point[self.bit];
        let rest: Vec<F> = point
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != self.bit)
            .map(|(_, &r)| r)
            .collect();
        let eq = eq_weights(&rest);
        // E_r at a pair's low member, x_b = 0: eq·(C·(1 − r_b) + S·r_b); at its high member,
        // x_b = 1: eq·(C·r_b + S·(r_b − 1)).
        let (one_minus, minus_one) = (F::ONE - r_b, r_b - F::ONE);
        let low = 1usize << self.bit;
        let mut kernel = vec![F::ZERO; 2 * eq.len()];
        for (k, ((&e, &c), &s)) in eq.iter().zip(self.cos).zip(self.sin).enumerate() {
            let x = ((k >> self.bit) << (self.bit + 1)) | (k & (low - 1));
            kernel[x] = e * (one_minus.mul_by_base(c) + r_b.mul_by_base(s));
            kernel[x | low] = e * (r_b.mul_by_base(c) + minus_one.mul_by_base(s));
        }
        Ok(RotateTables {
            x: self.x,
            kernel,
            vars: self.vars,
            bit: self.bit,
        })
    }
}

/// Shows the shape, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for Rotate<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rotate")
            .field("vars", &self.vars)
            .field("bit", &self.bit)
            .finish_non_exhaustive()
    }
}

/// The tables of a [`Rotate`]'s instance for one claim on its output, from [`Rotate::tables`]:
/// the public kernel `E_r` and `X`.
pub struct RotateTables<'a, F: PolyField> {
    x: &'a [F::Base],
    /// `E_r` at every entry, public.
    kernel: Vec<F>,
    vars: usize,
    bit: usize,
}

impl<F: PolyField> RotateTables<'_, F> {
    /// The instance for [`prove_batch`](crate::prove_batch): the product of the kernel, a public
    /// factor, and `X`, whose evaluation is the proof's one.
    pub fn instance(&self) -> Instance<'_, F> {
        Instance::new(vec![Table::ext(&self.kernel).public(), Table::base(self.x)])
    }

    /// The claim on `X̃` after the sumcheck, at the instance's point `r'`, its last `n`
    /// challenges ([`ProvenBatch::instance_point`]), with its one evaluation
    /// ([`ProvenBatch::instance_evals`]).
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `n`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when there is not one evaluation.
    ///
    /// [`ProvenBatch::instance_point`]: crate::ProvenBatch::instance_point
    /// [`ProvenBatch::instance_evals`]: crate::ProvenBatch::instance_evals
    pub fn claim(&self, point: &[F], evals: &[F]) -> Result<Claim<F>, SumcheckError> {
        if point.len() != self.vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.vars,
            });
        }
        match *evals {
            [value] => Ok(Claim {
                point: point.to_vec(),
                value,
            }),
            _ => Err(SumcheckError::EvaluationCount {
                len: evals.len(),
                expected: 1,
            }),
        }
    }
}

/// Shows the shape, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for RotateTables<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RotateTables")
            .field("vars", &self.vars)
            .field("bit", &self.bit)
            .finish_non_exhaustive()
    }
}
