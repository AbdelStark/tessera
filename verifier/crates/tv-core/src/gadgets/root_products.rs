// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Final-evaluation kernel of RFC-0050's raw root-product reductions.
//!
//! This evaluates a relation, not a proof. The stage's sumcheck multiplies it by
//! `eq(target, point)` and binds every operand evaluation to its canonical root
//! or input decomposition. The incoming coefficient claim is the normalization
//! equation's affine form, never a whole product reduced modulo Goldilocks.

use super::GadgetError;
use tv_field::Fp2;

/// Verifier-derived coefficient geometry for the complete byte-radix products.
#[derive(Clone, Copy, Debug)]
pub struct Products {
    root_limbs: usize,
    coefficients: usize,
}

impl Products {
    /// Derive from the public root width, 1 through 32. There are `ceil(w/8)`
    /// root digits and `ceil((2*w+55)/8)` full product coefficients (RFC-0050).
    ///
    /// # Errors
    /// `Internal` for a verifier-owned width outside 1 through 32.
    pub fn derive(width: u8) -> Result<Self, GadgetError> {
        if !(1..=32).contains(&width) {
            return Err(GadgetError::Internal);
        }
        let bits = usize::from(width)
            .checked_mul(2)
            .and_then(|n| n.checked_add(55))
            .ok_or(GadgetError::Internal)?;
        Ok(Self {
            root_limbs: usize::from(width).div_ceil(8),
            coefficients: bits.div_ceil(8),
        })
    }

    /// Root digits read by each instance, including the virtual narrowed top.
    pub const fn root_limbs(self) -> usize {
        self.root_limbs
    }

    /// Complete coefficient count on each side, including high zero coefficients.
    pub const fn coefficients(self) -> usize {
        self.coefficients
    }

    /// Operand evaluations per instance: root digits first, then four input limbs
    /// of widths `(16,16,16,7)`. This kernel does not discharge their range bindings.
    pub const fn evaluations(self) -> usize {
        self.root_limbs.saturating_add(4)
    }

    /// Evaluate coefficient `j` of `rho^2*u` or, with `increment`, `(rho+1)^2*u`
    /// from the instance's operand evaluations. The increment is the constant one
    /// at all field points, including points between real and padded rows.
    ///
    /// Symmetric square coefficients followed by the input convolution are used
    /// instead of the oracle's literal triple expansion. No full integer product
    /// is formed, truncated, converted to a field element, or re-decomposed.
    /// `None` means inconsistent verifier-derived coefficient/operand geometry.
    pub fn evaluate(self, j: usize, increment: bool, values: &[Fp2]) -> Option<Fp2> {
        if j >= self.coefficients || values.len() != self.evaluations() {
            return None;
        }
        let (roots, inputs) = values.split_at_checked(self.root_limbs)?;
        let mut coefficient = Fp2::ZERO;
        for (k, &u) in inputs.iter().enumerate() {
            let Some(s) = j.checked_sub(k.checked_mul(2)?) else {
                continue;
            };
            let mut square = Fp2::ZERO;
            for (i, &r) in roots.iter().enumerate() {
                let Some(h) = s.checked_sub(i) else {
                    continue;
                };
                if h < i {
                    continue;
                }
                let Some(&other) = roots.get(h) else {
                    continue;
                };
                let product = r * other;
                square += if h == i { product } else { product + product };
            }
            if increment {
                let r = roots.get(s).copied().unwrap_or(Fp2::ZERO);
                square += r + r;
                if s == 0 {
                    square += Fp2::ONE;
                }
            }
            coefficient += square * u;
        }
        Some(coefficient)
    }
}
