// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 NonNeg's affine lookup forms, including the narrowed virtual top.
//! The same public boxes specialize to RFC-0050's root input and root bytes.

use super::GadgetError;
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};

/// Public geometry of a decomposition in a box below `2^62 < P` (Lemma 0).
#[derive(Clone, Copy, Debug)]
pub struct Decomposition {
    top_bits: u8,
    limb_bits: u8,
    limbs: usize,
    radix: Fp2,
    top_inverse: Fp2,
}

/// Evaluations of the value, committed low limbs and prescribed range-lookup
/// key polynomials at their common verifier-derived point. The top is virtual.
#[derive(Clone, Copy)]
pub struct Values<'a> {
    /// The point, LSB-first.
    pub point: &'a [Fp2],
    /// The original value's multilinear extension at the point.
    pub value: Fp2,
    /// The committed low limbs, least significant first.
    pub low: &'a [Fp2],
    /// All range lookup keys, low limbs first and virtual top last.
    pub keys: &'a [Fp2],
}

/// A binding that remains after the affine forms match.
#[derive(Clone, Copy)]
pub struct Claim<'a> {
    /// `None` for the original input, `Some(i)` for limb `i`, including the top.
    pub limb: Option<usize>,
    /// The point.
    pub point: &'a [Fp2],
    /// The claimed evaluation, which need not be an integer at a non-Boolean point.
    pub value: Fp2,
    /// The prescribed Range width of a limb; `None` for the original input.
    pub range_bits: Option<u8>,
}

/// Matching affine forms, with every source and range claim still to discharge.
#[must_use = "bind the original value and all committed limbs, and check every range lookup"]
pub struct Checked<'a> {
    plan: Decomposition,
    values: Values<'a>,
}

impl Decomposition {
    /// Derive `ceil(bits/limb_bits)` limbs without allocating. A one-limb
    /// decomposition has no committed low limbs; its only key is the input.
    ///
    /// # Errors
    /// `Internal` for verifier-owned widths outside `1..=62` or `1..=32`.
    pub fn derive(bits: u8, limb_bits: u8) -> Result<Self, GadgetError> {
        if !(1..=62).contains(&bits) || !(1..=32).contains(&limb_bits) {
            return Err(GadgetError::Internal);
        }
        let limbs = usize::from(bits).div_ceil(usize::from(limb_bits));
        let shift = limbs
            .checked_sub(1)
            .and_then(|n| n.checked_mul(usize::from(limb_bits)))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(GadgetError::Internal)?;
        let top_scale = Fp::reduce(1u64.checked_shl(shift).ok_or(GadgetError::Internal)?);
        Ok(Self {
            top_bits: bits
                .checked_sub(u8::try_from(shift).map_err(|_| GadgetError::Internal)?)
                .ok_or(GadgetError::Internal)?,
            limb_bits,
            limbs,
            radix: Fp2::from(Fp::reduce(
                1u64.checked_shl(u32::from(limb_bits))
                    .ok_or(GadgetError::Internal)?,
            )),
            top_inverse: Fp2::from(top_scale.inverse().ok_or(GadgetError::Internal)?),
        })
    }

    /// The complete number of prescribed lookup keys, including the virtual top.
    pub const fn limbs(self) -> usize {
        self.limbs
    }

    /// Match every low key and the virtual top to their affine forms. The value
    /// and keys have already been fixed at the caller's specified transcript step.
    /// This does not range-check field evaluations as if they were integers.
    ///
    /// # Errors
    /// `Rejected { gadget }` for a wrong low/key count or any wrong affine key.
    pub fn check<'a>(
        self,
        gadget: GadgetId,
        values: Values<'a>,
    ) -> Result<Checked<'a>, GadgetError> {
        let reject = GadgetError::Rejected { gadget };
        if values.low.len().checked_add(1) != Some(self.limbs) || values.keys.len() != self.limbs {
            return Err(reject);
        }
        let (mut residual, mut scale) = (values.value, Fp2::ONE);
        for (&low, &key) in values.low.iter().zip(values.keys) {
            if low != key {
                return Err(reject);
            }
            residual -= scale * low;
            scale *= self.radix;
        }
        if values.keys.last().copied() != Some(residual * self.top_inverse) {
            return Err(reject);
        }
        Ok(Checked { plan: self, values })
    }
}

impl<'a> Checked<'a> {
    /// The original input followed by every limb's claim. The caller maps low
    /// limbs to the same node's canonical auxiliary roles and keeps the top as
    /// the virtual key; all Range widths include the narrowed top.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'a>> + '_ {
        let input = Claim {
            limb: None,
            point: self.values.point,
            value: self.values.value,
            range_bits: None,
        };
        core::iter::once(input).chain(self.values.keys.iter().enumerate().map(
            move |(limb, &value)| {
                let width = if limb.checked_add(1) == Some(self.plan.limbs) {
                    self.plan.top_bits
                } else {
                    self.plan.limb_bits
                };
                Claim {
                    limb: Some(limb),
                    point: self.values.point,
                    value,
                    range_bits: Some(width),
                }
            },
        ))
    }
}
