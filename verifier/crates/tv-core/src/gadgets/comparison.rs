// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemma 4 and RFC-0050's six grouped comparison arrays.
//!
//! These relations are affine in the raw coefficient tables and witness tables,
//! so their multilinear extensions satisfy the same equations at any common
//! point. Raw product coefficients must be extensions of their row-wise products,
//! not products of extensions: the caller proves those nonlinear reductions.
//!
//! The plan derives carries from inclusive public coefficient bounds. Those bounds
//! must already follow from bound inputs (SP-7). Every witness value still needs
//! its range lookup and its canonical source binding; [`Checked::claims`] retains
//! every raw input and witness claim, with the required range width. No input or
//! final carry disappears when a constant side or a high limb happens to be zero.

use super::GadgetError;
use crate::advice::ComparisonRole;
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};

/// Verifier-derived public comparison geometry and outgoing carry boxes.
#[derive(Debug)]
pub struct Comparison {
    bits: u8,
    radix: Fp2,
    left_carries: Vec<u8>,
    right_carries: Vec<u8>,
}

/// The identity of a comparison input, independent of its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// A raw coefficient on the left, whose public bound is established upstream.
    Left,
    /// A raw coefficient on the right.
    Right,
    /// One of RFC-0050's six canonical grouped arrays.
    Witness(ComparisonRole),
}

/// Claimed multilinear evaluations at one verifier-derived point. The six arrays
/// are left digits, left carries, right digits, right carries, differences, borrows.
/// These are evaluations of polynomials, not integers to range-check directly.
#[derive(Clone, Copy)]
pub struct Values<'a> {
    /// The common point, LSB-first.
    pub point: &'a [Fp2],
    /// Raw left coefficients.
    pub left: &'a [Fp2],
    /// Raw right coefficients.
    pub right: &'a [Fp2],
    /// The complete grouped witness, including both final carries and the final borrow.
    pub arrays: [&'a [Fp2]; 6],
}

/// One outstanding binding of a local comparison check.
#[derive(Clone, Copy)]
pub struct Claim<'a> {
    /// Which raw input or canonical witness role it names.
    pub input: Input,
    /// Its limb, least significant first.
    pub limb: usize,
    /// The common point.
    pub point: &'a [Fp2],
    /// The claimed evaluation; not a range-checked integer at a non-Boolean point.
    pub value: Fp2,
    /// The prescribed Range width for a witness table's lookup claim; `None` for
    /// raw coefficients, whose public bounds and nonlinear definitions are upstream.
    pub range_bits: Option<u8>,
}

/// A comparison whose affine equations passed, with all binding obligations intact.
#[must_use = "bind every raw input and witness claim, including every range lookup"]
pub struct Checked<'a> {
    plan: &'a Comparison,
    values: Values<'a>,
}

impl Comparison {
    /// Derive the complete comparison from public inclusive coefficient bounds.
    /// `bits` is the radix exponent, 1 through 32; both nonempty bound vectors have
    /// the complete limb count, including enough high zero limbs for both inputs.
    ///
    /// # Errors
    /// `Internal` for inconsistent verifier-owned dimensions, a bound at least
    /// `2^40`, a carry lookup wider than 32 bits, an equation's full lookup box at
    /// least `2^41`, or a nonzero final maximum carry; `Memory` for failed
    /// reservation. No proof metadata is read.
    pub fn derive(bits: u8, left: &[u64], right: &[u64]) -> Result<Self, GadgetError> {
        if !(1..=32).contains(&bits) || left.is_empty() || left.len() != right.len() {
            return Err(GadgetError::Internal);
        }
        left.len().checked_mul(8).ok_or(GadgetError::Memory)?;
        let radix = 1u64
            .checked_shl(u32::from(bits))
            .ok_or(GadgetError::Internal)?;
        Ok(Self {
            bits,
            radix: Fp2::from(Fp::reduce(radix)),
            left_carries: carry_widths(left, radix)?,
            right_carries: carry_widths(right, radix)?,
        })
    }

    /// Complete number of limbs in every one of the six arrays.
    pub fn limbs(&self) -> usize {
        self.left_carries.len()
    }

    /// Verifier-derived Range widths for each limb's outgoing left carry.
    pub fn left_carries(&self) -> &[u8] {
        &self.left_carries
    }

    /// Verifier-derived Range widths for each limb's outgoing right carry.
    pub fn right_carries(&self) -> &[u8] {
        &self.right_carries
    }

    /// Check left normalization, right normalization and subtraction in limb
    /// order, then both terminal carries and the terminal borrow, all at one point.
    /// The caller has absorbed the claimed values at their specified transcript step.
    ///
    /// # Errors
    /// `Rejected { gadget }` for any wrong array length or failed equation. A
    /// passing result still owes every source, range and raw-input reduction claim.
    pub fn check<'a>(
        &'a self,
        gadget: GadgetId,
        values: Values<'a>,
    ) -> Result<Checked<'a>, GadgetError> {
        let reject = GadgetError::Rejected { gadget };
        if values.left.len() != self.limbs()
            || values.right.len() != self.limbs()
            || values.arrays.iter().any(|a| a.len() != self.limbs())
        {
            return Err(reject);
        }
        let [ld, lc, rd, rc, difference, borrow] = values.arrays;
        let (mut carry_l, mut carry_r, mut previous_borrow) = (Fp2::ZERO, Fp2::ZERO, Fp2::ZERO);
        for limb in 0..self.limbs() {
            let at = |a: &[Fp2]| a.get(limb).copied().ok_or(GadgetError::Internal);
            if at(values.left)? + carry_l != at(ld)? + self.radix * at(lc)?
                || at(values.right)? + carry_r != at(rd)? + self.radix * at(rc)?
                || at(rd)? - at(ld)? - previous_borrow != at(difference)? - self.radix * at(borrow)?
            {
                return Err(reject);
            }
            carry_l = at(lc)?;
            carry_r = at(rc)?;
            previous_borrow = at(borrow)?;
        }
        if carry_l != Fp2::ZERO || carry_r != Fp2::ZERO || previous_borrow != Fp2::ZERO {
            return Err(reject);
        }
        Ok(Checked { plan: self, values })
    }

    fn width(&self, input: Input, limb: usize) -> Option<u8> {
        match input {
            Input::Left | Input::Right => None,
            Input::Witness(ComparisonRole::LeftCarry) => self.left_carries.get(limb).copied(),
            Input::Witness(ComparisonRole::RightCarry) => self.right_carries.get(limb).copied(),
            Input::Witness(ComparisonRole::Borrow) => Some(1),
            Input::Witness(
                ComparisonRole::LeftDigit | ComparisonRole::RightDigit | ComparisonRole::Difference,
            ) => Some(self.bits),
        }
    }
}

impl<'a> Checked<'a> {
    /// Every outstanding claim, raw left/right first, then the six arrays in
    /// RFC-0050 order. The driver maps witness roles to the same node/comparison's
    /// advice columns, discharges their specified lookups, and binds raw inputs.
    /// This iterator allocates nothing and never omits a constant or zero limb.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'a>> + '_ {
        let [ld, lc, rd, rc, difference, borrow] = self.values.arrays;
        [
            (Input::Left, self.values.left),
            (Input::Right, self.values.right),
            (Input::Witness(ComparisonRole::LeftDigit), ld),
            (Input::Witness(ComparisonRole::LeftCarry), lc),
            (Input::Witness(ComparisonRole::RightDigit), rd),
            (Input::Witness(ComparisonRole::RightCarry), rc),
            (Input::Witness(ComparisonRole::Difference), difference),
            (Input::Witness(ComparisonRole::Borrow), borrow),
        ]
        .into_iter()
        .flat_map(move |(input, values)| {
            values.iter().enumerate().map(move |(limb, &value)| Claim {
                input,
                limb,
                point: self.values.point,
                value,
                range_bits: self.plan.width(input, limb),
            })
        })
    }
}

fn carry_widths(bounds: &[u64], radix: u64) -> Result<Vec<u8>, GadgetError> {
    let mut widths = Vec::new();
    widths
        .try_reserve_exact(bounds.len())
        .map_err(|_| GadgetError::Memory)?;
    let (mut maximum, mut previous_box) = (0u64, 0u64);
    for &bound in bounds {
        if bound >= 1 << 40 {
            return Err(GadgetError::Internal);
        }
        maximum = bound
            .checked_add(maximum)
            .and_then(|x| x.checked_div(radix))
            .ok_or(GadgetError::Internal)?;
        let bits = u64::BITS.saturating_sub(maximum.leading_zeros()).max(1);
        if bits > 32 {
            return Err(GadgetError::Internal);
        }
        let full_box = 1u64
            .checked_shl(bits)
            .and_then(|x| x.checked_sub(1))
            .ok_or(GadgetError::Internal)?;
        let left = bound
            .checked_add(previous_box)
            .ok_or(GadgetError::Internal)?;
        let right = full_box
            .checked_mul(radix)
            .and_then(|x| x.checked_add(radix.checked_sub(1)?))
            .ok_or(GadgetError::Internal)?;
        if left >= 1 << 41 || right >= 1 << 41 {
            return Err(GadgetError::Internal);
        }
        widths.push(u8::try_from(bits).map_err(|_| GadgetError::Internal)?);
        previous_box = full_box;
    }
    if maximum != 0 {
        return Err(GadgetError::Internal);
    }
    Ok(widths)
}
