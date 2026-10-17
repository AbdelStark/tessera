// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RoPE's constants (RFC-0008 'Fixed-point generators', `Rope`'s frequencies and pairs): the
//! pairs `(C, S)` a `Rope` table holds for the `Rotate` relation. A `Rope` table is not a lookup
//! table; its digest still binds every pair.

use crate::TableError;
use crate::fixed::{TAU, cos_sin, exp_neg, ln};

/// The fractional bits of `smooth` in Llama-3's interpolated band.
const SMOOTH_BITS: u32 = 40;

/// A `Rope` table's constants: one Llama-3 scaled frequency for each of a head's `d_head / 2`
/// pairs of dimensions, and the pair `(C, S)` each gives at every position below `max_context`.
///
/// Row `j` of the table is the pair of position `⌊j / (d_head/2)⌋` and frequency
/// `j mod (d_head/2)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rope {
    max_context: u32,
    bits: u8,
    frequencies: Vec<u128>,
}

/// A `Rope` kind's parameters, checked against RFC-0008 'Table specifications'.
pub(crate) struct RopeParameters {
    pub(crate) theta: u32,
    pub(crate) factor: u16,
    pub(crate) low_freq_factor: u16,
    pub(crate) high_freq_factor: u16,
    pub(crate) original_max: u32,
    pub(crate) bits: u8,
    pub(crate) max_context: u32,
    pub(crate) d_head: u16,
}

impl Rope {
    /// The constants of checked parameters: every frequency, computed once.
    pub(crate) fn new(parameters: &RopeParameters) -> Result<Self, TableError> {
        let ln_theta = ln(u64::from(parameters.theta)).ok_or(TableError::Internal)?;
        let half = parameters
            .d_head
            .checked_div(2)
            .ok_or(TableError::Internal)?;
        let mut frequencies = Vec::new();
        frequencies
            .try_reserve_exact(usize::from(half))
            .map_err(|_| TableError::Memory)?;
        for i in 0..half {
            frequencies.push(frequency(parameters, ln_theta, i).ok_or(TableError::Internal)?);
        }
        Ok(Self {
            max_context: parameters.max_context,
            bits: parameters.bits,
            frequencies,
        })
    }

    /// The frequencies `ω_i` in Q64, `i` from 0 to `d_head/2 − 1`, after Llama-3's bands.
    pub fn frequencies(&self) -> &[u128] {
        &self.frequencies
    }

    /// The number of positions, `max_context`.
    pub fn positions(&self) -> u32 {
        self.max_context
    }

    /// The constants' fractional bits.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The number of pairs, `max_context · d_head / 2`: the table's `T`.
    pub fn pairs(&self) -> u64 {
        u64::from(self.max_context).saturating_mul(crate::spec::length(self.frequencies.len()))
    }

    /// The pair `(C, S)` of `position` and `frequency`, or `None` past the table.
    pub fn pair(&self, position: u32, frequency: usize) -> Option<(i64, i64)> {
        if position >= self.max_context {
            return None;
        }
        constants(*self.frequencies.get(frequency)?, position, self.bits)
    }

    /// The pair of row `index`, or `None` past the table's `T` rows.
    pub fn row(&self, index: u64) -> Option<(i64, i64)> {
        let half = crate::spec::length(self.frequencies.len());
        let position = u32::try_from(index.checked_div(half)?).ok()?;
        let frequency = usize::try_from(index.checked_rem(half)?).ok()?;
        self.pair(position, frequency)
    }
}

/// The pair of a position at the frequency `omega` (Q64): `(c, s) = cos_sin(position · ω)`, each
/// rounded half up to `bits` fractional bits as `(v · 2^bits + 2^61) >> 62`. `None` when the
/// angle is `2^100` or more, or `bits > 62`.
pub fn constants(omega: u128, position: u32, bits: u8) -> Option<(i64, i64)> {
    let angle = omega.checked_mul(u128::from(position))?;
    let (c, s) = cos_sin(angle)?;
    let scale = 1i128.checked_shl(u32::from(bits)).filter(|_| bits <= 62)?;
    let round = |v: i128| -> Option<i64> {
        let shifted = v
            .checked_mul(scale)?
            .checked_add(1 << 61)?
            .checked_shr(62)?;
        i64::try_from(shifted).ok()
    };
    Some((round(c)?, round(s)?))
}

/// `ω_i`: `(y, n) = exp_neg(⌊2i · ln(θ) / d_head⌋)` and `ω = y >> n`, the Q64 of
/// `θ^{−2i/d_head}`; then Llama-3 compares its wavelength `2π/ω` with `L = original_max`:
///
/// - kept when `TAU · high_freq_factor < L · ω`;
/// - divided, `⌊ω / factor⌋`, when `TAU · low_freq_factor > L · ω`;
/// - otherwise interpolated: with
///   `smooth = ⌊(L · ω − low_freq_factor · TAU) · 2^40 / ((high_freq_factor − low_freq_factor) · TAU)⌋`,
///   `⌊ω / factor⌋ + ⌊⌊ω · smooth / 2^40⌋ · (factor − 1) / factor⌋`.
fn frequency(parameters: &RopeParameters, ln_theta: u128, i: u16) -> Option<u128> {
    let exponent = u128::from(i)
        .checked_mul(2)?
        .checked_mul(ln_theta)?
        .checked_div(u128::from(parameters.d_head))?;
    let (y, n) = exp_neg(exponent)?;
    let omega = match u32::try_from(n) {
        Ok(shift) if shift < u128::BITS => y.checked_shr(shift)?,
        _ => 0,
    };
    let factor = u128::from(parameters.factor);
    let low = u128::from(parameters.low_freq_factor);
    let high = u128::from(parameters.high_freq_factor);
    let reach = u128::from(parameters.original_max).checked_mul(omega)?;
    if TAU.checked_mul(high)? < reach {
        return Some(omega);
    }
    let divided = omega.checked_div(factor)?;
    if TAU.checked_mul(low)? > reach {
        return Some(divided);
    }
    let smooth = reach
        .checked_sub(low.checked_mul(TAU)?)?
        .checked_mul(1 << SMOOTH_BITS)?
        .checked_div(high.checked_sub(low)?.checked_mul(TAU)?)?;
    let blend = omega
        .checked_mul(smooth)?
        .checked_shr(SMOOTH_BITS)?
        .checked_mul(factor.checked_sub(1)?)?
        .checked_div(factor)?;
    divided.checked_add(blend)
}
