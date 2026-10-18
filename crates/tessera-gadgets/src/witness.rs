// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The gadgets' honest witnesses (RFC-0009): the committed tensors of the one assignment each
//! lemma admits, computed from the gadget's input and hidden classes.
//!
//! The input and the classes are secret-class (05 §3): the caller lends them for the call, and
//! what is computed from them lives in `SecretBuf`s, which clear their memory (OBS-6). Control
//! flow depends on the lengths only (INV-ARCH-2): every element is computed whether or not an
//! earlier one lies outside the gadget's domain, and that outcome, the one that depends on the
//! values, is reported after the last element. An honest input never has it (INV-IR-2).

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_tables::Table;

use crate::non_neg::limb;
use crate::requant::Form;
use crate::{Column, Committed, GadgetError, NonNeg, Requant};

/// A gadget's committed tensors, each with one entry per element of its input.
///
/// The tensors are secret-class (05 §3), each in a `SecretBuf` that clears its memory (OBS-6),
/// and the witness lends them, as slices, to the code that holds it. The debug form shows the
/// tensors' names and lengths, never an entry.
pub struct Witness {
    tensors: Vec<(Committed, SecretBuf<i64>)>,
}

impl Witness {
    /// The witness of `tensors`, as given. An honest witness comes from [`Requant::witness`] or
    /// [`NonNeg::witness`]; this one may be any assignment, such as a cheating prover's.
    pub fn from_tensors(tensors: Vec<(Committed, SecretBuf<i64>)>) -> Self {
        Self { tensors }
    }

    /// The entries of `tensor`, or `None` for a tensor the witness does not hold.
    ///
    /// The entries are secret-class (05 §3), lent to the code that computes on them, such as
    /// [`Lookups::new`](crate::Lookups::new). Their holder puts no entry, and nothing derived
    /// from one, in a log, an error, a metric or a report; hands them only to code that
    /// `secret-access.toml` lists (OBS-1); and copies none out, since the clearing cannot reach a
    /// copy (OBS-6).
    pub fn tensor(&self, tensor: Committed) -> Option<&[i64]> {
        self.tensors
            .iter()
            .find(|(held, _)| *held == tensor)
            .map(|(_, values)| &values[..])
    }
}

/// Shows the tensors' names and lengths, never an entry (05 §3).
impl fmt::Debug for Witness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.tensors
                    .iter()
                    .map(|(tensor, values)| (tensor, values.len())),
            )
            .finish()
    }
}

impl Requant {
    /// The honest witness for `keys` and their `classes`, element by element:
    ///
    /// - **`Requant1`:** `y`, the table's value at `pack(c, z)`.
    /// - **`RequantSplit`:** `z_lo = z_off mod 2^s`, and `y`, the reduced table's value at
    ///   `⌊z / 2^s⌋`, which RFC-0009 Lemma 2 shows is `clamp(⌊(z + 2^{r_c−1}) / 2^{r_c}⌋, lo, hi)`.
    /// - **`RequantWide`:** `z_lo = z_off mod 2^s` when `s ≥ 1`, `h0` and `h1_off` the digits of
    ///   `⌊z_off / 2^s⌋` in radix `2^t`, `g` the saturation table's value at `2·h1` and `y` the
    ///   reduced table's at `w' = h0 + 2^t·g` (RFC-0024 Lemma 11).
    ///
    /// `classes` holds each element's class, its output channel's (RFC-0007). It is required
    /// when the gadget has several classes; for one, it may be left out, or hold zeros.
    ///
    /// # Errors
    ///
    /// In this order: [`GadgetError::Missing`] for a gadget of several classes without
    /// `classes`; [`GadgetError::Length`] for classes of another length than `keys`;
    /// [`GadgetError::Input`] when a key lies outside `[−2^{b−1}, 2^{b−1})` or a class is not
    /// below `n`, after every element is computed.
    pub fn witness(&self, keys: &[i64], classes: Option<&[u8]>) -> Result<Witness, GadgetError> {
        let classes = class_column(self.classes(), keys.len(), classes)?;
        let class = |i: usize| classes.map_or(0, |c| c[i]);
        let len = keys.len();
        let half = 1i64 << (self.key_bits() - 1);
        // Whether every key and class lies in the gadget's domain, folded without branching. The
        // tables check it: each gadget looks up a key derived from `z`, which its table holds
        // exactly when `z` lies in `[−2^{b−1}, 2^{b−1})`, and its output with the class.
        let mut inside = true;
        let tensors = match &self.form {
            Form::One { table, .. } => {
                let mut y = SecretBuf::with_capacity(len);
                for (i, &z) in keys.iter().enumerate() {
                    let value = table.lookup(z, class(i));
                    inside &= value.is_some();
                    y.push(value.unwrap_or(0));
                }
                vec![(Committed::Y, y)]
            }
            Form::Split { b, split, high, .. } => {
                let (b, split) = (*b, *split);
                let mask = (1i64 << split) - 1;
                // `⌊z / 2^s⌋ = ⌊z_off / 2^s⌋ − 2^{b−1−s}`: `2^{b−1}` is a multiple of `2^s`.
                let reduced_half = 1i64 << (b - 1 - split);
                let mut z_lo = SecretBuf::with_capacity(len);
                let mut y = SecretBuf::with_capacity(len);
                for (i, &z) in keys.iter().enumerate() {
                    // Wrapping, so that a key far outside cannot overflow before its check.
                    let z_off = z.wrapping_add(half);
                    // The reduced table holds `⌊z / 2^s⌋` exactly when `z` lies in its width,
                    // and the class exactly when it is below `n`: its lookup checks both.
                    let value = high.lookup((z_off >> split).wrapping_sub(reduced_half), class(i));
                    inside &= value.is_some();
                    z_lo.push(z_off & mask);
                    y.push(value.unwrap_or(0));
                }
                vec![(Committed::ZLo, z_lo), (Committed::Y, y)]
            }
            Form::Wide(wide) => {
                let (s, t) = (wide.split, wide.middle);
                let (low_mask, mid_mask) = ((1i64 << s) - 1, (1i64 << t) - 1);
                let top_half = 1i64 << (wide.top_bits - 1);
                let (sign, high) = (&wide.sign, &wide.high);
                let mut z_lo = wide.low.as_ref().map(|_| SecretBuf::with_capacity(len));
                let mut h0 = SecretBuf::with_capacity(len);
                let mut g = SecretBuf::with_capacity(len);
                let mut y = SecretBuf::with_capacity(len);
                for (i, &z) in keys.iter().enumerate() {
                    let z_off = z.wrapping_add(half);
                    let w_off = z_off >> s;
                    let middle_digit = w_off & mid_mask;
                    // `h1 = h1_off − 2^{k−1}`, with `h1_off = ⌊w_off / 2^t⌋`. The saturation
                    // table holds `2·h1` exactly when `z` lies in its width, and its lookup
                    // checks it; `w'` always lies in the output table, whose lookup checks the
                    // class.
                    let h1 = (w_off >> t).wrapping_sub(top_half);
                    let saturated = sign.lookup(h1.wrapping_mul(2), 0);
                    inside &= saturated.is_some();
                    let saturated = saturated.unwrap_or(0);
                    let value = high.lookup(middle_digit + saturated * (1i64 << t), class(i));
                    inside &= value.is_some();
                    if let Some(z_lo) = &mut z_lo {
                        z_lo.push(z_off & low_mask);
                    }
                    h0.push(middle_digit);
                    g.push(saturated);
                    y.push(value.unwrap_or(0));
                }
                let mut tensors = Vec::with_capacity(4);
                if let Some(z_lo) = z_lo {
                    tensors.push((Committed::ZLo, z_lo));
                }
                tensors.extend([(Committed::H0, h0), (Committed::G, g), (Committed::Y, y)]);
                tensors
            }
        };
        match inside {
            true => Ok(Witness { tensors }),
            false => Err(GadgetError::Input {
                rule: "every key lies in [−2^{b−1}, 2^{b−1}) and every class below n (INV-IR-2)",
            }),
        }
    }
}

impl NonNeg {
    /// The honest witness for the virtual values `values`: the limbs `v_j`, the digits of `v` in
    /// radix `2^L`, for `j < n − 1`. RFC-0009 Lemma 3 shows that they are the only limbs that
    /// pass, and that none do unless `0 ≤ v < 2^k`.
    ///
    /// # Errors
    ///
    /// [`GadgetError::Input`] when a value lies outside `[0, 2^k)`, after every element is
    /// computed.
    pub fn witness(&self, values: &[i64]) -> Result<Witness, GadgetError> {
        let committed = self.limbs() - 1;
        // A committed limb's table `Range { L }` exists, so `L ≤ 32`; without one, no mask.
        let mask = match committed {
            0 => 0,
            _ => (1i64 << self.limb_bits) - 1,
        };
        let bound = 1i64 << self.bits;
        let mut limbs: Vec<SecretBuf<i64>> = (0..committed)
            .map(|_| SecretBuf::with_capacity(values.len()))
            .collect();
        let mut inside = true;
        for &v in values {
            inside &= (0 <= v) & (v < bound);
            for (j, column) in (0..committed).zip(limbs.iter_mut()) {
                column.push((v >> (j * self.limb_bits)) & mask);
            }
        }
        match inside {
            true => Ok(Witness {
                tensors: (0..committed)
                    .map(|j| Committed::Limb(limb(j)))
                    .zip(limbs)
                    .collect(),
            }),
            false => Err(GadgetError::Input {
                rule: "every value lies in [0, 2^k) (RFC-0009 Lemma 3)",
            }),
        }
    }
}

/// The classes `classes`, checked against `len` keys of a gadget of `count` classes.
fn class_column(
    count: usize,
    len: usize,
    classes: Option<&[u8]>,
) -> Result<Option<&[u8]>, GadgetError> {
    match classes {
        Some(column) if column.len() != len => Err(GadgetError::Length {
            column: Column::Classes,
            len: column.len(),
            expected: len,
        }),
        Some(column) => Ok(Some(column)),
        None if count > 1 => Err(GadgetError::Missing {
            column: Column::Classes,
        }),
        None => Ok(None),
    }
}
