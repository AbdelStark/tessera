// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Malicious alternative-witness generators for adversarial gadget testing
//! (RFC-0009 'Testing Strategy', `07 §4`).
//!
//! This module is compiled only when `cfg(debug_assertions)` is enabled (test and debug builds)
//! and is absent from release builds (OBS-4). Every generator constructs a witness that violates
//! a specific uniqueness premise of RFC-0009 Lemmas 1–9 or RFC-0024 Lemma 11 while satisfying as
//! many other local constraints as possible, so tests can verify that the independent verifier
//! rejects every alternative witness.

use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_tables::Table;

use crate::non_neg::limb;
use crate::requant::Form;
use crate::{
    Argmax, ArgmaxEvaluation, ArgmaxWitness, BigIntCompare, Bracket, Committed, CompareWitness,
    GadgetError, Lookups, NonNeg, OneHot, OneHotWitness, Requant, RsqrtColumn, RsqrtRow, RsqrtTerm,
    RsqrtWitness, Side, Slack, SoftmaxColumn, SoftmaxEvaluation, SoftmaxRow, SoftmaxTerm,
    SoftmaxValue, SoftmaxWitness, Witness,
};
use tessera_tables::TableId;

/// Group pre-evaluated lookup columns into [`Lookups`] for adversarial test harnesses.
pub fn lookups_from_columns<B: PrimeField + PolyField<Base = B>>(
    columns: Vec<(TableId, SecretBuf<B>, Option<SecretBuf<B>>)>,
) -> Result<Lookups<B>, GadgetError> {
    Lookups::from_columns(columns)
}

/// Alternative witness perturbations for [`Requant`] (`Requant1`, `RequantSplit`, `RequantWide`;
/// RFC-0009 Lemmas 1–2, RFC-0024 Lemma 11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequantAlternative {
    /// Perturb the committed output `y` at `index` by `delta` (typically `-1` or `+1`).
    OutputDelta {
        /// Element index to perturb.
        index: usize,
        /// Signed perturbation added to the honest output `y`.
        delta: i64,
    },
    /// Perturb `z_lo` at `index` to `z_lo + 2^s` and adjust `y` (or `h0` in `RequantWide`) for
    /// `z_hi - 1` so the high lookup still matches a table entry and only `Range { s }` fails.
    SplitLowPlusPow2S {
        /// Element index to perturb.
        index: usize,
    },
    /// Perturb `z_lo` at `index` to `z_lo - 1`.
    SplitLowMinusOne {
        /// Element index to perturb.
        index: usize,
    },
    /// Perturb `h0` at `index` in `RequantWide` to `h0 + 2^t` and adjust `g` and `y` for
    /// `h1_off - 1`.
    WideMiddlePlusPow2T {
        /// Element index to perturb.
        index: usize,
    },
    /// Perturb `h0` at `index` in `RequantWide` to `h0 - 1`.
    WideMiddleMinusOne {
        /// Element index to perturb.
        index: usize,
    },
    /// Perturb the saturation flag `g` at `index` in `RequantWide` by `delta` (`-1` or `+1`).
    WideSaturationDelta {
        /// Element index to perturb.
        index: usize,
        /// Signed perturbation added to `g`.
        delta: i64,
    },
}

/// Alternative witness perturbations for [`NonNeg`] (RFC-0009 Lemma 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonNegAlternative {
    /// Perturb committed low limb `limb` at `index` to `v_j + 2^L`, compensating `v_{j+1} - 1`
    /// when `j + 1 < n - 1` so the virtual top limb is unchanged.
    LimbPlusPow2L {
        /// Low limb index `j < n - 1`.
        limb: u8,
        /// Element index to perturb.
        index: usize,
    },
    /// Perturb committed low limb `limb` at `index` to `v_j - 1`, compensating `v_{j+1} + 1`
    /// when `j + 1 < n - 1`.
    LimbMinusOne {
        /// Low limb index `j < n - 1`.
        limb: u8,
        /// Element index to perturb.
        index: usize,
    },
}

/// Alternative witness perturbations for [`BigIntCompare`] (RFC-0009 Lemma 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareAlternative {
    /// Construct a normalization and subtraction witness for inputs with `X > Y`.
    ///
    /// When `clear_final_borrow` is `false`, the subtraction equations hold at every limb and the
    /// final borrow is `1` (violating the terminal borrow zero-check). When `true`, the final
    /// borrow is forced to `0` (violating the top-limb subtraction equation).
    FalseComparison {
        /// Whether to force the final borrow `b_{m-1}` to `0`.
        clear_final_borrow: bool,
    },
    /// Perturb a normalized digit and its outgoing carry at `(side, limb, row)` while preserving
    /// the local normalization equation `raw + prev_carry = digit + 2^L * carry`.
    DigitCarryShift {
        /// Which input side (`Left` or `Right`) to perturb.
        side: Side,
        /// Limb index `j` to perturb.
        limb: usize,
        /// Row index to perturb.
        row: usize,
        /// When `true`, replaces `(d_j, c_j)` with `(d_j + 2^L, c_j - 1)`; when `false`,
        /// replaces `(d_j, c_j)` with `(d_j - 1, c_j + 1)`.
        add_radix: bool,
    },
}

/// Alternative witness and evaluation perturbations for [`SoftmaxRow`] (RFC-0009 Lemmas 5–6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxAlternative {
    /// Set product-tree leaves at masked positions (`i >= valid`) to `0` instead of `1`.
    MaskedLeavesZero,
    /// Set product-tree leaves at masked positions (`i >= valid`) to the unmasked difference
    /// `c_q - x_{q,i}` instead of `1`, optionally overriding `c_q` with `masked_max`.
    MaskedLeavesRealKeys {
        /// Optional override for the row maximum `c_q` (e.g. a future masked score).
        masked_max: Option<i64>,
    },
    /// Replace the causal maximum `c_q` with the second-largest valid score in `0..valid`.
    SecondLargestMax,
    /// Replace the causal maximum `c_q` with `honest_max + 1`, recomputing exponentials,
    /// reciprocal and slacks so all range/table lookups pass and only the product root fails.
    MaxPlusOne,
    /// Perturb the reciprocal `R_q` by `delta` (`-1` or `+1`), recomputing low slack limbs modulo
    /// `2^L` from the resulting field-reduced slacks.
    ReciprocalDelta(i64),
    /// On a padded row (`valid == 0`), set `column` (`Maximum` or `Reciprocal`) to a nonzero
    /// `value`.
    PaddedNonZero {
        /// Which padded scalar column to make nonzero.
        column: SoftmaxColumn,
        /// Nonzero value to place on the padded row.
        value: i64,
    },
}

/// Alternative witness perturbations for [`RsqrtRow`] (RFC-0009 Lemma 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsqrtAlternative {
    /// Perturb the root `ρ` at `row` by `delta` (`-1` or `+1`), recomputing root byte limbs and
    /// both `BigIntCompare` witnesses for the perturbed root.
    RootDelta {
        /// Row index to perturb.
        row: usize,
        /// Signed perturbation added to `ρ` (`-1` or `+1`).
        delta: i64,
        /// Whether to force the violated comparison bracket's final borrow to `0`.
        clear_final_borrow: bool,
    },
    /// On a padded row (`real[row] == false`), set `ρ` to a nonzero `root` and decompose its
    /// byte limbs (both comparison brackets still see `u = 0` and `0 <= 0`, isolating the
    /// padded-root zero-check).
    PaddedRoot {
        /// Padded row index (`real[row] == false`).
        row: usize,
        /// Nonzero root value to place on the padded row.
        root: i64,
    },
}

/// Alternative witness perturbations for [`OneHot`] (RFC-0009 Lemma 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneHotAlternative {
    /// Set indicator to `1` at two distinct positions `first` and `second`.
    TwoOnes {
        /// First position set to `1`.
        first: usize,
        /// Second position set to `1`.
        second: usize,
    },
    /// Set a single `1` at `index != token` inside `0..vocab`.
    WrongIndex {
        /// Wrong vocabulary index set to `1`.
        index: usize,
    },
    /// Set a single `1` at `index >= vocab` in the padded vocabulary tail.
    PaddedVocabularyOne {
        /// Padded vocabulary index (`vocab <= index < padded_len`) set to `1`.
        index: usize,
    },
    /// Set indicator at `index` to a non-Boolean `value` (e.g. `2` or `-1`).
    NonBoolean {
        /// Vocabulary index to perturb.
        index: usize,
        /// Non-Boolean field integer value.
        value: i64,
    },
}

/// Alternative witness perturbations for [`Argmax`] (RFC-0009 Lemma 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgmaxAlternative {
    /// Substitute the `OneHotWitness` of a later tied maximum `y' > y` so `Selected` and `OneHot`
    /// hold, isolating the strict-suffix `Range29` failure at the earlier tied index `y`.
    LaterTie {
        /// The later tied index `y' > y` whose `ℓ_{y'} == m`.
        later_index: usize,
    },
    /// Perturb the committed maximum `m` by `delta` (`-1` or `+1`).
    MaxDelta(i64),
}

impl Requant {
    /// Build an alternative witness for adversarial testing (RFC-0009 'Testing Strategy').
    pub fn malicious_witness(
        &self,
        keys: &[i64],
        classes: Option<&[u8]>,
        alt: RequantAlternative,
    ) -> Result<Witness, GadgetError> {
        let honest = self.witness(keys, classes)?;
        let mut tensors: Vec<(Committed, SecretBuf<i64>)> = Vec::new();
        for &role in &[Committed::ZLo, Committed::H0, Committed::G, Committed::Y] {
            if let Some(slice) = honest.tensor(role) {
                tensors.push((role, slice.iter().copied().collect()));
            }
        }
        fn find_mut(
            tensors: &mut [(Committed, SecretBuf<i64>)],
            role: Committed,
        ) -> Result<&mut SecretBuf<i64>, GadgetError> {
            tensors
                .iter_mut()
                .find(|(r, _)| *r == role)
                .map(|(_, buf)| buf)
                .ok_or(GadgetError::Shape {
                    rule: "requested Requant alternative role is absent from this pattern",
                })
        }
        let class_at = |i: usize| classes.map_or(0, |c| c[i]);
        match alt {
            RequantAlternative::OutputDelta { index, delta } => {
                let y = find_mut(&mut tensors, Committed::Y)?;
                y[index] += delta;
            }
            RequantAlternative::SplitLowPlusPow2S { index } => match &self.form {
                Form::Split { b, split, high, .. } => {
                    let z_off = keys[index] + (1i64 << (b - 1));
                    let bad_lo = (z_off & ((1i64 << split) - 1)) + (1i64 << split);
                    let bad_hi = (z_off - bad_lo) >> split;
                    let reduced_half = 1i64 << (b - 1 - split);
                    find_mut(&mut tensors, Committed::ZLo)?[index] = bad_lo;
                    if let Some(y_val) =
                        high.lookup(bad_hi.wrapping_sub(reduced_half), class_at(index))
                    {
                        find_mut(&mut tensors, Committed::Y)?[index] = y_val;
                    }
                }
                Form::Wide(wide) if wide.split > 0 => {
                    let z_off = keys[index] + (1i64 << (wide.b - 1));
                    let bad_lo = (z_off & ((1i64 << wide.split) - 1)) + (1i64 << wide.split);
                    let w_off = (z_off - bad_lo) >> wide.split;
                    let h0 = w_off & ((1i64 << wide.middle) - 1);
                    let h1_off = w_off >> wide.middle;
                    let top_half = 1i64 << (wide.top_bits - 1);
                    let h1 = h1_off.wrapping_sub(top_half);
                    find_mut(&mut tensors, Committed::ZLo)?[index] = bad_lo;
                    find_mut(&mut tensors, Committed::H0)?[index] = h0;
                    if let Some(g_val) = wide.sign.lookup(h1.wrapping_mul(2), 0) {
                        find_mut(&mut tensors, Committed::G)?[index] = g_val;
                        let w_prime = h0 + (g_val << wide.middle);
                        if let Some(y_val) = wide.high.lookup(w_prime, class_at(index)) {
                            find_mut(&mut tensors, Committed::Y)?[index] = y_val;
                        }
                    }
                }
                _ => {
                    return Err(GadgetError::Shape {
                        rule: "SplitLowPlusPow2S requires RequantSplit or RequantWide with s > 0",
                    });
                }
            },
            RequantAlternative::SplitLowMinusOne { index } => {
                find_mut(&mut tensors, Committed::ZLo)?[index] -= 1;
            }
            RequantAlternative::WideMiddlePlusPow2T { index } => {
                let Form::Wide(wide) = &self.form else {
                    return Err(GadgetError::Shape {
                        rule: "WideMiddlePlusPow2T requires RequantWide",
                    });
                };
                let z_off = keys[index] + (1i64 << (wide.b - 1));
                let w_off = z_off >> wide.split;
                let bad_h0 = (w_off & ((1i64 << wide.middle) - 1)) + (1i64 << wide.middle);
                let bad_h1_off = (w_off - bad_h0) >> wide.middle;
                let top_half = 1i64 << (wide.top_bits - 1);
                let bad_h1 = bad_h1_off.wrapping_sub(top_half);
                find_mut(&mut tensors, Committed::H0)?[index] = bad_h0;
                if let Some(g_val) = wide.sign.lookup(bad_h1.wrapping_mul(2), 0) {
                    find_mut(&mut tensors, Committed::G)?[index] = g_val;
                    let w_prime = bad_h0 + (g_val << wide.middle);
                    if let Some(y_val) = wide.high.lookup(w_prime, class_at(index)) {
                        find_mut(&mut tensors, Committed::Y)?[index] = y_val;
                    }
                }
            }
            RequantAlternative::WideMiddleMinusOne { index } => {
                find_mut(&mut tensors, Committed::H0)?[index] -= 1;
            }
            RequantAlternative::WideSaturationDelta { index, delta } => {
                find_mut(&mut tensors, Committed::G)?[index] += delta;
            }
        }
        Ok(Witness::from_tensors(tensors))
    }
}

impl NonNeg {
    /// Build an alternative witness for adversarial testing (RFC-0009 Lemma 3).
    pub fn malicious_witness(
        &self,
        values: &[i64],
        alt: NonNegAlternative,
    ) -> Result<Witness, GadgetError> {
        let honest = self.witness(values)?;
        let n = self.limbs();
        if n <= 1 {
            return Err(GadgetError::Shape {
                rule: "NonNeg alternative witness requires at least one committed low limb",
            });
        }
        let mut tensors: Vec<(Committed, SecretBuf<i64>)> = (0..n - 1)
            .map(|j| {
                let role = Committed::Limb(limb(j));
                let slice = honest.tensor(role).unwrap_or(&[]);
                (role, slice.iter().copied().collect())
            })
            .collect();
        let radix = 1i64 << self.limb_bits;
        match alt {
            NonNegAlternative::LimbPlusPow2L { limb: j, index } => {
                let j_usize = usize::from(j);
                if j_usize >= tensors.len() {
                    return Err(GadgetError::Shape {
                        rule: "NonNeg limb index out of range",
                    });
                }
                tensors[j_usize].1[index] += radix;
                if j_usize + 1 < tensors.len() {
                    tensors[j_usize + 1].1[index] -= 1;
                }
            }
            NonNegAlternative::LimbMinusOne { limb: j, index } => {
                let j_usize = usize::from(j);
                if j_usize >= tensors.len() {
                    return Err(GadgetError::Shape {
                        rule: "NonNeg limb index out of range",
                    });
                }
                tensors[j_usize].1[index] -= 1;
                if j_usize + 1 < tensors.len() {
                    tensors[j_usize + 1].1[index] += 1;
                }
            }
        }
        Ok(Witness::from_tensors(tensors))
    }
}

impl BigIntCompare {
    /// Build an alternative witness for adversarial testing (RFC-0009 Lemma 4).
    pub fn malicious_witness(
        &self,
        left: &[&[i64]],
        right: &[&[i64]],
        alt: CompareAlternative,
    ) -> Result<CompareWitness, GadgetError> {
        let n = self.limbs();
        if left.len() != n || right.len() != n {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare inputs have the parameterized limb count",
            });
        }
        let len = left[0].len();
        if left.iter().chain(right).any(|v| v.len() != len) {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare input columns have the same length",
            });
        }
        let limb_bits = self.limb_bits();
        let radix = 1i64 << limb_bits;
        let mask = (1i128 << limb_bits) - 1;
        let mut tensors: Vec<_> = self
            .constraints::<Goldilocks>()?
            .committed
            .into_iter()
            .map(|column| (column, SecretBuf::with_capacity(len)))
            .collect();
        for row in 0..len {
            for (side, input) in [left, right].into_iter().enumerate() {
                let mut carry = 0i128;
                for (j, column) in input.iter().enumerate() {
                    let sum = i128::from(column[row]) + carry;
                    let digit = (sum & mask) as i64;
                    carry = sum >> limb_bits;
                    let offset = side * 2 * n + 2 * j;
                    tensors[offset].1.push(digit);
                    tensors[offset + 1].1.push(carry as i64);
                }
            }
            let mut borrow = 0i64;
            for j in 0..n {
                let x = tensors[2 * j].1[row];
                let y = tensors[2 * n + 2 * j].1[row];
                let difference = y - x - borrow;
                borrow = i64::from(difference < 0);
                tensors[4 * n + 2 * j]
                    .1
                    .push((i128::from(difference) & mask) as i64);
                tensors[4 * n + 2 * j + 1].1.push(borrow);
            }
        }
        match alt {
            CompareAlternative::FalseComparison { clear_final_borrow } => {
                if clear_final_borrow {
                    let final_borrow_idx = 4 * n + 2 * (n - 1) + 1;
                    for b in tensors[final_borrow_idx].1.iter_mut() {
                        *b = 0;
                    }
                }
            }
            CompareAlternative::DigitCarryShift {
                side,
                limb,
                row,
                add_radix,
            } => {
                let side_idx = usize::from(side == Side::Right);
                let offset = side_idx * 2 * n + 2 * limb;
                if add_radix {
                    tensors[offset].1[row] += radix;
                    tensors[offset + 1].1[row] -= 1;
                } else {
                    tensors[offset].1[row] -= 1;
                    tensors[offset + 1].1[row] += 1;
                }
            }
        }
        Ok(CompareWitness::from_tensors(tensors))
    }
}

impl SoftmaxRow {
    /// Build an alternative witness for adversarial testing (RFC-0009 Lemmas 5–6).
    pub fn malicious_witness(
        &self,
        scores: &[i64],
        valid: usize,
        alt: SoftmaxAlternative,
    ) -> Result<SoftmaxWitness, GadgetError> {
        match alt {
            SoftmaxAlternative::MaskedLeavesZero => self.witness(scores, valid),
            SoftmaxAlternative::MaskedLeavesRealKeys { masked_max } => {
                let honest = self.witness(scores, valid)?;
                let Some(override_max) = masked_max else {
                    return Ok(honest);
                };
                self.witness_with_max_and_reciprocal_delta(scores, valid, override_max, 0)
            }
            SoftmaxAlternative::SecondLargestMax => {
                let honest = self.witness(scores, valid)?;
                let max = honest.tensor(SoftmaxColumn::Maximum).map_or(0, |v| v[0]);
                let second = scores[..valid]
                    .iter()
                    .copied()
                    .filter(|&x| x < max)
                    .max()
                    .unwrap_or(max - 1);
                self.witness_with_max_and_reciprocal_delta(scores, valid, second, 0)
            }
            SoftmaxAlternative::MaxPlusOne => {
                let honest = self.witness(scores, valid)?;
                let max = honest.tensor(SoftmaxColumn::Maximum).map_or(0, |v| v[0]);
                self.witness_with_max_and_reciprocal_delta(scores, valid, max + 1, 0)
            }
            SoftmaxAlternative::ReciprocalDelta(delta) => {
                let honest = self.witness(scores, valid)?;
                let max = honest.tensor(SoftmaxColumn::Maximum).map_or(0, |v| v[0]);
                self.witness_with_max_and_reciprocal_delta(scores, valid, max, delta)
            }
            SoftmaxAlternative::PaddedNonZero { column, value } => {
                let honest = self.witness(scores, valid)?;
                let mut tensors = honest.into_tensors();
                for (col, buf) in &mut tensors {
                    if *col == column {
                        buf[0] = value;
                    }
                }
                Ok(SoftmaxWitness::from_tensors(tensors))
            }
        }
    }

    /// Evaluate an alternative witness, applying leaf-column perturbations for
    /// [`SoftmaxAlternative::MaskedLeavesZero`] and [`SoftmaxAlternative::MaskedLeavesRealKeys`].
    pub fn malicious_evaluate<B: PrimeField + PolyField<Base = B>>(
        &self,
        scores: &[i64],
        valid: usize,
        witness: &SoftmaxWitness,
        alt: SoftmaxAlternative,
    ) -> Result<SoftmaxEvaluation<B>, GadgetError> {
        let mut evaluated = self.evaluate::<B>(scores, valid, witness)?;
        let max = B::from_i64(witness.tensor(SoftmaxColumn::Maximum).map_or(0, |v| v[0]));
        match alt {
            SoftmaxAlternative::MaskedLeavesZero => {
                for i in valid..scores.len() {
                    evaluated.leaves[i] = B::ZERO;
                }
            }
            SoftmaxAlternative::MaskedLeavesRealKeys { .. } => {
                for (i, &score) in scores.iter().enumerate().skip(valid) {
                    evaluated.leaves[i] = max - B::from_i64(score);
                }
            }
            _ => {
                let _ = SoftmaxTerm::Virtual(SoftmaxValue::Leaf);
            }
        }
        Ok(evaluated)
    }

    fn witness_with_max_and_reciprocal_delta(
        &self,
        scores: &[i64],
        valid: usize,
        maximum: i64,
        reciprocal_delta: i64,
    ) -> Result<SoftmaxWitness, GadgetError> {
        let exp = self.exp_table();
        let mut e = SecretBuf::with_capacity(scores.len());
        let mut sum = 0i64;
        for (i, &x) in scores.iter().enumerate() {
            let key = if i < valid {
                i128::from(maximum) - i128::from(x)
            } else {
                i128::from(exp.row_count() - 1)
            };
            let clamped = key.clamp(0, i128::from(exp.row_count() - 1)) as i64;
            let value = exp.lookup(clamped, 0).unwrap_or(0);
            sum += value;
            e.push(value);
        }
        let psi = i64::from(valid != 0);
        let denominator = sum.max(1) as u64;
        let honest_reciprocal = (self.mass_bound() / denominator) as i64 * psi;
        let reciprocal = honest_reciprocal + reciprocal_delta;
        let a_f = Goldilocks::from_u64_reduced(psi as u64)
            * Goldilocks::from_u64_reduced(self.mass_bound())
            - Goldilocks::from_i64(reciprocal) * Goldilocks::from_i64(sum);
        let b_f = Goldilocks::from_i64(sum) - Goldilocks::from_u64_reduced(psi as u64) - a_f;
        let mut tensors = vec![
            (SoftmaxColumn::Exponentials, e),
            (
                SoftmaxColumn::Reciprocal,
                core::iter::once(reciprocal).collect(),
            ),
            (SoftmaxColumn::Maximum, core::iter::once(maximum).collect()),
        ];
        let limb_bits = self.non_neg().limb_bits;
        let mask = (1u64 << limb_bits) - 1;
        for (slack, val_f) in [(Slack::A, a_f), (Slack::B, b_f)] {
            let raw = val_f.to_canonical_u64();
            for j in 0..self.non_neg().limbs() - 1 {
                let digit = ((raw >> (j * limb_bits)) & mask) as i64;
                tensors.push((
                    SoftmaxColumn::Limb(slack, j as u8),
                    core::iter::once(digit).collect(),
                ));
            }
        }
        Ok(SoftmaxWitness::from_tensors(tensors))
    }
}

impl RsqrtRow {
    /// Build an alternative witness for adversarial testing (RFC-0009 Lemma 7).
    pub fn malicious_witness(
        &self,
        u: &[i64],
        real: &[bool],
        alt: RsqrtAlternative,
    ) -> Result<RsqrtWitness, GadgetError> {
        let honest = self.witness(u, real)?;
        let mut roots: SecretBuf<i64> = honest
            .tensor(RsqrtColumn::Root)
            .unwrap_or(&[])
            .iter()
            .copied()
            .collect();
        let clear_final_borrow = match alt {
            RsqrtAlternative::RootDelta {
                row,
                delta,
                clear_final_borrow,
            } => {
                roots[row] += delta;
                clear_final_borrow
            }
            RsqrtAlternative::PaddedRoot { row, root } => {
                roots[row] = root;
                false
            }
        };
        let mut tensors = vec![(RsqrtColumn::Root, roots)];
        for (input, bits, n) in [
            (true, 16, self.input_nonneg().limbs()),
            (false, 8, self.digits_nonneg().limbs()),
        ] {
            for j in 0..n - 1 {
                let mut limb_buf = SecretBuf::with_capacity(u.len());
                let data = if input { u } else { &tensors[0].1[..] };
                for &v in data {
                    limb_buf.push((v >> (bits * j)) & ((1 << bits) - 1));
                }
                let col = if input {
                    RsqrtColumn::InputLimb(j as u8)
                } else {
                    RsqrtColumn::RootLimb(j as u8)
                };
                tensors.push((col, limb_buf));
            }
        }
        let value = |row: usize, t: RsqrtTerm| match t {
            RsqrtTerm::Input => Goldilocks::from_i64(u[row]),
            RsqrtTerm::RowSelector => Goldilocks::from_u64_reduced(u64::from(real[row])),
            RsqrtTerm::Column(c) => Goldilocks::from_i64(
                tensors
                    .iter()
                    .find(|(k, _)| *k == c)
                    .map_or(0, |(_, v)| v[row]),
            ),
        };
        let mut comparisons = Vec::new();
        for (idx, b) in [Bracket::Lower, Bracket::Upper].into_iter().enumerate() {
            let cmp = &self.comparisons()[idx];
            let limbs = cmp.limbs();
            let polynomials: Vec<_> = (0..limbs)
                .map(|j| self.product_relation::<Goldilocks>(b, j))
                .collect();
            let mut product: Vec<_> = (0..limbs)
                .map(|_| SecretBuf::with_capacity(u.len()))
                .collect();
            let mut constant: Vec<_> = (0..limbs)
                .map(|_| SecretBuf::with_capacity(u.len()))
                .collect();
            for (row, &active) in real.iter().enumerate() {
                for j in 0..limbs {
                    product[j].push(
                        polynomials[j]
                            .evaluate(|t| value(row, t))
                            .to_canonical_u64() as i64,
                    );
                    constant[j].push(if active {
                        (((self.n() + u128::from(b == Bracket::Upper)) >> (8 * j)) & 255) as i64
                    } else {
                        0
                    });
                }
            }
            let p: Vec<_> = product.iter().map(|v| &v[..]).collect();
            let c: Vec<_> = constant.iter().map(|v| &v[..]).collect();
            let comp_alt = CompareAlternative::FalseComparison { clear_final_borrow };
            comparisons.push(if b == Bracket::Lower {
                cmp.malicious_witness(&p, &c, comp_alt)?
            } else {
                cmp.malicious_witness(&c, &p, comp_alt)?
            });
        }
        Ok(RsqrtWitness::from_parts(tensors, comparisons))
    }
}

impl<F: PrimeField + PolyField<Base = F>> OneHot<F> {
    /// Build an alternative witness for adversarial testing (RFC-0009 Lemma 8).
    pub fn malicious_witness(
        &self,
        alt: OneHotAlternative,
    ) -> Result<OneHotWitness<F>, GadgetError> {
        let len = self.padded_len();
        let mut indicator: SecretBuf<F> = (0..len).map(|_| F::ZERO).collect();
        match alt {
            OneHotAlternative::TwoOnes { first, second } => {
                if first >= len || second >= len || first == second {
                    return Err(GadgetError::Shape {
                        rule: "TwoOnes requires two distinct indices in 0..padded_len",
                    });
                }
                indicator[first] = F::ONE;
                indicator[second] = F::ONE;
            }
            OneHotAlternative::WrongIndex { index } => {
                if index >= self.vocab() {
                    return Err(GadgetError::Shape {
                        rule: "WrongIndex requires an index in 0..vocab",
                    });
                }
                indicator[index] = F::ONE;
            }
            OneHotAlternative::PaddedVocabularyOne { index } => {
                if index < self.vocab() || index >= len {
                    return Err(GadgetError::Shape {
                        rule: "PaddedVocabularyOne requires vocab <= index < padded_len",
                    });
                }
                indicator[index] = F::ONE;
            }
            OneHotAlternative::NonBoolean { index, value } => {
                if index >= len {
                    return Err(GadgetError::Shape {
                        rule: "NonBoolean requires an index in 0..padded_len",
                    });
                }
                indicator[index] = F::from_i64(value);
            }
        }
        Ok(OneHotWitness::from_column(indicator))
    }
}

/// Evaluated malicious [`Argmax`] bundle: next-token indicator, committed maximum and suffix/key
/// columns.
pub type MaliciousArgmax<F> = (OneHotWitness<F>, ArgmaxWitness<F>, ArgmaxEvaluation<F>);

impl<F: PrimeField + PolyField<Base = F>> Argmax<F> {
    /// Build an alternative `(OneHotWitness, ArgmaxWitness)` pair and its evaluation for
    /// adversarial testing (RFC-0009 Lemma 9).
    pub fn malicious_evaluate(
        &self,
        logits: &[F],
        real: bool,
        honest_next_token: i64,
        alt: ArgmaxAlternative,
    ) -> Result<MaliciousArgmax<F>, GadgetError> {
        let onehot = self.domain();
        let honest_argmax = self.witness(logits, real)?;
        let honest_max = honest_argmax.maximum()[0];
        let (next_witness, argmax_witness) = match alt {
            ArgmaxAlternative::LaterTie { later_index } => {
                let next = onehot.witness(later_index as i64, real)?;
                (next, honest_argmax)
            }
            ArgmaxAlternative::MaxDelta(delta) => {
                let next = onehot.witness(honest_next_token, real)?;
                let bad_max = honest_max + F::from_i64(delta);
                (
                    next,
                    ArgmaxWitness::from_column(core::iter::once(bad_max).collect()),
                )
            }
        };
        let eval = self.evaluate(logits, &next_witness, &argmax_witness)?;
        Ok((next_witness, argmax_witness, eval))
    }
}
