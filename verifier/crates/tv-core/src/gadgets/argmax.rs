// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemma 9's range-key, selected-logit and strict-suffix reductions.
//! These are final relations, not proof acceptance. The certificate binder must
//! derive their public metadata, discharge the Range29 lookup and bind every role
//! to its canonical maximum/logits or the SAME next-token OneHot (RFC-0017).
//! The graph lowering stops at logits; these roles do not invent graph auxiliaries.

use super::GadgetError;
use crate::selectors::{lt, suffix_kernel};
use crate::sumcheck::{Instance, eq};
use crate::validation::GadgetId;
use tv_field::Fp2;

/// A certificate input still owed to its canonical source or producing reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The certificate's committed maximum on output rows.
    Maximum,
    /// The model's logits at the positions predicting those outputs.
    Logits,
    /// The recorded NEXT token's OneHot, shared by selected-logit and suffix checks.
    Indicator,
    /// The virtual strict suffix of that same next-token indicator.
    Suffix,
}

/// A typed source obligation; its point is vocabulary first, then output rows.
/// Maximum has row coordinates only. A general extension is not an integer.
#[derive(Clone, Copy)]
pub struct Claim<'a> {
    /// Which canonical certificate input the binder must prove.
    pub role: Role,
    /// Its verifier-derived point, LSB-first within each axis.
    pub point: &'a [Fp2],
    /// Its claimed multilinear extension.
    pub value: Fp2,
}

/// The Range29 lookup left by a key reduction, still owed to earlier LogUp.
#[derive(Clone, Copy)]
pub struct Range<'a> {
    /// The key's full vocabulary/row point.
    pub point: &'a [Fp2],
    /// The key extension at that point.
    pub key: Fp2,
    /// The Range value, checked to be zero locally.
    pub value: Fp2,
}
impl Range<'_> {
    /// The fixed RFC-0009 Range width, never proof-selected.
    pub const fn bits(&self) -> u8 {
        29
    }
}

/// Public final-relation geometry the certificate binder derives from validated inputs.
#[derive(Clone, Copy, Debug)]
pub struct Argmax {
    gadget: GadgetId,
    vocab: u32,
    rows: u32,
    vocabulary_vars: usize,
    row_vars: usize,
    variables: usize,
}

/// One key reduction, bound to its incoming Range29 claim before batching.
#[must_use = "check its Range29 lookup, reduce the key and discharge every returned role"]
pub struct Key<'a> {
    site: Argmax,
    target: &'a [Fp2],
    key: Fp2,
}

/// A selected-logit reduction bound to an already fixed incoming maximum.
#[must_use = "bind the maximum and both final roles, including the same next-token OneHot"]
pub struct Selected<'a> {
    site: Argmax,
    row_point: &'a [Fp2],
    maximum: Fp2,
}

/// A virtual suffix claim to reduce after the key relation leaves it.
#[must_use = "reduce this suffix to the SAME next-token OneHot as the selected-logit check"]
pub struct Suffix<'a> {
    site: Argmax,
    target: &'a [Fp2],
    claim: Fp2,
}

/// A final relation value and all of its still-unproved canonical input roles.
#[must_use = "check the batched relation and discharge every returned canonical role"]
pub struct Final<'a> {
    value: Fp2,
    claims: [Option<Claim<'a>>; 3],
}

impl Argmax {
    /// Derive a certificate's geometry from verifier-owned metadata. `rows` is the
    /// output-row domain, not a proof-supplied sequence length. The binder owns the
    /// gadget id, canonical Range29 table and mappings to logits and next-token rows.
    /// It must also establish OneHot and the true-integer logit bound `|l| < 2^28`.
    ///
    /// # Errors
    /// `Internal` unless `1 <= vocab <= 2^18` and `1 <= rows <= 2^17`.
    pub fn derive(gadget: GadgetId, vocab: u32, rows: u32) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab) || !(1..=1 << 17).contains(&rows) {
            return Err(GadgetError::Internal);
        }
        let bits = |n: u32| {
            usize::try_from(u32::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
                .map_err(|_| GadgetError::Internal)
        };
        let vocabulary_vars = bits(vocab)?;
        let row_vars = bits(rows)?;
        Ok(Self {
            gadget,
            vocab,
            rows,
            vocabulary_vars,
            row_vars,
            variables: vocabulary_vars
                .checked_add(row_vars)
                .ok_or(GadgetError::Internal)?,
        })
    }
    /// The verifier-owned gadget id used in local rejection payloads.
    pub const fn gadget(self) -> GadgetId {
        self.gadget
    }
    /// The real vocabulary length, excluding padding.
    pub const fn vocab(self) -> u32 {
        self.vocab
    }
    /// The number of actual output rows, excluding padding.
    pub const fn rows(self) -> u32 {
        self.rows
    }
    /// Full point length: vocabulary coordinates followed by row coordinates.
    pub const fn variables(self) -> usize {
        self.variables
    }
    /// The row-only point length used by maximum claims.
    pub const fn row_variables(self) -> usize {
        self.row_vars
    }
    fn point(self, point: &[Fp2]) -> Result<(&[Fp2], &[Fp2]), GadgetError> {
        if point.len() != self.variables {
            return Err(GadgetError::Internal);
        }
        point
            .split_at_checked(self.vocabulary_vars)
            .ok_or(GadgetError::Internal)
    }
    fn proof_values(self, values: &[Fp2], count: usize) -> Result<(), GadgetError> {
        if values.len() != count {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        Ok(())
    }
    /// Bind an incoming key/value claim from the prescribed Range29 lookup.
    /// Membership is checked by LogUp; this rejects a nonzero Range value.
    /// Wrong own point geometry is `Internal` before any proof error; a nonzero
    /// value is `Rejected` with this gadget's id.
    pub fn lookup(self, point: &[Fp2], key: Fp2, value: Fp2) -> Result<Key<'_>, GadgetError> {
        self.point(point)?;
        if value != Fp2::ZERO {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        Ok(Key {
            site: self,
            target: point,
            key,
        })
    }
    /// Bind the selected-logit sum to an incoming maximum fixed before gamma.
    /// Returns `Internal` for wrong verifier-owned row geometry.
    pub fn selected(self, row_point: &[Fp2], maximum: Fp2) -> Result<Selected<'_>, GadgetError> {
        if row_point.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        Ok(Selected {
            site: self,
            row_point,
            maximum,
        })
    }
    /// Bind a suffix claim, normally left at the key reduction's final point.
    /// Returns `Internal` for wrong verifier-owned full-point geometry.
    pub fn suffix(self, point: &[Fp2], claim: Fp2) -> Result<Suffix<'_>, GadgetError> {
        self.point(point)?;
        Ok(Suffix {
            site: self,
            target: point,
            claim,
        })
    }
}
impl Key<'_> {
    /// Degree-three reduction of the masked range key.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.key,
            vars: self.site.variables,
            evals: 3,
        }
    }
    /// The exact incoming lookup, still owed to the canonical Range29 instance.
    pub const fn lookup(&self) -> Range<'_> {
        Range {
            point: self.target,
            key: self.key,
            value: Fp2::ZERO,
        }
    }
    /// Evaluate `eq(target,r)*LT(r_v,vocab)*(m-l-suffix)`.
    /// Final evaluations are maximum, logit, suffix, in that order.
    /// Own geometry errors are `Internal`; a wrong count is `Rejected` with its id.
    pub fn evaluate<'a>(&self, point: &'a [Fp2], values: &[Fp2]) -> Result<Final<'a>, GadgetError> {
        let (v, row) = self.site.point(point)?;
        self.site.proof_values(values, 3)?;
        let m = values.first().copied().ok_or(GadgetError::Internal)?;
        let ell = values.get(1).copied().ok_or(GadgetError::Internal)?;
        let suffix = values.get(2).copied().ok_or(GadgetError::Internal)?;
        Ok(Final {
            value: eq(self.target, point).ok_or(GadgetError::Internal)?
                * lt(v, u64::from(self.site.vocab)).ok_or(GadgetError::Internal)?
                * (m - ell - suffix),
            claims: [
                Some(Claim {
                    role: Role::Maximum,
                    point: row,
                    value: m,
                }),
                Some(Claim {
                    role: Role::Logits,
                    point,
                    value: ell,
                }),
                Some(Claim {
                    role: Role::Suffix,
                    point,
                    value: suffix,
                }),
            ],
        })
    }
}
impl Selected<'_> {
    /// Degree-three selected-logit reduction over the full padded vocabulary/rows.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.maximum,
            vars: self.site.variables,
            evals: 2,
        }
    }
    /// The already fixed maximum's canonical commitment obligation.
    pub const fn input_claim(&self) -> Claim<'_> {
        Claim {
            role: Role::Maximum,
            point: self.row_point,
            value: self.maximum,
        }
    }
    /// Evaluate `eq(incoming_row,r_row)*h_next(r)*logit(r)`.
    /// The two operands are that same next-token indicator, then the logit.
    /// Own geometry errors are `Internal`; a wrong count is `Rejected` with its id.
    pub fn evaluate<'a>(&self, point: &'a [Fp2], values: &[Fp2]) -> Result<Final<'a>, GadgetError> {
        let (_, row) = self.site.point(point)?;
        self.site.proof_values(values, 2)?;
        let h = values.first().copied().ok_or(GadgetError::Internal)?;
        let ell = values.get(1).copied().ok_or(GadgetError::Internal)?;
        Ok(Final {
            value: eq(self.row_point, row).ok_or(GadgetError::Internal)? * h * ell,
            claims: [
                Some(Claim {
                    role: Role::Indicator,
                    point,
                    value: h,
                }),
                Some(Claim {
                    role: Role::Logits,
                    point,
                    value: ell,
                }),
                None,
            ],
        })
    }
}
impl Suffix<'_> {
    /// Degree-two reduction to the strict suffix of the same next-token OneHot.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.claim,
            vars: self.site.variables,
            evals: 1,
        }
    }
    /// Evaluate the independent public suffix kernel times the next-token indicator.
    /// Own geometry errors are `Internal`; a wrong count is `Rejected` with its id.
    pub fn evaluate<'a>(&self, point: &'a [Fp2], values: &[Fp2]) -> Result<Final<'a>, GadgetError> {
        let (v, row) = self.site.point(point)?;
        let (target_v, target_row) = self.site.point(self.target)?;
        self.site.proof_values(values, 1)?;
        let h = values.first().copied().ok_or(GadgetError::Internal)?;
        Ok(Final {
            value: eq(target_row, row).ok_or(GadgetError::Internal)?
                * suffix_kernel(target_v, v).ok_or(GadgetError::Internal)?
                * h,
            claims: [
                Some(Claim {
                    role: Role::Indicator,
                    point,
                    value: h,
                }),
                None,
                None,
            ],
        })
    }
}
impl Final<'_> {
    /// The value to compare with the batch's final claim, including public factors.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// Every source role still owed, in operand order, with its prescribed point.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'_>> {
        self.claims.iter().copied().flatten()
    }
}
