// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0017 `VocabMax`, `OneSidedLog2`, and deterministic runtime scoring table
//! interning (RFC-0017, RFC-0027, RFC-0030, RFC-0035, RFC-0050, RFC-0053, RFC-0058).
//!
//! Written from the specifications without reference to prover code (SP-6, RFC-0001).
//! Every check transitions through `#[must_use]` obligation tokens so no lookup,
//! forest-leaf, advice, exponential, or token-indicator evaluation claim can be
//! dropped silently.

use core::fmt;

use tv_codec::Rational;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::Transcript;
use tv_tables::{TableId, TableKind, exp_maps_to_zero};

use super::GadgetError;
use super::comparison::{self, Comparison};
use super::decomposition::{self, Decomposition};
use super::inverse_cdf::{self, StageVerifyError};
use crate::advice::ComparisonRole;
use crate::fiat_shamir::{self, Step};
use crate::protocol::{Decoding, FixedBits, PredictorSpec};
use crate::scoring::{
    MAX_BATCH_TEMPERATURES, MAX_EPS_LOG2, ScoringError, c_match, validate_decoding, vocab_exp_scale,
};
use crate::selectors::lt;
use crate::sumcheck::{self, Batch, Instance, Prefix, SumcheckProof, eq};
use crate::validation::GadgetId;

// ============================================================================
// VocabMax (RFC-0017 'D's decoded token', RFC-0030, RFC-0035)
// ============================================================================

/// Canonical source role owed after verifying [`VocabMax`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VocabMaxRole {
    /// Committed row-wise vocabulary maximum `c_t` on the output-row domain.
    Maximum,
    /// Lowered LM-head logits `ℓ_{t,v}` on the `(vocab, row)` domain.
    Logits,
}

/// One retained source evaluation obligation of [`VocabMax`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabMaxSourceClaim {
    /// Canonical input or certificate role.
    pub role: VocabMaxRole,
    /// Verifier-derived evaluation point, LSB-first.
    pub point: Vec<Fp2>,
    /// Claimed multilinear extension value.
    pub value: Fp2,
}

/// One retained `Exp_T` lookup obligation of [`VocabMax`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabMaxLookupClaim {
    /// Verifier-selected `TableKind::Exp` specification.
    pub kind: TableKind,
    /// `(vocab, row)` lookup point, LSB-first.
    pub point: Vec<Fp2>,
    /// Masked key evaluation `k̃(lookup_target)`.
    pub key: Fp2,
    /// Exponential value evaluation `ẽ(lookup_target)`.
    pub value: Fp2,
}

/// Verifier-derived `VocabMax` site and validated `Exp_T` table parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabMax {
    gadget: GadgetId,
    vocab: u32,
    rows: u32,
    vocab_vars: usize,
    row_vars: usize,
    vars: usize,
    sigma: u8,
    scale: Rational,
    min_key_bits: u8,
    key_bits: u8,
    k0: u64,
    exp_kind: TableKind,
}

/// Incoming leaf/lookup claims of [`VocabMax`] checked against the forest root `1 − ψ`.
#[must_use = "run VocabMax Stage 1 and discharge its lookup and source obligations"]
#[derive(Debug)]
pub struct VocabMaxClaims<'g> {
    site: &'g VocabMax,
    lookup_target: Vec<Fp2>,
    lookup_key: Fp2,
    lookup_value: Fp2,
    tree_target: Vec<Fp2>,
    claims: [Fp2; 3],
}

/// Stage-1 (`stage/1`) batched sumcheck of [`VocabMax`]: masked lookup key, masked
/// product-forest leaf, and padded-row maximum zero-check.
#[must_use = "verify VocabMax Stage 1 and discharge all returned obligations"]
#[derive(Debug)]
pub struct VocabMaxStage1<'g> {
    checked: VocabMaxClaims<'g>,
    zero_row: Vec<Fp2>,
    instances: [Instance; 3],
}

/// Completed [`VocabMax`] verification with retained `Exp_T` lookup and `(c_t, ℓ_{t,v})` claims.
#[must_use = "discharge the Exp_T lookup obligation and both Maximum and Logits source claims"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabMaxVerified {
    lookup: VocabMaxLookupClaim,
    sources: [VocabMaxSourceClaim; 2],
}

impl VocabMaxVerified {
    /// Retained `Exp_T` lookup key/value obligation at `lookup_target`.
    pub const fn lookup_claim(&self) -> &VocabMaxLookupClaim {
        &self.lookup
    }

    /// Retained `[Maximum(r_{1,row}), Logits(r_1)]` source obligations.
    pub const fn source_claims(&self) -> &[VocabMaxSourceClaim; 2] {
        &self.sources
    }
}

/// Exclusive magnitude bound `2^28` on lowered LM-head logit endpoints `|ℓ| < 2^28`
/// (RFC-0009 Lemma 5, RFC-0053 'Public range and catalog').
const LOGIT_BOUND: i32 = 1 << 28;

/// Select the narrowest `key_bits ∈ [min_key_bits, max_log2_rows]` whose `Exp_T` table
/// has valid RFC-0008 parameters and maps its maximum key `2^{key_bits} − 1` to `0`
/// (RFC-0017 '`D`'s decoded token', RFC-0030, RFC-0053).
///
/// Returns `(TableKind::Exp { .. }, min_key_bits, key_bits, k0)`.
///
/// # Errors
/// [`GadgetError::Internal`] when public bounds (`-(1 << 28) < logit_lo ≤ logit_hi < 1 << 28`,
/// `sigma ≤ 32`, `max_log2_rows ∈ 1..=32`) or temperature scale are invalid, or when no
/// width up to `max_log2_rows` reaches the zero tail.
pub fn select_vocab_exp_table(
    logits_exponent: i8,
    temperature: Rational,
    sigma: u8,
    logit_lo: i32,
    logit_hi: i32,
    max_log2_rows: u8,
) -> Result<(TableKind, u8, u8, u64), GadgetError> {
    if sigma > 32 || !(1..=32).contains(&max_log2_rows) {
        return Err(GadgetError::Internal);
    }
    let scale = vocab_exp_scale(logits_exponent, temperature).map_err(|_| GadgetError::Internal)?;
    let min_kb = logit_min_key_bits(logit_lo, logit_hi)?;
    let (kind, kb, k0) = find_qualifying_exp_table(scale, sigma, min_kb, max_log2_rows)?
        .ok_or(GadgetError::Internal)?;
    Ok((kind, min_kb, kb, k0))
}

impl VocabMax {
    /// Derive [`VocabMax`] from public model/policy metadata, automatically selecting the
    /// narrowest qualifying `Exp_T` width in `[min_key_bits, max_log2_rows]`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] if public dimensions, temperature, or `Exp_T` zero-tail
    /// requirements are violated.
    #[allow(
        clippy::too_many_arguments,
        reason = "all eight inputs are distinct public statement/claim-list scalars"
    )]
    pub fn derive(
        gadget: GadgetId,
        vocab: u32,
        rows: u32,
        logits_exponent: i8,
        temperature: Rational,
        sigma: u8,
        logit_lo: i32,
        logit_hi: i32,
        max_log2_rows: u8,
    ) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab) || !(1..=1 << 17).contains(&rows) {
            return Err(GadgetError::Internal);
        }
        let (exp_kind, min_key_bits, key_bits, k0) = select_vocab_exp_table(
            logits_exponent,
            temperature,
            sigma,
            logit_lo,
            logit_hi,
            max_log2_rows,
        )?;
        let TableKind::Exp { scale, .. } = exp_kind else {
            return Err(GadgetError::Internal);
        };
        let vocab_vars = log2_ceil(vocab)?;
        let row_vars = log2_ceil(rows)?;
        let vars = vocab_vars
            .checked_add(row_vars)
            .ok_or(GadgetError::Internal)?;
        Ok(Self {
            gadget,
            vocab,
            rows,
            vocab_vars,
            row_vars,
            vars,
            sigma,
            scale,
            min_key_bits,
            key_bits,
            k0,
            exp_kind,
        })
    }

    /// Derive [`VocabMax`] checking an explicit `key_bits` candidate against both the
    /// public logit span (`hi − lo < 2^{key_bits}`) and `Exp_T[2^{key_bits} − 1] == 0`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `key_bits < min_key_bits`, `key_bits > max_log2_rows`,
    /// `TableKind::Exp` rejects its parameters, or `Exp_T[K_0] != 0`.
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit key_bits validation requires all public Exp_T domain parameters"
    )]
    pub fn derive_with_key_bits(
        gadget: GadgetId,
        vocab: u32,
        rows: u32,
        logits_exponent: i8,
        temperature: Rational,
        sigma: u8,
        logit_lo: i32,
        logit_hi: i32,
        max_log2_rows: u8,
        key_bits: u8,
    ) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab)
            || !(1..=1 << 17).contains(&rows)
            || sigma > 32
            || logit_lo > logit_hi
            || !(1..=32).contains(&max_log2_rows)
        {
            return Err(GadgetError::Internal);
        }
        let scale =
            vocab_exp_scale(logits_exponent, temperature).map_err(|_| GadgetError::Internal)?;
        let min_key_bits = logit_min_key_bits(logit_lo, logit_hi)?;
        if key_bits < min_key_bits || key_bits > max_log2_rows {
            return Err(GadgetError::Internal);
        }
        let exp_kind = TableKind::Exp {
            key_bits,
            sigma,
            scale,
        };
        if exp_kind.check_parameters().is_err() {
            return Err(GadgetError::Internal);
        }
        let k0 = (1u64
            .checked_shl(u32::from(key_bits))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .ok_or(GadgetError::Internal)?;
        if exp_maps_to_zero(k0, sigma, scale) != Ok(true) {
            return Err(GadgetError::Internal);
        }
        let vocab_vars = log2_ceil(vocab)?;
        let row_vars = log2_ceil(rows)?;
        let vars = vocab_vars
            .checked_add(row_vars)
            .ok_or(GadgetError::Internal)?;
        Ok(Self {
            gadget,
            vocab,
            rows,
            vocab_vars,
            row_vars,
            vars,
            sigma,
            scale,
            min_key_bits,
            key_bits,
            k0,
            exp_kind,
        })
    }

    /// Verifier-owned gadget identifier.
    pub const fn gadget(&self) -> GadgetId {
        self.gadget
    }

    /// Reduced canonical scale `2^{logits_exponent} / T`.
    pub const fn scale(&self) -> Rational {
        self.scale
    }

    /// Minimum Range/Exp key width `max(1, bit_length(logit_hi − logit_lo))`.
    pub const fn min_key_bits(&self) -> u8 {
        self.min_key_bits
    }

    /// Selected `Exp_T` key width.
    pub const fn key_bits(&self) -> u8 {
        self.key_bits
    }

    /// Zero-tail padding key `K_0 = 2^{key_bits} − 1` with `Exp_T[K_0] = 0`.
    pub const fn k0(&self) -> u64 {
        self.k0
    }

    /// Selected `TableKind::Exp` specification.
    pub fn exp_kind(&self) -> &TableKind {
        &self.exp_kind
    }

    /// Vocabulary variable count `⌈log2 vocab⌉`.
    pub const fn vocab_variables(&self) -> usize {
        self.vocab_vars
    }

    /// Row variable count `⌈log2 rows⌉`.
    pub const fn row_variables(&self) -> usize {
        self.row_vars
    }

    /// Combined `(vocab, row)` variable count.
    pub const fn variables(&self) -> usize {
        self.vars
    }

    /// Verify the product-forest root claim `1 − ψ(forest_root_target)` and prepare the
    /// three Stage-1 claims (`lookup_key − K_0 · (1 − C)`, `tree_leaf − (1 − C)`, `0`).
    ///
    /// # Errors
    /// [`GadgetError::Internal`] on mismatched target point lengths;
    /// [`GadgetError::Rejected`] when `forest_root_claim != 1 − ψ(forest_root_target)`;
    /// [`GadgetError::Memory`] on failed reservation.
    #[allow(
        clippy::too_many_arguments,
        reason = "lookup, product-tree and forest-root targets are distinct leaf-stage outputs"
    )]
    pub fn check_claims<'g>(
        &'g self,
        lookup_target: &[Fp2],
        lookup_key: Fp2,
        lookup_value: Fp2,
        tree_target: &[Fp2],
        tree_leaf: Fp2,
        forest_root_target: &[Fp2],
        forest_root_claim: Fp2,
    ) -> Result<VocabMaxClaims<'g>, GadgetError> {
        if lookup_target.len() != self.vars
            || tree_target.len() != self.vars
            || forest_root_target.len() != self.row_vars
        {
            return Err(GadgetError::Internal);
        }
        let psi_root = lt(forest_root_target, u64::from(self.rows)).ok_or(GadgetError::Internal)?;
        if forest_root_claim != Fp2::ONE - psi_root {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        let c_lookup = self.selector(lookup_target)?;
        let c_tree = self.selector(tree_target)?;
        let k0_fp = Fp2::from(Fp::reduce(self.k0));
        let claims = [
            lookup_key - k0_fp * (Fp2::ONE - c_lookup),
            tree_leaf - (Fp2::ONE - c_tree),
            Fp2::ZERO,
        ];
        Ok(VocabMaxClaims {
            site: self,
            lookup_target: clone_vec(lookup_target)?,
            lookup_key,
            lookup_value,
            tree_target: clone_vec(tree_target)?,
            claims,
        })
    }

    fn selector(&self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        let (v_pt, row_pt) = point
            .split_at_checked(self.vocab_vars)
            .ok_or(GadgetError::Internal)?;
        let cv = lt(v_pt, u64::from(self.vocab)).ok_or(GadgetError::Internal)?;
        let rt = lt(row_pt, u64::from(self.rows)).ok_or(GadgetError::Internal)?;
        Ok(cv * rt)
    }
}

impl<'g> VocabMaxClaims<'g> {
    /// Derived Stage-1 claims `[masked_lookup_key, masked_tree_leaf, 0]`.
    pub const fn claims(&self) -> &[Fp2; 3] {
        &self.claims
    }

    /// Squeeze `stage/1/zero/{0..row_vars}` from `transcript` and build Stage 1.
    ///
    /// # Errors
    /// [`StageVerifyError`] on transcript or reservation failure.
    pub fn squeeze_stage1(
        self,
        transcript: &mut Transcript,
    ) -> Result<VocabMaxStage1<'g>, StageVerifyError> {
        let zero_row = squeeze_zero(transcript, Prefix::Stage(1), self.site.row_vars)?;
        Ok(self.prepare_stage1(zero_row)?)
    }

    /// Build Stage 1 from an already squeezed stage-1 row zero-check point.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `zero_row.len() != row_vars`.
    pub fn prepare_stage1(self, zero_row: Vec<Fp2>) -> Result<VocabMaxStage1<'g>, GadgetError> {
        if zero_row.len() != self.site.row_vars {
            return Err(GadgetError::Internal);
        }
        let [c0, c1, c2] = self.claims;
        let instances = [
            Instance {
                claim: c0,
                vars: self.site.vars,
                evals: 2,
            },
            Instance {
                claim: c1,
                vars: self.site.vars,
                evals: 2,
            },
            Instance {
                claim: c2,
                vars: self.site.row_vars,
                evals: 1,
            },
        ];
        Ok(VocabMaxStage1 {
            checked: self,
            zero_row,
            instances,
        })
    }
}

impl VocabMaxStage1<'_> {
    /// Prescribed sumcheck degree (`3`) of `VocabMax` Stage 1.
    pub const fn degree(&self) -> usize {
        3
    }

    /// Stage-1 instances in schedule order (two `(vocab, row)` instances and one front-padded row instance).
    pub const fn instances(&self) -> &[Instance; 3] {
        &self.instances
    }

    /// Evaluate Stage-1 instance `index` (`0..3`) at its unpadded instance point `point`.
    ///
    /// Note: `sumcheck::verify_batch` strips the `vocab_vars` front-padding challenges
    /// before invoking `evaluate` on instance `2` (`vars = row_vars`).
    ///
    /// # Errors
    /// [`GadgetError::Internal`] for an invalid `index` or `point` length;
    /// [`GadgetError::Rejected`] for a wrong `values` count.
    pub fn evaluate(
        &self,
        index: usize,
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<Fp2, GadgetError> {
        let site = self.checked.site;
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        match index {
            0 | 1 => {
                if point.len() != site.vars {
                    return Err(GadgetError::Internal);
                }
                if values.len() != 2 {
                    return Err(reject);
                }
                let c_at = *values.first().ok_or(GadgetError::Internal)?;
                let ell_at = *values.get(1).ok_or(GadgetError::Internal)?;
                let sel = site.selector(point)?;
                let target = if index == 0 {
                    &self.checked.lookup_target
                } else {
                    &self.checked.tree_target
                };
                let eq_t = eq(target, point).ok_or(GadgetError::Internal)?;
                Ok(eq_t * sel * (c_at - ell_at))
            }
            2 => {
                if point.len() != site.row_vars {
                    return Err(GadgetError::Internal);
                }
                if values.len() != 1 {
                    return Err(reject);
                }
                let c_at = *values.first().ok_or(GadgetError::Internal)?;
                let psi = lt(point, u64::from(site.rows)).ok_or(GadgetError::Internal)?;
                let eq_zr = eq(&self.zero_row, point).ok_or(GadgetError::Internal)?;
                Ok(eq_zr * (Fp2::ONE - psi) * c_at)
            }
            _ => Err(GadgetError::Internal),
        }
    }

    /// Run `sumcheck::verify_batch` under `stage/1` and enforce cross-instance evaluation equality.
    ///
    /// # Errors
    /// [`StageVerifyError`] on any sumcheck or cross-instance consistency failure.
    pub fn verify(
        self,
        transcript: &mut Transcript,
        proof: &SumcheckProof,
    ) -> Result<(Batch, VocabMaxVerified), StageVerifyError> {
        let batch = sumcheck::verify_batch(
            transcript,
            Prefix::Stage(1),
            self.degree(),
            &self.instances,
            proof,
            |i, pt, vals| self.evaluate(i, pt, vals).ok(),
        )?;
        let verified = self.finish(clone_vec(&batch.point)?, &proof.final_evals)?;
        Ok((batch, verified))
    }

    /// Enforce equality of `[c, ℓ, c, ℓ, c]` at `point` and return all retained obligations.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != vars`;
    /// [`GadgetError::Rejected`] when `final_evals` has length `!= 5` or disagrees across instances.
    pub fn finish(
        self,
        point: Vec<Fp2>,
        final_evals: &[Fp2],
    ) -> Result<VocabMaxVerified, GadgetError> {
        let site = self.checked.site;
        if point.len() != site.vars {
            return Err(GadgetError::Internal);
        }
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        if final_evals.len() != 5 {
            return Err(reject);
        }
        let c0 = *final_evals.first().ok_or(GadgetError::Internal)?;
        let l0 = *final_evals.get(1).ok_or(GadgetError::Internal)?;
        let c1 = *final_evals.get(2).ok_or(GadgetError::Internal)?;
        let l1 = *final_evals.get(3).ok_or(GadgetError::Internal)?;
        let c2 = *final_evals.get(4).ok_or(GadgetError::Internal)?;
        if c0 != c1 || c0 != c2 || l0 != l1 {
            return Err(reject);
        }
        let row_pt = point.get(site.vocab_vars..).ok_or(GadgetError::Internal)?;
        let row_point = clone_vec(row_pt)?;
        Ok(VocabMaxVerified {
            lookup: VocabMaxLookupClaim {
                kind: site.exp_kind.clone(),
                point: self.checked.lookup_target,
                key: self.checked.lookup_key,
                value: self.checked.lookup_value,
            },
            sources: [
                VocabMaxSourceClaim {
                    role: VocabMaxRole::Maximum,
                    point: row_point,
                    value: c0,
                },
                VocabMaxSourceClaim {
                    role: VocabMaxRole::Logits,
                    point,
                    value: l0,
                },
            ],
        })
    }
}

// ============================================================================
// OneSidedLog2 (RFC-0017 'In-circuit scoring (mixture)', RFC-0035, RFC-0050)
// ============================================================================

/// Canonical advice or input role owed by [`OneSidedLog2`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Log2Role {
    /// Committed match selector `m_t` on the output-row domain.
    MatchBit,
    /// Committed exponent key `I_t` (`Pow2(63)`) on the output-row domain.
    ExponentKey,
    /// Committed `P_t = 2^{I_t}` value on the output-row domain.
    Pow2Value,
    /// Committed fractional key `f_t` (`Exp2Frac(8, 16)`) on the output-row domain.
    FractionKey,
    /// Committed `E_t = E2[f_t]` value on the output-row domain.
    Exp2FracValue,
    /// Committed scaled numerator `q_t` on the output-row domain.
    ScaledNumerator,
    /// Committed low 16-bit limb `Z_{t,h}` (`0 ≤ h < k_Z − 1`) on the output-row domain.
    PartitionLow(usize),
    /// Committed low 16-bit limb `Q_{t,a}` (`0 ≤ a < k_Q − 1`) on the output-row domain.
    NumeratorLow(usize),
    /// Committed low 16-bit limb `P_{t,b}` (`0 ≤ b < 3`) on the output-row domain.
    Pow2Low(usize),
    /// `BigIntCompare` witness limb (`0..m`) on the output-row domain.
    Compare(ComparisonRole, usize),
    /// Vocabulary exponential table output `e_{t,v}` on the `(vocab, row)` domain.
    Exponentials,
    /// Recorded output token indicator `h_{y_t,v}` (`OneHot`) on the `(vocab, row)` domain.
    RecordedIndicator,
    /// Decoded token indicator `h_{ŷ_t,v}` (`SampledIndicator` / `Argmax`) on the `(vocab, row)` domain.
    DecodedIndicator,
}

/// One retained source or advice evaluation claim of [`OneSidedLog2`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Log2SourceClaim {
    /// Canonical advice or input role.
    pub role: Log2Role,
    /// Verifier-derived evaluation point, LSB-first.
    pub point: Vec<Fp2>,
    /// Claimed multilinear extension value.
    pub value: Fp2,
}

/// One retained LogUp lookup obligation (`Pow2(63)`, `Exp2Frac(8, 16)`, or `Range`) of [`OneSidedLog2`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Log2LookupClaim {
    /// Verifier-derived table kind.
    pub kind: TableKind,
    /// Row evaluation point, LSB-first.
    pub point: Vec<Fp2>,
    /// Claimed key multilinear evaluation.
    pub key: Fp2,
    /// Claimed value multilinear evaluation (`0` for `Range`).
    pub value: Fp2,
}

/// Public schema parameters of [`OneSidedLog2`] independent of `rows` (RFC-0017 items 4–5).
#[derive(Debug)]
pub struct Log2Params {
    vocab: u32,
    sigma: u8,
    eps_log2: u8,
    c_match_fixed: u64,
    w_z: u8,
    k_z: usize,
    top_z_bits: u8,
    w_q: u8,
    k_q: usize,
    top_q_bits: u8,
    m: usize,
    y_active: usize,
    v_limbs: [u64; 2],
    bounds: [Vec<u64>; 2],
    z_decomp: Decomposition,
    q_decomp: Decomposition,
    cmp: Comparison,
    lookup_specs: Vec<TableKind>,
    inv_top_z: Fp2,
    inv_top_q: Fp2,
    inv_2_48: Fp2,
}

impl Log2Params {
    /// Derive public widths, limb boxes, carry widths and lookup specs for `(vocab, sigma, eps_log2)`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] unless `1 ≤ vocab ≤ 2^18`, `sigma ≤ 32`, and `eps_log2 ≤ 16`;
    /// [`GadgetError::Memory`] on failed reservation.
    pub fn derive(vocab: u32, sigma: u8, eps_log2: u8) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab) || sigma > 32 || eps_log2 > MAX_EPS_LOG2 {
            return Err(GadgetError::Internal);
        }
        let c_match_fixed = c_match(eps_log2).map_err(|_| GadgetError::Internal)?;

        let max_z = u128::from(vocab)
            .checked_shl(u32::from(sigma))
            .ok_or(GadgetError::Internal)?;
        let w_z = bit_len_u128(max_z)?;
        let k_z = usize::from(w_z).div_ceil(16);
        let z_low_bits = k_z
            .checked_sub(1)
            .and_then(|n| n.checked_mul(16))
            .ok_or(GadgetError::Internal)?;
        let top_z_bits = w_z
            .checked_sub(u8::try_from(z_low_bits).map_err(|_| GadgetError::Internal)?)
            .ok_or(GadgetError::Internal)?;

        let max_e_plus_1 = (1u128
            .checked_shl(u32::from(sigma))
            .ok_or(GadgetError::Internal)?)
        .checked_add(1)
        .ok_or(GadgetError::Internal)?;
        let max_q = max_e_plus_1
            .checked_mul((1u128 << 17) - 1)
            .ok_or(GadgetError::Internal)?;
        let w_q = bit_len_u128(max_q)?;
        let k_q = usize::from(w_q).div_ceil(16);
        let q_low_bits = k_q
            .checked_sub(1)
            .and_then(|n| n.checked_mul(16))
            .ok_or(GadgetError::Internal)?;
        let top_q_bits = w_q
            .checked_sub(u8::try_from(q_low_bits).map_err(|_| GadgetError::Internal)?)
            .ok_or(GadgetError::Internal)?;

        let lhs_max = ((1u128
            .checked_shl(u32::from(w_z))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .and_then(|x| x.checked_add(u128::from(vocab)))
        .and_then(|x| x.checked_shl(16)))
        .ok_or(GadgetError::Internal)?;
        let rhs_max = ((1u128
            .checked_shl(u32::from(w_q))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .and_then(|x| x.checked_mul((1u128 << 63) - 1)))
        .ok_or(GadgetError::Internal)?;
        let m_bits = bit_len_u128(lhs_max)?.max(bit_len_u128(rhs_max)?);
        let m = usize::from(m_bits).div_ceil(16);
        let y_active = m.min(k_q.checked_add(3).ok_or(GadgetError::Internal)?);

        let z_decomp = Decomposition::derive(w_z, 16)?;
        let q_decomp = Decomposition::derive(w_q, 16)?;

        let top_z_box = (1u64
            .checked_shl(u32::from(top_z_bits))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .ok_or(GadgetError::Internal)?;
        let mut box_z = reserve_vec(k_z)?;
        box_z.extend(std::iter::repeat_n(65_535, k_z.saturating_sub(1)));
        box_z.push(top_z_box);

        let top_q_box = (1u64
            .checked_shl(u32::from(top_q_bits))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .ok_or(GadgetError::Internal)?;
        let mut box_q = reserve_vec(k_q)?;
        box_q.extend(std::iter::repeat_n(65_535, k_q.saturating_sub(1)));
        box_q.push(top_q_box);

        let box_p = [65_535u64, 65_535u64, 65_535u64, 32_767u64];
        let v_limbs = [u64::from(vocab & 0xFFFF), u64::from(vocab >> 16)];

        let mut a_left = reserve_vec(m)?;
        a_left.push(0);
        for j in 1..m {
            let idx = j.saturating_sub(1);
            let bz = box_z.get(idx).copied().unwrap_or(0);
            let bv = v_limbs.get(idx).copied().unwrap_or(0);
            a_left.push(bz.checked_add(bv).ok_or(GadgetError::Internal)?);
        }

        let mut a_right = reserve_vec(m)?;
        for j in 0..m {
            let mut sum = 0u64;
            for (a, &bq) in box_q.iter().enumerate() {
                for (b, &bp) in box_p.iter().enumerate() {
                    if a.checked_add(b) == Some(j) {
                        let prod = bq.checked_mul(bp).ok_or(GadgetError::Internal)?;
                        sum = sum.checked_add(prod).ok_or(GadgetError::Internal)?;
                    }
                }
            }
            a_right.push(sum);
        }

        let cmp = Comparison::derive(16, &a_left, &a_right)?;

        let total_lookups = 2usize
            .checked_add(k_z)
            .and_then(|n| n.checked_add(k_q))
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_add(m.checked_mul(6)?))
            .ok_or(GadgetError::Internal)?;
        let mut lookup_specs = reserve_vec(total_lookups)?;
        lookup_specs.push(TableKind::Pow2 { max_exp: 63 });
        lookup_specs.push(TableKind::Exp2Frac {
            frac_bits: 8,
            out_bits: 16,
        });
        for _ in 0..k_z.saturating_sub(1) {
            lookup_specs.push(TableKind::Range { bits: 16 });
        }
        lookup_specs.push(TableKind::Range { bits: top_z_bits });
        for _ in 0..k_q.saturating_sub(1) {
            lookup_specs.push(TableKind::Range { bits: 16 });
        }
        lookup_specs.push(TableKind::Range { bits: top_q_bits });
        for bits in [16, 16, 16, 15] {
            lookup_specs.push(TableKind::Range { bits });
        }
        append_cmp_specs(&mut lookup_specs, m, &cmp);

        let inv_top_z = inv_pow2(z_low_bits)?;
        let inv_top_q = inv_pow2(q_low_bits)?;
        let inv_2_48 = inv_pow2(48)?;

        Ok(Self {
            vocab,
            sigma,
            eps_log2,
            c_match_fixed,
            w_z,
            k_z,
            top_z_bits,
            w_q,
            k_q,
            top_q_bits,
            m,
            y_active,
            v_limbs,
            bounds: [a_left, a_right],
            z_decomp,
            q_decomp,
            cmp,
            lookup_specs,
            inv_top_z,
            inv_top_q,
            inv_2_48,
        })
    }

    /// Vocabulary size `V`.
    pub const fn vocab(&self) -> u32 {
        self.vocab
    }

    /// Exponential fractional precision `σ`.
    pub const fn sigma(&self) -> u8 {
        self.sigma
    }

    /// Mixture weight exponent `k = eps_log2` (`ε = 2^{−k}`).
    pub const fn eps_log2(&self) -> u8 {
        self.eps_log2
    }

    /// Public match cost `c_match(k)` in `FixedBits` units.
    pub const fn c_match(&self) -> u64 {
        self.c_match_fixed
    }

    /// Partition width `W_Z = bit_length(V · 2^σ)`.
    pub const fn w_z(&self) -> u8 {
        self.w_z
    }

    /// Limb count `k_Z = ⌈W_Z / 16⌉` of `Z_t`.
    pub const fn k_z(&self) -> usize {
        self.k_z
    }

    /// Narrowed Range width `W_Z − 16(k_Z − 1)` of `Z_{k_Z−1}`.
    pub const fn top_z_bits(&self) -> u8 {
        self.top_z_bits
    }

    /// Scaled-numerator width `W_Q = bit_length((2^σ + 1) · (2^{17} − 1))`.
    pub const fn w_q(&self) -> u8 {
        self.w_q
    }

    /// Limb count `k_Q = ⌈W_Q / 16⌉` of `q_t`.
    pub const fn k_q(&self) -> usize {
        self.k_q
    }

    /// Narrowed Range width `W_Q − 16(k_Q − 1)` of `Q_{k_Q−1}`.
    pub const fn top_q_bits(&self) -> u8 {
        self.top_q_bits
    }

    /// Comparison limb count `m = ⌈(W_Q + 63) / 16⌉`.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Active quadratic right-hand limb count `min(m, k_Q + 3)`.
    pub const fn y_active(&self) -> usize {
        self.y_active
    }

    /// Per-row committed scalar count `6 + (k_Z − 1) + (k_Q − 1) + 3 + 6m` (RFC-0017 item 5).
    pub fn committed_row_count(&self) -> usize {
        6usize
            .saturating_add(self.k_z.saturating_sub(1))
            .saturating_add(self.k_q.saturating_sub(1))
            .saturating_add(3)
            .saturating_add(self.m.saturating_mul(6))
    }

    /// Per-row row-equality count `3m + 6` (RFC-0017 item 5).
    pub fn row_equality_count(&self) -> usize {
        self.m.saturating_mul(3).saturating_add(6)
    }

    /// Prescribed lookup table specifications in RFC-0017 item 5 order (`2 + k_Z + k_Q + 4 + 6m` entries).
    pub fn lookup_specs(&self) -> &[TableKind] {
        &self.lookup_specs
    }

    /// Inclusive public `(left, right)` comparison limb bounds `(A_left, A_right)`.
    pub fn bounds(&self) -> (&[u64], &[u64]) {
        let [left, right] = &self.bounds;
        (left, right)
    }

    /// Verifier-derived `(left, right)` comparison carry widths.
    pub fn carry_widths(&self) -> (&[u8], &[u8]) {
        (self.cmp.left_carries(), self.cmp.right_carries())
    }
}

/// Verifier-derived `OneSidedLog2` checker over `rows` output positions and vocabulary `vocab`.
#[derive(Debug)]
pub struct OneSidedLog2 {
    gadget: GadgetId,
    rows: u32,
    vocab_vars: usize,
    row_vars: usize,
    vars: usize,
    params: Log2Params,
}

/// Claimed multilinear evaluations at `row_target` supplied to [`OneSidedLog2::check_row`].
#[derive(Clone, Copy, Debug)]
pub struct Log2RowValues<'a> {
    /// Verifier-derived row evaluation point (`row_vars` coordinates).
    pub row_target: &'a [Fp2],
    /// Claimed partition sum extension `Z(row_target)`.
    pub z_eval: Fp2,
    /// Claimed scaled numerator extension `q(row_target)`.
    pub q_eval: Fp2,
    /// Optional explicit committed row evaluations (`6 + (k_Z − 1) + (k_Q − 1) + 3 + 6m` entries).
    pub committed_evals: Option<&'a [Fp2]>,
    /// Claimed lookup key evaluations (`2 + k_Z + k_Q + 4 + 6m` entries) in RFC-0017 item 5 order.
    pub lookup_keys: &'a [Fp2],
    /// Claimed lookup value evaluations (`2 + k_Z + k_Q + 4 + 6m` entries; `[P, E, 0, …, 0]`).
    pub lookup_values: &'a [Fp2],
}

/// Row-level `OneSidedLog2` equations verified at `row_target`, with Stages 1–2 and all
/// lookup/source obligations still pending.
#[must_use = "run OneSidedLog2 Stages 1 and 2 and discharge all returned lookup and source claims"]
#[derive(Debug)]
pub struct Log2RowChecked<'g> {
    site: &'g OneSidedLog2,
    row_target: Vec<Fp2>,
    z_eval: Fp2,
    x_targets: Vec<Fp2>,
    y_targets: Vec<Fp2>,
    lookups: Vec<Log2LookupClaim>,
    sources: Vec<Log2SourceClaim>,
}

/// Stage-1 (`stage/1`) batched sumcheck of [`OneSidedLog2`] over the output-row domain.
#[must_use = "verify OneSidedLog2 Stage 1, then run Stage 2 and discharge all obligations"]
#[derive(Debug)]
pub struct Log2Stage1<'g> {
    checked: Log2RowChecked<'g>,
    zero_row: Vec<Fp2>,
    instances: Vec<Instance>,
}

/// Stage-1 verified state, holding `Z(r_1)` and `Y(r_1)` for Stage 2.
#[must_use = "run OneSidedLog2 Stage 2 and discharge all retained lookup and source obligations"]
#[derive(Debug)]
pub struct Log2Stage1Done<'g> {
    site: &'g OneSidedLog2,
    row_target: Vec<Fp2>,
    r1: Vec<Fp2>,
    z_eval: Fp2,
    z_r1: Fp2,
    y_r1: Fp2,
    lookups: Vec<Log2LookupClaim>,
    sources: Vec<Log2SourceClaim>,
}

/// Stage-2 (`stage/2`) batched sumcheck of [`OneSidedLog2`] over `(vocab, row)`.
#[must_use = "verify OneSidedLog2 Stage 2 and discharge all retained lookup and source obligations"]
#[derive(Debug)]
pub struct Log2Stage2<'g> {
    done1: Log2Stage1Done<'g>,
    zero_vt: Vec<Fp2>,
    instances: [Instance; 4],
}

/// Completed two-stage [`OneSidedLog2`] reduction with all lookup and source obligations intact.
#[must_use = "discharge every Pow2, Exp2Frac and Range lookup claim and every source/advice opening claim"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Log2Verified {
    lookups: Vec<Log2LookupClaim>,
    sources: Vec<Log2SourceClaim>,
}

impl Log2Verified {
    /// All `2 + k_Z + k_Q + 4 + 6m` lookup claims at `row_target`.
    pub fn lookup_claims(&self) -> &[Log2LookupClaim] {
        &self.lookups
    }

    /// All committed row advice and `(vocab, row)` input claims accumulated across `row_target`, `r_1` and `r_2`.
    pub fn source_claims(&self) -> &[Log2SourceClaim] {
        &self.sources
    }
}

impl OneSidedLog2 {
    /// Derive a verifier-owned [`OneSidedLog2`] checker from validated public dimensions.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] unless `1 ≤ vocab ≤ 2^18`, `1 ≤ rows ≤ 2^17`, `sigma ≤ 32`,
    /// and `eps_log2 ≤ 16`; [`GadgetError::Memory`] on failed reservation.
    pub fn derive(
        gadget: GadgetId,
        vocab: u32,
        rows: u32,
        sigma: u8,
        eps_log2: u8,
    ) -> Result<Self, GadgetError> {
        if !(1..=1 << 17).contains(&rows) {
            return Err(GadgetError::Internal);
        }
        let params = Log2Params::derive(vocab, sigma, eps_log2)?;
        let vocab_vars = log2_ceil(vocab)?;
        let row_vars = log2_ceil(rows)?;
        let vars = vocab_vars
            .checked_add(row_vars)
            .ok_or(GadgetError::Internal)?;
        Ok(Self {
            gadget,
            rows,
            vocab_vars,
            row_vars,
            vars,
            params,
        })
    }

    /// Verifier-owned gadget identifier.
    pub const fn gadget(&self) -> GadgetId {
        self.gadget
    }

    /// Output-row count `rows`.
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Vocabulary variable count `⌈log2 vocab⌉`.
    pub const fn vocab_variables(&self) -> usize {
        self.vocab_vars
    }

    /// Row variable count `⌈log2 rows⌉`.
    pub const fn row_variables(&self) -> usize {
        self.row_vars
    }

    /// Combined `(vocab, row)` variable count.
    pub const fn variables(&self) -> usize {
        self.vars
    }

    /// Public schema parameters (`c_match`, `w_z`, `w_q`, `m`, bounds, carry widths, lookup specs).
    pub const fn params(&self) -> &Log2Params {
        &self.params
    }

    /// Verify row-level decompositions (`Z_t`, `q_t`, and 63-bit `P_t`), Range value zeros,
    /// and `BigIntCompare` (`ψ_t · (Z_t + V) · 2^16 ≤ q_t · P_t`) at `row_target`.
    ///
    /// Active limb targets `X_1 .. X_{k_Z}` and `Y_0 .. Y_{y_active−1}` are derived from the
    /// comparison's normalization chains and sumchecked in Stage 1; inactive limbs (`X_0 = 0`,
    /// `X_j = V_{j−1} · ψ` for `j > k_Z`, and `Y_j = 0` for `j ≥ y_active`) are fixed to their
    /// exact public values so a prover cannot inject slack into un-sumchecked high limbs.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `row_target.len() != row_vars`;
    /// [`GadgetError::Rejected`] on any wrong slice length, non-zero Range lookup value,
    /// failed affine limb key, or failed comparison equation; [`GadgetError::Memory`] on failed reservation.
    pub fn check_row<'g>(
        &'g self,
        values: Log2RowValues<'_>,
    ) -> Result<Log2RowChecked<'g>, GadgetError> {
        if values.row_target.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        let p = &self.params;
        let reject = GadgetError::Rejected {
            gadget: self.gadget,
        };
        let expected_lookups = p.lookup_specs.len();
        if values.lookup_keys.len() != expected_lookups
            || values.lookup_values.len() != expected_lookups
        {
            return Err(reject);
        }

        let p_eval = *values.lookup_values.first().ok_or(GadgetError::Internal)?;
        let e_eval = *values.lookup_values.get(1).ok_or(GadgetError::Internal)?;
        let range_vals = values.lookup_values.get(2..).ok_or(GadgetError::Internal)?;
        if range_vals.iter().any(|&v| v != Fp2::ZERO) {
            return Err(reject);
        }

        let k_z = p.k_z;
        let k_q = p.k_q;
        let m = p.m;
        let z_low_len = k_z.saturating_sub(1);
        let q_low_len = k_q.saturating_sub(1);

        let z_end = 2usize.checked_add(k_z).ok_or(GadgetError::Internal)?;
        let q_end = z_end.checked_add(k_q).ok_or(GadgetError::Internal)?;
        let p_end = q_end.checked_add(4).ok_or(GadgetError::Internal)?;

        let i_key = *values.lookup_keys.first().ok_or(GadgetError::Internal)?;
        let f_key = *values.lookup_keys.get(1).ok_or(GadgetError::Internal)?;
        let z_keys = values
            .lookup_keys
            .get(2..z_end)
            .ok_or(GadgetError::Internal)?;
        let q_keys = values
            .lookup_keys
            .get(z_end..q_end)
            .ok_or(GadgetError::Internal)?;
        let p_keys = values
            .lookup_keys
            .get(q_end..p_end)
            .ok_or(GadgetError::Internal)?;
        let cmp_flat = values
            .lookup_keys
            .get(p_end..)
            .ok_or(GadgetError::Internal)?;

        let (m_opt, z_low, q_low, p_low) = if let Some(committed) = values.committed_evals {
            if committed.len() != p.committed_row_count() {
                return Err(reject);
            }
            let m_c = *committed.first().ok_or(GadgetError::Internal)?;
            let i_c = *committed.get(1).ok_or(GadgetError::Internal)?;
            let p_c = *committed.get(2).ok_or(GadgetError::Internal)?;
            let f_c = *committed.get(3).ok_or(GadgetError::Internal)?;
            let e_c = *committed.get(4).ok_or(GadgetError::Internal)?;
            let q_c = *committed.get(5).ok_or(GadgetError::Internal)?;
            if i_c != i_key
                || p_c != p_eval
                || f_c != f_key
                || e_c != e_eval
                || q_c != values.q_eval
            {
                return Err(reject);
            }
            let z_c_end = 6usize.checked_add(z_low_len).ok_or(GadgetError::Internal)?;
            let q_c_end = z_c_end
                .checked_add(q_low_len)
                .ok_or(GadgetError::Internal)?;
            let p_c_end = q_c_end.checked_add(3).ok_or(GadgetError::Internal)?;
            if committed.get(p_c_end..) != Some(cmp_flat) {
                return Err(reject);
            }
            (
                Some(m_c),
                committed.get(6..z_c_end).ok_or(GadgetError::Internal)?,
                committed
                    .get(z_c_end..q_c_end)
                    .ok_or(GadgetError::Internal)?,
                committed
                    .get(q_c_end..p_c_end)
                    .ok_or(GadgetError::Internal)?,
            )
        } else {
            (
                None,
                z_keys.get(..z_low_len).ok_or(GadgetError::Internal)?,
                q_keys.get(..q_low_len).ok_or(GadgetError::Internal)?,
                p_keys.get(..3).ok_or(GadgetError::Internal)?,
            )
        };

        let _z_ok = p.z_decomp.check(
            self.gadget,
            decomposition::Values {
                point: values.row_target,
                value: values.z_eval,
                low: z_low,
                keys: z_keys,
            },
        )?;
        let _q_ok = p.q_decomp.check(
            self.gadget,
            decomposition::Values {
                point: values.row_target,
                value: values.q_eval,
                low: q_low,
                keys: q_keys,
            },
        )?;

        // 63-bit unsigned decomposition of P_t = P_0 + 2^16 P_1 + 2^32 P_2 + 2^48 P_3 (P_3 ∈ Range(15)).
        let radix = Fp2::from(Fp::reduce(1u64 << 16));
        let mut p_rem = p_eval;
        let mut scale = Fp2::ONE;
        for (&low, &key) in p_low.iter().zip(p_keys) {
            if low != key {
                return Err(reject);
            }
            p_rem -= scale * low;
            scale *= radix;
        }
        if p_keys.get(3).copied() != Some(p_rem * p.inv_2_48) {
            return Err(reject);
        }

        let cmp_arrays = split_six(cmp_flat, m)?;
        let [cmp_ld, cmp_lc, cmp_rd, cmp_rc, _, _] = cmp_arrays;
        let psi_rt = lt(values.row_target, u64::from(self.rows)).ok_or(GadgetError::Internal)?;

        let mut x_targets = reserve_vec(m)?;
        let mut prev_lc = Fp2::ZERO;
        for j in 0..m {
            let ld = cmp_ld.get(j).copied().ok_or(GadgetError::Internal)?;
            let lc = cmp_lc.get(j).copied().ok_or(GadgetError::Internal)?;
            if j == 0 {
                x_targets.push(Fp2::ZERO);
            } else if j <= k_z {
                x_targets.push(ld + radix * lc - prev_lc);
            } else {
                let vj = p.v_limbs.get(j.saturating_sub(1)).copied().unwrap_or(0);
                x_targets.push(Fp2::from(Fp::reduce(vj)) * psi_rt);
            }
            prev_lc = lc;
        }

        let mut y_targets = reserve_vec(m)?;
        let mut prev_rc = Fp2::ZERO;
        for j in 0..m {
            let rd = cmp_rd.get(j).copied().ok_or(GadgetError::Internal)?;
            let rc = cmp_rc.get(j).copied().ok_or(GadgetError::Internal)?;
            if j < p.y_active {
                y_targets.push(rd + radix * rc - prev_rc);
            } else {
                y_targets.push(Fp2::ZERO);
            }
            prev_rc = rc;
        }

        let _cmp_ok = p.cmp.check(
            self.gadget,
            comparison::Values {
                point: values.row_target,
                left: &x_targets,
                right: &y_targets,
                arrays: cmp_arrays,
            },
        )?;

        let row_target = clone_vec(values.row_target)?;
        let mut lookups = reserve_vec(expected_lookups)?;
        for ((kind, &key), &val) in p
            .lookup_specs
            .iter()
            .zip(values.lookup_keys)
            .zip(values.lookup_values)
        {
            lookups.push(Log2LookupClaim {
                kind: kind.clone(),
                point: clone_vec(&row_target)?,
                key,
                value: val,
            });
        }

        let source_cap = p
            .committed_row_count()
            .checked_add(p.committed_row_count())
            .ok_or(GadgetError::Internal)?;
        let mut sources = reserve_vec(source_cap)?;
        if let Some(m_c) = m_opt {
            sources.push(Log2SourceClaim {
                role: Log2Role::MatchBit,
                point: clone_vec(&row_target)?,
                value: m_c,
            });
        }
        for (role, value) in [
            (Log2Role::ExponentKey, i_key),
            (Log2Role::Pow2Value, p_eval),
            (Log2Role::FractionKey, f_key),
            (Log2Role::Exp2FracValue, e_eval),
            (Log2Role::ScaledNumerator, values.q_eval),
        ] {
            sources.push(Log2SourceClaim {
                role,
                point: clone_vec(&row_target)?,
                value,
            });
        }
        for (h, &v) in z_low.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::PartitionLow(h),
                point: clone_vec(&row_target)?,
                value: v,
            });
        }
        for (a, &v) in q_low.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::NumeratorLow(a),
                point: clone_vec(&row_target)?,
                value: v,
            });
        }
        for (b, &v) in p_low.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::Pow2Low(b),
                point: clone_vec(&row_target)?,
                value: v,
            });
        }
        push_log2_cmp_sources(&mut sources, &row_target, cmp_arrays)?;

        Ok(Log2RowChecked {
            site: self,
            row_target,
            z_eval: values.z_eval,
            x_targets,
            y_targets,
            lookups,
            sources,
        })
    }
}

impl<'g> Log2RowChecked<'g> {
    /// Derived comparison left-hand targets `X_0 .. X_{m−1}` at `row_target`.
    pub fn x_targets(&self) -> &[Fp2] {
        &self.x_targets
    }

    /// Derived comparison right-hand targets `Y_0 .. Y_{m−1}` at `row_target`.
    pub fn y_targets(&self) -> &[Fp2] {
        &self.y_targets
    }

    /// Squeeze `stage/1/zero/{0..row_vars}` from `transcript` and build Stage 1 for `claimed_uhat`.
    ///
    /// # Errors
    /// [`StageVerifyError`] on transcript, reservation, or non-canonical `claimed_uhat ≥ P` failure.
    pub fn squeeze_stage1(
        self,
        transcript: &mut Transcript,
        claimed_uhat: FixedBits,
    ) -> Result<Log2Stage1<'g>, StageVerifyError> {
        let zero_row = squeeze_zero(transcript, Prefix::Stage(1), self.site.row_vars)?;
        Ok(self.prepare_stage1(zero_row, claimed_uhat)?)
    }

    /// Build Stage 1 from an already squeezed stage-1 row zero-check point and `claimed_uhat`.
    ///
    /// `claimed_uhat.0` is checked strictly canonical (`< P`) via [`Fp::from_canonical`]; a
    /// non-canonical `u64 ≥ P` is rejected as `GadgetError::Rejected` rather than reduced modulo `P`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `zero_row.len() != row_vars`;
    /// [`GadgetError::Rejected`] when `claimed_uhat.0 >= Fp::MODULUS`;
    /// [`GadgetError::Memory`] on failed reservation.
    pub fn prepare_stage1(
        self,
        zero_row: Vec<Fp2>,
        claimed_uhat: FixedBits,
    ) -> Result<Log2Stage1<'g>, GadgetError> {
        if zero_row.len() != self.site.row_vars {
            return Err(GadgetError::Internal);
        }
        let Some(uhat_fp) = Fp::from_canonical(claimed_uhat.0) else {
            return Err(GadgetError::Rejected {
                gadget: self.site.gadget,
            });
        };
        let p = &self.site.params;
        let vars = self.site.row_vars;
        let count = 4usize
            .checked_add(p.k_z)
            .and_then(|n| n.checked_add(p.y_active))
            .ok_or(GadgetError::Internal)?;
        let mut instances = reserve_vec(count)?;
        // 0: match booleanity (m_t = 0 for k = 0, else m_t - m_t^2 = 0)
        instances.push(Instance {
            claim: Fp2::ZERO,
            vars,
            evals: 1,
        });
        // 1: q_t - (Y_t + psi_t) * E_t = 0
        instances.push(Instance {
            claim: Fp2::ZERO,
            vars,
            evals: 3,
        });
        // 2 .. 2 + k_z: X_1 .. X_{k_z}
        for j in 1..=p.k_z {
            let claim = *self.x_targets.get(j).ok_or(GadgetError::Internal)?;
            let evals = if j < p.k_z { 1 } else { p.k_z };
            instances.push(Instance { claim, vars, evals });
        }
        // 2 + k_z .. 2 + k_z + y_active: Y_0 .. Y_{y_active - 1}
        let qp_evals = p.k_q.checked_add(4).ok_or(GadgetError::Internal)?;
        for j in 0..p.y_active {
            let claim = *self.y_targets.get(j).ok_or(GadgetError::Internal)?;
            instances.push(Instance {
                claim,
                vars,
                evals: qp_evals,
            });
        }
        // idx_pad_z: (psi_t - 1) * Z_t = 0
        instances.push(Instance {
            claim: Fp2::ZERO,
            vars,
            evals: 1,
        });
        // idx_uhat: sum_t B_t = claimed_uhat
        instances.push(Instance {
            claim: Fp2::from(uhat_fp),
            vars,
            evals: 3,
        });

        Ok(Log2Stage1 {
            checked: self,
            zero_row,
            instances,
        })
    }
}

impl<'g> Log2Stage1<'g> {
    /// Prescribed sumcheck degree (`3`) of `OneSidedLog2` Stage 1.
    pub const fn degree(&self) -> usize {
        3
    }

    /// Stage-1 instances in schedule order.
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Evaluate Stage-1 instance `index` at `point` from its slice of `final_evals`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] for an invalid `index` or `point` length;
    /// [`GadgetError::Rejected`] for a wrong `values` count.
    pub fn evaluate(
        &self,
        index: usize,
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<Fp2, GadgetError> {
        let site = self.checked.site;
        if point.len() != site.row_vars {
            return Err(GadgetError::Internal);
        }
        let p = &site.params;
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        let psi = lt(point, u64::from(site.rows)).ok_or(GadgetError::Internal)?;
        let x_end = 2usize.checked_add(p.k_z).ok_or(GadgetError::Internal)?;
        let y_end = x_end.checked_add(p.y_active).ok_or(GadgetError::Internal)?;
        let idx_pad_z = y_end;
        let idx_uhat = idx_pad_z.checked_add(1).ok_or(GadgetError::Internal)?;

        if index == 0 {
            if values.len() != 1 {
                return Err(reject);
            }
            let m_at = *values.first().ok_or(GadgetError::Internal)?;
            let eq_zr = eq(&self.zero_row, point).ok_or(GadgetError::Internal)?;
            let m_rel = if p.eps_log2 == 0 {
                m_at
            } else {
                m_at - m_at * m_at
            };
            Ok(eq_zr * m_rel)
        } else if index == 1 {
            if values.len() != 3 {
                return Err(reject);
            }
            let q_at = *values.first().ok_or(GadgetError::Internal)?;
            let y_at = *values.get(1).ok_or(GadgetError::Internal)?;
            let e_at = *values.get(2).ok_or(GadgetError::Internal)?;
            let eq_zr = eq(&self.zero_row, point).ok_or(GadgetError::Internal)?;
            Ok(eq_zr * (q_at - (y_at + psi) * e_at))
        } else if (2..x_end).contains(&index) {
            let j = index.saturating_sub(1); // 1 ..= k_z
            let vj = Fp2::from(Fp::reduce(
                p.v_limbs.get(j.saturating_sub(1)).copied().unwrap_or(0),
            ));
            let eq_rt = eq(&self.checked.row_target, point).ok_or(GadgetError::Internal)?;
            if j < p.k_z {
                if values.len() != 1 {
                    return Err(reject);
                }
                let zh = *values.first().ok_or(GadgetError::Internal)?;
                Ok(eq_rt * psi * (zh + vj))
            } else {
                if values.len() != p.k_z {
                    return Err(reject);
                }
                let z_at = *values.first().ok_or(GadgetError::Internal)?;
                let z_low = values.get(1..).ok_or(GadgetError::Internal)?;
                let z_top = derive_top_limb(z_at, z_low, p.inv_top_z);
                Ok(eq_rt * psi * (z_top + vj))
            }
        } else if (x_end..y_end).contains(&index) {
            let j = index.saturating_sub(x_end); // 0 .. y_active
            let expected = p.k_q.checked_add(4).ok_or(GadgetError::Internal)?;
            if values.len() != expected {
                return Err(reject);
            }
            let q_at = *values.first().ok_or(GadgetError::Internal)?;
            let q_low = values.get(1..p.k_q).ok_or(GadgetError::Internal)?;
            let p_at = *values.get(p.k_q).ok_or(GadgetError::Internal)?;
            let p_low_start = p.k_q.checked_add(1).ok_or(GadgetError::Internal)?;
            let p_low = values.get(p_low_start..).ok_or(GadgetError::Internal)?;
            let q_top = derive_top_limb(q_at, q_low, p.inv_top_q);
            let p_top = derive_top_limb(p_at, p_low, p.inv_2_48);

            let mut yj = Fp2::ZERO;
            for a in 0..p.k_q {
                let qa = if a < q_low.len() {
                    *q_low.get(a).ok_or(GadgetError::Internal)?
                } else {
                    q_top
                };
                for b in 0..4 {
                    if a.checked_add(b) == Some(j) {
                        let pb = if b < p_low.len() {
                            *p_low.get(b).ok_or(GadgetError::Internal)?
                        } else {
                            p_top
                        };
                        yj += qa * pb;
                    }
                }
            }
            let eq_rt = eq(&self.checked.row_target, point).ok_or(GadgetError::Internal)?;
            Ok(eq_rt * yj)
        } else if index == idx_pad_z {
            if values.len() != 1 {
                return Err(reject);
            }
            let z_at = *values.first().ok_or(GadgetError::Internal)?;
            let eq_zr = eq(&self.zero_row, point).ok_or(GadgetError::Internal)?;
            Ok(eq_zr * (psi - Fp2::ONE) * z_at)
        } else if index == idx_uhat {
            if values.len() != 3 {
                return Err(reject);
            }
            let m_at = *values.first().ok_or(GadgetError::Internal)?;
            let i_at = *values.get(1).ok_or(GadgetError::Internal)?;
            let f_at = *values.get(2).ok_or(GadgetError::Internal)?;
            let k_cost = Fp2::from(Fp::reduce(256u64.saturating_mul(u64::from(p.eps_log2))));
            let scale_256 = Fp2::from(Fp::reduce(256));
            let c_match_fp = Fp2::from(Fp::reduce(p.c_match_fixed));
            let mismatch = k_cost + scale_256 * i_at + f_at;
            Ok(psi * (m_at * c_match_fp + (Fp2::ONE - m_at) * mismatch))
        } else {
            Err(GadgetError::Internal)
        }
    }

    /// Run `sumcheck::verify_batch` under `stage/1` and enforce cross-instance evaluation equality.
    ///
    /// # Errors
    /// [`StageVerifyError`] on any sumcheck or cross-instance consistency failure.
    pub fn verify(
        self,
        transcript: &mut Transcript,
        proof: &SumcheckProof,
    ) -> Result<(Batch, Log2Stage1Done<'g>), StageVerifyError> {
        let batch = sumcheck::verify_batch(
            transcript,
            Prefix::Stage(1),
            self.degree(),
            &self.instances,
            proof,
            |i, pt, vals| self.evaluate(i, pt, vals).ok(),
        )?;
        let done = self.finish(clone_vec(&batch.point)?, &proof.final_evals)?;
        Ok((batch, done))
    }

    /// Enforce equality of all repeated evaluations at `point` and advance to Stage 2.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != row_vars`;
    /// [`GadgetError::Rejected`] when `final_evals` has the wrong length or disagrees across instances.
    pub fn finish(
        self,
        point: Vec<Fp2>,
        final_evals: &[Fp2],
    ) -> Result<Log2Stage1Done<'g>, GadgetError> {
        let site = self.checked.site;
        if point.len() != site.row_vars {
            return Err(GadgetError::Internal);
        }
        let p = &site.params;
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        let z_low_len = p.k_z.saturating_sub(1);
        let qp_stride = p.k_q.checked_add(4).ok_or(GadgetError::Internal)?;
        let x_evals_end = 4usize
            .checked_add(z_low_len)
            .and_then(|n| n.checked_add(p.k_z))
            .ok_or(GadgetError::Internal)?;
        let y_evals_end = x_evals_end
            .checked_add(
                p.y_active
                    .checked_mul(qp_stride)
                    .ok_or(GadgetError::Internal)?,
            )
            .ok_or(GadgetError::Internal)?;
        let expected_len = y_evals_end.checked_add(4).ok_or(GadgetError::Internal)?;
        if final_evals.len() != expected_len {
            return Err(reject);
        }

        let m_r1 = *final_evals.first().ok_or(GadgetError::Internal)?;
        let q_r1 = *final_evals.get(1).ok_or(GadgetError::Internal)?;
        let y_r1 = *final_evals.get(2).ok_or(GadgetError::Internal)?;
        let e_r1 = *final_evals.get(3).ok_or(GadgetError::Internal)?;

        let z_single_end = 4usize.checked_add(z_low_len).ok_or(GadgetError::Internal)?;
        let z_low_single = final_evals
            .get(4..z_single_end)
            .ok_or(GadgetError::Internal)?;
        let x_top_slice = final_evals
            .get(z_single_end..x_evals_end)
            .ok_or(GadgetError::Internal)?;
        let z_r1 = *x_top_slice.first().ok_or(GadgetError::Internal)?;
        let z_low_top = x_top_slice.get(1..).ok_or(GadgetError::Internal)?;
        if z_low_single != z_low_top {
            return Err(reject);
        }

        let qp_first_end = x_evals_end
            .checked_add(qp_stride)
            .ok_or(GadgetError::Internal)?;
        let qp_head = final_evals
            .get(x_evals_end..qp_first_end)
            .ok_or(GadgetError::Internal)?;
        for idx in 1..p.y_active {
            let start = x_evals_end
                .checked_add(idx.checked_mul(qp_stride).ok_or(GadgetError::Internal)?)
                .ok_or(GadgetError::Internal)?;
            let end = start.checked_add(qp_stride).ok_or(GadgetError::Internal)?;
            if final_evals.get(start..end) != Some(qp_head) {
                return Err(reject);
            }
        }
        let q_qp = *qp_head.first().ok_or(GadgetError::Internal)?;
        if q_qp != q_r1 {
            return Err(reject);
        }
        let q_low_r1 = qp_head.get(1..p.k_q).ok_or(GadgetError::Internal)?;
        let p_r1 = *qp_head.get(p.k_q).ok_or(GadgetError::Internal)?;
        let p_low_start = p.k_q.checked_add(1).ok_or(GadgetError::Internal)?;
        let p_low_r1 = qp_head.get(p_low_start..).ok_or(GadgetError::Internal)?;

        let tail = final_evals
            .get(y_evals_end..)
            .ok_or(GadgetError::Internal)?;
        let z_pad = *tail.first().ok_or(GadgetError::Internal)?;
        let m_uhat = *tail.get(1).ok_or(GadgetError::Internal)?;
        let i_r1 = *tail.get(2).ok_or(GadgetError::Internal)?;
        let f_r1 = *tail.get(3).ok_or(GadgetError::Internal)?;
        if z_pad != z_r1 || m_uhat != m_r1 {
            return Err(reject);
        }

        let mut sources = self.checked.sources;
        for (role, value) in [
            (Log2Role::MatchBit, m_r1),
            (Log2Role::ExponentKey, i_r1),
            (Log2Role::Pow2Value, p_r1),
            (Log2Role::FractionKey, f_r1),
            (Log2Role::Exp2FracValue, e_r1),
            (Log2Role::ScaledNumerator, q_r1),
        ] {
            sources.push(Log2SourceClaim {
                role,
                point: clone_vec(&point)?,
                value,
            });
        }
        for (h, &v) in z_low_top.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::PartitionLow(h),
                point: clone_vec(&point)?,
                value: v,
            });
        }
        for (a, &v) in q_low_r1.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::NumeratorLow(a),
                point: clone_vec(&point)?,
                value: v,
            });
        }
        for (b, &v) in p_low_r1.iter().enumerate() {
            sources.push(Log2SourceClaim {
                role: Log2Role::Pow2Low(b),
                point: clone_vec(&point)?,
                value: v,
            });
        }

        Ok(Log2Stage1Done {
            site,
            row_target: self.checked.row_target,
            r1: point,
            z_eval: self.checked.z_eval,
            z_r1,
            y_r1,
            lookups: self.checked.lookups,
            sources,
        })
    }
}

impl<'g> Log2Stage1Done<'g> {
    /// Squeeze `stage/2/zero/{0..vars}` from `transcript` and build Stage 2.
    ///
    /// # Errors
    /// [`StageVerifyError`] on transcript or reservation failure.
    pub fn squeeze_stage2(
        self,
        transcript: &mut Transcript,
    ) -> Result<Log2Stage2<'g>, StageVerifyError> {
        let zero_vt = squeeze_zero(transcript, Prefix::Stage(2), self.site.vars)?;
        Ok(self.prepare_stage2(zero_vt)?)
    }

    /// Build Stage 2 from an already squeezed `(vocab, row)` zero-check point.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `zero_vt.len() != vars`.
    pub fn prepare_stage2(self, zero_vt: Vec<Fp2>) -> Result<Log2Stage2<'g>, GadgetError> {
        if zero_vt.len() != self.site.vars {
            return Err(GadgetError::Internal);
        }
        let vars = self.site.vars;
        let instances = [
            Instance {
                claim: self.z_eval,
                vars,
                evals: 1,
            },
            Instance {
                claim: self.z_r1,
                vars,
                evals: 1,
            },
            Instance {
                claim: self.y_r1,
                vars,
                evals: 2,
            },
            Instance {
                claim: Fp2::ZERO,
                vars,
                evals: 3,
            },
        ];
        Ok(Log2Stage2 {
            done1: self,
            zero_vt,
            instances,
        })
    }
}

impl Log2Stage2<'_> {
    /// Prescribed sumcheck degree (`3`) of `OneSidedLog2` Stage 2.
    pub const fn degree(&self) -> usize {
        3
    }

    /// Stage-2 instances in schedule order (`Z(row_target)`, `Z(r_1)`, `Y(r_1)`, `m_t · (h_y − h_{ŷ}) = 0`).
    pub const fn instances(&self) -> &[Instance; 4] {
        &self.instances
    }

    /// Evaluate Stage-2 instance `index` (`0..4`) at `point` from its slice of `final_evals`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] for an invalid `index` or `point` length;
    /// [`GadgetError::Rejected`] for a wrong `values` count.
    pub fn evaluate(
        &self,
        index: usize,
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<Fp2, GadgetError> {
        let site = self.done1.site;
        if point.len() != site.vars {
            return Err(GadgetError::Internal);
        }
        let (v_pt, row_pt) = point
            .split_at_checked(site.vocab_vars)
            .ok_or(GadgetError::Internal)?;
        let cv = lt(v_pt, u64::from(site.params.vocab)).ok_or(GadgetError::Internal)?;
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        match index {
            0 => {
                if values.len() != 1 {
                    return Err(reject);
                }
                let ev = *values.first().ok_or(GadgetError::Internal)?;
                let eq_rt = eq(&self.done1.row_target, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_rt * cv * ev)
            }
            1 => {
                if values.len() != 1 {
                    return Err(reject);
                }
                let ev = *values.first().ok_or(GadgetError::Internal)?;
                let eq_r1 = eq(&self.done1.r1, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_r1 * cv * ev)
            }
            2 => {
                if values.len() != 2 {
                    return Err(reject);
                }
                let hy = *values.first().ok_or(GadgetError::Internal)?;
                let ev = *values.get(1).ok_or(GadgetError::Internal)?;
                let eq_r1 = eq(&self.done1.r1, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_r1 * hy * ev)
            }
            3 => {
                if values.len() != 3 {
                    return Err(reject);
                }
                let mt = *values.first().ok_or(GadgetError::Internal)?;
                let hy = *values.get(1).ok_or(GadgetError::Internal)?;
                let hyhat = *values.get(2).ok_or(GadgetError::Internal)?;
                let eq_vt = eq(&self.zero_vt, point).ok_or(GadgetError::Internal)?;
                Ok(eq_vt * mt * (hy - hyhat))
            }
            _ => Err(GadgetError::Internal),
        }
    }

    /// Run `sumcheck::verify_batch` under `stage/2` and enforce cross-instance evaluation equality.
    ///
    /// # Errors
    /// [`StageVerifyError`] on any sumcheck or cross-instance consistency failure.
    pub fn verify(
        self,
        transcript: &mut Transcript,
        proof: &SumcheckProof,
    ) -> Result<(Batch, Log2Verified), StageVerifyError> {
        let batch = sumcheck::verify_batch(
            transcript,
            Prefix::Stage(2),
            self.degree(),
            &self.instances,
            proof,
            |i, pt, vals| self.evaluate(i, pt, vals).ok(),
        )?;
        let verified = self.finish(clone_vec(&batch.point)?, &proof.final_evals)?;
        Ok((batch, verified))
    }

    /// Enforce equality of repeated `e` and `h_y` evaluations at `point` and finalize all obligations.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != vars`;
    /// [`GadgetError::Rejected`] when `final_evals` has length `!= 7` or disagrees across instances.
    pub fn finish(
        mut self,
        point: Vec<Fp2>,
        final_evals: &[Fp2],
    ) -> Result<Log2Verified, GadgetError> {
        let site = self.done1.site;
        if point.len() != site.vars {
            return Err(GadgetError::Internal);
        }
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        if final_evals.len() != 7 {
            return Err(reject);
        }
        let ev0 = *final_evals.first().ok_or(GadgetError::Internal)?;
        let ev1 = *final_evals.get(1).ok_or(GadgetError::Internal)?;
        let hy0 = *final_evals.get(2).ok_or(GadgetError::Internal)?;
        let ev2 = *final_evals.get(3).ok_or(GadgetError::Internal)?;
        let mt = *final_evals.get(4).ok_or(GadgetError::Internal)?;
        let hy1 = *final_evals.get(5).ok_or(GadgetError::Internal)?;
        let hyhat = *final_evals.get(6).ok_or(GadgetError::Internal)?;
        if ev0 != ev1 || ev0 != ev2 || hy0 != hy1 {
            return Err(reject);
        }

        let row_pt = point.get(site.vocab_vars..).ok_or(GadgetError::Internal)?;
        let row_point = clone_vec(row_pt)?;
        self.done1.sources.push(Log2SourceClaim {
            role: Log2Role::Exponentials,
            point: clone_vec(&point)?,
            value: ev0,
        });
        self.done1.sources.push(Log2SourceClaim {
            role: Log2Role::RecordedIndicator,
            point: clone_vec(&point)?,
            value: hy0,
        });
        self.done1.sources.push(Log2SourceClaim {
            role: Log2Role::DecodedIndicator,
            point,
            value: hyhat,
        });
        self.done1.sources.push(Log2SourceClaim {
            role: Log2Role::MatchBit,
            point: row_point,
            value: mt,
        });

        Ok(Log2Verified {
            lookups: self.done1.lookups,
            sources: self.done1.sources,
        })
    }
}

// ============================================================================
// Runtime Table Interning (RFC-0017 item 6)
// ============================================================================

/// Batch predictor mode governing runtime table interning (RFC-0017 item 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoringMode {
    /// Exact mode: no `OneSidedLog2` tables and no `Exp_T` for greedy requests.
    Exact,
    /// Bounded mixture mode (`ε = 2^{−eps_log2}`): greedy requests use `Exp_1`, and every
    /// output-bearing request interns `OneSidedLog2` lookup specs.
    Mixture {
        /// `k` in `ε = 2^{−k}` (`0..=16`).
        eps_log2: u8,
    },
}

impl ScoringMode {
    /// Extract [`ScoringMode`] from a [`PredictorSpec`].
    pub const fn from_predictor(predictor: PredictorSpec) -> Self {
        match predictor {
            PredictorSpec::Exact { .. } => Self::Exact,
            PredictorSpec::Mixture { eps_log2, .. } => Self::Mixture { eps_log2 },
        }
    }
}

/// Per-request decoding descriptor for runtime scoring table interning (RFC-0017 item 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoringRequest {
    /// Number of recorded output tokens (`0` skips runtime scoring tables for this request).
    pub outputs: u32,
    /// Request decoding rule (`Greedy` or `Temperature { t }`).
    pub decoding: Decoding,
}

/// Public configuration passed to [`intern_scoring_tables`] (RFC-0017 item 6).
#[derive(Clone, Copy, Debug)]
pub struct InternConfig<'a> {
    /// Existing `ClaimList.tables` specifications in catalog order.
    pub base_tables: &'a [TableKind],
    /// Vocabulary size `1..=2^18`.
    pub vocab: u32,
    /// Exponential precision `0..=32` (`exp_sigma`).
    pub sigma: u8,
    /// Public LM-head `logits_exponent` (`σ_ℓ = 2^{logits_exponent}`, RFC-0027).
    pub logits_exponent: i8,
    /// Lower bound of lowered LM-head logits.
    pub logit_lo: i32,
    /// Upper bound of lowered LM-head logits.
    pub logit_hi: i32,
    /// Policy ceiling `max_log2_rows` (`1..=32`).
    pub max_log2_rows: u8,
    /// Batch predictor mode (`Exact` or `Mixture { eps_log2 }`).
    pub mode: ScoringMode,
    /// Batch requests in statement order.
    pub requests: &'a [ScoringRequest],
}

/// One distinct effective temperature's interned `Exp_T` table entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpTableEntry {
    /// Effective rational temperature (`1/1` for greedy requests in `Mixture` mode).
    pub temperature: Rational,
    /// Interned `TableId` in the expanded catalog.
    pub table_id: TableId,
    /// Selected narrowest `TableKind::Exp` specification.
    pub kind: TableKind,
}

/// Per-request interned runtime `TableId`s in batch order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestTables {
    /// `Argmax` (`Range(min_key_bits)`) or `InverseCdf` (`2 + 2k + 12m` `Range` specs) `TableId`s.
    pub decoding_ids: Vec<TableId>,
    /// `OneSidedLog2` (`2 + k_Z + k_Q + 4 + 6m` specs) `TableId`s, or empty in `Exact` mode / `outputs == 0`.
    pub log2_ids: Vec<TableId>,
}

/// Expanded runtime table catalog and per-temperature / per-request `TableId` bindings (RFC-0017 item 6).
#[must_use = "bind all interned TableIds into the runtime lookup schedule"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternedScoringTables {
    /// Complete deduplicated `TableKind` catalog (`base_tables` followed by newly appended kinds).
    pub catalog: Vec<TableKind>,
    /// Distinct effective temperatures in first-seen output-bearing request order (`≤ 4`).
    pub exp_tables: Vec<ExpTableEntry>,
    /// Per-request decoding and `OneSidedLog2` `TableId`s in batch order.
    pub request_tables: Vec<RequestTables>,
}

/// Errors returned by [`intern_scoring_tables`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternError {
    /// Out-of-domain temperature, `eps_log2`, or unrepresentable `2^{logits_exponent} / T` scale.
    Scoring(ScoringError),
    /// More than [`MAX_BATCH_TEMPERATURES`] (`4`) distinct effective temperatures in one batch.
    TooManyTemperatures {
        /// Distinct effective temperature count encountered.
        count: usize,
    },
    /// A runtime lookup table's `log2_rows` exceeds `GadgetPolicy.max_log2_rows`
    /// (RFC-0053 'Table bounds').
    TableTooWide {
        /// Required `⌈log2 T⌉` of the runtime lookup table.
        log2_rows: u8,
        /// Policy ceiling `max_log2_rows`.
        max_log2_rows: u8,
    },
    /// No `key_bits ∈ [min_key_bits, max_log2_rows]` satisfies `Exp_T[2^{key_bits} − 1] == 0`.
    NoQualifyingExpWidth {
        /// Effective temperature that failed width selection.
        temperature: Rational,
    },
    /// Invalid public geometry or failed memory reservation.
    Gadget(GadgetError),
}

impl From<ScoringError> for InternError {
    fn from(err: ScoringError) -> Self {
        Self::Scoring(err)
    }
}

impl From<GadgetError> for InternError {
    fn from(err: GadgetError) -> Self {
        Self::Gadget(err)
    }
}

impl fmt::Display for InternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Scoring(err) => write!(f, "{err}"),
            Self::TooManyTemperatures { count } => {
                write!(f, "batch uses {count} distinct temperatures (maximum 4)")
            }
            Self::TableTooWide {
                log2_rows,
                max_log2_rows,
            } => write!(
                f,
                "runtime lookup table width {log2_rows} exceeds policy max_log2_rows {max_log2_rows}"
            ),
            Self::NoQualifyingExpWidth { temperature } => write!(
                f,
                "no qualifying Exp_T table width for temperature {}/{}",
                temperature.num(),
                temperature.den()
            ),
            Self::Gadget(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for InternError {}

/// Deterministically intern runtime scoring tables onto `ClaimList.tables` in RFC-0017 item 6 order:
///
/// 1. Distinct effective temperatures' narrowest `Exp_T` tables in first-seen output-bearing
///    request order (`T = 1` for greedy `Mixture` requests; skipped for greedy `Exact` requests;
///    at most 4 distinct temperatures per batch).
/// 2. For each output-bearing request in batch order: decoding `Range` tables (`Argmax`'s
///    `Range(min_key_bits)` for greedy, or `InverseCdf`'s `Range` specs for temperature), followed
///    by `OneSidedLog2`'s lookup specs in `Mixture` mode.
///
/// Every runtime table's `⌈log2 T⌉` is checked against `cfg.max_log2_rows` before deduplicating
/// against `base_tables` (`RFC-0053`), and `vocab_exp_scale` is validated on every request that
/// carries an effective temperature regardless of whether `outputs == 0` (`RFC-0017`).
///
/// # Errors
/// [`InternError`] if any parameter or logit endpoint (`|ℓ| < 2^28`) is out of domain, more than
/// 4 effective temperatures appear, a runtime lookup table exceeds `max_log2_rows`, no qualifying
/// `Exp_T` width exists, or memory reservation fails.
pub fn intern_scoring_tables(cfg: InternConfig<'_>) -> Result<InternedScoringTables, InternError> {
    if !(1..=1 << 18).contains(&cfg.vocab)
        || cfg.sigma > 32
        || !(1..=32).contains(&cfg.max_log2_rows)
    {
        return Err(InternError::Gadget(GadgetError::Internal));
    }
    let min_kb = logit_min_key_bits(cfg.logit_lo, cfg.logit_hi)?;
    if let ScoringMode::Mixture { eps_log2 } = cfg.mode
        && eps_log2 > MAX_EPS_LOG2
    {
        return Err(InternError::Scoring(ScoringError::EpsLog2 { eps_log2 }));
    }

    let one_temp = Rational::new(1, 1).ok_or(InternError::Gadget(GadgetError::Internal))?;
    for req in cfg.requests {
        validate_decoding(&req.decoding)?;
        let eff_temp = match (cfg.mode, req.decoding) {
            (ScoringMode::Exact, Decoding::Greedy) => None,
            (ScoringMode::Mixture { .. }, Decoding::Greedy) => Some(one_temp),
            (_, Decoding::Temperature { t }) => Some(t),
        };
        if let Some(temp) = eff_temp {
            let _ = vocab_exp_scale(cfg.logits_exponent, temp)?;
        }
    }

    let mut catalog = reserve_vec(cfg.base_tables.len().saturating_add(32))?;
    catalog.extend_from_slice(cfg.base_tables);

    let mut seen_temps: Vec<Rational> = reserve_vec(MAX_BATCH_TEMPERATURES)?;
    let mut exp_tables: Vec<ExpTableEntry> = reserve_vec(MAX_BATCH_TEMPERATURES)?;

    // Step 1: distinct effective temperatures in first-seen output-bearing request order.
    for req in cfg.requests {
        if req.outputs == 0 {
            continue;
        }
        let eff_temp = match (cfg.mode, req.decoding) {
            (ScoringMode::Exact, Decoding::Greedy) => None,
            (ScoringMode::Mixture { .. }, Decoding::Greedy) => Some(one_temp),
            (_, Decoding::Temperature { t }) => Some(t),
        };
        let Some(temp) = eff_temp else {
            continue;
        };
        if seen_temps.contains(&temp) {
            continue;
        }
        if seen_temps.len() >= MAX_BATCH_TEMPERATURES {
            return Err(InternError::TooManyTemperatures {
                count: seen_temps.len().saturating_add(1),
            });
        }
        seen_temps.push(temp);
        let scale = vocab_exp_scale(cfg.logits_exponent, temp)?;
        let (kind, _, _) = find_qualifying_exp_table(scale, cfg.sigma, min_kb, cfg.max_log2_rows)?
            .ok_or(InternError::NoQualifyingExpWidth { temperature: temp })?;
        let table_id = intern_kind(&mut catalog, kind.clone(), cfg.max_log2_rows)?;
        exp_tables.push(ExpTableEntry {
            temperature: temp,
            table_id,
            kind,
        });
    }

    // Step 2: per output-bearing request in batch order: decoding tables, then OneSidedLog2.
    let bounded = matches!(cfg.mode, ScoringMode::Mixture { .. });
    let has_temp_output = cfg
        .requests
        .iter()
        .any(|r| r.outputs > 0 && matches!(r.decoding, Decoding::Temperature { .. }));
    let cdf_params = if has_temp_output {
        Some(inverse_cdf::Params::derive(cfg.vocab, cfg.sigma, bounded)?)
    } else {
        None
    };
    let log2_params = match cfg.mode {
        ScoringMode::Exact => None,
        ScoringMode::Mixture { eps_log2 } => {
            Some(Log2Params::derive(cfg.vocab, cfg.sigma, eps_log2)?)
        }
    };

    let mut request_tables = reserve_vec(cfg.requests.len())?;
    for req in cfg.requests {
        if req.outputs == 0 {
            request_tables.push(RequestTables {
                decoding_ids: Vec::new(),
                log2_ids: Vec::new(),
            });
            continue;
        }
        let decoding_ids = match req.decoding {
            Decoding::Greedy => {
                let mut ids = reserve_vec(1)?;
                ids.push(intern_kind(
                    &mut catalog,
                    TableKind::Range { bits: min_kb },
                    cfg.max_log2_rows,
                )?);
                ids
            }
            Decoding::Temperature { .. } => {
                let cp = cdf_params.as_ref().ok_or(GadgetError::Internal)?;
                let widths = cp.range_widths();
                let mut ids = reserve_vec(widths.len())?;
                for &bits in widths {
                    ids.push(intern_kind(
                        &mut catalog,
                        TableKind::Range { bits },
                        cfg.max_log2_rows,
                    )?);
                }
                ids
            }
        };
        let log2_ids = if let Some(ref lp) = log2_params {
            let specs = lp.lookup_specs();
            let mut ids = reserve_vec(specs.len())?;
            for spec in specs {
                ids.push(intern_kind(&mut catalog, spec.clone(), cfg.max_log2_rows)?);
            }
            ids
        } else {
            Vec::new()
        };
        request_tables.push(RequestTables {
            decoding_ids,
            log2_ids,
        });
    }

    Ok(InternedScoringTables {
        catalog,
        exp_tables,
        request_tables,
    })
}

// --- Internal helpers ---

fn find_qualifying_exp_table(
    scale: Rational,
    sigma: u8,
    min_kb: u8,
    max_log2_rows: u8,
) -> Result<Option<(TableKind, u8, u64)>, GadgetError> {
    for kb in min_kb..=max_log2_rows {
        let kind = TableKind::Exp {
            key_bits: kb,
            sigma,
            scale,
        };
        if kind.check_parameters().is_err() {
            continue;
        }
        let k0 = (1u64
            .checked_shl(u32::from(kb))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .ok_or(GadgetError::Internal)?;
        if exp_maps_to_zero(k0, sigma, scale) == Ok(true) {
            return Ok(Some((kind, kb, k0)));
        }
    }
    Ok(None)
}

fn intern_kind(
    catalog: &mut Vec<TableKind>,
    kind: TableKind,
    max_log2_rows: u8,
) -> Result<TableId, InternError> {
    let log2_rows = kind
        .check_parameters()
        .map_err(|_| InternError::Gadget(GadgetError::Internal))?;
    if log2_rows > max_log2_rows {
        return Err(InternError::TableTooWide {
            log2_rows,
            max_log2_rows,
        });
    }
    if let Some(pos) = catalog.iter().position(|existing| existing == &kind) {
        return u32::try_from(pos)
            .map(TableId)
            .map_err(|_| InternError::Gadget(GadgetError::Internal));
    }
    let idx =
        u32::try_from(catalog.len()).map_err(|_| InternError::Gadget(GadgetError::Internal))?;
    catalog
        .try_reserve(1)
        .map_err(|_| InternError::Gadget(GadgetError::Memory))?;
    catalog.push(kind);
    Ok(TableId(idx))
}

fn logit_min_key_bits(logit_lo: i32, logit_hi: i32) -> Result<u8, GadgetError> {
    if logit_lo <= -LOGIT_BOUND || logit_lo > logit_hi || logit_hi >= LOGIT_BOUND {
        return Err(GadgetError::Internal);
    }
    let spread_i64 = i64::from(logit_hi)
        .checked_sub(i64::from(logit_lo))
        .ok_or(GadgetError::Internal)?;
    let spread = u64::try_from(spread_i64).map_err(|_| GadgetError::Internal)?;
    let raw = u8::try_from(u64::BITS.saturating_sub(spread.leading_zeros()))
        .map_err(|_| GadgetError::Internal)?;
    Ok(raw.max(1))
}

fn bit_len_u128(value: u128) -> Result<u8, GadgetError> {
    u8::try_from(u128::BITS.saturating_sub(value.leading_zeros()))
        .map_err(|_| GadgetError::Internal)
}

fn log2_ceil(value: u32) -> Result<usize, GadgetError> {
    usize::try_from(u32::BITS.saturating_sub(value.saturating_sub(1).leading_zeros()))
        .map_err(|_| GadgetError::Internal)
}

fn inv_pow2(shift_bits: usize) -> Result<Fp2, GadgetError> {
    let shift = u32::try_from(shift_bits).map_err(|_| GadgetError::Internal)?;
    let scale = 1u64.checked_shl(shift).ok_or(GadgetError::Internal)?;
    Ok(Fp2::from(
        Fp::reduce(scale).inverse().ok_or(GadgetError::Internal)?,
    ))
}

fn derive_top_limb(total: Fp2, low: &[Fp2], inv_top: Fp2) -> Fp2 {
    let radix = Fp2::from(Fp::reduce(1u64 << 16));
    let mut rem = total;
    let mut scale = Fp2::ONE;
    for &limb in low {
        rem -= scale * limb;
        scale *= radix;
    }
    rem * inv_top
}

fn reserve_vec<T>(len: usize) -> Result<Vec<T>, GadgetError> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| GadgetError::Memory)?;
    Ok(out)
}

fn clone_vec<T: Copy>(slice: &[T]) -> Result<Vec<T>, GadgetError> {
    let mut out = reserve_vec(slice.len())?;
    out.extend_from_slice(slice);
    Ok(out)
}

fn append_cmp_specs(out: &mut Vec<TableKind>, m: usize, cmp: &Comparison) {
    out.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, m));
    for &bits in cmp.left_carries() {
        out.push(TableKind::Range { bits });
    }
    out.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, m));
    for &bits in cmp.right_carries() {
        out.push(TableKind::Range { bits });
    }
    out.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, m));
    out.extend(core::iter::repeat_n(TableKind::Range { bits: 1 }, m));
}

fn split_six(flat: &[Fp2], m: usize) -> Result<[&[Fp2]; 6], GadgetError> {
    let (a0, rest) = flat.split_at_checked(m).ok_or(GadgetError::Internal)?;
    let (a1, rest) = rest.split_at_checked(m).ok_or(GadgetError::Internal)?;
    let (a2, rest) = rest.split_at_checked(m).ok_or(GadgetError::Internal)?;
    let (a3, rest) = rest.split_at_checked(m).ok_or(GadgetError::Internal)?;
    let (a4, a5) = rest.split_at_checked(m).ok_or(GadgetError::Internal)?;
    if a5.len() != m {
        return Err(GadgetError::Internal);
    }
    Ok([a0, a1, a2, a3, a4, a5])
}

fn push_log2_cmp_sources(
    out: &mut Vec<Log2SourceClaim>,
    point: &[Fp2],
    arrays: [&[Fp2]; 6],
) -> Result<(), GadgetError> {
    let roles = [
        ComparisonRole::LeftDigit,
        ComparisonRole::LeftCarry,
        ComparisonRole::RightDigit,
        ComparisonRole::RightCarry,
        ComparisonRole::Difference,
        ComparisonRole::Borrow,
    ];
    for (role, slice) in roles.into_iter().zip(arrays) {
        for (limb, &value) in slice.iter().enumerate() {
            out.push(Log2SourceClaim {
                role: Log2Role::Compare(role, limb),
                point: clone_vec(point)?,
                value,
            });
        }
    }
    Ok(())
}

fn squeeze_zero(
    transcript: &mut Transcript,
    prefix: Prefix,
    vars: usize,
) -> Result<Vec<Fp2>, StageVerifyError> {
    let mut out = reserve_vec(vars)?;
    for c in 0..vars {
        out.push(fiat_shamir::challenge(transcript, prefix, Step::Zero(c))?);
    }
    Ok(out)
}
