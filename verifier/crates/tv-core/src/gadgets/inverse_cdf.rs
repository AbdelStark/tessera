// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemma 10 and RFC-0058 Lemma 10' `InverseCdf` temperature-sampling
//! final checker: public limb/carry schema, row decomposition and comparison
//! checks, and the three-stage reduction chain over Goldilocks `Fp2`.
//!
//! Written from RFC-0009 ('InverseCdf (temperature sampling)', Lemma 10),
//! RFC-0058 ('InverseCdf columns, lookups and reductions', Lemma 10'),
//! RFC-0050 ('Complete comparison schema') and RFC-0035 ('Padded rows')
//! without reference to prover code (SP-6, RFC-0001).
//!
//! Every stage transitions through a `#[must_use]` token so no Range lookup,
//! committed row advice claim, exponential claim, or indicator obligation can
//! be silently dropped before commitment opening.

use tv_field::{Fp, Fp2};
use tv_hash::transcript::Transcript;

use super::GadgetError;
use super::comparison::{self, Comparison};
use super::decomposition::{self, Decomposition};
use crate::advice::ComparisonRole;
use crate::fiat_shamir::{self, Step};
use crate::selectors::{lt, suffix_kernel};
use crate::sumcheck::{self, Batch, Instance, Prefix, SumcheckError, SumcheckProof, eq};
use crate::validation::GadgetId;

/// Error returned when verifying a component sumcheck stage of a gadget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageVerifyError {
    /// Batched sumcheck round, length, degree, transcript or final-relation error.
    Sumcheck(SumcheckError),
    /// Gadget-local count or cross-instance evaluation consistency failure (`REJ-GADGET`).
    Gadget(GadgetError),
}

impl From<SumcheckError> for StageVerifyError {
    fn from(err: SumcheckError) -> Self {
        Self::Sumcheck(err)
    }
}

impl From<GadgetError> for StageVerifyError {
    fn from(err: GadgetError) -> Self {
        Self::Gadget(err)
    }
}

/// Canonical source or advice role owed by an [`InverseCdf`] verification chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Sampler PRF draw `u_t` on the output-row domain.
    Draw,
    /// Committed low 16-bit limb `u_{t,0}` on the output-row domain.
    DrawLow,
    /// Committed low 16-bit limb `Z_{t,h}` (`0 ≤ h < k − 1`) on the output-row domain.
    PartitionLow(usize),
    /// Committed low 16-bit limb `B_{t,h}` (`0 ≤ h < k − 1`) on the output-row domain.
    PrefixLow(usize),
    /// Lower comparison `2^32 · B ≤ u · Z` witness limb (`0..m`) on the output-row domain.
    LowerCompare(ComparisonRole, usize),
    /// Upper comparison `u · Z + ψ ≤ 2^32 · (B + S)` witness limb (`0..m`) on the output-row domain.
    UpperCompare(ComparisonRole, usize),
    /// Vocabulary exponential table output `e_{t,v}` on the `(vocab, row)` domain.
    Exponentials,
    /// Decoded token indicator `h_{ŷ_t,v}` on the `(vocab, row)` domain (`OneHot` in exact
    /// mode, separate `SampledIndicator` in bounded mode).
    Indicator,
}

/// One retained multilinear evaluation obligation against an input or committed column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceClaim {
    /// Canonical input or advice column role.
    pub role: Role,
    /// Verifier-derived evaluation point, LSB-first.
    pub point: Vec<Fp2>,
    /// Claimed multilinear extension value.
    pub value: Fp2,
}

/// One `Range { bits }` lookup obligation owed to earlier LogUp leaf verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeClaim {
    /// Verifier-derived Range table width in bits (`1..=32`).
    pub bits: u8,
    /// Row evaluation point, LSB-first.
    pub point: Vec<Fp2>,
    /// Claimed key multilinear evaluation.
    pub key: Fp2,
    /// Claimed value multilinear evaluation (`0` for `Range`).
    pub value: Fp2,
}

/// Public schema parameters of `InverseCdf` independent of the output-row count (RFC-0058).
#[derive(Debug)]
pub struct Params {
    vocab: u32,
    sigma: u8,
    bounded: bool,
    w_bits: u8,
    k: usize,
    m: usize,
    top_z_bits: u8,
    u_decomp: Decomposition,
    z_decomp: Decomposition,
    lower_bounds: [Vec<u64>; 2],
    upper_bounds: [Vec<u64>; 2],
    lower_cmp: Comparison,
    upper_cmp: Comparison,
    range_widths: Vec<u8>,
    inv_2_16: Fp2,
    inv_top_z: Fp2,
}

impl Params {
    /// Derive public widths, limb boxes, carry widths and comparison plans from `(vocab, sigma, bounded)`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] unless `1 ≤ vocab ≤ 2^18` and `sigma ≤ 32`;
    /// [`GadgetError::Memory`] if public metadata reservation fails.
    pub fn derive(vocab: u32, sigma: u8, bounded: bool) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab) || sigma > 32 {
            return Err(GadgetError::Internal);
        }
        let max_z = u128::from(vocab)
            .checked_shl(u32::from(sigma))
            .ok_or(GadgetError::Internal)?;
        let w_bits = u8::try_from(u128::BITS.saturating_sub(max_z.leading_zeros()))
            .map_err(|_| GadgetError::Internal)?;
        if !(1..=51).contains(&w_bits) {
            return Err(GadgetError::Internal);
        }
        let k = usize::from(w_bits).div_ceil(16);
        let m = k.checked_add(2).ok_or(GadgetError::Internal)?;
        let low_shift_bits = k
            .checked_sub(1)
            .and_then(|n| n.checked_mul(16))
            .ok_or(GadgetError::Internal)?;
        let top_z_bits = w_bits
            .checked_sub(u8::try_from(low_shift_bits).map_err(|_| GadgetError::Internal)?)
            .ok_or(GadgetError::Internal)?;

        let u_decomp = Decomposition::derive(32, 16)?;
        let z_decomp = Decomposition::derive(w_bits, 16)?;

        let box_u = [65_535u64, 65_535u64];
        let top_z_box = (1u64
            .checked_shl(u32::from(top_z_bits))
            .ok_or(GadgetError::Internal)?)
        .checked_sub(1)
        .ok_or(GadgetError::Internal)?;
        let mut box_z = reserve_vec(k)?;
        box_z.extend(std::iter::repeat_n(65_535, k.saturating_sub(1)));
        box_z.push(top_z_box);

        let mut box_p = reserve_vec(m)?;
        for j in 0..m {
            let mut sum = 0u64;
            for (i, &bu) in box_u.iter().enumerate() {
                for (h, &bz) in box_z.iter().enumerate() {
                    if i.checked_add(h) == Some(j) {
                        let prod = bu.checked_mul(bz).ok_or(GadgetError::Internal)?;
                        sum = sum.checked_add(prod).ok_or(GadgetError::Internal)?;
                    }
                }
            }
            box_p.push(sum);
        }

        let mut lower_left = reserve_vec(m)?;
        lower_left.push(0);
        lower_left.push(0);
        lower_left.extend_from_slice(&box_z);
        let lower_right = clone_vec(&box_p)?;

        let mut upper_left = reserve_vec(m)?;
        for (j, &bp) in box_p.iter().enumerate() {
            let extra = u64::from(j == 0);
            upper_left.push(bp.checked_add(extra).ok_or(GadgetError::Internal)?);
        }
        let max_s = 1u64
            .checked_shl(u32::from(sigma))
            .ok_or(GadgetError::Internal)?;
        let mut upper_right = reserve_vec(m)?;
        for j in 0..m {
            let base = if j >= 2 {
                box_z
                    .get(j.saturating_sub(2))
                    .copied()
                    .ok_or(GadgetError::Internal)?
            } else {
                0
            };
            let s_add = if j == 2 { max_s } else { 0 };
            upper_right.push(base.checked_add(s_add).ok_or(GadgetError::Internal)?);
        }

        let lower_cmp = Comparison::derive(16, &lower_left, &lower_right)?;
        let upper_cmp = Comparison::derive(16, &upper_left, &upper_right)?;

        let total_ranges = 2usize
            .checked_add(k.checked_mul(2).ok_or(GadgetError::Internal)?)
            .and_then(|n| n.checked_add(m.checked_mul(12)?))
            .ok_or(GadgetError::Internal)?;
        let mut range_widths = reserve_vec(total_ranges)?;
        range_widths.push(16);
        range_widths.push(16);
        for _ in 0..2 {
            range_widths.extend(std::iter::repeat_n(16, k.saturating_sub(1)));
            range_widths.push(top_z_bits);
        }
        append_cmp_widths(&mut range_widths, m, &lower_cmp);
        append_cmp_widths(&mut range_widths, m, &upper_cmp);

        let inv_2_16 = Fp2::from(
            Fp::reduce(1u64 << 16)
                .inverse()
                .ok_or(GadgetError::Internal)?,
        );
        let top_shift = u32::try_from(low_shift_bits).map_err(|_| GadgetError::Internal)?;
        let top_scale = 1u64.checked_shl(top_shift).ok_or(GadgetError::Internal)?;
        let inv_top_z = Fp2::from(
            Fp::reduce(top_scale)
                .inverse()
                .ok_or(GadgetError::Internal)?,
        );

        Ok(Self {
            vocab,
            sigma,
            bounded,
            w_bits,
            k,
            m,
            top_z_bits,
            u_decomp,
            z_decomp,
            lower_bounds: [lower_left, lower_right],
            upper_bounds: [upper_left, upper_right],
            lower_cmp,
            upper_cmp,
            range_widths,
            inv_2_16,
            inv_top_z,
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

    /// Whether `h_{ŷ_t}` is a separate bounded-mode `SampledIndicator` (`true`) or exact `OneHot` (`false`).
    pub const fn bounded(&self) -> bool {
        self.bounded
    }

    /// Partition bit width `W = bit_length(V · 2^σ)` (`1..=51`).
    pub const fn w_bits(&self) -> u8 {
        self.w_bits
    }

    /// Number of 16-bit limbs `k = ⌈W / 16⌉` (`1..=4`) of `Z_t` and `B_t`.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of 16-bit comparison limbs `m = k + 2` (`3..=6`).
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Narrowed Range width `W − 16(k − 1)` (`1..=16`) of the virtual top limb `Z_{k−1}` and `B_{k−1}`.
    pub const fn top_z_bits(&self) -> u8 {
        self.top_z_bits
    }

    /// Per-row committed scalar count `1 + 2(k − 1) + 12m` (RFC-0058).
    pub fn committed_row_count(&self) -> usize {
        1usize
            .saturating_add(self.k.saturating_sub(1).saturating_mul(2))
            .saturating_add(self.m.saturating_mul(12))
    }

    /// Prescribed `Range { bits }` widths in RFC-0058 order (`2 + 2k + 12m` entries).
    pub fn range_widths(&self) -> &[u8] {
        &self.range_widths
    }

    /// Inclusive public `(left, right)` coefficient bounds of the lower comparison `2^32 · B ≤ u · Z`.
    pub fn lower_bounds(&self) -> (&[u64], &[u64]) {
        let [left, right] = &self.lower_bounds;
        (left, right)
    }

    /// Verifier-derived `(left, right)` carry widths of the lower comparison.
    pub fn lower_carry_widths(&self) -> (&[u8], &[u8]) {
        (
            self.lower_cmp.left_carries(),
            self.lower_cmp.right_carries(),
        )
    }

    /// Inclusive public `(left, right)` coefficient bounds of the upper comparison `u · Z + ψ ≤ 2^32 · (B + S)`.
    pub fn upper_bounds(&self) -> (&[u64], &[u64]) {
        let [left, right] = &self.upper_bounds;
        (left, right)
    }

    /// Verifier-derived `(left, right)` carry widths of the upper comparison.
    pub fn upper_carry_widths(&self) -> (&[u8], &[u8]) {
        (
            self.upper_cmp.left_carries(),
            self.upper_cmp.right_carries(),
        )
    }
}

/// Verifier-derived `InverseCdf` instance over `rows` output positions and vocabulary `vocab`.
#[derive(Debug)]
pub struct InverseCdf {
    gadget: GadgetId,
    rows: u32,
    vocab_vars: usize,
    row_vars: usize,
    vars: usize,
    params: Params,
}

/// Claimed multilinear evaluations at `row_target` supplied to [`InverseCdf::check_row`].
#[derive(Clone, Copy, Debug)]
pub struct RowValues<'a> {
    /// Verifier-derived row evaluation point (`row_vars` coordinates).
    pub row_target: &'a [Fp2],
    /// Claimed sampler draw extension `u(row_target)`.
    pub u_eval: Fp2,
    /// Claimed partition sum extension `Z(row_target)`.
    pub z_eval: Fp2,
    /// Claimed strict prefix sum extension `B(row_target)`.
    pub b_eval: Fp2,
    /// Claimed selected exponential extension `S(row_target)`.
    pub s_eval: Fp2,
    /// Optional explicit committed row evaluations (`1 + 2(k − 1) + 12m` entries); when
    /// `Some`, checked equal to the corresponding entries of `range_keys`.
    pub committed_evals: Option<&'a [Fp2]>,
    /// Claimed `Range` lookup key evaluations (`2 + 2k + 12m` entries) in RFC-0058 order.
    pub range_keys: &'a [Fp2],
    /// Optional LogUp `Range` lookup value evaluations (`2 + 2k + 12m` entries), each required to be `0`.
    pub range_values: Option<&'a [Fp2]>,
}

/// Row-level `InverseCdf` equations verified at `row_target`, with Stages 1–3 and all
/// lookup/source obligations still pending.
#[must_use = "run Stages 1, 2 and 3 and discharge every returned Range and source obligation"]
#[derive(Debug)]
pub struct RowChecked<'g> {
    site: &'g InverseCdf,
    row_target: Vec<Fp2>,
    z_eval: Fp2,
    b_eval: Fp2,
    s_eval: Fp2,
    p_targets: Vec<Fp2>,
    ranges: Vec<RangeClaim>,
    sources: Vec<SourceClaim>,
}

/// Stage-1 (`stage/1`) batched sumcheck of `InverseCdf`: `m` limb-product relations `P_j`
/// and the padded-row partition zero-check `(1 − ψ_t) · Z_t = 0`.
#[must_use = "verify Stage 1, then run Stages 2 and 3 and discharge all retained obligations"]
#[derive(Debug)]
pub struct Stage1<'g> {
    checked: RowChecked<'g>,
    zero_row: Vec<Fp2>,
    instances: Vec<Instance>,
}

/// Stage-1 verified state, holding `Z(r_1)` and ready to prepare Stage 2.
#[must_use = "run Stage 2 and Stage 3 and discharge all retained Range and source obligations"]
#[derive(Debug)]
pub struct Stage1Done<'g> {
    site: &'g InverseCdf,
    row_target: Vec<Fp2>,
    r1: Vec<Fp2>,
    z_eval: Fp2,
    z_r1: Fp2,
    b_eval: Fp2,
    s_eval: Fp2,
    ranges: Vec<RangeClaim>,
    sources: Vec<SourceClaim>,
}

/// Stage-2 (`stage/2`) batched sumcheck of `InverseCdf` over `(vocab, row)`.
#[must_use = "verify Stage 2, then run Stage 3 and discharge all retained obligations"]
#[derive(Debug)]
pub struct Stage2<'g> {
    done1: Stage1Done<'g>,
    zero_vt: Vec<Fp2>,
    instances: Vec<Instance>,
}

/// Stage-3 (`stage/3`) strict-suffix kernel sumcheck of `InverseCdf` over `(vocab, row)`.
#[must_use = "verify Stage 3 and discharge all retained Range and source obligations"]
#[derive(Debug)]
pub struct Stage3<'g> {
    site: &'g InverseCdf,
    r2: Vec<Fp2>,
    instance: Instance,
    ranges: Vec<RangeClaim>,
    sources: Vec<SourceClaim>,
}

/// Completed three-stage `InverseCdf` reduction, holding every Range and source opening obligation.
#[must_use = "discharge every Range lookup claim and every canonical source/advice opening claim"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    ranges: Vec<RangeClaim>,
    sources: Vec<SourceClaim>,
}

impl Verified {
    /// All `2 + 2k + 12m` Range lookup obligations at `row_target`.
    pub fn range_claims(&self) -> &[RangeClaim] {
        &self.ranges
    }

    /// All committed row advice and input obligations accumulated at `row_target`, `r_1`, `r_2` and `r_3`.
    pub fn source_claims(&self) -> &[SourceClaim] {
        &self.sources
    }
}

impl InverseCdf {
    /// Derive a verifier-owned `InverseCdf` checker from validated public dimensions.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] unless `1 ≤ vocab ≤ 2^18`, `1 ≤ rows ≤ 2^17`, and `sigma ≤ 32`;
    /// [`GadgetError::Memory`] on failed reservation.
    pub fn derive(
        gadget: GadgetId,
        vocab: u32,
        rows: u32,
        sigma: u8,
        bounded: bool,
    ) -> Result<Self, GadgetError> {
        if !(1..=1 << 17).contains(&rows) {
            return Err(GadgetError::Internal);
        }
        let params = Params::derive(vocab, sigma, bounded)?;
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

    /// Public schema parameters (`w_bits`, `k`, `m`, bounds, carry widths, Range widths).
    pub const fn params(&self) -> &Params {
        &self.params
    }

    /// Verify row-level affine decompositions (`u`, `Z`, `B`), Range value zeros, and both
    /// `BigIntCompare` brackets (`2^32 · B ≤ u · Z` and `u · Z + ψ ≤ 2^32 · (B + S)`) at `row_target`.
    ///
    /// `P_j(row_target)` is derived uniquely from the lower comparison's right normalization
    /// chain (`y'_{L,j} + 2^16 · c_{R,j} − c_{R,j−1}`) rather than trusted from the proof, and
    /// becomes the target of Stage 1.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `row_target.len() != row_vars`;
    /// [`GadgetError::Rejected`] on any wrong slice length, non-zero Range value, failed
    /// affine limb key, or failed comparison equation; [`GadgetError::Memory`] on failed reservation.
    pub fn check_row<'g>(&'g self, values: RowValues<'_>) -> Result<RowChecked<'g>, GadgetError> {
        if values.row_target.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        let reject = GadgetError::Rejected {
            gadget: self.gadget,
        };
        let p = &self.params;
        let expected_ranges = p.range_widths.len();
        if values.range_keys.len() != expected_ranges {
            return Err(reject);
        }
        if let Some(rv) = values.range_values
            && (rv.len() != expected_ranges || rv.iter().any(|&v| v != Fp2::ZERO))
        {
            return Err(reject);
        }

        let k = p.k;
        let m = p.m;
        let low_count = k.saturating_sub(1);
        let z_end = 2usize.checked_add(k).ok_or(GadgetError::Internal)?;
        let b_end = z_end.checked_add(k).ok_or(GadgetError::Internal)?;
        let lower_end = b_end
            .checked_add(m.checked_mul(6).ok_or(GadgetError::Internal)?)
            .ok_or(GadgetError::Internal)?;

        let u_keys = values.range_keys.get(..2).ok_or(GadgetError::Internal)?;
        let z_keys = values
            .range_keys
            .get(2..z_end)
            .ok_or(GadgetError::Internal)?;
        let b_keys = values
            .range_keys
            .get(z_end..b_end)
            .ok_or(GadgetError::Internal)?;
        let lower_flat = values
            .range_keys
            .get(b_end..lower_end)
            .ok_or(GadgetError::Internal)?;
        let upper_flat = values
            .range_keys
            .get(lower_end..)
            .ok_or(GadgetError::Internal)?;

        let (u_low, z_low, b_low) = if let Some(committed) = values.committed_evals {
            if committed.len() != p.committed_row_count() {
                return Err(reject);
            }
            let z_c_end = 1usize.checked_add(low_count).ok_or(GadgetError::Internal)?;
            let b_c_end = z_c_end
                .checked_add(low_count)
                .ok_or(GadgetError::Internal)?;
            let cmp_committed = committed.get(b_c_end..).ok_or(GadgetError::Internal)?;
            let cmp_keys = values
                .range_keys
                .get(b_end..)
                .ok_or(GadgetError::Internal)?;
            if cmp_committed != cmp_keys {
                return Err(reject);
            }
            (
                committed.get(..1).ok_or(GadgetError::Internal)?,
                committed.get(1..z_c_end).ok_or(GadgetError::Internal)?,
                committed
                    .get(z_c_end..b_c_end)
                    .ok_or(GadgetError::Internal)?,
            )
        } else {
            (
                u_keys.get(..1).ok_or(GadgetError::Internal)?,
                z_keys.get(..low_count).ok_or(GadgetError::Internal)?,
                b_keys.get(..low_count).ok_or(GadgetError::Internal)?,
            )
        };

        let _u_checked = p.u_decomp.check(
            self.gadget,
            decomposition::Values {
                point: values.row_target,
                value: values.u_eval,
                low: u_low,
                keys: u_keys,
            },
        )?;
        let _z_checked = p.z_decomp.check(
            self.gadget,
            decomposition::Values {
                point: values.row_target,
                value: values.z_eval,
                low: z_low,
                keys: z_keys,
            },
        )?;
        let _b_checked = p.z_decomp.check(
            self.gadget,
            decomposition::Values {
                point: values.row_target,
                value: values.b_eval,
                low: b_low,
                keys: b_keys,
            },
        )?;

        let lower_arrays = split_six(lower_flat, m)?;
        let upper_arrays = split_six(upper_flat, m)?;
        let [_, _, lower_rd, lower_rc, _, _] = lower_arrays;

        // Derive P_j(row_target) from lower_cmp's right normalization chain:
        // P_j + c_{R,j-1} = y'_{L,j} + 2^16 * c_{R,j}.
        let radix = Fp2::from(Fp::reduce(1u64 << 16));
        let mut p_targets = reserve_vec(m)?;
        let mut prev_rc = Fp2::ZERO;
        for j in 0..m {
            let rd = lower_rd.get(j).copied().ok_or(GadgetError::Internal)?;
            let rc = lower_rc.get(j).copied().ok_or(GadgetError::Internal)?;
            p_targets.push(rd + radix * rc - prev_rc);
            prev_rc = rc;
        }

        let mut lower_left = reserve_vec(m)?;
        lower_left.push(Fp2::ZERO);
        lower_left.push(Fp2::ZERO);
        lower_left.extend_from_slice(b_keys);
        let _lower_ok = p.lower_cmp.check(
            self.gadget,
            comparison::Values {
                point: values.row_target,
                left: &lower_left,
                right: &p_targets,
                arrays: lower_arrays,
            },
        )?;

        let psi_rt = lt(values.row_target, u64::from(self.rows)).ok_or(GadgetError::Internal)?;
        let mut upper_left = reserve_vec(m)?;
        for (j, &pt) in p_targets.iter().enumerate() {
            upper_left.push(if j == 0 { pt + psi_rt } else { pt });
        }
        let mut upper_right = reserve_vec(m)?;
        for j in 0..m {
            let base = if j >= 2 {
                b_keys
                    .get(j.saturating_sub(2))
                    .copied()
                    .ok_or(GadgetError::Internal)?
            } else {
                Fp2::ZERO
            };
            upper_right.push(if j == 2 { base + values.s_eval } else { base });
        }
        let _upper_ok = p.upper_cmp.check(
            self.gadget,
            comparison::Values {
                point: values.row_target,
                left: &upper_left,
                right: &upper_right,
                arrays: upper_arrays,
            },
        )?;

        let row_target = clone_vec(values.row_target)?;
        let mut ranges = reserve_vec(expected_ranges)?;
        for (&bits, &key) in p.range_widths.iter().zip(values.range_keys) {
            ranges.push(RangeClaim {
                bits,
                point: clone_vec(&row_target)?,
                key,
                value: Fp2::ZERO,
            });
        }

        let source_cap = 1usize
            .checked_add(p.committed_row_count())
            .and_then(|n| n.checked_add(1usize.saturating_add(low_count).saturating_add(4)))
            .ok_or(GadgetError::Internal)?;
        let mut sources = reserve_vec(source_cap)?;
        sources.push(SourceClaim {
            role: Role::Draw,
            point: clone_vec(&row_target)?,
            value: values.u_eval,
        });
        sources.push(SourceClaim {
            role: Role::DrawLow,
            point: clone_vec(&row_target)?,
            value: *u_low.first().ok_or(GadgetError::Internal)?,
        });
        for (h, &v) in z_low.iter().enumerate() {
            sources.push(SourceClaim {
                role: Role::PartitionLow(h),
                point: clone_vec(&row_target)?,
                value: v,
            });
        }
        for (h, &v) in b_low.iter().enumerate() {
            sources.push(SourceClaim {
                role: Role::PrefixLow(h),
                point: clone_vec(&row_target)?,
                value: v,
            });
        }
        push_cmp_sources(&mut sources, &row_target, lower_arrays, Role::LowerCompare)?;
        push_cmp_sources(&mut sources, &row_target, upper_arrays, Role::UpperCompare)?;

        Ok(RowChecked {
            site: self,
            row_target,
            z_eval: values.z_eval,
            b_eval: values.b_eval,
            s_eval: values.s_eval,
            p_targets,
            ranges,
            sources,
        })
    }
}

impl<'g> RowChecked<'g> {
    /// Derived limb-product targets `P_0(row_target), …, P_{m−1}(row_target)` for Stage 1.
    pub fn p_targets(&self) -> &[Fp2] {
        &self.p_targets
    }

    /// Squeeze `stage/1/zero/{0..row_vars}` from `transcript` and build Stage 1.
    ///
    /// # Errors
    /// [`StageVerifyError`] on transcript or reservation failure.
    pub fn squeeze_stage1(
        self,
        transcript: &mut Transcript,
    ) -> Result<Stage1<'g>, StageVerifyError> {
        let zero_row = squeeze_zero(transcript, Prefix::Stage(1), self.site.row_vars)?;
        Ok(self.prepare_stage1(zero_row)?)
    }

    /// Build Stage 1 from an already squeezed stage-1 row zero-check point.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `zero_row.len() != row_vars`;
    /// [`GadgetError::Memory`] on failed reservation.
    pub fn prepare_stage1(self, zero_row: Vec<Fp2>) -> Result<Stage1<'g>, GadgetError> {
        if zero_row.len() != self.site.row_vars {
            return Err(GadgetError::Internal);
        }
        let m = self.site.params.m;
        let per_p_evals = self
            .site
            .params
            .k
            .checked_add(2)
            .ok_or(GadgetError::Internal)?;
        let count = m.checked_add(1).ok_or(GadgetError::Internal)?;
        let mut instances = reserve_vec(count)?;
        for &claim in &self.p_targets {
            instances.push(Instance {
                claim,
                vars: self.site.row_vars,
                evals: per_p_evals,
            });
        }
        instances.push(Instance {
            claim: Fp2::ZERO,
            vars: self.site.row_vars,
            evals: 1,
        });
        Ok(Stage1 {
            checked: self,
            zero_row,
            instances,
        })
    }
}

impl<'g> Stage1<'g> {
    /// Prescribed sumcheck degree (`3`) of Stage 1.
    pub const fn degree(&self) -> usize {
        3
    }

    /// Stage-1 instances in schedule order (`P_0 .. P_{m−1}`, then `(1 − ψ_t) · Z_t = 0`).
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
        let m = p.m;
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        if index < m {
            let expected = p.k.checked_add(2).ok_or(GadgetError::Internal)?;
            if values.len() != expected {
                return Err(reject);
            }
            let u_at = *values.first().ok_or(GadgetError::Internal)?;
            let u0_at = *values.get(1).ok_or(GadgetError::Internal)?;
            let z_at = *values.get(2).ok_or(GadgetError::Internal)?;
            let z_low = values.get(3..).ok_or(GadgetError::Internal)?;
            let u_limbs = [u0_at, (u_at - u0_at) * p.inv_2_16];

            let radix = Fp2::from(Fp::reduce(1u64 << 16));
            let mut rem = z_at;
            let mut scale = Fp2::ONE;
            for &zh in z_low {
                rem -= scale * zh;
                scale *= radix;
            }
            let z_top = rem * p.inv_top_z;

            let mut pj = Fp2::ZERO;
            for (i, &ui) in u_limbs.iter().enumerate() {
                for h in 0..p.k {
                    if i.checked_add(h) == Some(index) {
                        let zh = if h < z_low.len() {
                            *z_low.get(h).ok_or(GadgetError::Internal)?
                        } else {
                            z_top
                        };
                        pj += ui * zh;
                    }
                }
            }
            let eq_rt = eq(&self.checked.row_target, point).ok_or(GadgetError::Internal)?;
            Ok(eq_rt * pj)
        } else if index == m {
            if values.len() != 1 {
                return Err(reject);
            }
            let z_at = *values.first().ok_or(GadgetError::Internal)?;
            let psi = lt(point, u64::from(site.rows)).ok_or(GadgetError::Internal)?;
            let eq_zr = eq(&self.zero_row, point).ok_or(GadgetError::Internal)?;
            Ok(eq_zr * (Fp2::ONE - psi) * z_at)
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
    ) -> Result<(Batch, Stage1Done<'g>), StageVerifyError> {
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

    /// Enforce equality of repeated `[u, u_0, Z, Z_0..Z_{k−2}]` evaluations at `point`
    /// and advance to Stage 2.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != row_vars`;
    /// [`GadgetError::Rejected`] when `final_evals` has the wrong length or disagrees across instances.
    pub fn finish(
        self,
        point: Vec<Fp2>,
        final_evals: &[Fp2],
    ) -> Result<Stage1Done<'g>, GadgetError> {
        let site = self.checked.site;
        if point.len() != site.row_vars {
            return Err(GadgetError::Internal);
        }
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        let p = &site.params;
        let stride = p.k.checked_add(2).ok_or(GadgetError::Internal)?;
        let total =
            p.m.checked_mul(stride)
                .and_then(|n| n.checked_add(1))
                .ok_or(GadgetError::Internal)?;
        if final_evals.len() != total {
            return Err(reject);
        }
        let head = final_evals.get(..stride).ok_or(GadgetError::Internal)?;
        for j in 1..p.m {
            let start = j.checked_mul(stride).ok_or(GadgetError::Internal)?;
            let end = start.checked_add(stride).ok_or(GadgetError::Internal)?;
            if final_evals.get(start..end) != Some(head) {
                return Err(reject);
            }
        }
        let z_pad = *final_evals.last().ok_or(GadgetError::Internal)?;
        let u_r1 = *head.first().ok_or(GadgetError::Internal)?;
        let u0_r1 = *head.get(1).ok_or(GadgetError::Internal)?;
        let z_r1 = *head.get(2).ok_or(GadgetError::Internal)?;
        if z_pad != z_r1 {
            return Err(reject);
        }

        let mut sources = self.checked.sources;
        sources.push(SourceClaim {
            role: Role::Draw,
            point: clone_vec(&point)?,
            value: u_r1,
        });
        sources.push(SourceClaim {
            role: Role::DrawLow,
            point: clone_vec(&point)?,
            value: u0_r1,
        });
        let z_low = head.get(3..).ok_or(GadgetError::Internal)?;
        for (h, &zh) in z_low.iter().enumerate() {
            sources.push(SourceClaim {
                role: Role::PartitionLow(h),
                point: clone_vec(&point)?,
                value: zh,
            });
        }

        Ok(Stage1Done {
            site,
            row_target: self.checked.row_target,
            r1: point,
            z_eval: self.checked.z_eval,
            z_r1,
            b_eval: self.checked.b_eval,
            s_eval: self.checked.s_eval,
            ranges: self.checked.ranges,
            sources,
        })
    }
}

impl<'g> Stage1Done<'g> {
    /// Squeeze `stage/2/zero/{0..vars}` when `bounded == true` (nothing when `bounded == false`)
    /// and build Stage 2.
    ///
    /// # Errors
    /// [`StageVerifyError`] on transcript or reservation failure.
    pub fn squeeze_stage2(
        self,
        transcript: &mut Transcript,
    ) -> Result<Stage2<'g>, StageVerifyError> {
        let zero_vt = if self.site.params.bounded {
            squeeze_zero(transcript, Prefix::Stage(2), self.site.vars)?
        } else {
            Vec::new()
        };
        Ok(self.prepare_stage2(zero_vt)?)
    }

    /// Build Stage 2 from an already squeezed `(vocab, row)` zero-check point (empty when `!bounded`).
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `zero_vt` does not match `bounded`;
    /// [`GadgetError::Memory`] on failed reservation.
    pub fn prepare_stage2(self, zero_vt: Vec<Fp2>) -> Result<Stage2<'g>, GadgetError> {
        let bounded = self.site.params.bounded;
        let expected_zero = if bounded { self.site.vars } else { 0 };
        if zero_vt.len() != expected_zero {
            return Err(GadgetError::Internal);
        }
        let vars = self.site.vars;
        let count = if bounded { 7 } else { 4 };
        let mut instances = reserve_vec(count)?;
        instances.push(Instance {
            claim: self.z_eval,
            vars,
            evals: 1,
        });
        instances.push(Instance {
            claim: self.z_r1,
            vars,
            evals: 1,
        });
        instances.push(Instance {
            claim: self.b_eval,
            vars,
            evals: 2,
        });
        instances.push(Instance {
            claim: self.s_eval,
            vars,
            evals: 2,
        });
        if bounded {
            let z_row = zero_vt
                .get(self.site.vocab_vars..)
                .ok_or(GadgetError::Internal)?;
            let psi_zrow = lt(z_row, u64::from(self.site.rows)).ok_or(GadgetError::Internal)?;
            instances.push(Instance {
                claim: Fp2::ZERO,
                vars,
                evals: 1,
            });
            instances.push(Instance {
                claim: Fp2::ZERO,
                vars,
                evals: 1,
            });
            instances.push(Instance {
                claim: psi_zrow,
                vars,
                evals: 1,
            });
        }
        Ok(Stage2 {
            done1: self,
            zero_vt,
            instances,
        })
    }
}

impl<'g> Stage2<'g> {
    /// Prescribed sumcheck degree (`3`) of Stage 2.
    pub const fn degree(&self) -> usize {
        3
    }

    /// Stage-2 instances in schedule order.
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Evaluate Stage-2 instance `index` at `point` from its slice of `final_evals`.
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
                let ev = *values.first().ok_or(GadgetError::Internal)?;
                let lv = *values.get(1).ok_or(GadgetError::Internal)?;
                let eq_rt = eq(&self.done1.row_target, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_rt * ev * lv)
            }
            3 => {
                if values.len() != 2 {
                    return Err(reject);
                }
                let hv = *values.first().ok_or(GadgetError::Internal)?;
                let ev = *values.get(1).ok_or(GadgetError::Internal)?;
                let eq_rt = eq(&self.done1.row_target, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_rt * hv * ev)
            }
            4 if site.params.bounded => {
                if values.len() != 1 {
                    return Err(reject);
                }
                let hv = *values.first().ok_or(GadgetError::Internal)?;
                let eq_vt = eq(&self.zero_vt, point).ok_or(GadgetError::Internal)?;
                Ok(eq_vt * hv * (Fp2::ONE - hv))
            }
            5 if site.params.bounded => {
                if values.len() != 1 {
                    return Err(reject);
                }
                let hv = *values.first().ok_or(GadgetError::Internal)?;
                let eq_vt = eq(&self.zero_vt, point).ok_or(GadgetError::Internal)?;
                Ok(eq_vt * (Fp2::ONE - cv) * hv)
            }
            6 if site.params.bounded => {
                if values.len() != 1 {
                    return Err(reject);
                }
                let hv = *values.first().ok_or(GadgetError::Internal)?;
                let z_row = self
                    .zero_vt
                    .get(site.vocab_vars..)
                    .ok_or(GadgetError::Internal)?;
                let eq_zr = eq(z_row, row_pt).ok_or(GadgetError::Internal)?;
                Ok(eq_zr * hv)
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
    ) -> Result<(Batch, Stage3<'g>), StageVerifyError> {
        let batch = sumcheck::verify_batch(
            transcript,
            Prefix::Stage(2),
            self.degree(),
            &self.instances,
            proof,
            |i, pt, vals| self.evaluate(i, pt, vals).ok(),
        )?;
        let stage3 = self.finish(clone_vec(&batch.point)?, &proof.final_evals)?;
        Ok((batch, stage3))
    }

    /// Enforce equality of repeated `e` and `h_{ŷ}` evaluations at `point` and build Stage 3.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != vars`;
    /// [`GadgetError::Rejected`] when `final_evals` has the wrong length or disagrees across instances.
    pub fn finish(self, point: Vec<Fp2>, final_evals: &[Fp2]) -> Result<Stage3<'g>, GadgetError> {
        let site = self.done1.site;
        if point.len() != site.vars {
            return Err(GadgetError::Internal);
        }
        let reject = GadgetError::Rejected {
            gadget: site.gadget,
        };
        let expected_len = if site.params.bounded { 9 } else { 6 };
        if final_evals.len() != expected_len {
            return Err(reject);
        }
        let e0 = *final_evals.first().ok_or(GadgetError::Internal)?;
        let e1 = *final_evals.get(1).ok_or(GadgetError::Internal)?;
        let e2 = *final_evals.get(2).ok_or(GadgetError::Internal)?;
        let l_r2 = *final_evals.get(3).ok_or(GadgetError::Internal)?;
        let h0 = *final_evals.get(4).ok_or(GadgetError::Internal)?;
        let e3 = *final_evals.get(5).ok_or(GadgetError::Internal)?;
        if e0 != e1 || e0 != e2 || e0 != e3 {
            return Err(reject);
        }
        if site.params.bounded {
            let h1 = *final_evals.get(6).ok_or(GadgetError::Internal)?;
            let h2 = *final_evals.get(7).ok_or(GadgetError::Internal)?;
            let h3 = *final_evals.get(8).ok_or(GadgetError::Internal)?;
            if h0 != h1 || h0 != h2 || h0 != h3 {
                return Err(reject);
            }
        }

        let mut sources = self.done1.sources;
        sources.push(SourceClaim {
            role: Role::Exponentials,
            point: clone_vec(&point)?,
            value: e0,
        });
        sources.push(SourceClaim {
            role: Role::Indicator,
            point: clone_vec(&point)?,
            value: h0,
        });

        Ok(Stage3 {
            site,
            r2: point,
            instance: Instance {
                claim: l_r2,
                vars: site.vars,
                evals: 1,
            },
            ranges: self.done1.ranges,
            sources,
        })
    }
}

impl<'g> Stage3<'g> {
    /// Prescribed sumcheck degree (`2`) of Stage 3.
    pub const fn degree(&self) -> usize {
        2
    }

    /// The single Stage-3 strict-suffix instance.
    pub const fn instance(&self) -> Instance {
        self.instance
    }

    /// Evaluate Stage 3's strict-suffix relation `eq(r_{2,row}, row_pt) · suffix_kernel(r_{2,v}, v_pt) · h_{ŷ}` at `point`.
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != vars`;
    /// [`GadgetError::Rejected`] when `values.len() != 1`.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Fp2, GadgetError> {
        if point.len() != self.site.vars {
            return Err(GadgetError::Internal);
        }
        if values.len() != 1 {
            return Err(GadgetError::Rejected {
                gadget: self.site.gadget,
            });
        }
        let (v_pt, row_pt) = point
            .split_at_checked(self.site.vocab_vars)
            .ok_or(GadgetError::Internal)?;
        let (r2_v, r2_row) = self
            .r2
            .split_at_checked(self.site.vocab_vars)
            .ok_or(GadgetError::Internal)?;
        let kernel = suffix_kernel(r2_v, v_pt).ok_or(GadgetError::Internal)?;
        let hv = *values.first().ok_or(GadgetError::Internal)?;
        let eq_r2 = eq(r2_row, row_pt).ok_or(GadgetError::Internal)?;
        Ok(eq_r2 * kernel * hv)
    }

    /// Run `sumcheck::verify_batch` under `stage/3` and finalize all retained obligations.
    ///
    /// # Errors
    /// [`StageVerifyError`] on any sumcheck or evaluation-count failure.
    pub fn verify(
        self,
        transcript: &mut Transcript,
        proof: &SumcheckProof,
    ) -> Result<(Batch, Verified), StageVerifyError> {
        let instances = [self.instance];
        let batch = sumcheck::verify_batch(
            transcript,
            Prefix::Stage(3),
            self.degree(),
            &instances,
            proof,
            |i, pt, vals| {
                if i == 0 {
                    self.evaluate(pt, vals).ok()
                } else {
                    None
                }
            },
        )?;
        let verified = self.finish(clone_vec(&batch.point)?, &proof.final_evals)?;
        Ok((batch, verified))
    }

    /// Record the final `Indicator` opening obligation at `r_3` and return [`Verified`].
    ///
    /// # Errors
    /// [`GadgetError::Internal`] when `point.len() != vars`;
    /// [`GadgetError::Rejected`] when `final_evals.len() != 1`.
    pub fn finish(mut self, point: Vec<Fp2>, final_evals: &[Fp2]) -> Result<Verified, GadgetError> {
        if point.len() != self.site.vars {
            return Err(GadgetError::Internal);
        }
        if final_evals.len() != 1 {
            return Err(GadgetError::Rejected {
                gadget: self.site.gadget,
            });
        }
        let hv = *final_evals.first().ok_or(GadgetError::Internal)?;
        self.sources.push(SourceClaim {
            role: Role::Indicator,
            point,
            value: hv,
        });
        Ok(Verified {
            ranges: self.ranges,
            sources: self.sources,
        })
    }
}

// --- Internal helpers ---

fn log2_ceil(value: u32) -> Result<usize, GadgetError> {
    usize::try_from(u32::BITS.saturating_sub(value.saturating_sub(1).leading_zeros()))
        .map_err(|_| GadgetError::Internal)
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

fn append_cmp_widths(out: &mut Vec<u8>, m: usize, cmp: &Comparison) {
    out.extend(core::iter::repeat_n(16, m));
    out.extend_from_slice(cmp.left_carries());
    out.extend(core::iter::repeat_n(16, m));
    out.extend_from_slice(cmp.right_carries());
    out.extend(core::iter::repeat_n(16, m));
    out.extend(core::iter::repeat_n(1, m));
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

fn push_cmp_sources(
    out: &mut Vec<SourceClaim>,
    point: &[Fp2],
    arrays: [&[Fp2]; 6],
    make_role: fn(ComparisonRole, usize) -> Role,
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
            out.push(SourceClaim {
                role: make_role(role, limb),
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
