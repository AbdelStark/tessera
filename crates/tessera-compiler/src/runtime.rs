// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public per-request scoring metadata and runtime table derivation (RFC-0017, RFC-0030,
//! RFC-0053, RFC-0058).

use crate::{CompareColumn, CompileError, Lowering, Selection, analyze};
use tessera_codec::Rational;
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::{ClaimList, GadgetId, IntRange, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind};

/// Maximum distinct temperatures allowed in a single batch (RFC-0017 'Limits').
pub const MAX_BATCH_TEMPERATURES: usize = 4;

const REQUEST: &str = "the greedy request names a positive prompt within the model sequence";
const TEMP_REQUEST: &str =
    "each temperature request names a positive prompt within the model context";
const TEMPERATURE: &str = "a temperature lies in (0, 4] (RFC-0017)";
const BATCH_TEMPERATURES: &str = "a batch uses at most four distinct temperatures (RFC-0017)";
const METADATA: &str = "runtime selection preserves the model's canonical table catalog";
const RANGE: &str = "greedy logits have a public interval strictly within the signed 28-bit bound";
const VOCAB_RANGE: &str =
    "vocabulary logits have a public interval strictly within the signed 28-bit bound";

fn usage(rule: &'static str) -> CompileError {
    CompileError::Usage { rule }
}

/// One maximum source and its virtual lookup column, derived from public data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyCertificate {
    /// Identity after every reserved model node id.
    pub gadget: GadgetId,
    /// Exact physical range table in the complete runtime catalog.
    pub table: TableId,
    /// Positive prompt length; logit time `prompt - 1` predicts its first output.
    pub prompt: u32,
    /// Model time length, including the prompt and every output.
    pub positions: u32,
    /// Unpadded model vocabulary.
    pub vocab: u32,
    /// Inclusive logit interval from public range analysis, never from a witness.
    pub logits: IntRange,
    /// Width of the physical Range table; at most 29 bits.
    pub range_bits: u8,
    /// Padded output-plus-vocabulary lookup variable count.
    pub variables: u32,
}
impl GreedyCertificate {
    /// Both padding axes contribute zero-key lookup requests.
    pub fn entries(self) -> u64 {
        1u64 << self.variables
    }
}

/// Model selection plus the optional implicit Range table, without changing the claim list.
#[derive(Clone, Debug)]
pub struct RuntimePlan {
    selection: Selection,
    certificate: Option<GreedyCertificate>,
}
impl RuntimePlan {
    /// Extend checked model metadata for one request. Empty outputs add no maximum or table.
    ///
    /// # Errors
    /// `Usage` for inconsistent model/request metadata or identity overflow; `Model` for an
    /// unsupported public logit interval or physical table policy; range/table errors propagate.
    pub fn derive(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        prompt: u32,
        ordinal: u32,
    ) -> Result<Self, CompileError> {
        if selection.tables != claims.tables {
            return Err(usage(METADATA));
        }
        let shape = &lowering
            .graph
            .tensors
            .get(lowering.logits.0 as usize)
            .ok_or_else(|| usage(REQUEST))?
            .shape;
        let [positions, vocab] = shape.as_slice() else {
            return Err(usage(REQUEST));
        };
        if prompt == 0 || prompt > *positions {
            return Err(usage(REQUEST));
        }
        let mut selection = selection.clone();
        let certificate = if prompt < *positions {
            let gadget = GadgetId(
                u32::try_from(lowering.graph.nodes.len())
                    .ok()
                    .and_then(|n| n.checked_add(ordinal))
                    .ok_or_else(|| usage(REQUEST))?,
            );
            let logits = analyze(&lowering.graph, &lowering.tables)?
                .of(lowering.logits)
                .and_then(|r| r.bounded())
                .filter(|r| -(1i128 << 28) < r.lo && r.hi < 1i128 << 28)
                .ok_or(CompileError::Model { rule: RANGE })?;
            let span = u32::try_from(logits.hi - logits.lo)
                .map_err(|_| CompileError::Model { rule: RANGE })?;
            let range_bits = (u32::BITS - span.leading_zeros()).max(1) as u8;
            if range_bits > claims.gadget_policy.max_log2_rows {
                return Err(CompileError::Model {
                    rule: "the greedy physical Range width fits GadgetPolicy.max_log2_rows",
                });
            }
            let kind = TableKind::Range { bits: range_bits };
            let table = match selection.tables.iter().find(|t| t.kind == kind) {
                Some(table) => table.id,
                None => {
                    let id = TableId(
                        u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?,
                    );
                    let generated = GeneratedTable::new(id, kind)
                        .map_err(|error| CompileError::ModelTable { table: id, error })?;
                    selection.tables.push(generated.spec().clone());
                    id
                }
            };
            let bits = |n: u32| u32::BITS - n.saturating_sub(1).leading_zeros();
            let variables = bits(*positions - prompt) + bits(*vocab);
            if variables >= u64::BITS {
                return Err(usage(REQUEST));
            }
            Some(GreedyCertificate {
                gadget,
                table,
                prompt,
                positions: *positions,
                vocab: *vocab,
                logits,
                range_bits,
                variables,
            })
        } else {
            None
        };
        Ok(Self {
            selection,
            certificate,
        })
    }
    /// Complete catalog, preserving every model table id, kind and digest.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// A distinct certificate exists exactly when the request has outputs.
    pub const fn certificate(&self) -> Option<GreedyCertificate> {
        self.certificate
    }
}

/// Folds the model's public `logits_exponent` (`e_FinalNorm + e_lm`, RFC-0027) and a decoding
/// temperature `T ∈ (0, 4]` into the exact reduced rational scale `2^{logits_exponent} / T` of
/// `Exp_T` (RFC-0017 'Vocabulary softmax').
///
/// # Errors
/// [`CompileError::Usage`] unless `0 < T ≤ 4`; [`CompileError::Model`] if `2^{logits_exponent} / T`
/// exceeds [`Rational`]'s signed-64 numerator or unsigned-64 denominator.
pub fn vocab_exp_scale(
    logits_exponent: i8,
    temperature: Rational,
) -> Result<Rational, CompileError> {
    let raw_num = temperature.num();
    let raw_den = temperature.den();
    if raw_num <= 0 || (raw_num as u128) > 4 * u128::from(raw_den) {
        return Err(usage(TEMPERATURE));
    }
    let t_num = raw_num as u64;
    let overflow = || CompileError::Model {
        rule: "2^{logits_exponent} / T fits Rational's exact integer bounds",
    };
    if logits_exponent >= 0 {
        let exp = logits_exponent as u32;
        let cancel = t_num.trailing_zeros().min(exp);
        let shift = exp - cancel;
        if shift >= 63 {
            return Err(overflow());
        }
        let num_128 = u128::from(raw_den)
            .checked_shl(shift)
            .ok_or_else(overflow)?;
        let num = i64::try_from(num_128).map_err(|_| overflow())?;
        let den = t_num >> cancel;
        Rational::new(num, den).ok_or_else(overflow)
    } else {
        let neg = (-i32::from(logits_exponent)) as u32;
        let cancel = raw_den.trailing_zeros().min(neg);
        let shift = neg - cancel;
        if shift >= 64 {
            return Err(overflow());
        }
        let num = i64::try_from(raw_den >> cancel).map_err(|_| overflow())?;
        let den_128 = u128::from(t_num).checked_shl(shift).ok_or_else(overflow)?;
        let den = u64::try_from(den_128).map_err(|_| overflow())?;
        Rational::new(num, den).ok_or_else(overflow)
    }
}

/// Canonical committed columns of one [`Op::InverseCdf`](tessera_ir::Op::InverseCdf) row in
/// RFC-0058's allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CdfAdviceColumn {
    /// Bounded mode's sampled indicator `ĥ`, `pad(vocab)` entries, preceding `u_0`.
    SampledIndicator,
    /// `u_0`, the draw's low 16-bit limb.
    DrawLimb,
    /// `Z_j`, a committed low limb of the partition (`j < k − 1`).
    TotalLimb(u8),
    /// `B_j`, a committed low limb of the strict prefix (`j < k − 1`).
    BelowLimb(u8),
    /// A committed normalization or subtraction entry of the lower (`upper = false`) or upper
    /// (`upper = true`) comparison, grouped by array per RFC-0050.
    Comparison {
        /// `false` for `B·2^32 ≤ u·Z`; `true` for `u·Z + ψ ≤ (B + S)·2^32`.
        upper: bool,
        /// Array kind in RFC-0050 order.
        column: CompareColumn,
        /// Limb index `0 .. k + 2`.
        limb: u32,
    },
}

/// Derives RFC-0058's ordered [`CdfAdviceColumn`] schema and ordered `Range { bits }` lookup
/// widths for one `(vocab, sigma, bounded)` configuration.
///
/// # Errors
/// [`CompileError::Model`] if `vocab` (`1..=2^18`), `sigma` (`0..=32`), or comparison boxes leave
/// RFC-0009's bounds.
pub fn cdf_columns_and_requests(
    vocab: u32,
    sigma: u8,
    bounded: bool,
) -> Result<(Vec<CdfAdviceColumn>, Vec<u8>), CompileError> {
    let bad = || CompileError::Model {
        rule: "InverseCdf needs 1 ≤ vocab ≤ 2^18 and σ ≤ 32 within Goldilocks' centered range",
    };
    if !(1..=1 << 18).contains(&vocab) || sigma > 32 {
        return Err(bad());
    }
    let width = (64 - (u64::from(vocab) << sigma).leading_zeros()) as u8;
    let half_field = (Goldilocks::MODULUS - 1) / 2;
    if 1u64 << width.max(32) >= half_field {
        return Err(bad());
    }
    let k = width.div_ceil(16) as usize;
    let top_bits = width - ((k as u8 - 1) * 16);
    let full = (1u64 << 16) - 1;
    let top_box = (1u64 << top_bits) - 1;
    let sum_box = |h: usize| if h + 1 == k { top_box } else { full };
    let below_box = |j: usize| {
        if (2..k + 2).contains(&j) {
            sum_box(j - 2)
        } else {
            0
        }
    };
    let product_box = |j: usize| -> u64 {
        (0..2)
            .flat_map(|i| (0..k).map(move |h| (i, h)))
            .filter(|&(i, h)| i + h == j)
            .map(|(_, h)| full * sum_box(h))
            .sum()
    };
    let lower_left: Vec<u64> = (0..k + 2).map(below_box).collect();
    let lower_right: Vec<u64> = (0..k + 2).map(product_box).collect();
    let upper_left: Vec<u64> = (0..k + 2)
        .map(|j| product_box(j) + u64::from(j == 0))
        .collect();
    let upper_right: Vec<u64> = (0..k + 2)
        .map(|j| below_box(j) + if j == 2 { 1u64 << sigma } else { 0 })
        .collect();

    let mut columns = Vec::new();
    if bounded {
        columns.push(CdfAdviceColumn::SampledIndicator);
    }
    columns.push(CdfAdviceColumn::DrawLimb);
    for j in 0..(k - 1) as u8 {
        columns.push(CdfAdviceColumn::TotalLimb(j));
    }
    for j in 0..(k - 1) as u8 {
        columns.push(CdfAdviceColumn::BelowLimb(j));
    }

    let mut requests = vec![16u8, 16u8];
    for _ in 0..2 {
        requests.extend(core::iter::repeat_n(16u8, k - 1));
        requests.push(top_bits);
    }

    let carry_widths = |bounds: &[u64]| -> Result<Vec<u8>, CompileError> {
        let base = 1u64 << 16;
        let mut honest_carry = 0u64;
        let mut previous_box = 0u64;
        let mut equation_bound = 2 * base - 1;
        let mut widths = Vec::with_capacity(bounds.len());
        for &bound in bounds {
            if bound >= 1 << 40 {
                return Err(bad());
            }
            honest_carry = (bound + honest_carry) / base;
            let w = (64 - honest_carry.leading_zeros()).max(1) as u8;
            if w > 32 {
                return Err(bad());
            }
            let carry_box = (1u64 << w) - 1;
            equation_bound = equation_bound
                .max(bound + previous_box)
                .max(base - 1 + base * carry_box);
            previous_box = carry_box;
            widths.push(w);
        }
        if equation_bound >= 1 << 41 {
            return Err(bad());
        }
        Ok(widths)
    };

    for (upper, (left_b, right_b)) in [
        (false, (&lower_left, &lower_right)),
        (true, (&upper_left, &upper_right)),
    ] {
        let left_carries = carry_widths(left_b)?;
        let right_carries = carry_widths(right_b)?;
        let m = (k + 2) as u32;
        for (column, widths) in [
            (CompareColumn::LeftByte, vec![16u8; k + 2]),
            (CompareColumn::LeftCarry, left_carries),
            (CompareColumn::RightByte, vec![16u8; k + 2]),
            (CompareColumn::RightCarry, right_carries),
            (CompareColumn::Difference, vec![16u8; k + 2]),
            (CompareColumn::Borrow, vec![1u8; k + 2]),
        ] {
            for limb in 0..m {
                columns.push(CdfAdviceColumn::Comparison {
                    upper,
                    column,
                    limb,
                });
            }
            requests.extend(widths);
        }
    }
    Ok((columns, requests))
}

/// Public request geometry for runtime temperature decoding (`Temperature` or `Mixture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemperatureRequestSpec {
    /// Positive prompt length; logit position `prompt − 1` predicts the first output.
    pub prompt: u32,
    /// Total sequence length `prompt + output_len ≤ max_context`.
    pub positions: u32,
    /// Decoding temperature `T ∈ (0, 4]`.
    pub temperature: Rational,
    /// Whether `InverseCdf` runs in bounded mode (committing a separate sampled indicator `ĥ`).
    pub bounded: bool,
}

/// Derived vocabulary `Exp_T` table and row grouping for one distinct batch temperature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemperatureTable {
    /// Decoding temperature `T ∈ (0, 4]`.
    pub temperature: Rational,
    /// Folded rational scale `2^{logits_exponent} / T`.
    pub scale: Rational,
    /// Physical `Exp_T` table id in [`TemperaturePlan::selection`].
    pub table: TableId,
    /// Derived key width `key_bits`, satisfying `Exp_T[2^{key_bits} − 1] = 0`.
    pub key_bits: u8,
    /// Precision `σ` (`claims.quant.exp_sigma`).
    pub sigma: u8,
    /// Indices into the input request slice whose output tokens use this `Exp_T` table, in batch
    /// order.
    pub requests: Vec<usize>,
    /// Total real output tokens across `requests`.
    pub output_tokens: u32,
}

/// Derived certificate metadata for one output-bearing temperature request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemperatureCertificate {
    /// Request index in the batch.
    pub request_index: usize,
    /// Gadget identity assigned after all model nodes.
    pub gadget: GadgetId,
    /// Vocabulary `Exp_T` table id.
    pub exp_table: TableId,
    /// Ordered `Range` table ids for [`Op::InverseCdf`](tessera_ir::Op::InverseCdf)'s requests.
    pub range_tables: Vec<TableId>,
    /// Ordered committed columns of each output row (RFC-0058).
    pub columns: Vec<CdfAdviceColumn>,
    /// Prompt length.
    pub prompt: u32,
    /// Sequence length `prompt + outputs`.
    pub positions: u32,
    /// Unpadded vocabulary size.
    pub vocab: u32,
    /// Public inclusive logit interval.
    pub logits: IntRange,
}

/// Runtime catalog and grouped certificate layout for up to [`MAX_BATCH_TEMPERATURES`] temperatures
/// in `(0, 4]`, leaving `ClaimList`'s wire format untouched.
#[derive(Clone, Debug)]
pub struct TemperaturePlan {
    selection: Selection,
    temperatures: Vec<TemperatureTable>,
    certificates: Vec<TemperatureCertificate>,
}

impl TemperaturePlan {
    /// Derive the runtime `Exp_T` and `Range` tables and RFC-0058 column layouts for `requests`.
    ///
    /// # Errors
    /// [`CompileError::Usage`] if `selection.tables != claims.tables`, any request has invalid
    /// lengths or temperature outside `(0, 4]`, or the batch names more than
    /// [`MAX_BATCH_TEMPERATURES`] distinct temperatures; [`CompileError::Model`] if public logits
    /// exceed `|ℓ| < 2^28` or no `Exp_T` width within `GadgetPolicy.max_log2_rows` reaches a zero
    /// final row `Exp_T[K_0] = 0`.
    pub fn derive(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        requests: &[TemperatureRequestSpec],
        first_ordinal: u32,
    ) -> Result<Self, CompileError> {
        if selection.tables != claims.tables {
            return Err(usage(METADATA));
        }
        let shape = &lowering
            .graph
            .tensors
            .get(lowering.logits.0 as usize)
            .ok_or_else(|| usage(TEMP_REQUEST))?
            .shape;
        let [_lowered_positions, vocab] = shape.as_slice() else {
            return Err(usage(TEMP_REQUEST));
        };
        if *vocab != claims.arch.vocab {
            return Err(usage(METADATA));
        }
        let logits = analyze(&lowering.graph, &lowering.tables)?
            .of(lowering.logits)
            .and_then(|r| r.bounded())
            .filter(|r| -(1i128 << 28) < r.lo && r.hi < 1i128 << 28)
            .ok_or(CompileError::Model { rule: VOCAB_RANGE })?;
        let span = u32::try_from(logits.hi - logits.lo)
            .map_err(|_| CompileError::Model { rule: VOCAB_RANGE })?;
        let min_key_bits = (u32::BITS - span.leading_zeros()).max(1) as u8;
        let max_key_bits = claims.gadget_policy.max_log2_rows.min(32);
        let sigma = claims.quant.exp_sigma;

        let mut distinct: Vec<Rational> = Vec::new();
        for req in requests {
            if req.prompt == 0
                || req.prompt > req.positions
                || req.positions > claims.arch.max_context
            {
                return Err(usage(TEMP_REQUEST));
            }
            let _ = vocab_exp_scale(claims.quant.logits_exponent, req.temperature)?;
            if !distinct.contains(&req.temperature) {
                distinct.push(req.temperature);
            }
        }
        if distinct.len() > MAX_BATCH_TEMPERATURES {
            return Err(usage(BATCH_TEMPERATURES));
        }

        let mut selection = selection.clone();
        let mut intern = |kind: TableKind| -> Result<TableId, CompileError> {
            if let Some(existing) = selection.tables.iter().find(|t| t.kind == kind) {
                return Ok(existing.id);
            }
            let id = TableId(u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?);
            let generated = GeneratedTable::new(id, kind)
                .map_err(|error| CompileError::ModelTable { table: id, error })?;
            selection.tables.push(generated.spec().clone());
            Ok(id)
        };

        let mut temperatures = Vec::new();
        for temp in distinct {
            let active_reqs: Vec<usize> = requests
                .iter()
                .enumerate()
                .filter(|(_, r)| r.temperature == temp && r.prompt < r.positions)
                .map(|(idx, _)| idx)
                .collect();
            if active_reqs.is_empty() {
                continue;
            }
            let scale = vocab_exp_scale(claims.quant.logits_exponent, temp)?;
            let mut chosen: Option<(u8, TableKind)> = None;
            for key_bits in min_key_bits..=max_key_bits {
                let kind = TableKind::Exp {
                    key_bits,
                    sigma,
                    scale,
                };
                let probe_id = TableId(0);
                let generated = GeneratedTable::new(probe_id, kind.clone()).map_err(|error| {
                    CompileError::ModelTable {
                        table: probe_id,
                        error,
                    }
                })?;
                let last_is_zero = generated
                    .row(generated.row_count() - 1)
                    .is_some_and(|(_, value)| value == Goldilocks::ZERO);
                if last_is_zero {
                    chosen = Some((key_bits, kind));
                    break;
                }
            }
            let (key_bits, kind) = chosen.ok_or(CompileError::Model {
                rule: "the vocabulary Exp_T table fits GadgetPolicy.max_log2_rows with Exp_T[K_0] = 0 (RFC-0030)",
            })?;
            let table = intern(kind)?;
            let mut output_tokens = 0u32;
            for &idx in &active_reqs {
                output_tokens = output_tokens
                    .checked_add(requests[idx].positions - requests[idx].prompt)
                    .ok_or_else(|| usage(TEMP_REQUEST))?;
            }
            temperatures.push(TemperatureTable {
                temperature: temp,
                scale,
                table,
                key_bits,
                sigma,
                requests: active_reqs,
                output_tokens,
            });
        }

        let base_node = u32::try_from(lowering.graph.nodes.len())
            .ok()
            .and_then(|n| n.checked_add(first_ordinal))
            .ok_or_else(|| usage(TEMP_REQUEST))?;
        let mut certificates = Vec::new();
        for (req_idx, req) in requests.iter().enumerate() {
            if req.prompt == req.positions {
                continue;
            }
            let ordinal = u32::try_from(certificates.len())
                .ok()
                .and_then(|o| base_node.checked_add(o))
                .ok_or_else(|| usage(TEMP_REQUEST))?;
            let exp_table = temperatures
                .iter()
                .find(|t| t.temperature == req.temperature)
                .map(|t| t.table)
                .ok_or_else(|| usage(METADATA))?;
            let (columns, widths) = cdf_columns_and_requests(*vocab, sigma, req.bounded)?;
            let mut range_tables = Vec::with_capacity(widths.len());
            for bits in widths {
                range_tables.push(intern(TableKind::Range { bits })?);
            }
            certificates.push(TemperatureCertificate {
                request_index: req_idx,
                gadget: GadgetId(ordinal),
                exp_table,
                range_tables,
                columns,
                prompt: req.prompt,
                positions: req.positions,
                vocab: *vocab,
                logits,
            });
        }

        Ok(Self {
            selection,
            temperatures,
            certificates,
        })
    }

    /// Complete runtime table catalog, preserving every model table id, kind and digest.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Distinct active temperature tables and their grouped request indices.
    pub fn temperatures(&self) -> &[TemperatureTable] {
        &self.temperatures
    }

    /// Per-request certificate descriptions in batch order (omitting zero-output requests).
    pub fn certificates(&self) -> &[TemperatureCertificate] {
        &self.certificates
    }
}
