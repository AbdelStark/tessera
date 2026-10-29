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

/// One active request's exact-greedy lookup key column `M[τ] − L[τ, v]` (RFC-0053, RFC-0071).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyKeyColumn {
    /// Request ordinal `r` in `statement.batch.requests`.
    pub request_index: u32,
    /// Positive prompt length `prompt_len_r ≥ 1`; logit position `prompt − 1` predicts the first
    /// output.
    pub prompt: u32,
    /// Request sequence length `n_r = prompt_len_r + output_len_r`, with `output_len_r > 0`.
    pub positions: u32,
    /// Slot base offset `r · S` (`0` when `R = 1`).
    pub slot_offset: u32,
    /// Padded output-plus-vocabulary lookup variable count `⌈log2 output_len_r⌉ + ⌈log2 vocab⌉`.
    pub variables: u32,
}
impl GreedyKeyColumn {
    /// Both padding axes contribute zero-key lookup requests.
    pub fn entries(self) -> u64 {
        1u64 << self.variables
    }

    /// Output token count `n_r − prompt_len_r > 0`.
    pub fn output_len(self) -> u32 {
        self.positions - self.prompt
    }
}

/// Model selection plus the optional implicit Range table, without changing the claim list.
#[derive(Clone, Debug)]
pub struct RuntimePlan {
    selection: Selection,
    certificate: Option<GreedyCertificate>,
    key_columns: Vec<GreedyKeyColumn>,
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
        if selection.tables != claims.tables || lowering.graph.slots.is_some() {
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
        let mut key_columns = Vec::new();
        let certificate = if prompt < *positions {
            let gadget = GadgetId(
                u32::try_from(lowering.graph.nodes.len())
                    .ok()
                    .and_then(|n| n.checked_add(ordinal))
                    .ok_or_else(|| usage(REQUEST))?,
            );
            let (logits, range_bits, table) =
                intern_greedy_range_table(lowering, &mut selection, claims)?;
            let bits = |n: u32| u32::BITS - n.saturating_sub(1).leading_zeros();
            let variables = bits(*positions - prompt) + bits(*vocab);
            if variables >= u64::BITS {
                return Err(usage(REQUEST));
            }
            key_columns.push(GreedyKeyColumn {
                request_index: 0,
                prompt,
                positions: *positions,
                slot_offset: 0,
                variables,
            });
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
            key_columns,
        })
    }

    /// Derive exact-greedy runtime metadata for a batch of `requests` (`(prompt_len, output_len)`
    /// in slot order; RFC-0053, RFC-0071 'The greedy certificate and scoring').
    ///
    /// With one request (`R = 1`), `lowering.graph.slots` must be `None` and this matches
    /// [`Self::derive`] at ordinal `0`. With `R ≥ 2`, `lowering.graph.slots` must match the
    /// requests' lengths; one shared [`GreedyCertificate`] (`M` on `[N]` at ordinal `0` and one
    /// physical `Range` table) is emitted iff at least one request has `output_len > 0`, together
    /// with one [`GreedyKeyColumn`] per output-bearing request in batch order.
    ///
    /// # Errors
    /// `Usage` for empty `requests`, `prompt_len == 0`, length mismatch against `lowering` or
    /// `claims.arch.max_context`, or table/identity overflow; `Model` when logits exceed the
    /// signed 28-bit greedy bound or `GadgetPolicy.max_log2_rows`.
    pub fn derive_batch(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        requests: &[(u32, u32)],
    ) -> Result<Self, CompileError> {
        if requests.is_empty() {
            return Err(usage(REQUEST));
        }
        if selection.tables != claims.tables {
            return Err(usage(METADATA));
        }
        let shape = &lowering
            .graph
            .tensors
            .get(lowering.logits.0 as usize)
            .ok_or_else(|| usage(REQUEST))?
            .shape;
        let [token_len, vocab] = shape.as_slice() else {
            return Err(usage(REQUEST));
        };
        if let &[(prompt, output_len)] = requests {
            let n = prompt
                .checked_add(output_len)
                .ok_or_else(|| usage(REQUEST))?;
            if n != *token_len || n > claims.arch.max_context {
                return Err(usage(REQUEST));
            }
            return Self::derive(lowering, selection, claims, prompt, 0);
        }
        let Some(slots) = &lowering.graph.slots else {
            return Err(usage(METADATA));
        };
        if slots.lengths.len() != requests.len() || slots.token_len() != *token_len {
            return Err(usage(METADATA));
        }
        let bits = |n: u32| u32::BITS - n.saturating_sub(1).leading_zeros();
        let vocab_bits = bits(*vocab);
        let mut key_columns = Vec::new();
        let mut total_entries = 0u64;
        for (idx, &(prompt, output_len)) in requests.iter().enumerate() {
            let n_r = prompt
                .checked_add(output_len)
                .ok_or_else(|| usage(REQUEST))?;
            if prompt == 0 || n_r > claims.arch.max_context || slots.lengths[idx] != n_r {
                return Err(usage(REQUEST));
            }
            if output_len > 0 {
                let request_index = u32::try_from(idx).map_err(|_| usage(REQUEST))?;
                let slot_offset = request_index
                    .checked_mul(slots.slot_len)
                    .ok_or_else(|| usage(REQUEST))?;
                let variables = bits(output_len) + vocab_bits;
                if variables >= u64::BITS {
                    return Err(usage(REQUEST));
                }
                let col = GreedyKeyColumn {
                    request_index,
                    prompt,
                    positions: n_r,
                    slot_offset,
                    variables,
                };
                total_entries = total_entries
                    .checked_add(col.entries())
                    .ok_or_else(|| usage(REQUEST))?;
                key_columns.push(col);
            }
        }
        let mut selection = selection.clone();
        let certificate = if let Some(&first) = key_columns.first() {
            let gadget =
                GadgetId(u32::try_from(lowering.graph.nodes.len()).map_err(|_| usage(REQUEST))?);
            let (logits, range_bits, table) =
                intern_greedy_range_table(lowering, &mut selection, claims)?;
            Some(GreedyCertificate {
                gadget,
                table,
                prompt: first.prompt,
                positions: *token_len,
                vocab: *vocab,
                logits,
                range_bits,
                variables: first.variables,
            })
        } else {
            None
        };
        Ok(Self {
            selection,
            certificate,
            key_columns,
        })
    }

    /// Complete catalog, preserving every model table id, kind and digest.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// A certificate exists iff at least one request in the batch has outputs.
    pub const fn certificate(&self) -> Option<GreedyCertificate> {
        self.certificate
    }
    /// Logical exact-greedy key columns, one per output-bearing request in batch order.
    pub fn key_columns(&self) -> &[GreedyKeyColumn] {
        &self.key_columns
    }
}

fn intern_greedy_range_table(
    lowering: &Lowering,
    selection: &mut Selection,
    claims: &ClaimList,
) -> Result<(IntRange, u8, TableId), CompileError> {
    let logits = analyze(&lowering.graph, &lowering.tables)?
        .of(lowering.logits)
        .and_then(|r| r.bounded())
        .filter(|r| -(1i128 << 28) < r.lo && r.hi < 1i128 << 28)
        .ok_or(CompileError::Model { rule: RANGE })?;
    let span =
        u32::try_from(logits.hi - logits.lo).map_err(|_| CompileError::Model { rule: RANGE })?;
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
            let id = TableId(u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?);
            let generated = GeneratedTable::new(id, kind)
                .map_err(|error| CompileError::ModelTable { table: id, error })?;
            selection.tables.push(generated.spec().clone());
            id
        }
    };
    Ok((logits, range_bits, table))
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
        let max_log2_rows = claims.gadget_policy.max_log2_rows;
        let max_key_bits = max_log2_rows.min(32);
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
            if req.prompt < req.positions && !distinct.contains(&req.temperature) {
                distinct.push(req.temperature);
            }
        }
        if distinct.len() > MAX_BATCH_TEMPERATURES {
            return Err(usage(BATCH_TEMPERATURES));
        }

        let mut selection = selection.clone();
        let mut intern = |kind: TableKind| -> Result<TableId, CompileError> {
            if let Some(existing) = selection.tables.iter().find(|t| t.kind == kind) {
                if existing.log2_rows > max_log2_rows {
                    return Err(CompileError::Model {
                        rule: "runtime scoring tables fit GadgetPolicy.max_log2_rows",
                    });
                }
                return Ok(existing.id);
            }
            let id = TableId(u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?);
            let generated = GeneratedTable::new(id, kind)
                .map_err(|error| CompileError::ModelTable { table: id, error })?;
            if generated.log2_rows() > max_log2_rows {
                return Err(CompileError::Model {
                    rule: "runtime scoring tables fit GadgetPolicy.max_log2_rows",
                });
            }
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

/// Canonical committed columns of one mixture
/// [`OneSidedLog2`](https://github.com/AbdelStark/tessera/blob/main/docs/rfcs/RFC-0017-scoring.md)
/// row in allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Log2AdviceColumn {
    /// `m_t ∈ {0, 1}` (`0` when `eps_log2 == 0`), the match indicator bit.
    MatchBit,
    /// `I_t ∈ [0, 63)`, the integer part of the `log2` bound.
    IntExp,
    /// `P_t = 2^{I_t}`, looked up in `Pow2 { max_exp: 63 }`.
    Pow2Val,
    /// `f_t ∈ [0, 256)`, the `2^{−8}`-bit fractional exponent.
    FracExp,
    /// `E_t = Exp2Frac[f_t]`, looked up in `Exp2Frac { frac_bits: 8, out_bits: 16 }`.
    Exp2Val,
    /// `q_t = (Y_t + ψ_t) · E_t`, the scaled selected numerator.
    ScaledSelected,
    /// `Z_j`, a committed low 16-bit limb of `Z_t` (`j < k_Z − 1`).
    TotalLimb(u8),
    /// `Q_j`, a committed low 16-bit limb of `q_t` (`j < k_Q − 1`).
    ScaledLimb(u8),
    /// `P_j`, one of the three committed low 16-bit limbs of `P_t` (`j ∈ 0..3`).
    Pow2Limb(u8),
    /// A committed normalization or subtraction entry of `BigIntCompare`, grouped by array per
    /// RFC-0050.
    Comparison {
        /// Array kind in RFC-0050 order.
        column: CompareColumn,
        /// Limb index `0 .. m`.
        limb: u32,
    },
}

/// Derives the ordered [`Log2AdviceColumn`] schema and ordered [`TableKind`] lookup requests of
/// `OneSidedLog2` (RFC-0017 'In-circuit scoring (mixture)') for `(vocab, sigma)`.
///
/// # Errors
/// [`CompileError::Model`] if `vocab` (`1..=2^18`), `sigma` (`0..=32`), or comparison boxes leave
/// RFC-0009's bounds.
pub fn log2_columns_and_requests(
    vocab: u32,
    sigma: u8,
) -> Result<(Vec<Log2AdviceColumn>, Vec<TableKind>), CompileError> {
    let bad = || CompileError::Model {
        rule: "OneSidedLog2 needs 1 ≤ vocab ≤ 2^18 and σ ≤ 32 within Goldilocks' centered range",
    };
    if !(1..=1 << 18).contains(&vocab) || sigma > 32 {
        return Err(bad());
    }
    let total_width = (64 - (u64::from(vocab) << sigma).leading_zeros()) as u8;
    let max_q = ((1u128 << sigma) + 1) * ((1u128 << 17) - 1);
    let scaled_width = (128 - max_q.leading_zeros()) as u8;
    let half_field = (Goldilocks::MODULUS - 1) / 2;
    if 1u64 << total_width.max(scaled_width) >= half_field {
        return Err(bad());
    }

    let k_z = total_width.div_ceil(16) as usize;
    let k_q = scaled_width.div_ceil(16) as usize;
    let top_z_bits = total_width - ((k_z as u8 - 1) * 16);
    let top_q_bits = scaled_width - ((k_q as u8 - 1) * 16);
    let full = (1u64 << 16) - 1;
    let z_box = |h: usize| {
        if h + 1 == k_z {
            (1u64 << top_z_bits) - 1
        } else if h < k_z {
            full
        } else {
            0
        }
    };
    let q_box = |a: usize| {
        if a + 1 == k_q {
            (1u64 << top_q_bits) - 1
        } else if a < k_q {
            full
        } else {
            0
        }
    };
    let p_box = |b: usize| match b {
        0..=2 => full,
        3 => (1u64 << 15) - 1,
        _ => 0,
    };
    let v_limbs = [u64::from(vocab) & 0xffff, u64::from(vocab) >> 16];
    let max_left = (((1u128 << total_width) - 1) + u128::from(vocab)) << 16;
    let max_right = ((1u128 << scaled_width) - 1) * ((1u128 << 63) - 1);
    let total_bits = (128 - max_left.max(max_right).leading_zeros()) as usize;
    let m = total_bits.div_ceil(16);

    let left_bounds: Vec<u64> = (0..m)
        .map(|j| {
            if j == 0 {
                0
            } else {
                let h = j - 1;
                z_box(h) + v_limbs.get(h).copied().unwrap_or(0)
            }
        })
        .collect();
    let right_bounds: Vec<u64> = (0..m)
        .map(|j| {
            (0..k_q)
                .flat_map(|a| (0..4).map(move |b| (a, b)))
                .filter(|&(a, b)| a + b == j)
                .map(|(a, b)| q_box(a) * p_box(b))
                .sum()
        })
        .collect();

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
    let left_carries = carry_widths(&left_bounds)?;
    let right_carries = carry_widths(&right_bounds)?;

    let mut columns = vec![
        Log2AdviceColumn::MatchBit,
        Log2AdviceColumn::IntExp,
        Log2AdviceColumn::Pow2Val,
        Log2AdviceColumn::FracExp,
        Log2AdviceColumn::Exp2Val,
        Log2AdviceColumn::ScaledSelected,
    ];
    for j in 0..(k_z - 1) as u8 {
        columns.push(Log2AdviceColumn::TotalLimb(j));
    }
    for j in 0..(k_q - 1) as u8 {
        columns.push(Log2AdviceColumn::ScaledLimb(j));
    }
    for j in 0..3u8 {
        columns.push(Log2AdviceColumn::Pow2Limb(j));
    }

    let mut requests = vec![
        TableKind::Pow2 { max_exp: 63 },
        TableKind::Exp2Frac {
            frac_bits: 8,
            out_bits: 16,
        },
    ];
    requests.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, k_z - 1));
    requests.push(TableKind::Range { bits: top_z_bits });
    requests.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, k_q - 1));
    requests.push(TableKind::Range { bits: top_q_bits });
    requests.extend(core::iter::repeat_n(TableKind::Range { bits: 16 }, 3));
    requests.push(TableKind::Range { bits: 15 });

    let m_u32 = m as u32;
    for (column, widths) in [
        (CompareColumn::LeftByte, vec![16u8; m]),
        (CompareColumn::LeftCarry, left_carries),
        (CompareColumn::RightByte, vec![16u8; m]),
        (CompareColumn::RightCarry, right_carries),
        (CompareColumn::Difference, vec![16u8; m]),
        (CompareColumn::Borrow, vec![1u8; m]),
    ] {
        for limb in 0..m_u32 {
            columns.push(Log2AdviceColumn::Comparison { column, limb });
        }
        requests.extend(widths.into_iter().map(|bits| TableKind::Range { bits }));
    }

    Ok((columns, requests))
}

/// Decoding rule under `PredictorSpec::Mixture` (RFC-0017 'Decoding and the sampler PRF').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixtureDecoding {
    /// Greedy decoding (`ŷ_t` certified by bounded `Argmax`, `p̂_t` from `Exp_1` at `T = 1`).
    Greedy,
    /// Inverse-CDF temperature sampling at `T ∈ (0, 4]` (`ŷ_t` certified by bounded `InverseCdf`,
    /// `p̂_t` from `Exp_T`).
    Temperature(Rational),
}

impl MixtureDecoding {
    /// Effective temperature `T` of the smoothed vocabulary distribution `p̂_t` (`1` for greedy).
    pub const fn effective_temperature(self) -> Rational {
        match self {
            Self::Greedy => Rational::ONE,
            Self::Temperature(t) => t,
        }
    }
}

/// Public request geometry for runtime mixture scoring (RFC-0017).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixtureRequestSpec {
    /// Positive prompt length; logit position `prompt − 1` predicts the first output.
    pub prompt: u32,
    /// Total sequence length `prompt + output_len ≤ max_context`.
    pub positions: u32,
    /// Decoding rule (`Greedy` or `Temperature(T)`).
    pub decoding: MixtureDecoding,
}

/// Derived certificate metadata for one output-bearing mixture request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixtureCertificate {
    /// Request index in the batch.
    pub request_index: usize,
    /// Gadget identity assigned after all model nodes.
    pub gadget: GadgetId,
    /// Request decoding rule.
    pub decoding: MixtureDecoding,
    /// Vocabulary `Exp_T` table id (`Exp_1` for `Greedy`, `Exp_T` for `Temperature(T)`).
    pub exp_table: TableId,
    /// Physical `Range` table id for bounded `Argmax` (`Some` iff `Greedy`).
    pub argmax_range_table: Option<TableId>,
    /// Physical `Range` bit width for bounded `Argmax` (`Some` iff `Greedy`).
    pub argmax_range_bits: Option<u8>,
    /// Ordered `Range` table ids for bounded `InverseCdf` (non-empty iff `Temperature`).
    pub cdf_range_tables: Vec<TableId>,
    /// Ordered committed `CdfAdviceColumn`s for bounded `InverseCdf` (non-empty iff `Temperature`).
    pub cdf_columns: Vec<CdfAdviceColumn>,
    /// Ordered lookup table ids (`Pow2`, `Exp2Frac`, and `Range`s) for `OneSidedLog2`.
    pub log2_tables: Vec<TableId>,
    /// Ordered committed `Log2AdviceColumn`s for `OneSidedLog2`.
    pub log2_columns: Vec<Log2AdviceColumn>,
    /// Prompt length.
    pub prompt: u32,
    /// Sequence length `prompt + outputs`.
    pub positions: u32,
    /// Unpadded vocabulary size.
    pub vocab: u32,
    /// Public inclusive logit interval.
    pub logits: IntRange,
}

/// Runtime catalog and certificate layout for mixture scoring (`eps_log2 ∈ 0..=16`) with up to
/// [`MAX_BATCH_TEMPERATURES`] distinct `Exp_T` tables, leaving `ClaimList`'s wire format untouched.
#[derive(Clone, Debug)]
pub struct MixturePlan {
    selection: Selection,
    eps_log2: u8,
    temperatures: Vec<TemperatureTable>,
    certificates: Vec<MixtureCertificate>,
}

impl MixturePlan {
    /// Derive the runtime `Exp_T`, `Pow2`, `Exp2Frac`, and `Range` tables and bounded certificate
    /// layouts for `requests` at mixture exponent `eps_log2 ∈ 0..=16`.
    ///
    /// # Errors
    /// [`CompileError::Usage`] if `selection.tables != claims.tables`, `eps_log2 > 16`, any request
    /// has invalid lengths or temperature outside `(0, 4]`, or the batch names more than
    /// [`MAX_BATCH_TEMPERATURES`] distinct effective temperatures; [`CompileError::Model`] if
    /// public logits exceed `|ℓ| < 2^28` or any required runtime table exceeds
    /// `GadgetPolicy.max_log2_rows`.
    pub fn derive(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        requests: &[MixtureRequestSpec],
        eps_log2: u8,
        first_ordinal: u32,
    ) -> Result<Self, CompileError> {
        const MIXTURE_REQUEST: &str =
            "each mixture request names a positive prompt within the model context";
        if selection.tables != claims.tables {
            return Err(usage(METADATA));
        }
        if eps_log2 > 16 {
            return Err(usage("Mixture eps_log2 lies in 0..=16 (RFC-0017)"));
        }
        let shape = &lowering
            .graph
            .tensors
            .get(lowering.logits.0 as usize)
            .ok_or_else(|| usage(MIXTURE_REQUEST))?
            .shape;
        let [_lowered_positions, vocab] = shape.as_slice() else {
            return Err(usage(MIXTURE_REQUEST));
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
        let max_log2_rows = claims.gadget_policy.max_log2_rows;
        let max_key_bits = max_log2_rows.min(32);
        let sigma = claims.quant.exp_sigma;

        let mut distinct: Vec<Rational> = Vec::new();
        for req in requests {
            if req.prompt == 0
                || req.prompt > req.positions
                || req.positions > claims.arch.max_context
            {
                return Err(usage(MIXTURE_REQUEST));
            }
            let temp = req.decoding.effective_temperature();
            let _ = vocab_exp_scale(claims.quant.logits_exponent, temp)?;
            if req.prompt < req.positions && !distinct.contains(&temp) {
                distinct.push(temp);
            }
        }
        if distinct.len() > MAX_BATCH_TEMPERATURES {
            return Err(usage(BATCH_TEMPERATURES));
        }

        let mut selection = selection.clone();
        let mut intern = |kind: TableKind| -> Result<TableId, CompileError> {
            if let Some(existing) = selection.tables.iter().find(|t| t.kind == kind) {
                if existing.log2_rows > max_log2_rows {
                    return Err(CompileError::Model {
                        rule: "runtime scoring tables fit GadgetPolicy.max_log2_rows",
                    });
                }
                return Ok(existing.id);
            }
            let id = TableId(u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?);
            let generated = GeneratedTable::new(id, kind)
                .map_err(|error| CompileError::ModelTable { table: id, error })?;
            if generated.log2_rows() > max_log2_rows {
                return Err(CompileError::Model {
                    rule: "runtime scoring tables fit GadgetPolicy.max_log2_rows",
                });
            }
            selection.tables.push(generated.spec().clone());
            Ok(id)
        };

        let mut temperatures = Vec::new();
        for temp in distinct {
            let active_reqs: Vec<usize> = requests
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    r.decoding.effective_temperature() == temp && r.prompt < r.positions
                })
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
                    .ok_or_else(|| usage(MIXTURE_REQUEST))?;
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
            .ok_or_else(|| usage(MIXTURE_REQUEST))?;
        let mut certificates = Vec::new();
        for (req_idx, req) in requests.iter().enumerate() {
            if req.prompt == req.positions {
                continue;
            }
            let ordinal = u32::try_from(certificates.len())
                .ok()
                .and_then(|o| base_node.checked_add(o))
                .ok_or_else(|| usage(MIXTURE_REQUEST))?;
            let temp = req.decoding.effective_temperature();
            let exp_table = temperatures
                .iter()
                .find(|t| t.temperature == temp)
                .map(|t| t.table)
                .ok_or_else(|| usage(METADATA))?;

            let (argmax_range_table, argmax_range_bits, cdf_range_tables, cdf_columns) = match req
                .decoding
            {
                MixtureDecoding::Greedy => {
                    if min_key_bits > max_log2_rows {
                        return Err(CompileError::Model {
                            rule: "the greedy physical Range width fits GadgetPolicy.max_log2_rows",
                        });
                    }
                    let table = intern(TableKind::Range { bits: min_key_bits })?;
                    (Some(table), Some(min_key_bits), Vec::new(), Vec::new())
                }
                MixtureDecoding::Temperature(_) => {
                    let (cols, widths) = cdf_columns_and_requests(*vocab, sigma, true)?;
                    let mut tables = Vec::with_capacity(widths.len());
                    for bits in widths {
                        tables.push(intern(TableKind::Range { bits })?);
                    }
                    (None, None, tables, cols)
                }
            };

            let (log2_columns, log2_kinds) = log2_columns_and_requests(*vocab, sigma)?;
            let mut log2_tables = Vec::with_capacity(log2_kinds.len());
            for kind in log2_kinds {
                log2_tables.push(intern(kind)?);
            }

            certificates.push(MixtureCertificate {
                request_index: req_idx,
                gadget: GadgetId(ordinal),
                decoding: req.decoding,
                exp_table,
                argmax_range_table,
                argmax_range_bits,
                cdf_range_tables,
                cdf_columns,
                log2_tables,
                log2_columns,
                prompt: req.prompt,
                positions: req.positions,
                vocab: *vocab,
                logits,
            });
        }

        Ok(Self {
            selection,
            eps_log2,
            temperatures,
            certificates,
        })
    }

    /// Complete runtime table catalog, preserving every model table id, kind and digest.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Mixture weight exponent `k = eps_log2` (`ε = 2^{−k}`).
    pub const fn eps_log2(&self) -> u8 {
        self.eps_log2
    }

    /// Distinct active `Exp_T` tables (`T = 1` for greedy, `T` for temperature) and their grouped
    /// request indices.
    pub fn temperatures(&self) -> &[TemperatureTable] {
        &self.temperatures
    }

    /// Per-request certificate descriptions in batch order (omitting zero-output requests).
    pub fn certificates(&self) -> &[MixtureCertificate] {
        &self.certificates
    }
}
