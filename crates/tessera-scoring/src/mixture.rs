// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0017 mixture predictor scoring (`q_t = (1 − ε)·δ_{ŷ_t} + ε·p̂_t`) with the one-sided
//! `log2` bound `Û_r = Σ_{t ∈ r} B_t` in `2^{−8}`-bit (`FixedBits`) units (`INV-SC-1..3`).
//!
//! Each real output row `t` pairs:
//! - the model's recorded next-token indicator `h_{y_t}` (already proved `OneHot` by the model),
//! - a separate committed decoded indicator `h_{ŷ_t}` certified by bounded [`Argmax`] (`Greedy`)
//!   or bounded [`InverseCdf`] (`Temperature(T)`),
//! - `VocabMax` over `Exp_1` (`Greedy`) or `Exp_T` (`Temperature(T)`) producing `e_{t,v}`, and
//! - [`OneSidedLog2`] certifying `B_t ≥ ⌈−log2 q_t(y_t) · 2^8⌉` on mismatches and `c_match(k)` on
//!   certified matches, while exposing only the summed per-request `Û_r` publicly (`INV-SC-3`).

use core::fmt;

use tessera_codec::error::RequestId;
use tessera_codec::{Rational, SecretBuf, Zeroize};
use tessera_compiler::{Lowering, MAX_BATCH_TEMPERATURES, MixtureDecoding, vocab_exp_scale};
use tessera_field::{Field, Goldilocks as G, PrimeField};
use tessera_gadgets::{
    Argmax, ArgmaxRelations, BatchedArgmaxEvaluation, BatchedArgmaxSelector, BatchedCdfEvaluation,
    BatchedCdfSelector, BatchedLog2Evaluation, BatchedLog2Selector, CdfBatchColumns, CdfColumn,
    CdfRelations, GadgetError, InverseCdf, Log2BatchColumns, Log2Column, Log2VocabRelations,
    Lookups, OneHotWitness, OneSidedLog2, SampledRelations, VocabEvaluation, VocabMax,
    VocabRelations, VocabSelector, VocabWitness, c_match,
};
use tessera_ir::{Op, TensorId, WeightType};
use tessera_poly::PolyField;
use tessera_sumcheck::Elementwise;
use tessera_tables::{Table, TableKind, TableSpec};

use crate::{OutputRow, ScoringError};

fn usage(rule: &'static str) -> ScoringError {
    ScoringError::Usage { rule }
}

#[derive(Clone, Debug)]
struct MixtureTempEntry {
    temperature: Rational,
    vocab_max: VocabMax<G>,
    cdf: Result<(InverseCdf<G>, Vec<CdfColumn>), GadgetError>,
}

/// Mixture-mode scoring circuit on one native [`Lowering`], supporting greedy decoding (`T = 1`
/// for `p̂_t`) and up to [`MAX_BATCH_TEMPERATURES`] (`4`) distinct `Exp_T` tables at `eps_log2 ∈ 0..=16`
/// (RFC-0017).
pub struct MixtureScoring {
    entries: Vec<MixtureTempEntry>,
    argmax: Option<Argmax<G>>,
    log2: OneSidedLog2<G>,
    log2_columns: Vec<Log2Column>,
    eps_log2: u8,
    vocab: usize,
    n: u32,
    sources: OutputRow,
}

impl fmt::Debug for MixtureScoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixtureScoring")
            .field("vocab", &self.vocab)
            .field("positions", &self.n)
            .field("eps_log2", &self.eps_log2)
            .field("temperatures", &self.entries.len())
            .field("greedy_enabled", &self.argmax.is_some())
            .finish_non_exhaustive()
    }
}

impl MixtureScoring {
    /// Bind the model's `tokens`, existing `OneHot`, and `LmHead` logits, and validate `eps_log2 ∈ 0..=16`,
    /// `1..=4` `(temperature, Exp_T)` tables against `logits_exponent`, optional bounded `Argmax`
    /// `Range` table, and `aux_tables` (`Pow2 { 63 }`, `Exp2Frac { 8, 16 }`, and `Range`s).
    ///
    /// # Errors
    /// [`ScoringError::Usage`] for an invalid lowering, `eps_log2 > 16`, invalid/duplicate/missing
    /// `Exp_T` tables, or inconsistent `sigma`; [`ScoringError::Gadget`] for invalid `Exp_T`,
    /// `Argmax`, `InverseCdf`, or `OneSidedLog2` table specifications.
    pub fn new(
        lowering: &Lowering,
        logits_exponent: i8,
        eps_log2: u8,
        exp_tables: &[(Rational, TableSpec)],
        argmax_table: Option<&TableSpec>,
        aux_tables: &[TableSpec],
    ) -> Result<Self, ScoringError> {
        let bad = usage("mixture scoring needs original tokens, OneHot and LM-head logits");
        lowering.graph.validate().map_err(|_| bad)?;
        if c_match(eps_log2).is_err() {
            return Err(usage("Mixture eps_log2 lies in 0..=16 (RFC-0017)"));
        }
        let tensor = |id: TensorId| {
            lowering
                .graph
                .tensors
                .get(id.0 as usize)
                .filter(|t| t.id == id)
                .ok_or(bad)
        };
        let tokens = tensor(lowering.tokens)?;
        let h = tensor(lowering.one_hot)?;
        let logits = tensor(lowering.logits)?;
        let &[n, vocab] = h.shape.as_slice() else {
            return Err(bad);
        };
        let h_node = lowering
            .graph
            .producer(h.id)
            .and_then(|id| lowering.graph.nodes.get(id.0 as usize))
            .ok_or(bad)?;
        let l_node = lowering
            .graph
            .producer(logits.id)
            .and_then(|id| lowering.graph.nodes.get(id.0 as usize))
            .ok_or(bad)?;
        if n == 0
            || n > 1 << 17
            || tokens.shape != [n]
            || logits.shape != [n, vocab]
            || !matches!(h_node.op, Op::OneHot { vocab: v } if v == vocab)
            || h_node.inputs.as_slice() != [tokens.id]
            || !matches!(
                l_node.op,
                Op::MatMulWeight { weight } if weight.ty == WeightType::LmHead && weight.layer == 0
            )
            || logits.range.lo <= -(1 << 28)
            || logits.range.hi >= 1 << 28
        {
            return Err(bad);
        }
        if exp_tables.is_empty() || exp_tables.len() > MAX_BATCH_TEMPERATURES {
            return Err(usage(
                "a mixture batch registers 1..=4 Exp_T tables (RFC-0017)",
            ));
        }

        let spread = logits.range.hi - logits.range.lo;
        let max_span =
            u64::try_from(spread).map_err(|_| usage("public logit spread is non-negative"))?;
        let vocab_usize = vocab as usize;

        let argmax = match argmax_table {
            Some(spec) => Some(Argmax::<G>::with_span(vocab_usize, max_span, spec)?),
            None => None,
        };

        let mut entries: Vec<MixtureTempEntry> = Vec::with_capacity(exp_tables.len());
        let mut common_sigma: Option<u8> = None;
        for (temperature, exp_spec) in exp_tables {
            if entries.iter().any(|e| e.temperature == *temperature) {
                return Err(usage("each registered batch temperature is distinct"));
            }
            let expected_scale = vocab_exp_scale(logits_exponent, *temperature)
                .map_err(|_| usage("temperature lies in (0, 4] and folds into Rational"))?;
            if !matches!(exp_spec.kind, TableKind::Exp { scale, .. } if scale == expected_scale) {
                return Err(usage(
                    "Exp_T's scale equals 2^{logits_exponent} / T (RFC-0017)",
                ));
            }
            let vocab_max = VocabMax::<G>::new(vocab_usize, exp_spec)?;
            if spread >= i128::from(vocab_max.table().row_count()) {
                return Err(usage(
                    "Exp_T's key width covers the public logit spread (SP-7)",
                ));
            }
            let sigma = vocab_max.sigma();
            if let Some(prev) = common_sigma
                && prev != sigma
            {
                return Err(usage("all registered Exp_T tables share exp_sigma"));
            }
            common_sigma = Some(sigma);
            let cdf = InverseCdf::<G>::new(vocab_usize, sigma, aux_tables).and_then(|c| {
                let cols = c.constraints::<G>()?.committed;
                Ok((c, cols))
            });
            entries.push(MixtureTempEntry {
                temperature: *temperature,
                vocab_max,
                cdf,
            });
        }

        let sigma = common_sigma.ok_or_else(|| usage("at least one Exp_T table is registered"))?;
        let log2 = OneSidedLog2::<G>::new(vocab_usize, sigma, eps_log2, aux_tables)?;
        let log2_columns = log2.constraints::<G>()?.committed;

        Ok(Self {
            entries,
            argmax,
            log2,
            log2_columns,
            eps_log2,
            vocab: vocab_usize,
            n,
            sources: OutputRow {
                logits: logits.id,
                one_hot: h.id,
                tokens: tokens.id,
                logit_position: 0,
                token_position: 0,
            },
        })
    }

    /// Mixture weight exponent `k = eps_log2` (`ε = 2^{−k}`).
    pub const fn eps_log2(&self) -> u8 {
        self.eps_log2
    }

    /// Shared [`OneSidedLog2`] gadget of this batch.
    pub const fn log2(&self) -> &OneSidedLog2<G> {
        &self.log2
    }

    /// Instantiate a mixture request with `decoding`, `prompt_len ≥ 1`, and
    /// `prompt_len + output_len == n`.
    ///
    /// # Errors
    /// [`ScoringError::Usage`] if `decoding`'s required `Exp_T` or `Argmax` table was not
    /// registered or lengths do not partition the lowered sequence; [`ScoringError::Gadget`] if
    /// `Temperature` decoding's `InverseCdf` tables were absent from `aux_tables`.
    pub fn request(
        &self,
        id: RequestId,
        decoding: MixtureDecoding,
        prompt_len: u32,
        output_len: u32,
    ) -> Result<MixtureRequest<'_>, ScoringError> {
        let temp = decoding.effective_temperature();
        let entry_idx = self
            .entries
            .iter()
            .position(|e| e.temperature == temp)
            .ok_or_else(|| usage("the request's effective Exp_T temperature is registered"))?;
        match decoding {
            MixtureDecoding::Greedy => {
                if self.argmax.is_none() {
                    return Err(usage(
                        "greedy mixture requests require an Argmax Range table",
                    ));
                }
            }
            MixtureDecoding::Temperature(_) => {
                let _ = self.entries[entry_idx].cdf.as_ref().map_err(|&e| e)?;
            }
        }
        if prompt_len == 0 || prompt_len.checked_add(output_len) != Some(self.n) {
            return Err(usage(
                "a nonempty prompt and outputs cover the lowered sequence",
            ));
        }
        let padded = if output_len == 0 {
            0
        } else {
            (output_len as usize).next_power_of_two()
        };
        Ok(MixtureRequest {
            scorer: self,
            entry_idx,
            decoding,
            id,
            prompt_len,
            output_len,
            padded,
        })
    }
}

/// Public geometry of one mixture-mode request.
#[derive(Clone, Copy, Debug)]
pub struct MixtureRequest<'s> {
    scorer: &'s MixtureScoring,
    entry_idx: usize,
    decoding: MixtureDecoding,
    id: RequestId,
    prompt_len: u32,
    output_len: u32,
    padded: usize,
}

impl<'s> MixtureRequest<'s> {
    /// Padded output row count (`0` when `output_len == 0`).
    pub const fn padded_rows(&self) -> usize {
        self.padded
    }

    /// Decoding rule (`Greedy` or `Temperature(T)`).
    pub const fn decoding(&self) -> MixtureDecoding {
        self.decoding
    }

    /// `VocabMax` gadget for this request's smoothed distribution `p̂_t`.
    pub fn vocab_max(&self) -> &'s VocabMax<G> {
        &self.scorer.entries[self.entry_idx].vocab_max
    }

    /// Bounded `Argmax` gadget when `decoding == MixtureDecoding::Greedy`.
    pub const fn argmax(&self) -> Option<&'s Argmax<G>> {
        match self.decoding {
            MixtureDecoding::Greedy => self.scorer.argmax.as_ref(),
            MixtureDecoding::Temperature(_) => None,
        }
    }

    /// Bounded `InverseCdf` gadget when `decoding == MixtureDecoding::Temperature(_)`.
    pub fn cdf(&self) -> Option<&'s InverseCdf<G>> {
        match self.decoding {
            MixtureDecoding::Greedy => None,
            MixtureDecoding::Temperature(_) => self.scorer.entries[self.entry_idx]
                .cdf
                .as_ref()
                .ok()
                .map(|(c, _)| c),
        }
    }

    /// `OneSidedLog2` gadget of this batch.
    pub const fn log2(&self) -> &'s OneSidedLog2<G> {
        &self.scorer.log2
    }

    /// Original model source rows for `index`, or `None` on output padding (`RFC-0047`).
    ///
    /// # Errors
    /// [`ScoringError::Usage`] if `index >= padded_rows()`.
    pub fn row(&self, index: usize) -> Result<Option<OutputRow>, ScoringError> {
        if index >= self.padded {
            return Err(usage("an output row lies in its padded certificate domain"));
        }
        if index >= self.output_len as usize {
            return Ok(None);
        }
        let position = self.prompt_len + index as u32;
        Ok(Some(OutputRow {
            logit_position: position - 1,
            token_position: position,
            ..self.scorer.sources
        }))
    }

    /// Stream every real output row, compute its clearing bounded decoding witness (`Argmax` or
    /// `InverseCdf`), `VocabMax` witness, and `OneSidedLog2` witness, and accumulate only the
    /// per-request bound `Û_r = Σ_{t ∈ r} B_t` publicly (`INV-SC-1..3`).
    ///
    /// Unlike exact mode, recorded tokens that deviate from `D`'s decoded token `ŷ_t` succeed and
    /// contribute their one-sided `-log2 q_t(y_t)` bound to `Û_r`.
    ///
    /// # Errors
    /// [`ScoringError::Usage`] for wrong input counts/shapes or missing/unexpected `draw`;
    /// [`ScoringError::Gadget`] for invalid logits, indicators or draws; [`ScoringError::Memory`]
    /// on allocation failure.
    pub fn witness(
        self,
        inputs: impl IntoIterator<Item = MixtureInputs>,
    ) -> Result<MixtureCertificate<'s>, ScoringError> {
        let entry = &self.scorer.entries[self.entry_idx];
        let padded_vocab = self.scorer.vocab.next_power_of_two();
        let total_indicator_len = self
            .padded
            .checked_mul(padded_vocab)
            .ok_or_else(|| usage("mixture indicator dimensions fit usize"))?;

        let mut vocab_maxima = alloc_zeroed(self.padded)?;
        let mut argmax_maxima = match self.decoding {
            MixtureDecoding::Greedy => Some(alloc_zeroed(self.padded)?),
            MixtureDecoding::Temperature(_) => None,
        };
        let mut predicted_indicators = alloc_zeroed(total_indicator_len)?;

        let mut cdf_cols: Option<Vec<(CdfColumn, SecretBuf<G>)>> = match self.decoding {
            MixtureDecoding::Greedy => None,
            MixtureDecoding::Temperature(_) => {
                let (_, cols) = entry.cdf.as_ref().map_err(|&e| e)?;
                let mut bufs = Vec::with_capacity(cols.len());
                for &col in cols {
                    bufs.push((col, alloc_zeroed(self.padded)?));
                }
                Some(bufs)
            }
        };

        let mut log2_cols: Vec<(Log2Column, SecretBuf<G>)> =
            Vec::with_capacity(self.scorer.log2_columns.len());
        for &col in &self.scorer.log2_columns {
            log2_cols.push((col, alloc_zeroed(self.padded)?));
        }

        let mut rows = inputs.into_iter();
        let mut uhat_acc = SecretBuf::filled(0u64, 1);
        for index in 0..(self.output_len as usize) {
            let input = rows
                .next()
                .ok_or_else(|| usage("one input row is supplied per recorded output"))?;
            let row = compute_real_row(self, &input)?;
            vocab_maxima[index] = row.vocab_witness.maximum()[0];
            if let (Some(am_buf), Some(am_w)) =
                (argmax_maxima.as_mut(), row.argmax_witness.as_ref())
            {
                am_buf[index] = am_w.maximum()[0];
            }
            let ind_base = index * padded_vocab;
            predicted_indicators[ind_base..ind_base + padded_vocab]
                .copy_from_slice(row.predicted.indicator());
            if let (Some(cols), Some(cw)) = (cdf_cols.as_mut(), row.cdf_witness.as_ref()) {
                for ((_, buf), &val) in cols.iter_mut().zip(cw.values()) {
                    buf[index] = G::from_i64(val);
                }
            }
            for ((_, buf), &val) in log2_cols.iter_mut().zip(row.log2_witness.values()) {
                buf[index] = G::from_i64(val);
            }
            let mut row_bound = row.log2_witness.bound_units();
            let overflow = uhat_acc[0].checked_add(row_bound).is_none();
            uhat_acc[0] = uhat_acc[0].wrapping_add(row_bound);
            row_bound.zeroize();
            if overflow {
                return Err(usage("per-request Û fits u64"));
            }
        }
        if rows.next().is_some() {
            return Err(usage("no extra output row is supplied"));
        }

        // Fill padded rows (`output_len .. padded`) with canonical zero-row witnesses.
        if (self.output_len as usize) < self.padded {
            let zero_exps = SecretBuf::filled(G::ZERO, padded_vocab);
            let zero_h = self.scorer.log2.domain().witness(0, false)?;
            if let Some(ref mut cols) = cdf_cols {
                let (cdf, _) = entry.cdf.as_ref().map_err(|&e| e)?;
                let pad_cw = cdf.witness(&zero_exps, G::ZERO, &zero_h, false)?;
                for index in (self.output_len as usize)..self.padded {
                    for ((_, buf), &val) in cols.iter_mut().zip(pad_cw.values()) {
                        buf[index] = G::from_i64(val);
                    }
                }
            }
            let pad_log2 = self
                .scorer
                .log2
                .witness(&zero_exps, &zero_h, &zero_h, false)?;
            for index in (self.output_len as usize)..self.padded {
                for ((_, buf), &val) in log2_cols.iter_mut().zip(pad_log2.values()) {
                    buf[index] = G::from_i64(val);
                }
            }
        }

        Ok(MixtureCertificate {
            request: self,
            uhat: uhat_acc[0],
            vocab_maxima,
            argmax_maxima,
            predicted_indicators,
            cdf_columns: cdf_cols.map(CdfBatchColumns::from_columns),
            log2_columns: Log2BatchColumns::from_columns(log2_cols),
        })
    }
}

/// Regenerated source rows and optional PRF draw `u_t` for one real mixture output token.
pub struct MixtureInputs {
    /// Vocabulary-padded model logit row at `logit_position = token_position − 1`.
    pub logits: SecretBuf<G>,
    /// Recorded next-token indicator proved by the model's `OneHot` relation.
    pub next_token: OneHotWitness<G>,
    /// PRF draw `u_t ∈ [0, 2^32)` for temperature decoding, or `None` for greedy decoding.
    pub draw: Option<G>,
}

impl Drop for MixtureInputs {
    fn drop(&mut self) {
        if let Some(ref mut d) = self.draw {
            d.zeroize();
        }
    }
}

impl fmt::Debug for MixtureInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixtureInputs")
            .field("logits", &self.logits.len())
            .field("indicator", &self.next_token.indicator().len())
            .field("has_draw", &self.draw.is_some())
            .finish_non_exhaustive()
    }
}

/// Clearing certificate columns and public per-request `Û_r` claim for one mixture request
/// (`INV-SC-1..3`).
#[must_use = "commit the certificate columns and prove all VocabMax, decoding, OneSidedLog2 and PRF obligations"]
pub struct MixtureCertificate<'s> {
    request: MixtureRequest<'s>,
    uhat: u64,
    vocab_maxima: SecretBuf<G>,
    argmax_maxima: Option<SecretBuf<G>>,
    predicted_indicators: SecretBuf<G>,
    cdf_columns: Option<CdfBatchColumns<G>>,
    log2_columns: Log2BatchColumns<G>,
}

impl fmt::Debug for MixtureCertificate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixtureCertificate")
            .field("request", &self.request.id)
            .field("rows", &self.vocab_maxima.len())
            .field("uhat", &self.uhat)
            .finish_non_exhaustive()
    }
}

impl<'s> MixtureCertificate<'s> {
    /// Per-request public bound `Û_r` in units of `2^{−8}` bits (`FixedBits`; RFC-0017).
    pub const fn uhat(&self) -> u64 {
        self.uhat
    }

    /// Borrow the secret per-row `VocabMax` maximum column (`padded_rows` entries).
    pub fn vocab_maxima(&self) -> &[G] {
        &self.vocab_maxima
    }

    /// Borrow the secret per-row `Argmax` maximum column (`padded_rows` entries) when `Greedy`.
    pub fn argmax_maxima(&self) -> Option<&[G]> {
        self.argmax_maxima.as_deref()
    }

    /// Borrow the flat row-major committed decoded indicators `h_{ŷ_t}`
    /// (`padded_rows · pad(vocab)` entries).
    pub fn predicted_indicators(&self) -> &[G] {
        &self.predicted_indicators
    }

    /// Borrow all committed [`CdfColumn`]s across `padded_rows` when `Temperature`.
    pub const fn cdf_columns(&self) -> Option<&CdfBatchColumns<G>> {
        self.cdf_columns.as_ref()
    }

    /// Borrow all committed [`Log2Column`]s across `padded_rows`.
    pub const fn log2_columns(&self) -> &Log2BatchColumns<G> {
        &self.log2_columns
    }

    /// Assemble the complete power-of-two batch (`padded_rows`) of `VocabMax`, bounded `Argmax` or
    /// `InverseCdf`, and `OneSidedLog2` proof columns, lookup instances, and batched degree-3
    /// sumcheck relations from replayed real-row `inputs` (`output_len` items).
    ///
    /// # Errors
    /// [`ScoringError::Usage`] if `output_len == 0`, `inputs.len() != output_len`, or any replayed
    /// row disagrees with the committed certificate columns; [`ScoringError::Gadget`] or
    /// [`ScoringError::Memory`] on invalid shapes or allocation failure.
    pub fn prepare_batch(
        &self,
        inputs: &[MixtureInputs],
    ) -> Result<MixtureBatch<'s>, ScoringError> {
        let real_rows = self.request.output_len as usize;
        let padded_rows = self.request.padded;
        if real_rows == 0 || inputs.len() != real_rows {
            return Err(usage(
                "prepare_batch takes the exact nonempty real output rows of the request",
            ));
        }
        let entry = &self.request.scorer.entries[self.request.entry_idx];
        let padded_vocab = self.request.scorer.vocab.next_power_of_two();
        let total = padded_rows
            .checked_mul(padded_vocab)
            .ok_or_else(|| usage("batch dimensions fit usize"))?;

        let mut logits = SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        let mut recorded = SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        let mut exponentials =
            SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        let mut draws =
            SecretBuf::try_with_capacity(padded_rows).map_err(|_| ScoringError::Memory)?;

        let mut consistent = true;
        let mut replayed_uhat = SecretBuf::filled(0u64, 1);
        for (index, input) in inputs.iter().enumerate() {
            let row = compute_real_row(self.request, input)?;
            consistent &= self.vocab_maxima[index] == row.vocab_witness.maximum()[0];
            if let (Some(am_buf), Some(am_w)) =
                (self.argmax_maxima.as_ref(), row.argmax_witness.as_ref())
            {
                consistent &= am_buf[index] == am_w.maximum()[0];
            }
            let ind_base = index * padded_vocab;
            for v in 0..padded_vocab {
                consistent &=
                    self.predicted_indicators[ind_base + v] == row.predicted.indicator()[v];
            }
            if let (Some(cdf_cols), Some(cw)) =
                (self.cdf_columns.as_ref(), row.cdf_witness.as_ref())
            {
                for (&col, &val) in cw.columns().iter().zip(cw.values()) {
                    consistent &= cdf_cols
                        .column(col)
                        .is_some_and(|s| s[index] == G::from_i64(val));
                }
            }
            for (&col, &val) in row
                .log2_witness
                .columns()
                .iter()
                .zip(row.log2_witness.values())
            {
                consistent &= self
                    .log2_columns
                    .column(col)
                    .is_some_and(|s| s[index] == G::from_i64(val));
            }
            let mut rb = row.log2_witness.bound_units();
            replayed_uhat[0] = replayed_uhat[0].saturating_add(rb);
            rb.zeroize();

            logits.extend(input.logits.iter().copied());
            recorded.extend(input.next_token.indicator().iter().copied());
            exponentials.extend(row.vocab_witness.exponentials().iter().copied());
            draws.push(input.draw.unwrap_or(G::ZERO));
        }
        consistent &= replayed_uhat[0] == self.uhat;
        if !consistent {
            return Err(usage(
                "replayed mixture inputs match the committed certificate columns",
            ));
        }

        for _ in real_rows..padded_rows {
            for _ in 0..padded_vocab {
                logits.push(G::ZERO);
                recorded.push(G::ZERO);
                exponentials.push(G::ZERO);
            }
            draws.push(G::ZERO);
        }

        let mut vocab_maxima =
            SecretBuf::try_with_capacity(padded_rows).map_err(|_| ScoringError::Memory)?;
        vocab_maxima.extend(self.vocab_maxima.iter().copied());
        let vocab_witness = VocabWitness::from_columns(vocab_maxima, exponentials);
        let vocab_selector = entry.vocab_max.selector(padded_rows, real_rows)?;
        let vocab_eval =
            entry
                .vocab_max
                .evaluate_batch(&logits, padded_rows, real_rows, &vocab_witness)?;

        let mut predicted =
            SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        predicted.extend(self.predicted_indicators.iter().copied());

        let decoding_batch = match self.request.decoding {
            MixtureDecoding::Greedy => {
                let argmax = self
                    .request
                    .scorer
                    .argmax
                    .as_ref()
                    .ok_or_else(|| usage("greedy mixture batch holds Argmax"))?;
                let src_max = self
                    .argmax_maxima
                    .as_ref()
                    .ok_or_else(|| usage("greedy mixture certificate holds Argmax maxima"))?;
                let mut maxima =
                    SecretBuf::try_with_capacity(padded_rows).map_err(|_| ScoringError::Memory)?;
                maxima.extend(src_max.iter().copied());
                let selector = argmax.selector(padded_rows, real_rows)?;
                let eval = argmax.evaluate_batch(&logits, &predicted, &maxima, padded_rows)?;
                DecodingBatch::Greedy {
                    argmax,
                    selector,
                    maxima,
                    eval,
                }
            }
            MixtureDecoding::Temperature(_) => {
                let (cdf, _) = entry.cdf.as_ref().map_err(|&e| e)?;
                let src_cols = self.cdf_columns.as_ref().ok_or_else(|| {
                    usage("temperature mixture certificate holds CdfBatchColumns")
                })?;
                let cols = clone_cdf_columns(src_cols)?;
                let selector = cdf.selector(padded_rows, real_rows)?;
                let eval =
                    cdf.evaluate_batch(vocab_witness.exponentials(), &predicted, padded_rows)?;
                let lookups = cdf.evaluate_lookups(&selector, &draws, &eval, &cols)?;
                DecodingBatch::Temperature {
                    cdf,
                    selector,
                    draws,
                    eval,
                    columns: cols,
                    lookups,
                }
            }
        };

        let log2_cols = clone_log2_columns(&self.log2_columns)?;
        let log2_selector = self.request.scorer.log2.selector(padded_rows, real_rows)?;
        let log2_eval = self.request.scorer.log2.evaluate_batch(
            vocab_witness.exponentials(),
            &recorded,
            &log2_cols,
            &log2_selector,
        )?;
        let log2_lookups =
            self.request
                .scorer
                .log2
                .evaluate_lookups(&log2_selector, &log2_eval, &log2_cols)?;

        Ok(MixtureBatch {
            scorer: self.request.scorer,
            entry,
            uhat: self.uhat,
            vocab_selector,
            log2_selector,
            logits,
            recorded,
            predicted,
            vocab_witness,
            vocab_eval,
            decoding: decoding_batch,
            log2_eval,
            log2_columns: log2_cols,
            log2_lookups,
        })
    }
}

enum DecodingBatch<'s> {
    Greedy {
        argmax: &'s Argmax<G>,
        selector: BatchedArgmaxSelector<G>,
        maxima: SecretBuf<G>,
        eval: BatchedArgmaxEvaluation<G>,
    },
    Temperature {
        cdf: &'s InverseCdf<G>,
        selector: BatchedCdfSelector<G>,
        draws: SecretBuf<G>,
        eval: BatchedCdfEvaluation<G>,
        columns: CdfBatchColumns<G>,
        lookups: Lookups<G>,
    },
}

/// Complete power-of-two batch (`padded_rows`) of `VocabMax`, bounded `Argmax`/`InverseCdf`, and
/// `OneSidedLog2` proof obligations for one mixture request.
#[must_use = "prove VocabMax, bounded decoding, OneSidedLog2 relations/lookups, and the Û_r bound claim"]
pub struct MixtureBatch<'s> {
    scorer: &'s MixtureScoring,
    entry: &'s MixtureTempEntry,
    uhat: u64,
    vocab_selector: VocabSelector<G>,
    log2_selector: BatchedLog2Selector<G>,
    logits: SecretBuf<G>,
    recorded: SecretBuf<G>,
    predicted: SecretBuf<G>,
    vocab_witness: VocabWitness<G>,
    vocab_eval: VocabEvaluation<G>,
    decoding: DecodingBatch<'s>,
    log2_eval: BatchedLog2Evaluation<G>,
    log2_columns: Log2BatchColumns<G>,
    log2_lookups: Lookups<G>,
}

impl fmt::Debug for MixtureBatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixtureBatch")
            .field("padded_rows", &self.vocab_selector.padded_rows())
            .field("real_rows", &self.vocab_selector.real_rows())
            .field("uhat", &self.uhat)
            .finish_non_exhaustive()
    }
}

impl MixtureBatch<'_> {
    /// Padded row count `R = 2^{row_vars}`.
    pub const fn padded_rows(&self) -> usize {
        self.vocab_selector.padded_rows()
    }

    /// Active real output row count.
    pub const fn real_rows(&self) -> usize {
        self.vocab_selector.real_rows()
    }

    /// Public per-request `Û_r` claim in `2^{−8}`-bit units (`FixedBits`).
    pub const fn uhat(&self) -> u64 {
        self.uhat
    }

    /// Public per-request `Û_r` claim as a Goldilocks element.
    pub fn uhat_field(&self) -> G {
        G::from_u64_reduced(self.uhat)
    }

    /// `VocabMax` gadget for this batch's `Exp_T` table.
    pub const fn vocab_max(&self) -> &VocabMax<G> {
        &self.entry.vocab_max
    }

    /// `OneSidedLog2` gadget of this batch.
    pub const fn log2(&self) -> &OneSidedLog2<G> {
        &self.scorer.log2
    }

    /// Flat row-major model logits (`padded_rows · pad(vocab)`), zero on row and vocab padding.
    pub fn logits(&self) -> &[G] {
        &self.logits
    }

    /// Flat row-major recorded indicators `h_{y_t}` (`padded_rows · pad(vocab)`).
    pub fn recorded_indicators(&self) -> &[G] {
        &self.recorded
    }

    /// Flat row-major committed decoded indicators `h_{ŷ_t}` (`padded_rows · pad(vocab)`).
    pub fn predicted_indicators(&self) -> &[G] {
        &self.predicted
    }

    /// Per-row PRF draws `u_t` (`padded_rows`), when `Temperature`.
    pub fn draws(&self) -> Option<&[G]> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => None,
            DecodingBatch::Temperature { draws, .. } => Some(draws),
        }
    }

    /// Per-row `Argmax` maxima (`padded_rows`), when `Greedy`.
    pub fn argmax_maxima(&self) -> Option<&[G]> {
        match &self.decoding {
            DecodingBatch::Greedy { maxima, .. } => Some(maxima),
            DecodingBatch::Temperature { .. } => None,
        }
    }

    /// Evaluated bounded `Argmax` suffix, keys and `Range` lookups, when `Greedy`.
    pub const fn argmax_evaluated(&self) -> Option<&BatchedArgmaxEvaluation<G>> {
        match &self.decoding {
            DecodingBatch::Greedy { eval, .. } => Some(eval),
            DecodingBatch::Temperature { .. } => None,
        }
    }

    /// Evaluated bounded `InverseCdf` suffix and virtual sums, when `Temperature`.
    pub const fn cdf_evaluated(&self) -> Option<&BatchedCdfEvaluation<G>> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => None,
            DecodingBatch::Temperature { eval, .. } => Some(eval),
        }
    }

    /// Committed [`CdfBatchColumns`], when `Temperature`.
    pub const fn cdf_columns(&self) -> Option<&CdfBatchColumns<G>> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => None,
            DecodingBatch::Temperature { columns, .. } => Some(columns),
        }
    }

    /// Evaluated bounded `InverseCdf` `Range` [`Lookups`], when `Temperature`.
    pub const fn cdf_lookups(&self) -> Option<&Lookups<G>> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => None,
            DecodingBatch::Temperature { lookups, .. } => Some(lookups),
        }
    }

    /// Committed `VocabMax` columns (`maximum` and `exponentials`).
    pub const fn vocab_witness(&self) -> &VocabWitness<G> {
        &self.vocab_witness
    }

    /// Evaluated `VocabMax` keys, product-tree leaves, padding checks, roots and `Exp_T` lookups.
    pub const fn vocab_evaluated(&self) -> &VocabEvaluation<G> {
        &self.vocab_eval
    }

    /// Evaluated `OneSidedLog2` virtual sums `Z_t`, `Y_t` and per-row bounds `B_t`.
    pub const fn log2_evaluated(&self) -> &BatchedLog2Evaluation<G> {
        &self.log2_eval
    }

    /// Committed [`Log2BatchColumns`] across `padded_rows`.
    pub const fn log2_columns(&self) -> &Log2BatchColumns<G> {
        &self.log2_columns
    }

    /// Evaluated `OneSidedLog2` [`Lookups`] (`Pow2`, `Exp2Frac`, and `Range`s).
    pub const fn log2_lookups(&self) -> &Lookups<G> {
        &self.log2_lookups
    }

    /// Batched `VocabMax` sumcheck relations (`key`, `leaf`, `padding`).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn vocab_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<VocabRelations<'_, F>, ScoringError> {
        Ok(self.entry.vocab_max.relations(
            &self.vocab_selector,
            &self.logits,
            &self.vocab_witness,
        )?)
    }

    /// Bounded mode's batched `OneHot` obligations (`booleanity`, `vocabulary`, and row `count = ψ_t`)
    /// on the committed decoded indicator `h_{ŷ_t}`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn predicted_indicator_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<SampledRelations<'_, F>, ScoringError> {
        match &self.decoding {
            DecodingBatch::Greedy {
                argmax, selector, ..
            } => Ok(argmax.batched_sampled_relations(selector, &self.predicted)?),
            DecodingBatch::Temperature { cdf, selector, .. } => {
                Ok(cdf.batched_sampled_relations(selector, &self.predicted)?)
            }
        }
    }

    /// Batched bounded `Argmax` relations (`selected`, `key`) when `Greedy`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn argmax_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Option<ArgmaxRelations<'_, F>>, ScoringError> {
        match &self.decoding {
            DecodingBatch::Greedy {
                argmax,
                selector,
                maxima,
                eval,
            } => Ok(Some(argmax.batched_relations(
                selector,
                &self.logits,
                &self.predicted,
                maxima,
                eval,
            )?)),
            DecodingBatch::Temperature { .. } => Ok(None),
        }
    }

    /// Batched bounded `InverseCdf` contractions (`total`, `below`, `selected`) when `Temperature`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn cdf_contractions<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Option<CdfRelations<'_, F>>, ScoringError> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => Ok(None),
            DecodingBatch::Temperature {
                cdf,
                selector,
                eval,
                ..
            } => Ok(Some(cdf.batched_relations(
                selector,
                self.vocab_witness.exponentials(),
                &self.predicted,
                eval,
            )?)),
        }
    }

    /// Batched bounded `InverseCdf` row equalities when `Temperature`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn cdf_equality_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Option<Vec<Elementwise<'_, F>>>, ScoringError> {
        match &self.decoding {
            DecodingBatch::Greedy { .. } => Ok(None),
            DecodingBatch::Temperature {
                cdf,
                selector,
                draws,
                eval,
                columns,
                ..
            } => Ok(Some(
                cdf.equality_relations(selector, draws, eval, columns)?,
            )),
        }
    }

    /// Public weights reducing a batched suffix claim `L̃(ρ_vocab, ρ_row)` to `h_{ŷ_t}`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if `rho.len() != log2(pad(vocab)) + log2(padded_rows)`.
    pub fn predicted_suffix_weights<F: PolyField<Base = G>>(
        &self,
        rho: &[F],
    ) -> Result<Vec<F>, ScoringError> {
        let rows = self.vocab_selector.padded_rows();
        match &self.decoding {
            DecodingBatch::Greedy { argmax, .. } => Ok(argmax.batched_suffix_weights(rho, rows)?),
            DecodingBatch::Temperature { cdf, .. } => Ok(cdf.batched_suffix_weights(rho, rows)?),
        }
    }

    /// Batched `OneSidedLog2` vocabulary relations (`total`, `selected`, `match_indicator`).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn log2_vocab_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Log2VocabRelations<'_, F>, ScoringError> {
        Ok(self.scorer.log2.batched_vocab_relations(
            &self.log2_selector,
            self.vocab_witness.exponentials(),
            &self.recorded,
            &self.predicted,
            &self.log2_columns,
        )?)
    }

    /// Batched `OneSidedLog2` row equalities (`m_t` check, `q_t = (Y_t + ψ_t)·E_t`,
    /// `BigIntCompare`, and `(ψ_t − 1)·Z_t = 0`).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn log2_equality_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Vec<Elementwise<'_, F>>, ScoringError> {
        Ok(self.scorer.log2.equality_relations(
            &self.log2_selector,
            &self.log2_eval,
            &self.log2_columns,
        )?)
    }

    /// Batched degree-3 `OneSidedLog2` bound contraction `Σ_t B_t = Û_r` (RFC-0017).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn log2_bound_relation<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Elementwise<'_, F>, ScoringError> {
        Ok(self
            .scorer
            .log2
            .bound_relation(&self.log2_selector, &self.log2_columns)?)
    }
}

struct RealRowWitness {
    vocab_witness: VocabWitness<G>,
    argmax_witness: Option<tessera_gadgets::ArgmaxWitness<G>>,
    predicted: OneHotWitness<G>,
    cdf_witness: Option<tessera_gadgets::CdfWitness>,
    log2_witness: tessera_gadgets::Log2Witness,
}

fn compute_real_row(
    request: MixtureRequest<'_>,
    input: &MixtureInputs,
) -> Result<RealRowWitness, ScoringError> {
    let entry = &request.scorer.entries[request.entry_idx];
    let padded_vocab = request.scorer.vocab.next_power_of_two();
    if input.logits.len() != padded_vocab || input.next_token.indicator().len() != padded_vocab {
        return Err(usage(
            "logits and the original indicator have the padded vocabulary length",
        ));
    }

    let mut valid = true;
    let mut sum = SecretBuf::filled(G::ZERO, 1);
    for (v, &h) in input.next_token.indicator().iter().enumerate() {
        valid &= (h == G::ZERO) | (h == G::ONE);
        valid &= (v < request.scorer.vocab) | (h == G::ZERO);
        sum[0] += h;
    }
    valid &= sum[0] == G::ONE;
    if !valid {
        return Err(GadgetError::Input {
            rule: "the original next-token indicator is OneHot over the real vocabulary",
        }
        .into());
    }

    let vocab_witness = entry.vocab_max.witness(&input.logits, true)?;
    let (argmax_witness, predicted, cdf_witness) = match request.decoding {
        MixtureDecoding::Greedy => {
            if input.draw.is_some() {
                return Err(usage("greedy mixture rows take no PRF draw"));
            }
            let argmax = request
                .scorer
                .argmax
                .as_ref()
                .ok_or_else(|| usage("greedy mixture requests hold Argmax"))?;
            let (pred, am_witness) = argmax.sample(&input.logits, true)?;
            (Some(am_witness), pred, None)
        }
        MixtureDecoding::Temperature(_) => {
            let draw = input
                .draw
                .ok_or_else(|| usage("temperature mixture rows require a PRF draw"))?;
            let (cdf, _) = entry.cdf.as_ref().map_err(|&e| e)?;
            let pred = cdf.sample(vocab_witness.exponentials(), draw, true)?;
            let cw = cdf.witness(vocab_witness.exponentials(), draw, &pred, true)?;
            (None, pred, Some(cw))
        }
    };

    let log2_witness = request.scorer.log2.witness(
        vocab_witness.exponentials(),
        &input.next_token,
        &predicted,
        true,
    )?;

    Ok(RealRowWitness {
        vocab_witness,
        argmax_witness,
        predicted,
        cdf_witness,
        log2_witness,
    })
}

fn alloc_zeroed(len: usize) -> Result<SecretBuf<G>, ScoringError> {
    let mut buf = SecretBuf::try_with_capacity(len).map_err(|_| ScoringError::Memory)?;
    for _ in 0..len {
        buf.push(G::ZERO);
    }
    Ok(buf)
}

fn clone_cdf_columns(src: &CdfBatchColumns<G>) -> Result<CdfBatchColumns<G>, ScoringError> {
    let mut cols = Vec::new();
    for name in src.names() {
        let vals = src.column(name).unwrap_or(&[]);
        let mut buf = SecretBuf::try_with_capacity(vals.len()).map_err(|_| ScoringError::Memory)?;
        buf.extend(vals.iter().copied());
        cols.push((name, buf));
    }
    Ok(CdfBatchColumns::from_columns(cols))
}

fn clone_log2_columns(src: &Log2BatchColumns<G>) -> Result<Log2BatchColumns<G>, ScoringError> {
    let mut cols = Vec::new();
    for name in src.names() {
        let vals = src.column(name).unwrap_or(&[]);
        let mut buf = SecretBuf::try_with_capacity(vals.len()).map_err(|_| ScoringError::Memory)?;
        buf.extend(vals.iter().copied());
        cols.push((name, buf));
    }
    Ok(Log2BatchColumns::from_columns(cols))
}
