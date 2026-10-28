// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0017 exact temperature decoding certificates (`VocabMax` + `InverseCdf` + `Û = 0`),
//! reusing the recorded sequence's `OneHot` indicators and the runtime `Exp_T` / `Range` catalog.

use core::{fmt, hint::select_unpredictable};

use tessera_codec::error::{RequestId, TokenPosition};
use tessera_codec::{Rational, SecretBuf, Zeroize};
use tessera_compiler::{Lowering, MAX_BATCH_TEMPERATURES, vocab_exp_scale};
use tessera_field::{Field, Goldilocks as G, PrimeField};
use tessera_gadgets::{
    BatchedCdfEvaluation, BatchedCdfSelector, CdfBatchColumns, CdfColumn, CdfEvaluation,
    CdfRelations, CdfWitness, GadgetError, InverseCdf, Lookups, OneHotWitness, VocabEvaluation,
    VocabMax, VocabRelations, VocabSelector, VocabWitness,
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
struct TemperatureEntry {
    temperature: Rational,
    vocab_max: VocabMax<G>,
    cdf: InverseCdf<G>,
    columns: Vec<CdfColumn>,
}

/// Exact-mode temperature scoring circuit on one native [`Lowering`], supporting up to
/// [`MAX_BATCH_TEMPERATURES`] (`4`) distinct temperatures `T ∈ (0, 4]` per batch (RFC-0017).
pub struct ExactTemperature {
    entries: Vec<TemperatureEntry>,
    vocab: usize,
    n: u32,
    sources: OutputRow,
}

impl fmt::Debug for ExactTemperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExactTemperature")
            .field("vocab", &self.vocab)
            .field("positions", &self.n)
            .field("temperatures", &self.entries.len())
            .finish_non_exhaustive()
    }
}

impl ExactTemperature {
    /// Bind the model's `tokens`, existing `OneHot`, and `LmHead` logits, and validate `1..=4`
    /// `(temperature, Exp_T)` tables against `logits_exponent` (`RFC-0027`), `Exp_T[K_0] = 0`
    /// (`RFC-0030`), the public logit spread, and `InverseCdf`'s `Range` tables (`RFC-0058`).
    ///
    /// # Errors
    /// [`ScoringError::Usage`] for an invalid lowering, temperature outside `(0, 4]`, duplicate or
    /// empty/over-limit temperature list, or mismatched `Exp_T` scale/width; [`ScoringError::Gadget`]
    /// for invalid `Exp_T` (`Exp_T[K_0] != 0`) or `Range` specifications.
    pub fn new(
        lowering: &Lowering,
        logits_exponent: i8,
        exp_tables: &[(Rational, TableSpec)],
        range_tables: &[TableSpec],
    ) -> Result<Self, ScoringError> {
        let bad =
            usage("exact temperature scoring needs original tokens, OneHot and LM-head logits");
        lowering.graph.validate().map_err(|_| bad)?;
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
                "a temperature batch registers 1..=4 temperatures (RFC-0017)",
            ));
        }
        let spread = logits.range.hi - logits.range.lo;
        let vocab_usize = vocab as usize;
        let mut entries: Vec<TemperatureEntry> = Vec::with_capacity(exp_tables.len());
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
            let cdf = InverseCdf::<G>::new(vocab_usize, vocab_max.sigma(), range_tables)?;
            let columns = cdf.constraints::<G>()?.committed;
            entries.push(TemperatureEntry {
                temperature: *temperature,
                vocab_max,
                cdf,
                columns,
            });
        }

        Ok(Self {
            entries,
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

    /// Instantiate a request at a registered `temperature` over this lowering, with `prompt_len ≥ 1`
    /// and `prompt_len + output_len == n`.
    ///
    /// # Errors
    /// [`ScoringError::Usage`] if `temperature` was not registered or lengths do not partition the
    /// lowered sequence.
    pub fn request(
        &self,
        id: RequestId,
        temperature: Rational,
        prompt_len: u32,
        output_len: u32,
    ) -> Result<TemperatureRequest<'_>, ScoringError> {
        let entry_idx = self
            .entries
            .iter()
            .position(|e| e.temperature == temperature)
            .ok_or_else(|| usage("the request's temperature is registered in this batch"))?;
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
        Ok(TemperatureRequest {
            scorer: self,
            entry_idx,
            id,
            prompt_len,
            output_len,
            padded,
        })
    }
}

/// Public geometry of one exact-mode temperature request.
#[derive(Clone, Copy, Debug)]
pub struct TemperatureRequest<'s> {
    scorer: &'s ExactTemperature,
    entry_idx: usize,
    id: RequestId,
    prompt_len: u32,
    output_len: u32,
    padded: usize,
}

impl<'s> TemperatureRequest<'s> {
    /// Padded output row count (`0` when `output_len == 0`).
    pub const fn padded_rows(&self) -> usize {
        self.padded
    }

    /// Registered temperature of this request.
    pub fn temperature(&self) -> Rational {
        self.scorer.entries[self.entry_idx].temperature
    }

    /// Vocabulary maximum gadget for this request's `Exp_T` table.
    pub fn vocab_max(&self) -> &'s VocabMax<G> {
        &self.scorer.entries[self.entry_idx].vocab_max
    }

    /// `InverseCdf` gadget for this request's `Exp_T` table.
    pub fn cdf(&self) -> &'s InverseCdf<G> {
        &self.scorer.entries[self.entry_idx].cdf
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

    /// Stream and validate every real output row from its model `logits`, recorded `next_token`
    /// `OneHot`, and PRF `draw` `u_t`.
    ///
    /// Processes all real output rows before reporting the first mismatch in request-stream
    /// coordinates (`ERR-WITNESS-EXACT-MISMATCH`), selecting the first failing row index via
    /// `select_unpredictable`.
    ///
    /// # Errors
    /// [`ScoringError::Usage`] for wrong input counts/shapes; [`ScoringError::Gadget`] for invalid
    /// upstream logits, indicators or draws; [`ScoringError::Memory`] on allocation failure;
    /// [`ScoringError::ExactMismatch`] if any recorded token deviates from the inverse-CDF sample.
    pub fn witness(
        self,
        inputs: impl IntoIterator<Item = TemperatureInputs>,
    ) -> Result<TemperatureCertificate<'s>, ScoringError> {
        let entry = &self.scorer.entries[self.entry_idx];
        let mut maxima =
            SecretBuf::try_with_capacity(self.padded).map_err(|_| ScoringError::Memory)?;
        for _ in 0..self.padded {
            maxima.push(G::ZERO);
        }
        let mut cdf_cols: Vec<(CdfColumn, SecretBuf<G>)> = Vec::with_capacity(entry.columns.len());
        for &col in &entry.columns {
            let mut buf =
                SecretBuf::try_with_capacity(self.padded).map_err(|_| ScoringError::Memory)?;
            for _ in 0..self.padded {
                buf.push(G::ZERO);
            }
            cdf_cols.push((col, buf));
        }

        let mut rows = inputs.into_iter();
        let mut first = SecretBuf::try_with_capacity(1).map_err(|_| ScoringError::Memory)?;
        first.push(self.output_len);
        for index in 0..self.output_len {
            let input = rows
                .next()
                .ok_or_else(|| usage("one input row is supplied per recorded output"))?;
            let prepared = prepare_single(self, index as usize, &input, None)?;
            maxima[index as usize] = prepared.vocab_witness.maximum()[0];
            if let Some(ref cw) = prepared.cdf_witness {
                for ((_, buf), &val) in cdf_cols.iter_mut().zip(cw.values()) {
                    buf[index as usize] = G::from_i64(val);
                }
            }
            first[0] = select_unpredictable(
                (first[0] == self.output_len) & !prepared.matching,
                index,
                first[0],
            );
        }
        if rows.next().is_some() {
            return Err(usage("no extra output row is supplied"));
        }
        if first[0] != self.output_len {
            return Err(ScoringError::ExactMismatch {
                request: self.id,
                position: TokenPosition(self.prompt_len + first[0]),
            });
        }
        // Fill padded rows (`output_len .. padded`) from `InverseCdf`'s unique zero witness.
        if (self.output_len as usize) < self.padded {
            let padded_vocab = self.scorer.vocab.next_power_of_two();
            let zero_exps = SecretBuf::filled(G::ZERO, padded_vocab);
            let zero_h = entry.cdf.domain().witness(0, false)?;
            let pad_cw = entry.cdf.witness(&zero_exps, G::ZERO, &zero_h, false)?;
            for index in (self.output_len as usize)..self.padded {
                for ((_, buf), &val) in cdf_cols.iter_mut().zip(pad_cw.values()) {
                    buf[index] = G::from_i64(val);
                }
            }
        }
        Ok(TemperatureCertificate {
            request: self,
            maxima,
            cdf_columns: CdfBatchColumns::from_columns(cdf_cols),
        })
    }
}

/// Regenerated source rows and PRF draw `u_t` for one real output token.
pub struct TemperatureInputs {
    /// Vocabulary-padded model logit row at `logit_position = token_position − 1`.
    pub logits: SecretBuf<G>,
    /// The existing recorded next-token indicator proved by the model's `OneHot` relation.
    pub next_token: OneHotWitness<G>,
    /// PRF draw `u_t ∈ [0, 2^32)` produced by the Sampler PRF circuit at output index `t`.
    pub draw: G,
}

impl Drop for TemperatureInputs {
    fn drop(&mut self) {
        self.draw.zeroize();
    }
}

impl fmt::Debug for TemperatureInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemperatureInputs")
            .field("logits", &self.logits.len())
            .field("indicator", &self.next_token.indicator().len())
            .finish_non_exhaustive()
    }
}

/// Clearing certificate columns and `Û = 0` claim for one exact-mode temperature request.
#[must_use = "commit the certificate columns and prove all VocabMax, InverseCdf and PRF obligations"]
pub struct TemperatureCertificate<'s> {
    request: TemperatureRequest<'s>,
    maxima: SecretBuf<G>,
    cdf_columns: CdfBatchColumns<G>,
}

impl fmt::Debug for TemperatureCertificate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemperatureCertificate")
            .field("request", &self.request.id)
            .field("rows", &self.maxima.len())
            .finish_non_exhaustive()
    }
}

impl<'s> TemperatureCertificate<'s> {
    /// Per-request public bound in units of `2^-8` bits: exact mode always claims zero.
    pub const fn uhat(&self) -> u64 {
        0
    }

    /// Borrow the secret per-row maximum column (`padded_rows` entries) for commitment/proving.
    pub fn maxima(&self) -> &[G] {
        &self.maxima
    }

    /// Borrow all committed [`CdfColumn`]s across the request's `padded_rows` rows.
    pub const fn cdf_columns(&self) -> &CdfBatchColumns<G> {
        &self.cdf_columns
    }

    /// Recreate one real output row's proof operands from replayed sources and the fixed
    /// certificate columns.
    ///
    /// # Errors
    /// [`ScoringError::Usage`] for a padded/out-of-domain row; [`ScoringError::ExactMismatch`] if
    /// replayed inputs disagree with the certificate; [`ScoringError::Gadget`] for malformed shapes.
    pub fn prepare_row<'i>(
        &self,
        index: usize,
        inputs: &'i TemperatureInputs,
    ) -> Result<TemperatureRow<'s, 'i>, ScoringError> {
        let max = *self
            .maxima
            .get(index)
            .ok_or_else(|| usage("a proof row is a real output"))?;
        let prepared = prepare_single(self.request, index, inputs, Some(max))?;
        if !prepared.matching {
            return Err(ScoringError::ExactMismatch {
                request: self.request.id,
                position: TokenPosition(prepared.sources.token_position),
            });
        }
        Ok(prepared)
    }

    /// Assemble the complete power-of-two batch (`padded_rows`) of `VocabMax` and `InverseCdf`
    /// proof columns, lookup instances, and batched degree-3 sumcheck relations from replayed
    /// real-row `inputs` (`output_len` items).
    ///
    /// # Errors
    /// [`ScoringError::Usage`] if `output_len == 0` or `inputs.len() != output_len`;
    /// [`ScoringError::ExactMismatch`] if any replayed row disagrees with the certificate;
    /// [`ScoringError::Gadget`] or [`ScoringError::Memory`] on invalid shapes or allocation failure.
    pub fn prepare_batch(
        &self,
        inputs: &[TemperatureInputs],
    ) -> Result<TemperatureBatch<'s>, ScoringError> {
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
        let mut indicators =
            SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        let mut exponentials =
            SecretBuf::try_with_capacity(total).map_err(|_| ScoringError::Memory)?;
        let mut draws =
            SecretBuf::try_with_capacity(padded_rows).map_err(|_| ScoringError::Memory)?;

        for (index, input) in inputs.iter().enumerate() {
            let row = self.prepare_row(index, input)?;
            logits.extend(input.logits.iter().copied());
            indicators.extend(input.next_token.indicator().iter().copied());
            exponentials.extend(row.vocab_witness.exponentials().iter().copied());
            draws.push(input.draw);
        }
        for _ in real_rows..padded_rows {
            for _ in 0..padded_vocab {
                logits.push(G::ZERO);
                indicators.push(G::ZERO);
                exponentials.push(G::ZERO);
            }
            draws.push(G::ZERO);
        }

        let mut maxima =
            SecretBuf::try_with_capacity(padded_rows).map_err(|_| ScoringError::Memory)?;
        maxima.extend(self.maxima.iter().copied());
        let vocab_witness = VocabWitness::from_columns(maxima, exponentials);
        let vocab_selector = entry.vocab_max.selector(padded_rows, real_rows)?;
        let vocab_eval =
            entry
                .vocab_max
                .evaluate_batch(&logits, padded_rows, real_rows, &vocab_witness)?;
        let cdf_selector = entry.cdf.selector(padded_rows, real_rows)?;
        let cdf_eval =
            entry
                .cdf
                .evaluate_batch(vocab_witness.exponentials(), &indicators, padded_rows)?;
        let cdf_lookups =
            entry
                .cdf
                .evaluate_lookups(&cdf_selector, &draws, &cdf_eval, &self.cdf_columns)?;

        Ok(TemperatureBatch {
            entry,
            vocab_selector,
            cdf_selector,
            logits,
            indicators,
            draws,
            vocab_witness,
            vocab_eval,
            cdf_eval,
            cdf_columns: clone_batch_columns(&self.cdf_columns)?,
            cdf_lookups,
        })
    }
}

fn clone_batch_columns(src: &CdfBatchColumns<G>) -> Result<CdfBatchColumns<G>, ScoringError> {
    let mut cols = Vec::new();
    for name in src.names() {
        let vals = src.column(name).unwrap_or(&[]);
        let mut buf = SecretBuf::try_with_capacity(vals.len()).map_err(|_| ScoringError::Memory)?;
        buf.extend(vals.iter().copied());
        cols.push((name, buf));
    }
    Ok(CdfBatchColumns::from_columns(cols))
}

/// Prepared single-row `VocabMax` and `InverseCdf` proof operands.
#[must_use = "prove VocabMax, InverseCdf contractions/equalities/lookups, and source claims"]
pub struct TemperatureRow<'s, 'i> {
    entry: &'s TemperatureEntry,
    inputs: &'i TemperatureInputs,
    sources: OutputRow,
    vocab_witness: VocabWitness<G>,
    vocab_eval: VocabEvaluation<G>,
    cdf_witness: Option<CdfWitness>,
    cdf_eval: CdfEvaluation<G>,
    matching: bool,
}

impl Drop for TemperatureRow<'_, '_> {
    fn drop(&mut self) {
        self.matching.zeroize();
    }
}

impl fmt::Debug for TemperatureRow<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemperatureRow")
            .field("sources", &self.sources)
            .finish_non_exhaustive()
    }
}

impl TemperatureRow<'_, '_> {
    /// Original model tensor identities and teacher-forcing positions.
    pub const fn sources(&self) -> OutputRow {
        self.sources
    }

    /// Borrowed original next-token indicator.
    pub const fn next_token(&self) -> &OneHotWitness<G> {
        &self.inputs.next_token
    }

    /// Clearing `VocabMax` witness (`maximum` and `exponentials`).
    pub const fn vocab_witness(&self) -> &VocabWitness<G> {
        &self.vocab_witness
    }

    /// Evaluated `VocabMax` lookup keys, product leaves, padding check and root.
    pub const fn vocab_evaluated(&self) -> &VocabEvaluation<G> {
        &self.vocab_eval
    }

    /// Clearing `InverseCdf` row witness.
    pub const fn cdf_witness(&self) -> Option<&CdfWitness> {
        self.cdf_witness.as_ref()
    }

    /// Evaluated `InverseCdf` suffix `L` and sums `[Z, B, S]`.
    pub const fn cdf_evaluated(&self) -> &CdfEvaluation<G> {
        &self.cdf_eval
    }

    /// Single-row `InverseCdf` contractions (`Z`, `B`, `S`).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn cdf_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<CdfRelations<'_, F>, ScoringError> {
        Ok(self.entry.cdf.relations(
            self.vocab_witness.exponentials(),
            &self.inputs.next_token,
            &self.cdf_eval,
        )?)
    }
}

/// Complete power-of-two batch (`padded_rows`) of `VocabMax` and `InverseCdf` proof obligations.
#[must_use = "prove VocabMax, InverseCdf batched contractions, row equalities, and LogUp instances"]
pub struct TemperatureBatch<'s> {
    entry: &'s TemperatureEntry,
    vocab_selector: VocabSelector<G>,
    cdf_selector: BatchedCdfSelector<G>,
    logits: SecretBuf<G>,
    indicators: SecretBuf<G>,
    draws: SecretBuf<G>,
    vocab_witness: VocabWitness<G>,
    vocab_eval: VocabEvaluation<G>,
    cdf_eval: BatchedCdfEvaluation<G>,
    cdf_columns: CdfBatchColumns<G>,
    cdf_lookups: Lookups<G>,
}

impl fmt::Debug for TemperatureBatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemperatureBatch")
            .field("padded_rows", &self.vocab_selector.padded_rows())
            .field("real_rows", &self.vocab_selector.real_rows())
            .finish_non_exhaustive()
    }
}

impl TemperatureBatch<'_> {
    /// Padded row count `R = 2^{row_vars}`.
    pub const fn padded_rows(&self) -> usize {
        self.vocab_selector.padded_rows()
    }

    /// Active real output row count.
    pub const fn real_rows(&self) -> usize {
        self.vocab_selector.real_rows()
    }

    /// `VocabMax` gadget for this batch's `Exp_T` table.
    pub const fn vocab_max(&self) -> &VocabMax<G> {
        &self.entry.vocab_max
    }

    /// `InverseCdf` gadget for this batch's `Exp_T` table.
    pub const fn cdf(&self) -> &InverseCdf<G> {
        &self.entry.cdf
    }

    /// Flat row-major model logits (`padded_rows · pad(vocab)`), zero on row and vocab padding.
    pub fn logits(&self) -> &[G] {
        &self.logits
    }

    /// Flat row-major next-token indicators (`padded_rows · pad(vocab)`), zero on row padding.
    pub fn indicators(&self) -> &[G] {
        &self.indicators
    }

    /// Per-row PRF draws `u_t` (`padded_rows`), zero on row padding.
    pub fn draws(&self) -> &[G] {
        &self.draws
    }

    /// Committed `VocabMax` columns (`maximum` and `exponentials`).
    pub const fn vocab_witness(&self) -> &VocabWitness<G> {
        &self.vocab_witness
    }

    /// Evaluated `VocabMax` keys, product-tree leaves, padding checks, roots and `Exp_T` lookups.
    pub const fn vocab_evaluated(&self) -> &VocabEvaluation<G> {
        &self.vocab_eval
    }

    /// Evaluated batched `InverseCdf` suffixes `L` and virtual sums `Z_t`, `B_t`, `S_t`.
    pub const fn cdf_evaluated(&self) -> &BatchedCdfEvaluation<G> {
        &self.cdf_eval
    }

    /// Committed [`CdfBatchColumns`] across `padded_rows`.
    pub const fn cdf_columns(&self) -> &CdfBatchColumns<G> {
        &self.cdf_columns
    }

    /// Evaluated `Range` [`Lookups`] across `padded_rows`.
    pub const fn cdf_lookups(&self) -> &Lookups<G> {
        &self.cdf_lookups
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

    /// Batched degree-3 `InverseCdf` vocabulary contractions (`Z`, `B`, `S`; RFC-0058).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn cdf_contractions<F: PolyField<Base = G>>(
        &self,
    ) -> Result<CdfRelations<'_, F>, ScoringError> {
        Ok(self.entry.cdf.batched_relations(
            &self.cdf_selector,
            self.vocab_witness.exponentials(),
            &self.indicators,
            &self.cdf_eval,
        )?)
    }

    /// Batched `InverseCdf` row-equality zero-checks (both comparisons + `(ψ − 1)·Z = 0`).
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if operand shapes fail engine checks.
    pub fn cdf_equality_relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<Vec<Elementwise<'_, F>>, ScoringError> {
        Ok(self.entry.cdf.equality_relations(
            &self.cdf_selector,
            &self.draws,
            &self.cdf_eval,
            &self.cdf_columns,
        )?)
    }

    /// Public weights reducing a batched suffix claim `L̃(ρ_vocab, ρ_row)` to `ĥ`.
    ///
    /// # Errors
    /// Propagates [`GadgetError`] if `rho.len() != log2(pad(vocab)) + log2(padded_rows)`.
    pub fn suffix_weights<F: PolyField<Base = G>>(
        &self,
        rho: &[F],
    ) -> Result<Vec<F>, ScoringError> {
        Ok(self
            .entry
            .cdf
            .batched_suffix_weights(rho, self.vocab_selector.padded_rows())?)
    }
}

fn prepare_single<'s, 'i>(
    request: TemperatureRequest<'s>,
    index: usize,
    inputs: &'i TemperatureInputs,
    fixed_max: Option<G>,
) -> Result<TemperatureRow<'s, 'i>, ScoringError> {
    let sources = request
        .row(index)?
        .ok_or_else(|| usage("a proof row is a real output"))?;
    let entry = &request.scorer.entries[request.entry_idx];
    let padded = request.scorer.vocab.next_power_of_two();
    if inputs.logits.len() != padded || inputs.next_token.indicator().len() != padded {
        return Err(usage(
            "logits and the original indicator have the padded vocabulary length",
        ));
    }
    let mut valid = true;
    let mut sum = SecretBuf::filled(G::ZERO, 1);
    for (v, &h) in inputs.next_token.indicator().iter().enumerate() {
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

    let honest_vocab = entry.vocab_max.witness(&inputs.logits, true)?;
    let max_matches = fixed_max.is_none_or(|m| m == honest_vocab.maximum()[0]);
    let vocab_witness = match fixed_max {
        None => honest_vocab,
        Some(m) => {
            let mut max_col = SecretBuf::try_with_capacity(1).map_err(|_| ScoringError::Memory)?;
            max_col.push(m);
            let mut exp_col =
                SecretBuf::try_with_capacity(padded).map_err(|_| ScoringError::Memory)?;
            exp_col.extend(honest_vocab.exponentials().iter().copied());
            VocabWitness::from_columns(max_col, exp_col)
        }
    };
    let vocab_eval = entry
        .vocab_max
        .evaluate(&inputs.logits, true, &vocab_witness)?;
    let cdf_eval = entry
        .cdf
        .evaluate(vocab_witness.exponentials(), &inputs.next_token)?;
    let (cdf_witness, cdf_matches) = match entry.cdf.witness(
        vocab_witness.exponentials(),
        inputs.draw,
        &inputs.next_token,
        true,
    ) {
        Ok(cw) => (Some(cw), true),
        Err(e) if e == InverseCdf::<G>::MISMATCH => (None, false),
        Err(other) => return Err(other.into()),
    };
    let matching = max_matches & cdf_matches;

    Ok(TemperatureRow {
        entry,
        inputs,
        sources,
        vocab_witness,
        vocab_eval,
        cdf_witness,
        cdf_eval,
        matching,
    })
}
