// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0017 exact greedy scoring, reusing RFC-0009's Argmax on the recorded OneHot.
use core::{fmt, hint::select_unpredictable};
use tessera_codec::{
    SecretBuf, Zeroize,
    error::{ErrorCode, PublicContext, RequestId, TesseraError, TokenPosition},
};
use tessera_compiler::Lowering;
use tessera_field::{Field, Goldilocks as G, PrimeField};
use tessera_gadgets::{
    Argmax, ArgmaxEvaluation, ArgmaxRelations, ArgmaxWitness, GadgetError, OneHotWitness,
};
use tessera_ir::{Op, TensorId, WeightType};
use tessera_poly::PolyField;
use tessera_tables::TableSpec;

/// A public caller error, an upstream witness defect, or an exact-mode deviation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringError {
    /// Inconsistent public geometry or row counts (`ERR-USAGE`).
    Usage {
        /// The rule, never a secret value.
        rule: &'static str,
    },
    /// An upstream gadget/input invariant failed (`ERR-INTERNAL`).
    Gadget(GadgetError),
    /// The maximum certificate could not be allocated (`ERR-RES-MEMORY`).
    Memory,
    /// The recorded token deviates from the exact predictor (`Argmax` or `InverseCdf`).
    ExactMismatch {
        /// The public request identifier.
        request: RequestId,
        /// Absolute position in the request's token stream.
        position: TokenPosition,
    },
}
impl ScoringError {
    /// Stable error code from 04 §3.
    pub const fn code(self) -> ErrorCode {
        match self {
            Self::Usage { .. } => ErrorCode::Usage,
            Self::Gadget(_) => ErrorCode::Internal,
            Self::Memory => ErrorCode::ResMemory,
            Self::ExactMismatch { .. } => ErrorCode::WitnessExactMismatch,
        }
    }
    /// Only public identifiers; no token, logit, maximum or match vector.
    pub fn context(self) -> PublicContext {
        let mut context = PublicContext::default();
        if let Self::ExactMismatch { request, position } = self {
            context.request = Some(request);
            context.position = Some(position);
        }
        context
    }
}
impl fmt::Display for ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())?;
        match self {
            Self::Usage { rule } => write!(f, ": {rule}"),
            Self::Gadget(e) => write!(f, ": {e}"),
            Self::Memory => Ok(()),
            Self::ExactMismatch { request, position } => write!(f, ": {request}, {position}"),
        }
    }
}
impl std::error::Error for ScoringError {}
impl From<GadgetError> for ScoringError {
    fn from(e: GadgetError) -> Self {
        Self::Gadget(e)
    }
}
impl From<ScoringError> for TesseraError {
    fn from(e: ScoringError) -> Self {
        Self::new(e.code()).with_context(e.context()).with_source(e)
    }
}
fn usage(rule: &'static str) -> ScoringError {
    ScoringError::Usage { rule }
}

/// Original model rows read by one output certificate (RFC-0047).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRow {
    /// The model's logits tensor.
    pub logits: TensorId,
    /// The model's existing OneHot tensor, never a replacement indicator.
    pub one_hot: TensorId,
    /// The recorded token tensor whose OneHot index relation fixes the indicator.
    pub tokens: TensorId,
    /// `prompt_len - 1 + output_index`.
    pub logit_position: u32,
    /// `prompt_len + output_index`.
    pub token_position: u32,
}

/// An exact-greedy circuit on one native lowering. No model-level scoring field is
/// needed (RFC-0026). The enclosing circuit supplies its selected Range(29) spec
/// and assigns global gadget/certificate ids before any proof message.
pub struct ExactGreedy {
    gadget: Argmax<G>,
    vocab: usize,
    n: u32,
    sources: OutputRow,
}
impl fmt::Debug for ExactGreedy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExactGreedy")
            .field("vocab", &self.vocab)
            .field("positions", &self.n)
            .finish_non_exhaustive()
    }
}
impl ExactGreedy {
    /// Bind the model's token, existing indicator and LM-head logits identities.
    /// No protocol wire types or claim-list fields are changed.
    ///
    /// # Errors
    /// Usage for an inconsistent lowering/source role/shape or logits outside
    /// RFC-0009's public integer bound; Gadget for an invalid Range(29) spec.
    pub fn new(lowering: &Lowering, range: &TableSpec) -> Result<Self, ScoringError> {
        let bad = usage("exact scoring needs the original tokens, OneHot and LM-head logits");
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
            || !matches!(h_node.op,Op::OneHot{vocab:v} if v==vocab)
            || h_node.inputs.as_slice() != [tokens.id]
            || !matches!(l_node.op,Op::MatMulWeight{weight} if weight.ty==WeightType::LmHead && weight.layer==0)
            || logits.range.lo <= -(1 << 28)
            || logits.range.hi >= 1 << 28
        {
            return Err(bad);
        }
        let vocab = vocab as usize;
        Ok(Self {
            gadget: Argmax::new(vocab, range)?,
            vocab,
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
    /// Instantiate a request over this lowering, with at least one prompt token.
    /// Empty outputs need no certificate rows and have the public bound zero.
    ///
    /// # Errors
    /// Usage unless `prompt_len + output_len` is the lowering's sequence length
    /// and `prompt_len >= 1`, so every output has its preceding logit row.
    pub fn request(
        &self,
        id: RequestId,
        prompt_len: u32,
        output_len: u32,
    ) -> Result<ExactRequest<'_>, ScoringError> {
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
        Ok(ExactRequest {
            scorer: self,
            id,
            prompt_len,
            output_len,
            padded,
        })
    }
}

/// Public request geometry. Its row map refers to the original model sources.
#[derive(Clone, Copy, Debug)]
pub struct ExactRequest<'s> {
    scorer: &'s ExactGreedy,
    id: RequestId,
    prompt_len: u32,
    output_len: u32,
    padded: usize,
}
impl<'s> ExactRequest<'s> {
    /// The output certificate's padded row count, or zero for no outputs.
    pub const fn padded_rows(&self) -> usize {
        self.padded
    }
    /// Original source rows, or None for output padding. Padding must not read
    /// the model's last real but unused logit row (RFC-0047).
    ///
    /// # Errors
    /// Usage for an index outside this request's certificate domain.
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
    /// Prepare every real output from its original logit and next-token OneHot
    /// rows in order. The iterator can regenerate and release one row at a time;
    /// only the O(outputs) maximum column survives. All real rows are processed
    /// before reporting the first mismatch, in request-stream coordinates.
    ///
    /// The returned bound is a claim to prove, not a verifier verdict. The driver
    /// must prove OneHot on these same indicator sources, every Argmax key's
    /// Range(29) lookup, selected-logit and suffix reductions, and source openings.
    ///
    /// # Errors
    /// Usage for wrong input counts/shapes; Gadget for invalid upstream logits or
    /// indicators; Memory for the certificate; ExactMismatch for any deviation.
    pub fn witness(
        self,
        inputs: impl IntoIterator<Item = GreedyInputs>,
    ) -> Result<GreedyCertificate<'s>, ScoringError> {
        let mut maxima =
            SecretBuf::try_with_capacity(self.padded).map_err(|_| ScoringError::Memory)?;
        for _ in 0..self.padded {
            maxima.push(G::ZERO)
        }
        let mut rows = inputs.into_iter();
        let mut first = SecretBuf::try_with_capacity(1).map_err(|_| ScoringError::Memory)?;
        first.push(self.output_len);
        for index in 0..self.output_len {
            let input = rows
                .next()
                .ok_or_else(|| usage("one input row is supplied per recorded output"))?;
            let prepared = prepare(self, index as usize, &input, None)?;
            maxima[index as usize] = prepared.maximum.maximum()[0];
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
        Ok(GreedyCertificate {
            request: self,
            maxima,
        })
    }
}

/// One real output's regenerated source rows. Move the existing next-token
/// indicator here; do not construct a new indicator for a recomputed token.
/// These buffers clear on release and may be borrowed only for computation.
pub struct GreedyInputs {
    /// The vocabulary-padded logit row at the preceding model position.
    pub logits: SecretBuf<G>,
    /// The same next-token indicator proved by the model's OneHot relation.
    pub next_token: OneHotWitness<G>,
}

/// An exact request's maximum column and public zero claim. This is secret
/// certificate advice until committed/opened, not proof acceptance.
#[must_use = "commit the maximum column and prove all Argmax and source obligations"]
pub struct GreedyCertificate<'s> {
    request: ExactRequest<'s>,
    maxima: SecretBuf<G>,
}
impl fmt::Debug for GreedyCertificate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreedyCertificate")
            .field("request", &self.request.id)
            .field("rows", &self.maxima.len())
            .finish_non_exhaustive()
    }
}
impl<'s> GreedyCertificate<'s> {
    /// Per-request public bound in units of 2^-8 bits: exact mode always claims zero.
    pub const fn uhat(&self) -> u64 {
        0
    }
    /// Borrow the secret maximum column for commitment/proving only. Output
    /// padding is zero; it neither reads nor allocates replacement model rows.
    pub fn maxima(&self) -> &[G] {
        &self.maxima
    }
    /// Recreate one row's proof operands from replayed original sources and the
    /// fixed certificate maximum. No new indicator is generated.
    ///
    /// # Errors
    /// As request witness preparation, or Usage for a padded/out-of-domain row.
    /// ExactMismatch if replayed data no longer agrees with the fixed maximum.
    pub fn prepare_row<'i>(
        &self,
        index: usize,
        inputs: &'i GreedyInputs,
    ) -> Result<GreedyRow<'s, 'i>, ScoringError> {
        let max = *self
            .maxima
            .get(index)
            .ok_or_else(|| usage("a proof row is a real output"))?;
        let prepared = prepare(self.request, index, inputs, Some(max))?;
        if !prepared.matching {
            return Err(ScoringError::ExactMismatch {
                request: self.request.id,
                position: TokenPosition(prepared.sources.token_position),
            });
        }
        Ok(prepared)
    }
}

/// Borrowed original inputs and clearing Argmax columns for all row reductions.
#[must_use = "prove the selected/key/suffix relations, Range(29), and original source claims"]
pub struct GreedyRow<'s, 'i> {
    scorer: &'s ExactGreedy,
    inputs: &'i GreedyInputs,
    sources: OutputRow,
    maximum: ArgmaxWitness<G>,
    evaluated: ArgmaxEvaluation<G>,
    matching: bool,
}
impl Drop for GreedyRow<'_, '_> {
    fn drop(&mut self) {
        self.matching.zeroize();
    }
}
impl fmt::Debug for GreedyRow<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreedyRow")
            .field("sources", &self.sources)
            .finish_non_exhaustive()
    }
}
impl GreedyRow<'_, '_> {
    /// Original tensor identities and their teacher-forcing positions.
    pub const fn sources(&self) -> OutputRow {
        self.sources
    }
    /// The very same borrowed indicator. The enclosing proof also proves its
    /// OneHot/index relation against the recorded token source.
    pub const fn next_token(&self) -> &OneHotWitness<G> {
        &self.inputs.next_token
    }
    /// Clearing maximum advice; never log its values.
    pub const fn maximum(&self) -> &ArgmaxWitness<G> {
        &self.maximum
    }
    /// Clearing virtual suffixes and lookup keys; they are not new commitments.
    pub const fn evaluated(&self) -> &ArgmaxEvaluation<G> {
        &self.evaluated
    }
    /// The selected-logit and masked-key relations over the original inputs.
    ///
    /// # Errors
    /// Gadget errors for inconsistent upstream shapes or engine geometry.
    pub fn relations<F: PolyField<Base = G>>(
        &self,
    ) -> Result<ArgmaxRelations<'_, F>, ScoringError> {
        Ok(self.scorer.gadget.relations(
            &self.inputs.logits,
            &self.inputs.next_token,
            &self.maximum,
            &self.evaluated,
        )?)
    }
    /// The existing gadget's public suffix weights and canonical Range table.
    /// All returned proof relations still owe the suffix reduction and lookups.
    pub const fn gadget(&self) -> &Argmax<G> {
        &self.scorer.gadget
    }
}
fn prepare<'s, 'i>(
    request: ExactRequest<'s>,
    index: usize,
    inputs: &'i GreedyInputs,
    maximum: Option<G>,
) -> Result<GreedyRow<'s, 'i>, ScoringError> {
    let sources = request
        .row(index)?
        .ok_or_else(|| usage("a proof row is a real output"))?;
    let scorer = request.scorer;
    let padded = scorer.vocab.next_power_of_two();
    if inputs.logits.len() != padded || inputs.next_token.indicator().len() != padded {
        return Err(usage(
            "logits and the original indicator have the padded vocabulary length",
        ));
    }
    let honest = scorer.gadget.witness(&inputs.logits, true)?;
    let maximum = match maximum {
        None => honest,
        Some(value) => {
            let mut column = SecretBuf::try_with_capacity(1).map_err(|_| ScoringError::Memory)?;
            column.push(value);
            ArgmaxWitness::from_column(column)
        }
    };
    let evaluated = scorer
        .gadget
        .evaluate(&inputs.logits, &inputs.next_token, &maximum)?;
    let mut sums = SecretBuf::filled(G::ZERO, 2);
    let mut valid = true;
    let mut keys_in_range = true;
    for (v, ((&h, &ell), &key)) in inputs
        .next_token
        .indicator()
        .iter()
        .zip(inputs.logits.iter())
        .zip(evaluated.keys.iter())
        .enumerate()
    {
        valid &= (h == G::ZERO) | (h == G::ONE);
        valid &= (v < scorer.vocab) | (h == G::ZERO);
        sums[0] += h;
        sums[1] += h * ell;
        keys_in_range &= key.to_canonical_u64() < 1 << 29;
    }
    valid &= sums[0] == G::ONE;
    if !valid {
        return Err(GadgetError::Input {
            rule: "the original next-token indicator is OneHot over the real vocabulary",
        }
        .into());
    }
    let matching = keys_in_range & (sums[1] == maximum.maximum()[0]);
    Ok(GreedyRow {
        scorer,
        inputs,
        sources,
        maximum,
        evaluated,
        matching,
    })
}
