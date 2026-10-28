// SPDX-License-Identifier: Apache-2.0 OR MIT

//! OneHot and Argmax (RFC-0009 Lemmas 8–9). Teacher forcing lends the next token's existing
//! indicator to Argmax; Argmax commits only its maximum, never another indicator.

use crate::{GadgetError, Lookups, SampledRelations};
use core::{fmt, hint::select_unpredictable};
use tessera_codec::{SecretBuf, Zeroize};
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::{PolyField, eq_table, less_than_points};
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input, SumcheckError, Term};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

fn shape(rule: &'static str) -> GadgetError {
    GadgetError::Shape { rule }
}
pub(crate) fn relation(_: SumcheckError) -> GadgetError {
    shape("the choice relation's public shape fits its sumcheck")
}

/// OneHot's public vocabulary and index column. Tokens must be true integers in `[0,vocab)`
/// on real rows; on padded rows the token and indicator are zero (RFC-0007, RFC-0035).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHot<B = Goldilocks> {
    vocab: usize,
    indices: Vec<B>,
    valid: Vec<B>,
}

impl<B: PrimeField> OneHot<B> {
    /// A v1 vocabulary, `1 ≤ vocab ≤ 2^18`, padded to a power of two.
    ///
    /// # Errors
    /// Shape for another vocabulary or a field too small to make the sum of Boolean entries
    /// an integer: the padded length must be strictly below the field modulus.
    pub fn new(vocab: usize) -> Result<Self, GadgetError> {
        if !(1..=1 << 18).contains(&vocab) || vocab.next_power_of_two() as u64 >= B::MODULUS {
            return Err(shape("OneHot needs 1 ≤ vocab ≤ 2^18 and pad(vocab) < P"));
        }
        Ok(Self {
            vocab,
            valid: (0..vocab.next_power_of_two())
                .map(|i| B::from_u64_reduced(u64::from(i < vocab)))
                .collect(),
            indices: (0..vocab.next_power_of_two())
                .map(|i| B::from_u64_reduced(i as u64))
                .collect(),
        })
    }

    /// The number of real vocabulary entries.
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// The padded indicator's length.
    pub fn padded_len(&self) -> usize {
        self.indices.len()
    }
    pub(crate) fn vars(&self) -> usize {
        self.padded_len().trailing_zeros() as usize
    }
    /// The public vocabulary indicator `C_v = [v < vocab]`, over the padded vocabulary.
    pub(crate) fn valid(&self) -> &[B] {
        &self.valid
    }

    /// Compute a clearing indicator. `real` is the public row selector ψ. Control flow and
    /// addresses depend only on public sizes; no secret token indexes an allocation.
    ///
    /// # Errors
    /// Input unless a real token lies in `[0,vocab)`, or a padded token is zero. Reported after
    /// all entries have been computed, without the token or its position in the error (OBS-1).
    pub fn witness(&self, token: i64, real: bool) -> Result<OneHotWitness<B>, GadgetError> {
        let inside = if real {
            (0 <= token) & (token < self.vocab as i64)
        } else {
            token == 0
        };
        let indicator = (0..self.padded_len())
            .map(|v| B::from_u64_reduced(u64::from(real & (token == v as i64))))
            .collect();
        if !inside {
            return Err(GadgetError::Input {
                rule: "a real token is in [0,vocab), a padded token is zero (RFC-0009 Lemma 8)",
            });
        }
        Ok(OneHotWitness { indicator })
    }

    /// Build the four proof obligations using the sumcheck engine (RFC-0010). Bind booleanity
    /// to zero at the stage's zero-check point, count to ψ, and index to the committed token.
    /// The vocabulary zero-check pins padding to zero, excluding padded token indices (#378).
    ///
    /// # Errors
    /// Shape for a witness of another padded vocabulary length, or an invalid engine shape.
    pub fn relations<'a, F: PolyField<Base = B>>(
        &'a self,
        witness: &'a OneHotWitness<B>,
    ) -> Result<OneHotRelations<'a, F>, GadgetError> {
        self.check(&witness.indicator)?;
        let h = Input::Operand(&witness.indicator);
        let contracted = ElementwiseShape {
            vars: self.vars(),
            summed_vars: self.vars(),
        };
        Ok(OneHotRelations {
            booleanity: Elementwise::booleanity(h, self.vars()).map_err(relation)?,
            vocabulary: Elementwise::padding(h, &self.valid, self.vars()).map_err(relation)?,
            count: Elementwise::new(contracted, vec![h], vec![Term::new(F::ONE, vec![0])])
                .map_err(relation)?,
            index: Elementwise::new(
                contracted,
                vec![h, Input::Public(&self.indices)],
                vec![Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
        })
    }

    pub(crate) fn check(&self, values: &[B]) -> Result<(), GadgetError> {
        if values.len() != self.padded_len() {
            return Err(shape("a choice column has pad(vocab) entries"));
        }
        Ok(())
    }
}

/// A clearing committed indicator. Its holder follows the secret-access rules of
/// [`crate::Witness::tensor`]. Debug shows only the length.
pub struct OneHotWitness<B: Copy> {
    indicator: SecretBuf<B>,
}
impl<B: Copy> OneHotWitness<B> {
    /// An arbitrary assignment, including an adversarial one. Relations check its public shape.
    pub fn from_column(indicator: SecretBuf<B>) -> Self {
        Self { indicator }
    }
    /// Borrow the secret indicator for computation; never log or retain an uncleared copy.
    pub fn indicator(&self) -> &[B] {
        &self.indicator
    }
}
impl<B: Copy> fmt::Debug for OneHotWitness<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneHotWitness")
            .field("len", &self.indicator.len())
            .finish()
    }
}

/// Four obligations on the same committed h. Prove each and consume its operand claims.
#[must_use]
pub struct OneHotRelations<'a, F: PolyField> {
    /// `Σ eq(z,v) h_v(1-h_v)=0`, degree3 except at a scalar vocabulary.
    pub booleanity: Elementwise<'a, F>,
    /// `Σ eq(z,v)·(1−[v<vocab])·h_v=0`, pinning vocabulary padding to zero (#378).
    pub vocabulary: Elementwise<'a, F>,
    /// `Σ h_v=ψ`, a contraction over every padded vocabulary coordinate.
    pub count: Elementwise<'a, F>,
    /// `Σ v h_v=token`, whose public index factor the verifier evaluates itself.
    pub index: Elementwise<'a, F>,
}

/// Greedy decoding with lowest-index ties (RFC-0009 Lemma 9). Its range table is Range(29),
/// and input logits are true integers with `|ℓ| < 2^28`. Vocabulary padding never joins the max.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Argmax<B = Goldilocks> {
    domain: OneHot<B>,
    range: GeneratedTable,
}

impl<B: PrimeField> Argmax<B> {
    /// Use the canonical selected Range(29) table. Its id is retained for the LogUp instance.
    ///
    /// # Errors
    /// OneHot's shape errors; Shape unless `P>2^30` excludes negative-key modular aliases;
    /// Table for another table kind, Spec for a specification that does not regenerate.
    pub fn new(vocab: usize, table: &TableSpec) -> Result<Self, GadgetError> {
        let domain = OneHot::new(vocab)?;
        if B::MODULUS <= 1 << 30 {
            return Err(shape("Argmax needs P > 2^30 for its integer key boxes"));
        }
        if table.kind != (TableKind::Range { bits: 29 }) {
            return Err(GadgetError::Table {
                table: table.id,
                rule: "Argmax reads Range { 29 }",
            });
        }
        let range = GeneratedTable::from_spec(table).map_err(|error| GadgetError::Spec {
            table: table.id,
            error,
        })?;
        Ok(Self { domain, range })
    }

    /// Construct an [`Argmax`] over a physical `Range { bits }` (`1 ≤ bits ≤ 29`) covering public
    /// logit span `max_span = hi − lo < 2^bits` (RFC-0053 'Public range and catalog').
    ///
    /// # Errors
    /// [`OneHot::new`]'s shape errors; [`GadgetError::Shape`] unless `P > 2^30`;
    /// [`GadgetError::Table`] if `table` is not a `Range { bits }` with `1 ≤ bits ≤ 29` and
    /// `max_span < 2^bits`; [`GadgetError::Spec`] if `table` fails regeneration.
    pub fn with_span(vocab: usize, max_span: u64, table: &TableSpec) -> Result<Self, GadgetError> {
        let domain = OneHot::new(vocab)?;
        if B::MODULUS <= 1 << 30 {
            return Err(shape("Argmax needs P > 2^30 for its integer key boxes"));
        }
        let TableKind::Range { bits } = table.kind else {
            return Err(GadgetError::Table {
                table: table.id,
                rule: "Argmax reads a Range table",
            });
        };
        if !(1..=29).contains(&bits) || max_span >= (1u64 << bits) {
            return Err(GadgetError::Table {
                table: table.id,
                rule: "Argmax's Range table covers the public logit span within 1..=29 bits",
            });
        }
        let range = GeneratedTable::from_spec(table).map_err(|error| GadgetError::Spec {
            table: table.id,
            error,
        })?;
        Ok(Self { domain, range })
    }

    /// The canonical Range table, evaluated by structured formulas rather than a dense allocation.
    pub fn table(&self) -> &GeneratedTable {
        &self.range
    }

    /// The vocabulary's [`OneHot`] indicator domain.
    pub fn domain(&self) -> &OneHot<B> {
        &self.domain
    }

    /// Branch-free lowest-index greedy choice `ŷ = min{v < vocab : ℓ_v = max_{w < vocab} ℓ_w}`
    /// and its maximum `m`, for bounded mode (RFC-0009 Lemma 9, RFC-0017).
    ///
    /// Scans all `pad(vocab)` entries before reporting any out-of-domain logit or span overflow
    /// (INV-ARCH-2, OBS-1).
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for another column length; [`GadgetError::Input`] unless active
    /// logits satisfy `|ℓ| < 2^28`, padding is zero, and every active key fits `table`.
    pub fn sample(
        &self,
        logits: &[B],
        real: bool,
    ) -> Result<(OneHotWitness<B>, ArgmaxWitness<B>), GadgetError> {
        self.domain.check(logits)?;
        let mut maximum = -(1i128 << 28);
        let mut best = 0i64;
        let mut inside = true;
        for (v, &logit) in logits.iter().enumerate() {
            let x = logit.to_centered_i128();
            if real && v < self.domain.vocab {
                inside &= x.unsigned_abs() < 1 << 28;
                let better = x > maximum;
                maximum = select_unpredictable(better, x, maximum);
                best = select_unpredictable(better, v as i64, best);
            } else {
                inside &= x == 0;
            }
        }
        let rows = i128::from(self.range.row_count());
        if real {
            for (v, &logit) in logits.iter().enumerate().take(self.domain.vocab) {
                let lt = i128::from((v as i64) < best);
                let key = maximum - logit.to_centered_i128() - lt;
                inside &= (0 <= key) & (key < rows);
            }
        }
        let token = if real { best } else { 0 };
        let mut max_val = if real { maximum as i64 } else { 0 };
        maximum.zeroize();
        best.zeroize();
        let indicator = self.domain.witness(token, real)?;
        let values = core::iter::once(B::from_i64(max_val)).collect();
        max_val.zeroize();
        if !inside {
            return Err(GadgetError::Input {
                rule: "real logits have |ℓ|<2^28 within Argmax's Range table; vocabulary and row padding are zero",
            });
        }
        Ok((indicator, ArgmaxWitness { maximum: values }))
    }

    /// The honest maximum over real vocabulary entries only. `real` is the public row selector.
    /// Argmax borrows the next-token indicator separately: it generates no replacement token.
    ///
    /// # Errors
    /// Shape for another column length. Input unless real logits have |ℓ|<2^28 within `table`'s
    /// span, padded vocabulary entries are zero, and every logit of a padded row is zero. Errors
    /// show no value.
    pub fn witness(&self, logits: &[B], real: bool) -> Result<ArgmaxWitness<B>, GadgetError> {
        self.domain.check(logits)?;
        let mut maximum = -(1i128 << 28);
        let mut best = 0i64;
        let mut inside = true;
        for (v, &logit) in logits.iter().enumerate() {
            let x = logit.to_centered_i128();
            if real && v < self.domain.vocab {
                inside &= x.unsigned_abs() < 1 << 28;
                let better = x > maximum;
                maximum = select_unpredictable(better, x, maximum);
                best = select_unpredictable(better, v as i64, best);
            } else {
                inside &= x == 0;
            }
        }
        let rows = i128::from(self.range.row_count());
        if real {
            for (v, &logit) in logits.iter().enumerate().take(self.domain.vocab) {
                let lt = i128::from((v as i64) < best);
                let key = maximum - logit.to_centered_i128() - lt;
                inside &= (0 <= key) & (key < rows);
            }
        }
        let mut max_val = if real { maximum as i64 } else { 0 };
        maximum.zeroize();
        best.zeroize();
        let values = core::iter::once(B::from_i64(max_val)).collect();
        max_val.zeroize();
        if !inside {
            return Err(GadgetError::Input {
                rule: "real logits have |ℓ|<2^28 within Argmax's Range table; vocabulary and row padding are zero",
            });
        }
        Ok(ArgmaxWitness { maximum: values })
    }

    /// Compute virtual suffixes and keys for any assignment. All values derived from h/logits
    /// remain clearing buffers. The caller proves OneHot on this very same borrowed h.
    ///
    /// # Errors
    /// Shape for another logit/indicator length or a maximum column other than one entry.
    pub fn evaluate(
        &self,
        logits: &[B],
        next_token: &OneHotWitness<B>,
        witness: &ArgmaxWitness<B>,
    ) -> Result<ArgmaxEvaluation<B>, GadgetError> {
        self.domain.check(logits)?;
        self.domain.check(next_token.indicator())?;
        if witness.maximum.len() != 1 {
            return Err(shape("Argmax commits one maximum per row"));
        }
        let mut suffix = SecretBuf::filled(B::ZERO, self.domain.padded_len());
        let mut sum = B::ZERO;
        for (v, &h) in next_token.indicator.iter().enumerate().rev() {
            suffix[v] = sum;
            sum += h;
        }
        let keys = logits
            .iter()
            .enumerate()
            .map(|(v, &x)| self.domain.valid[v] * (witness.maximum[0] - x - suffix[v]))
            .collect();
        Ok(ArgmaxEvaluation { suffix, keys })
    }

    /// Bind the selected-logit contraction to committed m, and the key reduction to LogUp's
    /// key claim. The suffix column has its separate linear reduction below. Vocabulary padding
    /// is Range's public row0: the key reduction is `C_v·(m−ℓ−lt)`, not a product of MLEs.
    ///
    /// # Errors
    /// Shape for any malformed public column length or an invalid engine shape.
    pub fn relations<'a, F: PolyField<Base = B>>(
        &'a self,
        logits: &'a [B],
        next_token: &'a OneHotWitness<B>,
        witness: &'a ArgmaxWitness<B>,
        evaluated: &'a ArgmaxEvaluation<B>,
    ) -> Result<ArgmaxRelations<'a, F>, GadgetError> {
        self.domain.check(logits)?;
        self.domain.check(next_token.indicator())?;
        self.domain.check(&evaluated.suffix)?;
        self.domain.check(&evaluated.keys)?;
        if witness.maximum.len() != 1 {
            return Err(shape("Argmax commits one maximum per row"));
        }
        let vars = self.domain.vars();
        Ok(ArgmaxRelations {
            selected: Elementwise::new(
                ElementwiseShape {
                    vars,
                    summed_vars: vars,
                },
                vec![
                    Input::Operand(next_token.indicator()),
                    Input::Operand(logits),
                ],
                vec![Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
            key: Elementwise::new(
                ElementwiseShape::elementwise(vars),
                vec![
                    Input::Public(&self.domain.valid),
                    Input::Broadcast {
                        operand: &witness.maximum,
                        low_vars: vars,
                    },
                    Input::Operand(logits),
                    Input::Operand(&evaluated.suffix),
                ],
                vec![
                    Term::new(F::ONE, vec![0, 1]),
                    Term::new(-F::ONE, vec![0, 2]),
                    Term::new(-F::ONE, vec![0, 3]),
                ],
            )
            .map_err(relation)?,
        })
    }

    /// Public weights for reducing a suffix claim at rho to `Σ_v weight_v·h_v`:
    /// `weight_v=Σ_{i<v}eq(rho,i)`. Prove this dot product against the same committed h.
    /// The verifier evaluates the weights' MLE at the final point by [`suffix_kernel`].
    ///
    /// # Errors
    /// Shape unless rho has log2(pad(vocab)) coordinates.
    pub fn suffix_weights<F: PolyField<Base = B>>(&self, rho: &[F]) -> Result<Vec<F>, GadgetError> {
        suffix_table(rho, self.domain.vars())
    }

    /// Precompute the public vocabulary mask and row selector for a power-of-two batch of
    /// `padded_rows` rows (`real_rows ≤ padded_rows`).
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `padded_rows` is a positive power of two with
    /// `real_rows ≤ padded_rows`.
    pub fn selector(
        &self,
        padded_rows: usize,
        real_rows: usize,
    ) -> Result<BatchedArgmaxSelector<B>, GadgetError> {
        let padded_vocab = self.domain.padded_len();
        let total = padded_rows
            .checked_mul(padded_vocab)
            .ok_or_else(|| shape("Argmax batch selector fits usize"))?;
        if !padded_rows.is_power_of_two() || real_rows > padded_rows {
            return Err(shape(
                "Argmax batch needs power-of-two padded_rows ≥ real_rows",
            ));
        }
        let mut valid = Vec::with_capacity(total);
        let mut row_selector = Vec::with_capacity(padded_rows);
        for row in 0..padded_rows {
            valid.extend_from_slice(&self.domain.valid);
            row_selector.push(B::from_u64_reduced(u64::from(row < real_rows)));
        }
        Ok(BatchedArgmaxSelector {
            padded_rows,
            real_rows,
            valid,
            row_selector,
        })
    }

    /// Evaluate `padded_rows` rows' strict suffixes, masked keys and `Range` [`Lookups`] from flat
    /// row-major `logits`, `indicators` and per-row `maxima`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `padded_rows` is a positive power of two, `logits` and
    /// `indicators` have `padded_rows · pad(vocab)` entries, and `maxima` has `padded_rows`.
    pub fn evaluate_batch(
        &self,
        logits: &[B],
        indicators: &[B],
        maxima: &[B],
        padded_rows: usize,
    ) -> Result<BatchedArgmaxEvaluation<B>, GadgetError>
    where
        B: PolyField<Base = B>,
    {
        let padded_vocab = self.domain.padded_len();
        let total = padded_rows
            .checked_mul(padded_vocab)
            .ok_or_else(|| shape("Argmax batch evaluation fits usize"))?;
        if !padded_rows.is_power_of_two()
            || logits.len() != total
            || indicators.len() != total
            || maxima.len() != padded_rows
        {
            return Err(shape(
                "Argmax batch slices match padded_rows and pad(vocab)",
            ));
        }
        let mut suffix = SecretBuf::filled(B::ZERO, total);
        let mut keys = SecretBuf::with_capacity(total);
        for (row, &max) in maxima.iter().enumerate() {
            let base = row * padded_vocab;
            let mut sum = B::ZERO;
            for v in (0..padded_vocab).rev() {
                suffix[base + v] = sum;
                sum += indicators[base + v];
            }
            for v in 0..padded_vocab {
                keys.push(self.domain.valid[v] * (max - logits[base + v] - suffix[base + v]));
            }
        }
        let lookups = Lookups::from_columns(vec![(
            self.range.spec().id,
            keys.iter().copied().collect(),
            None,
        )])?;
        Ok(BatchedArgmaxEvaluation {
            lookups,
            suffix,
            keys,
        })
    }

    /// Batched `selected` contraction (`Σ_v ĥ(t,v)·ℓ(t,v) = m(t)`) and `key` reduction
    /// (`C_v·(m(t) − ℓ(t,v) − L(t,v))`) over `row_vars + vocab_vars`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for mismatched operand lengths or an unsupported engine shape.
    pub fn batched_relations<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a BatchedArgmaxSelector<B>,
        logits: &'a [B],
        indicators: &'a [B],
        maxima: &'a [B],
        evaluated: &'a BatchedArgmaxEvaluation<B>,
    ) -> Result<ArgmaxRelations<'a, F>, GadgetError> {
        let total = selector.valid.len();
        if logits.len() != total
            || indicators.len() != total
            || evaluated.suffix.len() != total
            || evaluated.keys.len() != total
            || maxima.len() != selector.padded_rows
        {
            return Err(shape("Argmax batched operands match selector's shape"));
        }
        let vocab_vars = self.domain.vars();
        let row_vars = selector.padded_rows.trailing_zeros() as usize;
        let vars = row_vars + vocab_vars;
        Ok(ArgmaxRelations {
            selected: Elementwise::new(
                ElementwiseShape {
                    vars,
                    summed_vars: vocab_vars,
                },
                vec![Input::Operand(indicators), Input::Operand(logits)],
                vec![Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
            key: Elementwise::new(
                ElementwiseShape::elementwise(vars),
                vec![
                    Input::Public(&selector.valid),
                    Input::Broadcast {
                        operand: maxima,
                        low_vars: vocab_vars,
                    },
                    Input::Operand(logits),
                    Input::Operand(&evaluated.suffix),
                ],
                vec![
                    Term::new(F::ONE, vec![0, 1]),
                    Term::new(-F::ONE, vec![0, 2]),
                    Term::new(-F::ONE, vec![0, 3]),
                ],
            )
            .map_err(relation)?,
        })
    }

    /// Bounded mode's batched `OneHot` obligations (booleanity, vocabulary, and row count `ψ_t`)
    /// on flat row-major greedy indicators `ĥ(t, v)` over `padded_rows` rows.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for mismatched operand lengths or an unsupported engine shape.
    pub fn batched_sampled_relations<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a BatchedArgmaxSelector<B>,
        indicators: &'a [B],
    ) -> Result<SampledRelations<'a, F>, GadgetError> {
        if indicators.len() != selector.valid.len() {
            return Err(shape(
                "Argmax batched sampled indicators match selector's shape",
            ));
        }
        let vocab_vars = self.domain.vars();
        let row_vars = selector.padded_rows.trailing_zeros() as usize;
        let vars = row_vars + vocab_vars;
        let h = Input::Operand(indicators);
        Ok(SampledRelations {
            booleanity: Elementwise::booleanity(h, vars).map_err(relation)?,
            vocabulary: Elementwise::padding(h, &selector.valid, vars).map_err(relation)?,
            count: Elementwise::new(
                ElementwiseShape {
                    vars,
                    summed_vars: vocab_vars,
                },
                vec![h],
                vec![Term::new(F::ONE, vec![0])],
            )
            .map_err(relation)?,
        })
    }

    /// Public weights reducing a batched suffix claim `L̃(ρ_vocab, ρ_row)` to
    /// `Σ_{t,v} W(t, v)·ĥ(t, v)`, where `W(t, v) = eq(ρ_row, t)·Σ_{i < v} eq(ρ_vocab, i)`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `rho` has `row_vars + log2(pad(vocab))` coordinates for a
    /// power-of-two `padded_rows`.
    pub fn batched_suffix_weights<F: PolyField<Base = B>>(
        &self,
        rho: &[F],
        padded_rows: usize,
    ) -> Result<Vec<F>, GadgetError> {
        if !padded_rows.is_power_of_two() {
            return Err(shape("Argmax batch has power-of-two padded_rows"));
        }
        let vocab_vars = self.domain.vars();
        let row_vars = padded_rows.trailing_zeros() as usize;
        if rho.len() != vocab_vars + row_vars {
            return Err(shape(
                "a batched suffix claim has log2(pad(vocab)) + log2(padded_rows) coordinates",
            ));
        }
        let v_weights = suffix_table(&rho[..vocab_vars], vocab_vars)?;
        let r_weights = eq_table(&rho[vocab_vars..])
            .map_err(|_| shape("the public row point fits the polynomial engine"))?;
        let mut out = Vec::with_capacity(padded_rows * v_weights.len());
        for rw in r_weights {
            for &vw in &v_weights {
                out.push(rw * vw);
            }
        }
        Ok(out)
    }
}

/// `w_v = Σ_{i<v} eq(rho, i)` over `2^vars` entries: the weights of a strict suffix's claim at
/// `rho`, shared by `Argmax` and `InverseCdf`.
pub(crate) fn suffix_table<F: PolyField>(rho: &[F], vars: usize) -> Result<Vec<F>, GadgetError> {
    if rho.len() != vars {
        return Err(shape("a suffix claim has log2(pad(vocab)) coordinates"));
    }
    let mut sum = F::ZERO;
    Ok(eq_table(rho)
        .map_err(|_| shape("the public suffix point fits the polynomial engine"))?
        .into_iter()
        .map(|w| {
            let before = sum;
            sum += w;
            before
        })
        .collect())
}

/// The public triangular kernel `Σ_{i<v} eq(rho,i)eq(point,v)`, in O(log vocab) time.
/// Coordinates are lowest bit first, as in RFC-0010. This binds the virtual suffix to h.
///
/// # Errors
/// Shape unless the two points have equal lengths, at most18 coordinates.
pub fn suffix_kernel<F: PolyField>(rho: &[F], point: &[F]) -> Result<F, GadgetError> {
    if rho.len() > 18 {
        return Err(shape(
            "suffix kernel points have the same length, at most18",
        ));
    }
    less_than_points(rho, point)
        .map_err(|_| shape("the suffix kernel's public points have equal lengths"))
}

/// The sole committed Argmax column. Debug exposes only its length.
pub struct ArgmaxWitness<B: Copy> {
    maximum: SecretBuf<B>,
}
impl<B: Copy> ArgmaxWitness<B> {
    /// Any assignment, including an adversarial one; public shape is checked by evaluate.
    pub fn from_column(maximum: SecretBuf<B>) -> Self {
        Self { maximum }
    }
    /// Borrow the secret maximum for computation only, under [`crate::Witness::tensor`]'s rules.
    pub fn maximum(&self) -> &[B] {
        &self.maximum
    }
}
impl<B: Copy> fmt::Debug for ArgmaxWitness<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArgmaxWitness")
            .field("len", &self.maximum.len())
            .finish()
    }
}

/// Virtual columns, bound by the key relation and suffix dot-product reduction.
#[must_use]
pub struct ArgmaxEvaluation<B: Copy> {
    /// `Σ_{w>v}h_w`, over the full padded vocabulary.
    pub suffix: SecretBuf<B>,
    /// `C_v·(m−ℓ_v−lt_v)`, with public zero keys at vocabulary padding.
    pub keys: SecretBuf<B>,
}
impl<B: Copy> fmt::Debug for ArgmaxEvaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArgmaxEvaluation")
            .field("len", &self.keys.len())
            .finish_non_exhaustive()
    }
}

/// The two nonlinear/linear reductions. The range lookup and suffix dot product are additional
/// obligations; this object alone does not certify a choice.
#[must_use]
pub struct ArgmaxRelations<'a, F: PolyField> {
    /// `Σ h_v ℓ_v=m`, over the same h certified by OneHot.
    pub selected: Elementwise<'a, F>,
    /// The pointwise masked key expression, with its claim coming from the Range lookup.
    pub key: Elementwise<'a, F>,
}

/// Public vocabulary mask and row selector over `padded_rows` rows for batched [`Argmax`]
/// reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedArgmaxSelector<B> {
    padded_rows: usize,
    real_rows: usize,
    valid: Vec<B>,
    row_selector: Vec<B>,
}

impl<B> BatchedArgmaxSelector<B> {
    /// Padded row count `R = 2^{row_vars}`.
    pub const fn padded_rows(&self) -> usize {
        self.padded_rows
    }

    /// Active row count `real_rows ≤ padded_rows`.
    pub const fn real_rows(&self) -> usize {
        self.real_rows
    }

    /// Public row selector `ψ_t = [t < real_rows]`.
    pub fn row_selector(&self) -> &[B] {
        &self.row_selector
    }
}

/// Evaluated virtual suffixes, masked keys and `Range` [`Lookups`] across `padded_rows` rows.
#[must_use]
pub struct BatchedArgmaxEvaluation<B: Copy> {
    /// `Range` lookup instance for `keys`.
    pub lookups: Lookups<B>,
    /// Flat row-major strict suffixes `L(t, v) = Σ_{w > v} ĥ(t, w)`.
    pub suffix: SecretBuf<B>,
    /// Flat row-major masked keys `C_v·(m(t) − ℓ(t, v) − L(t, v))`.
    pub keys: SecretBuf<B>,
}

impl<B: Copy> fmt::Debug for BatchedArgmaxEvaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchedArgmaxEvaluation")
            .field("lookups", &self.lookups)
            .field("keys", &self.keys.len())
            .finish_non_exhaustive()
    }
}
