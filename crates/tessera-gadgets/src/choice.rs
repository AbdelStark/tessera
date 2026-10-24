// SPDX-License-Identifier: Apache-2.0 OR MIT

//! OneHot and Argmax (RFC-0009 Lemmas 8–9). Teacher forcing lends the next token's existing
//! indicator to Argmax; Argmax commits only its maximum, never another indicator.

use crate::GadgetError;
use core::{fmt, hint::select_unpredictable};
use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::{PolyField, eq_table, less_than_points};
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input, SumcheckError, Term};
use tessera_tables::{GeneratedTable, TableKind, TableSpec};

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

    /// The canonical Range(29), evaluated by structured formulas rather than a dense allocation.
    pub fn table(&self) -> &GeneratedTable {
        &self.range
    }

    #[cfg(debug_assertions)]
    pub(crate) fn domain(&self) -> &OneHot<B> {
        &self.domain
    }

    /// The honest maximum over real vocabulary entries only. `real` is the public row selector.
    /// Argmax borrows the next-token indicator separately: it generates no replacement token.
    ///
    /// # Errors
    /// Shape for another column length. Input unless real logits have |ℓ|<2^28, padded
    /// vocabulary entries are zero, and every logit of a padded row is zero. Errors show no value.
    pub fn witness(&self, logits: &[B], real: bool) -> Result<ArgmaxWitness<B>, GadgetError> {
        self.domain.check(logits)?;
        let mut maximum = -(1i128 << 28);
        let mut inside = true;
        for (v, &logit) in logits.iter().enumerate() {
            let x = logit.to_centered_i128();
            if real && v < self.domain.vocab {
                inside &= x.unsigned_abs() < 1 << 28;
                maximum = select_unpredictable(x > maximum, x, maximum);
            } else {
                inside &= x == 0;
            }
        }
        let maximum = if real { maximum } else { 0 };
        let values = core::iter::once(B::from_i64(maximum as i64)).collect();
        if !inside {
            return Err(GadgetError::Input {
                rule: "real logits have |ℓ|<2^28; vocabulary and row padding are zero",
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
