// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vocabulary maximum certificate (RFC-0009 Lemma 5, RFC-0017 'Vocabulary softmax', RFC-0030,
//! RFC-0035).
//!
//! Unlike [`crate::SoftmaxRow`], vocabulary decoding (`Temperature` and `Mixture`) never divides
//! by the partition sum `Z_t`: [`crate::InverseCdf`] multiplies by `Z_t` instead, and `Mixture`
//! bounds `log2 Z_t` with `Pow2` and `Exp2Frac` (RFC-0017 'Vocabulary softmax'). So a vocabulary
//! row commits only its maximum `c_t` and its exponentials `e_{t,v}`, with no reciprocal `R` and
//! no bracket slacks `(a, b)`.
//!
//! On row `t` with public row selector `ψ_t ∈ {0, 1}` and vocabulary indicator `V_v = [v < vocab]`
//! over the padded vocabulary `[0, pad(vocab))`, let `C_{t,v} = ψ_t · V_v` and
//! `K_0 = 2^{key_bits} − 1`. Given true-integer logits `|ℓ_{t,v}| < 2^28` on active entries and
//! `0` on vocabulary and row padding:
//!
//! 1. `(κ_{t,v}, e_{t,v}) ∈ Exp_T` with `κ_{t,v} = C_{t,v}·(c_t − ℓ_{t,v}) + (1 − C_{t,v})·K_0`,
//!    where `Exp_T[K_0] = 0` pins masked and padded exponentials to `0` (RFC-0030);
//! 2. the product-tree leaves `L_{t,v} = C_{t,v}·(c_t − ℓ_{t,v}) + 1 − C_{t,v}` multiply to the
//!    public row root `1 − ψ_t`;
//! 3. the padding zero-check `(1 − ψ_t)·c_t = 0` pins a padded row's maximum to `0` (RFC-0035).
//!
//! Over a prime field with `P > 2^30`, `P ≥ 2^{key_bits} + 2^29`, and `2^σ ≤ (P − 1) / 2`, root `0`
//! on a real row (`ψ_t = 1`) forces `c_t = ℓ_{t,v*}` for some `v* < vocab`, because `L_{t,v} = 1`
//! on `v ≥ vocab`. Thus `|c_t| < 2^28`, so `c_t − ℓ_{t,v} ∈ (−2^29, 2^29)` cannot wrap into
//! `[0, K_0]` modulo `P` (`P − (2^29 − 2) ≥ 2^{key_bits} + 2 > K_0`), every lookup key
//! `κ_{t,v} ∈ [0, K_0]` forces `c_t ≥ ℓ_{t,v}`, and `c_t = max_{v < vocab} ℓ_{t,v}` with
//! `e_{t,v*} = Exp_T[0] = 2^σ ∈ [1, (P − 1) / 2]` uniquely (RFC-0009 Lemma 0).

use core::{fmt, hint::select_unpredictable};

use tessera_codec::{SecretBuf, Zeroize};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input, Term as SumcheckTerm};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::choice::{OneHot, relation};
use crate::{Affine, GadgetError, Lookup, Lookups};

/// Committed columns of the vocabulary maximum certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VocabColumn {
    /// One exponential per padded vocabulary entry; zero on vocabulary and row padding.
    Exponentials,
    /// One maximum per row; zero on padded rows.
    Maximum,
}

/// Virtual columns and zero-check targets of the vocabulary maximum certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VocabValue {
    /// `κ = C·(c − ℓ) + (1 − C)·K_0`, the `Exp_T` lookup key.
    Key,
    /// `L = C·(c − ℓ) + 1 − C`, the product-tree leaf.
    Leaf,
    /// Zero-check `(1 − ψ)·c = 0` on padded rows.
    PaddingMaximum,
}

/// A factor in a vocabulary certificate relation or affine lookup form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VocabTerm {
    /// The input logit `ℓ_{t,v}`.
    Logit,
    /// A committed column; [`VocabColumn::Maximum`] broadcasts across vocabulary positions.
    Column(VocabColumn),
    /// A virtual column bound by its relation.
    Virtual(VocabValue),
    /// The combined active-entry selector `C_{t,v} = ψ_t · [v < vocab]`.
    Active,
    /// The public row selector `ψ_t`.
    RowSelector,
}

/// A pointwise polynomial relation of [`VocabMax`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabRelation<F> {
    /// Virtual column defined by the expression, or the padding condition it zero-checks.
    pub target: VocabValue,
    /// `(coefficient, factors)`, an empty factor list denoting the constant one.
    pub terms: Vec<(F, Vec<VocabTerm>)>,
}

impl<F: PrimeField> VocabRelation<F> {
    /// Evaluate the relation polynomial at one entry or at a sumcheck's final point.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(VocabTerm) -> E) -> E {
        self.terms
            .iter()
            .fold(E::ZERO, |sum, (coefficient, factors)| {
                sum + factors
                    .iter()
                    .fold(E::from(*coefficient), |p, &t| p * value(t))
            })
    }
}

/// All obligations of [`VocabMax`], alongside each row's product-tree root `1 − ψ_t`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "prove the Exp lookup, product-tree roots, key/leaf reductions and padding check"]
pub struct VocabConstraints<F> {
    /// Committed columns in canonical order: exponentials, then maximum.
    pub committed: Vec<VocabColumn>,
    /// Lookup key, product leaf, and maximum padding zero-check.
    pub relations: Vec<VocabRelation<F>>,
    /// The single `(κ, e) ∈ Exp_T` lookup per padded vocabulary position.
    pub lookup: Lookup<F, VocabTerm>,
}

/// Vocabulary maximum and exponential certificate for one `Exp_T` table (RFC-0017).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabMax<B = Goldilocks> {
    domain: OneHot<B>,
    exp: GeneratedTable,
    sigma: u8,
}

impl<B: PrimeField> VocabMax<B> {
    /// Build a vocabulary maximum certificate over `1 ≤ vocab ≤ 2^18` and regenerated `exp`.
    ///
    /// # Errors
    /// [`OneHot::new`]'s shape errors; [`GadgetError::Shape`] unless `P > 2^30`,
    /// `P ≥ 2^{key_bits} + 2^29`, `σ ≤ 32`, and `2^σ ≤ (P − 1) / 2`; [`GadgetError::Table`] if
    /// `exp` is not [`TableKind::Exp`] or its last key `K_0` does not map to zero (RFC-0030);
    /// [`GadgetError::Spec`] if `exp` fails regeneration.
    pub fn new(vocab: usize, exp: &TableSpec) -> Result<Self, GadgetError> {
        let domain = OneHot::new(vocab)?;
        if B::MODULUS <= 1 << 30 {
            return Err(GadgetError::Shape {
                rule: "VocabMax needs P > 2^30 so c − ℓ cannot wrap modulo P",
            });
        }
        let TableKind::Exp { sigma, .. } = exp.kind else {
            return Err(GadgetError::Table {
                table: exp.id,
                rule: "VocabMax reads an Exp table",
            });
        };
        if sigma > 32 || (1u128 << sigma) > (u128::from(B::MODULUS) - 1) / 2 {
            return Err(GadgetError::Shape {
                rule: "VocabMax needs σ ≤ 32 and 2^σ ≤ (P − 1) / 2 (RFC-0008, RFC-0009 Lemma 0)",
            });
        }
        let generated = GeneratedTable::from_spec(exp).map_err(|error| GadgetError::Spec {
            table: exp.id,
            error,
        })?;
        if u128::from(B::MODULUS) < u128::from(generated.row_count()) + (1u128 << 29) {
            return Err(GadgetError::Shape {
                rule: "VocabMax needs P ≥ 2^{key_bits} + 2^29 so negative c − ℓ cannot wrap into Exp_T",
            });
        }
        if generated
            .row(generated.row_count() - 1)
            .is_none_or(|(_, value)| value != Goldilocks::ZERO)
        {
            return Err(GadgetError::Table {
                table: exp.id,
                rule: "VocabMax's last Exp key maps to zero (RFC-0030)",
            });
        }
        Ok(Self {
            domain,
            exp: generated,
            sigma,
        })
    }

    /// Number of real vocabulary entries.
    pub fn vocab(&self) -> usize {
        self.domain.vocab()
    }

    /// Padded vocabulary length `pad(vocab)`.
    pub fn padded_vocab(&self) -> usize {
        self.domain.padded_len()
    }

    /// Precision `σ` of `Exp_T`, where `Exp_T[0] = 2^σ`.
    pub const fn sigma(&self) -> u8 {
        self.sigma
    }

    /// Regenerated `Exp_T` table.
    pub const fn table(&self) -> &GeneratedTable {
        &self.exp
    }

    /// Masked lookup key `K_0 = 2^{key_bits} − 1`, where `Exp_T[K_0] = 0`.
    pub fn masked_key(&self) -> u64 {
        self.exp.row_count() - 1
    }

    /// Explicit degree-3 reductions and `Exp_T` lookup form.
    pub fn constraints<F: PrimeField>(&self) -> VocabConstraints<F> {
        use VocabColumn::{Exponentials, Maximum};
        use VocabTerm::{Active as C, Column as Col, Logit, RowSelector as Psi, Virtual as V};
        use VocabValue::{Key, Leaf, PaddingMaximum};
        let one = F::ONE;
        let k0 = F::from_u64_reduced(self.masked_key());
        let product_terms = |constant: F| {
            vec![
                (one, vec![C, Col(Maximum)]),
                (-one, vec![C, Logit]),
                (constant, vec![]),
                (-constant, vec![C]),
            ]
        };
        let relations = vec![
            VocabRelation {
                target: Key,
                terms: product_terms(k0),
            },
            VocabRelation {
                target: Leaf,
                terms: product_terms(one),
            },
            VocabRelation {
                target: PaddingMaximum,
                terms: vec![(one, vec![Col(Maximum)]), (-one, vec![Psi, Col(Maximum)])],
            },
        ];
        let affine = |term| Affine {
            constant: F::ZERO,
            terms: vec![(one, term)],
        };
        VocabConstraints {
            committed: vec![Exponentials, Maximum],
            relations,
            lookup: Lookup {
                table: self.exp.spec().id,
                key: affine(V(Key)),
                value: Some(affine(Col(Exponentials))),
            },
        }
    }

    /// Honest witness for one padded vocabulary row; `real` is the public row selector `ψ`.
    ///
    /// Scans all `pad(vocab)` entries branch-free before reporting any out-of-domain logit or key
    /// beyond `Exp_T`'s width (OBS-1, INV-ARCH-2).
    ///
    /// # Errors
    /// [`GadgetError::Shape`] if `logits.len() != pad(vocab)`; [`GadgetError::Input`] unless real
    /// logits satisfy `|ℓ_v| < 2^28`, padded entries are zero, and every active `c − ℓ_v` lies in
    /// `[0, 2^{key_bits})`.
    pub fn witness(&self, logits: &[B], real: bool) -> Result<VocabWitness<B>, GadgetError> {
        self.domain.check(logits)?;
        let vocab = self.domain.vocab();
        let mut maximum = -(1i128 << 28);
        let mut inside = true;
        for (v, &logit) in logits.iter().enumerate() {
            let x = logit.to_centered_i128();
            if real && v < vocab {
                inside &= x.unsigned_abs() < 1 << 28;
                maximum = select_unpredictable(x > maximum, x, maximum);
            } else {
                inside &= x == 0;
            }
        }
        let mut row_max = if real { maximum } else { 0 };
        maximum.zeroize();
        let k0 = i128::from(self.masked_key());
        let rows = i128::from(self.exp.row_count());
        let mut exponentials = SecretBuf::with_capacity(logits.len());
        for (v, &logit) in logits.iter().enumerate() {
            let key = if real && v < vocab {
                row_max - logit.to_centered_i128()
            } else {
                k0
            };
            let fit = (0 <= key) & (key < rows);
            let safe_key = select_unpredictable(fit, key, 0) as i64;
            let value = self.exp.lookup(safe_key, 0);
            inside &= fit & value.is_some();
            exponentials.push(B::from_i64(value.unwrap_or(0)));
        }
        let mut max_buf = SecretBuf::with_capacity(1);
        max_buf.push(B::from_i64(row_max as i64));
        row_max.zeroize();
        if !inside {
            return Err(GadgetError::Input {
                rule: "real logits have |ℓ| < 2^28, padding is zero, and every active c − ℓ fits Exp_T",
            });
        }
        Ok(VocabWitness {
            maximum: max_buf,
            exponentials,
        })
    }

    /// Evaluate one row's assignment (honest or adversarial) into its lookup columns, product
    /// leaves, padding zero-check value and public root `1 − ψ`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] if column lengths do not match a single padded vocabulary row.
    pub fn evaluate(
        &self,
        logits: &[B],
        real: bool,
        witness: &VocabWitness<B>,
    ) -> Result<VocabEvaluation<B>, GadgetError>
    where
        B: PolyField<Base = B>,
    {
        self.evaluate_batch(logits, 1, usize::from(real), witness)
    }

    /// Evaluate a power-of-two batch of `padded_rows` rows (`real_rows ≤ padded_rows`) into flat
    /// row-major lookup columns, product leaves, padding checks and per-row roots.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `padded_rows` is a positive power of two,
    /// `real_rows ≤ padded_rows`, `logits` and `witness.exponentials` have
    /// `padded_rows · pad(vocab)` entries, and `witness.maximum` has `padded_rows` entries.
    pub fn evaluate_batch(
        &self,
        logits: &[B],
        padded_rows: usize,
        real_rows: usize,
        witness: &VocabWitness<B>,
    ) -> Result<VocabEvaluation<B>, GadgetError>
    where
        B: PolyField<Base = B>,
    {
        let padded_vocab = self.domain.padded_len();
        let total_entries = padded_rows
            .checked_mul(padded_vocab)
            .ok_or(GadgetError::Shape {
                rule: "VocabMax batch size fits usize",
            })?;
        if !padded_rows.is_power_of_two()
            || real_rows > padded_rows
            || logits.len() != total_entries
            || witness.exponentials.len() != total_entries
            || witness.maximum.len() != padded_rows
        {
            return Err(GadgetError::Shape {
                rule: "VocabMax batch columns match padded_rows and pad(vocab)",
            });
        }
        let vocab = self.domain.vocab();
        let k0 = B::from_u64_reduced(self.masked_key());
        let mut keys = SecretBuf::with_capacity(total_entries);
        let mut leaves = SecretBuf::with_capacity(total_entries);
        let mut padding = SecretBuf::with_capacity(padded_rows);
        let mut roots = Vec::with_capacity(padded_rows);
        for row in 0..padded_rows {
            let real = row < real_rows;
            let psi = B::from_u64_reduced(u64::from(real));
            let max = witness.maximum[row];
            padding.push((B::ONE - psi) * max);
            roots.push(B::ONE - psi);
            let base = row * padded_vocab;
            for v in 0..padded_vocab {
                let c = B::from_u64_reduced(u64::from(real && v < vocab));
                let diff = max - logits[base + v];
                keys.push(c * diff + (B::ONE - c) * k0);
                leaves.push(c * diff + B::ONE - c);
            }
        }
        let values = witness.exponentials.iter().copied().collect();
        let lookups = Lookups::from_columns(vec![(
            self.exp.spec().id,
            keys.iter().copied().collect(),
            Some(values),
        )])?;
        Ok(VocabEvaluation {
            lookups,
            keys,
            leaves,
            padding,
            roots,
        })
    }

    /// Precompute the public selectors for `padded_rows` rows (`real_rows ≤ padded_rows`).
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `padded_rows` is a positive power of two with
    /// `real_rows ≤ padded_rows`.
    pub fn selector(
        &self,
        padded_rows: usize,
        real_rows: usize,
    ) -> Result<VocabSelector<B>, GadgetError> {
        let padded_vocab = self.domain.padded_len();
        let total = padded_rows
            .checked_mul(padded_vocab)
            .ok_or(GadgetError::Shape {
                rule: "VocabMax selector size fits usize",
            })?;
        if !padded_rows.is_power_of_two() || real_rows > padded_rows {
            return Err(GadgetError::Shape {
                rule: "VocabMax needs power-of-two padded_rows ≥ real_rows",
            });
        }
        let vocab = self.domain.vocab();
        let k0 = B::from_u64_reduced(self.masked_key());
        let mut active = Vec::with_capacity(total);
        let mut inactive_k0 = Vec::with_capacity(total);
        let mut inactive_one = Vec::with_capacity(total);
        let mut row_selector = Vec::with_capacity(padded_rows);
        for row in 0..padded_rows {
            let real = row < real_rows;
            row_selector.push(B::from_u64_reduced(u64::from(real)));
            for v in 0..padded_vocab {
                let is_active = real && v < vocab;
                active.push(B::from_u64_reduced(u64::from(is_active)));
                inactive_k0.push(if is_active { B::ZERO } else { k0 });
                inactive_one.push(if is_active { B::ZERO } else { B::ONE });
            }
        }
        Ok(VocabSelector {
            padded_rows,
            real_rows,
            active,
            inactive_k0,
            inactive_one,
            row_selector,
        })
    }

    /// Build the sumcheck [`Elementwise`] relations for one row or a power-of-two batch of rows:
    /// - `key`: reduces a LogUp key claim on `κ` to `c` and `ℓ`;
    /// - `leaf`: reduces a product-forest leaf claim on `L` to `c` and `ℓ`;
    /// - `padding`: zero-checks `(1 − ψ_t)·c_t = 0`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] if `logits` or `witness` lengths do not match `selector`.
    pub fn relations<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a VocabSelector<B>,
        logits: &'a [B],
        witness: &'a VocabWitness<B>,
    ) -> Result<VocabRelations<'a, F>, GadgetError> {
        let padded_vocab = self.domain.padded_len();
        let total = selector.active.len();
        if logits.len() != total
            || witness.exponentials.len() != total
            || witness.maximum.len() != selector.padded_rows
        {
            return Err(GadgetError::Shape {
                rule: "VocabMax operands match the public selector's shape",
            });
        }
        let vocab_vars = self.domain.vars();
        let row_vars = selector.padded_rows.trailing_zeros() as usize;
        let vars = row_vars + vocab_vars;
        let shape = ElementwiseShape::elementwise(vars);
        let max_input = Input::Broadcast {
            operand: &witness.maximum,
            low_vars: vocab_vars,
        };
        let key = Elementwise::new(
            shape,
            vec![
                Input::Public(&selector.active),
                Input::Public(&selector.inactive_k0),
                max_input,
                Input::Operand(logits),
            ],
            vec![
                SumcheckTerm::new(F::ONE, vec![0, 2]),
                SumcheckTerm::new(-F::ONE, vec![0, 3]),
                SumcheckTerm::new(F::ONE, vec![1]),
            ],
        )
        .map_err(relation)?;
        let leaf = Elementwise::new(
            shape,
            vec![
                Input::Public(&selector.active),
                Input::Public(&selector.inactive_one),
                max_input,
                Input::Operand(logits),
            ],
            vec![
                SumcheckTerm::new(F::ONE, vec![0, 2]),
                SumcheckTerm::new(-F::ONE, vec![0, 3]),
                SumcheckTerm::new(F::ONE, vec![1]),
            ],
        )
        .map_err(relation)?;
        let padding = Elementwise::padding(
            Input::Operand(&witness.maximum),
            &selector.row_selector,
            row_vars,
        )
        .map_err(relation)?;
        debug_assert_eq!(padded_vocab, 1 << vocab_vars);
        Ok(VocabRelations { key, leaf, padding })
    }
}

/// Committed maximum and exponential columns in clearing storage (OBS-6).
pub struct VocabWitness<B: Copy> {
    maximum: SecretBuf<B>,
    exponentials: SecretBuf<B>,
}

impl<B: Copy> VocabWitness<B> {
    /// Construct from arbitrary clearing columns (for batch assembly or adversarial tests).
    pub fn from_columns(maximum: SecretBuf<B>, exponentials: SecretBuf<B>) -> Self {
        Self {
            maximum,
            exponentials,
        }
    }

    /// Borrow the committed maximum column (`1` entry per row); never log or copy outside
    /// clearing storage.
    pub fn maximum(&self) -> &[B] {
        &self.maximum
    }

    /// Borrow the committed exponential column (`pad(vocab)` entries per row); never log or copy
    /// outside clearing storage.
    pub fn exponentials(&self) -> &[B] {
        &self.exponentials
    }
}

impl<B: Copy> fmt::Debug for VocabWitness<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabWitness")
            .field("rows", &self.maximum.len())
            .field("exponentials", &self.exponentials.len())
            .finish()
    }
}

/// Evaluated lookup columns, product leaves, padding zero-check values and public roots.
#[must_use]
pub struct VocabEvaluation<B: Copy> {
    /// `(κ, e)` lookup instance on `Exp_T`.
    pub lookups: Lookups<B>,
    /// Flat row-major lookup keys `κ_{t,v}`.
    pub keys: SecretBuf<B>,
    /// Flat row-major product-tree leaves `L_{t,v}`.
    pub leaves: SecretBuf<B>,
    /// Per-row padding zero-check values `(1 − ψ_t)·c_t`.
    pub padding: SecretBuf<B>,
    /// Public per-row product-tree roots `1 − ψ_t`.
    pub roots: Vec<B>,
}

impl<B: Copy> fmt::Debug for VocabEvaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabEvaluation")
            .field("lookups", &self.lookups)
            .field("leaves", &self.leaves.len())
            .field("rows", &self.roots.len())
            .finish_non_exhaustive()
    }
}

/// Public selector tables for single-row or batched [`VocabMax::relations`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabSelector<B> {
    padded_rows: usize,
    real_rows: usize,
    active: Vec<B>,
    inactive_k0: Vec<B>,
    inactive_one: Vec<B>,
    row_selector: Vec<B>,
}

impl<B> VocabSelector<B> {
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

/// Sumcheck obligations of [`VocabMax`]. Pair with LogUp on [`VocabEvaluation::lookups`] and the
/// product forest against [`VocabEvaluation::roots`].
#[must_use]
pub struct VocabRelations<'a, F: PolyField> {
    /// `κ(t, v) = C(t,v)·c(t) − C(t,v)·ℓ(t,v) + (1 − C(t,v))·K_0`.
    pub key: Elementwise<'a, F>,
    /// `L(t, v) = C(t,v)·c(t) − C(t,v)·ℓ(t,v) + (1 − C(t,v))`.
    pub leaf: Elementwise<'a, F>,
    /// `0 = Σ_t eq(z, t)·(1 − ψ_t)·c_t`.
    pub padding: Elementwise<'a, F>,
}
