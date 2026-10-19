// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SoftmaxRow's causal maximum certificate and reciprocal bracket (RFC-0009 Lemmas 5–6).
//! A row has a public valid prefix: `q + 1` entries for a real query, zero for a padded row,
//! layer or head (RFC-0030, RFC-0035). The driver derives that prefix from public dimensions.
//! Positions are in score-tensor order; advice's causal-block permutation is a layout concern.

use core::{fmt, hint::select_unpredictable};

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

use crate::{Affine, Committed, GadgetError, Lookups, NonNeg, Term};

/// Which reciprocal-bracket slack a limb belongs to (RFC-0009 relation 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slack {
    /// `a = ψ·2^{2σ} − R·s`.
    A,
    /// `b = s − ψ − a`.
    B,
}

/// Committed columns, in attention-tile order (RFC-0041).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoftmaxColumn {
    /// One exponential per score position; zero at every masked position.
    Exponentials,
    /// One reciprocal per row.
    Reciprocal,
    /// One causal maximum per row.
    Maximum,
    /// A committed NonNeg limb, low to high, A before B. The top limb is virtual.
    Limb(Slack, u8),
}

/// Virtual columns and zero-check targets of the certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoftmaxValue {
    /// `κ = C·(c − x) + (1 − C)·K_0`, the lookup key.
    Key,
    /// `L = C·(c − x) + 1 − C`, the product-tree leaf.
    Leaf,
    /// `s = Σ_i e_i`, including masked positions, whose e must be zero.
    Sum,
    /// A reciprocal-bracket slack.
    Slack(Slack),
    /// Zero-check `(1 − ψ)·c = 0`.
    PaddingMaximum,
    /// Zero-check `(1 − ψ)·R = 0`.
    PaddingReciprocal,
}

/// A factor in a pointwise relation or an affine lookup description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoftmaxTerm {
    /// The virtual score, at a position.
    Score,
    /// A committed value; row columns broadcast over key positions where needed.
    Column(SoftmaxColumn),
    /// A virtual value, separately bound by its relation.
    Virtual(SoftmaxValue),
    /// The public causal selector C at a position.
    Causal,
    /// The public row selector ψ.
    RowSelector,
}

/// A sum of pointwise products, with eq giving one additional degree in a sumcheck.
///
/// For Sum, sum the exponential over key positions. For every other target, apply the
/// expression pointwise. Padding targets equal zero. Other targets equal their virtual column.
/// **Never** replace a product's multilinear extension by a product of extensions (RFC-0030).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftmaxRelation<F> {
    /// What the expression defines, or the padding condition it zero-checks.
    pub target: SoftmaxValue,
    /// `(coefficient, factors)`, an empty factor list being the constant one.
    pub terms: Vec<(F, Vec<SoftmaxTerm>)>,
}

impl<F: PrimeField> SoftmaxRelation<F> {
    /// Evaluate at one Boolean vertex, or evaluate the relation's polynomial at a sumcheck's
    /// final point. This is not its target's multilinear extension away from Boolean vertices.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(SoftmaxTerm) -> E) -> E {
        self.terms
            .iter()
            .fold(E::ZERO, |sum, (coefficient, factors)| {
                sum + factors
                    .iter()
                    .fold(E::from(*coefficient), |p, &t| p * value(t))
            })
    }
}

/// One Exp or Range lookup. Affine terms may name virtual columns with separate obligations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftmaxLookup<F> {
    /// The canonical table id.
    pub table: TableId,
    /// Its key, affine in virtual and committed columns.
    pub key: Affine<F, SoftmaxTerm>,
    /// Its value, absent for Range.
    pub value: Option<Affine<F, SoftmaxTerm>>,
    /// True only for Exp: one tuple per padded score position; otherwise one per row.
    pub per_position: bool,
}

/// All obligations of the gadget. The product forest additionally has public root `1 − ψ`.
/// The driver must prove every relation and lookup and bind them to the same committed columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftmaxConstraints<F> {
    /// The advice columns, in RFC-0041 order.
    pub committed: Vec<SoftmaxColumn>,
    /// Lookup key, product leaf, row sum, A, B, then the two padding zero-checks.
    pub relations: Vec<SoftmaxRelation<F>>,
    /// Exp, reciprocal Range, A's limbs, then B's limbs (low limbs before top).
    pub lookups: Vec<SoftmaxLookup<F>>,
}

/// The certificate's public parameters and regenerated tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftmaxRow {
    exp: GeneratedTable,
    reciprocal: GeneratedTable,
    non_neg: NonNeg,
    sigma: u8,
    max_context: usize,
}

impl SoftmaxRow {
    /// Build from an Exp table and the reciprocal/NonNeg Range specifications in `tables`.
    /// `max_context` is the model's public bound, not a prover-supplied row length.
    ///
    /// # Errors
    /// Shape unless `1 ≤ max_context ≤ 2^17`, the exact Lemma 6 bound
    /// `(2·max_context − 1)·2^{2σ} < P` holds, limbs have 1–32 bits, and the required tables
    /// exist. Table if Exp's last key is nonzero. Spec for a specification that fails regeneration.
    pub fn new(
        exp: TableId,
        limb_bits: u8,
        max_context: usize,
        tables: &[TableSpec],
    ) -> Result<Self, GadgetError> {
        let spec = tables
            .iter()
            .find(|t| t.id == exp)
            .ok_or(GadgetError::Table {
                table: exp,
                rule: "SoftmaxRow's Exp specification exists",
            })?;
        let TableKind::Exp { sigma, .. } = spec.kind else {
            return Err(GadgetError::Table {
                table: exp,
                rule: "SoftmaxRow reads Exp",
            });
        };
        let shape_error = GadgetError::Shape {
            rule: "SoftmaxRow needs 1 ≤ max_context ≤ 2^17, (2·max_context−1)·2^{2σ} < P and 1 ≤ L ≤ 32",
        };
        if !(1..=1 << 17).contains(&max_context) || sigma > 31 || !(1..=32).contains(&limb_bits) {
            return Err(shape_error);
        }
        // Public bounds above make this at most 2^80, within u128.
        let bound = (2 * max_context as u128 - 1) * (1u128 << (2 * sigma));
        if bound >= u128::from(Goldilocks::MODULUS) {
            return Err(shape_error);
        }
        let exp = GeneratedTable::from_spec(spec)
            .map_err(|error| GadgetError::Spec { table: exp, error })?;
        if exp
            .row(exp.row_count() - 1)
            .is_none_or(|(_, value)| value != Goldilocks::ZERO)
        {
            return Err(GadgetError::Table {
                table: spec.id,
                rule: "SoftmaxRow's last Exp key maps to zero (RFC-0030)",
            });
        }
        let kind = TableKind::Range { bits: sigma + 1 };
        let spec = tables
            .iter()
            .find(|t| t.kind == kind)
            .ok_or(GadgetError::Shape {
                rule: "SoftmaxRow needs Range { σ+1 } for R",
            })?;
        let reciprocal = GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec {
            table: spec.id,
            error,
        })?;
        let non_neg = NonNeg::new(sigma + 18, limb_bits, tables)?;
        Ok(Self {
            exp,
            reciprocal,
            non_neg,
            sigma,
            max_context,
        })
    }

    /// Required Range definitions in first-use order, deduplicated. The caller interns these
    /// after the lowering's tables in gadget order (RFC-0006); Exp is already among those tables.
    ///
    /// # Errors
    /// Shape unless σ ≤ 31 and 1 ≤ L ≤ 32. The constructor checks the field/context bound too.
    pub fn range_kinds(sigma: u8, limb_bits: u8) -> Result<Vec<TableKind>, GadgetError> {
        if sigma > 31 || !(1..=32).contains(&limb_bits) {
            return Err(GadgetError::Shape {
                rule: "SoftmaxRow's σ ≤ 31 and 1 ≤ L ≤ 32",
            });
        }
        let k = sigma + 18;
        let n = k.div_ceil(limb_bits);
        let mut bits = vec![sigma + 1];
        if n > 1 && !bits.contains(&limb_bits) {
            bits.push(limb_bits);
        }
        let top = k - (n - 1) * limb_bits;
        if !bits.contains(&top) {
            bits.push(top);
        }
        Ok(bits
            .into_iter()
            .map(|bits| TableKind::Range { bits })
            .collect())
    }

    /// Tables in first-use order, each canonical id once. Repeated NonNeg limbs share tables.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        let mut tables = vec![&self.exp, &self.reciprocal];
        for t in self.non_neg.tables() {
            if tables.iter().all(|old| old.spec().id != t.spec().id) {
                tables.push(t);
            }
        }
        tables
    }

    /// The relational bound `R·Σe ≤ 2^{2σ}` emitted by Lemma 6. It applies only after all
    /// obligations in [`constraints`](Self::constraints), including the root, have been proved.
    pub fn mass_bound(&self) -> u64 {
        1u64 << (2 * self.sigma)
    }

    /// Explicit degree-3 reductions and row lookup forms (RFC-0009, RFC-0030, RFC-0035).
    pub fn constraints<F: PrimeField>(&self) -> SoftmaxConstraints<F> {
        use SoftmaxColumn::{Exponentials, Maximum, Reciprocal};
        use SoftmaxTerm::{Causal as C, Column as Col, RowSelector as Psi, Virtual as V};
        use SoftmaxValue::{Key, Leaf, PaddingMaximum, PaddingReciprocal, Slack as S, Sum};
        let one = F::ONE;
        let k0 = F::from_u64_reduced(self.exp.row_count() - 1);
        let relation = |target, terms| SoftmaxRelation { target, terms };
        let product_terms = |constant: F| {
            vec![
                (one, vec![C, Col(Maximum)]),
                (-one, vec![C, SoftmaxTerm::Score]),
                (constant, vec![]),
                (-constant, vec![C]),
            ]
        };
        let relations = vec![
            relation(Key, product_terms(k0)),
            relation(Leaf, product_terms(one)),
            relation(Sum, vec![(one, vec![Col(Exponentials)])]),
            relation(
                S(Slack::A),
                vec![
                    (F::from_u64_reduced(self.mass_bound()), vec![Psi]),
                    (-one, vec![Col(Reciprocal), V(Sum)]),
                ],
            ),
            relation(
                S(Slack::B),
                vec![
                    (one, vec![V(Sum)]),
                    (-one, vec![Psi]),
                    (-one, vec![V(S(Slack::A))]),
                ],
            ),
            relation(
                PaddingMaximum,
                vec![(one, vec![Col(Maximum)]), (-one, vec![Psi, Col(Maximum)])],
            ),
            relation(
                PaddingReciprocal,
                vec![
                    (one, vec![Col(Reciprocal)]),
                    (-one, vec![Psi, Col(Reciprocal)]),
                ],
            ),
        ];
        let affine = |term| Affine {
            constant: F::ZERO,
            terms: vec![(one, term)],
        };
        let mut lookups = vec![
            SoftmaxLookup {
                table: self.exp.spec().id,
                key: affine(V(Key)),
                value: Some(affine(Col(Exponentials))),
                per_position: true,
            },
            SoftmaxLookup {
                table: self.reciprocal.spec().id,
                key: affine(Col(Reciprocal)),
                value: None,
                per_position: false,
            },
        ];
        let mut committed = vec![Exponentials, Reciprocal, Maximum];
        let non_neg = self.non_neg.constraints::<F>();
        for slack in [Slack::A, Slack::B] {
            committed.extend(non_neg.committed.iter().filter_map(|c| match c {
                Committed::Limb(j) => Some(SoftmaxColumn::Limb(slack, *j)),
                _ => None,
            }));
            for lookup in &non_neg.lookups {
                let terms = lookup
                    .key
                    .terms
                    .iter()
                    .map(|&(weight, term)| {
                        (
                            weight,
                            match term {
                                Term::Input => V(S(slack)),
                                Term::Committed(Committed::Limb(j)) => {
                                    Col(SoftmaxColumn::Limb(slack, j))
                                }
                                _ => unreachable!("NonNeg reads input and limbs only"),
                            },
                        )
                    })
                    .collect();
                lookups.push(SoftmaxLookup {
                    table: lookup.table,
                    key: Affine {
                        constant: lookup.key.constant,
                        terms,
                    },
                    value: None,
                    per_position: false,
                });
            }
        }
        SoftmaxConstraints {
            committed,
            relations,
            lookups,
        }
    }

    fn shape(&self, len: usize, valid: usize) -> Result<(), GadgetError> {
        if !len.is_power_of_two()
            || len > self.max_context.next_power_of_two()
            || valid > self.max_context
            || valid > len
        {
            return Err(GadgetError::Shape {
                rule: "a padded score row has power-of-two length ≤ pad(max_context) and a public valid prefix ≤ min(length, max_context)",
            });
        }
        Ok(())
    }

    /// One padded row's honest witness. `valid` is the public causal prefix length `q+1`, or
    /// zero for a padded row/layer/head. Scores are true integers with |x| ≤ 2^21 (INV-IR-2).
    ///
    /// Every committed column is a clearing buffer. Invalid inputs are reported only after
    /// the whole row is processed, without values or failing positions in the error (OBS-1).
    ///
    /// # Errors
    /// Shape for an invalid public row shape; Input for an out-of-domain valid score or a
    /// valid key beyond Exp's width. Required NonNeg tables were checked at construction.
    pub fn witness(&self, scores: &[i64], valid: usize) -> Result<SoftmaxWitness, GadgetError> {
        self.shape(scores.len(), valid)?;
        let mut maximum = -(1i64 << 21);
        let mut inside = true;
        for &x in &scores[..valid] {
            inside &= x.unsigned_abs() <= 1 << 21;
            maximum = select_unpredictable(x > maximum, x, maximum);
        }
        let maximum = if valid == 0 { 0 } else { maximum };
        let mut e = SecretBuf::with_capacity(scores.len());
        let mut sum = 0i64;
        for (i, &x) in scores.iter().enumerate() {
            // The branch is on a public selector. Widen subtraction before a malformed input.
            let key = if i < valid {
                i128::from(maximum) - i128::from(x)
            } else {
                i128::from(self.exp.row_count() - 1)
            };
            let fit = (0 <= key) & (key < i128::from(self.exp.row_count()));
            let key = select_unpredictable(fit, key, 0) as i64;
            let value = self.exp.lookup(key, 0);
            inside &= fit & value.is_some();
            let value = value.unwrap_or(0);
            sum += value; // ≤ max_context·2^σ < 2^49, also on an invalid input.
            e.push(value);
        }
        let psi = i64::from(valid != 0);
        let denominator = select_unpredictable(sum == 0, 1, sum) as u64;
        let reciprocal = (self.mass_bound() / denominator) as i64 * psi;
        let a = i128::from(psi) * i128::from(self.mass_bound())
            - i128::from(reciprocal) * i128::from(sum);
        let b = i128::from(sum) - i128::from(psi) - a;
        // On malformed keys the safe substitutes still give a bounded positive row sum.
        let a = a as i64;
        let b = b as i64;
        let mut tensors = vec![
            (SoftmaxColumn::Exponentials, e),
            (
                SoftmaxColumn::Reciprocal,
                core::iter::once(reciprocal).collect(),
            ),
            (SoftmaxColumn::Maximum, core::iter::once(maximum).collect()),
        ];
        for (slack, value) in [(Slack::A, a), (Slack::B, b)] {
            let witness = self.non_neg.witness(&[value])?;
            for j in 0..self.non_neg.limbs() - 1 {
                let j = j as u8;
                let values = witness.tensor(Committed::Limb(j)).unwrap_or(&[]);
                tensors.push((
                    SoftmaxColumn::Limb(slack, j),
                    values.iter().copied().collect(),
                ));
            }
        }
        if !inside {
            return Err(GadgetError::Input {
                rule: "SoftmaxRow's valid scores have |x| ≤ 2^21 and every valid key names an Exp row",
            });
        }
        Ok(SoftmaxWitness { tensors })
    }

    /// Evaluate any assignment, including an adversarial one, into its lookup columns, product
    /// leaves and two padding checks. This constructs obligations; it does not verify them.
    ///
    /// # Errors
    /// Shape for a malformed row; Shape for a missing, extra, duplicate or wrong-length witness
    /// column. No secret value decides whether an assignment can be evaluated.
    pub fn evaluate<B: PrimeField + PolyField<Base = B>>(
        &self,
        scores: &[i64],
        valid: usize,
        witness: &SoftmaxWitness,
    ) -> Result<SoftmaxEvaluation<B>, GadgetError> {
        self.shape(scores.len(), valid)?;
        let constraints = self.constraints::<B>();
        if witness.tensors.len() != constraints.committed.len()
            || constraints.committed.iter().any(|c| {
                witness.tensors.iter().filter(|(held, _)| held == c).count() != 1
                    || witness.tensor(*c).is_none_or(|v| {
                        v.len()
                            != if *c == SoftmaxColumn::Exponentials {
                                scores.len()
                            } else {
                                1
                            }
                    })
            })
        {
            return Err(GadgetError::Shape {
                rule: "SoftmaxRow's witness holds each committed column once at its public length",
            });
        }
        let col = |c, i| B::from_i64(witness.tensor(c).map_or(0, |v| v[i]));
        let sum = (0..scores.len()).fold(B::ZERO, |s, i| s + col(SoftmaxColumn::Exponentials, i));
        let psi = B::from_u64_reduced(u64::from(valid != 0));
        let a =
            psi * B::from_u64_reduced(self.mass_bound()) - col(SoftmaxColumn::Reciprocal, 0) * sum;
        let b = sum - psi - a;
        let mut leaves = SecretBuf::with_capacity(scores.len());
        let mut keys = SecretBuf::with_capacity(scores.len());
        let k0 = B::from_u64_reduced(self.exp.row_count() - 1);
        for (i, &score) in scores.iter().enumerate() {
            let c = B::from_u64_reduced(u64::from(i < valid));
            let key = col(SoftmaxColumn::Maximum, 0) - B::from_i64(score);
            leaves.push(c * key + B::ONE - c);
            keys.push(c * key + (B::ONE - c) * k0);
        }
        let value = |term, i| match term {
            SoftmaxTerm::Score => B::from_i64(scores[i]),
            SoftmaxTerm::Column(c) => col(
                c,
                if c == SoftmaxColumn::Exponentials {
                    i
                } else {
                    0
                },
            ),
            SoftmaxTerm::Virtual(SoftmaxValue::Key) => keys[i],
            SoftmaxTerm::Virtual(SoftmaxValue::Sum) => sum,
            SoftmaxTerm::Virtual(SoftmaxValue::Slack(Slack::A)) => a,
            SoftmaxTerm::Virtual(SoftmaxValue::Slack(Slack::B)) => b,
            SoftmaxTerm::Virtual(SoftmaxValue::Leaf) => leaves[i],
            SoftmaxTerm::Virtual(_) => B::ZERO, // padding targets are zero-checks, never operands
            SoftmaxTerm::Causal => B::from_u64_reduced(u64::from(i < valid)),
            SoftmaxTerm::RowSelector => psi,
        };
        let columns = constraints
            .lookups
            .iter()
            .map(|lookup| {
                let n = if lookup.per_position { scores.len() } else { 1 };
                let keys = (0..n)
                    .map(|i| lookup.key.evaluate(|t| value(t, i)))
                    .collect();
                let values = lookup
                    .value
                    .as_ref()
                    .map(|f| (0..n).map(|i| f.evaluate(|t| value(t, i))).collect());
                (lookup.table, keys, values)
            })
            .collect();
        let padding = [SoftmaxColumn::Maximum, SoftmaxColumn::Reciprocal]
            .iter()
            .map(|&c| (B::ONE - psi) * col(c, 0))
            .collect();
        Ok(SoftmaxEvaluation {
            lookups: Lookups::from_columns(columns)?,
            leaves,
            padding,
            root: B::ONE - psi,
        })
    }
}

/// One row's advice, retained only in clearing buffers; Debug shows names and lengths.
pub struct SoftmaxWitness {
    tensors: Vec<(SoftmaxColumn, SecretBuf<i64>)>,
}

impl SoftmaxWitness {
    /// An arbitrary assignment, for proof generation or adversarial testing. `evaluate` checks
    /// the public shape. The holder applies the secret access rules of [`crate::Witness::tensor`].
    pub fn from_tensors(tensors: Vec<(SoftmaxColumn, SecretBuf<i64>)>) -> Self {
        Self { tensors }
    }

    /// Borrow a secret-class column for computation only: no logs, errors, metrics or reports;
    /// no retained copies outside clearing buffers (OBS-1, OBS-6).
    pub fn tensor(&self, column: SoftmaxColumn) -> Option<&[i64]> {
        self.tensors
            .iter()
            .find(|(c, _)| *c == column)
            .map(|(_, v)| &v[..])
    }
}

impl fmt::Debug for SoftmaxWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.tensors.iter().map(|(c, v)| (c, v.len())))
            .finish()
    }
}

/// Obligations on an evaluated assignment. All secret-derived columns clear when dropped.
/// Check the lookups by LogUp, the forest against root, and both padding checks against zero;
/// all descriptions must be reduced to the committed columns by the later stages.
#[must_use]
pub struct SoftmaxEvaluation<B: Copy> {
    /// Exp and row-range lookups, grouped by canonical table id.
    pub lookups: Lookups<B>,
    /// Row-major product leaves for a forest of height log2(row length).
    pub leaves: SecretBuf<B>,
    /// The two padding-zero-check columns: maximum, then reciprocal.
    pub padding: SecretBuf<B>,
    /// Public product root: zero on a real row, one on a padded one.
    pub root: B,
}

impl<B: Copy> fmt::Debug for SoftmaxEvaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SoftmaxEvaluation")
            .field("lookups", &self.lookups)
            .field("leaves", &self.leaves.len())
            .field("padding", &self.padding.len())
            .finish_non_exhaustive()
    }
}
