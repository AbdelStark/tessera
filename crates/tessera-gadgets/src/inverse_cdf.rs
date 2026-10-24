// SPDX-License-Identifier: Apache-2.0 OR MIT

//! InverseCdf, the temperature-sampling certificate (RFC-0009 Lemma 10), in RFC-0058's columns.
//!
//! One output token's row reads the vocabulary exponentials `e`, the draw `u` and an indicator
//! `ĥ`. Three vocabulary sums, `Z = Σ_v C_v·e_v`, `B = Σ_v e_v·L_v` with `L` the indicator's
//! strict suffix, and `S = Σ_v ĥ_v·e_v`, are virtual. The row commits the radix-`2^16` limbs of
//! `u`, `Z` and `B`, and two comparisons: `B·2^32 ≤ u·Z` and `u·Z + ψ ≤ (B + S)·2^32`. A padded
//! row's partition is pinned by `(1 − ψ)·Z = 0`. In bounded mode the gadget also proves OneHot's
//! booleanity, vocabulary and count relations on the sampled token's indicator.

use crate::choice::{OneHot, OneHotWitness, relation, suffix_table};
use crate::compare::array_order;
use crate::{
    Affine, BigIntCompare, Committed, CompareColumn, GadgetError, Lookup, NonNeg, Side, Term,
};
use core::{fmt, hint::select_unpredictable};
use tessera_codec::{SecretBuf, Zeroize};
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input};
use tessera_tables::{GeneratedTable, Table, TableSpec};

/// The radix of every decomposition and comparison (RFC-0058).
const LIMB_BITS: u8 = 16;
/// The draw's width: `u ∈ [0, 2^32)` (RFC-0017 'Sampler PRF').
const DRAW_BITS: u8 = 32;

/// One of RFC-0058's two comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    /// `B·2^32 ≤ u·Z`: the draw is at or past the token's lower edge, `C(ŷ − 1)`.
    Lower,
    /// `u·Z + ψ ≤ (B + S)·2^32`: the draw is below its upper edge, `C(ŷ)`.
    Upper,
}

/// A committed value of one row, after the bounded-mode indicator, in RFC-0058's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CdfColumn {
    /// `u_0`, the draw's low 16 bits. Its top limb is virtual.
    DrawLimb,
    /// `Z_j`, a low limb of the partition, `j < k − 1`.
    TotalLimb(u8),
    /// `B_j`, a low limb of the prefix below the token, `j < k − 1`.
    BelowLimb(u8),
    /// A column of one comparison.
    Compare(Edge, CompareColumn),
}

/// The factors of a row's affine forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CdfTerm {
    /// The draw `u`, which the PRF circuit commits (RFC-0017).
    Draw,
    /// `Z`, the virtual partition.
    Total,
    /// `B`, the virtual prefix below the token.
    Below,
    /// `S`, the token's virtual exponential.
    Selected,
    /// The public row selector `ψ` (RFC-0035).
    RowSelector,
    /// A committed column.
    Column(CdfColumn),
}

/// A pointwise zero-check of the row, a sum of products of affine factors.
///
/// Degree at most two, three with the sumcheck's equality factor. Prove the pointwise
/// polynomial: a product of multilinear extensions is not the extension of a product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdfRelation<F> {
    /// Public coefficients and affine factors; no factors means the constant one.
    pub terms: Vec<(F, Vec<Affine<F, CdfTerm>>)>,
}

impl<F: PrimeField> CdfRelation<F> {
    /// The polynomial at a Boolean vertex or at the sumcheck's final point.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(CdfTerm) -> E) -> E {
        self.terms.iter().fold(E::ZERO, |s, (c, fs)| {
            s + fs
                .iter()
                .fold(E::from(*c), |p, f| p * f.evaluate(&mut value))
        })
    }
}

/// A row's obligations. All must bind the same columns, sums, draw and selector.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "all equalities and lookups must be proved against the same advice"]
pub struct CdfConstraints<F> {
    /// The committed values in RFC-0058's order. The comparisons' inputs are never committed. In
    /// bounded mode the sampled token's indicator, `pad(vocab)` entries, precedes them.
    pub committed: Vec<CdfColumn>,
    /// The lower comparison's chains, the upper comparison's, then the padding zero-check
    /// `(ψ − 1)·Z = 0`.
    pub equalities: Vec<CdfRelation<F>>,
    /// The draw's, the partition's and the prefix's limbs, then each comparison's range checks.
    pub lookups: Vec<Lookup<F, CdfTerm>>,
}

/// The decomposition a limb belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sum {
    Draw,
    Total,
    Below,
}

/// RFC-0009 'InverseCdf' in RFC-0058's representation, for one vocabulary and one table `σ`.
///
/// At `σ = 32` and a vocabulary of 128,256 tokens a row commits 79 values and makes 82 range
/// requests, plus the sampled token's indicator in bounded mode. The caller supplies the range
/// tables; [`Self::tables`] reports those used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InverseCdf<B = Goldilocks> {
    domain: OneHot<B>,
    sigma: u8,
    width: u8,
    draw: NonNeg,
    sums: NonNeg,
    comparisons: [BigIntCompare; 2],
}

impl<B: PrimeField> InverseCdf<B> {
    /// [`Self::witness`]'s refusal of a token that is not the inverse-CDF sample of its draw:
    /// in exact mode, the deviation RFC-0017 reports as `ERR-WITNESS-EXACT-MISMATCH`.
    pub const MISMATCH: GadgetError = GadgetError::Input {
        rule: "the token is the inverse-CDF sample of its draw (RFC-0009 Lemma 10)",
    };

    /// The gadget for a v1 vocabulary, `1 ≤ vocab ≤ 2^18`, and the vocabulary table's
    /// `σ ≤ 32` (RFC-0008 'Table specifications').
    ///
    /// The partition's box has `W = bit_length(vocab·2^σ)` bits, and `Z` and `B` take
    /// `k = ⌈W/16⌉` limbs. Both comparisons use radix `2^16` and `k + 2` limbs.
    ///
    /// # Errors
    /// OneHot's shape errors; Shape for `σ > 32`, a field whose centered range does not hold
    /// the `2^max(W, 32)` boxes (RFC-0009 Lemma 0), or a missing range table; Spec for a table
    /// that does not regenerate.
    pub fn new(vocab: usize, sigma: u8, tables: &[TableSpec]) -> Result<Self, GadgetError> {
        let domain = OneHot::new(vocab)?;
        if sigma > 32 {
            return Err(GadgetError::Shape {
                rule: "InverseCdf's vocabulary table has σ ≤ 32 (RFC-0008)",
            });
        }
        let width = (64 - ((vocab as u64) << sigma).leading_zeros()) as u8;
        if 1u64 << width.max(DRAW_BITS) >= (B::MODULUS - 1) / 2 {
            return Err(GadgetError::Shape {
                rule: "InverseCdf's boxes fit the centered field range (RFC-0009 Lemma 0)",
            });
        }
        let draw = NonNeg::new(DRAW_BITS, LIMB_BITS, tables)?;
        let sums = NonNeg::new(width, LIMB_BITS, tables)?;
        let full = (1u64 << LIMB_BITS) - 1;
        let top = (1u64 << (u32::from(width) - (sums.limbs() - 1) * u32::from(LIMB_BITS))) - 1;
        let k = sums.limbs() as usize;
        let sum_box = |h: usize| if h + 1 == k { top } else { full };
        let below = |j: usize| {
            if (2..k + 2).contains(&j) {
                sum_box(j - 2)
            } else {
                0
            }
        };
        let product = |j: usize| {
            (0..2)
                .flat_map(|i| (0..k).map(move |h| (i, h)))
                .filter(|&(i, h)| i + h == j)
                .map(|(_, h)| full * sum_box(h))
                .sum::<u64>()
        };
        let lower_left: Vec<u64> = (0..k + 2).map(below).collect();
        let lower_right: Vec<u64> = (0..k + 2).map(product).collect();
        let upper_left: Vec<u64> = (0..k + 2).map(|j| product(j) + u64::from(j == 0)).collect();
        let upper_right: Vec<u64> = (0..k + 2)
            .map(|j| below(j) + if j == 2 { 1 << sigma } else { 0 })
            .collect();
        let comparisons = [
            BigIntCompare::new(LIMB_BITS, &lower_left, &lower_right, tables)?,
            BigIntCompare::new(LIMB_BITS, &upper_left, &upper_right, tables)?,
        ];
        Ok(Self {
            domain,
            sigma,
            width,
            draw,
            sums,
            comparisons,
        })
    }

    /// `W`, the bits of the partition's box `vocab·2^σ`.
    pub fn width(&self) -> u8 {
        self.width
    }

    /// `k`, the limbs of `Z` and of `B`; each comparison has `k + 2`.
    pub fn limbs(&self) -> usize {
        self.sums.limbs() as usize
    }

    /// The vocabulary's indicator domain, whose obligations `ĥ` must satisfy.
    pub fn domain(&self) -> &OneHot<B> {
        &self.domain
    }

    /// The range tables read, each once, in first-use order.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        let mut out: Vec<&GeneratedTable> = Vec::new();
        for t in self
            .draw
            .tables()
            .into_iter()
            .chain(self.sums.tables())
            .chain(self.comparisons.iter().flat_map(|c| c.tables()))
        {
            if !out.iter().any(|s| s.spec().id == t.spec().id) {
                out.push(t);
            }
        }
        out
    }

    /// The row's equalities and lookups over a field that holds its boxes.
    ///
    /// # Errors
    /// Shape when the field's centered range does not hold the decompositions' boxes or the
    /// comparisons' equations (RFC-0009 Lemma 0).
    pub fn constraints<F: PrimeField>(&self) -> Result<CdfConstraints<F>, GadgetError> {
        if 1u64 << self.width.max(DRAW_BITS) >= (F::MODULUS - 1) / 2 {
            return Err(GadgetError::Shape {
                rule: "InverseCdf's boxes fit the centered field range (RFC-0009 Lemma 0)",
            });
        }
        let mut out = CdfConstraints {
            committed: Vec::new(),
            equalities: Vec::new(),
            lookups: Vec::new(),
        };
        // Each decomposition's limbs as forms over the row's terms: committed low limbs, then
        // the virtual top.
        let mut digits: Vec<Vec<Affine<F, CdfTerm>>> = Vec::new();
        for which in [Sum::Draw, Sum::Total, Sum::Below] {
            let nonneg = self.nonneg(which);
            let c = nonneg.constraints::<F>();
            let n = nonneg.limbs() as usize;
            let mut limbs: Vec<_> = (0..n - 1)
                .map(|j| cell(limb_column(which, j as u8)))
                .collect();
            limbs.push(map_nonneg(c.virtuals[0].1.clone(), which)?);
            digits.push(limbs);
            for column in &c.committed {
                match column {
                    Committed::Limb(j) => out.committed.push(limb_column(which, *j)),
                    _ => return Err(unexpected()),
                }
            }
            for l in c.lookups {
                out.lookups.push(Lookup {
                    table: l.table,
                    key: map_nonneg(l.key, which)?,
                    value: None,
                });
            }
        }
        for (idx, edge) in [Edge::Lower, Edge::Upper].into_iter().enumerate() {
            let mut c = self.comparisons[idx].constraints::<F>()?;
            if c.lookups.iter().any(|lookup| {
                lookup.key.constant != F::ZERO
                    || !matches!(lookup.key.terms.as_slice(), [(one, column)]
                    if *one == F::ONE && !matches!(column, CompareColumn::Input(..)))
            }) {
                return Err(GadgetError::Shape {
                    rule: "comparison range keys are committed columns (RFC-0050)",
                });
            }
            // RFC-0050 groups the arrays, rather than interleaving digits and carries.
            c.committed
                .sort_unstable_by_key(|&column| array_order(column));
            c.lookups
                .sort_unstable_by_key(|lookup| array_order(lookup.key.terms[0].1));
            out.committed.extend(
                c.committed
                    .into_iter()
                    .map(|column| CdfColumn::Compare(edge, column)),
            );
            for eq in c.equalities {
                let mut terms = vec![(eq.constant, Vec::new())];
                for (coefficient, column) in eq.terms {
                    match column {
                        CompareColumn::Input(side, j) => terms.extend(
                            self.coefficient(&digits, edge, side, j)
                                .into_iter()
                                .map(|(c, factors)| (coefficient * c, factors)),
                        ),
                        _ => {
                            terms.push((coefficient, vec![cell(CdfColumn::Compare(edge, column))]))
                        }
                    }
                }
                out.equalities.push(CdfRelation { terms });
            }
            out.lookups.extend(c.lookups.into_iter().map(|l| {
                Lookup {
                    table: l.table,
                    key: Affine {
                        constant: l.key.constant,
                        terms: l
                            .key
                            .terms
                            .into_iter()
                            .map(|(c, column)| {
                                (c, CdfTerm::Column(CdfColumn::Compare(edge, column)))
                            })
                            .collect(),
                    },
                    value: None,
                }
            }));
        }
        // RFC-0058 'Padded rows': `(ψ − 1)·Z = 0` pins a padded row's partition, and with it
        // every limb, whatever the vocabulary certificate leaves in its exponentials.
        let mut padding = term(CdfTerm::RowSelector);
        padding.constant = -F::ONE;
        out.equalities.push(CdfRelation {
            terms: vec![(F::ONE, vec![padding, term(CdfTerm::Total)])],
        });
        Ok(out)
    }

    /// The input coefficient `j` of one side of one comparison, as a sum of products:
    ///
    /// - lower: `X_j = B_{j−2}`, `Y_j = P_j`;
    /// - upper: `X_j = P_j + [j = 0]·ψ`, `Y_j = B_{j−2} + [j = 2]·S`;
    ///
    /// with `P_j = Σ_{i + h = j} u_i·Z_h`, the coefficients of `u·Z`.
    fn coefficient<F: PrimeField>(
        &self,
        digits: &[Vec<Affine<F, CdfTerm>>],
        edge: Edge,
        side: Side,
        j: usize,
    ) -> Vec<(F, Vec<Affine<F, CdfTerm>>)> {
        let k = self.limbs();
        let (draw, total, below_digits) = (&digits[0], &digits[1], &digits[2]);
        let product = || {
            let mut terms = Vec::new();
            for (i, u) in draw.iter().enumerate() {
                for (h, z) in total.iter().enumerate() {
                    if i + h == j {
                        terms.push((F::ONE, vec![u.clone(), z.clone()]));
                    }
                }
            }
            terms
        };
        let below = || {
            if (2..k + 2).contains(&j) {
                vec![(F::ONE, vec![below_digits[j - 2].clone()])]
            } else {
                Vec::new()
            }
        };
        let mut terms = match (edge, side) {
            (Edge::Lower, Side::Left) | (Edge::Upper, Side::Right) => below(),
            (Edge::Lower, Side::Right) | (Edge::Upper, Side::Left) => product(),
        };
        match (edge, side, j) {
            (Edge::Upper, Side::Left, 0) => terms.push((F::ONE, vec![term(CdfTerm::RowSelector)])),
            (Edge::Upper, Side::Right, 2) => terms.push((F::ONE, vec![term(CdfTerm::Selected)])),
            _ => {}
        }
        terms
    }

    fn nonneg(&self, which: Sum) -> &NonNeg {
        match which {
            Sum::Draw => &self.draw,
            Sum::Total | Sum::Below => &self.sums,
        }
    }

    /// The honest token's indicator: `ŷ = min{v < vocab : C(v)·2^32 > u·Z}`, the indicator a
    /// bounded-mode row commits. In exact mode the row borrows the next recorded token's.
    ///
    /// The scan reads every padded entry and selects by masks; its addresses depend on the
    /// public vocabulary alone (INV-ARCH-2). `real` is the public row selector `ψ`.
    ///
    /// # Errors
    /// Shape for another length. Input, after the whole scan, unless the draw is in
    /// `[0, 2^32)`, a real row's exponentials are in `[0, 2^σ]` with a positive sum, and a
    /// padded row's are zero.
    pub fn sample(&self, weights: &[B], u: B, real: bool) -> Result<OneHotWitness<B>, GadgetError> {
        let (sums, inside) = self.integers(weights, u, None, real)?;
        let token = if real { sums.token } else { 0 };
        let indicator = self.domain.witness(token, real)?;
        match inside {
            true => Ok(indicator),
            false => Err(domain_error()),
        }
    }

    /// The row's committed values for the indicator `token`, which the caller proves to be a
    /// OneHot of this domain with count `ψ`.
    ///
    /// # Errors
    /// Shape for another length. Input, after every value is computed: [`Self::sample`]'s
    /// domain, an indicator that is not Boolean with count `ψ` and zero padding, or a token
    /// that is not the inverse-CDF sample of its draw, [`Self::MISMATCH`]. In exact mode the
    /// last is the deviation RFC-0017 reports as `ERR-WITNESS-EXACT-MISMATCH`.
    pub fn witness(
        &self,
        weights: &[B],
        u: B,
        token: &OneHotWitness<B>,
        real: bool,
    ) -> Result<CdfWitness, GadgetError> {
        self.domain.check(token.indicator())?;
        let (sums, inside) = self.integers(weights, u, Some(token.indicator()), real)?;
        let k = self.limbs();
        let limb = |x: u128, j: usize| ((x >> (16 * j)) & 0xffff) as i64;
        let top = |x: u128, n: usize| (x >> (16 * (n - 1))) as i64;
        let mut draw = [limb(sums.draw, 0), top(sums.draw, 2)];
        let total: SecretBuf<i64> = (0..k)
            .map(|j| {
                if j + 1 == k {
                    top(sums.total, k)
                } else {
                    limb(sums.total, j)
                }
            })
            .collect();
        let below: SecretBuf<i64> = (0..k)
            .map(|j| {
                if j + 1 == k {
                    top(sums.below, k)
                } else {
                    limb(sums.below, j)
                }
            })
            .collect();
        let mut columns = vec![CdfColumn::DrawLimb];
        let mut values = SecretBuf::with_capacity(1 + 2 * (k - 1) + 12 * (k + 2));
        values.push(draw[0]);
        for (j, &limb) in total[..k - 1].iter().enumerate() {
            columns.push(CdfColumn::TotalLimb(j as u8));
            values.push(limb);
        }
        for (j, &limb) in below[..k - 1].iter().enumerate() {
            columns.push(CdfColumn::BelowLimb(j as u8));
            values.push(limb);
        }
        // The coefficients, from digits in their boxes: each is below 2^33.
        let product = |j: usize| -> i64 {
            (0..2)
                .flat_map(|i| (0..k).map(move |h| (i, h)))
                .filter(|&(i, h)| i + h == j)
                .map(|(i, h)| draw[i] * total[h])
                .sum()
        };
        let lower = |j: usize| {
            if (2..k + 2).contains(&j) {
                below[j - 2]
            } else {
                0
            }
        };
        let selector = i64::from(real);
        let selected = sums.selected as i64;
        let mut fits = true;
        for (idx, edge) in [Edge::Lower, Edge::Upper].into_iter().enumerate() {
            let mut left = SecretBuf::with_capacity(k + 2);
            let mut right = SecretBuf::with_capacity(k + 2);
            for j in 0..k + 2 {
                let (x, y) = match edge {
                    Edge::Lower => (lower(j), product(j)),
                    Edge::Upper => (
                        product(j) + if j == 0 { selector } else { 0 },
                        lower(j) + if j == 2 { selected } else { 0 },
                    ),
                };
                left.push(x);
                right.push(y);
            }
            let left: Vec<&[i64]> = left.chunks(1).collect();
            let right: Vec<&[i64]> = right.chunks(1).collect();
            match self.comparisons[idx].witness(&left, &right) {
                Ok(compared) => {
                    let mut tensors = compared.into_tensors();
                    tensors.sort_unstable_by_key(|(column, _)| array_order(*column));
                    for (column, tensor) in tensors {
                        columns.push(CdfColumn::Compare(edge, column));
                        values.push(tensor[0]);
                    }
                }
                Err(GadgetError::Input { .. }) => fits = false,
                Err(other) => {
                    draw.zeroize();
                    return Err(other);
                }
            }
        }
        draw.zeroize();
        match (inside, fits) {
            (true, true) => Ok(CdfWitness { columns, values }),
            (false, _) => Err(domain_error()),
            (true, false) => Err(Self::MISMATCH),
        }
    }

    /// The draw, the sums and the honest token as integers, from inputs clamped into their
    /// boxes, and whether the inputs were inside them. With an indicator, also checks that it
    /// is Boolean with count `ψ` and zero padding, and takes `B` and `S` from it.
    fn integers(
        &self,
        weights: &[B],
        u: B,
        indicator: Option<&[B]>,
        real: bool,
    ) -> Result<(Integers, bool), GadgetError> {
        self.domain.check(weights)?;
        let vocab = self.domain.vocab();
        let limit = 1u64 << self.sigma;
        let mut inside = true;
        let d = u.to_canonical_u64();
        let fits = d < 1 << DRAW_BITS;
        inside &= fits;
        let draw = u128::from(select_unpredictable(fits, d, 0));
        let weight = |v: usize| -> u128 {
            let x = weights[v].to_canonical_u64();
            u128::from(select_unpredictable(x <= limit, x, 0))
        };
        let mut total = 0u128;
        for (v, &e) in weights.iter().enumerate().take(vocab) {
            let x = e.to_canonical_u64();
            inside &= if real { x <= limit } else { x == 0 };
            total += weight(v);
        }
        inside &= !real | (total > 0);
        let target = draw * total;
        let mut prefix = 0u128;
        let mut token = 0i64;
        let mut found = false;
        for v in 0..vocab {
            prefix += weight(v);
            let cross = !found & ((prefix << 32) > target);
            token = select_unpredictable(cross, v as i64, token);
            found |= cross;
        }
        let (mut below, mut selected) = (0u128, 0u128);
        if let Some(h) = indicator {
            let mut count = 0u64;
            let mut after = 0u128;
            for v in (0..h.len()).rev() {
                let x = h[v].to_canonical_u64();
                inside &= (x <= 1) & ((v < vocab) | (x == 0));
                let bit = u128::from(x == 1);
                count += x & 1;
                if v < vocab {
                    below += weight(v) * after;
                    selected += weight(v) * bit;
                }
                after += bit;
            }
            inside &= count == u64::from(real);
        }
        // A real row's prefix is at most Z; clamp so that a malformed one stays in its box.
        let below = select_unpredictable(below <= total, below, 0);
        Ok((
            Integers {
                draw,
                total,
                below,
                selected,
                token,
            },
            inside,
        ))
    }

    /// The indicator's strict suffix `L_v = Σ_{w > v} ĥ_w` and the sums `Z`, `B` and `S` in the
    /// field, for any assignment.
    ///
    /// # Errors
    /// Shape for another length.
    pub fn evaluate(
        &self,
        weights: &[B],
        token: &OneHotWitness<B>,
    ) -> Result<CdfEvaluation<B>, GadgetError> {
        self.domain.check(weights)?;
        self.domain.check(token.indicator())?;
        let mut suffix = SecretBuf::filled(B::ZERO, self.domain.padded_len());
        let mut after = B::ZERO;
        for (v, &h) in token.indicator().iter().enumerate().rev() {
            suffix[v] = after;
            after += h;
        }
        let valid = self.domain.valid();
        let (mut total, mut below, mut selected) = (B::ZERO, B::ZERO, B::ZERO);
        for (v, &e) in weights.iter().enumerate() {
            total += valid[v] * e;
            below += e * suffix[v];
            selected += token.indicator()[v] * e;
        }
        let sums = [total, below, selected].into_iter().collect();
        Ok(CdfEvaluation { suffix, sums })
    }

    /// The three contractions over the vocabulary: `Σ_v C_v·e_v = Z`, `Σ_v e_v·L_v = B` and
    /// `Σ_v ĥ_v·e_v = S`. Each leaves claims on its operands; `L`'s reduces to `ĥ` through
    /// [`Self::suffix_weights`]. `B` and `S` need no mask: OneHot's vocabulary relation makes
    /// `ĥ`, and so `L`, zero on the vocabulary's padding.
    ///
    /// # Errors
    /// Shape for another length or an engine shape the relations do not fit.
    pub fn relations<'a, F: PolyField<Base = B>>(
        &'a self,
        weights: &'a [B],
        token: &'a OneHotWitness<B>,
        evaluated: &'a CdfEvaluation<B>,
    ) -> Result<CdfRelations<'a, F>, GadgetError> {
        self.domain.check(weights)?;
        self.domain.check(token.indicator())?;
        self.domain.check(&evaluated.suffix)?;
        let vars = self.domain.vars();
        let contracted = ElementwiseShape {
            vars,
            summed_vars: vars,
        };
        let valid = Input::Public(self.domain.valid());
        let e = Input::Operand(weights);
        Ok(CdfRelations {
            total: Elementwise::new(
                contracted,
                vec![valid, e],
                vec![tessera_sumcheck::Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
            below: Elementwise::new(
                contracted,
                vec![e, Input::Operand(&evaluated.suffix)],
                vec![tessera_sumcheck::Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
            selected: Elementwise::new(
                contracted,
                vec![Input::Operand(token.indicator()), e],
                vec![tessera_sumcheck::Term::new(F::ONE, vec![0, 1])],
            )
            .map_err(relation)?,
        })
    }

    /// Bounded mode's obligations on the sampled token's own indicator: OneHot's booleanity,
    /// vocabulary and count relations (RFC-0009 Lemma 8, RFC-0058 'Parameters and inputs').
    /// They make `ĥ` the indicator of one `ŷ < vocab` on a real row and zero on a padded one.
    /// OneHot's index relation is not among them: bounded mode commits no token value, and the
    /// indicator carries `ŷ` itself.
    ///
    /// # Errors
    /// Shape for another length or an engine shape the relations do not fit.
    pub fn sampled_relations<'a, F: PolyField<Base = B>>(
        &'a self,
        token: &'a OneHotWitness<B>,
    ) -> Result<SampledRelations<'a, F>, GadgetError> {
        let all = self.domain.relations::<F>(token)?;
        Ok(SampledRelations {
            booleanity: all.booleanity,
            vocabulary: all.vocabulary,
            count: all.count,
        })
    }

    /// Public weights reducing a suffix claim `L̃(ρ)` to `Σ_v w_v·ĥ_v`, `w_v = Σ_{i<v} eq(ρ, i)`,
    /// as for `Argmax`. The verifier evaluates their extension with [`crate::suffix_kernel`].
    ///
    /// # Errors
    /// Shape unless `ρ` has `log2(pad(vocab))` coordinates.
    pub fn suffix_weights<F: PolyField<Base = B>>(&self, rho: &[F]) -> Result<Vec<F>, GadgetError> {
        suffix_table(rho, self.domain.vars())
    }
}

/// The integers of one row, in their boxes: derived values outside a buffer, cleared where they
/// are held (OBS-6).
struct Integers {
    draw: u128,
    total: u128,
    below: u128,
    selected: u128,
    token: i64,
}

impl Drop for Integers {
    fn drop(&mut self) {
        self.draw.zeroize();
        self.total.zeroize();
        self.below.zeroize();
        self.selected.zeroize();
        self.token.zeroize();
    }
}

fn domain_error() -> GadgetError {
    GadgetError::Input {
        rule: "the draw is in [0,2^32); a real row's exponentials are in [0,2^σ] with a positive sum, a padded row's zero; the indicator is Boolean with count ψ and zero padding",
    }
}

/// A row's committed values, in RFC-0058's order, in clearing storage (OBS-6). Debug shows the
/// count only.
pub struct CdfWitness {
    columns: Vec<CdfColumn>,
    values: SecretBuf<i64>,
}

impl CdfWitness {
    /// The committed columns, in RFC-0058's order: public.
    pub fn columns(&self) -> &[CdfColumn] {
        &self.columns
    }

    /// The values, aligned with [`Self::columns`]. Secret-class: the holder follows
    /// [`crate::Witness::tensor`]'s rules.
    pub fn values(&self) -> &[i64] {
        &self.values
    }

    /// One column's value, or `None` for a column the row does not commit. Secret-class.
    pub fn value(&self, column: CdfColumn) -> Option<i64> {
        self.columns
            .iter()
            .position(|&c| c == column)
            .map(|i| self.values[i])
    }
}

impl fmt::Debug for CdfWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CdfWitness")
            .field("len", &self.values.len())
            .finish()
    }
}

/// The virtual columns of a row, bound by the contractions and the suffix's dot product.
#[must_use]
pub struct CdfEvaluation<B: Copy> {
    /// `L_v = Σ_{w > v} ĥ_w`, over the padded vocabulary.
    pub suffix: SecretBuf<B>,
    /// `Z`, `B` and `S`, in that order.
    pub sums: SecretBuf<B>,
}

impl<B: Copy> fmt::Debug for CdfEvaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CdfEvaluation")
            .field("len", &self.suffix.len())
            .finish_non_exhaustive()
    }
}

/// The vocabulary contractions. With the indicator's OneHot obligations, the suffix's dot
/// product and the row's [`CdfConstraints`], they certify the token; none alone does.
#[must_use]
pub struct CdfRelations<'a, F: PolyField> {
    /// `Σ_v C_v·e_v = Z`.
    pub total: Elementwise<'a, F>,
    /// `Σ_v e_v·L_v = B`.
    pub below: Elementwise<'a, F>,
    /// `Σ_v ĥ_v·e_v = S`.
    pub selected: Elementwise<'a, F>,
}

/// Bounded mode's three OneHot obligations on the sampled token's indicator. Prove each against
/// the same committed `ĥ` that the contractions read.
#[must_use]
pub struct SampledRelations<'a, F: PolyField> {
    /// `Σ eq(z,v)·ĥ_v(1 − ĥ_v) = 0`.
    pub booleanity: Elementwise<'a, F>,
    /// `Σ eq(z,v)·(1 − C_v)·ĥ_v = 0`: no entry of the vocabulary's padding (#378).
    pub vocabulary: Elementwise<'a, F>,
    /// `Σ_v ĥ_v = ψ`.
    pub count: Elementwise<'a, F>,
}

fn limb_column(which: Sum, j: u8) -> CdfColumn {
    match which {
        Sum::Draw => CdfColumn::DrawLimb,
        Sum::Total => CdfColumn::TotalLimb(j),
        Sum::Below => CdfColumn::BelowLimb(j),
    }
}

/// A NonNeg form over the row's terms. NonNeg's forms read its input and its limbs alone; any
/// other term is refused rather than dropped.
fn map_nonneg<F: PrimeField>(a: Affine<F>, which: Sum) -> Result<Affine<F, CdfTerm>, GadgetError> {
    let terms = a
        .terms
        .into_iter()
        .map(|(c, t)| {
            let term = match t {
                Term::Input => match which {
                    Sum::Draw => CdfTerm::Draw,
                    Sum::Total => CdfTerm::Total,
                    Sum::Below => CdfTerm::Below,
                },
                Term::Committed(Committed::Limb(j)) => CdfTerm::Column(limb_column(which, j)),
                _ => return Err(unexpected()),
            };
            Ok((c, term))
        })
        .collect::<Result<_, _>>()?;
    Ok(Affine {
        constant: a.constant,
        terms,
    })
}

fn unexpected() -> GadgetError {
    GadgetError::Shape {
        rule: "a NonNeg decomposition reads its input and its limbs alone (RFC-0009 Lemma 3)",
    }
}

fn term<F: PrimeField>(t: CdfTerm) -> Affine<F, CdfTerm> {
    Affine {
        constant: F::ZERO,
        terms: vec![(F::ONE, t)],
    }
}

fn cell<F: PrimeField>(c: CdfColumn) -> Affine<F, CdfTerm> {
    term(CdfTerm::Column(c))
}
