// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact multi-limb reciprocal-square-root brackets (RFC-0009 Lemma 7).

use crate::compare::array_order;
use crate::{
    Affine, BigIntCompare, Committed, CompareColumn, CompareWitness, GadgetError, Lookup, NonNeg,
    Side, Term,
};
use core::{fmt, hint::select_unpredictable};
use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

/// Which exact inequality a comparison proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bracket {
    /// `rho^2 u <= psi N`.
    Lower,
    /// `psi (N+1) <= (rho+1)^2 u`.
    Upper,
}

/// Advice columns in root, input-limb, root-limb, lower/upper comparison order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RsqrtColumn {
    /// The root `rho`.
    Root,
    /// One of u's three committed 16-bit low limbs; the top seven bits are virtual.
    InputLimb(u8),
    /// A committed low byte of rho; its top byte is virtual.
    RootLimb(u8),
    /// A normalization, difference or borrow column of one bracket comparison.
    Compare(Bracket, CompareColumn),
}

/// Factors in the gadget's affine forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RsqrtTerm {
    /// Virtual `u = psi (sum res_i^2 + e)`, bound by the driver.
    Input,
    /// The public real-row selector psi (RFC-0035).
    RowSelector,
    /// A committed column.
    Column(RsqrtColumn),
}

/// A pointwise zero-check, a sum of products of affine factors.
///
/// Degree at most three, four including the sumcheck equality factor. Prove the pointwise
/// polynomial; a product of multilinear extensions is not the extension of a product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsqrtRelation<F> {
    /// Public coefficient and affine factors; no factors means the constant one.
    pub terms: Vec<(F, Vec<Affine<F, RsqrtTerm>>)>,
}
impl<F: PrimeField> RsqrtRelation<F> {
    /// Evaluate the polynomial at a Boolean vertex or the sumcheck's final point.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(RsqrtTerm) -> E) -> E {
        self.terms.iter().fold(E::ZERO, |s, (c, fs)| {
            s + fs
                .iter()
                .fold(E::from(*c), |p, f| p * f.evaluate(&mut value))
        })
    }
}

/// Every obligation must bind the same columns, input and public selector before acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "all equalities and lookups must be proved against the same advice"]
pub struct RsqrtConstraints<F> {
    /// Advice in canonical gadget order; comparison input sums are virtual, never committed.
    pub committed: Vec<RsqrtColumn>,
    /// Lower comparison, upper comparison, then the padded-root zero-check.
    pub equalities: Vec<RsqrtRelation<F>>,
    /// Root range, u limbs, root bytes, then lower/upper comparison range checks.
    pub lookups: Vec<Lookup<F, RsqrtTerm>>,
}

/// Public parameters for the exact integer bracket, with regenerated range tables.
///
/// RFC-0050 uses `ceil((2w+55)/8)` byte-radix limbs per comparison. At a 32-bit root it
/// commits 187 columns and performs 189 range lookups per row; it does not claim the
/// RFC's approximate cost estimate. Model tile/table integration belongs to the driver.
/// The caller supplies tables; [`Self::tables`] reports precisely those used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsqrtRow {
    n: u128,
    e: u64,
    bits: u8,
    root: GeneratedTable,
    input: NonNeg,
    digits: NonNeg,
    comparisons: [BigIntCompare; 2],
}
impl RsqrtRow {
    /// Requested owned heap of this public template, including nested comparison
    /// bounds, carry widths, table arrays and their payload capacities. Excludes
    /// inline `Self`, constructor scratch and every secret witness (RFC-0018).
    /// Returns `None` on size overflow and allocates nothing.
    pub fn heap_bytes(&self) -> Option<usize> {
        let bytes = crate::storage::table(&self.root)?
            .checked_add(self.input.heap_bytes()?)?
            .checked_add(self.digits.heap_bytes()?)?;
        self.comparisons
            .iter()
            .try_fold(bytes, |sum, compare| sum.checked_add(compare.heap_bytes()?))
    }

    /// Construct from claim-list dimension, precision, epsilon and range specifications.
    ///
    /// u's decomposition uses 16-bit limbs (top seven bits). Root bytes keep triple limb
    /// products below 2^40; fifteen base-256 limbs hold the complete product below 2^119,
    /// even when rho+1 is 2^32. No product is reduced as a whole into the field.
    ///
    /// # Errors
    /// Shape for d=0, kappa>31, e outside [1,2^55), a root needing more than 32 bits,
    /// or missing ranges. Spec for a table that does not regenerate.
    pub fn new(dim: u32, kappa: u8, e: u64, tables: &[TableSpec]) -> Result<Self, GadgetError> {
        let shape = GadgetError::Shape {
            rule: "RsqrtRow needs d>0, kappa<=31, 1<=e<2^55 and N/e<2^64 (RFC-0022/23)",
        };
        if dim == 0 || kappa > 31 || e == 0 || e >= 1 << 55 {
            return Err(shape);
        }
        let n = u128::from(dim) << (2 * u32::from(kappa));
        let quotient = n / u128::from(e);
        if quotient >= 1u128 << 64 {
            return Err(shape);
        }
        let bits = (128 - quotient.isqrt().leading_zeros()).max(1) as u8;
        let spec = tables
            .iter()
            .find(|s| s.kind == TableKind::Range { bits })
            .ok_or(shape)?;
        let root = GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec {
            table: spec.id,
            error,
        })?;
        let input = NonNeg::new(55, 16, tables)?;
        let digits = NonNeg::new(bits, 8, tables)?;
        let limbs = (2 * usize::from(bits) + 55).div_ceil(8);
        let lower = product_bounds(bits, false, limbs);
        let upper = product_bounds(bits, true, limbs);
        let constants = vec![255; limbs];
        let comparisons = [
            BigIntCompare::new(8, &lower, &constants, tables)?,
            BigIntCompare::new(8, &constants, &upper, tables)?,
        ];
        Ok(Self {
            n,
            e,
            bits,
            root,
            input,
            digits,
            comparisons,
        })
    }

    /// The public root range width, at least one and at most 32.
    pub fn root_bits(&self) -> u8 {
        self.bits
    }

    #[cfg(debug_assertions)]
    pub(crate) fn n(&self) -> u128 {
        self.n
    }

    #[cfg(debug_assertions)]
    pub(crate) fn input_nonneg(&self) -> &NonNeg {
        &self.input
    }

    #[cfg(debug_assertions)]
    pub(crate) fn digits_nonneg(&self) -> &NonNeg {
        &self.digits
    }

    #[cfg(debug_assertions)]
    pub(crate) fn comparisons(&self) -> &[BigIntCompare; 2] {
        &self.comparisons
    }

    #[cfg(debug_assertions)]
    pub(crate) fn product_relation<F: PrimeField>(&self, b: Bracket, j: usize) -> RsqrtRelation<F> {
        RsqrtRelation {
            terms: self.product::<F>(b, j),
        }
    }

    /// Tables used by this gadget, deduplicated in first-use order.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        let mut out = vec![&self.root];
        for t in self
            .input
            .tables()
            .into_iter()
            .chain(self.digits.tables())
            .chain(self.comparisons.iter().flat_map(|c| c.tables()))
        {
            if !out.iter().any(|s| s.spec().id == t.spec().id) {
                out.push(t);
            }
        }
        out
    }

    /// All pointwise equality and lookup obligations over a field that holds the limb boxes.
    ///
    /// # Errors
    /// Shape when the field is too small for u's 55-bit box or the comparison equations.
    pub fn constraints<F: PrimeField>(&self) -> Result<RsqrtConstraints<F>, GadgetError> {
        if (1u64 << 55) >= (F::MODULUS - 1) / 2 {
            return Err(GadgetError::Shape {
                rule: "RsqrtRow's limb box fits the centered field (RFC-0009 Lemma 0)",
            });
        }
        let mut out = RsqrtConstraints {
            committed: vec![RsqrtColumn::Root],
            equalities: Vec::new(),
            lookups: vec![Lookup {
                table: self.root.spec().id,
                key: column(RsqrtColumn::Root),
                value: None,
            }],
        };
        for (nonneg, is_input) in [(&self.input, true), (&self.digits, false)] {
            let c = nonneg.constraints::<F>();
            out.committed
                .extend(c.committed.into_iter().map(|c| match c {
                    Committed::Limb(j) => limb_column(is_input, j),
                    _ => unreachable!(),
                }));
            out.lookups.extend(c.lookups.into_iter().map(|l| Lookup {
                table: l.table,
                key: map_nonneg(l.key, is_input),
                value: None,
            }));
        }
        for (idx, b) in [Bracket::Lower, Bracket::Upper].into_iter().enumerate() {
            let mut c = self.comparisons[idx].constraints::<F>()?;
            // RFC-0050 groups arrays, rather than interleaving digits and carries.
            c.committed
                .sort_unstable_by_key(|&column| array_order(column));
            if c.lookups.iter().any(|lookup| {
                lookup.key.constant != F::ZERO
                    || !matches!(lookup.key.terms.as_slice(), [(one, column)]
                    if *one == F::ONE && !matches!(column, CompareColumn::Input(..)))
            }) {
                return Err(GadgetError::Shape {
                    rule: "root comparison range keys are committed columns (RFC-0050)",
                });
            }
            c.lookups
                .sort_unstable_by_key(|lookup| array_order(lookup.key.terms[0].1));
            out.committed.extend(
                c.committed
                    .into_iter()
                    .map(|col| RsqrtColumn::Compare(b, col)),
            );
            for eq in c.equalities {
                let mut terms = vec![(eq.constant, Vec::new())];
                for (coefficient, col) in eq.terms {
                    match col {
                        CompareColumn::Input(side, j) => {
                            let product = (b == Bracket::Lower && side == Side::Left)
                                || (b == Bracket::Upper && side == Side::Right);
                            if product {
                                terms.extend(
                                    self.product::<F>(b, j)
                                        .into_iter()
                                        .map(|(c, f)| (coefficient * c, f)),
                                );
                            } else {
                                let v = (((self.n + u128::from(b == Bracket::Upper)) >> (8 * j))
                                    & 255) as u64;
                                terms.push((
                                    coefficient * F::from_u64_reduced(v),
                                    vec![term(RsqrtTerm::RowSelector)],
                                ));
                            }
                        }
                        _ => terms.push((coefficient, vec![column(RsqrtColumn::Compare(b, col))])),
                    }
                }
                out.equalities.push(RsqrtRelation { terms });
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
                            .map(|(c, col)| (c, RsqrtTerm::Column(RsqrtColumn::Compare(b, col))))
                            .collect(),
                    },
                    value: None,
                }
            }));
        }
        let mut padding = term(RsqrtTerm::RowSelector);
        padding.constant = -F::ONE;
        out.equalities.push(RsqrtRelation {
            terms: vec![(F::ONE, vec![padding, column(RsqrtColumn::Root)])],
        });
        Ok(out)
    }

    fn product<F: PrimeField>(&self, b: Bracket, j: usize) -> Vec<(F, Vec<Affine<F, RsqrtTerm>>)> {
        let mut terms = Vec::new();
        for a in 0..usize::from(self.bits).div_ceil(8) {
            for c in 0..usize::from(self.bits).div_ceil(8) {
                for k in 0..4 {
                    if a + c + 2 * k != j {
                        continue;
                    }
                    let mut left = self.digit::<F>(false, a);
                    let mut right = self.digit::<F>(false, c);
                    if b == Bracket::Upper && a == 0 {
                        left.constant += F::ONE;
                    }
                    if b == Bracket::Upper && c == 0 {
                        right.constant += F::ONE;
                    }
                    terms.push((F::ONE, vec![left, right, self.digit::<F>(true, k)]));
                }
            }
        }
        terms
    }
    fn digit<F: PrimeField>(&self, input: bool, j: usize) -> Affine<F, RsqrtTerm> {
        let nonneg = if input { &self.input } else { &self.digits };
        if j + 1 == nonneg.limbs() as usize {
            map_nonneg(nonneg.constraints::<F>().virtuals[0].1.clone(), input)
        } else {
            column(limb_column(input, j as u8))
        }
    }

    /// Generate roots, low limbs and both exact comparison witnesses for public real/padded rows.
    ///
    /// Inputs are the already selected u, not the sum before adding epsilon. Real rows require
    /// e<=u<2^55; padded rows u=0. The driver derives selectors from public dimensions.
    /// Secret inputs are borrowed; every retained derived value lives in clearing storage (OBS-6).
    /// Root search uses exactly 32 trials per row, with branch-free selection.
    ///
    /// # Errors
    /// Shape for mismatched lengths; Input after the complete batch if a row is outside its domain.
    pub fn witness(&self, u: &[i64], real: &[bool]) -> Result<RsqrtWitness, GadgetError> {
        if u.len() != real.len() {
            return Err(GadgetError::Shape {
                rule: "RsqrtRow has one public selector per row",
            });
        }
        let mut roots = SecretBuf::with_capacity(u.len());
        let mut values = SecretBuf::with_capacity(u.len());
        let mut inside = true;
        for (&v, &active) in u.iter().zip(real) {
            let valid = if active {
                (v >= self.e as i64) & (v < 1 << 55)
            } else {
                v == 0
            };
            inside &= valid;
            let safe = select_unpredictable(valid, v, if active { self.e as i64 } else { 0 });
            let r = root(self.n, safe as u64);
            roots.push(if active { r as i64 } else { 0 });
            values.push(safe);
        }
        let mut tensors = vec![(RsqrtColumn::Root, roots)];
        // Build low limbs directly; top limbs are virtual in NonNeg's lookups.
        for (input, bits, n) in [
            (true, 16, self.input.limbs()),
            (false, 8, self.digits.limbs()),
        ] {
            for j in 0..n - 1 {
                let mut limb = SecretBuf::with_capacity(u.len());
                let data = if input {
                    &values[..]
                } else {
                    &tensors[0].1[..]
                };
                for &v in data {
                    limb.push((v >> (bits * j)) & ((1 << bits) - 1));
                }
                tensors.push((limb_column(input, j as u8), limb));
            }
        }
        let value = |row: usize, t: RsqrtTerm| match t {
            RsqrtTerm::Input => Goldilocks::from_i64(values[row]),
            RsqrtTerm::RowSelector => Goldilocks::from_u64_reduced(u64::from(real[row])),
            RsqrtTerm::Column(c) => Goldilocks::from_i64(
                tensors
                    .iter()
                    .find(|(k, _)| *k == c)
                    .map_or(0, |(_, v)| v[row]),
            ),
        };
        let mut comparisons = Vec::new();
        for (idx, b) in [Bracket::Lower, Bracket::Upper].into_iter().enumerate() {
            let limbs = self.comparisons[idx].limbs();
            let polynomials: Vec<_> = (0..limbs)
                .map(|j| RsqrtRelation {
                    terms: self.product::<Goldilocks>(b, j),
                })
                .collect();
            let mut product: Vec<_> = (0..limbs)
                .map(|_| SecretBuf::with_capacity(u.len()))
                .collect();
            let mut constant: Vec<_> = (0..limbs)
                .map(|_| SecretBuf::with_capacity(u.len()))
                .collect();
            for (row, &active) in real.iter().enumerate() {
                for j in 0..limbs {
                    product[j].push(
                        polynomials[j]
                            .evaluate(|t| value(row, t))
                            .to_canonical_u64() as i64,
                    );
                    constant[j].push(if active {
                        (((self.n + u128::from(b == Bracket::Upper)) >> (8 * j)) & 255) as i64
                    } else {
                        0
                    });
                }
            }
            let p: Vec<_> = product.iter().map(|v| &v[..]).collect();
            let c: Vec<_> = constant.iter().map(|v| &v[..]).collect();
            comparisons.push(if b == Bracket::Lower {
                self.comparisons[idx].witness(&p, &c)?
            } else {
                self.comparisons[idx].witness(&c, &p)?
            });
        }
        if !inside {
            return Err(GadgetError::Input {
                rule: "RsqrtRow has e<=u<2^55 on real rows and u=0 on padded rows",
            });
        }
        Ok(RsqrtWitness {
            tensors,
            comparisons,
        })
    }
}

/// Advice in zeroizing buffers. Debug reveals column names and lengths only.
pub struct RsqrtWitness {
    tensors: Vec<(RsqrtColumn, SecretBuf<i64>)>,
    comparisons: Vec<CompareWitness>,
}
impl RsqrtWitness {
    #[cfg(debug_assertions)]
    pub(crate) fn from_parts(
        tensors: Vec<(RsqrtColumn, SecretBuf<i64>)>,
        comparisons: Vec<CompareWitness>,
    ) -> Self {
        Self {
            tensors,
            comparisons,
        }
    }

    /// Transfer columns in RFC-0050 allocation order without copying their values.
    /// The holder applies the secret borrowing rules of [`Self::tensor`].
    pub fn into_tensors(self) -> Vec<(RsqrtColumn, SecretBuf<i64>)> {
        let mut tensors = self.tensors;
        for (bracket, comparison) in [Bracket::Lower, Bracket::Upper]
            .into_iter()
            .zip(self.comparisons)
        {
            tensors.extend(
                comparison
                    .into_tensors()
                    .into_iter()
                    .map(|(column, values)| (RsqrtColumn::Compare(bracket, column), values)),
            );
        }
        tensors.sort_unstable_by_key(|(column, _)| column_order(*column));
        tensors
    }

    /// Borrow a committed column. Values remain secret-class (05 §3).
    pub fn tensor(&self, c: RsqrtColumn) -> Option<&[i64]> {
        match c {
            RsqrtColumn::Compare(b, col) => self
                .comparisons
                .get(usize::from(b == Bracket::Upper))?
                .tensor(col),
            _ => self
                .tensors
                .iter()
                .find(|(k, _)| *k == c)
                .map(|(_, v)| &v[..]),
        }
    }
}
impl fmt::Debug for RsqrtWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RsqrtWitness")
            .field(
                "columns",
                &self
                    .tensors
                    .iter()
                    .map(|(c, v)| (*c, v.len()))
                    .collect::<Vec<_>>(),
            )
            .field("comparisons", &self.comparisons)
            .finish()
    }
}

fn column_order(column: RsqrtColumn) -> (u8, usize) {
    match column {
        RsqrtColumn::Root => (0, 0),
        RsqrtColumn::InputLimb(index) => (1, usize::from(index)),
        RsqrtColumn::RootLimb(index) => (2, usize::from(index)),
        RsqrtColumn::Compare(bracket, column) => {
            let (kind, index) = array_order(column);
            (3 + 6 * u8::from(bracket == Bracket::Upper) + kind, index)
        }
    }
}

fn product_bounds(bits: u8, upper: bool, limbs: usize) -> Vec<u64> {
    let count = usize::from(bits).div_ceil(8);
    let mut bounds = vec![0; limbs];
    for a in 0..count {
        for b in 0..count {
            for k in 0..4 {
                let digit = |i: usize| {
                    ((1u64 << usize::from(bits).saturating_sub(8 * i).min(8)) - 1)
                        + u64::from(upper && i == 0)
                };
                bounds[a + b + 2 * k] += digit(a) * digit(b) * if k == 3 { 127 } else { 65535 };
            }
        }
    }
    bounds
}
fn limb_column(input: bool, j: u8) -> RsqrtColumn {
    if input {
        RsqrtColumn::InputLimb(j)
    } else {
        RsqrtColumn::RootLimb(j)
    }
}
fn map_nonneg<F: PrimeField>(a: Affine<F>, input: bool) -> Affine<F, RsqrtTerm> {
    Affine {
        constant: a.constant,
        terms: a
            .terms
            .into_iter()
            .map(|(c, t)| {
                (
                    c,
                    match t {
                        Term::Input => {
                            if input {
                                RsqrtTerm::Input
                            } else {
                                RsqrtTerm::Column(RsqrtColumn::Root)
                            }
                        }
                        Term::Committed(Committed::Limb(j)) => {
                            RsqrtTerm::Column(limb_column(input, j))
                        }
                        _ => unreachable!(),
                    },
                )
            })
            .collect(),
    }
}
fn term<F: PrimeField>(t: RsqrtTerm) -> Affine<F, RsqrtTerm> {
    Affine {
        constant: F::ZERO,
        terms: vec![(F::ONE, t)],
    }
}
fn column<F: PrimeField>(c: RsqrtColumn) -> Affine<F, RsqrtTerm> {
    term(RsqrtTerm::Column(c))
}

// Exactly 32 trials, independent of the secret row's magnitude. Products are below 2^119.
fn root(n: u128, u: u64) -> u64 {
    let mut r = 0;
    for bit in (0..32).rev() {
        let trial = r | (1u64 << bit);
        let product = u128::from(trial) * u128::from(trial) * u128::from(u);
        r = select_unpredictable(product <= n, trial, r);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::root;
    #[test]
    fn million_roots_match_independent_integer_inequality() {
        let mut seed = 0x357d65a4be9d006au64;
        for i in 0..1_000_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed & ((1u64 << 55) - 1)).max(1);
            let (d, k) = [
                (1u128, 0),
                (4096, 30),
                (16384, 30),
                (u128::from(u32::MAX), 16),
            ][i % 4];
            let n = d << (2 * k);
            let r = u128::from(root(n, u));
            // Separate standard-library square root after integer division.
            assert_eq!(r, (n / u128::from(u)).isqrt());
            assert!(r * r * u128::from(u) <= n);
            assert!((r + 1) * (r + 1) * u128::from(u) > n);
        }
    }
}
