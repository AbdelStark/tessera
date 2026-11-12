// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `OneSidedLog2`, the mixture predictor's one-sided `-log2 q_t(y_t)` bound certificate
//! (RFC-0017 'In-circuit scoring (mixture)', RFC-0008, RFC-0009 Lemmas 0, 3 & 4, RFC-0035,
//! RFC-0050).
//!
//! On output row `t` with public row selector `ψ_t ∈ {0, 1}`, vocabulary size `V = vocab`,
//! vocabulary exponentials `e_{t,v} ∈ [0, 2^σ]` certified by [`crate::VocabMax`], recorded
//! next-token indicator `h_{y_t}` certified by [`crate::OneHot`], and `D`'s decoded indicator
//! `h_{ŷ_t}` certified by bounded [`crate::Argmax`] or [`crate::InverseCdf`]:
//!
//! - `Z_t = Σ_v C_v · e_{t,v}` is the virtual partition sum (`C_v = [v < V]`);
//! - `Y_t = Σ_v h_{y_t,v} · e_{t,v}` is the virtual recorded-token exponential `e_{t,y_t}`;
//! - `m_t ∈ {0, 1}` (`m_t = 0` when `k = eps_log2 = 0`) is the committed match bit, bound by the
//!   vocabulary zero-check `m_t · (h_{y_t,v} − h_{ŷ_t,v}) = 0`;
//! - `(I_t, P_t) ∈ Pow2 { max_exp: 63 }` (`I_t ∈ [0, 62]`, `P_t = 2^{I_t} ∈ [1, 2^{62}]`) and
//!   `(f_t, E_t) ∈ Exp2Frac { frac_bits: 8, out_bits: 16 }` (`f_t ∈ [0, 255]`,
//!   `E_t = ⌊2^{f_t/256} · 2^{16}⌋ ∈ [2^{16}, 2^{17})`);
//! - `q_t = (Y_t + ψ_t) · E_t ∈ [0, (2^σ + 1)(2^{17} − 1)] ⊂ [0, 2^{49})` is committed with the
//!   degree-2 equality `q_t − (Y_t + ψ_t) · E_t = 0` and decomposed in radix `2^{16}` via
//!   [`crate::NonNeg`];
//! - `Z_t` is decomposed in radix `2^{16}` via [`crate::NonNeg`] with padding check
//!   `(ψ_t − 1) · Z_t = 0`, and `P_t` is decomposed in radix `2^{16}` over a `63`-bit box
//!   (`P_0, P_1, P_2 ∈ Range { 16 }`, virtual `P_3 ∈ Range { 15 }`, unique when `P > 2^{63}`);
//! - [`crate::BigIntCompare`] over radix `2^{16}` proves
//!   `ψ_t · (Z_t + V) · 2^{16} ≤ q_t · P_t = (Y_t + ψ_t) · 2^{I_t} · E_t` with degree-2 pairwise
//!   limb products below `2^{34} << 2^{40}`;
//! - the virtual integer row bound in units of `2^{−8}` bits (`FixedBits`) is
//!   `B_t = ψ_t · (m_t · c_match(k) + (1 − m_t) · (256 · k + 256 · I_t + f_t))`, summed across
//!   each request's slot into the public claim `Û_r = Σ_{t ∈ r} B_t` while keeping per-token
//!   values hidden (`INV-SC-1..3`).

use core::{fmt, hint::select_unpredictable};

use tessera_codec::{SecretBuf, Zeroize};
use tessera_field::{Goldilocks, PrimeField};
use tessera_poly::PolyField;
use tessera_sumcheck::{Elementwise, ElementwiseShape, Input};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

use crate::choice::{OneHot, OneHotWitness, relation};
use crate::compare::array_order;
use crate::constraints::inv_pow2;
use crate::{
    Affine, BigIntCompare, Committed, CompareColumn, GadgetError, Lookup, Lookups, NonNeg, Side,
    Term,
};

/// Radix `2^{16}` shared by `Z_t`, `q_t`, `P_t` and `BigIntCompare`.
const LIMB_BITS: u8 = 16;

/// `Pow2 { max_exp: 63 }` (`I_t ∈ [0, 62]`, `2^{I_t} ∈ [1, 2^{62}]`; RFC-0008, RFC-0017).
pub const POW2_MAX_EXP: u8 = 63;

/// `Exp2Frac` input width `frac_bits = 8` (`256` rows; RFC-0017).
pub const EXP2_FRAC_BITS: u8 = 8;

/// `Exp2Frac` output scaling `out_bits = 16` (`E2[f] = ⌊2^{f/256} · 2^{16}⌋`; RFC-0017).
pub const EXP2_OUT_BITS: u8 = 16;

/// Maximum `eps_log2` (`ε = 2^{−k}`, `k ∈ 0..=16`; RFC-0017).
pub const MAX_EPS_LOG2: u8 = 16;

/// Inclusive upper bound on a single row's `B_t` in `2^{−8}`-bit units (`256·16 + 256·62 + 255`).
pub const MAX_ROW_BOUND: u64 = 256 * 16 + 256 * 62 + 255;

/// Exact `⌈−log2(1 − 2^{−k}) · 2^8⌉` for `k = 1..=16`, and `0` at `k = 0` (`m_t = 0`).
const C_MATCH: [u64; 17] = [0, 256, 107, 50, 24, 12, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1];

/// Public match cost `c_match(k)` in `2^{−8}`-bit units (`0` at `k = 0`,
/// `⌈−log2(1 − 2^{−k}) · 2^8⌉` for `1 ≤ k ≤ 16`; RFC-0017).
///
/// # Errors
/// [`GadgetError::Shape`] if `eps_log2 > 16`.
pub fn c_match(eps_log2: u8) -> Result<u64, GadgetError> {
    C_MATCH
        .get(usize::from(eps_log2))
        .copied()
        .ok_or(GadgetError::Shape {
            rule: "Mixture eps_log2 lies in 0..=16 (RFC-0017)",
        })
}

/// Committed advice columns of one [`OneSidedLog2`] row in canonical allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Log2Column {
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
    /// A committed normalization or subtraction column of `BigIntCompare`, in RFC-0050 array order.
    Compare(CompareColumn),
}

/// Factors appearing in [`OneSidedLog2`]'s affine lookup forms and row equalities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Log2Term {
    /// Virtual partition sum `Z_t = Σ_v C_v · e_{t,v}`.
    Total,
    /// Virtual recorded-token exponential `Y_t = Σ_v h_{y_t,v} · e_{t,v}`.
    Selected,
    /// Public row selector `ψ_t ∈ {0, 1}` (RFC-0035).
    RowSelector,
    /// A committed [`Log2Column`].
    Column(Log2Column),
}

/// A pointwise polynomial zero-check of one [`OneSidedLog2`] row (`Σ_i c_i · Π_j f_{i,j} = 0`).
///
/// Every relation has degree at most two (three with the sumcheck's `eq` factor).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Log2Relation<F> {
    /// Public coefficients and affine factor lists; an empty factor list denotes constant `1`.
    pub terms: Vec<(F, Vec<Affine<F, Log2Term>>)>,
}

impl<F: PrimeField> Log2Relation<F> {
    /// Evaluate the relation polynomial at one row or at a sumcheck's final point.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(Log2Term) -> E) -> E {
        self.terms.iter().fold(E::ZERO, |sum, (coeff, factors)| {
            sum + factors
                .iter()
                .fold(E::from(*coeff), |prod, f| prod * f.evaluate(&mut value))
        })
    }
}

/// Row-level committed columns, equalities and lookups of [`OneSidedLog2`].
///
/// Pair with [`OneSidedLog2::batched_vocab_relations`] (`Z_t`, `Y_t`, and
/// `m_t · (h_{y_t} − h_{ŷ_t}) = 0`) and [`OneSidedLog2::bound_relation`] (`Σ_t B_t = Û_r`).
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "prove all equalities, lookups, vocabulary relations and the Û_r sum relation"]
pub struct Log2Constraints<F> {
    /// Committed columns in canonical allocation order.
    pub committed: Vec<Log2Column>,
    /// Match-bit check (`m_t = 0` or `m_t(1 − m_t) = 0`), `q_t − (Y_t + ψ_t)·E_t = 0`,
    /// `BigIntCompare`'s chains, and `(ψ_t − 1)·Z_t = 0`.
    pub equalities: Vec<Log2Relation<F>>,
    /// `Pow2`, `Exp2Frac`, `Z_t` limbs, `q_t` limbs, `P_t` limbs, and `BigIntCompare` ranges.
    pub lookups: Vec<Lookup<F, Log2Term>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Decomp {
    Total,
    Scaled,
}

/// Mixture predictor one-sided `log2` bound gadget (RFC-0017 'In-circuit scoring (mixture)').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneSidedLog2<B = Goldilocks> {
    domain: OneHot<B>,
    sigma: u8,
    eps_log2: u8,
    c_match_units: u64,
    total_width: u8,
    scaled_width: u8,
    pow2: GeneratedTable,
    exp2_frac: GeneratedTable,
    exp2_values: Vec<u64>,
    total_nonneg: NonNeg,
    scaled_nonneg: NonNeg,
    pow2_low_range: GeneratedTable,
    pow2_top_range: GeneratedTable,
    comparison: BigIntCompare,
}

impl<B: PrimeField> OneSidedLog2<B> {
    /// Build [`OneSidedLog2`] for `1 ≤ vocab ≤ 2^18`, `σ ≤ 32`, and `eps_log2 ∈ 0..=16`.
    ///
    /// Requires `B::MODULUS > 2^63` so `P_t`'s `63`-bit limb box `[0, 2^63)` cannot alias modulo
    /// `P` (`RFC-0009` Lemma 0), and regenerates `Pow2 { max_exp: 63 }`,
    /// `Exp2Frac { frac_bits: 8, out_bits: 16 }`, and all required `Range` tables from `tables`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for invalid `vocab`, `sigma`, `eps_log2`, field modulus, or missing
    /// tables; [`GadgetError::Spec`] if any table specification fails regeneration.
    pub fn new(
        vocab: usize,
        sigma: u8,
        eps_log2: u8,
        tables: &[TableSpec],
    ) -> Result<Self, GadgetError> {
        let domain = OneHot::new(vocab)?;
        if sigma > 32 {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2's vocabulary table has σ ≤ 32 (RFC-0008)",
            });
        }
        let c_match_units = c_match(eps_log2)?;
        if B::MODULUS <= (1u64 << 63) {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 needs P > 2^63 for Pow2's 63-bit limb box (RFC-0009 Lemma 0)",
            });
        }
        let total_width = (64 - ((vocab as u64) << sigma).leading_zeros()) as u8;
        let max_q = ((1u128 << sigma) + 1) * ((1u128 << 17) - 1);
        let scaled_width = (128 - max_q.leading_zeros()) as u8;
        let half_field = (B::MODULUS - 1) / 2;
        if (1u64 << total_width.max(scaled_width)) >= half_field {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2's Z and q boxes fit the centered field range (RFC-0009 Lemma 0)",
            });
        }

        let find_regen =
            |kind: TableKind, rule: &'static str| -> Result<GeneratedTable, GadgetError> {
                let spec = tables
                    .iter()
                    .find(|s| s.kind == kind)
                    .ok_or(GadgetError::Shape { rule })?;
                GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec {
                    table: spec.id,
                    error,
                })
            };
        let pow2 = find_regen(
            TableKind::Pow2 {
                max_exp: POW2_MAX_EXP,
            },
            "OneSidedLog2 requires Pow2 { max_exp: 63 }",
        )?;
        let exp2_frac = find_regen(
            TableKind::Exp2Frac {
                frac_bits: EXP2_FRAC_BITS,
                out_bits: EXP2_OUT_BITS,
            },
            "OneSidedLog2 requires Exp2Frac { frac_bits: 8, out_bits: 16 }",
        )?;
        let exp2_values: Vec<u64> = (0..exp2_frac.row_count())
            .map(|f| {
                exp2_frac
                    .lookup(f as i64, 0)
                    .and_then(|v| u64::try_from(v).ok())
                    .filter(|&v| ((1 << 16)..(1 << 17)).contains(&v))
                    .ok_or(GadgetError::Shape {
                        rule: "Exp2Frac { 8, 16 } values lie in [2^16, 2^17)",
                    })
            })
            .collect::<Result<_, _>>()?;
        let total_nonneg = NonNeg::new(total_width, LIMB_BITS, tables)?;
        let scaled_nonneg = NonNeg::new(scaled_width, LIMB_BITS, tables)?;
        let pow2_low_range = find_regen(
            TableKind::Range { bits: 16 },
            "OneSidedLog2 requires Range { bits: 16 } for Pow2's low limbs",
        )?;
        let pow2_top_range = find_regen(
            TableKind::Range { bits: 15 },
            "OneSidedLog2 requires Range { bits: 15 } for Pow2's top limb",
        )?;

        let (left_bounds, right_bounds) = comparison_bounds(
            vocab,
            total_width,
            scaled_width,
            total_nonneg.limbs() as usize,
            scaled_nonneg.limbs() as usize,
        );
        let comparison = BigIntCompare::new(LIMB_BITS, &left_bounds, &right_bounds, tables)?;

        Ok(Self {
            domain,
            sigma,
            eps_log2,
            c_match_units,
            total_width,
            scaled_width,
            pow2,
            exp2_frac,
            exp2_values,
            total_nonneg,
            scaled_nonneg,
            pow2_low_range,
            pow2_top_range,
            comparison,
        })
    }

    /// Required [`TableKind`]s (`Pow2 { 63 }`, `Exp2Frac { 8, 16 }`, and deduplicated `Range`s) in
    /// first-use order.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] if `vocab`, `sigma`, or `B::MODULUS` violates [`Self::new`].
    pub fn table_kinds(vocab: usize, sigma: u8) -> Result<Vec<TableKind>, GadgetError> {
        let mut probe = Vec::with_capacity(34);
        probe.push(
            GeneratedTable::new(
                TableId(0),
                TableKind::Pow2 {
                    max_exp: POW2_MAX_EXP,
                },
            )
            .map_err(|error| GadgetError::Spec {
                table: TableId(0),
                error,
            })?
            .spec()
            .clone(),
        );
        probe.push(
            GeneratedTable::new(
                TableId(1),
                TableKind::Exp2Frac {
                    frac_bits: EXP2_FRAC_BITS,
                    out_bits: EXP2_OUT_BITS,
                },
            )
            .map_err(|error| GadgetError::Spec {
                table: TableId(1),
                error,
            })?
            .spec()
            .clone(),
        );
        for bits in 1..=32u8 {
            let id = TableId(1 + u32::from(bits));
            probe.push(
                GeneratedTable::new(id, TableKind::Range { bits })
                    .map_err(|error| GadgetError::Spec { table: id, error })?
                    .spec()
                    .clone(),
            );
        }
        let gadget = Self::new(vocab, sigma, 6, &probe)?;
        Ok(gadget
            .tables()
            .into_iter()
            .map(|t| t.spec().kind.clone())
            .collect())
    }

    /// Vocabulary [`OneHot`] domain.
    pub fn domain(&self) -> &OneHot<B> {
        &self.domain
    }

    /// Exponential precision `σ`.
    pub const fn sigma(&self) -> u8 {
        self.sigma
    }

    /// Mixture weight exponent `k = eps_log2` (`ε = 2^{−k}`).
    pub const fn eps_log2(&self) -> u8 {
        self.eps_log2
    }

    /// Match cost `c_match(k)` in `2^{−8}`-bit units.
    pub const fn c_match_units(&self) -> u64 {
        self.c_match_units
    }

    /// Bit width `W_Z` of `Z_t`'s `NonNeg` box.
    pub const fn total_width(&self) -> u8 {
        self.total_width
    }

    /// Bit width `W_Q` of `q_t`'s `NonNeg` box.
    pub const fn scaled_width(&self) -> u8 {
        self.scaled_width
    }

    /// Limb count `m` of `BigIntCompare`.
    pub fn comparison_limbs(&self) -> usize {
        self.comparison.limbs()
    }

    /// Regenerated tables read by [`OneSidedLog2`], deduplicated in first-use order.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        let mut out: Vec<&GeneratedTable> = vec![&self.pow2, &self.exp2_frac];
        for t in self
            .total_nonneg
            .tables()
            .into_iter()
            .chain(self.scaled_nonneg.tables())
            .chain([&self.pow2_low_range, &self.pow2_top_range])
            .chain(self.comparison.tables())
        {
            if !out.iter().any(|s| s.spec().id == t.spec().id) {
                out.push(t);
            }
        }
        out
    }

    /// Canonical committed columns, row equalities and lookups over `F`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `F::MODULUS > 2^63` and `F`'s centered range holds `Z_t`,
    /// `q_t`, and `BigIntCompare`'s equation boxes (`RFC-0009` Lemma 0).
    pub fn constraints<F: PrimeField>(&self) -> Result<Log2Constraints<F>, GadgetError> {
        if F::MODULUS <= (1u64 << 63) {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 needs P > 2^63 for Pow2's 63-bit limb box (RFC-0009 Lemma 0)",
            });
        }
        let half_field = (F::MODULUS - 1) / 2;
        if (1u64 << self.total_width.max(self.scaled_width)) >= half_field {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2's Z and q boxes fit the centered field range (RFC-0009 Lemma 0)",
            });
        }

        let mut out = Log2Constraints {
            committed: vec![
                Log2Column::MatchBit,
                Log2Column::IntExp,
                Log2Column::Pow2Val,
                Log2Column::FracExp,
                Log2Column::Exp2Val,
                Log2Column::ScaledSelected,
            ],
            equalities: Vec::new(),
            lookups: vec![
                Lookup {
                    table: self.pow2.spec().id,
                    key: cell(Log2Column::IntExp),
                    value: Some(cell(Log2Column::Pow2Val)),
                },
                Lookup {
                    table: self.exp2_frac.spec().id,
                    key: cell(Log2Column::FracExp),
                    value: Some(cell(Log2Column::Exp2Val)),
                },
            ],
        };

        // 1. Decompose Z_t and q_t via NonNeg(W_Z, 16) and NonNeg(W_Q, 16).
        let mut z_digits: Vec<Affine<F, Log2Term>> = Vec::new();
        let mut q_digits: Vec<Affine<F, Log2Term>> = Vec::new();
        for (which, nonneg) in [
            (Decomp::Total, &self.total_nonneg),
            (Decomp::Scaled, &self.scaled_nonneg),
        ] {
            let c = nonneg.constraints::<F>();
            let n = nonneg.limbs() as usize;
            let mut limbs: Vec<_> = (0..n - 1)
                .map(|j| cell(decomp_limb(which, j as u8)))
                .collect();
            limbs.push(map_nonneg(c.virtuals[0].1.clone(), which)?);
            for col in &c.committed {
                match col {
                    Committed::Limb(j) => out.committed.push(decomp_limb(which, *j)),
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
            match which {
                Decomp::Total => z_digits = limbs,
                Decomp::Scaled => q_digits = limbs,
            }
        }

        // 2. Decompose P_t into P_0, P_1, P_2 in Range(16) and virtual P_3 in Range(15).
        let mut p_digits: Vec<Affine<F, Log2Term>> = Vec::with_capacity(4);
        for j in 0..3u8 {
            let col = Log2Column::Pow2Limb(j);
            out.committed.push(col);
            out.lookups.push(Lookup {
                table: self.pow2_low_range.spec().id,
                key: cell(col),
                value: None,
            });
            p_digits.push(cell(col));
        }
        let p_top = Affine {
            constant: F::ZERO,
            terms: vec![
                (inv_pow2::<F>(48), Log2Term::Column(Log2Column::Pow2Val)),
                (
                    -inv_pow2::<F>(48),
                    Log2Term::Column(Log2Column::Pow2Limb(0)),
                ),
                (
                    -inv_pow2::<F>(32),
                    Log2Term::Column(Log2Column::Pow2Limb(1)),
                ),
                (
                    -inv_pow2::<F>(16),
                    Log2Term::Column(Log2Column::Pow2Limb(2)),
                ),
            ],
        };
        out.lookups.push(Lookup {
            table: self.pow2_top_range.spec().id,
            key: p_top.clone(),
            value: None,
        });
        p_digits.push(p_top);

        // 3. Match-bit constraint: m_t = 0 when k = 0; m_t * (1 - m_t) = 0 when 1 <= k <= 16.
        if self.eps_log2 == 0 {
            out.equalities.push(Log2Relation {
                terms: vec![(F::ONE, vec![cell(Log2Column::MatchBit)])],
            });
        } else {
            let mut one_minus_m = Affine {
                constant: F::ONE,
                terms: vec![(-F::ONE, Log2Term::Column(Log2Column::MatchBit))],
            };
            let _ = &mut one_minus_m;
            out.equalities.push(Log2Relation {
                terms: vec![(F::ONE, vec![cell(Log2Column::MatchBit), one_minus_m])],
            });
        }

        // 4. Scaled selected product: q_t - (Y_t + ψ_t) * E_t = 0.
        let y_plus_psi = Affine {
            constant: F::ZERO,
            terms: vec![
                (F::ONE, Log2Term::Selected),
                (F::ONE, Log2Term::RowSelector),
            ],
        };
        out.equalities.push(Log2Relation {
            terms: vec![
                (F::ONE, vec![cell(Log2Column::ScaledSelected)]),
                (-F::ONE, vec![y_plus_psi, cell(Log2Column::Exp2Val)]),
            ],
        });

        // 5. BigIntCompare in RFC-0050 grouped array order:
        //    LHS = ψ_t * (Z_t + V) * 2^16, RHS = q_t * P_t.
        let mut cmp = self.comparison.constraints::<F>()?;
        if cmp.lookups.iter().any(|lookup| {
            lookup.key.constant != F::ZERO
                || !matches!(lookup.key.terms.as_slice(), [(one, column)]
                    if *one == F::ONE && !matches!(column, CompareColumn::Input(..)))
        }) {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 comparison range keys are committed columns (RFC-0050)",
            });
        }
        cmp.committed
            .sort_unstable_by_key(|&column| array_order(column));
        cmp.lookups
            .sort_unstable_by_key(|lookup| array_order(lookup.key.terms[0].1));
        out.committed
            .extend(cmp.committed.into_iter().map(Log2Column::Compare));

        let v_limbs = [
            (self.domain.vocab() as u64) & 0xffff,
            (self.domain.vocab() as u64) >> 16,
        ];
        for eq in cmp.equalities {
            let mut terms = vec![(eq.constant, Vec::new())];
            for (coeff, col) in eq.terms {
                match col {
                    CompareColumn::Input(side, j) => {
                        for (c, factors) in
                            comparison_input(&z_digits, &q_digits, &p_digits, v_limbs, side, j)
                        {
                            terms.push((coeff * c, factors));
                        }
                    }
                    _ => terms.push((coeff, vec![cell(Log2Column::Compare(col))])),
                }
            }
            out.equalities.push(Log2Relation { terms });
        }
        out.lookups.extend(cmp.lookups.into_iter().map(|l| {
            Lookup {
                table: l.table,
                key: Affine {
                    constant: l.key.constant,
                    terms: l
                        .key
                        .terms
                        .into_iter()
                        .map(|(c, col)| (c, Log2Term::Column(Log2Column::Compare(col))))
                        .collect(),
                },
                value: None,
            }
        }));

        // 6. Padded-row partition check: (ψ_t - 1) * Z_t = 0 (RFC-0035, RFC-0058).
        let mut psi_minus_one = term(Log2Term::RowSelector);
        psi_minus_one.constant = -F::ONE;
        out.equalities.push(Log2Relation {
            terms: vec![(F::ONE, vec![psi_minus_one, term(Log2Term::Total)])],
        });

        Ok(out)
    }

    /// Generate the honest clearing witness for one output row.
    ///
    /// Control flow, memory addresses and shift counts depend on public sizes alone (`INV-ARCH-2`,
    /// `OBS-1`, `OBS-6`): all `63` `Pow2` exponents and `256` `Exp2Frac` rows are scanned with
    /// `select_unpredictable`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for mismatched slice lengths; [`GadgetError::Input`] (after scanning
    /// the entire row) if `weights`, `recorded` or `predicted` violate their real/padded domains.
    pub fn witness(
        &self,
        weights: &[B],
        recorded: &OneHotWitness<B>,
        predicted: &OneHotWitness<B>,
        real: bool,
    ) -> Result<Log2Witness, GadgetError> {
        self.domain.check(weights)?;
        self.domain.check(recorded.indicator())?;
        self.domain.check(predicted.indicator())?;

        let vocab = self.domain.vocab();
        let limit = 1u64 << self.sigma;
        let mut inside = true;
        let mut total = 0u128;
        let mut selected = 0u128;
        let mut rec_count = 0u64;
        let mut pred_count = 0u64;
        let mut same_indicator = true;

        for (v, ((&e_f, &h_rec_f), &h_pred_f)) in weights
            .iter()
            .zip(recorded.indicator())
            .zip(predicted.indicator())
            .enumerate()
        {
            let e_raw = e_f.to_canonical_u64();
            let h_rec = h_rec_f.to_canonical_u64();
            let h_pred = h_pred_f.to_canonical_u64();
            inside &= (h_rec <= 1) & ((v < vocab) | (h_rec == 0));
            inside &= (h_pred <= 1) & ((v < vocab) | (h_pred == 0));
            rec_count += h_rec & 1;
            pred_count += h_pred & 1;
            same_indicator &= h_rec == h_pred;

            if v < vocab {
                inside &= if real { e_raw <= limit } else { e_raw == 0 };
                let safe_e = u128::from(select_unpredictable(real & (e_raw <= limit), e_raw, 0));
                total += safe_e;
                selected += safe_e * u128::from(h_rec == 1);
            } else {
                inside &= e_raw == 0;
            }
        }
        inside &= (rec_count == u64::from(real)) & (pred_count == u64::from(real));
        inside &= !real | (total > 0);

        let e2_zero = self.exp2_values.first().copied().unwrap_or(1 << 16);
        let e2_max = u128::from(self.exp2_values.last().copied().unwrap_or(1 << 16));

        let lhs = if real {
            (total + (vocab as u128)) << 16
        } else {
            0
        };
        let num = if real { selected + 1 } else { 1 };

        // Stage 1: 63 public-shift iterations to find the minimal I_t where (I_t, 255) clears lhs.
        let mut found_i = false;
        let mut best_i = 0u64;
        let mut best_pow2 = 1u128;
        for i in 0..u64::from(POW2_MAX_EXP) {
            let pow2 = 1u128 << i;
            let clears = (num * pow2 * e2_max) >= lhs;
            let take = !found_i & clears;
            best_i = select_unpredictable(take, i, best_i);
            best_pow2 = select_unpredictable(take, pow2, best_pow2);
            found_i |= clears;
        }
        inside &= found_i;

        // Stage 2: 256 public-index iterations over Exp2Frac's values to find minimal f_t at best_i.
        let scaled_num = num * best_pow2;
        let mut found_f = false;
        let mut best_f = 0u64;
        let mut best_e2 = e2_zero;
        for (f, &e2_val) in self.exp2_values.iter().enumerate() {
            let clears = (scaled_num * u128::from(e2_val)) >= lhs;
            let take = !found_f & clears;
            best_f = select_unpredictable(take, f as u64, best_f);
            best_e2 = select_unpredictable(take, e2_val, best_e2);
            found_f |= clears;
        }
        inside &= found_f;

        let match_allowed = real & (self.eps_log2 > 0);
        let mut m_val = u64::from(select_unpredictable(
            match_allowed & same_indicator,
            true,
            false,
        ));
        let mut i_val = if real { best_i } else { 0 };
        let mut p_val = if real { best_pow2 as u64 } else { 1 };
        let mut f_val = if real { best_f } else { 0 };
        let mut e_val = if real { best_e2 } else { e2_zero };
        let mut q_val = if real {
            (selected + 1) * u128::from(e_val)
        } else {
            0
        };

        let mut bound_units = if real {
            let mismatch_cost = 256 * u64::from(self.eps_log2) + 256 * i_val + f_val;
            select_unpredictable(m_val == 1, self.c_match_units, mismatch_cost)
        } else {
            0
        };

        let k_z = self.total_nonneg.limbs() as usize;
        let k_q = self.scaled_nonneg.limbs() as usize;
        let limb = |x: u128, j: usize| ((x >> (16 * j)) & 0xffff) as i64;
        let top = |x: u128, n: usize| (x >> (16 * (n - 1))) as i64;

        let z_limbs: SecretBuf<i64> = (0..k_z)
            .map(|j| {
                if j + 1 == k_z {
                    top(total, k_z)
                } else {
                    limb(total, j)
                }
            })
            .collect();
        let q_limbs: SecretBuf<i64> = (0..k_q)
            .map(|j| {
                if j + 1 == k_q {
                    top(q_val, k_q)
                } else {
                    limb(q_val, j)
                }
            })
            .collect();
        let mut p_limbs = [
            limb(u128::from(p_val), 0),
            limb(u128::from(p_val), 1),
            limb(u128::from(p_val), 2),
            top(u128::from(p_val), 4),
        ];

        let m = self.comparison.limbs();
        let mut columns = Vec::with_capacity(6 + (k_z - 1) + (k_q - 1) + 3 + 6 * m);
        let mut values = SecretBuf::with_capacity(6 + (k_z - 1) + (k_q - 1) + 3 + 6 * m);
        for (col, val) in [
            (Log2Column::MatchBit, m_val as i64),
            (Log2Column::IntExp, i_val as i64),
            (Log2Column::Pow2Val, p_val as i64),
            (Log2Column::FracExp, f_val as i64),
            (Log2Column::Exp2Val, e_val as i64),
            (Log2Column::ScaledSelected, q_val as i64),
        ] {
            columns.push(col);
            values.push(val);
        }
        for (j, &val) in z_limbs[..k_z - 1].iter().enumerate() {
            columns.push(Log2Column::TotalLimb(j as u8));
            values.push(val);
        }
        for (j, &val) in q_limbs[..k_q - 1].iter().enumerate() {
            columns.push(Log2Column::ScaledLimb(j as u8));
            values.push(val);
        }
        for (j, &val) in p_limbs[..3].iter().enumerate() {
            columns.push(Log2Column::Pow2Limb(j as u8));
            values.push(val);
        }

        let v_limbs = [
            ((vocab as u64) & 0xffff) as i64,
            ((vocab as u64) >> 16) as i64,
        ];
        let psi = i64::from(real);
        let mut left = SecretBuf::with_capacity(m);
        let mut right = SecretBuf::with_capacity(m);
        for j in 0..m {
            let x = if j == 0 {
                0
            } else {
                let h = j - 1;
                let zh = z_limbs.get(h).copied().unwrap_or(0);
                let vh = v_limbs.get(h).copied().unwrap_or(0);
                psi * (zh + vh)
            };
            let y: i64 = (0..k_q)
                .flat_map(|a| (0..4).map(move |b| (a, b)))
                .filter(|&(a, b)| a + b == j)
                .map(|(a, b)| q_limbs[a] * p_limbs[b])
                .sum();
            left.push(x);
            right.push(y);
        }
        let left_refs: Vec<&[i64]> = left.chunks(1).collect();
        let right_refs: Vec<&[i64]> = right.chunks(1).collect();
        let compared = self.comparison.witness(&left_refs, &right_refs);

        total.zeroize();
        selected.zeroize();
        rec_count.zeroize();
        pred_count.zeroize();
        same_indicator.zeroize();
        best_i.zeroize();
        best_pow2.zeroize();
        best_f.zeroize();
        best_e2.zeroize();
        m_val.zeroize();
        i_val.zeroize();
        p_val.zeroize();
        f_val.zeroize();
        e_val.zeroize();
        q_val.zeroize();
        p_limbs.zeroize();

        let compared = match compared {
            Ok(c) => c,
            Err(other) => {
                bound_units.zeroize();
                return Err(other);
            }
        };
        let mut tensors = compared.into_tensors();
        tensors.sort_unstable_by_key(|(column, _)| array_order(*column));
        for (column, tensor) in tensors {
            columns.push(Log2Column::Compare(column));
            values.push(tensor[0]);
        }

        if !inside {
            bound_units.zeroize();
            return Err(GadgetError::Input {
                rule: "a real row's exponentials lie in [0, 2^σ] with positive sum and OneHot indicators; a padded row's are zero",
            });
        }
        let mut bound_buf = SecretBuf::with_capacity(1);
        bound_buf.push(bound_units);
        bound_units.zeroize();
        Ok(Log2Witness {
            columns,
            values,
            bound_units: bound_buf,
        })
    }

    /// Precompute public selectors for a power-of-two batch of `padded_rows` rows
    /// (`real_rows ≤ padded_rows`), checking that `padded_rows · MAX_ROW_BOUND < P` so the `Û_r`
    /// contraction cannot wrap modulo `P`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `padded_rows` is a positive power of two with
    /// `real_rows ≤ padded_rows` and `padded_rows · MAX_ROW_BOUND < B::MODULUS`.
    pub fn selector(
        &self,
        padded_rows: usize,
        real_rows: usize,
    ) -> Result<BatchedLog2Selector<B>, GadgetError> {
        let padded_vocab = self.domain.padded_len();
        let total = padded_rows
            .checked_mul(padded_vocab)
            .ok_or(GadgetError::Shape {
                rule: "OneSidedLog2 batch selector fits usize",
            })?;
        if !padded_rows.is_power_of_two()
            || real_rows > padded_rows
            || (padded_rows as u128) * u128::from(MAX_ROW_BOUND) >= u128::from(B::MODULUS)
        {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 batch needs power-of-two padded_rows ≥ real_rows with Σ B_t < P",
            });
        }
        let mut valid = Vec::with_capacity(total);
        let mut row_selector = Vec::with_capacity(padded_rows);
        for row in 0..padded_rows {
            valid.extend_from_slice(self.domain.valid());
            row_selector.push(B::from_u64_reduced(u64::from(row < real_rows)));
        }
        let ones = vec![B::ONE; padded_rows];
        Ok(BatchedLog2Selector {
            padded_rows,
            real_rows,
            valid,
            row_selector,
            ones,
        })
    }

    /// Evaluate virtual `Z_t`, `Y_t` and per-row bounds `B_t` across `padded_rows` rows in the
    /// field, plus the batch's field sum `Σ_t B_t`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `weights`, `recorded` and `columns` match `selector`.
    pub fn evaluate_batch(
        &self,
        weights: &[B],
        recorded: &[B],
        columns: &Log2BatchColumns<B>,
        selector: &BatchedLog2Selector<B>,
    ) -> Result<BatchedLog2Evaluation<B>, GadgetError> {
        let constraints = self.constraints::<B>()?;
        let padded_rows = selector.padded_rows;
        let padded_vocab = self.domain.padded_len();
        let total_entries = selector.valid.len();
        if weights.len() != total_entries || recorded.len() != total_entries {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 batch weights and indicators match selector's shape",
            });
        }
        check_columns_only(padded_rows, columns, &constraints.committed)?;

        let valid = self.domain.valid();
        let m_col = columns.column(Log2Column::MatchBit).unwrap_or(&[]);
        let i_col = columns.column(Log2Column::IntExp).unwrap_or(&[]);
        let f_col = columns.column(Log2Column::FracExp).unwrap_or(&[]);
        let c_f = B::from_u64_reduced(self.c_match_units);
        let k_f = B::from_u64_reduced(256 * u64::from(self.eps_log2));
        let scale_256 = B::from_u64_reduced(256);

        let mut totals = SecretBuf::with_capacity(padded_rows);
        let mut selecteds = SecretBuf::with_capacity(padded_rows);
        let mut bounds = SecretBuf::with_capacity(padded_rows);
        let mut sum_field = B::ZERO;
        for row in 0..padded_rows {
            let base = row * padded_vocab;
            let (mut z, mut y) = (B::ZERO, B::ZERO);
            for v in 0..padded_vocab {
                let e = weights[base + v];
                z += valid[v] * e;
                y += recorded[base + v] * e;
            }
            totals.push(z);
            selecteds.push(y);
            let psi = selector.row_selector[row];
            let m = m_col[row];
            let b_row =
                psi * (m * c_f + (B::ONE - m) * (k_f + scale_256 * i_col[row] + f_col[row]));
            bounds.push(b_row);
            sum_field += b_row;
        }
        let mut uhat_field = SecretBuf::with_capacity(1);
        uhat_field.push(sum_field);
        sum_field.zeroize();
        Ok(BatchedLog2Evaluation {
            totals,
            selecteds,
            bounds,
            uhat_field,
        })
    }

    /// Batched vocabulary relations over `row_vars + vocab_vars`:
    /// - `total`: `Σ_v C_v · e(t, v) = Z(t)`;
    /// - `selected`: `Σ_v h_y(t, v) · e(t, v) = Y(t)`;
    /// - `match_indicator`: `0 = Σ_{t,v} eq(z, (t, v)) · m(t) · (h_y(t, v) − h_ŷ(t, v))`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for mismatched operand shapes or an invalid engine configuration.
    pub fn batched_vocab_relations<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a BatchedLog2Selector<B>,
        weights: &'a [B],
        recorded: &'a [B],
        predicted: &'a [B],
        columns: &'a Log2BatchColumns<B>,
    ) -> Result<Log2VocabRelations<'a, F>, GadgetError> {
        let total_entries = selector.valid.len();
        let Some(match_bits) = columns.column(Log2Column::MatchBit) else {
            return Err(GadgetError::Shape {
                rule: "Log2BatchColumns holds MatchBit",
            });
        };
        if weights.len() != total_entries
            || recorded.len() != total_entries
            || predicted.len() != total_entries
            || match_bits.len() != selector.padded_rows
        {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 vocabulary operands match selector's shape",
            });
        }
        let vocab_vars = self.domain.vars();
        let row_vars = selector.padded_rows.trailing_zeros() as usize;
        let vars = row_vars + vocab_vars;
        let contracted = ElementwiseShape {
            vars,
            summed_vars: vocab_vars,
        };
        let e = Input::Operand(weights);
        let total = Elementwise::new(
            contracted,
            vec![Input::Public(&selector.valid), e],
            vec![tessera_sumcheck::Term::new(F::ONE, vec![0, 1])],
        )
        .map_err(relation)?;
        let selected = Elementwise::new(
            contracted,
            vec![Input::Operand(recorded), e],
            vec![tessera_sumcheck::Term::new(F::ONE, vec![0, 1])],
        )
        .map_err(relation)?;
        let match_indicator = Elementwise::new(
            ElementwiseShape::elementwise(vars),
            vec![
                Input::Broadcast {
                    operand: match_bits,
                    low_vars: vocab_vars,
                },
                Input::Operand(recorded),
                Input::Operand(predicted),
            ],
            vec![
                tessera_sumcheck::Term::new(F::ONE, vec![0, 1]),
                tessera_sumcheck::Term::new(-F::ONE, vec![0, 2]),
            ],
        )
        .map_err(relation)?;
        Ok(Log2VocabRelations {
            total,
            selected,
            match_indicator,
        })
    }

    /// Build [`Elementwise`] zero-checks over `row_vars` for every row equality of
    /// [`Log2Constraints`].
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `selector`, `evaluated` and `columns` agree on `padded_rows`
    /// and hold every committed [`Log2Column`] once.
    pub fn equality_relations<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a BatchedLog2Selector<B>,
        evaluated: &'a BatchedLog2Evaluation<B>,
        columns: &'a Log2BatchColumns<B>,
    ) -> Result<Vec<Elementwise<'a, F>>, GadgetError> {
        let constraints = self.constraints::<B>()?;
        check_batch_columns(selector, evaluated, columns, &constraints.committed)?;
        let row_vars = selector.padded_rows.trailing_zeros() as usize;
        let shape = ElementwiseShape::elementwise(row_vars);
        let mut inputs = vec![
            Input::Public(&selector.ones),
            Input::Public(&selector.row_selector),
            Input::Operand(&evaluated.totals),
            Input::Operand(&evaluated.selecteds),
        ];
        for &col in &constraints.committed {
            let slice = columns.column(col).ok_or(GadgetError::Shape {
                rule: "Log2BatchColumns holds every committed Log2Column",
            })?;
            inputs.push(Input::Operand(slice));
        }
        let term_index = |t: Log2Term| -> Result<usize, GadgetError> {
            match t {
                Log2Term::RowSelector => Ok(1),
                Log2Term::Total => Ok(2),
                Log2Term::Selected => Ok(3),
                Log2Term::Column(col) => constraints
                    .committed
                    .iter()
                    .position(|&c| c == col)
                    .map(|idx| 4 + idx)
                    .ok_or(GadgetError::Shape {
                        rule: "Log2Relation references a committed Log2Column",
                    }),
            }
        };

        let mut out = Vec::with_capacity(constraints.equalities.len());
        for eq in &constraints.equalities {
            let mut combined: Vec<(Vec<usize>, B)> = Vec::new();
            for &(coeff, ref factors) in &eq.terms {
                let mut expanded: Vec<(B, Vec<usize>)> = vec![(coeff, Vec::new())];
                for aff in factors {
                    let mut next = Vec::new();
                    for (w, mono) in expanded {
                        if aff.constant != B::ZERO {
                            next.push((w * aff.constant, mono.clone()));
                        }
                        for &(tc, term) in &aff.terms {
                            if tc != B::ZERO {
                                let idx = term_index(term)?;
                                let mut m = mono.clone();
                                m.push(idx);
                                m.sort_unstable();
                                next.push((w * tc, m));
                            }
                        }
                    }
                    expanded = next;
                }
                for (w, mono) in expanded {
                    let key = if mono.is_empty() { vec![0] } else { mono };
                    if let Some((_, acc)) = combined.iter_mut().find(|(k, _)| *k == key) {
                        *acc += w;
                    } else {
                        combined.push((key, w));
                    }
                }
            }
            let terms: Vec<tessera_sumcheck::Term<F>> = combined
                .into_iter()
                .filter(|&(_, w)| w != B::ZERO)
                .map(|(factors, w)| tessera_sumcheck::Term::new(F::from(w), factors))
                .collect();
            out.push(Elementwise::new(shape, inputs.clone(), terms).map_err(relation)?);
        }
        Ok(out)
    }

    /// Build the degree-3 contraction `Σ_t B_t = Û_r` over `row_vars` (`RFC-0017` Constraint 4 & 5),
    /// where `B_t = ψ_t · (m_t · c_match + (1 − m_t) · (256 · k + 256 · I_t + f_t))`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] if `columns` is missing `MatchBit`, `IntExp`, or `FracExp` at
    /// `selector.padded_rows`.
    pub fn bound_relation<'a, F: PolyField<Base = B>>(
        &'a self,
        selector: &'a BatchedLog2Selector<B>,
        columns: &'a Log2BatchColumns<B>,
    ) -> Result<Elementwise<'a, F>, GadgetError> {
        let rows = selector.padded_rows;
        let (Some(m_col), Some(i_col), Some(f_col)) = (
            columns.column(Log2Column::MatchBit),
            columns.column(Log2Column::IntExp),
            columns.column(Log2Column::FracExp),
        ) else {
            return Err(GadgetError::Shape {
                rule: "Log2BatchColumns holds MatchBit, IntExp and FracExp",
            });
        };
        if m_col.len() != rows || i_col.len() != rows || f_col.len() != rows {
            return Err(GadgetError::Shape {
                rule: "OneSidedLog2 bound columns match padded_rows",
            });
        }
        let row_vars = rows.trailing_zeros() as usize;
        let shape = ElementwiseShape {
            vars: row_vars,
            summed_vars: row_vars,
        };
        let inputs = vec![
            Input::Public(&selector.row_selector),
            Input::Operand(m_col),
            Input::Operand(i_col),
            Input::Operand(f_col),
        ];
        let k_scaled = 256 * (self.eps_log2 as i64);
        let c_minus_k = (self.c_match_units as i64) - k_scaled;
        let raw_terms = [
            (B::from_i64(k_scaled), vec![0]),
            (B::from_i64(c_minus_k), vec![0, 1]),
            (B::from_i64(256), vec![0, 2]),
            (B::from_i64(-256), vec![0, 1, 2]),
            (B::ONE, vec![0, 3]),
            (-B::ONE, vec![0, 1, 3]),
        ];
        let terms: Vec<tessera_sumcheck::Term<F>> = raw_terms
            .into_iter()
            .filter(|&(w, _)| w != B::ZERO)
            .map(|(w, factors)| tessera_sumcheck::Term::new(F::from(w), factors))
            .collect();
        Elementwise::new(shape, inputs, terms).map_err(relation)
    }

    /// Evaluate all lookups (`Pow2`, `Exp2Frac`, `Z_t` limbs, `q_t` limbs, `P_t` limbs, and
    /// `BigIntCompare` range checks) over `padded_rows` rows into [`Lookups`] grouped by table id.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless `selector`, `evaluated` and `columns` agree on `padded_rows`
    /// and hold every committed [`Log2Column`] once.
    pub fn evaluate_lookups(
        &self,
        selector: &BatchedLog2Selector<B>,
        evaluated: &BatchedLog2Evaluation<B>,
        columns: &Log2BatchColumns<B>,
    ) -> Result<Lookups<B>, GadgetError>
    where
        B: PolyField<Base = B>,
    {
        let constraints = self.constraints::<B>()?;
        check_batch_columns(selector, evaluated, columns, &constraints.committed)?;
        let eval_term = |row: usize, t: Log2Term| -> B {
            match t {
                Log2Term::Total => evaluated.totals[row],
                Log2Term::Selected => evaluated.selecteds[row],
                Log2Term::RowSelector => selector.row_selector[row],
                Log2Term::Column(col) => columns.column(col).map_or(B::ZERO, |vals| vals[row]),
            }
        };
        let raw = constraints
            .lookups
            .iter()
            .map(|lookup| {
                let keys: SecretBuf<B> = (0..selector.padded_rows)
                    .map(|row| lookup.key.evaluate(|t| eval_term(row, t)))
                    .collect();
                let values = lookup.value.as_ref().map(|val_form| {
                    (0..selector.padded_rows)
                        .map(|row| val_form.evaluate(|t| eval_term(row, t)))
                        .collect()
                });
                (lookup.table, keys, values)
            })
            .collect();
        Lookups::from_columns(raw)
    }
}

fn check_columns_only<B: Copy>(
    rows: usize,
    columns: &Log2BatchColumns<B>,
    expected: &[Log2Column],
) -> Result<(), GadgetError> {
    if columns.columns.len() != expected.len()
        || expected.iter().any(|&col| {
            columns.columns.iter().filter(|(c, _)| *c == col).count() != 1
                || columns.column(col).is_none_or(|v| v.len() != rows)
        })
    {
        return Err(GadgetError::Shape {
            rule: "Log2BatchColumns holds each committed Log2Column once at padded_rows",
        });
    }
    Ok(())
}

fn check_batch_columns<B: Copy>(
    selector: &BatchedLog2Selector<B>,
    evaluated: &BatchedLog2Evaluation<B>,
    columns: &Log2BatchColumns<B>,
    expected: &[Log2Column],
) -> Result<(), GadgetError> {
    let rows = selector.padded_rows;
    if evaluated.totals.len() != rows || evaluated.selecteds.len() != rows {
        return Err(GadgetError::Shape {
            rule: "OneSidedLog2 evaluated sums match padded_rows",
        });
    }
    check_columns_only(rows, columns, expected)
}

fn comparison_bounds(
    vocab: usize,
    total_width: u8,
    scaled_width: u8,
    k_z: usize,
    k_q: usize,
) -> (Vec<u64>, Vec<u64>) {
    let full = (1u64 << LIMB_BITS) - 1;
    let top_z_bits = u32::from(total_width) - ((k_z as u32 - 1) * u32::from(LIMB_BITS));
    let top_q_bits = u32::from(scaled_width) - ((k_q as u32 - 1) * u32::from(LIMB_BITS));
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
    let v_limbs = [(vocab as u64) & 0xffff, (vocab as u64) >> 16];

    let max_left = (((1u128 << total_width) - 1) + (vocab as u128)) << 16;
    let max_right = ((1u128 << scaled_width) - 1) * ((1u128 << 63) - 1);
    let max_both = max_left.max(max_right);
    let total_bits = (128 - max_both.leading_zeros()) as usize;
    let m = total_bits.div_ceil(usize::from(LIMB_BITS));

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
    (left_bounds, right_bounds)
}

fn comparison_input<F: PrimeField>(
    z_digits: &[Affine<F, Log2Term>],
    q_digits: &[Affine<F, Log2Term>],
    p_digits: &[Affine<F, Log2Term>],
    v_limbs: [u64; 2],
    side: Side,
    j: usize,
) -> Vec<(F, Vec<Affine<F, Log2Term>>)> {
    match side {
        Side::Left => {
            if j == 0 {
                return Vec::new();
            }
            let h = j - 1;
            let mut terms = Vec::new();
            if let Some(z_digit) = z_digits.get(h) {
                terms.push((F::ONE, vec![term(Log2Term::RowSelector), z_digit.clone()]));
            }
            if let Some(&v_val) = v_limbs.get(h)
                && v_val != 0
            {
                terms.push((
                    F::from_u64_reduced(v_val),
                    vec![term(Log2Term::RowSelector)],
                ));
            }
            terms
        }
        Side::Right => {
            let mut terms = Vec::new();
            for (a, q_digit) in q_digits.iter().enumerate() {
                for (b, p_digit) in p_digits.iter().enumerate() {
                    if a + b == j {
                        terms.push((F::ONE, vec![q_digit.clone(), p_digit.clone()]));
                    }
                }
            }
            terms
        }
    }
}

fn decomp_limb(which: Decomp, j: u8) -> Log2Column {
    match which {
        Decomp::Total => Log2Column::TotalLimb(j),
        Decomp::Scaled => Log2Column::ScaledLimb(j),
    }
}

fn map_nonneg<F: PrimeField>(
    a: Affine<F>,
    which: Decomp,
) -> Result<Affine<F, Log2Term>, GadgetError> {
    let terms = a
        .terms
        .into_iter()
        .map(|(c, t)| {
            let mapped = match t {
                Term::Input => match which {
                    Decomp::Total => Log2Term::Total,
                    Decomp::Scaled => Log2Term::Column(Log2Column::ScaledSelected),
                },
                Term::Committed(Committed::Limb(j)) => Log2Term::Column(decomp_limb(which, j)),
                _ => return Err(unexpected()),
            };
            Ok((c, mapped))
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

fn term<F: PrimeField>(t: Log2Term) -> Affine<F, Log2Term> {
    Affine {
        constant: F::ZERO,
        terms: vec![(F::ONE, t)],
    }
}

fn cell<F: PrimeField>(c: Log2Column) -> Affine<F, Log2Term> {
    term(Log2Term::Column(c))
}

/// Clearing committed columns and per-row bound of one [`OneSidedLog2`] row (`OBS-6`, `INV-SC-3`).
///
/// `Debug` exposes only the column count, never any secret value or per-row `B_t`.
pub struct Log2Witness {
    columns: Vec<Log2Column>,
    values: SecretBuf<i64>,
    bound_units: SecretBuf<u64>,
}

impl Log2Witness {
    /// Committed columns in canonical allocation order.
    pub fn columns(&self) -> &[Log2Column] {
        &self.columns
    }

    /// Secret-class column values aligned with [`Self::columns`].
    pub fn values(&self) -> &[i64] {
        &self.values
    }

    /// Borrow one committed column's value, or `None` if absent. Secret-class (`OBS-1`).
    pub fn value(&self, column: Log2Column) -> Option<i64> {
        self.columns
            .iter()
            .position(|&c| c == column)
            .map(|idx| self.values[idx])
    }

    /// Secret-class single-row `B_t` in `2^{−8}`-bit units (`INV-SC-3`: sum into `Û_r` across the
    /// request before declassifying; never reveal individual token bounds).
    pub fn bound_units(&self) -> u64 {
        self.bound_units[0]
    }
}

impl fmt::Debug for Log2Witness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Log2Witness")
            .field("len", &self.values.len())
            .finish()
    }
}

/// Column-major clearing storage of committed [`Log2Column`]s across `padded_rows` rows.
pub struct Log2BatchColumns<B: Copy> {
    columns: Vec<(Log2Column, SecretBuf<B>)>,
}

impl<B: Copy> Log2BatchColumns<B> {
    /// Assemble from `(Log2Column, SecretBuf<B>)` pairs in canonical order.
    pub fn from_columns(columns: Vec<(Log2Column, SecretBuf<B>)>) -> Self {
        Self { columns }
    }

    /// Borrow one committed column across the batch; secret-class (`OBS-1`, `OBS-6`).
    pub fn column(&self, column: Log2Column) -> Option<&[B]> {
        self.columns
            .iter()
            .find(|(c, _)| *c == column)
            .map(|(_, vals)| &vals[..])
    }

    /// Ordered [`Log2Column`] tags held by this batch.
    pub fn names(&self) -> impl Iterator<Item = Log2Column> + '_ {
        self.columns.iter().map(|&(c, _)| c)
    }
}

impl<B: Copy> fmt::Debug for Log2BatchColumns<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.columns.iter().map(|(c, vals)| (c, vals.len())))
            .finish()
    }
}

/// Public vocabulary mask and row selector over `padded_rows` rows for batched [`OneSidedLog2`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedLog2Selector<B> {
    padded_rows: usize,
    real_rows: usize,
    valid: Vec<B>,
    row_selector: Vec<B>,
    ones: Vec<B>,
}

impl<B> BatchedLog2Selector<B> {
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

/// Evaluated virtual sums `Z_t`, `Y_t`, per-row `B_t` (secret-class), and total field claim
/// `Σ_t B_t` across `padded_rows` rows.
#[must_use]
pub struct BatchedLog2Evaluation<B: Copy> {
    /// Per-row virtual partition sums `Z_t`.
    pub totals: SecretBuf<B>,
    /// Per-row virtual recorded-token exponentials `Y_t`.
    pub selecteds: SecretBuf<B>,
    /// Per-row secret bounds `B_t` in `2^{−8}`-bit units (`INV-SC-3`).
    pub bounds: SecretBuf<B>,
    uhat_field: SecretBuf<B>,
}

impl<B: Copy> BatchedLog2Evaluation<B> {
    /// Batch field sum `Σ_t B_t` (`Û_r` in `B` when the batch spans one request).
    pub fn uhat_field(&self) -> B {
        self.uhat_field[0]
    }
}

impl<B: Copy> fmt::Debug for BatchedLog2Evaluation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchedLog2Evaluation")
            .field("rows", &self.totals.len())
            .finish_non_exhaustive()
    }
}

/// Batched vocabulary contractions and match-indicator zero-check of [`OneSidedLog2`].
#[must_use]
pub struct Log2VocabRelations<'a, F: PolyField> {
    /// `Σ_v C_v · e(t, v) = Z(t)`.
    pub total: Elementwise<'a, F>,
    /// `Σ_v h_y(t, v) · e(t, v) = Y(t)`.
    pub selected: Elementwise<'a, F>,
    /// `0 = Σ_{t,v} eq(z, (t, v)) · m(t) · (h_y(t, v) − h_ŷ(t, v))`.
    pub match_indicator: Elementwise<'a, F>,
}
