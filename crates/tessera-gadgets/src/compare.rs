// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Carry normalization and comparison of integers larger than the field (RFC-0009 Lemma 4).

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_field::PrimeField;
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::{Affine, GadgetError, Lookup};

/// One of the two integers in `left ≤ right`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The left integer, `X`.
    Left,
    /// The right integer, `Y`.
    Right,
}

/// A limb column of a comparison. Limbs are least significant first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareColumn {
    /// An input limb or exact limb-product sum, before carry normalization.
    Input(Side, usize),
    /// A normalized base-`2^L` digit.
    Digit(Side, usize),
    /// The carry leaving a limb during normalization.
    Carry(Side, usize),
    /// A digit of `Y − X` modulo the full limb box.
    Difference(usize),
    /// The borrow leaving a subtraction limb.
    Borrow(usize),
}

/// The relations and lookups to prove for a comparison (RFC-0009 Lemma 4).
///
/// Inputs must already be true integers within the public bounds used by [`BigIntCompare::new`].
/// These bounds come from range analysis, re-derived by the verifier (SP-7), never from advice.
/// Every equality must be zero, and every lookup must pass; neither alone proves the comparison.
/// The equalities are the left normalization chain and its final carry, the right chain and its
/// final carry, then the subtraction chain and its final borrow. All chains start at zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompareConstraints<F> {
    /// The columns to commit: left digits/carries, right digits/carries, differences/borrows,
    /// each interleaved in ascending limb order. Input columns are not new commitments.
    pub committed: Vec<CompareColumn>,
    /// Zero-check relations, including both final carries and the final borrow.
    pub equalities: Vec<Affine<F, CompareColumn>>,
    /// Range lookups for every committed digit, carry, difference and borrow.
    pub lookups: Vec<Lookup<F, CompareColumn>>,
}

/// Parameters and range tables for exact comparison with carry normalization.
///
/// The input vectors represent `Σ_j a_j 2^{jL}`; their entries need not be normalized digits.
/// Both vectors have the same public limb count. Include enough zero high limbs that both final
/// carries vanish. An overflowing input has no passing witness: normalization never truncates it.
/// No whole integer is reduced modulo the field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigIntCompare {
    limb_bits: u8,
    bounds: [Vec<u64>; 2],
    carry_bits: [Vec<u8>; 2],
    tables: Vec<GeneratedTable>,
    equation_bound: u64,
}

impl BigIntCompare {
    /// Build a comparison from inclusive per-limb input bounds and `Range` table specifications.
    ///
    /// `1 ≤ limb_bits ≤ 32`; vectors are nonempty and equally long; every input bound is below
    /// `2^40` (RFC-0009). Carries use the narrowest nonempty bit range containing their honest
    /// bound, propagated from low limbs to high ones. All required `Range { bits }` tables must
    /// be present, including `Range(1)` and `Range(limb_bits)`. A needed range above 32 bits is
    /// refused, as are normalization equation boxes reaching `2^41`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for invalid parameters or a missing table; [`GadgetError::Spec`]
    /// for a table whose specification does not regenerate. Bounds are public, not witness data.
    pub fn new(
        limb_bits: u8,
        left_bounds: &[u64],
        right_bounds: &[u64],
        tables: &[TableSpec],
    ) -> Result<Self, GadgetError> {
        if !(1..=32).contains(&limb_bits)
            || left_bounds.is_empty()
            || left_bounds.len() != right_bounds.len()
        {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare has equally many nonempty limbs of 1 to 32 bits (RFC-0009)",
            });
        }
        let base = 1u64 << limb_bits;
        let mut carry_bits = [Vec::new(), Vec::new()];
        let mut equation_bound = 2 * base - 1;
        for (bounds, bits) in [left_bounds, right_bounds].into_iter().zip(&mut carry_bits) {
            let mut honest_carry = 0;
            let mut previous_box = 0;
            for &bound in bounds {
                if bound >= 1 << 40 {
                    return Err(GadgetError::Shape {
                        rule: "BigIntCompare input limb-product sums are below 2^40 (RFC-0009)",
                    });
                }
                honest_carry = (bound + honest_carry) / base;
                let width = (64 - honest_carry.leading_zeros()).max(1) as u8;
                if width > 32 {
                    return Err(GadgetError::Shape {
                        rule: "a BigIntCompare carry fits one Range table of at most 32 bits",
                    });
                }
                let carry_box = (1u64 << width) - 1;
                // Count the whole lookup box, not merely the honest carry (RFC-0009 Lemma 0).
                equation_bound = equation_bound
                    .max(bound + previous_box)
                    .max(base - 1 + base * carry_box);
                previous_box = carry_box;
                bits.push(width);
            }
        }
        if equation_bound >= 1 << 41 {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare normalization equation boxes are below 2^41 (RFC-0009)",
            });
        }
        let mut needed = vec![1, limb_bits];
        needed.extend(carry_bits.iter().flatten().copied());
        needed.sort_unstable();
        needed.dedup();
        let tables = needed
            .into_iter()
            .map(|bits| {
                let spec = tables
                    .iter()
                    .find(|spec| spec.kind == TableKind::Range { bits })
                    .ok_or(GadgetError::Shape {
                        rule: "BigIntCompare's digit, carry and borrow Range tables are present",
                    })?;
                GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec {
                    table: spec.id,
                    error,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            limb_bits,
            bounds: [left_bounds.to_vec(), right_bounds.to_vec()],
            carry_bits,
            tables,
            equation_bound,
        })
    }

    /// The number of input limbs on either side, including any zero high limbs.
    pub fn limbs(&self) -> usize {
        self.bounds[0].len()
    }

    #[cfg(debug_assertions)]
    pub(crate) fn limb_bits(&self) -> u8 {
        self.limb_bits
    }

    /// The range tables used, once each, in ascending bit-width order.
    pub fn tables(&self) -> &[GeneratedTable] {
        &self.tables
    }

    /// Describe normalization, subtraction and their range checks over `F`.
    ///
    /// # Errors
    /// [`GadgetError::Shape`] unless the full integer equation boxes lie strictly below
    /// `(F::MODULUS − 1)/2`, as required by RFC-0009 Lemma 0. Small-field experiments must use
    /// appropriately small limbs and input bounds. Production uses Goldilocks.
    pub fn constraints<F: PrimeField>(&self) -> Result<CompareConstraints<F>, GadgetError> {
        if self.equation_bound >= (F::MODULUS - 1) / 2 {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare equation boxes fit the centered field range (RFC-0009 Lemma 0)",
            });
        }
        let base = F::from_u64_reduced(1 << self.limb_bits);
        let mut constraints = CompareConstraints {
            committed: Vec::new(),
            equalities: Vec::new(),
            lookups: Vec::new(),
        };
        let mut commit = |column, bits| {
            let table = self
                .tables
                .iter()
                .find(|t| t.spec().kind == TableKind::Range { bits })
                .ok_or(GadgetError::Shape {
                    rule: "BigIntCompare's Range table was regenerated",
                })?;
            constraints.committed.push(column);
            constraints.lookups.push(Lookup {
                table: table.spec().id,
                key: form(vec![(F::ONE, column)]),
                value: None,
            });
            Ok::<_, GadgetError>(())
        };
        for (side, bits) in [Side::Left, Side::Right].into_iter().zip(&self.carry_bits) {
            for (j, &width) in bits.iter().enumerate() {
                let digit = CompareColumn::Digit(side, j);
                let carry = CompareColumn::Carry(side, j);
                commit(digit, self.limb_bits)?;
                commit(carry, width)?;
                let mut terms = vec![
                    (F::ONE, CompareColumn::Input(side, j)),
                    (-F::ONE, digit),
                    (-base, carry),
                ];
                if j > 0 {
                    terms.push((F::ONE, CompareColumn::Carry(side, j - 1)));
                }
                constraints.equalities.push(form(terms));
            }
            constraints.equalities.push(form(vec![(
                F::ONE,
                CompareColumn::Carry(side, self.limbs() - 1),
            )]));
        }
        for j in 0..self.limbs() {
            let difference = CompareColumn::Difference(j);
            let borrow = CompareColumn::Borrow(j);
            commit(difference, self.limb_bits)?;
            commit(borrow, 1)?;
            let mut terms = vec![
                (F::ONE, CompareColumn::Digit(Side::Right, j)),
                (-F::ONE, CompareColumn::Digit(Side::Left, j)),
                (-F::ONE, difference),
                (base, borrow),
            ];
            if j > 0 {
                terms.push((-F::ONE, CompareColumn::Borrow(j - 1)));
            }
            constraints.equalities.push(form(terms));
        }
        constraints.equalities.push(form(vec![(
            F::ONE,
            CompareColumn::Borrow(self.limbs() - 1),
        )]));
        Ok(constraints)
    }

    /// Generate the unique passing witness for a batch of comparisons, one column per limb.
    ///
    /// Each side has [`limbs`](Self::limbs) columns, all with the same number of comparisons.
    /// Inputs and the returned tensors are secret-class (05 §3); only the public shape appears
    /// in Debug/errors. All comparisons are computed before reporting invalid input (INV-ARCH-2).
    ///
    /// # Errors
    /// [`GadgetError::Shape`] for mismatched column counts/lengths; [`GadgetError::Input`] if an
    /// input is negative, exceeds its bound, overflows the full limb box, or has `X > Y`.
    pub fn witness(
        &self,
        left: &[&[i64]],
        right: &[&[i64]],
    ) -> Result<CompareWitness, GadgetError> {
        if left.len() != self.limbs() || right.len() != self.limbs() {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare inputs have the parameterized limb count",
            });
        }
        let len = left[0].len();
        if left.iter().chain(right).any(|v| v.len() != len) {
            return Err(GadgetError::Shape {
                rule: "BigIntCompare input columns have the same length",
            });
        }
        let n = self.limbs();
        let mask = (1i128 << self.limb_bits) - 1;
        let mut tensors: Vec<_> = self
            .constraints::<tessera_field::Goldilocks>()?
            .committed
            .into_iter()
            .map(|column| (column, SecretBuf::with_capacity(len)))
            .collect();
        let mut inside = true;
        for row in 0..len {
            for (side, input) in [left, right].into_iter().enumerate() {
                let mut carry = 0i128;
                for (j, column) in input.iter().enumerate() {
                    let raw = column[row];
                    inside &= (raw >= 0) & (i128::from(raw) <= i128::from(self.bounds[side][j]));
                    // Widen first: malformed i64 inputs cannot overflow the normalization sum.
                    let sum = i128::from(raw) + carry;
                    let digit = (sum & mask) as i64;
                    carry = sum >> self.limb_bits;
                    let offset = side * 2 * n + 2 * j;
                    tensors[offset].1.push(digit);
                    tensors[offset + 1].1.push(carry as i64);
                }
                inside &= carry == 0;
            }
            let mut borrow = 0i64;
            for j in 0..n {
                let x = tensors[2 * j].1[row];
                let y = tensors[2 * n + 2 * j].1[row];
                let difference = y - x - borrow;
                borrow = i64::from(difference < 0);
                tensors[4 * n + 2 * j]
                    .1
                    .push((i128::from(difference) & mask) as i64);
                tensors[4 * n + 2 * j + 1].1.push(borrow);
            }
            inside &= borrow == 0;
        }
        match inside {
            true => Ok(CompareWitness { tensors }),
            false => Err(GadgetError::Input {
                rule: "BigIntCompare inputs fit their bounds and limb box, and X ≤ Y (RFC-0009 Lemma 4)",
            }),
        }
    }
}

/// Committed comparison columns in zeroizing storage (05 §3 OBS-6).
///
/// Input columns are borrowed only during witness generation and are not retained.
pub struct CompareWitness {
    tensors: Vec<(CompareColumn, SecretBuf<i64>)>,
}

impl CompareWitness {
    #[cfg(debug_assertions)]
    pub(crate) fn from_tensors(tensors: Vec<(CompareColumn, SecretBuf<i64>)>) -> Self {
        Self { tensors }
    }

    /// Transfer the clearing columns without copying their values.
    /// The holder applies the secret borrowing rules of [`Self::tensor`].
    pub fn into_tensors(self) -> Vec<(CompareColumn, SecretBuf<i64>)> {
        self.tensors
    }

    /// Borrow a committed column, or `None` for an input or an absent limb.
    ///
    /// Entries remain secret-class; callers must not log them, copy them into uncleared storage,
    /// or lend them outside the modules listed in `secret-access.toml` (05 §3).
    pub fn tensor(&self, column: CompareColumn) -> Option<&[i64]> {
        self.tensors
            .iter()
            .find(|(which, _)| *which == column)
            .map(|(_, values)| &values[..])
    }
}

impl fmt::Debug for CompareWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.tensors.iter().map(|(column, v)| (column, v.len())))
            .finish()
    }
}

/// RFC-0050's array order of a comparison's committed columns: left digits, left carries, right
/// digits, right carries, differences, borrows, each by limb. Inputs sort after them.
pub(crate) fn array_order(column: CompareColumn) -> (u8, usize) {
    match column {
        CompareColumn::Digit(side, index) => (2 * u8::from(side == Side::Right), index),
        CompareColumn::Carry(side, index) => (1 + 2 * u8::from(side == Side::Right), index),
        CompareColumn::Difference(index) => (4, index),
        CompareColumn::Borrow(index) => (5, index),
        CompareColumn::Input(side, index) => (32 + u8::from(side == Side::Right), index),
    }
}

fn form<F: PrimeField>(terms: Vec<(F, CompareColumn)>) -> Affine<F, CompareColumn> {
    Affine {
        constant: F::ZERO,
        terms,
    }
}
