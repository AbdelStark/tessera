// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Recomposition` (RFC-0072 'The bit zero-checks', RFC-0066 '`BitDecompose`'): the bits of a
//! `BitDecompose { low, high }` recompose its input, as one zero-check over the input's variables
//! `y`, summed over the bit axis.
//!
//! The outputs `b`, of `low` bits, and `h`, of `high` (none when `high = 0`), take
//! `k_b = ⌈log2 low⌉` and `k_h = ⌈log2 high⌉` bit variables, the lowest of their tables. With
//! `k = max(k_b, k_h)` and `n_y` the input's variables, the instance on `k + n_y` variables is
//!
//! ```text
//! 0 = Σ_{(i, y) ∈ {0,1}^{k + n_y}} eq(z_y, y)·[2^{−k}·x̃(y) − W_b(i)·b̂(i ‖ y) − 2^{low}·W_h(i)·ĥ(i ‖ y)]
//! ```
//!
//! - **Zero-extended outputs.** An output of fewer bit variables than `k` takes the *high* ones:
//!   `b̂(i ‖ y) = Z_b(i)·b̃(i_{≥ k − k_b} ‖ y)` with `Z_b(i) = Π_{t < k − k_b}(1 − i_t)`, so bit `j`
//!   of `b` sits at `i = j·2^{k − k_b}`; `ĥ` likewise with `k_h`.
//! - **The weights.** `W_b(i) = 2^j` at `i = j·2^{k − k_b}` for `j < low`, 0 elsewhere; `W_h`
//!   likewise with `high` and `k_h`. Both parties use their multilinear extensions, which hold
//!   the factor `Z_b` (`Z_h`) themselves.
//! - **For each `y`** the sum over `i` is `x(y) − Σ_{j < low} 2^j·b_j(y) − 2^{low}·Σ_{j < high}
//!   2^j·h_j(y)`, RFC-0066's recomposition, and the `eq` over `y` zero-checks it.
//!
//! **Its tables.** `eq(z_y, ·)` and `x`, each broadcast over the bit variables; `E_b = eq(z_y, ·)·
//! Z_b`, a product of functions of disjoint variables and so multilinear; `W_b`, broadcast over
//! `y`; and `b`, broadcast over the low `k − k_b` bit variables; then the same three for `h`. The
//! terms are `2^{−k}·eq·x`, `−E_b·W_b·b` and `−2^{low}·E_h·W_h·h`: exactly the polynomial above.
//! Each variable has degree at most 2 in it, but its terms have three factors, so the batch that
//! proves it has degree 3 at least, as stage 1 has from booleanity, always beside it.
//!
//! **Its claims.** At the instance's point `r' = r'_i ‖ r'_y`, the broadcast tables' final
//! evaluations are `x̃(r'_y)`, `b̃(r'_{i, ≥ k − k_b} ‖ r'_y)` and `h̃(r'_{i, ≥ k − k_h} ‖ r'_y)`: the
//! claims on `x`, unless it is public, on `b` and on `h`. The verifier evaluates `eq`, the
//! zero-extensions' products and the weights itself.

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_field::{Field, PrimeField};
use tessera_poly::PolyField;

use crate::contract::eq_high;
use crate::{Claim, Input, Instance, SumcheckError, Table, Term};

/// `Recomposition` (RFC-0072 'The bit zero-checks'): the recomposition zero-check of a
/// `BitDecompose { low, high }` of `x`, summed over its bit axis's `k` variables, with its
/// outputs zero-extended on their high bit coordinates.
///
/// It borrows `x`, of `2^{n_y}` entries, an operand or a public table; `b`, of `2^{n_y + k_b}`;
/// and `h`, of `2^{n_y + k_h}`, when `high > 0`. Each is a tensor padded per dimension, its bit
/// axis innermost, over the prime subfield.
///
/// It is proven in the steps of the other relations: [`tables`](Self::tables) builds the
/// instance's tables at the zero-check point's `eq` coordinates, [`prove_batch`] proves
/// [`RecompositionTables::instance`] among stage 1's instances, of a degree at least 3, and
/// [`RecompositionTables::claims`] gives the claims on `x`, `b` and `h`.
///
/// The operands are secret-class (05 §3): the debug form shows the widths and the variables only.
///
/// [`prove_batch`]: crate::prove_batch
pub struct Recomposition<'a, F: PolyField> {
    x: Input<'a, F>,
    b: &'a [F::Base],
    h: Option<&'a [F::Base]>,
    low: u8,
    high: u8,
    rows: usize,
}

impl<F: PolyField> Clone for Recomposition<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Recomposition<'_, F> {}

/// `⌈log2 n⌉` for `n ≥ 1`, 0 for `n = 0`.
fn bit_vars(bits: u8) -> usize {
    u32::from(bits).next_power_of_two().trailing_zeros() as usize
}

impl<'a, F: PolyField> Recomposition<'a, F> {
    /// The recomposition of `x`, on `rows` variables, into `b`, its `low` result bits, and `h`,
    /// its `high` bits above them, `None` exactly when `high = 0`.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::BitWidths`] when `low` is 0, `low + high` exceeds 62, or
    /// `h` is given exactly when `high` is 0; [`SumcheckError::RelationShape`] when the instance's
    /// `2^{k + n_y}` entries cannot be counted in a `usize`; then [`SumcheckError::InputLength`]
    /// for the first of `x` (input 0), `b` (1) and `h` (2) that does not have its `2^{n_y}`,
    /// `2^{n_y + k_b}` or `2^{n_y + k_h}` entries, or an `x` read as neither an operand nor a
    /// public table.
    pub fn new(
        x: Input<'a, F>,
        b: &'a [F::Base],
        h: Option<&'a [F::Base]>,
        low: u8,
        high: u8,
        rows: usize,
    ) -> Result<Self, SumcheckError> {
        if low == 0 || u16::from(low) + u16::from(high) > 62 || h.is_some() != (high > 0) {
            return Err(SumcheckError::BitWidths { low, high });
        }
        let k = bit_vars(low).max(bit_vars(high));
        let vars = rows.saturating_add(k);
        if vars >= usize::BITS as usize {
            return Err(SumcheckError::RelationShape {
                vars,
                summed_vars: k,
            });
        }
        let x_len = match x {
            Input::Operand(values) | Input::Public(values) => values.len(),
            Input::Partner { operand, .. } | Input::Broadcast { operand, .. } => {
                return Err(SumcheckError::InputLength {
                    input: 0,
                    len: operand.len(),
                    vars,
                });
            }
        };
        let lengths = [
            (0, x_len, 1usize << rows),
            (1, b.len(), 1usize << (rows + bit_vars(low))),
            (
                2,
                h.map_or(0, <[_]>::len),
                h.map_or(0, |_| 1 << (rows + bit_vars(high))),
            ),
        ];
        for (input, len, expected) in lengths {
            if len != expected {
                return Err(SumcheckError::InputLength { input, len, vars });
            }
        }
        Ok(Self {
            x,
            b,
            h,
            low,
            high,
            rows,
        })
    }

    /// Its degree as a batch counts it, 3: its terms have three factors, each variable of degree
    /// at most 2 (RFC-0072 'Degree').
    pub fn degree(&self) -> usize {
        3
    }

    /// The bit axis's variables, `k = max(k_b, k_h)`, the lowest of the instance's.
    pub fn bit_vars(&self) -> usize {
        bit_vars(self.low).max(bit_vars(self.high))
    }

    /// The variables of its instance, `k + n_y`.
    pub fn vars(&self) -> usize {
        self.bit_vars() + self.rows
    }

    /// The number of evaluations its instance sends: `x`'s unless it is public, `b`'s, and `h`'s
    /// when `high > 0`.
    pub fn sent(&self) -> usize {
        usize::from(!self.x.is_public()) + 1 + usize::from(self.h.is_some())
    }

    /// The instance's tables for the zero-check at `point`, the `n_y` coordinates its `eq` takes,
    /// the highest of the instance's `k + n_y` coordinates of the zero-check point (RFC-0035
    /// 'Zero-check points'). The broadcast operands are laid out in buffers that clear themselves
    /// when they drop (05 §3 OBS-6).
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have `n_y` coordinates, before
    /// anything is computed.
    pub fn tables(&self, point: &[F]) -> Result<RecompositionTables<F>, SumcheckError> {
        if point.len() != self.rows {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.rows,
            });
        }
        let k = self.bit_vars();
        let vars = self.vars();
        let eq = eq_high(point, k);
        let x = match self.x {
            Input::Public(values) => XTable::Public(spread(values, k, vars).to_vec()),
            _ => XTable::Operand(spread(self.x_values(), k, vars)),
        };
        let b = Output::new(&eq, self.b, self.low, k, vars);
        let h = self.h.map(|h| Output::new(&eq, h, self.high, k, vars));
        // `2^{−k}` and `2^{low}`: 2 is invertible in every field of odd characteristic.
        let two = F::ONE.double();
        let scale = two
            .pow(k as u64)
            .inverse()
            .ok_or(SumcheckError::BitWidths {
                low: self.low,
                high: self.high,
            })?;
        Ok(RecompositionTables {
            eq,
            x,
            b,
            h,
            scale,
            lift: two.pow(u64::from(self.low)),
            vars,
            k,
            k_b: bit_vars(self.low),
            k_h: bit_vars(self.high),
        })
    }

    fn x_values(&self) -> &'a [F::Base] {
        match self.x {
            Input::Operand(values)
            | Input::Public(values)
            | Input::Partner {
                operand: values, ..
            }
            | Input::Broadcast {
                operand: values, ..
            } => values,
        }
    }
}

/// Shows the widths and the variables, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for Recomposition<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recomposition")
            .field("low", &self.low)
            .field("high", &self.high)
            .field("rows", &self.rows)
            .field("x_public", &self.x.is_public())
            .finish_non_exhaustive()
    }
}

/// `R` laid out at `2^vars` entries, constant in the `low` lowest variables: entry `e` is
/// `R[e >> low]`.
fn spread<B: PrimeField>(operand: &[B], low: usize, vars: usize) -> SecretBuf<B> {
    let mut out = SecretBuf::with_capacity(1usize << vars);
    for &value in operand {
        out.extend(core::iter::repeat_n(value, 1usize << low));
    }
    out
}

/// `x` laid out over the instance's variables: a secret operand or a public table.
enum XTable<B: Copy> {
    Operand(SecretBuf<B>),
    Public(Vec<B>),
}

/// One output's tables: `E = eq(z_y, ·)·Z`, the weights `W`, and the output broadcast over the
/// bit variables it does not have.
struct Output<F: PolyField> {
    extension: Vec<F>,
    weights: Vec<F::Base>,
    bits: SecretBuf<F::Base>,
}

impl<F: PolyField> Output<F> {
    /// The tables of an output of `count` bits on `k + n_y = vars` variables, `eq` the broadcast
    /// `eq(z_y, ·)` and `bits` the output's table of `2^{n_y + k_o}` entries.
    fn new(eq: &[F], bits: &[F::Base], count: u8, k: usize, vars: usize) -> Self {
        let pad = k - bit_vars(count);
        let low = (1usize << pad) - 1;
        let bit_mask = (1usize << k) - 1;
        let mut extension = Vec::with_capacity(eq.len());
        let mut weights = Vec::with_capacity(eq.len());
        for (e, &value) in eq.iter().enumerate() {
            let i = e & bit_mask;
            let on = i & low == 0;
            extension.push(if on { value } else { F::ZERO });
            let j = i >> pad;
            weights.push(if on && j < usize::from(count) {
                F::Base::from_u64_reduced(1u64 << j)
            } else {
                F::Base::ZERO
            });
        }
        Self {
            extension,
            weights,
            bits: spread(bits, pad, vars),
        }
    }
}

/// The tables of a [`Recomposition`]'s instance at one zero-check point, from
/// [`Recomposition::tables`].
///
/// The broadcast operands are secret-class, as the operands are (05 §3): the debug form shows the
/// variables only.
pub struct RecompositionTables<F: PolyField> {
    eq: Vec<F>,
    x: XTable<F::Base>,
    b: Output<F>,
    h: Option<Output<F>>,
    scale: F,
    lift: F,
    vars: usize,
    k: usize,
    k_b: usize,
    k_h: usize,
}

impl<F: PolyField> RecompositionTables<F> {
    /// The instance for [`prove_batch`](crate::prove_batch), a sum of products over the tables
    /// `[eq, x, E_b, W_b, b, E_h, W_h, h]`, `h`'s three when `high > 0`, with the terms
    /// `2^{−k}·eq·x`, `−E_b·W_b·b` and `−2^{low}·E_h·W_h·h`. Its evaluations are `x`'s unless it
    /// is public, `b`'s and `h`'s, in that order.
    pub fn instance(&self) -> Instance<'_, F> {
        let mut tables = vec![
            Table::ext(&self.eq).public(),
            match &self.x {
                XTable::Operand(values) => Table::base(values),
                XTable::Public(values) => Table::base(values).public(),
            },
        ];
        let mut terms = vec![Term::new(self.scale, vec![0, 1])];
        for (output, weight) in [(Some(&self.b), -F::ONE), (self.h.as_ref(), -self.lift)] {
            let Some(output) = output else {
                continue;
            };
            let at = tables.len();
            tables.push(Table::ext(&output.extension).public());
            tables.push(Table::base(&output.weights).public());
            tables.push(Table::base(&output.bits));
            terms.push(Term::new(weight, vec![at, at + 1, at + 2]));
        }
        Instance::sum_of_products(tables, terms)
    }

    /// The claims after the sumcheck, from the instance's point `r' = r'_i ‖ r'_y`, its last
    /// `k + n_y` challenges, and its evaluations: on `x` at `r'_y` unless it is public, on `b` at
    /// `(r'_{i, ≥ k − k_b}, r'_y)` and on `h` at `(r'_{i, ≥ k − k_h}, r'_y)`.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `k + n_y`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when the evaluations are not one per
    /// table that is not public.
    pub fn claims(&self, point: &[F], evals: &[F]) -> Result<Vec<Claim<F>>, SumcheckError> {
        if point.len() != self.vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.vars,
            });
        }
        let mut points = Vec::with_capacity(3);
        if matches!(self.x, XTable::Operand(_)) {
            points.push(point[self.k..].to_vec());
        }
        points.push(point[self.k - self.k_b..].to_vec());
        if self.h.is_some() {
            points.push(point[self.k - self.k_h..].to_vec());
        }
        if evals.len() != points.len() {
            return Err(SumcheckError::EvaluationCount {
                len: evals.len(),
                expected: points.len(),
            });
        }
        Ok(points
            .into_iter()
            .zip(evals)
            .map(|(point, &value)| Claim { point, value })
            .collect())
    }
}

/// Shows the variables, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for RecompositionTables<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecompositionTables")
            .field("vars", &self.vars)
            .field("bit_vars", &self.k)
            .finish_non_exhaustive()
    }
}
