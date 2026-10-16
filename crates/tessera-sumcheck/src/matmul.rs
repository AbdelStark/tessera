// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The matmul relations of RFC-0010 'Relations', stacked over layers or heads as 'Batching
//! within a stage' stacks per-layer instances.
//!
//! Both are one product: `MatMulWeight`'s `Y = X·Wᵀ` is `MatMulAct`'s `Z = A·B` with `A = X`, its
//! tokens the rows, and `B = Wᵀ`, `W` stored as `B`'s transpose, its outputs the columns. A
//! private `Product` builds the tables and the claims' points of both; the two public types
//! name the operands, which the stage routes to different commitments, and order the
//! evaluations as RFC-0010 lists the operands.

use core::fmt;

use tessera_poly::PolyField;

use crate::contract::{contract_low, contract_middle, eq_high, eq_weights};
use crate::{Claim, Instance, SumcheckError, Table};

/// An operand of a matmul relation, named as RFC-0010 'Relations' names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    /// The weights `W` of a [`MatMulWeight`].
    W,
    /// The activations `X` of a [`MatMulWeight`].
    X,
    /// The left operand `A` of a [`MatMulAct`].
    A,
    /// The right operand `B` of a [`MatMulAct`], stored as its [`RightLayout`] says.
    B,
}

/// `W`, `X`, `A` or `B`.
impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::W => "W",
            Self::X => "X",
            Self::A => "A",
            Self::B => "B",
        })
    }
}

/// The shape of a [`MatMulWeight`]: `2^{n_ℓ}` layers, each an activation matrix `X` of `2^{n_t}`
/// tokens by `2^{n_i}` inputs and a weight matrix `W` of `2^{n_o}` outputs by `2^{n_i}` inputs.
/// Each field is the log of two of its dimension, the number of variables it takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MatMulWeightShape {
    /// `n_ℓ`: the layer index's variables, the highest of every operand. The schedule pads a
    /// layer count that is not a power of two with zero layers.
    pub layer_vars: usize,
    /// `n_t`: the token index's variables, the rows of `X` and `Y`; 0 for a single token.
    pub token_vars: usize,
    /// `n_o`: the output index's variables, the rows of `W` and the columns of `Y`.
    pub out_vars: usize,
    /// `n_i`: the input index's variables, the lowest of both operands.
    pub in_vars: usize,
}

impl MatMulWeightShape {
    /// The shape of `Y = X·Wᵀ` as a product `A·B`: `X` is `A`, its tokens the rows, and `W` is
    /// `B` stored transposed, its outputs the columns.
    fn product(self) -> MatMulActShape {
        MatMulActShape {
            stack_vars: self.layer_vars,
            row_vars: self.token_vars,
            inner_vars: self.in_vars,
            col_vars: self.out_vars,
        }
    }
}

/// The shape of a [`MatMulAct`]: `2^{n_b}` stacked products `Z = A·B`, `A` of `2^{n_a}` rows by
/// `2^{n_j}` columns and `B` of `2^{n_j}` rows by `2^{n_c}` columns. Each field is the log of two
/// of its dimension, the number of variables it takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MatMulActShape {
    /// `n_b`: the stacking index's variables, layers by heads, the highest of every operand. The
    /// schedule pads a count that is not a power of two with zero products.
    pub stack_vars: usize,
    /// `n_a`: the variables of the rows of `A` and `Z`.
    pub row_vars: usize,
    /// `n_j`: the variables of the contracted index, the columns of `A` and the rows of `B`.
    pub inner_vars: usize,
    /// `n_c`: the variables of the columns of `B` and `Z`.
    pub col_vars: usize,
}

/// How a [`MatMulAct`]'s right operand `B` is stored, both row-major with the stacking index
/// highest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RightLayout {
    /// `B` itself, `(b, j, c)`: entry `(b·2^{n_j} + j)·2^{n_c} + c` is `B[b, j, c]`, as `V` in
    /// `PV`. Its claim is at `r_c ‖ r_j ‖ ρ_b`.
    Direct,
    /// Its transpose `K`, `(b, c, j)`: entry `(b·2^{n_c} + c)·2^{n_j} + j` is
    /// `K[b, c, j] = B[b, j, c]`, as `K` in `QKᵀ`. Since `B̃(b, j, c) = K̃(b, c, j)`, the claim
    /// lands on `K` with its coordinates reordered, at `r_j ‖ r_c ‖ ρ_b` (RFC-0010 'Relations').
    Transposed,
}

/// Checks that an operand of `len` entries has the `2^{Σ vars}` its shape gives, in that order:
/// that the count fits a `usize`, then that it is the length.
fn check_operand(operand: Operand, len: usize, vars: &[usize]) -> Result<(), SumcheckError> {
    let total = vars
        .iter()
        .try_fold(0usize, |total, &v| total.checked_add(v));
    let expected = total
        .and_then(|total| u32::try_from(total).ok())
        .and_then(|total| 1usize.checked_shl(total));
    match expected {
        None => Err(SumcheckError::OperandTooLarge {
            operand,
            vars: total.unwrap_or(usize::MAX),
        }),
        Some(expected) if expected != len => Err(SumcheckError::OperandLength {
            operand,
            len,
            expected,
        }),
        Some(_) => Ok(()),
    }
}

/// Checks that a point has `expected` coordinates.
fn check_point<F>(point: &[F], expected: usize) -> Result<(), SumcheckError> {
    if point.len() == expected {
        Ok(())
    } else {
        Err(SumcheckError::PointLength {
            len: point.len(),
            expected,
        })
    }
}

/// The two evaluations an instance of either relation sends, those of its operands in the order
/// RFC-0010 lists them.
fn two_evals<F: Copy>(evals: &[F]) -> Result<[F; 2], SumcheckError> {
    match *evals {
        [first, second] => Ok([first, second]),
        _ => Err(SumcheckError::EvaluationCount {
            len: evals.len(),
            expected: 2,
        }),
    }
}

/// `parts` concatenated: a claim's point, from the instance's point and the input claim's.
fn join<F: Copy>(parts: &[&[F]]) -> Vec<F> {
    let mut point = Vec::with_capacity(parts.iter().map(|part| part.len()).sum());
    for part in parts {
        point.extend_from_slice(part);
    }
    point
}

/// The degree of a matmul's sumcheck: 3 when `eq(ρ, ·)` multiplies the two operands on
/// stacking variables, 2 without them.
fn degree(stack_vars: usize) -> usize {
    if stack_vars > 0 { 3 } else { 2 }
}

/// One factor of a matmul's instance: an operand used as it is, or contracted into a table over
/// `F`.
enum Factor<'a, F: PolyField> {
    AsIs(&'a [F::Base]),
    Contracted(Vec<F>),
}

impl<F: PolyField> Factor<'_, F> {
    fn table(&self) -> Table<'_, F> {
        match self {
            Self::AsIs(values) => Table::base(values),
            Self::Contracted(values) => Table::ext(values),
        }
    }
}

/// The tables of a stacked product `Z = A·B` for an input claim on `Z̃` at `r_c ‖ r_a ‖ ρ`, on
/// the instance's variables `(j, b)`, `j` the low ones: `eq(ρ, ·)` broadcast over `j` when there
/// are stacking variables; `Ã(b, r_a, j)`, `A` contracted over its middle variables with
/// `eq(r_a, ·)`; and `B̃(b, j, r_c)`, `B` contracted over its low variables with `eq(r_c, ·)`, or
/// over its middle ones when stored transposed. An operand with no variable to contract,
/// `n_a = 0` or `n_c = 0`, is used as it is.
struct Product<'a, F: PolyField> {
    shape: MatMulActShape,
    layout: RightLayout,
    /// `eq(ρ, ·)` on the stacking variables, empty without them.
    eq: Vec<F>,
    left: Factor<'a, F>,
    right: Factor<'a, F>,
    /// `r_a`.
    r_rows: Vec<F>,
    /// `r_c`.
    r_cols: Vec<F>,
}

impl<'a, F: PolyField> Product<'a, F> {
    /// The tables for an input claim at `point = r_c ‖ r_a ‖ ρ`, `n_c + n_a + n_b` coordinates,
    /// checked before anything is computed. The operands' lengths are their shape's.
    fn new(
        left: &'a [F::Base],
        right: &'a [F::Base],
        layout: RightLayout,
        shape: MatMulActShape,
        point: &[F],
    ) -> Result<Self, SumcheckError> {
        let MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        } = shape;
        check_point(point, col_vars + row_vars + stack_vars)?;
        let (r_cols, rest) = point.split_at(col_vars);
        let (r_rows, rho) = rest.split_at(row_vars);
        let left = if row_vars == 0 {
            Factor::AsIs(left)
        } else {
            Factor::Contracted(contract_middle(left, &eq_weights(r_rows), 1 << inner_vars))
        };
        let right = if col_vars == 0 {
            Factor::AsIs(right)
        } else {
            let weights = eq_weights(r_cols);
            Factor::Contracted(match layout {
                RightLayout::Direct => contract_low(right, &weights),
                RightLayout::Transposed => contract_middle(right, &weights, 1 << inner_vars),
            })
        };
        let eq = if stack_vars > 0 {
            eq_high(rho, inner_vars)
        } else {
            Vec::new()
        };
        Ok(Self {
            shape,
            layout,
            eq,
            left,
            right,
            r_rows: r_rows.to_vec(),
            r_cols: r_cols.to_vec(),
        })
    }

    /// The instance: `eq(ρ, ·)` as a public factor when there are stacking variables, then
    /// `first` and `second`, whose evaluations the proof sends in that order.
    fn instance<'t>(
        &'t self,
        first: &'t Factor<'a, F>,
        second: &'t Factor<'a, F>,
    ) -> Instance<'t, F> {
        let mut tables = Vec::with_capacity(3);
        if !self.eq.is_empty() {
            tables.push(Table::ext(&self.eq).public());
        }
        tables.push(first.table());
        tables.push(second.table());
        Instance::new(tables)
    }

    /// The points of the claims on `A` and on `B` as stored, from the instance's point
    /// `r_j ‖ ρ_b`: `r_j ‖ r_a ‖ ρ_b`, and `r_c ‖ r_j ‖ ρ_b`, or `r_j ‖ r_c ‖ ρ_b` when `B` is
    /// stored transposed.
    fn claim_points(&self, point: &[F]) -> Result<[Vec<F>; 2], SumcheckError> {
        check_point(point, self.shape.inner_vars + self.shape.stack_vars)?;
        let (r_j, rho) = point.split_at(self.shape.inner_vars);
        let right = match self.layout {
            RightLayout::Direct => join(&[&self.r_cols, r_j, rho]),
            RightLayout::Transposed => join(&[r_j, &self.r_cols, rho]),
        };
        Ok([join(&[r_j, &self.r_rows, rho]), right])
    }
}

/// `MatMulWeight` (RFC-0010 'Relations'): a batch of tokens' activations times a weight
/// matrix's transpose, stacked over `2^{n_ℓ}` layers. Layer `ℓ` computes
/// `Y[ℓ, t, o] = Σ_i W[ℓ, o, i]·X[ℓ, t, i]`, `Y = X·Wᵀ`, one row of `X` and `Y` per token.
///
/// It borrows its operands over the prime subfield, flattened row-major with the layer index
/// highest (RFC-0010 'Multilinear extensions'): `W` of `2^{n_ℓ + n_o + n_i}` entries, entry
/// `(ℓ·2^{n_o} + o)·2^{n_i} + i` being `W[ℓ, o, i]`, and `X` of `2^{n_ℓ + n_t + n_i}`, entry
/// `(ℓ·2^{n_t} + t)·2^{n_i} + i` being `X[ℓ, t, i]`. A claim on `Ỹ` at the point
/// `r_o ‖ r_t ‖ ρ`, `ρ` on the layer variables, is the sum over `(i, ℓ)`, `i` the low variables,
///
/// ```text
/// Ỹ(ρ, r_t, r_o) = Σ_{ℓ, i} eq(ρ, ℓ)·W̃(ℓ, r_o, i)·X̃(ℓ, r_t, i),
/// ```
///
/// of degree 3 in the layer variables and 2 in the others; a single layer (`n_ℓ = 0`) has no
/// `eq`, and degree 2. Its sumcheck ends at the instance's point `r_i ‖ ρ_ℓ` with a claim on
/// each operand: `W̃` at `r_i ‖ r_o ‖ ρ_ℓ` and `X̃` at `r_i ‖ r_t ‖ ρ_ℓ`; `eq(ρ, ρ_ℓ)` is a public
/// factor the verifier evaluates. For a weight type stacked over every layer, the claim on `W`
/// is the single claim `W̃_T(ρ_ℓ, r_o, r_i)` that RFC-0012 'Opening by the folding pass' opens.
///
/// Three steps prove it, the first two for the stage driver around [`prove_batch`]:
///
/// 1. [`tables`](Self::tables) builds the instance's tables from the input claim's point, never
///    its value; [`MatMulWeightTables::instance`] lends them to [`prove_batch`] as one instance
///    among the stage's.
/// 2. [`prove_batch`] runs the stage's sumcheck, of a degree at least [`degree`](Self::degree).
/// 3. [`MatMulWeightTables::claims`] maps the instance's point and evaluations, from the
///    [`ProvenBatch`](crate::ProvenBatch), to the claims on `W` and `X`.
///
/// In maths this is [`MatMulAct`] with `A = X` and `B = W` stored transposed, and the two share
/// their code; the claims route to different commitments, so each has its own type.
///
/// The operands are secret-class values (05 §3): the debug form shows the shape only.
///
/// [`prove_batch`]: crate::prove_batch
pub struct MatMulWeight<'a, F: PolyField> {
    w: &'a [F::Base],
    x: &'a [F::Base],
    shape: MatMulWeightShape,
}

impl<F: PolyField> Clone for MatMulWeight<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for MatMulWeight<'_, F> {}

impl<'a, F: PolyField> MatMulWeight<'a, F> {
    /// The relation on the weights `w` and the activations `x`, row-major, of the shape `shape`.
    ///
    /// # Errors
    ///
    /// Operand by operand, `W` then `X`: [`SumcheckError::OperandTooLarge`] when its shape counts
    /// `2^n` entries with `n` at least the bits of a `usize`, so that no slice has them, then
    /// [`SumcheckError::OperandLength`] when it does not have `2^{n_ℓ + n_o + n_i}` entries (`W`)
    /// or `2^{n_ℓ + n_t + n_i}` (`X`).
    pub fn new(
        w: &'a [F::Base],
        x: &'a [F::Base],
        shape: MatMulWeightShape,
    ) -> Result<Self, SumcheckError> {
        let MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        } = shape;
        check_operand(Operand::W, w.len(), &[layer_vars, out_vars, in_vars])?;
        check_operand(Operand::X, x.len(), &[layer_vars, token_vars, in_vars])?;
        Ok(Self { w, x, shape })
    }

    /// The shape.
    pub fn shape(&self) -> MatMulWeightShape {
        self.shape
    }

    /// The degree of its sumcheck: 3 over stacked layers, 2 for a single layer. The stage's
    /// degree must be at least this.
    pub fn degree(&self) -> usize {
        degree(self.shape.layer_vars)
    }

    /// The variables of its instance, `n_i + n_ℓ`.
    pub fn vars(&self) -> usize {
        self.shape.in_vars + self.shape.layer_vars
    }

    /// The instance's tables for an input claim on `Ỹ` at `point = r_o ‖ r_t ‖ ρ`,
    /// `n_o + n_t + n_ℓ` coordinates, the lowest first.
    ///
    /// They are `eq(ρ, ℓ)` broadcast over the inputs, a public factor, when there are layer
    /// variables; `W̃(ℓ, r_o, i) = Σ_o eq(r_o, o)·W[ℓ, o, i]` and
    /// `X̃(ℓ, r_t, i) = Σ_t eq(r_t, t)·X[ℓ, t, i]`, each operand contracted over its middle
    /// variables in one pass, into a table over `F` of `2^{n_ℓ + n_i}` entries. An operand with
    /// no variable to contract, `W` with `n_o = 0` or `X` with `n_t = 0`, is used as it is, over
    /// the prime subfield.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have `n_o + n_t + n_ℓ`
    /// coordinates, before anything is computed.
    pub fn tables(&self, point: &[F]) -> Result<MatMulWeightTables<'a, F>, SumcheckError> {
        let product = Product::new(
            self.x,
            self.w,
            RightLayout::Transposed,
            self.shape.product(),
            point,
        )?;
        Ok(MatMulWeightTables {
            shape: self.shape,
            product,
        })
    }
}

/// Shows the shape, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for MatMulWeight<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulWeight")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// The tables of a [`MatMulWeight`]'s instance for one input claim, from
/// [`MatMulWeight::tables`]: `eq(ρ, ·)` on the layer variables when there are any,
/// `W̃(ℓ, r_o, i)` and `X̃(ℓ, r_t, i)`, each of `2^{n_i + n_ℓ}` entries.
///
/// The contractions are secret-class, as the operands are (05 §3): the debug form shows the
/// shape only.
pub struct MatMulWeightTables<'a, F: PolyField> {
    shape: MatMulWeightShape,
    /// `X` as `A`, `W` as `B` stored transposed.
    product: Product<'a, F>,
}

impl<F: PolyField> MatMulWeightTables<'_, F> {
    /// The instance for [`prove_batch`](crate::prove_batch): `eq(ρ, ·)` as a public factor over
    /// stacked layers, then `W̃(ℓ, r_o, i)` and `X̃(ℓ, r_t, i)`, whose evaluations the proof sends
    /// in that order.
    pub fn instance(&self) -> Instance<'_, F> {
        self.product
            .instance(&self.product.right, &self.product.left)
    }

    /// The claims on the operands after the sumcheck, from the instance's point `r_i ‖ ρ_ℓ`
    /// (its last `n_i + n_ℓ` challenges, [`ProvenBatch::instance_point`]) and its evaluations
    /// `[w, x]` ([`ProvenBatch::instance_evals`]): `W̃(r_i ‖ r_o ‖ ρ_ℓ) = w` and
    /// `X̃(r_i ‖ r_t ‖ ρ_ℓ) = x`.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `n_i + n_ℓ`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when there are not 2 evaluations.
    ///
    /// [`ProvenBatch::instance_point`]: crate::ProvenBatch::instance_point
    /// [`ProvenBatch::instance_evals`]: crate::ProvenBatch::instance_evals
    pub fn claims(&self, point: &[F], evals: &[F]) -> Result<MatMulWeightClaims<F>, SumcheckError> {
        let [x_point, w_point] = self.product.claim_points(point)?;
        let [w, x] = two_evals(evals)?;
        Ok(MatMulWeightClaims {
            w: Claim {
                point: w_point,
                value: w,
            },
            x: Claim {
                point: x_point,
                value: x,
            },
        })
    }
}

/// Shows the shape, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for MatMulWeightTables<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulWeightTables")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// The claims a [`MatMulWeight`]'s sumcheck leaves, one per operand, at the instance's point
/// `r_i ‖ ρ_ℓ`: what the weight opening and the next stage owe a check.
#[must_use]
#[derive(Clone, PartialEq, Eq)]
pub struct MatMulWeightClaims<F> {
    /// `W̃(r_i ‖ r_o ‖ ρ_ℓ)`, on `n_i + n_o + n_ℓ` variables.
    pub w: Claim<F>,
    /// `X̃(r_i ‖ r_t ‖ ρ_ℓ)`, on `n_i + n_t + n_ℓ` variables.
    pub x: Claim<F>,
}

/// Shows the claims' numbers of coordinates, never a value (05 §3).
impl<F> fmt::Debug for MatMulWeightClaims<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulWeightClaims")
            .field("w", &self.w)
            .field("x", &self.x)
            .finish()
    }
}

/// `MatMulAct` (RFC-0010 'Relations'): an activation matrix times an activation matrix, such as
/// `QKᵀ` and `PV`, stacked over `2^{n_b}` products, layers by heads. Product `b` computes
/// `Z[b, a, c] = Σ_j A[b, a, j]·B[b, j, c]`.
///
/// It borrows its operands over the prime subfield, flattened row-major with the stacking index
/// highest (RFC-0010 'Multilinear extensions'): `A` of `2^{n_b + n_a + n_j}` entries, entry
/// `(b·2^{n_a} + a)·2^{n_j} + j` being `A[b, a, j]`, and `B` of `2^{n_b + n_j + n_c}`, stored as
/// [`RightLayout`] says. A claim on `Z̃` at the point `r_c ‖ r_a ‖ ρ` is the sum over `(j, b)`,
/// `j` the low variables,
///
/// ```text
/// Z̃(ρ, r_a, r_c) = Σ_{b, j} eq(ρ, b)·Ã(b, r_a, j)·B̃(b, j, r_c),
/// ```
///
/// of degree 3 in the stacking variables and 2 in the others; a single product (`n_b = 0`) has
/// no `eq`, and degree 2. Its sumcheck ends at the instance's point `r_j ‖ ρ_b` with a claim on
/// each operand: `Ã` at `r_j ‖ r_a ‖ ρ_b`, and `B̃` at `r_c ‖ r_j ‖ ρ_b`, or, stored transposed,
/// `K̃` at `r_j ‖ r_c ‖ ρ_b`; `eq(ρ, ρ_b)` is a public factor the verifier evaluates.
///
/// It is proven in the three steps of [`MatMulWeight`]: [`tables`](Self::tables), then
/// [`prove_batch`](crate::prove_batch) on [`MatMulActTables::instance`] among the stage's
/// instances, then [`MatMulActTables::claims`]. Mapping grouped-query attention's heads onto
/// shared key and value heads is the schedule's: `B` holds a product's right operand for each
/// `b`.
///
/// The operands are secret-class values (05 §3): the debug form shows the shape and the layout
/// only.
pub struct MatMulAct<'a, F: PolyField> {
    a: &'a [F::Base],
    b: &'a [F::Base],
    layout: RightLayout,
    shape: MatMulActShape,
}

impl<F: PolyField> Clone for MatMulAct<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for MatMulAct<'_, F> {}

impl<'a, F: PolyField> MatMulAct<'a, F> {
    /// The relation on the left operand `a` and the right operand `b`, stored as `layout` says,
    /// of the shape `shape`.
    ///
    /// # Errors
    ///
    /// Operand by operand, `A` then `B`: [`SumcheckError::OperandTooLarge`] when its shape counts
    /// `2^n` entries with `n` at least the bits of a `usize`, so that no slice has them, then
    /// [`SumcheckError::OperandLength`] when it does not have `2^{n_b + n_a + n_j}` entries (`A`)
    /// or `2^{n_b + n_j + n_c}` (`B`).
    pub fn new(
        a: &'a [F::Base],
        b: &'a [F::Base],
        layout: RightLayout,
        shape: MatMulActShape,
    ) -> Result<Self, SumcheckError> {
        let MatMulActShape {
            stack_vars,
            row_vars,
            inner_vars,
            col_vars,
        } = shape;
        check_operand(Operand::A, a.len(), &[stack_vars, row_vars, inner_vars])?;
        check_operand(Operand::B, b.len(), &[stack_vars, inner_vars, col_vars])?;
        Ok(Self {
            a,
            b,
            layout,
            shape,
        })
    }

    /// The shape.
    pub fn shape(&self) -> MatMulActShape {
        self.shape
    }

    /// How `B` is stored.
    pub fn layout(&self) -> RightLayout {
        self.layout
    }

    /// The degree of its sumcheck: 3 over stacked products, 2 for a single one. The stage's
    /// degree must be at least this.
    pub fn degree(&self) -> usize {
        degree(self.shape.stack_vars)
    }

    /// The variables of its instance, `n_j + n_b`.
    pub fn vars(&self) -> usize {
        self.shape.inner_vars + self.shape.stack_vars
    }

    /// The instance's tables for an input claim on `Z̃` at `point = r_c ‖ r_a ‖ ρ`,
    /// `n_c + n_a + n_b` coordinates, the lowest first.
    ///
    /// They are `eq(ρ, b)` broadcast over `j`, a public factor, when there are stacking
    /// variables; `Ã(b, r_a, j)`, `A` contracted over its middle variables with `eq(r_a, ·)`;
    /// and `B̃(b, j, r_c)`, `B` contracted over its low variables with `eq(r_c, ·)`, or over its
    /// middle ones when stored transposed. Each has `2^{n_j + n_b}` entries. An operand with no
    /// variable to contract, `n_a = 0` or `n_c = 0`, is used as it is, over the prime subfield.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have `n_c + n_a + n_b`
    /// coordinates, before anything is computed.
    pub fn tables(&self, point: &[F]) -> Result<MatMulActTables<'a, F>, SumcheckError> {
        let product = Product::new(self.a, self.b, self.layout, self.shape, point)?;
        Ok(MatMulActTables { product })
    }
}

/// Shows the shape and the layout, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for MatMulAct<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulAct")
            .field("shape", &self.shape)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// The tables of a [`MatMulAct`]'s instance for one input claim, from [`MatMulAct::tables`]:
/// `eq(ρ, ·)` on the stacking variables when there are any, `Ã(b, r_a, j)` and `B̃(b, j, r_c)`,
/// each of `2^{n_j + n_b}` entries.
///
/// The contractions are secret-class, as the operands are (05 §3): the debug form shows the shape
/// and the layout only.
pub struct MatMulActTables<'a, F: PolyField> {
    product: Product<'a, F>,
}

impl<F: PolyField> MatMulActTables<'_, F> {
    /// The instance for [`prove_batch`](crate::prove_batch): `eq(ρ, ·)` as a public factor over
    /// stacked products, then `Ã(b, r_a, j)` and `B̃(b, j, r_c)`, whose evaluations the proof
    /// sends in that order.
    pub fn instance(&self) -> Instance<'_, F> {
        self.product
            .instance(&self.product.left, &self.product.right)
    }

    /// The claims on the operands after the sumcheck, from the instance's point `r_j ‖ ρ_b`
    /// (its last `n_j + n_b` challenges, [`ProvenBatch::instance_point`]) and its evaluations
    /// `[a, b]` ([`ProvenBatch::instance_evals`]): `Ã(r_j ‖ r_a ‖ ρ_b) = a`, and
    /// `B̃(r_c ‖ r_j ‖ ρ_b) = b` or, stored transposed, `K̃(r_j ‖ r_c ‖ ρ_b) = b`.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `n_j + n_b`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when there are not 2 evaluations.
    ///
    /// [`ProvenBatch::instance_point`]: crate::ProvenBatch::instance_point
    /// [`ProvenBatch::instance_evals`]: crate::ProvenBatch::instance_evals
    pub fn claims(&self, point: &[F], evals: &[F]) -> Result<MatMulActClaims<F>, SumcheckError> {
        let [a_point, b_point] = self.product.claim_points(point)?;
        let [a, b] = two_evals(evals)?;
        Ok(MatMulActClaims {
            a: Claim {
                point: a_point,
                value: a,
            },
            b: Claim {
                point: b_point,
                value: b,
            },
        })
    }
}

/// Shows the shape and the layout, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for MatMulActTables<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulActTables")
            .field("shape", &self.product.shape)
            .field("layout", &self.product.layout)
            .finish_non_exhaustive()
    }
}

/// The claims a [`MatMulAct`]'s sumcheck leaves, one per operand, at the instance's point
/// `r_j ‖ ρ_b`: what the next stage or the openings owe a check.
#[must_use]
#[derive(Clone, PartialEq, Eq)]
pub struct MatMulActClaims<F> {
    /// `Ã(r_j ‖ r_a ‖ ρ_b)`, on `n_j + n_a + n_b` variables.
    pub a: Claim<F>,
    /// The claim on `B` as stored: `B̃(r_c ‖ r_j ‖ ρ_b)`, or `K̃(r_j ‖ r_c ‖ ρ_b)` when stored
    /// transposed, on `n_j + n_c + n_b` variables.
    pub b: Claim<F>,
}

/// Shows the claims' numbers of coordinates, never a value (05 §3).
impl<F> fmt::Debug for MatMulActClaims<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatMulActClaims")
            .field("a", &self.a)
            .field("b", &self.b)
            .finish()
    }
}
