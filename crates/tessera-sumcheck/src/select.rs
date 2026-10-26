// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The selection relation (RFC-0072 'The selection relation', 'Combined selections'): the rows a
//! `Slice` or a `GatherPublic` takes of `x` along one axis, as one sumcheck over that axis
//! against a public 0/1 kernel, which leaves one claim on `x`.
//!
//! `x`'s `n = lo + k + hi` variables are its selected axis's `k`, the `lo` below them, of the axes
//! after it, and the `hi` above them, of the axes before it (RFC-0010 'Multilinear extensions':
//! the last axis takes the lowest variables). The output `y` takes rows `idx_j`, `j < len`, along
//! the axis: with `S(j, k) = [j < len]·[idx_j = k]` and a claim on `y` at `r_lo ‖ r_a ‖ r_hi`, `r_a`
//! the claim's coordinates of `y`'s axis,
//!
//! ```text
//! ỹ(r_lo ‖ r_a ‖ r_hi) = Σ_{k ∈ {0,1}^k} S̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)
//! ```
//!
//! of degree 2. A combined selection, `Σ_t a_t·y_t` over selections of one `x` along one axis at
//! one point, is the same sum with the kernel `K = Σ_t a_t·S_t` in place of `S`.
//!
//! - **Its tables**, both on `k`'s variables: the kernel `K̃(r_a, ·)`, a public factor, and
//!   `x̃(r_lo ‖ · ‖ r_hi)`, `x` contracted at the claim's coordinates of its other axes.
//! - **Its claim.** The instance's one evaluation, at its point `r'_k`, is `x̃(r_lo ‖ r'_k ‖ r_hi)`:
//!   the claim on `x` there.
//! - **The verifier's evaluation** of the kernel is `K̃(r_a, r'_k) = Σ_t a_t·Σ_{j < len_t}
//!   eq(r_a, j)·eq(r'_k, idx_{t,j})`, in `O(len·(⌈log2 len⌉ + k))`.
//! - **Padded rows.** `S` is 0 on `y`'s rows past `len`, so `y` is 0 there, and no index names a
//!   padded row of `x`: the schedule's indices are the operation's, inside `x`'s axis.

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

use crate::contract::{contract_low, eq_weights};
use crate::{Claim, Instance, SumcheckError, Table};

/// One selection of a kernel (RFC-0072 'Combined selections'): its coefficient `a_t` and its
/// rows, `index[j]` the row of `x`'s axis that output row `j` takes. A single selection's kernel
/// is one member of coefficient 1. Both are public.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected<'a, F> {
    /// The coefficient `a_t`.
    pub coefficient: F,
    /// The rows `idx_j`, `j < len`.
    pub index: &'a [usize],
}

/// The selection relation (RFC-0072 'The selection relation'): the rows of `x` along one axis
/// against a public kernel, `ỹ(r_lo ‖ r_a ‖ r_hi) = Σ_k K̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)`, of degree 2.
///
/// It borrows `x`, of `2^{lo + k + hi}` entries over the prime subfield, a tensor padded per
/// dimension whose selected axis takes the `k` variables above its `lo` lowest.
///
/// It is proven in the steps of the other relations: [`tables`](Self::tables) contracts `x` and
/// builds the kernel at the input claim's point, [`prove_batch`](crate::prove_batch) proves
/// [`SelectionTables::instance`] among the stage's instances, and [`SelectionTables::claim`]
/// gives the claim on `x`.
///
/// `x` is secret-class (05 §3): the debug form shows the variables only.
pub struct Selection<'a, F: PolyField> {
    x: &'a [F::Base],
    low: usize,
    axis: usize,
    high: usize,
}

impl<F: PolyField> Clone for Selection<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Selection<'_, F> {}

impl<'a, F: PolyField> Selection<'a, F> {
    /// The selection of rows of `x` along an axis of `axis` variables, with `low` variables
    /// below it and `high` above.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::RelationShape`] when `2^{low + axis + high}` cannot be counted in a
    /// `usize`; then [`SumcheckError::InputLength`] when `x` does not have that many entries.
    pub fn new(
        x: &'a [F::Base],
        low: usize,
        axis: usize,
        high: usize,
    ) -> Result<Self, SumcheckError> {
        let vars = low.saturating_add(axis).saturating_add(high);
        if vars >= usize::BITS as usize {
            return Err(SumcheckError::RelationShape {
                vars,
                summed_vars: 0,
            });
        }
        if x.len() != 1usize << vars {
            return Err(SumcheckError::InputLength {
                input: 0,
                len: x.len(),
                vars,
            });
        }
        Ok(Self { x, low, axis, high })
    }

    /// The degree of its sumcheck, 2: the kernel times `x`.
    pub fn degree(&self) -> usize {
        2
    }

    /// The variables of its instance, the selected axis's `k`.
    pub fn vars(&self) -> usize {
        self.axis
    }

    /// The instance's tables for a claim on the output at `point = r_lo ‖ r_a ‖ r_hi`, the
    /// lowest first, `r_a` of `rows` coordinates, the output's axis's: the kernel
    /// `K̃(r_a, ·) = Σ_t a_t·Σ_{j < len_t} eq(r_a, j)·[idx_{t,j} = ·]`, a public factor, and `x`
    /// contracted at `r_lo` and `r_hi`, in a buffer that clears itself when it drops (05 §3
    /// OBS-6).
    ///
    /// # Errors
    ///
    /// In this order, before anything is computed: [`SumcheckError::PointLength`] when the point
    /// does not have `lo + rows + hi` coordinates, or `2^rows` cannot be counted; then
    /// [`SumcheckError::SelectionRows`] for the first member that selects more than `2^rows` rows
    /// or names a row outside `x`'s axis of `2^k`.
    pub fn tables(
        &self,
        point: &[F],
        rows: usize,
        kernel: &[Selected<'_, F>],
    ) -> Result<SelectionTables<F>, SumcheckError> {
        let expected = self
            .low
            .checked_add(rows)
            .and_then(|v| v.checked_add(self.high))
            .filter(|_| rows < usize::BITS as usize);
        if Some(point.len()) != expected {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: expected.unwrap_or(usize::MAX),
            });
        }
        let width = 1usize << self.axis;
        for (member, selected) in kernel.iter().enumerate() {
            if selected.index.len() > 1usize << rows || selected.index.iter().any(|&k| k >= width) {
                return Err(SumcheckError::SelectionRows {
                    member,
                    len: selected.index.len(),
                    rows,
                    vars: self.axis,
                });
            }
        }
        let (r_lo, rest) = point.split_at(self.low);
        let (r_a, r_hi) = rest.split_at(rows);
        // K̃(r_a, k) = Σ_t a_t·Σ_{j : idx_{t,j} = k} eq(r_a, j).
        let eq_a = eq_weights(r_a);
        let mut table = vec![F::ZERO; width];
        for selected in kernel {
            for (&k, &e) in selected.index.iter().zip(&eq_a) {
                table[k] += selected.coefficient * e;
            }
        }
        // x̃(r_lo ‖ k ‖ r_hi): the low variables contracted with eq(r_lo, ·), then the high ones
        // with eq(r_hi, ·), each entry of x read once.
        let low = contract_low::<F>(self.x, &eq_weights(r_lo));
        let mut x = SecretBuf::with_capacity(width);
        x.resize(width, F::ZERO);
        for (block, &e) in low.chunks_exact(width).zip(&eq_weights(r_hi)) {
            for (sum, &t) in x.iter_mut().zip(block) {
                *sum += e * t;
            }
        }
        Ok(SelectionTables {
            kernel: table,
            x,
            r_lo: r_lo.to_vec(),
            r_hi: r_hi.to_vec(),
            vars: self.axis,
        })
    }
}

/// Shows the variables, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for Selection<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Selection")
            .field("low", &self.low)
            .field("axis", &self.axis)
            .field("high", &self.high)
            .finish_non_exhaustive()
    }
}

/// The tables of a [`Selection`]'s instance for one claim on its output, from
/// [`Selection::tables`]: the public kernel and `x` contracted at the claim's other coordinates.
///
/// The contraction is secret-class, as `x` is (05 §3): the debug form shows the variables only.
pub struct SelectionTables<F: PolyField> {
    /// `K̃(r_a, ·)`, public.
    kernel: Vec<F>,
    /// `x̃(r_lo ‖ · ‖ r_hi)`.
    x: SecretBuf<F>,
    r_lo: Vec<F>,
    r_hi: Vec<F>,
    vars: usize,
}

impl<F: PolyField> SelectionTables<F> {
    /// The instance for [`prove_batch`](crate::prove_batch): the product of the kernel, a public
    /// factor, and the contracted `x`, whose evaluation is the proof's one.
    pub fn instance(&self) -> Instance<'_, F> {
        Instance::new(vec![Table::ext(&self.kernel).public(), Table::ext(&self.x)])
    }

    /// The claim on `x̃` after the sumcheck, at `r_lo ‖ r'_k ‖ r_hi`, `r'_k` the instance's point,
    /// its last `k` challenges, with its one evaluation.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `k`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when there is not one evaluation.
    pub fn claim(&self, point: &[F], evals: &[F]) -> Result<Claim<F>, SumcheckError> {
        if point.len() != self.vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.vars,
            });
        }
        match *evals {
            [value] => Ok(Claim {
                point: [&self.r_lo[..], point, &self.r_hi[..]].concat(),
                value,
            }),
            _ => Err(SumcheckError::EvaluationCount {
                len: evals.len(),
                expected: 1,
            }),
        }
    }
}

/// Shows the variables, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for SelectionTables<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectionTables")
            .field("vars", &self.vars)
            .finish_non_exhaustive()
    }
}
