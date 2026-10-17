// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `Linear` relation of RFC-0010 'Relations': adds, residual chains, offsets and sums over an
//! axis, whose claims become claims on their inputs with public coefficients, with no sumcheck
//! and no message.
//!
//! A target that is a public linear combination of its inputs has, at any point, the
//! combination of its inputs' extensions: `T = Σ_k c_k·I_k` gives `T̃(r) = Σ_k c_k·Ĩ_k(r)`, each
//! input at the point its [`Placement`] gives. A [`LinearClaim`] holds such a combination with
//! one value; [`LinearClaim::substitute`] replaces a target by its inputs, so a chain of linear
//! relations, such as a residual stream over its layers, ends on the chain's first inputs at
//! the claim's point.

use core::fmt;

use tessera_field::{Field, PrimeField};
use tessera_poly::PolyField;

use crate::{Claim, SumcheckError};

/// How an input of a [`Linear`] relation sits among its target's variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// The input has the target's variables, `T(y) = … + c·I(y)`: its claim is at the target's
    /// point.
    Same,
    /// The input has `vars` of the target's variables, from `low_vars` on, and the target reads
    /// it at every value of the others: `T(y) = … + c·I(y_{k+1}, …, y_{k+j})`, such as a row's
    /// `c_q` against each entry of the row in `c_q − x_qi`, or a constant with `j = 0`. Its claim
    /// is at the coordinates `k` to `k + j − 1` of the target's point.
    Broadcast {
        /// `k`: the target's low variables the input does not have.
        low_vars: usize,
        /// `j`: the input's variables.
        vars: usize,
    },
    /// The target sums the input over the input's `vars` lowest variables, its other variables
    /// being the target's: `T(y) = … + c·Σ_x I(x ‖ y)`, a `Sum` over the last axis, such as a
    /// row's sum `s_q = Σ_i e_qi`. Since `T̃(r) = 2^j·Ĩ(½, …, ½ ‖ r)`, its claim is at
    /// `(½)^j ‖ r`, with its coefficient times `2^j`.
    Sum {
        /// `j`: the variables summed out.
        vars: usize,
    },
}

/// An input of a [`Linear`] relation: a polynomial the caller names, its public coefficient,
/// and how it sits among the target's variables.
#[derive(Clone, PartialEq, Eq)]
pub struct LinearInput<P, F> {
    /// The public coefficient `c`.
    pub coefficient: F,
    /// The input polynomial.
    pub poly: P,
    /// Where its variables sit among the target's.
    pub placement: Placement,
}

/// Shows the polynomial and its placement, never the coefficient.
impl<P: fmt::Debug, F> fmt::Debug for LinearInput<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearInput")
            .field("poly", &self.poly)
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

/// `Linear` (RFC-0010 'Relations'): a target that is a public linear combination of its inputs,
/// `T = Σ_k c_k·I_k`, each input placed among the target's variables ([`Placement`]): adds and
/// subtractions, residual chains, products by public scalars, offsets and sums over an axis.
///
/// A claim on the target at `r` becomes a claim on the combination of its inputs, each at its
/// own point with a public coefficient, `Σ_k c_k·s_k·Ĩ_k(r_k) = T̃(r)` with `s_k = 2^j` for a
/// sum over `j` variables and 1 otherwise ([`terms`](Self::terms),
/// [`LinearClaim::substitute`]). It costs no sumcheck and no message.
///
/// The polynomials are named by the caller's type `P`, such as the schedule's references to its
/// tensors. A public input, such as an offset's constants, is named like any other, and
/// [`LinearClaim::evaluate_public`] moves its value to the claim's side.
#[derive(Clone, PartialEq, Eq)]
pub struct Linear<P, F> {
    target_vars: usize,
    inputs: Vec<LinearInput<P, F>>,
}

impl<P: Clone, F: PolyField> Linear<P, F> {
    /// The relation of a target on `target_vars` variables to `inputs`.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::InputPlacement`] for the first input that does not sit within the
    /// target's variables: a broadcast whose `k + j` exceeds them, or a sum whose input's
    /// `target_vars + j` variables a `usize` cannot count entries of.
    pub fn new(target_vars: usize, inputs: Vec<LinearInput<P, F>>) -> Result<Self, SumcheckError> {
        for (input, value) in inputs.iter().enumerate() {
            let fits = match value.placement {
                Placement::Same => true,
                Placement::Broadcast { low_vars, vars } => low_vars
                    .checked_add(vars)
                    .is_some_and(|top| top <= target_vars),
                Placement::Sum { vars } => target_vars
                    .checked_add(vars)
                    .is_some_and(|all| all < usize::BITS as usize),
            };
            if !fits {
                return Err(SumcheckError::InputPlacement { input, target_vars });
            }
        }
        Ok(Self {
            target_vars,
            inputs,
        })
    }

    /// The target's variables.
    pub fn target_vars(&self) -> usize {
        self.target_vars
    }

    /// The inputs, in order.
    pub fn inputs(&self) -> &[LinearInput<P, F>] {
        &self.inputs
    }

    /// The terms a claim on the target at `point` becomes, one per input in order: its
    /// coefficient, times `2^j` for a sum over `j` variables, at the point its placement gives,
    /// `point` itself, its coordinates `k` to `k + j − 1`, or `(½)^j ‖ point`.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have the target's coordinates.
    pub fn terms(&self, point: &[F]) -> Result<Vec<LinearTerm<P, F>>, SumcheckError> {
        if point.len() != self.target_vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: self.target_vars,
            });
        }
        let half = F::from(F::Base::from_u64_reduced(F::Base::MODULUS / 2 + 1));
        Ok(self
            .inputs
            .iter()
            .map(|input| {
                let (coefficient, point) = match input.placement {
                    Placement::Same => (input.coefficient, point.to_vec()),
                    Placement::Broadcast { low_vars, vars } => {
                        (input.coefficient, point[low_vars..low_vars + vars].to_vec())
                    }
                    Placement::Sum { vars } => {
                        let scale = (0..vars).fold(F::ONE, |scale, _| scale.double());
                        let mut at = vec![half; vars];
                        at.extend_from_slice(point);
                        (input.coefficient * scale, at)
                    }
                };
                LinearTerm {
                    coefficient,
                    poly: input.poly.clone(),
                    point,
                }
            })
            .collect())
    }
}

/// Shows the target's variables and the inputs' polynomials and placements, never a
/// coefficient.
impl<P: fmt::Debug, F> fmt::Debug for Linear<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Linear")
            .field("target_vars", &self.target_vars)
            .field("inputs", &self.inputs)
            .finish()
    }
}

/// One term of a [`LinearClaim`], `c·P̃(r)`.
#[derive(Clone, PartialEq, Eq)]
pub struct LinearTerm<P, F> {
    /// The public coefficient `c`.
    pub coefficient: F,
    /// The polynomial `P`.
    pub poly: P,
    /// The point `r`, its lowest variable first.
    pub point: Vec<F>,
}

/// Shows the polynomial and the number of coordinates, never the coefficient (05 §3).
impl<P: fmt::Debug, F> fmt::Debug for LinearTerm<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearTerm")
            .field("poly", &self.poly)
            .field("vars", &self.point.len())
            .finish_non_exhaustive()
    }
}

/// A claim on a public linear combination of polynomials, each at its own point:
/// `Σ_t c_t·P̃_t(r_t) = v`.
///
/// A claim on a [`Linear`] target becomes one ([`substitute`](Self::substitute)), and a chain of
/// them stays one: a claim on a residual stream after `L` layers ends on the stream's first
/// state and every layer's update, at the claim's point. What consumes it, an opening or the
/// next stage's instances, checks the combination as a whole.
///
/// The value is protocol-class data, a function of secret-class tables (05 §3): the debug form
/// shows the terms' polynomials and numbers of coordinates only.
#[must_use]
#[derive(Clone, PartialEq, Eq)]
pub struct LinearClaim<P, F> {
    /// The terms, in order.
    pub terms: Vec<LinearTerm<P, F>>,
    /// The claimed value `v` of the combination.
    pub value: F,
}

impl<P, F: Field> LinearClaim<P, F> {
    /// The claim `P̃(r) = v` on one polynomial, with coefficient 1.
    pub fn single(poly: P, claim: Claim<F>) -> Self {
        Self {
            terms: vec![LinearTerm {
                coefficient: F::ONE,
                poly,
                point: claim.point,
            }],
            value: claim.value,
        }
    }

    /// Substitutes `relation` for its target: each term on `target`, `c·T̃(r)`, becomes the
    /// relation's [`terms`](Linear::terms) at `r`, each coefficient times `c`, in place. Terms on
    /// other polynomials keep their places, and the value is unchanged.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when a term on `target` has a point other than the
    /// relation's target's number of coordinates. The claim is then unchanged.
    pub fn substitute(&mut self, target: &P, relation: &Linear<P, F>) -> Result<(), SumcheckError>
    where
        P: PartialEq + Clone,
        F: PolyField,
    {
        let mut terms = Vec::with_capacity(self.terms.len() + relation.inputs.len());
        for term in &self.terms {
            if term.poly != *target {
                terms.push(term.clone());
                continue;
            }
            for input in relation.terms(&term.point)? {
                terms.push(LinearTerm {
                    coefficient: term.coefficient * input.coefficient,
                    ..input
                });
            }
        }
        self.terms = terms;
        Ok(())
    }

    /// Moves to the value each term whose polynomial `evaluate` evaluates, `Some(P̃(r))` for a
    /// public polynomial such as an offset's constants: `v ← v − c·P̃(r)`, and the term leaves.
    /// The other terms keep their order.
    pub fn evaluate_public(&mut self, mut evaluate: impl FnMut(&P, &[F]) -> Option<F>) {
        let mut value = self.value;
        self.terms
            .retain(|term| match evaluate(&term.poly, &term.point) {
                Some(at) => {
                    value -= term.coefficient * at;
                    false
                }
                None => true,
            });
        self.value = value;
    }

    /// `Σ_t c_t·P̃_t(r_t)`, each `P̃_t(r_t)` from `evaluate`: the claim holds when this is its
    /// value.
    pub fn combination(&self, mut evaluate: impl FnMut(&P, &[F]) -> F) -> F {
        self.terms.iter().fold(F::ZERO, |sum, term| {
            sum + term.coefficient * evaluate(&term.poly, &term.point)
        })
    }
}

/// Shows the terms' polynomials and numbers of coordinates, never a value (05 §3).
impl<P: fmt::Debug, F> fmt::Debug for LinearClaim<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearClaim")
            .field("terms", &self.terms)
            .finish_non_exhaustive()
    }
}
