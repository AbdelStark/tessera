// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Constraint descriptions (RFC-0009): a gadget's committed tensors, its virtual values and its
//! lookups, each lookup column an affine form over the gadget's input, its hidden class column,
//! its committed tensors and constants.
//!
//! The forms hold element by element. A multilinear extension is linear, so the same form gives a
//! column's extension at a point of its hypercube from the terms' extensions there, the constant
//! weighted by `Σ_x eq(ρ, x) = 1`: that is how a LogUp claim on a lookup's key or value column
//! becomes claims on `z`, `c` and the committed tensors.

use tessera_field::PrimeField;
use tessera_poly::PolyField;
use tessera_tables::TableId;

/// A committed tensor of a gadget (RFC-0009 'Committed'), one entry per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Committed {
    /// `y`, a requantization's output.
    Y,
    /// `z_lo`, the low part of the offset key: `RequantSplit`'s, and `RequantWide`'s when
    /// `s ≥ 1`.
    ZLo,
    /// `h0`, `RequantWide`'s middle part.
    H0,
    /// `g = clamp(h1, −2, 1)`, `RequantWide`'s top part saturated.
    G,
    /// The limb `v_j` of `NonNeg`, for `j < n − 1`.
    Limb(u8),
}

/// A virtual value of a gadget (RFC-0009 'Virtual'): an affine form the lookups read, never
/// committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Virtual {
    /// `z_off = z + 2^{b−1}`, the offset key.
    ZOff,
    /// `z_hi = (z_off − z_lo)·(2^s)^{−1}`, `RequantSplit`'s high part.
    ZHi,
    /// `w_off = (z_off − z_lo)·(2^s)^{−1}`, `RequantWide`'s key without its low part.
    WOff,
    /// `h1_off = (w_off − h0)·(2^t)^{−1}`, `RequantWide`'s top part, offset.
    H1Off,
    /// `h1 = h1_off − 2^{k−1}`, the top part, signed.
    H1,
    /// `w' = h0 + 2^t·g`, the reduced key of `RequantWide`'s output.
    WPrime,
    /// `v_{n−1} = (v − Σ_{j<n−1} v_j·2^{jL})·2^{−(n−1)L}`, `NonNeg`'s top limb.
    TopLimb,
}

/// What an affine form combines, element by element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    /// The gadget's input: a requantization's key `z`, or `NonNeg`'s `v`.
    Input,
    /// The hidden class column `c`, each element its output channel's class (RFC-0007).
    Class,
    /// A committed tensor.
    Committed(Committed),
}

/// `constant + Σ_i coefficient_i · term_i`, with public coefficients in the prime field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Affine<F, T = Term> {
    /// The constant.
    pub constant: F,
    /// The terms, each with its coefficient.
    pub terms: Vec<(F, T)>,
}

impl<F: PrimeField, T: Copy + PartialEq> Affine<F, T> {
    /// The form at one point, each term's value there given by `value`: an element's entry, or
    /// the term's multilinear extension at a point of its hypercube, over the prime field or an
    /// extension of it.
    pub fn evaluate<E: PolyField<Base = F>>(&self, mut value: impl FnMut(T) -> E) -> E {
        self.terms
            .iter()
            .fold(E::from(self.constant), |sum, &(coefficient, term)| {
                sum + value(term).mul_by_base(coefficient)
            })
    }

    /// Whether a term reads `term`.
    pub(crate) fn reads(&self, term: T) -> bool {
        self.terms.iter().any(|&(_, t)| t == term)
    }
}

/// One lookup of a gadget: `(key, value) ∈ table`, its columns affine forms. A table without
/// values, `Range`, takes the key alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup<F, T = Term> {
    /// The table.
    pub table: TableId,
    /// The key column.
    pub key: Affine<F, T>,
    /// The value column, `None` for a table without values.
    pub value: Option<Affine<F, T>>,
}

/// A gadget's constraints (RFC-0009): what it commits, its virtual values and its lookups, in
/// the order its section lists them. LogUp proves that every lookup's tuple is a row of its
/// table, except with probability `(N + T)/|Fp2|` (RFC-0008 'Relation'); the gadget's lemma says
/// that exactly one assignment of the committed tensors then passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraints<F> {
    /// The committed tensors.
    pub committed: Vec<Committed>,
    /// The virtual values, each as a form over the terms.
    pub virtuals: Vec<(Virtual, Affine<F>)>,
    /// The lookups.
    pub lookups: Vec<Lookup<F>>,
}

impl<F: PrimeField> Constraints<F> {
    /// Whether a lookup's key or value reads `term`.
    pub(crate) fn reads(&self, term: Term) -> bool {
        self.lookups.iter().any(|lookup| {
            lookup.key.reads(term) || lookup.value.as_ref().is_some_and(|v| v.reads(term))
        })
    }
}

/// `2^e` in `F`, for `e ≤ 63`.
pub(crate) fn pow2<F: PrimeField>(e: u32) -> F {
    F::from_u64_reduced(1 << e)
}

/// `2^{−e}` in `F`: `(1/2)^e`, with `1/2 = (P + 1)/2` for the odd prime `P`.
pub(crate) fn inv_pow2<F: PrimeField>(e: u32) -> F {
    F::from_u64_reduced(F::MODULUS / 2 + 1).pow(u64::from(e))
}

/// The form `constant + Σ terms`.
pub(crate) fn affine<F: Copy>(constant: F, terms: &[(F, Term)]) -> Affine<F> {
    Affine {
        constant,
        terms: terms.to_vec(),
    }
}

/// The committed tensor `tensor` alone.
pub(crate) fn tensor<F: PrimeField>(tensor: Committed) -> Affine<F> {
    affine(F::ZERO, &[(F::ONE, Term::Committed(tensor))])
}
