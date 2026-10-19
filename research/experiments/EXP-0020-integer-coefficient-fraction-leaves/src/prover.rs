// SPDX-License-Identifier: Apache-2.0 OR MIT
use crate::{Prefix, SumcheckProof};
use core::{fmt, fmt::Write as _, ops::Range};
use tessera_field::PrimeField;
pub struct ProvenBatch<F> {
    proof: SumcheckProof<F>,
    gamma: F,
    point: Vec<F>,
    vars: Vec<usize>,
    evals: Vec<Range<usize>>,
}

impl<F> ProvenBatch<F> {
    /// A batch proven elsewhere in the crate, as the tree levels are ([`crate::level`]): its
    /// messages, `γ`, its point, each instance's variables and the range of its final
    /// evaluations.
    pub(crate) fn from_parts(
        proof: SumcheckProof<F>,
        gamma: F,
        point: Vec<F>,
        vars: Vec<usize>,
        evals: Vec<Range<usize>>,
    ) -> Self {
        Self {
            proof,
            gamma,
            point,
            vars,
            evals,
        }
    }

    /// The messages.
    pub fn proof(&self) -> &SumcheckProof<F> {
        &self.proof
    }

    /// The messages, by value.
    pub fn into_proof(self) -> SumcheckProof<F> {
        self.proof
    }

    /// The batching challenge `γ`.
    pub fn gamma(&self) -> &F {
        &self.gamma
    }

    /// The challenges in round order, `r_1, …, r_{n_max}`: the batch's point, its lowest
    /// variable first.
    pub fn point(&self) -> &[F] {
        &self.point
    }

    /// The number of instances.
    pub fn instances(&self) -> usize {
        self.vars.len()
    }

    /// Instance `i`'s point, the last `n_i` challenges (front-padding), or `None` past the last
    /// instance.
    pub fn instance_point(&self, i: usize) -> Option<&[F]> {
        let vars = *self.vars.get(i)?;
        self.point.get(self.point.len() - vars..)
    }

    /// Instance `i`'s final evaluations, its tables' values at its point in the order of its
    /// tables, public factors left out; `None` past the last instance.
    pub fn instance_evals(&self, i: usize) -> Option<&[F]> {
        self.proof.final_evals.get(self.evals.get(i)?.clone())
    }
}

/// Shows the shapes: the evaluations are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenBatch<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenBatch")
            .field("proof", &self.proof)
            .field("vars", &self.vars)
            .finish_non_exhaustive()
    }
}

pub(crate) struct Labels {
    prefix: Prefix,
    text: String,
}

impl Labels {
    pub(crate) fn new(prefix: Prefix) -> Self {
        Self {
            prefix,
            text: String::with_capacity(32),
        }
    }

    fn build(&mut self, suffix: fmt::Arguments<'_>) -> &str {
        self.text.clear();
        // Writing to a `String` cannot fail.
        let _ = write!(self.text, "{}/{suffix}", self.prefix);
        &self.text
    }

    /// `<prefix>/gamma`.
    fn gamma(&mut self) -> &str {
        self.build(format_args!("gamma"))
    }

    /// `<prefix>/round/{j}`, round `j` counted from 0.
    pub(crate) fn round(&mut self, j: usize) -> &str {
        self.build(format_args!("round/{j}"))
    }

    /// `<prefix>/r/{j}`, round `j` counted from 0.
    pub(crate) fn challenge(&mut self, j: usize) -> &str {
        self.build(format_args!("r/{j}"))
    }

    /// `<prefix>/evals`.
    pub(crate) fn evals(&mut self) -> &str {
        self.build(format_args!("evals"))
    }
}

pub(crate) fn padding_scale<B: PrimeField>(exponent: usize) -> B {
    B::from_u64_reduced(1 << exponent)
}
