// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A sumcheck's messages (RFC-0010 'Sumcheck messages').

use core::fmt;

use tessera_codec::{Decode, Encode};

/// A sumcheck's messages, as RFC-0010 'Sumcheck messages' defines `SumcheckProof`.
///
/// Its TCE encoding is the struct's: `rounds` as `Vec<Vec<F>>`, then `final_evals` as `Vec<F>`.
///
/// Round messages and evaluations are protocol-class data (05 §3): the debug form shows how many
/// there are and how long each round is, never a value.
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub struct SumcheckProof<F> {
    /// Round `j`'s message, `j` from 0 as its label counts: the round polynomial's values
    /// `p(0), p(2), p(3), …, p(d)`, `d` of them, never `p(1)`.
    pub rounds: Vec<Vec<F>>,
    /// The final evaluations, instance by instance in the order of the batch, each instance's
    /// tables at its point in the order of its tables, public factors left out.
    pub final_evals: Vec<F>,
}

impl<F> fmt::Debug for SumcheckProof<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lengths: Vec<usize> = self.rounds.iter().map(Vec::len).collect();
        f.debug_struct("SumcheckProof")
            .field("round_lengths", &lengths)
            .field("final_evals", &self.final_evals.len())
            .finish()
    }
}
