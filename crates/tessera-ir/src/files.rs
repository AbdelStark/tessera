// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The auxiliary files that carry token sequences and `D`'s integer logits between the front-end,
//! the Rust engines and the tools that read them (RFC-0003 'Auxiliary schemas'). Both are
//! secret-class (05 §3): their debug forms show their sizes, never a token or a logit.

use core::fmt;

use tessera_codec::{Decode, Encode};

/// Recorded token sequences, artifact `0x0601`: one sequence of token ids per sequence.
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct TokenFile {
    /// One sequence of token ids per sequence, in order.
    pub sequences: Vec<Vec<u32>>,
}

impl fmt::Debug for TokenFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lengths: Vec<usize> = self.sequences.iter().map(Vec::len).collect();
        f.debug_struct("TokenFile")
            .field("sequences", &self.sequences.len())
            .field("lengths", &lengths)
            .finish_non_exhaustive()
    }
}

/// `D`'s integer logits, artifact `0x0602`: one row of `vocab` values per position.
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct LogitsFile {
    /// The vocabulary size, every row's length.
    pub vocab: u32,
    /// One row per position, in order.
    pub positions: Vec<Vec<i64>>,
}

impl fmt::Debug for LogitsFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogitsFile")
            .field("vocab", &self.vocab)
            .field("positions", &self.positions.len())
            .finish_non_exhaustive()
    }
}
