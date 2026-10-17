// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The logits file that carries `D`'s outputs from the Rust engines to the tools that read them
//! (RFC-0003 'Auxiliary schemas'). It is secret-class (05 §3): its debug form shows its sizes,
//! never a logit.

use core::fmt;

use tessera_codec::{Decode, Encode};

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
