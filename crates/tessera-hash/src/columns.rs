// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Chained column digests (RFC-0004 'Column digests'): a committed matrix is absorbed in blocks
//! of `B` rows, and each column keeps 32 bytes of state whatever the number of rows.
//!
//! ```text
//! d_j[0]   = H("column-init", commitment_id ‖ u64(j))
//! d_j[i+1] = H("column-step", d_j[i] ‖ symbols of rows iB .. min((i+1)B, m) − 1 at column j)
//! leaf_j   = d_j[ceil(m / B)]
//! ```
//!
//! The rows and codewords are secret-class (05 §3). The default path uses a zeroizing
//! serialization buffer; the allocation-free path streams each canonical symbol into the
//! hasher. Only the public column digests are retained.

use core::fmt;
use core::num::NonZeroUsize;

use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks, PrimeField};

use crate::{Digest, Domain, Hasher, hash};

/// Why a block of rows cannot be absorbed. The chains are unchanged when this is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnError {
    /// The block's length is not a whole number of rows.
    PartialRow {
        /// The number of symbols given.
        symbols: usize,
        /// The number of columns.
        columns: usize,
    },
    /// The block has no rows, or more than `B`.
    BlockSize {
        /// The number of rows given.
        rows: usize,
        /// The row block `B`.
        block: usize,
    },
    /// A block came after a block shorter than `B`, which must be the last.
    AfterFinalBlock,
}

impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialRow { symbols, columns } => {
                write!(f, "{symbols} symbols do not fill rows of {columns} columns")
            }
            Self::BlockSize { rows, block } => {
                write!(f, "a block holds 1 to {block} rows, not {rows}")
            }
            Self::AfterFinalBlock => write!(f, "a block followed the short final block"),
        }
    }
}

impl std::error::Error for ColumnError {}

/// The chains of every column of one committed matrix.
#[derive(Clone, Debug)]
pub struct ColumnChains {
    chains: Vec<Digest>,
    block: usize,
    closed: bool,
}

impl ColumnChains {
    /// `d_j[0] = H("column-init", commitment_id ‖ u64(j))` for each of `columns` columns, which
    /// will absorb rows in blocks of `block` (`B`, a public commitment parameter).
    pub fn new(commitment_id: &Digest, columns: NonZeroUsize, block: NonZeroUsize) -> Self {
        let chains = (0..columns.get() as u64)
            .map(|j| {
                Hasher::new(Domain::ColumnInit)
                    .update(commitment_id)
                    .update(&j.to_le_bytes())
                    .finalize()
            })
            .collect();
        Self {
            chains,
            block: block.get(),
            closed: false,
        }
    }

    /// Absorbs the next block of rows, row-major: B rows, or fewer for the last block.
    /// This path reuses one zeroizing serialization buffer across columns.
    pub fn absorb_block(&mut self, rows: &[Goldilocks]) -> Result<(), ColumnError> {
        let (columns, count) = self.check_block(rows)?;
        // One contiguous input per column: the chain, then the column's symbols in row order,
        // written over the same buffer column after column.
        let mut input = SecretBuf::filled(0u8, 32 + 8 * count);
        for (j, chain) in self.chains.iter_mut().enumerate() {
            let (head, symbols) = input.split_at_mut(32);
            head.copy_from_slice(chain);
            for (symbol, row) in symbols.chunks_exact_mut(8).zip(rows.chunks_exact(columns)) {
                symbol.copy_from_slice(&row[j].to_canonical_u64().to_le_bytes());
            }
            *chain = hash(Domain::ColumnStep, &input);
        }
        Ok(())
    }

    /// The same canonical chain update, without a serialization allocation. This trades
    /// small incremental hash updates for the exact one-block heap bound of streaming PCS.
    /// The checks and byte sequence are identical to [`Self::absorb_block`].
    pub fn absorb_block_no_alloc(&mut self, rows: &[Goldilocks]) -> Result<(), ColumnError> {
        let (columns, _) = self.check_block(rows)?;
        for (j, chain) in self.chains.iter_mut().enumerate() {
            let mut input = Hasher::new(Domain::ColumnStep);
            input.update(chain);
            for row in rows.chunks_exact(columns) {
                input.update(&row[j].to_canonical_u64().to_le_bytes());
            }
            *chain = input.finalize();
        }
        Ok(())
    }

    fn check_block(&mut self, rows: &[Goldilocks]) -> Result<(usize, usize), ColumnError> {
        let columns = self.chains.len();
        if self.closed {
            return Err(ColumnError::AfterFinalBlock);
        }
        if !rows.len().is_multiple_of(columns) {
            return Err(ColumnError::PartialRow {
                symbols: rows.len(),
                columns,
            });
        }
        let count = rows.len() / columns;
        if count == 0 || count > self.block {
            return Err(ColumnError::BlockSize {
                rows: count,
                block: self.block,
            });
        }
        self.closed = count < self.block;
        Ok((columns, count))
    }

    /// The leaves `leaf_j`, one per column.
    pub fn finish(self) -> Vec<Digest> {
        self.chains
    }
}
