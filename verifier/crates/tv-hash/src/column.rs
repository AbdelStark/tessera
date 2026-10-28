// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Chained column digests (RFC-0004 'Column digests').
//!
//! A committed matrix of `m` rows is hashed column by column, `B` rows at a time:
//!
//! ```text
//! d_j[0]   = H("column-init", commitment_id ‖ u64(j))
//! d_j[i+1] = H("column-step", d_j[i] ‖ symbols of rows iB .. min((i+1)B, m) − 1 at column j)
//! leaf_j   = d_j[ceil(m / B)]
//! ```
//!
//! Each symbol is its 8-byte TCE encoding, the canonical value little-endian, with no length
//! prefix: `B` and `m` fix how many there are. RFC-0011 chains each segment of the advice this
//! way and merges the segment digests of a column under `column-merge`.

use core::num::NonZeroU32;

use tv_field::Fp;

use crate::Digest;
use crate::domain::{Domain, DomainHasher, hash};

/// The bytes of a column step that fit the stack buffer: the chain value and 124 symbols, one
/// BLAKE3 chunk. Longer steps are hashed in pieces.
const STEP_BUFFER: usize = 1024;
/// Symbols per buffer of a long step.
const SYMBOLS_PER_PIECE: usize = STEP_BUFFER / 8;

/// The digest of column `column` of the commitment `commitment_id`, whose rows are chained
/// `block` at a time: `d_j[ceil(m / B)]` of RFC-0004, with `m = symbols.len()`.
///
/// `symbols` is the whole opened column, its `m` values in row order, `m` possibly 0, in which
/// case the digest is `d_j[0]`. Nothing fails here: `B ≥ 1` by type and every `Fp` is canonical.
/// That `m` is the committed matrix's row count is the caller's check, from the public layout.
pub fn column_digest(
    commitment_id: &Digest,
    column: u64,
    block: NonZeroU32,
    symbols: &[Fp],
) -> Digest {
    match column_digest_with_cancel(commitment_id, column, block, symbols, &mut || {
        Ok::<(), core::convert::Infallible>(())
    }) {
        Ok(digest) => digest,
        Err(never) => match never {},
    }
}

/// [`column_digest`] with cooperative cancellation (RFC-0019, SP-8).
///
/// Checks before work, between chained blocks and each 128-symbol piece of a long
/// block, and before returning a digest. No partial digest escapes on cancellation.
///
/// # Errors
/// Returns the callback's error unchanged as soon as a check fails.
pub fn column_digest_with_cancel<E>(
    commitment_id: &Digest,
    column: u64,
    block: NonZeroU32,
    symbols: &[Fp],
    check: &mut impl FnMut() -> Result<(), E>,
) -> Result<Digest, E> {
    check()?;
    let mut init = [0u8; 40];
    for (slot, byte) in init
        .iter_mut()
        .zip(commitment_id.iter().copied().chain(column.to_le_bytes()))
    {
        *slot = byte;
    }
    let mut chain = hash(Domain::ColumnInit, &init);
    // A step never holds more rows than the slice, so a block size beyond `usize` is one step.
    let rows = usize::try_from(block.get()).unwrap_or(usize::MAX);
    for step in symbols.chunks(rows) {
        check()?;
        chain = column_step(&chain, step, check)?;
    }
    check()?;
    Ok(chain)
}

/// `H("column-step", chain ‖ symbols)`.
fn column_step<E>(
    chain: &Digest,
    symbols: &[Fp],
    check: &mut impl FnMut() -> Result<(), E>,
) -> Result<Digest, E> {
    let mut buffer = [0u8; STEP_BUFFER];
    let (head, body) = buffer.split_at_mut(chain.len());
    for (slot, byte) in head.iter_mut().zip(chain) {
        *slot = *byte;
    }
    let short = symbols
        .len()
        .checked_mul(8)
        .is_some_and(|bytes| bytes <= body.len());
    if short {
        let len = encode(body, symbols);
        let used = len.saturating_add(chain.len());
        return Ok(hash(
            Domain::ColumnStep,
            buffer.get(..used).unwrap_or_default(),
        ));
    }
    let mut hasher = DomainHasher::new(Domain::ColumnStep);
    hasher.update(chain);
    for piece in symbols.chunks(SYMBOLS_PER_PIECE) {
        check()?;
        let len = encode(&mut buffer, piece);
        hasher.update(buffer.get(..len).unwrap_or_default());
    }
    Ok(hasher.finalize())
}

/// Writes the 8-byte encodings of `symbols` at the start of `out`, as many as fit, and returns
/// how many bytes it wrote.
fn encode(out: &mut [u8], symbols: &[Fp]) -> usize {
    let mut written: usize = 0;
    for (slot, symbol) in out.chunks_exact_mut(8).zip(symbols) {
        for (byte_slot, byte) in slot.iter_mut().zip(symbol.to_le_bytes()) {
            *byte_slot = byte;
        }
        written = written.saturating_add(8);
    }
    written
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn short_chain_blocks_long_blocks_and_empty_columns_are_cancellable() {
        let id = [29; 32];
        for (rows, block) in [(0usize, 1), (1024, 1), (4097, 124), (4097, u32::MAX)] {
            let symbols: Vec<_> = (0..rows).map(|i| Fp::reduce(i as u64)).collect();
            let block = NonZeroU32::new(block).unwrap();
            let expected = column_digest(&id, 9, block, &symbols);
            let mut polls = 0;
            assert_eq!(
                column_digest_with_cancel(&id, 9, block, &symbols, &mut || {
                    polls += 1;
                    Ok::<(), u32>(())
                }),
                Ok(expected)
            );
            assert!(polls >= 2 + rows.div_ceil(128));
            for stop in 1..=polls {
                let mut seen = 0;
                assert_eq!(
                    column_digest_with_cancel(&id, 9, block, &symbols, &mut || {
                        seen += 1;
                        if seen == stop { Err(73) } else { Ok(()) }
                    }),
                    Err(73)
                );
                assert_eq!(seen, stop);
            }
        }
    }
}
