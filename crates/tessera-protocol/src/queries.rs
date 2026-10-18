// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The shape of a proof's column queries (RFC-0036 '`ColumnQueries`', INV-PROTO-3): the
//! verifier draws queries of that shape, and the prover answers only queries of that shape.
//!
//! Every value of a shape is public. The counts are the parameter set's `τ_A` and `τ_W`
//! (RFC-0014 'Parameter sets'), the matrices those `weight_combined` opens, and the lengths the
//! commitments' layouts give: `n_A` for the advice, `n_T = 4·k_T` for matrix `T`. An index is a
//! `u32`, so no codeword is longer than `2^32` symbols.

use crate::matrix::WeightMatrix;
use crate::{ColumnQueries, MatrixQueries, ProtocolError, QueryFault};

/// The longest codeword a `u32` index reaches.
const MAX_LEN: u64 = 1 << 32;

/// The most indices a codeword's queries take, `2^16`: far above every parameter set of
/// RFC-0014, whose `τ_A` are 155, 193 and 8 and whose `τ_W` are 97, 122 and 8, and low enough
/// that a draw reserves no more than 256 KiB of indices per codeword.
const MAX_COUNT: u32 = 1 << 16;

/// An opened weight matrix's codeword, as the queries see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixCodeword {
    /// The matrix's id ([`WeightMatrix`]).
    pub matrix: u8,
    /// Its codeword's length, `n_T = 4·k_T`.
    pub len: u64,
}

/// The column queries of a proof: how many indices each codeword takes, and the codewords'
/// lengths.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QueryShape {
    /// `τ_A`, the advice indices.
    pub advice_count: u32,
    /// `n_A`, the advice codeword's length.
    pub advice_len: u64,
    /// `τ_W`, the indices of each opened matrix.
    pub matrix_count: u32,
    /// The opened matrices in `weight_combined`'s order, `matrix` strictly increasing.
    pub matrices: Vec<MatrixCodeword>,
}

impl QueryShape {
    /// Checks that queries can have this shape: `τ_A` and `τ_W` are 1 to `2^16`, every
    /// codeword holds 1 to `2^32` symbols, and the matrices are [`WeightMatrix`] ids in strictly
    /// increasing order. Its source, the party's own engine or checks, is at fault otherwise
    /// (`ERR-INTERNAL`).
    pub(crate) fn validate(&self) -> Result<(), ProtocolError> {
        let counts = [self.advice_count, self.matrix_count];
        if counts.iter().any(|count| !(1..=MAX_COUNT).contains(count)) {
            return Err(ProtocolError::Internal {
                rule: "the parameter set's query counts are 1 to 2^16",
            });
        }
        let mut lengths =
            core::iter::once(self.advice_len).chain(self.matrices.iter().map(|m| m.len));
        if lengths.any(|len| len == 0 || len > MAX_LEN) {
            return Err(ProtocolError::Internal {
                rule: "every queried codeword holds 1 to 2^32 symbols",
            });
        }
        let ids_valid = self
            .matrices
            .iter()
            .all(|m| WeightMatrix::from_id(m.matrix).is_some());
        let increasing = self.matrices.windows(2).all(|w| w[0].matrix < w[1].matrix);
        if !ids_valid || !increasing {
            return Err(ProtocolError::Internal {
                rule: "the opened matrices are weight matrix ids in increasing order",
            });
        }
        Ok(())
    }

    /// Whether `queries` have this shape, as a prover checks them before answering
    /// (INV-PROTO-3): in this order, the matrices are this shape's, in its order; the advice
    /// holds `τ_A` indices and each matrix `τ_W`; and every index is below its codeword's
    /// length, the advice's first, in draw order. Indices may repeat.
    pub fn check(&self, queries: &ColumnQueries) -> Result<(), QueryFault> {
        let same_matrices = queries.weights.len() == self.matrices.len()
            && queries
                .weights
                .iter()
                .zip(&self.matrices)
                .all(|(q, m)| q.matrix == m.matrix);
        if !same_matrices {
            return Err(QueryFault::Matrices);
        }
        let codewords =
            core::iter::once((None, self.advice_count, self.advice_len, &queries.advice)).chain(
                queries
                    .weights
                    .iter()
                    .zip(&self.matrices)
                    .map(|(q, m)| (Some(m.matrix), self.matrix_count, m.len, &q.indices)),
            );
        for (matrix, expected, _, indices) in codewords.clone() {
            if indices.len() as u64 != u64::from(expected) {
                return Err(QueryFault::Count {
                    matrix,
                    expected,
                    found: indices.len(),
                });
            }
        }
        for (matrix, _, len, indices) in codewords {
            if let Some(position) = indices.iter().position(|&i| u64::from(i) >= len) {
                return Err(QueryFault::Range {
                    matrix,
                    position,
                    len,
                });
            }
        }
        Ok(())
    }

    /// Queries of this shape, as a verifier draws them after `Round2Response` (RFC-0011
    /// 'Opening protocol' step 7): `τ_A` advice indices, then `τ_W` for each matrix in order,
    /// each uniform below its codeword's length and independent of the others, with
    /// replacement, from the operating system's CSPRNG.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::Internal`] for a shape no queries can have, before anything is drawn or
    /// reserved, and [`ProtocolError::Randomness`] if the random source fails.
    pub fn draw(&self) -> Result<ColumnQueries, ProtocolError> {
        self.draw_from(&mut || getrandom::u64().map_err(|_| ProtocolError::Randomness))
    }

    /// [`draw`](Self::draw) from the 64-bit words `word` gives.
    fn draw_from(
        &self,
        word: &mut impl FnMut() -> Result<u64, ProtocolError>,
    ) -> Result<ColumnQueries, ProtocolError> {
        self.validate()?;
        let advice = indices(self.advice_count, self.advice_len, word)?;
        let mut weights = Vec::with_capacity(self.matrices.len());
        for codeword in &self.matrices {
            weights.push(MatrixQueries {
                matrix: codeword.matrix,
                indices: indices(self.matrix_count, codeword.len, word)?,
            });
        }
        Ok(ColumnQueries { advice, weights })
    }
}

/// `count` indices uniform in `0..len`, `1 ≤ len ≤ 2^32`, by rejection: each word is masked to
/// the bits of `len`'s next power of two, and kept if it is below `len`. Each word is kept with
/// probability above 1/2, and a kept index is exactly uniform, with no reduction modulo `len`.
fn indices(
    count: u32,
    len: u64,
    word: &mut impl FnMut() -> Result<u64, ProtocolError>,
) -> Result<Vec<u32>, ProtocolError> {
    // len ≤ 2^32, so its next power of two neither overflows nor exceeds 2^32.
    let mask = len.next_power_of_two() - 1;
    let mut drawn = Vec::with_capacity(count as usize);
    while drawn.len() < count as usize {
        let candidate = word()? & mask;
        if candidate < len {
            // candidate < len ≤ 2^32.
            drawn.push(candidate as u32);
        }
    }
    Ok(drawn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(advice_len: u64, matrices: &[(u8, u64)]) -> QueryShape {
        QueryShape {
            advice_count: 3,
            advice_len,
            matrix_count: 2,
            matrices: matrices
                .iter()
                .map(|&(matrix, len)| MatrixCodeword { matrix, len })
                .collect(),
        }
    }

    /// A source that gives `words` in order, then fails.
    fn stub(words: &[u64]) -> impl FnMut() -> Result<u64, ProtocolError> + '_ {
        let mut next = words.iter();
        move || next.next().copied().ok_or(ProtocolError::Randomness)
    }

    /// A word at or above the length after masking is drawn again; below it, it is kept as
    /// masked. Honest randomness reaches the retry only for a length that is not a power of
    /// two, which no parameter set gives, so the stub drives it.
    #[test]
    fn a_masked_word_past_the_codeword_is_drawn_again() {
        // len 5: mask 7. 7, 6 and 5 are refused, 12 masks to 4.
        assert_eq!(indices(1, 5, &mut stub(&[7, 6, 5, 12])), Ok(vec![4]));
        // len 2^32: every word is kept, masked to its low 32 bits.
        let high = u64::MAX;
        assert_eq!(
            indices(2, 1 << 32, &mut stub(&[high, 1 << 32])),
            Ok(vec![u32::MAX, 0])
        );
        // len 1: every word gives 0.
        assert_eq!(indices(3, 1, &mut stub(&[9, 10, 11])), Ok(vec![0; 3]));
        // len 6: mask 7; 6 and 7 are refused.
        assert_eq!(
            indices(3, 6, &mut stub(&[6, 7, 5, 0, 14, 3])),
            Ok(vec![5, 0, 3])
        );
    }

    /// The draw takes the advice's words first, then each matrix's in order, and fails with
    /// the source.
    #[test]
    fn the_draw_follows_the_shape_and_its_source() {
        let s = shape(8, &[(1, 4), (8, 16)]);
        let words: Vec<u64> = (0..7).map(|i| 9 + i).collect();
        let queries = s.draw_from(&mut stub(&words)).expect("a draw");
        assert_eq!(queries.advice, vec![1, 2, 3]);
        let matrices: Vec<(u8, Vec<u32>)> = queries
            .weights
            .iter()
            .map(|m| (m.matrix, m.indices.clone()))
            .collect();
        assert_eq!(matrices, vec![(1, vec![0, 1]), (8, vec![14, 15])]);
        assert_eq!(
            s.draw_from(&mut stub(&words[..6])),
            Err(ProtocolError::Randomness)
        );
    }

    /// A shape with a count of 0 or past 2^16, a codeword of no symbol or of more than 2^32,
    /// or matrices out of order or unknown, is refused before any word is drawn: the draw would
    /// never end on an empty codeword, and would reserve gigabytes for a count near 2^32.
    #[test]
    fn a_shape_no_queries_can_have_is_internal() {
        let counted = |advice_count, matrix_count| QueryShape {
            advice_count,
            matrix_count,
            ..shape(8, &[(1, 4)])
        };
        for bad in [
            counted(0, 2),
            counted(3, 0),
            counted(MAX_COUNT + 1, 2),
            counted(3, MAX_COUNT + 1),
            counted(u32::MAX, 2),
            shape(0, &[]),
            shape((1 << 32) + 1, &[]),
            shape(8, &[(1, 0)]),
            shape(8, &[(1, (1 << 32) + 1)]),
            shape(8, &[(8, 4), (1, 4)]),
            shape(8, &[(1, 4), (1, 4)]),
            shape(8, &[(10, 4)]),
        ] {
            let outcome = bad.draw_from(&mut stub(&[]));
            assert!(
                matches!(outcome, Err(ProtocolError::Internal { .. })),
                "{bad:?}: {outcome:?}"
            );
        }
        assert!(shape(1 << 32, &[(9, 1 << 32)]).validate().is_ok());
        assert!(counted(1, 1).validate().is_ok());
        assert!(counted(MAX_COUNT, MAX_COUNT).validate().is_ok());
        // Every parameter set of RFC-0014: live-64/1, ni-100/1 and test-small/1.
        for set in crate::ParamSet::ALL {
            assert!(
                counted(set.advice_queries, set.weight_queries)
                    .validate()
                    .is_ok()
            );
        }
    }
}
