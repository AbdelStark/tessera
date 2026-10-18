// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `ColumnQueries`, `Round3Response` and `NiProof` (RFC-0036 '`ColumnQueries`, `Round3Response`
//! and `NiProof`').
//!
//! `MatrixQueries` and `MatrixColumns` are structs because TCE has no three-element tuple;
//! `MatrixQueries` encodes as the pair `(u8, Vec<u32>)` would. Opened columns and multiproofs are
//! protocol-class (05 §3), so their debug forms show counts only, as the queries' do.

use core::fmt;

use tessera_codec::{Decode, DecodeError, Encode, Reader};
use tessera_field::Goldilocks;
use tessera_hash::Digest;

use crate::matrix::{ByMatrix, decode_by_matrix, read_matrix};
use crate::{Round1, Round2Response};

/// Step 4: the verifier's column queries, drawn after `Round2Response` is fully received and
/// absorbed (schema `0x0512`, INV-PROTO-3).
///
/// Indices are in draw order and may repeat, since they are drawn with replacement
/// (RFC-0011 'Opening protocol' step 7). Every index is below its codeword's length: `n_A` for
/// the advice, `n_T = 4·k_T` for matrix `T`.
#[derive(Clone, PartialEq, Eq, Encode)]
pub struct ColumnQueries {
    /// The advice commitment's `τ_A` indices.
    pub advice: Vec<u32>,
    /// One entry per opened matrix, in `weight_combined`'s order, each with `τ_W` indices.
    pub weights: Vec<MatrixQueries>,
}

/// Counts only.
impl fmt::Debug for ColumnQueries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnQueries")
            .field("advice", &self.advice.len())
            .field("weights", &self.weights)
            .finish()
    }
}

/// `weights` is decoded by matrix id, which strictly increases.
impl Decode for ColumnQueries {
    // The advice's sequence, then the length of the weights'.
    const MIN_SIZE: usize = <Vec<u32>>::MIN_SIZE + u64::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let value = Self {
            advice: Decode::decode_from(reader)?,
            weights: decode_by_matrix(reader)?,
        };
        reader.leave();
        Ok(value)
    }
}

/// One opened matrix's queries.
///
/// It has no `Decode` of its own: an entry decodes only inside its message, which checks that
/// the ids strictly increase (RFC-0036 'Matrix ids').
#[derive(Clone, PartialEq, Eq, Encode)]
pub struct MatrixQueries {
    /// The matrix's id ([`WeightMatrix`](crate::WeightMatrix)).
    pub matrix: u8,
    /// Its `τ_W` indices, in draw order.
    pub indices: Vec<u32>,
}

/// The matrix and a count.
impl fmt::Debug for MatrixQueries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatrixQueries")
            .field("matrix", &self.matrix)
            .field("indices", &self.indices.len())
            .finish()
    }
}

/// Decoded only inside `ColumnQueries`, whose order it must keep.
impl ByMatrix for MatrixQueries {
    const MIN_SIZE: usize = u8::MIN_SIZE + <Vec<u32>>::MIN_SIZE;

    fn decode_after(reader: &mut Reader<'_>, previous: Option<u8>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let matrix = read_matrix(reader, previous)?;
        let indices = Decode::decode_from(reader)?;
        reader.leave();
        Ok(Self { matrix, indices })
    }

    fn matrix(&self) -> u8 {
        self.matrix
    }
}

/// Step 5: the answers to the queries (schema `0x0513`).
///
/// Each commitment's distinct queried indices are answered once, in increasing order, as
/// RFC-0011 step 7 and RFC-0012 say. The answer does not carry the indices, which the verifier
/// knows from its draw.
#[derive(Clone, PartialEq, Eq, Encode)]
pub struct Round3Response {
    /// `advice_columns[c]`: every non-zero row's symbol in the `c`-th distinct queried column,
    /// in segment order.
    pub advice_columns: Vec<Vec<Goldilocks>>,
    /// The canonical multiproof of the advice's distinct indices.
    pub advice_proof: MultiProof,
    /// One entry per opened matrix, in `weight_combined`'s order.
    pub weight_columns: Vec<MatrixColumns>,
}

/// Counts only: opened columns are protocol-class.
impl fmt::Debug for Round3Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Round3Response")
            .field("advice_columns", &self.advice_columns.len())
            .field(
                "advice_symbols",
                &self.advice_columns.first().map_or(0, Vec::len),
            )
            .field("advice_proof", &self.advice_proof)
            .field("weight_columns", &self.weight_columns)
            .finish()
    }
}

/// `weight_columns` is decoded by matrix id, which strictly increases.
impl Decode for Round3Response {
    // The advice's columns and proof, then the length of weight_columns.
    const MIN_SIZE: usize = <Vec<Vec<Goldilocks>>>::MIN_SIZE + MultiProof::MIN_SIZE + u64::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let value = Self {
            advice_columns: Decode::decode_from(reader)?,
            advice_proof: Decode::decode_from(reader)?,
            weight_columns: decode_by_matrix(reader)?,
        };
        reader.leave();
        Ok(value)
    }
}

/// One opened matrix's columns.
///
/// It has no `Decode` of its own: an entry decodes only inside its message, which checks that
/// the ids strictly increase (RFC-0036 'Matrix ids').
#[derive(Clone, PartialEq, Eq, Encode)]
pub struct MatrixColumns {
    /// The matrix's id ([`WeightMatrix`](crate::WeightMatrix)).
    pub matrix: u8,
    /// One column per distinct queried index, in increasing order: its non-zero data rows, and
    /// in `tessera-protocol/2` its blinding rows.
    pub columns: Vec<Vec<Goldilocks>>,
    /// The canonical multiproof of the matrix's distinct indices.
    pub proof: MultiProof,
}

/// The matrix and counts.
impl fmt::Debug for MatrixColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatrixColumns")
            .field("matrix", &self.matrix)
            .field("columns", &self.columns.len())
            .field("symbols", &self.columns.first().map_or(0, Vec::len))
            .field("proof", &self.proof)
            .finish()
    }
}

/// Decoded only inside `Round3Response`, whose order it must keep.
impl ByMatrix for MatrixColumns {
    const MIN_SIZE: usize = u8::MIN_SIZE + <Vec<Vec<Goldilocks>>>::MIN_SIZE + MultiProof::MIN_SIZE;

    fn decode_after(reader: &mut Reader<'_>, previous: Option<u8>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let matrix = read_matrix(reader, previous)?;
        let columns = Decode::decode_from(reader)?;
        let proof = Decode::decode_from(reader)?;
        reader.leave();
        Ok(Self {
            matrix,
            columns,
            proof,
        })
    }

    fn matrix(&self) -> u8 {
        self.matrix
    }
}

/// A Merkle multiproof: the siblings a verifier cannot compute, in RFC-0004's canonical order.
///
/// The wire form of `tessera_hash::MultiProof`, which has no TCE encoding.
#[derive(Clone, Default, PartialEq, Eq, Encode, Decode)]
pub struct MultiProof {
    /// The sibling digests, level by level from the leaves, left to right within a level.
    pub siblings: Vec<Digest>,
}

/// A count: siblings belong to an opening, protocol-class.
impl fmt::Debug for MultiProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiProof")
            .field("siblings", &self.siblings.len())
            .finish()
    }
}

/// A non-interactive proof (schema `0x0520`): the three prover messages as payloads, without
/// inner envelopes, and the nonces of RFC-0014 'Grinding'.
///
/// The column indices come from the transcript, so it holds no `ColumnQueries`. A tool that
/// extracts a message from it re-envelopes it at `NiProof`'s version.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct NiProof {
    /// The prover's commitments.
    pub round1: Round1,
    /// The proof of work under `seed-pow`, after which the seed is squeezed.
    pub seed_nonce: u64,
    /// The prover's messages from the seed to the openings.
    pub round2: Round2Response,
    /// The proof of work under `query-pow`, after which the indices are squeezed.
    pub query_nonce: u64,
    /// The answers to the squeezed indices.
    pub round3: Round3Response,
}
