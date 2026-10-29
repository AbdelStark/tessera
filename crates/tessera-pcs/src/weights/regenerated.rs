// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 weight roots and regenerated columns with one clearing codeword block.

use core::fmt;
use core::num::{NonZeroU32, NonZeroUsize};

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_hash::{
    ColumnChains, Digest, Domain, Hasher, MerkleTree, MultiProof, verify_multiproof,
};

use crate::{
    ColumnAnswer, CommitmentDescriptor, MatrixRows, ModelWeights, PcsError, RowEncoder,
    WeightCommitmentPublic, WeightLayout, WeightMatrix,
};

type Fp = Goldilocks;

/// The public metadata of a protocol/1 weight commitment (RFC-0040), without resident weights
/// or codewords. Openings regenerate rows from the caller's validated model.
///
/// Every root and answer is byte-identical to [`crate::WeightCommitment`]. Construction and
/// answering retain one block of encoded rows in a clearing buffer; only Merkle trees and
/// layout metadata persist. This is non-hiding, as protocol/1 requires.
pub struct RegeneratedWeightCommitment {
    public: WeightCommitmentPublic,
    layout: WeightLayout,
    row_block: NonZeroUsize,
    trees: Vec<MerkleTree>,
}

impl fmt::Debug for RegeneratedWeightCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegeneratedWeightCommitment")
            .field("public", &self.public)
            .field("row_block", &self.row_block)
            .finish_non_exhaustive()
    }
}

fn invalid(rule: &'static str) -> PcsError {
    PcsError::LayoutData { rule }
}

fn capacity(rows: u64, row_len: usize) -> PcsError {
    PcsError::TooLarge { rows, row_len }
}

fn zeros(len: usize) -> Result<SecretBuf<Fp>, PcsError> {
    let mut values = SecretBuf::try_with_capacity(len).map_err(|_| capacity(1, len))?;
    values.resize(len, Fp::ZERO);
    Ok(values)
}

impl RegeneratedWeightCommitment {
    /// Commits all matrices after validating every model weight (RFC-0040).
    ///
    /// The existing row encoder and column chains are used unchanged, without masks or
    /// column merging. No row survives a block. The public tree survives for canonical P5
    /// multiproofs; the model is neither copied nor retained.
    ///
    /// # Errors
    ///
    /// The layout's model validation errors precede allocation and reading. Unaddressable
    /// buffers return [`PcsError::TooLarge`]; encoding/layout failures return their typed error.
    pub fn commit(
        layout: &WeightLayout,
        row_block: NonZeroU32,
        epoch: u32,
        weights: &ModelWeights<'_>,
    ) -> Result<Self, PcsError> {
        let _commit = tessera_codec::span!(INFO, "commit_weights", epoch = epoch).entered();
        let _checked = layout.check(weights)?;
        let block = usize::try_from(row_block.get())
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| capacity(u64::from(row_block.get()), 1))?;
        let mut roots = [Digest::default(); 10];
        let mut trees = Vec::with_capacity(10);
        let mut elements = 0u64;
        for matrix in WeightMatrix::ALL {
            let leaves = encode_matrix(layout, block, epoch, matrix, weights, &[], |_, _| Ok(()))?;
            let tree = MerkleTree::new(leaves)
                .map_err(|_| invalid("the encoded weight columns form a power-of-two tree"))?;
            roots[matrix.index()] = tree.root();
            trees.push(tree);
            elements = elements
                .checked_add(
                    layout
                        .matrix(matrix)
                        .rows()
                        .checked_mul(layout.matrix(matrix).message_len() as u64)
                        .ok_or_else(|| {
                            capacity(
                                layout.matrix(matrix).rows(),
                                layout.matrix(matrix).message_len(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    capacity(
                        layout.matrix(matrix).rows(),
                        layout.matrix(matrix).message_len(),
                    )
                })?;
        }
        tessera_codec::metrics::committed(tessera_codec::metrics::Committed::Weight, elements);
        tessera_codec::event!(
            DEBUG,
            "pcs.weights_committed",
            matrices = trees.len(),
            epoch = epoch;
            "{matrices} weight matrices committed at epoch {epoch}"
        );
        Ok(Self {
            public: WeightCommitmentPublic {
                layout: *layout.params(),
                roots,
                epoch,
            },
            layout: layout.clone(),
            row_block: block,
            trees,
        })
    }

    /// The unchanged public weight artifact.
    pub fn public(&self) -> &WeightCommitmentPublic {
        &self.public
    }

    /// `R_W`, derived from the unchanged public artifact (RFC-0040).
    pub fn weights_root(&self) -> Digest {
        self.public.weights_root()
    }

    /// Regenerates the distinct queried columns, in increasing order, and their canonical
    /// multiproof (RFC-0012 'Column answers'). Repeated queries are answered once.
    ///
    /// Candidate answers stay in clearing buffers until the full regenerated matrix root
    /// matches this commitment. Changed matrix values cannot produce an answer under its old root.
    ///
    /// # Errors
    ///
    /// [`PcsError::NoQueries`], then [`PcsError::QueryRange`] for the first invalid query in
    /// draw order, then model validation/encoding errors. A regenerated root mismatch returns
    /// [`PcsError::LayoutData`] without releasing any candidate answer.
    pub fn answer(
        &self,
        matrix: WeightMatrix,
        weights: &ModelWeights<'_>,
        queries: &[u64],
    ) -> Result<ColumnAnswer, PcsError> {
        if queries.is_empty() {
            return Err(PcsError::NoQueries);
        }
        let tree = &self.trees[matrix.index()];
        let n = tree.leaves();
        if let Some((position, &query)) = queries.iter().enumerate().find(|&(_, &q)| q >= n as u64)
        {
            return Err(PcsError::QueryRange {
                position,
                query,
                columns: n,
            });
        }
        let mut distinct = queries.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        // Validate the complete model before allocating candidate answers.
        let _checked = self.layout.check(weights)?;
        let count = usize::try_from(self.layout.matrix(matrix).rows())
            .map_err(|_| capacity(self.layout.matrix(matrix).rows(), n))?;
        let mut candidates = (0..distinct.len())
            .map(|_| zeros(count))
            .collect::<Result<Vec<_>, _>>()?;
        let leaves = encode_matrix(
            &self.layout,
            self.row_block,
            self.public.epoch,
            matrix,
            weights,
            &distinct,
            |row, values| {
                for (column, &value) in candidates.iter_mut().zip(values) {
                    column[row] = value;
                }
                Ok(())
            },
        )?;
        let regenerated = MerkleTree::new(leaves)
            .map_err(|_| invalid("the regenerated weight columns form a power-of-two tree"))?;
        if regenerated.root() != tree.root() {
            return Err(invalid(
                "regenerated weights match the committed matrix root",
            ));
        }
        let proof = tree
            .open(&distinct)
            .map_err(|_| invalid("checked weight queries have a canonical multiproof"))?;
        self.check_answer(matrix, &distinct, &candidates, &proof)?;
        // Only checked answers become protocol-class. The candidate buffers clear on drop.
        let columns = candidates.iter().map(|column| column.to_vec()).collect();
        Ok(ColumnAnswer { columns, proof })
    }

    fn check_answer(
        &self,
        matrix: WeightMatrix,
        queries: &[u64],
        columns: &[SecretBuf<Fp>],
        proof: &MultiProof,
    ) -> Result<(), PcsError> {
        if columns.len() != queries.len()
            || columns
                .iter()
                .any(|c| c.len() as u64 != self.layout.matrix(matrix).rows())
        {
            return Err(invalid(
                "regenerated weight columns have the committed shape",
            ));
        }
        // The constructor obtained this block from a u32, before converting it to usize.
        let id = descriptor(
            &self.layout,
            self.row_block.get() as u32,
            self.public.epoch,
            matrix,
        )
        .commitment_id();
        let leaves = queries
            .iter()
            .zip(columns)
            .map(|(&query, column)| {
                let mut chain = Hasher::new(Domain::ColumnInit)
                    .update(&id)
                    .update(&query.to_le_bytes())
                    .finalize();
                for block in column.chunks(self.row_block.get()) {
                    let mut hash = Hasher::new(Domain::ColumnStep);
                    hash.update(&chain);
                    for value in block {
                        hash.update(&value.to_canonical_u64().to_le_bytes());
                    }
                    chain = hash.finalize();
                }
                (query, chain)
            })
            .collect::<Vec<_>>();
        let tree = &self.trees[matrix.index()];
        if !verify_multiproof(&tree.root(), tree.leaves() as u64, &leaves, proof) {
            return Err(invalid(
                "regenerated weight columns verify against the committed root",
            ));
        }
        Ok(())
    }
}

fn descriptor(
    layout: &WeightLayout,
    row_block: u32,
    epoch: u32,
    matrix: WeightMatrix,
) -> CommitmentDescriptor {
    let shape = layout.matrix(matrix);
    CommitmentDescriptor::WeightMatrix {
        layout: *layout.params(),
        matrix,
        log2_k: shape.log2_k() as u8,
        row_block,
        rows: shape.rows(),
        epoch,
    }
}

fn encode_matrix(
    layout: &WeightLayout,
    block: NonZeroUsize,
    epoch: u32,
    matrix: WeightMatrix,
    weights: &ModelWeights<'_>,
    queries: &[u64],
    mut answer: impl FnMut(usize, &[Fp]) -> Result<(), PcsError>,
) -> Result<Vec<Digest>, PcsError> {
    let shape = layout.matrix(matrix);
    let k = shape.message_len();
    let encoder = RowEncoder::new(k, u32::from(layout.params().log2_inv_rate), 0)?;
    let n = encoder.codeword_len();
    let descriptor = descriptor(
        layout,
        u32::try_from(block.get()).map_err(|_| capacity(block.get() as u64, n))?,
        epoch,
        matrix,
    );
    let columns = NonZeroUsize::new(n).ok_or_else(|| capacity(shape.rows(), n))?;
    let count = usize::try_from(shape.rows()).map_err(|_| capacity(shape.rows(), n))?;
    let rows_per_block = block.get().min(count);
    let symbols = rows_per_block
        .checked_mul(n)
        .ok_or_else(|| capacity(shape.rows(), n))?;
    let mut rows = layout.rows(matrix, weights)?;
    let mut row = zeros(k)?;
    let mut codewords = zeros(symbols)?;
    let mut selected = zeros(queries.len())?;
    let mut chains = ColumnChains::new(&descriptor.commitment_id(), columns, block);
    let mut none = || Fp::ZERO;
    let mut start = 0;
    while start < count {
        let here = rows_per_block.min(count - start);
        for within in 0..here {
            rows.read_row(start + within, &mut row)?;
            let codeword = &mut codewords[within * n..(within + 1) * n];
            encoder.encode_row(&row, &mut none, codeword)?;
            for (value, &query) in selected.iter_mut().zip(queries) {
                *value = codeword[query as usize];
            }
            answer(start + within, &selected)?;
        }
        chains
            .absorb_block_no_alloc(&codewords[..here * n])
            .map_err(|_| invalid("weight codeword blocks follow the committed row block"))?;
        codewords.fill(Fp::ZERO);
        row.fill(Fp::ZERO);
        start += here;
    }
    Ok(chains.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WeightLayoutParams, WeightShape, WeightShapes, WeightTensorRef};

    #[test]
    fn candidate_columns_are_checked_before_declassification() {
        let data = [1, 2, 3, 4, 5, 6];
        let weights = ModelWeights {
            tensors: WeightMatrix::WEIGHTS
                .into_iter()
                .map(|matrix| WeightTensorRef {
                    matrix,
                    layer: 0,
                    data: &data,
                })
                .collect(),
            channels: Vec::new(),
        };
        let layout = WeightLayout::new(
            WeightLayoutParams {
                log2_k: 4,
                log2_inv_rate: 2,
                mask_coeffs: 0,
                blinding_rows: 0,
            },
            WeightShapes {
                layers: 1,
                shapes: [WeightShape { out: 3, inputs: 2 }; 9],
                channels: Vec::new(),
            },
        )
        .unwrap();
        let commitment =
            RegeneratedWeightCommitment::commit(&layout, NonZeroU32::new(3).unwrap(), 0, &weights)
                .unwrap();
        let queries = [0, 3];
        let answer = commitment
            .answer(WeightMatrix::Embed, &weights, &queries)
            .unwrap();
        let mut candidates: Vec<SecretBuf<Fp>> = answer
            .columns
            .iter()
            .map(|c| c.iter().copied().collect())
            .collect();
        assert_eq!(
            commitment.check_answer(WeightMatrix::Embed, &queries, &candidates, &answer.proof),
            Ok(())
        );
        candidates[0][0] += Fp::ONE;
        assert_eq!(
            commitment.check_answer(WeightMatrix::Embed, &queries, &candidates, &answer.proof),
            Err(invalid(
                "regenerated weight columns verify against the committed root"
            ))
        );
        candidates[0][0] -= Fp::ONE;
        let mut bad_proof = answer.proof.clone();
        bad_proof.siblings[0][0] ^= 1;
        assert_eq!(
            commitment.check_answer(WeightMatrix::Embed, &queries, &candidates, &bad_proof),
            Err(invalid(
                "regenerated weight columns verify against the committed root"
            ))
        );
        assert_eq!(
            commitment.check_answer(
                WeightMatrix::Embed,
                &queries,
                &candidates[..1],
                &answer.proof
            ),
            Err(invalid(
                "regenerated weight columns have the committed shape"
            ))
        );
        candidates[0].clear();
        assert_eq!(
            commitment.check_answer(WeightMatrix::Embed, &queries, &candidates, &answer.proof),
            Err(invalid(
                "regenerated weight columns have the committed shape"
            ))
        );
    }
}
