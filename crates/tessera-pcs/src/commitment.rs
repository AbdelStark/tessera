// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference tensor commitment of RFC-0011: a resident matrix, committed and opened without
//! streaming ('Column digests and root', 'Opening protocol'). A weight matrix of RFC-0012 is the
//! same scheme with other parameters: rate 1/4, one chain per column, and no mask in
//! `tessera-protocol/1` (RFC-0040 'Encoding, leaves and roots').
//!
//! # Commitment
//!
//! The matrix has `2^row_vars` rows of `k` values. Its [`RowLayout`] lists the non-zero rows,
//! segment by segment; every other row is zero, and is never encoded, hashed or opened.
//! [`ReferenceCommitment::commit`] runs four steps:
//!
//! 1. It draws each non-zero row's `P_A` mask coefficients from the [`MaskSource`], `m_0` first,
//!    row by row in layout order, and encodes the row with them ([`RowEncoder`], RFC-0011
//!    'Encoding'): `n = 2^r·k` symbols on the coset `7·L`, `n = 2k` for the advice.
//! 2. It chains each segment's columns over that segment's codewords, `B` rows per step
//!    ([`ColumnChains`], RFC-0004 'Column digests'). This gives `d_j^(s)`, which for an empty
//!    segment is the chain's first digest, `H("column-init", commitment_id ‖ u64(j))`.
//! 3. It merges each column's segment digests,
//!    `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))`, for every `S ≥ 1`.
//! 4. It builds the Merkle tree over the `n` leaves ([`MerkleTree`]): its root is the
//!    commitment.
//!
//! [`ReferenceCommitment::commit_unmasked`] commits a weight matrix of `tessera-protocol/1`: no
//! mask is drawn in step 1, and its layout has one segment, whose chain's last digest is the leaf,
//! `leaf_j = d_j[⌈m/B⌉]`, with no `column-merge` in step 3 (RFC-0040). A matrix of no row has the
//! leaves `d_j[0]`.
//!
//! The commitment keeps what its openings need, resident: the rows, the masks, the codewords,
//! the leaves and the tree. That is `N·((1 + 2^r)·k + P)` field elements and `3n` digests for `N`
//! non-zero rows, which is why it is a reference for tests and measurements: the streaming
//! committer of RFC-0011 'Streaming procedure' holds one block of `B` codewords at a time and
//! 32 bytes of chain state per column and segment.
//!
//! # Openings
//!
//! [`ReferenceCommitment::open`] forms the combined rows of RFC-0011 steps 2 to 5
//! ([`opening`](crate::opening) has the algorithm). [`ReferenceCommitment::answer`] answers
//! the column queries of step 7. It takes the drawn indices in draw order, possibly repeated,
//! and returns the columns of their sorted, distinct set, in increasing index order, each with
//! one symbol per non-zero row in layout order, together with the canonical multiproof of that
//! set ([`MerkleTree::open`], RFC-0004 'Merkle trees').
//!
//! The protocol fixes the order of these calls, and this crate does not police it: the
//! combined rows are sent before the verifier draws the queries, and a commitment is opened and
//! answered once. Each further combination or column reveals more of the rows. With the
//! encoding mask alone, `τ_A ≤ P_A` opened columns are uniform and independent of the rows
//! (RFC-0013 'Hiding openings'); the combined rows reveal their combinations of the rows until
//! RFC-0013's blinding rows hide them (#149).
//!
//! # Secrets
//!
//! Rows, masks and codewords are secret-class (05 §3). Nothing here logs; errors carry lengths,
//! positions and public indices only ([`PcsError`]). A commitment's `Debug` form shows its
//! shape, and it has no `Display` and no `Clone`. Control flow depends on lengths, the public
//! layout and public indices, never on a row, a mask or a symbol (INV-ARCH-2). The commitment
//! takes its rows in their `Secret` and keeps them there, and holds the masks and codewords it
//! derives in `SecretBuf`s: all three are cleared when it drops (05 §3 OBS-1, OBS-6). The encoder
//! keeps nothing; it writes each codeword in the buffer it is given.

use core::fmt;
use core::num::NonZeroUsize;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, Goldilocks, GoldilocksExt2};
use tessera_hash::{ColumnChains, Digest, Domain, Hasher, MerkleTree, MultiProof};

use crate::ntt::TWO_ADICITY;
use crate::opening::combine;
use crate::{ClaimGroup, CombinedRow, MaskSource, PcsError, RowEncoder};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// The public parameters of a commitment (RFC-0011 'Encoding', 'Column digests and root'): the
/// code, the row block and the identifier the column chains bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    /// `log2 k`. A row holds `k = 2^log2_k` values, the evaluations of a polynomial of degree
    /// below `k` on the subgroup `H` of order `k`.
    pub log2_k: u32,
    /// `r`, the `log2` of the inverse rate: a row's codeword holds `n = 2^r·k` symbols. 1 for the
    /// advice, rate 1/2 (RFC-0011 'Encoding'), 2 for a weight matrix, rate 1/4 (RFC-0012
    /// 'Layout'). At least 1, with `log2_k + r ≤ 32`, so that `n ≤ 2^32`.
    pub log2_inv_rate: u32,
    /// `P`, the mask coefficients of each row: `P_A`, from 1 to `k`, for the advice, and 0 for a
    /// weight matrix in `tessera-protocol/1` (RFC-0040).
    ///
    /// Hiding needs `P_A ≥ τ_A`. Soundness needs a rate `ρ = (k + P_A)/n` well below 1: at
    /// `P_A = k` the code is all of `Fp^n`, and the parameter sets keep `ρ ≤ 0.51` (RFC-0011
    /// 'Encoding', RFC-0014 'Parameter sets'). Both are rules of the parameter sets, which
    /// this crate does not check.
    pub mask_coeffs: usize,
    /// `B`, the rows a column chain absorbs per step (RFC-0004 'Column digests'): the parameter
    /// set's row block, 32 in production and 4 at `test-small`, for the advice and the weights
    /// alike (RFC-0031 'The weight matrices' row block').
    pub row_block: NonZeroUsize,
    /// The 32 bytes bound into the first digest of every column chain. They identify the
    /// commitment's descriptor, so that columns of different commitments never collide
    /// (RFC-0011 'Column digests and root', RFC-0031).
    pub commitment_id: Digest,
}

/// `k`, `n` and `P` of parameters that passed their checks, and how a column's leaf follows from
/// its chains.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) mask_coeffs: usize,
    pub(crate) leaves: Leaves,
}

/// How a column's leaf follows from its chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Leaves {
    /// The advice's: `H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))` over every segment's chain
    /// (RFC-0011 'Column digests and root').
    Merged,
    /// A weight matrix's: its one chain's last digest, `d_j[⌈m/B⌉]`, with no `column-merge`
    /// (RFC-0040 'Encoding, leaves and roots').
    Chained,
}

impl CommitmentParams {
    /// `k` and `n = 2^r·k`, checked in this order: [`PcsError::InverseRate`] if `r = 0`, then
    /// [`PcsError::TooLong`] unless `n ≤ 2^32`.
    fn code(&self) -> Result<(usize, usize), PcsError> {
        if self.log2_inv_rate == 0 {
            return Err(PcsError::InverseRate {
                log2_inv_rate: self.log2_inv_rate,
            });
        }
        let log_n = self.log2_k.saturating_add(self.log2_inv_rate);
        let too_long = PcsError::TooLong { log_len: log_n };
        if log_n > TWO_ADICITY {
            return Err(too_long);
        }
        let k = 1usize.checked_shl(self.log2_k).ok_or(too_long)?;
        let n = 1usize.checked_shl(log_n).ok_or(too_long)?;
        Ok((k, n))
    }

    /// Checks the parameters of a masked commitment whose leaves merge its segments, the
    /// advice's, in this order: those of the code ([`PcsError::InverseRate`],
    /// [`PcsError::TooLong`]), then [`PcsError::NoMask`] if `P_A = 0` and
    /// [`PcsError::MaskTooLarge`] if `P_A > k`.
    pub(crate) fn shape(&self) -> Result<Shape, PcsError> {
        let (k, n) = self.code()?;
        if self.mask_coeffs == 0 {
            return Err(PcsError::NoMask);
        }
        if self.mask_coeffs > k {
            return Err(PcsError::MaskTooLarge {
                mask_coeffs: self.mask_coeffs,
                message_len: k,
            });
        }
        Ok(Shape {
            k,
            n,
            mask_coeffs: self.mask_coeffs,
            leaves: Leaves::Merged,
        })
    }

    /// Checks the parameters of an unmasked commitment of one chain per column, a weight
    /// matrix's in `tessera-protocol/1`, in this order: those of the code, then
    /// [`PcsError::UnexpectedMask`] unless `P = 0`.
    fn unmasked_shape(&self) -> Result<Shape, PcsError> {
        let (k, n) = self.code()?;
        if self.mask_coeffs != 0 {
            return Err(PcsError::UnexpectedMask {
                mask_coeffs: self.mask_coeffs,
            });
        }
        Ok(Shape {
            k,
            n,
            mask_coeffs: 0,
            leaves: Leaves::Chained,
        })
    }
}

/// Which rows of a matrix of `2^row_vars` rows are committed: the non-zero rows of each segment,
/// segments in layout order (RFC-0011 'Canonical advice layout').
///
/// The layout is public: the verifier derives it from public data, and the prover commits and
/// opens exactly the rows it lists. Rows increase strictly within and across segments, every
/// row is below `2^row_vars`, and a segment may be empty. Every row not listed is a zero row:
/// it is neither encoded, nor hashed, nor opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowLayout {
    row_vars: u32,
    segments: Vec<Vec<u64>>,
    rows: usize,
}

impl RowLayout {
    /// The layout of a matrix whose row index has `row_vars` bits and whose segments hold the
    /// non-zero rows `segments`, in layout order.
    ///
    /// # Errors
    ///
    /// In this order: [`PcsError::RowVars`] if `row_vars > 64`; [`PcsError::NoSegments`] if
    /// there is no segment; then, row by row in layout order, [`PcsError::RowOrder`] for a row
    /// that is not above the row before it, across segments too, and [`PcsError::RowRange`]
    /// for a row that is not below `2^row_vars`.
    pub fn new(row_vars: u32, segments: Vec<Vec<u64>>) -> Result<Self, PcsError> {
        if row_vars > u64::BITS {
            return Err(PcsError::RowVars { row_vars });
        }
        if segments.is_empty() {
            return Err(PcsError::NoSegments);
        }
        let mut previous: Option<u64> = None;
        for (segment, rows) in segments.iter().enumerate() {
            for (position, &row) in rows.iter().enumerate() {
                if previous.is_some_and(|before| row <= before) {
                    return Err(PcsError::RowOrder { segment, position });
                }
                // A shift by the full width overflows: every u64 is below 2^64.
                if row.checked_shr(row_vars).unwrap_or(0) != 0 {
                    return Err(PcsError::RowRange {
                        segment,
                        position,
                        row_vars,
                    });
                }
                previous = Some(row);
            }
        }
        let rows = segments.iter().map(Vec::len).sum();
        Ok(Self {
            row_vars,
            segments,
            rows,
        })
    }

    /// The bit count of a row index: the matrix has rows `0` to `2^row_vars − 1`.
    pub fn row_vars(&self) -> u32 {
        self.row_vars
    }

    /// The non-zero rows of each segment, in layout order.
    pub fn segments(&self) -> &[Vec<u64>] {
        &self.segments
    }

    /// `N`, the number of non-zero rows across all segments.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The non-zero rows in layout order: segment by segment, increasing.
    pub fn row_indices(&self) -> impl Iterator<Item = u64> + '_ {
        self.segments.iter().flatten().copied()
    }
}

/// The reference commitment to a resident matrix (RFC-0011 'Column digests and root'), with
/// everything its openings need: the non-zero rows, their masks and codewords, the leaves and
/// the Merkle tree.
///
/// Rows, masks and codewords are secret-class (05 §3): the `Debug` form shows the shape only,
/// there is no `Display` and no `Clone`, and all three are cleared when the commitment drops
/// (OBS-6).
pub struct ReferenceCommitment {
    params: CommitmentParams,
    shape: Shape,
    layout: RowLayout,
    /// The non-zero rows' values, `N × k`, in layout order.
    rows: Secret<Vec<Fp>>,
    /// Their mask coefficients, `N × P_A`, each row's `m_0` first.
    masks: SecretBuf<Fp>,
    /// Their codewords, `N × n`.
    codewords: SecretBuf<Fp>,
    /// `leaf_j` for `j < n`.
    leaves: Vec<Digest>,
    tree: MerkleTree,
}

/// Shows the shape: the rows, masks and codewords are secret.
impl fmt::Debug for ReferenceCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReferenceCommitment")
            .field("message_len", &self.shape.k)
            .field("codeword_len", &self.shape.n)
            .field("mask_coeffs", &self.shape.mask_coeffs)
            .field("row_block", &self.params.row_block)
            .field("row_vars", &self.layout.row_vars)
            .field("segments", &self.layout.segments.len())
            .field("rows", &self.layout.rows)
            .finish_non_exhaustive()
    }
}

impl ReferenceCommitment {
    /// Commits to the matrix whose non-zero rows, listed by `layout`, hold `rows` (`N` rows of
    /// `k` values, one after another, in layout order) and whose other rows are zero. Each row
    /// takes `P_A` mask coefficients from `mask`, row by row in layout order, `m_0` first, and
    /// each column's leaf merges its segments' chains (RFC-0011).
    ///
    /// The commitment takes the rows, secret-class values in their `Secret`, and keeps them for
    /// its openings (05 §3 OBS-1). It allocates `N·(n + P_A)` elements for the masks and
    /// codewords, cleared when it drops (OBS-6), and `3n` digests.
    ///
    /// # Errors
    ///
    /// In this order, before anything is drawn or computed: [`PcsError::InverseRate`] if
    /// `r = 0`, [`PcsError::TooLong`] unless `n = 2^r·k ≤ 2^32`, [`PcsError::NoMask`] if
    /// `P_A = 0`, [`PcsError::MaskTooLarge`] if `P_A > k`, and [`PcsError::MatrixLength`] unless
    /// `rows` holds `N·k` values. The layout was checked when it was built ([`RowLayout::new`]).
    /// On error nothing is drawn from `mask`, and `rows` is dropped.
    pub fn commit<M: MaskSource + ?Sized>(
        params: CommitmentParams,
        layout: RowLayout,
        rows: Secret<Vec<Fp>>,
        mask: &mut M,
    ) -> Result<Self, PcsError> {
        let shape = params.shape()?;
        Self::encode(params, shape, layout, rows, mask)
    }

    /// Commits to a weight matrix of `tessera-protocol/1` (RFC-0040 'Encoding, leaves and
    /// roots'): the matrix whose committed rows, listed by `layout`'s one segment, hold `rows`,
    /// and whose other rows are structurally zero. No row is masked, and each column is one chain
    /// over the committed rows, whose last digest is the column's leaf, `leaf_j = d_j[⌈m/B⌉]`:
    /// no `column-merge`. A matrix of no committed row has the leaves `d_j[0]`.
    ///
    /// It takes and keeps the rows as [`commit`](Self::commit) does, and allocates `N·n`
    /// elements for the codewords.
    ///
    /// # Errors
    ///
    /// In this order, before anything is computed: [`PcsError::InverseRate`] if `r = 0`,
    /// [`PcsError::TooLong`] unless `n = 2^r·k ≤ 2^32`, [`PcsError::UnexpectedMask`] unless
    /// `P = 0`, [`PcsError::Segments`] unless the layout has one segment, and
    /// [`PcsError::MatrixLength`] unless `rows` holds `N·k` values.
    pub fn commit_unmasked(
        params: CommitmentParams,
        layout: RowLayout,
        rows: Secret<Vec<Fp>>,
    ) -> Result<Self, PcsError> {
        let shape = params.unmasked_shape()?;
        if layout.segments.len() != 1 {
            return Err(PcsError::Segments {
                segments: layout.segments.len(),
            });
        }
        let mut none = || -> Fp { unreachable!("an unmasked encoding draws no mask coefficient") };
        Self::encode(params, shape, layout, rows, &mut none)
    }

    /// Step 1 of the commitment on checked parameters: the masks and the codewords, then steps
    /// 2 to 4.
    #[allow(clippy::disallowed_methods)]
    fn encode<M: MaskSource + ?Sized>(
        params: CommitmentParams,
        shape: Shape,
        layout: RowLayout,
        rows: Secret<Vec<Fp>>,
        mask: &mut M,
    ) -> Result<Self, PcsError> {
        let values = rows.expose_secret();
        // N·k with no product that could overflow.
        if !values.len().is_multiple_of(shape.k) || values.len() / shape.k != layout.rows {
            return Err(PcsError::MatrixLength {
                rows: layout.rows,
                row_len: shape.k,
                found: values.len(),
            });
        }
        // N·n = 2^r·N·k elements, a length no buffer has when it overflows.
        let symbols =
            values
                .len()
                .checked_mul(shape.n / shape.k)
                .ok_or(PcsError::CodewordLength {
                    expected: usize::MAX,
                    found: 0,
                })?;
        let encoder = RowEncoder::new(shape.k, params.log2_inv_rate, shape.mask_coeffs)?;
        // N·P ≤ N·k: a vector of N·k elements exists, so it does not overflow.
        let mut masks = SecretBuf::filled(Fp::ZERO, layout.rows * shape.mask_coeffs);
        for coefficient in masks.iter_mut() {
            *coefficient = mask.draw();
        }
        let mut codewords = SecretBuf::filled(Fp::ZERO, symbols);
        // The encoder draws P coefficients per row, row after row: the masks just drawn.
        let mut drawn = masks.iter().copied();
        let mut replay = || {
            drawn
                .next()
                .expect("the encoder draws P coefficients per row")
        };
        encoder.encode_rows(values, &mut replay, &mut codewords)?;
        Ok(Self::from_encoded(
            params, shape, layout, rows, masks, codewords,
        ))
    }

    /// Steps 2 to 4 of the commitment, from the codewords: the chains of each segment, the
    /// leaves and the tree. The shapes are the caller's to check: `codewords` holds `N` rows
    /// of `n` symbols, `rows` and `masks` `N` rows of `k` and `P`, and a commitment of chained
    /// leaves has one segment.
    pub(crate) fn from_encoded(
        params: CommitmentParams,
        shape: Shape,
        layout: RowLayout,
        rows: Secret<Vec<Fp>>,
        masks: SecretBuf<Fp>,
        codewords: SecretBuf<Fp>,
    ) -> Self {
        let n = shape.n;
        let columns = NonZeroUsize::new(n).expect("a codeword has n ≥ 2 symbols");
        // A step absorbs B rows of n symbols; a B too large for a usize covers any segment.
        let step = params.row_block.get().saturating_mul(n);
        let mut digests: Vec<Vec<Digest>> = Vec::with_capacity(layout.segments.len());
        let mut rest = &codewords[..];
        for segment in &layout.segments {
            let (here, after) = rest.split_at(segment.len() * n);
            let mut chains = ColumnChains::new(&params.commitment_id, columns, params.row_block);
            for block in here.chunks(step) {
                chains
                    .absorb_block(block)
                    .expect("whole rows, at most B of them, the short block last");
            }
            digests.push(chains.finish());
            rest = after;
        }
        let leaves: Vec<Digest> = match shape.leaves {
            Leaves::Chained => digests.pop().expect("a chained commitment has one segment"),
            Leaves::Merged => (0..n)
                .map(|j| {
                    let mut merge = Hasher::new(Domain::ColumnMerge);
                    for segment in &digests {
                        merge.update(&segment[j]);
                    }
                    merge.finalize()
                })
                .collect(),
        };
        let tree = MerkleTree::new(leaves.clone()).expect("n is a power of two");
        Self {
            params,
            shape,
            layout,
            rows,
            masks,
            codewords,
            leaves,
            tree,
        }
    }

    /// The commitment: the Merkle root over the `n` leaves.
    pub fn root(&self) -> Digest {
        self.tree.root()
    }

    /// The leaves, one per column, in column order: `leaf_j = H("column-merge", d_j^(1) ‖ … ‖
    /// d_j^(S))` for [`commit`](Self::commit), and the one chain's `d_j[⌈m/B⌉]` for
    /// [`commit_unmasked`](Self::commit_unmasked). Digests are public (05 §3).
    pub fn leaves(&self) -> &[Digest] {
        &self.leaves
    }

    /// The parameters committed with.
    pub fn params(&self) -> &CommitmentParams {
        &self.params
    }

    /// The layout of the non-zero rows.
    pub fn layout(&self) -> &RowLayout {
        &self.layout
    }

    /// `k`, the values in a row.
    pub fn message_len(&self) -> usize {
        self.shape.k
    }

    /// `n = 2^r·k`, the symbols in a codeword and the columns of the commitment.
    pub fn codeword_len(&self) -> usize {
        self.shape.n
    }

    /// The combined rows of RFC-0011 steps 2 to 5: one per group, in group order, then the
    /// proximity row. For group `g`, `u_g = Σ_row c_g(row)·A_row` over `Fp2` and
    /// `M_g = Σ_row c_g(row)·m_row`, with `c_g` from [`RowLayout::row_coefficients`] and the
    /// sums over the non-zero rows; the proximity row does the same with `ξ_row = eq(ξ*, row)`.
    /// Each `u` holds `k` values and each mask `P_A` coefficients.
    ///
    /// The claims' values never enter: the prover does not need them. The column points do
    /// not enter either, since `u_g` depends on the row coefficients only. They are checked so
    /// that an opening the verifier would refuse as malformed is refused here first. The cost
    /// is `(G + 1)·N·(k + P_A)` products of an `Fp2` coefficient by an `Fp` value, summed
    /// without reduction and reduced once per output, with no allocation in the loop.
    ///
    /// # Errors
    ///
    /// In this order, before anything is computed: [`PcsError::ColumnPoint`] for the first
    /// group, in group order, whose column point does not have `log2 k` coordinates, then the
    /// errors of [`RowLayout::row_coefficients`].
    #[allow(clippy::disallowed_methods)]
    pub fn open(
        &self,
        groups: &[ClaimGroup],
        xi_point: &[Fp2],
    ) -> Result<Vec<CombinedRow>, PcsError> {
        // log2_k ≤ 31.
        let expected = self.params.log2_k as usize;
        for (group, claims) in groups.iter().enumerate() {
            if claims.column_point.len() != expected {
                return Err(PcsError::ColumnPoint {
                    group,
                    expected,
                    found: claims.column_point.len(),
                });
            }
        }
        let coefficients = self.layout.row_coefficients(groups, xi_point)?;
        let u = combine(&coefficients, self.rows.expose_secret(), self.shape.k);
        let masks = combine(&coefficients, &self.masks, self.shape.mask_coeffs);
        Ok(u.into_iter()
            .zip(masks)
            .map(|(u, mask)| CombinedRow { u, mask })
            .collect())
    }

    /// The answer to the column queries of RFC-0011 step 7. `queries` holds the drawn column
    /// indices in draw order, possibly repeated and in any order. The answer holds, for each
    /// distinct index in increasing order, the column of every non-zero row's symbol at that
    /// index, in layout order, and the canonical multiproof of the distinct indices.
    ///
    /// # Errors
    ///
    /// [`PcsError::NoQueries`] if `queries` is empty, then [`PcsError::QueryRange`] at the
    /// first query, in draw order, that is `n` or more.
    pub fn answer(&self, queries: &[u64]) -> Result<ColumnAnswer, PcsError> {
        let n = self.shape.n;
        if queries.is_empty() {
            return Err(PcsError::NoQueries);
        }
        // n ≤ 2^32, so it fits a u64.
        if let Some(position) = queries.iter().position(|&query| query >= n as u64) {
            return Err(PcsError::QueryRange {
                position,
                query: queries[position],
                columns: n,
            });
        }
        let mut distinct = queries.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let columns = distinct
            .iter()
            .map(|&j| {
                // j < n, a usize: the cast is exact.
                let j = j as usize;
                self.codewords.iter().skip(j).step_by(n).copied().collect()
            })
            .collect();
        let proof = self
            .tree
            .open(&distinct)
            .expect("at least one index, each below n");
        Ok(ColumnAnswer { columns, proof })
    }
}

/// The answer to the column queries (RFC-0011 step 7): the columns of the distinct queried
/// indices, in increasing index order, and their canonical multiproof.
///
/// It does not carry the indices, which the verifier knows from its draw. Opened columns are
/// protocol-class (05 §3): the `Debug` form shows counts only.
#[derive(Clone, PartialEq, Eq)]
pub struct ColumnAnswer {
    /// One column per distinct queried index, in increasing index order. Column `j` holds
    /// symbol `j` of every non-zero row's codeword, segment by segment, in layout order.
    pub columns: Vec<Vec<Fp>>,
    /// The canonical multiproof of the distinct indices (RFC-0004 'Merkle trees').
    pub proof: MultiProof,
}

/// Shows the counts: opened columns are protocol-class.
impl fmt::Debug for ColumnAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnAnswer")
            .field("columns", &self.columns.len())
            .field("symbols", &self.columns.first().map_or(0, Vec::len))
            .field("siblings", &self.proof.siblings.len())
            .finish()
    }
}

#[cfg(test)]
mod proximity;
