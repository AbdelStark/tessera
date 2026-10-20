// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The streaming advice commitment, the persistent weight commitment, openings and the hiding
//! budget (RFC-0011, RFC-0012).
//!
//! The encoding of RFC-0011 'Encoding', canonical advice layouts, streaming commitment and
//! regenerated openings are here, with a resident reference, and the weight commitment as `tessera-protocol/1` commits it
//! (RFC-0040): its layout, its roots, `R_W` and its two artifacts, and opens it by the folding
//! pass (RFC-0012, RFC-0049).
//!
//! # Encoding
//!
//! - [`NttPlan`] holds the twiddles of one transform size, from `2^0` to `2^32` points, and runs
//!   the NTT, the inverse NTT and their coset variants in place, in natural order on both sides:
//!   `ntt` maps the coefficients of `f` to `f(ω_n^0), …, f(ω_n^{n−1})` with
//!   `ω_n = 7^((P − 1)/n)`, and `coset_ntt` to `f(s·ω_n^j)`.
//! - [`RowEncoder`] encodes rows of `k` values into codewords of `n = 2^r·k` symbols on the coset
//!   `7·L`, rate 1/2 for the advice and 1/4 for the weights, adding a vanishing mask of `P`
//!   coefficients drawn from a [`MaskSource`]: `encode_row` writes one codeword into a buffer the
//!   caller provides, `encode_rows` a block of them.
//!
//! ```
//! use tessera_field::{Goldilocks, PrimeField};
//! use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};
//!
//! let fp = Goldilocks::from_u64_reduced;
//! let row: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let encoder = RowEncoder::new(8, 1, 2)?;
//! let mut codeword = vec![Goldilocks::default(); encoder.codeword_len()];
//! let mut masks = [fp(11), fp(12)].into_iter();
//! let mut source = || masks.next().unwrap_or_default();
//! encoder.encode_row(&row, &mut source, &mut codeword)?;
//!
//! // The coefficients of g = f + (X^8 − 1)·m: the top half holds m.
//! let mut g = codeword.clone();
//! NttPlan::new(16)?.coset_intt(&mut g, COSET_SHIFT)?;
//! assert_eq!(&g[8..], &[fp(11), fp(12), fp(0), fp(0), fp(0), fp(0), fp(0), fp(0)]);
//! // On H the mask vanishes: folding g modulo X^8 − 1 gives f, whose values are the row.
//! let mut f: Vec<Goldilocks> = (0..8).map(|i| g[i] + g[i + 8]).collect();
//! NttPlan::new(8)?.ntt(&mut f)?;
//! assert_eq!(f, row);
//! # Ok::<(), tessera_pcs::PcsError>(())
//! ```
//!
//! # The reference commitment
//!
//! [`ReferenceCommitment`] commits to a matrix held in memory, for tests and measurements.
//! Its rows have `k = 2^log2_k` values, and its [`RowLayout`] lists the non-zero rows, segment
//! by segment, in increasing order. Rows not listed are zero, and are never encoded, hashed or
//! opened. Its parameters ([`CommitmentParams`]) are `k`, `P_A`, the row block `B` and the
//! `commitment_id`.
//!
//! - [`ReferenceCommitment::commit`] draws each non-zero row's `P_A` mask coefficients, row by
//!   row in layout order, and encodes the row. It chains each segment's columns, `B` rows per
//!   step (RFC-0004 'Column digests'), and merges each column's segment digests into
//!   `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))`. The Merkle root over the `n` leaves
//!   is the commitment (RFC-0011 'Column digests and root').
//! - [`ReferenceCommitment::open`] takes the claims grouped by column sub-point
//!   ([`ClaimGroup`], each claim a [`RowClaim`]: its row point and batching challenge `γ`) and
//!   the proximity point `ξ*`. It returns the combined rows of RFC-0011 steps 2 to 5, the
//!   groups' in group order and then the proximity row: `u = Σ_row c(row)·A_row` and
//!   `M = Σ_row c(row)·m_row` over the non-zero rows ([`CombinedRow`]). The coefficients are
//!   `c_g(row) = Σ_{claims of g} γ·eq(row_point, row)` for a group and `ξ_row = eq(ξ*, row)`
//!   for the proximity row ([`RowLayout::row_coefficients`]). The prover never needs the
//!   claims' values.
//! - [`ReferenceCommitment::answer`] takes the drawn column queries in draw order, possibly
//!   repeated. It returns the columns of their sorted, distinct set, in increasing order, each
//!   with every non-zero row's symbol in layout order, and the canonical multiproof of that
//!   set (RFC-0011 step 7, [`ColumnAnswer`]).
//!
//! The opening pass is `(G + 1)·N·(k + P_A)` products of an `Fp2` coefficient by an `Fp` value,
//! for `G` groups and `N` non-zero rows. They are summed unreduced, in 128 bits and a count of
//! carries, and reduced once per output. These reference opening calls take `γ`, `ξ*`, queries
//! and a row layout from their caller; they neither drive the transcript nor add RFC-0013's
//! blinding rows.
//!
//! ```
//! use std::num::NonZeroUsize;
//!
//! use tessera_codec::Secret;
//! use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
//! use tessera_pcs::{ClaimGroup, CommitmentParams, ReferenceCommitment, RowClaim, RowLayout};
//!
//! let fp = Goldilocks::from_u64_reduced;
//! let fp2 = |c0, c1| GoldilocksExt2::new(fp(c0), fp(c1));
//! // Rows of k = 4 values with P_A = 2 mask coefficients each; a column chain absorbs B = 32
//! // rows per step.
//! let params = CommitmentParams {
//!     log2_k: 2,
//!     log2_inv_rate: 1,
//!     mask_coeffs: 2,
//!     row_block: NonZeroUsize::new(32).expect("non-zero"),
//!     commitment_id: [7; 32],
//! };
//! // Of the rows 0 to 3, rows 0 and 1 form the first segment and row 3 the second; row 2 is
//! // zero.
//! let layout = RowLayout::new(2, vec![vec![0, 1], vec![3]])?;
//! // The rows are secret-class: the commitment takes them in their `Secret`.
//! let rows = Secret::new((1..=12).map(fp).collect::<Vec<Goldilocks>>());
//! let mut seed = 0;
//! let mut masks = || {
//!     seed += 1;
//!     fp(seed * 0x9E37_79B9)
//! };
//! let commitment = ReferenceCommitment::commit(params, layout, rows, &mut masks)?;
//!
//! // One group of one claim. The proximity point (1, 0) selects row 1, so the proximity row
//! // is row 1 itself.
//! let claim = RowClaim { row_point: vec![fp2(2, 0), fp2(1, 1)], gamma: fp2(4, 4) };
//! let groups = [ClaimGroup { column_point: vec![fp2(3, 1), fp2(5, 9)], claims: vec![claim] }];
//! let combined = commitment.open(&groups, &[fp2(1, 0), fp2(0, 0)])?;
//! assert_eq!(combined.len(), 2);
//! let row_1: Vec<GoldilocksExt2> = (5..=8).map(|x| fp2(x, 0)).collect();
//! assert_eq!(combined[1].u, row_1);
//! assert_eq!(combined[1].mask, [fp2(3 * 0x9E37_79B9, 0), fp2(4 * 0x9E37_79B9, 0)]);
//!
//! // The verifier draws columns 5, 2 and 5: the answer holds columns 2 and 5, of 3 symbols.
//! let answer = commitment.answer(&[5, 2, 5])?;
//! assert_eq!(answer.columns.len(), 2);
//! assert!(answer.columns.iter().all(|column| column.len() == 3));
//! # Ok::<(), tessera_pcs::PcsError>(())
//! ```
//!
//! # The weight commitment
//!
//! RFC-0040 commits the weights of `D` in ten matrices ([`WeightMatrix`]), on a layout that
//! follows from public data alone.
//!
//! - [`WeightLayout`] takes the parameter set's [`WeightLayoutParams`] and the shapes of `D`
//!   ([`WeightShapes`]): each weight type's `out × in`, the layers, and which weights have a
//!   `ChannelParams`. It gives each matrix's message width `k_T`, row index and committed rows
//!   ([`MatrixLayout`]), and writes the committed rows from `D`'s weights ([`ModelWeights`]).
//! - [`WeightCommitment::commit`] commits each matrix with
//!   [`ReferenceCommitment::commit_unmasked`]: rate 1/4, no mask, one chain per column from the
//!   matrix's `commitment_id` ([`CommitmentDescriptor`], RFC-0031), the Merkle root over its
//!   columns. [`WeightCommitmentPublic`] holds the layout, the ten roots and the epoch, and
//!   `R_W` is the hash of its encoding.
//! - [`WeightCommitmentState`] adds the masks' seed, drawn from the operating system's CSPRNG
//!   into a private `Secret` that only the state's own file receives, and each matrix's budget
//!   ([`TypeBudget`]). Each artifact is written by atomic rename, the state with mode 0600.
//!
//! `tessera-ir` gives the shapes from the architecture and lends the model's weights
//! (`IntegerModel::weight_shapes`, `IntegerModel::model_weights`).
//!
//! # The weight opening
//!
//! RFC-0012 opens a matrix by the folding pass, as RFC-0049 completes it for
//! `tessera-protocol/1`: one claim per matrix, no mask, no blinding row.
//!
//! - [`WeightFold::new`] is the one read of a matrix's committed rows a proof makes. It takes the
//!   claim's coordinates known when the pass starts, `r_o` for a weight type, and `ξ*_T`,
//!   squeezed right after the seed, and reads each row once through a [`MatrixRows`]: from `D`'s
//!   weights, [`WeightLayout::rows`] ([`WeightRows`]). It keeps the per-layer folds `F_ℓ` and the
//!   proximity row `u_ξ,T = Σ_j eq(ξ*_T, j)·W_T[row_j]`, over the committed rows by position.
//! - [`WeightFold::sumcheck_input`] is the matmul sumcheck's `W̃(ℓ, r_o, i)`, and
//!   [`WeightFold::combined_rows`] the evaluation row `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ` and the
//!   proximity row ([`WeightCombinedRows`]), from the folds alone (INV-WC-4).
//! - The column answers are the matrix's commitment's ([`ReferenceCommitment::answer`]), from
//!   the codewords it holds.
//!
//! ```
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_pcs::{
//!     ModelWeights, WeightFold, WeightLayout, WeightLayoutParams, WeightMatrix, WeightShape,
//!     WeightShapes, WeightTensorRef,
//! };
//!
//! let fp2 = |x: u64| GoldilocksExt2::from(Goldilocks::from_u64_reduced(x));
//! // One layer of 2 × 2 weights at k_W = 2: each row holds one output's two inputs.
//! let shapes = WeightShapes {
//!     layers: 1,
//!     shapes: [WeightShape { out: 2, inputs: 2 }; 9],
//!     channels: Vec::new(),
//! };
//! let params = WeightLayoutParams { log2_k: 1, log2_inv_rate: 2, mask_coeffs: 64, blinding_rows: 32 };
//! let layout = WeightLayout::new(params, shapes)?;
//! let w: [i8; 4] = [1, 2, 3, 4];
//! let weights = ModelWeights {
//!     tensors: WeightMatrix::WEIGHTS
//!         .iter()
//!         .map(|&matrix| WeightTensorRef { matrix, layer: 0, data: &w })
//!         .collect(),
//!     channels: Vec::new(),
//! };
//! // Q's rows are W[0] = (1, 2) and W[1] = (3, 4). The fold point r_o = (2), and ξ* = (3).
//! let mut rows = layout.rows(WeightMatrix::Q, &weights)?;
//! let fold = WeightFold::new(&layout, WeightMatrix::Q, &mut rows, &[fp2(2)], &[fp2(3)])?;
//! // W̃(r_o, i) = (1 − 2)·W[0][i] + 2·W[1][i].
//! assert_eq!(&fold.sumcheck_input()[..], &[fp2(5), fp2(6)]);
//! // One layer, no layer point: u_T = θ·F_0 at θ = 10, and u_ξ = (1 − 3)·W[0] + 3·W[1].
//! let combined = fold.combined_rows(&[], fp2(10))?;
//! assert_eq!(combined.evaluation.u, [fp2(50), fp2(60)]);
//! assert_eq!(combined.proximity.u, [fp2(7), fp2(8)]);
//! # Ok::<(), tessera_pcs::PcsError>(())
//! ```
//!
//! # The canonical advice layout
//!
//! [`AdviceLayout::new`] places every committed tensor of a proof unit in the advice matrix
//! (RFC-0011 'Canonical advice layout', RFC-0034, RFC-0041, RFC-0046, RFC-0047), from public data
//! alone (INV-PCS-1): the claim list's [`LayoutParams`], the request's length
//! `n = prompt_len + output_len`, and its tiles in production order, each with the shapes of the
//! values it commits ([`TileShape`]). The crate precedes the compiler and the gadgets (01 §3), so
//! the tiles come as plain data.
//!
//! - **Order.** A dense tensor's variables take RFC-0047's layout order: its padded axes, those
//!   whose length is not a power of two, in axis order, then its other axes in axis order
//!   ([`layout_order`]). [`DenseShape::new`] derives it, and a dense shape has no other. An
//!   attention tile's exponentials take RFC-0034's `(i_hi, h, q, i_lo)` ([`TensorShape::Scores`],
//!   [`Attention`]).
//! - **Allocation.** Each tensor of `n_X` variables takes the next offset aligned to
//!   `2^{max(n_X, log2 k)}` ([`Placement`]), within at most `2^32` rows (RFC-0047 'The maximum
//!   layout').
//! - **The map.** `Ã` has `ν = log2 k + ⌈log2 R⌉` variables, `R` the rows allocated (RFC-0047). A
//!   claim `X̃(r)` is `Ã(r' ‖ base_X)`, `r'` the claim's coordinates in the layout order
//!   ([`Placement::point`]); [`Placement::locate`] gives an element's row and column.
//! - **Zero rows.** Rows of padding, alignment gaps and a score tile's masked blocks hold nothing
//!   and are skipped ([`AdviceLayout::is_zero_row`]). [`AdviceLayout::rows`] counts the others:
//!   the advice descriptor's `rows` (RFC-0031).
//! - **Segments.** [`segment_layers`] gives each segment its layers, and [`Segment`] its rows,
//!   from its first tile's first row.
//!
//! [`LayoutParams::check`] refuses, as `Layout` (RFC-0007 step 6, RFC-0039), a claim list from
//! which no layout derives, such as one whose rows cannot hold one key against a tile's queries
//! and heads: `g·min(Q, N_max) > k`. [`AdviceLayout::new`] refuses a layout of more than `2^32`
//! rows as `Layout` too.
//!
//! ```
//! use tessera_field::{Field, Goldilocks};
//! use tessera_pcs::{AdviceLayout, Axis, DenseShape, LayoutParams, Padding, TensorShape, TileShape};
//!
//! // RFC-0047's order: a padded axis, here of 3 positions, is the most significant.
//! assert_eq!(DenseShape::new(vec![Axis::new(8), Axis::new(3)], Padding::Zero).order(), [1, 0]);
//!
//! // Rows of k = 16 values; one layer of query heads in pairs, blocks of 4 queries.
//! let params = LayoutParams {
//!     log2_k: 4,
//!     segments: 1,
//!     layers: 1,
//!     group: 2,
//!     query_block: 4,
//!     max_context: 16,
//! };
//! // Ten positions pad to N = 16: four blocks, and score rows of 2^w = 2 keys.
//! let attention = params.attention(10)?;
//! assert_eq!((attention.queries, attention.blocks, attention.key_bits), (4, 4, 1));
//! let tokens = DenseShape::new(vec![Axis::new(10)], Padding::Zero);
//! let mut tiles = vec![TileShape { layer: None, values: vec![TensorShape::Dense(tokens)] }];
//! for block in 0..attention.blocks {
//!     // The block's exponentials, then its reciprocals.
//!     let reciprocals = attention.row_shape(block, Padding::Zero);
//!     tiles.push(TileShape {
//!         layer: Some(0),
//!         values: vec![TensorShape::Scores { block }, TensorShape::Dense(reciprocals)],
//!     });
//! }
//! let layout = AdviceLayout::new(params, 10, &tiles)?;
//!
//! // The tokens fill row 0. Block 0's exponentials take rows 8 to 15, aligned: their keys 0 to
//! // 15, two per row, against queries 0 to 3. Past key 3 every position is masked, so rows 10 to
//! // 15 are zero rows.
//! let scores = layout.tensor(1, 0).expect("block 0's exponentials");
//! assert_eq!((scores.first_row(), scores.row_count()), (8, 8));
//! assert_eq!(scores.nonzero().len(), 1);
//! assert_eq!(scores.nonzero()[0], 8..10);
//! assert!(layout.is_zero_row(10));
//! // The element (h, q, i) = (1, 3, 2) lies in row 8 + i_hi = 9, column i_lo + 2·q + 8·h = 14.
//! assert_eq!(scores.locate(&[1, 3, 2])?, (9, 14));
//! // A claim on the tile's n_X = 7 variables is a claim on Ã's ν = 4 + 7.
//! let point = scores.point(&[Goldilocks::ONE; 7])?;
//! assert_eq!((point.len(), layout.variables()), (11, 11));
//! // Block 3 holds queries 12 to 15, all padding: nothing of it is committed. Of 65 rows, 15
//! // are.
//! assert!(layout.tensor(4, 0).expect("block 3's exponentials").nonzero().is_empty());
//! assert_eq!((layout.allocated_rows(), layout.rows()), (65, 15));
//! # Ok::<(), tessera_pcs::PcsError>(())
//! ```
//!
//! # Streaming advice commitment
//!
//! [`StreamingCommitter`] borrows a canonical [`AdviceLayout`], a shared [`RowEncoder`] and the
//! caller's per-proof [`MaskSource`]. Push each segment's committed rows in layout order, then
//! [`finish`](StreamingCommitter::finish) it into [`SegmentDigests`]. [`merge_segments`] checks
//! the complete ordered set and returns the [`AdviceCommitment`] that the protocol re-exports
//! in `Round1`; its fields and TCE bytes are unchanged.
//!
//! The stream owns one initialized secret codeword block and its column chain state, with no
//! allocation per row or block. It clears every flushed block and its allocation on abort.
//! The merge uses a public stack frontier rather than retaining a tree. Encoder tables, layout,
//! input row and PRG state remain borrowed baseline state. The per-proof seed lifecycle belongs
//! to the driver of RFC-0018, which supplies the mask stream.
//!
//! # Regenerated advice openings
//!
//! [`AdviceOpening::new`] builds public opening metadata from the complete ordered P1
//! [`SegmentDigests`], retaining a Merkle tree rather than the witness or its codewords. It samples
//! one column uniformly with the OS random source. The root-only [`merge_segments`] still uses
//! no heap. [`AdviceOpening::prepare`] batches [`AdviceClaim`]s: every [`AdvicePoint`] piece of
//! one original combination takes the same gamma power, starting at gamma^1 per group.
//!
//! [`OpeningGroups::begin`] starts P4. Push every retained row, in segment order, replaying P1's
//! per-proof mask stream. P4 combines blocks into clearing accumulators, then verifies the
//! sampled column's leaf and every completed combination. [`OpeningPass::finish`] releases a
//! [`CheckedOpening`] only on success; faults yield [`AdviceOpeningError::SelfCheck`], whose
//! code is `ERR-INTERNAL`, before any combined-row message exists. The self-check samples one
//! column; the independent verifier still checks the full proof.
//!
//! Absorb [`CheckedOpening::combined`] before drawing column queries. Then
//! [`CheckedOpening::answer`] starts P5 with a fresh replay of the same masks. Push each retained
//! row again; [`ColumnPass::finish`] returns the existing [`ColumnAnswer`] format, each distinct
//! queried column once in increasing order and its canonical multiproof. No codeword matrix is
//! retained. The driver owns witness regeneration, seed restart policy, resource admission,
//! transcript timing and the one opening/answer per proof; protocol/2 blinding is separate.
//! The [allocation report](https://github.com/AbdelStark/tessera/blob/main/docs/perf/regenerated-advice-openings.md)
//! accounts for the public tree, block scratch, combined rows and output copies.
//!
//! # Secrets
//!
//! Rows, mask coefficients and codewords are secret-class, and combined rows and opened
//! columns protocol-class (05 §3). Nothing here logs. Errors carry lengths, positions and public
//! indices only ([`PcsError`]). The `Debug` forms of the plan, the encoder and the commitment
//! show their sizes, and those of combined rows and answers their lengths. Control flow depends
//! on lengths, the public layout and public indices, never on values (INV-ARCH-2). The reference
//! commitment owns its rows in `Secret` and its masks and codewords in `SecretBuf`s, which clear
//! on drop. The streaming path borrows rows and retains only its active secret block until it
//! is hashed and cleared (05 §3 OBS-1, OBS-6).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible, and gives other crates'
//!   tests [`WeightCommitmentState`]'s `with_seed` and `sealed`, which this crate's own test build
//!   has without it. Tests only: a release build with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod commitment;
mod descriptor;
mod encoding;
mod error;
mod layout;
mod ntt;
mod opening;
mod streaming;
mod weights;

pub use commitment::{ColumnAnswer, CommitmentParams, ReferenceCommitment, RowLayout};
pub use descriptor::{AdviceLayoutParams, CommitmentDescriptor, WeightLayoutParams, WeightMatrix};
pub use encoding::{COSET_SHIFT, MaskSource, RowEncoder};
pub use error::PcsError;
pub use layout::{
    AdviceLayout, AdviceLayoutCounter, Attention, Axis, DenseShape, LayoutParams, Padding,
    Placement, Segment, TensorShape, TilePlacement, TileShape, layout_order, segment_layers,
};
pub use ntt::NttPlan;
pub use opening::{ClaimGroup, CombinedRow, RowClaim};
pub use streaming::opening::{
    AdviceClaim, AdviceOpening, AdviceOpeningError, AdvicePoint, CheckedOpening, ColumnPass,
    OpeningGroups, OpeningPass,
};
pub use streaming::{AdviceCommitment, SegmentDigests, StreamingCommitter, merge_segments};
pub use weights::{
    ChannelParamsRef, ChannelValues, MatrixKind, MatrixLayout, MatrixRows, ModelWeights,
    OpenedColumns, TypeBudget, WeightCombinedRows, WeightCommitment, WeightCommitmentPublic,
    WeightCommitmentState, WeightFold, WeightLayout, WeightRows, WeightShape, WeightShapes,
    WeightTensorRef,
};
