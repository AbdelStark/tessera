// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The streaming advice commitment, the persistent weight commitment, openings and the hiding
//! budget (RFC-0011, RFC-0012).
//!
//! The encoding of RFC-0011 'Encoding' and a non-streaming reference of its commitment and
//! opening are here so far.
//!
//! # Encoding
//!
//! - [`NttPlan`] holds the twiddles of one transform size, from `2^0` to `2^32` points, and runs
//!   the NTT, the inverse NTT and their coset variants in place, in natural order on both sides:
//!   `ntt` maps the coefficients of `f` to `f(ω_n^0), …, f(ω_n^{n−1})` with
//!   `ω_n = 7^((P − 1)/n)`, and `coset_ntt` to `f(s·ω_n^j)`.
//! - [`RowEncoder`] encodes rows of `k` values into codewords of `n = 2k` symbols on the coset
//!   `7·L`, adding a vanishing mask of `P` coefficients drawn from a [`MaskSource`]: `encode_row`
//!   writes one codeword into a buffer the caller provides, `encode_rows` a block of them.
//!
//! ```
//! use tessera_field::{Goldilocks, PrimeField};
//! use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};
//!
//! let fp = Goldilocks::from_u64_reduced;
//! let row: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let encoder = RowEncoder::new(8, 2)?;
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
//! carries, and reduced once per output. The transcript, the canonical layout, streaming and the
//! blinding rows of RFC-0013 are not here: `γ`, `ξ*`, the queries and the layout are inputs.
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
//! # Secrets
//!
//! Rows, mask coefficients and codewords are secret-class, and combined rows and opened
//! columns protocol-class (05 §3). Nothing here logs. Errors carry lengths, positions and public
//! indices only ([`PcsError`]). The `Debug` forms of the plan, the encoder and the commitment
//! show their sizes, and those of combined rows and answers their lengths. Control flow depends
//! on lengths, the public layout and public indices, never on values (INV-ARCH-2). The commitment
//! takes its rows in their `Secret` and keeps its masks and codewords in `SecretBuf`s, so all
//! three are cleared when it drops (05 §3 OBS-1, OBS-6).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod commitment;
mod encoding;
mod error;
mod ntt;
mod opening;

pub use commitment::{ColumnAnswer, CommitmentParams, ReferenceCommitment, RowLayout};
pub use encoding::{COSET_SHIFT, MaskSource, RowEncoder};
pub use error::PcsError;
pub use ntt::NttPlan;
pub use opening::{ClaimGroup, CombinedRow, RowClaim};
