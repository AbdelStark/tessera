// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The sumcheck engine, GKR layer protocols and stage batching (RFC-0010, RFC-0013).
//!
//! This version holds the generic batched sumcheck prover of RFC-0010 'Batching within a stage'
//! and 'Sumcheck messages': [`prove_batch`] proves the sums of products of up to
//! [`MAX_DEGREE`] multilinear tables, over many instances of different sizes, in one round
//! sequence. On it, [`prove_forests`] proves the product-tree levels of the leaf stage, for
//! batches of forests of stacked trees (RFC-0010 'Tree levels'), and returns the leaf claims the
//! next stage checks. The relations of the stages, masking (RFC-0013) and the stage schedule
//! build on them.
//!
//! # Protocol
//!
//! - A table of `2^n` entries is the multilinear `T̃(x_1, …, x_n)` equal to `T[Σ_i x_i·2^{i−1}]`
//!   on `{0, 1}^n`: `x_1` is the least significant bit of the index, and round `j` binds `x_j`
//!   (RFC-0010 'Multilinear extensions').
//! - An [`Instance`] is a product of [`Table`]s of one size, at most the batch's degree `d` of
//!   them. Instance `i` (counted from 0 in the order given) on `n_i` variables enters with weight
//!   `γ^i`, `γ` squeezed under `<prefix>/gamma` before the first round.
//! - Smaller instances are front-padded: constant in the lowest `n_max − n_i` variables, which the
//!   first rounds bind, so that an instance's own variables take the last `n_i` challenges. The
//!   batched sum is `Σ_i γ^i·2^{n_max − n_i}·c_i`.
//! - Round `j` (from 0) absorbs `p_j(0), p_j(2), …, p_j(d)` as TCE `Vec<F>` under
//!   `<prefix>/round/{j}`, never `p_j(1)`, which the verifier derives from its claim, and squeezes
//!   `r_j` under `<prefix>/r/{j}`.
//! - The final evaluations, each instance's tables at its point in the order of its tables, are
//!   absorbed as one TCE `Vec<F>` under `<prefix>/evals`. A table marked [`Table::public`], such
//!   as `eq(r, ·)`, is a public factor: the verifier evaluates it, so it is left out.
//!
//! The [`Prefix`] is `stage/{k}` or `tree/{ℓ}` (RFC-0014 'Transcript schedule'). The result,
//! [`ProvenBatch`], holds the [`SumcheckProof`] (TCE-encodable, as RFC-0010 defines it), `γ`,
//! the point, and each instance's point and evaluations, from which the caller forms its next
//! claims.
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_sumcheck::{Instance, Prefix, Table, prove_batch};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! let a: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let b: Vec<Goldilocks> = (11..=18).map(fp).collect();
//! let c: Vec<Goldilocks> = (1..=4).map(fp).collect();
//! // Σ_x a(x)·b(x) on 3 variables, and Σ_y c(y) on 2, front-padded by one round.
//! let instances = [
//!     Instance::new(vec![Table::base(&a), Table::base(&b)]),
//!     Instance::new(vec![Table::base(&c)]),
//! ];
//! let mut transcript = Transcript::new(&context);
//! let proven = prove_batch::<GoldilocksExt2, _>(&mut transcript, Prefix::Stage(0), 2, &instances)?;
//! assert_eq!(proven.proof().rounds.len(), 3);
//! assert!(proven.proof().rounds.iter().all(|round| round.len() == 2));
//! assert_eq!(proven.instance_point(1), Some(&proven.point()[1..]));
//! assert_eq!(proven.proof().final_evals.len(), 3);
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! # Product forests
//!
//! A [`Forest`] stacks `2^s` product trees of height `h` into one table of `2^{s+h}` leaves over
//! the prime subfield, entry `y·2^h + x` being leaf `x` of tree `y`. [`prove_forests`] proves a
//! batch of them from the roots to the leaves, one batched sumcheck of degree 3 per level under
//! `tree/{ℓ}`. Forests join where their heights match; a level where forests of more than one
//! tree join first squeezes their root point under `tree/{ℓ}/root/{c}`. The result,
//! [`ProvenForests`], holds each level's challenges and [`SumcheckProof`], and each forest's leaf
//! claim, a [`LevelClaim`] on its stacked leaves.
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_sumcheck::{Forest, prove_forests};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! // Four rows of four keys, each with a zero key: 2^2 trees of height 2.
//! let rows: Vec<Goldilocks> = [3, 0, 5, 1, 0, 2, 2, 7, 4, 4, 0, 9, 1, 1, 1, 0]
//!     .into_iter()
//!     .map(fp)
//!     .collect();
//! // One tree of 2^3 leaves, a level taller: it joins at level 0, the rows at level 1.
//! let tree: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let forests = [Forest::new(&rows, 2), Forest::<GoldilocksExt2>::new(&tree, 3)];
//! let mut transcript = Transcript::new(&context);
//! let proven = prove_forests(&mut transcript, &forests)?;
//! let levels = proven.levels();
//! assert_eq!(levels.len(), 3);
//! // The single tree joins at level 0 under no root challenge; the rows join at level 1 under two.
//! assert_eq!(levels[0].root_challenges().len(), 0);
//! assert_eq!(levels[1].root_challenges().len(), 2);
//! assert_eq!(levels[1].forests(), [0, 1]);
//! // A leaf claim per forest, on its stacked leaves: 2 + 2 and 0 + 3 variables.
//! let points: Vec<usize> = proven.leaf_claims().iter().map(|c| c.point.len()).collect();
//! assert_eq!(points, [4, 3]);
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! # Algorithm
//!
//! - **Round values.** On a pair `(a, b) = (T[2t], T[2t + 1])` a table is `a + z·Δ` with
//!   `Δ = b − a`, so its values at the nodes `0, 2, 3, …, d` come by adding `Δ`, with no
//!   multiplication, and a product of `k` tables costs `k − 1` multiplications per node. `p(1)`
//!   is never computed.
//! - **Prime-subfield first round.** An instance's first own round reads its tables unbound,
//!   whatever round that is, since padding rounds do not touch them. Tables over the prime
//!   subfield multiply there; when all of an instance's tables are, the round is summed there too
//!   and lifted once.
//! - **Padding.** A padding round contributes `2^{n_max − n_i − j − 1}·S_i` at every node.
//!   `S_i`, the instance's sum over its own hypercube, comes from its tables in one pass, which
//!   also yields its first own round at every node; the claims are never read. Padded tables are
//!   never materialized.
//! - **Binding.** An instance's first own round binds each borrowed table into a table over `F`
//!   of half its length, reserved once; later rounds bind in place. A round allocates its message
//!   and nothing else.
//! - **Loops.** A single table needs two sums, of its even and odd entries, for every node. Two
//!   to four tables over the prime subfield, or two or three over `F` at up to three nodes, are
//!   zipped pair by pair with their products in registers; other products are formed a block of
//!   pairs at a time in rows that stay in the first-level cache. Every loop walks its slices with
//!   `chunks_exact` and `zip`, without bounds checks, and sums over a range of pairs that a
//!   parallel backend can split (RFC-0018).
//!
//! Control flow depends on shapes and on which tables are over the prime subfield, never on a
//! value (INV-ARCH-2). Table entries are secret-class values and the messages protocol-class
//! (05 §3): no debug form or error of this crate shows either, only shapes.
//!
//! The code is generic over [`tessera_poly::PolyField`], so it runs over Goldilocks and its
//! extension, and over `F65537` and its extension for the soundness-calibration tests (07 §4),
//! with any transcript that implements [`SumcheckTranscript`].
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

mod error;
mod forest;
mod kernel;
mod proof;
mod prover;
mod transcript;

pub use error::SumcheckError;
pub use forest::{Forest, LevelClaim, ProvenForests, ProvenLevel, prove_forests};
pub use proof::SumcheckProof;
pub use prover::{Instance, Prefix, ProvenBatch, Table, prove_batch};
pub use tessera_poly::MAX_DEGREE;
pub use transcript::SumcheckTranscript;
