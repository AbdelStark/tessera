// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The sumcheck engine, GKR layer protocols and stage batching (RFC-0010, RFC-0013).
//!
//! This version holds the generic batched sumcheck prover of RFC-0010 'Batching within a stage'
//! and 'Sumcheck messages': [`prove_batch`] proves the sums of products of up to
//! [`MAX_DEGREE`] multilinear tables, and sums of weighted products over one shared table list,
//! over many instances of different sizes, in one round sequence. A batch's degree is at most its
//! [`Protocol`]'s bound at its [`Prefix`]: 10 at a reduction stage of `tessera-protocol/2`, 7
//! everywhere else ([`Protocol::max_degree`], RFC-0013 'Masked extensions' 'Degree'). On it,
//! [`prove_trees`] proves the levels of the leaf stage, where product forests of stacked trees and LogUp fraction trees
//! run together (RFC-0010 'Tree levels', RFC-0008 'Fraction tree'), and returns the leaf claims
//! later checks owe; [`prove_forests`] is its batch of forests alone. [`MatMulWeight`] and
//! [`MatMulAct`] build the instances of the matmul relations of RFC-0010 'Relations', stacked
//! over layers or heads, and map their sumchecks' results to claims on their operands;
//! [`Elementwise`] does the same for `Mul`, `SumSquares` and `ZeroCheck`, [`Rotate`] for RoPE,
//! [`Recomposition`] for a `BitDecompose`'s recomposition zero-check (RFC-0072), [`Selection`] for
//! a `Slice`'s or a `GatherPublic`'s rows against a public kernel (RFC-0072 'The selection
//! relation'), and [`Linear`] propagates claims through linear relations without a sumcheck. Masking (RFC-0013)
//! and the stage schedule build on them.
//!
//! # Protocol
//!
//! - A table of `2^n` entries is the multilinear `T̃(x_1, …, x_n)` equal to `T[Σ_i x_i·2^{i−1}]`
//!   on `{0, 1}^n`: `x_1` is the least significant bit of the index, and round `j` binds `x_j`
//!   (RFC-0010 'Multilinear extensions').
//! - An [`Instance`] is a product of [`Table`]s of one size, at most the batch's degree `d` of
//!   them, or a sum of weighted products over one list of tables, each [`Term`] a weight times
//!   at most `d` of them ([`Instance::sum_of_products`]). Instance `i` (counted from 0 in the
//!   order given) on `n_i` variables enters with weight `γ^i`, `γ` squeezed under
//!   `<prefix>/gamma` before the first round.
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
//! use tessera_sumcheck::{Instance, Prefix, Protocol, Table, prove_batch};
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
//! let proven = prove_batch::<GoldilocksExt2, _>(
//!     &mut transcript,
//!     Protocol::One,
//!     Prefix::Stage(0),
//!     2,
//!     &instances,
//! )?;
//! assert_eq!(proven.proof().rounds.len(), 3);
//! assert!(proven.proof().rounds.iter().all(|round| round.len() == 2));
//! assert_eq!(proven.instance_point(1), Some(&proven.point()[1..]));
//! assert_eq!(proven.proof().final_evals.len(), 3);
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! # The merge
//!
//! A leaf stage proven in token groups ends with each merged polynomial's claims at the points of
//! the groups that read it (RFC-0054 'The merge'). [`Merge`] holds those claims, with each
//! group's `γ_h` and `β`; [`prove_merge`] reduces them to one claim per polynomial at one point,
//! a degree-2 sumcheck under `merge/…` whose weights `W_i` the verifier computes itself.
//! [`prove_merge_two_pass`] sends the same messages from the tables each group's pass keeps
//! ([`MergeTables`]) and one regeneration of every polynomial.
//!
//! # Product forests
//!
//! A [`Forest`] stacks `2^s` product trees of height `h` into one table of `2^{s+h}` leaves over
//! the prime subfield, entry `y·2^h + x` being leaf `x` of tree `y`. [`prove_forests`] proves a
//! batch of them from the roots to the leaves, one batched sumcheck of degree 3 per level under
//! `tree/{ℓ}`. Forests join where their heights match; a level where forests of more than one
//! tree join first squeezes their root point under `tree/{ℓ}/root/{c}`. The result,
//! [`ProvenForests`], holds each level's challenges and [`SumcheckProof`], and each forest's leaf
//! claim, a [`Claim`] on its stacked leaves.
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
//! # Fraction trees
//!
//! A [`FractionTree`] holds a LogUp tree's levels, built bottom-up from its `2^h` leaves
//! `(n_x, d_x)` with `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)`, its leaf numerators 1 or over the
//! prime subfield and every other node over `F`. [`prove_trees`] takes a schedule of [`Tree`]s,
//! forests and fraction trees in any order: each joins where its height matches, a forest of
//! several trees under root challenges, a fraction tree on its root; `λ` follows `γ` at a level
//! where a fraction tree is present, and a fraction tree's level is one sum-of-products instance,
//! `eq·(N_0·D_1 + N_1·D_0 + λ·D_0·D_1)` over the shared tables `[eq, N_0, N_1, D_0, D_1]`, whose
//! evaluations `N_0, N_1, D_0, D_1` are the level's message. [`ProvenTrees`] holds each level and
//! each tree's leaf claims, a [`LevelClaim`].
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_sumcheck::{Forest, FractionTree, LeafNumerators, LevelClaim, Tree, prove_trees};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! // Leaves (1, x + 9) for x < 8 and (2, x + 9) for x < 2, as a lookup tree and a table tree.
//! let d = |x: usize| GoldilocksExt2::new(fp(x as u64 + 9), fp(1));
//! let lookups = FractionTree::new(3, LeafNumerators::Unit, d)?;
//! let numerators = [fp(2), fp(2)];
//! let rows = FractionTree::new(1, LeafNumerators::Base(&numerators), d)?;
//! let leaves: Vec<Goldilocks> = (1..=4).map(fp).collect();
//! let trees = vec![
//!     Tree::Fraction(lookups),
//!     Tree::Forest(Forest::new(&leaves, 2)),
//!     Tree::Fraction(rows),
//! ];
//! let proven = prove_trees(&mut Transcript::new(&context), trees)?;
//! // Heights 3, 2 and 1 join at levels 0, 1 and 2; every level has a fraction tree, so a λ.
//! let present: Vec<&[usize]> = proven.levels().iter().map(|level| level.trees()).collect();
//! assert_eq!(present, [&[0][..], &[0, 1], &[0, 1, 2]]);
//! assert!(proven.levels().iter().all(|level| level.lambda().is_some()));
//! // Four evaluations per fraction tree, two per forest.
//! assert_eq!(proven.levels()[2].proof().final_evals.len(), 4 + 2 + 4);
//! assert!(matches!(proven.leaf_claims()[1], LevelClaim::Product(_)));
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! # Matmul relations
//!
//! [`MatMulWeight`] is a batch of tokens' activations times a weight matrix's transpose,
//! `Y = X·Wᵀ`, and [`MatMulAct`] an activation matrix times an activation matrix, `B` stored as it
//! is or transposed ([`RightLayout`]), each stacked over `2^n` layers or heads whose index takes
//! the high variables. Each borrows its operands, row-major over the prime subfield, with their
//! shape. Stacked, a claim on the product at a point with the stacking coordinate `ρ` is a sum
//! over the inner index and the stacking index of `eq(ρ, ·)` times the two operands, each
//! contracted with `eq` at the point's other coordinates (RFC-0010 'Relations'):
//!
//! 1. `tables` builds the instance's tables from the input claim's point, never its value: the
//!    contractions, one pass over each operand they contract, and `eq(ρ, ·)` as a public factor;
//! 2. the stage's [`prove_batch`] proves the instance among others, of a degree at least the
//!    relation's: 3 when stacked, 2 otherwise;
//! 3. `claims` maps the instance's point and evaluations to one [`Claim`] per operand.
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_poly::evaluate_base;
//! use tessera_sumcheck::{MatMulWeight, MatMulWeightShape, Prefix, Protocol, prove_batch};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! // 2^2 layers, each 2^2 tokens' activations of 2^4 inputs by a 2^3 × 2^4 weight matrix.
//! let shape = MatMulWeightShape { layer_vars: 2, token_vars: 2, out_vars: 3, in_vars: 4 };
//! let w: Vec<Goldilocks> = (0..1 << 9).map(|k| fp(k * k + 1)).collect();
//! let x: Vec<Goldilocks> = (0..1 << 8).map(|k| fp(3 * k + 2)).collect();
//! let matmul = MatMulWeight::<GoldilocksExt2>::new(&w, &x, shape)?;
//! // An input claim on Ỹ at r_o ‖ r_t ‖ ρ, 3 + 2 + 2 coordinates; the prover never needs its
//! // value.
//! let point: Vec<GoldilocksExt2> =
//!     (1..=7).map(|k| GoldilocksExt2::new(fp(k), fp(7 * k))).collect();
//! let tables = matmul.tables(&point)?;
//! let mut transcript = Transcript::new(&context);
//! let degree = matmul.degree();
//! let instances = [tables.instance()];
//! let proven = prove_batch(&mut transcript, Protocol::One, Prefix::Stage(1), degree, &instances)?;
//! let at = proven.instance_point(0).expect("one instance");
//! let claims = tables.claims(at, proven.instance_evals(0).expect("one instance"))?;
//! // W at r_i ‖ r_o ‖ ρ_ℓ and X at r_i ‖ r_t ‖ ρ_ℓ, each its operand's extension there.
//! assert_eq!(claims.w.point[4..7], point[..3]);
//! assert_eq!(claims.x.point[4..6], point[3..5]);
//! assert_eq!(Ok(claims.w.value), evaluate_base(&w, &claims.w.point));
//! assert_eq!(Ok(claims.x.value), evaluate_base(&x, &claims.x.point));
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! # Elementwise relations
//!
//! An [`Elementwise`] relation on `n` variables is a sum of weighted products of its [`Input`]s
//! against `eq` at its target's point, `T̃(r) = Σ_x eq(r, x_high)·Σ_t w_t·Π_{f ∈ t} I_f(x)`: an
//! operand, read at each vertex, at each entry's RoPE partner, or broadcast over some variables;
//! or a public table, which the verifier evaluates. Its target has the high `n − m` variables:
//! all of them but for `SumSquares`, which sums the low `m`. The steps are the matmuls':
//! `tables` from the input claim's point, [`prove_batch`] on their instance, then `claims`, one
//! per operand input. A `ZeroCheck`'s point is the stage's, from [`zero_check_point`], and its
//! claim 0.
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_poly::evaluate_base;
//! use tessera_sumcheck::{Elementwise, Input, Prefix, Protocol, prove_batch, zero_check_point};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! // 2^2 rows of 2^3 entries, each row scaled by its own factor: R_q times the rows of PV.
//! let pv: Vec<Goldilocks> = (0..32).map(|k| fp(3 * k + 1)).collect();
//! let r: Vec<Goldilocks> = (0..4).map(|k| fp(k + 7)).collect();
//! let scaled = Elementwise::<GoldilocksExt2>::mul(
//!     Input::Broadcast { operand: &r, low_vars: 3 },
//!     Input::Operand(&pv),
//!     5,
//! )?;
//! // A one-hot row, whose booleanity is a zero-check at the stage's point.
//! let h: Vec<Goldilocks> = [0, 0, 1, 0, 0, 0, 0, 0].into_iter().map(fp).collect();
//! let boolean = Elementwise::<GoldilocksExt2>::booleanity(Input::Operand(&h), 3)?;
//!
//! let mut transcript = Transcript::new(&context);
//! let z: Vec<GoldilocksExt2> = zero_check_point(&mut transcript, 1, 3);
//! let point: Vec<GoldilocksExt2> = (1..=5).map(|k| GoldilocksExt2::from(fp(11 * k))).collect();
//! let tables = [scaled.tables(&point)?, boolean.tables(&z)?];
//! let instances = [tables[0].instance(), tables[1].instance()];
//! let proven = prove_batch(&mut transcript, Protocol::One, Prefix::Stage(1), 3, &instances)?;
//! // R at the high 2 coordinates of the instance's point, PV at all 5.
//! let at = proven.instance_point(0).expect("an instance");
//! let claims = tables[0].claims(at, proven.instance_evals(0).expect("an instance"))?;
//! assert_eq!(claims[0].point, at[3..]);
//! assert_eq!(Ok(claims[0].value), evaluate_base(&r, &claims[0].point));
//! assert_eq!(Ok(claims[1].value), evaluate_base(&pv, at));
//! # Ok::<(), tessera_sumcheck::SumcheckError>(())
//! ```
//!
//! A [`Linear`] relation, a target that is a public linear combination of its inputs, costs no
//! sumcheck: [`LinearClaim::substitute`] replaces a claim on its target by the combination of
//! its inputs, each at the point its [`Placement`] gives, so a residual stream's claim after
//! its last layer ends on its first state and every update.
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
//! - **Sums of products.** Each table that a term reads writes its values on the line through
//!   each pair into a row, once per block of pairs however many terms read it. The factors every
//!   term shares, such as `eq`, multiply once per pair, and terms of equal weight share one sum,
//!   their weight applied once per round: a fraction tree's level costs 5 products per pair and
//!   node, 3 with numerators of 1. A table no term reads, as those numerators, is never bound: it
//!   is evaluated once at the instance's point. Each table is bound once per round.
//! - **Contractions.** A matmul contracts each operand once, in order, one product of an element
//!   of `F` by a base element per entry, into a table over `F` of the instance's size. The middle
//!   variables are summed in panels of 8 rows, 4 columns at a time, the low ones 4 rows at a time,
//!   with 4 independent sums in registers: 0.9 to 1.4 ns per entry on one Apple M4 Max core,
//!   where a pass over the output per row of the operand takes 1.5 to 1.8 ns. With the sumcheck
//!   linear in the instance, a matmul costs `O(|A| + |B|)` field operations beyond the product it
//!   proves (`benches/matmul.rs`).
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

mod bound;
mod claim;
mod contract;
mod elementwise;
mod error;
mod forest;
mod fraction_rows;
mod kernel;
mod level;
mod linear;
mod matmul;
mod merge;
mod proof;
mod prover;
mod recompose;
mod relation_memory;
pub use relation_memory::RelationMemory;
mod rotate;
mod select;
mod terms;
mod transcript;
mod trees;

pub use bound::Protocol;
pub use claim::Claim;
pub use elementwise::{Elementwise, ElementwiseShape, ElementwiseTables, Input, zero_check_point};
pub use error::SumcheckError;
pub use forest::{Forest, ProvenForests, ProvenLevel, check_forests, prove_forests};
pub use fraction_rows::{FractionTreeRows, FractionTreeSource};
pub use linear::{Linear, LinearClaim, LinearInput, LinearTerm, Placement};
pub use matmul::{
    MatMulAct, MatMulActClaims, MatMulActShape, MatMulActTables, MatMulWeight, MatMulWeightClaims,
    MatMulWeightShape, MatMulWeightTables, Operand, RightLayout,
};
pub use merge::{
    MAX_MERGE_VARS, Merge, MergeClaim, MergePolynomial, MergeTables, largest_table_rounds,
    prove_merge, prove_merge_two_pass,
};
pub use proof::SumcheckProof;
pub use prover::window::{
    MAX_WINDOW_VARS, PreparedWindow, WindowError, WindowInstance, WindowMemory,
    WindowMetadataMemory, WindowPass, WindowShape, WindowSource, WindowTable,
};
pub use prover::{
    BatchMemory, BatchShape, Computed, ComputedState, Instance, Prefix, ProvenBatch, Table,
    TreeScope, prove_batch,
};
pub use recompose::{Recomposition, RecompositionTables};
pub use rotate::{Rotate, RotateTables};
pub use select::{Selected, Selection, SelectionTables};
pub use terms::Term;
/// The largest degree of any sumcheck, 10, `tessera-protocol/2`'s at a reduction stage: a
/// sumcheck's own bound is [`Protocol::max_degree`].
pub use tessera_poly::MAX_DEGREE;
pub use transcript::SumcheckTranscript;
pub use trees::{
    BoundedTree, FractionClaim, FractionTree, LeafNumerators, LevelClaim, PreparedTreeBatch,
    ProvenTreeLevel, ProvenTrees, Tree, prove_trees, prove_trees_in,
};
