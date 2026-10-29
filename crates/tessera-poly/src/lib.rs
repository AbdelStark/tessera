// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Multilinear tables, `eq` tables, MLE evaluation, univariate round polynomials and masking
//! helpers (RFC-0010).
//!
//! The conventions are those of RFC-0010 'Multilinear extensions':
//!
//! - A table of `2^n` values `f[0], …, f[2^n − 1]` is the multilinear polynomial
//!   `f̃(x_1, …, x_n)` equal to `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least
//!   significant bit of the index. A point `r = (r_1, …, r_n)` sets `x_i = r_i`, so slices of
//!   coordinates list the lowest variable first.
//! - `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))` and `f̃(r) = Σ_x f[x]·eq(r, x)`.
//! - Binding the lowest variable to `r` maps each pair `(f[2k], f[2k+1])` to
//!   `f[2k] + r·(f[2k+1] − f[2k])`: the table of `f̃(r, x_2, …, x_n)`. Round `j` of a sumcheck
//!   binds `x_j`, so a stage's final point lists its challenges in round order.
//!
//! The pieces:
//!
//! - **Tables** are slices of length `2^n`. [`bind_lowest`] binds a table over the prime subfield
//!   into a new table over the challenge field (the first round of a sumcheck over committed
//!   values); [`bind_lowest_in_place`] binds a table in place, without allocating (every later
//!   round). [`evaluate`] and [`evaluate_base`] give `f̃` at a point. A table of activations or
//!   weights is secret-class, and so are the tables derived from it: a bound table comes back in
//!   a [`SecretBuf`](tessera_codec::SecretBuf), and an evaluation keeps its partial sums in one,
//!   which clears its memory when it drops (05 §3 OBS-6).
//! - **`eq`**: [`eq_table`] builds the `2^n` values of `eq(r, ·)` with one multiplication each;
//!   [`eq`] evaluates `eq(a, b)` at two points.
//! - **Predicates** with `O(n)` closed forms, which the prover and the verifier each evaluate:
//!   [`less_than`] (`LT(r, a)`, RFC-0008 'Structured evaluation'), [`less_than_points`]
//!   (`LT2(x, y)`) and [`causal_selector`] (`C(q, i)`, RFC-0009 'SoftmaxRow').
//! - **Round polynomials**: [`RoundPoly`] holds a round message, the evaluations at
//!   `0, 2, …, d`, and evaluates it at a challenge from the round's claim.
//!
//! A sumcheck over a committed `Fp` table binds it into `Fp2` in the first round and in place
//! after, which ends at its value at the point of challenges:
//!
//! ```
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
//! use tessera_poly::{bind_lowest, bind_lowest_in_place, evaluate_base};
//!
//! let fp = Goldilocks::from_u64_reduced;
//! let table: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let point: Vec<GoldilocksExt2> =
//!     (0..3).map(|i| GoldilocksExt2::new(fp(i + 2), fp(i))).collect();
//! let mut bound = bind_lowest(&table, point[0])?;
//! for &r in &point[1..] {
//!     bind_lowest_in_place(&mut bound, r)?;
//! }
//! assert_eq!(bound[..], [evaluate_base(&table, &point)?]);
//! # Ok::<(), tessera_poly::PolyError>(())
//! ```
//!
//! The code is generic over the field traits of `tessera-field`, so it runs over `Goldilocks` and
//! `GoldilocksExt2` and over the calibration fields `F65537` and `F65537Ext2`; [`PolyField`] names
//! the prime subfield of each. Every input-dependent failure is a [`PolyError`]. Control flow
//! depends on lengths and on the public bound of [`less_than`] only, never on table values or
//! points (INV-ARCH-2). The masking helpers of RFC-0013 come with the masked sumcheck.

#![forbid(unsafe_code)]

mod eq;
mod error;
mod field;
mod multilinear;
mod predicates;
mod round;

pub use eq::{eq, eq_table};
pub use error::PolyError;
pub use field::PolyField;
pub use multilinear::{bind_lowest, bind_lowest_in_place, evaluate, evaluate_base, num_vars};
pub use predicates::{
    batch_causal_selector, causal_selector, causal_selector_bounded, less_than, less_than_points,
    slot_selector,
};
pub use round::{MAX_DEGREE, RoundPoly};
