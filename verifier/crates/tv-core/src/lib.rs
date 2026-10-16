// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation and lowering, range re-derivation, schedule derivation and every proof
//! check, run in the fixed order of RFC-0019.
//!
//! This version holds the sumcheck checks of RFC-0019 'Check order' steps 7 and 8: the rounds
//! of a sumcheck, batched instances with front-padding, and the levels of the leaf stage's
//! product trees (RFC-0010 'Multilinear extensions', 'Tree levels', 'Batching within a stage',
//! 'Sumcheck messages'). They are written from the specification without reference to the
//! prover's sumcheck code, so that the two agree only because both follow the specification
//! (SP-6, RFC-0001). Where the specification admits several algorithms this crate takes another
//! than the prover's: a round polynomial is evaluated by Newton's forward differences.
//!
//! | Item | What it checks |
//! |---|---|
//! | [`sumcheck::reduce`] | the rounds of one sumcheck, which reduce a claim on a sum to a claim at a point |
//! | [`sumcheck::verify_batch`] | a batched sumcheck: the batching challenge, front-padded instances, the rounds, and the final evaluations against the relation the caller supplies |
//! | [`tree::verify_product_level`] | one level of the leaf stage's product trees, and the claims it passes to the next level |
//! | [`sumcheck::eq`] | `eq(a, b)`, the public factor of RFC-0010 'Multilinear extensions' |
//!
//! The stage schedule (#114), the relations of the stages, the masked sumcheck of RFC-0013, the
//! LogUp fraction-tree levels (#62) and the verifier's driver and verdicts (#128) build on these.
//!
//! # Specification
//!
//! What this crate computes and rejects, written down from RFC-0010, RFC-0014 'Transcript
//! schedule' and 'Soundness budget', RFC-0004 'Transcript', RFC-0019 'Robustness' and 04 §2
//! before the code. The tests check every item against `spec/vectors/sumcheck/`, written by an
//! independent Python prover, and with inputs that violate exactly one check.
//!
//! ## Conventions
//!
//! - A table of `2^n` values `f[0], …, f[2^n − 1]` is the multilinear `f̃(x_1, …, x_n)` equal to
//!   `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. Round
//!   `j` binds `x_j` to its challenge, so the point a sumcheck ends at lists its challenges in
//!   round order, low variables first.
//! - `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))` for two points of the same length; the
//!   empty product is 1.
//! - Labels follow RFC-0014's patterns under a [`sumcheck::Prefix`], `stage/{k}` or `tree/{ℓ}`,
//!   with decimal indices counted from 0. RFC-0010 counts rounds from 1: its round `j` is
//!   absorbed under `<prefix>/round/{j − 1}`, and errors report it as round `j − 1`, the index
//!   its labels carry.
//! - A message is absorbed as TCE `Vec<Fp2>`: its length as a little-endian `u64`, then each
//!   element as `c0 ‖ c1`, 8 little-endian bytes each.
//!
//! ## Rounds
//!
//! [`sumcheck::reduce`]`(transcript, prefix, d, n, claim, rounds)` checks in this order, and the
//! first check that fails is the error:
//!
//! 1. `1 ≤ d ≤ 7` ([`MAX_DEGREE`](sumcheck::MAX_DEGREE), RFC-0014 'Soundness budget'), else
//!    `Degree`, before the transcript moves.
//! 2. Round by round, for `j = 0, …, n − 1`:
//!    1. round `j` is in the proof, else `MissingRound { round: j }`;
//!    2. it holds `d` evaluations `p(0), p(2), p(3), …, p(d)`, else
//!       `RoundLength { round: j, expected: d, found }`;
//!    3. it is absorbed under `<prefix>/round/{j}`; then `p(1) = claim − p(0)`,
//!       `r_j = challenge_fp2(<prefix>/r/{j})`, and the claim becomes `p(r_j)`, with `p` the
//!       polynomial of degree at most `d` through `p(0), p(1), …, p(d)`.
//! 3. The proof has no round past `n − 1`, else `ExtraRound { round: n }`.
//!
//! It returns the point `(r_0, …, r_{n−1})` and the last claim. With `n = 0` there is no round
//! and no transcript operation, and the claim comes back as it went in. Nothing here compares the
//! last claim with anything: [`sumcheck::verify_batch`] does, and so must any other caller, which
//! is why the result is `#[must_use]`.
//!
//! ## Batches
//!
//! [`sumcheck::verify_batch`]`(transcript, prefix, d, instances, proof, relation)` checks
//! instances `i = 0, …, m − 1` in schedule order, instance `i` a claim `c_i` on `n_i` variables
//! whose relation lists `k_i` final evaluations; `n_max` is the largest `n_i`. In this order:
//!
//! 1. Before the transcript moves: `1 ≤ d ≤ 7`, else `Degree`; at least one instance, else
//!    `NoInstances`; `Σ k_i` fits a `usize`, else `EvaluationTotal`.
//! 2. `γ = challenge_fp2(<prefix>/gamma)`.
//! 3. The claim is `Σ_i γ^i · 2^{n_max − n_i} · c_i`: an instance with fewer variables is
//!    front-padded, constant in the lowest `n_max − n_i` variables, which the first
//!    `n_max − n_i` rounds bind.
//! 4. The rounds of [`sumcheck::reduce`] over `n_max` variables from that claim, under the same
//!    prefix, with its errors.
//! 5. The proof holds `Σ k_i` final evaluations, else
//!    `EvaluationCount { round: n_max, expected, found }`.
//! 6. They are absorbed as one TCE `Vec<Fp2>` under `<prefix>/evals`.
//! 7. Instance `i`'s point is the last `n_i` challenges and its evaluations the next `k_i` of the
//!    proof's, instance by instance; `relation(i, point, evaluations)` is its relation's value
//!    `g_i` there, public factors (`eq`, selectors, constants) included, or `None` when the
//!    caller cannot evaluate it: `Relation { instance: i }`.
//! 8. `Σ_i γ^i · g_i` equals the last claim, else `FinalMismatch { round: n_max }`. A batch of
//!    `n_max = 0` has no rounds and checks its claim here directly.
//!
//! It returns `γ` and the point. The scale `2^{n_max − n_i}` enters the claim only: the padded
//! polynomial takes, at the stage's point, the value of the instance's own polynomial at its
//! own point.
//!
//! ## Product-tree levels
//!
//! [`tree::verify_product_level`]`(transcript, ℓ, claims, proof)` checks level `ℓ` of the leaf
//! stage for the product-tree instances present there, in schedule order, each a claim
//! `Ṽ(r_i) = v_i` on its current level over `n_i = |r_i|` variables:
//!
//! 1. to 8. Those of [`sumcheck::verify_batch`] under `tree/{ℓ}`, of degree 3, with `c_i = v_i`,
//!    `k_i = 2`, the evaluations `u_0, w_0, u_1, w_1, …`, and `g_i = eq(r_i, r'_i)·u_i·w_i`,
//!    `r'_i` the last `n_i` challenges: the check is
//!    `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i = claim`.
//! 9. `ρ = challenge_fp2(tree/{ℓ}/line)`.
//!
//! It returns `γ`, the point, `ρ`, and, in the same order, each instance's claim on its next
//! level: `Ṽ_{ℓ+1}(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)` on `n_i + 1` variables, `ρ` the lowest, the
//! pair bit of `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`. The caller says which instances are
//! present. Each is a forest of `2^s` trees of one height `h`, stacked with the tree index in the
//! high `s` variables; a single tree has `s = 0`. With `H` the tallest forest's height, a forest
//! joins at level `H − h`, with its root claim `Σ_y v_y·eq(r, y)` on `s` variables. Its public
//! root values `v_y` give the claim, at the root point `r` that the caller squeezes under
//! `tree/{ℓ}/root/{c}` before the level. After the forest's last level, the claims returned
//! are its leaf claims.
//!
//! The check of step 8 is symmetric in `u_i` and `w_i`, and a `u_i` whose `w_i` is zero does
//! not enter it: such changes pass the level. They do not pass the next claim, which absorbs
//! both values before `ρ` and places them on the line through `Ṽ_{ℓ+1}(0 ‖ r'_i)` and
//! `Ṽ_{ℓ+1}(1 ‖ r'_i)`, so the next level, or the stage that consumes the leaf claims, rejects
//! them. A level that passed has proven nothing until its next claims are checked, so its result
//! is `#[must_use]`.
//!
//! RFC-0010 runs the LogUp fraction trees in the same levels and under the same labels; how one
//! level orders and lays out instances of both kinds is open (RFC-0010 OQ-0010-1). This check
//! takes levels whose instances are all product trees, and LogUp-GKR (#62) generalizes it.
//!
//! ## Errors
//!
//! [`sumcheck::SumcheckError`] says whose fault an error is:
//!
//! - The proof's: `MissingRound`, `RoundLength`, `ExtraRound`, `EvaluationCount` and
//!   `FinalMismatch`, a `Reject(Sumcheck { stage, round })` in 04 §2, with `round` from
//!   [`rejected_round`](sumcheck::SumcheckError::rejected_round). A failed final check is
//!   reported at round `n_max`, one past the last round's index.
//! - The verifier's: `Degree`, `NoInstances`, `EvaluationTotal` and `Relation` judge what the
//!   caller passes in, the schedule's degree and instances and its relation; `Label` would mean
//!   a label built here was not ASCII, and labels are built from decimal indices; `Transcript`
//!   is a challenge that found no value after 16 squeezes, with probability below `2^-496`
//!   (RFC-0004 'Failure modes'). All are `Inconclusive(ERR-INTERNAL)`, never a verdict on a
//!   proof.
//!
//! ## Robustness and soundness
//!
//! Every input yields a value or a typed error: nothing here panics, indexes unchecked or
//! computes sizes unchecked (RFC-0019 'Robustness'). Work and memory are bounded by the proof's
//! size and the caller's instances, never by a number the proof states: the rounds stop at the
//! first missing one, so a point holds at most as many challenges as the proof has rounds; a
//! message is encoded for the transcript only after its length check, at most `8 + 16·7` bytes
//! a round and `8 + 16·Σ k_i` for the evaluations; `2^{n_max − n_i}` takes one squaring per bit.
//!
//! A batch of `m` instances over `n_max` rounds of degree `d` accepts a false claim with
//! probability at most `(n_max·d + m)/|Fp2|` (RFC-0010 'Soundness of the schedule'). A false
//! claim either fails the final check or moves into the final evaluations, which the next stage,
//! or the openings, then check.
//!
//! # Features
//!
//! - `test-params` accepts the `test-small` parameter set, which certifies nothing. Tests only: a
//!   release build with it fails to compile.

#![forbid(unsafe_code)]
// Proofs come from a hostile prover and must be rejected with typed errors, never by panicking
// (RFC-0019 'Robustness'). Size arithmetic is checked or saturating, conversions are explicit,
// and field arithmetic is exempt through its own types (`clippy.toml`).
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::as_conversions
    )
)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-params", not(debug_assertions)))]
compile_error!(
    "the `test-params` feature accepts parameter sets that certify nothing; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod fiat_shamir;
mod interpolation;
pub mod sumcheck;
pub mod tree;
