// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation and lowering, range re-derivation, schedule derivation and every proof
//! check, run in the fixed order of RFC-0019.
//!
//! This version holds the sumcheck checks of RFC-0019 'Check order' steps 7 and 8: the rounds
//! of a sumcheck, batched instances with front-padding, and the levels of the leaf stage's
//! product trees (RFC-0010 'Multilinear extensions', 'Tree levels', 'Batching within a stage',
//! 'Sumcheck messages'). It also holds the opening checks of the advice commitment, steps 9 and
//! 11: the group evaluations, then the queried columns' leaves, multiproof and consistency with
//! the combined rows (RFC-0011 'Opening protocol'). They are written from the specification
//! without reference to the prover's code, so that the two agree only because both follow the
//! specification (SP-6, RFC-0001). Where the specification admits several algorithms this crate
//! takes another than the prover's: a round polynomial is evaluated by Newton's forward
//! differences; a combined row is evaluated at its column point by merging pairs in one pass,
//! with no table of `eq`; and the row coefficients come from tables of `eq` over 8 bits of the
//! row index at a time.
//!
//! | Item | What it checks |
//! |---|---|
//! | [`sumcheck::reduce`] | the rounds of one sumcheck, which reduce a claim on a sum to a claim at a point |
//! | [`sumcheck::verify_batch`] | a batched sumcheck: the batching challenge, front-padded instances, the rounds, and the final evaluations against the relation the caller supplies |
//! | [`tree::verify_product_level`] | one level of the leaf stage's product trees, and the claims it passes to the next level |
//! | [`sumcheck::eq`] | `eq(a, b)`, the public factor of RFC-0010 'Multilinear extensions' |
//! | [`opening::check_evaluations`] | the advice opening's combined rows: their shape, and each group's claims against its combined row (RFC-0011 step 6) |
//! | [`opening::Evaluated::check_columns`] | the answer to the column queries: its shape, the leaves and the multiproof, and every combined row against every queried column (RFC-0011 step 8) |
//!
//! The stage schedule (#114), the relations of the stages, the masked sumcheck of RFC-0013, the
//! LogUp fraction-tree levels (#62) and the verifier's driver and verdicts (#128) build on these;
//! so do the advice layout (#116), the transcript steps that squeeze the opening's challenges and
//! queries, the hiding of RFC-0013 (blinding rows, `u'_g`, `q_g`) and the weight commitment
//! (RFC-0012).
//!
//! # Specification
//!
//! What the sumcheck checks compute and reject, written down from RFC-0010, RFC-0014 'Transcript
//! schedule' and 'Soundness budget', RFC-0004 'Transcript', RFC-0019 'Robustness' and 04 §2
//! before the code. The tests check every item against `spec/vectors/sumcheck/`, written by an
//! independent Python prover, and with inputs that violate exactly one check. The opening checks
//! follow, in 'Openings'.
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
//! ## Openings
//!
//! What the opening checks of the advice commitment compute and reject, written down from
//! RFC-0011 ('Canonical advice layout', 'Encoding', 'Column digests and root', 'Opening protocol'
//! steps 1 to 8, 'Failure modes'), RFC-0004 ('Column digests', 'Merkle trees'), RFC-0002
//! ('Constants'), RFC-0019 ('Check order' steps 9 to 11, 'Robustness') and 04 §2 before the
//! code. The tests check every item against `spec/vectors/pcs/opening.json`, written by an
//! independent Python oracle, and with inputs that violate exactly one check. The caller derives
//! the layout and runs the transcript, which squeezes the batching challenges `γ`, the point `ξ*`
//! and the queries; these checks take them as inputs.
//!
//! ### Conventions
//!
//! - `k = 2^log2_k` and `n = k·2^log2_inv_rate`. `H` and `L` are the subgroups of `Fp^*` of
//!   order `k` and `n`, and `ω_m = 7^((P − 1)/m)` generates the one of order `m`. Column `j` of
//!   the commitment is checked at `x_j = 7·ω_n^j`, on the coset `7·L`, which is disjoint from
//!   `H`.
//! - The advice polynomial `Ã` has the column index in its low `log2_k` variables and the row
//!   index in its high `row_vars`, so a claim `Ã(column_point ‖ row_point) = value` lists the
//!   column point first. A point lists its coordinates lowest variable first: coordinate `t`
//!   sets bit `t` of an index, and `eq(r, x) = Π_t (r_t·x_t + (1 − r_t)·(1 − x_t))`.
//! - The layout lists the non-zero rows of each of the `S` segments in layout order, `N` rows in
//!   all. Every other row is zero, and is neither committed nor summed over. A column holds one
//!   symbol per non-zero row, segment by segment, in layout order.
//! - Group `g`'s row coefficients are `c_g(row) = Σ_{claims of g} γ·eq(row_point, row)`, and the
//!   proximity coefficients are `ξ_row = eq(ξ*, row)`. The verifier computes both itself, for
//!   every non-zero row.
//! - A combined row is `u`, `k` values in `Fp2`, and `M`, `P_A` coefficients in `Fp2`. `enc(u)`
//!   is the polynomial of degree below `k` that takes the values `u` on `H`,
//!   `enc(u)(ω_k^i) = u_i`, and the row's value at column `j` is
//!   `enc(u)(x_j) + Z_H(x_j)·M(x_j)`, with `Z_H(X) = X^k − 1` and `M(X) = Σ_{i < P_A} M_i·X^i`.
//! - `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))`, where `d_j^(s)` is the chained digest
//!   [`column_digest`](tv_hash::column::column_digest)`(commitment_id, j, B, ·)` of segment `s`'s
//!   symbols at column `j`, possibly none. The root is the Merkle root over the `n` leaves.
//!
//! ### The verifier's inputs
//!
//! [`opening::check_evaluations`]`(opening, combined)` first checks the [`opening::Opening`] the
//! verifier built from public data and the transcript, before it reads the proof, in this order.
//! Each failure is a fault of the verifier:
//!
//! 1. `log2_inv_rate ≥ 1` and `log2_k + log2_inv_rate ≤ 32`, so that `L` exists and `n ≥ 2k`
//!    symbols determine `enc(u) + Z_H·M`, of degree below `k + P_A ≤ 2k`; and `n` is addressable.
//!    Else `CodeLength`.
//! 2. `1 ≤ P_A ≤ k`, else `MaskCoefficients`. RFC-0011 admits `P_A = k`, though at rate 1/2
//!    the code is then all of `Fp^n` and the proximity check proves nothing: the parameter sets
//!    keep `ρ = (k + P_A)/n ≤ 0.51`, and RFC-0019 step 2 accepts the parameter set before these
//!    checks run.
//! 3. `B ≥ 1`, else `RowBlock`.
//! 4. `row_vars ≤ 64`, the bits of a `u64` row index, else `RowVars`.
//! 5. At least one segment, else `NoSegments`. A segment may be empty.
//! 6. Row by row in layout order: the row is above the one before it, across segments too, else
//!    `RowOrder { segment, position }`; then it is below `2^row_vars`, else
//!    `RowRange { segment, position }`.
//! 7. Group by group: its column point has `log2_k` coordinates, else
//!    `ColumnPoint { group, length }`; then claim by claim, its row point has `row_vars`, else
//!    `RowPoint { group, claim, length }`.
//! 8. `ξ*` has `row_vars` coordinates, else `XiPoint { length }`.
//!
//! There may be no group, and a group may have no claim. Such a group claims 0 and weighs no row,
//! so its combined row must open to 0 at its column point and be 0 at every queried column.
//!
//! ### The evaluation check
//!
//! Then [`opening::check_evaluations`] checks the combined rows of the proof (RFC-0011 steps 5
//! and 6, RFC-0019 step 9), in this order:
//!
//! 1. Their shape: `G + 1` rows, the groups' in group order and then the proximity row; then row
//!    by row, `u` holds `k` values and `M` holds `P_A`. Else `Evaluation { group: None }`.
//! 2. Group by group, for `g = 0, …, G − 1`:
//!    `Σ_{claims of g} γ·value = Σ_{i < k} u_g[i]·eq(column_point_g, i)`, else
//!    `Evaluation { group: Some(g) }`.
//!
//! It returns the rows as an [`opening::Evaluated`], which is `#[must_use]`: they are bound to
//! the claims, and to the commitment only once the columns are checked.
//!
//! ### The column check
//!
//! [`opening::Evaluated::check_columns`]`(queries, columns, siblings)` checks the answer to the
//! queries (RFC-0011 steps 7 and 8, RFC-0019 step 11). The queries are the `τ` drawn indices in
//! draw order, which may repeat. In this order:
//!
//! 1. The queries, which are the verifier's: at least one, else `NoQueries`; each below `n`,
//!    else `QueryRange { position }` at the first in draw order that is not. `D` is their sorted
//!    set without repeats, and `j_0` its smallest element.
//! 2. The answer's shape (step 8.1): one column per element of `D`, then each of `N` symbols,
//!    else `MerklePath { column: j_0 }`.
//! 3. The leaves and the multiproof (step 8.2): the answer's column `i` is column `D_i`, its
//!    symbols are split among the segments by their row counts, and its leaf is recomputed; then
//!    [`verify_multiproof`](tv_hash::merkle::verify_multiproof)`(root, n, [(D_i, leaf_i)…],
//!    siblings)`. Missing or extra siblings and any other root, from a wrong leaf or sibling or
//!    one out of place, are `MerklePath { column: j_0 }`: a multiproof fails as a whole, never at
//!    one column.
//! 4. Consistency (step 8.3): for each `j` of `D` in increasing order, every combined row, the
//!    groups' in group order and then the proximity row, satisfies
//!    `Σ_row c(row)·col_j[row] = enc(u)(x_j) + Z_H(x_j)·M(x_j)`, with `c` the row's `c_g` or `ξ`
//!    and the sum over the non-zero rows. The first `j` at which some row fails is
//!    `Proximity { column: j }`.
//!
//! ### Opening errors
//!
//! [`opening::OpeningError`] says whose fault an error is, and
//! [`code`](opening::OpeningError::code) gives its identifier in 04 §2 and §3. The commitment
//! class of the rejection is the caller's, the advice commitment here.
//!
//! | Error | Fault | Verdict (04 §2) |
//! |---|---|---|
//! | `Evaluation { group }` | the proof's | `Reject(Evaluation { commitment })`, `REJ-PCS-EVAL`; `group` locates the failure and is not part of the reason |
//! | `MerklePath { column }` | the proof's | `Reject(MerklePath { commitment, column })`, `REJ-PCS-MERKLE` |
//! | `Proximity { column }` | the proof's | `Reject(Proximity { commitment, column })`, `REJ-PCS-PROXIMITY` |
//! | `Memory` | the verifier's: memory for a buffer the parameters size could not be reserved | `Inconclusive(ERR-RES-MEMORY)` |
//! | `CodeLength`, `MaskCoefficients`, `RowBlock`, `RowVars`, `NoSegments`, `RowOrder`, `RowRange`, `ColumnPoint`, `RowPoint`, `XiPoint`, `NoQueries`, `QueryRange` | the verifier's: its own inputs | `Inconclusive(ERR-INTERNAL)` |
//! | `Internal` | the verifier's: a step that the checks before it make infallible failed, a bug of this crate that no input reaches | `Inconclusive(ERR-INTERNAL)` |
//!
//! A column is below `n ≤ 2^32`, so it is a `u32`, as 04 §2 writes it.
//!
//! ### Cost and memory
//!
//! Every length the proof gives is compared with the parameters or the layout before anything is
//! allocated or computed from it, so a proof of any shape costs the verifier no more than an
//! honest one; nothing panics, indexes unchecked or computes sizes unchecked (RFC-0019
//! 'Robustness').
//!
//! - The evaluation check allocates nothing. It reduces each `u_g` to its value at the column
//!   point in one pass, merging each pair of values with the coordinate of its level as soon as
//!   both are known: `k − 1` products in `Fp2` per group, and 33 values of state.
//! - The column check allocates the sorted queries and the leaves (`τ` and `|D|` entries), the
//!   coefficients of one combined row (`N` values in `Fp2`), one codeword (`n` values in `Fp2`),
//!   and the NTT's `n/2` twiddles. Hashing the leaves reads `|D|·N` symbols. Then, one combined row
//!   at a time: its coefficients, at `⌈row_vars/8⌉` products per claim and non-zero row, from
//!   tables of `γ·eq` over 8 bits of the row index; its codeword, an inverse NTT of size `k` on
//!   `H`, the mask folded in as `g = f + (X^k − 1)·M`, and a coset NTT of size `n` with shift 7;
//!   and `|D|` inner products of length `N`. A row is checked only at the positions before the
//!   first failure found so far, which keeps the reported column exact: it is the smallest
//!   failing position over the rows.
//! - So the column check costs `G + 1` NTTs of sizes `k` and `n` in `Fp2`, `(G + 1)·|D|·N`
//!   products of an `Fp2` coefficient by an `Fp` symbol, and the hashing of `|D|·N` symbols.
//!   Holding one codeword at a time, it needs about `16·n + 16·N + 4·n` bytes beyond its inputs.
//! - Measured kernel by kernel on one core of an Apple M4 Max at `k = 2^22`, `n = 2^23`,
//!   `N = 5.8×10^5`, `|D| = 193`, `G + 1 = 17` and 40 claims: re-encoding takes 0.6 s per
//!   combined row, about 11 s in all, and dominates; the consistency 1.2 to 1.3 ns per product,
//!   about 2.4 s; the leaves 0.9 s; the coefficients 0.4 s; the evaluation check 0.5 s. The rows
//!   are independent of one another, so both large loops parallelize across them: RFC-0019's
//!   performance plan gives them about 5 s and 2 s on 32 cores.
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
pub mod opening;
pub mod sumcheck;
pub mod tree;
