// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation and lowering, range re-derivation, schedule derivation and every proof
//! check, run in the fixed order of RFC-0019.
//!
//! This version holds the sumcheck checks of RFC-0019 'Check order' steps 7 and 8: the rounds
//! of a sumcheck, batched instances with front-padding, and the levels of the leaf stage, where
//! product forests and LogUp fraction trees run together (RFC-0010 'Multilinear extensions',
//! 'Tree levels', 'Batching within a stage', 'Sumcheck messages'). Around those levels it holds
//! LogUp-GKR's own checks of step 7: the roots before the first level and the leaf checks after
//! the last (RFC-0008 'Fraction tree'). It also holds the opening checks of the advice
//! commitment, steps 9 and 11: the group evaluations, then the queried columns' leaves,
//! multiproof and consistency with the combined rows (RFC-0011 'Opening protocol'), and the
//! claim list: its decoding, its validation, RFC-0019 steps 2 and 3, and its Llama lowering
//! (RFC-0007, RFC-0039, RFC-0044; 'Claim lists' below). From the lowered graph it derives the
//! stage schedule and checks the reduction stages of step 8 and the leaf stage's splits
//! (RFC-0010, RFC-0037; [`schedule`] and [`stages`], whose documentation specifies them). They are
//! written from the specification without reference to the prover's code, so that the two agree
//! only because both follow the specification (SP-6, RFC-0001). Where the specification admits
//! several algorithms this crate takes another than the prover's: a round polynomial is evaluated
//! by Newton's forward differences; a combined row is evaluated at its column point by merging
//! pairs in one pass, with no table of `eq`; and the row coefficients come from tables of `eq`
//! over 8 bits of the row index at a time.
//!
//! | Item | What it checks |
//! |---|---|
//! | [`sumcheck::reduce`] | the rounds of one sumcheck, which reduce a claim on a sum to a claim at a point |
//! | [`sumcheck::verify_batch`] | a batched sumcheck: the batching challenge, front-padded instances, the rounds, and the final evaluations against the relation the caller supplies |
//! | [`logup::check_roots`] | LogUp's challenges `α` and `β`, then the roots of every instance's lookup tree and table tree, and the claims the trees enter the levels with |
//! | [`tree::verify_level`] | one level of the leaf stage, its instances product forests and LogUp fraction trees, and the claims it passes to the next level |
//! | [`tree::verify_product_level`] | one level of the leaf stage whose instances are all product forests |
//! | [`logup::check_leaves`] | the lookup trees' value evaluations, then each LogUp instance's leaf checks, and the claims they leave for later stages |
//! | [`sumcheck::eq`] | `eq(a, b)`, the public factor of RFC-0010 'Multilinear extensions' |
//! | [`opening::check_evaluations`] | the advice opening's combined rows: their shape, and each group's claims against its combined row (RFC-0011 step 6) |
//! | [`opening::Evaluated::check_columns`] | the answer to the column queries: its shape, the leaves and the multiproof, and every combined row against every queried column (RFC-0011 step 8) |
//! | [`weights::derive()`] | compact independent layouts and descriptor hashes for all ten weight matrices (RFC-0040) |
//! | [`weights::check_public`] | bind the public matrix roots, layout and epoch to trusted `R_W` (RFC-0040) |
//! | [`weights::check_evaluations`] | weight combined-row shapes and the schedule-derived evaluation claim (RFC-0012, RFC-0049) |
//! | [`weights::Evaluated::check_columns`] | direct weight-column authentication and both combined rows' consistency, with structural and committed-position indices kept distinct |
//! | [`claims::ClaimList`] | the claim list's schema, decoded from its sealed envelope and encoded back for `claim_list_digest` (RFC-0007, RFC-0026, RFC-0042) |
//! | [`protocol`] | independently bounded protocol/1 message and envelope decoding, field-based encoding and public-statement hashing (RFC-0036); syntax only |
//! | [`params::ParamSet`] | the parameter sets the verifier accepts and the layout each prescribes (RFC-0014, RFC-0026) |
//! | [`validation::validate`] | claim-list validation, RFC-0007 'Verifier validation' in RFC-0039's order: steps 1 to 7, and RFC-0019 step 2 before them |
//! | [`advice::Layout`] | complete public advice geometry, source locations, zero rows, segments and weighted whole-claim opening images (RFC-0050) |
//! | [`model_openings::Claims`] | every complete-model opening obligation, preserving original source points, layer coefficients and external certificate debts |
//! | [`validation::check_public`] | steps 2.1 to 2.3, which give the public part the lowering reads |
//! | [`lowering::lower`] | the Llama lowering of a claim list's public part, node by node (RFC-0044) |
//! | [`schedule::derive()`] | the stage schedule of a lowered graph: the leaf stage's claims, each reduction stage's instances with their relations, variables and degrees, and where every claim goes (RFC-0010, RFC-0037) |
//! | [`leaf::ModelSchedule`] / [`leaf::Schedule::verify`] | independently derived model lookup counts and Softmax forest roots; the shared tree/LogUp loop, with all leaf claims retained |
//! | [`leaf::split::Layout`] | canonical model lookup forms, aligned blocks and first-read source order; the combined split absorption and column identities, retaining all reduction/opening debts |
//! | [`stages::verify`] | the leaf stage's splits, then each reduction stage: its layer and zero-check points, its batched sumcheck with the public factors, its splits; and the claim pool |
//! | [`selectors`] | the public factors: `LT`, the causal selector `C̃`, the row selector `ψ̃`, layer weights `w̃`, `Rotate`'s kernel, and RFC-0035's padding zero-check |
//! | [`gadgets`] | final checks for requantization, NonNeg, BigIntCompare, OneHot, roots, causal Softmax and Argmax, with canonical source/range obligations (RFC-0009/0050) |
//! | [`soundness::SoundnessReport::evaluate`] | independent integer-only soundness calculator for every row of RFC-0014's table (RFC-0014, RFC-0049) |
//!
//! The masked sumcheck of RFC-0013 (#159), the leaf stage's integration of the trees with
//! the stages (#130), the
//! regeneration of the tables and the structured evaluation of `t̃` (#100), and the verifier's
//! driver and verdicts (#128) build on these; so do the transcript steps that squeeze the
//! opening's challenges and queries, and the hiding of RFC-0013 (blinding
//! rows, `u'_g`, `q_g`), which take the claim pool. [`weights`] adds the protocol1 weight
//! opening components; the full driver still joins them and [`gadgets`] to the schedule,
//! earlier LogUp/forest claims and canonical openings. No component result is a verdict.
//! `ConstraintSystemDigest` (#120) builds on the lowering. Validation already derives the
//! complete lookup catalog and checks the advice layout at `max_context`.
//!
//! # Claim lists
//!
//! The claim list is the verifier's own input: [`claims`] decodes it, [`validation`] checks it
//! before any proof message is read, and [`lowering`] rebuilds the graph `D` lowers to from its
//! fields alone, with code written from RFC-0044 and the documents it cites, not from the
//! compiler (RFC-0001, INV-CL-2). Each module's documentation specifies what it computes and
//! rejects, in order, and the tests check every rule against `spec/vectors/claims/`, the
//! compiler's golden lists, and `spec/vectors/lowering/`, an independent oracle's graphs, and
//! with claim lists that break exactly one rule.
//!
//! # Specification
//!
//! What the sumcheck and LogUp checks compute and reject, written down from RFC-0008 'Relation'
//! and 'Fraction tree', RFC-0010, RFC-0014 'Transcript schedule' and 'Soundness budget',
//! RFC-0004 'Transcript', RFC-0019 'Check order' and 'Robustness' and 04 §2 before the code. The
//! tests check every item against `spec/vectors/sumcheck/`, written by an independent Python
//! prover, and with inputs that violate exactly one check. The opening checks follow, in
//! 'Openings'.
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
//! ## Tree levels
//!
//! The leaf stage proves its trees level by level from their roots (RFC-0010 'Tree levels'). An
//! instance of a level is one of two kinds:
//!
//! - a product forest: `2^s` product trees of one height, stacked with the tree index in the high
//!   `s` variables (a single tree has `s = 0`), with its claim `Ṽ(r_i) = v_i` on its current
//!   level, a [`tree::TreeClaim`] over `n_i = |r_i|` variables;
//! - a LogUp fraction tree, a single tree, with its claims `Ñ(r_i) = p_i` on its current level's
//!   numerators and `D̃(r_i) = q_i` on its denominators, at one point, a [`tree::FractionClaim`]
//!   over `n_i = |r_i|` variables.
//!
//! Level `ℓ + 1` of a tree has twice the entries of level `ℓ`, and entry `x` of level `ℓ`
//! combines the pair `2x` (left) and `2x + 1` (right): `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`
//! in a product tree, `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)` in a fraction tree (RFC-0008
//! 'Fraction tree'). The pair bit is therefore the lowest variable of level `ℓ + 1`: entry
//! `2x + b` is `Ṽ_{ℓ+1}(b ‖ x)`, and likewise `N(b ‖ x)` and `D(b ‖ x)`.
//!
//! [`tree::verify_level`]`(transcript, ℓ, claims, proof)` checks level `ℓ` of the leaf stage for
//! the instances present there, in schedule order, numbered from 0 across both kinds. Instance
//! `i` lists `k_i` final evaluations: 2 for a forest, 4 for a fraction tree. In this order:
//!
//! 1. Before the transcript moves: at least one instance, else `NoInstances`; `Σ k_i` fits a
//!    `usize`, else `EvaluationTotal`.
//! 2. `γ = challenge_fp2(tree/{ℓ}/gamma)`.
//! 3. When at least one instance is a fraction tree, `λ = challenge_fp2(tree/{ℓ}/lambda)`. A
//!    level of forests alone squeezes no `λ`.
//! 4. Instance `i` enters with `c_i = v_i` for a forest and `c_i = p_i + λ·q_i` for a fraction
//!    tree, and the level's claim is `Σ_i γ^i · 2^{n_max − n_i} · c_i`, front-padded as in a
//!    batch.
//! 5. The rounds of [`sumcheck::reduce`], of degree 3, over `n_max` variables, under `tree/{ℓ}`,
//!    with their errors.
//! 6. The proof holds `Σ k_i` final evaluations, else `EvaluationCount { round: n_max, … }`:
//!    instance by instance, `u_i, w_i` for a forest and `N_0, N_1, D_0, D_1` for a fraction tree.
//! 7. They are absorbed as one TCE `Vec<Fp2>` under `tree/{ℓ}/evals`.
//! 8. With `r'_i` the last `n_i` challenges, `Σ_i γ^i · eq(r_i, r'_i) · g_i` equals the last
//!    claim, else `FinalMismatch { round: n_max }`, where `g_i = u_i·w_i` for a forest and
//!    `g_i = N_0·D_1 + N_1·D_0 + λ·D_0·D_1` for a fraction tree.
//! 9. `ρ = challenge_fp2(tree/{ℓ}/line)`.
//!
//! Steps 1, 2 and 4 to 8 are those of [`sumcheck::verify_batch`] of degree 3. It returns `γ`, `λ`
//! when it was squeezed, the point, `ρ`, and, in the same order, each instance's claims on its
//! next level, on `n_i + 1` variables at `ρ ‖ r'_i`, `ρ` the lowest: `u_i + ρ·(w_i − u_i)` for a
//! forest; `N_0 + ρ·(N_1 − N_0)` on the numerators and `D_0 + ρ·(D_1 − D_0)` on the denominators
//! for a fraction tree.
//!
//! [`tree::verify_product_level`]`(transcript, ℓ, claims, proof)` is this check for a level whose
//! instances are all forests: it squeezes no `λ`, and its errors, its transcript and the claims it
//! returns are those of [`tree::verify_level`] on the same claims.
//!
//! The caller, which holds the schedule, says which instances are present. With `H` the height of
//! the tallest forest or fraction tree, an instance of height `h` joins at level `H − h`:
//!
//! - a forest with its root claim `Σ_y v_y·eq(r, y)` on `s` variables, from its public root
//!   values `v_y`, at the root point `r` that the caller squeezes under `tree/{ℓ}/root/{c}` before
//!   the level's `γ`, the last `s` of those challenges;
//! - a fraction tree with its root `(n, d)` on no variable, from [`logup::check_roots`].
//!
//! At the levels after it joins, an instance carries the claims the level before returned. After
//! the last level, `H − 1`, they are its leaf claims: a forest's go to the stage that consumes
//! them, a fraction tree's to [`logup::check_leaves`]. A fraction tree of height 0, a single leaf,
//! joins no level, and its root claims are its leaf claims.
//!
//! A level that passed has proven nothing until its next claims are checked, so its result is
//! `#[must_use]`. Some changes of the evaluations pass the check of step 8:
//!
//! - It is symmetric in `u_i` and `w_i`, and a `u_i` whose `w_i` is zero does not enter it.
//! - It is symmetric in the children, the pairs `(N_0, D_0)` and `(N_1, D_1)`. A change `δ` of
//!   one evaluation adds `γ^i·eq(r_i, r'_i)·δ` times its cofactor to it: `D_1` for `N_0`, `D_0` for
//!   `N_1`, `N_1 + λ·D_1` for `D_0` and `N_0 + λ·D_0` for `D_1`. It passes when that cofactor is
//!   zero.
//!
//! Such changes do not pass the next claims, which the level absorbs before `ρ` and places on the
//! line through the two children: the next level rejects them, or after the last level the stage
//! that consumes a forest's leaf claims, or the leaf checks of a fraction tree.
//!
//! ## LogUp roots
//!
//! A LogUp instance proves that each of its `N ≥ 1` lookups `a_0, …, a_{N−1}` is a row of its
//! table `t_0, …, t_{T−1}`, each a pair compressed to `key + β·value`, with committed
//! multiplicities `m_j` (RFC-0008 'Relation'). It has two fraction trees (RFC-0008 'Fraction
//! tree'):
//!
//! - the lookup tree, of `2^{h_a}` leaves, `h_a = ⌈log2 N⌉`: leaf `i < N` is `(1, α − a_i)`, and
//!   the padding leaves repeat row 0, `(1, α − t_0)`, which the multiplicity of row 0 counts;
//! - the table tree, of `2^{h_t}` leaves, `h_t = ⌈log2 T⌉`: leaf `j < T` is `(−m_j, α − t_j)`, and
//!   the padding leaves repeat row 0 too, `(−m_j, α − t_0)`, with multiplicity 0 from an honest
//!   prover.
//!
//! A root's `d` is the product of its tree's leaf denominators, and when it is not zero, `n/d` is
//! the sum of its leaves' fractions. So when neither root denominator is zero,
//! `n_a·d_t + n_t·d_a = 0` says `Σ_i 1/(α − a_i) − Σ_j m_j/(α − t_j) = 0` over every leaf, the
//! identity of 'Relation'. A padding row is a copy of row 0, so the multiplicity a prover commits
//! there only counts more lookups of row 0: padding admits no tuple that is not a row. (Padding
//! with a row of key and value 0 would admit the lookup `(0, 0)`, which nothing checks a padding
//! multiplicity against.)
//!
//! [`logup::check_roots`]`(transcript, m, roots)` runs RFC-0014 'Transcript schedule' row 4 and
//! RFC-0019 'Check order' step 7.1 for the leaf stage's `m ≥ 0` LogUp instances, numbered from 0
//! in schedule order. A leaf stage of product forests alone has `m = 0`: it still squeezes `α`
//! and `β` and absorbs the roots and the values empty (RFC-0010 'Tree levels'). In this order:
//!
//! 1. `α = challenge_fp2(logup/alpha)`, then `β = challenge_fp2(logup/beta)`.
//! 2. The roots message holds `4m` values, `n_a, d_a, n_t, d_t` for each instance in order, else
//!    `RootsLength { instance, expected: 4m, found }`. A message too short names the first
//!    instance whose four values it does not hold, and one too long the last instance, which its
//!    extra values follow. With no instance, a message that holds values names none:
//!    `RootsWithoutInstance { found }`.
//! 3. It is absorbed as one TCE `Vec<Fp2>` under `logup/roots`, even when empty.
//! 4. Instance by instance, for `i = 0, …, m − 1`:
//!    1. `d_a ≠ 0`, else `LookupDenominator { instance: i }`;
//!    2. `d_t ≠ 0`, else `TableDenominator { instance: i }`;
//!    3. `n_a·d_t + n_t·d_a = 0`, else `Unbalanced { instance: i }`.
//!
//! It returns `α`, `β` and each instance's root claims, the claims its trees enter the levels
//! with, on no variable: `Ñ = n_a`, `D̃ = d_a` for the lookup tree and `Ñ = n_t`, `D̃ = d_t` for
//! the table tree. The roots are the prover's word until the levels and the leaf checks bind them
//! to the leaves, so the result is `#[must_use]`.
//!
//! ## LogUp leaf checks
//!
//! After its last level, an instance's lookup tree holds the leaf claims `Ñ(ρ_a) = p_a` and
//! `D̃(ρ_a) = q_a`, and its table tree `Ñ(ρ_t) = p_t` and `D̃(ρ_t) = q_t`, on `h_a` and `h_t`
//! variables. The lookup tree's numerators are all 1, and its denominators are
//! `α − kẽy − β·valũe`, polynomials of `2^{h_a}` entries whose padding slots hold row 0's key and
//! value. The table tree's numerators are `−m̃`, over `2^{h_t}` committed entries, and its
//! denominators `α − t̃`, with `t̃ = kẽy_t + β·valũe_t` public and its padding rows copies of row
//! 0 (RFC-0008 'Fraction tree').
//!
//! [`logup::check_leaves`]`(transcript, challenges, leaves, values, table)` runs the last step of
//! RFC-0014 'Transcript schedule' row 5 and RFC-0019 'Check order' step 7.3 for the `m ≥ 0`
//! instances in schedule order, after the last tree level, or right after the roots when no tree
//! has a level. `leaves[i]` holds instance `i`'s leaf claims and whether its table has values;
//! `table(i, ρ_t)` is `t̃(ρ_t)`, which the caller evaluates from the public table (RFC-0008
//! 'Structured evaluation'), its padding rows copies of row 0, or `None` when it cannot. In this
//! order:
//!
//! 1. The values message holds one value `v` for each instance whose table has values, in
//!    schedule order, `k` in all, else `ValuesLength { instance, expected: k, found }`. A message
//!    too short names the first instance with values whose value it lacks, and one too long the
//!    last instance. With no instance, a message that holds values names none:
//!    `ValuesWithoutInstance { found }`. An instance whose table has no values, such as a `Range`
//!    table's, sends none, and its `v` is 0.
//! 2. It is absorbed as one TCE `Vec<Fp2>` under `logup/values`, even when empty.
//! 3. Instance by instance, for `i = 0, …, m − 1`, the lookup tree first:
//!    1. `p_a = 1`, else `LeafNumerator { instance: i }`;
//!    2. `table(i, ρ_t)` gives `t̃(ρ_t)`, else `TableUnavailable { instance: i }`;
//!    3. `q_t = α − t̃(ρ_t)`, else `TableEvaluation { instance: i }`.
//!
//! It returns, for each instance, the claims left for later stages: `kẽy(ρ_a) = α − q_a − β·v` on
//! the lookup-side key polynomial, which is virtual and goes to the stage that produces the keys;
//! `valũe(ρ_a) = v` on the value polynomial, committed advice, for a table with values only; and
//! `m̃(ρ_t) = −p_t` on the multiplicity polynomial, committed advice. Nothing here checks them, so
//! the result is `#[must_use]`. No check reads `v` either: a false `v` passes the leaf checks and
//! leaves false claims on the value and key polynomials, which the stages and openings that take
//! them reject.
//!
//! ## Errors
//!
//! [`sumcheck::SumcheckError`] says whose fault an error of a sumcheck or a tree level is:
//!
//! - The proof's: `MissingRound`, `RoundLength`, `ExtraRound`, `EvaluationCount` and
//!   `FinalMismatch`, a `Reject(Sumcheck { stage, round })` in 04 §2 for a reduction stage and a
//!   `Reject(TreeLevel { level, round })` for a tree level (RFC-0033), with `round` from
//!   [`rejected_round`](sumcheck::SumcheckError::rejected_round). A failed final check is
//!   reported at round `n_max`, one past the last round's index.
//! - The verifier's: `Degree`, `NoInstances`, `EvaluationTotal` and `Relation` judge what the
//!   caller passes in, the schedule's degree and instances and its relation; `Label` would mean
//!   a label built here was not ASCII, and labels are built from decimal indices; `Transcript`
//!   is a challenge that found no value after 16 squeezes, with probability below `2^-496`
//!   (RFC-0004 'Failure modes'). All are `Inconclusive(ERR-INTERNAL)`, never a verdict on a
//!   proof.
//!
//! [`logup::LookupError`] does the same for the LogUp roots and leaf checks:
//!
//! - The proof's, as [`is_rejection`](logup::LookupError::is_rejection) says: `RootsLength`,
//!   `LookupDenominator`, `TableDenominator`, `Unbalanced`, `ValuesLength`, `LeafNumerator` and
//!   `TableEvaluation`, a `Reject(Lookup { table })` in 04 §2, whose `table` is that of the
//!   instance [`rejected_instance`](logup::LookupError::rejected_instance) gives; and
//!   `RootsWithoutInstance` and `ValuesWithoutInstance`, which name no instance and so no table
//!   for `Reject(Lookup)`, and whose reason the verifier's driver (#128) gives.
//! - The verifier's: `TableUnavailable` judges the caller's table evaluation; `Label` would mean
//!   that one of the four fixed labels was not ASCII; `Transcript` is `α` or `β` without a value
//!   after 16 squeezes. All are `Inconclusive(ERR-INTERNAL)`.
//!
//! ## Robustness and soundness
//!
//! Every input yields a value or a typed error: nothing here panics, indexes unchecked or
//! computes sizes unchecked (RFC-0019 'Robustness'). Work and memory are bounded by the proof's
//! size and the caller's instances, never by a number the proof states: the rounds stop at the
//! first missing one, so a point holds at most as many challenges as the proof has rounds; a
//! message is encoded for the transcript only after its length check, at most `8 + 16·7` bytes
//! a round, `8 + 16·Σ k_i` for a level's evaluations, `8 + 64·m` for the roots and `8 + 16·k` for
//! the values; `2^{n_max − n_i}` takes one squaring per bit. The checks of the roots and the
//! leaves invert nothing. They allocate, after the count checks, the root claims of the `m`
//! instances and the claims the leaf checks return, one copy of each leaf claim's point.
//!
//! A batch of `m` instances over `n_max` rounds of degree `d` accepts a false claim with
//! probability at most `(n_max·d + m)/|Fp2|` (RFC-0010 'Soundness of the schedule'). A false
//! claim either fails the final check or moves into the final evaluations, which the next stage,
//! or the openings, then check. A tree level is such a batch of degree 3; at a level with fraction
//! trees, `Σ_i γ^i·c_i` has total degree at most `m` in `γ` and `λ`, so `λ` adds nothing to the
//! bound. A lookup that is not a row of its table makes the roots balance for at most `N + T − 1`
//! values of `α`, the compression by `β` included, so the root check accepts a false instance with
//! probability at most `(N + T)/|Fp2|` (RFC-0008 'Relation', RFC-0014 'Soundness budget'). A zero
//! leaf denominator makes its root's `d` zero, which the root check rejects unless a level's claim
//! is false. The leaf checks compare exactly what the verifier can compute; what they cannot, the
//! key, value and multiplicity claims, they return for the stages and openings that can.
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
//! Model verification uses [`advice::opening::Commitment::bind`] with its independently derived
//! [`advice::Layout`] and validated encoding parameters. It derives the descriptor identifier
//! and rejects a different Round1 `layout_digest`. [`advice::opening::Commitment::prepare`]
//! then groups the verifier's original combination claims, with one `γ^(t+1)` per claim across
//! all its tensor/layer/tile pieces. It does not require individual piece values. Its must-use
//! evaluation state authenticates the canonical segment-merged columns against that same root.
//! Counts and source coefficients operate on compact geometry, without expanding a row map
//! or all attention images. The explicit-layout [`opening`] API below remains the low-level
//! primitive; both paths share shape, evaluation, authentication and consistency checks.
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
//! - `test-params` accepts the `test-small` parameter set, which certifies nothing
//!   ([`params::ParamSet`] has no such variant without it). Tests only: a release build with it
//!   fails to compile. The golden claim lists of `spec/vectors/claims/` name `test-small/1`, so
//!   the tests that validate them as they are run with `cargo test -p tv-core --features
//!   test-params`; without the feature the same lists are refused as `Reject(Params)`, and every
//!   other claim-list test runs on their twins under `ni-100/1`.

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

pub mod advice;
pub mod claims;
mod fiat_shamir;
pub mod gadgets;
mod interpolation;
pub mod leaf;
pub mod logup;
pub mod lowering;
pub mod model_openings;
pub mod opening;
pub mod params;
pub mod protocol;
mod range_checks;
pub mod ranges;
pub mod runtime;
pub mod schedule;
pub mod selection;
pub mod selectors;
pub mod soundness;
pub mod stages;
pub mod sumcheck;
pub mod tree;
pub mod validation;
pub mod weights;
