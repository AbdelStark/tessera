# EXP-0027: Batched mask opening: rank, distinguisher, soundness and the 405B count

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0040](../../ideas/IDEA-0040-batched-mask-opening.md) |
| Evidence level | E0: exact arithmetic over F65537 and Goldilocks, exact linear algebra, a Monte Carlo count of accepted cheating proofs, the compiler's counts |
| Hardware | none for the result; CPU execution only (Apple M4 Max, macOS) |
| Pre-registered on | 2026-10-25 |

## Hypothesis

The masked batching sumcheck of IDEA-0040 has the following properties:
- **Zero knowledge.** With its pad `S` laid out like the mask block `M`, it reveals nothing beyond the views that precede it, for every `C`, every challenge draw and every registered configuration.
- **Power of the test.** Without `S`, with `M̃(r*)` revealed alone, or with `S` restricted to `C`'s support, a distinguisher wins on configurations built to expose each.
- **Soundness.** A false mask claim survives with probability at most `(N − 1)/q + 1/q + 2m/q`.
- **Size.** At 405B it replaces the mask claims' opening groups by one, saving at least 90% of their proof bytes.

## Success criterion

**The verdict.**
- **`validated`** only if all four parts pass:
  - **(a) Rank.**
    - The functional extraction matches the executable prover on every draw.
    - The leak dimension of the candidate layout L2, defined in the method, is 0 on every draw of every family, field and size.
  - **(b) Distinguisher**, over F65537.
    - The control arm OLD and the candidate arm CAND are indistinguishable on every draw of REAL, TIGHT and UNCLAIMED.
    - Each power control is distinguishable on at least 99% of its draws: NOS and SPLIT on TIGHT, L1 and NOS on UNCLAIMED.
    - At draw 0, each arm the exact test calls distinguishable shows an empirical advantage of at least 0.95, and each other arm at most 0.05 in absolute value, over 1,000 sampled views per witness.
  - **(c) Soundness**, over F65537.
    - **The count.** For S1 and S2, the composite cheater's accepted count `k` lies within `4·sqrt(P·T·(1 − P))` of `P·T` and at most `B·T + 4·sqrt(B·T)`.
    - **The terms.** `P` is the exact success probability of that cheater, `T` the trials, and `B = (N + 2m + 1)/q`.
    - **The controls.** The honest prover is accepted in 100% of its trials. NC1 and NC2 succeed in at least 99% of theirs.
  - **(d) Count.** The 405B count completes from the compiler's report. The candidate's added proof bytes are at most 10% of the per-claim baseline B1''s bytes.
- **`refuted`** if any of these holds:
  - a CAND draw is distinguishable in any configuration, which is a leak under the candidate;
  - an accepted count in (c) exceeds `B·T + 4·sqrt(B·T)`;
  - at 405B, the candidate's added proof bytes reach B1''s.
- **`inconclusive`** otherwise: a failed correctness check, a distinguishable OLD draw (the toy is then wrong), a power control below 99%, a count in (c) outside the `P` band but under the bound, or an interrupted run.

**Predictions that do not decide the verdict.** They are recorded and analysed whatever they show.
- **P-a1 (DENSE).** `rank_full = rank_L1 = 2m + 1` on at least 98% of F65537 draws and every Goldilocks draw.
- **P-a2 (SINGLE).** `rank_full = m + 1` and `rank_L1 = 1` on every draw.
- **P-a3 (SUBCUBE, `j` fixed variables).** `rank_full = 2m + 1 − j` and `rank_L1 = 2m + 1 − 2j`, on at least 98% of F65537 draws and every Goldilocks draw.
- **P-a4 (PROTO, POLY, ROWS).** `rank_L1 = rank_L2` and leak dimension 0 for L1, on at least 98% of F65537 draws and every Goldilocks draw.
- **P-a5 (EQ1).** `rank_full = m + 1` on at least 98% of F65537 draws and every Goldilocks draw.
- **P-b1.** NOS and SPLIT on REAL are indistinguishable on at least 90% of draws: `M`'s spare directions absorb them.
- **P-b2.** L1 on REAL and TIGHT is indistinguishable on every draw.

**What does not follow.** No timing, GPU or memory result.
- **The field.** The prime fields stand for `Fp2`. The rule that `S` is uniform over `Fp2`, two `Fp` blocks, is RFC-0013's and is not tested here.
- **Lemma 5.** The proximity row's free rows with `S`'s rows added are argued, not tested.
- **The scale.** The toy protocol is small.

These criteria and the method are frozen before the experiment's code.

### Amendment, 2026-10-25: worker streams

**The defect.** The method's seeding rule makes the parallel streams overlap. SplitMix64 advances its state by `0x9E3779B97F4A7C15` per draw, so the stream seeded with `seed + w·0x9E3779B97F4A7C15` is the stream of `seed` shifted by `w` draws.
- **Part (b).** Its draws, one task each, reuse one another's values shifted by one position.
- **Part (c).** Its eight trial chunks are shifted copies of one another.
- **Part (a).** Each cell's draws come from one stream, so within a cell they are independent; only different cells overlap.

**How it was found.** The first run, made under the frozen rule, showed it: (c)'s per-round counts of accepted cheating proofs were lumpy, 114 against 61 expected in one round. Its outputs are kept as `results/run-{a,b,c}.registered-seeding.json`.

**The amended rule.** Worker `w` of a part is seeded with the `(w + 1)`-th output of SplitMix64 started at the part's seed. The outputs pass SplitMix64's mixer, so the streams start at unrelated states.
- **Part (c)'s setup** (layout, `M`, `S`, the claims) stays SplitMix64 at the part's seed, and its trial chunks are workers `1, 2, …`.
- **Part (b)'s empirical streams** take their seeds from the same rule.

Nothing else changes: the hypothesis, the criteria, the configurations, the draw and trial counts. The run under this amendment is the one judged; the first run is reported beside it.

## Method

### Common definitions

- **Fields.** `q = 65537` and Goldilocks `p = 2^64 − 2^32 + 1`, with arbitrary-precision integers. Uniform elements come by rejection sampling from SplitMix64 draws.
- **Seeds.** (a) `0x45800270001`, (b) `0x45800270002`, (c) `0x45800270003`. Worker `w` of a parallel part starts its stream at `seed + w·0x9E3779B97F4A7C15 mod 2^64`.
- **The cube.** `m` variables. Position `x ∈ [0, 2^m)` has bit `i` as variable `i + 1`, so the lowest variable is bit 0. `f̃(r) = Σ_x f[x]·eq(r, x)`.
- **The batching sumcheck.** It binds variable 1 first.
  - Round `j` sends `s_j(X) = A_j + B_j·X + Q_j·X^2`, computed from the tables `C` and `z` folded by `r_1..r_{j−1}`.
  - The verifier checks `2A_j + B_j + Q_j` against the running claim, which starts at `v + ρ·Γ_S`.
  - At the end it checks `s_m(r_m) = C̃(r*)·y`, computing `C̃(r*)` itself.
  - `y` is checked against the true `z̃(r*)`: an ideal opening.
- **The revealed functionals of a vector `S`.** `Γ_S` is `C` itself. Round `j` gives `B_j` and `Q_j`, as vectors `w` with `⟨w, S⟩` equal to the coefficient computed on `z = S`. The final value gives `eq(r*, ·)`. That is `2m + 2` listed functionals, of which at most `2m + 1` are independent when `C̃(r*) ≠ 0`.

### (a) Rank experiment

**The draws.** For each family, `m ∈ {6, 8, 10}` and each field:
- 200 draws at `m = 6` and `m = 8`, 50 at `m = 10`;
- each draw has a fresh `C`, with fresh `α` and claim points, and fresh challenges `r_1..r_m`.

**The correctness check.** On every draw, a fresh uniform vector `S` run through the executable prover gives `⟨w, S⟩` for every listed functional exactly.

**The layouts.** `T_layout` is the family's layout support: used coefficient slots and the rows' entries for PROTO, POLY and ROWS, and the whole cube for the synthetic families. `T_spread` is every position whose row index, its bits from `c = m − 4` up, has Hamming weight at most 2.

**What each draw records.**
- `rank_full`, over the whole cube;
- `rank_L1`, over `supp C`;
- `rank_L2`, over `T_layout`;
- `rank_L3`, over `T_spread`;
- each layout's **leak dimension**, `rank(w|_{T_S ∪ T_layout}) − rank(w|_{T_S})`: the combinations of revealed values whose `S` part cancels while `M`'s part does not.

**Families.**

| Family | `C` |
|---|---|
| PROTO | a protocol-like layout. `m = 6`: one masking polynomial (`n = 2`) and rows of widths 1, 2, 3, 4. `m = 8`: polynomials `n = 2, 3, 4` and rows 1 to 6. `m = 10`: polynomials `n = 2` to 7 and rows 1 to 8, then 7 and 6 |
| POLY | masking polynomials alone, `n` cycling through 2 to 7 while a block fits |
| ROWS | rows alone, widths cycling through 1 to `m − 2` while a row fits |
| DENSE | every entry uniform non-zero |
| SPARSE(`s`) | `s ∈ {2, 3, 5, 2m}` random positions with uniform non-zero values |
| SUBCUBE(`j`) | `j ∈ {1, 2, m/2}` random variables fixed to random bits; dense inside |
| SINGLE | one position |
| EQ1 | `eq(p, ·)` at a uniform point |
| BOOLPT(`k`) | `eq(p, ·)` with `k ∈ {1, 2}` coordinates Boolean |

**The layout rules of PROTO, POLY and ROWS.**
- **A masking polynomial of `n` rounds** occupies an aligned block of `2^⌈log2 8(n+1)⌉` slots.
  - Slot 0 is `a_0`. Slots `8j..8j + d_j` hold `g_{j,0..d_j}`.
  - The degrees are `d_j = 3` and `d_n = 7`.
  - Its claim puts `r_j^m` on each used slot and 1 on `a_0`.
- **A row of width `c`** occupies an aligned block of `2^c` entries. Its claim is `eq(p, ·)` on the block.
- **Placement.** Blocks go in decreasing size order. Every other position is a public zero.
- **The combination.** `C = Σ_i α^i·c_i`.

### (b) Distinguisher experiment, over F65537

**The toy protocol.** Every value is computed by one executable implementation.

**Stages.** Each stage is a masked sumcheck as RFC-0013 defines it.
- **The sum.** It proves `H = Σ_x P(x)·A'(x)` over `n` variables, with `P` public and uniform, and `A'(x) = Ã(x) + Z(x)·R̃_A(x)`.
  - `Z(x) = Π_{i=2}^{n} x_i(1 − x_i)`.
  - `R_A` is uniform over all `n` variables, outside `M`.
- **The mask.** `d_j = 2` for `j < n` and `d_n = 4`. The prover commits `G = a_0 + Σ_j g_j(x_j)` with uniform coefficients.
- **The messages.**
  - The prover sends `Γ`, and the verifier draws `ρ_k`.
  - The rounds send every coefficient.
  - The prover sends `A'(r)`, and the verifier checks `p_n(r_n) = P̃(r)·A'(r) + ρ_k·G(r)`.
  - The mask claim is `G(r)`, which that check determines.
- **The witnesses.** Two `A` with equal `H`: `A_2 = A_1 + Δ`, with `Σ P·Δ = 0` and `Δ ≠ 0`.

**Rows.** They model what RFC-0032 reveals of a tree-level mask row (`docs/zk/simulator.md` §4.2).
- **Width `c = 1` ("root-like").** The view holds `Ṽ(q) + q(1 − q)·R̃(q)` and the claim `R̃(p)`.
- **Width `c ≥ 2`.** The view holds `Ṽ(b ‖ q) + Z(q)·R̃(b ‖ q)` for `b ∈ {0, 1}`, with `q ∈ F^{c−1}` and `Z(q) = Π q_i(1 − q_i)`, and the claim `R̃(p)`.
- **Draws.** The points are uniform, and the two witnesses' `V` are independent and uniform.

**Configurations.**
- **REAL.**
  - Stages with `n = 2, 3, 4`, and rows of widths 1, 2, 2, 3, 4, 5, laid out by (a)'s rules.
  - The coefficients sit in 8-slot rounds with `d_j` as above. `m = 8`, and there are 9 claims, stages first.
- **TIGHT.** Six root-like rows in a cube of 16 entries, `m = 4`.
- **UNCLAIMED.**
  - Four claimed root-like rows at `[0, 8)`.
  - Four unclaimed root-like rows at `[8, 16)`. Each of these is revealed at its masking point and at a second point, standing for an opening elsewhere, but takes no part in `C`.
  - `m = 4`, and `T_layout` is the whole cube.

**Arms.**

| Arm | The batching view |
|---|---|
| OLD | none: the earlier view alone |
| CAND | `Γ_S`, the rounds and `y`, with `S` uniform on `T_layout` |
| L1 | as CAND, with `S` uniform on `supp C` only |
| NOS | `z = M`, no `Γ_S` and no `ρ`; the rounds and `y` |
| SPLIT | as CAND, but with `M̃(r*)` and `S̃(r*)` in place of `y` |

**The draws.** 100 per configuration. Each draw has fresh public tables, points, challenges and witness pairs.

**The exact test.**
- Every value is computed as an affine form in the masking randomness, with one constant per witness: `y = A·s + b_i`.
- A draw is **distinguishable** when `b_1 − b_2 ∉ colspan(A)`. The elimination then returns `λ` with `λ^T A = 0` and `λ^T(b_1 − b_2) ≠ 0`.
- The affine forms are checked against plain numeric runs of the same code on random masks, on the first 10 sampled views.

**The empirical test**, at draw 0 of each configuration and arm.
- 1,000 numeric views per witness, with fresh masks. Every view must pass the verifier's checks.
- The statistic is `D(y) = [λ^T y = λ^T b_1]`, with `λ` the arm's own vector when it has one, else the NOS vector of the same configuration and draw, else a uniform vector.
- Advantage is `Pr[D = 1 | w_1] − Pr[D = 1 | w_2]`.

### (c) Soundness, over F65537

**Layouts.** S1 is REAL's (`N = 9` claims, `m = 8`), over `T = 10^6` trials. S2 is TIGHT's (`N = 6`, `m = 4`), over `T = 2·10^6` trials.

**The composite cheater** has three stages:
1. **The claims.** It errs by `e_i`, the coefficients of `E(α) = κ·Π_{k=1}^{N−1}(α − a_k)`, with distinct uniform roots and uniform non-zero `κ`.
2. **The pad's sum.** It sends `Γ_S' = ⟨C, S⟩ + E(α)/ρ_0`, for a guess `ρ_0` drawn before `ρ`.
3. **The rounds.** While its claim is false, it sends `s_j + δ_j·(X − a_j)(X − b_j)`, with distinct uniform `a_j, b_j` such that `a_j b_j + (1 − a_j)(1 − b_j) ≠ 0`, and `δ_j` set so the round check passes. It is honest once a challenge lands on a root. `y` is always honest.

**Its exact success probability** is `P = 1 − (1 − (N − 1)/q)·(1 − 1/q)·(1 − 2/q)^m`. Each accepted trial records which event made it pass.

**Controls**, on S1 with `10^4` trials each:
- the honest prover;
- **NC1**, with `Γ_S` absorbed after `ρ`, so the prover cancels `E(α)` exactly;
- **NC2**, with `α` drawn before the claims, so the prover picks errors with `E(α) = 0`.

### (d) The 405B count

**The schedule.** A scratch crate in this directory depends on `crates/tessera-compiler` by path. It lowers Llama-3.1-405B in `servable` with two shift classes and `B_max = 32`, as `crates/tessera-bench/tests/commitment_report.rs` does, and prints the report's stage count, tables, product-tree leaves and stage tables. `count.py` checks them against `docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json`.

**The batch.** `T = S = 16,384`, `k = 2^22`, 126 layers padded to 128, 128 heads.

**LogUp instances.** One per table with lookups.
- **The lookups.** `N_i` is `lookups_per_token·T`, plus the padded layers' `SoftmaxRow` reads of the protocol calculator (`crates/tessera-protocol/src/soundness/llama.rs`). `Exp` has `2^42`.
- **The heights.** `h_L = max(2, ⌈log2 N_i⌉)` and `h_T = max(2, ⌈log2 rows⌉)`.
- **A sensitivity variant** adds 1 to every `h_L`: RFC-0045's aligned blocks can raise a height by one.

**Forests.**
- **Fraction forests.** Two per instance, each with a padding tree, so `s = 1`.
- **The `SoftmaxRow` forest.** `s = 28`, `h = 14`.
- **The levels.** `H` is the tallest height. Batch level `ℓ = 0..H − 1` has `n(ℓ) = max(s_i + t_i)` rounds over the instances present there.

**Masked sumchecks.**
- the reduction stages, whose rounds are the stage tables' largest instance at this batch and at least 22, of degree at most 7;
- the layer sumcheck: 22 rounds of degree 4 (RFC-0068);
- the identity sumcheck: 2 rounds of degree 4;
- the `H` tree levels: `n(ℓ)` rounds of degree 3, the raised round of degree 7.

**Mask claims.**
- one per masked sumcheck;
- two per fraction-forest level `t = 0..h − 1`;
- one per `SoftmaxRow` level `t = 1..h − 1`.

**Groups before.** B1 is one group per claim. B1' shares a group among a level's claims of equal `n_i`, so equal column sub-point, and gives each `G_k` its own.

**`M`'s layout.**
- the coefficient slots `Σ(1 + 8n_k)`;
- the used coefficients `Σ(1 + Σ(d_j + 1))`;
- the tree-level entries `2^{min(s + t, 22)}` per claim;
- the rows, unpacked (one `Fp2` row per tree-level claim) and packed;
- `m = 22 + ⌈log2 rows⌉`.

**Bytes**, under `live-64` (`τ_A = 155`, `P_A = 160`) and `ni-100` (193, 200).
- **Per group.**
  - a combined row of `2^22 × 16` bytes;
  - `M_g` at `P_A × 16`;
  - `q_g` at 16;
  - a blinding pair: two `Fp` rows, `2·τ_A × 8` bytes of column symbols.
- **The candidate.**
  - **Its group.** One group.
  - **Its messages.** The batching's `(2m + 2) × 16` bytes.
  - **Its rows.** `S`'s rows, `2·τ_A × 8` bytes per `Fp2` row.

**What else it reports.**
- **The prover's cost.** `S`'s committed elements, as a share of RFC-0011's 953,580 non-zero rows.
- **The soundness terms added.** `(N_claims − 1 + 2m + 1)/q`, with `q = p^2`.

## Reproduction

```sh
python3 -I research/experiments/EXP-0027-batched-mask-opening/batch.py all research/experiments/EXP-0027-batched-mask-opening/results --workers 8
cargo run --release --locked --manifest-path research/experiments/EXP-0027-batched-mask-opening/schedule/Cargo.toml -- research/experiments/EXP-0027-batched-mask-opening/results/schedule-405b.json
python3 -I research/experiments/EXP-0027-batched-mask-opening/count.py research/experiments/EXP-0027-batched-mask-opening/results/schedule-405b.json research/experiments/EXP-0027-batched-mask-opening/results/count-405b.json
```

Environment: Apple M4 Max (16 cores), macOS 26.6.2 arm64, Python 3.15.0, rustc and cargo 1.95.0, 2026-10-25. Each result file records its own environment line and UTC times.

## Results

**Provenance.**
- **The plan.** It was frozen before any experiment code: `results/README.preregistered.txt`, SHA-256 `79c531f7…` at 2026-10-25T19:19:41Z (`results/preregistration.txt`).
- **The amendment.** It was frozen before the rerun: `results/README.amendment-preregistered.txt`, SHA-256 `cefd62a0…` at 19:33:21Z (`results/preregistration-amendment.txt`).
- **No commit.** Neither file is in a commit: this unit makes no git writes. `experiment.md` asks for a commit before code, which the digests stand in for until the maintainer commits.
- **Development runs.** Smoke runs on other seeds and small sizes preceded the registered runs.
  - One development run checked that REAL's NOS leak was not a harness artifact, on variant configurations.
  - Only the registered runs are reported.
- **The judged run.** It is the amended run, 19:33:28Z to 19:34:38Z, with eight worker processes; E0, so the wall time is not a measurement. The first run under the frozen seeding rule (19:30:33Z) is kept as `results/run-{a,b,c}.registered-seeding.json`.

### (a) Rank, `results/run-a.json`

**Correctness.** The functional extraction matched the executable prover on all 13,500 of 13,500 draws, in both fields at `m = 6, 8, 10`. L2's leak dimension is 0 on every draw.

The table gives each family's ranks against `m`, over 900 draws each, both fields and all three sizes:

| Family | `rank_full` | `rank_L1` (`S` on `supp C`) | `rank_L3` (spread rows) | leak L1 | leak L3 |
|---|---|---|---|---|---|
| PROTO, POLY, ROWS, DENSE | `2m` on every draw | `2m` | `2m` | 0 | 0 |
| SUBCUBE(`j`), `j ∈ {1, 2, m/2}` | `2m − j` on every draw | `2m − 2j` on every draw | 3 to `2m − j` | `j` | 0 to 12 |
| SINGLE | `m + 1` | 1 | 3 to `m + 1` | `m` | 0 to 8 |
| EQ1 | `m + 1` | `m + 1` | `m + 1` | 0 | 0 |
| BOOLPT(`k`) | `m + 1` | `m + 1 − k`, one F65537 draw in 200 one lower | `m + 1 − k` to `m + 1` | `k` | 0 to 2 |
| SPARSE(`s`), `s ∈ {2, 3, 5}` | `2m − 9` to `2m` | at most `s` | varies | 5 to 17 | 0 to 14 |
| SPARSE(`2m`) | `2m − 1` to `2m` | `2m − 4` to `2m` | `2m − 2` to `2m` | 0 to 4 | 0 to 2 |

**The predictions.**
- **P-a1 fails by one.** DENSE reaches `2m`, not `2m + 1`, on every draw.
  - The last round's three coefficients depend on `S` only through the two values of its folded table, and the round check fixes one of them.
  - So `2m` is the most any `C` can give: `Γ_S`, two per round for `m − 1` rounds, and one in the last.
- **P-a3 fails by the same one.** SUBCUBE gives exactly `2m − j` and `2m − 2j` on every draw.
- **P-a2, P-a4 and P-a5 hold** on every draw: SINGLE `m + 1` and 1; PROTO, POLY and ROWS with `rank_L1 = rank_L2` and leak 0; EQ1 `m + 1`.

### (b) Distinguisher, `results/run-b.json`

Distinguishable draws out of 100, by exact linear algebra:

| Configuration (`m`; view entries, earlier → with the batching) | OLD | CAND | L1 | NOS | SPLIT |
|---|---|---|---|---|---|
| REAL (8; 59 → 85) | 0 | 0 | 0 | 100 | 0 |
| TIGHT (4; 12 → 26) | 0 | 0 | 0 | 100 | 100 |
| UNCLAIMED (4; 16 → 30) | 0 | 0 | 100 | 100 | — |

**The draws.**
- **The model.** On every draw and arm, both witnesses give the same matrix `A`.
- **The empirical test.** At draw 0, over 1,000 numeric views per witness:
  - each distinguishable arm has advantage 1.000, and each other arm 0.000;
  - all 28,000 views pass the verifier's checks;
  - the affine model matches the numeric runs on every entry checked.
- **The predictions.** P-b1 fails for NOS, which is distinguishable on all 100 REAL draws, and holds for SPLIT (0 of 100). P-b2 holds.

### (c) Soundness, `results/run-c.json`

| Layout | `N` | `m` | Trials | Accepted | `P·T ± 4σ` | `B·T + 4·sqrt(B·T)` | at `α` (predicted) | at `ρ` (predicted) | per round (predicted each) |
|---|---|---|---|---|---|---|---|---|---|
| S1 (REAL) | 9 | 8 | 10^6 | 366 | 381.4 ± 78.1 | 476.4 | 114 (122.1) | 10 (15.3) | 37, 35, 26, 27, 31, 27, 31, 28 (30.5) |
| S2 (TIGHT) | 6 | 4 | 2·10^6 | 421 | 427.2 ± 82.7 | 543.3 | 142 (152.6) | 31 (30.5) | 65, 62, 60, 61 (61.0) |

**The controls.**
- the honest prover: 10,000 of 10,000 accepted;
- NC1 (`Γ_S` after `ρ`): 9,999 of 10,000 accepted;
- NC2 (`α` before the claims): 10,000 of 10,000 accepted.

**The first run, under the frozen seeding.** It gave 317 (S1) and 376 (S2), within the band too. Its per-round counts were lumpy: S2's rounds read 34, 16, 114, 16 against 61 expected each. That is the stream overlap the amendment removes. Its parts (a) and (b) gave the same verdict-deciding outcomes.

### (d) The 405B count, `results/schedule-405b.json`, `results/count-405b.json`

**Checks.** The compiler's tables, stage count and stage tables equal the committed `docs/perf/v0.2` report, and the calculator's tables equal the compiler's. The advice has 953,580 non-zero rows of 2,328,576 allocated, ν = 44.

| Quantity | Value |
|---|---|
| LogUp instances, fraction forests, `SoftmaxRow` forest | 25, 50, `s = 28`, `h = 14`; tallest tree 42 (`Exp`'s lookup tree) |
| Masked sumchecks | 46: reduction stages of 42 and 35 rounds, the layer sumcheck (22), the identity (2), 42 tree levels of `ℓ + 1` rounds; 1,004 rounds |
| Masking coefficients | 4,562 `Fp2` (8,078 slots in 8-slot rounds) |
| Tree-level mask claims | 2,621, of which 697 at full width `2^22`; 3.26·10^9 `Fp2` entries |
| Mask claims | 2,667 |
| Opening groups | B1 2,667; B1' 612; candidate 1 |
| Blinding rows (`Fp`) | 5,334; 1,224; 2 |
| Proof bytes, `live-64` | B1 166.7 GiB; B1' 38.25 GiB; candidate 65.85 MiB packed (70.21 MiB unpacked): 99.83% saved against B1' |
| Proof bytes, `ni-100` | B1' 38.25 GiB; candidate 66.30 MiB packed |
| The candidate's parts | one group 64.0 MiB; batching messages 1,056 B (`m = 32`); `S`'s opened symbols 1.93 MB (`live-64`), 2.41 MB (`ni-100`) |
| `M` and `S` | packed: 779 `Fp2` rows each, `S` 1,558 `Fp` rows, 0.163% of the advice's non-zero rows, 6.53·10^9 committed elements; unpacked: 2,622 `Fp2` rows, `m = 34`, 0.550% |
| `C`'s support | 3.26·10^9 `Fp2` entries; the verifier's `C̃(r*)` about 8.8·10^4 terms; the batching prover at most 2.6·10^10 `Fp2` multiplications |
| Soundness added | `(2,666 + 64 + 1)/q ≈ 2^-116.6`, with `q = p^2` |
| Sensitivity, every lookup tree one taller | 2,718 claims, B1' 558, candidate 65.97 MiB |

## Analysis

**Against the criterion.** Every part passes on the judged run.

| Part | Requirement | Result |
|---|---|---|
| (a) | correctness | 100% |
| (a) | L2 leak dimension 0 | every draw |
| (b) | OLD and CAND indistinguishable | 0 of 300 draws distinguishable each |
| (b) | the four power controls | 100% of their draws |
| (b) | empirical advantages | 1.000 and 0.000 |
| (c) | both counts within `4σ` of the exact prediction and under the bound | −0.8σ and −0.3σ |
| (c) | the three controls | as required |
| (d) | the candidate's bytes against B1''s | 0.17% |

The first run satisfies the same criteria.

**What the experiment shows.**

1. **Lemma B holds on an executable protocol.**
   - **The test.** It runs masked stages, tree-level-like mask rows and the batching.
   - **The candidate never leaks.** With the pad laid out like `M`, the batching view never separates two witnesses whose earlier views are identically distributed. That is exact for every draw, not statistical.
2. **The rank question has an answer, and it is not the one the issue posed.**
   - **The rank.** The map from `S` to the batching's revealed values reaches `2m`, the most its shape allows, for every protocol-like `C` on every draw.
   - **Where it drops.**
     - On sub-cube supports: `j` on `M`'s side, `2j` when `S` is restricted to `C`'s support.
     - On single `eq` claims: `m + 1`.
     - On single entries and very sparse `C`.
   - **What a drop means.** A drop on `M`'s side is a public relation and costs nothing. A leak needs `S`'s side to drop further than `M`'s: `S` must miss a position where `M` is not public.
   - **The rule that follows.** Full rank is not the condition. Coverage is: `S` must be uniform wherever `M`'s layout holds a mask value.
3. **What the protocol produces.**
   - **At 405B.** `C` combines 2,667 claims on disjoint blocks, which is the protocol-like case.
   - **The layout versus `C`'s support.** They differ only where a challenge or a claim coordinate is 0 or 1:
     - `r_j = 0` empties a round's slots `e ≥ 1`;
     - a Boolean coordinate empties half a row.
   - **The probability.** At most `(1,004 + 2·Σ_t c_t)/q`: a round's challenge at 0, or a claim coordinate at 0 or 1.
   - **The rule removes it.** Laying `S` out by `M`'s layout, which is public and fixed before any challenge, removes even that event.
   - **The cost of laying `S` by `C`.** UNCLAIMED shows it: a non-public block in the cube that no claim reads is exposed by L1 on every draw. That rules out putting other masks, the column masks or tensor masks, in `M`'s cube unless their claims join `C` and `S` covers them.
4. **`S` is necessary, and so is the rule against `M̃(r*)` alone.**
   - **NOS leaks on REAL**, on every draw, against prediction P-b1.
     - The leak uses the root-like row, a two-entry row revealed at two points with no spare direction. Every LogUp forest's root level has such rows (RFC-0032).
     - A development run without those rows found no leak.
     - So the bare batching sumcheck is not zero-knowledge for the protocol's `M`.
   - **`M̃(r*)` alone leaks on TIGHT but not on REAL.** One more functional, spread over every position, is absorbed as soon as `eq(r*)` reaches one spare direction anywhere in `M`. 405B's `M` has many: any tree-level row of more than 3 entries.
     - **The rule stays, for robustness.** It keeps Lemma B free of any rank condition, and it covers layouts with no spare direction at all, as TIGHT shows.
     - **A corrected reason.** An earlier wording of this paragraph gave the rule a reason that does not follow; the independent reading below found it.
5. **Soundness.**
   - **The counts.** The composite cheater's acceptance matches its exact probability and stays under `(N + 2m + 1)/q`.
   - **The breakdown.** The per-event counts match the three terms.
   - **The order rules are necessary.** NC1 and NC2 show both are: `Γ_S` before `ρ`, and the claims before `α`.
6. **The count corrects the issue's figure.**
   - **The claims.** 2,667 mask claims, against #489's projection of about 200 groups. The tree-level claims are 2,621 by themselves.
   - **Per-claim openings** would add 166.7 GiB, and 38.25 GiB even with shared points.
   - **The candidate** adds 65.85 MiB, an `S` of 0.16% of the advice rows, and 32 rounds.
7. **A cheaper pad looks possible.** L3, `S` on the rows whose row index has Hamming weight at most 2, has leak 0 for every protocol-like `C` on every draw.
   - **The saving.** At 405B's 10 row bits that is 56 `Fp2` rows, against 779.
   - **Why it is not a result yet.** L3 leaks for sparse, sub-cube and single-entry `C`. It needs a rank lemma and the protocol's `C` to satisfy it, so it is a follow-up.

**Threats to validity.**
- **Scale.** `m ≤ 10` in (a) and `m ≤ 8` in (b), against 32 at 405B. Lemma B's argument does not depend on the scale. The rank observations do.
- **The field.**
  - **What it stands for.** The prime fields stand for `Fp2`, so masks that are uniform over `Fp` only (Lemma 2.2) are not tested.
  - **The requirement.** `S` must be two `Fp` blocks, uniform over `Fp2`, as RFC-0013 requires of every mask.
- **The earlier view.** The toy reveals the stages' masked final values and the rows' masked points. It does not reveal the openings' combined rows, the proximity row or the columns. `S`'s presence in them (Lemmas 5 and 6) is argued in IDEA-0040, not tested.
- **The challenges.** They are drawn independently. Exactness for every fixed challenge sequence implies it for any adaptive choice of challenges (Method).
- **One cheater.** (c) measures one cheating strategy. The bound is analytic. The measurement shows this strategy has no acceptance path outside the three terms and meets the bound, not that no strategy beats it.
- **(d)'s rules.**
  - **The heights.** `⌈log2 N⌉`, the protocol crate's rule. RFC-0045's aligned blocks could raise one: the sensitivity run moves B1' from 612 to 558 and the claims from 2,667 to 2,718.
  - **The degrees.** The reduction stages' degrees are RFC-0013's bound of 7.
  - **The levels.** The tree levels are batched as RFC-0010 says, one masked sumcheck per batch level.
- **Seeding.** The defect in the frozen seeding rule is disclosed and amended. The first run reaches the same verdict.

### Independent reading, 2026-10-25

A fresh agent read Lemma B and the mechanism cold against RFC-0011, RFC-0013, RFC-0032 and `docs/zk/simulator.md`. It was asked for attacks as ADV-V and ADV-P.

**No break.**
- **Lemma B's proof is sound.** That includes the adaptive Fiat–Shamir challenges, both coin-toss modes, programming of the random oracle and the PRG hybrid.
- **The composition is sound.** It holds with `y`'s group, with the proximity row (exact under R1; `p^-2` under the tensor generator), and with the columns.
- **The ordering is sound.** `α` comes after every claim, and `Γ_S` after `α` and before `ρ`.

**Two specification gaps, for the RFC.**
- **`S` depends on `M` after the batching.** `S_T = ρ⁻¹(z_T − M_T)`.
  - **The consequence.** A later message that reveals a functional of `S` alone or of `M` alone, inside the span of the batching's functionals, reveals that functional of `M`.
  - **Today's later uses are covered.** `y`'s group by `Q_y`, the proximity row by `Q_ξ`, the columns by the encoding masks.
  - **The rule.** `M` and `S` enter nothing revealed except as `z`, or under a fresh designated block. Lemma 5 must not count `S`'s rows as free beyond the batching's span. The activation mask rows still give its second free row.
- **How `G_k(r_k)` is carried is unspecified.** RFC-0013 says "opened", RFC-0068 "checks … plus `ρ_zk·G(ρ')`", and Lemma 4 "fixed by the final check". No label carries it.
  - **Soundness.** Either way `v_k` is fixed before `α`, so the batching stays sound.
  - **If it is derived.** `ρ_k = 0` needs a rule, and every masked final check becomes vacuous, so the stages' `Reject(Sumcheck)` and `Reject(TreeLevel)` move to the batching.
  - **The fix.** Send `G_k(r_k)` under `…/evals`. It adds no information.

**Clarifications.**
- **The soundness sum leaves out two terms:**
  - the `γ` term of `y`'s group, `1/q`;
  - `S`'s rows in the proximity bound. `ϑ` stays 22: 2,328,576 + 2^11 rows < 2^22.
- **The calculator** should list the batching as a stage, as RFC-0068 does.
- **Its order** must be pinned between RFC-0014's rows 6 and 7.
- **`Fp2`.** `S` over `Fp2` is necessary for the proof as written, and sufficient with `S^(0)`, `S^(1)` independent. `y`'s pieces take the coefficients `(1, X, ρ, ρX)`.
- **`M`'s layout.**
  - The masking polynomials' functionals use exactly `d_j + 1` slots per round. All 8 would raise that round's error to `7/q`.
  - Committed-tensor masks and reduction-stage whole-row masks stay out of `M` unless `S` takes their whole rows.
- **R5 and R10.**
  - R5 needs a coverage clause, since the batching's functionals are dependent through public relations.
  - R10's check (#488) must not count `r*` as a revealed point of each `R_t`.
- **The simulator's §8** projects about 200 masked sumchecks and 2,000 mask rows. It should be recounted from these figures: 46 sumchecks and 2,621 tree-level rows.

None of this changes the verdict, which the pre-registered criterion decides. It sets what the RFC must state.

## Verdict

`validated`. IDEA-0040 becomes `validated`. The next step is an RFC amending RFC-0013, RFC-0011, RFC-0032 and RFC-0014, with `docs/zk/simulator.md` R11 and Lemma 5, under #489.

## Follow-ups

**What the RFC must state.**
- `M`'s cube and layout:
  - only claimed masks;
  - public zeros elsewhere;
  - the column variables lowest;
  - exactly `d_j + 1` slots per round in `C`.
- `S`:
  - uniform over `Fp2` wherever `M`'s layout holds a mask value;
  - committed in Round 1, from the per-proof PRG.
- The order: `α` after the last mask claim, `Γ_S` before `ρ`.
- The labels, and the rows of RFC-0014's schedule.
- The final value `y = (M̃ + ρS̃)(r*)` as one claim in one more group. `M̃(r*)` and `S̃(r*)` are never sent apart.
- The verifier's `C̃(r*)`.
- The calculator's terms: `(N − 1)/q`, `2m/q`, `1/q` for `ρ`, `1/q` for `y`'s group, and `S`'s rows in the proximity term; the batching listed as a stage.
- The cost: one group, the `S` rows, `m` rounds.
- Lemma B, and the simulator's added step and its `1/q` term.
- `G_k(r_k)` sent under each masked sumcheck's `…/evals`, so the stages' final checks keep their rejections.
- The order between RFC-0014's rows 6 and 7.
- The coverage rule: `M` and `S` enter nothing revealed except as `z` or under a fresh designated block. That rule, with a coverage clause in R5, and R10 not counting `r*`.
- `S` over `Fp2`, with `y`'s pieces at `(1, X, ρ, ρX)`.
- simulator.md §4.4, §7 and §8 recounted from this experiment's figures.

**Next experiments and tests.**
- **Lemma 5 with `S`'s rows.** The proximity row with `S`'s rows added, under R1 and the tensor generator: argued here, owed a proof.
- **#162's distinguisher suite.** It should carry this experiment's negative controls:
  - NOS;
  - `M̃(r*)` alone;
  - `S` on `C`'s support with an unclaimed block;
  - `S` drawn over `Fp`.
- **A cheaper pad.** L3 with a rank lemma would cut `S` from 1,558 `Fp` rows to about 112.
- **Absorbing the other mask-only groups.** RFC-0068's column masks and the two reduction stages' fold groups could join the batching. That is three groups, 192 MiB, saved for their rows in `S`.
- **No H100 run is needed** to settle any of this. The batching prover's cost, at most 2.6·10^10 `Fp2` multiplications, is a v0.5 measurement.
