# EXP-0027: Batched mask opening: rank, distinguisher, soundness and the 405B count

| | |
|---|---|
| Status | `planned` |
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

Environment: recorded in the result files.

## Results

Not run yet.

## Analysis

Not run yet.

## Verdict

Not run yet.

## Follow-ups

Not run yet.
