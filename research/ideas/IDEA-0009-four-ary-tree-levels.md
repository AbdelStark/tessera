# IDEA-0009: Four-ary product-tree levels: half the levels for the same multiplications

| | |
|---|---|
| Status | `triaged` (2026-10-16); first action is a spec question to the RFC-0018 owner, not an experiment |
| Origin | new, from reading [RFC-0018 'Pass structure'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure) ("Leaf levels") against [tools/cost_model.py](../tools/cost_model.py); touches RISK-10, OQ-0018-1 and RFC-0018 Alternatives Considered 3; composes with the [SUR-0003](../surveys/SUR-0003-passes.md#candidate-ideas) seed "Tile-folded leaf levels (Hobbit)" |
| Lever | `stream` (the number of tree levels that exceed memory and are streamed, hence the passes); glue (tree rounds and per-round all-reduces, half as many); fold bytes per leaf behind `c_π`; proof size of the tree rounds |
| Spec surfaces | RFC-0010 'Relations' (`ProductTreeLevel`, degree 3 → a `ProductTreeLevel4` of degree 5) and 'Tree levels' (steps 2–4: four evaluations per instance and two line challenges); RFC-0014 'Transcript schedule' step 5 (a second `tree/{ℓ}/line` label); optionally RFC-0008 'Fraction tree' (four-child nodes, degree 5). The `tree/{ℓ}/evals` message layout and the labels are TCE message schemas, tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)): a superseding RFC and a protocol version, cheap before the `tessera-protocol/1` label freeze at v0.3 |
| Priority | 0.37 under the `counted` profile (−4.9% × 0.3 / 4 under the cost model's leaf-only reading of RFC-0018, with the `c_π` input from EXP-0006's component split; 1.5 if the literal every-level reading is confirmed), re-scored 2026-10-16; 0.43 and 2.3 under the corrected profile at 12 ps |

## Mechanism

**Relation.** A level of the `SoftmaxRow` forest combines four children instead of two. RFC-0010 'Tree levels' puts the pair bit lowest; this puts the two child bits lowest:

```
V_ℓ[x] = Π_{b ∈ {0,1}^2} V_{ℓ+1}[b ‖ x]
ProductTreeLevel4:  Ṽ_ℓ(r) = Σ_x eq(r, x) · Ṽ_{ℓ+1}(00 ‖ x) · Ṽ_{ℓ+1}(01 ‖ x) · Ṽ_{ℓ+1}(10 ‖ x) · Ṽ_{ℓ+1}(11 ‖ x)
```

The summand is a product of five multilinears, degree 5 per variable: inside RFC-0014's `d ≤ 7` row of the soundness budget and the engine's `MAX_DEGREE = 7` (`crates/tessera-poly/src/round.rs`, verified 2026-10-16). The sumcheck still binds one variable per round, so a level over `2^n` nodes takes `n` rounds, and the folded state halves per round exactly as today.

**Level protocol** (RFC-0010 'Tree levels', steps 2–4 replaced; step 1 and the root claim unchanged):

2. A sumcheck of degree 5 over `n_max` variables proves `Σ_x eq(r_i, x)·Π_b Ṽ_{t_i+1}(b ‖ x)` for every instance `i`, messages under `tree/{ℓ}/round/{j − 1}` and `tree/{ℓ}/r/{j − 1}` as today (five values per round instead of three).
3. The prover sends `u_i^{00}, u_i^{01}, u_i^{10}, u_i^{11} = Ṽ_{t_i+1}(b ‖ r'_i)` per instance, one TCE `Vec<Fp2>` under `tree/{ℓ}/evals`. The verifier checks `Σ_i γ^i · eq(r_i, r'_i) · Π_b u_i^b` against the last claim.
4. Two line challenges `ρ_1, ρ_2` are squeezed (`tree/{ℓ}/line/0`, `tree/{ℓ}/line/1`); each instance continues with `Ṽ_{t_i+1}(ρ_1, ρ_2 ‖ r'_i) = Σ_b eq((ρ_1, ρ_2), b) · u_i^b`, the restriction of a multilinear in two variables to a point, on `n_i + 2` variables.

**Heights.** A tree of height `h` has `⌊h/2⌋` four-ary levels; an odd `h` keeps one binary level at the root (the smallest level, where it costs nothing that matters). At 16K context the `SoftmaxRow` tree has `h = 14`: 7 four-ary levels instead of 14 binary ones. Forest batching (RFC-0010 'Batching within a stage', trees of different heights joining where heights match) is unchanged with height counted in four-ary levels; a forest whose height parity differs joins one level later with a binary top.

**Prover.** `crates/tessera-sumcheck/src/forest.rs` builds a level as the two halves `V_{t+1}[2z]`, `V_{t+1}[2z + 1]` (`LEVEL_DEGREE = 3`, verified); the four-ary builder builds it as four quarters and hands `prove_batch` five tables (eq and the four quarters) at degree 5. `kernel.rs` zips extension-field tables in registers up to `ZIP_EXT_NODES = 3` nodes and uses the `Scratch` row loop beyond (verified): a degree-5 level with 6 nodes takes the row loop, so the kernel needs no new path, only a measurement (kill criterion (b)).

**Optional extension to the `Exp` fraction tree** (RFC-0008 'Fraction tree', internal nodes `(n, d) = (n_L d_R + n_R d_L, d_L d_R)`): a four-child node is `d = d_1 d_2 d_3 d_4`, `n = Σ_i n_i Π_{j ≠ i} d_j`, degree 5 in each of the two sums; at the leaf level, where lookup numerators are 1, `n = (d_1 + d_2) d_3 d_4 + (d_3 + d_4) d_1 d_2`. Both trees stream in the same regeneration passes over the attention tiles (RFC-0018 pass P2 covers "lookup keys and values and softmax keys"), so the pass count only falls if both go four-ary.

**Where the passes go.** RFC-0018 'Leaf levels': "A level that does not fit is proven by streaming", `c = ⌈log2(N_level · 32 B / M)⌉` streamed rounds per such level. A binary tree whose leaf level has `c` streamed rounds has levels of `N, N/2, N/4, …` nodes, so the levels above the leaves need `c − 1, c − 2, …, 1` streamed rounds each until one fits. A four-ary tree has levels of `N, N/4, N/16, …`: `c, c − 2, c − 4, …` rounds. Half as many non-fitting levels, about half as many passes.

## Why it could work

- **Compute-neutral for the product tree** (E0, own count with the kernel's rule of `k − 1` multiplications per node for `k` tables at `d + 1` nodes; primary: `crates/tessera-sumcheck/src/kernel.rs`). Round 1 of a binary level over `N` leaves: 3 tables, 4 nodes, 2 multiplications per node, `N/2` pairs: `8/2 = 4` `Fp` multiplications per leaf; the level above: `8/4 = 2`; two binary levels: 6 per leaf. One four-ary level: 5 tables, 6 nodes, 4 multiplications per node, `N/4` nodes: `24/4 = 6` per leaf. Equal.
- **Fewer folds** (E0, own count): a table of `m` entries costs about `m` entry-folds over all its rounds (`m/2 + m/4 + …`). Two binary levels: `3 · N/2 + 3 · N/4 = 2.25 N` entry-folds; one four-ary level: `5 · N/4 = 1.25 N` (−44%). Over a whole tree the same ratio holds (`3N` → `1.67N`). [SUR-0003](../surveys/SUR-0003-passes.md) finding 6 (zkSpeed's profile, secondary) says per-pass sumcheck cost is bytes moved, not multiplications: the fold count is the part of this that pays on an H100.
- **Fewer rounds** (E0): rounds, round messages and per-round NCCL all-reduces (RFC-0018 'Partitioning') halve for the tree levels: `2 log N` rounds of two binary levels become `log N`. Message values per pair of levels: `2 · 3 log N = 6 log N` → `5 log N`, plus four evaluations instead of `2 × 2`. SUR-0003 finding 5 puts the all-reduce floor at 20–30 µs per round (E0, secondary); fewer rounds is the only lever on it that is not a protocol change elsewhere.
- **Passes** (E0, own arithmetic from RFC-0018's formula, primary; the reading of the RFC is inferred, see "Expected gain"): at 16K, `c = 9` (`cost_model.py`, E0), the levels `h − 1 … h − 8` hold `1.1×10^12 × 32 B = 35 TB` down to `0.27 TB` and do not fit 0.2 TB either, so the literal count is `9 + 8 + … + 1 = 45` streamed passes for a binary tree and `9 + 7 + 5 + 3 + 1 = 25` for a four-ary one; at 128K (`c = 12`): 78 against `12 + 10 + 8 + 6 + 4 + 2 = 42`; at 4K (`c = 7`): 28 against 16; at 1K (`c = 5`): 15 against 9. `cost_model.py` and RISK-10 count `c` passes, the leaf level only.
- **Fraction tree, if it follows** (E0, own count, same rule, the `eq` factor counted as one multiplication per sum): two binary levels of the `(n, d)` tree cost about `6 + 5 = 11` `Fp` multiplications per leaf in round 1 (leaf level with numerators 1: 3 per node × 4 nodes over `N/2`; next level: 5 per node × 4 nodes over `N/4`); one four-ary level with numerators 1: 9 per node × 6 nodes over `N/4` = 13.5 per leaf (+23%). The generation pass counted about 8 → 9 (+12%, E0); the two counts differ in how the shared products `d_1 d_2`, `d_3 d_4` are reused, which the E1 prototype settles. Either way the increase lands inside the lookup term `c_ℓ`, of which tree levels are a fraction, and the fraction tree's fold count falls as the product tree's does.
- **Prior art on arity** (secondary; SUR-0003's [Jolt26] and [PP25]): the streaming provers in the corpus bind one variable per round and treat the arity of a product layer as a free design parameter; none reports a measurement for a `2^w`-ary product layer, so the compute neutrality above is our own count until the experiment.

## Expected gain

The model's stream term is `c(S) · p_s · A · S / native(S)` with `c(S)` the leaf level's streamed rounds. The idea's pass count is not expressible with `--set` directly; since the term is linear in the pass count, a scenario with `P` passes is `--set stream_pass_ps_per_score = 7 · P / c(S)`, one command per context (`c(S) = 5, 7, 9, 12` at 1K, 4K, 16K, 128K from `cost_model.py --json`, E0).

**Reading 1, literal RFC-0018 ("A level that does not fit is proven by streaming": every non-fitting level streamed; inferred from the RFC text, a question for the RFC-0018 owner).** Binary baseline `P = 15, 28, 45, 78`; four-ary `P = 9, 16, 25, 42`:

```sh
python3 -I research/tools/cost_model.py --contexts 1024   --set stream_pass_ps_per_score=21     # binary, literal
python3 -I research/tools/cost_model.py --contexts 1024   --set stream_pass_ps_per_score=12.6   # four-ary
python3 -I research/tools/cost_model.py --contexts 4096   --set stream_pass_ps_per_score=28
python3 -I research/tools/cost_model.py --contexts 4096   --set stream_pass_ps_per_score=16
python3 -I research/tools/cost_model.py --contexts 16384  --set stream_pass_ps_per_score=35
python3 -I research/tools/cost_model.py --contexts 16384  --set stream_pass_ps_per_score=19.444444
python3 -I research/tools/cost_model.py --contexts 131072 --set stream_pass_ps_per_score=45.5
python3 -I research/tools/cost_model.py --contexts 131072 --set stream_pass_ps_per_score=24.5
```

Output of the four-ary runs (the `Δ` column is against the corrected profile, which counts `c` passes; the literal binary runs give totals `9.7`, `21.0`, `80.1`, `600.0`):

```
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.76      9.2   Δ   +3.8%
   4K     3.25       4.46    0.021    4.0       5.30     17.0   Δ  +21.2%
  16K     3.06      16.82    0.020    4.0      31.22     55.1   Δ  +56.9%
 128K     1.99      87.48    0.013    4.0     272.74    366.2   Δ +113.7%
```

| Context | Δμ (four-ary against literal binary) | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | 9.72 → 9.22, −5.2% (E0) | tree rounds: `6 log N → 5 log N` values per two levels, about −17% of the tree-round messages, which are kilobytes of a 2 GB proof: ≈ 0 (E0) | tree rounds halve and each round's check goes from a degree-3 to a degree-5 interpolation: about −20% of the tree-level verifier work, itself a small share (E0) |
| 4K | 21.0 → 17.0, −18.9% (E0) | same | same |
| 16K | 80.1 → 55.1, −31.2% (E0) | same | same |
| 128K | 600 → 366, −39.0% (E0) | same | same |

**Reading 2, the model's and RISK-10's (leaf level only; the levels above proven from stored or resident state).** The pass count is unchanged; the gain is the lower fold count per pass and per leaf. Guess (E0, no measurement): `p_s` 7 → 6 ps per score per pass and `c_π` 12 → 10 ps per leaf, if about half of each is fold bandwidth and the folds fall by 44%:

```sh
python3 -I research/tools/cost_model.py --set leaf_ps=10 --set stream_pass_ps_per_score=6
```

```
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.11    0.021    4.0       0.36      8.8   Δ   -1.0%
   4K     3.25       4.35    0.021    4.0       1.99     13.6   Δ   -3.1%
  16K     3.06      16.41    0.020    4.0       9.63     33.1   Δ   -5.7%
 128K     1.99      85.35    0.013    4.0      66.79    158.1   Δ   -7.7%
```

**Composed with Hobbit tile folding** ([SUR-0003](../surveys/SUR-0003-passes.md#candidate-ideas) seed; [PP25] primary, E1 on its own hardware): one pass per non-fitting level instead of `c_level`. Under reading 1 the binary tree needs 9 passes at 16K and the four-ary tree 5: stream term `11.24 → 6.24`, `μ(16K)` 35.1 → 30.2 (−14%, E0, `9 → 5` passes at `1.249` per pass). Under reading 2 both need one pass and the idea adds nothing to Hobbit on passes.

Inputs: `c(S)`, `p_s = 7 ps`, `c_π = 12 ps` from the corrected profile of `cost_model.py` (E0, RISK-10's estimates, projections per 08 §1); pass counts E0 own arithmetic; fold and multiplication counts E0 own counts; the `p_s` and `c_π` reductions of reading 2 are E0 guesses with no source. Which reading RFC-0018 intends decides whether the headline is −31% or −6% at 16K; it is the first question for the refuters and the RFC's owner.

### Re-scored under the `counted` profile (2026-10-16)

[EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md) priced the leaf at `c_π ∈ [45, 117]` ps (central 73; E0 count fitted to E1 within 2%) and split the cost by component (E1 on one Apple M4 Max core; predicted 31.1 ns per leaf: `Fp2` rounds 40%, their folds 20%, the first round 15%, its folds 12%, the `eq` tables 12%, the level build 2%). That split replaces reading 2's unsourced guess for `c_π` with an estimate (E0 over the E1 split):

- rounds (55%): under the kernel's as-written rule of `d` evaluations per round (`p(1)` is never computed; EXP-0006's count), a binary tree's levels cost `3N + 1.5N + … = 6N` round multiplications and a four-ary tree's `5N + 1.25N + … = 6.67N`: **+11%**, not the equality of the `d + 1`-node count under "Why it could work", which charged four nodes to the binary level and six to the four-ary one;
- folds (32%): `3N → 1.67N` entry-folds, **−44%**, as counted above;
- `eq` tables (12%): `N/2 + N/4 + … = N` entries built for the binary tree against `N/4 + N/16 + … = N/3`: **−67%**;
- level build (2%): the same `N` multiplications either way.

Weighted, `0.55 × 1.11 + 0.32 × 0.56 + 0.12 × 0.33 + 0.02 = 0.85`: `c_π` 73 → 62 ps (−15%; the note's guess was −17%). `p_s` 7 → 6 stays a guess, since EXP-0006 streams nothing. Additions are not counted: they are a third of the int32 work per EXP-0006, and the five-table summand has a different addition count, for the E1 prototype to settle.

```sh
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=62.0 --set stream_pass_ps_per_score=6      # reading 2, EXP-0006 split
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=60.83 --set stream_pass_ps_per_score=6     # reading 2, the note's proportional guess (×10/12)
python3 -I research/tools/cost_model.py --profile counted --contexts 16384 --set stream_pass_ps_per_score=35         # reading 1, literal binary (45 passes)
python3 -I research/tools/cost_model.py --profile counted --contexts 16384 --set stream_pass_ps_per_score=19.444444  # reading 1, four-ary (25 passes)
python3 -I research/tools/cost_model.py --profile counted --contexts 16384 --set stream_pass_ps_per_score=3.888889   # reading 1 with Hobbit: 5 passes against the baseline's 9
```

| Context | `counted` baseline μ | Reading 2, EXP-0006 split (`c_π` 62, `p_s` 6) | Reading 2, proportional (`c_π` 60.83, `p_s` 6) | Reading 1, literal binary → four-ary |
|---|---|---|---|---|
| 1K | 17.3 | −1.2% (17.1) | −1.3% | 18.2 → 17.7, −2.7% |
| 4K | 30.0 | −3.1% (29.1) | −3.3% | 36.9 → 33.0, −10.6% |
| 16K | 78.8 | −4.9% (74.9) | −5.2% | 123.7 → 98.8, −20.1% |
| 128K | 373.4 | −6.1% (350.5) | −6.5% | 802.0 → 568.2, −29.2% |

All E0 (the cost model; the `c_π` reduction over EXP-0006's E1 split; `p_s` 6 a guess; pass counts own arithmetic). Reading 1 composed with Hobbit tile folding at 16K: 78.8 → 73.8 (−6.3%, from −14% under the corrected profile). Reading 1's percentages fall because the baseline grows (the absolute stream saving, 25.0 of `μ(16K)`, is the same); reading 2's barely move, for a reason worth recording: under the corrected profile most of its −5.7% came from `p_s` (streaming is 32% of `μ(16K)` there), under `counted` most of its −4.9% comes from `c_π` (19.0% of `μ(16K)` there, against 14.3% for streaming), so the headline now rests on the sourced input rather than the guessed one.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Each round errs `d/|Fp2|` with `d = 5` instead of 3, over half as many rounds: per pair of binary levels `2 · 3n → 5n`, lower in total. Each level's two line challenges add `2/|Fp2|` (the restriction of a multilinear in two variables: a wrong `u_i^b` passes with probability at most `2/|Fp2|`) against today's `1/|Fp2|` per level over twice as many levels. Root claims (`s/|Fp2|` per forest) and the `γ` batching (`m/|Fp2|`) are unchanged. The soundness calculator (RFC-0014 'Soundness calculator') takes the new degree and round count; the total stays below the 2^-110 of RFC-0010 'Soundness of the schedule' (E0, from the per-round bound). | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None. The root is the same product over the same leaves `k_qi`; Lemma 5 (max certificate) argues from the product being zero in a field, not from the tree's shape. The fraction tree's root `(n, d)` is likewise the same rational sum. | pass |
| Zero knowledge (G-5, RFC-0013) | OQ-0013-2 is open today: a level reveals `Ṽ_{ℓ+1}(0 ‖ r'_i)` and `Ṽ_{ℓ+1}(1 ‖ r'_i)`, points with a Boolean coordinate where the mask's `Z` vanishes. The four-ary level reveals four such values per level over half as many levels: the same count of revealed values per tree (`2 · h` → `4 · h/2 = 2h`), the same open question, not worse. The Libra masking polynomial of RFC-0013 for a tree round would have degree 5. Unknown until OQ-0013-2 is resolved; its resolution must not need per-level masks whose count the change does not reduce (kill criterion (c)). | unknown |
| Verifier independence (SP-6) | `tv-core` implements the degree-5 level check and the two-variable line reduction from the specification; no shared code, no new dependency. | pass |
| Post-quantum (hash-based commitments) | Unchanged: a sumcheck relation and transcript labels only; commitments untouched. | pass |

## Prior rejection check

- [baseline.md §7](../baseline.md#7-already-rejected): no row concerns tree arity. The RFC-0010 row ("layer-by-layer GKR; committing every intermediate; independent per-layer challenges; one monolithic sumcheck") and the RFC-0008 row (one-hot, univariate, base-field challenges) do not cover it.
- RFC-0018 Alternatives Considered 3, "Multi-variable sumcheck rounds (binding several variables per round to cut streamed passes). Rejected for v1: message size and prover work grow as `(d + 1)^w` per round": the nearest rejection. It does not cover this idea: a `2^w`-ary level still binds one variable per round; its message grows linearly in the arity (`2^w + 1` values per round, 5 for `w = 2`, against `(d + 1)^w = 16`), and round-1 multiplications are equal, not `(d + 1)^w`. What is new is the observation that arity, not variables per round, is the parameter that halves the non-fitting level count. "Reconsider if RISK-10 measurements exceed 50% overhead": under reading 1 the streaming term is already 3.3× the attention term at 16K (E0).
- RFC-0018 Alternative 2 (per-tile challenges) is untouched; the idea composes with SUR-0003's Hobbit seed, which answers that objection separately.
- RFC-0010 Alternatives Considered 1–5 and RFC-0008 Alternatives Considered 1–6 do not mention arity (read on 2026-10-16).
- Not excluded by a known result: the lookup separation principle concerns per-lookup cost, not the tree over the lookups.

## Kill criterion

Any one of: (a) RFC-0018's owner confirms reading 2 (only the leaf level is streamed; the levels above are proven from stored or resident folded state) and the E1 prototype shows no byte saving per leaf, so neither passes nor bandwidth move; (b) the four-ary level runs more than 10% slower per leaf than two binary levels in the kernel (the row loop at 6 nodes × 5 tables over `Fp2` losing what the halved fold count gains); (c) the resolution of OQ-0013-2 needs per-level masks whose count the change does not reduce, so the zero-knowledge cost of tree levels stays and the tier-A change buys only the fold bytes.

## Cheapest experiment

**Hypothesis.** One four-ary level over `2^22` leaves is within ±10% of two binary levels in prover time per leaf on this CPU, folds at most 0.6× the entries, and is accepted by an independent verifier-side recomputation that rejects a flipped evaluation.

**Method.** E1 here: a research crate under `research/experiments/EXP-NNNN-four-ary-levels/` with its own `[workspace]`, depending on `tessera-sumcheck` and `tessera-field` by path (README rule 2). Build a `SoftmaxRow`-shaped forest of `2^22` leaves of 22-bit integers from a seeded generator (rule 6); prove the top two binary levels through `prove_batch` (3 tables, degree 3, twice) and one four-ary level (5 tables, degree 5, once) in the same harness on the same leaves (rule 4); count ns per leaf and entry-folds; recompute the verifier's check (`Σ_i γ^i eq(r_i, r'_i) Π_b u_i^b` against the last claim, then the two-variable line reduction) in the experiment's own code, and assert that flipping one `u_i^b` rejects. Plus the E0 pass count per level from RFC-0018's formula for both arities at 1K–128K as a table, and a question to the RFC-0018 owner on reading 1 against reading 2.

**Evidence reachable.** E1 for compute neutrality and fold count (the machine named in the report); E0 for passes. Effort `m` for the experiment (days: a kernel path exists for 6 nodes, the forest builder needs a four-quarter variant); `l` for adoption (a tier-A RFC superseding RFC-0010's level protocol and RFC-0014's label schedule).

**Unknown until an H100 run.** Whether per-pass cost is bandwidth-bound as SUR-0003 finding 6 says, so that the 44% fewer folds show in `p_s` and `c_π`; the all-reduce latency saved per removed round; and the actual pass count per level, which v0.5's pass counter measures (RISK-10).

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Degree 5 sits inside RFC-0014's budget row (d ≤ 7) and the engine's MAX_DEGREE = 7 (`crates/tessera-poly/src/round.rs`); per pair of binary levels the round error 3n + 3(n − 1) becomes 5n and the line step 2·(1/q') becomes one 2/q' (a wrong u^b vector differs from the truth by a non-zero bilinear polynomial in (ρ_1, ρ_2), Schwartz–Zippel degree 2), so the total is not worse (E0, primary, refuter's count); the verifier still recomputes eq(r_i, r'_i) itself and checks Σ γ^i eq Π_b u_i^b against the last claim, as RFC-0010 'Tree levels' does today. Both line challenges must be squeezed after tree/{ℓ}/evals is absorbed, which needs a second label in the RFC-0014 schedule, the tier-A change the note names. SP-3 is unchanged: Lemma 5 (RFC-0009) argues from a zero product in a field, not from the tree's arity. Zero knowledge stays at OQ-0013-2 with one caveat the note understates: at the level where a single tree joins, a four-ary level reveals the four raw quarter-products of the row's keys where a binary level reveals the two half-products, so the open question's resolution must cover four Boolean-coordinate reveals per level and unmasked degree-5 round messages; the total count of revealed values per tree (2h) is the same. |
| prior art | survives | Not in the specification: RFC-0010 'Relations' and 'Tree levels' fix binary pairing (V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x+1], degree 3, one line challenge ρ), and the prover hard-codes it (`LEVEL_DEGREE = 3`, `crates/tessera-sumcheck/src/forest.rs`) while the engine allows degree 7. Not rejected: RFC-0018 Alternatives Considered 3 rejects binding several variables per round for (d + 1)^w growth, whereas a 2^w-ary level still binds one variable per round and sends 2^w + 1 values, so the distinction the note draws holds; RFC-0010 Alternatives 1–5 and baseline.md §7 do not mention arity, and [Jo26] concerns per-lookup cost, not the tree. Higher fan-in GKR layers are a known free parameter in the literature, so the arity itself is not new; new is using it to halve the count of non-fitting levels. The spec question the note raises is real: RFC-0018 'Leaf levels' says 'A level that does not fit is proven by streaming' for every level while its own count (c ≈ 8) and the cost model charge the leaf level only, so the headline −31% at 16K (reading 1) against −6% (reading 2) turns on an RFC-0018 clarification, not on prior art. |
| cost model | weakened | The gain is bimodal on an unresolved spec reading: under reading 1 the four-ary tree reruns to 80.1 → 55.1 at 16K (−31.2%) and 600 → 366 at 128K, under reading 2 to −5.7% at 16K from `leaf_ps=10, stream_pass_ps_per_score=6`, both reproduced. Reading 1 follows from RFC-0018 'Leaf levels' read literally for GKR (every level whose two tables exceed 0.2 TB needs its own c_level passes), but the same paragraph totals 'c ≈ 8 streamed passes' at 16K and RISK-10 and `cost_model.py` count the leaf level only; if reading 1 is right the baseline's streaming term is 5× understated (μ(16K) = 80 > the SC-2 ceiling) and the dominant fix is SUR-0003's Hobbit tile folding, which takes the same 9 passes to 1 while four-ary alone takes them to 5. Under reading 2 the −5.7% rests on p_s 7 → 6 and c_π 12 → 10, labeled by the note as guesses with no source, i.e. below the model's input noise. The pass arithmetic (45 → 25, 78 → 42) and the compute-neutrality count are sound E0; the idea's value is conditional on a question for the RFC-0018 owner that must be answered before any score is meaningful, and on whether folds are the bandwidth driver (unmeasured). |
| feasibility | survives | The E1 prototype is feasible without touching the engine: MAX_DEGREE = 7, `prove_batch` accepts 5 tables at degree 5 (`crates/tessera-sumcheck/src/prover.rs`), and an Fp2 eq beside four Fp quarters takes the block path of `product_sums` with NODES = 8 (`kernel.rs`); `build_levels` is private (`forest.rs`), so the experiment reimplements a four-quarter level build, a small job. Effort `m` for the experiment holds. Adoption is `l`: tier-A changes to RFC-0010 relations, RFC-0014 labels and `tv-core`'s level check, plus OQ-0013-2 `unknown`. The headline gain rests on reading 1 of RFC-0018 'Leaf levels', which describes streaming (regenerating leaves tile by tile) for leaf levels only; streaming a non-leaf level would mean regenerating its product nodes per pass, which no RFC text specifies, so the −31% figure is a question for the owner rather than a measurable claim, and the measurable part (−44% folds) is reading 2's −6%. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −5.7% μ(16K) under the cost model's own leaf-only reading of RFC-0018 (E0 cost model; inputs `leaf_ps` 12 → 10 and `stream_pass_ps_per_score` 7 → 6 are the note's guesses with no source); −31.2% μ(16K) under the literal every-level reading (E0 pass arithmetic 45 → 25), which no RFC text specifies and which Hobbit tile folding (9 passes → 1) would dominate | 0.3: compute neutrality and the pass arithmetic are sound E0, but the headline turns on an unanswered spec question and the reading-2 inputs have no source; zero knowledge stays at OQ-0013-2 (`unknown`, not worse) | `l`: tier-A changes (RFC-0010 relations, a second RFC-0014 line label, `tv-core`'s level check) and OQ-0013-2 open; the E1 prototype is `m` here, no hardware | 5.7 × 0.3 / 4 = 0.43 (31.2 × 0.3 / 4 = 2.3 if reading 1 is confirmed, in which case the baseline's streaming term is 5× understated and the queue reorders around tile folding) |
| Re-scored 2026-10-16 under `counted` (`c_π` 73 ps, EXP-0006): −4.9% μ(16K) under the leaf-only reading (E0 cost model at `leaf_ps=62.0`, from EXP-0006's E1 component split under the kernel's `d`-evaluations rule, and `stream_pass_ps_per_score=6`, still a guess); −20.1% μ(16K) under the literal every-level reading (E0 pass arithmetic 45 → 25) | 0.3: unchanged; the `c_π` input now has a source, but the spec question is unanswered, `p_s` 6 has none, and the round multiplications are +11% rather than equal | `l`: unchanged | 4.9 × 0.3 / 4 = 0.37 (0.39 with the proportional guess; 20.1 × 0.3 / 4 = 1.5 if reading 1 is confirmed) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.43. First action: an issue asking the RFC-0018 owner whether 'A level that does not fit is proven by streaming' charges every non-fitting GKR level or the leaf level only (the paragraph's own c ≈ 8 and `cost_model.py` say leaf only); the E1 prototype (compute neutrality, −44% folds) is worth running only once that answer and IDEA-0008's repricing of c_π are in. Experiments: none yet. RFC or issue: none. Ledger line updated on: pending (the consolidation agent adds the ledger line).

Re-score 2026-10-16 (`ideate (score)` under the `counted` profile): `triaged` at 0.37 (from 0.43), 1.5 under reading 1 (from 2.3); the first action is unchanged, the issue to the RFC-0018 owner. The headline now rests on `c_π` (−15% by EXP-0006's split) rather than on the guessed `p_s`, and the E1 prototype's compute-neutrality hypothesis should be restated as "within +11% in multiplications" under the kernel's `d`-evaluations rule. Ledger line updated on: 2026-10-16.
