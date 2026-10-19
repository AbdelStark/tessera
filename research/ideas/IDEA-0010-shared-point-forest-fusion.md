# IDEA-0010: Shared-point fusion of the `SoftmaxRow` forest with the `Exp` fraction tree

| | |
|---|---|
| Status | `triaged` (updated 2026-10-19; EXP-0018 inconclusive) |
| Origin | [SUR-0004](../surveys/SUR-0004-attention.md) seeds "Base-field leaf levels of the `SoftmaxRow` product tree" and "Share the first product-tree pass with the `Exp` fraction tree's first level"; [SUR-0003](../surveys/SUR-0003-passes.md) seed "Small-value first rounds" ([DDBDT26]); prior-art check against `crates/tessera-sumcheck` (`forest.rs`, `kernel.rs`, `benches/throughput.rs`) |
| Lever | `c_π` (the eq-table build and the leaf-level reductions: about 10–15% of the forest's compute by this note's count, E0; the generation pass counted 25–30%, which this note does not reproduce); `stream` (`p_s`) only against an implementation that streams the forest and the fraction tree in separate passes, which the specification does not describe |
| Spec surfaces | None normative. RFC-0010 'Tree levels' and 'Batching within a stage', RFC-0008 'Batching' and RFC-0018 'Fusion' already describe the protocol this idea exploits; the change is confined to the prover's kernel (`tessera-sumcheck`, later `tessera-gpu`). Messages, points, labels and the verifier are unchanged, so no tier-A surface (02 §1) moves and no RFC is required; OQ-0010-1 (how fraction and product trees are ordered inside one level) must be resolved as its text intends for mechanism 1 to apply |
| Priority | 0.87 (−2.9% × 0.3 / 1 under the `counted` profile, `c_π` 73 ps, re-scored 2026-10-16 after [EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md); 0.33 under the corrected profile at 12 ps) |

## Mechanism

Setting. In the leaf stage, the `SoftmaxRow` forest (one product tree per query row, leaves `k_qi = c_q − x_qi`, RFC-0009 'SoftmaxRow' relation 2) and the `Exp` fraction tree (leaves `(1, α − k_qi − β·e_qi)` for the lookups of relation 1 and `(−m_j, α − t_j)` for the table rows, RFC-0008 'Fraction tree') are proven level by level in the same batched sumchecks (RFC-0008 'Batching'; RFC-0010 'Tree levels'; OQ-0010-1 states that both kinds run "in the same levels, under one label set per level"). At level `ℓ` every joined instance `i` ends at the point `r'_i` = the last `n_i` round challenges (RFC-0010 'Tree levels' step 3), and its claim for the next level sits at `(ρ ‖ r'_i)` with the shared line challenge `ρ` as the lowest coordinate (step 4). Two instances with `n_1 ≤ n_2` therefore have points whose last `n_1` coordinates coincide and whose lowest coordinate is the same `ρ`: `eq(r_2, x) = eq(ρ, x_0) · eq(m, x_mid) · eq(r_1-tail, x_tail)` where `m` is the `n_2 − n_1` middle challenges that instance 1 never sees.

Three prover-only changes, none of which alters a message:

1. **Shared eq tables.** Build `eq(ρ ‖ shared tail, ·)` once per level for the whole batch and derive each joined instance's `eq(r_i, ·)` from it: identical when `n_i` is equal, a tensor product with a `2^{n_2 − n_1}`-entry middle table when it is not (the `eq_table` build in `crates/tessera-poly/src/eq.rs` doubles coordinate by coordinate, so the shared tail is an intermediate of the larger build when coordinates are processed from the tail). The table is read once per pair per round instead of once per instance. What this does **not** do: it does not fold the per-node `eq` multiplication across instances. Today each instance computes `Σ_x eq(r_i, x)·(product at node z)` and `γ^i` is applied to the instance sum; a fused summand `eq·(γ_1·n + γ_2·d + γ_3·V_L V_R)` would have to multiply `V_L V_R ∈ Fp` by `γ_3 ∈ Fp2` per node, which costs the same two `Fp` multiplications as multiplying it by `eq`, and would turn the fraction tree's `Fp2` terms into `Fp2 × Fp2` products before the shared multiplication (own count, E0). The generation pass's "one eq multiplication for the whole batch" saving is therefore nil in multiplications; the saving of this mechanism is the build (one table per level instead of one per joined instance) and the memory traffic of one table.
2. **Integer round 1 at the leaf level.** Leaves are integers in `[0, 2^22)` (scores `|x_qi| < 2^21`, RFC-0009) held in the prime subfield (`forest.rs` builds every level in `Fp`). On the line through a pair, the forest's value at node `z` is `a + z·Δ` with `|a + z·Δ| < 2^24` for `z ≤ 3`, so `V_L·V_R` at every node is an integer below `2^48` and needs no field reduction; `eq_x · (V_L V_R)_x` is two `64 × 48`-bit products below `2^112` per `Fp2` component, accumulated in `u128` and reduced to Goldilocks once per `2^16` terms instead of once per term, after the small-value prover of [DDBDT26]. The lift of the folded table into `Fp2` happens once, at the bind after round 1, exactly where `bind_lowest` lifts today (`benches/throughput.rs`, "one bind_lowest pass, Fp into Fp2"). Level `h − 1` holds 44-bit products: `V_L V_R` still fits `u128` unreduced, but `eq × (V_L V_R)` does not, so only the product's own reduction is saved there. From level `h − 2` the values are full field elements and the generic kernel runs.
3. **One streamed pass per level for both kinds.** When a leaf level is streamed (RFC-0018 'Leaf levels'), each pass regenerates the tile's scores once and produces the forest's leaves `k_qi` and the fraction tree's leaves `(1, α − k_qi − β·e_qi)` from the same keys, accumulating both instances' round messages in that pass; the folded states written at the last streamed round are produced in the same pass. This is what RFC-0018 'Fusion' and the single batched sumcheck per level already imply; the mechanism names it so the kernel does not run the forest's pass separately. Against the specification as written it moves nothing; against a kernel that streams the two kinds apart it saves one pass of `p_s` per streamed round over the forest's 16 B per leaf.

The verifier's work is unchanged: it still computes `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i` per level with its own `eq` (RFC-0010 step 3). Control flow depends on shapes and variable counts only, never on a leaf value (INV-ARCH-2).

## Why it could work

- **The eq build is 10% of the standalone forest's time (E1, primary, this machine).** `cargo bench -p tessera-sumcheck --bench throughput` on one Apple M4 Max core, rustc 1.95.0, 2026-10-16: `prove_forests` at 32.2–32.5 ns per leaf for a forest of 2^10 trees of 2^12 leaves, one tree of 2^22 leaves, and a mixed batch of 2^22 leaves; "the eq tables alone" 3.26–3.30 ns per leaf in all three cases (10.1%). Mechanism 1 removes the forest's share of that build when the fraction tree builds the same table anyway, so its ceiling is about 10% of the forest's CPU time, less the small middle tables.
- **The leaf-level product kernel is reduction-bound enough for mechanism 2 to show (E0, own count; E1 anchors).** The same bench measures the engine's degree-3 product of three `Fp` tables at 24.6 ns per entry and "`Fp2` public × 2 `Fp` tables" at 27.8–28.2 ns per entry; the forest's leaf level is the latter shape. One Goldilocks reduction is about a third of a multiplication; mechanism 2 removes about three reductions per node per pair at the leaf level and one at level `h − 1`, which is 2–3 `Fp`-multiplication equivalents per leaf out of about 46 for the whole forest (own count from the kernel's loops, E0): 4–6%.
- **The prime-subfield path is real and already in use (E1, primary, code comment).** `kernel.rs` reports that the zipped prime-subfield loop runs 20 to 45% faster than the row loop, measured on an M4 Max core; `forest.rs` reports the halves-based level build at 1.5× the whole-level build. What is new here is cross-instance sharing of the `eq` table and unreduced integer accumulation, not the base-field path, which SUR-0004's first seed asked for and which is already implemented (that seed's own killer).
- **Production LogUp-GKR provers ship base-field first layers and a batch-wide `eq` (E0, primary, code read in SUR-0002).** [Stwo26]'s `prove_batch` generates `eq_evals` once per batch and runs one batched sumcheck per level; [SP1GPU26] stores first-layer numerators in the base field and fuses two sumcheck rounds per kernel. Neither shares the table with a product tree, because neither has one; the mechanism transfers because Tessera's product forest and fraction tree sit in one batch by design.
- **[DDBDT26] (E1, primary, abstract-level in SUR-0003)** reports over an order of magnitude on Spartan and 1.7–2.2× on a high-degree Shout sumcheck from small-value first rounds, on CPU. Tessera's case is narrower: the values are small for one level and a half, and the `eq` factor is a full `Fp2` element, so the saving is reductions, not multiplications.

Keys [DDBDT26], [Stwo26] and [SP1GPU26] are defined in the source rows of [SUR-0003](../surveys/SUR-0003-passes.md) and [SUR-0002](../surveys/SUR-0002-lookups.md); their bibliography lines are merged by the consolidation agent.

## Expected gain

Mechanisms 1 and 2 together: about 10% (eq build) + 4–6% (reductions) = 10–15% of the forest's compute, so `leaf_ps` 12 → 10.2 as the headline (−15%, E0 own count over E1 anchors) and 12 → 10.8 as the floor (−10%). The generation pass's −25% (`leaf_ps = 9`) is kept as the upper row for the record; this note's count does not support it (the per-node sharing it assumed costs as much as it saves). Mechanism 3 moves nothing against the specification.

```sh
python3 -I research/tools/cost_model.py --set leaf_ps=10.2    # headline: −15% of c_π
python3 -I research/tools/cost_model.py --set leaf_ps=10.8    # floor: −10% of c_π
python3 -I research/tools/cost_model.py --set leaf_ps=9       # generation pass: −25% of c_π
python3 -I research/tools/cost_model.py --set leaf_ps=42.5    # −15% of a repriced 50 ps (compare --set leaf_ps=50)
python3 -I research/tools/cost_model.py --set leaf_ps=70.5    # −15% of a repriced 83 ps (compare --set leaf_ps=83)
```

Headline scenario, `leaf_ps = 10.2` against the corrected profile (the tool's output, 2026-10-16):

```
scenario: leaf_ps=10.2
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.11    0.021    4.0       0.42      8.9   Δ   -0.3%
   4K     3.25       4.37    0.021    4.0       2.32     14.0   Δ   -0.7%
  16K     3.06      16.45    0.020    4.0      11.24     34.8   Δ   -1.1%
 128K     1.99      85.56    0.013    4.0      77.93    169.5   Δ   -1.1%
```

| Context | Δμ (headline, `leaf_ps` 12 → 10.2) | Δμ (floor, 12 → 10.8) | Δμ (generation pass, 12 → 9) | Δ proof size | Δ verification |
|---|---|---|---|---|---|
| 1K | −0.3% (8.9 → 8.9) | −0.2% | −0.5% (8.9 → 8.8) | 0 | 0 |
| 4K | −0.7% (14.1 → 14.0) | −0.5% | −1.2% (14.1 → 13.9) | 0 | 0 |
| 16K | −1.1% (35.1 → 34.8) | −0.7% | −1.8% (35.1 → 34.5) | 0 | 0 |
| 128K | −1.1% (171.4 → 169.5) | −0.7% | −1.9% (171.4 → 168.2) | 0 | 0 |

If `c_π` is repriced upward, as SUR-0002's count did for `c_ℓ`, the same 15% is worth more: at 50 ps the baseline is `μ(16K) = 42.9` and `leaf_ps = 42.5` gives 41.4 (−3.5%, 128K −3.8%); at 83 ps the baseline is 49.7 and `leaf_ps = 70.5` gives 47.1 (−5.2%, 128K −5.5%). The generation pass's figures at those prices (50 → 37: 42.9 → 40.3, −6.1%; 83 → 62: 49.7 → 45.4, −8.7%) assume the −25% this note does not reproduce. Every input is E0; the 12 ps is an assumption (baseline §2, RISK-10/RFC-0020 gate), the 10% eq share and the kernel timings are E1 on an Apple M4 Max core and may not transfer to the H100's int32 pipeline (`u128` accumulation is emulated there).

Mechanism 3's `p_s` share is not expressible in the model's `p_s` because the model already assumes one pass per streamed round over all attention leaves; it would be a correction of an implementation that ran two, worth at most 16 B per leaf per streamed round of the 32 B the fold writes (E0, RFC-0018 'Leaf levels').

### Re-scored under the `counted` profile (2026-10-16)

[EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md) priced the standalone leaf at `c_π ∈ [45, 117]` ps (central 73; E0 count fitted to E1 within 2% on one Apple M4 Max core), and `tools/cost_model.py` carries it as the `counted` profile with `c_c` 94 and `c_ℓ` 101 ps. The same fractions of `c_π` as above, applied to 73 ps. The saving is a share of the standalone count, which includes the forest's own `eq` tables: 3.0 of 25.5 `Fp`-multiplication equivalents by EXP-0006's count, 11.6% predicted and 10.3% measured (E1), so mechanism 1's ceiling is confirmed on a second harness.

```sh
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=62.05   # headline: −15% of 73 ps
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=65.7    # floor: −10%
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=54.75   # generation pass: −25%, not reproduced by this note
```

| Context | `counted` baseline μ | Δμ (headline, 73 → 62.05) | Δμ (floor, 73 → 65.7) | Δμ (generation pass, 73 → 54.75) |
|---|---|---|---|---|
| 1K | 17.3 | −0.9% (17.2) | −0.6% | −1.5% |
| 4K | 30.0 | −2.0% (29.4) | −1.3% | −3.3% |
| 16K | 78.8 | −2.9% (76.5) | −1.9% | −4.8% |
| 128K | 373.4 | −3.1% (361.7) | −2.1% | −5.2% |

All E0 (the cost model over E0 counts fitted to E1; the H100 translation is EXP-0006's). Under `counted` the `c_π` share of `μ(16K)` is 19.0%, not the 31% EXP-0006 reports with `c_c` and `c_ℓ` at their assumed values: all three unit costs rise together. So the headline is −2.9%, below the −3.5% to −5.2% projected above for 50–83 ps against the corrected profile. Across EXP-0006's interval the headline scales with `c_π`: about −1.8% of `μ(16K)` at 45 ps and −4.6% at 117 ps (E0, approximate: the baseline total moves too). Two facts from EXP-0006 bear on the experiment. The redundant per-instance `eq` folds of its fraction proxy cost 7.5% of the proxy's time per lookup (E1): mechanism 1's issue against `tessera-sumcheck`, quantified. And IDEA-0008's second kill-criterion clause asks exactly what hypothesis (a) below measures, the marginal cost of the forest inside a shared-point batch, so this experiment also decides whether the standalone 73 ps or a marginal figure is the right model input for `c_π`.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None. The sumcheck, its degree, its points `r'_i` and `(ρ ‖ r'_i)`, its messages and the verifier's checks are those of RFC-0010 'Tree levels'; the prover computes the same sums by another arrangement. Integer accumulation is exact (every partial sum is below `2^128` by construction and reduced to the same field element), so the messages are bit-identical. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None. Relations 1 and 2 of `SoftmaxRow` and Lemma 5 are untouched; no new committed value, no prover-declared range. | pass |
| Zero knowledge (G-5, RFC-0013) | None. Masks per stage and level are unchanged; what a tree level reveals is what RFC-0010 already reveals (OQ-0013-2 stands as before, neither better nor worse). Secret-class leaves and levels stay inside the kernel; shared tables are public `eq` values (05 §3). | pass |
| Verifier independence (SP-6) | None. No verifier code changes; `tv-*` keeps computing its own `eq(r_i, r'_i)` from the specification. | pass |
| Post-quantum (hash-based commitments) | None. No commitment or hash changes. | pass |

Timing: whether two instances share a table depends on their variable counts and the schedule, never on a value (INV-ARCH-2 holds).

## Prior rejection check

- [baseline.md §7](../baseline.md#7-already-rejected): no row covers prover-side table sharing or integer accumulation. The RFC-0009 row rejects cheaper *certificates* (two-table bracket, one-hot maximum) and reopens on "a certificate cheaper than one product leaf per score"; this idea keeps the certificate and lowers the leaf's cost, so that row neither covers nor reopens it.
- RFC-0010 'Batching within a stage' and 'Tree levels' write `eq(r_i, x)` per instance and do not forbid one table for equal or nested points; RFC-0010's Alternatives Considered (layer-by-layer GKR, committing intermediates, independent per-layer challenges, one monolithic sumcheck) do not touch it.
- RFC-0018 'Fusion' names pass fusion across P2/P3 and the tree build, not instance-level sharing within a level; mechanism 3 is its reading, not an alternative to it.
- SUR-0004's first seed ("base-field leaf levels") is already implemented in `tessera-sumcheck` (levels in the prime subfield, zipped base-field loops): that seed's own killer fired, and this idea does not claim it. SUR-0004's second seed ("share the first product-tree pass with the `Exp` tree's first level") carried the killer "RFC-0010 may already stack them in one stage"; OQ-0010-1 confirms that it does, which is why mechanism 3 moves nothing against the specification.
- New now: the E1 measurement that the eq build is 10% of the forest's time, and the count that per-node eq sharing is worthless, which turns a 25–30% seed into a 10–15% one before triage.

## Kill criterion

The marginal cost of the forest inside a shared-point `prove_batch` with a fraction-shaped instance (two `Fp2` and two `Fp` tables at the same point) is at least 90% of its standalone cost on the same machine (E1), and the integer round 1 over 22-bit leaves is not at least 10% faster than the reduced kernel at the leaf level. Either alone halves the idea; both together refute it. A shape kill also exists: if the schedule gives the forest and the `Exp` tree levels that never coincide (the forest's padded leaf count is about `2^42` for 126 layers × 128 heads × 2^14 rows × 2^14 padded leaves, and the combined fraction tree's `N + T ≈ 2.2×10^12 + 2^22` pads to `2^42`; whether they join at the same level or one apart is a schedule fact to read, E0), the shared table degrades to the nested case with a middle table, which costs the mechanism about half its build saving.

## Cheapest experiment

Hypothesis: on one M4 Max core, (a) proving the forest of 2^22 leaves inside a `prove_batch` with a fraction-shaped instance at the same point, with one eq table per level, costs at most 90% of the forest's standalone 32.2–32.5 ns per leaf in marginal terms; (b) a research-crate prototype of round 1 over 22-bit leaves with `u128` accumulation and one reduction per `2^16` terms matches the field kernel bit for bit and runs at least 10% faster than the engine's "`Fp2` public × 2 `Fp` tables" loop (27.8–28.2 ns per entry today).

Method: extend `crates/tessera-sumcheck/benches/throughput.rs` in a research workspace (rule 2: own `[workspace]`, path dependencies on `tessera-field`, `tessera-poly`, `tessera-sumcheck`) with the batched case and the prototype; seeded synthetic leaves (rule 6); fastest of several runs, baseline and candidate in the same harness (rule 4); report ns per leaf and the ratio. Reaches E1 for the CPU share. Effort `s` (days, no RFC). What stays unknown until an H100 run: whether `u128` emulation on the int32 pipeline keeps mechanism 2's saving, the HBM traffic saved by one eq table, and the true `c_π` the 12 ps stands for (RFC-0020 gate).

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | All three mechanisms are prover-side rearrangements: eq tables are public, so sharing one across joined instances leaks nothing and changes no message; the integer round 1 is exact because leaves k_qi ∈ [0, 2^22) (RFC-0009 abs(x) < 2^21; RFC-0005 keys ≤ 2^22), nodes a + z·Δ for z ∈ {0, 2, 3} stay below 2^24 and products below 2^48, so the Goldilocks reduction is deferred, not skipped, and the messages are bit-identical (E0, primary, refuter's bound). An overflow would only make an honest message fail p(0) + p(1) = claim; it can never help ADV-P. One margin to fix in the experiment: abs(eq_x) < 2^64 times abs(V_L·V_R) < 2^48 over 2^16 signed terms bounds the accumulator at 2^128, one bit over i128, so reduce every 2^15 terms or keep separate sign accumulators; the pre-registered bit-for-bit test catches it either way. Control flow depends on shapes only (INV-ARCH-2), the verifier's eq(r_i, r'_i) check (RFC-0010) is untouched, and OQ-0013-2 is neither better nor worse. |
| prior art | weakened | Mechanism 3 is already the specification (RFC-0008 'Batching'; RFC-0010 'All LogUp fraction trees and all product trees, level by level … batched across instances'; RFC-0018 'Fusion'), as the note concedes. Mechanism 1 is implied by the protocol: RFC-0010 'Tree levels' step 4 puts every instance's next claim at (ρ ‖ r'_i) with r'_i the last n_i challenges, so instances with equal variable counts share one point by construction, and building eq once per batch is what Stwo's `prove_batch` already does ([Stwo26], read in SUR-0002); only the prototype builds it per instance (`crates/tessera-sumcheck/src/forest.rs`), so this is an implementation fix, not a design idea. Mechanism 2 is [DDBDT26]'s small-value trick again (overlapping IDEA-0005), applied where the note itself finds the values small for one and a half levels. Nothing rejects it (baseline.md §7 and RFC-0010 Alternatives are silent), but what is new is only the E1 measurement that the eq build is 10% of the forest and the count that per-node eq sharing saves no multiplications, which caps the gain at about 1% of μ. |
| cost model | weakened | The headline `leaf_ps=10.2` reruns to −0.3 / −0.7 / −1.1 / −1.1%, the floor to −0.7% at 16K, and even the disowned −25% to −1.8%, all below the model's input noise while c_π carries a 3–7× uncertainty (IDEA-0008) and is 7.0% of μ(16K) today. The note's own count already demoted the seed from 25–30% to 10–15% of the forest by showing per-node eq sharing costs what it saves, and mechanism 3 moves nothing against the specification; the 10% eq share is E1 on an M4 core and the u128 accumulation of mechanism 2 is emulated on the int32 pipe, so neither share is known to transfer. The gain becomes visible only through IDEA-0008's repricing (−3.5% at 50 ps, −5.2% at 83 ps; reruns 42.9 → 41.4 and 49.7 → 47.1), so its priority is a function of that experiment, not of its own numbers. Cheap (effort `s`, prover-only) and honestly bounded, but at the stated price it is inside noise. |
| feasibility | survives | Prover-only, no spec surface, effort `s`, E1 reachable: `eq_table` (`crates/tessera-poly/src/eq.rs`), the `bind_lowest` bench (`crates/tessera-sumcheck/benches/throughput.rs`) and `prove_forests` building one eq per joined instance (`forest.rs`) are all present, and `Instance`/`Table` are public so a research crate can pass one shared eq slice to several instances of `prove_batch`. Mechanism 2's u128 round-1 loop is standalone code. Caveats: the kernel loops are `pub(crate)` (`kernel.rs`), so the prototype's baseline is `prove_batch`'s own round or a reimplementation, not the engine's inner loop; the 'fraction-shaped instance' is a product proxy for a sum-of-products relation (same caveat as IDEA-0008); and the ceiling of 10–15% of a c_π that is itself 7% of μ bounds the payoff at about −1% before IDEA-0008's repricing. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −1.1% μ(16K), −0.3% at 1K (E0 cost model at `leaf_ps=10.2`; E1 eq-build share 10% of 32.2–32.5 ns per leaf on one M4 Max core); −3.5% to −5.2% μ(16K) conditional on IDEA-0008 repricing c_π to 50–83 ps | 0.3: mechanisms 1 and 3 are implementation of what the specification already describes, not gains over it; the E1 eq share and the int32-emulated u128 trick are not known to transfer to the GPU | `s`: prover-only, no RFC, no tier-A surface, E1 here in a research workspace over `tessera-sumcheck` | 1.1 × 0.3 / 1 = 0.33 (3.5–5.2 × 0.3 / 1 = 1.0–1.6 after IDEA-0008's repricing) |
| Re-scored 2026-10-16 under `counted` (`c_π` 73 ps, EXP-0006): −2.9% μ(16K), −0.9% at 1K (E0 cost model at `leaf_ps=62.05`; about −1.8% to −4.6% across `c_π ∈ [45, 117]`) | 0.3: unchanged; EXP-0006 confirms the `eq` share (10.3% measured, 11.6% predicted, E1 on the same machine), but mechanisms 1 and 3 remain implementation of the specification and the GPU transfer is unmeasured | `s`: unchanged | 2.9 × 0.3 / 1 = 0.87 (0.57 at the floor scenario, 1.44 at the generation pass's −25%) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.33, to be re-scored after IDEA-0008. Mechanism 1 (one eq table per batch, not per instance) is an implementation fix to `prove_forests` worth an issue against `tessera-sumcheck` regardless of the score; the experiment (mechanisms 1 and 2 against `prove_batch`'s own round, bit-for-bit) runs only if the repricing lifts c_π's share. Experiments: none yet. RFC or issue: none (prover-only; an issue against `tessera-sumcheck` if validated). Ledger line updated on: pending consolidation.

Re-score 2026-10-16 (`ideate (score)` under the `counted` profile): `triaged` at priority 0.87 (from 0.33), ahead of IDEA-0011's 0.83 by less than the uncertainty of either gain, with IDEA-0002's re-score under `counted` queued and able to overtake both. The experiment (hypotheses (a) and (b), effort `s`) is now worth running on its own numbers, and its hypothesis (a) is also the marginal-cost test IDEA-0008's kill criterion names for whether the standalone `c_π` overstates the model input. Ledger line updated on: 2026-10-16.


Experiment triage 2026-10-19: [EXP-0012](../experiments/EXP-0012-integer-leaf-round-window/README.md) hosts mechanism 2 as its v1 observation, with a deliberately fixed 32-block signed-i128 cadence. It is 9.6–11.8% slower per whole-forest leaf than the same-harness split-eq reference (E1), with bit-identical production messages/claims and independent verifier acceptance; the isolated leaf level is also slower. This does not test this note's 2^16-term cadence or its level-h−1 extension, and does not judge hypothesis (a), shared eq, or streamed-pass fusion. Keep the remaining idea triaged; do not infer an H100 timing result or a global refutation of mechanism 2 from the v1 case.


Experiment triage 2026-10-19: [EXP-0018](../experiments/EXP-0018-shared-point-marginal-forest/README.md) tests hypothesis (a) against the current production split-eq kernel, with a separately judged literal full-eq generic proxy. The primary result is `inconclusive`: median paired marginal ratios are 100.51%, 100.11% and 98.51% on the three judged 2^22-leaf shapes (E1 CPU). Every retained pointwise ratio is at least 90%, and two shapes pass the refutation rule; s10h12 retains 4.75% marginal CV after its single registered rerun, despite base-arm CVs below 1.45%, so the all-shape kill cannot fire. The proxy is also `inconclusive`: median ratios 89.68%, 89.99%, 89.20%, but every empirical min/max band crosses 90%. These bands are sensitivity ranges, not confidence intervals. All 30 large-shape workloads replay exactly and pass independent tv-core checks; all 480 raw samples and six full-series reruns are retained.

The measured production points coincide after the forest's join level, with the fresh-root-point exception at the join (s8h14 joins at global level 8, s10h12 at 10; s0h22 shares the empty root point). Current production already avoids full eq tables and their folds. Its split-eq factor builds remain per instance, so their exact build count in B is S+F. The literal generic proxy saves one full eq build while still folding the borrowed eq table independently in each instance (E0 source/loop count). This supersedes the historical full-eq implementation assumptions in this note; it does not establish a new H100 marginal c_pi or change the current counted 41.8 ps leaf default.

Decision after EXP-0018: retain `triaged`, priority 0.87 and confidence 0.3. These retained scores are the historical triage values, not a new quantitative gain estimate from an inconclusive experiment. Neither a validation nor the hypothesis (a) half-score kill is licensed. Mechanism 2's specified 2^16-term cadence remains untested by EXP-0012's 32-block observation. A new registered noise study or the separate accumulator experiment is needed before changing this decision; no production promotion follows from this CPU result.
