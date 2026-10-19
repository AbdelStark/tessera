# IDEA-0004: Calibrate `c_c` before the v0.1 gate: operation-and-byte count of encode plus chained hashing, fitted to a CPU measurement

| | |
|---|---|
| Status | `inconclusive` (2026-10-16, [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md): the fit passed and the counted interval is `[58, 157]` ps, whose floor sits on the boundary of RFC-0020's "proceed" band, so the branch cannot be named from counts; the 30 ps assumption is excluded at every point of the interval) |
| Origin | RISK-1 and RISK-8 ([00 §9](../../docs/spec/00-overview.md#9-risks)); OQ-0004-1 ([SPEC.md open questions](../../SPEC.md#open-questions)); the method of [SUR-0002](../surveys/SUR-0002-lookups.md) [E0-this] and its candidate "Calibrate the op count on a CPU (E1) against Stwo's `prove_batch`"; [SUR-0003](../surveys/SUR-0003-passes.md) "Re-derive `p_s` from bytes"; the sensitivity table of [baseline.md §2](../baseline.md#2-the-cost-model), where `c_c` is 19–21% of `μ` at every context (E0) |
| Lever | `c_c` (its uncertainty, not its value); the SC-2 ceilings ([00 §7](../../docs/spec/00-overview.md#7-success-criteria)) as a decision input |
| Spec surfaces | None change. The result feeds [RFC-0020 'Decision rule'](../../docs/rfcs/RFC-0020-cost-model.md#proposed-design) a year early; if it lands in the `ρ_c ∈ (2, 4]` branch, the SC-2 amendment that RFC-0020 already prescribes goes through an RFC (00 §7 is not tier A per [02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)). No parameter set, schema or verdict changes |
| Priority | 25 on the decision-value reading (projection exposure +41% × 0.6 / 1); 0 on the rubric's literal Δμ reading (see Score) |

## Mechanism

Produce a calibrated interval for `c_c`, the picoseconds per committed 64-bit element of the advice commitment ([RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#proposed-design): inverse NTT of size `k = 2^22`, coset NTT of size `2k`, then the `B = 32`-row chained BLAKE3 column digests of [RFC-0004 'Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#proposed-design)), in three steps that use only the hardware at hand.

1. **Count (E0).** Per committed element at rate 1/2:
   - butterflies: size-`k` inverse NTT `(k/2)·22 / k = 11`, size-`2k` coset NTT `(2k/2)·23 / k = 23`, total 34, plus 2 coset-shift multiplications;
   - int32 instructions per butterfly on the GPU's integer pipeline: one Goldilocks multiplication (12–18 instructions; [SUR-0002](../surveys/SUR-0002-lookups.md) assumed 18) plus one addition and one subtraction with their conditional corrections (about 4 each): 20–26;
   - hash: 16 bytes of codeword symbols per committed element plus the 32-byte chain state amortized over `8B = 256` bytes (factor 1.125), at about 15 int32 operations per byte ([RFC-0004 Alternatives Considered 1](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#alternatives-considered)): 240–270 operations; equivalently 5 BLAKE3 compressions per 32 symbols (one 288-byte input per column per block, as `tessera_hash::ColumnChains::absorb_block` builds it);
   - bytes through HBM per committed element: 48 (row resident on chip, written once as witness and once as codeword, read once by the hasher) to about 216 (radix-16 NTTs in 4 passes each: `4 × 16 + 4 × 32`, plus the witness read and the hash read).
2. **Measure here (E1), one harness.** The three kernels the GPU will run, as the CPU code already implements them: `cargo bench -p tessera-pcs --bench throughput` (inverse NTT 2^22, coset NTT 2^23, `RowEncoder::encode_row` at `P = 200`) and a scratch program over `tessera_hash::ColumnChains` (the `B`-row chain, swept over `B ∈ {8, …, 128}`), in ns per committed element on one core, then across cores. Fit the CPU ns to the butterfly and byte counts of step 1: the fit validates the *counts* (butterflies per element, bytes per element, compressions per symbol); the per-butterfly instruction figure is translated to the int32 pipeline from the generated code of `tessera_field::Goldilocks::mul` (`reduce128`) rather than from the ARM timing, since a 64-bit scalar multiply is one instruction on the CPU and about eight on the GPU.
3. **Project (E0) and publish.** `c_c` at the H100 int32 peak (1.67×10^13 ops/s: 132 SMs × 64 lanes × 1.98 GHz) and at the 60% utilization that [VYTA25] (via SUR-0002) measures for field-multiplication kernels; the HBM floor at 3.35 TB/s peak and at 80%; the interval `[max(compute, bytes) at peak, at utilization]` becomes the planning value, and the SC-2 ceilings are checked against its upper end, as SUR-0002 recommends for `c_ℓ`. The v0.1 measurement (E2) then confirms or replaces it; the method's own validation is whether the E2 figure lands inside the published interval.

## Why it could work

- E1, primary, this session (Apple M4 Max, one performance core, cargo 1.95.0, `cargo bench -p tessera-pcs --bench throughput`, fastest of 5 runs): `encode_row, P = 200` at `k = 2^22`: **32.85 ns per committed element** (16.42 ns per codeword symbol); `ntt 2^23` 12.38 ns per point; `intt 2^20` 10.19; `intt 2^23` 12.10; `coset_ntt 2^23` 14.31. A previous run in the generation pass gave 32.03 / 12.06 / 9.67 / 11.86 / 13.64: reproducible within 5%.
- E1, primary, this session (same machine, scratch harness in the session scratchpad over `tessera_hash::ColumnChains::absorb_block`, `2^20` columns, fastest of 5): `B = 8`: 13.06 ns per symbol; `B = 16`: 10.38; **`B = 32`: 9.77 ns per symbol = 19.55 ns per committed element, 0.82 GB/s of symbols**; `B = 64`: 9.79; `B = 128`: 9.55. The generation pass measured 9.34 / 18.67 at `B = 32`. Above `B = 16` the chain cost is flat, so the 32-byte chain overhead is not what limits it; the per-column 288-byte BLAKE3 call is.
- E1 total: **52.4 ns per committed element on one core, encoder:hash = 63:37**. Across 12 performance cores and ideal scaling this is about 4.4 ns, 150× the 30 ps assumption; the CPU figure is a calibration input, never the baseline (rule 4), and says nothing about the GPU by itself.
- E0, primary, own count (step 1 of the mechanism): NTT `34 × (20–26) + 2 × 18 ≈ 720–920` int32 operations; hash 240–270; **total 960–1,190 operations per committed element → 57–71 ps at the int32 peak, 96–119 ps at 60% utilization**. Bytes 48–216 per element → 14–64 ps at HBM peak, 18–81 ps at 80%. Both axes put the kernel as specified at 2–4× the 30 ps assumption, the same shape as SUR-0002's finding for `c_ℓ` (40 ps assumed, 56–130 ps counted; E0, primary).
- E0, `claim`: the 30 ps itself is the source's assumption ([Source26] §5 "to be measured"; [baseline.md §2](../baseline.md#2-the-cost-model) table). RISK-8 restates it as "BLAKE3 must sustain about 530 GB/s per H100" (16 B per element at 30 ps), a hash-only floor that leaves nothing for the NTT.
- External anchors: none primary for Goldilocks NTT + BLAKE3 chained hashing on an H100. [RFC-0020 Alternatives Considered 2](../../docs/rfcs/RFC-0020-cost-model.md#alternatives-considered) says a roofline alone cannot settle it; [VYTA25] (via SUR-0002; primary, measured on an A40 and a 377-bit field) supplies only the pipeline and the utilization figure.

## Expected gain

The idea's own Δμ is zero: it moves no term, it replaces the `[30, 100]` ps interval on `c_c` with a calibrated one before the GPU exists. What it buys is a decision: whether the SC-2 ceilings hold under the counted `c_c`, a year before RFC-0020's gate would say so.

```sh
python3 -I research/tools/cost_model.py            # the idea itself: no parameter moves
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | 0 (E0: no term moved) | 0 | 0 |
| 4K | 0 (E0) | 0 | 0 |
| 16K | 0 (E0) | 0 | 0 |
| 128K | 0 (E0) | 0 | 0 |

The exposure the calibration resolves, as cost-model scenarios (E0, primary, `research/tools/cost_model.py`, corrected profile as baseline; the scenario values are the lower and upper ends of the E0 count above):

```sh
python3 -I research/tools/cost_model.py --set commit_ps=60
```

```text
scenario: commit_ps=60
   S   linear  attention  weights  regen  streaming    total
   1K     4.72       1.55    0.021    4.0       0.42     10.7   Δ  +20.6%
   4K     4.65       6.10    0.021    4.0       2.32     17.1   Δ  +21.5%
  16K     4.38      22.98    0.020    4.0      11.24     42.6   Δ  +21.2%
 128K     2.85     119.49    0.013    4.0      77.93    204.3   Δ  +19.2%
```

```sh
python3 -I research/tools/cost_model.py --set commit_ps=90
```

```text
scenario: commit_ps=90
   S   linear  attention  weights  regen  streaming    total
   1K     6.13       1.96    0.021    4.0       0.42     12.5   Δ  +41.2%
   4K     6.04       7.73    0.021    4.0       2.32     20.1   Δ  +43.1%
  16K     5.69      29.13    0.020    4.0      11.24     50.1   Δ  +42.5%
 128K     3.70     151.49    0.013    4.0      77.93    237.1   Δ  +38.3%
```

`commit_ps=100` (the 60%-utilization end) gives +48.1% / +50.2% / +49.6% / +44.7% and `μ(16K) = 52.6`. At `commit_ps = 90` the 16K projection (50.1) and at 60 the 1K projection (10.7, ceiling 12) sit against the SC-2 ceilings of 12 and 45 (00 §7): `ρ_c ∈ (2, 4]` is RFC-0020's "proceed with re-plan" branch, which pulls OQ-0004-1, NG-5 and multi-variable rounds into scope. Inputs: `c_c` scenario values E0 (own count); the rest of the model E0 as in baseline §2; the CPU figures E1 (this machine) enter only through the fit, never as a projection.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None: the protocol, the code rate, `τ` and the hash are unchanged; the idea measures the specified commitment and changes no parameter | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None: no relation, table or witness changes | pass |
| Zero knowledge (G-5, RFC-0013) | None: the masking rows (`P_A`) stay in the encoder as specified; the measurement runs on synthetic rows from a seeded generator (rule 6), so no secret enters a report | pass |
| Verifier independence (SP-6) | None: prover-side measurement; nothing under `verifier/` is touched | pass |
| Post-quantum (hash-based commitments) | None: BLAKE3 chained digests and the Merkle tree stay | pass |

Two levers examined under this lens and found closed, for the record (not part of the idea): small-value commitments (NG-5) have no sound instantiation in Goldilocks without a field or commitment change, because integer packing needs the digits bound by their own commitment (no subfield for ring switching; pay-per-bit needs a norm-aware lattice commitment, [DBK+26] via SUR-0002, outside the hash-based stance of [baseline.md §1](../baseline.md#1-what-an-improvement-must-preserve)); and OQ-0004-1's "hash only the systematic symbols" is unsound for the Ligero test, since a column check restricted to message positions is satisfied by every matrix. A rate above 1/2 (rate 2/3 on a `3·2^21` domain, `3 | p − 1`) buys about −20% of `c_c` for +60% of `τ_A` and about +0.4 GB of proof under unique decoding (E0, `research/tools/proximity.py`), a bad trade on the scoreboard.

## Prior rejection check

- [RFC-0020 Alternatives Considered 2](../../docs/rfcs/RFC-0020-cost-model.md#alternatives-considered) rejects an *analytical roofline only*, because "NTT, hashing and fraction-tree efficiency depend on implementation details a roofline cannot predict". This idea is the roofline plus a same-harness CPU calibration of the counts on the code that exists, feeding the gate instead of replacing it. RFC-0020 Alternatives Considered 1 rejects *waiting* for end-to-end numbers for the same reason this idea exists: "discovering then that a unit cost is 5× off would waste the GPU work in between".
- [baseline.md §7](../baseline.md#7-already-rejected): no row covers a calibration; the hash row (SHA-256, SHA-3, BLAKE2s, Poseidon2) reopens on "GPU measurements show a field-native hash within the 30 ps budget", which this idea would inform, not reopen.
- SUR-0002 'Candidate ideas' proposes the same method for `c_ℓ`; nothing in the corpus has applied it to `c_c`. New now: the first E1 figures for the specified encoder and chain (above), and SUR-0002's demonstration that the `c_ℓ` assumption was 2–3× low under the same counting method.

None rejects it, checked.

## Kill criterion

A primary H100 measurement of a batched Goldilocks NTT of 2^23 points plus BLAKE3 over 16 bytes per element at or below 30 ps per committed element (for example published ICICLE or sppark Goldilocks NTT figures on an H100 combined with a measured BLAKE3 kernel at 16 B per element), which would make the calibration moot; or a CPU-to-count fit that fails on the kernels already measured (the butterfly or byte counts do not explain the ARM timings within 2×), which would void the projection method.

## Cheapest experiment

Hypothesis: the counted `c_c` of the commitment as specified lies in `[57, 119]` ps on an H100 (E0), that is `ρ_c ∈ (1.9, 4]`, and the CPU timings of the existing encoder and chain are explained by the butterfly and byte counts within 2×.

Method, effort `s` (days, no RFC), E1 reachable here: extend today's harness (the `tessera-pcs` throughput bench plus the `ColumnChains` scratch program) with the op-count script (butterflies, bytes, compressions per element, int32 instructions per Goldilocks multiplication read from the generated code of `tessera_field::Goldilocks::mul`), the `B` sweep already run, `Fp` against `Fp2` rows (the `G + 1 ≤ 17` combined rows of the opening pass, [RFC-0011 'Opening protocol'](../../docs/rfcs/RFC-0011-advice-commitment.md#proposed-design)), and thread scaling to 12 cores; report ns per committed element per kernel, the encoder:hash split, the fit residuals and the projected interval with both utilization ends. Baseline and candidate in the same harness (rule 4): there is no candidate here, the baseline is the specified kernel and the pre-registered success criterion is the fit.

Unknown until an H100 run (E2, the RFC-0020 v0.1 gate): the achieved int32 utilization of a Goldilocks radix-2/radix-16 NTT and of BLAKE3 at 288-byte inputs, the degree of fusion the real kernel reaches (48 against 216 bytes per element), and whether the encoder or the hasher binds first; the interval is designed to contain those.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The idea measures the commitment exactly as specified (inverse NTT, coset NTT, B = 32 chains) and changes no parameter, message, challenge or verifier check, so there is nothing for ADV-P or ADV-V to attack. The measurement runs on seeded synthetic rows (research rule 6), so no secret-class value reaches a report (05 §3). Its two asides are correct as stated: hashing only systematic symbols is unsound for the Ligero column test (every matrix satisfies a check restricted to message positions), and a rate above 1/2 lowers the per-query bits of RFC-0014, which the note rejects on cost rather than proposing. |
| prior art | survives | RFC-0020 Alternatives Considered 2 rejects an analytical roofline alone and Alternative 1 rejects waiting for end-to-end numbers; nothing in RFC-0020's protocol (H100 microbenchmarks only, 'compared with the CPU reference') or elsewhere schedules a CPU-side calibration of c_c, and baseline.md §7 has no row on it. The 30 ps figure is an unmeasured assumption (08 §2 'assumed'; RFC-0020), so a calibrated interval is not already in the corpus. Prior art for the method is the corpus's own SUR-0002 [E0-this] applied to c_ℓ; applying it to c_c with the first E1 numbers on the real encoder and `ColumnChains` is the new part. It moves no term (Δμ = 0 by construction), so it competes as a decision input, not as a design improvement, and is scored on the SC-2 exposure it resolves. |
| cost model | survives | The idea moves no term; the lens question is whether the uncertainty it removes matters: `commit_ps=60/90/100` rerun to +20.6 / +41.2 / +48.1% at 1K and +21.2 / +42.5 / +49.6% at 16K, so the [30, 119] ps interval alone decides whether μ(16K) sits at 35 or 53 against the SC-2 ceiling of 45 (baseline.md) and which branch of RFC-0020's decision rule fires (ρ_c ∈ (2, 4]). c_c is 19–21% of μ at every context, the second-largest unit-cost uncertainty after c_ℓ, and no survey or idea in the corpus has a primary GPU figure for a Goldilocks NTT plus BLAKE3 chaining. Inputs are labeled honestly (E1 M4 timings enter only through the fit, the 30 ps is marked `claim`). The one caveat: the fit validates butterfly and byte counts, not the GPU instruction cost per butterfly, which the note states; the projection interval is wide by design, which is fine for a calibration whose deliverable is the interval. |
| feasibility | survives | Every kernel the calibration measures exists and runs here: `cargo bench -p tessera-pcs --bench throughput` (`crates/tessera-pcs/benches/throughput.rs`), `tessera_hash::ColumnChains::absorb_block` (`crates/tessera-hash/src/columns.rs`) and `Goldilocks::mul` via `reduce128` (`crates/tessera-field/src/goldilocks.rs`) for the instruction count. No spec surface changes, no RFC, effort `s`, E1 reachable as claimed. Two caveats: the ARM-to-int32 translation is a hand count (ARM64 does a 64-bit multiply in one `mul`/`umulh` pair, so the fit validates butterfly and byte counts only), and the 12-core scaling point needs a parallel harness the single-core bench lacks. The fit's success criterion (counts explain ARM timings within 2×) is loose enough that the method is unlikely to be voided, which limits how much the E1 adds over the E0 count. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| Δμ of its own: 0. Scoreboard projection it resolves: μ(16K) +21% to +48% (`commit_ps` 60 to 100; the central E0 count of 88 ps gives +41%), i.e. whether μ(16K) is 35 or 53 against the SC-2 ceiling of 45 (E0 count on H100; E1 fit on one M4 Max core: 32.85 ns encode + 19.55 ns chain per committed element) | 0.6 → 0.5 after EXP-0002: the counts are validated (E1 fit ratios 0.83–1.00 on four kernels, primary), but the GPU int32 translation spans 1.6× by itself (969 to 1,574 ops per element) and the utilization another 1.67×, so the delivered interval `[58, 157]` ps covers all three RFC-0020 bands; a CPU calibration cannot narrow it further | `s`: spent (EXP-0002, one session); what remains is E2 | Rubric Δμ reading: 0. Decision-value reading after EXP-0002: the exposure is half resolved: the 30 ps assumption is excluded at every point of the interval (ρ_c ≥ 1.9) and the SC-2 ceiling at 16K fails above 70 ps, which every point of the interval but the unreachable floor exceeds; the branch (re-plan or redesign) stays with the v0.1 H100 run. Priority as an experiment here: 0 (nothing left that E1 can settle); as an input to RFC-0020's gate: the interval is the planning value to carry |

## Decision and links

Triage 2026-10-16: `triaged`; highest-ranked experiment of the batch on the decision-value reading, with IDEA-0008 as its c_π twin. Next step: the pre-registered E1 calibration (op-and-byte count fitted to the `tessera-pcs` bench and `ColumnChains`), carrying IDEA-0003's byte table as a line item.

Experiment 2026-10-16: [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md), `done: inconclusive` against its pre-registered criterion. What it settled (E1, primary, one M4 Max performance core): the operation counts explain the production kernels within 2× (predicted/measured 0.997 for `intt` 2^22, 0.826 for `coset_ntt` 2^23, 0.975 for `encode_row`, 0.834 for the `B = 32` chain at 2^23 columns), so the kill criterion's "fit fails" clause did not fire; `encode_row` 32.09 ns and the chain 19.22 ns per committed element reproduce the triage figures within 3%. What it delivered (E0, primary): `c_c ∈ [58, 157]` ps per committed element on an H100, `ρ_c ∈ [1.9, 5.2]`, the floor being a radix-16, lazily reduced kernel at 100% of the int32 peak and the ceiling the radix-2 code as written at 60%; the hash is 19–26% of the count, the NTTs the rest; HBM bytes (48–216 per element, 14–81 ps) never bind. Why `inconclusive` and not `validated`: the floor, 57.9 ps, is in the `(30, 60]` band of the criterion, so the counts alone cannot say whether RFC-0020's "proceed" branch is excluded, although the 30 ps assumption is. Status `inconclusive`; the next evidence is E2 (the v0.1 gate, or any primary H100 figure for a batched Goldilocks NTT of 2^23 points plus BLAKE3 at 288-byte inputs). RFC or issue: none; the interval should enter RFC-0020's gate as the planning value for `c_c`, which is a consolidation decision. Ledger line updated on: pending consolidation.
