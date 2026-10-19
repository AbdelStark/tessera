# IDEA-0024: Tensor-core limb-sliced radix-16 coset NTT for the advice encoder (INT8 MMA products, exact SIMT recombination)

| | |
|---|---|
| Status | `rejected` (2026-10-17, triage: refuted at E0 on the cost-model and feasibility lenses by the note's own consistent recount; see Decision and links) |
| Origin | [SUR-0011](../surveys/SUR-0011-field.md) finding [LZG+26] (TensorZKP: sumcheck and encoder recast as matrix products on tensor cores over a 32-bit prime) and its "Contested or open" item "whether tensor-core arithmetic can take a 64-bit field at all"; [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md) `count.py` (the whole width of `c_c ∈ [58, 157]` ps is radix and reduction strategy, E0); [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels) row "Limb-decomposed folds" and [RFC-0012 'Opening by the folding pass'](../../docs/rfcs/RFC-0012-weight-commitment.md#opening-by-the-folding-pass) step 1, which already split `Fp2` coefficients into 8-bit limbs for INT8 GEMMs with exact recombination, for folds but not for the encoder; [IDEA-0015](IDEA-0015-lazy-radix-encoder-ntt.md), whose lazy radix-16 count is the SIMT floor this idea tries to undercut; RISK-1 and RISK-8 ([00 §9](../../docs/spec/00-overview.md#9-risks)). New as a design idea |
| Lever | `c_c` (the encoder's share, 74–81% of the committed element by operation count: 721 of 969 int32 ops at the lazy radix-16 floor, 1,274 of 1,574 as written, E0 from EXP-0002): moves the in-stage DFT products of every NTT stage from the int32 pipe to the INT8 tensor pipe and leaves recombination, reduction and the inter-stage twiddle on SIMT. Secondary: the int32 pipe it would free is the pipe the BLAKE3 chains run on (RISK-8) |
| Spec surfaces | None normative. A prover kernel under [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels), row "Coset NTT (sizes 2^10–2^23) and RS encode with vanishing mask" (which demands canonical output); [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding) (code, coset `7·L`, order) unchanged; [RFC-0002 'Representation and reduction'](../../docs/rfcs/RFC-0002-field-arithmetic.md#representation-and-reduction) already grants GPU kernels unreduced intermediate ranges with canonicalization before hashing. Not tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)); no RFC |
| Priority | 0 (rejected; the consistent recount gives Δμ(16K) +0.9% at peak INT8 throughput to +21.5% at 60% utilization under `counted`, a loss at every context, E0) |

## Mechanism

Prover kernel only; the codeword is bit-identical to today's `RowEncoder::encode_row` (`crates/tessera-pcs/src/encoding.rs`, `ntt.rs`) after canonicalization. Five parts.

1. **Stage decomposition.** The size-`2^23` coset NTT (the coset shift `7^i` folded into the coefficient scaling as `RowEncoder`'s `spread` does, so no shifted twiddles) runs as five radix-16 stages and one radix-8 stage (`5 × 4 + 3 = 23` levels, E0, checked). The size-`2^22` inverse NTT that precedes it in the encoder runs as five radix-16 stages and one radix-4 stage (`22 = 5 × 4 + 2`). Per committed element the encoder therefore does `5.5` point-stages on the one `2^22` point and `2 × 5.75` on the two `2^23` symbols: **17 point-stages** (E0; the quantity every per-stage count below is multiplied by).
2. **The in-stage DFT as an INT8 MMA.** In a radix-16 stage, the 16-point DFT of every group is one constant `16 × 16` matrix `T` (`T_{ij} = ω_16^{ij}` mod `p`; the entries are powers of `2^12` but are treated as arbitrary 64-bit constants) applied to a `16 × N` operand matrix whose `N` dimension runs over the groups of a row and the 32 rows of the block buffer (`N ≈ 2^19 × 32 = 2^24` per stage, so the MMA tiles are full). Every 64-bit operand and every constant is split into eight signed 8-bit limbs (the operand split is a byte view of the word: no instructions; balanced signed limbs keep every partial product below `128^2`). The 64 limb-pair products are INT8 × INT8 → INT32 MMAs (`mma.sync` / `wgmma` m16n8k16, or the CUTLASS equivalent, as RFC-0018 Alternative 5 allows for fused limb-decomposed work), with the products of equal byte offset `k = i + j` accumulated into one INT32 accumulator: 15 accumulators per output element, offsets 0 to 14. **Accumulator bound:** at most 8 limb pairs per offset × 16 terms × `128^2 = 2^21 < 2^31` (E0, checked by Python this session), so INT32 accumulation is exact and order-independent, as [00 §4](../../docs/spec/00-overview.md#4-two-modes-one-circuit) and RFC-0018 'Witness engine' already rely on for the INT8 GEMMs of `D`. MACs per output element per stage: `16 × 64 = 1,024` (radix-16), 512 (radix-8), 256 (radix-4); per committed element `(5 × 1,024 + 256) + 2 × (5 × 1,024 + 512) = 16,640` (E0, this session; the generation pass's 5,632 counted one `2^23` output point only).
3. **SIMT per output element per stage.** Combine the 15 accumulators at 8-bit offsets into a 133-bit integer (three 64-bit words; about 45 int32 ops: a shift pair and two adds with carry per accumulator), reduce 133 → 64 bits with `2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`, `2^128 ≡ −2^32` (mod `p`) (E0, the three identities checked by Python this session; about 15 ops), multiply by the inter-stage Cooley–Tukey twiddle (one full multiplication per non-leading output, `15/16 × 18 ≈ 17` ops, fully reduced as in EXP-0002's lazy low end), store (about 4 ops): **about 81 int32 ops per output element per stage** (E0, hand count, no SASS).
4. **Canonicalize** every symbol below `p` before `ColumnChains` hashes it (INV-TCE-1), exactly as today; control flow data-independent (RFC-0002 'Timing', INV-ARCH-2): every correction a mask or select.
5. **Optional refinement, not counted:** in the first stage of the inverse NTT the message limbs of INT8 and INT20 values are mostly zero, so that stage's MMA uses one to three data limbs.

Memory: with three stages resident in shared memory per group (`16^3 = 4,096` points, 32 KiB), each NTT is two HBM passes of 16 B per point; about 96 B per committed element, 29 ps at the HBM3 peak of `3.35 × 10^12` B/s (E0), below the compute figures below, as in EXP-0002.

## Why it could work

- **E0, primary** ([EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md) `count.py`, counts fitted to E1 within 0.83–1.00 on four production kernels): the encoder is 721–1,274 int32 ops per committed element and the hash 248–300; the int32 pipe is the only pipe the counted kernel uses, so any work that leaves it is a direct saving if what remains is smaller.
- **E0, secondary** (vendor documentation): H100 SXM dense INT8 tensor rate 1,979 TOPS (`989.5 × 10^12` MAC/s), the rate [08 §1](../../docs/spec/08-performance-budget.md#1-cost-model) defines the native term `R` on, against `1.67 × 10^13` int32 lane-ops/s: a 59× ratio. The 16,640 MACs per committed element of part 2 cost 16.8 ps at peak and 33.6 ps at 50% utilization (E0, this session), which would hide under a SIMT remainder of 45–80 ps if that remainder were the only other cost. In a fused P1 pass they add tensor-pipe load next to the native forward's own INT8 GEMMs (`native(16K) ≈ 0.74` ms per token, E0 from the cost model).
- **E1, primary via SUR-0011** ([LZG+26], abstract only; GPU and field unnamed on the landing page, a 32-bit prime per the search index): tensor-core inner products at 25 ps per pair and degree-2 sumcheck at 120 ps per term at `2^25`, 36.2× over SIMT implementations: the recast works for a prime that fits the accumulator. A 64-bit field needs the limb split, which [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels) and [RFC-0012](../../docs/rfcs/RFC-0012-weight-commitment.md#opening-by-the-folding-pass) step 1 already specify for folds with exact recombination (primary, the specification). Caution from the same finding: TensorZKP's Spielman encoder costs 345 ps per element, 3.7–11× Tessera's `c_c` range; that encoder is sparse and not matrix-shaped, which a radix-16 DFT stage is.
- **E1, primary on another GPU via SUR-0005** ([Uulong26]): the only public Goldilocks GPU NTT runs at 10.6–17.3 ps per radix-2 butterfly and is memory-bound at large sizes; with 34 butterflies per committed element that is 360–590 ps, far above the counted floor, so the SIMT path has room and another pipe is worth pricing.
- **`claim`, outside the bibliography (to be added and read):** NTT over 32- to 64-bit moduli on INT8 tensor cores by limb decomposition is reported in the fully homomorphic encryption literature (TensorFHE, HPCA 2023); no Goldilocks or Reed–Solomon-encoding instance is known to this note.
- **Against, E0, this session (the consistent recount).** EXP-0002's lazy radix-16 structure costs, per 16 points and one radix-16 stage, 15 full twiddle multiplications (18 ops), 17 cheap multiplications by `2^(12j)` (8 ops), and 32 butterflies' 32 additions and 32 subtractions (4 ops each): 662 ops, **41.4 per point per stage**; times 17 point-stages that is 703, matching `count.py`'s 721 with the spread multiplication. The MMA variant replaces the cheap multiplications and the butterfly adds (24.5 ops per point-stage) with recombination and reduction (about 60) and keeps the twiddle (17) and the store: about 81 per point-stage, **1.96× the lazy SIMT floor**. The generation pass's figure of 492 ops "against 721" compared 6 stages of one `2^23` point with 17 point-stages per committed element; on one unit the idea does not undercut IDEA-0015 by operation count. It breaks even only if recombination plus reduction fit in about 20 int32 ops per point-stage (E0, `41.4 − 17 − 4`), a third of the hand count.

## Expected gain

The cost model's knob is `commit_ps`. Baseline: the `counted` profile (`c_c = 94` ps, EXP-0002's radix-2 count at the int32 peak; [baseline.md §2](../baseline.md#2-the-cost-model)), since under `corrected` the assumed 30 ps is already below the counted floor.

**Scenario A, the generation pass's derivation (E0, unit-inconsistent, kept for the record).** Encoder 492 ops (82 × 6 stages, one `2^23` point) plus hash 248–300 gives 740–792 ops: 74–79 ps at 60% utilization, against 94–157 for 969–1,574 ops; it set `commit_ps` 72 (conservative), 64 and 45 (the −50% reading against the centre). Run 2026-10-17:

```sh
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=72
```

```text
baseline: profile 'counted'
   S   linear  attention  weights  regen  streaming    total
   1K     9.05       3.70    0.021    4.0       0.42     17.2
   4K     8.91      14.59    0.021    4.0       2.32     29.8
  16K     8.40      54.98    0.020    4.0      11.24     78.6
 128K     5.46     285.91    0.013    4.0      77.93    373.3

scenario: commit_ps=72
   S   linear  attention  weights  regen  streaming    total
   1K     8.03       3.40    0.021    4.0       0.42     15.9   Δ   -7.7%
   4K     7.91      13.39    0.021    4.0       2.32     27.6   Δ   -7.4%
  16K     7.45      50.47    0.020    4.0      11.24     73.2   Δ   -6.9%
 128K     4.84     262.44    0.013    4.0      77.93    349.2   Δ   -6.5%
```

`--set commit_ps=64`: Δμ −10.5% / −10.1% / −9.5% / −8.8% at 1K / 4K / 16K / 128K; `--set commit_ps=45`: −17.2% / −16.4% / −15.5% / −14.4% (E0, run 2026-10-17). Under `corrected` with `--set commit_ps=20` (the same −33% applied to the assumed 30 ps): −6.8% / −7.2% / −7.1% / −6.4%.

**Scenario B, the consistent recount (E0, this session; the note's central estimate).** 81 SIMT ops per point-stage × 17 point-stages + spread = 1,403 encoder ops, + 248 hash = 1,651 ops per committed element: 98.9 ps at the int32 peak, 164.8 ps at [VYTA25]'s 60% utilization; the tensor pipe's 16.8–33.6 ps runs concurrently and is hidden. Against the counted centre (94 ps, radix-2 at peak) and the centre's own utilization reading, run 2026-10-17:

```sh
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=97
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=162
```

```text
scenario: commit_ps=97
   S   linear  attention  weights  regen  streaming    total
   1K     9.19       3.75    0.021    4.0       0.42     17.4   Δ   +1.1%
   4K     9.05      14.75    0.021    4.0       2.32     30.1   Δ   +1.0%
  16K     8.53      55.60    0.020    4.0    11.24     79.4   Δ   +0.9%
 128K     5.54     289.11    0.013    4.0    77.93    376.6   Δ   +0.9%

scenario: commit_ps=162
   S   linear  attention  weights  regen  streaming    total
   1K    12.21       4.64    0.021    4.0       0.42     21.3   Δ  +23.8%
   4K    12.02      18.29    0.021    4.0       2.32     36.7   Δ  +22.8%
  16K    11.33      68.94    0.020    4.0    11.24     95.5   Δ  +21.5%
 128K     7.36     358.46    0.013    4.0    77.93    447.8   Δ  +19.9%
```

The like-for-like comparison the model cannot express directly: against IDEA-0015's lazy radix-16 kernel at the same utilization (969 ops: 57.9 ps at peak, 96.5 at 60%), scenario B is **+70% of `c_c`** (1,651 / 969, E0). If recombination plus reduction cost 24 ops instead of 60 (an optimistic SASS), the total is 1,294 ops (+34% against lazy radix-16; 77.5 ps at peak, Δμ(16K) about −2% against the counted centre); only below about 20 ops per point-stage does the idea beat the SIMT floor.

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | +1.1% at peak, +23.8% at 60% (scenario B, E0); −7.7% to −17.2% claimed by scenario A (E0, unit-inconsistent) | 0 (E0: same codeword, same `τ_A`) | 0 (E0: verifier unchanged) |
| 4K | +1.0% / +22.8% (B); −7.4% to −16.4% (A) | 0 | 0 |
| 16K | +0.9% / +21.5% (B); −6.9% to −15.5% (A) | 0 | 0 |
| 128K | +0.9% / +19.9% (B); −6.5% to −14.4% (A) | 0 | 0 |

Inputs: the per-operation int32 figures and the 17 point-stages are E0 hand counts (EXP-0002's method, fitted to E1 on one M4 Max core for the SIMT kernels; the recombination and reduction counts of part 3 are this note's own, no SASS); the H100 int32 and INT8 peaks are `secondary` (documentation); the 60% utilization is [VYTA25]'s E1 figure for another field on an A40; the tensor-pipe time is E0 at a `secondary` peak. Proof size and verification do not move: the commitment bytes are identical.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None. The codeword is the same `RS[n, k + P_A]` on `7·L`, symbol for symbol: INT8 × INT8 → INT32 MMA is exact, the accumulator bound `2^21 < 2^31` is proven (part 2), the 133-bit recombination and the Goldilocks reduction are exact integer identities (part 3), and every symbol is canonical before hashing (INV-TCE-1). A kernel bug changes the codeword and fails the column check (RFC-0011 step 8.3) or the TCE decode: completeness failures for the honest prover, never an acceptance. Enforced by the bit-exact differential test against `NttPlan` / `RowEncoder` and `spec/vectors/encoding/encoding.json` | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None: no relation, table, witness or range changes; the encoder is downstream of the committed values and deterministic | pass |
| Zero knowledge (G-5, RFC-0013) | None: mask coefficients enter the same positions above `k` from the same stream in the same order; the kernel is deterministic, INT32 accumulation is order-independent, and control flow depends on lengths and indices only (INV-ARCH-2), so no timing channel is added | pass |
| Verifier independence (SP-6) | None: the verifier never runs this kernel; `tv-field`'s own NTT re-encodes the 17 combined rows of RFC-0011 'Encoding' unchanged; nothing is copied in either direction; the research harness depends on prover crates only (research rule 2) | pass |
| Post-quantum (hash-based commitments) | None: BLAKE3 column chains and the Merkle root are untouched | pass |

No floating point anywhere (INV-DM-6, INV-ARCH-2: the FP16, TF32 and FP64 DMMA variants are excluded by design; the DMMA route with three 22-bit limbs and 9 FMAs per product would cost 25–50 ps per element at 67 TFLOPS and would not be hidden, E0). Redaction (05 §3, INV-ARCH-6): the kernel touches `Secret` activations exactly where today's encoder does, through plain `&mut [Fp]` below the `Secret<T>` boundary (`expose_secret` named only in `crates/tessera-codec/src/secret.rs`, per `secret-access.toml`, as IDEA-0015 checked on 2026-10-16); no new logging surface; experiments run on seeded synthetic rows (research rule 6). Unsafe code: a CUDA kernel lives in `tessera-gpu`'s permitted modules with `// SAFETY:` comments; the research workspace has none.

## Prior rejection check

- [baseline.md §7](../baseline.md#7-already-rejected) and [RFC-0011 Alternatives Considered 3](../../docs/rfcs/RFC-0011-advice-commitment.md#alternatives-considered) keep the RS/NTT encoder ("NTT is fast on GPU") against Brakedown codes; this idea keeps the code and changes only the pipe that computes it, so the row supports rather than covers it.
- [RFC-0002 Alternatives Considered](../../docs/rfcs/RFC-0002-field-arithmetic.md#alternatives-considered) (Montgomery form, extension degree, other fields) decide representation, not the execution unit; RFC-0002 'Representation and reduction' permits unreduced intermediates on GPU with canonicalization before hashing, which part 3's 133-bit intermediate uses.
- [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels) lists the coset NTT as a custom kernel and reserves limb-decomposed INT8 GEMMs for folds; [RFC-0018 Alternative 5](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#alternatives-considered) keeps CUTLASS open for fused limb-decomposed work. Nothing rejects the technique for the encoder.
- Adjacent ideas: [IDEA-0015](IDEA-0015-lazy-radix-encoder-ntt.md) is SIMT-only (radix and lazy reduction; `triaged`, priority 1.5) and is the floor this note is measured against; [IDEA-0003](IDEA-0003-fused-encode-hash-tiles.md) is the byte side (fusion, L2 residency; `shelved`) and composes with this one, the stage's operand matrix being its tile. SUR-0011 recorded the 64-bit tensor-core question as open and [LZG+26]'s 345 ps Spielman figure as a caution.

None rejects it, checked. What is new: a priced answer to SUR-0011's open question for the one matrix-shaped kernel of the commitment, and the finding of this note's own recount that the SIMT remainder, not the tensor work, decides it.

## Kill criterion

Any of: (a) a SASS count of the per-stage SIMT part (15-accumulator recombination, 133 → 64-bit reduction, inter-stage twiddle, store) above about 41 int32 ops per output element per stage, the lazy radix-16 floor of EXP-0002, which this note's own hand count (about 81) already exceeds: the idea is then refuted at E0 unless a recombination below 20 ops is exhibited; (b) the MMA fragment layout forcing a shared-memory transpose per stage whose instructions exceed the ALU saved, or INT8 tensor utilization in the stage below 20%, so that 16.8 ps becomes 84 ps and is no longer hidden; (c) the RFC-0020 v0.1 gate measuring a SIMT encoder at or below 45 ps per committed element; (d) the fused P1 pass's tensor pipe already saturated by witness regeneration at the measured utilization.

## Cheapest experiment

Hypothesis (pre-registered here): a limb-sliced radix-16 stage can be written so that its SIMT remainder (recombination, reduction, twiddle, store) is at most 20 int32 ops per output element per stage; if it cannot, the idea is refuted by count.

- **E0 here, effort `s` (a day):** extend EXP-0002's `count.py` with the limb-sliced stage on the consistent per-committed-element unit (17 point-stages; MACs, SIMT recombination, reduction and twiddle ops, HBM bytes per stage), try the recombination variants (fewer accumulators by widening offsets to 16 bits where the bound `2^21` allows summing adjacent offsets before the shift; reduction folded into the shift-adds), and re-run `cost_model.py` at the resulting `commit_ps`. This settles kill criterion (a).
- **E1 here, correctness only, effort `s` (days):** a bit-exact CPU emulation in a scratch Cargo workspace (empty `[workspace]`, research rule 2) over `tessera-field` and `tessera-pcs`: eight-limb products accumulated in `i32`, recombined in `u128` or three `u64` words, reduced by the three identities, twiddled, canonicalized; checked symbol for symbol against `NttPlan::coset_ntt`, `NttPlan::intt` and `RowEncoder::encode_row` on seeded rows (SplitMix64, EXP-0002's seeds) and against `spec/vectors/encoding/encoding.json`, with the accumulator maximum recorded against the `2^21` bound. No timing claim: an M4 has no INT8 MMA of this shape.
- **E2 (the gain itself, H100, RFC-0020 v0.1 runner, effort `l`):** IDEA-0015's SIMT radix-16 kernel against the MMA stage on the same block buffer, Nsight int32-pipe and tensor-pipe utilization recorded. Pre-registered prediction from scenario B: the MMA kernel is slower per committed element than the lazy SIMT kernel at equal utilization (1,651 against 969 int32 ops); the experiment runs only if the E0 step finds a remainder below 20 ops.

Unknown until an H100 run: achieved tensor and int32 utilization in one fused stage, fragment-layout transposes, and whether the hash or the encoder binds first.

## Refutations

Sweep triage of 2026-10-17: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Prover-only kernel: the verifier never runs it and re-encodes the combined rows itself (RFC-0011 'Sizes' verifier-work row, docs/rfcs/RFC-0011-advice-commitment.md:105), so any arithmetic error is a completeness failure at the column check, never an acceptance. The exactness claims hold: 8 limb pairs × 16 terms × 128² = 2^21 < 2^31, and the identities 2^64 ≡ 2^32 − 1, 2^96 ≡ −1, 2^128 ≡ −2^32 (mod p) were re-checked in Python (E0, primary); canonicalization before hashing is the RFC-0002 grant (docs/rfcs/RFC-0002-field-arithmetic.md:88). Mask coefficients enter the same positions through a linear map and control flow depends on sizes only, so INV-ZK-1 and INV-ARCH-2 are untouched; no second witness, challenge order, leak or trust surface is created. Cost note outside this lens: a balanced signed limb split of a 64-bit operand needs borrow propagation, not a free byte view. |
| prior art | weakened | Nothing in the RFCs rejects a tensor-core encoder: RFC-0018:10 and the 'Kernels' row (RFC-0018:58) assign the coset NTT to custom kernels, source principle 10 (RFC-0018:14) says to run matrix-shaped work on tensor cores, RFC-0018:64 and RFC-0012:51 already specify limb-decomposed INT8 GEMMs with exact recombination for folds, and baseline §7 has no execution-unit row (verified). As a technique it is not new: INT8-tensor-core NTT by limb decomposition is the TensorFHE line of work (HPCA 2023), which the note lists unread as `claim`, and SUR-0011 already declined to seed from the LZG+26 line on exactly this ground ('a 64-bit field does not fit tensor-core accumulators without limbs, and the per-element figures do not beat the budget', research/surveys/SUR-0011-field.md:74) while leaving the 64-bit question open at line 65. New is only the priced application to the radix-16 DFT stage, and the note's own consistent recount (81 vs 41.4 int32 ops per point-stage, 1.96× the IDEA-0015 SIMT floor, IDEA-0024:30) says the prior dismissal was right at E0; the idea survives this lens only as the answer to SUR-0011's open question, not as a design candidate. |
| cost model | refuted | The note's own consistent recount (lines 30, 60-83) refutes it: the SIMT remainder of a limb-sliced stage is about 81 int32 ops per point-stage against the 41.4 of EXP-0002's lazy radix-16 floor, so the encoder costs 1,651 ops per committed element against 969 (+70% of `c_c` against IDEA-0015) and the rerun gives Δμ(16K) +0.9% (97 ps at peak) to +21.5% (162 ps at 60% utilization) against the `counted` centre, a loss at every context. The lazy floor re-derives from EXP-0002 count.py:121-123 (15 full twiddles, 17 cheap multiplications, 64 add/sub per 16 points per stage = 662 ops = 41.4 per point-stage; × 17 point-stages + spread = 721, matching README:90), so the recount is consistent; the only gain scenario (A, −7.7% to −17.2%) compares 6 stages of one 2^23-point transform with 17 point-stages per committed element and is unit-inconsistent, as the note says. The tensor work (16.8-33.6 ps) is hidden only because the SIMT remainder exceeds the whole SIMT kernel it replaces; combining 15 INT32 accumulators at 8-bit offsets into a 133-bit integer and reducing it cannot plausibly fit the 20 ops the break-even needs (15 shift-add-with-carry pairs alone exceed it). `c_c` is 30% of μ(16K) under `counted` (baseline §2), so the term matters, but the idea moves it the wrong way; evidence levels are stated honestly (E0 hand counts, secondary peaks, E1 utilization from another field). |
| feasibility | refuted | The only measurement that can show the gain is E2 on an H100 (effort `l`): an M4 has no INT8 MMA of the m16n8k16 shape, so the note's own E1 step is correctness-only, and the E0 step it proposes can only confirm what the note already computed. That recount (mechanism part 3) puts the SIMT remainder at about 81 int32 ops per output per stage against the 41.4-op lazy radix-16 floor of EXP-0002 `count.py` (`ops_low`, research/experiments/EXP-0002-commit-cost-calibration/count.py:120), which trips kill criterion (a) at E0 (scenario B: +70% of `c_c`, Δμ(16K) +0.9% to +21.5%). The target the kill clause leaves open, recombination plus reduction under 20 ops, is unreachable by count: folding 15 INT32 accumulators at 8-bit offsets into a 133-bit integer needs at least 14 add-with-carry steps on 32-bit lanes (about 28 ops even after pairing adjacent offsets, which the 2^21 bound allows) before the roughly 15-op reduction. Nothing touches a tier-A surface (prover kernel under RFC-0018 'Kernels', docs/rfcs/RFC-0018-gpu-prover-runtime.md:58); the refutation is that the cheapest experiment cannot reach the claimed evidence and the E0 evidence already available says the idea loses (E0, primary: own arithmetic over the note's part 3 and EXP-0002's count). |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| +0.9% μ(16K) at peak INT8 throughput to +21.5% at 60% utilization under `counted` (a loss at every context; E0 own recount of the SIMT remainder, 81 vs 41.4 int32 ops per point-stage, `commit_ps` 97-162 against 94); the generation pass's −7% to −10% rests on a unit mismatch (6 stages of one point against 17 point-stages per committed element); the tensor work itself 16.8-33.6 ps (E0 on secondary H100 peaks) | 0.6 that the loss is real: the recount is consistent with EXP-0002's E1-fitted count and the carry-chain lower bound (about 28 ops against a 20-op break-even) is arithmetic; an H100 (E2) could change the utilization, not the op count | `s` for the E0 count extension, `l` (an H100) for the only measurement that could overturn it; no RFC (prover kernel under RFC-0018 'Kernels') | 0: rejected on the hard gate (an E0 impossibility for the mechanism as stated, confirming SUR-0011:74's prior dismissal with the new recount); would be about −2 if the generation pass's −7.7% were taken at face value |

## Decision and links

Triage 2026-10-17: `rejected`. Hard gate: the cost-model and feasibility lenses refute the idea at E0 from the note's own consistent recount (81 int32 ops of SIMT remainder per point-stage against the 41.4-op lazy radix-16 floor of EXP-0002, kill criterion (a)); the break-even the kill clause leaves open (recombination plus reduction under 20 ops per point-stage) lies below the carry-chain lower bound of about 28 ops, which makes the mechanism as stated impossible by count; and SUR-0011:74 had already declined INT8 limb-sliced NTTs on this ground, so the note adds the priced confirmation and nothing that reopens it. Soundness survives (the kernel is exact and prover-only), so the record stands as the answer to SUR-0011's open 64-bit question. Reopens if an E2 measurement on an H100 shows the SIMT remainder under the lazy floor, or a recombination-plus-reduction scheme under 20 int32 ops per point-stage is exhibited (E0 suffices). Nothing is queued.

Experiments: none yet. RFC or issue: none; the kernel is a prover implementation choice under RFC-0018 'Kernels'. Ledger line updated on: pending consolidation (2026-10-17).
