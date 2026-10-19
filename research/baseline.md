# Baseline: Tessera as a cost structure

Status: projections (E0) derived from the specification on 2026-10-16. The RFC-0020 gate replaces every unit cost with a measurement; until then this page is the thing to beat and the yardstick every idea is scored against. Terms are named as the cost model names them, so that an idea can say which one it moves.

## 1. What an improvement must preserve

The statement (00 §3): a committed integer model `D` produced the recorded tokens, with no prover freedom upstream of the logits and a public bound `Û` downstream. The properties no saving may buy back:

| Property | Target | Where it is fixed |
|---|---|---|
| Soundness | ≥ 100 bits for every non-interactive component, ≥ 64 bits per live proximity audit (G-4) | [RFC-0014](../docs/rfcs/RFC-0014-interaction-and-soundness.md#soundness-budget) |
| Uniqueness | exactly one passing witness upstream of the logits (SP-3, Lemma 0) | [RFC-0009](../docs/rfcs/RFC-0009-gadgets.md) |
| Zero knowledge | statistical, against a malicious verifier (G-5) | [RFC-0013](../docs/rfcs/RFC-0013-zero-knowledge.md) |
| Verifier independence | the verifier shares no code with the prover (SP-6) | [RFC-0001](../docs/rfcs/RFC-0001-workspace-and-verifier-independence.md) |
| Post-quantum | hash-based commitments, no pairings, no trusted setup | [RFC-0011](../docs/rfcs/RFC-0011-advice-commitment.md), [RFC-0004](../docs/rfcs/RFC-0004-hashing-and-transcript.md) |
| True integers | every lookup key and gadget input is a true integer in table range, derived by the verifier from public data (SP-7) | [RFC-0006](../docs/rfcs/RFC-0006-compiler-ir.md) |

## 2. The cost model

The proof multiplier `μ` is proving GPU time over the GPU time of an ideal teacher-forced INT8 forward pass ([08 §1](../docs/spec/08-performance-budget.md#1-cost-model)):

```text
μ(S) = [1.15 (C_lin c_c + K_lin c_ℓ) + 1.15 A S (c_c + c_ℓ + c_π) + W/T] / native(S) + n_regen + stream(S)
native(S) = (F_lin + F_att S) / R
stream(S) = c(S) · p_s · A · S / native(S),   c(S) = ⌈log2(N_level(S) · 32 B / free HBM)⌉
```

| Term | Meaning | Source value | Corrected value | Fixed by |
|---|---|---|---|---|
| `c_c` | ps per committed 64-bit element: rate-1/2 RS encode, BLAKE3 chained column hashing | 30 | 30 (assumed) | RFC-0020 gate (RISK-1, RISK-8) |
| `c_ℓ` | ps per LogUp-GKR lookup: combined fraction tree in `Fp2` | 40 | 40 (assumed) | RFC-0020 gate |
| `c_π` | ps per product-tree leaf (softmax max certificate) | 12 | 12 (assumed) | RFC-0020 gate |
| glue | sumcheck rounds, `eq` tables | 1.15 | 1.15 | RFC-0020 gate |
| `C_lin`, `K_lin` | committed elements and lookups per token, linear layers | 1.729×10^7 (137,216 per layer per token × 126 layers, the stated count the pinned vector records; the chart's linear column fits 1.760×10^7) | 4.116×10^7 (326,656 per layer per token: the default `servable` profile with every split, [EXP-0011](experiments/EXP-0011-servable-count/README.md); RISK-2's 220,160 is the `minimal` profile's count) | v0.2 commitment report |
| `A` | scores per token per unit context, heads × layers / 2 | 8,064 | 8,064 | model architecture |
| `F_lin`, `F_att` | native INT8 ops per token: 2 per parameter; 512 per score | 8.12×10^11, 4.13×10^6 | same | — |
| `R` | native rate, 60% of 1,979 TOPS (H100 SXM dense INT8) | 1.19×10^15 op/s | same | RFC-0020 gate |
| `W/T` | weight fold pass per token, 0.24 GPU-s per 16K-token batch | 1.5×10^-5 s | same | RFC-0020 gate |
| `n_regen` | witness regenerations in full-forward equivalents (P1 witness, P3 stages, P4 opening) | 0 | 4 (RISK-9) | v0.3 count, v0.5 measurement |
| `stream` | streamed leaf-level rounds, `p_s` = 7 ps per score per pass, 0.2 TB free HBM | 0 | on (RISK-10) | RFC-0018, v0.5 |

The executable form is [tools/cost_model.py](tools/cost_model.py). Its golden check reproduces the source's §5 chart within ±0.05 per term (the RFC-0020 criterion), reading the pinned vector `spec/vectors/cost-model/cost-model.json` (PR #240: every input with its provenance, the two-decimal chart series, the §8 table; `tessera bench model --source-defaults` is the normative evaluation) when the file is present and the embedded chart-data table otherwise. The inputs are the vector's: the stated count `137,216 × 126 = 1.729×10^7` commitments and lookups per token, `F_lin = 2 × 405,853,388,800` and `F_att = 512 × 8,064` reconstructed; the model then stays within 0.036 of the chart's linear column and 0.041 of its totals. The least-squares fit of that column, `1.760×10^7`, is recorded in the vector and in the tool for reference only (an earlier version of this page carried a `1.757×10^7` fit tuned to the tolerance).

A third profile, `counted`, carries the unit costs at the centre of the operation-count intervals the loop derived on 2026-10-16, each an E0 count fitted to an E1 measurement on an Apple M4 Max and translated to the H100's integer pipeline at the 60% utilization of [VYTA25]: `c_c` 94 ps ([EXP-0002](experiments/EXP-0002-commit-cost-calibration/README.md), interval [58, 157]), `c_ℓ` 101 ps ([SUR-0002](surveys/SUR-0002-lookups.md), 76–169), `c_π` 73 ps ([EXP-0006](experiments/EXP-0006-product-leaf-cost/README.md), [45, 117]). It is a planning profile, not the RFC-0020 reference, and it says what the assumptions hide:

| Profile | `μ` 1K | 4K | 16K | 128K | SC-2 (`μ(1K) ≤ 12`, `μ(16K) ≤ 45`) |
|---|---|---|---|---|---|
| `source` (08 §3) | 3.2 | 6.5 | 18.8 | 88.7 | holds |
| `corrected` (`servable` count of EXP-0011, RISK-9, RISK-10) | 10.4 | 15.6 | 36.6 | 172.3 | holds, by 1.15× and 1.23× |
| `counted` (unit costs 3–6× the assumptions) | 21.6 | 34.2 | 82.7 | 376.0 | fails both, by 1.80× and 1.84× |

Since 2026-10-17 the `corrected` and `counted` profiles count the default `servable` profile with every split, 326,656 commitments and lookups per layer per token ([EXP-0011](experiments/EXP-0011-servable-count/README.md), E0 from RFC-0005's range cells under RFC-0006's rule; RISK-2's 220,160 is the `minimal` profile's figure). Idea scores recorded before that date were computed against the RISK-2 count (`μ(16K)` 35.1 corrected, 78.6 counted) and say so in their notes; a re-score moves a percentage by the ratio of the two baselines, about 4–5% of itself at 16K and 18–25% at 1K.

`python3 -I research/tools/cost_model.py --profile counted --sensitivity` prints the shares at those prices; the RFC-0020 v0.1 gate measures them (E2) and decides between the profiles. Under `counted` the three unit costs are 81% of `μ(16K)` against 56% under `corrected`, and since 2026-10-16 the unit-cost ideas are scored against these shares (E0):

| Context | `c_c` | `c_ℓ` | `c_π` | regen | streaming | `c_c` halved | `c_ℓ` halved | `c_π` halved | `A` halved | streaming → 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1K | 33.1% | 35.5% | 5.8% | 23.1% | 2.4% | −16.5% | −17.8% | −2.9% | −12.1% | −2.4% |
| 4K | 31.6% | 34.0% | 13.3% | 13.3% | 7.7% | −15.8% | −17.0% | −6.6% | −28.7% | −7.7% |
| 16K | 29.7% | 31.9% | 19.0% | 5.1% | 14.3% | −14.9% | −16.0% | −9.5% | −42.8% | −14.3% |
| 128K | 27.6% | 29.6% | 20.9% | 1.1% | 20.9% | −13.8% | −14.8% | −10.4% | −49.6% | −20.9% |

The `c_π` share of 31% and the −15.7% for halving it that [EXP-0006](experiments/EXP-0006-product-leaf-cost/README.md) reports are at 73 ps with `c_c` and `c_ℓ` at their assumed values; when all three rise together, `c_π` is 19.0% at 16K and halving it is worth −9.5%.

### Baseline tables (E0)

`python3 research/tools/cost_model.py`:

| Context | Profile | Linear | Attention | Weights | Regen | Streaming | μ |
|---|---|---|---|---|---|---|---|
| 1K | source | 2.03 | 1.13 | 0.021 | 0 | 0 | 3.2 |
| 4K | source | 1.99 | 4.46 | 0.021 | 0 | 0 | 6.5 |
| 16K | source | 1.88 | 16.82 | 0.020 | 0 | 0 | 18.7 |
| 128K | source | 1.22 | 87.48 | 0.013 | 0 | 0 | 88.7 |
| 1K | corrected | 4.82 | 1.13 | 0.021 | 4.0 | 0.42 | 10.4 |
| 4K | corrected | 4.75 | 4.46 | 0.021 | 4.0 | 2.32 | 15.6 |
| 16K | corrected | 4.47 | 16.82 | 0.020 | 4.0 | 11.24 | 36.6 |
| 128K | corrected | 2.91 | 87.48 | 0.013 | 4.0 | 77.93 | 172.3 |

The v1.0 ceilings are `μ(1K) ≤ 12` and `μ(16K) ≤ 45` (SC-2). The corrected 16K figure already spends most of the ceiling; the corrected 128K figure says the design has no long-context story until RISK-10 and the per-score cost move.

### Where the time goes (sensitivity, corrected profile)

`python3 research/tools/cost_model.py --sensitivity`:

| Context | `c_c` | `c_ℓ` | `c_π` | weights | regen | streaming |
|---|---|---|---|---|---|---|
| 1K | 20.6% | 27.5% | 1.9% | 0.2% | 45.1% | 4.7% |
| 4K | 21.5% | 28.7% | 4.6% | 0.1% | 28.5% | 16.5% |
| 16K | 21.2% | 28.3% | 7.0% | 0.1% | 11.4% | 32.0% |
| 128K | 19.2% | 25.6% | 7.5% | 0.0% | 2.3% | 45.5% |

Gain in `μ` from one lever, 2× better (or removed where marked):

| Lever | 1K | 4K | 16K | 128K |
|---|---|---|---|---|
| `c_c` halved | −10.3% | −10.8% | −10.6% | −9.6% |
| `c_ℓ` halved | −13.7% | −14.4% | −14.2% | −12.8% |
| `c_π` halved | −0.9% | −2.3% | −3.5% | −3.7% |
| `C_lin` halved | −8.0% | −5.0% | −1.9% | −0.2% |
| `K_lin` halved | −10.6% | −6.6% | −2.5% | −0.3% |
| `A` halved (fewer scores proven) | −9.2% | −25.3% | −41.7% | −50.1% |
| glue → 1.0 | −6.5% | −7.2% | −7.4% | −6.8% |
| regen → 0 | −45.1% | −28.5% | −11.4% | −2.3% |
| streaming → 0 | −4.7% | −16.5% | −32.0% | −45.5% |

Reading: the regeneration passes (RISK-9) are the largest single lever at short context; the number of scores proven (`A`) and the streamed passes (RISK-10) dominate from 4K up; `c_ℓ` is the largest unit cost at every context; `c_π` and the linear-layer counts are second-order. An idea is worth an experiment when it moves a lever that is large at the context it targets.

## 3. Primitive baseline (E0, assumed)

| Primitive | Current design | Assumed cost | Notes |
|---|---|---|---|
| Committed element | row of width 2^22, RS rate 1/2 over Goldilocks by coset NTT, vanishing-mask coefficients, chained BLAKE3 column digests with row block `B = 32` | 30 ps | about 530 GB/s of hashing per H100 inside the budget (RISK-8) |
| Lookup | LogUp-GKR, one combined fraction tree per table, leaves `(1, α − a)` and `(−m, α − t)`, `Fp2` challenges, degree-3 sumcheck per level, batched across instances | 40 ps | `(N + T)/|Fp2|` error, about 2^-90 at frontier volume (RISK-5) |
| Product-tree leaf | max certificate `Π (c_q − x_qi) = 0` per row, GKR over `Fp2` | 12 ps | one leaf per score |
| Sumcheck round | degree ≤ 7, `p(0) + p(1)` check, about 10^4 rounds per 405B 16K proof | 15% of commit + lookup | |
| Weight fold | limb-decomposed INT8 GEMM on tensor cores, one pass per proof unit | 0.24 GPU-s per 405B unit | opening and sumcheck fold are the same pass |
| Native rate | INT8 GEMM at 60% of 1,979 TOPS | 1.19×10^15 op/s | the denominator of `μ` |

## 4. Parameters, proof size, verification (E0)

| Quantity | `live-64/1` | `ni-100/1` | Where |
|---|---|---|---|
| Advice: `log2 k`, rate, mask `P_A`, `τ_A` | 22, 1/2, 160, 155 (64.3 bits) | 22, 1/2, 200, 193 (80.1 + 20 bits) | [RFC-0014](../docs/rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets) |
| Weights: `k_W`, rate, `P_W`, `τ_W` | 2^20, 1/4, 2^16, 97 (64.0 bits) | same, 122 (80.5 + 20 bits) | same |
| Per-query bits `b(ρ') = −log2((1 + ρ')/2)` (unique decoding) | 0.415 (advice), 0.660 (weights) | same | RFC-0014 'Soundness budget' |
| Projection under the Draft [RFC-0021](../docs/rfcs/RFC-0021-johnson-regime-proximity.md) (Johnson regime, uniform `Fp2` coefficients, `η_A` 0.003, `η_W` 0.005; E0, not yet accepted) | `τ_A` 130 (0.494 bits per query, field term 82.2 bits, list bound 49), `τ_W` 68 (0.942 bits, 86.0 bits); opened columns 0.72 → 0.60 GB, proof about 1.88 GB | `τ_A` 163, `τ_W` 85; opened columns 0.90 → 0.76 GB | [IDEA-0012](ideas/IDEA-0012-johnson-regime-tau-linear-error.md), [EXP-0003](experiments/EXP-0003-johnson-linear-tau/README.md), [EXP-0005](experiments/EXP-0005-list-decoding-extractor/README.md) |
| Opened advice columns | 155 × 5.8×10^5 rows × 8 B ≈ 0.72 GB | about 0.90 GB | [RFC-0011](../docs/rfcs/RFC-0011-advice-commitment.md#sizes-at-the-default-parameters-16k-tokens-16k-context-405b) |
| Combined rows | ≤ 17 × 2^22 × 16 B ≈ 1.14 GB | same | same |
| Weight openings | < 0.3 GB | | [08 §5](../docs/spec/08-performance-budget.md#5-proof-size-and-verification-budgets) |
| Proof total | about 2 GB; gate 4 GiB, target 2 GiB | about 2.2 GB | SC-3 |
| Verification | ≤ 300 s on 32 cores, ≤ 64 GiB RAM: hash 0.72 GB of columns, `(G + 1) τ_A` inner products of length 5.8×10^5, `G + 1` coset NTTs of size 2^23 over `Fp2`, table MLEs | | SC-3, RFC-0011 |

The calculator for these parameters under alternative proximity bounds is [tools/proximity.py](tools/proximity.py).

## 5. Scoreboard

"Beats every known protocol" means, in order:

1. **Statement level.** For a dense model of at least 70B parameters, the lowest `μ` at 1K, 4K and 16K context and the smallest proof, among systems that prove the full statement (uniqueness upstream of the logits, transcript anchor, `Û`) with ≥ 100-bit soundness, zero knowledge, hash-based commitments and an independent verifier, measured end to end (E3) under a published protocol.
2. **Primitive level.** Per-lookup, per-committed-element, per-product-leaf, per-sumcheck-round and per-hashed-byte costs at or below the best published figures for the same field size and security level, measured on the same hardware class (E2).
3. **Verification.** Verification time and proof size at or below the best published figures for the same statement strength.

Values as of 2026-10-16, from [landscape.md](landscape.md). Tessera's are projections.

| Metric | Tessera (projected, E0) | Best known | Holder | Gap |
|---|---|---|---|---|
| `μ` at 1K, dense 405B | 10.4 (corrected, `servable` count), 3.2 (source), 21.6 (counted) | none at this scale; about 10^5 for a 400B MoE at 1K tokens | VerInf | 4 orders of magnitude in Tessera's favor, on paper |
| `μ` at 16K, dense | 35 (corrected), 19 (source) | about 100–300× against ideal prefill for a 31B model (source's estimate from throughput) | Attestable | Tessera leads on paper, unmeasured |
| Throughput, one H100, ~30B dense | not stated (8×H100 design point) | 53–77 tokens/s proved | Attestable | the only measured frontier-adjacent number |
| Proof size, 16K tokens | about 2 GB | 4.35–7.92 MiB | Attestable | about 300× against Tessera; recursion is NG-3 |
| Verification time | ≤ 300 s target | 157–648 ms | Attestable | about 500× against Tessera |
| Soundness | 100 bits NI, 64 bits live | 100 bits | Attestable | parity |
| Zero knowledge | statistical, malicious verifier | stated | several | parity claimed, unverified for others |
| Post-quantum | hash-based | hash-based | Attestable | parity |
| Statement | uniqueness upstream, anchor, `Û` | `y = F(W, x, r)` with floating-point nonlinearities | Attestable | Tessera's is stronger |
| Verifier | independent code, specification-first | formal verification claimed | Attestable | the bar to clear (NG-7) |

Where Tessera trails (proof size, verification time), the gap is structural (Ligero-family openings, no recursion) and is a research target; where it leads, the lead is a projection until v0.1 measures the unit costs and v0.5 measures a proof.

### Standing after the first day of the loop (2026-10-16)

What the loop can settle on this machine is E0 and E1; levels 1 and 2 of the definition above need an H100 (E2) and a proof (E3), which the RFC-0020 gate and the v0.5 milestone deliver, so no claim of "beats every known protocol" can be made from `research/` alone, and none is. The day's results move the projections in both directions, each with its evidence level:

| Metric | Before | After the day's results | Evidence |
|---|---|---|---|
| Proof size, 16K tokens | about 2.0 GB | about 1.88 GB under the Draft [RFC-0021](../docs/rfcs/RFC-0021-johnson-regime-proximity.md) (`τ_A` 155 → 130); still about 250× Attestable's | E0, primary statements |
| Verification work | hash 0.72 GB, 17 × 155 inner products | hash 0.60 GB, 17 × 130 inner products (−16%) under RFC-0021; the 300 s target unchanged | E0 |
| `μ` at 16K | 36.6 (corrected profile, `servable` count, assumed unit costs) | 82.7 under the `counted` profile (`c_c` 94, `c_ℓ` 101, `c_π` 73 ps from EXP-0002, SUR-0002 and EXP-0006); 42 to 67 if only `c_c` moves; the SC-2 ceiling (45) holds only for kernels near the integer pipeline's peak | E0 counts fitted to E1 |
| Soundness accounting | `n/|Fp2|` field term on a tensor generator | the row's field term corrected (`20 · n/|Fp2|`, 2^-100.7, still within target) and, under RFC-0021, a Johnson-regime row with a machine-checked bound | E0, primary |
| Largest open levers | — | tile-folded streaming (−28% / −42% of `μ` at 16K / 128K, E0, conditional), certified block skipping (−42% to −74% at 16K if half to three quarters of blocks skip; hard sparsity unmeasured), `n_regen` 4 → 2 (−22% at 1K) | E0, from SUR-0003 and SUR-0004 |

The honest reading: the design's lead on `μ` is a projection whose unit-cost inputs the day's counts moved up by 2–4×, while its structural deficits (proof size, verification time) moved down by 6% and 16% on paper. The scoreboard is re-evaluated at every consolidation; the ledger's queue is ordered by what would move it most.

## 6. Levers: which technique class moves which term

| Technique class | Terms | Current anchor |
|---|---|---|
| Faster hashing, cheaper encoding, lower rate, fewer hashed bytes | `c_c` | RFC-0004, RFC-0011 |
| Small-value commitments (ring switching, lattice) | `c_c` | NG-5 |
| LogUp micro-architecture, alternative lookup arguments, tuple tables, fewer lookups per op | `c_ℓ`, `K_lin` | RFC-0008 |
| Cheaper or shared max certificates | `c_π` | RFC-0009 `SoftmaxRow` |
| Range analysis, rescale placement, wider accumulators, quantization co-design | `C_lin`, `K_lin` (RISK-2) | RFC-0005, RFC-0006 |
| Certified block skipping, sparsity certificates, proof-aware `D` in bounded mode | `A` | source §9 problem 1b |
| Pass fusion, checkpoint placement, streaming-friendly sumcheck, proof units, multi-GPU partitioning | `regen`, `stream` (RISK-9, RISK-10) | RFC-0018 |
| Proximity analysis, rate choice, grinding, recursion | `τ`, size, verify | RFC-0014, NG-3 |
| Structured tables, batching, verifier algorithms | verify | RFC-0008, RFC-0019 |
| Masking cost, hiding budget | zk overhead, size | RFC-0013 |
| Field and extension choice, GPU arithmetic | every unit cost | RFC-0002 |

## 7. Already rejected

Each RFC's "Alternatives Considered" is a decision with a reason. An idea that re-proposes one must say what is new. The column "Reopens if" is the evidence that would.

| Alternative | Rejected in | Reason given | Reopens if |
|---|---|---|---|
| M31/QM31 (Stwo), BabyBear with degree-4 extension | RFC-0002 | about twice the linear-layer rescales, harder RMSNorm | measured rescale counts and GPU arithmetic show a net win at 4K and 32K (source §9 problem 5) |
| 254-bit pairing field; Montgomery form for the prover; cubic or quartic challenge extension | RFC-0002 | 10× slower; special-form reduction is faster; 2.25× cost on `Fp2` work | — |
| SHA-256, SHA-3, BLAKE2s, Poseidon2 for the commitment path | RFC-0004 | GPU throughput; Poseidon2 about 10× slower than BLAKE3 on GPUs, shorter cryptanalysis record | GPU measurements show a field-native hash within the 30 ps budget, or recursion (NG-3) enters scope |
| One streaming hasher per column | RFC-0004 | 16 GB of hasher state for 2^23 columns | — |
| One-hot lookups (Lasso, Shout) | RFC-0008 | several committed chunks per lookup with hash-based commitments; the separation result says activation structure cannot lower per-lookup cost | a hash-based instantiation measures below LogUp-GKR per lookup on GPU at Tessera's table sizes |
| Univariate lookups (Plookup, Caulk, cq) | RFC-0008 | need univariate, usually pairing-based commitments | — |
| Base-field LogUp challenges | RFC-0008 | 2^-28 error at frontier volume | — |
| Two-table softmax bracket; tolerance-band softmax; one-hot row maximum; rsqrt by table; Newton iteration | RFC-0009 | cost (two lookups and commitments per score); witness slack; 30 ps per one-hot entry against 12 ps per leaf; range; prover-chosen iterates | a certificate cheaper than one product leaf per score with a uniqueness proof |
| Layer-by-layer GKR; committing every intermediate; independent per-layer challenges; one monolithic sumcheck | RFC-0010 | depth grows with layers; the cost driver; weight claims on 126 points; dependent relations | — |
| Storing the encoded witness; FRI-family PCS (BaseFold, WHIR) for the advice; Brakedown codes; systematic encoding; per-tensor commitments | RFC-0011 | terabytes; folding needs the vector resident or many passes; NTT is fast on GPU; hiding; thousands of roots | a folding commitment with a one-pass streaming prover, or recursion (NG-3) |
| ZK-STARK wrapper; committing round messages; per-layer masked evaluations; doubling tensors; honest-verifier ZK | RFC-0013 | second proof system; thousands of commitments; budget; 2× commitment cost; adversarial verifier | recursion enters scope |
| Non-interactive only; verifier-chosen seed; Johnson-bound (list-decoding) proximity analysis; `Fp4` LogUp; public beacon | RFC-0014 | 2.5× columns; adversarial challenges; field-size terms erode the margin and conjectures are unacceptable; 2× lookup cost | a provable interleaved-code bound whose field-size term keeps ≥ 64 bits at `n = 2^23` over `Fp2` ([tools/proximity.py](tools/proximity.py) quantifies it). Fired on 2026-10-16 for the Johnson-bound entry: [IDEA-0012](ideas/IDEA-0012-johnson-regime-tau-linear-error.md) is `validated` by [EXP-0003](experiments/EXP-0003-johnson-linear-tau/README.md) under [BCHKS25] and [DKT26] (77–80 one-shot bits at `τ_A` 129–130, E0), pending the list-decoding extractor argument and the switch to uniform `Fp2` proximity coefficients |
| End-to-end measurement only; roofline only; B200 first | RFC-0020 | too late; implementation-dependent; the comparison is stated for H100 | OQ-0020-1 |
