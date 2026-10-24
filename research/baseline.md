# Baseline: Tessera as a cost structure

Status: projections (E0), reconciled with main on 2026-10-19. The RFC-0020 gate replaces every unit cost with a measurement; until then this page is the thing to beat and the yardstick every idea is scored against. Terms are named as the cost model names them, so that an idea can say which one it moves.

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

Reconciled on 2026-10-19 with main at `6d9de83` (compiler report unchanged from c2c8a69). All figures below are E0 projections. The normative implementation is `tessera-bench::mu` (RFC-0020); `tessera bench model --source-defaults --json` exposes the original design inputs, while `--plan` is still pending (#87). The research tool now uses the same formula and the compiler’s committed counts, checked by [check_cost_model.py](tools/check_cost_model.py) against the normative Rust implementation on 72 scenarios, including nonzero P4/P5, table costs, short score blocks and exact memory boundaries.

```text
S′ = S + min(S, b)
native(S) = (F_lin + F_att S) / R
μ(S) = [glue (C c_c + K c_ℓ + A S (c_c + c_ℓ + c_π) + A min(S,b) c_c)
        + W/T + (C + A S′) (c_a + c_r) + glue M (c_c + c_ℓ)/T] / native(S)
       + n_regen + c(S) p_s A S / native(S)
c(S) = max(0, ceil(log2(A S T bytes_per_leaf / free_HBM)))
```

| Input | Current planning value | Evidence and scope |
|---|---|---|
| C / K | 45,566,660 / 45,440,678 per token | Compiler report, `servable`, two classes, B_max=32; includes checkpoints, one-hot and row scalars |
| M | 7,693,467,648 table rows per proof | Same report; excludes unallocated gadget range tables (#118/#122) |
| A / b | 8,064 / 512 | RFC-0034; diagonal masked halves are committed, never looked up or multiplied |
| F_lin / F_att | 811,698,487,296 / 4,128,768 ops | Report’s exact parameter count; source-defaults instead uses 811,706,777,600 linear ops |
| R / T / W | 1.1874e15 ops/s / 16,384 tokens / 0.24 GPU-s | Assumptions of RFC-0020 |
| n_regen | 5 | RFC-0018 P1–P5; #248 |
| p_s / free_HBM / bytes_per_leaf | 7 ps / 0.2 TB / 32 B | Attention leaf level only: 5/7/9/12 passes at 1K/4K/16K/128K (#250) |
| c_a (P5) | c_c: 30 / 94 ps | Full-re-encode planning bound, not a measurement; no glue on this term |
| c_r (P4) | 0, explicitly unpriced | Unknown H100 cost; use `--set row_combination_ps=...` (#252) |
| c_c / c_ℓ / c_π, corrected | 30 / 40 / 12 ps | Assumed targets |
| c_c / c_ℓ / c_π, counted | 94 / 101 / 41.8 ps | Counts fitted to E1; c_π repriced after #258’s split-eq kernel landed, using EXP-0008’s ratio; no H100 measurement |
| glue | 1.15 | Sumcheck overhead assumption; zero-knowledge overhead is not this entire 15% |

Counts come from [the current compiler report](../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json), refreshed by `cargo test -p tessera-bench --test commitment_report`. They supersede the 326,656 rescale-only count of EXP-0011, which described one class before RFC-0022’s classed residual update. The current per-layer rescale count is 359,424 at two classes, plus 808 row scalars; total commitments and lookups differ. One class still has 326,656 rescale elements, so that historical finding remains correct within its stated setting. [The report guide](../docs/perf/v0.2/README.md) gives all class counts and table bounds.

**Scope limitation:** P4 at zero, the rest of the leaf-stage streaming (#329), and the unallocated range tables are omissions. These totals are scenarios, not full proving-time bounds. P5 at c_c bounds re-encoding alone; it does not bound the omitted work. Weight column answers use the epoch’s stored codewords (about 15.2 TB, #251); evaluating them without that store must instead be charged to W. CPU ratios do not establish GPU costs.

### Baseline tables (E0)

Reproduce with `python3 -I research/tools/cost_model.py --profile counted` (or `corrected`). Historical raw results under experiments are immutable snapshots; rerun their pinned commit to recover their old model totals.

| Profile | μ(1K) | μ(4K) | μ(16K) | μ(128K) | SC-2 |
|---|---|---|---|---|---|
| `source` | 3.2 | 6.5 | 18.7 | 88.7 | original design only |
| `corrected` | 14.7 | 20.9 | 45.6 | 202.9 | misses 1K and 16K ceilings |
| `counted` | 32.3 | 46.8 | 102.4 | 436.3 | misses 1K and 16K ceilings |

| Context | Profile | Linear | Attention | Weights | Regen | Streaming | P5 | P4 | Tables | μ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1K | corrected | 5.330 | 1.340 | 0.021 | 5.000 | 0.421 | 2.530 | 0.000 | 0.055 | 14.697 |
| 4K | corrected | 5.248 | 4.668 | 0.021 | 5.000 | 2.319 | 3.556 | 0.000 | 0.054 | 20.866 |
| 16K | corrected | 4.945 | 17.016 | 0.020 | 5.000 | 11.240 | 7.365 | 0.000 | 0.051 | 45.637 |
| 128K | corrected | 3.214 | 87.606 | 0.013 | 5.000 | 77.926 | 29.139 | 0.000 | 0.033 | 202.932 |
| 1K | counted | 14.849 | 3.922 | 0.021 | 5.000 | 0.421 | 7.928 | 0.000 | 0.153 | 32.294 |
| 4K | counted | 14.622 | 13.529 | 0.021 | 5.000 | 2.319 | 11.143 | 0.000 | 0.151 | 46.786 |
| 16K | counted | 13.778 | 49.186 | 0.020 | 5.000 | 11.240 | 23.078 | 0.000 | 0.142 | 102.444 |
| 128K | counted | 8.956 | 253.020 | 0.013 | 5.000 | 77.926 | 91.303 | 0.000 | 0.092 | 436.310 |

### Reconciliation and sensitivity (E0)

The old minimal comparison (about 8.8 versus 9.4 at 1K) mixed a rounded rescale count and omitted work; it was not a discrepancy between two evaluations of identical inputs. The current differential check matches every term to Rust within 2e-13 relative tolerance. Source golden totals still pass ±0.05 without a fitted count. With P5, streaming and P4 disabled, the current report reproduces its 11.747 / 27.032 at 1K / 16K; adding the missing work yields the table above.

| Lever, counted profile | Δμ(1K) | Δμ(4K) | Δμ(16K) | Δμ(128K) |
|---|---|---|---|---|
| commit_ps (c_c) | -14.2% | -13.8% | -13.0% | -12.0% |
| lookup_ps (c_ℓ) | -14.2% | -14.0% | -13.6% | -12.9% |
| leaf_ps (c_π) | -0.9% | -2.4% | -4.2% | -5.1% |
| commits_per_token (C_lin) | -20.7% | -14.1% | -6.1% | -0.9% |
| lookups_per_token (K_lin) | -11.9% | -8.1% | -3.5% | -0.5% |
| scores per token (A) | -9.5% | -22.6% | -38.5% | -48.7% |
| glue factor → 1.0 | -7.6% | -7.9% | -8.0% | -7.8% |
| regen → 0 | -15.5% | -10.7% | -4.9% | -1.1% |
| streaming → 0 | -1.3% | -5.0% | -11.0% | -17.9% |
| column_answer_ps (P5) | -12.3% | -11.9% | -11.3% | -10.5% |
| row_combination_ps (P4) | +0.0% | +0.0% | +0.0% | +0.0% |
| table rows (M) | -0.2% | -0.2% | -0.1% | -0.0% |

P5 and P4 are independent costs: reducing commit_ps alone does not silently reduce either. The count levers reduce P4/P5 arithmetic as well as commitments and lookups. The validated pruned P5 result belongs in column_answer_ps, rather than being folded into commit_ps with a second glue charge. The model cannot certify SC-2 before E2 measurements and #329’s accounting.

### Survey-requested scenarios, refreshed 2026-10-19 (E0)

[survey_scenarios.py](tools/survey_scenarios.py) reproduces the [full inputs and outputs](tools/results/survey-scenarios-2026-10-19.json). These replace the historical SUR-0005 scenario totals, using current compiler counts and P5 accounting. The external GPU anchors motivate stress inputs, not measured Tessera H100 costs. Each cell lists μ at 1K / 4K / 16K / 128K.

| c_c | P5 held at94ps | Full-re-encode P5 explicitly set to c_c |
|---|---|---|
| 150ps | 37.8 / 54.5 / 118.3 / 498.9 | 42.5 / 61.1 / 132.0 / 553.3 |
| 240ps | 46.6 / 66.8 / 143.8 / 599.5 | 58.9 / 84.1 / 179.6 / 741.3 |
| 360ps | 58.3 / 83.3 / 177.7 / 733.6 | 80.7 / 114.8 / 243.0 / 991.9 |

Zero knowledge needs a separate additive line. Current [RFC-0013's Cost](../docs/rfcs/RFC-0013-zero-knowledge.md#cost) estimates **1.6×10^10 base-field mask elements**, twice SUR-0008's historical input. At16K tokens and context, charging those elements at `1.15×94ps` for commitment plus `94ps` for P5 gives **3.2336 GPU-s**, or **Δμ=0.26650**, about **0.260%** of the current partial counted baseline. This is an E0 subset estimate, not measured time, a complete ZK total, or a universal factor. The script checks the incremental model calculation against the direct dimensional formula. The 1.15 sumcheck glue stays unchanged; adding this subset to it does not turn the entire15% into ZK overhead.

Mask counts must come from the eventual stage schedule (#113/#149/#154); the compiler report currently excludes them. Extra masked-sumcheck work, current reserved opening rows and the weight masking budget still need separate accounting. Consequently the old `zk≈1.001` recommendation is historical and cannot be copied into the current model as a measured or complete cost. The normative formula and default partial projections above remain unchanged.

## 3. Primitive baseline (E0, assumed)

| Primitive | Current design | Assumed cost | Notes |
|---|---|---|---|
| Committed element | row of width 2^22, RS rate 1/2 over Goldilocks by coset NTT, vanishing-mask coefficients, chained BLAKE3 column digests with row block `B = 32` | 30 ps | about 530 GB/s of hashing per H100 inside the budget (RISK-8) |
| Lookup | LogUp-GKR, one combined fraction tree per table, leaves `(1, α − a)` and `(−m, α − t)`, `Fp2` challenges, degree-3 sumcheck per level, batched across instances | 40 ps | `(N + T)/\|Fp2\|` error, about 2^-90 at frontier volume (RISK-5) |
| Product-tree leaf | max certificate `Π (c_q − x_qi) = 0` per row, GKR over `Fp2` | 12 ps | one leaf per score |
| Sumcheck round | degree ≤ 7, `p(0) + p(1)` check, about 10^4 rounds per 405B 16K proof | 15% of commit + lookup | |
| Weight fold | limb-decomposed INT8 GEMM on tensor cores, one pass per proof | 0.24 GPU-s per 405B proof | opening and sumcheck fold are the same pass |
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
| `μ` at 1K, dense 405B | 14.7 corrected / 32.3 counted (two-class compiler counts; §2 scope) | none at this scale; about 10^5 for a 400B MoE at 1K tokens | VerInf | 4 orders of magnitude in Tessera's favor, on paper |
| `μ` at 16K, dense | 45.6 corrected / 102.4 counted (§2 scope) | about 100–300× against ideal prefill for a 31B model (source's estimate from throughput) | Attestable | Tessera leads on paper, unmeasured |
| Throughput, one H100, ~30B dense | not stated (8×H100 design point) | 53–77 tokens/s proved | Attestable | vendor claim; not independently reproduced here |
| Proof size, 16K tokens | about 2 GB | 4.35–7.92 MiB | Attestable | about 300× against Tessera; recursion is NG-3 |
| Verification time | Historical projection about 13 s on one core (E0, SUR-0012; not recomputed for the current layout); ≤ 300 s on 32 cores is the release budget | 157–648 ms on unnamed reference CPU (claim) | Attestable | Different hardware and evidence; no measured ratio established |
| Soundness | 100 bits NI, 64 bits live | 100 bits | Attestable | parity |
| Zero knowledge | statistical, malicious verifier | stated | several | parity claimed, unverified for others |
| Post-quantum | hash-based | hash-based | Attestable | parity |
| Statement | uniqueness upstream, anchor, `Û` | `y = F(W, x, r)` with floating-point nonlinearities | Attestable | Tessera's is stronger |
| Verifier | independent code, specification-first | formal verification not established by the primary pages checked 2026-10-17 ([SUR-0012](surveys/SUR-0012-verifier.md)) | Attestable | No assurance comparison established; NG-7 remains Tessera's own v1 non-goal |

Where Tessera trails (proof size, verification time), the gap is structural (Ligero-family openings, no recursion) and is a research target; where it leads, the lead is a projection until v0.1 measures the unit costs and v0.5 measures a proof.

### Standing after the first day of the loop (2026-10-16)

What the loop can settle on this machine is E0 and E1; levels 1 and 2 of the definition above need an H100 (E2) and a proof (E3), which the RFC-0020 gate and the v0.5 milestone deliver, so no claim of "beats every known protocol" can be made from `research/` alone, and none is. The day's results move the projections in both directions, each with its evidence level:

| Metric | Before | After the day's results | Evidence |
|---|---|---|---|
| Proof size, 16K tokens | about 2.0 GB | about 1.88 GB under the Draft [RFC-0021](../docs/rfcs/RFC-0021-johnson-regime-proximity.md) (`τ_A` 155 → 130); still about 250× Attestable's | E0, primary statements |
| Verification work | hash 0.72 GB, 17 × 155 inner products | hash 0.60 GB, 17 × 130 inner products (−16%) under RFC-0021; the 300 s target unchanged | E0 |
| `μ` at 16K | Historical 36.6 / 82.7 corrected / counted | Current 45.6 / 102.4 with two-class compiler counts, P5, streaming and the landed split-eq repricing; P4 and #329 remain unpriced | E0, §2 |
| Soundness accounting | `n/\|Fp2\|` field term on a tensor generator | the row's field term corrected (`20 · n/\|Fp2\|`, 2^-100.7, still within target) and, under RFC-0021, a Johnson-regime row with a machine-checked bound | E0, primary |
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
