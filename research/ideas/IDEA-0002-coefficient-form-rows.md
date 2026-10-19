# IDEA-0002: Coefficient-form rows: drop the interpolation NTT from the advice and weight encoders

| | |
|---|---|
| Status | `triaged` (reviewed 2026-10-19; discussion only, not promoted) |
| Origin | new: this round's commitment lens, from the cost line of [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding) ("an inverse NTT of size `k` and a coset NTT of size `2k`" per row) and the four steps of `crates/tessera-pcs/src/encoding.rs`; motivated by RISK-1 and RISK-8 ([00 §9](../../docs/spec/00-overview.md#9-risks)) and by [SUR-0001](../surveys/SUR-0001-proximity.md) Question 3, where the rate couples `τ` to `c_c`, so `c_c` has to fall without touching the rate |
| Lever | `c_c` (the encoder's share of the committed-element cost: one of the three NTT kernels per row disappears); `verify` (the `G + 1` inverse NTTs of size `2^22` over `Fp2` that `enc(u_g)` needs today disappear) |
| Spec surfaces | [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding), ['Opening protocol'](../../docs/rfcs/RFC-0011-advice-commitment.md#opening-protocol) step 8.3, the verifier-work row of ['Sizes'](../../docs/rfcs/RFC-0011-advice-commitment.md#sizes-at-the-default-parameters-16k-tokens-16k-context-405b) and the first bullet of 'Testing Strategy' ("the mask vanishes on `H`"); [RFC-0012 'Layout'](../../docs/rfcs/RFC-0012-weight-commitment.md#layout), the mask bullet; [RFC-0013 'Hiding openings'](../../docs/rfcs/RFC-0013-zero-knowledge.md#hiding-openings), the 'Columns' bullet (restated lemma); the commitment descriptor behind `commitment_id` ([RFC-0004 'Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests)). The codeword and the verifier's column check are protocol semantics (tier A, [02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)): a superseding RFC and a new protocol version are required |
| Priority | Historical 2026-10-16 score; not a current adoption estimate: 1.0 (−5.9% × 0.7 / 4 under the `counted` profile, re-scored 2026-10-16; 0.93 with the fifth pass priced against honest baselines, 1.26 with the pruned column pass on both sides; 0.63 at triage under `corrected`); the experiment's E1 split is in EXP-0007, adoption is a tier-A RFC |

## Current-spec review, 2026-10-19

[RFC-proposal discussion #393](https://github.com/AbdelStark/tessera/issues/393) records the design and missing acceptance gates. Status remains `triaged`: EXP-0007 validated output pruning (IDEA-0016), not this complete protocol or its security. No numbered RFC or production change has been adopted.

The discussion recommends coefficient rows with **natural semantic columns** and the existing derive-key chain. EXP-0007 gathers column `j` from DIF leaf `bitrev(j)`; it does not validate the historical mechanism's proposed bit-reversed semantic column mapping. The old mechanism and triage below are historical proposals where they differ from this review.

A protocol string alone cannot bind this format for weights: RFC-0031's weight descriptor has no protocol/statement field, and RFC-0040's `R_W` hashes the public payload, not its envelope version. The discussion proposes a fixed format discriminator in both layout structs, so claim list, descriptors, `commitment_id` and hashed weight-public payload bind basis/order/chain mode. This needs an accepted superseding tier-A RFC, explicit protocol/schema versions under 09 §3, and new roots/artifacts; no existing root may be reinterpreted. Protocol 2 already has ZK semantics and is not silently reassigned.

Preserve [RFC-0047](../../docs/rfcs/RFC-0047-advice-order.md): stable padded-axis order, permutation of relation claim coordinates into layout order, `ν`, maximum allocated rows checked before allocation, GQA and sequence bounds. The basis change does not rename logical tensor indices.

The mask matrix `diag(x_j^k)·Vandermonde` has full row rank at at most `P` distinct nonzero points (E0), but this proves only marginal column hiding. The joint distribution with `Fp2` combined rows, blinding claims and persistent weight masks still needs the complete malicious-verifier simulator and independent review. `docs/zk/simulator.md`, required by RFC-0013, is absent in this checkout. This security gate is **unknown**. A deliberately non-ZK-only scope may defer it only while barring the format in ZK mode. IDEA-0014's keyed chain is excluded pending its separate unknown gate.

Historical μ/priority/confidence figures below are not modern adoption evidence. P1 `commit_ps`, P5 `column_answer_ps` and P4 `row_combination_ps` must be priced separately; five regenerations, unpriced P4 and incomplete #329 remain visible. EXP-0007 is historical E1 CPU kernel evidence; there is no E2 gain or full-protocol ZK result here. Next: resolve #393's proof, version/binding and independent-vector gates, then preregister any required experiment.

## Mechanism

**Today** ([RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding), read 2026-10-16). A row's `k` values are the evaluations of `f` (degree `< k`) on the subgroup `H` of order `k`; the mask is `Z_H(X)·m(X)` with `Z_H = X^k − 1` and `m` of degree `< P`; the codeword is `g = f + Z_H·m` on the coset `7·L`, `|L| = n = 2k`, symbol `j` at `x_j = 7·ω_n^j`. `encoding.rs` runs, per row: (1) a Cooley–Tukey pass of size `k` with inverse twiddles (the interpolation, output bit-reversed), (2) a spread of the coefficients to even positions with the `7^i/k` scaling, (3) the mask at `−m_i·7^i` in the even and `m_i·7^{k+i}` in the odd position, (4) a Gentleman–Sande pass of size `n`. The verifier (step 8.3) checks `Σ_row c_g(row)·col_j[row] = enc(u_g)(x_j) + Z_H(x_j)·M_g(x_j)`, where `enc(u)` interpolates `u` on `H`: an inverse NTT of size `k` over `Fp2` and a coset NTT of size `n` per combined row.

**Change.** A row's `k` values *are* the coefficients `f_0, …, f_{k−1}` of `f(X) = Σ_i f_i X^i`. The mask occupies coefficients `k, …, k + P − 1`: `g(X) = f(X) + X^k·m(X)`. The codeword is `g` on the coset `7·L` as before. Concretely:

- Encoder: step 1 is deleted. The buffer of size `n` receives `f_i·7^i` in position `i < k`, `m_i·7^{k+i}` in position `k + i` for `i < P`, zero above (`7^i` from one precomputed table, as the `spread` table today), then one Gentleman–Sande pass of size `n` (step 4 unchanged). Two kernels per row (one NTT and the hash) instead of three.
- Codeword order. A Gentleman–Sande pass on natural-order coefficients yields evaluations in bit-reversed order: position `t` holds `g(7·ω_n^{rev_n(t)})`. Two ways to keep RFC-0011's "evaluations in natural order": (a) one bit-reversal permutation pass per row (a memory pass, no arithmetic), or (b) define symbol `j` as `g(7·ω_n^{rev_n(j)})`, so the verifier checks column `j` at `x_j = 7·ω_n^{rev_n(j)}`: a public relabelling of columns, zero cost. Option (b) is the proposal; (a) is the fallback if the relabelling is judged to complicate the verifier. Today's encoder avoids the permutation because the Cooley–Tukey step hands over its output already bit-reversed; the new encoder has no such step, so this choice must be made explicitly.
- The code is unchanged: `RS[n, k + P]` on `7·L`, every codeword a polynomial of degree `< k + P ≤ n` evaluated on the coset, so [RFC-0014](../../docs/rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets)'s `ρ' = (k + P)/n`, `τ_A`, `τ_W` and the proximity bounds apply verbatim. Only the linear map from the message to the codeword changes (a public change of basis on the message: coefficients instead of values on `H`).
- Verifier: step 8.3 becomes `Σ_row c_g(row)·col_j[row] = Σ_{i<k} u_g[i]·x_j^i + x_j^k·M_g(x_j)`, which is one coset NTT of size `n` over `Fp2` per combined row from the zero-padded coefficient vector (`tv-field::ntt::coset_ntt` takes coefficients and `Fp2` implements `NttValue`, read 2026-10-16); the inverse NTT of size `k` that `enc(u_g)` needs today disappears. `Z_H(x_j)` becomes `x_j^k`.
- Everything upstream of the commitment is a data vector: the evaluation check of step 6 (`⟨u_g, eq(column sub-point)⟩`) is over the message vector, every claim in [RFC-0010](../../docs/rfcs/RFC-0010-proof-schedule.md) is an MLE inner product `Σ_x f[x]·eq(r, x)`, and [RFC-0012 'Opening by the folding pass'](../../docs/rfcs/RFC-0012-weight-commitment.md#opening-by-the-folding-pass) reveals `u_T` as a combination of fold rows `F_ℓ` checked against `⟨u_T, eq ⊗ eq⟩`. None of them reads `f` at a point of `H`; none changes.
- Descriptor: the message basis is part of the commitment's definition, so the descriptor hashed into `commitment_id` (or the protocol version string) pins it; nothing else in [RFC-0004](../../docs/rfcs/RFC-0004-hashing-and-transcript.md) changes.

## Why it could work

- Operation count (E0, primary, own count 2026-10-16). Butterflies per committed element today: `(k/2·log k + n/2·log n)/k = 11 + 23 = 34` at `k = 2^22`; in coefficient form `23`: −32% of the encoder's arithmetic. The hash is untouched.
- Encoder kernel split (E1, primary, this session: Apple M4 Max, one core, cargo 1.95.0, `cargo bench -p tessera-pcs --bench throughput`, fastest of 5 runs): `encode_row` at `k = 2^22`, `P = 200` costs 32.79 ns per row value (137.52 ms per row); `intt` is 10.05 ns per point at `2^20` and 12.18 at `2^23`, so the size-`k` interpolation step is about 11 of the 32.79 ns (33–35%), consistent with the butterfly count; `coset_ntt` of size `2^23` alone is 14.08 ns per point = 28.2 ns per committed element, so an encoder that keeps only that pass lands near 22–28 ns per committed element.
- Encoder against hash (E1, primary, this session, same core, scratch harness over `tessera_hash::ColumnChains`, `B = 32`, `2^21` columns, fastest of 4 blocks): the chained column hash costs 9.62 ns per codeword symbol = 19.24 ns per committed element at rate 1/2. Row-level `c_c` proxy on this CPU: `32.79 + 19.24 = 52.0` ns per committed element, encoder share 63%; with the encoder −32% the proxy falls to about 41 ns (−21%). The GPU ratio is unknown (RISK-1, RISK-8): the −21% carries over only if the encoder keeps the same share there (E0 extrapolation).
- Bytes (E0, primary, own count): step 1 is a read-and-write pass over `8 B` per committed element (16 B) that a fused kernel can keep L2-resident for `k = 2^22` (32 MB); the size-`n` pass and the hash read at least `16 B` per committed element each. If the GPU kernel is bandwidth-bound the saving is one pass of 16 B out of roughly 50–60 B, the same order as the arithmetic count.
- Framework (E0, primary): the tensor-code commitment of [GLSTW23] needs only a linear code `Enc`; its evaluation check is an inner product on the message and its consistency test compares `Enc` of the combined message with the opened columns. Nothing in that analysis, nor in RFC-0011's Ligero-style bound ([AHIV17]), names the message basis. The verifier work (E0, own count): per combined row `2^21·22 + 2^22·23 = 1.43×10^8` butterflies over `Fp2` today against `0.97×10^8` after, −32% of the verifier's NTT work; second order next to hashing 0.72 GB of columns. Note: RFC-0011's verifier-work row lists only "`G + 1` coset NTTs of size `2^23`" and omits the inverse NTT of size `2^22` that `enc(u_g)` requires (barycentric evaluation at `τ_A` points would cost `τ_A·(G + 1)·k ≈ 10^10` `Fp2` operations, far more); this idea removes exactly the unlisted step, so the superseding RFC should correct that row either way.

## Expected gain

Historical 2026-10-16 projections, retained for provenance. They are not current gain estimates; see the 2026-10-19 review and [discussion #393](https://github.com/AbdelStark/tessera/issues/393).

Scenario: `c_c` −20% (the encoder at 63% of `c_c`, as measured on CPU, times the encoder's −32%), from 30 to 24 ps per committed element.

```sh
python3 -I research/tools/cost_model.py --set commit_ps=24
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: commit_ps=24
   S   linear  attention  weights  regen  streaming    total
   1K     3.02       1.05    0.021    4.0       0.42      8.5   Δ   -4.1%
   4K     2.97       4.14    0.021    4.0       2.32     13.5   Δ   -4.3%
  16K     2.80      15.59    0.020    4.0      11.24     33.7   Δ   -4.2%
 128K     1.82      81.08    0.013    4.0      77.93    164.8   Δ   -3.8%
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −4.1% (E0 from the model; inputs E1 CPU, E0 extrapolated to GPU) | 0 (E0: no message or parameter changes) | −32% of the verifier's NTT butterflies (E0, own count), a few percent of `verify`, which hashing the opened columns dominates (E0) |
| 4K | −4.3% (same) | 0 | same |
| 16K | −4.2% (same) | 0 | same |
| 128K | −3.8% (same) | 0 | same |

Pessimistic bracket, if the GPU encoder holds only a third of the CPU's share of `c_c` (`c_c` −10%): `python3 -I research/tools/cost_model.py --set commit_ps=27` gives Δμ −2.1% / −2.2% / −2.1% / −1.9% at 1K / 4K / 16K / 128K (E0). The scenario moves `c_c` only; `c_ℓ`, `c_π`, counts, `regen` and `stream` are untouched, which is why the gain is nearly flat across contexts (`c_c` enters both the linear and the attention term).

Inputs: `c_c = 30 ps` is the RFC-0020 assumption (E0, RISK-1); the 63:37 split and the −32% are E1 on an Apple M4 Max core (primary, this session); their transfer to the H100 is E0 until the v0.1 gate ([RFC-0020](../../docs/rfcs/RFC-0020-cost-model.md)) measures the kernel.

## Security gate

Current assessment, 2026-10-19; discussion #393 does not close these gates.

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | E0: coefficient/message injectivity and unchanged RS dimension support the existing distance inputs. The accepted version, exact column equation/order, descriptor/payload binding and independent verifier/conformance tests remain to be specified and reviewed. No complete protocol validation is claimed. | unknown (acceptance gates open) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | No gadget, table or true-integer range change is requested. This is not new evidence about gadget soundness. | pass (unchanged surface only) |
| Zero knowledge (G-5, RFC-0013) | Full row rank of `[x_j^{k+i}]` proves marginal hiding only. Joint simulation with combined rows, `Fp2` blinders, claims-before-batching order and persistent epoch budgets is unreviewed; the complete required simulator is absent. | unknown (blocks ZK adoption) |
| Verifier independence (SP-6) | A future verifier must be written from the accepted RFC with independent oracle vectors, not copied from the prover. No implementation is supplied here. | conditional; implementation gate open |
| Post-quantum (hash-based commitments) | The recommended coefficient-only format retains the current derive-key chain and Merkle primitive. No new post-quantum assurance is established here. | unchanged assumption |

Fiat–Shamir (INV-FS-1..4): retain the existing schedule, including blinded claims before batching and combined rows before queries. A new layout discriminator and version must be explicitly bound; the old descriptor does not gain those fields automatically.

## Prior rejection check

- [RFC-0011 Alternatives Considered](../../docs/rfcs/RFC-0011-advice-commitment.md#alternatives-considered) 4 rejects *systematic* encoding because "opened systematic symbols are witness entries, which would need a separate hiding layer". Coefficient form is not systematic: every symbol `g(x_j)` is a full linear combination of all `k` message coefficients plus the mask, so the hiding argument above carries over unchanged. The item does not cover this idea.
- [baseline.md §7](../baseline.md#7-already-rejected), the RFC-0011 row, lists the same five alternatives (storing the encoded witness, FRI-family PCS, Brakedown codes, systematic encoding, per-tensor commitments); none concerns the message basis. RFC-0012 and RFC-0013 'Alternatives Considered' have no item on it either (read 2026-10-16).
- None covers the idea, after the check. What is new is the measured split of the encoder's cost (E1 above): the RFCs chose evaluation form without recording a reason, presumably because it is the textbook Ligero layout.

## Kill criterion

The v0.1 Nsight profile of the GPU encode-and-hash kernel shows the size-`k` inverse step at less than about 10% of the kernel's time (for example because it runs L2-resident inside a single HBM pass of a bandwidth-bound kernel), which makes `Δc_c < 3%` and Δμ below the cost model's input uncertainty; or a protocol step is found that consumes `f`'s values on `H` (none found in RFC-0010 to RFC-0013: every evaluation claim is an MLE inner product over the data vector).

## Cheapest experiment

Hypothesis (pre-registered here): in the same harness on the same core, a `CoefficientRowEncoder` (steps 2–4 of `encoding.rs` without the Cooley–Tukey step, mask written at the `k + i` positions only) encodes a `k = 2^22`, `P = 200` row in 30 ± 5% less time than `RowEncoder`, and on the verifier side `tv-field`'s `coset_ntt` of size `2^23` over `Fp2` alone costs 30 ± 5% less than `intt` of size `2^22` followed by `coset_ntt` of size `2^23`.

Method: a scratch Cargo workspace (empty `[workspace]` table) depending on `tessera-pcs`, `tessera-field` and `tv-field` by path; both encoders timed on the same seeded rows, fastest of 5; a decoding test that the coefficient-form codeword lies in `RS[n, k + P]` (degree check by interpolation on random subsets, as RFC-0011's 'Testing Strategy' does for the current encoder) and that symbol `j` equals `g(x_j)` under the chosen column order; a hiding test that the mask's values at `P` random positions have full rank over `F65537`.

Evidence level reachable here: E1 (Apple M4 Max, one core). Effort: `s`. What stays unknown until an H100 run: the encoder's share of the fused GPU encode-and-hash kernel (E2, the v0.1 gate of RFC-0020), which decides whether the CPU's −21% on `c_c` transfers.

## Refutations

Historical 2026-10-16 triage. Its security conclusions are superseded by the current Security gate and 2026-10-19 review; a previous “survives” is not a present full-security pass.

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | RFC-0011's proximity argument is stated on codewords of RS[n, k+P] and unique decoding, not on the message basis: coefficient form replaces the split g = f + Z_H·m by g = f + X^k·m, both unique for a uniquely decoded g, so the extractor and the (1 − δ)^τ_A + n/q' bound carry over unchanged (E0, primary, refuter's derivation). Both evaluation checks are inner products on the message vector (RFC-0011 step 6, RFC-0012 step 3) and read no value of f on H, so no claim changes and no second witness appears. The hiding lemma of RFC-0013 'Hiding openings' transfers: the mask's contribution at τ ≤ P distinct non-zero x_j ∈ 7·L is diag(x_j^k)·Vandermonde, full row rank, so opened symbols stay uniform; Q_g, M_g and the challenge schedule are untouched (INV-FS-1..4). The superseding RFC must pin the verifier-side formula of step 8.3 with x_j^k in place of Z_H(x_j) and option (b)'s column relabelling in the descriptor: a prover/verifier disagreement on symbol order is a completeness failure, not a soundness hole. |
| prior art | survives | Not in the specification and not rejected: RFC-0011 'Encoding' fixes evaluation form on H with the Z_H·m mask and the opening check interpolates enc(u) on H, but no RFC records a reason for that basis; RFC-0011 Alternatives Considered 4 rejects only systematic encoding, which this is not; RFC-0012/0013 Alternatives and baseline.md §7 have no item on the message basis. Brakedown-style tensor commitments ([GLSTW23]) need only a linear code and never read the message at points of H, consistent with every claim in RFC-0010 being an MLE inner product. Coefficient-form RS is textbook coding theory; new is Tessera-specific: the measured 63:37 encoder:hash split (E1, primary) and the observation that RFC-0011's verifier-work row omits the size-k inverse NTT the current check needs. Tier-A change (new commitment descriptor), as the note says. |
| cost model | survives | `--set commit_ps=24` reruns to Δμ −4.1 / −4.3 / −4.2 / −3.8% at 1K / 4K / 16K / 128K and the pessimistic `commit_ps=27` to −2.1% (E0). c_c is 19–21% of μ at every context (baseline.md §2) and the saving is a fraction of c_c rather than an absolute figure, so it is robust to the 2–4× uncertainty on c_c itself: under IDEA-0004's count dropping 11 of 34 butterflies is about −24% of the kernel's compute, under IDEA-0003's byte table it removes 16–32 B of 108 B, and the note's −20% sits inside both; if c_c is repriced to 60–90 ps the same fraction is worth −6% to −9% of μ. Modest, but larger than the input noise, flat across contexts, and the one commitment-lens gain that grows if c_c is worse than assumed. |
| feasibility | survives | The E1 experiment is feasible here at effort `s`: the four-step encoder is in `crates/tessera-pcs/src/encoding.rs`, the bench it extends exists (`crates/tessera-pcs/benches/throughput.rs`: `intt`, `coset_ntt`, `encode_row` at k = 2^22, P = 200) and the verifier already has `coset_ntt` over Fp2 (`verifier/crates/tv-field/src/ntt.rs`). Adoption is `l`: the codeword is tier-A protocol semantics, the verifier's opening check it changes has landed (`verifier/crates/tv-core/src/opening.rs`, PR #234), the codeword conformance vectors pin today's encoding (`crates/tessera-pcs/tests/vectors.rs`, `spec/vectors/pcs/generate.py`, `spec/vectors/encoding/`), and four RFCs (0011, 0012, 0013, the RFC-0004 descriptor) plus `gen-vectors` move; option (b)'s relabelling touches query-index semantics on both sides. No hardware needed; the only unknown (the GPU encoder share) is correctly deferred to E2. |

## Score

Historical 2026-10-16 scores and confidence values below are retained, not recomputed or endorsed under the current partial baseline. Unknown security gates block promotion regardless of projected gain.

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −4.2% μ(16K), −4.1% at 1K (E0 cost model; inputs: E1 encoder:hash split on one M4 Max core, E0 transfer by butterfly count); `verify` also loses G + 1 inverse NTTs of size 2^22 over Fp2 (E0) | 0.6: the mechanism survives all four lenses and the restated hiding lemma is a standard full-rank argument, but the encoder share is measured on a CPU and its share of the fused GPU kernel (the kill criterion) is unknown until E2 | `l` for adoption: a tier-A superseding RFC to RFC-0011/0012/0013 and the commitment descriptor, with the verifier's opening check and the conformance vectors moving; the experiment itself is `s`, E1 here, no hardware | 4.2 × 0.6 / 4 = 0.63 |
| Re-score 2026-10-16 under `counted` (`c_c` 94 ps, EXP-0002): `c_c` −20% gives −6.6% / −6.3% / −5.9% / −5.5% of μ at 1K / 4K / 16K / 128K (−3.3% / −3.2% / −3.0% / −2.8% if the GPU encoder holds a third of the CPU share); with the fifth pass priced (regen 5, `c_c` 167 ps, EXP-0007) the honest coefficient-form baseline is −5.8% / −5.3% against the honest evaluation-form one at 1K / 16K (21.3 vs 22.6, 92.6 vs 97.8), and −7.2% / −7.2% with the pruned column pass on both sides (19.2 vs 20.7, 83.6 vs 90.1); E0 cost model | 0.7: EXP-0007 measured the same-harness E1 split (`encode_row` 139.6 ms against the coset transform alone 99.8 ms per 2^23-symbol row: the interpolation NTT is 28.5% of today's encoder time, near the 32% assumed); the GPU share stays E2 | unchanged (`l`, tier-A RFC) | 5.9 × 0.7 / 4 = 1.0 |

## Decision and links

Triage 2026-10-16: `triaged`, priority 0.63, the second-ranked design idea of the batch after IDEA-0011. Next step: the E1 same-harness bench (encoder with and without the size-k inverse NTT, column hash unchanged) as a pre-registered experiment; the kill criterion (GPU encoder share below about 10% of the fused kernel) waits for RFC-0020 v0.1. Experiments: none yet. RFC or issue: none yet (a superseding RFC to RFC-0011/0012/0013 if validated). Ledger line updated on: pending (the consolidation agent writes it). Re-scored 2026-10-16 (queue item 9) under the `counted` profile with the fifth regeneration pass of [EXP-0007](../experiments/EXP-0007-pruned-column-answer-pass/README.md) priced: priority 0.63 → 1.0 (bracket 0.93–1.26), confidence 0.6 → 0.7 on EXP-0007's same-harness split; the remaining unit of work is the tier-A descriptor RFC, into which IDEA-0014's keyed-chain clause folds (queue item 12).

Review 2026-10-19: discussion [#393](https://github.com/AbdelStark/tessera/issues/393) supersedes the historical plan to fold in keyed chaining without a separate security gate. Status remains `triaged`; no promotion or current gain re-score.
