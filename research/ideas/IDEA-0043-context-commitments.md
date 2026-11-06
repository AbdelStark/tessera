# IDEA-0043: Context commitments: prove a long request as binary-aligned chunks linked by committed keys and values

| | |
|---|---|
| Status | `triaged` |
| Origin | [#470](https://github.com/AbdelStark/tessera/issues/470), OQ-0057-1, RFC-0057 Alternative 3; ledger item 54 (shared prefixes); [SUR-0017](../surveys/SUR-0017-chunked-requests.md) |
| Lever | `size` per proof (the 4 GiB gate of 08 §5) and the existence of a statement above about 24K tokens; no Δμ credited |
| Spec surfaces | A new statement form (tier A: statement fields, messages, transcript labels, verifier checks): RFC-0007/0039 statement, RFC-0036 messages, RFC-0014 schedule and soundness budget, RFC-0019 checks, RFC-0037 claim atoms, RFC-0011 commitments, RFC-0012-style budget state, RFC-0013 hiding; supersedes RFC-0057 'Whole requests'. A superseding RFC is required |
| Priority | correctness and feasibility prerequisite for every target above 24K tokens (08 §3's 128K row); numerical score 0.9 on the size metric (below) |

## Mechanism

A transformer position reads earlier positions only through attention over their keys and values. So a request split into token ranges needs exactly one binding across proofs: each chunk's attention reads the post-RoPE keys `K_ℓ` and the values `V_ℓ` of the chunks before it. Everything else (embeddings, residual stream, norms, MLP, sampling) is per position.

1. **Chunks.** A request of `S` positions is cut into binary-aligned blocks: chunk `b` holds positions `[s_b, s_b + C_b)`, `C_b = 2^{a_b}`, with `s_b` a multiple of `C_b`. Any `S` is a sequence of such blocks, the prover's planner choosing `a_b` from the gate (below). Alignment makes each block a subcube of the position axis, which step 3 needs.
2. **The context commitment.** Chunk `b`'s proof commits, in `Round1` beside its advice, a second Ligero matrix `Κ_b` holding its post-RoPE keys and its values, `[L, 2, H_kv, C_b, d_head]` INT8, in RFC-0011's layout rules, with its own root `R^K_b`, vanishing masks and `B_K` blinding rows. In chunk `b` these two tensors are declared committed in `Κ_b` instead of virtual or advice: `Rotate` proves its output claim against `Κ_b` (RFC-0037 'Rotate'), and the value `Requant`'s output claims move there. Its proof opens `Κ_b` like any committed atom: evaluation and proximity combined rows, `τ_A` columns.
3. **Reading the past.** In chunk `c`, the key and value tensors the attention reads span positions `[0, s_c + C_c)`. The atom model of RFC-0037 sees them as the sum of one atom per earlier chunk and the chunk's own. A claim at `(ℓ, h, i, d)` with the position point `i = (i_hi, i_lo)` splits as
   `K̃(ℓ, h, i, d) = Σ_{b ≤ c} eq(i_hi^{(b)}, s_b / C_b) · K̃_b(ℓ, h, i_lo^{(b)}, d)`,
   where `i_lo^{(b)}` is the low `a_b` coordinates of `i` and `i_hi^{(b)}` the rest. Each `eq` factor is public. One stage split sends the `c` values `K̃_b(…)` under a label of its own, and each becomes a committed atom on `Κ_b`, with the point shared by every atom on that commitment. Positions past the last real key are padded zero, as RFC-0035 says.
4. **Opening earlier commitments.** Chunk `c` opens each `Κ_b`, `b < c`, at its claims: one evaluation pair and one proximity pair of blinding rows of `Κ_b` per consuming proof, a mask row per claim in `tessera-protocol/2`, and `τ_A` columns of `Κ_b`. The combined rows of several `Κ_b` of one width batch into one row with a challenge, so the size grows by one combined row (`2^{log2 k}` `Fp2`) plus `τ_A` short columns per earlier chunk.
5. **The statement chain.** Chunk `c`'s statement adds: the position offset `s_c` (RoPE's angles and the causal selector read it), the chunk's own tokens, and the digest of chunk `c − 1`'s statement together with the roots `R^K_0 … R^K_{c−1}`. The verifier of a chunked request verifies the chunks in order and accepts the request when every chunk accepts and each statement names its predecessor's digest and roots. A chunk's sampled token feeds the next chunk's first input exactly as RFC-0015's anchor already chains positions.
6. **The budget.** `Κ_b` is opened by its producer and by every later chunk: at most `N − b` openings, `N` the chunk count. Each opening burns four blinding rows (its evaluation and proximity pairs), and in `tessera-protocol/2` one mask row per claim, so `B_K ≥ (N − b)·(4 + claims per opening)` rows, and at most `τ_A·(N − b)` columns are revealed against its masks' `P`; both are fixed at commit time. Its secret state (masks seed, opened columns) follows RFC-0012's persisted state machine and is cleared when the request's last chunk is proven. A shared prefix (ledger item 54) is the same object with an operator-chosen consumer count.

## Why it could work

- The binding is the only cross-position dependency of the template (RFC-0037 'Layers and kinds'; the lowering's attention reads `K`, `V` through `Transpose`/`Expand`/`Reshape` only), so linking keys and values is complete, not a partial binding (E0, primary: the specification).
- Persistent, hiding, repeatedly opened Ligero commitments are already specified and budgeted for the weights (RFC-0012 'Budget state machine', RFC-0064), including their proximity per proof (RFC-0049).
- Linking is small. `Κ_b` holds `2·L·H_kv·d_head = 258,048` INT8 per position at 405B (126 layers, 8 KV heads, 128): 4.2 GB at 16K positions, about 1,000 committed rows at `k = 2^{22}`. An opening costs one combined row (67 MB, shared across earlier chunks) and `τ_A` columns of about 8 KB (about 1 MB per earlier chunk) (E0).
- No published system binds a KV cache across separately verified proofs: zkLLM, zkGPT, VerInf and DeepProve prove a sequence in one proof or chunk by layers, and DeepProve calls KV caching "poison for SNARKs" ([SUR-0017](../surveys/SUR-0017-chunked-requests.md): [DeepProveChunk26] primary code, [Lagrange25b] secondary). zkComposer commits each token partition's KV values and feeds them to later partitions, but within one session, over Hyrax (not post-quantum) and outside its theorems ([SGO+26b], E1 primary).
- VerInf states the same hiding rule for a commitment opened in several proofs: the union of revealed columns is the budget ([Petrie26b], primary), which is RFC-0012's and step 6's.
- Separate Ligero proofs linked through challenges drawn after both commitments are fixed are deployed and machine-checked in longfellow-zk ([LongfellowMac26], primary formal proof).
- The failure mode is known and recent: Jolt's verifier did not bind execution to a public trusted-advice commitment ([QED26], primary disclosure, checked 2026-11-06), and OpenVM's CVE-2026-62333 let a prover change a committed memory root ([OpenVM26], primary advisory). The chain check of step 5 is where this idea can fail, and its adversarial tests come first.

## Expected gain

The metric moved is proof size per proof against 08 §5's 4 GiB gate, and the existence of a statement above 24K tokens. [EXP-0032](../experiments/EXP-0032-chunk-proof-sizes/README.md) measured the exact `ModelAdviceLayout` non-zero rows of whole 405B `servable` requests (`ni-100` layout) at nine sizes from 4K to 64K (E1, RFC-0057's three rows reproduced exactly). The rows are `c + a·n + b·n²` with `c ≈ 36,800` rows per proof, `a ≈ 25.7` per token and `b ≈ 1.90×10^{-3}` per token², within 3% (the registered two-term fit missed by 18% at 4K and the experiment is `refuted` as registered). A chunk of `C` tokens at offset `P` adds the rows of its positions, its scores against every earlier key included, plus the per-proof constant. At 1,240 bytes of opened columns per row, a non-column part of at most 2.02 GB and 0.07 GB of linking, the planner takes the largest aligned block that fits the gate ([plan.py](../experiments/EXP-0032-chunk-proof-sizes/plan.py)):

| Request length | Proofs | Total | Largest proof | Chunk sizes |
|---|---:|---:|---:|---|
| 16K | 1 | 3.3 GB | 3.05 GiB | 16K |
| 32K | 3 | 9.9 GB | 3.18 GiB | 16K, 2 × 8K |
| 64K | 9 | 31.1 GB | 3.83 GiB | 16K, 4 × 8K, 4 × 4K |
| 128K | 31 | 109.6 GB | 3.94 GiB (about 3.98 with the per-proof constant) | 16K, 4 × 8K, 14 × 4K, 12 × 2K |

Bytes are E0 Tessera projections on E1 rows; the byte model is RFC-0057's. SUR-0017's independent E0 fit (exact causal score count, unaligned chunks of 3.3K–4.1K tokens at the end of a 128K request) gives 32–40 proofs and about 45 GB of opened columns summed over any chunking, 90% of it scores; the two counts differ in alignment and in the non-column bytes assumed, not in the conclusion. The non-column part is the conservative maximum of RFC-0057's three rows and does not shrink with the chunk, so small late chunks are overestimated. Δμ: the total score work is the same `S²/2` pairs as one proof; each proof adds its table side and weight fold, about 0.74 GPU-s (RFC-0057 'What a unit costs'), 23 GPU-s over 31 proofs at 128K (E0).

The table's lesson: at 128K the committed scores, not the linking, set the chunk size (2K tokens late). The lever that would shrink the proof count is committing fewer scores (#393, IDEA-0039's score instances), which this idea makes measurable per chunk.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Adds per chunk: the split's field term (one claim per earlier chunk at one point, `≤ 1/\|Fp2\|` each from the sent values' random combination), and per opened `Κ_b` an evaluation term `1/\|Fp2\|`, a proximity term `ϑ·n/\|Fp2\|` and the `τ_A` query term, as for any committed matrix. Every `Κ_b` root is fixed in an earlier statement that this chunk's transcript absorbs before any challenge, so its openings use fresh challenges. Each consuming proof tests `Κ_b`'s proximity itself, so no chunk's soundness rests on another session's challenges | pass (to be written as an RFC-0014 term) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Earlier keys and values are committed outputs of their producer's `Rotate` and `Requant`, whose witnesses are unique there; the consumer reads them as committed INT8 and range-checks nothing new | pass |
| Zero knowledge (G-5, RFC-0013) | Each opening of `Κ_b` reveals a masked evaluation and `τ_A` columns: hidden by a fresh blinding pair and mask row per opening and by the masks' `P` against at most `τ_A·(N − b)` revealed columns. The budget is fixed at commit time. The simulator argument (#145) must cover a commitment opened across sessions | unknown |
| Verifier independence (SP-6) | New statement fields, split and opening checks, written in `tv-core` from the RFC | pass |
| Post-quantum (hash-based commitments) | Ligero and BLAKE3 only | pass |

## Prior rejection check

RFC-0057 Alternative 3 names exactly this partition and defers it to #470; it is not rejected. Alternative 1 (layer-range units) is rejected for reasons that do not apply: units shrink neither the proof nor the scores, while token chunks shrink both. baseline.md §7 has no row on sequence chunking. New since RFC-0057: the binary alignment that makes the split a product of public `eq` factors, and the observation that keys and values are the only cross-position state.

## Kill criterion

If the exact per-chunk layout (EXP-0032) shows that a late chunk at 128K cannot fit the 4 GiB gate at any aligned size of at least 1,024 tokens, or that the non-column part alone exceeds the gate, the chunked statement does not reach 128K under the gate and #470 must restate the target instead.

## Cheapest experiment

EXP-0032 (run 2026-11-06, see its report): the exact `ModelAdviceLayout` rows of whole 405B `servable` requests at nine sizes from 4K to 64K, a fit, and a binary-aligned chunk plan under the gate, a chunk charged the rows its positions add to the whole request. E1 rows, E0 bytes, effort `s`. Still open: an exact chunk layout, which needs a lowering at an offset with cached keys, and per-chunk proving time (RFC-0020's H100 gate).

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The two dangerous readings were checked. (1) Trusting the producer's proximity test across sessions: avoided, since every consumer tests `Κ_b` again with its own challenges. (2) Prover-chosen split values: they are sent before the challenge that combines them, and each lands on a commitment fixed in an earlier statement. Open: the RFC must state the chain check precisely, since a verifier that accepts chunk `c` without having accepted chunks `0..c−1` would certify keys nobody proved. |
| prior art | survives | Deferred, not rejected, by RFC-0057 Alt. 3. SUR-0017 finds no system that links a KV cache across separately verified proofs; zkComposer does it inside one Hyrax session. zkVM continuations hand committed state between segments, and two of them broke in 2026 on exactly that handoff ([QED26], [OpenVM26]) |
| cost model | survives | Linking adds about 2% of a 4 GiB proof (E0); the proof count is set by scores, which the idea does not worsen. Δμ is not this idea's lever |
| feasibility | weakened | Tier A: a new statement form, labels, messages and verifier checks, plus a persisted secret state per request. Effort `l` |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| Statements above 24K tokens exist at all; 128K in 31 proofs under the gate (E0 bytes on E1 rows) | 0.6: the mechanism follows the spec's own atom model and RFC-0012's budget; numbers are a fit | `l`: tier-A RFC, weight 4 | correctness/feasibility prerequisite for v1.0's 128K target; ordered by #470, not by Δμ |

## Decision and links

Experiments: [EXP-0032](../experiments/EXP-0032-chunk-proof-sizes/README.md) (`done: refuted` as registered; kill criterion not met). RFC or issue: [#470](https://github.com/AbdelStark/tessera/issues/470). Ledger line updated on: 2026-11-06.
