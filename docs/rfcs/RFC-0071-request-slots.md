# RFC-0071: A Batch of Requests in One Proof Unit: Request Slots on One Token Axis

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends, in both protocols:
  - [RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering': one graph covers one batch;
  - [RFC-0044](RFC-0044-lowering-nodes.md#graph-inputs) 'Graph inputs' and 'A layer': the token axis and the scores of a batch;
  - [RFC-0035](RFC-0035-padded-rows.md#the-row-selector) 'The row selector': a request's real rows;
  - [RFC-0030](RFC-0030-masked-exponentials.md#the-causal-selector) 'The causal selector' and its root claim;
  - [RFC-0034](RFC-0034-causal-blocks.md#zero-rows) 'Zero rows': per request;
  - [RFC-0037](RFC-0037-stage-schedule.md#rotate) '`Rotate`': a position is a row's place in its request;
  - [RFC-0041](RFC-0041-execution-plan.md) 'Non-Goals' and 'The plan': a batch's tiles;
  - [RFC-0047](RFC-0047-advice-order.md#the-maximum-layout) 'The maximum layout' and 'A request's sequence length';
  - [RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-request-and-sources) 'Public request and sources': each request's certificate on its own rows;
  - [RFC-0017](RFC-0017-scoring.md#in-circuit-scoring-mixture) 'In-circuit scoring' and RFC-0035's output rows: each request's scoring window;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 4, and [03 §4](../spec/03-data-model.md#4-statement-and-protocol-types): the batch's rules.
- Origin: [#357](https://github.com/AbdelStark/tessera/issues/357)

## Summary

A statement covers a batch of requests (`PublicStatement.batch`, 03 §4), and the audit packs many into one batch (RFC-0016 'Batching'). No document defines how one proof unit proves several: RFC-0006 lowers "one sequence", RFC-0035's row selector has one length `n`, RFC-0041 orders one sequence's tiles. So a batch of one request is the only one a prover can prove.

This RFC puts each request in a **slot** of one token axis. With `S = 2^{⌈log2 max_r n_r⌉}`, request `r` of the batch's `R` holds rows `r·S` to `r·S + n_r − 1`, its positions `0` to `n_r − 1`. The rows between a request's end and the next slot are padded rows, as RFC-0035 pads a sequence today.
- **Per-token relations are unchanged.** Every relation that works row by row sees a longer token axis, whose high bits name the request.
- **Position-aware ones read a row's place in its slot.** Those are attention's keys, the causal selector, RoPE, the row selector, the greedy certificate and scoring. Each of their public extensions becomes a sum over the requests, `O(R·log S)` for the verifier.
- **A batch of one request is today's proof,** bit for bit.

At 405B, a 16K-token audit batch of 16 requests of 1K takes about 2.06 GB of proof. Proved one at a time, the same requests take 25.4 GB, because each proof pays its combined rows and weight openings again.

## Motivation

- **Audits batch requests.** RFC-0016 packs a sample's requests by first-fit decreasing up to `max_batch_tokens`, 16,384 by default, and each batch is one statement. Requests are typically far shorter than a batch: a 16K batch at 1K of context holds 16.
- **What a proof costs regardless of its size.** The advice opening's combined rows, `G + 1 ≤ 17` rows of `2^{22}` `Fp2` values, are 1.14 GB in every proof. The weights' openings add about 0.35 GB (08 §5). A proof per request pays both each time.
- **The cost model already assumes batches.** RFC-0056 prices `T` tokens at `S` positions of context as `T/S` whole sequences of `S`, and RFC-0054 and RFC-0055 index one padded token axis of the batch. The lowering, the selectors, the layout and both workspaces' code follow one sequence. The verifier's driver refuses a batch of two (`driver.rs`, `let [request] = …`), and so does the CLI's `prove --sample` (#357).

## Goals

- One proof unit for a batch of whole requests, in statement order.
- No new relation, challenge, message or opening group: the schedule's stages and claims are the same for every batch.
- The verifier derives every slot from public data: the batch descriptor's lengths.
- A batch of one request proves and verifies exactly as today, so `tessera-protocol/1`'s artifacts and vectors are unchanged.

## Non-Goals

- **A request longer than one proof holds**: chunks and their linking (#470). A batch holds whole requests from their first token (RFC-0057).
- **Several predictors in one batch.** `PublicStatement` holds one `PredictorSpec` for the batch, and this RFC keeps it. Per-request temperatures stay OQ-0036-1.
- **Packing short requests into one slot.** A slot holds one request ('Alternatives Considered' 4).

## Proposed Design

### The slots

A batch is `statement.batch.requests`, `R ≥ 1` requests in that order. Request `r` has `n_r = prompt_len_r + output_len_r` positions (RFC-0047).
- **The slot length** is `S = 2^{b}`, with `b = ⌈log2 max_r n_r⌉`.
- **The token axis.** Request `r`'s position `p < n_r` is row `τ = r·S + p` of the batch's token axis. The axis has `N = (R − 1)·S + n_{R−1}` rows, and is padded, as every axis is, to `2^{⌈log2 N⌉} = 2^{b + ⌈log2 R⌉}` rows.
  - Its low `b` variables are the position within a slot.
  - Its high `⌈log2 R⌉` variables are the request.
- **Real and padded rows.** A row is **real** when its position is below its request's length: `τ mod S < n_{⌊τ/S⌋}`. Every other row of the padded axis is a **padded row**, with RFC-0035's values: the rows between `r·S + n_r` and `(r + 1)·S`, and the axis's padding. A padded row's token is 0 and its activations are 0, and no real output reads it.
- **One request.** With `R = 1`, `N = n` and every row below `n` is real: the token axis is today's.

### The lowering

One graph covers one batch (RFC-0006 'Llama lowering'). `lower(model, lengths)` takes the batch's lengths in order, and `lower(model, n)` is its one-request case. Every shape below has `N` where RFC-0044 has `n`:
- **Inputs.** The tokens `[N]`, committed advice, and the positions `[N]`, public: position `τ mod S` at a real row and 0 at a padded one.
- **Per-token nodes.** These are the activations `[N, channels]` and every node that works row by row: matmuls by weights, norms, requantizations, SiLU and the residual. They apply to each row as before. Their padded rows hold RFC-0035's values.
- **Attention.** Keys are read within a slot.
  - The scores are `[n_heads, N, n_max]` with `n_max = max_r n_r`. Entry `(h, τ, j)` is query row `τ` against key `j` of `τ`'s own request: `Σ_d Q[h, τ, d]·K[h, ⌊τ/S⌋·S + j, d]`.
  - On the padded axes this is a matmul batched over the head and the request, `[n_heads, R, S, d_head] × [n_heads, R, d_head, S]`. The token axis's reshape into `[R, S]` is the identity on the cube, since `S` is a power of two.
  - `SoftmaxRow` reduces each row's `n_max` keys under the causal selector below. `MatMulAct(e, v)` reads values within the slot as the scores read keys.
- **RoPE.** It rotates each row by its position `τ mod S`, the public positions input. The rotate kernel's `C` and `S` read the same position (RFC-0037 '`Rotate`'): the verifier derives them from the batch's lengths, never from the row index alone.
- **Logits** are `[N, vocab]`; the next token at row `τ` is row `τ + 1`'s within the same request.

With `R = 1`, `N = n` and `n_max = n`, and the graph is RFC-0044's.

### The public selectors

Each public predicate that names a length becomes a sum over the requests. Write a point of the padded token axis as `(r_s, r_q)`: `r_s` the slot's `b` variables, `r_q` the request's.
- **The row selector** (RFC-0035) is `ψ(ℓ, h, τ) = [ℓ < n_layers]·[h < n_heads]·[τ real]`, with

  ```text
  ψ̃ = LT(r_ℓ, n_layers) · LT(r_h, n_heads) · Σ_{r < R} eq(r_q, r) · LT(r_s, n_r)
  ```

  For the output gadgets and the scoring bound, the real rows are request `r`'s positions `prompt_len_r − 1` to `n_r − 2`, those whose next token the statement records, in each slot.
- **The causal selector** (RFC-0030) is `C(τ, j) = [j ≤ τ mod S < n_{⌊τ/S⌋}]`. Its extension over the score instance's query variables `(r_s, r_q)` and key variables `r_k` is `Σ_{r<R} eq(r_q, r)·C_r(r_s, r_k)`, with `C_r` RFC-0030's selector for one sequence of `n_r`.
- **`SoftmaxRow`'s root claim** is `1 − LT(r_ℓ, n_layers)·LT(r_h, n_heads)·Σ_{r<R} eq(r_q, r)·LT(r_s, n_r)`. A real row's tree has root 0, and every other tree root 1.
- **Zero rows** (RFC-0034). A score tile's row is skipped by the rule of one sequence, applied to the query block's request with its length `n_r`. A query block lies within one slot ('The plan').

Each sum has `R` terms of `O(log S)` each. With `R = 1` each is today's formula: `eq` over zero variables is 1.

### The greedy certificate and scoring

**Greedy.** Each request keeps RFC-0053's certificate, with ordinal `r`, its index in the batch:
- its maximum `M` on its own rows, real on positions `prompt_len_r − 1 ≤ p < n_r − 1` of its slot and zero elsewhere;
- its key column and its key relation, whose time coefficient is `Σ_{u < output_len_r} eq(P_row, u)·eq(r_time, (r, prompt_len_r − 1 + u))`, with `(r, p)` row `r·S + p`;
- its observation instance when `output_len_r = 0`.

The selected and token-binding relations run once over the whole token axis, as zero-checks:
- **Selected.** Its `active(τ)` is 1 exactly on the rows of the windows above.
- **Token binding.** It reads every real row's token.

`Hplus[τ, v] = H[τ + 1, v]` is read only at active rows, where `τ + 1` is the same request's next position, so no relation reads across a slot.

**Scoring** (RFC-0017):
- request `r`'s `Û_r` sums its window's rows, positions `prompt_len_r − 1` to `n_r − 2` of its slot;
- the batch's `Û` is the sum of the `Û_r`, in statement order, as RFC-0017 states.

### The plan and the layout

- **The plan's tiles** are independent of the batch (RFC-0041 'The plan', 'Independence of `n`'). An `Attention` tile stands for its query blocks of `Q' = min(Q, S)` rows. The batch's padded token axis expands it into `2^{⌈log2 N⌉}/Q'` tiles in row order: slot by slot, and within a slot by position. Since `Q'` divides `S`, every query block lies within one request. With `R = 1`, `Q' = min(Q, 2^{⌈log2 n⌉})`, as today.
- **The advice layout** is a function of the claim list and the batch descriptor (INV-PCS-1). Its production order is the plan's, so a batch's rows follow the token axis.
- **The maximum layout.** A batch's layout fits in at most `2^{32}` rows (RFC-0047). The claim list's check at `n = max_context` (RFC-0007 step 6) does not cover a batch: its token axis can be longer than `max_context`. So the verifier checks the batch's own layout at statement binding.
- **Token groups** (RFC-0054). They interleave the padded token axis's rows, as RFC-0054 states for a batch: "its requests' positions in claim-list order". Interleaving by the low bits spreads each slot's rows over every group.

### The statement

RFC-0019 step 4 checks the batch, each check a `Reject(StatementMismatch { field: Batch })`:
- `R ≥ 1`, and no request named twice (03 §4, `RequestKey`);
- each request's `prompt_len_r ≥ 1` and `n_r ≤ max_context`, as before;
- the batch's advice layout, derived from the claim list and the batch descriptor, fits the maximum layout.

The requests' order in `statement.batch` is the slots' order. RFC-0016's batches list their requests in `RequestKey` order. The statement digest and the advice commitment's `layout_digest` bind the order (RFC-0031).

The tokens a prover proves are its requests', concatenated in batch order (02 §2.2 `ProofBatch`). Request `r`'s prompt then outputs go to its slot's real rows. The anchor binds each request's frames to its own rows (RFC-0015, INV-ANC-1).

### Soundness and zero knowledge

- **Soundness.** No challenge, message or relation is added. The selectors are public and the verifier computes them; a prover that put a real value in a padded row is caught as RFC-0035 catches it, whatever the row's place. The token axis's `⌈log2 R⌉` request variables add one round each to the relations over it. RFC-0014's calculator counts them through `rounds`, from the statement's sizes.
- **Zero knowledge.** `tessera-protocol/2`'s masks are per tensor and per row, and the slots are public. A request variable is a token variable: a claim's point is random in it, as in every token variable, so no mask's `Z` vanishes more often (R8).

### Cost at 405B

From the compiler's advice layout of one sequence (`ModelAdviceLayout`, `servable` with two classes, `B_max = 32`), at `n` = 256, 1K, 2K, 4K, 8K and 16K:
- **The fit.** The non-zero rows fit `a + b·n + c·n²` within 3.5%, with `a ≈ 52,600` (the tables' rows), `b ≈ 23.0` per token and `c ≈ 1.95·10^{−3}` per score.
- **A batch's projected rows.** Its per-token rows count the padded token axis, its score rows each slot's `S²`, and its tables once: `a + b·R·S + c·R·S²`.
- **The proof** (`tessera-protocol/1`). It is the combined rows, `17 × 2^{22} × 16` bytes, plus the advice columns at `τ_A = 155` and 8 bytes per row, plus 0.35 GB of weight openings. The figures below are projections from the fit, not measurements of a batched layout.

| Batch of 16,384 tokens | One proof (rows, bytes) | One proof per request |
|---|---|---|
| 1 × 16K | 953,580 rows, 2.67 GB | the same |
| 4 × 4K | about 560,000 rows, 2.19 GB | 6.85 GB |
| 8 × 2K | about 495,000 rows, 2.10 GB | 13.0 GB |
| 16 × 1K | about 462,000 rows, 2.06 GB | 25.4 GB |
| 64 × 256 | about 437,000 rows, 2.03 GB | 100 GB |

- **The prover.** RFC-0056's model prices the same batches. A 16K batch at 1K of context costs `μ ≈ 13.8` (`minimal`) and 15.7 (`servable`) per token. A batch of one 1K request alone costs between the 256- and the 2,048-token rows, 16.1 to 14.2 (`minimal`), before its proof's fixed costs, which it then pays alone (RFC-0056 'At 405B').
- **The verifier.** Every selector costs `O(R·log S)` field operations, about `10^3` for 64 requests. Its other work grows with the proof's rows and rounds.
- **Padding.** A slot is as long as the batch's longest request. Requests of very different lengths in one batch pad the shorter ones to `S`. RFC-0016's first-fit decreasing packs requests in decreasing length, so a batch's requests are close in length. A slot-aware packing is OQ-0071-1.

## Alternatives Considered

1. **One proof per request.**
   - **The cost.** Each proof pays its combined rows (1.14 GB) and weight openings (0.35 GB) again: 25.4 GB for 16 requests of 1K.
   - **Why rejected.** Batching was the audit's premise (RFC-0016).
2. **Concatenate the requests' tokens with no slots.**
   - **The idea.** Request `r` takes rows `Σ_{s<r} n_s` onward, which pads the whole axis once.
   - **Why rejected.** A query's keys then start at an offset that is not a power of two. The scores read `K` through an addition of offsets, the causal selector becomes a union of intervals at arbitrary starts, and a tile's query block can straddle two requests. Its per-token rows save at most the slots' padding, while the scores and their key axis keep the longest request's length either way.
3. **Each request its own instance of every relation, padded to its own power of two.**
   - **The idea.** This is #357's first option.
   - **Why rejected.** Each relation's claims multiply by `R`, at points of their own, so the claim pool and the opening groups grow with the batch. Slots keep one instance per relation and add only variables.
4. **Slots of several sizes.**
   - **The idea.** Pack each request in an aligned block of its own power of two, the largest first, so that short requests share a slot's length.
   - **What it would save.** The padding of mixed lengths.
   - **Why not now.** Every per-request selector would then name a block of its own size. The attention relation becomes a sum of instances of different key widths. Left as OQ-0071-1.

## Drawbacks

- **Padding.** A batch of mixed lengths pads every request to its longest.
- **The verifier.** Its selectors cost `O(R)` more evaluations each, small beside its hashing.
- **More to keep in step.** The lowering, the selectors, the greedy certificate and scoring each read the slots in both workspaces, which a differential test must keep equal ('Testing Strategy').

## Migration / Rollout

1. **This RFC.** It amends RFC-0006, RFC-0017, RFC-0019, RFC-0030, RFC-0034, RFC-0035, RFC-0037, RFC-0041, RFC-0044, RFC-0047, RFC-0053 and 03 §4. A batch of one request is unchanged, so no artifact, vector or proof of `tessera-protocol/1` changes.
2. **Implementation** ([#551](https://github.com/AbdelStark/tessera/issues/551)). Both workspaces lower, select, certify and score per slot. Then the verifier's driver and the CLI's `prove --sample` lift their one-request refusal. Until then a proof unit covers one request, and both refuse a batch of two with `ERR-USAGE`, as they do today.

## Testing Strategy

- **Vectors** from an independent oracle, for batches of 2 and 3 requests of different lengths:
  - the lowering's shapes and positions;
  - `ψ̃`, the causal selector and the root claim at random points;
  - the greedy windows and `Û_r`.
- **Equivalence.** A batch of one request gives today's claim list, plan, layout and proof bytes. The tiny models' existing vectors pass unchanged.
- **Adversarial**, each asserting its rejection:
  - a real value in a padded row between two slots;
  - a token of request 1 bound to request 0's frames;
  - a key read across a slot;
  - a statement naming a request twice;
  - a batch whose layout exceeds the maximum.
- **End to end.** A batch of two requests of different lengths proves and verifies through the CLI, NI and Live, and an exact mismatch in the second request is refused.

## Open Questions

- **OQ-0071-1. Slot-aware batching.** Should RFC-0016's packing count the padded token axis, `R·S`, rather than the requests' tokens, and group requests by `⌈log2 n_r⌉`? Packing by decreasing length keeps a batch's lengths close, but a long request with short ones pads them all to its length.

## References

- [#357](https://github.com/AbdelStark/tessera/issues/357); RFC-0006, RFC-0016, RFC-0030, RFC-0034, RFC-0035, RFC-0041, RFC-0044, RFC-0047, RFC-0053, RFC-0054, RFC-0056, RFC-0057.
