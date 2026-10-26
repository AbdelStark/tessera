# RFC-0071: A Batch of Requests in One Proof Unit: Request Slots on One Token Axis

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amended by: [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('The slots', in `tessera-protocol/2`: a slot holds four rows at least, #488), [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('Soundness and zero knowledge', in `tessera-protocol/2`: one mask per tensor kind, at its data claim's column sub-point, [#554](https://github.com/AbdelStark/tessera/issues/554))
- Amends, in both protocols where they apply:
  - [RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Operation set' (`MatMulAct`, `SoftmaxRow`), 'Range analysis' and 'Llama lowering': one graph covers one batch, which carries its slot length and its requests' lengths, and attention reads within a slot;
  - [RFC-0044](RFC-0044-lowering-nodes.md#graph-inputs) 'Graph inputs' and 'A layer': a batch's token axis, positions and scores;
  - [RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions) 'Relations', 'Multilinear extensions' and 'Tree levels', [RFC-0030](RFC-0030-masked-exponentials.md#the-causal-selector) 'The causal selector' and its root claim, [RFC-0035](RFC-0035-padded-rows.md#the-row-selector) 'The row selector': sums over the requests;
  - [RFC-0034](RFC-0034-causal-blocks.md#zero-rows) 'Causal layout' and 'Zero rows', [RFC-0046](RFC-0046-advice-segments.md#short-query-blocks) 'Short query blocks', [RFC-0041](RFC-0041-execution-plan.md) 'Non-Goals', 'Tiles', 'The plan' and 'Drawbacks': a slot's keys and query blocks;
  - [RFC-0037](RFC-0037-stage-schedule.md#rotate) '`Rotate`': a row's position is its place in its request;
  - [RFC-0047](RFC-0047-advice-order.md#the-maximum-layout) 'The maximum layout' and 'A request's sequence length', and [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) rule 4: the padded rows between slots;
  - [RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-request-and-sources) 'Public request and sources', the native relations and 'Schedule, transcript and budget': one maximum for the batch, a key column and key relation per request with outputs, one observation when none has, and their registration order;
  - [RFC-0017](RFC-0017-scoring.md#in-circuit-scoring-mixture) 'In-circuit scoring': each request's window;
  - [RFC-0016](RFC-0016-audit-sampling.md#batching) 'Motivation' and 'Batching': packing by slot class;
  - [RFC-0057](RFC-0057-one-proof-unit.md) 'Non-Goals': several requests in one unit are RFC-0071's;
  - [RFC-0015](RFC-0015-transcript-anchor.md#circuits) 'Circuits' and 'Prover pre-check': a request's frames bind its own rows, and a request that fails the pre-check is left out of its batch;
  - [RFC-0009](RFC-0009-gadgets.md#softmaxrow) '`SoftmaxRow`' and [RFC-0061](RFC-0061-sampling-randomness.md) 'Per batch': a batch's causal mask, and the packing by pin it awaits;
  - [RFC-0054](RFC-0054-token-groups.md#groups) 'Groups', and [RFC-0056](RFC-0056-native-working-sets.md#open-questions) 'Working sets with the native relations', 'Score instances without the score tensor' and OQ-0056-2: the batch's token index and form;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 4, [03 §4](../spec/03-data-model.md#4-statement-and-protocol-types) and [02 §2.2](../spec/02-public-api.md): the batch's rules, its tokens, and `prove --sample`'s pre-check.
- Origin: [#357](https://github.com/AbdelStark/tessera/issues/357)

## Summary

A statement covers a batch of requests (`PublicStatement.batch`, 03 §4), and the audit packs many into one batch (RFC-0016 'Batching'). No document defines how one proof unit proves several: RFC-0006 lowers "one sequence", RFC-0035's row selector has one length `n`, RFC-0041 orders one sequence's tiles. So a batch of one request is the only one a prover can prove.

This RFC puts each request of a batch of `R ≥ 2` in a **slot** of one token axis. With `S = 2^{⌈log2 max_r n_r⌉}`, request `r` holds rows `r·S` to `r·S + n_r − 1`, its positions `0` to `n_r − 1`. The rows past a request's end in its slot are padded rows, which RFC-0035 already defines.
- **Per-token relations are unchanged.** Every relation that works row by row sees a longer token axis, whose high bits name the request.
- **Position-aware ones read a row's place in its slot.** Attention reads keys and values within the query's slot. The causal selector, the row selector, `SoftmaxRow`'s root claim and RoPE's positions name each request's length. Each public extension becomes a sum over the requests.
- **The graph's nodes, the claim list and the plan are the same for every batch.** A batch of one request is today's proof, bit for bit.

Two parts are per request: the greedy certificate's key relations and the scoring windows. The audit packs requests by slot class, so that a batch's slots are its requests' own powers of two.

At 405B, a 16K-token batch of 16 requests of 1K takes about 2.07 GB of `tessera-protocol/1` proof, computed from the compiler's layout. Proved one at a time, the same requests take 25.4 GB, because each proof pays its combined rows and weight openings again.

## Motivation

- **Audits batch requests.** RFC-0016 packed a sample's requests by first-fit decreasing up to `max_batch_tokens`, a memory bound, so that at most one batch fell below `min_batch_tokens` (16,384 by default). Each batch is one statement. Requests are typically far shorter than a batch.
- **What a proof costs regardless of its size.** The advice opening's combined rows are at most `17 × 2^{22}` `Fp2` values, 1.14 GB, in every `tessera-protocol/1` proof, and 21 rows in `tessera-protocol/2`. The weights' openings add about 0.35 GB (08 §5). A proof per request pays both each time.
- **The cost model already assumes batches.** RFC-0056 prices `T` tokens at `S` positions of context as `T/S` whole sequences of `S`. RFC-0054 and RFC-0055 index one padded token axis of the batch.
- **What follows one sequence.** The lowering, the selectors, the layout and both workspaces' code. The verifier's driver refuses a batch of two (`let [request] = …`), and so does the CLI's `prove --sample` (#357).

## Goals

- One proof unit for a batch of whole requests, in statement order.
- No new kind of relation, challenge, message or opening group. The graph's nodes, the claim list and the plan do not depend on the batch.
  - **What grows with it.** The token axis's variables, and the per-request parts: each request's key column and key relation, its `Û_r`, and the selectors' terms.
- The verifier derives every slot from public data: the batch descriptor's lengths.
- A batch of one request proves and verifies exactly as today, so no artifact, vector or proof of a proof unit changes. The audit's packing does change ('The statement and the batching').

## Non-Goals

- **A request longer than one proof holds**: chunks and their linking (#470). A batch holds whole requests from their first token (RFC-0057).
- **Batches of other predictors.** A batch of several requests uses exact greedy (`Exact { Greedy }`), the predictor RFC-0053's certificate covers. The sampler PRF's counter and stream class are per request (RFC-0017, RFC-0061), and the mixture's scoring is not yet implemented. Every other request is a batch of its own until the sampling work (#144) states them per slot. `PublicStatement` keeps one `PredictorSpec` per batch (OQ-0036-1).
- **Several requests in one slot.** A slot holds one request, and a batch's slots one length ('Alternatives Considered' 4).

## Proposed Design

### The slots

A batch is `statement.batch.requests`, `R ≥ 1` requests in that order, with `n_r = prompt_len_r + output_len_r` positions each (RFC-0047).
- **The slot length** is `S = 2^{b}`, with `b = ⌈log2 max_r n_r⌉`, and `b ≥ 2` in `tessera-protocol/2`, whose masked tensors need two variables (RFC-0013 'Masked extensions'). Write `n_max = max_r n_r`, and `n_r = 0` for the empty slots `R ≤ r < 2^{⌈log2 R⌉}`.
- **One request.** With `R = 1` the token axis is today's, of `n` rows, and `S = 2^{⌈log2 n⌉}` is its padded length, at least 4 in `tessera-protocol/2`: every rule below reduces to today's, since `⌊τ/S⌋ = 0`.
- **The token axis.** Request `r`'s position `p < n_r` is row `τ = r·S + p`. For `R ≥ 2` the graph's token axis has `N = R·S` rows, every slot whole, so that a slot's reads stay inside the axis. It is padded, as every axis is, to `2^{b + ⌈log2 R⌉}` rows. With `R = 1`, `N = n`.
  - A point of it is `(r_p, r_β)`: `r_p` the `b` variables of the position within a slot, lowest, and `r_β` the `⌈log2 R⌉` variables of the request.
- **Real and padded rows.** A row is **real** when its position is below its request's length: `τ mod S < n_{⌊τ/S⌋}`, with `⌊τ/S⌋ < R`. Every other row of the padded axis is a **padded row**: the rows between `r·S + n_r` and `(r + 1)·S`, and the axis's own padding.
- **A padded row's values** are RFC-0035's at every node, whatever the node computes on a real row:
  - the token 0, and `OneHot`'s output the zero vector, since `Σ_v h_v = ψ = 0` there;
  - every activation 0, and a lookup output its table's value at key 0;
  - in `SoftmaxRow`, every key masked, so that `e` is 0 on the row.

  No real output reads a padded row. Inside the graph these rows lie between two requests, where with one request they lay past `n`, outside it.

The notation is this RFC's own: `S` here is the slot length, not RFC-0034's sequence length or RFC-0037's sine, and `N` is unpadded, where RFC-0041's is padded.

### The lowering

One graph covers one batch (RFC-0006 'Llama lowering'). `lower(model, lengths)` takes the batch's lengths in order; `lower(model, n)` is its case `R = 1`. The graph's nodes and ids are RFC-0044's for every batch. Only shapes and two operations' reads depend on it.
- **Inputs.** The tokens `[N]`, committed advice, and the positions `[N]`, public: position `τ mod S` at a real row, 0 at a padded one. Their values lie in `[0, n_max − 1]`, which fixes the input's type.
- **Per-token nodes.** These are the activations `[N, channels]` and every node RFC-0054 'Groups' lists as keeping the token axis: matmuls by weights, norms, requantizations, SiLU and the residual. Each applies to each real row as before, and holds a padded row's values at a padded row.
- **Attention reads within a slot.** RFC-0044's head layouts are unchanged: `Transpose`, `Expand`, `Reshape`, whose token axis stays whole. Q, K and V reach the scores as `[n_heads, N, d_head]`. Two operations then read rows of one slot, by the graph's slot length `S`. A batch's graph carries it, a public parameter the batch descriptor fixes, and a graph of one request carries none (RFC-0006). With `R = 1`, `⌊τ/S⌋ = 0`, and both are RFC-0006's operations unchanged:
  - **The scores.** `MatMulAct(q, k)` gives `[n_heads, N, n_max]`, whose entry `(h, τ, j)` is `Σ_d q[h, τ, d]·k[h, ⌊τ/S⌋·S + j, d]`: query row `τ` against key `j` of `τ`'s own request.
  - **The weighted values.** `MatMulAct(e, v)` contracts the key axis the same way: `Σ_j e[h, τ, j]·v[h, ⌊τ/S⌋·S + j, d]`.
  - **On the padded cube.** Each is a matmul batched over the head and the request variables `r_β`, which the query and the key share:

    ```text
    Z̃(r_h, r_p, r_β, r_j) = Σ_{h, β, d} eq((r_β, r_h), (β, h)) · Q̃(h, (r_p, β), d) · K̃(h, (r_j, β), d)
    ```

    `r_j` is the key's `b` variables. This is RFC-0010's stacked `MatMulAct`, `Σ_{s, j} eq(ρ, s)·Ã(s, r_a, j)·B̃(s, j, r_c)`, its stacking variables `s = (β, h)`, the request's lowest, as they lie in the operands' token and head axes. Its rounds bind the contracted index first, then `β`, then `h`: `(d, β, h)` for the scores and `(j, β, h)` for the weighted values. The weighted values are the same with `e` and `v`.
  - **`SoftmaxRow`** reduces each row `(h, τ)` over its `n_max` keys under the causal selector below. A row's keys past its request's length, and every key of a padded row, are masked.
- **RoPE.** It rotates each real row by its position `τ mod S`. Its constants are zero on every padded row, as they are today on the rows past `n` ('The public selectors').
- **Logits** are `[N, vocab]`. The next token at a real row `τ` is row `τ + 1`'s, in the same request, since a window ends at `n_r − 2` ('The greedy certificate and scoring').

### The public selectors

Each public predicate that names a length becomes a sum over the requests. Each sum has `R` terms. With `R = 1` each is today's formula.
- **The row selector** (RFC-0035):

  ```text
  ψ̃ = LT(r_ℓ, n_layers) · LT(r_h, n_heads) · Σ_{r < R} eq(r_β, r) · LT(r_p, n_r)
  ```

  For the output gadgets and the scoring bound, a request's rows are its positions `prompt_len_r − 1` to `n_r − 2`, those whose next token the statement records.
- **The causal selector** (RFC-0030, RFC-0010 'Multilinear extensions'). It is `C(τ, j) = [j ≤ τ mod S < n_{⌊τ/S⌋}]`, keys counted within the query's request. Its extension over a score instance's query variables `(r_p, r_β)` and key variables `r_j` is `Σ_{r<R} eq(r_β, r)·C_r(r_p, r_j)`, with `C_r` RFC-0030's selector for one sequence of `n_r`.
- **`SoftmaxRow`'s root claim** (RFC-0030, RFC-0010 'Tree levels'). It is `1 − LT(r_ℓ, n_layers)·LT(r_h, n_heads)·Σ_{r<R} eq(r_β, r)·LT(r_p, n_r)`. A real row's tree has root 0, and every other tree root 1.
- **RoPE's kernel** (RFC-0037 '`Rotate`'). It reads each real row's position and is zero on padded rows. The verifier's inner products run over the real rows, `Σ_r eq(r_β, r)·eq(r'_β, r)·Σ_{p<n_r}(…)`: `O(Σ_r n_r·d_head/2)` operations, as one sequence's `O(n·d_head/2)`.

The verifier evaluates `ψ̃`, the causal selector and the root claim in `O(R·log S)` each.

### The greedy certificate and scoring

**Greedy** (RFC-0053), one certificate for the batch:
- **One maximum.** `M` is one tensor on the padded token axis, real on every request's window, positions `prompt_len_r − 1 ≤ p < n_r − 1` of slot `r`, and zero elsewhere. It is one source, in one `GreedyMaximum` tile, with RFC-0053's certificate id of ordinal 0, the graph's node count. When no request of the batch has an output, there is no `M`: no tile, no key column, no `Selected`, only the observation below.
- **A key column and a key relation per request with outputs,** in batch order: request `r`'s logical key column, named by its ordinal `r`, follows the model's lookup columns and the earlier requests' key columns. A request without outputs has neither, as RFC-0053 omits them. The key relation reads `M`, `L` and `Hplus` with the time coefficient `Σ_{u < output_len_r} eq(P_row, u)·eq(r_time, (prompt_len_r − 1 + u, r))`, the time point `(r_p, r_β)` of row `r·S + p`.
- **The registration order** (RFC-0053 'Schedule, transcript and budget'): token binding, then the key relations in batch order, then `Selected`; or, when no request has an output, token binding then the logit observation.
- **Selected,** once over the whole token axis, with `M`. Its `active(τ)` is 1 exactly on the windows.
- **Token binding,** once over every row of the token axis, real or padded, as RFC-0053 binds every row. With `OneHot`'s `Σ_v h_v = ψ`, which puts a padded row's one-hot at zero (RFC-0035), it keeps a padded row's token at 0.
- **The increment.** `Hplus` and its carries run over the whole token axis's `b + ⌈log2 R⌉` time variables. It is read only at active rows, where `τ + 1` is the same request's next position, since a window ends at position `n_r − 2`.
- **The observation instance** of a zero-output request is added once, when no request of the batch has an output. Otherwise a key relation already reads `L`.
- **Scoring's sums** (below) are registered after the greedy relations, one per request, in batch order.

One `M`, which every key relation reads within its own window and `Selected` pins on every window, leaves no entry of the certificate outside a relation.

**Scoring** (RFC-0017), for a predictor that scores, once its batches are admitted (#144). Request `r`'s `Û_r` sums its window's rows, positions `prompt_len_r − 1` to `n_r − 2` of its slot. Its sum runs over the whole token axis with the public factor `eq(x_β, r)·[prompt_len_r − 1 ≤ x_p ≤ n_r − 2]`, so its claim on the per-token bounds is at a random point, never at Boolean request coordinates (R8). The batch's `Û` is the sum of the `Û_r`.

### The plan and the layout

- **The plan's tiles** are the same for every batch (RFC-0041 'The plan').
  - **Query blocks.** An `Attention` tile stands for its query blocks of `Q' = min(Q, S)` rows. The batch's padded token axis expands it into `2^{b + ⌈log2 R⌉}/Q'` tiles in row order: slot by slot, and within a slot by position. `Q'` divides `S`, so every query block lies within one request.
  - **Keys.** A tile's keys are its slot's `S` positions: RFC-0034's causal layout takes the slot's `b` variables for the key index, `w = min(b, log2 k − log2 Q' − log2 g)` (RFC-0046 'Short query blocks').
  - **Zero rows.** RFC-0034's rule applies to the query block's request, with its length `n_r`.
- **The advice layout** is a function of the claim list and the batch descriptor (INV-PCS-1), in the plan's production order, so a batch's rows follow the token axis. RFC-0011 rule 4 applies to the padded rows between slots as to the axis's end: a row is skipped when every position it holds is a padded row's public zero. A row of a lookup output whose padded value is not 0 (RFC-0035) is committed. RFC-0047's order puts a padded axis outermost, and with `R` a power of two the token axis is not one: another padded axis, such as `d_ff`, then goes outermost, and a slot's padding lies inside its rows and is committed.
- **The maximum layout.** A batch's layout must fit in at most `2^{32}` rows (RFC-0047). The claim list's check at `n = max_context` (RFC-0007 step 6) does not cover a batch, whose token axis can be longer than `max_context`. So the verifier computes the row count of the batch's complete runtime layout, RFC-0053's maximum and key columns included, from the claim list and the batch descriptor, before it allocates anything, and refuses one past the bound at statement binding. The packer bounds counted tokens, not rows, so a prover whose packed batch would exceed the maximum layout splits it in two by the same packing rule, a case reached only with a `max_batch_tokens` far beyond any memory bound.
- **Token groups** (RFC-0054 'Groups'). The batch's token index is the padded token axis's row. A prover's planner groups a batch with `g_T ≤ b` token bits, so that each group fixes low position bits only and holds every slot's positions alike, as RFC-0054's equal load needs. With `g_T > b` a group would hold one position of some slots, and the positions' keys differ in number. The rule is the planner's, and changes no verdict: the verifier checks only RFC-0054's bounds. A planner that cannot meet its memory budget within it fails with `ERR-RES-MEMORY`.

### The statement and the batching

**The statement.** RFC-0019 step 4 checks the batch, each check a `Reject(StatementMismatch { field: Batch })`:
- at least one request, and no request named twice: no two with an equal `RequestKey`, `(session_id, request_id)` (03 §4, RFC-0063);
- for `R ≥ 2`, the predictor `Exact { Greedy }`: a batch of any other predictor holds one request, until the sampling work (#144) states the sampler and the scoring per slot;
- each request's `prompt_len_r ≥ 1` and `n_r ≤ max_context`. A request without a prompt would start its window at position −1, in the slot before it;
- the batch's advice layout, as computed above, within the maximum layout.

The requests' order in `statement.batch` is the slots' order, `RequestKey` order for an audit's batch (RFC-0016). The statement digest and the advice commitment's `layout_digest` bind it (RFC-0031).

**The tokens.** A prover proves its requests' tokens concatenated in batch order (02 §2.2 `ProofBatch`), each request's prompt then outputs. They go to its slot's real rows. The anchor binds request `r`'s frames to its rows `r·S + p` (RFC-0015 'Circuits', INV-ANC-1). A per-request read, the anchor's as `Û_r`'s, is a sum over the whole token axis with public coefficients that select its slot, never a slice at a Boolean request coordinate, where `tessera-protocol/2`'s `Z` vanishes (R8).

**The batching** (RFC-0016). First-fit decreasing on total tokens puts a later, shorter request into the first batch it fits, so a batch can mix a request of 16,400 tokens with 16 of 1,000. Its 17 slots, padded to 32, would hold 32,768 rows each, 32 times its tokens. RFC-0016's packing therefore runs per **slot class**:
- **A class.** Request `r`'s class is `c_r = ⌈log2 max(L_r, 1)⌉`, with `L_r` its tokens. Within each of RFC-0016's groups, the requests of one class are packed together, the classes in decreasing order.
- **Counted tokens.** A batch of `R` requests of class `c` counts `2^{c + ⌈log2 R⌉}` tokens, the rows its token axis spans, against `max_batch_tokens`, which then bounds its memory.
- **The packing** is first-fit decreasing within a class, the requests in RFC-0016's order: each goes into the first batch whose counted tokens stay at most `max_batch_tokens` with it, else into a new batch. A request of class `c > ⌊log2 max_batch_tokens⌋` is a batch on its own, a test that cannot overflow.
- **The minimum.** In counted tokens, at most one batch per class has fewer than `min_batch_tokens`. Adding one request to a batch at most doubles its counted tokens, so a later batch under the minimum would have had its first request fit in an earlier one, under `2·min_batch_tokens ≤ max_batch_tokens`. In real tokens, a class's batches can fall below it.
- **The order.** Batches are numbered class by class in decreasing order, each class's in the order they were opened. This is the order `prove --sample --batch K` numbers (02 §2.2).
- **One group.** The packer has no per-request predictor, so until #144 every request is packed in one group as exact greedy, as 02 §2.2 states; a request the prover would prove under another predictor is left out of its batch, as an unprovable request is below, and stays unproven until #144: the statement admits no other predictor for two or more, and `prove --sample` proves only packed batches.
- **The cost.** A batch's slots are then its requests' own power of two, under twice each request's length, and its slots' count under twice its requests'. So its real tokens are more than a quarter of its counted tokens: 9 requests of 513 count 16,384 and hold 4,617.
- **An unprovable request.** A request the prover cannot prove, one whose pre-check fails or whose recorded outputs are not exact greedy, is left out of its batch, which is proven without it. RFC-0016's report names it as unproven. The verifier's accounting checks only that each proven request was sampled and not proven before, so the rest of the batch is accepted.
- **What changes with it.** The audit's batches, so `spec/vectors/audit/`'s packing cases, `tessera.audit-batches/1` reports and the batch indices of `prove --sample --batch K`. Both workspaces' packers change with #551. Until then a proof unit holds one request, so every provable batch is a single request either way.

### Soundness and zero knowledge

- **Soundness.**
  - **Nothing new to prove.** No challenge, message or kind of relation is added. The selectors are public and the verifier computes them.
  - **Padded rows.** A prover that puts a real value in a padded row is caught as RFC-0035 catches it, whatever the row's place.
  - **No reads across a slot.** Attention reads within a slot. `Hplus` is read only inside a window, which `prompt_len_r ≥ 1` keeps in its slot. Token binding pins every row, and the anchor ties each request's frames to its own rows.
  - **The rounds.** The request variables add `⌈log2 R⌉` rounds to each relation over the token axis. RFC-0014's calculator counts them through `rounds`, from the statement's sizes.
- **Zero knowledge.** In `tessera-protocol/2` each committed tensor kind's mask row is shared by every layer and attention tile of the kind, at its data claim's column sub-point (RFC-0013 'Committed tensors', [#554](https://github.com/AbdelStark/tessera/issues/554)); a claim at one layer takes its layer tensor's own row (RFC-0069), and a virtual tensor's mask takes its low variables, layer variables among them when they fit. The slots are public. A request variable is a token variable: a claim's point is random in it, so no mask's `Z` vanishes more often (`docs/zk/simulator.md` R8). In an attention-tiled tensor the request bits are part of its query block's, so they are row bits, not mask variables; in another tensor they are its own variables, mask variables where its column sub-point holds them. R10's check does not cover batches yet (RFC-0013 'Committed tensors').

### Cost at 405B

**The method.** From the compiler's advice layout of one sequence (`ModelAdviceLayout`, `servable` with two classes, `B_max = 32`), with the attention tiles' rows counted apart. A batch of `R` requests of `S` tokens has:
- one 16K sequence's rows outside the attention tiles: the per-token rows and the tables, 227,820;
- plus `R` times one sequence of `S`'s attention-tile rows, at `Q' = min(512, S)`.

| `n` | `Q'` | Rows | Attention-tile rows |
|---|---|---|---|
| 256 | 256 | 60,692 | 7,056 |
| 1,024 | 512 | 76,727 | 15,120 |
| 2,048 | 512 | 106,884 | 34,272 |
| 4,096 | 512 | 179,295 | 84,672 |
| 16,384 | 512 | 953,580 | 725,760 |
| 16,384 | 256 | 1,163,244 | 935,424 |

**The proof.** It is `tessera-protocol/1`'s combined rows, at most `17 × 2^{22} × 16` bytes, plus the advice columns at `live-64`'s `τ_A = 155` and 8 bytes a row, plus about 0.35 GB of weight openings. Sumcheck messages (under 10 MB) and the token groups' level messages (about 120 MB at 16K, 08 §5) are left out of both columns.

| Batch of 16,384 tokens | One proof (rows, bytes) | One proof per request |
|---|---|---|
| 1 × 16K | 953,580, 2.67 GB | the same |
| 4 × 4K | 566,508, 2.19 GB | 6.85 GB |
| 8 × 2K | 501,996, 2.11 GB | 13.0 GB |
| 16 × 1K | 469,740, 2.07 GB | 25.4 GB |
| 64 × 256 | 679,404, 2.33 GB | 100 GB |

At 64 × 256, `Q' = 256` doubles the attention tiles, whose rows are mostly per tile. In `tessera-protocol/2` every proof carries 21 combined rows (RFC-0068, RFC-0013's folds, RFC-0070), four more than here, so each proof per request costs 0.27 GB more, and the batch once.

**The prover.** RFC-0056 prices the same batches. A 16K batch at 1K of context costs `μ ≈ 13.8` (`minimal`) and 15.7 (`servable`) per token (RFC-0056 'At 405B'). A 1K request alone pays its proof's fixed costs by itself.

**The verifier.** Every selector costs `O(R·log S)` field operations, and RoPE's kernel `O(Σ n_r·d_head/2)`, as one sequence's. Its other work grows with the proof's rows and rounds.

## Alternatives Considered

1. **One proof per request.**
   - **The cost.** Each proof pays its combined rows (up to 1.14 GB) and weight openings (0.35 GB) again: 25.4 GB for 16 requests of 1K.
   - **Why rejected.** Batching is the audit's premise (RFC-0016).
2. **Concatenate the requests' tokens with no slots.**
   - **The idea.** Request `r` takes rows `Σ_{s<r} n_s` onward, which pads the whole axis once.
   - **Why rejected.** A query's keys then start at an offset that is not a power of two. The scores read `K` through an addition of offsets, the causal selector becomes a union of intervals at arbitrary starts, and a query block can straddle two requests. With slot classes, slots cost at most the rounding of each request to its power of two.
3. **Each request its own instance of every relation, padded to its own power of two.**
   - **The idea.** This is #357's first option.
   - **Why rejected.** Each relation's claims multiply by `R`, at points of their own, so the claim pool and the opening groups grow with the batch. Slots keep one instance per relation and add only variables.
4. **Slots of several sizes in one batch.**
   - **The idea.** Pack each request in an aligned block of its own power of two, the largest first, so that requests of different classes share a batch.
   - **Why rejected.** Every selector would then name a block of its own size. The attention relation becomes a sum of instances of different key widths. Packing by class gives the same slot sizes.
5. **A maximum per request** (RFC-0053's per-request sources).
   - **Why rejected.** `Selected` would then pin only the maxima's sum at each row. A prover could raise one request's maximum at a row of its window and hide the difference in another request's maximum, outside that request's window, where nothing reads it. A token that is not the argmax would then pass. One `M`, pinned on every window, leaves no such entry.

## Drawbacks

- **Batches per class.** Packing by class makes more batches than first-fit decreasing across classes: at most one undersized batch per class rather than per group.
- **The overhead calculator.** RFC-0016's calculator assumes batches of `min_batch_tokens` real tokens. Packed by class, a batch holds more than a quarter of its counted tokens, and a request proven under another predictor than exact greedy is a proof of its own, so it understates both.
- **Empty slots.** A batch of `R` requests pads its slots to `2^{⌈log2 R⌉}`. Their rows are padded rows, skipped when public zeros, but the relations' rounds count them.
- **More to keep in step.** The lowering, the selectors, the greedy certificate, scoring and the packing each read the slots in both workspaces. A differential test must keep them equal ('Testing Strategy').

## Migration / Rollout

1. **This RFC** amends the documents in its header. A batch of one request is unchanged, so no artifact, vector or proof of a proof unit changes. The audit's packing changes ('The statement and the batching').
2. **Implementation** ([#551](https://github.com/AbdelStark/tessera/issues/551)). Both workspaces lower, select, certify, score and pack per slot. Then the verifier's driver and the CLI's `prove --sample` lift their one-request refusal. Until then a proof unit covers one request, and both refuse a batch of two with `ERR-USAGE`, as today. An empty batch's verdict moves from `Inconclusive(ERR-USAGE)` to `Reject(StatementMismatch { field: Batch })`. The packing's vectors (`spec/vectors/audit/`), the `tessera.audit-batches/1` schema's descriptions and its example report change with it.

## Testing Strategy

- **Vectors** from an independent oracle, for batches of 2 and 3 requests of different lengths:
  - the lowering's shapes and positions;
  - `ψ̃`, the causal selector, the root claim and RoPE's kernel at random points;
  - the greedy windows, `Û_r`, and the packing by class.
- **Equivalence.** A batch of one request gives today's claim list, plan, layout and proof bytes, and every vector of a proof unit passes unchanged. The audit's packing vectors are regenerated by class.
- **Adversarial**, each asserting its rejection:
  - a real value in a padded row between two slots;
  - a key read across a slot;
  - a split maximum, raised in one window and compensated outside another;
  - request 1's token bound to request 0's frames;
  - an empty batch, a request with `prompt_len = 0`, a request named twice, a batch of two requests under another predictor than exact greedy, and a batch whose layout exceeds the maximum;
  - in `tessera-protocol/2`, a per-request read at a Boolean request coordinate, told apart by #162's distinguisher (R8).
- **End to end.** A batch of two requests of different lengths proves and verifies through the CLI, NI and Live. An exact mismatch in the second request is refused.

## Open Questions

- **OQ-0071-1. Classes across batches.** Should a class too small for `min_batch_tokens` join the next class up, padding its requests to that class's slots? It would trade padding for fewer undersized batches.

## References

- [#357](https://github.com/AbdelStark/tessera/issues/357); RFC-0006, RFC-0009, RFC-0010, RFC-0011, RFC-0015, RFC-0016, RFC-0017, RFC-0019, RFC-0030, RFC-0034, RFC-0035, RFC-0037, RFC-0041, RFC-0044, RFC-0046, RFC-0047, RFC-0053, RFC-0054, RFC-0056, RFC-0057, RFC-0061.
