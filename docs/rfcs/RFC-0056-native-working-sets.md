# RFC-0056: Native Relations in the Working Sets: Score Instances Without the Score Tensor

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-23
- Target milestone: v0.3 (working sets, plan and cost model), v0.5 (GPU backend)
- Amended by: [RFC-0057](RFC-0057-one-proof-unit.md) ('The batch's form': a chunk needs [#470](https://github.com/AbdelStark/tessera/issues/470)), [RFC-0071](RFC-0071-request-slots.md) (OQ-0056-2: the batch, answered)
- Supersedes:
  - [RFC-0055](RFC-0055-windowed-stages.md) 'Native relations', which left them to [#458](https://github.com/AbdelStark/tessera/issues/458); its 405B windows and its `c_w` bands, which leave the native relations out; its one window per stage, for stages whose rounds bind more than eight own variables of an instance (Alternative 7);
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model) 'Executable cost model': the stages' tables by growth, holding and width, the plan at a context, and the windows;
  - [RFC-0006](RFC-0006-compiler-ir.md#outputs) 'Outputs': the report counts the native relations and their residents;
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) 'Pass structure': what stage 1 holds for `SoftmaxRow`.
- Scope: the prover and the cost model. Messages, labels, the verifier and the soundness bound are RFC-0010's, in `tessera-protocol/1` and `/2`.
- Origin: [#458](https://github.com/AbdelStark/tessera/issues/458), from RFC-0055's review

## Summary

The CPU prover composes native relations into stage 1 ([RFC-0050](RFC-0050-gadget-advice-columns.md), [RFC-0009](RFC-0009-gadgets.md#softmaxrow), [RFC-0053](RFC-0053-greedy-runtime-boundary.md)):
- each root's coefficient products and padding checks;
- each `SoftmaxRow`'s key, forest and reciprocal relations;
- a request's token relation and its greedy certificate, or its observed logits.

The compiler's schedule did not hold them, so RFC-0055 sized the reduction stages without them. Among them, `SoftmaxRow`'s key and forest relations are sums over the score tensor. A 405B batch of 16K tokens would hold 8.8 TB of their tables at 1K context and 141 TB at 16K.

This RFC makes four changes, none of which touches the messages:
- **The score instances.** They are proven without the score tensor. The rounds over the key index read the INT8 queries, the keys folded over their bound bits, and the causal mask's closed form, with prefix sums. The remaining rounds run on row tables. The scores are `Q·Kᵀ`, bilinear, so the round polynomials are the same: an exact check against a literal prover passed on 336 runs ([EXP-0026](../../research/experiments/EXP-0026-causal-score-rounds/README.md)). Their working set becomes about 45 GB for 16K tokens.
- **Shared tables.** Instances of one relation of one tensor at one point hold one set of tables, and so do a root's coefficient products. Each root's thirty products leave thirty identical claims on the norm's sum of squares, which stage 2 proves as thirty identical instances over the residual.
- **The windows.** A stage's rounds take one window per eight own variables they bind of an instance, each a regeneration: eight is where a second window starts to pay at 405B.
- **The report and the model.** The commitment report counts the native relations, from a compiler-side derivation that a prover test holds equal to the prover's. The cost model sizes each stage at its batch and context.

**At 405B and 16K tokens**, with the native relations counted:
- **`minimal`.** Stages 1 and 2 hold 6.4 and 6.6 TB, and each takes one window, of up to eight own variables. Its regenerations stay at eight.
- **`servable`.** Stage 1 holds 11.9 TB, and its rounds bind ten own variables, so it takes two windows. Stage 2, at 1.1 TB, now exceeds memory too. Nine regenerations in all.
- **The projections.** The corrected projections are about 13.8 and 29.1 for `minimal`, unchanged, and 15.7 and 31.0 for `servable`.

## Motivation

[#458](https://github.com/AbdelStark/tessera/issues/458), found by RFC-0055's independent review:
- **Two schedules.** `tessera-prover`'s `ModelNativePlan` composes the native relations into stage 1 with `derive_composed_schedule`, while the commitment report read `derive_schedule`, which holds none.
- **The score instances.** `SoftmaxRow`'s key and forest relations prove `Σ_x eq(z, x)·M(x)·(c(row(x)) − s(x))` over `x = (ℓ, h, q, i)`, the score tensor's `h·S²` entries per sequence and layer. The prover builds two tables of that size: the scores, and the row maxima broadcast.
- **The window cost.** A window of RFC-0055 over them would bind 9 to 13 own variables, at 115 to 584 `Fp` multiplications per index against about 40 for the rounds it answers.

Composing the native relations also showed duplicated work. Each root's `2·⌈(2w + 55)/8⌉` coefficient products, thirty at 405B, end at one point. Each leaves a claim on the norm's sum of squares there. Stage 2 proves each claim as its own `SumSquares` instance over the residual: thirty identical instances per norm. At 405B they hold 33.1 TB of `minimal`'s stage 2, which holds 38.6 TB unshared.

## Goals

- Every relation the prover proves is counted in the report and the model.
- No stage holds a table that grows with the product of the batch and the context.
- No change to the messages, the transcript, the verifier, soundness or zero knowledge.

## Non-Goals

- Fewer passes for the stages: RFC-0055's windows stand.
- The protocol's schedule. Merging identical claims in RFC-0037 would change the messages for what the prover can do without changing them.

## Proposed Design

### Working sets with the native relations

`tessera_compiler::schedule::model_natives(lowering, request)` derives the native relations from the lowering: the descriptors (tag, node, variables, degree, evaluations, zero-check, outputs), what each proves, and the graph sources they replace.
- **The mirror.** It mirrors `ModelNativePlan`'s derivation. It walks the roots and softmax rows in node order, with RFC-0050's root width from the compiler's own range analysis. Then it appends a request's relations as `derive_runtime` does: the tokens, and the greedy certificate's key and selected relations for a request with outputs, or the observed logits for one without.
- **The check.** `tessera-prover`'s complete-proof tests assert that its descriptors, and the schedule `derive_composed_schedule` builds from them, equal the prover's on every conformance model, for the model alone and with each kind of request.

**The report** (`stage_tables`) reads the composed schedule with a request with outputs, the larger of the two:
- **Tables.** A graph instance holds one table per evaluation it leaves. A native instance holds its evaluations.
- **Classes.** Each class is `count + count_per_bit·b` tables of `2^{log2_entries}` entries of `bytes` bytes, each grown as its class grows, with `b` the bits of a request's positions:
  - **Growth.** With a batch of `T` tokens at context `S`, each padded to a power of two, a table gains no variables (`fixed`), `log2 T` (`tokens`), `log2 max(T, S)` (`keys`, the key positions the batch attends), or `log2 T + log2 S` (`tokens_and_context`, the score tensor). The greedy relations' evaluations grow by two per bit of a request's positions, `log2 min(T, S)`.
  - **The batch's form.** The growths assume that a batch with `T ≥ S` holds `T/S` whole sequences of `S` tokens, and that one with `T < S` is a chunk of one sequence whose earlier keys are cached. Proving such a chunk needs its cached keys and values bound to the tokens before it, across proofs, which no document specifies yet ([RFC-0057](RFC-0057-one-proof-unit.md), [#470](https://github.com/AbdelStark/tessera/issues/470)): the figures at `T < S` are projections for that statement. How a proof unit holds several requests is [#357](https://github.com/AbdelStark/tessera/issues/357)'s.
  - **Classification.** The compiler classifies each instance from the schedules of `m` and `m/2` positions, `m` the largest power of two at or below the sequence. It refuses one that grows by anything other than 0, 1 or 2 variables, or whose table count grows by anything other than 0 or 1.
  - **Holding.** Windowable, an elementwise operand a window may fold (RFC-0055). Resident, held whole: a matrix product's folded operand, or one of the residents below. Computed, never held: the score instances.
  - **Width.** 16 bytes for an `Fp2` table, 1 for an INT8 operand held as it is.
- **Residents.** Each stage with score instances lists what their prover holds, per `SoftmaxRow` site:
  - the INT8 queries, `H·d` per token and layer;
  - the INT8 keys, `H_kv·d` per key position and layer;
  - the score and row-maximum row tables, `H` entries of `Fp2` each per token and layer.

### Score instances without the score tensor

Write the key and forest sums as `Σ_x eq(z, x)·M(x)·(c(ℓ, h, q) − s(x))`, with `s = Q_{ℓ,h,q}·K_{ℓ,g(h),i}` and `M` the causal mask `i ≤ q < n` on real heads and layers. RFC-0010 binds the key index `i` first, its lowest variable.

**Rounds over the key bits**, `j = 1, …, log2 n_pad`.
- **The split.** Split `i` and `q` at bit `j`, the low parts the first `j` bits.
- **The mask.** With the low key bits at `(r_{<j}, X)`, the mask's extension is `1` for `i_hi < q_hi`, `LT((r_{<j}, X), q_lo + 1)` for `i_hi = q_hi`, and `0` for `i_hi > q_hi`, on a real row.
- **The score.** Its extension is `Q_q·K̃((r_{<j}, X), i_hi)`, with `K̃` the keys folded over their bound bits.
- **The round.** So

  ```text
  p_j(X) = eq(z_{≤j}, (r_{<j}, X)) · Σ_{ℓ,h,q real} eq(z_{ℓ,h,q}, (ℓ, h, q))
               · [ c(ℓ,h,q)·(A(q_hi) + e(q_hi)·LT(·, q_lo + 1))
                 − Q_q · (B_X(q_hi) + e(q_hi)·LT(·, q_lo + 1)·K̃((r_{<j}, X), q_hi)) ]
  ```

  with `e(t) = eq(z_{i,>j}, t)`, `A(t) = Σ_{b<t} e(b)` and `B_X(t) = Σ_{b<t} e(b)·K̃((r_{<j}, X), b)`, both prefix sums over the key blocks.
- **The folded keys.** `K̃` is not held. Each round folds the INT8 keys over its bound bits one key block at a time, `O(H_kv·n·d)` per layer, and adds the block to the prefix sums.
- **Its cost.** `O(H_kv·n·d)` for the folds and the prefix sums, `O(H·n·d)` for the dot products, and `O(H·n·j)` for `LT`'s closed form, per layer and per evaluation point.

**After the key bits.** The tables are row tables over `(ℓ, h, q)`:
- `eq(z_i, r_i)·eq(z_{ℓ,h,q}, ·)`;
- the mask's `LT(r_i, q + 1)` on a real row;
- the row maximum `c`;
- `Q_q·K̃(r_i)`.

The remaining rounds run as RFC-0010's prover runs them, and the final evaluations of the scores and the maxima are the row tables' at the final point.

**What it holds.** The residents above: about 2.6 MB per token and 0.13 MB per key position at 405B. That is 45 GB for 16K tokens at up to 16K of context, and 60 GB at 128K. The score tensor is never built.

**Its work.** `O(L·d·(H·T + H_kv·max(T, S))·log S)` multiplications by `Fp2` values, most of them INT8 values times `Fp2` vectors.
- **Against the literal prover.** EXP-0026 counted 0.027 of its multiplications at `d = 4` and `n = 2^{10}`. The ratio scales as `d·log n/n`: at 405B's `d = 128`, about 0.9 at 1K of context and 0.08 at 16K. The structured rounds save memory at every context, and arithmetic only at long ones.
- **In `μ`.** Neither prover's sumcheck arithmetic has a term of its own; `μ`'s glue prices it, as RFC-0020 does every stage's.

**The check** ([EXP-0026](../../research/experiments/EXP-0026-causal-score-rounds/README.md), E0). Against a literal prover over the materialized tensors, with arbitrary values on every padded position:
- **Equality.** Every round polynomial at 0 to 3 and both final evaluations are equal in `Fp2` on 336 registered runs: 1 to 16 positions, grouped key heads, and padded heads and layers.
- **Faults.** Nine planted faults are each detected.

### Shared tables

**Shared instances.** Instances of one relation of one tensor at one point, with one layer weighting, have one summand, one variable count and one claim `S`. The prover holds one set of tables for them and computes one round polynomial `p_j`. With `Γ = Σ γ^i` over their batching coefficients, it adds:
- `Γ·2^{p−j}·S` in each of their padding rounds `j ≤ p`, `p = n_max − n` (RFC-0010's front padding);
- `Γ·p_j` in each of their own rounds.

That is what the instances add one by one, so the messages do not change.
- **The duplicates at 405B.** Each root's thirty coefficient products leave thirty identical claims on the norm's sum of squares. Stage 2 proves them as thirty `SumSquares` instances, which now share one set of tables: 6.6 TB in `minimal`'s stage 2 instead of 38.6 TB.

**A root's products.** A root's `2·⌈(2w + 55)/8⌉` coefficient products are distinct relations over the same tables: its digits, its value and its input at one point. The prover holds those `⌈w/8⌉ + 4` tables once and computes each product's round polynomial from them.

The report counts shared tables once.

### The windows

A window binds the own variables of each instance among its rounds, and its grid costs `3·1.5^v` `Fp` multiplications per pair at `v` of them (RFC-0055 'Cost'). Splitting a wide window trades a regeneration for less grid arithmetic. Whether that pays depends on `v` and on the stage's windowed entries per token against the native time.

**The rule.** A stage's `c` rounds take `k = ⌈v/8⌉` windows, `v` the most own variables they bind of one instance, each binding at most eight:
- **Front padding.** The rounds before an instance's own bind none of its variables.
- **The passes.** Each window adds a fold pass, a regeneration. The second window's grid rides the first's fold pass (RFC-0055 Alternative 7).
- **The folds.** Each fold pass folds the stage's windowed entries again.

**The unit cost per window** (`c_w`, `window_ps`). A window's fold and its grid's excess over the rounds it answers, per windowed entry, from RFC-0055's two sources:
- **The count** (SUR-0007, E0). Two `Fp` multiplications per entry for the fold, plus `max(0, 3·1.5^v − 38)/2` for the grid, at 0.5 to 1 ps each.
- **The measurement** (EXP-0012, E1). The fold at 1 to 2 ps, plus `6·(r(v) − 1)` ps for the grid. `r(v) = (5.3 + 18.45·g^{v−5})/16.95` is a leaf level's time against the in-memory one, extrapolated from five own variables at `g` = 1.5 to 1.64 per variable. At four or fewer, EXP-0012 measured no excess.

| `v` | 4 or fewer | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c_w` by the count, ps | 1 to 2 | 1 to 2 | 1 to 2 | 4.3 to 8.6 | 11 to 21 | 20 to 41 | 35 to 70 |
| `c_w` by the measurement, ps | 1 to 2 | 3.4 to 4.4 | 6.7 to 8.6 | 12 to 15 | 19 to 27 | 30 to 45 | 46 to 75 |

**The cap at 405B.** A window costs about 1 on `μ`. `servable`'s stage 1 binds ten own variables, and `minimal`'s stages eight and seven. Two windows instead of one change `μ` by, from the count's lower bound to the measurement's upper one:
- **`servable`'s stage 1:** −0.09 to −1.3 at 1K, and −0.01 to −1.1 at 16K;
- **`minimal`'s stage 1:** +0.4 to +0.8;
- **`minimal`'s stage 2:** +0.7 to +1.0.

So at 405B a second window pays only beyond eight, and `MAX_WINDOW_VARS = 8` is a calibration for 405B's stages, not a law.
- **Smaller stages.** For a stage as large as `minimal`'s stage 1, nine own variables split into five and four would change `μ` by −0.27 to +0.40 by estimate.
- **A planner.** It could choose each stage's windows by cost once OQ-0055-1 measures the grid (OQ-0056-1).
- **The model's charge.** The model charges every window's grid on every windowed entry. A second window's grid runs on entries the first already folded, so the charge errs high.

### At 405B

Two classes, `B_max = 32`, 16K tokens, 0.2 TB:

| Profile, stage | Held tables at 1K to 128K | Rounds at 1K, 4K, 16K, 128K | Windows | Own variables of the windowed entries at 16K |
|---|---:|---|---|---|
| `minimal`, 1 | 6.14 to 6.43 TB | 11, 13, 15, 18 | 1 | 8 on 78%, 4 or fewer on 22% |
| `minimal`, 2 | 6.60 TB | 7 | 1 | 7 on 67%, 5 on 33% |
| `minimal`, 3 | 8.7 GB | none | | |
| `servable`, 1 | 11.63 to 11.92 TB | 11, 13, 15, 18 | 2 | 10 on 37%, 8 on 51%, 4 or fewer on 12%; at most 5 per window |
| `servable`, 2 | 1.11 TB | 3 | 1 | 3 |

- **What stage 1 holds.** At 16K context, `minimal`'s 6.41 TB are 5.2 TB of elementwise instances, 1.17 TB of the greedy certificate's relations over the logits, and 45 GB of residents.
- **Rounds by context.** The score instances set each stage 1's largest variable count, which grows with the context. So the window counts more rounds at longer contexts, each instance front-padded the more. The keys and the greedy relations grow the held tables a little.
- **Own variables do not grow.** The own variables a window binds stay put: memory sets them, and the score instances hold only their residents.
- **`servable`'s stage 2.** The roots' products now feed it, so it is windowed too: one more regeneration than RFC-0055 counted. Its stage 1's second window is another.
- **The batch.** A batch of 2,048 tokens gives `servable` 14.3 and 29.5: its stage 1 takes one window of seven own variables, and its stage 2 fits. That is two regenerations fewer, which outweigh the weight pass shared by eight times fewer tokens. `minimal` is cheapest at 16K tokens, at 14.2 and 29.6 at 2,048.

**The windows' arithmetic** (the bands above, over each profile's windowed entries):
- **`minimal`.** `c_w` is about 6 to 12 ps by the count and 12 to 17 by the measurement, so 6 to 17 ps. That adds 0.42 to 1.20 to `μ` at 1K and 0.40 to 1.14 at 16K.
- **`servable`.** Its windows bind at most five own variables. `c_w` is 1 to 2 ps by the count and 2.4 to 3.5 by the measurement, so 1 to 3.5 ps per window. That adds 0.13 to 0.47 at 1K and 0.13 to 0.45 at 16K.

### Soundness and zero knowledge

- **Soundness.** The prover sends RFC-0010's round polynomials and evaluations. The structured rounds, the shared tables and the windows only compute them differently. The soundness calculator does not change.
- **Zero knowledge.** RFC-0013's masked rounds add their part to the same polynomials. The structured rounds and the windows compute only the unmasked part, as RFC-0055's window does.
- **Secrets.** The INT8 queries and keys, the folded keys, the prefix sums and the row tables derive from secret tensors: `SecretBuf`, cleared (OBS-6), with branch-free arithmetic on the values (INV-ARCH-2).

### The executable model

`tessera-bench`'s `StageTable` carries the classes: a `Growth` (`Fixed`, `Tokens`, `Keys`, `TokensAndContext`), a `Holding` (`Windowable`, `Resident`, `Computed`), `log2_entries`, `count`, `count_per_bit` and `bytes`.

`stage_plan(counts, memory, context, batch_tokens)` sizes each stage at the batch and the context, with `b` and `s` their bits:
- **Variables.** A table gains `0`, `b`, `max(b, s)` or `b + s` variables, by its growth, and counts `count + count_per_bit·min(b, s)` tables.
- **What it holds.** The windowable and resident tables at their widths, and the residents (`ReductionStage::residents`). A computed instance sets `n_max` and holds nothing; a resident is no instance and sets none of the rounds.
- **The rounds.** RFC-0055's rule over the held tables.
- **The windows.** `⌈v/MAX_WINDOW_VARS⌉`, `MAX_WINDOW_VARS = 8`, with `v` the most own variables the rounds bind (`own_vars`).

`mu` reads the plan at its context. Each window adds a regeneration and `window_ps` per windowed entry, so the streaming term is `window_ps · Σ_w k_w·E_w / T`.

## Alternatives Considered

1. **Prove the score instances inside RFC-0054's token groups.** A group's scores fit with its forests, so the instances would cost no pass. But each group would reduce its key and forest claims to its own point, the merge would merge the scores and maxima instead, and `MatMulAct`'s `QKᵀ` would move a stage earlier: a change to the messages and the schedule, for what the structured prover does without one. Rejected.
2. **RFC-0055's windows over the score instances.** One window binds 9 to 13 own variables. Several windows of at most six cost a regeneration each, three at 16K, and each grid about once more the rounds it answers. Over the score tensor, those rounds alone are about 4 on `μ(16K)`. Rejected.
3. **Commit the keys.** The key and forest claims would become openings, but committing `A·S` more elements per token costs about 6 on `μ(16K)`. Rejected.
4. **Merge identical claims in the schedule.** It would remove the duplicated `SumSquares` instances from the messages. But it changes RFC-0037's derivation in both workspaces, the transcript and the vectors, for what shared tables save without a change. Rejected; a smaller transcript is not worth a protocol change here.
5. **Leave the native relations out of the report.** The model would then size stage 1 below the prover's working set, and miss `servable`'s windowed stage 2. Rejected.
6. **One window per stage, whatever it binds** (RFC-0055 as first written). `servable`'s stage 1 would bind ten own variables, at up to 1.3 more on `μ`. Rejected.
7. **A cap of five or six own variables** (RFC-0055 Alternative 7's 'five or fewer'). `minimal`'s windows of seven and eight would split, at 0.4 to 1.0 more on `μ` each. Rejected.

## Drawbacks

- **Two more regenerations for `servable`.** Its stage 2 exceeds memory once the roots' products feed it, and its stage 1 takes a second window: nine against RFC-0055's seven. `μ` rises by 2.
- **Wider windows.** The residents and the native relations' tables leave less memory for the windows. `minimal`'s bind eight and seven own variables, so the grid's arithmetic adds up to 1.2 to its `μ(1K)` at the upper estimates.
- **A mirror.** The compiler derives the native relations again, beside the prover. The prover's tests hold the two equal, and a future native relation must enter both.
- **Estimates.** The bands and the cap rest on a count and on a CPU measurement of windows of three to five variables, extrapolated (OQ-0056-1).
- **A small overcount.** The roots' and the reciprocal's padding checks count tables the products and the reciprocal relation already hold. They are row-sized, so the overcount is negligible, and it errs high.

## Migration / Rollout

- **v0.3.**
  - **The report and the model** (this RFC's change).
  - **The CPU prover.** It keeps its literal score instances and its unshared tables, which fit at the conformance models' sizes.
  - **The structured score instances and shared tables.** They land in the prover with the literal prover as their equivalence oracle ([#461](https://github.com/AbdelStark/tessera/issues/461)).
- **v0.5.** The GPU backend proves the score instances from its residents, shares tables, and windows the stages by the rule above ([#459](https://github.com/AbdelStark/tessera/issues/459)).

## Testing Strategy

- **The mirror.** The compiler's native descriptors and composed schedule equal `ModelNativePlan`'s, for the model alone and with a request with and without outputs, on every conformance model the complete-proof tests prove (`tessera-prover`, `model_leaf`).
- **Working sets.**
  - **Against the schedule.** The report's classes equal the composed schedule's tables, shared ones once, at 4, 32 and 256 positions, the score instances' growth of two and the greedy relations' per-bit tables included. A report of 12 positions has the classes of 8 (`tessera-compiler`).
  - **At 405B.** The same holds at 16,384 positions, and the model's working sets, rounds and windows equal those computed from the schedule and the residents (`tessera-bench`).
  - **The sharing rule.** These tests apply the rule; they do not prove it. [#461](https://github.com/AbdelStark/tessera/issues/461)'s equivalence tests do: the shared prover's messages equal the unshared one's byte for byte.
- **The classification.** A growth of 0, 1 or 2 variables and a count growth of 0 or 1 are accepted, and any other is refused. Classes part by growth, holding and width.
- **The structured prover.** EXP-0026: exact equality on 336 runs and 9 of 9 planted faults detected. The production prover's equivalence tests come with #461.
- **The model.** Each growth, the per-bit tables, a computed instance's padding, a resident wider than every instance that sets none of the rounds, and the windows' split at nine own variables and at seventeen, front padding excluded, each with its regenerations and folds in `μ`. The cost-model report states each class, resident and window under its name.
- **The documents' figures.** `tessera-bench`'s tests pin the projections, the working sets, rounds and windows, the windows' terms, and the batches of 256 and 2,048 tokens, from the committed reports' classes. The `c_w` table, the cap's deltas and the shares of own variables follow from the formulas above and those classes, and are not pinned.

## Open Questions

- **OQ-0056-1: the window cap.** Eight own variables rests on SUR-0007's count and EXP-0012's extrapolation, at 405B's stages. At those estimates a second window saves 0.01 to 1.3 on `servable`'s stage 1 and costs 0.4 to 1.0 on `minimal`'s stages. OQ-0055-1's H100 measurement of the grid settles the cap, or replaces it by a choice per stage by cost. Owner: @AbdelStark. Resolve with v0.5's measurements.
- **OQ-0056-2: the batch.** `servable`'s corrected `μ` is lower at 2,048 tokens than at the scenario's 16K: 14.3 and 29.5 against 15.7 and 31.0. The projections keep the source's scenario. How a proof unit holds several requests, and so which batches a prover can choose, is [#357](https://github.com/AbdelStark/tessera/issues/357)'s. Owner: @AbdelStark. Answered by [RFC-0071](RFC-0071-request-slots.md): a batch is `R` whole sequences in slots of `S`, packed by slot class, which this model's `T/S` sequences of `S` price.

## References

- [#458](https://github.com/AbdelStark/tessera/issues/458); RFC-0055 and its review, where the gap was found.
- [IDEA-0039](../../research/ideas/IDEA-0039-causal-score-rounds.md) and [EXP-0026](../../research/experiments/EXP-0026-causal-score-rounds/README.md): the structured score rounds and their exact check.
- [RFC-0009](RFC-0009-gadgets.md#softmaxrow) 'SoftmaxRow', [RFC-0050](RFC-0050-gadget-advice-columns.md) and [RFC-0053](RFC-0053-greedy-runtime-boundary.md): the native relations.
- [SUR-0007](../../research/surveys/SUR-0007-sumcheck.md) and [EXP-0012](../../research/experiments/EXP-0012-integer-leaf-round-window/README.md): the grid's count and measurement.
