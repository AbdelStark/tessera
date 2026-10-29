# RFC-0055: Windowed Rounds: Reduction Stages Larger Than Memory

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-23
- Target milestone: v0.3 (working sets, plan and cost model), v0.5 (GPU backend)
- Supersedes:
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) 'Pass structure', its P3 row, for a reduction stage larger than memory, and its pass counter's labels;
  - [RFC-0054](RFC-0054-token-groups.md#reduction-stages) 'Reduction stages', which left them to [#452](https://github.com/AbdelStark/tessera/issues/452), and its Drawbacks' line on them;
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model) 'Executable cost model': the reduction stages' plan, their fold passes and their windows' arithmetic;
  - [RFC-0006](RFC-0006-compiler-ir.md#outputs) 'Outputs': the commitment report lists each reduction stage's tables;
  - [RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages) 'Sumcheck messages': a pointer to the windows, its messages unchanged.
- Amended by: [RFC-0056](RFC-0056-native-working-sets.md) ('Native relations'; the 405B working sets, windows and `c_w` bands, which now count the native relations; 'The executable model': classes by growth, holding and width, the plan at a context, and a fold pass per window; Alternative 7, adopted beyond eight own variables per window)
- Scope: the prover and the cost model. Messages, labels, the verifier and the soundness bound are RFC-0010's, in `tessera-protocol/1` and `/2`.
- Origin: [#452](https://github.com/AbdelStark/tessera/issues/452)

## Summary

A reduction stage runs one batched sumcheck over every layer and token ([RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages)). The instances of the compiler's stage schedule ([RFC-0037](RFC-0037-stage-schedule.md)) hold tables of every layer and token. A 405B batch of 16K tokens gives `minimal`'s stages 1 and 2 working sets of 4.1 and 5.5 TB, and `servable`'s stage 1 one of 9.6 TB, against about 0.2 TB of free HBM. Under RFC-0010 as written, each round whose tables do not fit streams them: one regeneration pass per round, 12 for `minimal` and 8 for `servable`. With the native relations counted ([RFC-0056](RFC-0056-native-working-sets.md#at-405b)), the stages hold 6.4, 6.6 and 11.9 TB at 16K of context, and streaming would cost 18 to 25 and 14 to 21 regenerations, by context.

This RFC answers a stage's first `c` rounds from a **window** instead. One pass accumulates each elementwise instance's products on a small grid. A second pass folds its tables at the window's challenges, after which they fit. The messages are RFC-0010's, so the verifier, the transcript, soundness and zero knowledge do not change.

Such a stage then costs two passes instead of one, whatever its size: one more regeneration per stage, and one per further window where RFC-0056 splits a wide one. With the native relations RFC-0056 counts, that is +2 on `μ` for `minimal` at 405B and +3 for `servable`. The corrected 405B projections become about 13.8 at 1K and 29.1 at 16K for `minimal`, and 15.7 and 31.0 for `servable`. SC-2's 1K ceiling of 12 no longer holds at the corrected projection for either profile.

The prover also composes native gadget relations into stage 1 ([RFC-0050](RFC-0050-gadget-advice-columns.md)). Among them, `SoftmaxRow`'s key and forest relations span the score tensor, which grows with context: 8.8 TB at 1K and 141 TB at 16K for the same batch. A window cannot serve them at an acceptable cost ('Native relations'). [RFC-0056](RFC-0056-native-working-sets.md) proves them without the score tensor, counts every native relation, and restates the figures below.

## Motivation

[#452](https://github.com/AbdelStark/tessera/issues/452), found while resolving #329, sized three elementwise products from the model's shape and found each larger than memory. The compiler's schedule knows every instance of the graph. Its tables at 405B, two shift classes, `B_max = 32` and a 16K-token batch, one table per evaluation an instance leaves:

| Profile, stage | Tables | At 16 B per entry | Front padding of the largest |
|---|---:|---:|---|
| `minimal`, 1 | 2.5×10^11 entries | 4.08 TB | 3.85 TB unpadded; 0.21 TB padded by 4 |
| `minimal`, 2 | 3.4×10^11 | 5.50 TB | 4.40 TB unpadded; 1.10 TB padded by 2 |
| `minimal`, 3 | 5.4×10^8 | 8.7 GB | fits |
| `servable`, 1 | 6.0×10^11 | 9.57 TB | 4.40 TB unpadded; 4.95 TB padded by 2; 0.21 TB padded by 6 |
| `servable`, 2 | 5.5×10^8 | 8.7 GB | fits |

These tables do not depend on context; with the native relations, stage 1's do, a little ([RFC-0056](RFC-0056-native-working-sets.md#at-405b)). They are the elementwise instances' operands, such as the norms' squares, `a·u`, RoPE's rotations and the one-hot checks, and the matrix products' folded operands. Each is linear in the tokens or independent of them.

RFC-0010 binds the lowest variable first. A smaller instance is front-padded: constant in the stage's first `p = n_max − n_i` rounds. Under RFC-0010 as written, every round whose tables do not fit streams them, regenerating the stage's tensors. A stage then streams until its tables, each folded by `2^{max(0, j − p)}` after round `j`, fit: 6 rounds for each of `minimal`'s stages and 8 for `servable`'s stage 1. That adds 12 and 8 regenerations to `μ`, which RFC-0018's estimate of at most one forward pass for all of P3 does not count.

## Goals

- Every reduction stage's schedule instances cost at most two regenerations, whatever their size.
- No change to the messages, the transcript, the verifier, soundness or zero knowledge.
- The commitment report gives each stage's tables, and the cost model sizes and prices each stage's plan from them.

## Non-Goals

- Fewer regenerations than one per stage. Fusing passes across stages is RFC-0018's 'Fusion', and the dependencies in 'Pass structure' below rule it out for the windows.
- The leaf stage, which RFC-0054 proves in token groups.
- The native gadget relations ('Native relations'), which [RFC-0056](RFC-0056-native-working-sets.md) handles ([#458](https://github.com/AbdelStark/tessera/issues/458)).

## Proposed Design

### Working sets

A reduction stage's instance `i` of `n_i` variables holds one table of `2^{n_i}` entries per evaluation it leaves: `MatMulWeight`'s folded `W̃` and `X̃`, `Mul`'s two operands, `SumSquares`'s input. An entry is charged 16 bytes, one `Fp2`. A table enters the stage as base-field integers, 8 bytes each, and the first round writes half as many `Fp2` values beside them.

The `eq` factors are not stored as tables. They are kept split, `O(2^{n_i/2})` ([DDBDT26] §6). Public factors are evaluated from their closed forms, or held as tables of their own variables, which stay small: RoPE's angles are positions by frequencies.

**The report.** The compiler's commitment report lists each reduction stage's tables by size (`stage_tables`). Each class is `count` tables of `2^{log2_entries}` entries, per token of the batch or fixed, and windowable or not ('The window').
- **Classification.** The compiler derives the classes from the schedules of `n` and `n/2` positions, `n` the report's sequence. A per-token instance gains one variable when the sequence doubles, and a fixed one, a weight's folded table, none. The report refuses a graph where an instance does anything else.
- **At a batch.** A batch of `T` tokens, padded to `2^b`, gives a per-token table `log2_entries + b` variables. `tessera-bench` evaluates the classes at the batch.
- **An odd sequence.** When `n` is not a power of two, every table counts as per token, at `2^{n_i − ⌊log2 n⌋}` entries per token: an upper bound at every larger batch.
- **Native relations.** [RFC-0056](RFC-0056-native-working-sets.md) adds them from the composed schedule. It extends each class with a growth with the context, a holding (windowable, resident or computed) and a width, and lists the score instances' residents.

### The window

Let `B` be the memory budget. A stage whose tables exceed `B` takes a window of `c` rounds: the fewest after which its tables fit, each windowed table folded by `2^{v_i}`:

```text
v_i = min(n_i, max(0, c − p_i)),   p_i = n_max − n_i,   Σ_i 2^{n_i − v_i} · 16 · tables_i ≤ B,   c < n_max
```

`v_i` counts instance `i`'s own variables among the stage's first `c` rounds.
- **What a window folds.** An elementwise instance's tables, which a regeneration yields block by block. A matrix product's tables are folds over another index, such as `X̃(r_t, i)`, so they stay whole (`v_i = 0`) and count in full. So does any instance with `v_i = 0`, and any the prover chooses to keep resident.
- **The bound on `c`.** `c < n_max` leaves every windowed instance at least its last round in memory. That round's evaluation carries RFC-0013's masking term, which the grid does not compute.

**The grid pass (P3).** Write instance `i`'s summand as `eq(z, x_E) · Π_k f_k(x)`.
- **The factors.** `x_E` are the variables the relation's `eq` covers: all of them for `Mul`, the rows for `SumSquares`, the tokens for `OneHotCount` and `OneHotIndex`. The `f_k` are its tables and its public factors that depend on `x`: the causal mask, RoPE's kernel `E_r`, `OneHotIndex`'s index weights.
- **The degree.** `δ_i` is the summand's degree per variable once `eq` is factored out: 2 for `Mul` and the zero-checks, and 3 for a native product.

The prover regenerates the stage's tensors once. For each windowed instance it accumulates, for every point `u ∈ {0, …, δ_i}^{v_i}`:

```text
A_i(u) = Σ_y eq(z_{E ∩ y}, y_E) · Π_k f̃_k(u, y)
```

- **The sum.** `y` runs over the instance's variables above its window, and `eq(z_{E ∩ y}, y_E)` is the `eq` factor's part over those of them it covers.
- **The extension.** `f̃_k(u, y)` extends a block of `2^{v_i}` consecutive entries of `f_k` to the grid. A block is consecutive in the regenerated order: the window's variables are the lowest ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)), a row-major tensor's last index.
- **Size.** The grid holds `(δ_i + 1)^{v_i}` accumulators per instance: 6,561 `Fp2` at `δ = 2` and `v = 8`, about 100 KB.
- **Its inputs.** The pass needs the claim points only. That is the previous stage's final point, and in stage 1 also its layer point and zero-check point. It needs no claim value and no `γ`.
- **When it runs.** Between the previous stage's final evaluations and its split values, which the stage's tensors evaluate ([RFC-0037](RFC-0037-stage-schedule.md#splits)). The grid gives them: a split value is an instance's sum `S_i`, the grid's Boolean corner weighted by the window's `eq` factor. Stage 1 follows the leaf stage, whose split values the merge or the leaf stage itself computes. One pass therefore serves both, as P3 does for a stage that fits.

**The window's rounds.** Rounds 1 to `c` send RFC-0010's round polynomials, and each instance enters them as RFC-0010's prover adds it.
- **A padding round.** In round `j ≤ p_i`, instance `i` is constant: it adds `γ^i·2^{p_i − j}·S_i` at every node.
- **An own round.** Round `j > p_i` reads `A_i` weighted, coordinate by coordinate:
  - by the Lagrange basis on `{0, …, δ_i}` at `r_k` for each own variable already bound,
  - by that basis at `X` for the round's own variable,
  - and by the sum over `{0, 1}` for the window's later variables;

  times the `eq` factor's part on the window's coordinates that `x_E` covers, which is linear in each, and times `γ^i`.

**The fold pass (P3b).** After round `c`, the prover regenerates the stage's tensors again and binds each windowed table, its public factors included, at its window's challenges:

```text
f_k(r_window, y) = Σ_{w ∈ {0,1}^{v_i}} eq(r_window, w) · f_k(w, y)
```

That is one product of an `Fp2` weight by a base-field entry per entry. The folded tables, `2^{n_i − v_i}` entries each, fit by the choice of `c`.
- **The rounds after.** Rounds `c + 1` to `n_max` run in memory as RFC-0010's prover runs them, and the stage ends with RFC-0010's evaluations.
- **Binding in place.** The folded tables are the prover's own, not borrowed from a witness, so it binds them in place and their size stays the budget's. RFC-0010's resident prover binds its first round out of place.

**Secrets.** The accumulators and the folded tables derive from secret tables. They live in `SecretBuf`, as every derived table does ([05 §3](../spec/05-observability.md#3-data-classification-and-redaction), OBS-6), and their arithmetic stays branch-free on the values (INV-ARCH-2).

### Pass structure

| Pass | When | Regenerates | Native cost |
|---|---|---|---|
| P3 stage regeneration | Per reduction stage | Folds of the tensors the stage reads; for a windowed stage, its window's grid and the previous stage's split values | ≤ 1× forward in total for the stages that fit; at most 1× forward per windowed stage |
| P3b window fold | Per windowed stage, after its window's rounds | The windowed tables, which it binds at the window's challenges | At most 1× forward per windowed stage |

Neither pass fuses with another:
- **The previous pass.** A stage's grid pass needs its claim points, which follow the last round of the stage before. That is RFC-0054's merge for stage 1.
- **The next pass.** Its fold pass needs its own window's challenges, which follow the grid pass.
- **Across stages.** The next stage's grid pass needs this stage's final point, which follows its fold pass.

Each windowed stage therefore adds one regeneration, and each further window RFC-0056 splits off one more. The pass counter labels it `p3b` ([05 §4](../spec/05-observability.md#4-metrics)).

At 405B, 16K tokens and 0.2 TB, with the native relations counted ([RFC-0056](RFC-0056-native-working-sets.md#at-405b)):
- **`minimal`.** Stages 1 and 2 take windows of 15 and 7 rounds at 16K context, +2 regenerations.
- **`servable`.** Stage 1's 15 rounds take two windows, and stage 2 a window of 3 rounds, +3.
- **Their own variables.** In `minimal`, stage 1's window binds eight own variables on 78% of its entries and stage 2's seven on 67%, the rest five or fewer. In `servable`, stage 1's `a·u` binds ten, and the instances padded by two bind eight, so RFC-0056 splits its rounds into two windows of five ('The windows'). The matrix products stay whole.
- **Without the native relations.** This RFC first counted the compiler's graph schedule alone: windows of 6 and 6 rounds for `minimal`, and 8 for `servable`'s stage 1, +2 and +1.

### Native relations

The prover composes native gadget relations into stage 1 (RFC-0050; [RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels': the consuming stage proves a forest's leaf table). They include:
- each root's coefficient products, of degree 4, and its padding checks;
- `SoftmaxRow`'s key and forest relations, its reciprocal and its padding checks.

`SoftmaxRow`'s key and forest relations span the score tensor: two tables of `h·S²` entries per sequence and layer, about 8.8 TB at 1K context and 141 TB at 16K for a 405B batch of 16K tokens.
- **Their window.** It would bind 9 to 13 own variables, at 115 to 584 `Fp` multiplications per index against about 40 for the rounds it answers.
- **Several windows.** They would cost a regeneration each, three at 16K.

[RFC-0056](RFC-0056-native-working-sets.md) proves them without the score tensor, from the INT8 queries and keys and the causal mask's closed form, in memory linear in the tokens. It also counts every native relation in the report and the model. The windows above serve the instances that stay elementwise.

### Cost

**Passes.** One regeneration per windowed stage, one per window under RFC-0056: `p_w` in [RFC-0020](RFC-0020-cost-model.md#executable-cost-model)'s `μ`. Like the merge's pass, it counts when the model counts regenerations.

**Arithmetic.** For a window of `v` own variables, `δ = 2` and `eq` factored out:
- **The grid, counted.** It costs `1.5^v` products of small values and `1.5^v` scalings by an `Fp2` weight per index, `3 · 1.5^v` `Fp` multiplications: 15, 23, 34, 51 and 77 at `v` = 4 to 8. The per-round prover over the same rounds costs about 33 to 38 ([SUR-0007](../../research/surveys/SUR-0007-sumcheck.md), E0). By this count, the grid costs no more than the rounds it replaces up to six own variables.
- **The grid, measured.** [EXP-0012](../../research/experiments/EXP-0012-integer-leaf-round-window/README.md) measured integer windows on one CPU core (E1).
  - **Whole product forests.** 8% faster than the in-memory prover at `v = 3`, 2% faster at 4, and 19% slower at 5. At 5, the leaf level cost 1.4 times the in-memory one.
  - **Extrapolated.** The grid's time grew by 1.39 per variable from 3 to 4 and by 1.64 from 4 to 5. At 1.5 to 1.64 per variable, a level would cost 2.0 to 2.1 times the in-memory one at six own variables, and 4.0 to 5.1 times at eight.
  - **Its limits.** The measurement outweighs the count, but its windows were of three to five variables, and these are of six and eight. Its i128 accumulators would also overflow at six for its keys, so a larger window needs wider sums or more reductions.
- **The fold.** Two `Fp` multiplications per entry.

**The unit cost.** The model's `c_w` (`window_ps`) is the cost per entry of a windowed table: the fold, plus the grid beyond the rounds it answers.
- **The fold.** RFC-0054 prices the merge's two folds, four `Fp` multiplications per element, at 2 to 4 ps, so 0.5 to 1 ps per multiplication. The fold then costs 1 to 2 ps per entry.
- **The grid's excess.** `c_π = 12` ps prices a product-tree leaf over all its levels, about 12 ps per pair at the leaf level. Read at that rate, the measured excess is about 6 to 7 ps per entry at six own variables and 18 to 25 at eight.
- **Per own variable.** [RFC-0056](RFC-0056-native-working-sets.md#the-windows) tabulates both estimates for windows of up to ten own variables, and caps a window at eight.
- **`minimal`.** Its windows bind eight and seven own variables on most of their entries ([RFC-0056](RFC-0056-native-working-sets.md#at-405b)). `c_w` is about 6 to 12 ps by the count and 12 to 17 by the measurement, so 6 to 17 ps. That adds 0.42 to 1.20 to `μ` at 1K and 0.40 to 1.14 at 16K.
- **`servable`.** Its stage 1's rounds bind ten own variables, in two windows of five. `c_w` is 1 to 3.5 ps per window, which adds 0.13 to 0.47 at 1K and 0.13 to 0.45 at 16K.
- **Without the native relations.** This RFC first gave 1 to 8 ps and 5 to 17 ps, for windows of six and eight own variables.
- **Measurement.** v0.5 measures the grid on H100 (OQ-0055-1).

### Soundness and zero knowledge

- **Soundness.** The prover sends RFC-0010's round polynomials and evaluations. Only the order in which it computes them changes, so the stage's error stays `(n·d + m)/|Fp2|`, and the soundness calculator does not change.
- **Zero knowledge.** RFC-0013's masked rounds add the masking polynomial's part to the same polynomials.
  - **The window's share.** It computes only the unmasked part.
  - **The masking term.** `c < n_max` keeps every windowed instance's last round, where the masked extension's term `Z·R̃` no longer vanishes, outside the window.
- **No new trust.** A prover that computes a window wrongly sends a wrong round polynomial, which RFC-0010's checks reject as they reject any other.

### The executable model

`tessera-bench` reads the report's classes into `Counts::reduction_stages`, one `ReductionStage { tables: Vec<StageTable> }` per stage. [RFC-0056](RFC-0056-native-working-sets.md#the-executable-model) adds the classes' growth, holding and width, the residents, the context, and a window per eight own variables.

`stage_plan(counts, memory, batch_tokens)` sizes each stage at the batch:
- **Its working set.** Its tables at `STAGE_BYTES_PER_ENTRY` = 16.
- **Its window.** `c`, by the rule above, 0 when its tables fit.
- **The entries it folds.** Those of the windowed tables.
- **Refusals.** It refuses a stage that no window of at most `n_max − 1` rounds fits (`free_bytes`, `TooSmall`), and a batch beyond `2^31` tokens (`batch_tokens`, `TooLarge`).

`mu` adds two things:
- **The regeneration term.** One fold pass per windowed stage, when `n_regen` is positive.
- **The streaming term.** `window_ps · Σ_w E_w / T`, with `E_w` the windowed tables' entries at the batch.

Each projection lists the stages' working sets and windows.

## Alternatives Considered

1. **RFC-0010 as written: stream every round that does not fit.** One regeneration per round, 12 for `minimal` and 8 for `servable` at 405B and 16K tokens, against 2 and 1. Rejected.
2. **Token groups and a merge per stage** (#452's first direction, RFC-0054 applied to a stage). It costs the same one more pass per oversized stage, the merge's, and changes the protocol for it: per-group messages and challenges, merge rounds, a rejection, a union bound, and OQ-0054-1's zero-knowledge gap in every stage. Rejected for the schedule's instances: the same passes, more protocol. The score instances may still need groups, since their windows cost more (#458).
3. **Groups carried through the stages** (#452's second direction). Each stage still needs one pass over all groups, since its shared challenges follow the stage before. A shared tensor's claims, a weight's above all, land at `G` points, each a pass over the resident weights, unless the matrix products batch across groups with shared challenges for the weights' variables. And each committed tensor's `G` claim groups cost a combined row each. RFC-0054 Alternative 6 rejected the same for the leaf stage. Rejected.
4. **A smaller batch.** 256 tokens fit `minimal`'s stages, 0.10 TB each, and the native relations counted. But the weight pass and the table side, shared by 64 times fewer tokens, cost more than the windows save: `μ` is 16.1 at 1K and 31.2 at 16K, and 14.2 and 29.6 at 2,048 tokens, against 13.8 and 29.1 at 16K tokens. `servable` is the exception: at 2,048 tokens its stage 1 takes one window and its stage 2 fits, for 14.3 and 29.5 against 15.7 and 31.0 (RFC-0056 OQ-0056-2). Rejected as a rule. The planner chooses the batch, and the model prices either.
5. **Tables in host memory.** At 8 bytes per base-field entry, a stage holds 2.0 to 4.8 TB, beyond 1 TiB of host RAM. Each round would also cross PCIe, slower than a regeneration. Rejected.
6. **Multi-variable rounds or the univariate skip.** Both change the messages, the verifier and the soundness bound for what a window gets without them ([RFC-0018](RFC-0018-gpu-prover-runtime.md#alternatives-considered) Alternative 3, SUR-0007). Rejected.
7. **Two windows per stage, to keep each at five own variables or fewer.** A second window's grid can ride the first's fold pass, but it needs a second fold pass: another regeneration, about 1 on `μ`. Adopted by [RFC-0056](RFC-0056-native-working-sets.md#the-windows) beyond eight own variables: for `servable`'s stage 1, whose rounds bind ten, it saves 0.01 to 1.3. For `minimal`'s windows of eight and seven it costs 0.4 to 1.0. Rejected for windows of eight or fewer.

## Drawbacks

- **One more regeneration per stage larger than memory.** At 405B and 16K tokens, with the native relations counted, that is +2 on `minimal`'s `μ`, about 14% of its `μ(1K)` and 7% of its `μ(16K)`, and +3 on `servable`'s. SC-2's 1K ceiling fails at the corrected projection for both profiles: 15% and 31% above it before the column answers.
- **The grid's arithmetic.** It grows by 1.5 to 1.64 per own variable. EXP-0012 measured it above the in-memory rounds from five own variables on, and `minimal`'s windows bind seven and eight. It adds up to 1.2 to `minimal`'s `μ(1K)` and up to 0.47 to `servable`'s.
- **A second prover path.** Windowed and resident rounds must produce byte-identical messages, which the equivalence tests below hold.
- **The model's charge.** It counts one forward pass per fold pass. A fold regenerates only the stage's tensors, but they span every layer, so the charge is an upper estimate.
- **Native relations.** Counted, and the score instances proven without the score tensor, by RFC-0056.

## Migration / Rollout

- **v0.3.**
  - **The report and the model** (this RFC's change): the commitment report's `stage_tables`, `tessera-bench`'s `stage_plan`, the fold passes in `μ` and the projections, and the pass counter's `p3b` label.
  - **The CPU prover.** It keeps every stage resident: the conformance models' stages fit, and its pass plan ([#132](https://github.com/AbdelStark/tessera/issues/132)) refuses one that does not.
  - **The native relations.** [RFC-0056](RFC-0056-native-working-sets.md) counts them ([#458](https://github.com/AbdelStark/tessera/issues/458)).
- **v0.5.**
  - **The windowed rounds.** They land in `tessera-sumcheck` as a prover path with the equivalence tests below ([#459](https://github.com/AbdelStark/tessera/issues/459)).
  - **The GPU backend.** It windows the stages that exceed its budget, and the planner chooses `c`.

## Testing Strategy

- **Working sets.**
  - **At every batch.** The report's classes, evaluated at `T` tokens, equal the schedule's tables at `T` positions, for both profiles at 4, 32 and 256 positions (`tessera-compiler`).
  - **Odd sequences.** A report of 12 positions bounds the schedule's tables at 12 and 32.
  - **Each refusal.** A unit test violates each of the classification's conditions in turn: another relation, another table count, a growth of two variables, a shrinking table, and a table smaller than the sequence.
  - **At 405B.** The two-class reports at 16K tokens equal the schedule's tables at 16,384 positions, and the model's working sets and windows equal those computed from the schedule's own variables and padding (`tessera-bench`).
  - **End to end.** A compiled plan's projections size every stage the report lists (`tessera-cli`).
- **The model.**
  - **Windows.** `stage_plan` grows a stage's window as the memory halves, counting front padding, folds no matrix product, and stops one round short of the stage's last. It refuses a stage that cannot fit, and a batch beyond `2^31` tokens.
  - **`μ`.** One regeneration per windowed stage, counted only when `n_regen` is positive, and `window_ps` per windowed entry.
  - **The documents.** Their figures are held by `tessera-bench`'s tests.
- **The windowed prover** (with its implementation, [#459](https://github.com/AbdelStark/tessera/issues/459)).
  - **Equivalence.** Its messages equal the resident prover's byte for byte, and the proofs verify, across:
    - every relation kind, including those whose `eq` covers only some variables and those with public factors;
    - front-padded instances, and windows of 1 to `n_max − 1` rounds;
    - instances kept resident beside windowed ones.
  - **Memory.** Under a forced budget, the stage never exceeds it, and the pass counter counts P3 and P3b once each.
  - **Adversarial.** A fixed algebraic corruption of a grid, with the original claim and regenerated source unchanged, must produce `Reject(Sumcheck)` in the regression fixture. Assert the final relation mismatch at round `n_max`: RFC-0010 omits `p(1)` and reconstructs it from the running claim, so a well-formed corrupted round need not be rejected at the round that first uses the grid. This test does not change the protocol or its soundness bound.

## Open Questions

- **OQ-0055-1: the grid on H100.** The window's arithmetic at six and eight own variables, and whether its scalings are bound by bandwidth or by multiplications, decide `c_w`.
  - **The disagreement.** EXP-0012's CPU measurement and SUR-0007's count disagree by about twice.
  - **The cap.** The measurement also settles RFC-0056's cap of eight own variables per window (OQ-0056-1).
  - **Owner and deadline.** @AbdelStark; resolve with v0.5's measurements.
- **OQ-0054-2** stays open. The windows do not share the merge's pass: stage 1's grid pass needs the claim points the merge leaves.

## References

- [#452](https://github.com/AbdelStark/tessera/issues/452), its first-order table and its three directions; [#329](https://github.com/AbdelStark/tessera/issues/329) and RFC-0054, where the question arose.
- [SUR-0007](../../research/surveys/SUR-0007-sumcheck.md): the windows' arithmetic and the pass counts, from [DDBDT26] and [BCFFMMZ25].
- [EXP-0012](../../research/experiments/EXP-0012-integer-leaf-round-window/README.md): integer windows of one to five variables measured against the split-`eq` prover on CPU.
- [DDBDT26] Dao, DeStefano, Bagad, Domb and Thaler, "Speeding Up Sum-Check Proving (Extended Version)", ePrint 2026/587: §5's round batching, a window of `v` rounds answered from a `(d + 1)^v`-point grid accumulated in one pass, and §6's `eq` factored out.
- [BCFFMMZ25] Baweja, Chiesa, Fedele, Fenzi, Mishra, Mopuri and Zitek-Estrada, "Time-Space Trade-Offs for Sumcheck", TCC 2025, ePrint 2025/1473.
