# Stage schedules

`schedule.json` holds each conformance model's stage schedule ([RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md#stages) 'Stages', [RFC-0037](../../../docs/rfcs/RFC-0037-stage-schedule.md)) at the lengths 1, 4, 5 and `max_context`: the stage count, the leaf stage's claims, each reduction stage's instances, and the claim pool. The schedule is a function of the lowered graph alone, which the prover and the verifier derive independently and must agree on (INV-SCH-1). These vectors hold both to one schedule.

## How they are made

`cargo xtask gen-vectors` writes the file from the compiler's derivation, `tessera_compiler::derive_schedule`, of each model of [`../models/`](../models/README.md), after the models' oracle. It is no independent oracle's: like the golden claim lists of `../claims/`, it records what the prover derives, so that the verifier, which derives its schedule from the claim list with code of its own, is compared with the prover's on data and never on code (RFC-0001). `cargo xtask gen-vectors --check` fails when the file differs from what the compiler derives.

## `schedule.json`

`cases` holds one case per model and length.

| Field | Meaning |
|---|---|
| `model`, `n` | the conformance model and the sequence's length |
| `layers`, `layer_vars` | `L` and `λ = ⌈log2 L⌉` |
| `stage_count` | the number of stages, the leaf stage included |
| `sites[]` | the leaf stage's claims, in order: `{node, input}`, input `input` of a gadget's node, layer 0's for a template position, then `{logits: true}` |
| `stages[]` | stage `index`, from 0, the leaf stage first: its `rounds` `n_max`, its zero-check point's coordinates `zero_vars`, the values its split claims send, `splits` (for the leaf stage, every layer's), and its `instances` |
| `stages[].instances[]` | in order: the `relation`'s kind (`MatMulWeight`, `MatMulAct`, `Elementwise`, `Rotate`, `SumSquares` or `ZeroCheck`), the `node` whose relation it is (layer 0's for a template position), the `tensor` its claim states (layer 0's for a stacked kind), whether it is `stacked` over the layers, its variables `vars`, its claim's `point` (a zero-check's: its coordinates of the zero-check point) and its claim's `layers`, the layer point of `eq(ρ, ·)`, or `null` |
| `pool[]` | the claim pool, in the order the stages leave it: the `stage`, the `commitment` (`advice` or `weight`, the class and multiplier vectors being the weights'), the number of `atoms`, and the `point` |

A coordinate is written `leaf:{s}:{i}` for coordinate `i` of leaf claim `s`'s point, `layer:{i}` for the layer point `ρ_1`'s, `zero:{i}` for stage 1's zero-check point's, `round:{k}:{j}` for the challenge of round `j` of stage `k`, and `half` for `½`.

## `degrees.json`

`degrees.json` holds each reduction stage's `tessera-protocol/2` round degree ([RFC-0013](../../../docs/rfcs/RFC-0013-zero-knowledge.md#masked-extensions) 'Masked extensions' 'Degree', [#565](https://github.com/AbdelStark/tessera/issues/565)) on the composed schedule `tessera-protocol/2` proves: `derive_composed_schedule` with `model_natives` for the request, never `derive_schedule`'s graph-only schedule, which lacks the natives. Its 44 cases are each conformance claim list of `../claims/`, at `schedule.json`'s `n` of 1, 4, 5 and `max_context`, for the requests `none` and `prefill` at every `n` and `generate` at `n ≥ 2`: the claim lists in `claim-lists.json`'s order, then `n` ascending, then `none`, `prefill` and `generate`. `cargo xtask gen-vectors` writes it from the compiler's schedules and RFC-0013's table (`tessera_compiler::schedule::protocol2_stages`), and checks what the RFC says it records: `[10, 7]` in the 22 cases of the two `servable` lists and `[10, 7, 7]` in the 22 of the `minimal` and `affine` lists.

| Field | Meaning |
|---|---|
| `description`, `cases` | the file, as `schedule.json`'s |
| `cases[]` | `model`, the claim list's `name`, as `schedule.json` names it; `n`, the sequence's positions; `length`, the positions its schedule is lowered at, `max(n, 3)`, since `n` = 1, 2 and 3 share the layout of a 3-row token axis padded to 4; `request`, `none`, `prefill` or `generate`; and `stages[]` |
| `stages[]` | the reduction stages, `index` from 1: `rounds`, `tessera-protocol/2`'s, `max(max_i n_i, log2 k)` when the stage has a fold and `max_i n_i` otherwise, `log2 k` the claim list's `layout.advice.log2_k`; `d`, the largest of its instances' and its folds' (2); `folds`, one per instance whose claim is a virtual part the stage before left, none for a native or a zero-check; and `instances[]`, in schedule order |
| `stages[].instances[]` | `relation`, the table's string; `stacked`, true when the instance's layer weights run over `λ ≥ 1` coordinates or it is a `MatMulAct` with two heads or more, false otherwise; `public`, one Boolean per operand, `[]` for a native; `constant`, `scalar`, `channel` or `layer` for an `Equality.ConstMul` and `null` otherwise; and the row's `d_1`, `s` and `d` |

`stacked` is not `schedule.json`'s: that field records layer weights `eq(ρ, ·)`, `Layers::Eq`, which is true at a template position of a graph with `λ = 0`, where this one is false, and false for a `MatMulAct` stacked over heads alone, where this one is true.

## Who checks them

- `verifier/crates/tv-core` (`tests/schedule_vectors.rs`, [#114](https://github.com/AbdelStark/tessera/issues/114)) lowers each golden claim list of `../claims/` at each length, derives its schedule, and compares it with the case field by field, every coordinate of every point included. It reports every item that differs.
- `crates/tessera-prover` (`tests/verifier_stages.rs`) proves the stages of each model over the CPU engine's witness and verifies them with the verifier's stage checks.
- `degrees.json` is a differential vector, as `schedule.json` is: `verifier/crates/tv-core` derives each schedule and each instance's row with code of its own and compares every field, once #565's verifier half lands. `cargo xtask gen-vectors --check` fails when the file differs from what the compiler derives, and `crates/tessera-prover` (`tests/protocol2_degrees.rs`) measures each row of RFC-0013's table on its tiny masked instance.
