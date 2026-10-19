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

## Who checks them

- `verifier/crates/tv-core` (`tests/schedule_vectors.rs`, [#114](https://github.com/AbdelStark/tessera/issues/114)) lowers each golden claim list of `../claims/` at each length, derives its schedule, and compares it with the case field by field, every coordinate of every point included. It reports every item that differs.
- `crates/tessera-prover` (`tests/verifier_stages.rs`) proves the stages of each model over the CPU engine's witness and verifies them with the verifier's stage checks.
