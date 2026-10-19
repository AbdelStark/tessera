# Transcript schedule vectors

`schedule.json` pins the transcript schedule of `tessera-protocol/1` ([RFC-0014 'Transcript schedule'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#transcript-schedule) and ['Grinding'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#grinding)): for each case, every label one proof unit absorbs or squeezes under, in order, and for a fixed test seed every challenge, seed, column index and proof-of-work nonce drawn at each. `crates/tessera-protocol/tests/schedule_vectors.rs` checks that its expansion gives the same steps and that a prover and a verifier held to the schedule draw the same outputs; `crates/tessera-prover/tests/transcript_sizes.rs` checks that the conformance models' sizes are those the prover's derivations give.

[`generate.py`](generate.py) writes it. The script is the independent oracle: it shares no code with `crates/`, `verifier/` or `python/`. It expands the labels from RFC-0014's table as [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) ('Tree levels', 'Batching within a stage', 'Sumcheck messages'), [RFC-0035](../../../docs/rfcs/RFC-0035-padded-rows.md), [RFC-0036 'What the transcript absorbs'](../../../docs/rfcs/RFC-0036-protocol-messages.md#what-the-transcript-absorbs), [RFC-0037 'Messages and transcript'](../../../docs/rfcs/RFC-0037-stage-schedule.md#messages-and-transcript) and [RFC-0045 'The split'](../../../docs/rfcs/RFC-0045-leaf-layout.md#the-split) amend it, and with RFC-0049's placement of each weight matrix's proximity point `pcs/weights/{T}/xi`: right after the seed, for each matrix with committed rows, in RFC-0040's order. It runs the transcript of [`../transcript/generate.py`](../transcript/generate.py), whose BLAKE3 and `H(domain, bytes)` are those of [`../hash/generate.py`](../hash/generate.py), checked against the official test vectors first. Its messages come from SplitMix64 with the seed the file states, so `python3 spec/vectors/transcript-schedule/generate.py` reproduces the file byte for byte.

## Cases

| Section | Cases |
|---|---|
| `cases` | Sizes that reach every rule of the schedule, in a live session and in non-interactive proofs, with no proof of work and with proofs of work of 6 to 10 bits; no tree, no stage and no matrix; fraction trees only; trees of height 0; `live-64/1` |
| `models` | The conformance models of [`../models/`](../models/README.md), each in a live session and in a non-interactive proof, at `test-small/1` |

The `ni-100/1` sets' 20-bit proofs of work would take the oracle minutes of hashing in Python. The cases grind for fewer bits under the same rules, as a hypothetical set: `seed_pow` and `query_pow` state the bits a case uses, and `parameter_sets` RFC-0014's sets.

## A case

| Field | Holds |
|---|---|
| `mode`, `params`, `seed_pow`, `query_pow` | The mode, the parameter set and the bits of its two proofs of work |
| `sizes` | What the schedule is expanded from: the leaf stage's `trees` in the levels' order, each a `fraction` tree of a `height` or a `forest` of `2^trees_log` trees of a `height`; `layer_vars`, the coordinates of stage 1's layer point; the reduction `stages`, each with its `rounds` and its zero-check point's `zero_vars`; `advice_vars`, the advice proximity point's coordinates; `proximity`, each matrix with committed rows and its point's `vars`; `opened`, each opened matrix and its `codeword` length |
| `provenance` | For a conformance model, where each size comes from |
| `context`, `context_tce` | The `SessionContext` and its TCE encoding: the transcript starts from `H("transcript-init", context_tce)` |
| `coin_toss` | In a live session, the shares `s_V` and `s_P` whose `H("coin-toss", s_V ‖ s_P)` is absorbed under `seed` |
| `steps` | Every step, in order: its `label` and its `op` |
| `replayed`, `state`, `counter` | How many steps the oracle ran, then the transcript's state and squeeze counter |

A step's `op` is one of:

| `op` | The transcript | The step records |
|---|---|---|
| `absorb` | absorbs `tce` under the label | `tce`, the absorbed bytes |
| `digest` | squeezes 32 bytes: the non-interactive seed | `out` |
| `fp2` | draws `count` challenges in `Fp2` one after the other: a challenge, or a proximity point's coordinates in draw order | `values`, each `[c0, c1]` |
| `indices` | draws `count` indices of `log_n` bits (`challenge_indices`) | `values` |
| `grind` | finds the smallest nonce whose `H("pow", state ‖ Ascii(label) ‖ nonce)` has `bits` leading zero bits, and absorbs it | `phase`, `bits`, `nonce` |

Decimal strings carry every integer a step draws. The messages absorbed are stand-ins: `TCE(Vec<Fp2>)` of one element, except a live session's seed, the coin toss of its shares, and its `queries`, a `TCE(ColumnQueries)` of the parameter set's counts below each codeword's length.

## The conformance models

Each model's sizes are its first sequence's, from its public part:

- **Derived here:** `layer_vars`, `⌈log2 n_layers⌉`; `proximity`, every matrix with committed rows at `test-small/1` ([RFC-0040](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md#the-matrices)) and `⌈log2 m_T⌉` coordinates for its `m_T` committed rows; `opened`, each of those matrices, with its codeword of `4·k_T` symbols. Every weight type has a `MatMulWeight` claim, and `ChannelParams` a claim from each site that reads a class or a multiplier.
- **Stated in the oracle, derived by the prover:** the `trees`, each LogUp instance's lookup and table trees from `tessera-prover`'s leaf layout of the requantization sites ([RFC-0045](../../../docs/rfcs/RFC-0045-leaf-layout.md)), the only gadget leaves included in this fixture; and the `stages`, from `tessera-compiler`'s stage schedule ([RFC-0037](../../../docs/rfcs/RFC-0037-stage-schedule.md)). `transcript_sizes.rs` derives them again and fails when the requantization layout or the stage schedule changes, until the oracle states the new sizes. The other gadgets' instances, the `SoftmaxRow` product forests and the gadgets' zero-checks join with complete gadget/leaf integration (#126, #130), which extend that test and the oracle with them.
- **Advice point:** `advice_vars = ν − log2 k`, derived through the independent public lowering and complete advice-layout oracles (RFC-0047/0050), at `test-small/1`, Q=512 and one segment. Every conformance case replays all steps, including `pcs/advice/xi` and subsequent column queries. The leaf tree sizes remain explicitly frozen from the prover's derivations; their order follows the complete canonical table catalog.
