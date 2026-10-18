# Weight commitment vectors

`weights.json` pins the weight commitment of [RFC-0040](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md) at `test-small/1`:
- for each conformance model of [`../models/`](../models/README.md), and for one synthetic model, each of the ten matrices' message width, committed rows, `commitment_id` and root;
- then `R_W` and the model's `WeightCommitmentPublic` artifact (schema `0x0301`), each also written as `<model>.twr`;
- and rows encoded at rate 1/4, the code of every weight matrix.

## How they are made

[`generate.py`](generate.py) is the independent oracle. It is written from RFC-0040 ('The matrices', 'Encoding, leaves and roots', 'WeightCommitmentPublic'), [RFC-0012](../../../docs/rfcs/RFC-0012-weight-commitment.md#layout) 'Layout', [RFC-0031](../../../docs/rfcs/RFC-0031-commitment-id.md#the-descriptor) 'The descriptor', [RFC-0004](../../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests) ('Column digests', 'Merkle trees'), [RFC-0011](../../../docs/rfcs/RFC-0011-advice-commitment.md#encoding) 'Encoding', RFC-0003 and 03 §2 alone, and computes with Python's integers. It shares no code with `crates/`, `verifier/` or `python/`. It takes from the other oracles:

- BLAKE3, `H(domain, bytes)`, the Merkle levels and `TCE(CommitmentDescriptor)` from [`../hash/generate.py`](../hash/generate.py), whose BLAKE3 is checked against the official test vectors first;
- the roots of unity, Horner's rule and the masked polynomial from [`../encoding/generate.py`](../encoding/generate.py);
- the radix-2 NTT from [`../pcs/generate.py`](../pcs/generate.py);
- the `IntegerModel` types and encoder from [`../models/generate.py`](../models/generate.py).

It reads each model with its own TCE decoder, and checks that the models oracle's encoder writes the artifact back byte for byte and that its digest is the one `../models/models.json` lists. Two parts run faster than their definitions, and each is checked against them before anything is written:

- **Codewords.** A row's polynomial comes from an inverse NTT, and its codeword from NTTs of size `k` on the four cosets `7·ω_n^s·H` that make up `7·L`. The first and last committed rows of every matrix are checked against the definition, Horner's rule at each point: every symbol for the first row of each size, eight symbols for the others.
- **Hashing.** Every column of a matrix advances its chain at the same time, so the column chains and the Merkle levels run as one BLAKE3 compression over all of them: each 32-bit word of the state holds one lane per column, 64 bits apart, in one Python integer. Every input is a single block (`column-init`, `column-step` with `B = 4`, `merkle-node`). The lanes are checked against the hash oracle's `H` on random inputs first. For every matrix, one column's chain is recomputed with `H`, and for every model, one matrix's Merkle levels.

`python3 spec/vectors/weights/generate.py` rewrites every file here byte for byte, in about 30 seconds. It reads `../models/`, so it runs after that oracle; `cargo xtask gen-vectors` runs the oracles in name order.

## Parameters

Every model is committed at `test-small/1` ([RFC-0014](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets), [RFC-0026](../../../docs/rfcs/RFC-0026-claim-list-layout.md#schema)), with no mask and no blinding row, as `tessera-protocol/1` commits them:

| Field | Value |
|---|---|
| `layout` | the claim list's `WeightLayoutParams`: `log2_k` 8, `log2_inv_rate` 2, `mask_coeffs` 64, `blinding_rows` 32 |
| `row_block` | `B`, 4 |
| `epoch` | 0, the first commitment |

## `weights.json`

| Field | Meaning |
|---|---|
| `encoding[]` | a row of `k` values (`row`), encoded with the mask `m_0, …, m_{P−1}` (`mask`, `P` = `mask_coeffs`) into `codeword`, `n = 4k` symbols: `g = f + (X^k − 1)·m` at `7·ω_n^j`, as in [`../encoding/README.md`](../encoding/README.md#conventions). `P = 0` for `k = 2^0` to `2^6`, a weight row of `tessera-protocol/1`; then `P` of 1, `k/2` and `k` for `k` of 4 and 16 |
| `models[].name`, `arch` | the model and its `ArchSpec` sizes |
| `models[].model`, `model_digest` | a conformance model's artifact and its envelope digest |
| `models[].tensors[]` | each weight: `matrix`, `layer`, `rows` (outputs) and `cols` (inputs). For a conformance model, `offset` is where its first entry lies in the artifact, from the envelope's first byte, so that a reader can take the weights without decoding `IntegerModel`; for the synthetic model, `data` lists the entries, row-major |
| `models[].channels[]` | each `ChannelParams`: `matrix`, `layer`, `kind` (`class` or `multiplier`), `count` and `offset` of its values, `u8` or `u16` |
| `models[].matrices[]` | each matrix, in `WeightMatrix` order: `log2_k` (`log2 k_T`), `row_vars` (the bits of the row index), `rows` (`m_T`) and `committed_rows` (in increasing order); for a weight type, `out`, `in`, `layers`, `a` (`a_T`) and `h` (`h_T`); for `ChannelParams`, `channel_width` (`C`); then `descriptor` (`TCE(CommitmentDescriptor::WeightMatrix)`), `commitment_id` and `root` |
| `models[].public` | `TCE(WeightCommitmentPublic)`: the layout, the ten roots and the epoch |
| `models[].weights_root` | `R_W = H("weights-root", public)` |
| `models[].file`, `bytes`, `digest` | the `.twr` artifact, `public` sealed in its envelope, its size and its envelope digest |

Values in `encoding` are `"0x"` and 16 hex digits; digests and byte strings are hex.

## The synthetic model

It has three layers, `d_model` 12, two heads of 6, one KV head, `d_ff` 20 and a vocabulary of 300, and no `ChannelParams`. Its weights are uniform in `[−127, 127]`, both ends included. It covers what the conformance models do not:

- an embedding wider than `k_W`, `in_pad = 512`, so `k_T = in_pad`;
- widths and a layer count that are not powers of two, and rows that hold padded outputs;
- a message narrower than `k_W` (`K` and `V`, `k_T = 128`);
- `C = 1`: the `ChannelParams` matrix commits no row, and its leaves are the chains' first digests `d_j[0]`.

## Who checks them

- `crates/tessera-pcs` (`tests/weights_vectors.rs`) reproduces every value: the rate-1/4 encodings bit for bit; each matrix's layout and committed rows from its shape; each `commitment_id` from its descriptor; each root from the weights it reads at their offsets; then `public`, `R_W` and the `.twr` artifact.
- `crates/tessera-ir` (`tests/weights.rs`) decodes each conformance model, takes its shapes from its `ArchSpec`, commits its weights, and compares every matrix, `R_W` and the `.twr` artifact.
- `verifier/crates/tv-codec` (`tests/weights.rs`) opens each `.twr` envelope, decodes its `WeightCommitmentPublic` and recomputes `R_W`, never seeing a weight.
