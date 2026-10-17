# Conformance models

The tiny conformance models of [07 §5](../../../docs/spec/07-testing-strategy.md#5-differential-and-conformance-tests): Llama-3.1-architecture models with random INT8 weights, each sealed as an `IntegerModel` artifact (schema `0x0101`, [03 §2](../../../docs/spec/03-data-model.md#2-model-types)). With them are token sequences, and what `D` computes on each.

| Model | Shape | Profile | Residual |
|---|---|---|---|
| `tiny-2-servable.d` | 2 layers, `d_model` 64, 4 heads of 16, 2 KV heads, `d_ff` 192, vocabulary 512 | `servable` | INT20 |
| `tiny-2-minimal.d` | as above, other weights | `minimal` | INT20 |
| `tiny-4-servable.d` | 4 layers, `d_model` 256, 8 heads of 32, 2 KV heads, `d_ff` 384, vocabulary 512 | `servable` | INT16 |

The 16-bit residual is legal, since `residual_bits` lies in 2 to 31, and it catches an engine that assumes INT20.

## How they are made

[`generate.py`](generate.py) is the independent oracle. It computes `D`'s forward pass in Python's integers, written from [RFC-0005](../../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics) 'Integer semantics' as RFC-0022 (the saturating residual) and RFC-0023 (`κ`) amend it. It takes RFC-0008's tables from [`../tables/generate.py`](../tables/generate.py) and BLAKE3 from [`../hash/generate.py`](../hash/generate.py), and shares no code with `crates/`, `verifier/` or `python/`.

- **Weights.** SplitMix64 draws them, from integers only, so every platform draws the same: each entry is the sum of four values in `[−31, 31]`. Output channels take one to four shift classes, and the two rows of a RoPE pair share theirs.
- **Constants.** Each site's shift is the least that keeps its outputs on these sequences inside the output's range. A few sites are pushed past it on purpose, so that the sequences reach every clamp `D` has:
  - an INT8 requantization;
  - the Silu table;
  - the residual's saturation, at INT20 and at INT16 (RFC-0022);
  - an exponential of 0.

  The oracle fails unless each push clamps. `κ` follows RFC-0023, and RMSNorm's `e` is `⌈d · ε · 4^{10}⌉`.
- **Checks before writing.** The encoder must write the two models of [`../model/`](../model/README.md) byte for byte. The constants it chose, applied afresh, must give the same values.

`python3 spec/vectors/models/generate.py` rewrites every file here byte for byte, in about 7 seconds. `cargo xtask gen-vectors --check` runs it on a copy and fails when a file differs.

## `models.json`

| Field | Meaning |
|---|---|
| `schema`, `version` | `0x0101`, `1`: the artifacts' schema |
| `models[].name`, `file`, `bytes`, `digest` | the artifact, its size, and its envelope digest: BLAKE3 `derive_key("tessera/v1/envelope")` over the header and payload, the file's last 32 bytes |
| `models[].profile`, `shape` | the model's profile and `ArchSpec` sizes |
| `models[].events` | how often the sequences reach each clamp, over all sequences: `requant_clamped`, `silu_clamped`, `residual_saturated`, `exp_zero` |
| `models[].sequences[].tokens` | a sequence, at positions 0 to `n − 1` |
| `models[].sequences[].residuals` | the residual stream before the first layer and after each layer, `[n_layers + 1][n][d_model]` |
| `models[].sequences[].logits` | `ℓ = W_lm y`, `[n][vocab]` |

## Who checks them

- `crates/tessera-compiler` (`tests/lower.rs`) lowers each model and runs the graph on its exact interpreter. It checks every value against its analyzed range, and the logits bit for bit against `logits`, as does its token-by-token reference.
- `crates/tessera-e2e` opens each artifact under the `IntegerModel` limits, checks it re-encodes to its bytes, and checks that the compiler validates, lowers and selects gadgets for every sequence.
- `verifier/crates/tv-codec` (`tests/models.rs`) opens each envelope under the `IntegerModel` limits with the test kit's BLAKE3 and the index's digest.
- `python/` (`tests/test_conformance_models.py`) decodes each artifact with `tessera-quant`'s reader and encodes it back to its bytes.
- To come: the CPU engine (#82) reproduces `residuals` and `logits`, the PyTorch reference (#89) the same, and #91 compares the two on every committed tensor.
