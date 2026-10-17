# RFC-0005: Integer Model D: W8A8 Quantization, Calibration, and Fidelity Evaluation

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.2

## Summary

The Python front-end `tessera-quant` turns a Llama-3.1 checkpoint into the integer model `D`, a network defined entirely by integer operations with exact rounding rules. `D` uses INT8 weights and activations, power-of-two requantization shifts by default, an INT20 residual stream, an integer RMSNorm, RoPE with 14-bit constants, max-shift softmax over a 2^24-key exponential table, a SiLU table and INT28 logits. Two compilation profiles exist: `servable` (default; every matmul operand is INT8, so exact mode can run on tensor cores) and `minimal` (fewest commitments; bounded mode only). This RFC fixes `D`'s semantics bit for bit, the calibration procedure, the determinism rules, and the fidelity evaluation that gates any change.

## Motivation

The statement certifies `D`, not the floating-point model (source §6, principle 1). `D` must be:

- **exact**: its outputs are a function of integers with no unspecified rounding, so the CPU reference, the PyTorch reference, the GPU witness engine and the circuit agree on every bit (source §7, "Exact mode");
- **accurate**: in bounded mode `Û` prices `D`'s mismatch with the deployment (source §9, problem 4); W8A8 is expected to be accurate enough at 400B scale (source §4, last row of the table);
- **cheap to prove**: requantization placement drives commitment count (source §5, §7 "Typed integer dataflow").

These pull against each other. RISK-2, RISK-3 and RISK-7 in [00 §9](../spec/00-overview.md#9-risks) record where.

## Goals

- A normative integer semantics for every operation of a dense Llama-3.1 block.
- A deterministic pipeline from checkpoint to `IntegerModel` artifact.
- Fidelity measured with documented hygiene and confidence intervals, gating every change to quantization.
- Profiles that make the serving and proving trade-off explicit.

## Non-Goals

- Per-token dynamic activation scales (research, source §9 problem 4).
- Mixture-of-experts layers (NG-1) and non-Llama families.
- Weight-rounding algorithms beyond round-to-nearest (GPTQ, AWQ). They change weight values only and can be added later without touching the circuit.
- Floating-point proofs (approximate sum-check, source §3).

## Proposed Design

### Pipeline

1. **Import.** Load safetensors shards; validate shapes and configuration against `ArchSpec` (`ERR-QUANT-ARCH` on mismatch). Never unpickle.
   - The configuration is read exactly: a decimal such as `rms_norm_eps = 1e-05` becomes the rational it writes. Anything outside the Llama-3.1 family is refused: RoPE scaling other than `llama3`, an activation other than SiLU, biases, tied embeddings, weights other than BF16.
   - Each shard's header is checked before any tensor is read: its size bounded, each tensor's dtype BF16 and shape the architecture's, its byte range its shape's, and the ranges tiling the data to the end of the file.
   - The shards hold exactly the architecture's tensors, each where the index places it, and an index places tensors only in shards of its own directory.
2. **Fold.** Fold each RMSNorm gain into the columns of the weights that consume the normalized activations (Q, K, V after the attention norm; gate, up after the MLP norm; LM head after the final norm). This is exact in real arithmetic.
3. **Calibrate.** Run the BF16 model on the calibration set; record per-channel activation maxima at every quantization point.
4. **Migrate.** Apply SmoothQuant migration with strength `α` to each input of a linear layer; apply QK equalization (scale `W_q` rows by `λ_c` and `W_k` rows by `1/λ_c`, equal within each RoPE pair) so scores are unchanged in real arithmetic.
5. **Quantize.** Choose scales (below), quantize weights by round-to-nearest-even into `[−127, 127]`.
6. **Assemble.** Emit the integer graph, `QuantSpec` and weights as an `IntegerModel` (schema `0x0101`) with `Provenance`.

### Scale policy

Each tensor `t` has a real scale `σ_t` with `real ≈ int · σ_t`.

| Quantity | `pot` mode (default) | `affine` mode |
|---|---|---|
| Activations (INT8) | Per tensor, `σ = 2^e`, `e = ceil(log2(max|a| / 127))` | Per tensor, `σ = max|a| / 127` |
| Weights (INT8) | Per output channel, `σ_c = 2^{e_c}`, rounded up so no weight clips | Per output channel, `σ_c = max|w_c| / 127` |
| Residual (INT20) | Per model, `σ_res = 2^e`, with one bit of headroom over the calibration maximum | Same |
| Requantization | Shift `r_c = e_out − e_in − e_c` per output channel, grouped into shift classes | 16-bit multiplier `M_c` and shift `r` |

In `pot` mode a requantization is a pure shift. The quantizer limits each tensor to at most 4 consecutive weight-exponent classes (channels below the window are raised to its lowest exponent), and the lookup key is the tuple `(class_c, acc)`, so one tuple table per tensor kind serves every channel while the channel-to-class assignment stays hidden ([RFC-0009](RFC-0009-gadgets.md#requant1), [RFC-0007](RFC-0007-claim-list.md#public-and-hidden-quantization-data)). In `affine` mode the key is `acc · M_c`, which is wide and needs the split gadget ([RFC-0009](RFC-0009-gadgets.md#requantsplit)). The compiler may use `affine` per tensor where fidelity requires it; the commitment report shows the cost.

### Integer semantics

All operations are on integers; `rq(x, r, lo, hi) = clamp(floor((x + 2^{r−1}) / 2^r), lo, hi)` for `r ≥ 1` (round half up, saturate). Shapes refer to one token at position `p`; batching and causal masking are as in the reference model.

| Step | Operation | Range (worst case) |
|---|---|---|
| Embed | `x0 = onehot(token)^T · E` (INT8 row), `res = x0 · 2^{s_e}` | INT20 |
| Attn norm | `s = Σ res_i^2`; `ρ = max{ρ ∈ ℕ : ρ^2 (s + e) ≤ d · 2^{2κ}}`, `κ = 16`, `e` = exact integer form of `d·ε/σ_res^2` rounded up; `y_i = rq(res_i · ρ, r_n, −128, 127)` | `s < 2^54`; key `< 2^37` |
| Q, K, V | `acc = W y` (INT32) | `< 2^29` |
| RoPE | `(a, b) ↦ (a·C − b·S, a·S + b·C)` with `C, S` 14-bit constants from the RoPE table ([RFC-0008](RFC-0008-logup-and-tables.md#table-kinds)); then `q = rq(·, r_q, −128, 127)`, same for `k` | key `< 2^44` |
| V | `v = rq(acc, r_v,c, −128, 127)` | key `< 2^29` |
| Scores | `x_qi = Σ_j q_j k_ij` over `d_head = 128` for `i ≤ p` (GQA: 16 query heads per KV head) | `−2^21 < x < 2^21` |
| Softmax | `c_q = max_i x_qi`; `e_qi = T_exp[c_q − x_qi]`, `T_exp[k] = round(2^σ · exp(−k · σ_q σ_k / √d_head))`, `σ = 16`; `s_q = Σ_i e_qi`; `R_q = floor(2^{2σ} / s_q)` | keys `≤ 2^22`; `s_q < 2^{σ+17}` |
| Attention out | `o = R_q · Σ_i e_qi v_i` | `o ∈ [−2^39, 127 · 2^32]` using `R_q s_q ≤ 2^{2σ}`; `−2^39` is attained (one position, `v = −128`) |
| O projection | `servable`: `o8 = rq(o, r_o, −128, 127)`, `acc = W_O o8`. `minimal`: `acc = W_O o` | `< 2^29` or `< 2^60` |
| Residual | `res += rq(acc, r_res, −2^19, 2^19 − 1)` | INT20 |
| MLP norm | as attention norm | |
| Gate, up | `g = W_gate y`, `u = W_up y` | `< 2^29` |
| SiLU | `a = T_silu[g]` (rescale-absorbing table, INT8 output) | key `< 2^29` |
| Product | `servable`: `u8 = rq(u, r_u, −128, 127)`, `h = rq(a · u8, r_h, −128, 127)`, `acc = W_down h`. `minimal`: `acc = W_down (a · u)` | `< 2^30` or `< 2^58` |
| Residual | `res += rq(acc, r_res, −2^19, 2^19 − 1)` | INT20 |
| Final norm, logits | `ℓ = W_lm y` | `< 2^28` (INT28) |

Every constant (`s_e`, `e`, all shifts, `σ_q σ_k / √d_head` as an exact rational for the table generator) is recorded in `QuantSpec` as an integer or rational ([03 INV-DM-6](../spec/03-data-model.md#6-invariants)).

### Profiles

| Constraint | `servable` | `minimal` |
|---|---|---|
| S1: every matmul operand is INT8 | Required | Not required for O and down projections |
| S2: every accumulator fits INT32 in the worst case | Required | Not required |
| S3: every nonlinearity is a table lookup or an operation in the table above | Required | Required |
| Exact mode | Available | Not available (`ERR-COMPILE-PROFILE` if requested) |
| Witness GEMMs | INT8 tensor cores | Limb-decomposed GEMMs for O and down projections |

`servable` is the default so that one `D` serves both modes ("two modes, one circuit", source §7). The `minimal` profile reproduces the source's commitment-minimal dataflow (137,216 commitments per layer per token at 405B before RISK-2 corrections). The v0.2 commitment report costs both.

`minimal` has one bit of margin at 405B. Its O-projection accumulator `W_O · o` reaches `127 · 2^14 · 2^39 < 2^60`, 61 bits, one below the 62 that RFC-0006's split gadget accepts ([RFC-0006](RFC-0006-compiler-ir.md#gadget-selection)). A model with four times the hidden size, or `σ = 17`, would fail to compile in that profile with `ERR-COMPILE-RANGE`.

### Calibration

- **Default calibration set:** 512 sequences of 2,048 tokens drawn with a fixed seed from a pinned revision of the C4 `en` validation split, identified by dataset name, revision and the BLAKE3 digest of the selected documents.
- **Deployment calibration:** an operator may supply its own set (recommended for bounded mode); it is recorded the same way.
- **Migration strength:** `α` is chosen from `{0.5, 0.6, 0.7, 0.8}` by minimizing mean KL divergence to the BF16 reference on a 64-sequence split held out from the calibration set. The chosen value is recorded.
- **Statistics:** activations are captured from a deterministic BF16 forward pass and reduced in float64 on CPU. The statistics file is an artifact with a digest; everything after it is exactly reproducible on any machine.

### Determinism

- `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds for every RNG from the `--seed` argument.
- Statistics to `IntegerModel` is CPU-only float64 with explicit rounding, hence bit-reproducible.
- Checkpoint to statistics is reproducible on identical hardware and library versions; across hardware it is best-effort. The statistics digest in `Provenance` makes the dependency explicit.
- `Provenance` records: checkpoint shard digests, calibration set identity, statistics digest, `α`, library versions (Python, torch, transformers, safetensors, numpy), and the `tessera-quant` version.

### Fidelity evaluation

`tessera-quant eval` compares `D` (run by the integer reference) with the BF16 reference on data disjoint from calibration:

| Metric | Data | Report |
|---|---|---|
| Perplexity ratio `ppl(D) / ppl(ref)` | WikiText-2 test; 1,024 held-out C4 validation documents | Point estimate and 95% bootstrap CI over documents |
| Teacher-forced top-1 agreement | 512 prompts, 256 reference-generated tokens each | Rate and 95% Wilson CI |
| Mean per-token KL(ref ‖ D) | Same | Mean and 95% bootstrap CI |
| Bounded-mode `Û` per token | Reference-sampled outputs scored by the RFC-0017 predictor | Mean and 95% bootstrap CI |

Hygiene rules: calibration and evaluation documents are disjoint by content hash (checked); no evaluation data informs any choice (including `α`); every report includes seeds, data identities and library versions.

Gates (fail the change if the upper CI bound exceeds the gate):

| Model | Perplexity ratio | Top-1 agreement (lower CI bound) |
|---|---|---|
| Llama-3.1-8B | ≤ 1.05 | ≥ 0.90 |
| Llama-3.1-70B, 405B | ≤ 1.03 | ≥ 0.93 |

If `pot` fails a gate, the compiler switches the worst tensors (ranked by per-tensor KL contribution) to `affine` until the gate passes, and the report records the switched tensors and the added commitments.

### Python packaging

`python/pyproject.toml`: `requires-python = ">=3.11"`; dependencies `torch>=2.4,<3` (deterministic algorithms, `float8` dtypes for FP8 deployment proxies), `safetensors>=0.4` (safe loading), `transformers>=4.45` (Llama-3.1 RoPE scaling and configuration parsing), `numpy>=1.26`, `datasets>=2.20` (pinned revisions). Exact versions are locked in `uv.lock`, which resolves for the supported platforms of [09 §8](../spec/09-release-and-versioning.md#8-supported-platforms). Lint with `ruff`, types with `pyright` in strict mode, tests with `pytest` and `hypothesis`, on Python 3.11 to 3.13.

PyTorch comes from its own index, through exactly one of two extras:

- `cpu`: CPU wheels, for CI and the tiny conformance models;
- `cu130`: CUDA 13.0 wheels on Linux, for calibrating 8B and up.

Both resolve to the same PyTorch release, and a test of the package fails when `uv.lock` holds two. Calibration on CPU and on GPU then runs the same library, so a scale cannot differ between them through a PyTorch upgrade. The package's version is the prover release train's ([09 §1](../spec/09-release-and-versioning.md#1-what-is-versioned)); `cargo xtask lint` fails when it differs from the workspace's.

## Alternatives Considered

1. **Prove floating point directly** (FP8/BF16 with hint-and-verify). Rejected: pays per multiply-accumulate (about 90 constraints per MAC in the zkVM proposal, source §4) and non-associativity removes exactness.
2. **Per-token dynamic activation scales.** Considered for accuracy, especially in bounded mode. Rejected for v1: each scale is a per-row committed scalar needing a max-abs certificate and per-row requantization tables, which multiplies table instances. Listed as research.
3. **Affine scales everywhere.** Considered for accuracy. Rejected as the default because every requantization key becomes `acc · M_c` and needs the split gadget, roughly doubling linear-layer commitments. Kept as a per-tensor fallback.
4. **W4A8 or W4A16.** Rejected: proving cost follows committed activations, not weight bits, and INT4 weights complicate exact INT8 tensor-core serving.
5. **One profile only.** `minimal` only would make exact mode unservable at INT8 speed (RISK-3). `servable` only would forgo the cheapest bounded-mode circuit. Two profiles cost one compiler flag.
6. **Percentile-clipped activation scales.** Considered; may improve resolution. Not the default because saturation of outliers in large models is a known accuracy hazard; available through `--act-scale-percentile` and evaluated by the same gates.

## Drawbacks

- Power-of-two scales lose up to half a bit of resolution per tensor against affine scales.
- Static per-tensor activation scales are less accurate than per-token scales on models with activation outliers.
- 405B calibration needs a multi-GPU BF16 forward pass, which is outside the prover's hardware budget.

## Migration / Rollout

v0.2 delivers the pipeline for 8B and the tiny conformance models; 70B and 405B follow in v0.5 and v1.0 with the same code. Changing any rule in the semantics table changes `D` for every model and is a protocol-affecting change (new claim lists), handled by a superseding RFC.

## Testing Strategy

- Unit tests for `rq` at boundaries: ties (`x = k·2^r + 2^{r−1}`), negative ties, saturation at both ends, `r = 1` and `r = 40`.
- RMSNorm `ρ`: brute-force check of the defining inequality for 10^6 random `s` and edge cases `s = 0`, `s = 2^54 − 1`.
- Integer reference (PyTorch, int64 tensors) against the Rust CPU engine on all conformance models: every intermediate tensor bit-identical ([07 §5](../spec/07-testing-strategy.md#5-differential-and-conformance-tests)).
- Range assertions: the reference engine asserts each tensor's declared range on the 8B nightly run.
- Determinism: two runs of statistics-to-`IntegerModel` give identical digests.
- Fidelity: the gates above, run on every PR that touches `python/tessera_quant/quantize` or `calib`, on 8B with a reduced set (128 documents) and in full nightly.
- Data hygiene: disjointness test on content hashes.

## Open Questions

- **OQ-0005-1.** Does calibration on the deployment's own traffic reduce bounded-mode `Û` materially against the generic C4 set? Owner: @AbdelStark. Resolve in the v0.4 fidelity study ([RFC-0017](RFC-0017-scoring.md)); if the reduction exceeds 20%, deployment calibration becomes the documented default for bounded mode.

## References

- Source §4 (W8A8 accuracy at 400B), §6 principles 1, 3, 4, §7 ("Integer model D for Llama-3.1-405B", "Typed integer dataflow"), §9 problem 4, §10.
- Xiao et al., "SmoothQuant", ICML 2023.
- "Power-of-two INT8 scales for deterministic inference", 2026 (source §11).
- Grattafiori et al., "The Llama 3 Herd of Models", 2024 (architecture and RoPE scaling).
