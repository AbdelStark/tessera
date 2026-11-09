# Tessera adapter for the inference-proofs observatory

Draft of `systems/tessera` for [inference-proofs](https://github.com/ai-verification-observatory/inference-proofs) (#741). `entrypoint` implements adapter/1 with the standard library only.

1. **Upstream tools and arguments.** See the table at the top of `entrypoint`: `tessera-quant compile` (or a prebuilt D), `tessera compile --params ni-100 --max-log2-rows 32`, `tessera commit-weights`, `tools/greedy_tokens.py`, `tessera prove --backend gpu`, the `statement_pin` example of `tv-cli` and `tessera-verify verify --tokens`.
2. **Verdicts.** `tessera-verify` exit 0 → `accept`; 10 (any `REJ-*`, including `REJ-TOKENS` for a proof of other tokens) → `reject`; 64 (usage) → `invalid_call`; anything else stops the adapter.
3. **Files.** `commitment_id` is the weights root of `tessera commit-weights`. Verifier files: the claim list `claims.tcl` and the public weight commitment `weights.twr`. Prover files: D (`model.d`), the execution plan and the private weight state `weights.twc`.
4. **`proofs`.** One proof per request, covering the prompt and the continuation; no incremental proofs, so no steady boundary.
5. **Toolchain.** Rust 1.95.0 (`rust-toolchain.toml`), CUDA 12.8 or later for `--backend gpu`, Python 3.13 with `uv` for `tessera-quant`.
6. **CPU binding.** `tessera prove --threads` defaults to every CPU the adapter sees.
7. **Deviations.** The proof is of Tessera's integer model D, not of the BF16 checkpoint; D's fidelity is reported separately. No zero knowledge in this configuration (`tessera-protocol/1`). The Docker image (RFC-0014 B1–B7) is not built yet: runs use the harness's in-process mode on a GPU host.
