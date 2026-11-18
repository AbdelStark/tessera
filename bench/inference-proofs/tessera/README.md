# Tessera adapter

[Tessera](https://github.com/AbdelStark/tessera/tree/326f1c3a46edbd528caba2c658be8b17660f3b25) proves that a committed integer language model produced a token stream: a sumcheck, GKR and LogUp proof over Goldilocks under a Ligero-style commitment, checked by an independent verifier (`tessera-verify`) that shares no code with the prover. `entrypoint` implements adapter/1 with the standard library only.

1. **Upstream tools and arguments.** The table at the top of `entrypoint`:
   - `commit`: `tessera-quant compile` builds the integer model D from the workload's checkpoint, then `tessera compile --params ni-100 --max-log2-rows 32` and `tessera commit-weights`.
   - `prove`: per request, `python/tools/greedy_tokens.py` computes D's exact-greedy continuation, then `tessera prove --backend gpu` proves it.
   - `verify`: the `statement_pin` example of `tv-cli`, then `tessera-verify verify --tokens`.
2. **Verdicts** ([02-public-api.md §2.4 at the pin](https://github.com/AbdelStark/tessera/blob/326f1c3a46edbd528caba2c658be8b17660f3b25/docs/spec/02-public-api.md)):

   | `tessera-verify` exit | Verdict |
   |---|---|
   | 0 | `accept` |
   | 10, any `REJ-*`, including `REJ-TOKENS` for a proof of other tokens | `reject` |
   | 64, usage | `invalid_call` |
   | anything else | the adapter stops; no verdict is guessed |

3. **Files.**
   - `commitment_id` is the weights root of `tessera commit-weights`. `verify` passes the harness's recorded `commitment_id` to `--weights-root`, never anything `prove` wrote.
   - Verifier files: the claim list `claims.tcl` and the public weight commitment `weights.twr`.
   - Prover files: D (`model.d`), the execution plan and the private weight state `weights.twc`.
4. **`proofs`.** One proof per request, covering the prompt and the continuation. There are no incremental proofs, so there is no steady boundary.
5. **Image** (RFC-0014 B1–B7). The base is CUDA 12.8.1 devel on Ubuntu 24.04, by digest. The fetch stage adds Tessera at the pin and checks its `git archive` digest, the Rust 1.95.0 installer, and C4's validation split at `tessera-quant`'s pinned revision into the Hub's cache layout. It then fetches the Rust crates through `Cargo.lock` and the Python packages, with torch for CUDA 13.0, through `python/uv.lock`. The build stage compiles `tessera` (with the CUDA kernels), `tessera-verify` and `statement_pin` with `--network=none`. `commit` reads C4 from the image with `HF_HUB_OFFLINE=1`. Fixed-alpha calibration still validates all held-out windows; `TESSERA_REFERENCE_DEVICE=cuda` uses the upstream reference's guarded exact products on the GPU.
6. **Environment.** `TESSERA_MEMORY_GIB` sets the prover's `--memory-budget`. It defaults to 15/16 of the host's RAM or the cgroup limit. The earlier A100 run at 1,024 + 64 peaked at 1.79 TiB. It used a 11,000 GiB planner budget because the current resident estimate overcharges reusable storage; this is not a physical-memory limit. The H100 grouped run uses a 9,000-GiB logical budget under a 1,750-GiB hard allocation, with 24 prover CPUs and eight verifier CPUs. Reproductions must record both the planner budget and the enforced host allocation. `TESSERA_BACKEND` is `gpu` (default) or `cpu`, and `TESSERA_THREADS` defaults to every CPU the adapter sees.
7. **Deviations.**
   - The proof is of Tessera's integer model D, not of the BF16 checkpoint. D's fidelity is reported separately.
   - There is no zero knowledge in this configuration (`tessera-protocol/1`).
   - Two production proofs of one request differ in their bytes, because the commitment masks are fresh randomness (`deterministic_proofs: false`).
8. **Measured.** The full 1,024 + 64 H100 adapter workload completed with independent acceptance, checked derived metrics and the official C1 changed-token rejection. The [report and selected records](../../../docs/perf/mvp/h100-full-workload.md#completed-8b-adapter-workload) separate native proof timing from the adapter timing, which includes greedy generation. The run is a single nonconforming development observation; the driver differs from the profile. The separate small fidelity diagnostic shows a substantial gap between D and BF16. This is adapter evidence, not Level 1 admission or model qualification.

`uv run pytest systems/tessera/tests` runs on a laptop. The upstream tools are stand-ins, so none of their output is captured from a run.
