# MVP benchmark: Tessera beside FlashProve and SASH (#743)

Status: first numbers, 2026-11-10. Development runs on rented hardware, measured by Tessera's own scripts ([`scripts/prove-gpu.sh`](../../../scripts/prove-gpu.sh), `/usr/bin/time`), not yet by the observatory's harness. Read the caveats before any figure.

## Headline

Tessera proves a real Llama-3.1-8B request on one H200, and the independent verifier accepts it. At the shape it can run today, a 16-position request, it is about 70 times slower than SASH per token per Gbit and about 77,000 times slower than FlashProve. It cannot yet run the observatory's 1,088-position workload on one 1 TB host.

| System | Model | Shape | Prove time per token per Gbit | Proof size per token per Gbit | Overhead vs serving | Evidence |
|---|---|---|---|---|---|---|
| FlashProve | Llama 4 Maverick, 17B active | observatory workload | 68.2 µs | 223 B | 61× | external, inferenceproofs.ai v0.1, marked not yet checked |
| SASH | Llama 3.1 8B | 1,024 + 64 | 75.4 ms | 640 kB | 15,760× | external, inferenceproofs.ai v0.1, marked not yet checked |
| **Tessera** | Llama 3.1 8B (integer model D, W8) | **8 + 8** | **5.26 s** | **4.38 MB** | **about 638,000×** | measured, n = 1, development run |
| Tessera | Llama 3.2 1B (integer model D, W8) | 8 + 8 | 17.1 s | 23.4 MB | about 577,000× | measured, n = 1, development run, before the GPU P4 |

Units as the observatory defines them ([README](README.md#units)): per token covered (prompt plus continuation) and per Gbit of active weights, 64.24 Gbit for 8B and 11.99 Gbit for 1B. The overhead is the prover's GPU-seconds per token over vLLM's GPU-seconds per token covered at batch 1 on the 1,024 + 64 shape ([serving-8b-h100.json](serving-8b-h100.json), [serving-1b-h100.json](serving-1b-h100.json)).

## Caveats

1. **Different shapes.** Tessera's rows cover 16 positions; SASH's and the serving baseline's cover 1,088. Tessera's cost at 16 positions is mostly fixed: the table side (10.1 G table rows at 8B, [commitment report](commitment-report-8b.json)) does not depend on the request length, so its per-token figures fall as the request grows. These rows are not a like-for-like comparison and must not be quoted as one.
2. **Not the harness.** The observatory times calls from outside the container, binds CPUs and samples NVML. Tessera's adapter ([`bench/inference-proofs/tessera`](../../../bench/inference-proofs/tessera)) passes `iproofs validate --all`, but no harness run exists: the harness runs real systems only from their pinned image, and Tessera's image is not built yet.
3. **Different hardware.** Tessera ran on an H200 NVL (141 GB) with 82 host CPUs and 1 TB of RAM; the serving baseline on an H100 SXM under host contention; SASH and FlashProve on the observatory's H200.
4. **What is proved.** Tessera proves its integer model D, quantized from the BF16 checkpoint by `tessera-quant` (servable profile, affine scales, migration strength 1/2), not the BF16 model; D's fidelity is measured separately. `tessera-protocol/1` at `ni-100/1`, no zero knowledge. The verifier pins the public prompt and continuation (`--tokens`).
5. **n = 1.** One run per row, no repetitions.

## Llama-3.1-8B on one H200 (2026-11-10)

[unsloth/Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/tree/4699cc75b550f9c6f3173fb80f4703b62d946aa5), the observatory's `l31-8b-c1024` checkpoint. One request: an 8-token prompt (BOS and the first 7 tokens of the pipeline's fixed public text) and D's 8-token exact-greedy continuation. `tessera prove --backend gpu --memory-budget 900` at commit 09172b9.

| Phase | Wall time |
|---|---|
| Weight recommit | 10 min 51 s |
| Memory plan (resident 5.8 TB refused; physical plan of 369 GB found in 22 s) and witness | 1 min 3 s |
| P1 advice commitment, 17,004 rows (GPU encode and hash) | 3 min 0 s |
| P2 leaf stage: lookup and table trees (GPU level rounds) | 21 min 35 s |
| P3 reduction stages 1 to 3 | 29 min 25 s |
| P4 openings (GPU sampled symbols) | 9 min 27 s |
| P5 column answers (GPU re-encode) | 14 min 50 s |
| **Prove, total** | **1 h 30 min 11 s** (5,411 s), peak host RSS 386 GiB |
| Verify (`tessera-verify verify --tokens`) | 2 min 15 s, 8.4 GiB, **Accept** ([verdict](verdict-8b-h200.json)) |
| Verify with one continuation token changed | **Reject `REJ-TOKENS`**, exit 10 ([verdict](verdict-8b-h200-wrong-token.json)) |
| Proof | 4,498,007,074 bytes |

## The observatory's workload does not fit yet

At 1,024 + 64 positions, the prover's resident plan is 10.8 TB: 34.7 G tensor, 17.3 G lookup, 10.1 G table and 90.5 G advice cells at 64 bytes each. A 38-minute physical checkpoint search found no plan under the 900 GiB budget, and the proof stopped with `ERR-RES-MEMORY` before the witness. At 16 positions the same model's resident plan is 5.8 TB and its physical plan 369 GB, so the advice matrix (71.3 G cells at 16 positions, mostly the table side's multiplicities) dominates at both lengths.

## What to do next, ranked by this profile

1. **The fixed table side** (#729, #745): it sets the advice matrix's size, most of P2 and the floor of every phase. Restricting tables to the workload's ranges or a structured table evaluation (#691) shrinks every phase at once.
2. **P3 stage 1** (27 min at 8B): the reduction sumchecks still run on the CPU (#733).
3. **P5** (15 min): now slower than P4; it re-encodes every row on the GPU for 193 columns and can reuse P4's pass.
4. **Memory at 1,088 positions** (#728, #739): token groups and windows (RFC-0054, RFC-0055), or a smaller physical floor, before the observatory's workload can run on one host.
5. **A harness run** (#741): build the adapter image (RFC-0014 B1–B7) and run `iproofs run --hosted`.
