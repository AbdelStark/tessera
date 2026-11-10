# MVP benchmark: Tessera beside FlashProve and SASH (#743)

Status: first numbers, 2026-11-10. Development runs on rented hardware, measured by Tessera's own scripts ([`scripts/prove-gpu.sh`](../../../scripts/prove-gpu.sh), `/usr/bin/time`), not yet by the observatory's harness. Read the caveats before any figure.

## Headline

Tessera proves a real Llama-3.1-8B request on one H200, and the independent verifier accepts it. At the shape it can run today, a 16-position request, it is about 37 times slower than SASH per token per Gbit and about 41,000 times slower than FlashProve, after the same day's profiled fixes halved the 8B proof (90 to 48 minutes). It cannot yet run the observatory's 1,088-position workload on one 1 TB host.

| System | Model | Shape | Prove time per token per Gbit | Proof size per token per Gbit | Overhead vs serving | Evidence |
|---|---|---|---|---|---|---|
| FlashProve | Llama 4 Maverick, 17B active | observatory workload | 68.2 µs | 223 B | 61× | external, inferenceproofs.ai v0.1, marked not yet checked |
| SASH | Llama 3.1 8B | 1,024 + 64 | 75.4 ms | 640 kB | 15,760× | external, inferenceproofs.ai v0.1, marked not yet checked |
| **Tessera** | Llama 3.1 8B (integer model D, W8) | **8 + 8** | **2.82 s** | **4.38 MB** | **57,800×** (546,000× against serving at 1,024 + 64) | measured, n = 1, development run at 405d552 |
| Tessera | Llama 3.2 1B (integer model D, W8) | 8 + 8 | 6.92 s | 23.3 MB | 73,000× (939,000× against serving at 1,024 + 64) | measured, n = 1, development run at 405d552 |

Units as the observatory defines them ([README](README.md#units)): per token covered (prompt plus continuation) and per Gbit of active weights, 64.24 Gbit for 8B and 11.99 Gbit for 1B. The overhead is the prover's GPU-seconds per token over vLLM 0.32.0's GPU-seconds per token covered at batch 1, both on the same idle H200 NVL, first at the proof's own 8 + 8 shape, then at the workload's 1,024 + 64 ([serving-8b-h200-8x8.json](serving-8b-h200-8x8.json), [serving-8b-h200-c1024.json](serving-8b-h200-c1024.json), [serving-1b-h200-8x8.json](serving-1b-h200-8x8.json), [serving-1b-h200-c1024.json](serving-1b-h200-c1024.json); median of 8 requests after 2 warm-ups, `gpu_memory_utilization` 0.9). Serving at 8 + 8 costs 3.14 ms per token at 8B and 1.14 ms at 1B; at 1,024 + 64, 0.332 ms and 0.088 ms. The earlier H100 baselines (`serving-*-h100.json`) ran under host contention and are kept for the record only.

## Caveats

1. **Different shapes.** Tessera's rows cover 16 positions; SASH's and the serving baseline's cover 1,088. Tessera's cost at 16 positions is mostly fixed: the table side (10.1 G table rows at 8B, [commitment report](commitment-report-8b.json)) does not depend on the request length, so its per-token figures fall as the request grows. These rows are not a like-for-like comparison and must not be quoted as one.
2. **Not the harness.** The observatory times calls from outside the container, binds CPUs and samples NVML. Tessera's adapter ([`bench/inference-proofs/tessera`](../../../bench/inference-proofs/tessera)) passes `iproofs validate --all`, but no harness run exists: the harness runs real systems only from their pinned image, and Tessera's image is not built yet.
3. **Different hardware.** Tessera and its serving baselines ran on one H200 NVL (141 GB) with 82 host CPUs and 1 TB of RAM; SASH and FlashProve on the observatory's H200 (SXM), with its own serving rows.
4. **What is proved.** Tessera proves its integer model D, quantized from the BF16 checkpoint by `tessera-quant` (servable profile, affine scales, migration strength 1/2), not the BF16 model; D's fidelity is measured separately. `tessera-protocol/1` at `ni-100/1`, no zero knowledge. The verifier pins the public prompt and continuation (`--tokens`).
5. **n = 1.** One run per row, no repetitions.

## Llama-3.1-8B on one H200 (2026-11-10)


[unsloth/Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/tree/4699cc75b550f9c6f3173fb80f4703b62d946aa5), the observatory's `l31-8b-c1024` checkpoint. One request: an 8-token prompt (BOS and the first 7 tokens of the pipeline's fixed public text) and D's 8-token exact-greedy continuation. `tessera prove --backend gpu --memory-budget 900`, first at commit 09172b9, then at 405d552 with the GPU weight encoding (d8b0567) and the parallel stage loads (405d552) that a `gdb` stack profile pointed to.

| Phase | 09172b9 | 405d552 |
|---|---|---|
| Weight recommit | 10 min 51 s | **1 min 20 s** (GPU rate-1/4 encoding) |
| Memory plan (resident 5.8 TB refused; physical plan of 369 GB found in 22 s) and witness | 1 min 3 s | 53 s |
| P1 advice commitment, 17,004 rows (GPU encode and hash) | 3 min 0 s | 2 min 57 s |
| P2 leaf stage: lookup and table trees (GPU level rounds) | 21 min 35 s | 21 min 51 s |
| P3 reduction stages 1 to 3 | 29 min 25 s | **8 min 32 s** (stage 1's resident loads on every core) |
| P4 openings (GPU sampled symbols) | 9 min 27 s | 9 min 18 s |
| P5 column answers (GPU re-encode, GPU weight answers) | 14 min 50 s | **3 min 30 s** |
| **Prove, total** | 1 h 30 min 11 s (5,411 s) | **48 min 22 s** (2,902 s), peak host RSS 386 GiB |
| Verify (`tessera-verify verify --tokens`) | 2 min 15 s, 8.4 GiB, **Accept** ([verdict](verdict-8b-h200.json)) | 2 min 20 s, **Accept** ([verdict](verdict-8b-h200-v2.json)) |
| Verify with one continuation token changed | **Reject `REJ-TOKENS`**, exit 10 ([verdict](verdict-8b-h200-wrong-token.json)) | |
| Proof | 4,498,007,074 bytes | 4,498,009,922 bytes |

The proofs differ in their bytes because each draws fresh masks and challenges; both verify.

## Llama-3.2-1B on one H200 (2026-11-10)

The run of [README.md](README.md#llama-32-1b-on-one-h200-the-first-accepted-real-model-proof-2026-11-10), repeated on an otherwise idle host: the same D, tokens, claim list and weights.

| Phase | 60a6ea5 (shared host, CPU P4) | 09172b9 (GPU P4) | 405d552 (GPU weights, parallel stage loads) |
|---|---|---|---|
| Weight recommit, plan (physical, 190 GB) and witness | 1 min 23 s | 3 min 6 s | 1 min 3 s |
| P1 advice commitment, 8,687 rows | 1 min 37 s | 1 min 38 s | 1 min 34 s |
| P2 leaf stage | 11 min 16 s | 9 min 52 s | 9 min 52 s |
| P3 reduction stages | 5 min 43 s | 5 min 28 s | 3 min 39 s |
| P4 openings | 29 min 10 s | **4 min 14 s** | 4 min 15 s |
| P5 column answers | 3 min 32 s | 3 min 37 s | 1 min 43 s |
| **Prove, total** | 54 min 36 s | 27 min 55 s (1,675 s) | **22 min 7 s** (1,327 s), peak host RSS 176 GiB |
| Verify `--tokens` | 2 min 8 s, Accept | 2 min 9 s, **Accept** ([verdict](verdict-1b-h200-p4.json)) | Accept |
| Proof | 4.48 GB | 4,478,696,586 bytes | |

## The observatory's workload does not fit yet

At 1,024 + 64 positions, the prover's resident plan is 10.8 TB: 34.7 G tensor, 17.3 G lookup, 10.1 G table and 90.5 G advice cells at 64 bytes each. A 38-minute physical checkpoint search found no plan under the 900 GiB budget, and the proof stopped with `ERR-RES-MEMORY` before the witness. At 16 positions the same model's resident plan is 5.8 TB and its physical plan 369 GB, so the advice matrix (71.3 G cells at 16 positions, mostly the table side's multiplicities) dominates at both lengths.

## What to do next, ranked by this profile

A stack-sampling profile of the 1B proof (`gdb` launching the prover, since the container forbids attaching; one sample every 20 s) shows most of the remaining time on one host thread.

1. **The fixed table side** (#729, #745): it sets the advice matrix's size, most of P2 (22 min at 8B) and the floor of every phase. Restricting tables to the workload's ranges or a structured table evaluation (#691) shrinks every phase at once.
2. **P2's single-threaded host work**: between short bursts on every core (level regeneration and round messages), one thread prepares the window grids and allocates, fills and clears multi-gigabyte `SecretBuf` tables.
3. **P4 and P5's row pipeline** (13 min at 8B): one thread regenerates advice rows and copies them to the device from pageable memory; overlapping regeneration, pinned uploads and the GPU would bound them by the device.
4. **P1's row generation** (3 min at 8B): `visit_rows` feeds the GPU encoder from one thread.
5. **Memory at 1,088 positions** (#728, #739): token groups and windows (RFC-0054, RFC-0055), or a smaller physical floor, before the observatory's workload can run on one host.
6. **A harness run** (#741): build the adapter image (RFC-0014 B1–B7) and run `iproofs run --hosted`.
