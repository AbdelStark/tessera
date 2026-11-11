# MVP benchmark: Tessera beside FlashProve and SASH (#743)

Status: first numbers, 2026-11-11. Development runs on rented hardware, measured by Tessera's own scripts ([`scripts/prove-gpu.sh`](../../../scripts/prove-gpu.sh), `/usr/bin/time`), including the observatory's own workload, and by the observatory's harness itself: a development run of the adapter on the same workload, accepted ([below](#the-adapter-through-the-observatorys-harness)). Read the caveats before any figure.

## Headline

Tessera proves the observatory's own workload: Llama-3.1-8B, a 1,024-token prompt and its 64-token exact-greedy continuation, on one GPU, accepted by the independent verifier with the public tokens pinned (and `REJ-TOKENS` for one changed token). Per token per Gbit it takes 313 ms, about 4.2 times SASH's 75.4 ms and about 4,600 times FlashProve's 68.2 µs, and its proof is 62.5 kB, a tenth of SASH's 640 kB. Its overhead against vLLM on the same GPU at the same shape is 28,100× (SASH: 15,760× against serving on its H200). It needs a host with 2 TB of RAM: the proof peaked at 1.79 TiB. The observatory's own harness reproduces the result: on the same A100 with its profile's 15 prover CPUs it measured 8 h 57 min to prove (461 ms per token per Gbit, 41,500× vLLM), 177 s to verify, and the verdict `accept`.

| System | Model | Shape | Prove time per token per Gbit | Proof size per token per Gbit | Overhead vs serving | Evidence |
|---|---|---|---|---|---|---|
| FlashProve | Llama 4 Maverick, 17B active | observatory workload | 68.2 µs | 223 B | 61× | external, inferenceproofs.ai v0.1, marked not yet checked |
| SASH | Llama 3.1 8B | 1,024 + 64 | 75.4 ms | 640 kB | 15,760× | external, inferenceproofs.ai v0.1, marked not yet checked |
| **Tessera, observatory harness** | Llama 3.1 8B (integer model D, W8) | **1,024 + 64** | **461 ms** | **62.5 kB** | **41,500×** (against vLLM on the same A100) | harness-measured, n = 1, `iproofs run --hosted` development run at ba61755, one A100 80 GB, the profile's 15 prover CPUs, [run.json](../../../bench/inference-proofs/tessera/runs/2026-11-11-l31-8b-c1024-hosted-a100/run.json) |
| Tessera | Llama 3.1 8B (integer model D, W8) | 1,024 + 64 | 313 ms | 62.5 kB | 28,100× (against vLLM on the same A100) | measured, n = 1, development run at 6d16160, one A100 80 GB, all 128 CPUs |
| Tessera | Llama 3.1 8B (integer model D, W8) | 64 + 64 | 443 ms | 571 kB | 10,700× (85,800× against serving at 1,024 + 64) | measured, n = 1, development run at 699d3d8, one H200 NVL |
| Tessera | Llama 3.1 8B (integer model D, W8) | 8 + 8 | 2.82 s | 4.38 MB | 57,800× (546,000× against serving at 1,024 + 64) | measured, n = 1, development run at 405d552 |
| Tessera | Llama 3.2 1B (integer model D, W8) | 8 + 8 | 6.92 s | 23.3 MB | 73,000× (939,000× against serving at 1,024 + 64) | measured, n = 1, development run at 405d552 |

Units as the observatory defines them ([README](README.md#units)): per token covered (prompt plus continuation) and per Gbit of active weights, 64.24 Gbit for 8B and 11.99 Gbit for 1B. The overhead is the prover's GPU-seconds per token over vLLM 0.32.0's GPU-seconds per token covered at batch 1, both on the same idle H200 NVL, first at the proof's own shape, then at the workload's 1,024 + 64 ([serving-8b-h200-64x64.json](serving-8b-h200-64x64.json), [serving-8b-h200-8x8.json](serving-8b-h200-8x8.json), [serving-8b-h200-c1024.json](serving-8b-h200-c1024.json), [serving-1b-h200-8x8.json](serving-1b-h200-8x8.json), [serving-1b-h200-c1024.json](serving-1b-h200-c1024.json); median of 8 requests after 2 warm-ups, `gpu_memory_utilization` 0.9). Serving 8B costs 2.66 ms per token at 64 + 64 and 3.14 ms at 8 + 8, and 1B 1.14 ms at 8 + 8; at 1,024 + 64, 0.332 ms and 0.088 ms. The earlier H100 baselines (`serving-*-h100.json`) ran under host contention and are kept for the record only.

## Caveats

1. **Shapes.** Tessera's first row covers the workload's 1,088 positions, as SASH's does; its other rows cover 128 and 16. Tessera's cost is mostly fixed at these lengths: the table side (10.1 G table rows at 8B, [commitment report](commitment-report-8b.json)) does not depend on the request length, so its per-token figures fall as the request grows (8 times the positions cost 1.26 times the time: 2,902 s to 3,646 s). These rows are not a like-for-like comparison and must not be quoted as one; the per-token figures at 1,088 positions would be lower again, if it fit.
2. **Harness, but a development run.** The observatory times calls from outside the container, binds CPUs and samples NVML. Its row comes from that harness, but as a development install of the adapter into the provider's `nvidia/cuda` image, so the record is `local/`, `nonconforming` (an A100 80 GB, not the profile's H200) and never release-eligible; a release run needs the pinned adapter image (RFC-0014 B1–B7), which is not built yet. The other Tessera rows are Tessera's own measurements.
3. **Different hardware.** Tessera's full-workload proof ran on one A100 80 GB PCIe of a rented 6-GPU host (128 CPUs, a 2.07 TB container), and its other rows on one H200 NVL (82 CPUs, a 558 GiB container); each overhead divides by vLLM on the same GPU. SASH and FlashProve ran on the observatory's H200 (SXM), with its own serving rows. An A100 is slower than an H200, so the full-workload row's time is not flattered by the hardware.
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

## Llama-3.1-8B at the observatory's workload, 1,024 + 64 (2026-11-11)

The workload's checkpoint at its pinned revision, quantized here (C4 calibration, 128 windows, migration strength 1/2), compiled with `--params ni-100 --max-log2-rows 32`, and one request: 1,024 prompt tokens (BOS and the fixed public text) and D's 64-token exact-greedy continuation from the CUDA integer reference (7,475 s: it recomputes every position at every step). `tessera prove --backend gpu --memory-budget 11000` at 6d16160 on one A100 80 GB PCIe. The budget admits the resident route: the physical planner found no plan under 1,850 GiB in 58 minutes, and the resident plan charges 10.9 TB, against the 1.79 TiB the proof used.

| Phase | Wall time |
|---|---|
| Weight recommit and memory plan | 1 min 48 s |
| Witness, 32 layers at 1,088 positions | 14 min 2 s |
| P1 advice commitment, 21,582 rows (GPU) | 21 min 9 s |
| P2 leaf stage: lookup and table trees | **4 h 6 min 59 s** |
| P3 reduction stages 1 to 3 | 34 min 12 s |
| P4 openings | 26 min 16 s |
| P5 column answers | 20 min 41 s |
| **Prove, total** | **6 h 5 min 7 s** (21,907 s), peak host RSS 1.79 TiB |
| Verify (`tessera-verify verify --tokens`, 1,088 public tokens) | 2 min 58 s, 8.1 GiB, **Accept** ([verdict](verdict-8b-a100-c1024.json)) |
| Verify with the last continuation token changed | **Reject `REJ-TOKENS`**, exit 10 ([verdict](verdict-8b-a100-c1024-wrong.json)) |
| Proof | 4,370,880,874 bytes |

Serving baseline on the same A100: vLLM 0.32.0, batch 1, 0.778 s per request, 0.715 ms per token covered ([serving-8b-a100-c1024.json](serving-8b-a100-c1024.json)). P2 is two thirds of the proof; it is the next target.

## Llama-3.1-8B at 64 + 64 positions (2026-11-10)

The same D, claim list and weights; a 64-token prompt (BOS and the first 63 tokens of the fixed public text) and D's 64-token exact-greedy continuation (generated by the CUDA integer reference in 274 s). `tessera prove --backend gpu --memory-budget 540` at 699d3d8, within the container's 558 GiB.

| Phase | Wall time |
|---|---|
| Weight recommit | 1 min 6 s |
| Memory plan (resident 5.9 TB refused; physical plan of 461 GiB, P2 its peak, found in 19 s) and witness | 1 min 29 s |
| P1 advice commitment, 17,007 rows | 3 min 59 s |
| P2 leaf stage | 26 min 43 s |
| P3 reduction stages | 12 min 32 s |
| P4 openings | 10 min 21 s |
| P5 column answers | 4 min 31 s |
| **Prove, total** | **1 h 0 min 46 s** (3,646 s), peak host RSS 441 GiB |
| Verify (`tessera-verify verify --tokens`) | 2 min 22 s, 8.8 GiB, **Accept** ([verdict](verdict-8b-h200-128.json)) |
| Proof | 4,699,361,834 bytes |

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

## Memory at the observatory's workload

The rented container sees the host's 1 TB but is allowed 558 GiB (`/sys/fs/cgroup/memory.max`, 599,568,416,768 bytes); earlier runs' 900 GiB budgets overstated it. An attempt at 1,024 + 64 with a budget large enough to admit the resident plan (9,000 GiB) computed the witness in 8 minutes and committed 21,584 advice rows, then ran its leaf stage for 3 h 29 min without finishing and was killed by the container's OOM killer at 584 GB (`memory.events` oom_kill 1).

At 1,024 + 64 positions, the prover's resident plan is 10.8 TB: 34.7 G tensor, 17.3 G lookup, 10.1 G table and 90.5 G advice cells at 64 bytes each. A 38-minute physical checkpoint search found no plan under the 900 GiB budget, and the proof stopped with `ERR-RES-MEMORY` before the witness. At 16 positions the same model's resident plan is 5.8 TB and its physical plan 369 GB, so the advice matrix (71.3 G cells at 16 positions, mostly the table side's multiplicities) dominates at both lengths.

## The adapter through the observatory's harness

### Accepted at 1,024 + 64 on one A100 (2026-11-11)

`iproofs run --system tessera --workload l31-8b-c1024 --profile prover-h200-hosted --hosted --gpu 0 --allow-nonconforming` at inference-proofs 7d40d85, the adapter at ba61755 installed into the provider's `nvidia/cuda` image on the rented A100 host (128 CPUs, a 2.07 TB container), with `TESSERA_MEMORY_GIB=11000` in the container's environment so that the prover admits the resident route. The harness bound the prover to CPUs 0–14 and the verifier to CPUs 15–22. Record: [run.json](../../../bench/inference-proofs/tessera/runs/2026-11-11-l31-8b-c1024-hosted-a100/run.json), [manifest](../../../bench/inference-proofs/tessera/runs/2026-11-11-l31-8b-c1024-hosted-a100/manifest.json); its warnings are the single repetition and the profile mismatch (GPU model, memory, driver).

| Harness metric | Value |
|---|---|
| commit (quantize, compile, commit weights) | 832.7 s, 27.5 GB host peak |
| prove (greedy continuation, then `tessera prove --backend gpu`) | 32,249 s (8 h 57 min), 1.96 TB host peak, GPU 62.8 GB peak, 1.2 % mean GPU utilization |
| verify | 176.5 s (176.7 s on one CPU), 9.25 GB, **accept** |
| proof | 4.37 GB, 4.02 MB per token covered |
| per token per Gbit | 461 ms prove, 62.5 kB proof, 2.53 ms verify |
| GPU-s per token | 29.6, against vLLM's 0.715 ms per token on the same A100: 41,500× |
| verifier download | 11,750 bytes |

The prover's time is 1.47 times the 128-CPU run's (21,907 s): the harness's 15 CPUs slow its host-bound phases, which is where the GPU sits idle (point 2 below).

### Stopped at memory admission in a 558 GiB container (2026-11-10)

`iproofs run --system tessera --workload l31-8b-c1024 --profile prover-h200-hosted --hosted --gpu 0 --allow-nonconforming` at inference-proofs 7d40d85, inside the rented container (a development install of the adapter at 36aae28 into the provider's `nvidia/cuda` image, so the record is `local/`, never release-eligible, and `nonconforming`: an H200 NVL, not the profile's SXM). The harness's self-test passed. Record: [bench/inference-proofs/tessera/runs/](../../../bench/inference-proofs/tessera/runs/2026-11-10-l31-8b-c1024-hosted-2/run.json).

| Step | Harness clock | Outcome |
|---|---|---|
| info, setup | 0.1 s | passed |
| commit: `tessera-quant compile` of the workload's pinned checkpoint on CUDA (C4 calibration, 128 windows), `tessera compile --params ni-100`, `tessera commit-weights` | 749.4 s | passed: commitment `dc66ed3a…`, prover files 8.03 GB, verifier files 11.8 kB |
| prove: D's greedy continuation of the workload's 1,024-token prompt, then `tessera prove --backend gpu` on the profile's 15 prover CPUs | 7,386.3 s | **failed `ERR-RES-MEMORY`**: no physical plan fits the container's 523 GiB default budget |

So the harness runs Tessera's adapter end to end up to the prover's memory admission, and records the refusal. On a 2 TB host the same prover completes the workload ([above](#llama-31-8b-at-the-observatorys-workload-1024--64-2026-11-11)), but only with a budget that admits the resident route: a harness run there needs the physical planner to admit what the resident route uses (1.79 TiB), or the adapter to pass that budget.

## What to do next, ranked by this profile

A stack-sampling profile of the 1B proof (`gdb` launching the prover, since the container forbids attaching; one sample every 20 s) shows most of the remaining time on one host thread.

1. **P2 at the workload** (4 h 7 min of 6 h 5 min): the leaf stage's lookup and table trees over 18.5 G lookups. **The fixed table side** (#729, #745) sets the advice matrix's size and the floor of every phase. Restricting tables to the workload's ranges or a structured table evaluation (#691) shrinks every phase at once.
2. **P2's single-threaded host work**: between short bursts on every core (level regeneration and round messages), one thread prepares the window grids and allocates, fills and clears multi-gigabyte `SecretBuf` tables.
3. **P4 and P5's row pipeline** (13 min at 8B): one thread regenerates advice rows and copies them to the device from pageable memory; overlapping regeneration, pinned uploads and the GPU would bound them by the device.
4. **P1's row generation** (3 min at 8B): `visit_rows` feeds the GPU encoder from one thread.
5. **Memory at 1,088 positions** (#728, #739): the proof peaks at 1.79 TiB, and the planner's resident charge (10.9 TB) overstates it six times while its physical search admits nothing under 1,850 GiB. Token groups and windows (RFC-0054, RFC-0055), or a smaller physical floor, would bring the workload to an ordinary 1 TB GPU host.
6. **A release harness run** (#741): build the pinned adapter image (RFC-0014 B1–B7) and run it on the profile's H200; the development run above is accepted but `local/` and `nonconforming`.
