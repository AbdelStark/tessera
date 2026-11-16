# H100 full-workload benchmark

Status: preparation in progress, 2026-11-16. No completed full-workload H100 measurement is claimed on this page yet.

The measured source will be the merge commit
[`326f1c3a46edbd528caba2c658be8b17660f3b25`](https://github.com/AbdelStark/tessera/commit/326f1c3a46edbd528caba2c658be8b17660f3b25).
It includes the accepted [P4 device combination](p4-device-combine.md),
[P3 window arithmetic](p3-window-arithmetic.md), and the allocation-contract fixes.
The earlier 1B 8 + 8 H100 proof remains a separate measurement; it does not establish performance at 1,024 + 64.

## Measurement protocol

- **Workload:** one 1,024-token prompt and 64 exact-greedy continuation tokens. The 8B run uses the observatory's pinned `l31-8b-c1024` checkpoint and harness-generated prompt. The 1B run uses Tessera's fixed public prompt and its existing committed integer model.
- **Proof:** non-interactive `tessera-protocol/1`, `ni-100/1`, no zero knowledge. The independent verifier receives the trusted statement digest, weight root, and public token stream. Changing one continuation token must produce `REJ-TOKENS`.
- **Calibration:** the 8B checkpoint is quantized with the servable profile, affine scales, alpha 1/2, 128 C4 windows of 2,048 tokens and 32 held-out windows. The fixed alpha still runs held-out validation. Calibration is measured separately when the adapter receives the resulting prebuilt D.
- **8B adapter:** [upstream PR #406](https://github.com/ai-verification-observatory/inference-proofs/pull/406), an unchanged Tessera source wrapped by adapter/1. The successful image build pins `sha256:541c01d49eccb0f8efe366df189825658a31a3730adfaf78e6ad3fba9dfeafaf` from clean adapter commit `a506c80f92904af3517261390c4236cb6581c884`. Apptainer runs the existing hosted harness; `--containall --cleanenv --no-init` exposes the image environment to the harness through `/proc/1/environ`.
- **Timing:** retain the harness's external elapsed time, including exact-greedy generation in `prove`. Also retain the prover's own run manifest, with separate witness, commitment, leaf, reduction, opening, and column-answer timings. These are different timing scopes.
- **Memory:** record the Slurm allocation and observed peak RSS separately from the logical planner budget. The historical 8B run used 11,000 GiB to admit a resident estimate that overcharges reusable storage; that budget is not physical RAM.
- **Serving:** the pinned vLLM 0.32.0 engine, on the same GPU at batch 1 and the same 1,024 + 64 shape. Report the request count, warm-ups, range and median. BF16 serving and proving the integer D are different computations; the ratio measures deployment overhead, not identical-model runtime.
- **Evidence:** one completed proof per model initially, with the sample count explicit. Derive per-token and per-Gbit quantities from raw observations. Keep every failed attempt with its exit code.

The H100 SXM node reports driver 580.173.02. The observatory's `prover-h100-hosted` profile pins 610.57.04 and requires 500 GiB free disk; the initial node-local check found about 298 GiB free. These are development runs and cannot be represented as a conforming observatory release or Level 1 admission.

## Preparation evidence

The merged GPU prover and the independent verifier built from a clean checkout. The observatory harness checked the exact checkpoint file set, every SHA256 and the aggregate model digest. The refreshed adapter passed 1,553 harness/system tests locally; 16 tests requiring a GPU or real containers were skipped. Its upstream CI passed all six Python 3.12/3.13 shards, the fake end-to-end test, validation and type checks.

| Attempt | Outcome | Follow-up |
|---|---|---|
| Checkpoint preparation 6475 | Exit 1: the shared-home storage quota was exhausted | Download to a private directory on node-local temporary storage; preparation 6482 completed successfully |
| Initial local harness suite | Eight failures: GNU `timeout` absent on macOS | Install coreutils; all 37 affected tests and the full suite pass |
| 1B token generation 6485 | Exit 1: raw checkpoint has tied embeddings, incompatible with the prepared D | Use the existing prepared checkpoint |
| Serving-image conversion 6498 | Slurm `OUT_OF_MEMORY` (`0:125`); compressor exit 137 after exceeding its 32 GiB allocation | Cap compression at 8 GiB and eight workers with Apptainer's documented build option |
| 1B token generation 6491 | Exit 1: H100 NVL driver 570.211.01 does not support the pinned CUDA 13.0 PyTorch runtime | Use the CPU integer reference for input generation; keep proof measurements on the H100 SXM |

Accepted proof artifacts and raw Criterion samples are retained. Regenerable test-build intermediates were cleared to free shared-home storage. Older proof copies are being archived locally with SHA256 verification before any cluster copy is removed.
