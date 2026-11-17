# H100 full-workload benchmark

Status: the full 1B workload is independently accepted; the 8B proof retry and larger fidelity diagnostic remain in progress, 2026-11-17. The completed smaller diagnostic shows a substantial fidelity gap.

The measured source is the merge commit
[`326f1c3a46edbd528caba2c658be8b17660f3b25`](https://github.com/AbdelStark/tessera/commit/326f1c3a46edbd528caba2c658be8b17660f3b25).
It includes the accepted [P4 device combination](p4-device-combine.md),
[P3 window arithmetic](p3-window-arithmetic.md), and the allocation-contract fixes.
The earlier 1B 8 + 8 H100 proof remains a separate measurement; it does not establish performance at 1,024 + 64.

## Completed 1B workload

Job 6518 completed one 1,024 + 64 proof on one H100 SXM, using 32 prover CPUs
from a 64-CPU, 1,900-GiB Slurm allocation. It reuses the prepared integer D,
compiled plan and weight state. The independent verifier
[accepted](verdict-1b-h100-c1024.json) with the statement, weight root and public
tokens pinned. Changing the last continuation token returned
[`REJ-TOKENS`, exit 10](verdict-1b-h100-c1024-wrong-token.json).

| Measurement | 1B, 1,024 + 64 |
|---|---:|
| Native prove wall time | 5,688.45 s |
| Independent verify wall time | 141.30 s |
| Proof size | 4,414,731,722 bytes |
| Peak prover host RSS | 734,852,516 KiB (700.81 GiB) |
| Recorded device peak | 64,039,550,976 bytes (59.64 GiB) |
| Peak verifier RSS | 8,621,312 KiB |
| Prove time per covered token | 5.22835 s |
| Prove time per covered token per Gbit | 436.157 ms |
| Proof bytes per covered token per Gbit | 338,496 bytes |
| vLLM median request time | 0.180919 s |
| Native proof / serving time | 31,441.9× |

The denominator is 1,088 covered positions and 1,498,415,104 active parameters
at 8 bits, or 11.987320832 Gbit, including the materialized output head. Native
proving excludes the separately measured 2,347.70 s input generation. The
[serving baseline](serving-1b-h100-c1024.json) uses eight timed requests after two
warm-ups; its range is 0.163821–0.181900 s. The baseline uses the original BF16
checkpoint, while the proof certifies D. Both ran on the same allocated GPU.
These are development observations with one proof sample, not a controlled
comparison against the earlier 8 + 8 runs.

| Prover pass | Seconds | Share of native wall time |
|---|---:|---:|
| P1 | 586.193 | 10.3% |
| P2, leaf stage | 3,435.487 | 60.4% |
| P3, reduction stages | 583.154 | 10.3% |
| P4, openings | 425.750 | 7.5% |
| P5, column answers | 428.467 | 7.5% |

The table sums `tessera_pass_seconds`; it does not add overlapping phase metrics.
This resident execution records no P3b pass. Passes total 5,459.05 s; the
remaining 229.40 s includes work outside those pass scopes. The leaf stage
remains the largest measured cost. Public [observations](h100-full-workload.csv)
and [pass totals](h100-full-workload-phases.csv) retain the numeric values;
private run manifests and model artifacts stay in the local archive.

## Measurement protocol

- **Workload:** one 1,024-token prompt and 64 exact-greedy continuation tokens. The 8B run uses the observatory's pinned `l31-8b-c1024` checkpoint and harness-generated prompt. The 1B run uses Tessera's fixed public prompt and its existing committed integer model.
- **Execution:** the merged hybrid CPU/CUDA backend. The integer witness and reduction-window arithmetic still run on the CPU; these measurements do not establish completion of the remaining device-offload roadmap.
- **Proof:** non-interactive `tessera-protocol/1`, `ni-100/1`, no zero knowledge. The independent verifier receives the trusted statement digest, weight root, and public token stream. Changing one continuation token must produce `REJ-TOKENS`.
- **Calibration:** the 8B checkpoint is quantized with the servable profile, affine scales, alpha 1/2, 128 C4 windows of 2,048 tokens and 32 held-out windows. The fixed alpha still runs held-out validation. Calibration is measured separately when the adapter receives the resulting prebuilt D.
- **Fidelity diagnostic:** separately evaluate the fresh D against BF16 on CPU, on a second node. Use 32 C4 documents of at most 256 tokens and 32 C4 prompts of 64 tokens, each followed by 64 BF16-greedy tokens; seed 20261116 and 1,000 bootstrap resamples. The existing evaluator redraws and verifies D's calibration set and excludes its documents. This subset does not replace the full RFC-0005 fidelity gates or the tensor-by-tensor differential checks of #312.
- **8B adapter:** [upstream PR #406](https://github.com/ai-verification-observatory/inference-proofs/pull/406), an unchanged Tessera source wrapped by adapter/1. The successful image build pins `sha256:541c01d49eccb0f8efe366df189825658a31a3730adfaf78e6ad3fba9dfeafaf` from clean adapter commit `a506c80f92904af3517261390c4236cb6581c884`. Apptainer runs the existing hosted harness; `--containall --cleanenv --no-init` exposes the image environment to the harness through `/proc/1/environ`.
- **Timing:** retain the harness's external elapsed time, including exact-greedy generation in `prove`. Keep the prover's run manifest privately, and extract the public benchmark timings for witness, commitment, leaf, reduction, opening, and column answers. These are different timing scopes.
- **Memory:** record the Slurm allocation and observed peak RSS separately from the logical planner budget. The historical 8B run used 11,000 GiB to admit a resident estimate that overcharges reusable storage; that budget is not physical RAM.
- **Serving:** the pinned vLLM 0.32.0 engine through Tessera's existing Python-API baseline script, on the same GPU at batch 1 and the same 1,024 + 64 shape. Use deterministic synthetic prompts and report the request count, warm-ups, range and median. This is not the observatory's OpenAI-server serving protocol. BF16 serving and proving the integer D are different computations; the ratio measures deployment overhead, not identical-model runtime.
- **Evidence:** one completed proof per model initially, with the sample count explicit. Derive per-token and per-Gbit quantities from raw observations. Keep every failed attempt with its exit code.

The H100 SXM node reports driver 580.173.02. The observatory's `prover-h100-hosted` profile pins 610.57.04 and requires 500 GiB free disk; the initial node-local check found about 298 GiB free. These are development runs and cannot be represented as a conforming observatory release or Level 1 admission.

## Completed input preparation

Fresh 8B quantization (6483) completed all 32 held-out windows in 8,384.62 s,
with a peak RSS of 27,238,480 KiB. Its 8,032,307,959-byte model was copied
to durable shared storage and independently rehashed. Model bytes and their
file digests remain in the private artifact archive (OBS-6). This is calibration evidence,
not a fidelity score or proof acceptance.

The first full 8B harness attempt (6517) passed all six measurement self-tests,
then failed while resolving its prompt corpus, before setup, commitment or proving.
The original Gutenberg endpoint returned HTTP 503. The mirror at
`gutenberg.pglaf.org` supplied all eight files with the original pinned hashes;
the unchanged harness then materialized a 1,024-token prompt from the verified cache.
The failed run remains retained, and the retry uses a separate results directory.

The separate [BF16 serving baseline](serving-8b-h100-c1024-6517.json) in allocation 6517 completed successfully:
median 0.436434 s per request, range 0.435963–0.438387 s across eight timed
requests after two warm-ups, at batch 1 and 1,024 + 64. No proof-to-serving
ratio is claimed before a full proof completes.

## 8B resource retry

The second attempt resolved the verified inputs, compiled D and committed its
weights, then was killed during proving in its 1,900-GiB allocation. No proof
was accepted. Native compilation took 173.376 s and weight commitment 920.197 s;
the failed harness process reported 1,892,095,732 KiB peak RSS. The original run,
manifest, prepared tokens and compilation artifacts are retained privately.

The next attempt tests whether allocator retention contributes to that peak.
It keeps the same source, image, D, workload shape and 24 prover CPUs, with
`MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=131072` and
`MALLOC_TRIM_THRESHOLD_=131072`, and requests 1,910 GiB from Slurm. These
[glibc settings](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html)
limit arena count and keep allocation/release thresholds fixed. Their effective
values were checked inside the pinned image. A separate 30-second cgroup sampler
retains anonymous, file and kernel memory, including on failure. This is an
experiment in resource use; it does not establish a memory fix or a speedup.

## Fidelity diagnostic

The original 32-document/32-prompt diagnostic remains running on CPU.
Before observing any scores, a smaller diagnostic was specified: four C4
documents of at most 128 tokens and four 32-token C4 prompts followed by
32 BF16-greedy tokens, seed 20261116 and 1,000 bootstrap resamples. It completed
in 5,596.43 s on eight CPUs with 96 GiB allocated, with 41,253,864 KiB peak RSS.
The frozen evaluator redrew and verified the calibration selection and excluded
its documents. No model setting was changed in response to these scores.

| Metric | Estimate | 95% interval | Scored tokens |
|---|---:|---:|---:|
| Perplexity ratio, D / BF16 | 6.19351 | 4.79357–9.08874 | 508 |
| Teacher-forced top-1 agreement | 42.1875% (54 / 128) | 33.9821–50.8482% | 128 |
| Mean KL(ref ‖ D), nats per token | 2.18717 | 1.51523–3.09050 | 128 |

BF16 perplexity is 15.9772 and D's is 98.9547. The intervals use document
bootstrap for perplexity and KL, and Wilson for agreement. The
[numeric observations](fidelity-8b-h100-small.csv) preserve the unrounded values.
The source is `allenai/c4`, `en`, validation revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, using its eight validation files.
The CPU evaluator used Python 3.12.3, PyTorch 2.14.1+cu130, Transformers 5.20.1,
NumPy 2.5.4, safetensors 0.8.0 and blake3 1.0.11. The complete evaluation
specification and report remain in the private archive.

This small sample cannot qualify the model. It is also far outside
[RFC-0005's targets](../../rfcs/RFC-0005-integer-model.md#fidelity-evaluation)
of at most 1.05 perplexity ratio and at least 90% agreement at the lower
confidence bound. Proof acceptance establishes execution of D; it does not
establish BF16 fidelity. Full qualification and the tensor differential checks
of #312 remain unmet.

## Preparation evidence

The merged GPU prover and the independent verifier built from a clean checkout. The observatory harness checked the exact checkpoint file set, every SHA256 and the aggregate model digest. The refreshed adapter passed 1,553 harness/system tests locally; 16 tests requiring a GPU or real containers were skipped. Its upstream CI passed all six Python 3.12/3.13 shards, the fake end-to-end test, validation and type checks.

The pinned image passed its CPU preflight (6512): adapter metadata, source and adapter commits, setup and schemas. Its GPU preflight (6507) compiled, committed and proved the tiny conformance model with production `ni-100/1`; the independent verifier [accepted with the statement, weight root and public tokens pinned](verdict-h100-image-preflight.json). The vLLM image also imported successfully and allocated a CUDA tensor on the H100. These images contain PyTorch 2.14.1+cu130 (Tessera) and 2.13.0+cu130 (vLLM 0.32.0).

A separate hosted calibration probe (6513) passed `ST-CGROUP` and `ST-WATCH` with the actual Slurm allocation's cgroup mounted read-only into the container. The complete GPU measurement self-test (6531) subsequently passed all six hosted checks: clock, sampler, jitter, cgroup, release watcher and NVML. Sampling was 20.2 Hz with no gaps over 150 ms and p99 lateness 0.229 ms; NVML reported no foreign process on the allocated GPU and supported energy measurement. The full benchmark repeats this check in its own allocation. The launch wrappers normalize PyTorch's UUID to NVML's `GPU-` form; lookup was checked on the allocated device before requeuing the full jobs.

The 1B input generator completed in 2,347.70 s on 16 allocated CPUs (6510), using the existing guarded exact-FP64 CPU reference. Its TokenFile decodes to 1,024 prompt positions and 64 continuation positions, matching the generator's recorded output. This is input preparation, not a completed proof or a serving measurement.

| Attempt | Outcome | Follow-up |
|---|---|---|
| Checkpoint preparation 6475 | Exit 1: the shared-home storage quota was exhausted | Download to a private directory on node-local temporary storage; preparation 6482 completed successfully |
| Initial local harness suite | Eight failures: GNU `timeout` absent on macOS | Install coreutils; all 37 affected tests and the full suite pass |
| 1B token generation 6485 | Exit 1: raw checkpoint has tied embeddings, incompatible with the prepared D | Use the existing prepared checkpoint |
| Image preflight 6505 | Exit 64: the probe used an unsupported `tessera-verify --version` flag, after source-pin, schema and setup checks passed | Record the verifier binary digest and rerun the probe |
| Serving-image conversion 6498 | Slurm `OUT_OF_MEMORY` (`0:125`); compressor exit 137 after exceeding its 32 GiB allocation | Cap compression at 8 GiB and eight workers with Apptainer's documented build option |
| 1B token generation 6491 | Exit 1: H100 NVL driver 570.211.01 does not support the pinned CUDA 13.0 PyTorch runtime | Use the CPU integer reference for input generation; keep proof measurements on the H100 SXM |
| 1B token generation 6494 | Slurm `TIMEOUT` after one hour; no token file produced | Retain the attempt and use the existing guarded exact-FP64 reference, with CUDA generation available in the H100 job |
| 8B harness 6517 | Exit 4 before proving: the pinned prompt corpus returned HTTP 503 | Fetch from the Project Gutenberg mirror, require all eight original SHA256 hashes, and check prompt generation with the unchanged harness before retrying |
| 8B harness 6532 | Slurm `OUT_OF_MEMORY` (`0:125`), harness exit 4: one OOM kill during `prove`, 4 h 34 min 52 s after resolution began | Retain the failed run and prepared inputs; retry the unchanged source/image with explicit allocator settings and 1,910 GiB of physical RAM |

Accepted proof artifacts and raw Criterion samples are retained. Regenerable test-build intermediates were cleared to free shared-home storage. Five older proofs were archived locally and hash-verified. Four redundant cluster copies were removed after a second remote SHA256 check; adjacent archive manifests retain their locations and digests. This freed 17.9 GB from shared-home storage. The latest accepted P3/P4 proofs remain on the cluster.
