# Llama-3.1-8B calibration and scale preflight

This is preparation evidence for [#633](https://github.com/AbdelStark/tessera/issues/633)
and the full fidelity work in [#93](https://github.com/AbdelStark/tessera/issues/93).
It does not establish a fidelity gate, a best migration strength, a GPU proof
backend or a certifying model proof. The real-proof obligations in
[#617](https://github.com/AbdelStark/tessera/issues/617) and
[#629](https://github.com/AbdelStark/tessera/issues/629) remain.

## Source and calibration

The source is `NousResearch/Meta-Llama-3.1-8B` at immutable revision
`1f47e50cdbe801ad8a5174156ec3a0655108fb9f`, described in the
[real-model ramp guide](real-model-ramp.md). All four BF16 shards match the
[pinned LFS checksums](checkpoints.sha256): 291 source tensors and 8,030,261,248
parameters. The checkpoint is already untied. Its 32 layers have width 4,096,
MLP width 14,336, 32 query heads, 8 KV heads and a vocabulary of 128,256.

The existing deterministic BF16 capture completed the full RFC-0005 dataset:
512 calibration sequences of 2,048 tokens, seed 0, and 64 reserved held-out
sequences. The data is C4 `en` validation at
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, in the specified eight-file order.
The selection digest is
`8c7f3d086b03487c1f4503b9e1967470c70b90f4407b589a86900246e824fc52`.
No held-out forward pass is part of the capture.

The [public calibration report](../perf/v0.2/calibration-llama31-8b-mps.json)
contains architecture, point shapes, dataset identity, digests and environment;
it contains no activation maxima. Its SHA-256 is
`93382568df77ce3546fed03f88e9377df58383beedebe4efc9a459f1bebaa318`.

The private safetensors artifact has 386 named points and 2,695,168 float64
channels, occupying 21,593,544 bytes. An independent metadata/value audit checked
all expected shapes, contiguous byte ranges, finite nonnegative maxima and at
least one positive channel at every point, without printing values. Its embedded
selection identity matches the report. Both files were mode 0600.

| Statistics identity | Value |
| --- | --- |
| RFC-0005 domain-separated BLAKE3 | `432dbac32234d087655e013698e0f955be3188090848ed1090d645838a0198f8` |
| SHA-256 of private file | `606b6d02bfd5b553e9ed05763819d4998fe7223c33bc2fb79bfe48e433a5b721` |

## Environment and measurement scope

Apple M1 Max, 64 GiB, MPS, batch size 1, deterministic algorithms. The report
records Python 3.13.18, torch 2.14.1, transformers 5.20.1, safetensors 0.8.0,
NumPy 2.5.4 and blake3 1.0.11. The capture's wall time was 35,710.743 seconds.
Reported peak process RSS was 2,272,624,640 bytes; sampled MPS allocation and
driver maxima were 16,870,040,064 and 19,437,060,096 bytes. These are separate
measurements; the sampled values are not a guaranteed whole-run device peak.
Other owned validation/model jobs were active, so this is not a throughput
qualification.

The capture uses the existing `tessera_quant.capture.calibrate` and
`write_calibration` functions. Checkpoint configuration and safetensors are read
through the normal importer; no pickle or remote model code is used. The
[ramp guide](real-model-ramp.md) gives the pinned download, preparation and
calibration procedure. Statistics stay outside the checkout; only the existing
`tessera.calibration/1` public report is published here.

## Scale preflight

All 16 actual full-statistics candidates were attempted through the unchanged
`transformed_checkpoint`, `choose` and `_admissible_alphas` paths. A candidate
was released before constructing the next one; no held-out reference was run.

| Profile | Scale mode | 1/2 | 3/5 | 7/10 | 4/5 |
| --- | --- | --- | --- | --- | --- |
| Servable | pot | infeasible | infeasible | infeasible | infeasible |
| Servable | affine | admissible | admissible | admissible | infeasible |
| Minimal | pot | admissible | admissible | admissible | admissible |
| Minimal | affine | admissible | admissible | admissible | admissible |

The five refusals are typed `QuantizationInfeasibleError` / `ERR-QUANT-CALIB`:
derived scale/table infeasibility, without private values or exception text.
No unexpected error was suppressed. Every admitted candidate contains 226 INT8
weight tensors and 8,029,995,008 entries. RMSNorm gains are folded into consumers;
the source checkpoint's 65 norm vectors account for the 266,240-entry difference.

The [complete receipt](qualification-8b/preflight-results.json) records every
attempt's outcome and time, the original statistics/report identity and every
quantizer module's SHA-256. Those module hashes were rechecked at the end.
The [launch manifest](qualification-8b/preflight-launch.json) pins numeric source
at main `e78f2508a9294e57d203d4e3400589b82d5d0eac` and the original runner hash.
Elapsed time was 1,574.199 seconds and process peak RSS was 27,896,037,376 bytes
(Darwin `ru_maxrss`). Concurrent validation/model jobs make these observations,
rather than an isolated performance comparison.

Servable's default power-of-two configuration has no admissible strength for
these actual statistics. The explicit affine diagnostic below does not change
that default, select the minimum-KL strength, or qualify Minimal fidelity.

## Fixed diagnostic artifact

The explicitly fixed Servable/affine `alpha=1/2` candidate exported successfully:
8,032,307,955 bytes, mode 0600, with all 226 matrices / 8,029,995,008 INT8
entries compared to the original candidate after mmap readback. Architecture,
complete quantization metadata and provenance also agree. The
[export receipt](qualification-8b/export-receipt.json) records the rechecked
checkpoint BLAKE3/SHA-256, original full C4 selection/statistics, numeric source
hashes and exact export-runner hash.

| Artifact identity | Value |
| --- | --- |
| SHA-256 | `364c633550d773e4c14d8be98c5a81fd859a2cd9a417a07d5c391d155286f1ca` |
| RFC-0003 envelope digest | `e0408c73164ae09647726775462985302b8668b797ab94aac6ce282f8ce5ab59` |

Export and full readback took 256.602 seconds, with process peak RSS
29,738,893,312 bytes. The separate Rust `tessera inspect --json` returned exit 0
in 24.699 seconds, reporting peak child RSS 16,070,983,680 bytes. Its
[public geometry](qualification-8b/rust-inspect.json) agrees with the Python
receipt on envelope digest/length, architecture/profile, every matrix's
dimensions and all checkpoint/selection/statistics identities. The
[Rust invocation receipt](qualification-8b/rust-inspect-receipt.json) pins the
CLI binary hash. This checks the Rust artifact decoder; it is not an independent
proof-verifier verdict.

A subsequent real `tessera compile` at `ni-100`, `max_log2_rows=32` and
`query_block=512` completed with exit 0 in 267.279 seconds and peak child RSS
16,067,952,640 bytes. It wrote `claims.tcl`, `plan.tep` and
`commitment-report.json`. The [compile receipt](qualification-8b/compile-receipt.json)
pins all three output hashes, the unchanged D hash and the pre-existing release
CLI's binary hash. Compilation/codec behavior is unchanged by the advice-row
adapter change in PR #632; this binary is not credited with that new adapter.
The generated report also contains source-scenario cost projections; those are
not measured proving/GPU performance. Compilation alone does not exercise the
runtime leaf-layout guard, allocate a proof's resident state or produce a proof.

No candidate weights or activation maxima are published. This artifact is a
fixed-strength diagnostic: all-admissible-strength held-out selection, the
RFC-0005 fidelity gates, full real-model memory/soundness accounting and a
certifying-parameter proof with independent acceptance remain outstanding.

## Reproduction

Use the pinned checkpoint, tokenizer and eight C4 files from the
[ramp guide](real-model-ramp.md), and its full calibration command, before this
sequence. Set `TESSERA_PYTHON` to the Python executable of the recorded environment,
`TESSERA_CHECKPOINT_DIR` to the untied 8B checkpoint, `TESSERA_CALIBRATION_DIR`
to the private capture directory and `TESSERA_C4_DIR` to the directory containing
`en/`. Run from the checkout root, without Python's `-O` option. Both output
directories below must be new. Keep model/statistics/output files out of Git.

```sh
PYTHONPATH=python "$TESSERA_PYTHON" docs/models/qualification-8b/verify_calibration.py \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR"
PYTHONPATH=python "$TESSERA_PYTHON" docs/models/qualification-8b/preflight.py \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR" \
  --output /tmp/tessera-8b-preflight-replay
PYTHONPATH=python "$TESSERA_PYTHON" docs/models/qualification-8b/export_fixed.py \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR" \
  --c4 "$TESSERA_C4_DIR" \
  --preflight /tmp/tessera-8b-preflight-replay/preflight-results.json \
  --output /tmp/tessera-8b-export-replay
target/debug/tessera inspect /tmp/tessera-8b-export-replay/candidate-alpha-1of2.d --json
target/release/tessera compile \
  --model /tmp/tessera-8b-export-replay/candidate-alpha-1of2.d --params ni-100 \
  --max-log2-rows 32 --query-block 512 --out-dir /tmp/tessera-8b-compiled-replay
PYTHONPATH=python "$TESSERA_PYTHON" docs/models/qualification-8b/check_export_guards.py \
  --calibration "$TESSERA_CALIBRATION_DIR" \
  --preflight /tmp/tessera-8b-preflight-replay/preflight-results.json \
  --output /tmp/tessera-8b-export-guard-replay
```

The independent [calibration audit](qualification-8b/verify_calibration.py) uses
stdlib `struct` and BLAKE3, with no torch/safetensors decoder or production
`points()` function. It exclusively creates `verification.json`; preserve an
existing receipt before repeating it. The
[checked receipt](qualification-8b/calibration-verification.json) agrees byte for
byte between the original audit and this portable replay against the same input.

The [preflight replay](qualification-8b/preflight.py) keeps the measured runner's
numeric calls and order, adding portable path arguments. Its script hash therefore
differs from the original local runner. Both record every imported Python module's
hash; freeze those files while a run is active. Only derived
`QuantizationInfeasibleError` is an omittable outcome. Input/resource failures
stop the run with a redacted error type. A partial results file is not completion:
the final receipt must contain all four profile/mode groups, all 16 strengths and
the unchanged-source check.

The [fixed exporter](qualification-8b/export_fixed.py) requires that complete
receipt, matching quantizer source and original full-statistics digest. It checks
all four pinned source LFS checksums, re-draws the original 512+64 windows and
checks their selection digest before constructing Servable/affine `alpha=1/2`.
It streams the private artifact, verifies the envelope and complete metadata,
and compares every INT8 entry in bounded chunks. Checkpoint, calibration and
source digests are rechecked at the end. No held-out forward or KL score is run.
Six corrupt receipt cases (source, missing group, duplicate group, duplicate
strength, inconsistent outcomes and report digest) were rejected before model
construction, with no artifact written. A valid-receipt control reached the
model-construction boundary; disabling each targeted guard in turn made its
exact-rule assertion fail (6/6 runtime guard mutations caught). The
[guard-check receipt](qualification-8b/export-guard-checks.json) pins the tested
export-runner and [test-runner](qualification-8b/check_export_guards.py) hashes.
These test qualification metadata checks, not proof acceptance. Every fault starts
from the otherwise valid complete receipt, so a different invalid field cannot
conceal a missing guard.

These scripts are qualification evidence for #633. The production `quantize`
pipeline still performs all-admissible-strength held-out selection; this explicit
fixed export does not replace that stage or establish RFC-0005 fidelity.
