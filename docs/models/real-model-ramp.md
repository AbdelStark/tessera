# Reproducing the real-model ramp

This is the preparation path for the 1–10B integrity work, tracked in
[#617](https://github.com/AbdelStark/tessera/issues/617). **No certifying real-model proof
has been generated or independently verified yet.** Synthetic `test-small` proofs certify
nothing. Protocol One is non-hiding; this work establishes no zero-knowledge claim.

## Pinned checkpoints

The following sources have been downloaded and their complete weight files checked against
the pinned Hugging Face LFS SHA-256 values. The pinned model revisions, rather than moving
branch names, also select the configuration and tokenizer files.

| Ramp | Source and revision | Physical source parameters | Preparation |
|---|---|---|---|
| 1B | [NousResearch/Llama-3.2-1B](https://huggingface.co/NousResearch/Llama-3.2-1B/tree/a9745ffc3556f145a830ac0c203509ba860582a3), `a9745ffc3556f145a830ac0c203509ba860582a3` | 1,235,814,400 | Tied embedding/head; explicit conversion below |
| 8B | [NousResearch/Meta-Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/tree/1f47e50cdbe801ad8a5174156ec3a0655108fb9f), `1f47e50cdbe801ad8a5174156ec3a0655108fb9f` | 8,030,261,248 | Already untied; use directly |

The first has 16 layers and width 2048; the second has 32 layers and width 4096.
Both use the supported Llama-3 RoPE and BF16 safetensors format. The 1B source has 146
tensors; materializing its head produces 147 tensors and 1,498,482,688 logical parameters.
That increase counts an explicit copy of the tied embedding, not newly trained parameters.
The 8B source has 291 tensors. No model weights or tokenizer data are redistributed here.

The [full 8B calibration and preflight evidence](llama31-8b-calibration.md)
records the completed 512 × 2048 capture and all 16 existing-policy candidates.
Eleven candidates satisfy the scale/table bounds; this is preparation evidence,
with no completed held-out strength selection, fidelity gate or real-model proof.

From the repository root, choose an artifact directory outside the checkout:

```sh
export TESSERA_REPO="$PWD"
export TESSERA_MODEL_DIR="$HOME/tessera-models"
cd "$TESSERA_REPO/python"
uv sync --locked --extra cpu --python 3.13
uv run --no-sync python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["TESSERA_MODEL_DIR"])
for name, repo, revision in [
    ("llama-3.2-1b", "NousResearch/Llama-3.2-1B", "a9745ffc3556f145a830ac0c203509ba860582a3"),
    ("llama-3.1-8b", "NousResearch/Meta-Llama-3.1-8B", "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"),
]:
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=root / name,
        allow_patterns=["*.json", "*.safetensors", "README.md", "LICENSE*", "NOTICE", "USE_POLICY.md"],
    )
PY
(cd "$TESSERA_MODEL_DIR" && shasum -a 256 -c "$TESSERA_REPO/docs/models/checkpoints.sha256")
```

The [checksum file](checkpoints.sha256) covers the five large weight files. Download all other required files
from the same immutable revision. The normal importer checks their configuration and tensor
geometry; these checks do not replace the source weight checksums.

## Explicit 1B preparation

RFC-0005 still refuses tied source checkpoints. This repository tool performs a separate,
explicit conversion before import:

```sh
cd "$TESSERA_REPO/python"
PYTHONPATH=. uv run --no-sync python tools/materialize_tied_head.py prepare \
  "$TESSERA_MODEL_DIR/llama-3.2-1b" "$TESSERA_MODEL_DIR/llama-3.2-1b-prepared"
PYTHONPATH=. uv run --no-sync python tools/materialize_tied_head.py verify \
  "$TESSERA_MODEL_DIR/llama-3.2-1b" "$TESSERA_MODEL_DIR/llama-3.2-1b-prepared"
```

The source must otherwise pass the existing architecture and BF16 safetensors checks,
with exactly `lm_head.weight` omitted. The converter validates every source shard before
copying data, preserves every existing tensor bit, and writes the explicit head from the
embedding bytes. It changes only the top-level `tie_word_embeddings` boolean; all other
configuration bytes, including exact numeric lexemes, remain unchanged. Duplicate config
members, unsupported architectures, conflicting heads, malformed headers and unsafe shard
paths are refused. No pickle or model code is loaded.

Copies and comparisons use chunks of at most 1 MiB. Output files have mode 0600 in a new
0700 directory; existing destinations and paths within the source are refused. Failed
conversions retain a private `.incomplete` marker and staged files, with no public
`config.json`. Configuration is published last after complete byte and source-identity
checks. A valid `preparation.json` receipt reports public counts, never raw tensor values.
Source files are read only. This Python utility promises neither constant-time execution
nor zeroized allocator buffers, and its receipt is not a cryptographic attestation.

The converted output uses a complete shard index and no single `model.safetensors` file;
otherwise Hugging Face's loader could prefer the incomplete single-file weights over the
index. Verification also works on the earlier single-file conversion used by the fixed
candidate. Identical tensor bits do not imply identical checkpoint-file digests: shard
names, headers and placement are part of `Provenance.checkpoint`. Do not substitute this
directory for an existing D's pinned reference. Retain that original snapshot, or produce
a new D and its correctly bound provenance through the pipeline.

## Actual preparation evidence

On Apple M1 Max/64 GiB, Python 3.13.18 and the locked CPU environment, the actual 1B
conversion writes 2,996,965,376 bytes of tensor payload. The initial run took 16.403 seconds
and reached 371,851,264 bytes peak process RSS; a repeat after the chunk-release correction
took 17.253 seconds and reached 356,679,680 bytes. These include Python/library imports and exclude
OS file-cache memory. It is a single component observation under concurrent model/fuzz
work, not a throughput or whole-pipeline qualification. Live tensor-data comparison
scratch is at most two 1 MiB chunks, beside checked public headers and placement metadata.
A lifetime-counting regression found four simultaneous returned chunks before explicit
release was added; it now bounds the live pair, separately from the read-size guard.
The observed process-RSS difference is not credited as a qualified memory reduction.

The complete earlier conversion and new output both match the same original tensor bits,
including the explicit head. Hugging Face loaded all expected parameters from source and
prepared checkpoints without missing, unexpected or mismatched keys. Two public teacher-
forcing probes produced byte-identical BF16 logits, repeated after the chunk-release fix.
This is finite BF16 equivalence evidence, not integer
fidelity, alpha selection, a full-context benchmark or a proof. An initial probe stopped
at an overly specific test assumption (`missing_keys` is a set in this loader); its
diagnostic was retained, and the corrected check requires all three key sets to be empty.

Adversarial tests cover both sharded and single-file inputs, signed-zero/NaN payload bits,
long decimals, escaped/nested/duplicate keys, each changed tensor/config/tokenizer input,
unsafe existing external shards, malformed ranges/dtypes, bounded reads, source changes,
private permissions, failed writes, native loader discovery and redacted CLI errors.
All 459 Python cases pass locally, including 30 preparation cases and the existing
conformance traces; strict types and formatting pass. The unchanged Exp-core timing gate
ran with only the two own model computations temporarily paused, and both resumed in a
`finally` block. Fuzz workers stayed active. Fifteen meaningful planted source faults
fail assertions, with passing no-op/restored controls: lossy numeric config; global boolean
replacement; skipped whole-config comparison; duplicate config acceptance; omitted byte
equality; skipped original tensors; unbounded copy reads; unbounded comparison reads;
omitted source-identity check; public file modes; ambiguous single-file discovery; leaked
exception payload; unchecked external shard paths; retained comparison chunks; and
opening nonregular configurations. All faults were repeated against the final source,
with passing original/restored controls. The source is restored byte for byte afterward.

## Calibration and integer model

The next pipeline operation is RFC-0005's complete C4 draw: 512 calibration windows and
64 held-out windows of 2048 tokens, seed 0. The dataset revision is
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. Use the same source files and tokenizer for
the entire run; a shorter smoke test does not stand in for this calibration.

For a fresh run, the public CLI entry point is:

```sh
cd "$TESSERA_REPO/python"
uv run --no-sync tessera-quant compile \
  --checkpoint "$TESSERA_MODEL_DIR/llama-3.2-1b-prepared" \
  --calibration "$TESSERA_REPO/docs/perf/examples/calibration-spec-c4.json" \
  --profile servable --scales affine --seed 0 --device cpu \
  --out "$TESSERA_MODEL_DIR/llama-3.2-1b.d" \
  --report "$TESSERA_MODEL_DIR/llama-3.2-1b-calibration.json"
```

Use the untied 8B directory for that ramp. The Python API also supports `device="mps"`
for BF16 calibration/reference work on Metal hosts, and the locked CUDA extra is a
separate installation choice ([Python setup](../../python/README.md#setup)). There is
no Metal proof backend.

**This command is an entry point, not evidence of a completed real-model run.** The
current 1B calibration completed at the full 512-window size. Alpha selection now
preflights every prescribed strength before held-out evaluation and scores only those
that satisfy the unchanged scale/table bounds. Invalid input and scoring errors remain
fatal. On this checkpoint all four `servable`/`pot` candidates are infeasible;
`servable`/`affine` admits 1/2 and 3/5 and refuses 7/10 and 4/5. This construction
check does not measure KL or establish which admissible strength is best.
The proper-provenance diagnostic D uses a fixed admissible affine alpha of 1/2; it is
not selected as best and is not fidelity-qualified. Its original full 64-window held-out
evaluation remains running. Complete 8B calibration remains running too. Do not reuse
their reports with newly sharded checkpoint digests. Preparation and candidate preflight
do not establish RFC-0005's fidelity gates.

## Proving and independent verification

Once a correctly calibrated D exists, the real release commands begin with:

```sh
cd "$TESSERA_REPO"
cargo build -p tessera-cli --release --locked
target/release/tessera compile --model "$TESSERA_MODEL_DIR/model.d" --params ni-100 \
  --max-log2-rows 32 --query-block 512 --out-dir "$TESSERA_MODEL_DIR/compiled-ni100"
target/release/tessera commit-weights --model "$TESSERA_MODEL_DIR/model.d" \
  --plan "$TESSERA_MODEL_DIR/compiled-ni100/plan.tep" --out "$TESSERA_MODEL_DIR/weights.twc"
```

Both operations completed on the actual fixed 1B candidate: ten matrix roots, with weight
commitment taking 679.94 seconds and 7,465,140,224 bytes peak RSS. That commitment is not a
proof. The following one-prompt/one-continuation `prove` attempt returned exit 70 after
1263.743 seconds, wrote no proof, and reported the unchanged RFC-0045 leaf-layout rule:

> a site's real layers read the same tensors over the same axes

The public per-layer Requant selections disagree at one template site. The existing
guard stays in place. [Draft RFC-0076 / PR #620](https://github.com/AbdelStark/tessera/pull/620)
proposes a remedy; its non-author maintainer review and acceptance decision remain
required before adoption. See [#617](https://github.com/AbdelStark/tessera/issues/617) for
the exact public reproduction and remaining implementation/count/proof obligations.
No actual memory refusal was observed in that attempt; do not relabel the leaf failure.

A separate [complete runtime sizing experiment](../../research/experiments/EXP-0030-real-runtime-sizing/README.md)
uses the same public fixed-candidate metadata and the proposed common-family plans
in an isolated scratch adapter. The unchanged CPU planner charges a zero-lookup
base floor of 2.56 TiB at two positions and 3.80 TiB at 2,048 positions (E0); all
nine policy/length cases refuse tested budgets through 1 TiB. Current advice-row,
table-tree and tensor charges therefore require further memory work before this
64 GiB host can follow that CPU plan. These are conservative planner charges,
not observed peak RAM; the actual proof attempt above still stopped at the leaf
guard. This experiment neither adopts RFC-0076 nor discharges a complete
soundness budget, fidelity or proof obligation.

The ordinary advice adapter now uses one reusable clearing field row and public
source-index metadata proportional to tensor rank ([measurement and tests](../../crates/tessera-prover/benches/advice-rows/README.md)).
PCS already consumes streamed rows. This removes the adapter's whole padded
column allocation; source tiles and other prover state still have their own
lifetimes. The conservative planner charges above remain unchanged while
[#629](https://github.com/AbdelStark/tessera/issues/629) tracks the complete memory model.

After that blocker is resolved, proving must use the real token artifact and independent
request/manifest context described by the [exact CLI flow](../testing/exact-cli.md).
The ordinary independently built verifier then needs caller-trusted statement and weight
root pins ([verifier CLI](../verifier/reference-cli.md)); extracting a pin from the proof
does not establish an independent expectation. A complete actual-count soundness budget,
certifying-parameter `Accept`, exact tamper rejection, full fidelity and resource
qualification remain required. The tiny test fixture and prepared BF16 probes satisfy
none of those real-model proof obligations.
