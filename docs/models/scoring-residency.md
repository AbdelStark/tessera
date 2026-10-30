# Candidate construction and BF16 reference residency

[#635](https://github.com/AbdelStark/tessera/issues/635) separates two allocations
in RFC-0005 held-out strength selection. After complete preflight, each candidate
is now assembled before loading its BF16 reference model or first reference
window. Its float64 transformation maps are released first. The reference is
exhausted/closed before constructing another candidate or reconstructing the winner.
One INT8 candidate and one reference window remain live during scoring.

All requested strengths are still preflighted before any held-out forward. The
same admitted strengths, all held-out windows, per-position float64 KL scalars,
final mean order and smallest-strength tie rule determine the winner. Input,
resource, scoring and reconstruction errors remain fatal. No profile/mode
fallback, integer semantic, encoding, proof parameter or acceptance rule changes.

## Actual 8B boundary probe

Inputs are the [pinned full 8B calibration](llama31-8b-calibration.md): 512 × 2048
calibration windows and the original 64 held-out windows, with statistics digest
`432dbac32234d087655e013698e0f955be3188090848ed1090d645838a0198f8` and selection
digest `8c7f3d086b03487c1f4503b9e1967470c70b90f4407b589a86900246e824fc52`.
The diagnostic requests the fixed Servable/affine strength 1/2. It does not score
the other strengths or qualify the full alpha selection.

The [original-order receipt](scoring-residency/baseline.json) records a controlled
abort: after requested-strength preflight and one finite 2048 × 128256 BF16
window, candidate construction was entered with the BF16 model live and
16,073,122,304 MPS tensor bytes allocated. The probe's overlap guard stopped it
before allocating scoring migration scratch.

The [new-order receipt](scoring-residency/candidate.json) records both construction
entries with zero live BF16 models and zero MPS tensor bytes. Migration maps were
dead when the BF16 model loaded. The first full held-out BF16 window was finite.
At the first integer-inference boundary, the probe exported D and matched both
identities of the previous complete fixed candidate:

| Identity | Value |
| --- | --- |
| Bytes | 8,032,307,955 |
| SHA-256 | `364c633550d773e4c14d8be98c5a81fd859a2cd9a417a07d5c391d155286f1ca` |
| Envelope digest | `e0408c73164ae09647726775462985302b8668b797ab94aac6ce282f8ce5ab59` |

It then deliberately stopped before integer inference. All BF16 models and
migration maps were released on both controlled exits. There was no KL score,
selected winner, fidelity verdict or model proof from these probes.

The old control took 233.923 s and reported peak process RSS 28,456,009,728 bytes;
the new probe took 315.289 s and reported 30,035,787,776 bytes. **These are not
comparable before/after time or peak-RSS measurements:** the old probe stops before
the overlapping construction, while the new one completes construction and writes
the artifact. MPS allocation/driver figures are event samples, not a guaranteed
whole-run device peak. The demonstrated change is elimination of a live BF16
model during candidate construction, not a measured whole-process speedup.

Inactive MPS allocator cache pages can remain after closing a reference: the new
probe ended at zero allocated tensor bytes but 19,470,614,528 driver bytes. The
lifetime contract is not a physical bound for the complete scoring job. The long
runner below explicitly discards inactive Metal cache pages between constructions;
it does not change live tensors or numerical rules.

## Checks and reproduction

The resource test tracks weak references to actual BF16 models and transformation
maps, and guards checkpoint reads; the original code fails it with
`BF16 model overlaps candidate scratch`. It also checks release before subsequent
strengths and the winner. Three isolated faults are caught: early reference
loading, retained migration maps and a retained BF16 model. Two controls pass:
unchanged code, and creating a lazy iterator before construction without advancing
it. Thus the check follows live resources, not merely function-call order. The
[mutation receipt](scoring-residency/mutations.json) records each executed source
hash. Isolated mutation runs disable third-party pytest plugin auto-loading to
avoid an unrelated already-imported-plugin warning; the complete 490-case suite
runs with the normal project configuration.

Local checks: all 490 Python cases, strict pyright against the pinned Python 3.13
environment, ruff check/format and repository/PR gates. Existing tests compare
strength selection with the full-trace/eager divergence oracle in both profiles
and modes; they also cover all-infeasible preflight, fatal resource/scoring errors,
ragged weighting, missing/extra windows and byte identity.

The [portable probe](scoring-residency/probe.py) retains the measured runner's
numeric calls and resource guards, adding portable paths and formatting. Its hash
therefore differs from the archived measured runner; both receipts pin the
executed numerical package and original runner. The original package was frozen
from `e5b1bf2cedcdd66c440d1f37d34168a59c395d4a`; the candidate differs only in
`assemble.py`. Keep these source snapshots immutable during each run. Use the
recorded MPS/CPython 3.13 environment and run without `-O`:

```sh
PYTHONPATH=/path/to/frozen/baseline/python "$TESSERA_PYTHON" \
  docs/models/scoring-residency/probe.py --label baseline --output /tmp/old-residency \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR" \
  --c4 "$TESSERA_C4_DIR"
PYTHONPATH=/path/to/frozen/candidate/python "$TESSERA_PYTHON" \
  docs/models/scoring-residency/probe.py --label candidate --output /tmp/new-residency \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR" \
  --c4 "$TESSERA_C4_DIR"
```

After the boundary probe, the [full selection runner](scoring-residency/full_selection.py)
calls production `quantize` with all four strengths and all original 64 × 2048
held-out windows. It records progress and full per-strength aggregate scores,
exports only the actual selected winner, and checks complete artifact readback.
It uses two CPU threads and an MPS memory fraction of 0.5, clearing inactive Metal
cache pages only when no model tensors are allocated. All errors remain fatal.
Freeze the package and runner for this long job; the comparison helper is copied
and hashed at launch. A partial score/progress file is not completion.

```sh
PYTHONPATH=/path/to/frozen/candidate/python "$TESSERA_PYTHON" \
  docs/models/scoring-residency/full_selection.py --output /tmp/full-8b-selection \
  --checkpoint "$TESSERA_CHECKPOINT_DIR" --calibration "$TESSERA_CALIBRATION_DIR" \
  --c4 "$TESSERA_C4_DIR"
```

Both output directories must be new and private. Weights, statistics and raw
tokens/logits remain outside Git. This preparation does not complete [#93](https://github.com/AbdelStark/tessera/issues/93)'s
full fidelity matrix/CI, [#617](https://github.com/AbdelStark/tessera/issues/617)'s
real-model proof, or [#629](https://github.com/AbdelStark/tessera/issues/629)'s
complete bounded CPU resident execution. Their original requirements remain.
