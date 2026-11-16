# P4 device combination and the upload regression

Scope: the `mvp/e2e` continuation from `beeb662`, RFC-0011 opening step 4 and
RFC-0018 P4. No proof format, transcript schedule, parameter set or verifier
acceptance rule changes.

## Regression and fix

The handoff bisected tiny-model GPU self-check failures to `75cb7aa`'s pinned,
double-buffered uploads. Removing only that staging path made the tiny proofs
pass again. This change removes the staging API and its host-memory charge,
keeping the producer-thread advice pipeline. The exact fault inside the pinned
transfer path has not been identified.

P4 combines the already-uploaded advice block before encoding overwrites it.
The GPU accumulates `sum_r c[g][r] * A[r][i]` in canonical Goldilocks extension
pairs. Mask combinations and both sampled checks remain on the host. P5 continues
to select columns without accumulating combinations.

## H100 runs

Lucid `h100-sxm-8-0`, one H100 SXM 80 GB, 32 allocated CPUs (Xeon Platinum
8480CL), 400 GiB scheduler memory, `--memory-budget 300`. Same existing Llama
3.2 1B integer-model artifacts, 8 prompt and 8 continuation tokens, protocol/1,
ni-100/1. This is a small-shape engineering run, not the observatory's 1024 + 64
workload and not a zero-knowledge proof.

Historical baseline at beeb662: 1735.26 s; job 6433 with staging removed and
P4 device combination: 1455.42 s, independent Accept. Its P4 time was
153.893 s against 410.740 s; P3 was 210.792 s against 213.574 s. These single
runs changed two implementation choices and do not isolate the combine kernel's
speedup. The baseline still contained the tiny-model staging regression.

Final hardened-tree job 6444 proved in 1483.66 s, with P4 at 153.526 s and
P3 at 220.977 s; peak host RSS was 237349892 KiB (226.35 GiB), and the proof
was 4478799994 bytes. All seven commitment tests, both tiny-model CPU/GPU byte
comparisons (848014 and 916638 bytes), and GPU clippy passed before this run.
The first verification command passed the binary TokenFile to `--tokens`, which
requires decimal text: it returned `Inconclusive(ERR-USAGE)`, exit 64. Job 6458
verified the existing proof with the corrected public-token file and statement
and weight-root pins: **Accept**, 154.92 s, 8746752 KiB peak RSS.
[Run manifest](records/1b-h100-p4.json), [verdict](verdict-1b-h100-p4.json).

The run manifest records base `beeb662` because the measured tree was uncommitted.
Its full patch SHA-256 is `79798ce94fc866c60f78271f051ec4bd8d366e63a4114e0d6057ea030ad6e45f`; the patch is retained on Lucid at
`runs/p4-final-6444/source.patch`.

## Review and tests

- **Field and bounds (RFC-0002, INV-FIELD-5):** `Device::combine_rows` checks
  products before CUDA launch; row/accumulator lengths and coefficient shapes
  are validated. Each thread owns one column of each group, and every add/mul
  uses the existing canonical field operations. The H100 kernel test compares
  two accumulated blocks with a plain host sum at widths 33, 1000, 2048 and
  4097 and group counts 3, 8, 9 and 17, including field boundaries.
- **Trust boundary:** malformed row, coefficient and accumulator shapes,
  multiplication overflow and a mismatched download buffer are explicitly
  rejected in `combined_rows_reject_bad_shapes_before_launch`.
- **Transcript and commitments (INV-FS-1, RFC-0011):** no message or challenge
  moves. `OpeningPass::finish` collects sums before checking the sampled merged
  leaf and every combined row. A host combining sink equals the scalar path
  across block sizes 1, 2, 3 and 64; dropping any of three claim groups or the
  proximity group returns `AdviceOpeningError::SelfCheck`.
- **Adversarial controls:** disabling the combined-row self-check makes the
  dropped-group test fail; dropping the sink's accumulated sums makes its honest
  control fail. Both mutants were killed, and the restored control passed.
- **Secrets (OBS-6):** downloaded symbols and sums live in `SecretBuf`; persistent
  device sums remain owned until successful clearing, including copy/clear errors.
  Short-block buffers attempt all clears after failure. The returning symbol Vec
  is allocated only after every word has been validated, so errors leave no partial
  uncleared Vec. No secret enters diagnostics.
- **Memory (RFC-0018):** coefficient word copies fit the existing flush allowance
  for raw combination coefficients; the downloaded accumulator is smaller than
  the existing publication-copy allowance and is dropped before publication.
  Device sums add `16 * groups * k` bytes; no per-block host row copy is added.
- **Secret-independent execution (INV-ARCH-2):** launch shape, group chunks and
  all addresses depend on public sizes/indices. The generated sm_90 SASS uses
  selects/predication for field corrections; branches traverse public columns,
  groups and rows. Full disassembly retained on Lucid in `lucid/p4-combine.sass`.
- **Verifier independence (SP-6):** no verifier edits. The verifier workspace's
  ordinary and test-parameter suites passed, with clippy and formatting.

Local PCS/prover suites and clippy passed. Repository lint, isolation,
test-feature refusal, mask-rank, dependency policy (deny/vet) and all regenerated
conformance vectors passed.
The complete serialized prover workspace, including doctests, passed, followed
by workspace clippy, formatting and lint (1786 passing tests; 6 ignored). The x86 branchless gate passed all
15 field operations. All 12 feature-gated public CPU API tests passed, as did
the explicit CLI subprocess test that accepts an honest proof and rejects
tampering. Log directory:
`/tmp/tessera-e2e-validation` on the development host.

The GPU equivalence script now stops at a failed prove command and retains its
full error log; filtering progress lines no longer hides the failing command.
The performance skill records actual-hardware equivalence as a transfer-change
requirement (continuous-learning, INV-FIELD-5).
