# Ordinary row adapter: verification and adversarial review

Scope: [#630](https://github.com/AbdelStark/tessera/issues/630), implementing the
existing RFC-0011/0034/0047/0050 layout through one clearing row. The planner,
parameter sets, transcript, schemas and independent verifier are unchanged.
This is a diff-level review, not a new soundness argument or a real-model proof.

## Evidence

- The original scalar scatter passed the row differential test before replacement.
  The allocation regression was observed red on that implementation: 16,408 bytes
  exceeded the 256-byte field row plus 1,024-byte metadata allowance. It also used
  different row addresses. The streaming implementation passes both conditions,
  release checks and an allocation-count bound for 64 and 4,096 rows.
- [Source-row tests](../../../tessera-pcs/tests/source_rows.rs) exhaust all ranks
  0–3, axis lengths 1–5 and row splits, including padded source shapes and empty
  sources, against the preserved scalar `locate`. Score tests cover small groups,
  blocks and every key split. Slot tests retain the public zero-row schedule.
  Metadata-only cases cover 32-bit axes, the largest valid 2^63-cell tensor and
  empty sources whose trailing strides need not be addressable. The 32-bit-only
  overflow assertion is not executed on this 64-bit host.
- The compiler's [frozen oracle test](../../../tessera-compiler/tests/advice_layout.rs)
  derives expected indices from `spec/vectors/advice/layout.json`'s independent
  bit permutations, without calling `locate` to predict them.
- Prover [advice tests](../../src/advice.rs) compare complete emitted rows at
  signed boundaries, preserve row order, reject wrong source lengths and stop
  at the first sink failure. The metadata-allocation error retains
  `ERR-RES-MEMORY`. Existing `model_openings` tests additionally cover missing
  roles, source failures, public padding and complete row schedules.
- The final implementation passed
  `public_cpu_api_proofs_verify_in_ni_and_fresh_live_sessions` with the independent
  verifier's explicitly non-certifying test parameters. A separately built
  verifier-workspace CLI also passed the ignored
  `public_cli_binary_accepts_cpu_proof_and_rejects_tampering` test. There is no
  conditional skip in these selected acceptance runs.

The seeded public CPU fixture (`MaskRng::Seeded(17)`, test features only) produced
identical bytes with the original main adapter and the final streaming adapter.
Both also passed the fixture's internal comparison across 1 and 2 threads:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Proof TCE | 900,574 | `b22e602d8d69b774e855d156fef7f30252e79c83f94be6b4d4069d0ff2b0bf27` |
| Claim-list TCE | 1,985 | `89fafa076920df4212afeef4c9587ab00851d77b5a1bb9c761462314706eea7f` |
| Execution-plan TCE | 2,351 | `b69a9c0f19b37fead6143a5047daf8b2fd713a466de37bd0513e6ac03caf4362` |

The original source is main `5ccc3d8d50f1ee907cd7e8db5967cc15976ba46a`.
The final code hashes are recorded by the [mutation report](verification/mutations.json).
[Raw mutation logs](verification/mutation-logs.tar.gz) contain all 23 original,
faulted and restored runs; their hashes match the report.

## Fault injection

All **19/19** deliberately incorrect implementations compiled and failed their
runtime tests. Both original controls and both restored controls passed. The
runner rejects a compilation failure, timeout or surviving fault as evidence.

| Faults | Required observation |
| --- | --- |
| Dense axis order; score query shift; score key high bits; natural stride; source index | Scalar/frozen row-index or row-value mismatch |
| Source rank; source-axis bound; tensor row offset; inclusive row end | Exact typed error expectations fail |
| Metadata reservation forced to overflow | Honest source-row tests fail with the allocation refusal |
| Source padding; trailing row padding; omitted row zeroing | Wrong padding or iterator length |
| Reversed row order; changed field sign | Ordered row or signed-field differential fails |
| Swallowed sink error | Required immediate error fails |
| Leaked row storage; allocation per row; whole-column allocation | Release, allocation-count or peak bound fails |

## Preconditions and scoped soundness review

| Surface | Enforcement and result |
| --- | --- |
| Placement validity | `Placement` has private fields and is built by validated layout constructors. `row_indices` checks row subtraction/range, rank and every source-axis capacity before yielding an index. Pass. |
| Integer/index arithmetic | Local indices are widened to `u128`; valid layouts have at most 2^32 rows and `log2_k ≤ 31`. Real source strides use checked products. An empty source uses zero strides and always yields padding. Every yielded coordinate is below its source length, bounding its flat offset. Pass. |
| Canonical field values | The same `Goldilocks::from_i64` conversion is retained. Boundary-row comparison and the field-sign fault test pin it. No field arithmetic or integer range argument changes. Pass. |
| Padding and public schedule | Only domain-length padding yields `None`; causal/slot values still come from the witness. `Placement::nonzero` alone chooses emitted rows. The full row is reset before reuse, including the unvisited trailing columns. Pass. |
| Secrets and lifetime | Values stay in the borrowed witness and one `SecretBuf` field row. Axis metadata contains only public lengths, strides and bit positions. No new secret accessor, value-dependent branch, log or error context. Reuse/release is measured; unchanged `SecretBuf` drop clears the entire allocation (OBS-6). Freed memory is not inspected. Pass. |
| Encoding, transcript, commitments and masks | No change to TCE, labels, row order, masks, challenges, opening checks or PCS parameters. Full seeded proof bytes agree; independent NI/Live and CLI/tamper acceptance passed. Pass within the unchanged protocol. |
| Verifier independence and robustness | No verifier file or dependency changes. Isolation passes; source errors remain typed and reservation failure maps to memory. The verifier's unchanged deadline test has a contention issue tracked separately below. Pass for this diff. |
| New proof-system assumptions | No change to sumcheck, GKR, LogUp, gadgets, masking or their soundness bounds. These retain their existing obligations; this review does not discharge real-model soundness or formal zero knowledge. Unchanged. |

## Validation limits and retained failures

Two ordinary verifier-suite runs failed the existing deadline-observer assertion
under concurrent load: `Claims` was the last observed check instead of
`Statement`, after the expected `Inconclusive(Time)`. The same default-feature
binary passed the exact test 3/3 with the identified owned competing jobs paused
and resumed. The `test-params` driver also passed it 3/3 in a separate isolated
replay (binary SHA-256 `7a908521ca2fd4b575e988b3291ad2b2981da4ee71ab62562626553543d8d5cd`). The rest of both verifier suites was run with that one case
partitioned out; it is not treated as an omitted test. Production limits and the
fixture remain unchanged. [#631](https://github.com/AbdelStark/tessera/issues/631)
records the failure, binary hash and deterministic-clock follow-up.

An initial full prover run passed its test binaries, then rustdoc failed with
`E0460` while sibling feature builds refreshed its dependencies. Source is now
frozen and Cargo checks sharing a target directory are serialized. The sequential
rerun passed both doctests, including the public CPU API example; final workspace
and CI checks remain required before merge.

The [measurement record](measurements/README.md) retains conflicting timing
results and confirms repeated allocation counts. No stable throughput, complete
prover-memory bound, certifying real-model proof or fidelity result is claimed.
[#629](https://github.com/AbdelStark/tessera/issues/629) and
[#617](https://github.com/AbdelStark/tessera/issues/617) remain open.
