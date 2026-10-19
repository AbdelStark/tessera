# EXP-0024: Bounded exact multiplicity representations

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0037](../../ideas/IDEA-0037-bounded-multiplicity-columns.md) |
| Evidence level | E0 exact arithmetic/equivalence; E1 CPU allocation measurements only |
| Hardware | Local Apple CPU, identified by the run; no GPU |
| Pre-registered on | 2026-10-20 |

## Hypothesis

Publicly bounded dense-or-sorted weighted updates expose exactly the canonical
multiplicity column while avoiding a dense wide-table allocation when the public
update bound is much smaller than the table domain.

## Success criterion

Validated only if all of the following pass: (1) dense, sorted and row-window
results match an independent map/reference on every small-domain logical row;
(2) wide-domain row access matches the reference at every contributed key and
its in-range neighbours plus seeded windows, with zero table padding; (3) all
out-of-domain, count-overflow/modulus and capacity faults are rejected; (4) equal
public shapes with different synthetic key distributions choose identical modes,
capacities and logical lengths; (5) sorted insertion and sorting allocate no
additional heap beyond their preallocated public capacity, and dropping each arm
retains zero allocations; (6) at logical size `2^32`, with at most 4,096 weighted
updates, measured candidate storage is at most 128 KiB, against the analytic
32 GiB dense baseline. No dense 32 GiB allocation is attempted.

Refuted if any correctness, public-metadata, rejection or memory-bound criterion
fails. Inconclusive if a toolchain/allocator failure prevents adjudication, or
the baseline/reference cannot be reproduced. A failure must not be repaired by
relaxing these criteria; append a dated amendment before any changed experiment.

This experiment does not establish #112's full witness integration or its total
checkpoint-plus-one-working-tile bound. It does not prove timing ZK, test a
production proof, or credit a GPU/time/proof-size saving.

## Method

- One standalone Rust harness, safe code only, with the existing clearing
  `SecretBuf` and allocation-counter library. No root-workspace dependency on it.
- Baseline: dense `u64` counters. Candidates: fixed-capacity weighted updates
  sorted in place, and requested-row-window recounting. Exact arithmetic uses
  checked integers below Goldilocks; logical table padding cannot receive a key.
- Public domains include 1, 2, 3, 128, 256, 768, 65,536, 1,048,576 and `2^32`;
  update counts include 0, 1, 17, 513 and 4,096. Include duplicates, all-equal,
  alternating, boundary and seeded keys, zero-weight updates and weighted padding.
- SplitMix64 seed `0x0112_0024_0050_0001`. Independently accumulate in a standard
  map of synthetic data and hand-check one weighted column before randomized
  comparisons. Small domains are checked exhaustively; wide domains use the
  exact finite support, neighbours and at least 64 seeded windows of width 64.
- Include a row-0 tail equal to `next_power_of_two(N)-N` for a positive public
  lookup count, and table-tree padding zeros. Record exact row-window replay
  visits; no claim of free passes.
- Allocation measurements: one thread, no other local measurements; three
  warm-ups and ten measured iterations. Report min/median/max/CV of bytes and
  allocation counts. Repeat a measurement once if CV exceeds 3%; if still noisy,
  its memory criterion is inconclusive. Wall-clock speed is not a criterion.
- Test both the pinned Rust toolchain and MSRV 1.88.0. Functional faults must fail
  targeted checks with passing control/restoration; their count is scoped to
  this experiment, not production-code or full-proof soundness coverage.

## Reproduction

From the repository root, after the committed frozen plan is verified:

```sh
cargo run --manifest-path research/experiments/EXP-0024-bounded-multiplicities/Cargo.toml --release --locked
cargo +1.88.0 run --manifest-path research/experiments/EXP-0024-bounded-multiplicities/Cargo.toml --release --locked
```

Environment and raw measurements will be recorded with the results. There is no
experiment code yet at pre-registration.

## Results

All finite semantic checks passed under Rust 1.95.0 and MSRV 1.88.0, with identical JSON results on Apple M4 Max / macOS 26.6.2 / arm64. There are 225 synthetic cases and 83,897,106 exact row comparisons (E0). The window arm performed 26,790,330 update visits across the registered fixture matrix (E0), not a production pass count.

| Case | Measured peak bytes | Allocations | Retained after drop | Evidence |
|---|---|---|---|---|
| Dense, 65,536 rows / 4,096 updates | 524,288 | 1 | 0 bytes / 0 allocations | E1 |
| Sorted, same input | 65,536 | 1 | 0 bytes / 0 allocations | E1 |
| Sorted, `2^32` rows / 4,096 updates | 65,536 | 1 | 0 bytes / 0 allocations | E1 |
| Window, 65,536 rows / width 1,024 | 8,192 | 1 | 0 bytes / 0 allocations | E1 |

Every report has three warm-ups and ten identical measured samples: minimum, median and maximum equal the stated values; allocation CV is zero (E1). Insertion and sorting made zero additional heap allocations. The wide dense baseline is **analytic** 34,359,738,368 bytes / 32 GiB (E0); it was not allocated.

Raw data: [pinned](results/pinned.json), [MSRV](results/msrv.json), [environment](results/environment.json), [source faults](results/source-mutations.json). All 14 physical experiment-source faults were assertion-killed, with passing control and restored runs. These are test-sensitivity checks, not production/full-proof coverage.

## Analysis

Threats: synthetic weighted updates are not full model regeneration; allocation
bytes are not RSS, timing ZK or hardware throughput; sparse access does not permit
skipping secret-zero commitment rows; the public update bound must still be
derived and enforced in production. Windowed recounting may multiply replay cost.

## Verdict

`validated` for the frozen bounded counter criteria only. Wide sorted storage is 65,536 bytes, below the registered 128 KiB cap, with identical canonical queries and public metadata. No timing, RSS, GPU or full-source memory claim follows. IDEA-0037 retains zero performance credit; complete #112 integration is next.

## Follow-ups

Integrate complete canonical root/Softmax columns and bounded multiplicities. Derive/enforce the public update bound before allocation, expose only canonical logical row access through clearing buffers, and retain the prohibition on skipping secret-zero commitment rows. Then measure the full witness-source working set independently, including replay inputs, outputs, temporaries, counter state and public metadata. No table may silently omit requests, padding or the final row-0 tail.
An H100 run would settle GPU storage/recount costs, not these exact CPU semantics.
