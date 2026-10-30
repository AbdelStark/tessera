# M1 Max row-visitor measurement, 2026-10-30

The reproducible result is bounded storage and fewer allocations. Timing varied
materially across runs on this busy machine; no stable throughput improvement
is established. All runs are retained, including the unfavorable comparison.

Apple M1 Max, 64 GiB unified memory, macOS 15.6.1, Rust 1.95.0
(`59807616e`, 2026-04-14), release profile, one visitor thread, unlocked clocks.
Criterion 0.5.1: 1 s warmup, 2 s measurement, 10 samples per case.

| Positions | Peak before → after (bytes) | Allocations before → after | Total allocated before → after (bytes) |
| --- | ---: | ---: | ---: |
| 3 | 56,787 → 56,787 | 15,402 → 8,168 | 1,737,462 → 1,677,562 |
| 8 | 65,664 → 65,592 | 29,489 → 10,212 | 2,217,732 → 2,006,924 |
| 16 | 131,200 → 79,680 | 52,065 → 13,531 | 3,058,084 → 2,574,828 |

The original executable reproduced all original allocation figures exactly in
the paired replay. Every after allocation run also matched. The visitor includes
existing tile/source work, which dominates its smallest peaks. The pure adapter
regression separately bounds field storage to 256 bytes plus at most 1,024 bytes
of public metadata on the heap for the two-axis 64-row and 4,096-row fixtures. The original implementation
failed at 16,408 bytes on the smaller fixture. Source storage is outside that
measurement. Success and sink failure both release adapter storage; an allocation
count bound detects allocating a fresh field row even if the allocator reuses
its address.

## Timing variability

Times below are Criterion's point estimates in microseconds, for 3/8/16 positions.
The reported change is Criterion's bootstrapped comparison, not a ratio inferred
from the displayed slope estimates.

| Run | Before (µs) | After (µs) | Criterion change, 3/8/16 |
| --- | --- | --- | --- |
| Initial comparison | 690.12 / 1,196.0 / 2,122.2 | 698.85 / 1,030.7 / 1,725.5 | +3.1% (not significant) / −15.5% / −19.3% |
| Allocation-error fix snapshot, saved baseline | Same original baseline | 882.86 / 1,410.6 / 2,086.4 | +22.4% / +17.5% / −1.6% |
| Back-to-back original/snapshot binaries | 1,072.2 / 2,233.8 / 3,466.1 | 775.28 / 1,062.8 / 1,796.9 | −32.0% / −54.4% / −48.4% |

The first comparison preceded the final allocation-error classification fix;
successful row mapping and field storage were the same. That comparison
then contradicted the initial timing trend, so the unchanged original executable and
snapshot executable were measured back to back. The large spread even for the
original executable makes a general speedup claim unwarranted. This change is
retained for the measured memory bound; end-to-end proof throughput remains
unqualified.

Raw confidence intervals and outliers:
[original baseline](before-criterion.txt),
[initial after](after-criterion.txt),
[final against saved baseline](after-final-unpaired-criterion.txt),
[paired original](paired-before-criterion.txt),
[paired final](paired-after-criterion.txt).
[All samples](all-samples.csv) preserve all 150 iteration-count/time pairs;
[samples.csv](samples.csv) retains the initial 60 pairs separately. Allocation
logs are [before](before-allocation.txt) and [after](after-allocation.txt).

## Provenance and conditions

The original source was main `5ccc3d8d50f1ee907cd7e8db5967cc15976ba46a`:

- `advice.rs`: `c506f328c2c2ff7fd8a22174e2714980b096bdd5c3c1ead14e94e720e124a9d3`
- `layout.rs`: `6bdf64763cb20fa2e84bbb3ad3040dbadf60d318ae997750d33aa0aafc972013`

Measured timing snapshot source:

- `advice.rs`: `12aef4c357ec0ffcb35eac52f60b95586041a999fdfd3c7eb43a4ad50139d1de`
- `layout.rs`: `8037f5b24021338edf1c3744e8ee60af7eea40ac41867fd0472074594bd0d288`

Original/snapshot executable SHA-256 values in the paired run:

- Original: `22a6d6021cc228e4cc794214e104985ff6b00165a9e36a664b3171eca3a72ebf`
- Snapshot: `e3104885ac4d9cf3539c535c2e9ccbc05ba479e6290c053583544997cf4212c2`

The benchmark source and fixture were identical. An initial smoke-only invocation
without `--bench` is excluded. The two owned model jobs were paused for each
measurement and resumed in `finally`. After runs additionally paused the
identified check process groups started for this change, including their
children. All were resumed; the paired pause lasted 28.365 s. No fuzz campaign
or unrelated job was paused. Other system activity, thermal state, unlocked
clocks and scheduling on a heterogeneous CPU remain limitations.

To reproduce the original baseline, put this isolated benchmark workspace into
a separate checkout of the pinned main commit, then use the commands in the
[benchmark guide](../README.md). Preserve the original binary as well as its
baseline so both versions can be replayed under the same conditions. These are
synthetic adapter results, with no real-model proof or complete-memory claim.

The final correctness edit avoids constructing unused, overflowing strides for an
empty source on a 32-bit host. After that edit the release visitor was rebuilt
and reproduced the complete after allocation log exactly. Its `advice.rs` hash
remained `12aef4c357ec0ffcb35eac52f60b95586041a999fdfd3c7eb43a4ad50139d1de`;
its `layout.rs` hash was
`af3c5621dcc63fc1201e836624640ab9b5f557f6508f073ab2178e8417e53599`,
and the executable hash was
`931f1abf1c03a7729769231082d6996fd0ed5718768986d650af1238f2d2b7f5`.
The timing table describes the preceding snapshot, not a final throughput
qualification. The final heap bound, allocation counts and proof byte identity
are checked by the [verification record](../VERIFICATION.md).
