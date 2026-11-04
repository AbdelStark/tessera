# Wider checked table reads (#679)

The CPU planner can expand generated-table reads from 65,536 to at most
1,048,576 rows when spare public allowance covers the complete scratch charge.
It first preserves the existing feasible window cuts and the cache depth selected
with 65,536-row reads. Wider reads cannot displace that cache or add a window
round. Tight budgets retain smaller reads, down to the existing 256-row fallback.
Re-selection starts from that fallback and can shrink again.

This is an internal runtime policy under RFC-0018, with unchanged canonical
counts, fraction identities, transcript schedule and proof format. Both grouped
physical-route refusal guards remain. It establishes no real-model proof
acceptance, whole-proof speedup, fidelity or resource qualification.

## Experiment

Measurements used Rust 1.95.0 on an Apple M1 Max with 64 GiB RAM. Two frozen 1B
proof runs and an 8B calibration capture shared the host. Each candidate has 20
paired observations: ten baseline-first and ten candidate-first, with candidate
order rotated independently. The reported ratios are medians of paired
candidate/baseline timings; intervals bootstrap those ratios with 10,000 draws.
These intervals describe this contended run, not an isolated-host guarantee.

The preregistered primary metric is the complete bounded fraction component.
The adapter uses production buffer lifetimes and base-field compression, calling
the production `FractionTreeRows::root` and `read_level`. The unchanged standalone
count reader remains a diagnostic. Selection requires at least 5% target gain
and no primary control regression above 3%. Counts, roots and ordered nodes
compare exactly across chunks, including partial tails and zero denominators.

Ratios for 1,048,576-row reads versus 65,536-row reads:

| Public workload | Root ratio (95% interval) | Level ratio (95% interval) |
|---|---:|---:|
| Weighted target: tiny-4, 64 positions, 1,048,576 rows, 147,457 fixed updates | 0.803 (0.675–0.856) | 0.863 (0.782–0.956) |
| Few-update control: tiny-2, 2 positions, 8,388,608 rows, 1,155 fixed updates | 0.900 (0.873–0.936) | 0.901 (0.828–0.989) |
| Large materialized dense control: target counts, 1,048,576 rows | 0.893 (0.830–0.987) | 0.911 (0.770–0.991) |

An initial actual dense column had only 65,536 rows, so every proposed cap clamped
to the same effective size. Its root ratio was 1.001 with interval upper bound
1.018. That control alone could not support widening. The large dense control
materializes canonical counts once into `SecretBuf`, outside timing, and uses
the same `copy_from_slice` kernel as dense production storage. It retains the
weighted column as an untimed correctness oracle. It exercises actual read sizes
from 65,536 through 1,048,576; it does not change production storage selection.

The standalone reader has a substantial few-update regression: 1,048,576-row
caps took 2.00x and 2.86x baseline time for prefixes of 262,177 and 1,048,609 rows.
The high-update target prefixes took 0.224x and 0.147x baseline time. The standalone
caller reuses/resizes one output; fraction regeneration allocates fresh base/raw
buffers per read. That difference does not establish the cause of the regression.
The change is supported by the complete-component results and makes no claim of
a universal count-reader improvement.

## Work and memory

A weighted reader visits all `U` fixed-capacity updates for every chunk, independent
of secret keys and counts. Reading `R` rows at chunk `C` reads
`16*U*ceil(R/C)` input bytes, zeroes `8*R` output bytes, and performs keyed integer
read/modify/write operations. There are no field products in that reader. Public
chunk widening reduces repeated scans; it introduces no secret sparsity branch.

Fraction regeneration also converts counts, constructs denominators and folds
nodes. Its source and fold buffers increase with `C`, so fewer scans alone cannot
predict total time. For the height-20 control, the regenerator reports 4,195,176
scratch bytes at 65,536 rows and 67,109,736 at 1,048,576 rows. The CPU planner's
conservative complete scratch charge is separate and includes overlapping caller
storage; the selection checks it with overflow-safe arithmetic before use.

The M1 Max's advertised aggregate bandwidth is an optimistic DRAM-only bound,
not a single-core promise. The weighted records can be cache-resident, and masked
keyed writes have serial dependencies. These results do not measure aggregate
bandwidth or field multiplication throughput.

## Evidence and exclusions

The [benchmark README](../../crates/tessera-prover/benches/count-reads/README.md)
gives reproduction commands. Reviewable data lives beside it under
[`measurements/wider-chunks`](../../crates/tessera-prover/benches/count-reads/measurements/wider-chunks/provenance.json):
source/compiler pins, preregistration, all 1,120 raw balanced paired observations
and summaries. Raw Criterion distributions and full logs remain in the operator's
durable experiment archive.

An early schedule coupled pair order to candidate rotation, so each candidate
always ran in one position. Those timings are excluded. An extracted schedule
control passes, while restoring that coupling fails its ten/ten guard. Another
early adapter used generic extension arithmetic rather than production base
compression; those timings are also excluded. The valid study uses the corrected
adapter and balanced schedule. The initial unsupported 512-position fixture was
refused; the target uses the existing 64-position context without changing it.

Planner regressions cover preserved cuts/cache depth, one-byte-under shrinkage,
checked overflow, typed memory refusal and actual allocations at the new maximum.
The cache-boundary regression was observed failing against the old ceiling before
the implementation changed. Complete proof comparisons, independent NI/Live
acceptance and exact tamper rejection remain required before this policy ships.
