# Merkle opening allocation benchmark

Criterion 0.5.1 measures one public tree of 65,536 leaves with singleton, sparse,
contiguous and all-leaf query sets. Tree construction is outside timing; opening
and dropping the proof are inside. Setup logs record sibling length and capacity.
This uses no model weights or tokens and does not measure a whole prover.

Run from the repository checkout for its Rust 1.95 pin:

```sh
cargo bench --manifest-path crates/tessera-hash/benches/openings/Cargo.toml --locked -- --save-baseline before
```

The standalone lockfile pins this benchmark's graph. Its CLI/criterion dependencies
need not match the production workspace graph. The unchanged benchmark source used
for both recorded runs is retained in local evidence; formatting changes affect no
workload or algorithm.

## Work and hardware limits

Opening copies digests and indices; it hashes no parents and performs no field
products. At S sibling digests, unavoidable digest traffic is at least 64*S bytes
(32 read, 32 write), plus sorting, index traffic and allocation/copy overhead.
For the sparse-193 fixture S=1606, so digest traffic alone is 102,784 bytes.
M1 Max's up-to 400 GB/s aggregate bandwidth gives an optimistic DRAM lower bound
of about 0.257 microseconds; this is not a single-core promise. Measured latency
is tens of microseconds, consistent with branch/index/allocator and cache latency
costs rather than aggregate bandwidth saturation. The added cardinality pass
reads adjacent public indices and uses integer XOR/leading-zero counts only.

Host: Apple M1 Max, 10 CPU cores, 64 GiB. Model calibration and other host work
continued unchanged; raw samples retain the resulting uncertainty. See Apple's
[M1 Max specification](https://support.apple.com/en-ie/111900).

## Recorded results

| Query set | Before | After | Siblings/capacity before -> after |
|---|---:|---:|---|
| One | 2.032 us | 1.712 us | 16/16 -> 16/16 |
| Sparse 8 | 3.600 us | 2.685 us | 72/128 -> 72/72 |
| Sparse 193 | 33.637 us | 30.677 us | 1606/2048 -> 1606/1606 |
| Contiguous 193 | 4.833 us | 4.618 us | 14/16 -> 14/14 |
| All | 1.078 ms | 0.621 ms | 0/0 -> 0/0 |

Criterion found significant improvements for one, sparse-8 and all-leaf cases;
sparse-193 and contiguous-193 changes were indistinguishable from noise. The
initial exact-capacity candidate regressed on all leaves; the final implementation
returns the canonical empty proof after validated distinct coverage is complete.
No throughput improvement is claimed for whole proofs or real models. Retained
sibling capacity drops by 43.75% and 21.58% for sparse-8 and sparse-193, respectively.

Raw [baseline](measurements/before-summary.json),
[initial candidate](measurements/candidate-summary.json),
[pre-shortcut](measurements/after-initial-summary.json) and
[final](measurements/after-summary.json) samples/logs preserve all attempts.
Resource planning uses a worst-case public bound before queries are drawn,
including repetitions; observed adjacency/collisions never reduce that bound.

## Cardinality and memory argument

For a nonempty sorted distinct query set in a complete tree of height h, let
`d_i = bit_length(q_i XOR q_(i-1))`. Its path union has `h + 1 + sum(d_i)`
nodes: each subsequent leaf joins the previous path at their lowest common
ancestor and adds `d_i` new nodes. A path union with D leaves therefore needs
`S = h + 2 + sum(d_i) - 2*D` missing children. The implementation starts at h
and adds `d_i - 2`, preserving nonnegative prefix counts. The count never
exceeds the existing tree's node count. A contiguous interval uses the dyadic
complements: `popcount(first) + popcount(N - last - 1)`. Validated coverage of
all N leaves has no siblings. Digest order still comes from the original
level-by-level traversal; counting changes only its reservation.

The independent set-frontier tests cover every nonempty subset through 16 leaves,
reversed/repeated inputs, clustered and sparse sets, and every addressable index
height without allocating large trees. Before Fiat–Shamir draws exist, PCS uses
`min(N - 1, min(draws, N) * h)`, since repeated draws can leave only one distinct
leaf. Subtracting the maximum distinct count from N would undercharge that case.
The PCS plan includes the draw-sized query copy, private columns and codeword,
both Merkle index frontiers, exact-reserved sibling upper bound, and simultaneous
private/public column copies. Caller state and other phases remain additional.
