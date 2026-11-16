# P3 window arithmetic on Lucid H100

Scope: RFC-0055 window grids and RFC-0010 sumcheck evaluation. The change uses
successive additions to evaluate each affine line at the same public integer
nodes. It removes one extension-field multiplication per extended grid cell and
`degree * columns` multiplications per generic Boolean pair. The generic loop
also releases one `columns`-element secret scratch buffer. All remaining private
buffers retain their existing clearing ownership. No transcript, degree, node,
mask, proof encoding or verifier rule changes.

## Arithmetic benchmark

Lucid job 6448, `h100-sxm-8-0`, Xeon Platinum 8480CL host; one H100 allocated,
32 allocated CPUs, 400 GiB scheduler memory. These are CPU arithmetic benchmarks,
with synthetic resident sources and one worker; regeneration does no work. The
measured path includes grid preparation, folds, proof production and clearing.
Criterion uses 10 samples, 1 s warmup and a 2 s target measurement; slow cases
extend the actual measurement duration. The resident product is an unchanged
control. `cut` is the grid benchmark's public cut parameter.

Both versions use the expanded benchmark fixture, including a weighted sum of
products with repeated factors at degree 4. The original code runs first, the
candidate second, then the original is restored for a final control. All values
below are Criterion time estimates in milliseconds; the final column compares
the candidate with the restored control.

| Case | Before | Candidate | Restored control | Change |
|---|---:|---:|---:|---:|
| product/resident | 4.965 | 5.032 | 5.009 | +0.5% |
| product/grid/3 | 11.903 | 10.947 | 11.914 | -8.1% |
| product/grid/4 | 16.119 | 14.532 | 16.155 | -10.0% |
| product/grid/5 | 23.043 | 19.845 | 22.993 | -13.7% |
| product/grid/6 | 33.242 | 28.078 | 33.218 | -15.5% |
| product/grid/7 | 48.650 | 40.778 | 48.726 | -16.3% |
| product/grid/8 | 77.100 | 59.876 | 71.753 | -16.6% |
| sum-of-products/grid/0 | 24.797 | 20.899 | 25.183 | -17.0% |
| sum-of-products/grid/4 | 73.505 | 66.330 | 73.239 | -9.4% |
| sum-of-products/grid/8 | 1130.700 | 1007.100 | 1129.900 | -10.9% |

The first product/grid/8 baseline was noisy; its restored control is used for
the comparison. Generic/grid/0's control also has outliers. The stable controls
and unchanged resident case support the arithmetic gain; these microbenchmarks
do not establish an end-to-end percentage. Raw logs are retained on Lucid at
`runs/p3-bench-6448/{before,after,control}.log`.

Reproduce each benchmark from its respective source tree:

```sh
cargo bench --locked --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml --bench windowed -- --save-baseline before
cargo bench --locked --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml --bench windowed -- --baseline before
```

## Equivalence and adversarial review

- The independent multilinear oracle checks every extended grid cell at field
  boundaries, 0–3 variables and radices 2–8, including nonzero extension limbs.
- Existing literal-engine comparisons cover proof rounds, final evaluations,
  challenge points and the subsequent transcript challenge across degrees,
  mixed public/private factors, unequal arities, padding and grid-node challenges.
- Doubling the extension step fails the boundary oracle. Omitting the extra step
  from node 0 to node 2 fails the literal-transcript comparison. Both mutants were
  killed; restored window controls and full serialized sumcheck suites pass.
- All new branches and indices depend on public loop positions. Arithmetic uses
  the unchanged canonical field operations covered by the x86 branchless gate.
- No verifier implementation changes. No new data or secret is logged or retained.

## End-to-end run

Pending final Lucid run: final P3/P4 code plus the CPU hash and table-generation
allocation corrections, both tiny CPU/GPU comparisons, GPU-feature clippy, then the same
1B 8 + 8 request as the [P4 measurement](p4-device-combine.md), with pinned
statement, weight root and public tokens at verification. Jobs 6460 and 6461 were cancelled during proving when Linux CI exposed the CPU
hash and table-generation memory-contract bugs. Their incomplete runs are excluded
from measurements; the final run waits for the Linux regression checks.
