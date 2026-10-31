# Weighted count reads

Measure the existing `MultiplicityColumn::read_rows` with different public chunk sizes.
Setup runs a validated integer model on a public zero-token workload, builds clearing
checkpoints/cache and one weighted count column, then drops the model/cache/cores before
Criterion timing. It checks prefix and domain-end counts with partial chunk tails against
a complete interval read. The benchmark publishes only public dimensions and timing.

The default is the public `tiny-2-servable` conformance model with two positions. Optional
`COUNT_BENCH_MODEL`, `COUNT_BENCH_PLAN` and `COUNT_BENCH_POSITIONS` select a local model and
matching compiled metadata. A supplied plan uses its existing gadget policy. Without one,
the benchmark derives a synthetic `bench-counts/1` claim list solely for source metadata;
it never generates a proof. `COUNT_BENCH_PREFIX_ROWS` defaults to `2^18 + 33`.

Run from the repository checkout so its Rust pin applies:

```sh
cargo bench --manifest-path crates/tessera-prover/benches/count-reads/Cargo.toml --locked -- --save-baseline before
```

The three cases read the same interval at 256, 4096 and 65536 rows per call. The current
CPU reference uses 256 as its feasible fallback. The budgeted policy can select a larger
chunk only after charging all scratch and keeping its reference cuts. This benchmark
measures the unchanged reader at each configuration; whole-proof checks cover the policy.
The actual-model option measures a source workload, not recorded-greedy fidelity, proof
acceptance or whole-prover memory. Keep operator timings for secret models on the prover's
side of TB-1; public conformance inputs make reviewable benchmark data.

## Work and hardware limits

A weighted column retains `U` fixed 16-byte updates. Reading `R` rows with chunk `C`
visits exactly `U*ceil(R/C)` updates, plus zeroing `R` output `u64`s. Each visit has two
update-word loads, subtraction/comparison, masked index/count selection and a count add;
there are no field products. No secret key, zero count or unused update shortens the loop.
The input-record traffic model is `16*U*ceil(R/C)` bytes and output clearing adds `8*R`;
keyed output read/modify/write adds cache traffic. The arithmetic intensity is zero field
products per byte, with integer work and possible store dependencies instead.

Local hardware inventory: Apple M1 Max, 10 CPU cores and 64 GiB RAM. Apple specifies up to
400 GB/s aggregate memory bandwidth for [M1 Max](https://support.apple.com/en-ie/111900).
Dividing the traffic model by that peak gives an optimistic DRAM-only lower bound, not a
single-core promise: the update buffer may be cache-resident, and masked keyed writes can
be dependency-bound. Record observed throughput and classify the limiting resource after
baseline measurement. Concurrent model calibration remains running; record that contention
and require an improvement beyond measurement noise before changing production policy.

Any later policy must charge all source scratch, preserve the existing feasible small-
budget path and window cuts, and retain canonical proofs under NI/Live comparisons. See
[RFC-0018](../../../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler).

## Initial configuration measurements

The public fixture selected table32: 8,388,608 logical rows, 1,155 fixed weighted updates
and 18,480 backing bytes. Prefix and domain-end intervals both passed exact comparisons.
On the contended M1Max host, the same 262,177-row prefix measured:

| Read chunk | Fixed update visits | Criterion slope |
|---|---:|---:|
| 256 | 1,183,875 | 14.872 ms |
| 4,096 | 75,075 | 1.0381 ms |
| 65,536 | 5,775 | 0.16954 ms |

The 256/65,536 comparison is 87.72x for this reader workload, not a whole-prover speedup.
The shrinking repeated scan and serial masked update dependencies dominate the small
cached update buffer; this is not a measurement of aggregate DRAM bandwidth or field-multiply
throughput. See [raw baseline data](measurements/before-summary.json) and
[Criterion output](measurements/before-criterion.txt) for uncertainty and provenance.
