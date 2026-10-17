# Performance measurements

This directory holds measured performance: the CPU microbenchmarks' baselines now, and the v0.1 H100 unit-cost reports of [RFC-0020](../rfcs/RFC-0020-cost-model.md#measurement-protocol-v01-gate) once they are measured. Every number in [08](../spec/08-performance-budget.md) stays a projection until a file here replaces it. Every JSON file here is a report whose JSON Schema is in [`spec/schemas/`](../../spec/schemas/README.md), and `cargo xtask lint` validates it. [`examples/`](examples/) holds example reports: `tessera bench model --source-defaults --json`'s and the calibration's. Tests keep them current. [`v0.2/`](v0.2/README.md) holds the compiler's commitment reports for Llama-3.1-405B, which settle RISK-2 and the table bound `B_max`.

## CPU microbenchmarks

`tessera-bench`'s `primitives` benchmark times the prover's primitives at fixed sizes, single-threaded ([08 §6](../spec/08-performance-budget.md#6-regression-gates)):

| Benchmark | Work per run | Unit |
|---|---|---|
| `field/fp-mul`, `field/fp2-mul` | `2^20` independent Goldilocks or `Fp2` products | product |
| `field/fp-batch-inverse` | one batch inversion of `2^16` elements | element |
| `ntt/fp` | one forward NTT of `2^20` points, its plan built once | point |
| `hash/column-chains` | one block of `B = 32` rows of `2^14` columns into their chains ([RFC-0004](../rfcs/RFC-0004-hashing-and-transcript.md#column-digests)) | symbol |
| `merkle/multiproof` | 64 canonical multiproofs of 155 indices in a tree of `2^20` leaves | opening |
| `sumcheck/product-3` | a degree-3 sumcheck of three `Fp` tables of `2^20` entries | entry |
| `logup/prove` | the LogUp stage of `2^18` lookups into `2^16` rows with values | lookup |

Each runs with RFC-0020's measurement protocol: 3 warm-up runs, then 10 measured runs. It reports their median, minimum, maximum and coefficient of variation, in nanoseconds per unit. A coefficient above 3% is flagged as noisy, and the run should be repeated.

```sh
cargo bench -p tessera-bench --bench primitives                     # print the table
cargo bench -p tessera-bench --bench primitives -- \
  --runner-class NAME --out docs/perf/baselines/NAME.json            # also write a baseline
```

A relative `--out` is relative to the repository root, although cargo runs the benchmark from `crates/tessera-bench/`. Without `--bench`, as under `cargo test --benches`, each benchmark runs once at small sizes as a smoke test, and nothing is written.

### Baselines

A baseline is a `tessera.microbenchmarks/1` report, one per runner class. It has the following fields:

- `schema`;
- `runner_class`, a lowercase token naming the machine type;
- `os` and `arch`;
- `protocol`, the warm-up and measured run counts;
- `measurements`, each with:
  - `name`, the key that a candidate is compared by;
  - `unit` and `units`, the work per run;
  - `ns_per_unit`: `median`, `min`, `max` and `cv`;
  - `noisy`.

Every number is finite.

| File | Runner class | How it was measured |
|---|---|---|
| [`baselines/github-ubuntu-24.04.json`](baselines/github-ubuntu-24.04.json) | GitHub-hosted `ubuntu-24.04` (x86_64) | The `bench` workflow ([`.github/workflows/bench.yml`](../../.github/workflows/bench.yml)) runs the benchmark when `crates/tessera-bench/` changes, or on demand, and uploads the report as an artifact. |

The regression gate of 08 §6 compares each benchmark's median with its baseline on the same runner class, and a regression above 5% fails. That gate runs on a runner with locked clocks (#167). GitHub-hosted runners share their hosts, so their baseline records where the primitives stand, not a threshold to gate on.
