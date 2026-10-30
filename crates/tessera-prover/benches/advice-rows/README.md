# Canonical ordinary advice rows

This isolated workspace measures the actual `tessera_prover::advice::visit_rows`
used by P1, P4 and P5. Criterion and its lockfile enter neither production
workspace's dependency graph. The fixed public synthetic `tiny-2-servable.d`
fixture, `bench-rows/1` layout, sequence lengths 3/8/16 and segment 0 exclude
multiplicity tables. Setup and witness construction precede measurement; the
sink consumes each borrowed row without retaining it. These are adapter
measurements, not proof timings or certifying parameter sets.

```sh
CARGO_TARGET_DIR=/tmp/tessera-advice-rows cargo bench \
  --manifest-path crates/tessera-prover/benches/advice-rows/Cargo.toml \
  --bench advice_rows --locked -- --save-baseline before
# After changing the adapter, use the same target directory and fixture:
CARGO_TARGET_DIR=/tmp/tessera-advice-rows cargo bench \
  --manifest-path crates/tessera-prover/benches/advice-rows/Cargo.toml \
  --bench advice_rows --locked -- --baseline before
ADVICE_ROW_ALLOCATION=1 CARGO_TARGET_DIR=/tmp/tessera-advice-rows cargo bench \
  --manifest-path crates/tessera-prover/benches/advice-rows/Cargo.toml \
  --bench advice_rows --locked
```

Cargo supplies Criterion's `--bench` flag. Invoking the executable directly
requires `--bench`; otherwise Criterion performs smoke tests without timing.
Allocation counts include `WitnessSource::tile` and its borrowed column setup,
so their peak includes existing tile allocations beyond the row adapter.

For an ordinary column with `P` padded cells, `V` source values and row width
`K`, the old adapter allocated `8P` bytes of clearing field storage. It now uses
`8K` bytes plus public metadata proportional to source rank. Source tile storage,
PCS encoding, masks and proof state are separate allocations. The preserved
scalar scatter is a test-only oracle in `advice.rs`.

The old adapter zeroed and eventually cleared `8P` bytes and wrote converted
source values; it also allocated placement metadata for each dense source value.
The new adapter zeroes each emitted row and converts only its mapped source
positions, then clears its one row on release. Index extraction and source
strides use public shapes only. There is no new field multiplication or secret
index selection. The expected gain is allocation/indexing work and storage, not
a faster arithmetic kernel. A bandwidth-only lower bound for traffic `B` is
`B / bandwidth`; the M1 Max's vendor peak of
[400 GB/s](https://support.apple.com/en-mo/111901) is shared memory bandwidth,
not a measured single-core rate or a proof-throughput estimate.

The [recorded M1 Max run](measurements/README.md) includes raw samples, allocation
counts, source hashes and measurement limitations.

## Verification

The [verification record](VERIFICATION.md) retains the fault-injection results,
proof digests and scope of the independent checks.

`source_rows` exhausts ranks 0–3, lengths 1–5 and row splits against the preserved
`locate` map; score tests exercise every small key split and partial block.
Slotted cases preserve natural indices and public zero-row schedules. The
compiler's frozen `advice/layout.json` tests invert the oracle's bit permutation
independently of `locate`. Maximum-size metadata tests sample a 2^63-cell tensor
without materializing it. Prover tests compare complete rows at signed integer
boundaries, detect stale padding and changed order, and bound adapter allocations
for increasing row counts on success and sink failure.

The field row remains a `SecretBuf`: its unchanged destructor clears the entire
allocation, including spare capacity (OBS-6). Allocation tests verify reuse and
release; they do not inspect freed memory. Source/sink errors and complete proof
acceptance are additionally exercised by `model_leaf` and the independent
verifier with explicitly test-only parameters.

`mutate.py` plants 19 independent faults in a scratch copy and retains original
and restored controls, logs and hashes. Compilation errors and timeouts do not
count as caught faults. Copy the committed tree into a scratch repository first:

```sh
python3 crates/tessera-prover/benches/advice-rows/mutate.py \
  --repo /absolute/path/to/scratch/tessera \
  --out /tmp/tessera-row-mutations \
  --target-dir /tmp/tessera-row-mutation-target
```

This change leaves all planner charges and parameter sets unchanged. It does not
resolve the full [real-model memory obligation](https://github.com/AbdelStark/tessera/issues/629)
or the [real-model leaf-layout failure](https://github.com/AbdelStark/tessera/issues/617).
