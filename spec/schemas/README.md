# Report schemas

JSON reports are not TCE artifacts. Each carries a `"schema"` identifier, `tessera.<name>/<version>`, and its JSON Schema lives here as `<name>-v<version>.schema.json` ([03 §5](../../docs/spec/03-data-model.md#5-artifact-catalog)).

| Schema | Written by | Committed instances |
|---|---|---|
| [`microbenchmarks-v1`](microbenchmarks-v1.schema.json) | `cargo bench -p tessera-bench --bench primitives -- --runner-class NAME --out FILE` | [`docs/perf/baselines/`](../../docs/perf/baselines/) |
| [`cost-model-v1`](cost-model-v1.schema.json) | `tessera bench model --out FILE` | [`docs/perf/examples/cost-model-source-defaults.json`](../../docs/perf/examples/cost-model-source-defaults.json), which a `tessera-cli` test keeps current, up to the last bits of numbers that platforms round differently |
| [`calibration-v1`](calibration-v1.schema.json) | `tessera_quant.capture.calibrate`, beside the secret statistics file | [`docs/perf/examples/calibration-tiny.json`](../../docs/perf/examples/calibration-tiny.json), which a `tessera-quant` test keeps current but for its statistics digest and environment, which depend on the platform |
| [`commitment-report-v1`](commitment-report-v1.schema.json) | `tessera_compiler::report`, with the projections of `tessera-bench`'s cost model | [`docs/perf/examples/commitment-report-405b.json`](../../docs/perf/examples/commitment-report-405b.json), which a `tessera-bench` test keeps current; `TESSERA_WRITE_EXAMPLES=1` rewrites it |
| [`unit-costs-v1`](unit-costs-v1.schema.json) | `tessera bench unit-costs`, with the H100 measurement of [RFC-0020](../../docs/rfcs/RFC-0020-cost-model.md#measurement-protocol-v01-gate) | none yet; `docs/perf/v0.1/` will hold the measurement |

`cargo xtask lint` validates every committed report under `docs/perf/` against the schema its identifier names. It fails on a report without an identifier, or on an identifier without a schema file. It also checks every schema here.

## The keyword subset

The validator implements a small subset of JSON Schema 2020-12, chosen so that it needs no third-party crate. A schema that uses any other keyword fails the lint, so no constraint is ever silently ignored.

| Kind | Keywords |
|---|---|
| Annotations | `$schema`, `$id`, `title`, `description`, `$comment` |
| References | `$defs`, and `$ref` to `#/$defs/<name>` |
| Shape | `type` (`integer` is a number without a fraction), `properties`, `required`, `additionalProperties`, `items` |
| Values | `const`, `enum` (numbers compare by value), `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength`, `maxLength`, `minItems`, `maxItems` |

A report version is frozen once a release writes it: a later change gets a new version and a new file, and a reader accepts the current version and the one before it. Until that release, the version's file changes in place with its writer ([09 §3](../../docs/spec/09-release-and-versioning.md#3-protocol-and-schema-versions)).
