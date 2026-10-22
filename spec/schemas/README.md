# Report schemas

JSON reports, and the specifications `tessera-quant` reads, are not TCE artifacts. Each carries a `"schema"` identifier, `tessera.<name>/<version>`, and its JSON Schema lives here as `<name>-v<version>.schema.json` ([03 §5](../../docs/spec/03-data-model.md#5-artifact-catalog)).

| Schema | Written by | Committed instances |
|---|---|---|
| [`microbenchmarks-v1`](microbenchmarks-v1.schema.json) | `cargo bench -p tessera-bench --bench primitives -- --runner-class NAME --out FILE` | [`docs/perf/baselines/`](../../docs/perf/baselines/) |
| [`cost-model-v1`](cost-model-v1.schema.json) | `tessera bench model --out FILE` | [`docs/perf/examples/cost-model-source-defaults.json`](../../docs/perf/examples/cost-model-source-defaults.json), which a `tessera-cli` test keeps current, up to the last bits of numbers that platforms round differently |
| [`calibration-v1`](calibration-v1.schema.json) | `tessera_quant.capture.write_calibration`, beside the secret statistics file | [`docs/perf/examples/calibration-tiny.json`](../../docs/perf/examples/calibration-tiny.json), which a `tessera-quant` test keeps current but for its statistics digest and environment, which depend on the platform |
| [`fidelity-v1`](fidelity-v1.schema.json) | `tessera_quant.fidelity.evaluate_fidelity` (`FidelityReport.as_json`) | [`docs/perf/examples/fidelity-tiny.json`](../../docs/perf/examples/fidelity-tiny.json), whose shape a `tessera-quant` test keeps current; its numbers depend on the platform's BF16 arithmetic |
| [`calibration-spec-v1`](calibration-spec-v1.schema.json) | an operator, for `tessera-quant compile --calibration` (02 §2.1), which reads it strictly | [`docs/perf/examples/calibration-spec-c4.json`](../../docs/perf/examples/calibration-spec-c4.json), RFC-0005's default calibration, which a `tessera-quant` test reads |
| [`eval-spec-v1`](eval-spec-v1.schema.json) | an operator, for `tessera-quant eval --dataset` (02 §2.1), which reads it strictly | none yet: the 8B gates' data is [#93](https://github.com/AbdelStark/tessera/issues/93)'s |
| [`commitment-report-v1`](commitment-report-v1.schema.json) | `tessera_compiler::report`, with the projections of `tessera-bench`'s cost model | the 405B reports in [`docs/perf/v0.2/`](../../docs/perf/v0.2/README.md), which a `tessera-bench` test keeps current; `TESSERA_WRITE_EXAMPLES=1` rewrites them |
| [`soundness-v1`](soundness-v1.schema.json) | `tessera params check --json`, RFC-0014's soundness calculator | [`docs/perf/examples/soundness-live-64.json`](../../docs/perf/examples/soundness-live-64.json) and [`soundness-ni-100.json`](../../docs/perf/examples/soundness-ni-100.json), the 405B reports, which a `tessera-cli` test keeps current; the oracle of [`spec/vectors/soundness/`](../vectors/soundness/README.md) writes the same reports |
| [`verdict-v1`](verdict-v1.schema.json) | `tv_core::Verdict::to_json`, deterministic protocol/1 core report | [`docs/perf/examples/verdict-test-small.json`](../../docs/perf/examples/verdict-test-small.json), from the synthetic CPU integration fixture; test parameters certify nothing |
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
