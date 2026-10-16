# Cost-model vectors

`cost-model.json` holds the source's projections of the proof multiplier `μ` for Llama-3.1-405B, the inputs they rest on, and the model of [08 §1](../../../docs/spec/08-performance-budget.md#1-cost-model) evaluated on those inputs. It is the golden data of [RFC-0020 'Executable cost model'](../../../docs/rfcs/RFC-0020-cost-model.md#executable-cost-model). The source is [`proofs-of-inference-sota-path-forward.md`](../../../proofs-of-inference-sota-path-forward.md) at the repository root:
- its §5 states the model, the unit costs and the counts, and charts `μ` at eight contexts (lines 95–134);
- its §8 tabulates proof time, sampling rate and detection probability at four contexts (lines 238–249).

[`generate.py`](generate.py) writes the file. It is the independent oracle: it evaluates 08 §1 term by term in Python's floating point and shares no code with `crates/`. Before writing anything, it checks every claim below. `python3 spec/vectors/cost-model/generate.py` reproduces the file byte for byte.

## What the tests check

`crates/tessera-bench/tests/golden.rs`:
- The figures here are the document's: the chart's lines, the chart-data table and the §8 table, parsed from the markdown.
- `tessera_bench::mu` with `Counts::SOURCE_405B` and `UnitCosts::SOURCE` equals this file's `model` to 1e-12, relative.
- The model is within `tolerances.chart` (±0.05) of the chart's linear, attention and total columns at every context, and within `tolerances.weight_pass` (±0.0005) of the weight pass.
- The calculators, applied to §8's own `μ` column, give its sampling and detection columns within `tolerances.percentage_points` (0.1). They give its proof times within half the unit of the last printed digit.

## Provenance of the inputs

Each entry of `inputs` gives its `value`, the `line` it comes from, the `text` there, verbatim, a `note` on how the value follows from that text when it is not the text's own number, and one of three provenances:

- `stated`: the source gives the number. These are the unit costs, the glue, the native rate (60% of 1,979 TOPS), and the 0.24 GPU-s weight pass per 16,384-token batch (line 101).
- `derived`: computed from stated numbers.
  - 137,216 committed elements and lookups per layer per token (line 97), times 126 layers: `C = K = 17,289,216`.
  - `A = 128 · 126 / 2 = 8,064` scores per token per unit of context (line 99).
  - `F_att = 512 · A`, at 512 native operations per score (line 99).
- `reconstructed`: `F_lin`, which the source writes as `8.1×10^11` (line 97), to two significant digits.
  - Two operations per parameter of Llama-3.1-405B, `2 · 405,853,388,800`, reproduce the chart.
  - The attention column alone, which does not depend on `C`, pins `F_lin` within 0.1% of that value.
  - `8.1×10^11` misses the attention column by 0.11.

Contexts are binary: 1K is 1,024 tokens.

## Known deviations of the source

- **Linear column.** The chart's linear column corresponds to about `1.76×10^7` committed elements per token (least squares, `chart.fitted_commits_per_token`). That is 1.8% above the stated `137,216 · 126 = 1.729×10^7`. The model keeps the stated count, so its linear term runs up to 0.036 below the chart, inside ±0.05.
- **§8 rounding.** §8 is computed from its one-decimal `μ` column. From the model's `μ` instead, 16K detection would be 83.41% against the printed 83.3%. So the calculators are checked from the table's `μ`, and the model is checked against the chart.

## The file

| Field | Meaning |
|---|---|
| `source` | the source document, from the repository root |
| `inputs` | name → `{value, provenance, line, text, note}`, `note` when the value is not the text's own number |
| `chart` | `lines`; `contexts` in tokens; `total`, `attention`, `linear` (lines 113–115, two decimals); `weight_pass` and `total_one_decimal` (the chart-data table, lines 127–134); `fitted_commits_per_token` |
| `section_8` | `lines`; the scenario `gpus`, `eta`, `target_overhead`, `violations`; `rows` of `context`, `mu`, `proof_time_s` with `proof_time_half_unit_s`, `sampling_percent`, `detection_percent` and `detection_approximate` (≈100%) |
| `tolerances` | `chart`, `weight_pass`, `percentage_points` |
| `model` | per context: `linear`, `attention`, `weights` and `total` from the inputs |
| `calculators` | per §8 row: from its `mu`, `proof_time_s`, `sampling_rate` and `detection` as fractions |
