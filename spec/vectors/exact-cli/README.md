# Exact CLI fixture

Synthetic public model and tokens for the [real CLI suite](../../../docs/testing/exact-cli.md).
`generate.py` uses the independent integer oracle under `../models/`; it invokes
neither Rust nor `tessera_quant`. `expected.json` records the nonzero reference
logits, exact greedy continuation, public request metadata and artifact hashes.

The existing secret-class IntegerModel and TokenFile encodings are unchanged.
These bytes contain only synthetic test data. Three layers and five positions
exercise independent padding axes. The synthetic RMS epsilon and reciprocal
precision bound all carry tables under the 18-bit policy used by the test.

```sh
python3 spec/vectors/exact-cli/generate.py --check
```
