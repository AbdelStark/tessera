# Canary fixture

The model and tokens of the OBS-3 canary scan ([05 §3](../../../docs/spec/05-observability.md#3-data-classification-and-redaction)), run by `e2e_redaction_canary_scan` in the [exact-CLI suite](../../../docs/testing/exact-cli.md#the-canary-scan). `generate.py` uses the independent integer oracle under `../models/`; it invokes neither Rust nor `tessera_quant`. Every value is synthetic test data.

- `canary-reference.d`: a two-layer model of width 2 and vocabulary 4,096. Three runs of eight INT8 weights, in the embedding, a gate projection and the LM head, each read little-endian as a distinctive 64-bit value: every byte nonzero, the last negative. The LM head's run holds the rows of tokens `0xa3c` to `0xa3f`, longer than the drawn rows, so that the greedy continuation picks one of them.
- `recorded.ttk`: a prompt of three distinctive token ids and the model's greedy continuation of three. A distinctive id has four decimal digits that do not read as a year, and a hex form that mixes digits and letters.
- `mismatch.ttk`: the same tokens with the last output replaced by `0xa3c`, which the model does not choose; proving with it fails with `ERR-WITNESS-EXACT-MISMATCH`.
- `canaries.json`: the runs, with their positions and values, the token ids, and both files' SHA-256.

```sh
python3 spec/vectors/canary/generate.py --check
```
