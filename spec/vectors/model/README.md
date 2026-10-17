# Model vectors

`integer-model.json` pins the encoding of `IntegerModel`, artifact `0x0101` ([03 §2](../../../docs/spec/03-data-model.md#2-model-types)). Each of its `cases` is a model of one layer, sealed in its envelope ([RFC-0003](../../../docs/rfcs/RFC-0003-canonical-encoding.md#envelope)), with its value:
- `servable-pot`: the servable profile, power-of-two scales;
- `minimal-affine`: the minimal profile, with affine multipliers.

Between them the models hold every weight type, every rescale site and quantization point, both profiles and a map of seven entries, so that each enum tag and each canonical order is pinned. Their fields hold distinct values: they pin encodings, not a calibrated model.

- `cargo xtask gen-vectors` writes them from the Rust types of `tessera-ir`, and a prover test fails while they are stale.
- The Python writer and reader of `tessera-quant` check them in both directions: decoding `hex` under the `IntegerModel` limits gives `value`, and encoding `value` gives `hex`.
- `value` follows the codec vectors' conventions ([`../codec/README.md`](../codec/README.md#values)): integers as decimal strings, a struct as an object keyed by field name, an enum value as `{"Variant": fields}`, a `Map` as its entries in canonical order.
