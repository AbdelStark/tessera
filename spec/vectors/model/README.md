# Model vectors

`integer-model.json` pins the encoding of `IntegerModel`, artifact `0x0101` ([03 §2](../../../docs/spec/03-data-model.md#2-model-types)): a model of one layer, sealed in its envelope ([RFC-0003](../../../docs/rfcs/RFC-0003-canonical-encoding.md#envelope)), and its value.

- `cargo xtask gen-vectors` writes it from the Rust types of `tessera-ir`, and a prover test fails while it is stale.
- The Python writer and reader of `tessera-quant` check it in both directions: decoding `hex` under the `IntegerModel` limits gives `value`, and encoding `value` gives `hex`.
- `value` follows the codec vectors' conventions ([`../codec/README.md`](../codec/README.md#values)): integers as decimal strings, a struct as an object keyed by field name, an enum value as `{"Variant": fields}`, a `Map` as its entries in canonical order.
