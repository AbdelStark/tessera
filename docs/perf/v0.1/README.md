# v0.1 H100 unit costs

The measurements of [RFC-0020 'Measurement protocol'](../../rfcs/RFC-0020-cost-model.md#measurement-protocol-v01-gate), one report per unit cost as its microkernel lands. The decision under RFC-0020's [decision rule](../../rfcs/RFC-0020-cost-model.md#decision-rule) waits for `c_c`, `c_ℓ` and `c_π` together ([#66](https://github.com/AbdelStark/tessera/issues/66)).

| Unit | Assumed | Measured (median) | `ρ` | Report |
|---|---|---|---|---|
| `c_c` | 30 ps | 134.7 ps | 4.49 | [c_c on H100](cc-h100.md) |
| `c_ℓ` | 40 ps | not measured | | [#63](https://github.com/AbdelStark/tessera/issues/63) |
| `c_π` | 12 ps | 84.6 ps | 7.05 | [c_π on H100](pi-h100.md) |
