# Codec conformance vectors

These files pin the Tessera Canonical Encoding ([RFC-0003](../../../docs/rfcs/RFC-0003-canonical-encoding.md)) for every implementation:
- the prover's `tessera-codec`;
- the verifier's `tv-codec`;
- the Python reader.

`cargo xtask gen-vectors` writes them from cases defined next to the prover's types, and a prover test fails while they are stale. Every implementation checks them with its own decoders.

| File | What it pins |
|---|---|
| `golden.json` | Encodings of values: decoding `hex` as `type` gives `value`. Every proper prefix of `hex` is `ERR-CODEC-TRUNCATED`. |
| `noncanonical.json` | `cases`: inputs that decoding as `type` must reject with `code`, each for one reason. `accepted`: inputs sitting exactly at a limit, which decoding must accept. |
| `envelope.json` | Envelopes (RFC-0003 'Envelope'): the payload a reader returns, or the `code` and `offset` it must report, in the order the RFC fixes. |

Each file states the `limits` (`max_total_bytes`, `max_vec_len`, `max_depth`; RFC-0003 'Limits') its cases are read under.

## Values

`golden.json` gives each value in a form that does not depend on the language:

| Type | JSON |
|---|---|
| Integers, `Fp` | decimal strings, so that 64-bit values survive JSON |
| `bool` | `true` or `false` |
| `Fp2` | `[c0, c1]` |
| Arrays, `Vec`, `Bytes`, `Digest`, tuples | arrays |
| `Option` | `null`, or `{"some": value}` |
| `Ascii` | a string |
| `Rational` | `{"num": "…", "den": "…"}` |
| `Map` | its entries in canonical order, as `[key, value]` pairs |
| A struct | an object keyed by field name |
| An enum value | `{"Variant": fields}`: `null` for a unit variant, an array for a tuple variant, an object for a struct variant |

## Type names

- **Primitives:**
  - `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`;
  - `bool`;
  - `Fp`, a Goldilocks element (RFC-0002);
  - `Fp2`, an element of `Fp[X]/(X^2 - 7)`, encoded as `c0` then `c1`.
- **Composites:**
  - `Vec<T>` and `[T; N]`;
  - `Option<T>`;
  - tuples such as `(u8, Bytes)`;
  - `Map<K, V>`, `Ascii` and `Rational`.

  Two names are aliases: `Bytes` is `Vec<u8>`, and `Digest` is `[u8; 32]`.
- **Nesting shorthands,** for the depth cases. Each is exactly `n` composite levels, each counting one level of depth:
  - `Option^n<T>` is `n` nested `Option`s around `T`: `Option^2<u8>` is `Option<Option<u8>>`.
  - `[u8; 1]^n` is `n` nested one-element arrays around a `u8`: `[u8; 1]^2` is `[[u8; 1]; 1]`.
  - `pair^n` is `n` nested pairs, each adding a `u8` on the right, around a `u8`: `pair^2` is `((u8, u8), u8)`.
- **Test schemas.** Two types exist only for these vectors, as the `schemas` entry of `golden.json` describes them:
  - **`Sample`** is a struct whose fields encode in declaration order: `flag: bool, small: u8, list: Vec<u16>, maybe: Option<i32>, name: Ascii, digest: [u8; 32]`.
  - **`Shape`** is an enum: `Empty`, `Point(u32)`, `Box { width: u16, height: u16 }`, `Nested(Vec<Shape>)`. A value encodes as a `u16` tag equal to the variant's declaration index (0 to 3), then the variant's fields.

## Envelope cases

Each case of `envelope.json` names:
- the `schema` id a reader expects, and the `version` its writers emit, so it accepts `version` and `version - 1` but never 0;
- the payload `type`;
- the `hex` input.

A reader checks the input, then decodes the payload as `type`. It must return `payload`, or fail with `code` at `offset`, counted from the start of the envelope.
- **Limits.** A case reads under the file's `limits`, unless it names RFC-0003 defaults: `"limits": "message"` or `"integer-model"`. Each implementation maps those names to its own constants, which `named_limits` states, so a wrong constant fails.
- **Faults.** A rejection has one fault, except the cases that pin which of two faults is reported.
- **The `EnvelopeProbe` schema.** Most cases expect a test schema, id `0xFFF0` at version 3, which is not in the catalog of [03 §5](../../../docs/spec/03-data-model.md#5-artifact-catalog). With it, the vectors can show version 2 accepted and version 1 rejected.

## Depth

What counts as one level is defined in RFC-0003 'Limits'. Each of these is one level:
- structs and enums, including `Option`;
- tuples and map entries;
- arrays (and so `Digest`);
- `Vec` (and so `Bytes`);
- `Ascii`, `Rational` and `Fp2`.

The depth cases sit at a limit and one level past it, for each kind.
