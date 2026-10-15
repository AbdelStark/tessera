# RFC-0003: Canonical Encoding and Schema-Versioned Artifacts

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1

## Summary

Every artifact and protocol message in Tessera is serialized with the Tessera Canonical Encoding (TCE): fixed-width little-endian integers, canonical field elements, length-prefixed sequences, tagged unions, and no optional or unknown fields. Each value has exactly one encoding. Artifacts are wrapped in a versioned envelope with a BLAKE3 digest. Canonicality makes transcripts unambiguous and lets two independent implementations (prover and verifier) agree byte for byte.

## Motivation

Fiat–Shamir soundness requires that the transcript commits to exactly what the verifier parses. A format with several encodings of one value (varints with redundant forms, optional fields, unordered maps, non-canonical field elements) lets a prover present the verifier one parse while hashing another, or fork the transcript. The verifier is an independent implementation (RFC-0001), so the format must also be simple enough to implement twice without shared code. The source requires a "public claim list" from which the verifier recompiles constraints (source §6, principle 12); that list is only public and auditable if its encoding is unambiguous.

## Goals

- One encoding per value; decoders reject everything else.
- Decoding cost and memory bounded by declared limits before allocation.
- Streaming read and write for artifacts of hundreds of gigabytes (`IntegerModel` for 405B).
- Implementable in under 1,000 lines per language.
- Explicit schema identifiers and versions on every artifact.

## Non-Goals

- Schema evolution through optional or unknown fields. Evolution happens by version bumps ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)).
- Zero-copy access or random access by field.
- Human readability; `tessera inspect` renders artifacts as text.

## Proposed Design

### Primitive types

| Type | Encoding | Decoder rejects |
|---|---|---|
| `u8`, `u16`, `u32`, `u64` | Little-endian, fixed width | Truncation |
| `i8`, `i16`, `i32`, `i64` | Two's complement, little-endian, fixed width | Truncation |
| `bool` | One byte, `0x00` or `0x01` | Any other byte |
| `Fp` | `u64` little-endian | Values `≥ P` |
| `Fp2` | `Fp` `c0`, then `Fp` `c1` | Either coefficient non-canonical |
| `Digest` | 32 raw bytes | Truncation |
| `[T; N]` | `N` encodings of `T`, concatenated, no length | Truncation |
| `Vec<T>` | `u64` length `n`, then `n` encodings of `T` | `n` above the limit; `n · min_size(T)` above the remaining bytes |
| `Bytes` | `Vec<u8>` | As `Vec` |
| `Ascii` | `Vec<u8>` whose bytes are in `0x20..=0x7E` | Any other byte |
| `Rational` | `i64` numerator, `u64` denominator; denominator `> 0`; `gcd(|num|, den) = 1` | Zero denominator; non-reduced fraction |

### Composite types

- **Struct:** fields concatenated in declaration order, no padding, no field tags.
- **Enum (tagged union):** `u16` tag equal to the variant's declaration index (from 0), then the variant's fields as a struct. Unknown tags are rejected.
- **`Option<T>`:** encoded as the enum `{ None, Some(T) }`, that is a `u16` tag 0 or 1 followed by the payload. It expresses a value that is semantically absent; it never expresses a field that later versions may add.
- **Map:** not a TCE type. Schemas use `Vec<(K, V)>` sorted strictly ascending by the byte encoding of `K`. Decoders reject unsorted or duplicate keys.
- **Floating point:** not a TCE type. Quantities that would be floats are `Rational` or fixed-point integers with the scale named in the schema ([03 INV-DM-6](../spec/03-data-model.md#6-invariants)).

### No optional fields

A schema version defines exactly which bytes follow which. There are no trailing extension areas, no "ignore unknown fields" rule, and no defaults for missing data. Adding, removing, reordering or retyping a field produces a new schema version. This rule (COMPAT-4 in [02](../spec/02-public-api.md#6-compatibility-rules)) is what makes every accepted byte string canonical.

### Envelope

```rust
pub struct Envelope {
    pub magic: [u8; 4],          // b"TSRA"
    pub envelope_version: u16,   // 1
    pub schema_id: u16,          // see 03-data-model §5
    pub schema_version: u16,
    pub reserved: u16,           // must be 0
    pub payload_len: u64,
    pub payload: [u8; payload_len],
    pub digest: Digest,          // BLAKE3 derive_key("tessera/v1/envelope") over all preceding bytes
}
```

The digest trails the payload so writers can stream. Readers either verify the digest in a streaming pre-pass before decoding (files) or decode while hashing and discard the result on mismatch (messages). Nothing decoded from an envelope is acted on before its digest verifies.

### Auxiliary schemas

Two schemas carry data between the Python front-end and Rust tools and are added to the artifact catalog in [03 §5](../spec/03-data-model.md#5-artifact-catalog):

```rust
pub struct TokenFile { pub sequences: Vec<Vec<u32>> }                  // schema 0x0601
pub struct LogitsFile { pub vocab: u32, pub positions: Vec<Vec<i64>> } // schema 0x0602, integer logits of D
```

### Limits

```rust
pub struct DecodeLimits {
    pub max_total_bytes: u64,  // whole envelope
    pub max_vec_len: u64,      // any single Vec length
    pub max_depth: u16,        // nesting depth of composite types
}
```

Every decoder takes limits explicitly. Defaults: verifier messages 8 GiB total, `max_vec_len` 2^34, depth 32; `IntegerModel` 1 TiB total. Length checks happen before allocation, so a hostile length prefix costs nothing.

So that independent decoders return the same code for every input:

- **Order of checks.** `max_total_bytes` is checked before decoding starts. For each sequence (`Vec`, `Bytes`, `Ascii`, map), the declared length is checked against `max_vec_len` (`ERR-CODEC-LIMIT`), then `length × min_size(T)` against the bytes left (`ERR-CODEC-TRUNCATED`), both before any element is read. An array is checked as a whole against the bytes left the same way.
- **Depth.** Each value whose encoding holds other values adds one level while it is decoded:
  - structs and enums, including `Option`;
  - tuples and map entries;
  - arrays, so `Digest`;
  - `Vec`, so `Bytes`;
  - `Ascii`, `Rational` and `Fp2`.

  Integers, `bool` and `Fp` add none. A value beyond `max_depth` is `ERR-CODEC-LIMIT`. A limit above 256 counts as 256: decoding recurses once per level, and deeper recursion could exhaust a thread's stack.
- **Empty elements.** A sequence's element type encodes to at least one byte: a sequence of empty values would carry only a count and cost time without consuming input.

### Implementations

| Implementation | Location | Construction |
|---|---|---|
| Prover side | `crates/tessera-codec` | `Encode`/`Decode` traits plus a derive macro for structs and enums (`crates/tessera-codec-derive`) |
| Verifier side | `verifier/crates/tv-codec` | Hand-written decoders and the encoders needed to recompute digests; no shared macro |
| Python | `python/tessera_quant/tce.py` | Writer and reader for `IntegerModel`, `TokenFile`, `LogitsFile` |

Each implementation is tested against the same golden vectors (RFC-0001, "Shared artifacts that are not code").

### Invariants

- **INV-TCE-1.** For every value `v` of a schema type, `decode(encode(v)) = v`.
- **INV-TCE-2.** For every byte string `b` that decodes successfully, `encode(decode(b)) = b`.
- **INV-TCE-3.** Decoding never allocates more than `min(declared length, remaining bytes / min_size)` elements.
- **INV-TCE-4.** A decoder never reads past `payload_len`; leftover bytes inside the payload are an error.

### Failure modes

| Failure | Code |
|---|---|
| Wrong magic or schema id | `ERR-CODEC-MAGIC` |
| Unsupported envelope, schema or protocol version | `ERR-CODEC-VERSION` |
| Non-canonical value, bad tag, unsorted map, trailing bytes | `ERR-CODEC-NONCANONICAL` |
| Truncated input | `ERR-CODEC-TRUNCATED` |
| Digest mismatch | `ERR-CODEC-DIGEST` |
| Length above `max_vec_len`, input above `max_total_bytes`, or nesting beyond `max_depth` | `ERR-CODEC-LIMIT` |

## Alternatives Considered

1. **Protocol Buffers.** Considered for tooling and cross-language support. Rejected: field order, default-value omission and varint forms give one message several encodings, and unknown-field preservation conflicts with canonicality.
2. **Deterministic CBOR (RFC 8949 §4.2).** Considered because it is standardized and can be made canonical. Rejected: a conforming decoder must handle many major types, floats and indefinite lengths, and must enforce preferred serialization, all of which enlarges the verifier's trusted base for no gain.
3. **SSZ.** Considered: canonical and fixed-layout. Rejected: variable-size offsets and Merkleization rules add parsing logic Tessera does not need.
4. **bincode or postcard.** Rejected: their wire formats are defined by Rust crate versions, not by a specification, and postcard uses varints.
5. **safetensors for weights.** Considered for `IntegerModel` because it is the checkpoint format the front-end reads. Rejected for tier-A artifacts: the JSON header is not canonical (key order, whitespace, number forms).
6. **JSON everywhere.** Rejected for anything hashed or verified. Kept for human-facing reports only.

## Drawbacks

- A custom format needs its own inspection tool and documentation.
- Version bumps for every layout change produce more versions than a flexible format would.
- Fixed-width integers waste bytes for small values; irrelevant next to tensor payloads.

## Migration / Rollout

Lands in v0.1 before any artifact is written. The envelope's `envelope_version` allows a future envelope change without ambiguity.

## Testing Strategy

- Golden vectors in `spec/vectors/codec/` for every schema in the catalog: minimal values, maximal lengths within limits, every enum variant, nested vectors, edge-case integers (`0`, `MAX`, `MIN`).
- Property tests: INV-TCE-1 over random values of every schema; INV-TCE-2 over random byte strings (accepted ones must re-encode identically).
- Truncation: for one vector per schema, every prefix is rejected with `ERR-CODEC-TRUNCATED`.
- Non-canonical corpus: `Fp` values `≥ P`, `bool` value 2, unknown enum tag, unsorted map, non-reduced rational, ASCII byte `0x7F`, trailing payload byte, non-zero `reserved`, each rejected with the right code by all three implementations.
- Hostile lengths: a `Vec` length of 2^63 with 16 remaining bytes is rejected without allocation (asserted with a counting allocator in Rust).
- Fuzzing: `cargo-fuzz` targets for every verifier-side decoder (nightly 1 h; 24 CPU-hours before v1.0).
- Cross-language: the Python writer's `IntegerModel` for each conformance model decodes in both Rust readers to identical values.

## Open Questions

None.

## References

- Source §6 principle 12, §7 (claim list).
- RFC 8949, Concise Binary Object Representation, §4.2 (deterministic encoding).
- [03 Data model](../spec/03-data-model.md), [04 Error model](../spec/04-error-model.md).
