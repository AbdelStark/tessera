# Protocol message vectors

`protocol.json` pins schema version 1 of every protocol message, `tessera-protocol/1` ([RFC-0036](../../../docs/rfcs/RFC-0036-protocol-messages.md#versions-and-envelopes)): its encoding in TCE ([RFC-0003](../../../docs/rfcs/RFC-0003-canonical-encoding.md)), its envelope, the coin toss of a live session, and the inputs every reader must refuse. `crates/tessera-protocol/tests/vectors.rs` checks every case, and the verifier's decoders will check them in `tv-core` (#128). Every `Round1` carries protocol version 0, the claim list's until `tessera-protocol/1` is assigned ([09 §3](../../../docs/spec/09-release-and-versioning.md#3-protocol-and-schema-versions)).

[`generate.py`](generate.py) writes it. The script is the independent oracle: it shares no code with `crates/`, `verifier/` or `python/`. It writes each message's schema out field by field, from RFC-0036, [RFC-0037 'Messages and transcript'](../../../docs/rfcs/RFC-0037-stage-schedule.md#messages-and-transcript), [03 §4](../../../docs/spec/03-data-model.md#4-statement-and-protocol-types) and [RFC-0017](../../../docs/rfcs/RFC-0017-scoring.md#decoding-and-the-sampler-prf), and encodes by it. Its BLAKE3 and `H(domain, bytes)` are those of [`../hash/generate.py`](../hash/generate.py), checked against the official test vectors first. Its inputs come from SplitMix64 with the seed the script states, so `python3 spec/vectors/protocol/generate.py` reproduces the file byte for byte.

Before it writes anything, a separate decoder of the same schemas, which applies RFC-0003's checks in the order RFC-0003 'Limits' fixes, checks every output:
- every message decodes back to its value, and every proper prefix of it is `ERR-CODEC-TRUNCATED` at the offset of the first check that needs a byte past the prefix, which the encoder records in decoding order and the decoder must find too;
- every refused input fails with the code and offset it was built for, at the place the encoder recorded;
- every envelope opens to its value, and every refused envelope fails as it was built to.

## Sections

| Section | A case | Cases |
|---|---|---|
| `messages` | `name`, `type`, `value`, `payload` (its TCE bytes), `truncated` and `envelope`; a `Round1` adds `statement_digest`, `H("statement", TCE(statement))` | every message; `Round1` with each predictor and decoding, `ColumnQueries` with and without matrices |
| `coin_toss` | `verifier_share` `s_V`, `prover_share` `s_P`, `commitment` `H("coin-commit", s_V)` and `seed` `H("coin-toss", s_V ‖ s_P)` | the messages' shares, all-zero and all-one shares, random ones |
| `refused` | `type`, `hex`, `code`, `offset`, `why`, and `limits` when the case has its own | trailing bytes, truncation, enum tags, rationals, lengths, field elements, matrix ids, depth, total size |
| `accepted` | `type`, `hex`, `limits` and `why` | each message at its deepest level's `max_depth`; inputs exactly at `max_total_bytes` and `max_vec_len` |
| `envelopes` | `schema`, the message a reader expects, `hex`, `code`, `offset` and `why` | schema versions 2 and 0, another schema, a payload refused, a changed payload, a short digest |

`truncated` gives the offset of every proper prefix's `ERR-CODEC-TRUNCATED`, as runs `[length, offset]`: the prefixes from `length` bytes up to the next run's length fail at `offset`, the first run's length being 0.

Every case is read under the file's `limits`, RFC-0003's defaults for messages, unless it states its own. `schemas` gives each message's schema id and version ([03 §5](../../../docs/spec/03-data-model.md#5-artifact-catalog)), and `matrices` the names of the weight matrices in their declaration order.

## Values

| Type | JSON |
|---|---|
| `u8`, `u16`, `u32`, `u64`, a matrix id | decimal strings |
| `Fp` | a decimal string, its canonical value |
| `Fp2` | `[c0, c1]` |
| `[u8; 16]`, `[u8; 32]`, `Digest` | lowercase hex strings |
| `Rational` | `{"num": "…", "den": "…"}` |
| `Vec<T>` | arrays |
| A struct | an object keyed by field name, in declaration order |
| A struct of one unnamed field, `StageId` or `FixedBits` | its field's value |
| An enum value | `{"Variant": fields}`: `null` for a unit variant, an object for a struct variant |

## Rules beyond RFC-0003

- **Matrix ids,** as [RFC-0036 'Matrix ids'](../../../docs/rfcs/RFC-0036-protocol-messages.md#round2response) states them. A `u8` matrix id is the declaration index of [RFC-0031](../../../docs/rfcs/RFC-0031-commitment-id.md#the-descriptor)'s `WeightMatrix`, 0 to 9. In `Round2Response.weight_combined`, `ColumnQueries.weights` and `Round3Response.weight_columns` the ids strictly increase. A decoder reads each entry's id, its first byte, and checks it before the rest of the entry, as a map's key is checked: an id above 9, or one not above the previous entry's, is `ERR-CODEC-NONCANONICAL` at that byte.
- **Depth.** `StageId` and `FixedBits` are structs, so each takes a level of its own; `session_id`, the shares and every `Digest` are arrays, one level each.
- **Not decoding's.** Every length the schedule or the statement fixes, such as the stages' count or the scoring claims', is checked after decoding, and so is the protocol version `Round1` names (`ERR-PROTO-VERSION`): none of them is a decoding error.
