# Audit sampling vectors

`generate.py` writes `audit.json` from [RFC-0016](../../../docs/rfcs/RFC-0016-audit-sampling.md) ('Policy', 'Sample derivation', 'Batching') and [RFC-0063 'Which requests a period draws from'](../../../docs/rfcs/RFC-0063-request-boundaries.md#which-requests-a-period-draws-from), with RFC-0059's frame lengths, RFC-0004's domains and RFC-0003's encoding and envelope. It shares no code with `crates/`, `verifier/` or `python/`. Its BLAKE3 is [`hash/`](../hash/generate.py)'s, checked against the official vectors first.

| Section | What it pins |
|---|---|
| `rates`, `refused_rates` | A policy file's decimal rates and their thresholds, `⌊rate · 2^64⌋` capped at `2^64 − 1`, and texts that are no rate |
| `candidates` | Random logs of a period and, in most cases, the one before it, as entries of five kinds; each period's candidates, ended or not, with their lengths. A frame whose `length` is not `22 + 4n` makes its request's length `2^64 − 1` |
| `seeds` | Coin tosses over a manifest digest, the sample seed `H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`, and the selection value of each key, `u64_le(H("audit-select", seed ‖ session_id ‖ u64(request_id))[0..8])` |
| `samples` | Candidates with their lengths, the sample a policy draws from them, its TCE `payload`, every proper prefix `ERR-CODEC-TRUNCATED`, and its envelope (schema `0x0403`, version 1) |
| `refused`, `accepted` | Payloads a reader refuses, each with its code and offset, which the oracle's own reader finds; and payloads at the edges of the rules, which it accepts |
| `batches` | Requests with groups and tokens, and the batches first-fit decreasing packs under each bound, checked against RFC-0016's bounds |

The candidates' entries are written by kind: `open`, `carry` and `close` name a session; `frame` adds its request, direction, index and ciphertext length; `end` adds its request. A test builds each period's `DigestLogEntry` values from them, numbering `seq` from 0 across the two periods, and checks them against a manifest it signs over them.

`crates/tessera-protocol/tests/audit.rs` checks the rates, seeds, values, samples, refusals and batches; `crates/tessera-anchor/tests/period.rs` checks the candidates.
