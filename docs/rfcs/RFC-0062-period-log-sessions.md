# RFC-0062: The Period Log and Its Sessions

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0015](RFC-0015-transcript-anchor.md#deployment-assumptions):
    - 'Goals' and 'Deployment assumptions' (A-ANCHOR-1, A-ANCHOR-2): the period's log, the recorder's carries, and who writes on the recorded path;
    - 'Log format': the `SessionCarry` entry, and a request's entries read from its period's log;
    - 'Verifier checks' 2, 'Prover pre-check' and INV-ANC-2: the session rules over the period's log;
  - [RFC-0016](RFC-0016-audit-sampling.md#sample-derivation) 'Sample derivation': the manifest's signature is checked first, and the sample is derived from the verified log;
  - [RFC-0060](RFC-0060-log-manifests.md) 'Scope': where a chain of manifests is required (nowhere, for the anchor);
  - [RFC-0059](RFC-0059-frame-headers-and-nonces.md#session-identity) 'Session identity' and 'Verifier check 2, amended': what it left to #498, and which request and kind a break names;
  - [RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies) 'Which pin applies': a request's frames are its frames in the period's log;
  - [RFC-0019](RFC-0019-verifier.md#inputs) 'Inputs' and 'Check order' steps 1 and 4: the verified log replaces the batch's entries and their inclusion proofs;
  - [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Malformed', 'Anchor' and [04 §2](../spec/04-error-model.md#2-verdicts): `Inclusion` is retired, `Frames` covers the new frame rules, and `Session` is new;
  - [04 §3](../spec/04-error-model.md#3-error-codes): `ERR-CODEC-DIGEST` covers a period's log.
- Origin: [#498](https://github.com/AbdelStark/tessera/issues/498)
- Amended by: [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from):
  - 'The period's log', 'Security Argument' and 'Drawbacks': a request's frames span its end's period and the one before, which settles requests that cross a boundary;
  - 'The session rules': they hold in both periods of a request's window;
  - 'Testing Strategy': the first unknown tag is 5.

## Summary

A batch's frames are bound to their session's open entry: its key commitment keys the AES circuit (RFC-0015, RFC-0059). A batch carries its requests' entries and their sessions' opens, each with an inclusion proof against the statement's manifest. #137's reviews found that nothing says what the log must show of a session:
- **Zero or several opens.** With two, the prover chooses which key commitment becomes a public input.
- **Duplicate frames.** Two frames can carry one `(session_id, request_id, direction, frame_index)` with different counters. The prover supplies the one it can prove, and contiguity, distinct counters and inclusion all pass.
- **Sessions across periods.** A session opened in period `k` and used in period `k + 1` has its open in another manifest than the statement's.

An inclusion proof shows that an entry is in the log, never that another is not; short of the whole log, no set of them shows that a session has one open or a frame no duplicate. This RFC has the verifier hold the period's whole log, which it already needs to derive the audit sample (RFC-0016, INV-AUD-2):
- **The verified log.** The verifier checks the period's log against the manifest's root, count and `seq` range once, before it derives the sample, and keeps that log. Every later use reads that copy. A batch supplies no entries, and the audit needs no inclusion proof.
- **A carry entry.** At the start of each period, the recorder logs a `SessionCarry` for every session still open, with its key commitment, so each period's log holds the key of every session it uses. The recorder only adds entries.
- **The session rules.** In a period's log, a session used in it has exactly one open or carry, before its other entries; no two of its frames share a position; and no two share a counter.

## Motivation

[#498](https://github.com/AbdelStark/tessera/issues/498).
- **Two opens.** Each open commits a key. A prover holding two keys for one session decrypts each recorded ciphertext under the one that suits it, and supplies that open with its proof: the frames' tokens then depend on a choice made at proof time.
- **Duplicate frames.** A serving stack that sends two frames for one position, say one with the tokens it served and one with the tokens `D` would produce, has both logged. It proves the one that passes. Every check of RFC-0015 and RFC-0059 passes, since the batch omits the other.
- **Sessions across periods.** RFC-0015 requires the batch's entries to verify against the statement's manifest. An implementation that accepted an earlier manifest's open without its chain could be handed an open from any log.
- **Recorder rules.** RFC-0059 refused a rule such as "drop frames without an open": it lets frames be served without being logged. A passive tap cannot drop anything either. The rules here are the verifier's, over what the recorder logged.

## Goals

- **One key per session per period**, fixed by the period's log.
- **One frame per position**: a request's frames in a period are exactly its frames in the log.
- **One verified log**: every use of a period's entries, selection included, reads the copy the verifier checked.
- **No recorder rule that removes anything** from the log.

## Non-Goals

- **Requests whose frames span two periods**, and frames of a request logged after its period ends: #496.
- **The order of a request's inbound against its outbound frames**: #496, as its comment there says.
- **Counters across periods.** INV-ANC-3 across a period boundary stays the serving stack's ('The session rules').
- **The verifier's, the recorder's and the pre-check's implementations**: #150, #139 and #146.

## Proposed Design

### The period's log

- **What it is.** The entries of a manifest, `seq` from `first_seq` to `first_seq + entry_count − 1`, in `seq` order, each `DigestLogEntry` in its TCE. It is public (03 §5).
- **On disk.** A period's log is the concatenation of its entries' TCE encodings, with no envelope and no length prefix. Each entry's length is fixed by its tag, so the stream cuts one way. The reference recorder writes it as `<period_start_ns>.tdl` (RFC-0015 'Reference recorder'). Its integrity comes from the manifest's root, not from the file.
- **How it arrives.** The recorder delivers each period's log, with its manifest, to the verifiers on a path that does not pass through the prover (A-ANCHOR-2). A prover that could withhold the log of a period could keep that period from being audited at all.
- **Coverage.** A statement needs no chain of manifests, but the audit's coverage does.
  - **Linking.** The verifier links each manifest it receives to its predecessor (RFC-0060 'The chain').
  - **Unaudited periods.** A gap in the chain is an unaudited period. So is a period whose verified log the verifier does not hold within its policy's `log_deadline_ns` of the end of the last period it holds.
  - **Reporting.** The verifier reports each unaudited period, by the manifest it has or by the gap's place in the chain (OQ-0062-2).
- **The check.** The verifier recomputes RFC 9162's root over the log's leaves `SHA-256(0x00 ‖ TCE(entry))` (RFC-0060) and requires:
  - exactly `entry_count` entries;
  - the `i`-th entry's `seq` equal to `first_seq + i`;
  - the root equal to `entries_root`.
- **When.** Once per manifest, after the manifest's signature has passed under a pinned recorder key and before the verifier derives the audit sample (RFC-0016), before it uses any entry.
- **The verified log.** The check yields a `VerifiedLog`, the only form in which the verifier uses a log: the entries and the manifest's digest, which nothing but the check constructs. Selection, the anchor checks of every batch and the report all read it. RFC-0019 step 4 requires its digest to equal the manifest's identity, `statement.manifest_digest` and the sample's. A statement that names another manifest than the sample's is `StatementMismatch` (RFC-0016 check 1). A `VerifiedLog` that does not match the manifest the verifier was given is the verifier's own input, `Inconclusive(ERR-USAGE)`. A pass bit alone would let a later use read entries no check covered.
- **The store.** The verifier keeps each `VerifiedLog` beside its manifest, as it keeps its ledger. When it loads one, it checks the manifest's signature under its pinned recorder keys and the log against the manifest again, so a store rewritten whole is refused. The store is the verifier's own state, as trusted as the ledger.
- **Its fault.** The log is the verifier's own input, fetched before any proof exists. A log that fails to decode is `Inconclusive(ERR-CODEC-*)`. One that fails the check is `Inconclusive(ERR-CODEC-DIGEST)`. No sample is derived from it, and no proof of its period is judged until a verified copy is held.
- **A request's entries.** A request `(session_id, request_id)` has as its frames every `Frame` entry of the verified log with that session and request, and [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from) extends them to its window across a boundary. Its session's key commitment is the session's open or carry in the log. The batch supplies no entries of its own.
- **Its lengths.** RFC-0015 check 2's "every frame present" now reads over those frames. The statement's `prompt_len` and `output_len` equal the token counts of all the request's `In` and `Out` frames in the period, so no frame inside the period can be left out. A request's frames in the period before are [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from)'s.
- **Inclusion proofs.** RFC-0060's proofs stay defined, for parties that hold one entry, such as a client checking its own request. The audit no longer uses them.

### The carry entry

`DigestLogEntry` gains a variant, tag 3:

```rust
SessionCarry { seq: u64, t_ns: u64, session_id: [u8; 16], key_commitment: [u8; 32] },
```

- **When the recorder writes it.** At the start of each period of non-zero length, before the period's other entries, in `session_id` order, with `t_ns` the period's start. It writes one for every session that, in the previous period of non-zero length, kept rule 1 of 'The session rules' and has no close, and that had its open or a frame in that period or in the one of non-zero length before it.
- **What it copies.** The `key_commitment` of that open or carry.
- **Idle sessions.** A session with neither its open nor a frame in two consecutive periods of non-zero length is no longer carried, and its later requests cannot be proven.
  - **Determinism.** The rule depends only on the log, so a serving stack can tell when a session has lapsed and must open a new one.
  - **The margin.** Where a frame lands, on either side of a boundary, depends on clocks and path delays. A stack should treat a session as lapsed at the second boundary after its last frame or open was sealed, less the recorder's `ε`.
  - **Short periods.** Lapses count periods, not time, so short periods lapse sessions sooner.
- **A zero-length period** holds no entries, since each entry lies in its period (RFC-0060), so it carries nothing. The next period carries the sessions open at its start.
- **Why a carry and not an older manifest.** An open in an earlier period would need that period's manifest, the chain of manifests between, and their logs, to show that no second open came between. A carry puts the session's key in the period that uses it.

### The session rules

For a session `s` with an entry in the period's verified log:
1. **One key.** Exactly one `SessionOpen` or `SessionCarry` entry for `s`, with a lower `seq` than every other entry of `s` in the period.
2. **One frame per position.** No two `Frame` entries of `s` with one `(request_id, direction, frame_index)`.
3. **One use per counter.** No two `Frame` entries of `s` with one `frame_seq`.

The rules are properties of the session in the period, checked in their order, so a session that breaks rules 1 and 3 is named under rule 1.
- **Who is named.** The verifier checks requests in `statement.batch` order. For each, it checks its session's rules first, then RFC-0059's checks of its own frames, then its sampler key (RFC-0061). The first failure is the verdict: `Reject(Anchor { request_id: Some(r), kind })`, `r` the request being checked.
- **The kinds.** Rule 1 is `Session`, a new kind (tag 4). Rules 2 and 3 are `Frames`.
- **One verdict.** A session that breaks a rule names the first request of the batch in that session, whichever frames of the session broke it. With [RFC-0063](RFC-0063-request-boundaries.md#the-requests-rules)'s window, a break in one period names the first request of the batch whose window holds a frame in that period ('Who is named'). In RFC-0059's 'Verifier check 2, amended', this replaces two things. The third item, the open before the session's frames, is now rule 1 and `Session`. The naming sentences, under which the later of two requests sharing a counter was named, give way to this rule.
- **What the rules do not include.**
  - **A close.** A session's close ends its carries. A frame logged after its close, on the other path, is still bound to the period's open or carry, so a close needs no rule.
  - **Counters across periods.** The reference cipher never reuses one (INV-ANC-3), but RFC-0059 lets inbound and outbound frames reach the recorder in an order other than their sealing. A bound that a carry set from the period before could then refuse an honest frame sealed before the boundary and logged after it. Counters are confidentiality, not soundness, so the rule stays within a period.
- **A request that breaks a rule** cannot be proven. A sampled one is the prover's failure in the audit report (RFC-0016): `Rejected` if a proof includes it, `Unproven(DeadlineMissed)` if none does. Every entry on the recorded path is the serving stack's to send (A-ANCHOR-1), so a duplicate or a stray open is not an excuse.

### What the rules give

- **The key.** Rule 1 leaves each session one `key_commitment` in the period, which the circuits take as a public input.
- **The frames.** Rule 2 leaves each position of a request one frame, so the request's ciphertexts, and through the key its tokens, are fixed by the log. With the lengths taken over the period's frames, none of them can be left out.
- **The counters.** Rule 3 extends RFC-0059's distinct counters from the batch's frames to all the session's frames in the period.

### The pre-check

RFC-0015 'Prover pre-check' now reads the verified log as the verifier will. For each request it is to prove, it checks the session rules and the lengths, and recomputes every frame digest and the session's key commitment against the log's entries. A failure is `ERR-WITNESS-ANCHOR`, and the batch is not proven.

## Security Argument

- **No choice of log.** The log hashes to the signed `entries_root` with `entry_count` leaves and contiguous `seq`, so no entry can be added, removed or reordered (RFC-0060). The verifier checks it once and every use reads that copy, so no later input can stand in for it.
- **No choice of key.** Rule 1 leaves one commitment per session per period, and the verifier takes it from the log itself.
- **No choice of frames.** Rule 2 leaves one entry per position, and the verifier reads the request's frames from the log, with the lengths over all of them. A batch cannot omit a frame of the period.
- **The carry.** It comes from the recorder, honest by A-ANCHOR-2. A wrong or missing carry can make a session's later requests unprovable. It cannot bind a frame to a key the session did not commit, as long as the recorder copies faithfully.
- **Fail-closed.** Every rule refuses what the prover could choose between, and none removes a frame from the log.
- **What was left: requests that cross a boundary.** A request read in each period alone had an earlier part that passed as the whole request, and a later part that refused an honest prover. [RFC-0063](RFC-0063-request-boundaries.md) settles both with a request-end entry and a window of two periods.
- **Where a chain is needed.** With carries, each statement needs its own manifest and that manifest's log, and no chain of manifests, under A-ANCHOR-2. The audit's coverage does need one, to find a period that never arrived ('Coverage'). That answers where RFC-0060 left a chain required.

## Alternatives Considered

1. **Non-membership proofs.** Proving "no other open" or "no duplicate" against a sorted tree, such as a sparse Merkle tree keyed by position. It needs another tree from the recorder, while the verifier already holds the log. Rejected.
2. **Recorder rules that drop.** Dropping a second open or a duplicate frame, or frames without an open, lets frames be served without being logged (RFC-0059). Rejected.
3. **Opens from earlier manifests.** It needs the chain of manifests and their logs back to the open. Rejected for the carry.
4. **The first open or the first frame wins.** Under rule 1 or 2, the prover would choose which one comes first, since it sends them. Rejected for refusing both.
5. **A counter carried across periods** (this RFC's first draft). It refuses honest frames that reach the recorder out of their sealing order across a boundary. Rejected; counters stay within a period.
6. **A log supplied with each batch, checked and cached as a pass** (the first draft). A later batch could supply other entries under the cached pass. Rejected for one verified copy.

## Drawbacks

- **The verifier holds the whole period.** At 10^7 entries of about 88 bytes, that is about 0.9 GB per period. RFC 9162's tree then takes `2·10^7 − 1` SHA-256 calls, about `4·10^7` compressions, once per manifest. RFC-0016's selection read the log already.
- **Carry entries.** One per open session per period.
- **Idle sessions** lapse after two periods with neither an open nor a frame, and a serving stack must open new ones.
- **One broken session** makes every request of it in the period unprovable.
- **Requests in flight at a boundary.** Before RFC-0063, two things went wrong:
  - a sampled later part of a request that spans two periods could not be proven, and counted against an honest prover;
  - its earlier part could be accepted while its tail escaped.

  RFC-0063 settles both.

## Migration / Rollout

- **#498, with this RFC.**
  - `tessera-anchor`: `DigestLogEntry::SessionCarry` and its encoding.
  - The vectors of `spec/vectors/anchor/` and `manifest/`.
- **#150.**
  - The verified log and its store.
  - The session rules and the lengths.
  - `AnchorFailure::Session`, and `SamplerKey` from RFC-0061, in `tv-core` and in `spec/schemas/verdict-v1.schema.json`.
- **#139**: the recorder's carries and lapses.
- **#146**: the pre-check.

## Testing Strategy

- **Encoding.** `SessionCarry` round trips, and its TCE matches the oracle's. Tag 5 is refused (RFC-0063's `RequestEnd` took tag 4).
- **The rules (#150).** Each break alone, with the request it names:
  - a session with no open or carry, with two opens, with an open and a carry, with its open after a frame;
  - two frames at one position;
  - two frames with one counter, including across two requests of the batch.
- **Not refused.**
  - A frame logged after its session's close in the same period.
  - A carried session's frame whose counter is below a frame's of the previous period.
- **The log (#150).** One entry missing, one added, two swapped, a wrong count: each refused at load, `Inconclusive(ERR-CODEC-DIGEST)`. No proof of that period is judged.
- **The lengths.** A statement whose lengths cover fewer frames than the period holds for the request is refused, `Frames`.
- **The carries (#139).** A session idle for two periods is not carried. A session whose period broke rule 1 is not carried. A zero-length period carries nothing.

## Open Questions

- **OQ-0062-1.** Should the verifier check the rules for every session of the period and report the breaks of sessions no batch includes? They show a serving stack's misbehaviour without a proof. Owner: @AbdelStark. Resolve with the audit report's format in #150.

- **OQ-0062-2.** How should the audit report record an unaudited period, a gap in the chain or a log not verified by `log_deadline_ns`? `tessera.audit-report/1` is per sample, and such a period has none. What should `log_deadline_ns` default to? Owner: @AbdelStark. Resolve with #150.

## References

- [RFC-0015](RFC-0015-transcript-anchor.md); [RFC-0016](RFC-0016-audit-sampling.md); [RFC-0019](RFC-0019-verifier.md); [RFC-0043](RFC-0043-rejection-payloads.md); [RFC-0059](RFC-0059-frame-headers-and-nonces.md); [RFC-0060](RFC-0060-log-manifests.md); [RFC-0061](RFC-0061-sampling-randomness.md).
- RFC 9162 §2.1.
