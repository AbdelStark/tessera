# RFC-0063: Request Boundaries in the Log

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0015](RFC-0015-transcript-anchor.md#frames):
    - 'Deployment assumptions', A-ANCHOR-2: the recorder sequences inbound frames before the cluster can read them, and logs request ends;
    - 'Frames' and 'Log format': the request-end frame and its `RequestEnd` entry, and the order of a session's close;
    - 'Verifier checks' 2 and 'Prover pre-check': a request's window across a period boundary, its one end, and its inbound frames before its outbound ones;
  - [RFC-0016](RFC-0016-audit-sampling.md#sample-derivation) 'Sample derivation' and INV-AUD-2: the requests a period draws from, their lengths, and the previous period's log;
  - [RFC-0019](RFC-0019-verifier.md#inputs) 'Inputs' and 'Check order' steps 1 and 4: the previous period's verified log;
  - [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Anchor': `Frames` and `Session` over a request's window;
  - [RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies) 'Which pin applies': containment over the request's window;
  - [RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log) 'The period's log', 'The session rules', 'Security Argument', 'Drawbacks' and 'Testing Strategy': the rules over a request's window, and the tag of `RequestEnd`.
- Origin: [#496](https://github.com/AbdelStark/tessera/issues/496)

## Summary

The verifier reads a request's frames from the period's verified log, and the statement's lengths count all of them (RFC-0062). No frame inside a period can be left out. What remains is the period boundary. A request whose frames cross it is read in each period alone:
- **Its later part** lacks the frames before the boundary. Contiguity from `frame_index = 0` fails, and an honest prover is refused.
- **Its earlier part** starts at `frame_index = 0`, and its lengths count its own frames. It passes as though it were the whole request, and the tokens after the boundary escape the proof. Nothing in the log says where a request ends.

This RFC marks the end and reads a request across the boundary:
- **The request's end.** The serving stack sends a cleartext request-end frame after the request's last frame, and the recorder logs it as a `RequestEnd` entry.
- **A request belongs to the period of its end.** Its frames lie in that period and the one before. The verifier reads both periods' verified logs, and applies RFC-0062's rules to both.
- **Every request the log shows is drawn from.** A request with an end in a period is a candidate of that period, whatever comes before or after the end. One with a frame and no end in time is an unterminated candidate. Every logged frame lies in a candidate the sample draws at its bucket's rate, so a request cannot escape the audit by where it places its end, or by leaving it out.
- **Inbound before outbound.** Every inbound frame of a request is logged before its first outbound one. The recorder sequences inbound frames before the cluster can read them.

## Motivation

[#496](https://github.com/AbdelStark/tessera/issues/496), with its comments:
- **Trailing frames.** RFC-0015 asked for "every frame of every sampled request present", which no batch could show. RFC-0062 settled that within a period. Where a request ends, which no entry records, still decides what "every frame" means at a boundary.
- **The cheated tail.** A prover can serve an honest first part before a boundary and a cheated tail after it. The first part is accepted as the whole request, and the tail fails as an honest tail would.
- **Late inbound frames.** No rule orders a request's inbound frames against its outbound ones. The serving stack, which seals inbound frames itself, can serve outputs it chose, then log a late inbound frame whose tokens make those outputs `D`'s.

## Goals

- **One extent per request.** A request's frames are fixed by the log, wherever a boundary falls.
- **Honest requests in flight at a boundary are provable.**
- **No escape.** Every logged frame lies in a candidate the sample can draw, wherever the prover puts a request's end, or whether it sends one.
- **The sample still depends only on what was logged before its seed** (INV-AUD-1).

## Non-Goals

- **Requests longer than a period.** A request whose frames span three periods of non-zero length cannot be proven. Periods are hours long, requests seconds or minutes.
- **The verifier's and the recorder's implementations**: #150 and #139.

## Proposed Design

### The request's end

```text
RequestEndFrame = b"TRQE" ‖ session_id ‖ u64_le(request_id)      28 bytes, cleartext
RequestEnd { seq: u64, t_ns: u64, session_id: [u8; 16], request_id: u64 }      DigestLogEntry, tag 4
```

- **Who sends it.** The serving stack, on the outbound path, after the request's last frame in either direction. The recorder logs it like a session open, copying its fields.
- **Request ids.** A session never reuses a `request_id`. A reuse within one window breaks rule 1 or 5. A reuse two periods apart makes two candidates with disjoint frames, each its own ledger entry.
- **A session's close** goes on the outbound path after the session's last request end.
- **Why a frame of its own.**
  - **Not an empty frame.** An empty outbound frame could have marked the end, but RFC-0059 allows empty frames anywhere, and a stack that sent one mid-stream would end its request early.
  - **Not a header flag.** A flag in the frame header would change RFC-0059's header and entry for every frame.

### The periods

For a period `P` of non-zero length, `P − 1` names the last period of non-zero length before it in the chain of manifests (RFC-0060), reached through `prev_manifest` and any zero-length manifests between. A period of zero length holds no entries (RFC-0060) and has no candidates. A period with no period of non-zero length before it has no `P − 1`.

### Which requests a period draws from

The candidates of a period `P`, from which RFC-0016 selects, are of two kinds:
1. **Ended:** every `(session_id, request_id)` with a `RequestEnd` in `P`.
2. **Unterminated:** every `(session_id, request_id)` with a frame in `P − 1` and no `RequestEnd` in `P − 1` or `P`.

**A candidate's frames** are all its `Frame` entries in `P − 1` and `P`, wherever they lie against its end. Its length `L` (RFC-0016) is their token count, the same count its statement must carry. Where the prover puts an end therefore cannot move the request between buckets.

Three properties follow:
- **Every frame is drawn from.** Take a frame of `r` in period `Q`, and `Q + 1` the next period of non-zero length.
  - **If `r` has an end in `Q`,** the frame is among the frames of `r`'s candidate of `Q`: before the end and in its window, or after it and refused by rule 2.
  - **Otherwise, if `r` has an end in `Q + 1`,** the frame is among the frames of that candidate.
  - **Otherwise** it is among the frames of an unterminated candidate of `Q + 1`.

  So every logged frame belongs to a candidate that the sample draws at its bucket's rate.
- **The seed.** The candidates of `P` and their lengths depend on the logs of `P − 1` and `P` alone, both signed before `P`'s coin toss (INV-AUD-1).
- **A pair in two periods.** A request can be a candidate of two periods: one whose frames span three periods, one with two ends, one with a tail after its end, or an unterminated one whose frames lie in two periods. Each draw is independent and its own ledger entry (INV-AUD-3). Whichever is drawn, no frame lies in two accepted candidates, since rules 1 and 2 refuse a candidate whose frames another end bounds.

### The previous period's log

- **What the verifier holds.** To derive `P`'s sample and to check its batches, it holds `P − 1`'s verified log (RFC-0062) beside `P`'s.
- **Binding.** The two are bound by the chain: `P`'s `prev_manifest` names `P − 1`'s manifest, through any zero-length periods between, which the verifier also holds. A `previous_log` that is not that period's is the verifier's own input, `Inconclusive(ERR-USAGE)`.
- **No previous period.** A period with no period of non-zero length before it has no previous log.
- **Order.** No sample of `P` is derived before `P − 1`'s log is verified. If the verifier does not hold `P − 1`'s verified log, `P` is unaudited. `P`'s own log, once verified, still serves `P + 1`, so one missing log leaves out two periods, not every later one.

### The request's rules

For a candidate `r` of `P`, over its frames in `P − 1` and `P`:
1. **One end.** Exactly one `RequestEnd` for `r` in `P − 1` and `P`, and it lies in `P`.
2. **Before its end.** Every frame of `r` has a lower `seq` than its end, and there is at least one.
3. **Inbound first.** Every inbound frame of `r` has a lower `seq` than its first outbound frame.
4. **Its session in both periods.** For each of `P − 1` and `P` that holds a frame of `r`, the session keeps RFC-0062's rules in that period. Where both do, the session's open or carry in each carries the same key commitment. A period that holds none of `r`'s frames is not checked for it.
5. **One frame per position across the boundary.** No two frames of `r` share a `(direction, frame_index)`. In each direction, the indices are exactly `0, …, n − 1`, each once.

Then RFC-0015's check 2 reads over these frames: the lengths, and plaintext lengths (RFC-0059).
- **The kinds.** Rule 4 is `Session` where RFC-0062's rule 1 fails, or where the two periods' key commitments differ. RFC-0062's rules 2 and 3 keep their own kind under rule 4, `Frames`, as RFC-0062 'The session rules' names them. Rules 1, 2, 3 and 5 are `Frames`.
- **The order.** The verifier checks requests in `statement.batch` order. For each, it checks rule 4 with RFC-0062's session rules first: in `P − 1`, then in `P`, then the key commitment across both. Then come the other rules and RFC-0059's frame checks, then the sampler key (RFC-0043's order).
- **Who is named.** The first request, in batch order, whose check fails. A session's break in one period is seen only by requests whose window holds a frame in that period (rule 4), so it names the first such request of the batch, which may follow other requests of the same session.
- **One counter across the window.** RFC-0059's batch rule holds over both periods: no two frames of the batch's requests in one session share a `frame_seq`, in `P − 1` and `P` together. A break is `Frames` and names the session's first request in the batch. An honest stack never reuses a counter (INV-ANC-3), so this refuses no honest batch; lower counters in a later period, as a carried session's reordered frames may show, are not a break.
- **Unterminated candidates** break rule 1, and a candidate whose end comes before all its frames breaks rule 2.

### Inbound before outbound

- **The duty.** The recorder sequences each inbound frame, assigning its `seq`, before the cluster can read it: it stores and forwards inbound traffic. A passive tap whose copy of the inbound path may lag the cluster does not meet this duty.
- **Why rule 3 then holds.** An honest request's outbound frames are produced after the cluster has read its inbound frames, so rule 3 holds for every honest request. The request end, sent after the request's last frame, comes after both.
- **What it gives.** Rule 3 refuses an inbound frame logged after outputs were served, which would turn those outputs into `D`'s.
- **What it does not.** The serving stack seals inbound frames itself. It can still compute outputs privately, search for a prompt suffix that yields the outputs it wants, and log the prompt and suffix before serving. Rule 3 makes that search cost the request's latency; it does not prevent it. Binding prompts to the client's text is outside the anchor (A-ANCHOR-1).

## Security Argument

- **One extent.** Rules 1, 2 and 5 put a request's frames between the start of its window and its one end, each position once. Neither a part of a request nor its frames past the end pass as the whole request.
- **No escape.** Every frame belongs to a candidate ('Which requests a period draws from'):
  - an end placed before the request's frames leaves a candidate that rule 2 refuses;
  - an omitted end leaves an unterminated candidate, drawn at its bucket's rate and unprovable.

  Cheating in any frame is caught with the probability of any cheated request.
- **One key across the boundary.** Rule 4 applies RFC-0062's rules in both periods of the window, so the session has one key in each, and the two are equal.
- **A late inbound frame** after the outputs is refused by rule 3, by the log's own `seq`, under the recorder's ingress duty.
- **The seed.** The candidates and their lengths come from entries signed before the seed is drawn, so the prover cannot shape a request's bucket after learning it.
- **Fail-closed.** Every rule refuses, and none removes a frame from the log.

## Alternatives Considered

1. **A request belongs to the period of its first frame.** Its length, and so its bucket, would depend on the next period's log, signed after the seed of its own. The prover could then lengthen a request's tail to move it between buckets. Rejected.
2. **Ended candidates that need a frame before the end** (this RFC's first draft). An end logged before the request's frames left the request in no candidate of any period. An exhaustive search over logs of up to five events found 320 to 470 such logs, depending on which end is chosen. Rejected: every end makes a candidate, and the same search finds no escape.
3. **An empty outbound frame as the end.** It overloads frames RFC-0059 allows anywhere. Rejected.
4. **A final flag in the frame header.** It changes every header and entry. Rejected for a frame of its own.
5. **A request confined to one period, with the stack holding requests near a boundary.** It costs latency at every boundary, and a request still in flight would be lost. Rejected.
6. **Counts in the end frame.** The counts of inbound and outbound frames would repeat what rule 5 and the lengths already fix. Rejected.

## Drawbacks

- **Two periods' logs.** The verifier holds the previous period's verified log while it audits a period, and derives no sample of a period before the previous one is verified.
- **One more frame per request**, 28 bytes, and one more entry.
- **Long requests.** A request that spans three periods of non-zero length cannot be proven.
- **The recorder's ingress duty.** A passive tap that may lag the cluster on the inbound path is not enough.

## Migration / Rollout

- **#496, with this RFC.**
  - `tessera-anchor`: `RequestEndFrame` and `DigestLogEntry::RequestEnd`.
  - The vectors of `spec/vectors/anchor/` and `manifest/`. `manifest/`'s first period holds a request whose end lies in the second.
- **#150**: the candidates, the previous log and its binding, and the rules.
- **#139**: the recorder logs request ends and stores and forwards inbound frames.
- **#146**: the pre-check reads the window as the verifier will.

## Testing Strategy

- **Encoding.** `RequestEndFrame` round trips; 27 and 29 bytes and another magic are refused. `RequestEnd` matches the oracle's TCE. Tag 5 is refused.
- **The candidates (#150).** An independent oracle enumerates logs of a request's frames and ends across periods. The candidate set and lengths match it, and every frame lies in some candidate. The cases include:
  - an end before the request's frames;
  - frames after the end;
  - two ends;
  - no end;
  - a zero-length previous period.
- **The rules (#150).** Each break alone:
  - no end;
  - two ends;
  - an end in `P − 1`;
  - a frame after the end;
  - an end before every frame;
  - a late inbound frame;
  - two key commitments across the boundary;
  - two opens in `P − 1`;
  - one position logged in both periods;
  - a gap in the indices.
- **The window.** A request whose frames lie in `P − 1` and `P` with its end in `P` is accepted, under a temperature pin too (RFC-0061). Its part in `P − 1`, read alone, is not a candidate of `P − 1`.
- **The previous log.** One that is not `prev_manifest`'s is `Inconclusive(ERR-USAGE)`. A log's first period needs none.

## Open Questions

None.

## References

- [RFC-0015](RFC-0015-transcript-anchor.md); [RFC-0016](RFC-0016-audit-sampling.md); [RFC-0019](RFC-0019-verifier.md); [RFC-0043](RFC-0043-rejection-payloads.md); [RFC-0059](RFC-0059-frame-headers-and-nonces.md); [RFC-0060](RFC-0060-log-manifests.md); [RFC-0061](RFC-0061-sampling-randomness.md); [RFC-0062](RFC-0062-period-log-sessions.md).
