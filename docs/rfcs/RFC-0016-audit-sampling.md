# RFC-0016: Commit-Then-Sample Audit Protocol

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies) ('Batching': a batch's temperature requests share one pinned sampler key), [RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log) ('Sample derivation': the manifest's signature first, then the sample from the period's verified log), [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from) ('Sample derivation': the requests a period draws from, ended and unterminated, and their lengths)
- Clarified by: [#140](https://github.com/AbdelStark/tessera/issues/140), the implementation and its review:
  - the encodings of the policy, the coin toss and the sample;
  - the coin toss: the reveal's timing, the policy fixed first, one commitment per manifest, the prover's share and its record;
  - which candidates are selected: lengths past every bucket, and none for a period of zero length;
  - detection at the lowest rate of a bucket and the buckets after it;
  - the batch bounds, as first-fit decreasing meets them
- Amended by: [RFC-0067](RFC-0067-verifier-ledger-encoding.md) ('Audit report': a report from the ledger alone and its exit code; 'Calculators': the verifier's plan without a cost model), [RFC-0071](RFC-0071-request-slots.md) ('Batching': requests are packed by slot class, so that a batch's slots are its requests' own powers of two)

## Summary

Proving every request is not affordable; proving an unpredictable sample of committed requests is. Production traffic is recorded continuously. At the end of each period the recorder freezes and signs the log. Only then do the parties toss a coin for a sampling seed, from which a public, deterministic rule selects requests at per-length-bucket rates. Every selected request must be proven in full, in batches of at least 16K tokens, save at most one per slot class ('Batching', [RFC-0071](RFC-0071-request-slots.md)). The verifier checks that each proof covers only sampled requests and that every sampled request is accounted for. This RFC fixes the policy, the selection rule, batching, the audit report and the detection and overhead calculators.

## Motivation

"Sub-10% is reachable as amortized treaty overhead, through committed transcripts and unpredictable request sampling, not as a per-request cost" (source §1). The source's argument for why this sampling is sound where entry sampling is not: "every request is committed at the recorder before the sample is drawn, and a sampled request is proven in full. A cheater must risk each tampered request independently, so detection grows with the number of violations, not with their size" (source §8). It also requires "sampled requests are proven in batches of at least 16K tokens so the weight pass amortizes" (source §7).

## Goals

- A sample neither party controls and the prover cannot predict before the freeze.
- Deterministic selection both parties compute independently.
- Complete accounting: every sampled request ends as accepted, rejected, unproven or inconclusive.
- Calculators for detection probability and amortized overhead from public parameters.

## Non-Goals

- Choosing rates. Rates are a policy input the parties agree on; Tessera computes their consequences.
- Sampling within a request (rejected by the source for inference, §4).

## Proposed Design

### Policy

```rust
pub struct AuditPolicy {
    pub period_ns: u64,                           // recorder period, for example 24 h
    pub buckets: Vec<RateBucket>,                 // max_tokens strictly increasing
    pub min_batch_tokens: u32,                    // 16384
    pub max_batch_tokens: u32,                    // memory bound, for example 262144; at least 1 and 2 · min_batch_tokens
    pub proof_deadline_ns: u64,                   // time after the sample is fixed
    pub params: ParamSetId,
}
pub struct RateBucket { pub max_tokens: u32 /* max total tokens */, pub threshold: u64 /* rate as threshold over 2^64 */ }
```

Rates are stratified by request length because proving cost rises with context (source §8 table): a policy may sample short requests at 10% and 128K-token requests at 0.4% for the same budget.

- **Encoding.** TCE, the fields in order; a `RateBucket` is its `u32` then its `u64`, the pair the policy's first version wrote. Beyond RFC-0003's checks, a decoder refuses:
  - a bucket whose `max_tokens` is not above the previous bucket's: `ERR-CODEC-NONCANONICAL` at the bucket's first byte;
  - a `max_batch_tokens` of 0, or below `2 · min_batch_tokens`: `ERR-CODEC-LIMIT` at `max_batch_tokens`. 'Batching' needs the factor 2.
- **A threshold** `t` selects with probability `t / 2^64`, so the highest, `2^64 − 1`, leaves out one request in `2^64`.
- **Policy files.** `tessera.audit-policy/1` ([`spec/schemas/audit-policy-v1.schema.json`](../../spec/schemas/audit-policy-v1.schema.json)) is the policy as JSON, each bucket's rate a decimal string. The rate is `0`, `1`, or `0.` or `1.` followed by one to 19 digits, and at most 1. Its threshold is `⌊rate · 2^64⌋`, capped at `2^64 − 1`, computed exactly, so no floating point enters the sample (INV-DM-6).

### Sample derivation

1. The recorder publishes `LogManifest` for the period ([RFC-0015](RFC-0015-transcript-anchor.md#log-format)). The verifier checks its signature under a pinned recorder key, then checks the period's log against it and keeps the verified copy, from which every later step reads the entries ([RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log)). It derives no sample before the previous period's log is verified too ([RFC-0063](RFC-0063-request-boundaries.md#the-previous-periods-log)).
2. The parties run the coin toss of [RFC-0014](RFC-0014-interaction-and-soundness.md#session-order-live-mode) over the manifest: the verifier commits to `s_V`, the prover reveals `s_P`, the verifier reveals `s_V`.
   - **The shares.** Each is 32 bytes, fresh from the party's CSPRNG for this toss; `c_V = H("coin-commit", s_V)`. `s_P` is never a function of the manifest, even with a key of the prover's. A share the verifier could learn before it commits would let it choose `s_V`, and with it the sample.
   - **The policy comes first.** The parties fix the policy before the verifier sends `c_V`, and the prover records it with its answer. A policy changed once the seed is known could steer which bucket a request falls in.
   - **The reveal comes last.** The verifier reveals `s_V` only after step 1: the manifest's signature has passed and it holds both periods' verified logs. A prover that knew both shares before the manifest was signed could choose what to log until the seed suited it.
   - **One commitment per manifest.** The prover answers a commitment once the manifest is published, and answers one per manifest: the first `c_V` it receives for a manifest gets a fresh `s_P`, the same `c_V` again gets the same `s_P`, and any other is refused (`ERR-PROTO-ORDER`). A verifier that aborted after seeing `s_P`, to draw a sample it likes better, draws none.
     - **Kept first.** The prover keeps its answer, `(manifest, policy, c_V, s_P)`, durably before it sends the share, where every responder it runs finds it. Replicas that each answered a different `c_V` would hand the verifier several samples to choose from.
     - **One transcript.** The prover takes the transcript over a manifest from its record, with the revealed `s_V`, and accepts no other. A verifier holding the shares of other tosses, over other manifests or over none, could otherwise present whichever gives the sample it wants: with 1,000 such tosses it could put a request of rate 0.004 in the sample with probability 0.98.
     - **Further verifiers** of one prover adopt the sample of the verifier that tossed, and so trust it, as further verifiers adopt the issuer's pins ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins) 'One issuer'). A tosser that publishes a sample of its own choosing accepts nothing false: the prover proves the sample of its record. Its requests then look unproven to those who adopted the other sample, and the prover holds nothing it can show them. A record the prover signs would settle that; it is the verifier's audit accounting's to define ([#151](https://github.com/AbdelStark/tessera/issues/151)).
   - **The transcript.** Both parties keep `CoinTossTranscript { verifier_commitment: c_V, prover_share: s_P, verifier_share: s_V }`, TCE in that order. Its check is `c_V = H("coin-commit", s_V)`; a reveal that fails it is `ERR-PROTO-COINTOSS`.
3. `sample_seed = H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`, `manifest_digest` the manifest's identity (RFC-0060).
4. Request `(session_id, request_id)`, a candidate of the period ([RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from): it has a `RequestEnd` in the period, or a frame in the previous period and no end in either), with total length `L` over all its frames in the two periods, is selected iff `u64_le(H("audit-select", sample_seed ‖ session_id ‖ u64(request_id))[0..8]) < rate(bucket(L))`.
   - **The bucket.** `bucket(L)` is the first bucket whose `max_tokens` is at least `L`, and `rate` is its threshold.
   - **A period of zero length** has no candidates. It holds no entry, and the next period of non-zero length draws the unterminated requests of the period before it (RFC-0063), so each is drawn once.
   - **Past every bucket.** A candidate longer than every bucket is selected whatever the seed, so that no length escapes the sample. A rate below the last bucket's for longer requests would let a prover lengthen the requests it cheats on. A policy whose last bucket covers the longest context served never meets the case.
   - **A frame with no length.** A frame's tokens are `n` for a `length` of `22 + 4n` ([RFC-0059](RFC-0059-frame-headers-and-nonces.md)). A candidate with a frame of any other `length` has length `2^64 − 1`, past every bucket, and is selected; RFC-0015 check 2 refuses it.
5. The result is the public artifact:

```rust
pub struct AuditSample {                          // schema 0x0403
    pub manifest_digest: Digest, pub policy: AuditPolicy,
    pub coin_toss: CoinTossTranscript,            // c_V, s_P, s_V
    pub selected: Vec<RequestKey>,                // sorted by (session_id, request_id)
}
pub struct RequestKey { pub session_id: [u8; 16], pub request_id: u64 }
```

TCE, the fields in order. `selected` strictly increases: a decoder refuses a key that is not above the one before it with `ERR-CODEC-NONCANONICAL` at the key's first byte.

The domains `audit-sample` and `audit-select` join RFC-0004's domain table.

### Batching

The prover packs selected requests into batches by slot class, then first-fit decreasing, with the request order and tie-breaking fixed by `RequestKey` order, so that batch composition is reproducible. A batch of several requests puts each in a slot of its longest request's power of two ([RFC-0071](RFC-0071-request-slots.md#the-statement-and-the-batching)), so a batch holds requests of one class.

- **Groups.** The requests proven under exact greedy form one group. Until the sampling work (#144) states the sampler and the scoring per slot, every other request is a batch on its own; RFC-0061's sampler keys will then group the temperature requests.
- **Classes.** A request of `L` tokens has class `c = ⌈log2 max(L, 1)⌉`. Within the greedy group, the classes are packed one after another, in decreasing order.
- **Counted tokens.** A batch of `R` requests of class `c` counts `2^{c + ⌈log2 R⌉}` tokens, the rows its token axis spans.
- **First-fit decreasing.** Within a class, requests go in decreasing `L`, ties in `RequestKey` order. Each goes into the first batch, in the order the batches were opened, whose counted tokens stay at most `max_batch_tokens` with it, else into a new batch.
- **A request whose `2^c` exceeds `max_batch_tokens`** is a batch on its own, which takes no other request.
- **The minimum.** In counted tokens, at most one batch of each class has fewer than `min_batch_tokens`. Two such batches would count under `2 · min_batch_tokens ≤ max_batch_tokens` together, so the later one's first request would have fit in the earlier one. In real tokens a class's batches can fall below it.
- **A batch's statement** lists its requests in `RequestKey` order.
- **The CLI.** `tessera audit batch` writes a sample's batches as the `tessera.audit-batches/1` report, each request with the `prompt_len` and `output_len` its statement carries, and `tessera prove --sample --batch K` proves the `K`-th, reading the batches again from the sample and the logs (02 §2.2). Until RFC-0061's pins are implemented, both pack every exact-greedy request in one group, and every other request alone.

### Verifier checks

For each proof:

1. `statement.manifest_digest` equals the sample's `manifest_digest`.
2. Every request in `statement.batch` is in `selected` and has not appeared in an earlier accepted batch.
3. The anchor checks of RFC-0015 pass for every request.

Checks 1 and 2 are statement fields of RFC-0019 step 4: `Reject(StatementMismatch { field: ManifestDigest })` and `{ field: Batch }`, a replayed request included. Check 3 fails as `Reject(Anchor)` ([RFC-0043](RFC-0043-rejection-payloads.md#the-audit-samples-checks)).

**The parameter set.** Before these, the claim list's parameter set must be the one the sample's policy names ([#530](https://github.com/AbdelStark/tessera/issues/530)).
- **Why.** The policy fixes the soundness every proof of the audit gives. A proof under a weaker set, such as `live-64/1` under a policy of `ni-100/1`, would otherwise count toward the audit.
- **Where.** It is RFC-0019 step 2's check, which accepts only the policy's set for a proof of an audit sample. Another set is `Reject(Params { name })`, naming the claim list's. The accounting runs it first ([RFC-0067](RFC-0067-verifier-ledger-encoding.md#the-accounting)), so no proof of another set is recorded.
- **The prover** refuses to prove a sample's batch under another set, with `ERR-USAGE` before anything is committed (`tessera prove --sample`).
- **A set the verifier does not allow.** The step's other half refuses it, for the claim list and so for the policy that names it: no proof of such an audit is accepted, whatever its parameter set. The accounting compares names only, after acceptance.

At the deadline, the verifier emits the audit report.

### Audit report

```rust
pub enum RequestOutcome { Accepted { uhat: FixedBits }, Rejected(RejectReason), Unproven(UnprovenReason), Inconclusive(VerifierError) }
pub enum UnprovenReason { DeadlineMissed, ProverReportedMismatch, ProverReportedAnchorMismatch }
pub struct AuditReport { pub sample: Digest, pub manifest_digest: Digest, pub outcomes: Vec<(RequestKey, RequestOutcome)>, pub summary: AuditSummary }
pub struct AuditSummary { pub selected: u64, pub accepted: u64, pub rejected: u64, pub unproven: u64, pub inconclusive: u64, pub max_uhat: Option<FixedBits> }
```

- **Who writes it.** The verifier, at the sample's deadline: `tessera-verify audit report` (01 §2 stage 8, #151).
- **`sample`** is the sample seed, and `manifest_digest` the period's. `outcomes` holds one entry per selected request, in `selected`'s order.
- **`summary`** counts the outcomes. `selected` is their number and the sum of the other four counts. `max_uhat` is the largest accepted `Û`, and `None` when no request is accepted.
- **The prover's own reports.**
  - A prover that returns `ERR-WITNESS-EXACT-MISMATCH` for a request has disclosed a deviation. The report records it as `ProverReportedMismatch`.
  - A prover whose pre-check fails, `ERR-WITNESS-ANCHOR` (RFC-0015 'Prover pre-check'), has disclosed that its transcripts do not match the log. The report records it as `ProverReportedAnchorMismatch`.
- **Consequences** of any outcome are policy, outside Tessera.

`tessera.audit-report/1` ([`spec/schemas/audit-report-v1.schema.json`](../../spec/schemas/audit-report-v1.schema.json)) is the JSON rendering. Digests are lowercase hex, and `Û` is FixedBits as `uhat_fixed`. A rejection is rendered as `tessera.verdict/1` renders a verdict's reason. [`docs/perf/examples/audit-report.json`](../perf/examples/audit-report.json) shows one outcome of each kind.

### Calculators

`tessera audit plan` computes, from a policy, a traffic profile and the cost model's `μ` per bucket, the figures below.
- **The traffic profile.** `tessera.audit-traffic/1` ([`spec/schemas/audit-traffic-v1.schema.json`](../../spec/schemas/audit-traffic-v1.schema.json)) gives one entry per bucket of the policy, in its order:
  - its requests over a period, `n_b`;
  - their mean length, `L_b`: above the previous bucket's `max_tokens` and at most its own;
  - the violating requests among them, `v_b ≤ n_b`.
- **`μ_b`** is the cost model's at `L_b` (RFC-0020 'Executable cost model'), from an execution plan's counts or the source's.

- **Detection probability** for `v_b` violating requests per bucket: `1 − Π_b (1 − s̲_b)^{v_b}`, where `s̲_b = min_{c ≥ b} s_c` is the lowest rate of bucket `b` and every later one.
  - **Why the minimum.** The serving stack can pad a request it cheats on into a later bucket, for instance with inbound tokens it seals itself (RFC-0063 'Inbound before outbound'). So a violation of bucket `b` is detected at `s̲_b` at best.
  - **An example.** Under rates of 10%, 2% and 0.4% for 1K, 16K and 128K tokens, 100 violations of 1K tokens left where they are would be detected with probability above 0.9999. Padded past 16K tokens, they are detected with probability 0.33.
  - **No escape past the buckets.** A request padded past the last bucket is always selected (step 4), so padding never goes below the last bucket's rate.
- **Amortized overhead:** `Σ_b n_b L_b s_b μ_b η / Σ_b n_b L_b`, with `η` the prefill-to-serving ratio ([00 §10](../spec/00-overview.md#10-three-meanings-of-overhead)).
- **Rates for a target overhead**, as in the source's table: for 10% overhead at uniform context, `s = 0.1 / (μ η)`. Each bucket's target rate is `min(1, O/(μ_b·η))`, with `O` the target overhead.
- **Detection in place**, `1 − Π_b (1 − s_b)^{v_b}`, is what the policy detects if no violation leaves its bucket. The plan reports it beside the detection above, labelled as the upper bound it is.

**The report.** `tessera.audit-plan/1` ([`spec/schemas/audit-plan-v1.schema.json`](../../spec/schemas/audit-plan-v1.schema.json)) holds, per bucket:
- `s_b` and `s̲_b`;
- the traffic and `μ_b`;
- the expected selections `n_b·s_b`;
- the target rate;
- both detections over the bucket's own violations.

Then it holds the totals. A profile that is not one well-formed entry per bucket, or that has no request, is `ERR-USAGE`.

**The arithmetic** is IEEE 754 binary64, with `ln_1p` and `exp_m1` so that small rates keep their precision. These are planning figures. None enters the sample, whose thresholds are exact integers, or any TCE artifact (INV-DM-6).

The verifier side has an independent implementation used by `tessera-verify audit plan` (#151).
- **Detection** depends on the policy and the traffic only, so the verifier computes it from them alone.
- **The overhead and the target rates** need `μ`, which is the prover's cost model's (RFC-0020). The verifier reads them from a `tessera.cost-model/1` report whose contexts are the traffic's mean lengths (`--mu`), and without one it reports detection only.
- **Agreement.** CI checks that the two implementations agree within `10^{-12}`, relative, since platforms round `ln_1p` and `exp_m1` differently in the last bits.

### Invariants

- **INV-AUD-1.** No sampling randomness exists before the manifest is signed.
- **INV-AUD-2.** Selection is a deterministic function of `sample_seed`, the policy, and the entries of the manifest and of the period before it, read from their verified logs ([RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from)).
- **INV-AUD-3.** Each selected request appears in at most one accepted proof.

## Alternatives Considered

1. **Entry sampling inside a computation** (the training proposal's per-entry sampling, `k = 4,605` per tensor). Rejected for inference by the source: it misses deviations below about 1% of entries, and a covert channel needs only a few (source §4).
2. **Verifier-chosen samples.** Considered: unpredictable to the prover, simpler. Rejected: the prover cannot check the verifier respected the agreed rate, and a verifier could target specific requests; the coin toss lets both parties check the sample.
3. **Sampling announced before generation.** Rejected: the prover would behave only on announced requests.
4. **A uniform rate.** Rejected as the only option: long-context requests cost about 28 times more to prove than 1K requests at 405B (source §8), so uniform rates waste budget or under-sample short traffic.

## Drawbacks

- Detection is probabilistic; at long context and low rates a small number of violations can go undetected (source §8: 31.4% daily detection at 128K for 100 violations in 10^6).
- The prover must retain plaintext transcripts and session keys for every request until the period's sample is fixed and its deadline passes.
- **Stratified rates detect at the lowest later rate.** Against a prover that pads, a bucket's high rate buys no detection beyond the lowest rate of the buckets after it ('Calculators'). It still samples honest short traffic more densely, for the same budget. A policy that wants each bucket's rate as its detection rate needs rates that do not fall as length grows.

## Migration / Rollout

v0.4 delivers the policy, selection, batching, report and calculators, exercised with the reference recorder. Retention tooling for production transcripts is integration work.

## Testing Strategy

- Selection determinism: prover and verifier implementations select identical sets for 1,000 random seeds and synthetic manifests.
- Rate accuracy: over 10^6 synthetic requests, each bucket's selected fraction lies within a 99.9% binomial interval of its rate.
- Conformance vectors ([`spec/vectors/audit/`](../../spec/vectors/audit/README.md)), from an independent oracle: candidates and lengths from two periods' logs, rate thresholds, seeds, selection values, samples with their encodings and refusals, and batches.
- Unpredictability: changing one byte of `s_P` or `s_V` changes about half of the selection decisions (avalanche test).
- Batching: the packing by class is reproducible and respects both bounds, in counted tokens.
- Accounting: a proof containing an unselected request is rejected; a request in two batches is rejected the second time; a missing proof yields `Unproven(DeadlineMissed)`.
- Calculators: closed-form detection matches a Monte Carlo simulation within 0.5 percentage points; the source's table (§8) is reproduced from its `μ` values and `η = 0.3`.

## Open Questions

None. Rates, periods and deadlines are policy inputs.

## References

- Source §1, §4 (entry sampling), §7 ("Anchor, sampling, verifier"), §8 (sampling table and soundness argument).
- [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0020](RFC-0020-cost-model.md).
