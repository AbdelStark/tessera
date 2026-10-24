# RFC-0016: Commit-Then-Sample Audit Protocol

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies) ('Batching': a batch's temperature requests share one pinned sampler key), [RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log) ('Sample derivation': the manifest's signature first, then the sample from the period's verified log), [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from) ('Sample derivation': the requests a period draws from, ended and unterminated, and their lengths)

## Summary

Proving every request is not affordable; proving an unpredictable sample of committed requests is. Production traffic is recorded continuously. At the end of each period the recorder freezes and signs the log. Only then do the parties toss a coin for a sampling seed, from which a public, deterministic rule selects requests at per-length-bucket rates. Every selected request must be proven in full, in batches of at least 16K tokens. The verifier checks that each proof covers only sampled requests and that every sampled request is accounted for. This RFC fixes the policy, the selection rule, batching, the audit report and the detection and overhead calculators.

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
    pub buckets: Vec<(u32 /* max total tokens */, u64 /* rate as threshold over 2^64 */)>,  // ascending
    pub min_batch_tokens: u32,                    // 16384
    pub max_batch_tokens: u32,                    // memory bound, for example 262144
    pub proof_deadline_ns: u64,                   // time after the sample is fixed
    pub params: ParamSetId,
}
```

Rates are stratified by request length because proving cost rises with context (source §8 table): a policy may sample short requests at 10% and 128K-token requests at 0.4% for the same budget.

### Sample derivation

1. The recorder publishes `LogManifest` for the period ([RFC-0015](RFC-0015-transcript-anchor.md#log-format)). The verifier checks its signature under a pinned recorder key, then checks the period's log against it and keeps the verified copy, from which every later step reads the entries ([RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log)). It derives no sample before the previous period's log is verified too ([RFC-0063](RFC-0063-request-boundaries.md#the-previous-periods-log)).
2. The parties run the coin toss of [RFC-0014](RFC-0014-interaction-and-soundness.md#session-order-live-mode) over the manifest: the verifier commits to `s_V`, the prover reveals `s_P`, the verifier reveals `s_V`.
3. `sample_seed = H("audit-sample", s_V ‖ s_P ‖ manifest_digest)`.
4. Request `(session_id, request_id)`, a candidate of the period ([RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from): it has a `RequestEnd` in the period, or a frame in the previous period and no end in either), with total length `L` over all its frames in the two periods, is selected iff `u64_le(H("audit-select", sample_seed ‖ session_id ‖ u64(request_id))[0..8]) < rate(bucket(L))`.
5. The result is the public artifact:

```rust
pub struct AuditSample {                          // schema 0x0403
    pub manifest_digest: Digest, pub policy: AuditPolicy,
    pub coin_toss: CoinTossTranscript,            // c_V, s_P, s_V
    pub selected: Vec<RequestKey>,                // sorted by (session_id, request_id)
}
```

The domains `audit-sample` and `audit-select` join RFC-0004's domain table.

### Batching

The prover packs selected requests into batches by first-fit decreasing on total tokens, with the request order and tie-breaking fixed by `RequestKey` order, so that batch composition is reproducible. Requests that decode with temperature are first grouped by the pinned sampler key that applies to each, and each group is packed on its own ([RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies)). A batch has at least `min_batch_tokens` tokens unless the remaining selected requests total fewer, and at most `max_batch_tokens`. Each batch is one proof whose `PublicStatement.batch` lists its requests.

### Verifier checks

For each proof:

1. `statement.manifest_digest` equals the sample's `manifest_digest`.
2. Every request in `statement.batch` is in `selected` and has not appeared in an earlier accepted batch.
3. The anchor checks of RFC-0015 pass for every request.

Checks 1 and 2 are statement fields of RFC-0019 step 4: `Reject(StatementMismatch { field: ManifestDigest })` and `{ field: Batch }`, a replayed request included. Check 3 fails as `Reject(Anchor)` ([RFC-0043](RFC-0043-rejection-payloads.md#the-audit-samples-checks)).

At the deadline, the verifier emits the audit report.

### Audit report

```rust
pub enum RequestOutcome { Accepted { uhat: FixedBits }, Rejected(RejectReason), Unproven(UnprovenReason), Inconclusive(VerifierError) }
pub enum UnprovenReason { DeadlineMissed, ProverReportedMismatch, ProverReportedAnchorMismatch }
pub struct AuditReport { pub sample: Digest, pub outcomes: Vec<(RequestKey, RequestOutcome)>, pub summary: AuditSummary }
```

`tessera.audit-report/1` is the JSON rendering. A prover that returns `ERR-WITNESS-EXACT-MISMATCH` for a request has disclosed a deviation; the report records it as `ProverReportedMismatch`. Consequences of any outcome are policy, outside Tessera.

### Calculators

`tessera audit plan` computes, from a policy and a traffic profile (request counts per bucket) and the cost model's `μ` per bucket:

- **Detection probability** for `v_b` violating requests per bucket: `1 − Π_b (1 − s_b)^{v_b}`.
- **Amortized overhead:** `Σ_b n_b L_b s_b μ_b η / Σ_b n_b L_b`, with `η` the prefill-to-serving ratio ([00 §10](../spec/00-overview.md#10-three-meanings-of-overhead)).
- **Rates for a target overhead**, as in the source's table: for 10% overhead at uniform context, `s = 0.1 / (μ η)`.

The verifier side has an independent implementation used by `tessera-verify audit plan`; CI checks they agree.

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

## Migration / Rollout

v0.4 delivers the policy, selection, batching, report and calculators, exercised with the reference recorder. Retention tooling for production transcripts is integration work.

## Testing Strategy

- Selection determinism: prover and verifier implementations select identical sets for 1,000 random seeds and synthetic manifests.
- Rate accuracy: over 10^6 synthetic requests, each bucket's selected fraction lies within a 99.9% binomial interval of its rate.
- Unpredictability: changing one byte of `s_P` or `s_V` changes about half of the selection decisions (avalanche test).
- Batching: first-fit decreasing output is reproducible and respects both bounds.
- Accounting: a proof containing an unselected request is rejected; a request in two batches is rejected the second time; a missing proof yields `Unproven(DeadlineMissed)`.
- Calculators: closed-form detection matches a Monte Carlo simulation within 0.5 percentage points; the source's table (§8) is reproduced from its `μ` values and `η = 0.3`.

## Open Questions

None. Rates, periods and deadlines are policy inputs.

## References

- Source §1, §4 (entry sampling), §7 ("Anchor, sampling, verifier"), §8 (sampling table and soundness argument).
- [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0020](RFC-0020-cost-model.md).
