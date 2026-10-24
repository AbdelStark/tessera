# RFC-0043: Rejection Payloads

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Amended by: [RFC-0060](RFC-0060-log-manifests.md) (the inclusion proofs' encoding; `ManifestSignature` covers a manifest's other checks alone), [RFC-0059](RFC-0059-frame-headers-and-nonces.md#verifier-check-2-amended) ('Anchor': `Frames` also covers a session's repeated counter, an impossible frame length, and a session-open entry after one of its frames)
- Supersedes, on acceptance:
  - [04 §2](../spec/04-error-model.md#2-verdicts) 'Verdicts': `ArtifactKind`, `PowPhase` and `AnchorFailure` are defined here, `ParamSetName` is RFC-0014's `ParamSetId`, and `Anchor`'s request is optional;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 1: which decoding failures reject;
  - [RFC-0015](RFC-0015-transcript-anchor.md#verifier-checks) 'Verifier checks' and [RFC-0016](RFC-0016-audit-sampling.md#verifier-checks) 'Verifier checks': the rejection each check returns.
- Origin: [#353](https://github.com/AbdelStark/tessera/issues/353)

## Summary

04 §2 types four rejection payloads with names that no document defines: `ArtifactKind` in `Malformed`, `PowPhase` in `ProofOfWork`, `AnchorFailure` in `Anchor`, and `ParamSetName` in `Params`. This RFC defines each. It also settles what RFC-0015 and RFC-0016 leave open:
- the request a manifest-level anchor failure names;
- the rejection of a batch that holds a request outside the audit sample, or one already proven.

RFC-0039 did the same for `StatementMismatch`.

## Motivation

- **The verdict is tier A** ([02 §1](../spec/02-public-api.md#1-stability-tiers)), and `tv-core` returns it (#128). An undefined payload leaves each implementation to choose one.
- **One reason per input** ([RFC-0019](RFC-0019-verifier.md#goals)). Two verifiers must name the same payload for the same failure.
- **Gaps between the documents.**
  - **Which decoding failures reject.** RFC-0019 step 1 rejects any input that fails to decode (`Reject(Malformed)`). 04 §5 rejects only proofs, and gives other inputs `ERR-CODEC-*`.
  - **Manifest-level anchor failures.** RFC-0015 makes every anchor failure `Reject(Anchor { request_id, kind })`, with a `u64` request. Its first check, the manifest's signature, concerns no request.
  - **The audit sample's checks.** RFC-0016's checks name no rejection.

## Goals

- A definition and a TCE encoding for every payload of `RejectReason`, public data only ([05 §3](../spec/05-observability.md#3-data-classification-and-redaction)).
- A rejection for every check of RFC-0015 and RFC-0016.

## Non-Goals

- The encoding of the anchor's inclusion proofs, which [RFC-0060](RFC-0060-log-manifests.md#the-tree-over-a-periods-entries) fixes: `Vec<[u8; 32]>`, at most 64 digests.

## Proposed Design

### Malformed

`Malformed { artifact: ArtifactKind, code: ErrorCode }`:

```rust
pub struct ArtifactKind(pub SchemaId);     // the artifact's schema id, 03 §5
```

- **The artifacts.** `Malformed` names the prover's artifacts, whose bytes the verifier decodes:
  - `ProverSeedShare`, `Round1`, `Round2Response` and `Round3Response` in live mode;
  - `NiProof` in non-interactive mode;
  - the `DigestLogEntry`s that come with a proof's batch ([RFC-0015](RFC-0015-transcript-anchor.md#log-format)).
- **The code.** It is the decoder's `ERR-CODEC-*` code (RFC-0003). A `NiProof` that fails inside its `Round2Response` is `Malformed { artifact: NiProof, … }`: the artifact is the bytes the verifier was given.
- **The verifier's own inputs.** The claim list, the manifest, the audit sample, the ledger, the weight commitment ([RFC-0040](RFC-0040-weight-commitment-artifacts.md)) and the recorder keys are not the prover's. One that fails to decode is `Inconclusive(ERR-CODEC-*)`: the verifier could not evaluate, and no proof was judged (04 §5).

Its TCE encoding is the schema id, a `u16`.

### Proof of work

`ProofOfWork { phase: PowPhase }`:

```rust
pub enum PowPhase { Seed, Queries }        // NiProof's seed_nonce, then its query_nonce
```

`Seed` is `NiProof.seed_nonce` under `seed-pow`, and `Queries` is `NiProof.query_nonce` under `query-pow`: each short of its leading zero bits, as [RFC-0014](RFC-0014-interaction-and-soundness.md#grinding) fixes them. Its encoding is RFC-0003's enum: a `u16` tag, `Seed` 0 and `Queries` 1.

### Parameter sets

`Params { name: ParamSetId }`:

04 §2's `ParamSetName` is [RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets)'s `ParamSetId(pub Ascii)`: the parameter set the claim list names, such as `live-64/1`, that the verifier does not allow. A claim list's `params` is within its decode limits, so the payload is too.

### Anchor

`Anchor { request_id: Option<u64>, kind: AnchorFailure }`:

```rust
pub enum AnchorFailure {
    ManifestSignature,       // RFC-0015 check 1: the signature, its scheme or its key, or the
                             // manifest's other checks alone (RFC-0060 'The chain')
    Inclusion,               // check 2: an entry's inclusion proof against entries_root
    Frames,                  // check 2: a request's frames, not contiguous from 0 or not its lengths;
                             // a counter repeated in the session, an impossible length, or the
                             // session's open after one of its frames (RFC-0059)
}
```

- **The request.** `request_id` is `None` for `ManifestSignature`, which concerns the manifest, and the request's id otherwise. The verifier runs checks 1 and 2 in the order of RFC-0015. It checks the requests in `statement.batch` order, a request's session-open entry before its frames, and the first failure is the verdict.
- **RFC-0015's other checks.**
  - **Check 3**, the batch's requests against the audit sample, is RFC-0016's (below).
  - **Check 4**, the circuits' public inputs, cannot fail on its own. The verifier itself makes the entries' digests, both commitments, the counter blocks and the plaintexts' prefixes the circuits' public inputs ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-circuits)). A proof computed against others fails its sumchecks.
- **The encoding.** `AnchorFailure` is a `u16` tag. `request_id` is RFC-0003's `Option<u64>`.

### The audit sample's checks

RFC-0016's checks compare the statement with what the verifier's inputs give. They are statement fields of RFC-0019 step 4 ([RFC-0039](RFC-0039-statement-fields.md#the-checks)):

| RFC-0016 check | Rejection |
|---|---|
| 1. `statement.manifest_digest` equals the sample's | `StatementMismatch { field: ManifestDigest }` |
| 2. every request of `statement.batch` is in `selected`, and none was in an earlier accepted batch (the ledger's `proven`) | `StatementMismatch { field: Batch }` |
| 3. the anchor checks of RFC-0015 | `Anchor { … }`, above |

## Alternatives Considered

1. **An enum of artifact kinds** in place of the schema id. It would repeat 03 §5's catalog in a second list that must follow every new artifact.
2. **A request for every anchor failure**, the batch's first for a manifest failure. A rejection would then name a request that has nothing to do with the failure.
3. **A rejection of its own for a replayed request.** It is a statement whose batch the verifier does not expect, as a request outside the sample is, and `Batch` names it. The ledger, which the verifier keeps, decides which requests are proven.

## Drawbacks

- `Malformed` names the outer artifact, not the message inside an `NiProof` that failed. The code and the decoder's offset locate it.

## Migration / Rollout

v0.3, for #353: this RFC and its amendments. `tv-core`'s verdict types and drivers return them (#128). The anchor's kinds come with its implementation in v0.4.

## Testing Strategy

- **Every payload.** For each rejection, an input that breaks exactly its check is rejected with that payload (#128, #135):
  - a truncated `Round2Response`, `NiProof` and `DigestLogEntry`, each with its schema id;
  - each nonce one bit short;
  - an unknown parameter set;
  - a bad manifest signature, with no request;
  - a bad inclusion proof and a missing frame, with their request;
  - a request outside the sample, and a replayed one, both as `Batch`.
- **The verifier's inputs.** A claim list, manifest and ledger that fail to decode are each `Inconclusive(ERR-CODEC-*)`.

## Open Questions

None.

## References

- [04 §2](../spec/04-error-model.md#2-verdicts), [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0016](RFC-0016-audit-sampling.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0036](RFC-0036-protocol-messages.md), [RFC-0039](RFC-0039-statement-fields.md).
