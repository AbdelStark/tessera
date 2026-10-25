# RFC-0067: The Verifier Ledger's Encoding and the Audit Report

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends:
  - [RFC-0019](RFC-0019-verifier.md#ledger) 'Ledger': the TCE encoding of `VerifierLedger` (schema `0x0701`), each accepted request's `Û`, and how the file is written;
  - [RFC-0016](RFC-0016-audit-sampling.md#audit-report) 'Audit report': what a report from the ledger alone can state, and the command's exit code;
  - [RFC-0016](RFC-0016-audit-sampling.md#calculators) 'Calculators': the verifier's audit plan without a cost model.
- Origin: [#151](https://github.com/AbdelStark/tessera/issues/151)

## Summary

RFC-0019 sketched the ledger with a `BitSet` type that TCE does not have, and gave no rule for the order of its records. RFC-0016's report needs each accepted request's `Û`, which the sketch did not keep. This RFC fixes the encoding:
- `BitSet` is `Bytes`, least significant bit first, with no trailing zero byte;
- each record of `proven` carries one `Û` per accepted request, and the records are sorted by sample.

It also states how the file is written: atomically, under a lock. It states what `tessera-verify audit report` can say from the ledger alone, and what the verifier's audit plan prints without a cost model.

## Motivation

The verifier must refuse a request proven twice and report every selected request at the deadline (RFC-0016 INV-AUD-3, #151). The ledger is where that state persists across sessions. It is a tier-A artifact (02 §1), so its bytes must be one encoding, which RFC-0019's sketch left open.

## Goals

- One canonical encoding of `VerifierLedger`, decoded under RFC-0003's rules and the rules below.
- A ledger write that leaves the old ledger or the new one, never part of either.
- An audit report computable from the sample and the ledger.

## Non-Goals

- **The weight fields** (`weight_roots`, `successors`, `used_blinding_rows`): their encoding is the sketch's, and their order and uniqueness rules are #158's.
- **Several verifiers** of one sample, and a signed record of the prover's share (#140's review).
- **The policy's parameter set** against the proof's: no RFC compares them yet (#530).

## Proposed Design

### The encoding

```rust
pub struct VerifierLedger {                   // schema 0x0701, version 1, .tvl
    pub weight_roots: Vec<(Digest, Vec<(u32, Digest)>)>,
    pub successors: Vec<(Digest, Digest)>,
    pub used_blinding_rows: Vec<(Digest, u32, u8, BitSet)>,
    pub derived: Vec<Digest>,                  // the samples this verifier derived, strictly increasing
    pub proven: Vec<ProvenSample>,             // by strictly increasing sample
}
pub struct ProvenSample {
    pub sample: Digest,                        // the AuditSample's identity, its envelope's digest
    pub accepted: BitSet,                      // indices into the sample's selected; not empty
    pub uhat: Vec<FixedBits>,                  // one Û per accepted index, by increasing index
}
```

- **`BitSet`** is TCE `Bytes`. Index `i` is bit `i mod 8` of byte `⌊i/8⌋`. The last byte, when there is one, is not zero: the empty set is the empty string. A last byte of 0 is `ERR-CODEC-NONCANONICAL` at that byte, checked as the set is read, before any rule of the record that holds it: `[0x00]` is refused at its byte, not as an empty set.
- **`derived`** is strictly increasing, else `ERR-CODEC-NONCANONICAL` at the first digest out of order.
- **`proven`**:
  - each record's `sample` is greater than the one before it, else `ERR-CODEC-NONCANONICAL` at its first byte;
  - an empty `accepted` is `ERR-CODEC-NONCANONICAL` at its length, since a record exists only once a request is accepted;
  - a `uhat` of another length than `accepted`'s count of set bits is `ERR-CODEC-NONCANONICAL` at its length.
- **`sample`** is the identity of the verifier's own derivation of the sample (RFC-0016 INV-AUD-2), and is in `derived`, else `ERR-CODEC-NONCANONICAL` at the record's first byte.
- **Request order.** `selected`, and every list of requests here, sorts by `session_id` as bytes, then by `request_id` as an integer: ids 1 and 256 sort as numbers, not as their little-endian encodings (RFC-0016's `(session_id, request_id)`).

### The verifier's own sample

A sample is the verifier's when it derived it itself, with its own share. One whose coin toss merely opens could be anyone's: a prover can choose `s_V'`, set `c_V' = H(s_V')`, and attach a policy of rate 0.
- **Derivation.** The verifier derives a sample only by revealing its own toss over the period's verified window. The toss holds the share it committed after the policy was fixed, and that policy (RFC-0016 'The policy comes first'). The derivation adds the sample's identity to `derived` and writes the ledger before the sample leaves the verifier.
- **Accounting.** The checks below refuse a sample whose identity is not in `derived` with `Inconclusive(ERR-USAGE)`, since the verifier's own input is wrong, before any check of the statement.
- **The report** refuses such a sample too (`ERR-USAGE`, exit 64). The ledger therefore tells "my sample, nothing proven" from "not my sample".
- **A lost toss.** A share kept only in memory dies with its process. Since the prover answers one commitment per manifest, a verifier that restarts between commitment and reveal loses that period's audit. Keeping the share across restarts is open (#529).

### Updates

- **When.** The ledger changes only after `Accept` (RFC-0019 step 13). The accepted requests join their sample's record, with their `Û`.
- **The accounting checks run again** before the change, so no request is recorded twice. A batch of no request records nothing.
- **The file.** The verifier writes the new ledger to a temporary file beside the old one, syncs it, renames it over the old one, and syncs the directory. The file then holds the old ledger or the new one whenever the writer stops. The ledger in memory changes only once the file holds the new one, and a failed write means the verdict is not `Accept`.
  - **The directory's sync** is part of the write. If it fails, the write fails with `ERR-RES-IO`: after a power loss the old ledger could return, and with it a request accepted again (INV-AUD-3).
  - **The path.** The ledger is a regular file, neither a symbolic link nor a file of several hard links, else `ERR-USAGE`. Two names for one ledger would give two locks. Its directory is resolved once when the ledger is opened. A path that is not a regular file, such as a FIFO, is refused before it is opened.
- **One writer.** A lock file, `<name>.lock`, created exclusively, keeps a second writer out while the first holds the ledger.
  - A lock left by a crash refuses later writers with `ERR-RES-IO`. Only the operator removes it, after checking that no verifier runs.
  - A reader of a snapshot takes no lock.

### The accounting

RFC-0016 'Verifier checks' 1 and 2 run in RFC-0039's field order:
1. **The manifest.** Another manifest digest is `StatementMismatch { field: ManifestDigest }`.
2. **Each request.** It must be in `selected`, absent from the sample's record, and named once by the batch. Any of these failures is `StatementMismatch { field: Batch }`, a replayed request included.

Every failure of check 2 gives one reason, so its checks need no order among themselves. A verifier must still bound their cost in the batch's size, which the prover chooses: `tv-core` looks each request up by binary search and finds repeats by sorting, at `O(n log n)`.

### The report from the ledger

`tessera-verify audit report --ledger --sample` writes `tessera.audit-report/1` from the sample and the ledger alone.
- **Accepted.** A request the ledger records accepted for the sample is `Accepted { uhat }`.
- **Every other request** is `Unproven(DeadlineMissed)`.
  - The ledger keeps acceptances only, so a request whose proofs were all rejected or inconclusive is reported as unproven too.
  - A verifier that keeps its verdicts reports `Rejected`, `Inconclusive` and the prover's own reports through `tv_core::audit::AuditReport::new`, which takes one outcome per selected request.
- **No clock.** The command does not compare the time with the deadline: no artifact records when a sample was fixed. Running it at the deadline is the operator's.
- **The exit code** is 0 when every selected request is accepted, and 10 otherwise, as `params` exits 10 for a target missed.

### The verifier's audit plan

`tessera-verify audit plan --policy --traffic [--mu]` follows RFC-0016 'Calculators'.
- **The cost inputs.** `η` and the target overhead come from the cost model's scenario, and a target overhead of 0 is refused.
- **Without `--mu`** the plan computes detection only. The figures that need `μ`, `η` or the target are `null`, which `audit-plan-v1` allows. [`docs/perf/examples/audit-plan-detection.json`](../perf/examples/audit-plan-detection.json) is an example.

## Alternatives Considered

1. **A sorted list of indices in place of `BitSet`.** It is smaller for sparse records but needs its own canonical rules. A sample's selection is dense, since every selected request is meant to be proven, so a bitmap is near optimal.
2. **Keeping every verdict in the ledger.** It would let the report state rejections, but the ledger would grow with every proof, rejected ones included. Verdicts are the verifier's own logs (05 §2).

## Drawbacks

- A report from the ledger alone cannot tell a rejected proof from a missing one.
- A stale lock needs an operator, and so does a temporary file left by a crash.
- A verifier that restarts between a toss's commitment and its reveal loses that period's audit (#529).

## Migration / Rollout

#151 implements the encoding, the writes, the accounting, the report and the plan in `tv-core` and `tessera-verify`. The weight fields' rules follow with #158.

## Testing Strategy

- **Encoding.** Every rule above, refused at its offset; round trips of seeded ledgers.
- **Writes.** A write interrupted at each of its four steps leaves the old ledger byte for byte; a second writer is refused; creating over an existing file is refused.
- **Accounting.** A request outside the sample, one in two accepted batches, and one named twice in a batch are each refused, as is another manifest. A batch of 2,000,000 requests is checked within a time bound. A sample whose toss opens but which the verifier did not derive is refused, by the accounting and by the report.
- **The file.** A failed directory sync, a symbolic link, a second hard link and a FIFO are each refused.
- **Report.** A missing proof is `Unproven(DeadlineMissed)`.
- **Sampling.** The prover's and the verifier's samples agree over 1,000 seeds, with keys that share sessions and ids whose numeric and byte orders differ (`tessera-protocol`'s `audit_verifier.rs`, DEP-2). Their audit plans agree within `10^{-12}` relative over random profiles.

## Open Questions

None.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0016](RFC-0016-audit-sampling.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0043](RFC-0043-rejection-payloads.md).
