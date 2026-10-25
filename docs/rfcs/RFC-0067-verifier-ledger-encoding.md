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
    pub derived: Vec<(Digest, Digest)>,        // (manifest, the one sample derived for it), by strictly increasing manifest
    pub proven: Vec<ProvenSample>,             // by strictly increasing sample
}
pub struct ProvenSample {
    pub sample: Digest,                        // the AuditSample's identity, its envelope's digest
    pub accepted: BitSet,                      // indices into the sample's selected; not empty
    pub uhat: Vec<FixedBits>,                  // one Û per accepted index, by increasing index
}
```

- **`BitSet`** is TCE `Bytes`. Index `i` is bit `i mod 8` of byte `⌊i/8⌋`. The last byte, when there is one, is not zero: the empty set is the empty string. A last byte of 0 is `ERR-CODEC-NONCANONICAL` at that byte, checked as the set is read, before any rule of the record that holds it: `[0x00]` is refused at its byte, not as an empty set.
- **`derived`** holds one sample per manifest, by strictly increasing manifest digest, else `ERR-CODEC-NONCANONICAL` at the first entry out of order.
- **`proven`**:
  - each record's `sample` is greater than the one before it, else `ERR-CODEC-NONCANONICAL` at its first byte;
  - an empty `accepted` is `ERR-CODEC-NONCANONICAL` at its length, since a record exists only once a request is accepted;
  - a `uhat` of another length than `accepted`'s count of set bits is `ERR-CODEC-NONCANONICAL` at its length.
- **`sample`** is the identity of the verifier's own derivation of the sample (RFC-0016 INV-AUD-2), and is one of `derived`'s samples, else `ERR-CODEC-NONCANONICAL` at the record's first byte.
- **Request order.** `selected`, and every list of requests here, sorts by `session_id` as bytes, then by `request_id` as an integer: ids 1 and 256 sort as numbers, not as their little-endian encodings (RFC-0016's `(session_id, request_id)`).

### The verifier's own sample

A sample is the verifier's when it derived it itself, with its own share. One whose coin toss merely opens could be anyone's: a prover can choose `s_V'`, set `c_V' = H(s_V')`, and attach a policy of rate 0.
- **Derivation.** The verifier derives a sample only by revealing its own toss over the period's verified window. The toss holds the share it committed after the policy was fixed, and that policy (RFC-0016 'The policy comes first'). The derivation adds `(manifest, sample)` to `derived` and writes the ledger before the sample leaves the verifier.
- **One sample per manifest.** Once the verifier has revealed `s_V`, a prover can choose a new share `s_P'` whose sample suits it. So:
  - a toss is revealed against one prover share, the first. Another is refused with `ERR-PROTO-ORDER`, as the prover answers one commitment per manifest (RFC-0016 'Sample derivation');
  - a manifest that `derived` already holds takes no other sample, `ERR-PROTO-ORDER`, which also refuses a second toss over one manifest;
  - the same sample again changes nothing.

  Without the rule, one period could have two samples, and `proven`, keyed by sample, would accept a request under each.
- **Accounting.** The checks below refuse a sample that `derived` does not list under its own manifest with `Inconclusive(ERR-USAGE)`, since the verifier's own input is wrong, before any check of the statement.
- **The report** refuses such a sample too (`ERR-USAGE`, exit 64). The ledger therefore tells "my sample, nothing proven" from "not my sample". A report built from outcomes the caller supplies must agree with the ledger, else `ERR-USAGE`: `Accepted` exactly for the requests the ledger records accepted, with their `Û`.
- **The pending toss** ([#529](https://github.com/AbdelStark/tessera/issues/529)). The prover answers one commitment per manifest, so a share kept only in memory would die with its process, and with it the period's audit. The verifier keeps it in a file of its own, `VerifierToss` (schema `0x0702`, `.tvt`), between the commitment and the reveal.
  - **The encoding.** TCE: `manifest_digest`, `policy`, `commitment`, then `share`, 32 bytes. The share is secret-class until the reveal (05 §3). A decoder refuses a commitment that is not `H("coin-commit", share)` with `ERR-PROTO-COINTOSS`.
  - **Before the commitment leaves.** The verifier writes the file as it writes the ledger: a synced temporary file, renamed into place, the directory synced. It holds mode 0600 and is created exclusively. Only then is the commitment sent. A file already at the path is refused with `ERR-USAGE`: a toss is never silently replaced, and a second commitment over one manifest would be the prover's to refuse anyway.
  - **The reveal.** It reads the file without following a link, and refuses one whose manifest is not the window's (`ERR-USAGE`). It derives the sample and records it in the ledger's `derived`, as 'Derivation' says, before the sample leaves the verifier.
  - **Cleared after the record.** Only once the ledger is written is the file overwritten with zeros, synced and removed. A verifier that stops between the record and the removal reveals again over the same prover share, which changes nothing, then removes the file.
  - **What is refused, not repaired.** A lost file loses the period's audit, as before: nothing can recreate its share. A file that fails its envelope's digest is `ERR-CODEC-DIGEST`, and one that does not open its commitment is `ERR-PROTO-COINTOSS`. Neither is replaced.
  - **What the file does not bind.** The commitment binds the share alone, and the envelope's digest is no MAC. A file whose manifest or policy is edited and resealed derives another sample, which is not the prover's, so nothing is accepted under it, and the period's audit is lost. Whoever can write the verifier's files can do as much to the ledger.
  - **A second ledger.** A file kept past its record, after a stop, reveals the same sample into any ledger it is given. One ledger per verifier ('One ledger per verifier') stays the operator's to keep. Before the record, a new share after a restart is harmless: `s_V` has not left the verifier.

### Updates

- **When.** The ledger changes only after `Accept` (RFC-0019 step 13). The accepted requests join their sample's record, with their `Û`.
- **The accounting checks run again** before the change, so no request is recorded twice. A batch of no request records nothing.
- **The file.** The verifier writes the new ledger to a temporary file beside the old one, syncs it, renames it over the old one, and syncs the directory. The file then holds the old ledger or the new one whenever the writer stops. The ledger in memory changes only once the file holds the new one, and a failed write means the verdict is not `Accept`.
  - **The directory's sync** is part of the write. If it fails, the write fails with `ERR-RES-IO`: after a power loss the old ledger could return, and with it a request accepted again (INV-AUD-3).
  - **The path.** The ledger is a regular file, neither a symbolic link nor a file of several hard links, else `ERR-USAGE`. Two names for one ledger could be written as two ledgers.
    - Its directory is resolved once when the ledger is opened.
    - A path that is not a regular file, such as a FIFO, is refused before it is opened, by metadata that does not follow links. The file opened must then be the one checked, by device and inode.
    - The rules hold at open and again before every write, since a hard link added later would make a second ledger, and for a reader of a snapshot, which takes no lock.
    - A missing ledger is `ERR-RES-IO`.
  - **Off Unix**, where the standard library offers neither a directory sync nor a link count, a writer has no durable rename to rely on, and takes no lock. The guarantees above are Unix's (09 §8 supports no other host). On a Unix where `std` cannot lock a directory, every writer is refused.
- **One writer.** A writer holds an exclusive kernel lock on the ledger's directory, resolved once when the ledger is opened, for as long as it holds the ledger (#534).
  - **The lock** is the operating system's advisory file lock on the directory itself, taken through `std::fs::File::try_lock` (Rust 1.89): `flock` on Linux and macOS. A second writer does not wait: its `try_lock` fails, and it is refused with `ERR-RES-IO`, whether it opens the held ledger or creates another in the held directory.
  - **No stale lock.** The system releases the lock when its process ends, a crash included, so no operator step clears it. A child process the host is spawning at that moment holds a copy of its open files until its program starts, so a lock can outlive its holder by that instant; a writer then refused with `ERR-RES-IO` tries again.
  - **What an operator cannot do.** Removing or renaming a file in the directory does not release the lock, and neither does renaming the directory, which keeps its inode. A second writer is refused by every name the ledger has.
  - **What it can.** An operator who moves the ledger into another directory makes a second ledger there, which another writer can hold: one ledger per verifier ('One ledger per verifier') is the operator's to keep, as it is for a copy. A writer stalled past its checks recreates nothing there.
  - **One writer per directory.** The lock covers the directory, so a directory holds one ledger's writer at a time, the verifier's one ledger. Two paths to one directory, mount points included, give one lock.
  - **A reader of a snapshot** takes no lock.
  - **Before each rename** a writer checks that the ledger is still the file it last read or wrote, and that the directory's path still names the directory it locked, each by device and inode. Otherwise it writes nothing, `ERR-RES-IO`, and the verdict is not `Accept`.
    - An operator who replaced the file meanwhile left a ledger the writer has not read.
    - An operator who put a new directory at the path, and moved the ledger into it, would let a second writer lock the new one while this writer's file checks still passed.
    - The lock, held throughout, keeps every other writer of the locked directory out, so the check and the rename leave no window for one. One case remains: an operator who replaces the directory between the check and the rename, and moves the writer's temporary file into the new one. Closing it needs a rename relative to the locked directory's handle, which `std` does not offer.
  - **Local file systems.** A network file system may not honour `flock`. The ledger lives on a local one.
  - **One ledger per verifier.** A toss and its sample belong to the verifier's one ledger. A second ledger file would hold its own `derived` and `proven`, and could accept a request the first holds. So a toss records its sample only in the first ledger it is revealed into, by canonical path, and is refused by any other (`ERR-USAGE`). The operator keeps one ledger, opened by one path.
  - **A failed directory sync after the rename** leaves a file that may hold the new record while the verdict is not `Accept`. A report from the ledger then counts the request as accepted. A report built from kept verdicts disagrees with that ledger, and is refused (`ERR-USAGE`) until the operator reconciles the two.
  - **One limit.** The ledger is written and read under the same decode limits, `tv-codec`'s message limits, so a ledger a writer accepts is one every reader opens.
  - A batch of no request changes nothing and writes nothing.
  - A write that would make the ledger larger than the limits it is reopened under is refused, `ERR-CODEC-LIMIT`, and the ledger is left as it was.
- **Trust.** The ledger is the verifier's own state. Any file presented as the ledger is believed, `derived` included, so it is protected like the verifier's keys.

### The accounting

RFC-0016 'Verifier checks' 1 and 2 run in RFC-0039's field order, after RFC-0016's check of the parameter set:
0. **The parameter set.** The claim list's must be the one the sample's policy names, else `Reject(Params { name })`, naming the claim list's. It is RFC-0019 step 2's check, so it comes before the statement's fields.
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
- A temporary file left by a crash needs an operator. The lock does not: the system releases it with the process.
- The lock is advisory and local: a network file system may not honour it, and a process that writes the ledger without taking it is not kept out.
- One ledger writer per directory, since the lock covers the directory.
- A verifier that loses its pending toss file between the commitment and the reveal loses that period's audit: the share exists nowhere else.

## Migration / Rollout

#151 implements the encoding, the writes, the accounting, the report and the plan in `tv-core` and `tessera-verify`. The weight fields' rules follow with #158.

## Testing Strategy

- **Encoding.** Every rule above, refused at its offset; round trips of seeded ledgers.
- **Writes.** A write interrupted at each of its four steps leaves the old ledger byte for byte; a second writer is refused; creating over an existing file is refused.
- **Accounting.** A request outside the sample, one in two accepted batches, and one named twice in a batch are each refused, as is another manifest. A batch of 2,000,000 requests is checked within a time bound. A sample whose toss opens but which the verifier did not derive is refused, by the accounting and by the report. A second prover share for one toss, and a second toss over one manifest, are refused. A second writer is refused while the first holds the directory, from the same process and from another, by every name of the ledger and after the directory is renamed; a writer stalled before its rename loses no record; a killed holder's lock is released with no operator step; a replaced ledger or directory is not written over (#534). A claim list of another parameter set than the policy's is `Reject(Params)` and records nothing, ahead of every check of the statement and after the sample's own (#530).
- **The file.** A failed directory sync, a symbolic link, a second hard link and a FIFO are each refused.
- **Report.** A missing proof is `Unproven(DeadlineMissed)`.
- **Sampling.** The prover's and the verifier's samples agree over 1,000 seeds, with keys that share sessions and ids whose numeric and byte orders differ (`tessera-protocol`'s `audit_verifier.rs`, DEP-2). Their audit plans agree within `10^{-12}` relative over random profiles.

## Open Questions

None.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0016](RFC-0016-audit-sampling.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0043](RFC-0043-rejection-payloads.md).
