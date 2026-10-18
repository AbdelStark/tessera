---
name: soundness-review
description: Adversarial checklist for code that decides whether a proof is accepted or that handles secrets. Use when writing or reviewing field arithmetic, encodings, hashing and transcripts, sumcheck, GKR or LogUp, commitments, gadgets, zero knowledge or verifier checks.
---

# Soundness review

Read the change twice: as a malicious prover (ADV-P) with a large grinding budget who wants a false statement accepted, and as a malicious verifier (ADV-V) who wants weights, prompts or outputs. Apply every item below to the change and record pass or fail with evidence (`file:line`, test name). An item without evidence is a fail.

## Field and integers (RFC-0002)
- Every public operation returns a canonical value in `[0, P)`; decoders reject `x ≥ P`.
- Arithmetic on secret values is branch-free: masks and selects, no data-dependent `if` (INV-ARCH-2). It must stay branch-free in the binary too. On x86_64, LLVM turns a plain mask select inside a loop into a jump on the data, so a select on secret values goes through `core::hint::select_unpredictable`. `cargo xtask check-branchless` reads the field's x86_64 loops (#242).
- Integers enter the field only through `from_i64` and leave through `to_centered_i128`, and only as true integers. Every quantity compared modulo `P` is below `2^62` in absolute value, so Lemma 0 of RFC-0009 applies.
- Integer products and sums have a stated bound that rules out overflow; witness code uses checked or widened arithmetic, never silent wrapping.

## Encoding and transcript (RFC-0003, RFC-0004)
- One encoding per value: `encode(decode(b)) = b` for every accepted `b`; lengths are checked against limits and remaining bytes before allocation.
- Every prover message is absorbed, as TCE bytes, before any challenge that depends on it is squeezed (INV-FS-1).
- Challenges are exactly uniform: rejection sampling, never reduction modulo `P` of a wider integer.
- Labels are unique and follow the schedule of RFC-0014; the prover and verifier derive them from the same table.
- The verifier derives every value the protocol says to derive; it never reads such a value from the proof.

## Sumcheck, GKR and LogUp (RFC-0008, RFC-0010)
- Each round polynomial has at most the relation's degree; the verifier checks `p(0) + p(1) = claim` every round and recomputes the final evaluation from its own public factors.
- Batching challenges are drawn after the claims they combine are fixed; smaller instances are scaled by `2^(n_max − n_i)`.
- A check that passes on the way to a later one returns what that check still owes in a `#[must_use]` type: a sumcheck's last claim, a GKR level's next claims, the leaf claims. A caller that drops them accepts what nothing checked.
- LogUp: the verifier rejects a zero root denominator on either side and checks that the roots' fractions sum to zero. Multiplicities stay below `P`, and the error bound counts `N + T` over `|Fp2|`.
- Padding a prover can set must be harmless by construction. Each entry that pads a table to a power of two is either public and checked, or a copy of an admissible entry: RFC-0008 pads both LogUp trees with row 0. A padding row with new values, such as key and value 0, is a new table row, and a committed multiplicity on it admits a lookup that is not in the table.
- Every new round or challenge is counted by the soundness calculator (RFC-0014).

## Commitments (RFC-0011, RFC-0012)
- Column queries are drawn after every combined row is fixed; `τ` comes from the parameter set.
- The verifier re-encodes combined rows and checks consistency on every queried column; Merkle multiproofs are canonical (extra, missing or reordered siblings rejected).

## Gadgets (RFC-0009)
- Every committed value upstream of the logits has exactly one passing witness. The change carries an alternative-witness test for each gadget it touches.
- Gadget preconditions (such as `s ≤ r − 1`) are checked at claim-list validation, by the verifier, from public data.
- A range argument counts every term of the quantities it compares. A packed class term `2^b·c` multiplies a key's range by the class count `n`, so a split identity compares values up to `n·2^b`, not `2^b`. RFC-0009 Lemma 2's split dropped that term, which let `RequantSplit` wrap at `n = 4, b = 62` (#118).

## Zero knowledge (RFC-0013)
- Every value sent to the verifier is public, uniformly masked, or determined by values already sent (INV-ZK-1).
- A masked extension `Ṽ + Z·R̃` with `Z = Π x_i(1 − x_i)` equals `Ṽ` at any point with a Boolean coordinate, such as a GKR level's pair bit: an evaluation revealed there needs masking of its own (RFC-0013 OQ-0013-2).
- Each mask row, blinding row and masking polynomial is used in exactly one revealed combination; masking randomness comes from the per-proof seed, never from the transcript.

## Secrets (05 §3, 06 §7)
- Every secret-class value of the 05 §3 table lives in `Secret<T>` from the moment it is read, and every table derived from one in a `SecretBuf` until a message carries it (OBS-6); none reaches `Debug`, `Display`, an error, a metric, a log or a report.
- `expose_secret` appears only in the readers `secret-access.toml` lists with a reason; a borrower (`borrows = true`) computes on tables a reader lends it and names no accessor. A listed file hands neither the value nor a reference to it to code outside the list: no function returns it, and no trait impl or exported macro exposes it. An engine takes a secret table by reference to its `Secret` (`Table::secret_base`), never as a slice its caller exposed.
- A secret that moves or grows lives on the heap (`Secret<Box<_>>`, `Secret<Vec<_>>` sized up front), so zeroization reaches its only copy (OBS-2). A buffer of derived values is a `SecretBuf`, which also clears the allocation it leaves when it grows; a plain `Vec` of them, a copy taken out (`to_vec`), or a derived array kept in a struct and never cleared is a finding.
- Randomness comes from `getrandom`; only the `test-rng` feature pins it, in test builds.

## Verifier robustness (RFC-0019)
- Every input path returns `Accept`, `Reject(reason)` or `Inconclusive(error)`: no panic, no allocation beyond `ResourceLimits`, no unbounded loop.
- The check order is the fixed order of RFC-0019, so one invalid proof always yields one reason.
- Each error names whose fault it is: the proof's (`Reject(reason)`), or the verifier's own on values it derived itself, such as a tree size from public data or indices from the transcript (`Inconclusive(ERR-INTERNAL)`).

## Evidence
- For each check the change adds or modifies, a test that violates exactly that check and asserts the exact `RejectReason` or error code, seen red before the fix.
- Property and fuzz oracles judge what was accepted without the code under test. They re-encode each accepted value from the value itself (rationals reduced, maps sorted, ranges checked), measure it against the limits, and predict error offsets. An oracle that echoes the input's own fields passes a decoder that skipped a check.
- A mutation run in a scratch copy: disable, reorder or shift each check, one mutant at a time, with a no-op control. Every mutant that changes behaviour fails a test, and the pull request records the count caught.
- A mutation run proves only that the checks present are tested. The review also lists each precondition the soundness argument needs and the line that enforces it; a missing one has no mutant to catch it.
- Wrap attacks are probed over Goldilocks at the boundary widths, every class count included. The test field F65537 hides them: there `2^16 − P = −1`, while Goldilocks leaves a window of `2^64 − P = 2^32 − 1`.
- A branch that honest hashing reaches only with negligible probability, such as a rejection-sampling retry or the cap on a squeeze loop, is driven by a stub source in a unit test. Without one, a mutant of the branch survives every test.
