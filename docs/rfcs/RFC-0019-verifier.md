# RFC-0019: Independent Verifier

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (driver), v0.4 (zero-knowledge checks, anchor, scoring), v1.0 (hardening)
- Amended by: [RFC-0031](RFC-0031-commitment-id.md) ('Check order', step 4: the advice commitment's `layout_digest`), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) ('Check order', step 7, in `tessera-protocol/2`), [RFC-0033](RFC-0033-tree-level-rejections.md) ('Check order', step 7: the reasons of the leaf stage's sumchecks), [RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies) ('Inputs' and 'Check order', step 4: the pinned sampler keys)

## Summary

`tessera-verify` is the whole trusted base of a Tessera audit. It is a separate Rust workspace with no dependency on prover code, no `unsafe`, and a short allowlist of third-party crates. It decodes inputs under explicit resource limits, validates the claim list by lowering it with its own code, replays the transcript, checks every tree level, stage, opening, column, anchor binding and scoring claim in a fixed order, and returns `Accept`, `Reject(reason)` or `Inconclusive(error)`. It never panics on any input. A persistent ledger tracks weight-commitment epochs, used blinding rows and audit accounting across sessions. The code is organized so each check maps to a predicate in a written verifier specification, the starting point for later mechanized proofs.

## Motivation

"Keep the verifier small and independent. It recompiles every constraint from the public claim list, shares no code with the prover, and is the target of formal verification" (source §6, principle 12). "Formally verify the uniqueness lemmas and the verifier core; differential-test against an independent reference" is the source's assurance item, "required for treaty use" (source §9, problem 9). The verifier is what a counterparty state actually runs; its failure modes are the system's failure modes.

## Goals

- Every check of the protocol implemented once, independently of the prover.
- Deterministic verdicts: the same inputs always yield the same verdict and reason.
- No panic, unbounded allocation or unbounded time on any input.
- Verification of a 16K-token, 16K-context 405B batch within SC-3 (≤ 300 s, ≤ 64 GiB on 32 cores).
- A written specification that states every check as a predicate.

## Non-Goals

- Mechanized proofs (NG-7; OQ-0019-1).
- GPU acceleration: CPU verification meets SC-3, and a GPU stack would enlarge the trusted base.
- Browser or WebAssembly builds.

## Proposed Design

### Crates

| Crate | Responsibility |
|---|---|
| `tv-field` | `Fp`, `Fp2`, roots of unity, NTT for re-encoding combined rows |
| `tv-codec` | TCE decoders with `DecodeLimits`; encoders only where digests require them |
| `tv-hash` | BLAKE3 and SHA-256, implemented in the crate ([Dependencies](#dependencies)); BLAKE3 domains, chained column digests, Merkle multiproofs, transcript replay with the TCE encoding of its `SessionContext`, proof-of-work checks; SHA-256 Merkle for recorder logs |
| `tv-tables` | Integer-only table generators and structured multilinear evaluation |
| `tv-core` | Claim-list validation and lowering, range re-derivation, schedule derivation, sumcheck and tree verification, gadget final checks, opening and column checks, anchor and scoring checks, soundness calculator, audit plan |
| `tv-transport` | Stream and directory transports for live sessions (only crate allowed to link `rustls`) |
| `tv-cli` | `tessera-verify` binary, JSON verdict output |
| `tv-testkit` | Test support: a reader for `spec/vectors/` and a seeded generator, with no third-party dependency; other verifier crates may use it only as a dev-dependency, so it never reaches `tessera-verify` |

### Dependencies

The verifier implements its two hash functions itself, in safe Rust from their specifications: BLAKE3 for every Tessera hash (RFC-0004) and SHA-256 (FIPS 180-4) for the recorder logs (RFC-0015). Taking the `blake3` and `sha2` crates would have brought seventeen crates into the trusted base, with unsafe SIMD code and about 250,000 lines to audit. The official BLAKE3 vectors and NIST's SHA-256 vectors check the two implementations, and a differential test checks BLAKE3 against the prover's crate. The allowlist of [RFC-0001](RFC-0001-workspace-and-verifier-independence.md#enforcement) applies to everything else: `ed25519-dalek` 2.x, `rayon` 1.x, `thiserror` 2.x; `clap` 4.x, `serde` 1.x and `serde_json` 1.x in `tv-cli` only; `rustls` 0.23.x in `tv-transport` only. `zeroize` 1.x clears the heap-resident unrevealed Live coin-toss share in `tv-core`; the verifier lockfile pins 1.8.2, covered by imported safe-to-deploy audits without exemptions. Unix entropy is read directly from `/dev/urandom`, with an explicit I/O failure elsewhere. Every crate declares `#![forbid(unsafe_code)]`; `tv-core` additionally denies `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::indexing_slicing` and `clippy::arithmetic_side_effects` on size computations (field arithmetic is exempt through its own types).

### Inputs

```rust
pub struct VerifyInputs {
    pub claims: ClaimList,                    // decoded from claims.tcl
    pub statement: PublicStatement,           // from the proof's Round1
    pub manifest: LogManifest,
    pub sample: AuditSample,
    pub anchor_entries: Vec<(DigestLogEntry, InclusionProof)>,
    pub recorder_keys: Vec<PublicKey>,        // pinned
    pub sampler_keys: Vec<PinnedSamplerKey>,  // pinned, checked at load against the policy (RFC-0061)
    pub sampler_policy: SamplerPolicy,        // W, G, b_max (RFC-0061)
    pub ledger: VerifierLedger,
    pub weights: WeightCommitmentPublic,      // the epoch's weight commitment, from WEIGHTS.twr (RFC-0040)
}
```

### Check order

Checks run in this order; the first failure determines the verdict.

1. Decode every input under `ResourceLimits`: an artifact of the prover's that fails is `Reject(Malformed)`, one of the verifier's own inputs `Inconclusive(ERR-CODEC-*)` ([RFC-0043](RFC-0043-rejection-payloads.md#malformed)), and an exhausted resource `Inconclusive(ERR-RES-*)`. The sampler pins and policy load first, and a pin list or policy that breaks a rule of [RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins) is `Inconclusive(ERR-USAGE)`.
2. Accept the parameter set (`Reject(Params)`).
3. Validate and lower the claim list: [RFC-0007](RFC-0007-claim-list.md#verifier-validation) steps 1–7, including table regeneration and range re-derivation (`Reject(StatementMismatch)`).
4. Bind the statement: claim-list digest; `weights_root` and `epoch` equal to the ledger's current root for that model, which `weights` hashes to, with the claim list's weight layout ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-verifiers-weight-inputs)); manifest signature under a pinned key; sample membership and no prior acceptance of the batch's requests; each request's `prompt_len + output_len` at most the claim list's `max_context` ([RFC-0047](RFC-0047-advice-order.md#a-requests-sequence-length), `StatementMismatch { field: Batch }`); anchor entries' inclusion proofs ([RFC-0015](RFC-0015-transcript-anchor.md#verifier-checks), [RFC-0016](RFC-0016-audit-sampling.md#verifier-checks)), with each temperature request's pinned sampler key ([RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies), `Reject(Anchor { kind: SamplerKey })`); the advice commitment's `layout_digest` equal to the `commitment_id` the verifier derives ([RFC-0031](RFC-0031-commitment-id.md#layout_digest), `Reject(StatementMismatch)`).
5. Weight reservation: declared blinding rows unused in this epoch according to the ledger (`Reject(Budget)`).
6. Transcript: rebuild `SessionContext`, absorb `Round1`, derive or check the seed (coin-toss transcript, or proof-of-work nonce in non-interactive mode; `Reject(ProofOfWork)`).
7. Tree levels, in this order:
   1. The LogUp roots: for each instance, `d_a ≠ 0`, `d_t ≠ 0` and `n_a·d_t + n_t·d_a = 0` ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)) (`Reject(Lookup)`).
   2. The levels of the fraction trees and the product forests, from the root (`Reject(TreeLevel { level, round })`, [RFC-0033](RFC-0033-tree-level-rejections.md)).
   3. Each LogUp instance's leaf checks: the lookup tree's numerator claim is 1, and the table tree's denominator claim is `α − t̃(ρ)` (`Reject(Lookup)`).

   In `tessera-protocol/2` ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)), step 7.1 is the LogUp identity sumcheck, with its final check (`Reject(LogUpIdentity { round })`, [RFC-0033](RFC-0033-tree-level-rejections.md)). Step 7.2 runs masked, and step 7.3 reads the padding trees' coordinate and the masked extensions of keys, values and multiplicities.
8. Stages in schedule order, including masked-sumcheck terms and gadget final checks (`Reject(Sumcheck)`, `Reject(Gadget)`).
9. Opening groups for advice and every weight matrix: `Σ γ v = ⟨u'_g, eq⟩` per group (`Reject(Evaluation)`).
10. Column queries: live, the verifier draws them now from its CSPRNG; non-interactive, from the transcript after the query proof-of-work.
11. Columns: leaf recomputation, multiproofs against each matrix's root in `weights.roots` for the weights, consistency of every combined row with every opened column (`Reject(MerklePath)`, `Reject(Proximity)`).
12. Scoring: per-request `Û_r` and the batch total against `accept_bound` (`Reject(UBoundExceeded)`).
13. Update the ledger (used blinding rows, opened weight columns, proven requests) and return `Accept(report)`.

In live mode, steps 1–6 complete before `SeedReveal` is sent and steps 7–9 before `ColumnQueries`; step 10 draws the queries `ColumnQueries` carries, and steps 11–13 run after `Round3Response`.

### Current protocol/1 composition

`tv-core::driver` checks trusted input coherence, parameter name/mode/layout,
claim-list validation and statement fields before seed processing. It then runs
the complete runtime leaf/split/native/graph reductions, derives original opening
groups and required matrices, enforces their soundness budget before evaluation
checks, fixes every combined row before query generation, and checks every
queried advice/weight column. Exact proofs establish an unexplained-information
bound of zero; the scoring circuit and persistent ledger are later work.

The current proof unit contains one exact-greedy request. Unsupported batches or
predictors return Usage. The driver never treats a supplied statement or weight
artifact as its own trust anchor. Live entropy and session state are verifier-owned;
Round3 consumes the session. To reconstruct borrowed opening state, Round3 currently
replays its immutable cached Round2; it cannot draw new queries. Full allocation
accounting and deadlines inside arithmetic loops remain #182. The reference CLI
authenticates an externally supplied digest of the whole `PublicStatement`
before using the decoded statement as its expected input. Its measured load,
pin and verification trace is optional execution context; deterministic core
JSON retains null internal check durations. Anchor, ledger, audit commands and
live transport remain later work, as the target milestones below require.

### Ledger

```rust
pub struct VerifierLedger {                   // schema 0x0701, persisted, verifier-local
    pub weight_roots: Vec<(Digest /*claim list*/, Vec<(u32 /*epoch*/, Digest /*root*/)>)>,  // linked by WeightLinkProofs
    pub used_blinding_rows: Vec<(Digest, u32, u8, BitSet)>,  // (root, epoch, matrix, rows)
    pub proven: Vec<(Digest /*sample*/, BitSet /*selected index*/)>,
}
```

The ledger is updated only after `Accept` and is written atomically. `tessera-verify weights link` checks a `WeightLinkProof` and appends the new epoch.

### Robustness

- Every length is checked against limits and remaining bytes before allocation ([RFC-0003](RFC-0003-canonical-encoding.md#limits)).
- Size arithmetic uses checked operations; overflow is `Reject(Malformed)`.
- Recursion depth in decoders is bounded; lowering is iterative.
- A wall-clock limit aborts with `Inconclusive(ERR-RES-TIME)`.
- A panic anywhere is caught at the CLI boundary and reported as `Inconclusive(ERR-INTERNAL)`, never as `Accept`; tests assert no panic is reachable from fuzzed inputs.

### Performance plan (16K tokens, 16K context, 405B, `live-64`)

| Work | Size | Estimate (32 cores) |
|---|---|---|
| Decode and hash columns | about 1.18 GB advice + 0.3 GB weights (complete RFC-0050 layout) | recalibrate at the complete row count |
| Column consistency | `(G + 1) · τ` inner products over 953,580 rows | recalibrate at the complete row count |
| Re-encode combined rows | `G + 1 ≤ 17` coset NTTs of size 2^23 over `Fp2`, plus weights | about 5 s |
| Tables | ≤ 20 dense cores of ≤ 2^24 values | about 2 s |
| Lowering and range re-derivation | 405B claim list | about 5 s |
| Stages and trees | about 10^4 rounds | < 1 s |

Total about 15 s, well inside SC-3; the measured figure is published at v1.0.

### Verifier specification

`docs/verifier/spec.md` states, for each step above, the predicate the code checks, the soundness lemma it relies on (with references to RFC-0009's uniqueness lemmas, RFC-0011's proximity bound and RFC-0014's budget), and the exact correspondence to functions in `tv-core`. It is written so a reviewer can audit the code against it line by line and so a later mechanization can target it.

## Alternatives Considered

1. **Verifier inside the prover workspace.** Rejected by RFC-0001.
2. **Accepting a prover-supplied constraint system.** Rejected by RFC-0007 (SP-7).
3. **GPU-accelerated verification.** Rejected: CPU verification fits SC-3 with a wide margin, and GPU drivers would join the trusted base.
4. **Stateless verifier.** Rejected: blinding-row reuse, epoch continuity and duplicate request proofs can only be checked across sessions.

## Drawbacks

- The verifier contains a full lowering and range analysis, so its trusted base is larger than a pure checker's.
- The ledger is state that must be backed up; losing it loses reuse detection, not soundness.

## Migration / Rollout

v0.3: decoding, claim-list validation, transcript, trees, stages, openings, columns, non-interactive and live drivers without zero knowledge. v0.4: masked checks, anchor, scoring, ledger. v1.0: fuzzing campaign, resource-limit hardening, the written specification, reproducible builds.

## Testing Strategy

- Tamper corpus ([07 §4](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)): each mutation rejected with the expected reason at the expected step.
- Determinism: verifying the same inputs 100 times gives the same verdict and report bytes.
- Fuzzing: `cargo-fuzz` targets for every decoder, claim-list validation, multiproof verification and the full `verify_non_interactive` entry point; 1 hour nightly, 24 CPU-hours before v1.0, with no panic, no limit violation and no accept of a mutated proof.
- Limits: inputs crafted to exceed each limit yield the documented `Inconclusive` or `Reject` without exceeding memory caps (measured with a counting allocator).
- Isolation: `cargo xtask check-isolation` (RFC-0001).
- Performance: the 405B verification time and peak memory are measured at v1.0 against SC-3.

## Open Questions

- **OQ-0019-1.** Which route to mechanized proofs: extraction of `tv-core` into Lean 4 (for example through hax or Aeneas) or a separate Lean model tied to the code by refinement tests? Owner: @AbdelStark. Resolve in v1.1 planning, after the written specification exists.

## References

- Source §6 principle 12, §9 problem 9.
- [RFC-0001](RFC-0001-workspace-and-verifier-independence.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [04 Error model](../spec/04-error-model.md).
