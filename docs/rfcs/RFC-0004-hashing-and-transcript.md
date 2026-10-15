# RFC-0004: Hashing, Merkle Trees, and the Fiat–Shamir Transcript

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1

## Summary

Tessera uses BLAKE3 with per-domain derived keys as its only hash function for commitments, Merkle trees, envelopes, proof-of-work and Fiat–Shamir. Commitment leaves are chained column digests that need 32 bytes of state per column, which keeps streaming commitment of terabyte witnesses feasible. The transcript is a chaining-value construction with labeled absorbs, deterministic rejection sampling of challenges in `Fp` and `Fp2`, and an optional proof-of-work step. This RFC fixes all three so the prover, the GPU kernels and the independent verifier compute identical bytes.

## Motivation

The source sets 30 ps per committed 64-bit element, including rate-1/2 encoding and Merkle hashing, as the dominant unit cost (source §5). At rate 1/2 that is 16 codeword bytes hashed per committed element, about 530 GB/s per H100. The hash must be fast on GPUs, standard enough for a treaty verifier, and simple to implement independently. Fiat–Shamir must be specified to the byte because the prover can grind it with very large budgets (source §2, adversary OC5), and every challenge must derive from one seed (source §7, "one seed drives every stage").

## Goals

- One hash primitive, domain-separated, with 128-bit collision resistance.
- Streaming column commitments with constant state per column.
- A transcript whose challenges are exactly uniform in `Fp`, `Fp2` and power-of-two index ranges.
- Proof-of-work grinding with a verifier cost of one hash.
- Byte-identical behavior across CPU, GPU and the verifier.

## Non-Goals

- Arithmetization-friendly hashing for recursion (non-goal NG-3).
- The recorder's hash (SHA-256), fixed by RFC-0015.

## Proposed Design

### Hash function and domains

`H(domain, bytes) = BLAKE3.derive_key(context = "tessera/v1/" + domain, key_material = bytes)`, 32-byte output. Domains:

| Domain | Input | Used by |
|---|---|---|
| `envelope` | Envelope header and payload | RFC-0003 |
| `column-init` | `commitment_id: Digest ‖ column: u64` | Column digests |
| `column-step` | `chain: Digest ‖ symbols: [Fp; ≤ B]` | Column digests |
| `merkle-node` | `left: Digest ‖ right: Digest` | Merkle trees |
| `column-merge` | `segment digests of one column, in segment order` | Advice leaves (RFC-0011) |
| `weights-root` | `TCE(WeightLayoutParams) ‖ matrix roots ‖ epoch` | Weight root (RFC-0012) |
| `transcript-init` | `TCE(SessionContext)` | Transcript |
| `transcript-absorb` | `state ‖ Ascii(label) ‖ TCE(value)` | Transcript |
| `transcript-squeeze` | `state ‖ Ascii(label) ‖ counter: u64` | Transcript |
| `transcript-ratchet` | `state ‖ out: Digest` | Transcript |
| `pow` | `state ‖ Ascii(label) ‖ nonce: u64` | Grinding |
| `coin-commit` | `share_v: Digest` | Verifier's seed-share commitment (RFC-0014) |
| `coin-toss` | `share_v: Digest ‖ share_p: Digest` | Live seed (RFC-0014) |
| `table` | `TCE(TableSpec without digest) ‖ piece parameters ‖ core values` | Table digests (RFC-0008) |
| `audit-sample`, `audit-select` | Sampling seed and per-request selection inputs | Audit sampling (RFC-0016) |
| `claim-list` | `TCE(ClaimList)` | Claim-list digest (RFC-0007) |
| `statement` | `TCE(PublicStatement)` | `SessionContext.statement_digest` (RFC-0007) |

Context strings are distinct ASCII strings, so outputs of different domains are independent under the standard assumption on BLAKE3's key derivation mode. The Rust crate `blake3` 1.x is used on CPU (prover and verifier, allowlisted); a CUDA implementation in `tessera-gpu` covers `column-step` and `merkle-node`.

### Column digests

A committed matrix has `m` rows of codewords of length `n` (RFC-0011). Rows are produced in blocks of `B` rows. `B` is a public commitment parameter, recorded in the commitment's layout; the advice commitment uses 32 ([RFC-0011](RFC-0011-advice-commitment.md#column-digests-and-root)). For column `j`:

```text
d_j[0]   = H("column-init",  commitment_id ‖ u64(j))
d_j[i+1] = H("column-step",  d_j[i] ‖ symbols of rows iB .. min((i+1)B, m) − 1 at column j)
leaf_j   = d_j[ceil(m / B)]
```

The symbols of a step are their 8-byte TCE encodings, concatenated with no length prefix: `B` and `m` fix how many there are. The prover keeps one 32-byte chain value per column: 256 MiB for `n = 2^23`. The verifier, given an opened column (all `m` symbols), recomputes the chain. `commitment_id` binds the digest to the commitment's descriptor (shape, rate, `B`, purpose), so columns of different commitments never collide.

### Merkle trees

- Binary tree over `n` leaves (`n` a power of two), node `= H("merkle-node", left ‖ right)`, root at level `log2 n`.
- **Batch openings** use a canonical multiproof: given the sorted, deduplicated set of queried leaf indices, the proof lists, level by level from the leaves upward and left to right within a level, exactly the sibling digests the verifier cannot compute itself. The verifier rejects a multiproof with extra or missing digests, and an empty set of opened leaves.
- The verifier takes `n` from public data, never from the prover. The nodes at level `k` are the leaves of a tree of `n / 2^k` leaves with the same root, so a prover choosing `n` could open internal nodes as leaves.

```rust
pub struct MultiProof { pub siblings: Vec<Digest> }   // canonical order as above
pub fn verify_multiproof(root: &Digest, n: u64, leaves: &[(u64, Digest)], proof: &MultiProof) -> bool;
```

### Transcript

```rust
pub enum Mode { Live, NonInteractive }  // TCE tags 0 and 1, in declaration order

pub struct SessionContext {
    pub protocol: Ascii,              // "tessera-protocol/1"
    pub mode: Mode,
    pub params: ParamSetId,           // RFC-0014 'Parameter sets'
    pub statement_digest: Digest,     // H("statement", TCE(PublicStatement))
    pub session_id: [u8; 16],
}

pub struct Transcript { state: Digest, counter: u64 }

impl Transcript {
    pub fn new(ctx: &SessionContext) -> Self;                         // state = H("transcript-init", TCE(ctx))
    pub fn absorb<T: Encode>(&mut self, label: &str, v: &T);  // state = H("transcript-absorb", state ‖ Ascii(label) ‖ TCE(v))
    pub fn squeeze(&mut self, label: &str) -> Digest;
    pub fn challenge_fp(&mut self, label: &str) -> Fp;
    pub fn challenge_fp2(&mut self, label: &str) -> Fp2;
    pub fn challenge_indices(&mut self, label: &str, log_n: u32, count: u32) -> Vec<u64>;
    pub fn grind(&mut self, label: &str, bits: u8) -> u64;    // prover: finds nonce, absorbs it
    pub fn check_grind(&mut self, label: &str, bits: u8, nonce: u64) -> bool; // verifier
}
```

- `squeeze`: `out = H("transcript-squeeze", state ‖ Ascii(label) ‖ counter)`; `counter += 1`; `state = H("transcript-ratchet", state ‖ out)`; return `out`. A label is TCE `Ascii`, its length then its bytes, so no two label and value pairs share an input.
- `challenge_fp`: repeat `squeeze`; scan its four 8-byte little-endian words in order; return the first word `< P`. Each word is rejected with probability below 2^-32, so the output is exactly uniform on `Fp`.
- `challenge_fp2`: repeat `squeeze` until its first two words are both `< P`; return them as `(c0, c1)`. Exactly uniform on `Fp2`.
- `challenge_indices`: each squeezed word masked to its low `log_n` bits (`log_n ≤ 64`), drawn with replacement, in order, four per squeeze; `count = 0` squeezes nothing. Exactly uniform for power-of-two domains, so nothing is rejected.
- `grind`: the prover searches `nonce = 0, 1, 2, …` for the first value whose `H("pow", state ‖ Ascii(label) ‖ nonce)` has its first `bits` bits zero (most significant bit of byte 0 first); it then absorbs `nonce` under `label`. `check_grind` recomputes one hash and, when the nonce meets the requirement, absorbs it identically; an invalid nonce is not absorbed, and the verifier rejects the proof (`Reject(ProofOfWork)`) without using the transcript again. The honest prover uses the smallest valid nonce, so its output is deterministic; the verifier accepts any nonce that meets the bit requirement, since soundness accounting already charges the prover for every attempt.

Labels are ASCII strings built only from the tier-A label patterns of [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule), which fix the full schedule per protocol version.

### Invariants

- **INV-FS-1.** Every prover message is absorbed before any challenge that depends on it is squeezed.
- **INV-FS-2.** Every challenge is a deterministic function of the `SessionContext`, the seed and prior prover messages (INV-ARCH-4).
- **INV-FS-3.** Absorbed values are TCE bytes; nothing is absorbed in any other form.
- **INV-FS-4.** Distinct protocol steps use distinct labels; the label table has no duplicates.

### Failure modes

| Failure | Handling |
|---|---|
| Prover and verifier label tables diverge | Every challenge differs; the first sumcheck check fails; caught by transcript vectors in CI |
| Grinding nonce invalid | `Reject(ProofOfWork)` |
| Merkle multiproof non-canonical | `Reject(MerklePath)` |
| GPU kernel hash differs from CPU | Differential test failure; release blocker |

## Alternatives Considered

1. **SHA-256.** Considered for standardization and CPU hardware acceleration. Rejected for the commitment path: about 40 integer operations per byte against about 15 for BLAKE3, which would consume the whole 30 ps committed-element budget on an H100 (RISK-8). Kept for the recorder (RFC-0015), where hardware support matters more than throughput.
2. **SHA-3 / Keccak-256.** Standardized; rejected for GPU throughput and a 1600-bit state that is awkward in registers.
3. **Poseidon2 over Goldilocks.** Considered because it makes recursive compression cheap. Rejected for v1: recursion is out of scope (NG-3), field-native hashing is roughly an order of magnitude slower than BLAKE3 on GPUs for this input volume, and its cryptanalysis record is shorter. The hash sits behind a trait, so a superseding RFC can add it for recursion.
4. **BLAKE2s.** Considered: fast on GPUs and used by Stwo. Rejected in favor of BLAKE3, which has fewer rounds (7 against 10), the same 32-bit word design, and a built-in key-derivation mode for domain separation.
5. **A STROBE/Keccak duplex transcript (Merlin).** Rejected: it adds a second primitive and its specification is tied to one library.
6. **One streaming hasher per column instead of chained digests.** Rejected: a BLAKE3 hasher state is about 1.9 KB, which is 16 GB for 2^23 columns, against 256 MiB for 32-byte chains.

## Drawbacks

- BLAKE3 is not a NIST standard. Treaty parties that require FIPS primitives would need a superseding RFC with SHA-256 and a slower prover.
- Chained column digests fix the row-block size `B` in the commitment descriptor; changing `B` changes the commitment.

## Migration / Rollout

v0.1. The protocol label table is first frozen at `tessera-protocol/1` (v0.3).

## Testing Strategy

- BLAKE3: official test vectors for `derive_key` on CPU and GPU implementations.
- Domains: a test asserts all context strings are distinct and that `H(d1, x) ≠ H(d2, x)` on sample inputs.
- Column digests: random matrices with `m` not a multiple of `B`; CPU, GPU and verifier chains agree; changing any symbol changes the leaf.
- Merkle: random trees of size 2^1 to 2^20; single and batch openings verify; a multiproof with one extra, missing or reordered sibling is rejected.
- Transcript vectors in `spec/vectors/transcript/`: scripted absorb and squeeze sequences with expected challenges, checked by prover and verifier crates.
- Rejection paths: a test-only hash stub forces words `≥ P` to exercise every retry branch of `challenge_fp` and `challenge_fp2`, and the multi-squeeze path of `challenge_indices`, which never rejects.
- Uniformity smoke test: 2^20 `challenge_fp` outputs reduced modulo 2^16 pass a chi-square test at significance 0.001.
- Grinding: a valid nonce for `bits = 16` verifies; the same nonce fails for `bits = 24` unless it happens to have 24 leading zero bits (probability 1/256, so the test pins a nonce that has not); a valid but non-minimal nonce is accepted by `check_grind`.

## Open Questions

- **OQ-0004-1.** If the v0.1 measurement shows BLAKE3 column hashing alone above 20 ps per committed element on H100, is reducing the hashed volume (for example hashing only systematic symbols plus a separate parity check) sound and worth it? Owner: @AbdelStark. Resolve at the v0.1 gate ([RFC-0020](RFC-0020-cost-model.md#decision-rule)).

## References

- Source §2 (adversary), §5 (unit costs), §7 (one seed).
- O'Connor, Aumasson, Neves, Wilcox-O'Hearn, "BLAKE3: one function, fast everywhere", 2020.
- Canetti et al., "Fiat–Shamir: from practice to theory", STOC 2019; Ben-Sasson, Chiesa, Spooner, "Interactive Oracle Proofs", TCC 2016 (state-restoration soundness).
