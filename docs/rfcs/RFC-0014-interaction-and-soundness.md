# RFC-0014: Interaction Model, Soundness Budget, and Parameter Selection

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (protocol), v0.4 (zero-knowledge messages), v1.0 (live transport)

## Summary

Tessera runs in two modes over the same messages. **Live mode** is a bilateral session: the prover commits, a coin toss fixes one seed, the prover sends every sumcheck and combined row, and the verifier draws column queries after everything else is fixed. **Non-interactive mode** derives the seed and the queries from the transcript, each behind 20 bits of proof-of-work. This RFC fixes the message schemas, the session order, the transcript schedule, the soundness accounting and three named parameter sets: `live-64` (64-bit live proximity, at least 100-bit algebraic soundness), `ni-100` (at least 100 bits throughout) and `test-small`.

## Motivation

The source's interaction design: "Round 1: every commitment; weights by reference. Round 2: one seed from the verifier or a beacon, from which all sumcheck challenges follow by Fiat–Shamir over the 128-bit extension. Round 3: live column queries after all sumcheck messages. Targets: at least 100 bits for the algebraic parts, at least 64 bits for proximity per live audit" (source §7). Its decisions table makes the bilateral live protocol the default, with an optional transferable proof (source §10). Two numeric gaps in the source need closing: `τ = 100` gives 41.5 bits at rate 1/2 under unique-decoding analysis (RISK-4), and the LogUp challenge error near 2^-90 falls below 100 bits when grindable (RISK-5).

## Goals

- Exact message order and contents for both modes.
- A soundness calculator whose output is the authority for every parameter set.
- Every security target of [06 §4](../spec/06-security.md#4-security-properties) met with stated margins.

## Non-Goals

- Public randomness beacons as a seed source. Authenticating beacon output would put pairing-based signature verification into the verifier's trusted base; deferred.
- Recursive compression of non-interactive proofs (NG-3).

## Proposed Design

### Session order (live mode)

| Step | Direction | Message (schema) | Content |
|---|---|---|---|
| 0 | V → P | `SeedCommit` (`0x0501`) | `session_id`, `c_V = H("coin-commit", s_V)` for a fresh 32-byte `s_V` |
| 1 | P → V | `Round1` (`0x0510`) + `ProverSeedShare` (`0x0502`) | Commitments; fresh 32-byte `s_P` |
| 2 | V → P | `SeedReveal` (`0x0503`) | `s_V`; prover checks `c_V` |
| 3 | P → V | `Round2Response` (`0x0511`) | Tree levels, stages, masked sums, combined rows, `Û` claims |
| 4 | V → P | `ColumnQueries` (`0x0512`) | `τ_A` advice indices; `τ_W` indices per weight matrix |
| 5 | P → V | `Round3Response` (`0x0513`) | Opened columns, multiproofs |
| 6 | V | — | `Verdict` |

`seed = H("coin-toss", s_V ‖ s_P)` (the `coin-commit` and `coin-toss` domains are listed in RFC-0004). Because the verifier commits first and the prover's share accompanies `Round1`, the seed is uniform if either party is honest, and it is fixed only after every commitment.

In **non-interactive mode** the prover produces `NiProof` (`0x0520`) = `Round1`, a seed proof-of-work nonce, `Round2Response`, a query proof-of-work nonce and `Round3Response`; the seed and queries are squeezed from the transcript.

### Messages

```rust
pub struct Round1 {
    pub protocol_version: u16,
    pub session_id: [u8; 16],
    pub statement: PublicStatement,             // 03 §4
    pub advice: AdviceCommitment,               // RFC-0011
    pub weights: WeightReservation,             // epoch and the blinding rows this proof uses, RFC-0012
}
pub struct Round2Response {
    pub logup: LogUpProof, pub product_trees: TreeProof,   // leaf stage, RFC-0008/0010
    pub stages: Vec<StageProof>,                            // RFC-0010, masked per RFC-0013
    pub advice_blinding: Vec<Fp2>,                          // q_g per claim group, before the challenges (RFC-0013)
    pub weight_blinding: Vec<Fp2>,                          // q_T per matrix, before θ_T (RFC-0012)
    pub advice_combined: Vec<CombinedRow>,                  // u'_g, M_g per group, plus proximity row
    pub weight_combined: Vec<WeightCombinedRows>,           // per matrix
    pub scoring: ScoringClaims,                             // per-request Û claims, RFC-0017
}
pub struct ColumnQueries { pub advice: Vec<u32>, pub weights: Vec<(u8, Vec<u32>)> }  // live: τ_A, and τ_W per matrix, in draw order
pub struct Round3Response {
    pub advice_columns: Vec<Vec<Fp>>, pub advice_proof: MultiProof,   // one per distinct queried index, increasing
    pub weight_columns: Vec<(u8, Vec<Vec<Fp>>, MultiProof)>,          // likewise per matrix
}
pub struct NiProof { pub round1: Round1, pub seed_nonce: u64, pub round2: Round2Response,
                     pub query_nonce: u64, pub round3: Round3Response }
```

### Transcript schedule

Tier-A label patterns, in order; `{k}`, `{j}`, `{ℓ}`, `{c}`, `{T}` are decimal indices, counted from 0, or matrix names.

| Order | Operation | Label |
|---|---|---|
| 1 | init | `SessionContext` (protocol, mode, params, statement digest, session id) |
| 2 | absorb | `round1` |
| 3 | live: absorb `seed`; NI: grind `seed-pow`, then squeeze | `seed` |
| 4 | squeeze `Fp2` × 2, then absorb the fraction trees' roots ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)) | `logup/alpha`, `logup/beta`, `logup/roots` |
| 5 | per tree level `ℓ`: squeeze the root point of the forests that join at `ℓ` ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)), squeeze `γ_ℓ`, and `λ_ℓ` at a level with fraction trees, absorb round messages, squeeze challenges, absorb evaluations, squeeze the line challenge; after the last level, absorb the lookup trees' value evaluations | `tree/{ℓ}/root/{c}`, `tree/{ℓ}/gamma`, `tree/{ℓ}/lambda`, `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}`, `tree/{ℓ}/evals`, `tree/{ℓ}/line`, `logup/values` |
| 6 | per stage `k`: absorb masked sum, squeeze `ρ_k`, squeeze `γ_k`, rounds, final evaluations | `stage/{k}/zk-sum`, `stage/{k}/zk-rho`, `stage/{k}/gamma`, `stage/{k}/round/{j}`, `stage/{k}/r/{j}`, `stage/{k}/evals` |
| 7 | absorb the advice groups' blinding claims, then each weight matrix's; squeeze the opening challenges | `pcs/advice/blinding`, `pcs/weights/{T}/blinding`, `pcs/advice/gamma`, `pcs/advice/xi`, `pcs/weights/{T}/gamma`, `pcs/weights/{T}/xi` |
| 8 | absorb combined rows | `pcs/advice/combined`, `pcs/weights/{T}/combined` |
| 9 | absorb scoring claims | `scoring` |
| 10 | live: absorb `ColumnQueries`; NI: grind `query-pow`, then squeeze indices | `queries`, `pcs/advice/queries`, `pcs/weights/{T}/queries` |

Both sides construct labels from these patterns only; the full expanded list for a claim list is part of the derived constraint system and covered by transcript vectors.

### Soundness budget

Security level is stated in bits: `λ` bits means an adversary spending `2^t` hash evaluations succeeds with probability at most `2^{t − λ}`, or, for one-shot events the prover cannot retry, with probability at most `2^{−λ}`.

| Component | Error bound | Live mode | Non-interactive mode |
|---|---|---|---|
| Sumcheck and tree rounds (degree `d ≤ 7`) | `d/|Fp2|` per round, round-by-round | ≥ 125 bits per round under Fiat–Shamir | same |
| All rounds, union bound (405B, 16K tokens, about 10^4 rounds) | `Σ d/|Fp2|` | ≥ 112 bits | ≥ 112 bits |
| Product-forest root claim (`2^s` trees, `s ≤ 32`) | `s/|Fp2|` per forest | ≥ 120 bits | same |
| LogUp identity (`N + T ≤ 2^{38}` per instance) | `(N + T)/|Fp2|` | about 90 bits, one shot: `α` derives from a seed the prover cannot retry | 90 + `g_seed` = 110 bits |
| Tuple compression | included in the LogUp bound (joint polynomial in `α, β`) | — | — |
| Opening evaluation checks | `T_g/|Fp2|` per claim group of `T_g` claims, `1/|Fp2|` per weight matrix ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)) | ≥ 110 bits | same |
| Proximity per commitment | `((1 + ρ')/2)^τ + n/|Fp2|` | `τ · b(ρ') ≥ 64` bits, one shot | `τ · b(ρ') + g_query ≥ 100` bits |
| Weight linking | `ν/|Fp2|` | ≥ 120 bits | ≥ 120 bits |
| Hash collisions | BLAKE3, 256-bit output | 128 bits | 128 bits |

`b(ρ') = −log2((1 + ρ')/2)` is the unique-decoding bound per query; `ρ' = (k + P)/n` includes the vanishing-mask coefficients. The bound takes the queries to be independent. Both modes draw column indices uniformly with replacement, and a repeated index is checked once at no cost in soundness ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7). The verifier also rejects a zero LogUp root denominator, of a lookup tree or a table tree, and any zero denominator.

### Grinding

In non-interactive mode the seed is squeezed only after a nonce whose `H("pow", state ‖ Ascii("seed-pow") ‖ nonce)` has 20 leading zero bits; the queries likewise after `query-pow`. Each grinding step costs the honest prover about 2^20 BLAKE3 evaluations (milliseconds) and adds 20 bits to every bound an adversary could otherwise retry: the LogUp identity (RISK-5) and the proximity test.

### Parameter sets

| Parameter | `live-64/1` | `ni-100/1` | `test-small/1` |
|---|---|---|---|
| Mode | Live | Non-interactive | Either |
| Challenge field | `Fp2` | `Fp2` | `Fp2` |
| Advice: `log2 k`, rate, mask `P_A` | 22, 1/2, 160 | 22, 1/2, 200 | 10, 1/2, 16 |
| Advice `τ_A` | 155 (`64.3` bits) | 193 (`80.1 + 20` bits) | 8 |
| Weights: `k_W`, rate, mask `P_W`, blinding `B_W` | 2^20, 1/4, 2^16, 2048 | 2^20, 1/4, 2^16, 2048 | 2^8, 1/4, 64, 16 |
| Weights `τ_W` | 97 (`64.0` bits) | 122 (`80.5 + 20` bits) | 8 |
| `g_seed`, `g_query` | 0, 0 | 20, 20 | 0, 0 |
| Row block `B`, segments `S` | 32, 8 | 32, 8 | 4, 1 |
| Accepted by release verifier | Yes | Yes | No (`test-params` feature only) |

A parameter set is identified by its name and version as the table's header writes them, `live-64/1`, `ni-100/1` or `test-small/1`: `pub struct ParamSetId(pub Ascii)`. The identifier is bound into every transcript through `SessionContext` ([RFC-0004](RFC-0004-hashing-and-transcript.md#transcript)) and into every claim list ([RFC-0007](RFC-0007-claim-list.md)); a new version of a set is a new identifier.

The source's `τ = 100` at rate 1/2 gives 41.5 bits under unique decoding; `live-64` uses 155 to meet the 64-bit target, at about 1.55× the source's column bytes (RISK-4). Non-interactive proofs need 193 advice columns with grinding instead of the source's "roughly 2.5×" (250) without it.

### Invariants

- **INV-PROTO-1.** A prover answers at most one seed per `Round1`. If a session aborts after `ProverSeedShare`, the prover discards that `Round1`, its advice commitment and its masking randomness; a retry starts with fresh randomness and a fresh weight reservation.
- **INV-PROTO-2.** The seed is fixed only after `Round1` is complete; the verifier's share is committed before the prover's share is seen.
- **INV-PROTO-3.** The verifier sends `ColumnQueries` only after `Round2Response` is fully received and absorbed. The prover rejects queries that come early, that are out of range, or whose count per matrix is not the parameter set's `τ_A` or `τ_W`. Repeated indices are allowed, since queries are drawn with replacement, and each is answered once (RFC-0011 step 7). A verifier therefore never opens more distinct columns than the count, which bounds what the encoding masks must hide.
- **INV-PROTO-4.** `session_id` is fresh per session and bound into `SessionContext`.
- **INV-PROTO-5.** Every parameter used by either side comes from the named parameter set in the claim list; there are no free parameters at runtime.

### Live transport

Messages are TCE envelopes. Two transports are defined; neither is part of the transcript:

- **Stream:** length-prefixed envelopes (u64 length, then envelope) over mutually authenticated TLS 1.3. Prover side uses `rustls` 0.23.x; the verifier side uses a separate crate `tv-transport` (the only verifier crate allowed to depend on `rustls`, per the allowlist), so verification logic never links a TLS stack.
- **Directory:** each message is a file `NN-<schema>.tce` in a shared or transferred directory, for air-gapped exchange.

Session deadlines (per message) are configuration, not protocol; defaults are 1 hour for `Round2Response` and 10 minutes for every other message (OQ-0014-1).

### Soundness calculator

`tessera params check` (prover side) and `tessera-verify params` (verifier side, independent code) take a parameter set and statement sizes (lookups and table rows per instance, rounds, degrees, committed rows) and print every row of the soundness table with its bits and the minimum. CI runs both for every parameter set at the 405B 16K-token sizes and fails if any bound misses its target or the two calculators disagree.

## Alternatives Considered

1. **Non-interactive only.** Rejected as the default: the source's default is the bilateral live protocol (source §10), which reaches 64-bit proximity with 155 columns instead of 193 plus grinding, and keeps proofs non-transferable unless the parties want them transferable.
2. **Verifier-chosen seed without a coin toss.** Rejected: a malicious verifier could pick challenges adversarially, and the masked-sumcheck simulation of RFC-0013 assumes uniform challenges.
3. **Johnson-bound (list-decoding) proximity analysis.** Considered: about 0.5 bits per query at rate 1/2 instead of 0.415. Rejected: provable bounds in that regime carry field-size terms that erode the margin, and conjectured bounds are not acceptable for treaty use.
4. **A larger extension field for LogUp only.** Considered to remove reliance on grinding for RISK-5. Rejected: the fraction tree would run in `Fp4`, more than doubling the cost of the dominant lookup term; 20 bits of grinding costs milliseconds.
5. **Public randomness beacon.** Deferred (Non-Goals).

## Drawbacks

- Live mode requires both parties online for the duration of a proof (tens of seconds to minutes per batch).
- Non-interactive proofs are about 25% larger in advice columns.
- Security accounting depends on the claim list's sizes; the calculator must be rerun for every model and batch-size class.

## Migration / Rollout

v0.3: messages, both modes, transcript schedule and the calculator, without zero-knowledge items (`test-params` gating). v0.4: zero-knowledge fields, `tessera-protocol/2`. v1.0: stream transport, deadlines, `tessera-protocol` frozen.

## Testing Strategy

- State machines: every out-of-order message is rejected with `ERR-PROTO-ORDER` on both sides; a second `SeedReveal` for the same `Round1` is refused by the prover (`ERR-PROTO-REUSE`); a reveal that does not match `c_V` yields `ERR-PROTO-COINTOSS`.
- Transcript vectors: for each conformance model and both modes, the full sequence of labels and challenges matches pinned vectors on both sides.
- Grinding: a non-interactive proof with a seed nonce one bit short of 20 is rejected with `Reject(ProofOfWork)`.
- Calculator: both implementations agree on every row for 1,000 random size profiles; `live-64` and `ni-100` meet their targets at 405B sizes; a profile with `N + T = 2^{50}` makes the LogUp row fail its target, proving the check is live.
- Transports: the stream and directory transports deliver byte-identical envelopes; a TLS peer without a valid client certificate is refused.

## Open Questions

- **OQ-0014-1.** What per-message deadlines and retry policy should live sessions use in production? Owner: @AbdelStark. Resolve at v1.0 with the parties' operational requirements; until then the defaults above apply.

## References

- Source §2 (security targets), §7 ("Interaction and soundness"), §10 (verification model).
- Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity gaps for Reed–Solomon codes", FOCS 2020.
- Canetti et al., "Fiat–Shamir: from practice to theory", STOC 2019; Ben-Sasson, Chiesa, Spooner, TCC 2016 (round-by-round and state-restoration soundness).
- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md).
