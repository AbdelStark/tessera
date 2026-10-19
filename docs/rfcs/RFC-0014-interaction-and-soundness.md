# RFC-0014: Interaction Model, Soundness Budget, and Parameter Selection

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (protocol), v0.4 (zero-knowledge messages), v1.0 (live transport)
- Amended by: [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) ('Transcript schedule', rows 4 and 5, and 'Soundness budget', in `tessera-protocol/2`), [RFC-0035](RFC-0035-padded-rows.md) ('Transcript schedule', row 6: the zero-check point)

## Summary

Tessera runs in two modes over the same messages. **Live mode** is a bilateral session: the prover commits, a coin toss fixes one seed, the prover sends every sumcheck and combined row, and the verifier draws column queries after everything else is fixed. **Non-interactive mode** derives the seed and the queries from the transcript, each behind 20 bits of proof-of-work. This RFC fixes the message schemas, the session order, the transcript schedule, the soundness accounting and three named parameter sets: `live-64` (64-bit live proximity; at least 100 bits for every challenge a prover can retry, and at least 64 for the one-shot LogUp identity, 86 at the 405B statement), `ni-100` (at least 100 bits throughout) and `test-small`.

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
| 0 | V → P | `SeedCommit` (`0x0501`) | `session_id`, chosen by the verifier, and `c_V = H("coin-commit", s_V)` for a fresh 32-byte `s_V` |
| 1 | P → V | `Round1` (`0x0510`) + `ProverSeedShare` (`0x0502`) | Commitments; fresh 32-byte `s_P` |
| 2 | V → P | `SeedReveal` (`0x0503`) | `s_V`; prover checks `c_V` |
| 3 | P → V | `Round2Response` (`0x0511`) | Tree levels, stages, masked sums, combined rows, `Û` claims |
| 4 | V → P | `ColumnQueries` (`0x0512`) | `τ_A` advice indices; `τ_W` indices per weight matrix |
| 5 | P → V | `Round3Response` (`0x0513`) | Opened columns, multiproofs |
| 6 | V | — | `Verdict` |

`seed = H("coin-toss", s_V ‖ s_P)` (the `coin-commit` and `coin-toss` domains are listed in RFC-0004). Because the verifier commits first and the prover's share accompanies `Round1`, the seed is uniform if either party is honest, and it is fixed only after every commitment.

In **non-interactive mode** the prover produces `NiProof` (`0x0520`) = `Round1`, a seed proof-of-work nonce, `Round2Response`, a query proof-of-work nonce and `Round3Response`; the seed and queries are squeezed from the transcript.

### Messages

[RFC-0036](RFC-0036-protocol-messages.md) defines every message field by field, with the prover's and the verifier's session state machines. Its schema version 1 is `tessera-protocol/1`, which has none of the zero-knowledge items below: the weight reservation, the blinding claims and the masked sums. `tessera-protocol/2` adds them in schema version 2. The struct below lists the messages' parts:

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
| 3 | live: absorb `seed`; NI: grind `seed-pow`, then squeeze. In `tessera-protocol/1`, then squeeze each weight matrix's proximity point ([RFC-0049](RFC-0049-weight-proximity.md#the-schedule)) | `seed`, `pcs/weights/{T}/xi` |
| 4 | squeeze `Fp2` × 2, then absorb the fraction trees' roots ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)); in `tessera-protocol/2`, the masked LogUp identity sumcheck in place of the roots ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#the-logup-identity-sumcheck)) | `logup/alpha`, `logup/beta`, `logup/roots`; `tessera-protocol/2`: `logup/identity/zk-sum`, `logup/identity/zk-rho`, `logup/identity/gamma`, `logup/identity/round/{j}`, `logup/identity/r/{j}`, `logup/identity/evals` |
| 5 | per tree level `ℓ`: squeeze the root point of the forests that join at `ℓ` ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)), squeeze `γ_ℓ`, and `λ_ℓ` at a level with fraction trees, absorb round messages, squeeze challenges, absorb evaluations, squeeze the line challenge; after the last level, absorb the lookup trees' value evaluations | `tree/{ℓ}/root/{c}`, `tree/{ℓ}/gamma`, `tree/{ℓ}/lambda`, `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}`, `tree/{ℓ}/evals`, `tree/{ℓ}/line`, `logup/values`; `tessera-protocol/2` adds `tree/{ℓ}/zk-sum` and `tree/{ℓ}/zk-rho`, absorbed and squeezed before `tree/{ℓ}/gamma` ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#masked-tree-levels)) |
| 6 | per stage `k`: absorb masked sum, squeeze `ρ_k`, in stage 1 squeeze the layer point ([RFC-0037](RFC-0037-stage-schedule.md#the-layer-point)), squeeze the zero-check point if the stage holds a `ZeroCheck` ([RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage)), squeeze `γ_k`, rounds, final evaluations, absorb the split values ([RFC-0037](RFC-0037-stage-schedule.md#splits)); the leaf stage's split values, stage 0's, follow `logup/values` | `stage/{k}/zk-sum`, `stage/{k}/zk-rho`, `stage/1/layer/{c}`, `stage/{k}/zero/{c}`, `stage/{k}/gamma`, `stage/{k}/round/{j}`, `stage/{k}/r/{j}`, `stage/{k}/evals`, `stage/{k}/split` |
| 7 | absorb the advice groups' blinding claims, then each weight matrix's; squeeze the opening challenges | `pcs/advice/blinding`, `pcs/weights/{T}/blinding`, `pcs/advice/gamma`, `pcs/advice/xi`, `pcs/weights/{T}/gamma`; `pcs/weights/{T}/xi` in `tessera-protocol/2` only |
| 8 | absorb combined rows | `pcs/advice/combined`, `pcs/weights/{T}/combined` |
| 9 | absorb scoring claims | `scoring` |
| 10 | live: absorb `ColumnQueries`; NI: grind `query-pow`, then squeeze indices | `queries`, `pcs/advice/queries`, `pcs/weights/{T}/queries` |

Both sides construct labels from these patterns only; the full expanded list for a claim list is part of the derived constraint system and covered by transcript vectors. What each label absorbs, the spelling of `{T}` as a matrix's name, and the labels `tessera-protocol/1` leaves out (`zk-sum`, `zk-rho` and the blinding claims) are [RFC-0036](RFC-0036-protocol-messages.md#what-the-transcript-absorbs)'s. In `tessera-protocol/2`, rows 4 and 5 are [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)'s: a masked LogUp identity sumcheck in place of `logup/roots`, and masked tree levels.

### Soundness budget

Security level is stated in bits: `λ` bits means an adversary spending `2^t` hash evaluations succeeds with probability at most `2^{t − λ}`, or, for one-shot events the prover cannot retry, with probability at most `2^{−λ}`.

| Component | Error bound | Live mode | Non-interactive mode |
|---|---|---|---|
| Sumcheck and tree rounds (degree `d ≤ 7`) | `d/\|Fp2\|` per round, round-by-round | ≥ 125 bits per round under Fiat–Shamir | same |
| All rounds, union bound (405B, 16K tokens, about 10^4 rounds) | `Σ d/\|Fp2\|` | ≥ 112 bits | ≥ 112 bits |
| Product-forest root claim (`2^s` trees, `s ≤ 32`) | `s/\|Fp2\|` per forest | ≥ 120 bits | same |
| LogUp identity (`N + T ≤ 2^{38}` per instance) | `(N + T)/\|Fp2\|`; `2(N + T)/\|Fp2\|` in `tessera-protocol/2`, which counts zero denominators (RFC-0032) | about 90 bits, one shot: `α` derives from a seed the prover cannot retry | 90 + `g_seed` = 110 bits |
| Tuple compression | included in the LogUp bound (joint polynomial in `α, β`) | — | — |
| Opening evaluation checks | `T_g/\|Fp2\|` per claim group of `T_g` claims, `1/\|Fp2\|` per weight matrix ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)) | ≥ 110 bits | same |
| Proximity per commitment | `((1 + ρ')/2)^τ + ϑ·n/\|Fp2\|` | `τ · b(ρ') ≥ 64` bits, one shot; the field term about 100 bits | `τ · b(ρ') + g_query ≥ 100` bits; the field term about 100 bits, which no proof of work covers |
| Weight linking | `ν/\|Fp2\|` | ≥ 120 bits | ≥ 120 bits |
| Hash collisions | BLAKE3, 256-bit output | 128 bits | 128 bits |

`b(ρ') = −log2((1 + ρ')/2)` is the unique-decoding bound per query; `ρ' = (k + P)/n` includes the vanishing-mask coefficients.

The field term is the tensor generator's. The proximity coefficients are `eq(ξ*, row)` for a random point `ξ*` with one coordinate per bit of the row index ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 3), not independent uniform values. Folding those `ϑ` coordinates one at a time costs one line proximity gap each. The false-witness probability is therefore `ϑ·n/|Fp2|`, with `n` the codeword length (Diamond and Posen, 'Proximity Testing with Logarithmic Randomness' (2024); Diamond and Gruen, 'Proximity Gaps in Interleaved Codes' (2025), Theorems 2 and 3).
- **`ϑ` for advice** is `ν − log2 k`, every bit of the row index: about 20 at 405B, 16K tokens and 16K context, whose 5.8×10^5 non-zero rows span a row index of 20 or 21 bits. The term is then `21 · 2^23/2^128 ≈ 2^{−100.6}`.
- **`ϑ_T` for weight matrix `T`** is `⌈log2⌉` of its data and blinding rows: at most 18 at 405B. In `tessera-protocol/1`, which commits no blinding rows and no mask for the weights, it is `⌈log2 m_T⌉` and the rate is `k_T / n_T`; a matrix with no committed rows has no proximity row ([RFC-0049](RFC-0049-weight-proximity.md#the-calculator)). The term is at most `18 · 2^22/2^128 ≈ 2^{−101.8}`.
- **Both targets hold.** `live-64` and `ni-100` keep about 100 bits each, with no grinding: `ξ*` is squeezed after the stages' final evaluations, which a prover can vary without redoing a proof of work ('Grinding'), so the term meets its target alone.

The bound holds in the unique-decoding regime only, `δ < (1 − ρ')/2`, where this table works. The bound takes the queries to be independent. Both modes draw column indices uniformly with replacement, and a repeated index is checked once at no cost in soundness ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7). In `tessera-protocol/1` the verifier also rejects a zero LogUp root denominator, of a lookup tree or a table tree, and any zero denominator.

### Grinding

In non-interactive mode the seed is squeezed only after a nonce whose `H("pow", state ‖ Ascii("seed-pow") ‖ nonce)` has 20 leading zero bits; the queries likewise after `query-pow`. Each grinding step costs the honest prover about 2^20 BLAKE3 evaluations (milliseconds) and adds 20 bits to the bounds whose challenges it alone guards:

- **`seed-pow`** to the LogUp identity (RISK-5): `α` and `β` are squeezed right after the seed, so a new pair costs a new nonce.
- **`query-pow`** to each commitment's column queries, `((1 + ρ')/2)^τ`: the indices are squeezed right after its nonce.

No proof of work covers any other bound. Their challenges follow prover messages that a cheating prover can vary without redoing a proof of work: the tree levels' and stages' challenges follow their own round messages, and the opening challenges of row 7 (`pcs/advice/gamma`, `pcs/advice/xi`, `pcs/weights/{T}/xi`) the stages' final evaluations. Each of those bounds meets its target alone, in the `2^{t − λ}` sense of 'Soundness budget'; the soundness calculator gives them no grinding bits.

### Parameter sets

| Parameter | `live-64/1` | `ni-100/1` | `test-small/1` |
|---|---|---|---|
| Mode | Live | Non-interactive | Either |
| Challenge field | `Fp2` | `Fp2` | `Fp2` |
| Advice: `log2 k`, rate, mask `P_A` | 22, 1/2, 160 | 22, 1/2, 200 | 10, 1/2, 16 |
| Advice `τ_A` | 155 (`64.3` bits) | 193 (`80.1 + 20` bits) | 8 |
| Weights: `k_W`, rate, mask `P_W`, blinding `B_W` | 2^20, 1/4, 2^16, 4096 | 2^20, 1/4, 2^16, 4096 | 2^8, 1/4, 64, 32 |
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

`tessera params check` (prover side) and `tessera-verify params` (verifier side, independent code) take a parameter set and statement sizes (lookups and table rows per instance, rounds, degrees, committed rows) and print every row of the soundness table with its bits, its target, whether it reaches the target, and the minimum. The committed rows give each commitment's `ϑ`: the advice row index's bits, and each weight matrix's. CI runs both for every parameter set at the 405B 16K-token sizes and fails if any bound misses its target or the two calculators disagree.

So that the two agree bit for bit, the rest of this section specifies the calculator to the integer. It computes with integers only: floating point decides nothing, and no platform's `log2` can move a bit. Both calculators reproduce the reports of [`spec/vectors/soundness/`](../../spec/vectors/soundness/README.md), which an independent oracle writes, and print the `tessera.soundness/1` report ([schema](../../spec/schemas/soundness-v1.schema.json)).

**Statement sizes.**

| Size | What it counts |
|---|---|
| `protocol` | The `n` of `tessera-protocol/n`: 1, or 2 with the zero-knowledge items |
| `rounds` | Every round of every stage and tree level, as a count and a degree `d`; in `tessera-protocol/2`, the LogUp identity sumcheck's too |
| `stages` | The instances `m` each reduction stage batches; in `tessera-protocol/2`, the identity sumcheck is one more |
| `tree_levels` | The instances `m` each tree level batches: its fraction trees and product forests |
| `zero_checks` | The variables `n_i` of each `ZeroCheck` ([RFC-0035](RFC-0035-padded-rows.md#zero-check-points)) |
| `forests` | Each product forest's `s`: it holds `2^s` trees |
| `lookups` | Each LogUp instance's lookups `N` and table rows `T` |
| `claim_groups` | Each advice claim group's claims `T_g` |
| `advice_rows` | The rows the advice row index spans, `2^{ν − log2 k}` |
| `matrices` | Each opened weight matrix, in `WeightMatrix` order: its message width `k_T`, its committed data rows `m_T` and the variables `ν_T` of its polynomial |

A calculator refuses, with `ERR-USAGE`: a `protocol` other than 1 or 2; `τ_A` or `τ_W` outside `1..=2^16`; a codeword of no symbol or of more than `2^32`, the advice's `n_A = 2^{log2 k + log2 inv_rate}` or a matrix's `n_T = k_T·2^{log2 inv_rate}`; `advice_rows = 0`; in `tessera-protocol/2`, a matrix with `m_T + B_W = 0` (in protocol/1 a matrix with no committed row has no proximity or queries row, [RFC-0049](RFC-0049-weight-proximity.md#the-calculator)); matrices not in strictly increasing `WeightMatrix` order.

**Rows.** `q = |Fp2| = p^2`, `p = 2^64 − 2^32 + 1`. A numerator is an exact integer that saturates at `2^128 − 1`, and a maximum over no item is 0.

| Row | Error bound `ε` | Grinding | Target |
|---|---|---|---|
| Sumcheck and tree rounds | `D/q`, `D` the largest of: each `d` of a nonzero count; each stage's and tree level's `m`; each `n_i`; 1 when there is a tree level, its line challenge; 1 in `tessera-protocol/2` when there is a stage or a level, its mask's `ρ` | — | `bits` |
| All rounds, union bound | `U/q`, `U = Σ count·d + Σ_stages m + Σ_levels (m + 1) + Σ n_i`, plus the number of stages and levels in `tessera-protocol/2` | — | `bits` |
| Product-forest root claims | `max s/q` | — | `bits` |
| LogUp identity | `v·max(N + T)/q`, `v = 1`, and 2 in `tessera-protocol/2` | `g_seed` | `one_shot` |
| Opening evaluation checks | `max(max T_g, 1 when a matrix is opened)/q` | — | `bits` |
| Proximity queries, per commitment | `((n + k + P)/(2n))^τ`, `((1 + ρ')/2)^τ`: the advice's `n_A`, `k_A = 2^{log2 k}`, `P_A`, `τ_A`; each matrix's `n_T`, `k_T`, `P_W`, `τ_W` | `g_query` | `one_shot` |
| Proximity field term, per commitment | `ϑ·n/q`, `ϑ_A = ⌈log2 advice_rows⌉`, `ϑ_T = ⌈log2(m_T + B_W)⌉`, `⌈log2 m_T⌉` in protocol/1 | — | `bits` |
| Weight linking | `max ν_T/q` | — | `bits` |
| Hash collisions | `2^{−128}` | — | `bits` |

A report lists the rows in this order, the two proximity rows for the advice first and then for each matrix in order. A row's bits are `λ = g + max(0, −log2 ε)` for the grinding `g` that covers it: an error above 1 counts as 1. A row of `ε = 0` has no term: it has no bits, and it passes.

- **Per item, the largest.** A row over several items, instances, forests or groups, takes the largest item's error, not their sum. The items are fixed before their challenge, and the verifier accepts only if every false one passes: the probability is at most the bound of any one of them. The union row is the exception, the sum RFC-0010 'Soundness of the schedule' states.
- **Batching and lines.** A stage of `n` rounds of degree `d` batching `m` instances errs with probability at most `(n·d + m)/q`, as RFC-0010 states. A tree level adds its line challenge: `(n·d + m + 1)/q`, since with fraction trees `γ` and `λ` are a polynomial of total degree `m`, not `m − 1`, and the line challenge errs with probability `1/q` of its own.
- **The union row counts every term of the rounds row**, so the rounds row never fails alone.

**Targets.** SP-1 (06 §4). Each parameter set states two targets. `bits` is 100, for every row a prover can retry by varying its messages. `one_shot` is for the rows a proof of work covers, the LogUp identity and the column queries:

- in a live session their challenges come from the coin-tossed seed and the verifier's draw, once: `live-64` gives them 64 bits, SP-1's per live audit;
- a non-interactive proof retries them at its proofs of work's cost: `ni-100` gives them 100 bits, like every other row.

`test-small` has no targets: no row passes or fails.

**Bits.** A row's bits are a FixedBits lower bound on `λ`, in units of `2^{−8}` bits ([RFC-0036](RFC-0036-protocol-messages.md#round2response)), computed by LOG2.

`LOG2(a, b, F)`, for integers `a ≥ b ≥ 1` and `W = 62`:

1. `e` is the largest integer with `b·2^e ≤ a`.
2. `y = ⌊a·2^W / (b·2^e)⌋`, so that `2^W ≤ y < 2^{W+1}`.
3. `L = e`. Then `F` times: `y = ⌊y^2 / 2^W⌋` and `L = 2L`; if then `y ≥ 2^{W+1}`, `y = ⌊y/2⌋` and `L = L + 1`.
4. The result is `L`.

Every step rounds down, so `L ≤ 2^F·log2(a/b)`. Each rounding costs `log2 y` at most `2^{−W}/ln 2`, which the later steps scale by `2^{−i}`, so `L > 2^F·log2(a/b) − 1 − 4.33·2^{F−W}`. `y^2 < 2^{126}` fits a 128-bit integer; step 2's dividend needs 190 bits.

- **A field-form row**, `ε = c/q`: `256·g` when `c ≥ q`, and `LOG2(q, c, 8) + 256·g` otherwise.
- **A queries row**: with `A = 2n` and `B = n + k + P`, and `L = LOG2(A, B, 48)` when `B < A`, the per-query bits are `⌊L/2^{40}⌋` and the row's `⌊τ·L/2^{40}⌋ + 256·g`. When `B ≥ A`, `ρ' ≥ 1`: the per-query bits are 0 and the row's `256·g`.
- **The hash row**: 32,768, 128 bits exactly.

Every FixedBits `r`, per-query bits included, then satisfies `r ≤ 256·λ < r + 1 + 2^{−20}`: at most `2^{−8}·(1 + 2^{−20})` bits below the true value, never above. With `τ ≤ 2^{16}`, the queries' rounding costs at most `2^{−24}` bits beyond FixedBits' own.

**Decisions.** A row passes exactly when `λ ≥ t` for its target `t`, decided by comparing integers:

- **field form:** `t ≤ g`, or `c ≤ ⌊q / 2^{t−g}⌋`, which is `c·2^{t−g} ≤ q`;
- **queries:** `t ≤ g`, or `A^τ ≥ 2^{t−g}·B^τ`;
- **hash:** `t ≤ 128`.

A FixedBits comparison would be sound but not exact: it fails a row up to `2^{−8}` bits above its target. Since `q < 2^{128}`, real numerators lie that close. `c = 2^{28}` gives `100 − 6.7·10^{−10}` bits and fails 100, while `c = 2^{28} − 1` gives `100 + 4.7·10^{−9}` and passes, and integers decide both by the definition, with no precision argument. LOG2 may decide a queries row first without changing the outcome, and the powers are large only near a tie:

- `τ·L ≥ 2^{48}·(t − g)` passes;
- `τ·(L + 2) ≤ 2^{48}·(t − g)` fails;
- only a row within `τ·2^{−47}` bits of its target compares the powers.

**The minimum** is the row with the least margin `r − 256·t`, the first of any that tie; under a set without targets, the row with the fewest bits; rows without a term are not counted. When every target is equal, as at `ni-100`, it is the row with the fewest bits. A report meets its targets when every row passes.

**The 405B 16K-token, 16K-context statement** is the default: Llama-3.1-405B, `servable`, one batch of 16,384 tokens at 16,384 positions of context, under `tessera-protocol/1`. Each size comes from this RFC's text, from the compiler's commitment report ([`docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json`](../perf/v0.2/commitment-report-405b-servable-c2-b32.json)) or from the architecture by its RFC's rule. Where none gives a count, the default takes a bound:

| Size | Value | Source |
|---|---|---|
| `rounds` | 10,000 of degree 7 | this table's "about 10^4 rounds" of "degree `d ≤ 7`" |
| `stages` | 2, of at most 64 instances | `servable`'s two reduction stages (RFC-0010 'Stages'); the compiler's schedule batches 18 and 4 before the gadgets' relations join it |
| `tree_levels` | 42, of at most 31 instances | the height of the tallest tree, the `Exp` lookup tree; two trees for each of the 15 instances, and the `SoftmaxRow` forest |
| `zero_checks` | 8, of 42 variables | RFC-0035's four padding zero-checks, `OneHot`'s booleanity and the first checkpoint's equality, at most the score tensor's 42 variables |
| `forests` | `s = 28` | the `SoftmaxRow` forest, `2^7` layers by `2^7` heads by `2^{14}` rows |
| `lookups` | the report's 15 instances | `N` the lookups per token times 16,384, `T` the rows; `Exp`'s `N` is `2^{42}`, every position of the padded score tensor ([RFC-0030](RFC-0030-masked-exponentials.md#cost)) |
| `claim_groups` | 16, of at most 64 claims | RFC-0011's target for `G`; the compiler's pool holds 38 claims before the gadgets' join |
| `advice_rows` | `2^{21}` | the 21-bit row index of 'Soundness budget' |
| `matrices` | the ten of RFC-0040 at `k_W = 2^{20}` | each `k_T = 2^{20}`; `m_T` from 'The matrices', `ChannelParams`' 882 from [RFC-0012](RFC-0012-weight-commitment.md#column-answers); `ν_T` the padded dimensions' bits |

`--lookups N` and `--table-rows T` replace the instances with one of `N` lookups and `T` rows, a flag left out taking `Exp`'s. The CLI's other sizes are fixed until the compiler's schedule gives them.

**At the default statement.** The calculator's lower bounds, in bits, against this RFC's figures:

| Row | `live-64` | `ni-100` | This RFC |
|---|---|---|---|
| Sumcheck and tree rounds | 121.99 | 121.99 | ≥ 125: the batching challenges `m/q` count too |
| All rounds, union bound | 111.86 | 111.86 | ≥ 112: `7·10^4/q` alone gives 111.90 |
| LogUp identity | 85.99 | 105.99 | about 90, and 110: the `Exp` instance holds `2^{42}`, not `2^{38}` |
| Advice queries | 64.32 | 80.09 + 20 | 64.3, and 80.1, which is 80.098 rounded to nearest |
| Weight queries | 65.76 | 82.72 + 20 | 64.0 and 80.5 at the masked rate; protocol/1's weights are unmasked ([RFC-0049](RFC-0049-weight-proximity.md#the-calculator)) |
| Field terms | 100.60, weights 101.91 | the same | about 100; 100.6 and 101.8 |

Every row meets its target at both sets. `live-64`'s minimum is the advice queries, 64.32 bits for 64; `ni-100`'s the advice queries, 100.09 bits for 100.

## Alternatives Considered

1. **Non-interactive only.** Rejected as the default: the source's default is the bilateral live protocol (source §10), which reaches 64-bit proximity with 155 columns instead of 193 plus grinding, and keeps proofs non-transferable unless the parties want them transferable.
2. **Verifier-chosen seed without a coin toss.** Rejected: a malicious verifier could pick challenges adversarially, and the masked-sumcheck simulation of RFC-0013 assumes uniform challenges.
3. **Johnson-bound (list-decoding) proximity analysis.** Considered: about 0.5 bits per query at rate 1/2 instead of 0.415. Rejected: provable bounds in that regime carry field-size terms that erode the margin, and conjectured bounds are not acceptable for treaty use. For the tensor generator RFC-0011 draws its coefficients with, no list-decoding bound is known at all.
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
- Calculator: both implementations agree on every row for 1,000 random size profiles, and both reproduce the reports of `spec/vectors/soundness/`; `live-64` and `ni-100` meet their targets at 405B sizes; the proximity row's field term grows by `n/|Fp2|` with each row-index bit a profile's rows add; a profile with `N + T = 2^{50}` makes `ni-100`'s LogUp row fail its target, 98 bits for 100, proving the check is live (at `live-64` it gives 78 one-shot bits, which the 64-bit one-shot target accepts); for every row, a size or parameter that makes exactly that row miss its target.
- Transports: the stream and directory transports deliver byte-identical envelopes; a TLS peer without a valid client certificate is refused.

## Open Questions

- **OQ-0014-1.** What per-message deadlines and retry policy should live sessions use in production? Owner: @AbdelStark. Resolve at v1.0 with the parties' operational requirements; until then the defaults above apply.

## References

- Source §2 (security targets), §7 ("Interaction and soundness"), §10 (verification model).
- Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity gaps for Reed–Solomon codes", FOCS 2020.
- Diamond, Posen, "Proximity Testing with Logarithmic Randomness", ePrint 2023/630, IACR Communications in Cryptology 1(1), 2024.
- Diamond, Gruen, "Proximity Gaps in Interleaved Codes", ePrint 2024/1351, IACR Communications in Cryptology 1(4), 2025: Theorem 2 (interleaving keeps line proximity gaps within unique decoding), Theorem 3 and Definition 2 (tensor gaps for `ϑ` random values at `ϑ·ε/q`).
- Canetti et al., "Fiat–Shamir: from practice to theory", STOC 2019; Ben-Sasson, Chiesa, Spooner, TCC 2016 (round-by-round and state-restoration soundness).
- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md).
