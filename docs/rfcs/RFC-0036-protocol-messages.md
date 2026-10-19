# RFC-0036: Protocol Messages and Session State Machines

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#messages) 'Messages': every message's layout, for `tessera-protocol/1`;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#session-order-live-mode) 'Session order': who chooses `session_id`, and what each seed message holds;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule': the bytes rows 2, 3 and 7 to 10 absorb, and the spelling of `{T}`;
  - [02 §3.1 and §3.2](../spec/02-public-api.md#3-rust-interfaces-tier-b): the order of the live session's calls;
  - [04 §3](../spec/04-error-model.md#3-error-codes): `ERR-PROTO-QUERIES`;
  - [01 §5](../spec/01-architecture.md#5-prover-phases) step 2: what `Round1` holds.
- Origin: [#98](https://github.com/AbdelStark/tessera/issues/98)

## Summary

RFC-0014 names the protocol's eight messages, and several of the types they hold have never been defined: `WeightReservation`, `LogUpProof`, `TreeProof`, `WeightCombinedRows`, `ScoringClaims`, and the three seed messages. No document says which fields belong to `tessera-protocol/1`, or what the session's states are. Without them neither workspace can encode a message, and the verifier cannot be written from the specification. This RFC defines:

- **The layouts.** Every message of `tessera-protocol/1`, field by field, in TCE ([RFC-0003](RFC-0003-canonical-encoding.md)), as schema version 1.
- **The absorbed bytes.** What each label of the transcript schedule absorbs.
- **The state machines.** The prover's and the verifier's live sessions, with the error each refusal returns, including `ERR-PROTO-QUERIES` for column queries that `INV-PROTO-3` refuses.

`tessera-protocol/2` adds RFC-0013's and RFC-0032's fields in schema version 2 of the same schemas, defined with them.

## Motivation

#98 implements the messages and the session state machines. Surveying the documents for it found:

- **Undefined types.** `WeightReservation`, `LogUpProof`, `TreeProof`, `WeightCombinedRows` and `ScoringClaims` appear only in RFC-0014's struct. RFC-0010 OQ-0010-1 says the types that carry the tree levels "are still to be written".
- **Two protocols in one struct.** RFC-0014's `Round2Response` holds blinding claims and masked sums, which RFC-0013 adds in v0.4 behind the protocol version bump. RFC-0003 forbids optional fields, so one schema version cannot serve both.
- **Two session orders.** RFC-0014 sends `SeedCommit` first. 02 §3.2's `LiveVerifier` sends it in answer to `Round1`.
- **Unspecified transcript bytes.** The labels `round1`, `seed`, `pcs/advice/combined`, `pcs/weights/{T}/combined`, `scoring` and `queries` have no stated value type, against INV-FS-3.
- **No state machine** for either side, and no error for queries that `INV-PROTO-3` refuses.
- **No three-element tuples in TCE.** RFC-0014 writes `(u8, Vec<Vec<Fp>>, MultiProof)`.
- **A misplaced commitment.** 01 §5 puts an anchor key commitment in Round 1. RFC-0015 logs it at session open and checks it against that public value.

## Proposed Design

### Versions and envelopes

- **Schema version 1** of `SeedCommit`, `ProverSeedShare`, `SeedReveal` (`0x0501`–`0x0503`), `Round1`, `Round2Response`, `ColumnQueries`, `Round3Response` (`0x0510`–`0x0513`) and `NiProof` (`0x0520`) is `tessera-protocol/1`: sound and not hiding ([RFC-0013](RFC-0013-zero-knowledge.md#migration--rollout) 'Migration / Rollout').
- **Schema version 2** is `tessera-protocol/2`. It adds the weight reservation, masked sums, blinding claims and the LogUp identity sumcheck of RFC-0013 and RFC-0032. Those RFCs' issues define its layouts (#148, #149, #154). A tier-A schema change is a protocol change ([03 §5](../spec/03-data-model.md#5-artifact-catalog)), so the two never mix.
- **Envelopes.** Each message travels in an [RFC-0003 envelope](RFC-0003-canonical-encoding.md#envelope), schema id and version in its header.
  - `NiProof` holds `Round1`, `Round2Response` and `Round3Response` as payloads, without inner envelopes, at its own version.
  - A transport or a test that writes `ProverSeedShare` beside `Round1` writes two envelopes.
- **Field elements.** `Fp` and `Fp2` encode as RFC-0003 says; `Digest` is 32 raw bytes.

### Seed messages

```rust
pub struct SeedCommit { pub session_id: [u8; 16], pub commitment: Digest }  // c_V = H("coin-commit", s_V)
pub struct ProverSeedShare { pub session_id: [u8; 16], pub share: [u8; 32] } // s_P
pub struct SeedReveal { pub session_id: [u8; 16], pub share: [u8; 32] }      // s_V
```

- **Choices.** In live mode the verifier chooses `session_id` and `s_V`, each fresh from its CSPRNG for every session (INV-PROTO-4). The prover chooses `s_P` fresh for every `Round1`. It derives `s_P` from its per-proof secret seed through a PRG under a label of its own ([RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency) 'Crash consistency'), so that a prover that resumes after a crash sends the `Round1` and the `s_P` it sent, byte for byte, and never a second share with one `Round1`. The per-proof seed is fresh for every proof and an abort discards it (INV-PROTO-1), so a retry has a fresh `s_P`.
- **The seed.** `seed = H("coin-toss", s_V ‖ s_P)`. RFC-0004's `coin-commit` and `coin-toss` domains take the shares as their 32 bytes.
- **Session binding.** Each seed message names its session, so neither side can mistake one session's share for another's.

### `Round1`

```rust
pub struct Round1 {
    pub protocol_version: u16,          // 0 until tessera-protocol/1 is assigned (09 §3), then 1
    pub session_id: [u8; 16],
    pub statement: PublicStatement,     // 03 §4
    pub advice: AdviceCommitment,       // RFC-0011: { root, layout_digest }
}
```

- **`session_id`.** In live mode it is `SeedCommit`'s. In non-interactive mode the prover chooses it fresh, and `SessionContext` binds it either way.
- **`protocol_version`.** It is the protocol's number, `tessera-protocol/{protocol_version}`, the one the claim list states ([RFC-0007](RFC-0007-claim-list.md#schema)). It is 0 until `tessera-protocol/1` is assigned, when the first end-to-end proof verifies ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)), and 1 from then on. A reader of another version answers `ERR-PROTO-VERSION`.
- **No reservation.** `tessera-protocol/1` burns no blinding rows ([RFC-0012](RFC-0012-weight-commitment.md#migration--rollout) 'Migration / Rollout'). Its epoch is `statement.weights_epoch`, so `Round1` holds no `WeightReservation`; version 2 adds one.
- **`PublicStatement`** is 03 §4's. The verifier compares it with the statement its own inputs give: the audit sample's batch, the manifest, the claim list, and the ledger's root and epoch ([RFC-0019](RFC-0019-verifier.md#check-order) step 4). A difference is `Reject(StatementMismatch)`, naming the first field that differs in declaration order: `claim_list_digest`, `weights_root`, `weights_epoch`, `manifest_digest`, `batch`, `predictor`.

### `Round2Response`

```rust
pub struct Round2Response {
    pub logup: LogUpProof,
    pub trees: TreeProof,
    pub stages: Vec<StageProof>,
    pub advice_combined: Vec<CombinedRow>,
    pub weight_combined: Vec<WeightCombinedRows>,
    pub scoring: ScoringClaims,
}
pub struct LogUpProof { pub roots: Vec<Fp2>, pub values: Vec<Fp2> }
pub struct TreeProof { pub levels: Vec<SumcheckProof>, pub splits: Vec<Fp2> }
pub struct StageProof { pub stage: StageId, pub sumcheck: SumcheckProof, pub splits: Vec<Fp2> } // StageId(u16)
pub struct SumcheckProof { pub rounds: Vec<Vec<Fp2>>, pub final_evals: Vec<Fp2> } // RFC-0010
pub struct CombinedRow { pub u: Vec<Fp2>, pub mask: Vec<Fp2> }
pub struct WeightCombinedRows { pub matrix: u8, pub evaluation: CombinedRow, pub proximity: CombinedRow }
pub struct ScoringClaims { pub requests: Vec<FixedBits> }
pub struct FixedBits(pub u64);                                                  // units of 2^-8 bits
```

- **`logup`.**
  - `roots` holds `n_a, d_a, n_t, d_t` of every LogUp instance in schedule order, 4 per instance, as RFC-0008 'Fraction tree' sends them under `logup/roots`.
  - `values` holds `valũe(ρ_a)` of every instance whose table has values, in schedule order, under `logup/values`.
  - Both are absorbed even when empty.
- **`trees`.**
  - `levels[ℓ]` is level `ℓ` of the leaf stage, from the roots: the one sumcheck of RFC-0010 'Tree levels' that its fraction trees and product forests share.
  - RFC-0014 named the field `product_trees`; its levels hold the fraction trees too.
  - `splits` holds the values of the leaf stage's split claims, stage 0's ([RFC-0037](RFC-0037-stage-schedule.md#splits)).
  - The verifier reads the leaf stage in schedule order: `logup.roots`, then `trees.levels`, then `logup.values`. The order of the fields is not that of the transcript.
- **`stages`.**
  - `stages[i]` is reduction stage `i + 1`, the leaf stage being stage 0.
  - Its `sumcheck` holds the messages of RFC-0010 'Sumcheck messages', and `splits` the values of the claims it splits ([RFC-0037](RFC-0037-stage-schedule.md#splits)).
- **`advice_combined`.**
  - It holds RFC-0011's `G + 1` combined rows, the groups in group order, then the proximity row ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 5).
  - Each row's `u` holds `k` values. Its `mask` holds the `P_A` coefficients of the combined mask polynomial, constant first.
- **`weight_combined`.**
  - It holds one entry for every weight matrix the proof opens, `matrix` strictly increasing. An entry is the matrix's evaluation combined row and its proximity combined row ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass) 'Opening by the folding pass').
  - In `tessera-protocol/1`, a matrix with no committed rows is public zero and is not opened: it has no entry in `weight_combined`, `ColumnQueries.weights` or `Round3Response.weight_columns` ([RFC-0049](RFC-0049-weight-proximity.md#the-schedule)). Its root remains in `WeightCommitmentPublic` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)). Claims on that public-zero matrix are checked as zero, not discarded ([RFC-0037](RFC-0037-stage-schedule.md#claims-and-atoms)). This omission does not apply to a masked matrix with reserved blinding rows.
  - Each `u` holds `k_T` values. Each `mask` holds the combined mask's coefficients, none in `tessera-protocol/1`, whose weight encoding has no mask (RFC-0012 'Migration / Rollout').
- **`scoring`.**
  - It holds one claimed `Û_r` per request, in `statement.batch` order ([RFC-0017](RFC-0017-scoring.md#in-circuit-scoring-mixture) step 5), in units of `2^{−8}` bits. It is 0 in exact mode.
  - `FixedBits` is also the type of `Reject(UBoundExceeded)`'s values ([04 §2](../spec/04-error-model.md#2-verdicts)).

**Shapes.** The schedule and the statement fix every length the decoder cannot check alone:
- the stages, one per reduction stage, `stages[i].stage = i + 1`;
- the matrices of `weight_combined`, those that have claims and are opened under the rule above;
- the scoring claims, one per request;
- each stage's split values, one per value its split claims send ([RFC-0037](RFC-0037-stage-schedule.md#splits)).

The verifier checks them when it decodes `Round2Response`, after RFC-0019 step 6, and a mismatch is `Reject(Malformed)`. Shapes inside a part are that part's checks: a sumcheck's rounds and evaluations are RFC-0010's (`Reject(Sumcheck)`, `Reject(TreeLevel)`), and the combined rows' are RFC-0011 step 6's (`Reject(Evaluation)`), for the weight matrices' rows as for the advice's.

**Matrix ids.** A `u8` matrix id is the declaration index of [RFC-0031](RFC-0031-commitment-id.md#the-descriptor)'s `WeightMatrix`, RFC-0012's table order: `Embed` 0, `Q` 1, `K` 2, `V` 3, `O` 4, `Gate` 5, `Up` 6, `Down` 7, `LmHead` 8, `ChannelParams` 9. A label's `{T}` is the variant's name as written there, so `pcs/weights/Q/gamma` and `pcs/weights/ChannelParams/xi`. Three sequences are keyed by matrix id: `weight_combined`, `ColumnQueries.weights` and `Round3Response.weight_columns`. In each, the ids are 0 to 9 and strictly increase. A decoder reads each entry's id, its first byte, before the rest of the entry, as RFC-0003 reads a map's key, and rejects an id above 9, or one not above the previous entry's, with `ERR-CODEC-NONCANONICAL` at that byte.

### `ColumnQueries`, `Round3Response` and `NiProof`

```rust
pub struct ColumnQueries { pub advice: Vec<u32>, pub weights: Vec<MatrixQueries> }
pub struct MatrixQueries { pub matrix: u8, pub indices: Vec<u32> }
pub struct Round3Response {
    pub advice_columns: Vec<Vec<Fp>>,
    pub advice_proof: MultiProof,
    pub weight_columns: Vec<MatrixColumns>,
}
pub struct MatrixColumns { pub matrix: u8, pub columns: Vec<Vec<Fp>>, pub proof: MultiProof }
pub struct MultiProof { pub siblings: Vec<Digest> }                          // RFC-0004, canonical order
pub struct NiProof {
    pub round1: Round1,
    pub seed_nonce: u64,
    pub round2: Round2Response,
    pub query_nonce: u64,
    pub round3: Round3Response,
}
```

- **`ColumnQueries`.**
  - `advice` holds the advice commitment's `τ_A` indices in draw order.
  - `weights` holds one entry per opened matrix, in `weight_combined`'s order, each with `τ_W` indices in draw order.
  - Every index is below its codeword's length: `n_A` for the advice and `n_T = 4·k_T` for matrix `T`.
  - Indices may repeat ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7).
- **`Round3Response`.** It answers each distinct queried index once, in increasing order, as RFC-0011 step 7 and RFC-0012 say.
  - `advice_columns[c]` holds every non-zero row's symbol in that column, in segment order.
  - An entry of `weight_columns` does the same for one matrix: its non-zero data rows, and in `tessera-protocol/2` its blinding rows.
  - Each `MultiProof` covers its commitment's distinct indices.
  - `MatrixQueries` and `MatrixColumns` are structs because TCE has no three-element tuple. `MatrixQueries` encodes as the pair `(u8, Vec<u32>)` would.
- **`NiProof`.**
  - `seed_nonce` and `query_nonce` are the proof-of-work nonces of [RFC-0014](RFC-0014-interaction-and-soundness.md#grinding) 'Grinding'.
  - The column indices come from the transcript, so `NiProof` holds no `ColumnQueries`. Its parts are checked as in live mode.

### What the transcript absorbs

Every value is absorbed as its TCE payload, under the label of [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' (INV-FS-3):

| Row | Label | Absorbed |
|---|---|---|
| 2 | `round1` | `TCE(Round1)` |
| 3 | `seed` | Live: the 32 bytes of `seed`. Non-interactive: the grind of `seed-pow` absorbs `seed_nonce` (RFC-0004 `grind`), then the seed is squeezed under `seed`. |
| 3 | `pcs/weights/{T}/xi` | In `tessera-protocol/1`, after `seed`, squeeze each nonempty weight matrix's proximity point in matrix-id order; nothing is absorbed. Empty unmasked matrices make no call ([RFC-0049](RFC-0049-weight-proximity.md#the-schedule)). |
| 4, 5 | `logup/roots`, `tree/{ℓ}/…`, `logup/values` | As RFC-0008 and RFC-0010 state: `TCE(Vec<Fp2>)` of `logup.roots`, each level's messages, then `TCE(Vec<Fp2>)` of `logup.values` |
| 6 | `stage/{k}/…` | Each stage's messages, as RFC-0010 states. `tessera-protocol/1` has no masked sum, so `zk-sum` and `zk-rho` are not used. |
| 4 to 6 | `stage/{k}/split` | `TCE(Vec<Fp2>)` of the stage's `splits`: stage 0's, `trees.splits`, after `logup/values`, and stage `k`'s after `stage/{k}/evals` ([RFC-0037](RFC-0037-stage-schedule.md#splits)) |
| 7 | `pcs/advice/gamma`, `pcs/advice/xi`, then `pcs/weights/{T}/gamma` per opened matrix in id order | Nothing is absorbed: `tessera-protocol/1` has no blinding claim, so these challenges are squeezed directly. Its weight proximity points were squeezed at row 3 (RFC-0049); this table does not move `tessera-protocol/2` challenges. |
| 8 | `pcs/advice/combined` | `TCE(Vec<CombinedRow>)` of `advice_combined` |
| 8 | `pcs/weights/{T}/combined` | `TCE(WeightCombinedRows)` of the matrix's entry, matrix by matrix in id order |
| 9 | `scoring` | `TCE(ScoringClaims)` |
| 10 | `queries` | Live: `TCE(ColumnQueries)`. Non-interactive: the grind of `query-pow` absorbs `query_nonce`, then the indices are squeezed under `pcs/advice/queries` and `pcs/weights/{T}/queries` (RFC-0004 `challenge_indices`). |

`ProverSeedShare`, `SeedCommit`, `SeedReveal` and `Round3Response` are not absorbed. The seed covers the shares, and nothing is squeezed after the columns.

### The prover's live session

States: `AwaitCommit`, `AwaitReveal`, `AwaitQueries`, `Done`. A refused message leaves the state as it was. Abort is possible in any state.

| State | Message | Checks | Action | Next |
|---|---|---|---|---|
| `AwaitCommit` | `SeedCommit` | none | Runs the witness pass and commits. Sends `Round1` with the commit's `session_id`, and `ProverSeedShare` with a fresh `s_P`. | `AwaitReveal` |
| `AwaitReveal` | `SeedReveal` | `session_id` is the session's (`ERR-PROTO-ORDER`); `H("coin-commit", s_V) = c_V` (`ERR-PROTO-COINTOSS`, then abort) | Derives the seed and proves the tree levels, stages and openings. Sends `Round2Response`. | `AwaitQueries` |
| `AwaitQueries` | `SeedReveal` | — | Refused with `ERR-PROTO-REUSE`: it would answer a second seed for one `Round1` (INV-PROTO-1). | unchanged |
| `AwaitQueries` | `ColumnQueries` | The matrices are `weight_combined`'s, the counts are `τ_A` and `τ_W`, and every index is in range (`ERR-PROTO-QUERIES`) | Answers the distinct indices. Sends `Round3Response`. | `Done` |
| any | any other | — | Refused with `ERR-PROTO-ORDER` | unchanged |

**Abort.** After `Round1`, an abort discards that `Round1`, its advice commitment and its randomness. A retry is a new session, with a fresh `s_P` and fresh randomness (INV-PROTO-1). A `SeedReveal` that fails its commitment is an abort, so a verifier cannot try a second `s_V` against one `Round1`.

### The verifier's live session

States: `Start`, `AwaitRound1`, `AwaitRound2`, `AwaitRound3`, `Done`. The verifier ends a session at the first failed check, with the verdict that check gives.

| State | Message | Action | Next |
|---|---|---|---|
| `Start` | — | Draws `session_id` and `s_V`, and sends `SeedCommit` | `AwaitRound1` |
| `AwaitRound1` | `Round1`, `ProverSeedShare` | Both name the session. Runs [RFC-0019](RFC-0019-verifier.md#check-order) steps 1 to 6: decoding, parameters, claim list, statement, transcript and seed. Sends `SeedReveal`. | `AwaitRound2` |
| `AwaitRound2` | `Round2Response` | Runs steps 7 to 9. Step 10 draws the column queries from its CSPRNG and absorbs them under `queries`. Sends `ColumnQueries`. | `AwaitRound3` |
| `AwaitRound3` | `Round3Response` | Runs steps 11 to 13 and returns the verdict | `Done` |

- **Out of order.** A message the state does not expect, or one that names another session, ends the session with `Inconclusive(ERR-PROTO-ORDER)`.
- **Version.** A `protocol_version` other than its own ends it with `Inconclusive(ERR-PROTO-VERSION)`.
- **Order of the checks on `Round1`.** The verifier checks, in this order: that the state is `AwaitRound1`, that `Round1` names the session, that `ProverSeedShare` names it, that `protocol_version` is its own, and then RFC-0019 steps 1 to 6. The first check that fails ends the session with its code, so that every verifier reports the same one.
- **Only acceptance certifies** ([04 §1](../spec/04-error-model.md#1-principles)), so a prover gains nothing by forcing either verdict.

### Errors

[04 §3](../spec/04-error-model.md#3-error-codes) gains one code:

| Code | Class | Meaning | Raised by | Exit |
|---|---|---|---|---|
| `ERR-PROTO-QUERIES` | Protocol | Column queries for another set of matrices, of another count than the parameter set's `τ_A` or `τ_W`, or with an index past the codeword | Prover | 20 |

Queries that arrive before `Round2Response` is sent are `ERR-PROTO-ORDER`, as any early message is.

### Changes to other documents

- **RFC-0014.** 'Messages' points here for the layouts. 'Session order' says the verifier chooses `session_id`. 'Transcript schedule' points here for the absorbed values and `{T}`.
- **02 §3.1 and §3.2.** The calls follow RFC-0014's order:
  - the prover's `start(batch, &SeedCommit)` returns `Round1` and `ProverSeedShare`, and `on_seed(&SeedReveal)` returns `Round2Response`;
  - the verifier's `start()` returns `SeedCommit`, and `on_round1(&Round1, &ProverSeedShare)` returns `SeedReveal`.
- **03 §4** points here for the protocol messages.
- **04 §3** lists `ERR-PROTO-QUERIES`, and so do `tessera-codec`'s `ErrorCode` and `spec/vectors/errors/codes.json`.
- **01 §5.** In `tessera-protocol/1`, Round 1 holds the statement and the advice commitment. `tessera-protocol/2` adds the weight reservation and the masking commitments. The anchor's key commitment is the session-open entry's ([RFC-0015](RFC-0015-transcript-anchor.md#frames)).
- **RFC-0010 OQ-0010-1.** The types it awaited are `TreeProof` and `LogUpProof` here.

## Alternatives Considered

1. **One schema version for both protocols, with the zero-knowledge fields empty in `tessera-protocol/1`.** This was rejected. RFC-0003 forbids optional fields, and `tessera-protocol/2` changes the LogUp proof's structure, not only its lengths. Empty vectors would also leave every `tessera-protocol/1` reader one more rule to enforce.
2. **02's session order**, with the verifier committing after `Round1`. It is also sound: `Round1` is fixed before the commitment, and the commitment comes before the prover's share. This was rejected because it costs one more round trip before the seed, and RFC-0014's order is the protocol's.
3. **Separate `LogUpProof` and `TreeProof` levels.** The fraction trees and the product forests share every level's sumcheck (RFC-0010 'Tree levels'), so each level is one message, and `TreeProof` holds it.
4. **A new reason in place of `ERR-PROTO-QUERIES`.** Malformed queries are the verifier's fault, and the prover refuses them. No reject reason applies, since the prover issues no verdict.

## Drawbacks

- `NiProof` holds three messages without inner envelopes. A tool that extracts `Round2Response` from it must re-envelope it at `NiProof`'s version.
- Schema version 2 is not defined yet. A `tessera-protocol/2` reader waits on #148, #149 and #154.

## Migration / Rollout

v0.3 (#98): the TCE types in `tessera-protocol`, the prover's and the verifier's state machines, `ERR-PROTO-QUERIES`, and golden vectors for every message, written by an independent oracle. The verifier's decoders follow in `tv-core` (#128). v0.4 defines schema version 2.

## Testing Strategy

- **Golden vectors.** For every message of schema version 1, the encoding of a small instance with every field filled, its envelope, and its decoding back. Both workspaces check them.
- **Round trips.** `decode(encode(m)) = m` for random messages, and every non-canonical variant is rejected: an unsorted `weight_combined`, a non-canonical field element, trailing bytes.
- **State machines.** Each refusal of the tables above, with its exact code:
  - out-of-order messages give `ERR-PROTO-ORDER`;
  - a second `SeedReveal` gives `ERR-PROTO-REUSE`;
  - a reveal that fails its commitment gives `ERR-PROTO-COINTOSS` and an abort;
  - queries of the wrong count, matrices or range give `ERR-PROTO-QUERIES`. The two paths differ: matrix ids reordered, repeated or unknown on the wire are a decoding error, `ERR-CODEC-NONCANONICAL` ('Matrix ids'), and a `ColumnQueries` that decodes but names other matrices than `weight_combined`'s is `ERR-PROTO-QUERIES`;
  - another `protocol_version` gives `ERR-PROTO-VERSION`.
- **Transcript.** The bytes each label absorbs equal the table above, checked against the transcript vectors of #125.

## Open Questions

- **OQ-0036-1.** RFC-0017 allows up to 4 temperatures per batch, but `PredictorSpec` holds one `Decoding`, and RFC-0007 speaks of a temperature per request. This RFC does not change `PublicStatement`. The question belongs to the v0.4 sampling work (#144), which must settle it before temperature decoding ships.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0017](RFC-0017-scoring.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0031](RFC-0031-commitment-id.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md).
