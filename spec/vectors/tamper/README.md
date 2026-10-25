# Tamper corpus

`protocol-1.json` is the SC-4 tamper corpus of `tessera-protocol/1` ([07 §4](../../../docs/spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)): mutations of a valid tiny proof, one per verifier check, each with the verdict [RFC-0019 'Check order'](../../../docs/rfcs/RFC-0019-verifier.md#check-order) gives it. [docs/testing/tamper-corpus.md](../../../docs/testing/tamper-corpus.md) says how to run and regenerate it.

## Files

| File | What | Written by |
|---|---|---|
| `claims.tcl`, `weights.twr` | The claim list and the public weight artifact of the [exact-cli fixture](../exact-cli/README.md) at `test-small/1`: the caller's trusted inputs | `tamper-base` |
| `ni.tnp` | A valid non-interactive proof | `tamper-base` |
| `live-round1.tce`, `live-round2.tce`, `live-round3.tce` | A valid live session's prover messages | `tamper-base` |
| `live-seed.bin`, `live-queries.tce` | That session's coin-toss seed and the fixed queries it answered | `tamper-base` |
| `base.json` | The request metadata, the statement digest, the lookup instances' `TableId`s and two 20-bit proof-of-work boundaries | `tamper-base` |
| `protocol-1.json` | The mutations and their verdicts | `generate.py` |

`tamper-base` is the `tessera-e2e` example of that name. Its files are synthetic producer output: the corpus's inputs, never evidence. The verifier re-derives the lookup instances from the claim list and refuses the corpus if they differ.

[`generate.py`](generate.py) is the independent oracle. It parses the base messages from RFC-0003's encoding, chooses every mutation, and predicts every verdict from the specification. It shares no code with `crates/`, `verifier/` or `python/`, never runs either workspace, and reads no verdict from a verifier. Its envelope hash and transcript are those of [`../transcript/generate.py`](../transcript/generate.py).

## A case

| Field | Meaning |
|---|---|
| `id` | Unique; `ni-` cases judge the non-interactive proof, `live-` cases the recorded live session |
| `step` | RFC-0019's step: `1`, `4`, `6`, `7.1`, `7.2`, `7.3`, `8`, `9`, `11`, `12`, or `order` for a case of two or more mutations |
| `family` | The kind of mutation |
| `reseal` | `true`: offsets count from the payload's first byte, and the consumer recomputes `payload_len` and the digest. `false`: offsets count from the envelope's first byte, and nothing is recomputed |
| `edits` | Per target file, in decreasing offset. A literal edit replaces the bytes `before` at `offset` with `after`, either possibly empty. A `shift_fp2` edit adds `delta` to `count` consecutive `Fp2` values from `offset`, coordinate-wise modulo `P` |
| `expected` | The verdict object of the core JSON report: `status` and the `reason` with its [RFC-0043](../../../docs/rfcs/RFC-0043-rejection-payloads.md) payload |

The caller's expected statement comes from `claims.tcl`, `weights.twr` and `base.json`'s request, never from a proof. A recorded live case replays `live-seed.bin` and `live-queries.tce` in the live verifier's three phases, each judging its own message under its own schema; a live verifier draws its own queries.

## Families

| Family | Step | Mutation | Verdict |
|---|---|---|---|
| `decode` | 1 | A field element `P`, a trailing byte, a missing byte, a changed envelope digest | `REJ-MALFORMED` with the codec's code |
| `leaf-groups` | 1 | `Round1.leaf_groups` past the token axis's `t` bits, past the layer index's `n_ℓ` bits, or 15 bits in all; within them, one token bit, one layer bit, or `(t, n_ℓ)` on the ungrouped bases; an ungrouped `TreeProof` with a group, a table root, a merge round or a merge evaluation (RFC-0054 'Messages') | `REJ-MALFORMED`: `ERR-CODEC-LIMIT` past a bound, on `Round1` in a live session; `ERR-CODEC-NONCANONICAL` for the counts a grouping within its bounds fixes, judged after the statement, and for a grouped field, on `Round2Response` in a live session |
| `statement` | 4 | Each statement field, each request field, the predictor, the layout digest | `REJ-STATEMENT` with the field |
| `fiat-shamir` | 6 | The seed nonce, which test-small's zero-bit predicate accepts | `REJ-TREE` in group 0 at level 1, round 1: the nonce is absorbed |
| `logup-root` | 7.1 | Each instance's zero `d_a`, zero `d_t`, and a changed `n_a` | `REJ-LOOKUP` with its table |
| `tree-coefficient`, `tree-evaluation`, `tree-round-length` | 7.2 | Per level, a round value, an evaluation, the last round one value short, or a round at level 0, which has none | `REJ-TREE` in group 0, the ungrouped stage's, with the level and the round of RFC-0010 'Sumcheck messages' |
| `leaf-split` | 7.3 | Each `logup/values` entry; the first and last `stage/0/split` values | `REJ-SUMCHECK` stage 0, round 0 (RFC-0045 'The check') |
| `stage-coefficient`, `stage-evaluation`, `stage-round-length` | 8 | Per stage, as for a level | `REJ-SUMCHECK` with the stage and round |
| `combined-row` | 9 | Every advice and weight group's row, every entry shifted by one | `REJ-PCS-EVAL` with the commitment |
| `merkle-path` | 11 | A sibling and a column symbol, advice and each opened matrix | `REJ-PCS-MERKLE`, column 0 |
| `proximity` | 11 | A proximity row's first entry, advice and each opened matrix | `REJ-PCS-PROXIMITY`, column 0 |
| `scoring` | 12 | `Û = 1` in exact mode | `REJ-UHAT` |
| `check-order` | — | Two mutations of adjacent checks, at every step boundary | The earlier check's |

`generate.py`'s comments derive each prediction. Two rest on the protocol's algebra as well as on the check order:
- A round value that changes is caught by its level's or stage's final check, round `n_max`, since the verifier derives `p_j(1)` from the claim.
- An opening group's column sub-point can hold fixed bits of a tensor's offset, where `eq` gives an entry weight 0. Shifting every entry by `δ` moves `⟨u, eq⟩` by exactly `δ` at any point, so the group's own check fails.

Steps 11 and 12 are judged only in the recorded live session: there, Round2's last values do not move the fixed queries, while in a non-interactive proof they would.

`base.json`'s `pow` holds a valid and an invalid nonce, under `ni-100/1`'s 20 bits, for both phases on a fixed context. Test-small's predicate has no invalid nonce. Each consumer checks both nonces with its own transcript.

## Not covered by `tessera-protocol/1`

- Anchors, the manifest signature, the ledger and the weight reservation (steps 4 and 5): v0.4. The manifest digest stands for the anchor here.
- The LogUp identity sumcheck of `tessera-protocol/2` (`REJ-LOGUP-IDENTITY`).
- The masks' batching of `tessera-protocol/2` (`REJ-MASK-BATCH`, RFC-0070).
- The LogUp leaf checks of step 7.3 and gadget final checks (`REJ-GADGET`): a single change fails an earlier check first. The alternative-witness tests reach them with a consistent cheating prover.

```sh
python3 spec/vectors/tamper/generate.py           # rewrite protocol-1.json from the bases
python3 spec/vectors/tamper/generate.py --check   # compare; cargo xtask gen-vectors --check runs it too
```
