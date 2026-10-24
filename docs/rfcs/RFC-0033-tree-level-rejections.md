# RFC-0033: Rejection Reasons for the Leaf Stage's Sumchecks

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [04 §2](../spec/04-error-model.md#2-verdicts): `RejectReason` gains `TreeLevel` and `LogUpIdentity`;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order', step 7: the reasons of a failed tree level and of `tessera-protocol/2`'s identity sumcheck;
  - [RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages) 'Sumcheck messages': the reason of a failed tree level.
- Amended by: [RFC-0054](RFC-0054-token-groups.md#messages) ('Messages'): `TreeLevel` gains `group`, and `RejectReason` gains `LeafMerge`.
- Origin: [#225](https://github.com/AbdelStark/tessera/issues/225)

## Summary

The leaf stage's sumchecks have no rejection reason of their own.

- **Tree levels.** RFC-0019 step 7 reports a failed tree level as `Reject(Sumcheck)`, whose fields are `{ stage: StageId, round: u32 }`. A tree level has no `StageId`: its labels are `tree/{ℓ}/…`, and `Round2Response` carries the trees outside `stages`.
- **The identity sumcheck.** `tessera-protocol/2`'s LogUp identity sumcheck ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#the-logup-identity-sumcheck)) is batched over every LogUp instance, so its failure names neither a stage nor a table.

This RFC adds two variants, so that one invalid proof yields one reason on every verifier:

```rust
TreeLevel { level: u32, round: u32 },                            // REJ-TREE
LogUpIdentity { round: u32 },                                    // REJ-LOGUP-IDENTITY
```

## Motivation

Check order is fixed so that a given invalid proof always yields the same reason (04 §2). For a tree level, the reason did not exist. Two verifiers could report the same proof differently, and the tamper corpus (07 §4) could not pin a tree level's failure. `tv-core` reports a level's failures with a round counted within the level, and leaves the mapping to the driver (#128). The level `ℓ` is what locates the failure, and `Sumcheck` has no field for it.

## Proposed Design

- **`TreeLevel { level, round }`**, code `REJ-TREE`, exit 10.
  - A tree level fails: its round messages, its final evaluations, or its final check, in RFC-0010 'Sumcheck messages''s order.
  - `level` is the `ℓ` its labels carry, `tree/{ℓ}/…`.
  - `round` counts from 0 as the labels do, with `n_max` for the evaluations' count and the final check.
  - It covers the levels of fraction trees and of product forests alike, which RFC-0010 batches into one sumcheck per level.
- **`LogUpIdentity { round }`**, code `REJ-LOGUP-IDENTITY`, exit 10, in `tessera-protocol/2` only.
  - The identity sumcheck fails, with `round` counted as for a tree level.
  - The sumcheck combines every instance, so a failure names no table.
- **Unchanged.** `Sumcheck { stage, round }` stays the reason of a reduction stage, numbered as its `stage/{k}/…` labels are. `Lookup { table }` stays the reason of `tessera-protocol/1`'s LogUp roots and of the LogUp leaf checks, which name an instance.

The variants follow `Sumcheck` in 04 §2's declaration. No artifact encodes `RejectReason` yet, so its order changes in place (09 §3).

## Alternatives Considered

1. **Reserve a `StageId` for the leaf stage**, with `round` counted across its levels: the rounds of levels `0` to `ℓ − 1`, plus the round within level `ℓ`. Rejected: the types would not change, but the reason would no longer name the level. Recovering it would need every level's round count, which depends on the forests' heights.
2. **`Lookup` for the identity sumcheck.** Rejected: `Lookup` names a table, and the identity sumcheck's failure has none.

## Drawbacks

- Two more variants in a tier-A enum.

## Migration / Rollout

v0.3, before the verifier's driver (#128) assigns reasons, and before the tamper corpus (#135) pins them. `LogUpIdentity` is used from `tessera-protocol/2` (v0.4).

## Testing Strategy

- **The tamper corpus** (07 §4, #135) pins each failure:
  - one round coefficient and one evaluation per tree level, each `Reject(TreeLevel { level, round })` with its level and round;
  - in `tessera-protocol/2`, one round coefficient of the identity sumcheck, `Reject(LogUpIdentity { round })`.
- **The driver** (#128) maps `tv-core`'s level errors to `TreeLevel` with the level it verified, and a stage's to `Sumcheck`.

## Open Questions

None.

## References

- [04 §2](../spec/04-error-model.md#2-verdicts), [07 §4](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md).
- [#225](https://github.com/AbdelStark/tessera/issues/225), [#59](https://github.com/AbdelStark/tessera/issues/59), [#128](https://github.com/AbdelStark/tessera/issues/128), [#135](https://github.com/AbdelStark/tessera/issues/135).
