# RFC-0039: Statement Fields

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [04 §2](../spec/04-error-model.md#2-verdicts) 'Verdicts': `StatementField`, the payload of `Reject(StatementMismatch)`, is defined here;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation': every step names the field it rejects with. Step 1 defers to RFC-0019's order. Step 2 states every architecture and quantization rule. Step 5 checks the gadget policy first and the table set last.
- Amended by: [RFC-0065](RFC-0065-masked-channel-params.md) ('The checks', step 2.4: in `tessera-protocol/2`, affine `ChannelParams` sites only, until #486), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('The checks', step 2.4: lifted)
- Origin: [#346](https://github.com/AbdelStark/tessera/issues/346)

## Summary

`Reject(StatementMismatch { field })` means that the verifier cannot certify the statement a proof is for. Either the claim list breaks a validation rule, or the proof's `PublicStatement` differs from the statement the verifier's own inputs give. 04 §2 types the payload as `StatementField`, which no document defines. This RFC defines it: one variant per check, in check order, holding the public identifier that locates the failure. It also completes RFC-0007's validation steps, some of whose rules exist only in the compiler, so that two verifiers name the same field for the same input.

## Motivation

- **The verdict is tier A** ([02 §1](../spec/02-public-api.md#1-stability-tiers)). `tv-core` will return it (#107, #128), and `tv-tables` already maps its errors to `Reject(StatementMismatch)` (#100). While the payload is undefined, each implementation chooses its own.
- **One reason per input.** [RFC-0019](RFC-0019-verifier.md#goals) promises that "the same inputs always yield the same verdict and reason". With a payload, the reason depends on which check fails first. Every check that can fail must therefore be written down, and so must its place in the order.
- **Rules only the compiler knows.** RFC-0007 step 2 states four architecture rules, and `tessera-compiler` checks three more. Neither RFC-0007 nor 03 §2 gathers the rules on the public quantization data: they are split between 03 §2, RFC-0025 'Lowering' and the compiler's `check_public`. The independent verifier is written from the specification ([RFC-0001](RFC-0001-workspace-and-verifier-independence.md)), so the rules must be written there.
- **The gadget policy.** RFC-0007's testing strategy rejects a `GadgetPolicy` with `max_log2_rows = 40` "at the expected validation step", but no step checks the gadget policy.

## Goals

- A payload for every check that returns `Reject(StatementMismatch)`, holding public data only ([05 §3](../spec/05-observability.md#3-data-classification-and-redaction)).
- The same field from every correct verifier for the same claim list and statement.
- Every rule of claim-list validation stated in RFC-0007.

## Non-Goals

- The payloads of `Malformed`, `ProofOfWork`, `Anchor` and `Params` (#353).
- New rules. Every rule below is one that a document states or that `tessera-compiler` applies. `rms_eps_int` and `rsqrt_kappa` are constants `D` states. RFC-0005 and RFC-0023 give the quantizer's rule for them, which validation does not enforce: another value gives a valid `D` of lower precision ([RFC-0023](RFC-0023-norm-precision.md#proposed-design)).

## Proposed Design

### The type

```rust
pub enum StatementField {
    // The claim list: RFC-0007 'Verifier validation', RFC-0019 step 3
    ProtocolVersion,                 // step 1
    Architecture,                    // step 2
    Quantization,                    // step 2
    Rescaling { site: Site },        // step 2 (RFC-0038)
    Profile,                         // step 3
    Table { table: TableId },        // steps 4 and 5
    GadgetPolicy,                    // step 5
    Gadget { gadget: GadgetId },     // step 5
    Layout,                          // step 6
    Template,                        // step 7
    // The statement: RFC-0019 step 4, PublicStatement's fields in declaration order (RFC-0036)
    ClaimListDigest,
    WeightsRoot,
    WeightsEpoch,
    ManifestDigest,
    Batch,
    Predictor,
    // The advice commitment: RFC-0019 step 4 (RFC-0031)
    LayoutDigest,
}
```

Its TCE encoding is RFC-0003's: a `u16` tag, the variant's declaration index, then the payload's fields. `Site` is encoded as 03 §2 declares it, and `TableId` and `GadgetId` as `u32`. Every payload is public: a site type, a table's index or a gadget's index, never a value of the model.

### The checks

[RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation' states each check's rules, and this RFC gives each check its field. The verifier runs the checks in the order below, and the first failure is the verdict:

| Step | Check | Field |
|---|---|---|
| RFC-0019 step 2 | `params` names an allowed parameter set | none: `Reject(Params)` |
| 1 | `protocol_version` is one the verifier implements | `ProtocolVersion` |
| 2.1 | the architecture's consistency | `Architecture` |
| 2.2 | the public quantization data's rules | `Quantization` |
| 2.3 | every layer's sites rescaled alike ([RFC-0038](RFC-0038-uniform-layers.md#the-rule)) | `Rescaling { site }`, the first site in 03 §2's declaration order that breaks it |
| 2.4 | none: RFC-0065's refusal, in `tessera-protocol/2`, of a site with a `ChannelParams` that is not affine, is lifted by [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) | none |
| 3 | the profile's constraints S1–S3 | `Profile` |
| 4 | each table's regeneration | `Table { table }`, the first in `TableId` order |
| 5.1 | `max_log2_rows` from 1 to 32 | `GadgetPolicy` |
| 5.2 | each gadget-backed node's key, inputs, preconditions and `Exp` table | `Gadget { gadget }`, the first in node order, numbered as [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) numbers gadgets |
| 5.3 | the derived tables against the claim list's, id for id | `Table { table }`, the first id whose two specifications differ, or that only one side has |
| 6 | the commitment parameters and the advice layout | `Layout` |
| 7 | the scoring and anchor templates | `Template` |
| RFC-0019 step 4 | the proof's `PublicStatement` against the one the verifier's inputs give | the first field that differs, in declaration order ([RFC-0036](RFC-0036-protocol-messages.md#round1)): `ClaimListDigest`, `WeightsRoot`, `WeightsEpoch`, `ManifestDigest`, `Batch`, `Predictor`. `Batch` includes a request outside the audit sample or already proven ([RFC-0043](RFC-0043-rejection-payloads.md#the-audit-samples-checks)) |
| RFC-0019 step 4 | the weight commitment's layout against the claim list's ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-verifiers-weight-inputs)) | `Layout` |
| RFC-0019 step 4 | `Round1`'s `layout_digest` against the `commitment_id` the verifier derives ([RFC-0031](RFC-0031-commitment-id.md#layout_digest)) | `LayoutDigest` |

Within one check, the order of its rules does not matter: they all give the same field. Three consequences of the table:

- **The gadget policy's other values.** `limb_bits` and `query_block` are checked where the lowering uses them. A limb gadget's `Range { bits: limb_bits }` table follows RFC-0008's rules like any other, so a bad `limb_bits` fails as `Gadget`. `query_block` shapes the advice layout ([RFC-0034](RFC-0034-causal-blocks.md#causal-layout)), so a bad one fails as `Layout`.
- **A lookup into a `Rope` table** fails at 5.3: no lowering derives a `Rope` table for a lookup.
- **A wide key with a narrow key's tables** passes 5.2, since the verifier selects the gadget from the key's range, and fails at 5.3, where its tables differ.

## Alternatives Considered

1. **A variant per rule.** A rejection would name the exact rule, at the cost of about forty variants, each a tier-A promise. Two implementations would also have to agree on the order of every rule within a check. One variant per check names the datum a reader must look at, and the verifier's report can still quote the rule's text.
2. **A point for `Quantization`.** `Quantization { point }` would name the first offending point. But several rules concern a list as a whole: a missing point, an unsorted list, a site in neither list. Naming one point would need a further convention for each of them.
3. **Free text.** A string can hold anything, including a value of the model. It also gives no stable identifier for tests and tooling.

## Drawbacks

- `Architecture` and `Quantization` do not say which rule failed. The verifier's audit trail (05 §7) can.
- A rejection's field depends on the order of the checks, which this RFC fixes. A verifier that runs them in another order is not wrong about the verdict, but it is wrong about the reason, so the order is tested.

## Migration / Rollout

v0.3, for #346:

1. this RFC, 04 §2's definition and RFC-0007's steps;
2. `tv-core`'s verdict types return `StatementField` (#107, #128).

## Testing Strategy

- **Every variant.** For each variant, a claim list or statement that breaks exactly its check, which `tv-core` rejects with that field (#107, #128).
- **The order.** For each pair of adjacent checks, a claim list that breaks both, rejected with the earlier check's field.
- **The gadget policy.** `max_log2_rows` of 0, 33 and 40, each rejected as `GadgetPolicy` (#107), as `tessera-compiler`'s `select` refuses 0 and 33.

## Open Questions

None.

## References

- [04 §2](../spec/04-error-model.md#2-verdicts), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0025](RFC-0025-affine-sites.md), [RFC-0031](RFC-0031-commitment-id.md), [RFC-0036](RFC-0036-protocol-messages.md), [RFC-0038](RFC-0038-uniform-layers.md).
