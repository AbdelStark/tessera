# RFC-0042: SiLU Exponents in the Claim List

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0007](RFC-0007-claim-list.md#schema) 'Schema' and 'Verifier validation' step 2: the claim list states each layer's `Silu` input exponent;
  - [RFC-0026](RFC-0026-claim-list-layout.md#schema) 'Schema': the field's place;
  - [03 §2](../spec/03-data-model.md#quantization) 'SiLU': where the claim list states the exponent.
- Origin: [#358](https://github.com/AbdelStark/tessera/issues/358)

## Summary

Each layer's SiLU input exponent `in_exp` is public, and until now the claim list stated it only inside the layer's `Silu` table. But the lowering shares a table among the layers whose definitions are equal (RFC-0006 'Llama lowering'). So a table list does not always say which table each layer reads, and two models whose layers differ in `in_exp` can have byte-identical claim lists. This RFC adds the exponents to the claim list, one per layer, as `IntegerModel` already holds them beside its public quantization data.

## Motivation

Take three layers alike but for their exponents, with one score scale and the same `Silu` site classes and output exponent:

- `in_exp = (a, b, a)` lowers to the tables `[Rope, Exp, Silu(a), Silu(b)]`, and layer 2 reads `Silu(a)`;
- `(a, b, b)` lowers to the same four tables, and layer 2 reads `Silu(b)`.

The two models compute different functions at layer 2, and their claim lists were equal. That broke two invariants:

- INV-CL-3: "two claim lists with the same digest produce the same constraint system in any correct verifier";
- INV-DM-3: every value the verifier rebuilds constraints from is bound by `claim_list_digest`.

The verifier's lowering (RFC-0007 step 5, #107) had no rule to recover `in_exp`. Two correct verifiers could read one claim list two ways, and parties could disagree on what an accepted proof certifies.

The `Exp` and `Requant` tables do not have the problem. Each layer's definition follows from data the claim list states elsewhere: its score scale and its shift classes. `in_exp` is the only per-layer quantity that the claim list stated in a table alone.

## Goals

- A claim list that determines every layer's `Silu` table from public data alone.
- No change to `D`'s schema or to the tables.

## Non-Goals

- Changing how tables are shared.

## Proposed Design

```rust
pub struct ClaimList {                       // schema 0x0201, version 1
    pub protocol_version: u16,
    pub arch: ArchSpec,
    pub profile: Profile,
    pub quant: PublicQuantSpec,
    pub silu_in_exponents: Vec<(u32, i8)>,   // each layer's Silu input exponent, sorted by layer
    pub tables: Vec<TableSpec>,
    pub gadget_policy: GadgetPolicy,
    pub layout: CommitmentLayout,
    pub params: ParamSetId,
}
```

- **Its place.** `IntegerModel`'s `QuantSpec` holds `silu_in_exponents` beside its public part (03 §2), and the claim list now holds it beside its copy of that part, `quant`.
- **Its table.** A layer's `Silu` table still holds `in_exp` in its definition (RFC-0008). The lowering builds each layer's table from the stated exponent, and RFC-0007 step 5.3 checks the claim list's tables against the derived ones, id for id. A claim list whose exponent and table disagree fails there (`Table { table }`, [RFC-0039](RFC-0039-statement-fields.md#the-checks)).
- **Validation.** Step 2's quantization rules gain one: `silu_in_exponents` holds one exponent per layer, sorted by layer. `tessera-compiler` already applies it to `D` (`Quantization`).
- **The verifier's lowering** reads the architecture, the profile, `quant` and `silu_in_exponents`: what `tessera-compiler` reads from `D` (RFC-0006 'Llama lowering'). It reads them from the claim list alone.

## Alternatives Considered

1. **One `Silu` table per layer, never shared.** The tables would map to layers by position. But every layer would add a LogUp instance with its own multiplicities, up to 126 at 405B, for an exponent that costs one byte to state.
2. **A rule for the verifier, such as reusing the first match**, together with a compiler refusal of the models it cannot tell apart. This constrains quantization to work around a defect of the encoding.
3. **The exponents in `PublicQuantSpec`.** That would move them inside `D`'s public part, changing `IntegerModel`'s schema and every model vector, while giving the claim list the same bytes up to order.

## Drawbacks

- The exponent is stated twice in the claim list: in its field and in its layer's table. Step 5.3 keeps the two in agreement.

## Migration / Rollout

v0.3, for #358:

1. this RFC and its amendments;
2. `tessera-compiler` emits the field, and lowers a claim list from its own fields (`PublicModel::of_claims`);
3. the claim-list vectors of `spec/vectors/claims/` are regenerated;
4. the verifier's claim-list validation reads the field (#107).

## Testing Strategy

- **The example.**
  - The two models above list the same tables, and their claim lists differ.
  - Each claim list lowers, from its fields alone, to its model's graph and tables.
- **Every conformance model.** Lowering its claim list gives the graph that lowering `D` gives.

## Open Questions

None.

## References

- [RFC-0006](RFC-0006-compiler-ir.md#llama-lowering), [RFC-0007](RFC-0007-claim-list.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0026](RFC-0026-claim-list-layout.md), [RFC-0039](RFC-0039-statement-fields.md).
