# SUR-0014: Public gadget columns before model advice offsets

| | |
|---|---|
| Status | `current (2026-10-19)` |
| Agenda topic | `compiler`, bounded specification prerequisite for `C_lin`, `K_lin`, size and verify |
| Questions | "How tight is worst-case range analysis against measured ranges at 405B, and what would replace the split gadget (two commitments, two lookups) by one lookup where typical ranges fit a 2^32 table (RISK-2: up to +60% commitments)?" |
| Period covered | Primary circuit-interface material from 2023 through 2026; current accepted Tessera RFCs at `5f201fe` |
| Produced by | `autoresearch survey compiler`, 2026-10-19; prerequisite of #106, tracked in #407 |

## Method

This unit answers a prerequisite of the quoted standing question: which committed
values and lookup families the compiler is counting. It does not measure typical
ranges, choose a quantizer, replace RequantSplit, or claim a proving-speed gain.

Read RFC-0009's decomposition, comparison and root specifications, RFC-0041's tile
and value order, and RFC-0011/0047's public layout rules. No new model-layout code
has been written and no prover source has been inspected for this unit.

Searches on 2026-10-19: ePrint "arithmetic circuit witness layout canonical verifier
constraints"; arXiv "zero knowledge circuit witness representation range check
limbs"; Plonky3's official repository for `BaseAir` / `width`. Read the explicit
column-width interface in pinned Plonky3 code, Kimchi's range-check layout section,
and the abstract of Coglio–McCarthy–Smith. The paper's arXiv HTML endpoint failed;
no theorem or full-paper result is credited. These sources explain circuit
contracts; none specifies Tessera's auxiliary columns. Primary Tessera documents
are decisive for that local question.

## Findings

All findings are E0, primary, with no hardware measurement. Dates and URLs are in
the bibliography. Numeric examples below are own analytic counts, not benchmarks.

| Source | What it is | Mechanism | Numbers | Term | Applicability to Tessera | Caveats |
|---|---|---|---|---|---|---|
| [TesseraGadgets26] | Accepted gadget semantics | Integer boxes and comparison chains fix values, while auxiliary root representation remains open | At root width 32, the example below permits one versus two committed auxiliary digits (E0) | `C_lin`, `K_lin` | direct: pin roles and lookups before assigning offsets | This is a format ambiguity, not a false-proof finding |
| [TesseraTiles26] | Accepted tile/value order | Allocate graph outputs before gadget auxiliaries in their specified order | No new numeric claim | size, verify | direct: distinguish graph outputs from extra gadget values | A group named "comparison witnesses" still needs a concrete ordered schema |
| [TesseraAdvice26] | Accepted public advice layout | Offsets, variable order and zero rows are derived from public inputs | One extra scalar auxiliary tensor costs one row when its domain fits a row (E0) | size, verify | direct: an unspecified auxiliary changes the claim map | Later alignment may absorb a gap; individual auxiliary offsets still differ |
| [P3Air26] | Official AIR interface, pinned code | The circuit interface declares its column width | No performance result | `C_lin`, verify | adapt: make column roles and count an explicit public contract | Different arithmetization; do not share an implementation with the independent verifier |
| [KimchiLayout26] | Official range-check specification | Limb-to-column placement is stated alongside constraints | No borrowed numeric estimate | `K_lin`, size | adapt: specify representation as well as integer relations | Its field, gate types and lookup scheme are not Tessera's |
| [CoglioCircuits23] | Primary paper abstract | Circuit correctness concerns the computation encoded by constraints | No borrowed quantitative result | verify | adapt: test semantic correctness separately from format agreement | Abstract only; no formal verification of Tessera is implied |

## Synthesis

The external sources support explicit circuit interfaces; they do not settle a
Tessera-specific layout choice. RFC-0041 requires a fixed auxiliary order, while
RFC-0009's `RsqrtRow` permits any auxiliary root decomposition with a box below the
field modulus. `BigIntCompare` also leaves sufficient high zero limbs and parts of
the normalization representation implicit. Uniqueness of the integer result alone
does not determine the number of committed columns.

An E0 counterexample to a unique representation uses a root in a 32-bit box:

- A: commit its low 16-bit digit; derive its high digit virtually and range-check
  it, as `NonNeg` does.
- B: commit both 16-bit digits and enforce exact recomposition in the stated
  32-bit box.

Both boxes are below Goldilocks, both decompositions are unique for a fixed root,
and both can feed the same base-2^16 product arithmetic. They use the same range
table families; B has one additional auxiliary tensor (all counts E0). At a scalar
row this is one additional advice row. This exhibits the representation gap; it is
not an implemented full-model proof or evidence that current validation accepts a
different complete claim list. Complete family ordering must also be fixed.

RFC-0009's rejected table/Newton alternatives and RFC-0011's rejected per-tensor
commitments are not reopened. The existing root bracket, exact limb boxes, final
carries, row selectors and shared advice commitment remain requirements. No speed,
proof-size saving, soundness-bit increase or zero-knowledge result is credited.

## Candidate ideas

- Pin an ordered public gadget-column contract, including virtual top limbs,
  normalization shapes, carries and range-family requests. Terms: `C_lin`,
  `K_lin`, size and verify. Credited gain: zero (E0); value is making #106's
  independent derivation well-defined. First killer: an already accepted exact
  schema, an omitted committed value, or a box/carry/padding counterexample.

The external sources produce no separate performance idea in this bounded unit:
their circuit formats are not drop-in replacements for Tessera.

## Staleness triggers

An accepted resolution of #407; a new RFC fixing these auxiliary roles; a changed
gadget witness representation or table policy. Refresh against that schema before
using this note to implement either layout. Other compiler standing questions
remain outside this unit.

## References

[TesseraGadgets26], [TesseraTiles26], [TesseraAdvice26], [P3Air26],
[KimchiLayout26], [CoglioCircuits23], resolved in [bibliography.md](../bibliography.md).
