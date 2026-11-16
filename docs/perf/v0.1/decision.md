# The v0.1 decision: stop and redesign

[RFC-0020's decision rule](../../rfcs/RFC-0020-cost-model.md#decision-rule), fixed before any measurement, applied on 2026-11-09 to the three unit costs measured on one H100 ([#66](https://github.com/AbdelStark/tessera/issues/66)).

| Unit | Assumed | Measured (median) | `ρ = measured / assumed` | Report |
|---|---|---|---|---|
| `c_c` | 30 ps | 134.7 ps | 4.49 | [c_c on H100](cc-h100.md) |
| `c_ℓ` | 40 ps | 297.0 ps | 7.42 | [c_ℓ on H100](l-h100.md) |
| `c_π` | 12 ps | 84.6 ps | 7.05 | [c_π on H100](pi-h100.md) |

**Outcome: Stop and redesign.** Every `ρ` exceeds 4, and the rule needs only one. GPU prover work stops, and each affected component needs a superseding RFC before v0.5 work resumes. Here that is all three: the advice commitment's encoding ([RFC-0011](../../rfcs/RFC-0011-advice-commitment.md)), LogUp ([RFC-0008](../../rfcs/RFC-0008-logup-and-tables.md)) and the tree levels ([RFC-0010](../../rfcs/RFC-0010-proof-schedule.md#tree-levels)).

## Why the measured ratios govern

Each report also estimates what a better kernel could reach for the same algorithm:

| Unit | Floor estimate | `ρ` at the floor |
|---|---|---|
| `c_c` | 79 ps | at least 2.6 |
| `c_ℓ` | 80–100 ps | about 2–2.5 |
| `c_π` | above 30 ps | above 2.5 |

At those floors the outcome would be "Proceed with re-plan". The rule takes measurements, and it was written before them so that the outcome could not shape the criteria. A floor estimate is a projection, and swapping it for a measurement would do exactly that. Two further reasons make the stricter reading the right one:

- **Both readings demand a redesign.** At the floors, the 405B projection still misses [SC-2](../../spec/00-overview.md#7-success-criteria) by a factor of about two or more, and a re-plan that misses SC-2 needs an RFC amending it (below). Neither reading lets v0.5 go ahead as planned.
- **The gap is structural, not tuning.** RFC-0011's encoding costs 34 Goldilocks butterflies per committed element. LogUp's trees are in `Fp2` from the leaves up. The tree levels move their bound tables through memory every round. Kernel work buys a factor of 1.5 to 4. The assumptions need 4 to 7.

## The restated projection

The research cost model (`research/tools/cost_model.py`, `counted` profile: the 405B commitment report's counts, every correction of [research/baseline.md](../../../research/baseline.md)) at the measured unit costs, with the column answers priced as committing:

| Context | 1K | 4K | 16K | 128K |
|---|---|---|---|---|
| `μ` at the assumed costs (`counted` profile) | 32.4 | 46.9 | 102.5 | 436.4 |
| `μ` at the measured costs | **58.3** | **85.2** | **187.1** | **785.3** |
| SC-2 ceiling | 12 | | 45 | |

```sh
python3 -I research/tools/cost_model.py --profile counted \
  --set commit_ps=134.7 --set lookup_ps=297.0 --set leaf_ps=84.6 --set column_answer_ps=134.7
```

At 405B the projection misses SC-2 by 4.9× at 1K and 4.2× at 16K. Linear-layer commitments and lookups dominate at short contexts. Attention's trees dominate from 4K on.

These remain projections (E0) built on measured unit costs (E1). No end-to-end GPU proof exists. The other units of the protocol (glue, weight pass, native rate, streaming pass, `c_a`, `c_r`) are not measured. With row combination priced at zero, they could only raise these figures.

## What changes

- **Stopped:** the production v0.5 GPU prover as planned (#165, #166, #168, #171, #172, #173 and #174, at their full scope). The microkernels, their tests and the benchmark stay as the harness that will measure the redesign.
- **Set the same day:** the MVP priority ([MVP plan](../../roadmap/MVP.md), [#722](https://github.com/AbdelStark/tessera/issues/722)).
  - The MVP builds the end-to-end GPU path on the current protocol, at the narrow scope of its own issues, as the measured baseline the redesign needs.
  - It is a measurement, not a resumption of v0.5. The redesign is scored against what it measures (#745).
- **Next.** The redesign starts as research ([research/LEDGER.md](../../../research/LEDGER.md)), one superseding RFC per component, each scored against the measured costs, not the assumed ones:
  - **The commitment.** Commit fewer elements (NG-5's small-value commitments, ring switching), or use an encoding with fewer field multiplications per element: a cheaper code, a smaller field for the codeword, or a lower rate.
  - **LogUp.** Use fewer lookups per operation (`K_lin`), keep the leaves' fractions in the base field longer, or adopt a cheaper lookup argument.
  - **The tree levels.** Use fewer `Fp2` passes per level (Gruen's trick, binding from the base level), or use cheaper or shared max certificates (RFC-0009 `SoftmaxRow`).
- **Unchanged.** SC-2's ceilings stand until an RFC amends them. Nothing here relaxes soundness, zero knowledge or verifier independence: a redesign that buys speed with any of them is rejected ([research/baseline.md §1](../../../research/baseline.md#1-what-an-improvement-must-preserve)).
