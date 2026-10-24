# RFC-0049: The Weights' Proximity Point Before the Stages

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-19
- Target milestone: v0.3
- Amended by: [RFC-0052](RFC-0052-channel-params-claims.md) ('Non-Goals': opening `ChannelParams` across several claims in one pass); [RFC-0057](RFC-0057-one-proof-unit.md) ('Motivation': the weight read is per proof)
- Supersedes, on acceptance:
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' rows 3 and 7: where `pcs/weights/{T}/xi` is squeezed in `tessera-protocol/1`;
  - [RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass) 'Opening by the folding pass': the proximity row's coefficients and when the fold computes it;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) 'Soundness calculator': a weight matrix's `ϑ_T` and mask in `tessera-protocol/1`.
- Origin: [#108](https://github.com/AbdelStark/tessera/issues/108)

## Summary

RFC-0012 promises one pass over the weights per proof, producing both the sumcheck folds and the opening's combined rows (INV-WC-4). The schedule broke the promise:
- The folds run inside the `MatMulWeight` stage, which needs `W̃(ℓ, r_o, i)` at its point.
- The proximity row `Σ_row eq(ξ*_T, row)·W_row` needs `ξ*_T`, which row 7 squeezed after every stage.
- So the proximity row needed a second read of every weight, about 0.24 GPU-seconds per 405B proof (RFC-0018).

This RFC squeezes each matrix's `ξ*_T` right after the seed. The fold pass then computes both rows. It also fixes which rows `ξ*_T` indexes, and the calculator's count for `tessera-protocol/1`.

## Motivation

- **The pass is the cost.** The weight fold is RFC-0018's pass W, the one pass a proof makes over the 405 GB of weights. A second pass for the proximity row doubles it.
- **The point can come early.** The proximity test needs `ξ*_T` drawn after the matrix's rows are fixed, and the column queries drawn after the combined rows (RFC-0011 'Opening protocol'). `R_W` is fixed before the session opens, so every point after the seed satisfies the first condition. The queries stay last.
- **Ambiguities.**
  - **Which rows `ξ*_T` indexes.** RFC-0012 says "one coordinate per bit of the matrix's row index, data and blinding rows together". RFC-0040 defines a structural row index, while the calculator counts `⌈log2(m_T + B_W)⌉` coordinates.
  - **What `tessera-protocol/1` counts.** It commits no blinding rows and no masks for the weights (RFC-0040), and the calculator still counted both.

## Goals

- One pass over the weights per proof: INV-WC-4 holds.
- One reading of the proximity coefficients.

## Non-Goals

- `tessera-protocol/2`'s blinding rows and masks, and where their labels go (#147).
- Several claims on one matrix, such as `ChannelParams` claims from several sites ([#386](https://github.com/AbdelStark/tessera/issues/386), resolved by [RFC-0052](RFC-0052-channel-params-claims.md)).

## Proposed Design

### The schedule

In `tessera-protocol/1`:
- **Row 3.** Right after the seed, the transcript squeezes `pcs/weights/{T}/xi` for each weight matrix `T` with committed rows, in RFC-0040's matrix order. A matrix with no committed rows has no proximity row and squeezes no point.
- **Row 7** keeps `pcs/advice/gamma`, `pcs/advice/xi` and `pcs/weights/{T}/gamma`, and no longer squeezes the weights' `xi`.
- **Nothing else moves.**
  - The advice's `ξ*` stays at row 7: the advice is committed in `Round1`, and its opening pass regenerates its rows anyway (RFC-0018 pass P4).
  - The combined rows are still sent at row 8, and the queries still come at row 10.

### The proximity row

`ξ*_T` has `⌈log2 m_T⌉` coordinates, one per bit of a committed row's position. `m_T` is the matrix's committed rows (RFC-0040), and row `j` is the `j`-th committed row in RFC-0040's order. The proximity row is

```text
u_ξ,T = Σ_{j < m_T} eq(ξ*_T, j)·W_T[row_j]
```

and the fold pass computes it beside the folds `F_ℓ`, in the same read of each row. At a queried column, the verifier checks `u_ξ,T`'s codeword against the column's answers weighted by `eq(ξ*_T, j)`, as RFC-0011 step 3 does for the advice.

- **Why positions.** The column answers carry one symbol per committed row, so positions index them directly. Skipped rows are public zeros and add nothing.

### The calculator

In `tessera-protocol/1` a weight matrix has:
- `ϑ_T = ⌈log2 m_T⌉`;
- no mask, so its rate is `k_T / n_T`.

A matrix with no committed rows has no proximity row and no queries row. In `tessera-protocol/2` the counts stay those of RFC-0014, with `B_W` and `P_W`.

### Soundness

- **The proximity argument is unchanged.** It needs `ξ*_T` independent of the committed rows, and the queries drawn after `u_ξ,T` is sent. `R_W` predates the session, `ξ*_T` follows the seed, and the queries come at row 10.
- **Live mode.** The seed is the coin toss, so `ξ*_T` is uniform even against a prover that chose its weights adversarially.
- **Non-interactive mode.** `ξ*_T` is squeezed from a transcript that follows the seed's proof of work, so re-drawing it costs that work. The prover's choice of weights is fixed in `R_W` before any squeeze.

## Alternatives Considered

1. **A second pass over the weights.** It keeps the schedule and doubles RFC-0018's pass W at 405B, against INV-WC-4.
2. **Structural row indices.** These are `ϑ_T = row_vars`, as the advice uses. They add coordinates for skipped rows that the column answers do not carry.

## Drawbacks

- `tessera-protocol/1` and `/2` may place the weights' `xi` differently, depending on how #147 masks the proximity row.

## Migration / Rollout

v0.3:
1. this RFC and its amendments;
2. the folding-pass opening computes `u_ξ,T` in the fold pass (#108);
3. the transcript schedule squeezes the weights' `xi` after the seed (#125);
4. the calculator counts `ϑ_T` and the mask by protocol version.

## Testing Strategy

- **One pass.** An instrumented counter shows one read of each weight per proof, producing the folds, `u_T` and `u_ξ,T`.
- **Binding.** A weight mutated after the commitment fails the proximity check at a queried column.
- **Schedule.** The expanded schedule has each `pcs/weights/{T}/xi` after `seed` and before the first stage label, and none at row 7.
- **Calculator.**
  - `ϑ_T = ⌈log2 m_T⌉` and an unmasked rate in protocol/1.
  - An empty `ChannelParams` adds no rows and is not refused.

## Open Questions

None.

## References

- [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure), [RFC-0040](RFC-0040-weight-commitment-artifacts.md).
