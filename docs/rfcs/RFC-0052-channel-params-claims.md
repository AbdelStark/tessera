# RFC-0052: Opening Several Claims on `ChannelParams`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-20
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass) 'Opening by the folding pass': opening `ChannelParams` at `m ≥ 1` evaluation claims in one pass;
  - [RFC-0036](RFC-0036-protocol-messages.md#round2response) '`Round2Response`': `WeightCombinedRows.evaluations` as `Vec<CombinedRow>`;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool) 'The claim pool' and 'Invariants': `ChannelParams` claims in the pool;
  - [RFC-0045](RFC-0045-leaf-layout.md#open-questions) 'Open Questions': resolves the `ChannelParams` opening question;
  - [RFC-0049](RFC-0049-weight-proximity.md#non-goals) 'Non-Goals': opening `ChannelParams` across several claims in one pass.
- Amended by: [RFC-0065](RFC-0065-masked-channel-params.md) ('Opening by the single-pass fold': in `tessera-protocol/2`, each claim's own mask row and the matrix's blinding pair), which supersedes 'Non-Goals' for `tessera-protocol/2`
- Origin: [#386](https://github.com/AbdelStark/tessera/issues/386)

## Summary

Each weight matrix `Embed`..`LmHead` has one claim in the claim pool (INV-SCH-5), while `ChannelParams` receives one claim per site in the layer template that reads channel classes (in Stage 0) or affine multipliers (in Stage 1 or 2). Because different sites belong to different weight types `t_i`, have different output widths `out_{t_i}`, or sit in different stages and instances, their claims on `ChannelParams` lie at distinct points on `Ã_CP`, with distinct column sub-points when `k_CP > C · L_pad` or when the claims come from different stages. This RFC specifies how each site's claim maps to a point on `Ã_CP` and how `ChannelParams` opens all `m` of its claims in one pass over its committed rows:

- **Point on `Ã_CP`.** Claim `i ∈ 0..m` on weight type `t_i` (with padded output width `2^{c_i} ≤ C = 2^γ`), at channel point `r_{c, i}` (`c_i` coordinates) and layer point `ρ_i` (`λ` coordinates), evaluates `Ã_CP` at `point_i = r_{c, i} ‖ 0^{γ − c_i} ‖ ρ_i ‖ vertex_4(t_i)`.
- **One evaluation row per claim.** `WeightCombinedRows` carries `evaluations: Vec<CombinedRow>`, with one evaluation row per claim on the matrix in pool order (`1` for `Embed`..`LmHead`, `m` for `ChannelParams`), followed by `proximity: CombinedRow`.
- **Single-pass fold.** For `ChannelParams`, the prover accumulates all `m` folds `F_i = Σ_{j < m_CP} eq(row_point_i, r_j) · W_CP[row_j]` and the proximity row `u_{ξ, CP}` in one pass over the `m_CP` committed rows, then scales each `F_i` by `θ_CP` (`pcs/weights/ChannelParams/gamma`) to form `evaluations[i] = θ_CP · F_i`.

## Motivation

- **Multiple reading sites.** In the Llama layer template ([RFC-0044](RFC-0044-lowering-nodes.md)), up to seven per-layer weights (`Q`, `K`, `V`, `O`, `Gate`, `Up`, `Down`) have a `ChannelParams` entry ([03 §2](../spec/03-data-model.md#quantization)). A `pot` site of several classes reads its class tensor in Stage 0 ([RFC-0045](RFC-0045-leaf-layout.md#the-split)), stacked over layers at `ρ_1` ([RFC-0037](RFC-0037-stage-schedule.md#the-layer-point)). An affine site reads its multiplier tensor in a `Mul(Leading)` instance in Stage 1 or 2 ([RFC-0025](RFC-0025-affine-sites.md)), stacked at that instance's layer coordinates. Each site leaves its own claim in the claim pool.
- **Distinct column points.** `ChannelParams` has shape `[16, L_pad, C]` (`L_pad = 2^λ`, `C = 2^γ`), flattened row-major as `f = (t · L_pad + ℓ) · C + c`, with message width `k_CP = min(k_W, 16 · L_pad · C)` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)). When `k_CP > C` (for example at 405B where `k_CP = 2^{20}` and `C = 2^{16}`, or at 8B where `k_CP = 2^{20}` and `C · L_pad = 2^{17}`), the column index `f mod k_CP` includes layer bits and, when `k_CP > C · L_pad`, type bits `t_i`. Even when two Stage-0 sites of the same width share `r_{c, i}` and `ρ_1`, their type bits `t_i` differ whenever `k_CP > C · L_pad`, and claims from different stages or different widths have different `r_{c, i}` or `ρ_i`. Thus the claims on `ChannelParams` do not share a single column point.
- **Schedule-fixed message shapes.** Grouping claims by runtime equality of `column_point_i` would make the number of combined rows depend on challenge values rather than on the schedule alone ([RFC-0036](RFC-0036-protocol-messages.md#round2response) 'Shapes'). Giving `WeightCombinedRows` one evaluation row per claim in pool order fixes every length from the schedule.

## Goals

- Define the exact point on `Ã_CP` for every claim on `ChannelParams`.
- Open all `m` claims on `ChannelParams` in one read of each committed row (INV-WC-4).
- Keep the transcript schedule ([RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule)) and the soundness calculator ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator)) unchanged.

## Non-Goals

- Changing the commitment layout of `ChannelParams` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)).
- `tessera-protocol/2`'s blinding rows and masks for `ChannelParams` (#147).

## Proposed Design

### Point on `Ã_CP`

Let `ChannelParams` have `C = 2^γ` and `L_pad = 2^λ`, so the flattened tensor `[16, L_pad, C]` has `ν_CP = γ + λ + 4` variables, ordered lowest bit first ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)): `γ` channel bits, `λ` layer bits, and `4` weight-type bits.

A site that reads the channel parameters of weight type `t_i ∈ 0..9` (with output width `out_{t_i}` and padded output width `out_pad_{t_i} = 2^{c_i} ≤ C = 2^γ`) leaves a claim `P̃_{t_i}(r_{c, i} ‖ ρ_i) = v_i` in the claim pool, where:

- `r_{c, i}` has `c_i = log2(out_pad_{t_i})` coordinates (lowest bit first);
- `ρ_i` has `λ = log2(L_pad)` coordinates (`0^λ` if a weight without layers ever had a `ChannelParams`);
- `vertex_4(t_i) = (Fp2::from(t_i & 1), Fp2::from((t_i >> 1) & 1), Fp2::from((t_i >> 2) & 1), Fp2::from((t_i >> 3) & 1))` is the 4-bit boolean vertex of `t_i`.

Because every entry of `ChannelParams` at `c ∈ [out_{t_i}, C)` (and in particular at `c ∈ [2^{c_i}, C)`, where at least one of the high `γ − c_i` channel bits is `1`) and at `ℓ ∈ [n_layers, L_pad)` is `0` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)), the multilinear extension `Ã_CP` of `ChannelParams` at

```text
point_i = r_{c, i} ‖ 0^{γ − c_i} ‖ ρ_i ‖ vertex_4(t_i)
```

satisfies

```text
Ã_CP(point_i) = Σ_{ℓ < n_layers} eq(ρ_i, ℓ) · P̃_{t_i, ℓ}(r_{c, i}) = v_i.
```

Splitting `point_i` at `log2(k_CP)` gives:

- `column_point_i = point_i[..log2(k_CP)]` (`log2(k_CP)` coordinates);
- `row_point_i = point_i[log2(k_CP)..]` (`ν_CP − log2(k_CP)` coordinates).

### Pool order of `ChannelParams` claims

Claims on `ChannelParams` enter the claim pool in schedule order ([RFC-0037](RFC-0037-stage-schedule.md#splits)):

1. **Stage 0 (leaf stage).** Each gadget site in the layer template whose lookups read a channel-class tensor produces one stacked claim at channel sub-point `r_{c, i} = π_c(r_s)` and layer point `ρ_1`, in node order ([RFC-0045](RFC-0045-leaf-layout.md#the-split)).
2. **Reduction stages (`k ≥ 1`).** Each `Mul(Leading)` instance whose factor is an affine multiplier tensor produces one claim at that instance's `(r_{c, i}, ρ_i)`, in stage and instance order ([RFC-0037](RFC-0037-stage-schedule.md#stages-and-their-instances)).

Let `m` be the total number of `ChannelParams` claims in the pool. When `m = 0`, no weight has a `ChannelParams`, `m_CP = 0`, and `ChannelParams` is not opened ([RFC-0036](RFC-0036-protocol-messages.md#round2response), [RFC-0049](RFC-0049-weight-proximity.md#the-schedule)).

### `WeightCombinedRows`

In [RFC-0036](RFC-0036-protocol-messages.md#round2response), `WeightCombinedRows` is amended to:

```rust
pub struct WeightCombinedRows {
    pub matrix: u8,
    pub evaluations: Vec<CombinedRow>,
    pub proximity: CombinedRow,
}
```

- For `Embed`..`LmHead` (`matrix ∈ 0..9`), `evaluations` has length `1`.
- For `ChannelParams` (`matrix == 9`), `evaluations` has length `m ≥ 1`, one `CombinedRow` per claim in pool order.
- Each `CombinedRow` in `evaluations` and `proximity` has `u.len() == k_T` and, in `tessera-protocol/1`, `mask.is_empty()`.
- Under `pcs/weights/{T}/combined` (row 8 of the transcript schedule), the transcript absorbs `TCE(WeightCombinedRows)` as before.

### Opening by the single-pass fold

For any opened weight matrix `T` with `m ≥ 1` claims `(column_point_i, row_point_i, v_i)` (`i ∈ 0..m`, with `m = 1` for `Embed`..`LmHead`):

1. **Single-pass fold.** As the prover streams the committed rows `j ∈ 0..m_T` (with structural row index `r_j`), it accumulates the `m` evaluation folds and the proximity row in one pass:

   ```text
   F_i = Σ_{j < m_T} eq(row_point_i, r_j) · W_T[row_j]    for i ∈ 0..m,
   u_{ξ, T} = Σ_{j < m_T} eq(ξ*_T, j) · W_T[row_j].
   ```

   For `Embed`..`LmHead`, `F_0` is the layer-weighted sum `Σ_ℓ eq(ρ_ℓ, ℓ) · F_ℓ` of the per-layer matmul folds. For `ChannelParams`, each committed row `j` is read once and added into `u_{ξ, CP}` and into each `F_i` whose `eq(row_point_i, r_j)` is non-zero.
2. **Scaling by `θ_T`.** After row 7 squeezes `θ_T` under `pcs/weights/{T}/gamma`, the prover sets

   ```text
   evaluations[i].u = θ_T · F_i    for i ∈ 0..m.
   ```

3. **Evaluation check (RFC-0019 step 9).** For each `i ∈ 0..m`, the verifier checks `evaluations[i].u.len() == k_T`, `evaluations[i].mask.is_empty()` (in `tessera-protocol/1`), and

   ```text
   ũ_{T, i}(column_point_i) = θ_T · v_i,
   ```

   rejecting any mismatch with `Reject(Evaluation { commitment: Weight(T) })`.
4. **Column check (RFC-0019 step 11).** The verifier encodes each `evaluations[i]` (`i ∈ 0..m`) and `proximity` into codewords `w_0, …, w_{m−1}, w_m` on `7 · L_T`. At each queried column `q`, with column symbols `c[0..m_T]`, it checks:
   - for each `i ∈ 0..m` (`row = i` in `Reject(Proximity)`):

     ```text
     w_i[q] = θ_T · Σ_{j < m_T} eq(row_point_i, r_j) · c[j];
     ```

   - for `proximity` (`row = m` in `Reject(Proximity)`):

     ```text
     w_m[q] = Σ_{j < m_T} eq(ξ*_T, j) · c[j].
     ```

   A mismatch on row `r ∈ 0..=m` is `Reject(Proximity { commitment: Weight(T), column: q, row: r as u16 })`.

### Soundness and the calculator

- Each evaluation row `evaluations[i]` combines a single claim (`T = 1` claim in that combined row), scaled by `θ_T`. In `tessera-protocol/1` there is no blinding row, so `θ_T · v_i = ũ_{T, i}(column_point_i)` and the column check `w_i[q] = θ_T · Σ_j eq(row_point_i, r_j) · c[j]` hold with `θ_T ≠ 0` (or `θ_T = 0` with probability `1 / |Fp2|`).
- Therefore each evaluation row has batching error at most `1 / |Fp2|`, identical to a single-claim matrix opening (`matrix_opening = 1` in [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator)), and `ChannelParams`' proximity row is unchanged (`ϑ_CP = ⌈log2 m_CP⌉`).

## Alternatives Considered

1. **Group `ChannelParams` claims that happen to share `column_point_i` at runtime.** Rejected: when `k_CP ≤ C`, claims of the same output width in Stage 0 could share `column_point_i`, whereas at `k_CP > C · L_pad` they do not because `t_i` enters `column_point_i`. Grouping by runtime equality of `column_point_i` would make the number of combined rows depend on runtime challenge values rather than on the schedule, and at 405B (`k_CP = 2^{20} = 16 · C`) claims with different `(t_i, ℓ_lo)` have distinct `column_point_i` anyway.
2. **Run a sumcheck over `ChannelParams` to reduce all `m` claims to one point.** Rejected: adds an extra sumcheck stage or instance for a tensor whose entire opening is at most `m ≤ 7` rows of `k_CP` field elements.

## Drawbacks

- `Round2Response` carries `m` evaluation `CombinedRow`s for `ChannelParams` instead of `1` (`m ≤ 7` in the Llama layer template).

## Migration / Rollout

v0.3, for #386:
1. this RFC and its amendments to RFC-0012, RFC-0036, RFC-0037, RFC-0045 and RFC-0049;
2. `WeightCombinedRows.evaluations: Vec<CombinedRow>` in `tessera-protocol` and the protocol conformance vectors;
3. `WeightLayout::channel_point`, multi-claim `WeightFold::channel_params` and `WeightOpening` in `tessera-pcs`, and multi-claim `ChannelParams` verification in `tv-core`, with updated `spec/vectors/weights-opening/` conformance vectors.

## Testing Strategy

- **Single pass.** Opening `ChannelParams` on a model whose sites read classes and multipliers at several points reads each committed row of `ChannelParams` once and produces all `m` evaluation rows plus the proximity row.
- **Adversarial.** Tampering with a class or multiplier entry after commitment, or with any of the `m` claimed values `v_i` or `evaluations[i]`, is rejected by the verifier with `Reject(Evaluation)` or `Reject(Proximity)` naming the exact row `0..=m`.
- **Conformance.** `spec/vectors/protocol/` and `spec/vectors/weights-opening/` match in both workspaces.

## Open Questions

None.

## References

- [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0025](RFC-0025-affine-sites.md), [RFC-0036](RFC-0036-protocol-messages.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0049](RFC-0049-weight-proximity.md).
- [#386](https://github.com/AbdelStark/tessera/issues/386).
