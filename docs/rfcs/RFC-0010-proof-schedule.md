# RFC-0010: Batched Constant-Depth Sumcheck/GKR Proof Schedule

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3

## Summary

After all advice is committed and the seed is fixed, the prover runs a fixed sequence of stages. Each stage is one batched sumcheck (or one batched GKR tree level) covering every layer, head and query block, with shared challenges. Claims flow from lookup and product-tree leaves back through virtual relations until they land on committed tensors, where they are batched into openings. Because every non-arithmetic output is committed, the longest virtual chain inside a layer is a handful of relations, so the number of stages is independent of the number of layers. Shared challenges put every layer's claims on one weight type at a single point, which is what lets the weight commitment open with one folding pass.

## Motivation

The source's "constant-depth proof" (source §7) is what removes the sequential dependency of layer-by-layer GKR: "the longest chain inside a layer is about six stages ... Each stage runs as one sumcheck batched over all layers, heads and query blocks with shared challenges. Proof depth no longer grows with model depth." The weight-commitment design depends on it: "Batching every layer's instance of a weight type with shared challenges lands all its weight claims on one common point per type" (source §7, "Weights opened by the folding pass"). This RFC turns those sentences into a schedule both sides derive identically.

## Goals

- A deterministic stage schedule derived from the constraint system by prover and verifier independently.
- Stage count independent of `n_layers`.
- Every claim on a committed tensor of a given kind lands on one point per stage.
- Sumcheck messages and their verification fully specified.

## Non-Goals

- Zero-knowledge masking of stages (RFC-0013).
- Kernel-level scheduling (RFC-0018).

## Proposed Design

### Multilinear extensions

A table of `2^n` values `f[0], …, f[2^n − 1]` defines the multilinear polynomial `f̃(x_1, …, x_n)` that equals `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. A point `r = (r_1, …, r_n)` sets `x_i = r_i`, and

```text
eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))        f̃(r) = Σ_{x ∈ {0,1}^n} f[x]·eq(r, x)
```

- Tensors are flattened row-major, so their last index takes the lowest variables, and a stacked tensor's layer index the highest. Function notation such as `W̃(ℓ, o, i)` names the indices in row-major order. As a coordinate vector, low variables first, the same point is `r_i ‖ r_o ‖ ρ_ℓ`, as RFC-0011 writes `r ‖ base_X`.
- Sumcheck round `j` binds `x_j` to its challenge `r_j`: each pair `(f[2k], f[2k+1])` becomes `f[2k] + r_j·(f[2k+1] − f[2k])`. A stage's final point therefore lists its challenges in round order.
- Two public predicates have `O(n)` closed forms, which the prover and the verifier each evaluate:
  - `LT(r, a) = Σ_{h < a} eq(r, h)`, for an integer `0 ≤ a ≤ 2^n` ([RFC-0008](RFC-0008-logup-and-tables.md#structured-evaluation));
  - the causal selector `C(q, i) = Σ_{i' ≤ q'} eq(q, q')·eq(i, i')`, over pairs of `n`-bit positions ([RFC-0009](RFC-0009-gadgets.md#softmaxrow)).

### Relations

Every constraint in the derived constraint system is one of these relation kinds, each a sumcheck identity over the Boolean hypercube:

| Kind | Identity (claim on target at point `r`) | Degree per variable |
|---|---|---|
| `MatMulWeight` | `Ỹ(r_o) = Σ_i W̃(r_o, i) X̃(i)` | 2 |
| `MatMulAct` | `Z̃(r_a, r_c) = Σ_j Ã(r_a, j) B̃(j, r_c)` | 2 |
| `Elementwise` (`Mul`, `Rotate`, scalar-by-row) | `Z̃(r) = Σ_x eq(r, x) Ã(x) B̃(x)` (constants evaluated by the verifier) | 3 |
| `SumSquares` | `Ũ(r) = Σ_x eq(r, x_row) X̃(x)^2` | 3 |
| `Linear` (sums, adds, residual chains, selectors) | Evaluated directly from input claims at the same point; no sumcheck | — |
| `ZeroCheck` (`OneHot` booleanity) | `0 = Σ_x eq(r, x) h̃(x)(1 − h̃(x))` | 3 |
| `ProductTreeLevel` | `V_ℓ(r) = Σ_x eq(r, x) V_{ℓ+1}(x, 0) V_{ℓ+1}(x, 1)` | 3 |
| `FractionTreeLevel` | LogUp-GKR level (RFC-0008) | 3 |

Claims on a `Linear` target become claims on its inputs at the same point with public coefficients, so linear chains (residual stream sums, `c_q − x_qi`, offsets) cost nothing.

### Stages

- **Leaf stage.** All LogUp fraction trees and all product trees, level by level from the root, batched across instances. Output: claims on lookup keys (virtual), lookup values and multiplicities (committed), product-tree leaves (virtual softmax keys).
- **Reduction stages.** A virtual claim is assigned to the stage equal to its distance from the leaf stage in the dependency DAG of virtual tensors. Stage `k` proves every relation instance whose target claims arose in stage `k − 1`, across all layers, heads and blocks, in one batched sumcheck. Its outputs are claims on inputs: committed tensors go to the opening pool; virtual tensors go to stage `k + 1`.
- **Termination.** The schedule ends when no virtual claim remains. Every chain ends at committed tensors because every non-arithmetic output is committed (INV-IR-3).

For the `servable` Llama template the longest chain is: requantization key of the residual update ← O-projection (`MatMulWeight`) ← attention output requantization key ← `R_q · PV` (`Elementwise`) ← `PV` (`MatMulAct`) ← committed `e`, `v`. In the `minimal` template one more stage appears because the attention output is not committed. The compiler's commitment report lists the stage count; it must be identical for 2-, 4-, 8- and 126-layer models of the same template.

### Batching within a stage

- Instances of one stage are combined with powers of a batching challenge `γ` drawn after the stage's input claims are fixed (label `stage/{k}/gamma`).
- Instances with fewer variables than the stage maximum `n_max` are treated as functions constant in the extra leading variables; their claims are scaled by `2^{n_max − n_i}` (the standard front-padding argument), so all instances share one round sequence.
- **Stacked layers.** Per-layer instances of one relation are stacked: the layer index becomes `⌈log2 n_layers⌉` extra sumcheck variables (padded layers are zero). A stage therefore ends with one claim per stacked tensor, at one point that includes a layer coordinate `ρ_ℓ`. For weights this yields a single claim `W̃_T(ρ_ℓ, r_o, r_i)` per weight type, opened with one folding pass ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass)); for zero knowledge it means one masked evaluation per tensor kind per proof rather than one per layer ([RFC-0013](RFC-0013-zero-knowledge.md#stacked-layers)).
- All instances share the stage's round challenges `r_1 … r_{n_max}`; different relations and tensors of different shapes are combined with powers of `γ` as above.

### Sumcheck messages

- Round `j` sends the round polynomial `p_j` of degree `d` as its evaluations at `{0, 2, 3, …, d}`; the verifier derives `p_j(1) = claim_j − p_j(0)`.
- After absorbing the message, the challenge `r_j ∈ Fp2` is squeezed (label `stage/{k}/round/{j}`).
- At the end, the prover sends the claimed evaluations of every input polynomial at the final point (label `stage/{k}/evals`). The verifier checks that the stage's batched combination of these evaluations, with public factors (`eq`, selectors, constants) it computes itself, equals the last round's claim.

```rust
pub struct SumcheckProof { pub rounds: Vec<Vec<Fp2>>, pub final_evals: Vec<Fp2> }
pub struct StageProof { pub stage: StageId, pub sumcheck: SumcheckProof }
pub struct Claim { pub poly: PolyRef, pub point: Vec<Fp2>, pub value: Fp2 }

pub fn derive_schedule(cs: &ConstraintSystem) -> StageSchedule;    // prover and verifier each implement this
pub fn prove_stage(stage: &Stage, w: &mut dyn WitnessSource, tr: &mut Transcript)
    -> (StageProof, Vec<Claim>);
```

`WitnessSource` regenerates the evaluation tables a stage needs from committed data and weights on demand (recompute, never store; RFC-0018). Stages whose tables exceed device memory (the leaf levels of fraction and product trees at production sizes) are proven by streaming their early rounds over the regenerated witness, one pass per streamed round, until the folded state fits in memory ([RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure)); the messages are identical to those of the in-memory prover.

### Claim pool and openings

All claims on committed advice are collected in a pool keyed by polynomial. Claims on the same polynomial at different points are reduced to one point by a final batched sumcheck over the advice matrix (RFC-0011). Claims on weights are combined per weight type (RFC-0012). Claims on channel parameters (hidden shift classes) join the weight openings.

### Soundness of the schedule

For a stage with `n` rounds of degree `d` and `m` batched instances, the error is at most `(n·d + m)/|Fp2|`. Each tree level contributes likewise. The total over all stages and levels is summed by the soundness calculator (RFC-0014); for 405B at 16K tokens it is below 2^-110.

### Invariants

- **INV-SCH-1.** The schedule is a function of the constraint system only; prover and verifier derive it independently and must agree.
- **INV-SCH-2.** Every virtual claim is consumed by exactly one relation instance in the next stage.
- **INV-SCH-3.** Every claim produced at a committed polynomial reaches the opening pool.
- **INV-SCH-4.** The number of stages does not depend on `n_layers`, sequence length or batch size.

## Alternatives Considered

1. **Layer-by-layer GKR** (zkLLM, classic GKR). Rejected: proof depth grows with the 126 layers, each layer's challenges differ, and weight claims land on 126 different points per type.
2. **Commit every intermediate** (VerInf, DeepProve). Rejected: the cost driver the design exists to remove (source §4).
3. **Independent challenges per layer within a stage.** Rejected: loses the common point per weight type and multiplies weight openings by the layer count.
4. **One monolithic sumcheck over all relations.** Rejected: relations depend on claims produced by earlier relations, so a single round sequence cannot serve them all.
5. **Sending full round-polynomial coefficients.** Rejected: one redundant element per round; the derived `p(1)` form is standard.

## Drawbacks

- Shared challenges couple all layers: the prover must hold or regenerate every layer's evaluation tables for the current round, which is why stages recompute witness values on tensor cores.
- Front-padding wastes rounds for small instances in large stages.

## Migration / Rollout

The sumcheck engine, matmul relation and product tree land in v0.1 (needed for unit-cost measurement); the full schedule lands in v0.3 with the CPU prover and verifier.

## Testing Strategy

- Each relation kind: honest proofs verify for random instances with 1–24 variables; flipping any round evaluation or final evaluation is rejected.
- Batching: stages mixing instances of 3, 7 and 12 variables verify; dropping the `2^{n_max − n_i}` scaling is rejected.
- Schedule derivation: prover and verifier schedules (stage count, relation kinds, variable counts) are equal on all conformance models.
- INV-SCH-4: stage count equal for tiny models with 2, 4 and 8 layers of the same template, and in the 405B-shaped commitment report.
- Common point: after the `MatMulWeight` stage, each weight type has exactly one claim, at a point whose layer coordinate is shared by all stacked tensors of that stage (asserted in tests).
- Soundness calibration over `F65537`: a prover lying about one round succeeds at the rate `d/|F|` within confidence bounds.

## Open Questions

None.

## References

- Source §6 principle 8, §7 ("Constant-depth proof", "Weights opened by the folding pass").
- Goldwasser, Kalai, Rothblum, STOC 2008; Thaler, CRYPTO 2013 (matmul sumcheck); Lund, Fortnow, Karloff, Nisan, 1992 (sumcheck).
- [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md).
