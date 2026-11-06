# IDEA-0043: Exact structured bottom levels for Range table fraction trees

| | |
|---|---|
| Status | `validated` |
| Origin | [#691](https://github.com/AbdelStark/tessera/issues/691); [SUR-0017](../surveys/SUR-0017-structured-table-fractions.md) candidate C0; EXP-0030 real 1B geometry |
| Lever | `regen` on the table side of LogUp (leaf scans of generated `Range` trees); no relation, transcript, parameter or opening change |
| Spec surfaces | None normative. RFC-0008 'Fraction tree' and RFC-0010 'Tree levels' fix the messages, and this changes only how the prover computes them. INV-ARCH-2 (01 §6) is respected as written. RFC-0018 'Memory scheduler' must charge the new scratch. No tier-A change. |
| Priority | Performance, after the correctness prerequisites. E0 expectation: about `4(h − c)` leaf scans become about one on every Range tree. Range is 33.7% of the real 1B table rows (SUR-0017), so table-side work falls by at most that share. Numerical score deferred until EXP-0032 counts it. |

## Mechanism

**The structure.** Take a `Range{bits = h}` table tree: `2^h` leaves, leaf `j` is `(−m_j, α − j)`, and there is no padding because `T = 2^h`. At level `t + 1` with block size `B = 2^{h−t−1}`:

- `D_{t+1}[e] = P_B(α − B·e)`, where `P_B(z) = Π_{s<B}(z − s)`;
- `N_{t+1}[e] = −Σ_{i: ⌊a_i/B⌋ = e} Π_{s∈[eB,(e+1)B), s≠a_i}(α − s)`;

where `a_i` runs over the instance's lookup column, its `2^{h_a}` entries including the row-0 padding. That is RFC-0008's canonical `m_j`, counted lookup by lookup.

**Upper levels `t < c` are unchanged.** They keep the existing dense path, with the cache at depth `c = cache_depth(h, L)` built by one public-index leaf scan.

**Bottom levels `t ≥ c` are structured.** Level `t`'s sumcheck (RFC-0010 step 2, the fraction-tree summand `g = N_0 D_1 + N_1 D_0 + λ D_0 D_1`, pair bit lowest, variables bound lowest first) is computed from three objects, never from the `2^{t+1}`-entry level:

1. **Denominator polynomials.** For `b ∈ {0, 1}`, `H_b(y) = P_B(α − B(2y + b))`, degree `B` in the node index `y`. Binding a variable to `ρ` maps `H ↦ (1 − ρ)·H(2u) + ρ·H(2u + 1)`, a classical division-free Taylor shift ([vzGG97]).
2. **Suffix moments.** `M_k = Σ_{u ∈ {0,1}^m} eq(r_{>j}, u)·u^k` for `k ≤ 2B`, by Pascal's rule bit by bit. The `λ D_0 D_1` part of round `j` at `X` is the dot product of `(H_0^X·H_1^X)`'s coefficients with `M`, times the bound prefix `Π_{i<j} eq(r_i, ρ_i)·eq(r_j, X)`.
3. **Fixed-capacity numerator entries.** One entry per lookup slot: its pair bit, its node position and its weight `−Π_{s≠a_i}(α − s)`, the product taken over its block without division.
   - In each round an entry contributes `weight · (X or 1 − X) · eq(r_{>j}, position) · H_{1−b}^X(position)`.
   - Binding multiplies the weight by `ρ` or `1 − ρ` and shifts the position.
   - Every select is arithmetic (`b·x + (1 − b)·y`), the loop count is the public `2^{h_a}`, and no memory address depends on a key.

**Level evaluations.** The `tree/{ℓ}/evals` values `N(0‖r'), N(1‖r'), D(0‖r'), D(1‖r')` are the entries' summed weights per pair bit and the constants `H_b(0)` after the last binding. The verifier's leaf check `q = α − t̃(ρ)` is unchanged.

**Choosing the route.** Range trees with `T ≥ 2^{h_min}` take the structured route. The choice uses public data only (`h`, `h_a`, `c`, the allowance), and the generic `FractionTreeRows` path remains the reference.

## Why it could work

- **Exactness is algebraic.** Every message is a fixed field value, so a correct evaluator is byte-identical. A private 168-case scratch check (heights 1–7, every level, α inside the table, no lookups, duplicate keys) matched a dense reference exactly. That is ideation evidence, not EXP-0032's registered result.
- **The redundancy is large and public.** The current path costs `1 + Σ_{t=c}^{h−1} 4·(1 + c_t) ≥ 1 + 4(h − c)` leaf scans per tree (SUR-0017, E0 code reading). On the real candidate (EXP-0030, E0):

  | Table | Height `h` | Lookup entries | Scans at `c = 20` |
  |---|---:|---:|---:|
  | `Range{30}` | 30 | 66 | ≥ 41 |
  | `Range{29}` | 29 | 81,920 | ≥ 37 |
  | `Range{27}` | 27 | 528 | ≥ 29 |
  | `Range{26}` | 26 | 131,336 | ≥ 25 |
  | `Range{25}` | 25 | 131,336 | ≥ 21 |

- **The bottom levels are cheap in the structured form.** Their block sizes are at most `2^{h−c−1}` (512 at `h = 30`), so the polynomial and moment work per round is small, and the entry work is `O(t·2^{h_a})` per level.

## Expected gain

The cost model has no table-side term. The registered experiment counts field multiplications instead: baseline from the code's scan formula with `c_t = 0` (its most favourable case), candidate from the harness's own instrumented operation counts, both on the 21 real Range instances. No time, RSS or GPU conversion is credited. Whole-proof effect: at most the Range share of table-side work, until P2 is profiled on the real model.

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K–128K | not modelled (table side outside `μ`) | 0 | 0 |

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None: identical messages, challenges and checks | pass |
| Uniqueness upstream of the logits (SP-3) | None: the witness and multiplicities are unchanged | pass |
| Zero knowledge (G-5, RFC-0013) | None in `tessera-protocol/1`. Protocol/2's masked tree levels are out of scope, and the structured route would need its masked terms added. | pass (scope: protocol/1) |
| INV-ARCH-2 / OBS-6 (secret handling) | Control flow and addresses public. Entry weights and positions are secret-derived and live in clearing storage. Arithmetic selects only. | pass, to be checked by the registered fault "branch on pair bit" being absent in the harness and by code review |
| Verifier independence (SP-6) | Prover-only | pass |
| Post-quantum | Unchanged | pass |

## Prior rejection check

- No RFC's Alternatives Considered or `baseline.md` "Already rejected" row covers structured table-side evaluation.
- RFC-0018's bounded regeneration and RFC-0054 groups bound memory, not leaf-scan count.
- #663 (PR #664) optimized the per-scan constant: aligned subtree compaction, 0.67–0.71× per scan, E1 synthetic. It did not change the number of scans.

## Kill criterion

Any mismatch, in any registered case, between the candidate's round messages, evaluations or roots and the dense reference. Or a counted candidate cost for the real Range instances above half the conservative baseline.

## Cheapest experiment

[EXP-0032](../experiments/EXP-0032-structured-range-fractions/README.md): E0, Python oracle over Goldilocks `Fp2`, effort `m`.

- **Exactness.** Every level of every registered small case against a dense reference.
- **Independent check.** Of each produced level transcript.
- **Faults.** Seven registered faults, each to be caught.
- **Counts.** Closed-form operation counts, validated against instrumented counts on small cases and then evaluated on the real geometry.

Unknown until later: wall time in Rust, the production integration (#641), the valued tables (V1), and any GPU figure.

## Refutations

None.

[EXP-0032](../experiments/EXP-0032-structured-range-fractions/README.md) validated the idea at E0 on 2026-11-06:

- exact messages on 8,736 levels;
- all seven faults caught;
- a summed counted ratio of 0.0948 on the 21 real Range instances, and 0.0795 when the cheaper route is chosen per tree.

**Amendment the experiment forces.** The structured route loses on small trees with many lookups, by up to 4,360× on `Range{6}`. The route must be chosen per tree by comparing the two public closed forms, not by a height threshold alone.
