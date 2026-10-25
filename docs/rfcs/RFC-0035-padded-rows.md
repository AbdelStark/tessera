# RFC-0035: Padded Rows Satisfy Every Row Gadget

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Amended by: [RFC-0058](RFC-0058-inverse-cdf-columns.md#padded-rows) ('InverseCdf's padding zero-check pins its partition `Z` too), [RFC-0066](RFC-0066-bits-and-sha256.md) ('Zero-check points': its zero-checks run in stage 1)
- Supersedes, on acceptance:
  - [RFC-0009](RFC-0009-gadgets.md) 'RsqrtRow', 'SoftmaxRow' relation 3, 'OneHot' and 'InverseCdf': the row selector `ψ`;
  - [RFC-0017](RFC-0017-scoring.md#in-circuit-scoring-mixture) 'In-circuit scoring', constraints 3 and 4: the row selector `ψ`;
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations' and 'Batching within a stage': the padding zero-check, and the point every `ZeroCheck` takes;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule', row 6: `stage/{k}/zero/{c}`;
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout', rules 1 and 4: what padded rows hold, and which rows are skipped.
- Origin: [#322](https://github.com/AbdelStark/tessera/issues/322)

## Summary

RFC-0011 pads every dimension to a power of two with zeros. A sequence's length and a model's layer count rarely are powers of two, so padded rows exist in every tensor indexed by token, position or layer. Five row relations had no satisfying witness on such a row, so the honest prover could not complete a proof for most sequences:

- `RsqrtRow`;
- `SoftmaxRow`'s bracket;
- `OneHot`;
- `InverseCdf`;
- the scoring bound of RFC-0017.

This RFC keeps padded rows as they are and makes every row relation accept them:

- **The row selector.** `ψ` is 1 on a real row and 0 on a padded one. It is public, and its extension is a product of `LT` predicates.
- **The relations.** Each relation that a padded row fails multiplies its constants by `ψ`. A padded row is then the relation's instance at zero input, which the padded row's witness satisfies.
- **Padding zero-checks.** Where that instance leaves a committed value free, a zero-check `(1 − ψ)·w = 0` pins it to 0, so every committed value upstream of the logits keeps exactly one passing witness. The values are `RsqrtRow`'s root, `SoftmaxRow`'s maximum and reciprocal, and `InverseCdf`'s vocabulary maximum. [RFC-0058](RFC-0058-inverse-cdf-columns.md#padded-rows) adds `InverseCdf`'s virtual partition `Z`.
- **Zero-check points.** Each stage draws one point for its `ZeroCheck` instances before `γ`. That also fixes where `OneHot`'s booleanity check takes its point, which no document said.

No commitment or lookup is added, only four small zero-check instances, and RFC-0058's fifth.

## Motivation

#322 found the failures while resolving #279. On an all-zero row:

- **`RsqrtRow`.** `u = Σ res² + e = e` with `ρ = 0`, so the bracket `N + 1 ≤ (ρ + 1)²·u` needs `e ≥ N + 1`, which is false. The limbs of `u`, zero on a padded row, do not recompose to `e` either.
- **`SoftmaxRow`.** `s_q = 0`, so `b_q = s_q − 1 − a_q = −1 − 2^{2σ}`, which `NonNeg` refuses. The slack `a_q = 2^{2σ}` does not match its zero limbs.
- **`OneHot`.** The zero vector fails `Σ_v h_v = 1`.
- **`InverseCdf`.** Its vocabulary certificate looks up `Exp_T` at key 0, whose value is not 0. Its second comparison needs `1 ≤ 0`.
- **Scoring bound.** RFC-0017's constraint 3 needs `2^{16} ≥ V·2^{16}`, which is false. Its `b_t` would add the constant `k` to `Û` for every padded row.

#322 weighed three options:

1. **Dummy positions.** Compute padded rows as tokens of the sequence. This costs up to twice the work for a length just above a power of two.
2. **Row selectors.** This RFC's choice.
3. **Padding by whole requests.** This does not remove the padding within a sequence.

## Proposed Design

### The row selector

A gadget's rows are indexed by layer, by head where its instances are per head, and by token or position within one sequence of `n` positions. The row selector is

```text
ψ(ℓ, h, t) = [ℓ < n_layers] · [h < n_heads] · [t < n],        ψ̃ = LT(r_ℓ, n_layers) · LT(r_h, n_heads) · LT(r_t, n)
```

with the head factor only for per-head rows. `LT` has an `O(n)` closed form ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)), so the verifier evaluates `ψ̃` at any point in `O(log)` time. For the output gadgets, `Argmax` and `InverseCdf`, and for the scoring bound, `t` runs over the positions whose next token the statement records. On a real row `ψ = 1`, and every relation is the one RFC-0009 or RFC-0017 states.

### What a padded row holds

On a padded row, every gadget computes its witness from the row's inputs as on a real row, with `ψ = 0` in the relations below. The token is 0 and the activations are 0. A lookup at key 0 returns its table's value there, which is 0 for every table but three: the scoring bound's `Exp2Frac` and `Pow2`, and the AES S-box. The AES and SHA-256 circuits of RFC-0015 also add their own constants. RFC-0011 rule 4 skips a row only when every entry of it is 0, so a padded row that holds one of these values is committed. A padded row's witness is therefore determined, as a real row's is.

No real output reads a padded row:

- a real query attends only to keys `i ≤ q < n` ([RFC-0030](RFC-0030-masked-exponentials.md#the-causal-selector));
- every other gadget works row by row;
- the scoring bound sums `b_t`, which is 0 on a padded row.

### The relations

- **`OneHot`.** `Σ_v h_v = ψ_t`. On a padded row, booleanity and a zero sum leave `h = 0`, and `Σ_v v·h_v = t` gives `t = 0`. Lemma 8 holds on real rows unchanged.
- **`RsqrtRow`.** `u = ψ·(Σ_i res_i² + e)`, and the comparisons become `ρ²u ≤ ψ·N` and `ψ·(N + 1) ≤ (ρ + 1)²·u`.
  - On a padded row `u = 0`. Its range-checked limbs, a box below `P` (RFC-0009 'The limb box'), recompose to 0 only as zeros, and both comparisons are `0 ≤ 0`, whose `BigIntCompare` witnesses are zero (Lemma 4).
  - The root `ρ` is left free, and the padding zero-check pins it.
- **`SoftmaxRow`.** Relation 3 takes `a_q = ψ_q·2^{2σ} − R_q·s_q` and `b_q = s_q − ψ_q − a_q`.
  - On a padded row `s_q = 0`, so `a_q = b_q = 0` whatever `R_q`.
  - Relations 1 and 2 already mask a padded row: RFC-0030's selector holds `q < n`, so every key is `K_0`, every `e` is 0 and the tree's root is 1.
  - `c_q` and `R_q` are left free, and the padding zero-check pins them.
- **`InverseCdf`.**
  - Its vocabulary certificate masks a padded row as `SoftmaxRow` masks a padded query: keys `ψ·k_v + (1 − ψ)·K_0`, leaves `ψ·k_v + 1 − ψ`, root 1.
  - The second comparison becomes `u·Z + ψ ≤ C_≤ · 2^{32}`.
  - On a padded row `e = 0`, `Z = 0` and `C_≤ = C_< = 0`, so both comparisons are `0 ≤ 0`.
  - The certificate's maximum `c` is left free, and the padding zero-check pins it.
  - [RFC-0058](RFC-0058-inverse-cdf-columns.md#padded-rows) also pins the partition `Z`, so the gadget's witness on a padded row does not depend on the exponentials the certificate gives there.
- **`Argmax`** needs nothing: on a padded row `h = 0`, so `Σ_v h_v·ℓ_v = m` gives `m = 0`, and every key `g_v` is 0.
- **Scoring bound (RFC-0017).**
  - Constraint 3 takes `(e_{y_t} + ψ)·2^{I_t}·E2[f_t] ≥ ψ·(Z + V)·2^{16}`, and constraint 4 takes `b_t = ψ·(m_t·c_match + (1 − m_t)·(k + I_t + f_t/2^8))`.
  - On a padded row `b_t = 0`, so `Û` counts real tokens only.
  - `m_t`, `I_t` and `f_t` stay free there, as they are on real rows, where RFC-0017's one-sided freedom already allows the prover to choose them. They are downstream of the logits, and no choice moves `Û`.
- **Every other gadget** works element by element, and a padded row satisfies it as given.

### The padding zero-check

For each committed value `w` that the relations above leave free on a padded row (`RsqrtRow`'s `ρ`, `SoftmaxRow`'s `c_q` and `R_q`, `InverseCdf`'s `c`), stage 1 proves the `ZeroCheck`

```text
0 = Σ_x eq(z, x) · (1 − ψ(x)) · w̃(x)
```

The relation has degree 3, and the verifier evaluates its public factor `1 − ψ̃` at the stage's final point. The instance's claim on `w̃` joins the opening pool. [RFC-0058](RFC-0058-inverse-cdf-columns.md#padded-rows) adds the same check on `InverseCdf`'s partition `Z`, which is virtual: its claim `Z̃(r′)` is reduced by the partition's contraction `Σ_v C_v·e_v` in the next stage (RFC-0010 'Stages'), not opened. A non-zero `w` on a padded row makes the sum a non-zero multilinear polynomial in `z`, which vanishes with probability at most `n_w/|Fp2|`.

### Zero-check points

`RFC-0010` lists `ZeroCheck` as a relation kind, but no document said where its point `z` comes from. Each stage `k` that holds `ZeroCheck` instances draws one point just before `γ_k`, after `ρ_k` in `tessera-protocol/2`: coordinates `z_c` for `c = 0, 1, …` under `stage/{k}/zero/{c}`, one per variable of its largest `ZeroCheck`. An instance of `n_i` variables takes the last `n_i` coordinates, as forests take their root points ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)). `OneHot`'s booleanity check runs in stage 1 with the padding zero-checks, at the same point. So do [RFC-0066](RFC-0066-bits-and-sha256.md#where-the-zero-checks-run)'s zero-checks, after these, in node order: `BitDecompose`'s padding check of a committed input, booleanity, bit-padding and recomposition checks, and `AssertEqual`'s.

### Soundness

On a real row every relation is unchanged, and so is every lemma. A padded row's witness is determined, and no real output depends on it. Each `ZeroCheck` adds `n_i/|Fp2|` to its stage's error. The soundness calculator counts it ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-budget)).

### Changes to other documents

- **RFC-0009.** `RsqrtRow`, `SoftmaxRow` relation 3, `OneHot` and `InverseCdf`, as above.
- **RFC-0017.** Constraints 3 and 4.
- **RFC-0010.** The padding zero-check in 'Relations', and the zero-check point in 'Batching within a stage'.
- **RFC-0014.** Row 6 of the transcript schedule.
- **RFC-0011.** Rule 1 points here for what a padded row holds, and rule 4 skips a row only when all of it is 0.

## Alternatives Considered

1. **Dummy positions.** Compute padded rows as the sequence's next tokens. Every gadget is then complete without change. Rejected: up to twice the work for a length just above a power of two.
2. **Selectors that switch a padded row's relations off.** Multiplying each relation by `ψ` would make a padded row vacuous. The committed values there would then be free, with many passing witnesses where the uniqueness invariant asks for one, and `RsqrtRow`'s limbs among them. Rejected: zeroing the constants keeps a padded row's instance a real one, and the zero-check pins the few values it leaves free.
3. **A canonical witness for padded rows.** Commit, on a padded row, the witness of a real computation on fixed inputs. `OneHot` would then need a one-hot of some token, whose embedding is not 0. Every layer would compute on it, which is option 1 again.

## Drawbacks

- Every row relation that a padded row fails names `ψ`, one more public factor for the prover and verifier to evaluate.
- Four zero-check instances join stage 1, and RFC-0058 adds a fifth.

## Migration / Rollout

v0.3: the gadgets' relations (#119, #122, #123, #126), the stage schedule's zero-check point (#125), and the verifier's checks (#128).

## Testing Strategy

- **Completeness.** For each row gadget, an instance whose rows include padded ones verifies, with a padded row holding the witness this RFC defines.
- **Adversarial.** A non-zero `ρ`, `c_q`, `R_q` or `c`, or `InverseCdf`'s `Z` (RFC-0058), on a padded row is rejected by the padding zero-check. A non-zero `h` on a padded row is rejected by `OneHot`, and a non-zero `b_t` by the scoring bound.
- **End to end.** A sequence whose length is not a power of two proves and verifies, once the CPU prover exists (#133).

## Open Questions

None.

## References

- [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0017](RFC-0017-scoring.md), [RFC-0030](RFC-0030-masked-exponentials.md).
