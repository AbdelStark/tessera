# RFC-0024: Wide Requantization: Keys Too Wide for Their Shift

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0005](RFC-0005-integer-model.md#profiles) 'Profiles': `minimal`'s down-projection residual;
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': the refusal of keys that `RequantSplit` cannot take;
  - [RFC-0009](RFC-0009-gadgets.md#summary-of-costs) 'Summary of costs', which gains `RequantWide`;
  - [00 §9](../spec/00-overview.md#9-risks) RISK-2: the `minimal` profile's upper bound, and the projections that follow from it.
- Origin: [#292](https://github.com/AbdelStark/tessera/issues/292)

## Summary

`RequantSplit` takes a requantization key wider than one table by committing its low bits, but only while the split stays below every class's rounding bit. `minimal`'s down projection breaks that: its residual key reaches about `2^{54}` at 8B and `2^{58}` at 405B, against shifts near 17, so `minimal` does not compile at Llama's sizes. This RFC adds `RequantWide`. It splits below the rounding bit as far as it may, keeps a middle part as wide as the output's range needs, and reduces the top part to its saturation, `clamp(h1, −2, 1)`. The cost is four commitments and five lookups per element, into existing table kinds. With it, `minimal` compiles at 8B and 405B, and its projected `μ(1K)` rises from 9.4 to 10.0.

## Motivation

[RFC-0009](RFC-0009-gadgets.md#requantsplit)'s `RequantSplit` computes `y = clamp(⌊(z + 2^{r−1}) / 2^r⌋, lo, hi)` for a key `z` of `b` bits. It commits the low `s = b + ⌈log2 n⌉ − B_max` bits, `n` the number of shift classes, and looks up the rest in a table of `B_max` bits. The rest's floor is the requantization only when `2^{r_c − 1}` is a multiple of `2^s`, so Lemma 2 needs `s ≤ r_c − 1` for every class. #80's selection enforces it.

`minimal` reads `a · u` unrequantized in the down projection ([RFC-0005](RFC-0005-integer-model.md#profiles)). The accumulator's public bound is then `127 · d_ff · 128 · (127 · 128 · d)`, and the residual update takes it whole ([RFC-0022](RFC-0022-saturating-residual.md#saturating-residual)). The compiler's range analysis gives:

| Model | Key bits `b` | Split `s` at `B_max = 32`, one class | Shift it needs |
|---|---|---|---|
| 8B | 55 | 23 | `r ≥ 24` |
| 405B | 59 | 27 | `r ≥ 28` |

The residual site's shift follows from the scales, `r = e_res − e_acc` with `e_acc = e_a + e_y + e_up + e_down,c`. #292's illustrative exponents give about 17, far below either. No choice of scale closes the gap:

- a coarser residual loses precision at every layer;
- the weights' exponents are already the finest that hold them ([RFC-0005](RFC-0005-integer-model.md#scale-policy)).

The bound is a worst case, but the circuit must give the one right output for every key inside it. Beyond about `2^{r+19}`, that output is the saturated one.

## Goals

- A gadget for every requantization key of at most 62 bits whose tables fit `B_max`, whatever its shift.
- One passing witness per key, by a lemma in the form of RFC-0009's.
- Existing table kinds only: no new kind, schema or transcript label.
- Cost only where `RequantSplit` cannot go.

## Non-Goals

- A change to `minimal`'s dataflow or to RFC-0005's semantics.
- Keys over 62 bits, which no gadget keeps as true integers.
- A change to `B_max`.

## Proposed Design

### RequantWide

Inputs as `Requant1`: a virtual key `z`, a true integer of `b ≤ 62` signed bits, and the hidden class `c` of a site whose shifts lie between `r_min` and `r_max`, with bounds `lo ≤ hi`. Output: committed `y`.

Three public parameters come from the key's range and the site:

- the low part, below every class's rounding bit: `s = min(r_min − 1, B_max)`;
- the middle part, past which every class saturates: `t = (r_max − s) + ⌈log2 M⌉`, where `M = max(|lo|, |hi|, 1)`;
- the top part: `k = b − s − t`.

Committed values: `z_lo` when `s ≥ 1`, `h0`, `g` and `y`. Virtual values:

- `z_off = z + 2^{b−1}`;
- `w_off = (z_off − z_lo) · (2^s)^{−1}`;
- `h1_off = (w_off − h0) · (2^t)^{−1}`, and `h1 = h1_off − 2^{k−1}`;
- `w' = h0 + 2^t · g`.

Lookups:

1. `z_lo ∈ [0, 2^s)`, into `Range { s }`; absent when `s = 0`.
2. `h0 ∈ [0, 2^t)`, into `Range { t }`.
3. `h1_off ∈ [0, 2^k)`, into `Range { k }`.
4. `(pack(0, 2·h1), g) ∈ Requant { k + 1, [1], −2, 1 }`, so `g = clamp(h1, −2, 1)`. The key `pack(0, 2·h1)` is `2·h1_off`.
5. `(pack(c, w'), y) ∈ Requant { t + 2, r_c − s, lo, hi }`.

Cost: 4 commitments and 5 lookups per element, or 3 and 4 when `s = 0`.

Preconditions, checked at gadget selection and again by the verifier's validation:

- `b ≤ 62`;
- `t + 2 + ⌈log2 n⌉ ≤ B_max`;
- `k + 1 ≤ B_max`.

When `RequantSplit`'s conditions fail and these hold, `k ≥ 3`.

**Lemma 11.** Given `z` and `c`, exactly one witness passes, and its output is `y = clamp(⌊(z + 2^{r_c − 1}) / 2^{r_c}⌋, lo, hi)`.

*Proof of uniqueness.* Lookups 1 to 3 make `z_lo`, `h0` and `h1_off` integers in `[0, 2^s)`, `[0, 2^t)` and `[0, 2^k)`. The virtual definitions give `2^{s+t} · h1_off + 2^s · h0 + z_lo ≡ z_off (mod P)`. Both sides lie in `[0, 2^b)` with `b ≤ 62`, so by Lemma 0 they are equal over the integers. The three values are therefore the digits of `z_off` in the mixed radix `(2^s, 2^t)`, and they are unique. Lookups 4 and 5 are functions of their keys, so `g` and `y` are unique too.

*Proof of the value.* Let `w = ⌊z / 2^s⌋ = 2^t · h1 + h0`; this uses that `2^{b−1}` is a multiple of `2^{s+t}`, since `k ≥ 1`. Each class has `s ≤ r_c − 1`, so nested floor division gives `⌊(z + 2^{r_c−1}) / 2^{r_c}⌋ = q(w)`, as in Lemma 2. Here `q(x) = ⌊(x + 2^{r'−1}) / 2^{r'}⌋` with `r' = r_c − s ≤ r_max − s`, and `q` is nondecreasing.

- **`−2 ≤ h1 ≤ 1`:** `g = h1` and `w' = w`.
- **`h1 ≥ 2`:** `w ≥ 2^{t+1}`, and `g = 1` gives `w' = h0 + 2^t ≥ 2^t`. Both `q(w)` and `q(w')` are at least `q(2^t) = 2^{t−r'} ≥ 2^{⌈log2 M⌉} ≥ hi`, so both clamp to `hi`.
- **`h1 ≤ −3`:** `w ≤ −2^{t+1} − 1`, and `g = −2` gives `w' = h0 − 2^{t+1} ≤ −2^t − 1`. Both are at most `q(−2^t − 1) = −2^{t−r'} ≤ −M ≤ lo`, so both clamp to `lo`.

In every case lookup 5 returns `clamp(q(w), lo, hi)`. ∎

**Why lookup 3.** Lookup 4 alone pins `2·h1_off`, not `h1_off`, to a key of its table. That table also holds the odd keys `2m + 1`, which the field element `h1_off = m + 1/2` reaches.

- Suppose `h0 ≥ 2^{t−1}`. Then `h0 − 2^{t−1}` with such a half-integer passes lookups 1, 2, 4 and 5, and returns another `y`.
- Lookup 3 makes `h1_off` an integer.
- An exhaustive search over `F_65537` checked five configurations, with one, two and four classes, symmetric and asymmetric bounds, and `s = 0`. Every key and class was tried. Without lookup 3, every configuration had alternative outputs; with it, none did.

### Selection

[RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection' takes `RequantWide` after `RequantSplit`. Of the gadgets that apply, the first is the cheapest:

| Condition | Gadget | Tables | Cost per element |
|---|---|---|---|
| `b + ⌈log2 n⌉ ≤ B_max` | `Requant1` | `Requant { b, r_c, lo, hi }` | 1 commitment, 1 lookup |
| otherwise, `s = b + ⌈log2 n⌉ − B_max`, with `s ≤ B_max`, `b ≤ 62` and `s ≤ r_c − 1` for every class | `RequantSplit` | `Range { s }`, `Requant { b − s, r_c − s, lo, hi }` | 2 commitments, 2 lookups |
| otherwise, with `b ≤ 62`, `t + 2 + ⌈log2 n⌉ ≤ B_max` and `k + 1 ≤ B_max` | `RequantWide` | `Range { s }` when `s ≥ 1`, `Range { t }`, `Range { k }`, `Requant { k + 1, [1], −2, 1 }`, `Requant { t + 2, r_c − s, lo, hi }` | 4 commitments, 5 lookups |
| otherwise | `ERR-COMPILE-RANGE` | — | — |

The verifier selects by the same rule from public data ([RFC-0007](RFC-0007-claim-list.md#verifier-validation), step 5). The claim list's tables include the new gadget's, all of existing kinds, so no schema changes.

### At Llama's shapes

The compiler's selection gives the following for `minimal`'s down-projection residual, at the test shapes' residual shift of 17. With one class, the fused key saturates at INT20. With four classes, RFC-0022's update `u = rq(acc, r_c, ±(2^{20} − 1))` saturates.

| Model | Classes | `b` | `s` | `t` | `k` | Largest table |
|---|---|---|---|---|---|---|
| 8B | 1 | 55 | 16 | 20 | 19 | the reduced `Requant { 22, [1], INT20 }`: `2^{22}` rows |
| 8B | 4 | 55 | 16 | 24 | 15 | the reduced `Requant { 26, r_c − 16, ±(2^{20} − 1) }`: `2^{28}` rows |
| 405B | 1 | 59 | 16 | 20 | 23 | the saturation's `Requant { 24, [1], −2, 1 }`: `2^{24}` rows |
| 405B | 4 | 59 | 16 | 24 | 19 | the reduced table, as at 8B: `2^{28}` rows |

Every other site keeps its gadget. With one class, the middle `Range { 20 }` is the norms' split table, shared.

### Cost

`RequantWide` costs two commitments and three lookups more than `RequantSplit` would. In `minimal`, only the down projection's residual takes it: at 405B, 16,384 elements per layer per token.

RISK-2's upper bound for `minimal` therefore rises. It was 220,160 commitments and 220,160 lookups per layer per token. It is now 252,928 commitments and 269,312 lookups. The executable cost model ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)) computes the corrected 405B projections at both counts:

| `minimal`, 405B | Before | After |
|---|---|---|
| Linear-layer term at 1K | 3.3 | 3.9 |
| `μ(1K)`, `μ(16K)` | 9.4, 24.9 | 10.0, 25.4 |
| The same, with the column answers priced as committing | 11.0, 31.4 | 11.8, 32.1 |
| 16K, adding RISK-10's upper streaming estimate | 43 | 44 |

`minimal` remains the cheaper profile: `servable` gives 11.0 and 26.3. Under SC-2's 1K ceiling of 12, with the column answers priced as committing, its margin shrinks from 8% to 2%.

### Changes to other documents

- RFC-0006 'Gadget selection': the table above, and its text.
- RFC-0009: the section 'RequantWide' with Lemma 11, 'Summary of costs' and 'Testing Strategy'.
- RFC-0005 'Profiles': `minimal`'s down-projection residual takes `RequantWide`.
- 00 §7 SC-2, 00 §9 RISK-2 and RISK-9, 08 §1 and §3, and the summary in `SPEC.md`: `minimal`'s count and projections.
- The glossary.

## Alternatives Considered

1. **Requantize `a · u` in `minimal`**, as `servable` does.
   - The product's key is about `2^{33}` at 8B and `2^{35}` at 405B, so each element takes `RequantSplit`: two commitments per element of `d_ff`. The residual's key then fits one lookup.
   - At 405B that is 122,880 commitments per layer per token, against `RequantWide`'s 65,536 at the residual.
   - It also changes `D`'s semantics by a rounding.
2. **Accept that `minimal` does not compile at Llama's sizes**, and keep it as a profile for counting only. But its count is the reason it exists.
3. **A `Clamp` table kind**, `key ↦ clamp(key, lo, hi)`, would do the work of lookups 3 and 4 in one. That saves one lookup per element, 0.14 of `μ(1K)` at 405B by the executable model. A new kind needs generators in both workspaces, an oracle and vectors, so it can follow if a measured profile shows the lookup matters.
4. **A split at the rounding bit with limbs**, as #292 sketched. It would commit `z mod 2^{r−1}`, then certify the clamp of the remaining high part, about 39 bits, with `NonNeg` limbs and case indicators. That costs as much or more, and it needs a degree-2 selection. The saturation by `clamp(h1, −2, 1)` needs neither.

## Drawbacks

- The site costs four commitments and five lookups per element, twice `RequantSplit`'s, and `minimal`'s projected `μ(1K)` rises by about 0.6.
- The verifier's selection gains a third requantization gadget.
- With four classes, the reduced table has `2^{28}` rows.

## Migration / Rollout

The compiler's selection takes the gadget in v0.2. Its constraints land in v0.3 with `Requant1` and `RequantSplit` ([#118](https://github.com/AbdelStark/tessera/issues/118)). No proof has been produced, so no claim list changes meaning.

## Testing Strategy

- **Selection** (`crates/tessera-compiler/tests/select.rs`):
  - the boundary between `RequantSplit` and `RequantWide`, at the rounding bit and at the low part's `B_max`, with the tables on each side;
  - each precondition, refused at its boundary;
  - `minimal` 8B and 405B with one class and with four.
- **A property test.** The five lookups return `clamp(rq(z, r_c), lo, hi)` for every class. The keys span every width the gadget takes, including both ends of the range, and `lo` and `hi` are drawn independently.
- **With #118:**
  - completeness on 10^4 random keys, including both ends;
  - Lemma 11 by exhaustion for `b ≤ 14` over `F_65537`: only the true digits pass;
  - the odd-key witness of 'Why lookup 3', rejected;
  - without lookup 3, the same witness accepted, which shows the lookup is necessary (RFC-0009 'Testing Strategy').

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0020](RFC-0020-cost-model.md), [RFC-0022](RFC-0022-saturating-residual.md).
- [#292](https://github.com/AbdelStark/tessera/issues/292).
