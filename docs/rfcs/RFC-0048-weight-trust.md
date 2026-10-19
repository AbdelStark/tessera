# RFC-0048: What the Weights Root Vouches For

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-19
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [00 §3](../spec/00-overview.md#3-the-statement) 'The statement': the assumption on `R_W`;
  - [06 §5](../spec/06-security.md#5-threats-and-mitigations) 'Threats and mitigations': committed values outside their domains;
  - [RFC-0009](RFC-0009-gadgets.md#proposed-design) 'Proposed Design': what the requantization lemmas take from the class domain;
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md) 'Verifier': what the trusted root covers.
- Origin: [#369](https://github.com/AbdelStark/tessera/issues/369)

## Summary

The prover commits the weights and the `ChannelParams`, and the verifier sees only `R_W`. RFC-0006's range analysis assumes:
- weights in `[−127, 127]`;
- classes in `[0, n)`;
- multipliers in `[1, 2^{16})`;
- padding positions equal to 0.

No document said who guarantees them. This RFC separates two properties:
- **Uniqueness (SP-3)** holds for arbitrary committed values. Every gadget's uniqueness argument rests on a box of digits that maps injectively below `P`, or on a table being a function of its key, never on an input's range. A malformed commitment cannot let the prover adapt outputs request by request.
- **Fidelity and totality** need the domains. Fidelity means `D` computes the Llama its claim list describes; totality means every input has a proof. They are an assumption on `R_W`, which `tessera commit-weights` discharges for the party that publishes it.

## Motivation

#118's review showed a class `c = 1 + 3·2^{−6}` passing a `Requant1` lookup. It reads the row of class 1 at `z + 3`, and the output differs from class 1's at 24 of 64 keys. The question was whether such a commitment breaks soundness, or only changes the model. The answer decides between:
- a stated assumption;
- a one-time proof of every committed weight's domain, about `4·10^{11}` range lookups at 405B.

## Goals

- State what the verifier's guarantee does not depend on: uniqueness, for any committed values.
- State the assumption the remaining properties rest on, and who discharges it.

## Non-Goals

- A proof of the committed weights' domains: it buys no SP-3 property.
- The uniqueness conditions found beside this question, which are not about weights. Each has its own issue:
  - `SoftmaxRow`'s reciprocal needs a bound on `σ` (#376);
  - `RsqrtRow`'s limbs need a box below `P` (#377);
  - token ids need to lie in `[0, vocab)` (#378).

## Proposed Design

### Uniqueness for arbitrary committed values

A malformed `R_W` makes the virtual tensors downstream of the weights arbitrary field elements: matmul accumulators, `Rotate` outputs, affine keys, the first checkpoint, norm sums and logits. Every other tensor is a polynomial in lookup outputs with public coefficients, so it stays in its range whatever `R_W` is. Each gadget stays unique:

| Gadget | Why it is unique for any committed values |
|---|---|
| `Requant1`, `Silu`, `Exp` | The key is one field element. Table keys are distinct integers below `2^{32} < P`, so at most one row passes, and its value is fixed. |
| `RequantSplit` | Two passing pairs give congruent integers in `[0, n·2^b)`, which is below `P` (RFC-0009 Lemma 2), so they are equal, and so are their digits. |
| `RequantWide` | The parts form a box `[0, 2^b) ⊂ [0, P)` with `b ≤ 62`. The class enters only lookup 5's key. |
| `NonNeg`, `BigIntCompare` | Limb boxes below `P`, with `k ≤ 62`. |
| `SoftmaxRow`'s maximum (Lemma 5) | The tree forces `c` to equal some score. Two candidates differ by `d` with `d` and `−d` both in `[0, 2^{key_bits})`, so `d = 0`. |
| `OneHot` | Booleanity and the sum leave one 1, and `Σ v·h_v = t` fixes its index. |

**Padding inside committed rows** is also safe:
- A padded weight multiplies an activation that a lookup or `OneHot` pins, and every lookup block covers its tensor's padded positions (RFC-0045).
- A wholly padded row or layer is structurally zero and public.
- Values at padded positions change which function `R_W` names, never how many witnesses pass.

### The assumption

`R_W` is trusted to be the root of a `D` that `tessera commit-weights` accepted (#102). That is a `D` whose:
- weights lie in `[−127, 127]`;
- classes lie in `[0, n)` for their site's `n`;
- multipliers lie in `[1, 2^{16})`;
- padding positions hold 0;
- layer count is at least 1.

**Who discharges it.** The party that publishes `R_W` (03 §5, RFC-0040).
- A verifier that holds `D`, such as an auditor, recomputes `R_W` with its own code and discharges it itself.
- A verifier that does not hold `D` takes it from the publisher, as it takes the claim that `D` is the model it wants audited.

**What a malformed `R_W` can do, and what it cannot.**
- **It cannot adapt outputs.** Uniqueness holds, so every request's logits are one fixed function of `R_W` and the request.
- **It can name a function that is not the claim list's Llama.** The verifier cannot see the weights, so it cannot tell a malformed model from a well-formed one of poor quality either.
- **It can make inputs unprovable.**
  - A key outside every table has no witness.
  - An unproven sampled request is reported as unproven ([RFC-0016](RFC-0016-audit-sampling.md)). Under the assumption, that is the prover's fault.

### The lemmas' two halves

RFC-0009's requantization lemmas (1, 2 and 11) take the class domain only for their value half. Each now states two halves:
- **Uniqueness:** for any field `z` and `c`, at most one witness passes.
- **Value:** for `c ∈ [0, n)` and `z` in its range, the witness's `y` is `rq(z, r_c)`.

## Alternatives Considered

1. **A one-time well-formedness proof per weight epoch.** It would make a range lookup on every committed weight, class and multiplier, and a zero-check on padding. That is about `4·10^{11}` lookups at 405B. It buys fidelity without trusting the publisher, but no uniqueness, which already holds. It can be added later as an optional audit of `R_W` without changing any proof.
2. **In-circuit range checks per request.** These cost the same lookups on every proof, for the same fidelity.

## Drawbacks

- A verifier that trusts neither the publisher nor its own copy of `D` gets uniqueness and nothing about fidelity.

## Migration / Rollout

v0.3, for #369: this RFC and its amendments. `tessera commit-weights` already enforces the domains (#102). The verifier changes nothing.

## Testing Strategy

- **Uniqueness under arbitrary values.** For each implemented gadget, Goldilocks probes with arbitrary keys and classes find at most one passing witness. The `Requant1` example's class `1 + 3·2^{−6}` passes with exactly one `y`.
- **The assumption's enforcement.** `commit-weights` refuses a weight of −128, a class at `n`, a multiplier of 0 and zero layers (#102's tests).

## Open Questions

None.

## References

- [00 §3](../spec/00-overview.md#3-the-statement), [06](../spec/06-security.md), [RFC-0006](RFC-0006-compiler-ir.md#range-analysis), [RFC-0009](RFC-0009-gadgets.md), [RFC-0016](RFC-0016-audit-sampling.md), [RFC-0024](RFC-0024-wide-requantization.md), [RFC-0030](RFC-0030-masked-exponentials.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0045](RFC-0045-leaf-layout.md).
