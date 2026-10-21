# IDEA-0038: Original-source greedy boundary bindings

| | |
|---|---|
| Status | `triaged` |
| Origin | #128/#130 integration; RFC-0009/0017/0037; packet-only draft PR439 |
| Lever | Correctness of the first complete CPU proof; zero performance credit |
| Spec surfaces | Runtime scoring advice, implicit tables, native schedule and original linear opening claims; an RFC is required before production acceptance |
| Priority | 0 performance credit (E0); required correctness frontier |

## Mechanism

Keep the actual model logits `L[t,v]`, next-token OneHot `H[t+1,v]` and
committed maximum `M[t]` at their original identities. The maximum has the model
time domain and is zero outside output-logit positions. For a request with
`n=prompt+outputs`, those positions are `[prompt-1,n-1)`. Output lookup keys
retain the smaller padded `(output,vocabulary)` domain and mask both padding
axes. Never introduce a fake model maximum node or an externally supplied logit.

The time shift has an exact linear opening decomposition. For padded time width
`b`, cyclic `Hnext[t]=H[t+1 mod 2^b]` at point `r` is the sum over carry lengths:
`alpha_k=prod(i<k,r_i)*(1-r_k)`, source point `(0^k,1,r[k+1:])`, plus
`alpha_wrap=prod(r_i)` at the all-zero source point. Vocabulary coordinates are
unchanged. A whole linear claim retains one value and one PCS batching power
over its weighted original-source points; no prover-selected per-image values.
The active output mask excludes the cyclic wrap from every real certificate.

Use public multilinear coefficients to express the masked key's evaluation as
a linear functional of original `L`, `Hnext` and `M`. The selected-logit identity
is a zero-check on `sum_v Hnext[t,v]*L[t,v]=M[t]` at active times; its maximum
term also constrains all inactive/padded maximum entries to zero. A separate
weighted OneHot zero-check binds `sum_v v*H[t,v]` directly to the committed token
input. The candidate degrees are respectively two, three and two (E0); these
are candidate native relations, not a current full-protocol bound.

A public proved logit interval `[lo,hi]` permits physical Range width
`max(1,bit_length(hi-lo)) <= 29`: the selected-logit equality makes `m` an actual
logit, and `m-l-lt` lies in `[-(hi-lo)-1,hi-lo]`. A nonnegative valid key therefore
fits that width; a negative Goldilocks encoding cannot. The original logical
29-bit Argmax predicate and tie-breaking remain. No secret-observed range is used.

## Seeds and lens pass

| Lens / alternative | Decision |
|---|---|
| Lookup: materialize Range(29) | Correct comparison baseline; unsuitable as a tiny test's physical table |
| Lookup: public proved logit span | Candidate; no extra limb commitments, exact completeness test required |
| Lookup: NonNeg(29) low limbs | Defer; changes prescribed auxiliary columns and lookup counts |
| Max certificate: original-source linear key | Candidate; fixes every source identity before challenges |
| Schedule: separate output/suffix reductions | Valid alternative; introduces dependent stages absent from the current native interface |
| Counts: duplicate model-time/vocabulary axes | Defer; original tensor-product domains can exceed the existing variable cap and expand work |
| Commitment: weighted original-point shift | Candidate; preserve one whole claimed value across all PCS images |
| Commitment: commit another shifted OneHot | Defer; avoid another vocabulary-sized secret column |
| Soundness: substitute a zero logit or maximum | Reject; drops the actual model/greedy obligation |
| Parameters: weaken tiny ranges/presets | Reject; public bounds must follow the actual validated model |
| ZK: claim current protocol/1 is hiding | Reject; protocol/2 masking is a separate required gate |
| Hardware / wild paradigm | No GPU, new PCS or folding is needed for this bounded correctness question |

## Why it could work

[TesseraGreedy26] fixes the lowest-index maximum, the actual next-token OneHot,
linear suffix and sumcheck/PCS binding requirements (E0, primary). The increment
partitions the Boolean time cube by its first zero, giving the source points
above. Public selectors may be multilinearized before multiplication; their
Boolean values, source identities and per-variable degrees remain explicit.

## Expected gain

No mu, proof-size, verification-time or GPU credit at any context (E0 accounting
decision). The candidate removes missing bindings, not a measured bottleneck.
An extra linear-claim group must be charged and the complete dynamic soundness
budget checked before any accepting production path. No cost-model inputs change.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| G-4 soundness | Original-source identities and degrees must match; full composed budget remains untested | unknown; E0 first |
| SP-3 uniqueness | Selected equality fixes the maximum; explicit inactive/padded maximum zeros and OneHot token identity | unknown; exhaust adversarial boxes |
| G-5 zero knowledge | No protocol/1 hiding claim; all new views must enter the later protocol/2 mask accounting | unchanged requirement |
| SP-6 independence | Literal Python integer/cube oracle precedes independent Rust integration | pass by method |
| Post-quantum | Existing hash commitments only | unchanged |

## Prior rejection check

Baseline section 7 and RFC-0009/0011/0037 alternatives do not reject exact
original-point linear combinations. This is not a new PCS or permission to omit
source claims. RFC-0050 deliberately leaves scoring auxiliaries unspecified;
the runtime maximum/table/order contract still needs a superseding RFC.

## Kill criterion

Any mismatch in a public kernel, shifted opening, Boolean-cube identity,
derived-range completeness, tie-break rejection or declared per-variable degree
refutes the candidate as written. Any dropped original source or uncharged
acceptance predicate blocks promotion to an accepting implementation.

## Cheapest experiment

[EXP-0025](../experiments/EXP-0025-greedy-boundary/README.md), exact finite-field
and integer calculations (E0), no CPU timing or H100 inference. Test the native
polynomials against literal tensor/index formulas before changing runtime code.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| Soundness | weakened | PCS images must share one original value; full FS/budget integration is still owed |
| Prior art | survives | Composition of the accepted Argmax, OneHot, sumcheck and linear PCS obligations |
| Cost model | survives with zero credit | Additional groups and rows cannot be called a speedup |
| Feasibility | survives provisionally | Public increment has logarithmically many original points; finite oracle can refute it |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 performance credit (E0) | 0.6 for the algebra; integration unvalidated | m, RFC and CPU conformance required | 0; correctness gate overrides performance queue |

## Decision and links

Triaged for EXP-0025. No accepted RFC, accepting driver or performance result.
Ledger updated 2026-10-21.
