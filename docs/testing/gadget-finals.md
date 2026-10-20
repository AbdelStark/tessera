# Gadget final checks

`tv_core::gadgets` implements the independent final relations of
[RFC-0009](../rfcs/RFC-0009-gadgets.md), including the amendments for wide
requantization, masked exponentials and canonical root advice. It does not
return a whole-proof verdict. The verifier driver (#128) joins these checks to
the earlier LogUp/product forests, graph stages and commitment openings.

## Run the checks

From the repository root:

```sh
cargo test --manifest-path verifier/Cargo.toml -p tv-core --locked
cargo test -p tessera-prover --locked --test gadget_comparison --test gadget_root_products --test gadget_requant --test gadget_onehot --test gadget_root --test gadget_softmax --test gadget_argmax --test gadget_argmax_outputs
cargo xtask gen-vectors --check
cargo xtask check-isolation
```

For fuzzing, use the pinned nightly in `verifier/fuzz/` and follow its
[README](../../verifier/fuzz/README.md). The existing `sumcheck` target routes
`GDG` scripts to all gadget families and output-row views. Ordinary `cargo test
--locked` in that workspace replays every generated gadget seed and checks
million-element input rejection with a measured allocator.

The [fixture documentation](../../spec/vectors/gadget-finals/README.md) describes
the independent integer/direct-cube oracles and their producer consumers.
Verifier tests passed before producer implementation comparisons. Producer
adapters then checked actual witness columns, lookup rows and CPU proof messages,
including final transcript state and squeeze counter.

The native graph schedule also derives OneHot booleanity in stage 1, in node
order with the embedding equality check (RFC-0035, RFC-0037). Its degree-three
sumcheck reads the original indicator and returns that same source to the opening
pool. The shared zero-check point has the largest zero-check domain, including
the vocabulary axis. Independent synthetic proofs and actual CPU witness proofs
exercise this path; a non-Boolean indicator fails even when downstream values
are recomputed consistently. This does not discharge the other OneHot relations
or the remaining gadget/source and lookup composition below.

## Required composition

| Component | Checked relations | Obligations retained for the driver |
|---|---|---|
| Requant1/Split/Wide/TableMap | Every canonical affine key and value, including class projection and wide saturation | Earlier LogUp claims; graph input/output; class weights and auxiliary openings |
| NonNeg | Every limb key and the narrowed virtual top | Input and limb sources; exact Range memberships |
| BigIntCompare | Both normalizations, subtraction, both final carries and final borrow | Raw product coefficients, all six canonical arrays and their Range memberships |
| RsqrtRow | Canonical decompositions, both full-integer brackets and degree-four product reductions | Root/input/auxiliary sources; graph sum of squares; root padding and residual feature/time padding |
| OneHot | Booleanity, vocabulary padding, row sum, token index | Original token claim and the same indicator source at each final point |
| SoftmaxRow | Masked lookup key, masked maximum-forest leaf, reciprocal product and both NonNeg decompositions | Earlier forest/LogUp claims; maximum, reciprocal and exponential sources; both row-padding checks |
| Argmax | Masked Range29 key, selected logit and strict suffix | Certificate maximum, logits and the same next-token OneHot; output views bind original graph rows |

`Plan` derives graph sites, identities, tables and geometry from the verifier's
private lowered graph and selection. Scoring owns Argmax's global certificate id
and maximum commitment; those are not proof-selected graph auxiliaries. For an
output row `u`, its logit is graph row `prompt_len - 1 + u`, and its indicator is
the existing OneHot at `prompt_len + u`.

All successful check states that carry unresolved claims are `#[must_use]`.
A successful scalar relation is not proof that its source values are committed
or that lookup membership holds. A driver must preserve each claim's identity,
point and value, pair each lookup with the prescribed earlier leaf claim, and
route each graph/advice/weight/certificate claim to its reduction or opening.

Incoming evaluations must be fixed before sumcheck batching. Lookup and maximum
forest points can differ. Instance callbacks receive the suffix of the final
point after front-padding; do not remove the padding twice. For layer templates,
fold fixed incoming layer claims before gamma and reduce nonlinear relations over
the layer axis. Late per-layer evaluations at a known layer point can cancel.
Stage assembly and the aggregate soundness/resource counts belong to the driver.
The component tests do not assert a complete native-model transcript or verdict.

The simplified root input `u = S + epsilon*psi` requires proved zero residual
padding on **both** feature and time axes. The checker returns that zero-check
obligation separately. A layer template also needs the real-layer selector in
`psi`. Canonical physical row placement alone does not establish these zeros.

## Rejection order and alternatives

Local affine failures and malformed final-evaluation counts produce
`GadgetError::Rejected { gadget }` with the canonical owner id. Wrong
verifier-owned point geometry is `Internal`; failed allocation is `Memory`.
No error includes field evaluations or secret values.

The blanket `Reject(Gadget)` wording in #124 must be read with RFC-0019's normative
first-failure order: illegal lookup membership fails LogUp in step 7; a failed
nonlinear batched relation fails Sumcheck at its exact stage and round. A mixed
sumcheck batch cannot identify one guilty gadget. Relabeling either failure as
Gadget would violate that order. InverseCdf alternatives belong to #160.

Adversarial fixtures cover low-limb wrap and underflow, narrowed-top `u + P`, full
products/constants beyond Goldilocks, root and reciprocal alternatives, wrong
maxima, masked keys/leaves, omitted padding axes, two/wrong OneHot entries and
later Argmax ties. Omission proofs retain true source evaluations and pass a
control with exactly the tested obligation removed. Perturbing a final evaluation
must reject or leave a false canonical source claim for the later opening check;
it must not silently discharge that claim.
