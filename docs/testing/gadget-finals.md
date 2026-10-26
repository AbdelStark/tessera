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
cargo test -p tessera-e2e --locked --test alternative_witnesses
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

The native graph schedule derives all four OneHot relations in stage 1, in node
order before the embedding equality check (RFC-0009, RFC-0035, RFC-0037). Every
check reads the original indicator; the index check also reads the original token
source. All final source claims reach the opening pool, including the token's
row-coordinate projection. The common zero-check point covers the full indicator
domain. Count and index use its row suffix.

For padded vocabulary size `W`, the implementation proves these zero sums over
the full token/vocabulary cube:

```text
sum eq(z,(v,t)) * H(v,t) * (1-H(v,t))                    = 0
sum eq(z,(v,t)) * (1-LT(v,vocab)) * H(v,t)               = 0
sum eq(z_row,t) * (H(v,t) - psi(t)/W)                   = 0
sum eq(z_row,t) * (index(v)*H(v,t) - T(t)/W)            = 0
```

The repeated public selector and token terms sum over `W` positions, giving
exactly RFC-0009's per-row count and index equations. `W` is nonzero in the field.
This formulation needs no new token evaluation between the zero-point draw and
batching. It introduces no new transcript label, commitment or advice column.
Booleanity and vocabulary have degree three; the row equations have degree two.
The CPU's generic index kernel uses the conservative degree-three bound already
shared with booleanity in the same batch.

Independent proofs isolate each premise, including a one in vocabulary padding
with the matching out-of-vocabulary token, and a wrong token for vocabulary one.
Every other OneHot premise still holds for each alternative. The actual CPU
engine also proves scalar and ragged-vocabulary model variants with public
synthetic embedding/head weights. These are graph stage proofs, not complete
model proofs: the remaining gadget, lookup and opening composition below is
still required.

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

RFC-0066's `BitDecompose` and `AssertEqual` have no final check here: their relations are stage
1's zero-checks (RFC-0072 'The bit zero-checks'), so a false bit fails stage 1's final check,
`Reject(Sumcheck { stage: 1, round: n_1 })`, never `Reject(Gadget)`. Their alternatives are the
prover's stage tests (`crates/tessera-prover/tests/bits/mod.rs`, part of `tests/stages.rs`): of every Boolean assignment
of a decomposition's bits and padded slot only the honest one passes, nor does any non-Boolean
assignment that recomposes; only equal tensors pass an `AssertEqual`; and each check's violation
passes once exactly that check is removed. The verifier's mirror is #150's and #159's.

Adversarial fixtures cover low-limb wrap and underflow, narrowed-top `u + P`, full
products/constants beyond Goldilocks, root and reciprocal alternatives, wrong
maxima, masked keys/leaves, omitted padding axes, two/wrong OneHot entries and
later Argmax ties. Omission proofs retain true source evaluations and pass a
control with exactly the tested obligation removed. Perturbing a final evaluation
must reject or leave a false canonical source claim for the later opening check;
it must not silently discharge that claim.
