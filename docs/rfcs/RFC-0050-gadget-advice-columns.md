# RFC-0050: Canonical Gadget Advice Columns

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-19
- Target milestone: v0.3
- Supersedes in the listed sections:
  - [RFC-0009](RFC-0009-gadgets.md) 'BigIntCompare' and 'RsqrtRow': the exact auxiliary representation, column order and range requests;
  - [RFC-0022](RFC-0022-saturating-residual.md#complete-norms) 'Complete norms': the minimum root lookup width;
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': the complete auxiliary range-table requests; 'Range analysis': minimum root width;
  - [RFC-0041](RFC-0041-execution-plan.md#production-order) 'Production order': the ordered gadget values following node outputs.
- Origin: [#407](https://github.com/AbdelStark/tessera/issues/407), prerequisite of the remaining [#106](https://github.com/AbdelStark/tessera/issues/106)

## Summary

The advice layout must be derived independently from public data (INV-PCS-1,
SP-6). RFC-0041 places a node's committed outputs before its gadget values, in
RFC-0009's order. The latter specifies integer relations but leaves parts of a
comparison witness and auxiliary root decomposition open. Different correct
representations give different advice offsets and table descriptors.

This RFC fixes the current ClaimList lowering's auxiliary roles, shapes, order and
range requests. It retains the original integer operations and the original
`(rho+1)^2*u` bracket polynomial. It makes the migration from the current scalar
reference explicit. No message field, challenge label, table kind or cryptographic
assumption is added.

## Motivation

For a root in a 32-bit box, one committed low 16-bit digit with a virtual high
digit and two committed 16-bit digits can both give exact recomposition in a box
below Goldilocks. They can use the same table families while allocating different
auxiliary tensors (E0 analytic example). Uniqueness of a result does not define
its representation.

Likewise, "enough high zero limbs" for BigIntCompare does not define a public limb
count. A layout cannot infer offsets from a prover's chosen representation. The
decision must precede the complete model layout and its maximum-size check.

## Goals

- A complete ordered auxiliary list for every gadget of the current lowering.
- Public shapes, integer boxes and lookup families, with no secret-dependent size.
- The same semantic role at the same advice location in independent implementations.
- Exact model counts, replacing approximate Rsqrt costs before parameter checks.

## Non-Goals

- Implementing the model layout, committer, gadget verifier or proof drivers.
- Adding scoring or anchor nodes to the claim list (RFC-0026), or defining a future
  sampler's auxiliary schema.
- A new root algorithm, new proof round or a claim of performance gain.
- Claiming that a model/preset meets G-4 without its complete exact size profile.
- Protocol/2 masking/blinding rows, which remain separately required by RFC-0013.

## Proposed Design

### Role identity, domains and graph outputs

A gadget auxiliary is identified by its graph node and the role below, including
array indices. A role is unique within its node. These identities are derived
metadata, not new TCE tags or proof-supplied identifiers.

Allocate a node's committed graph outputs once, in graph output order, then its
auxiliaries in this RFC's order. A lookup value or root already in the graph is
not allocated again. Public values, raw product sums and virtual top limbs do not
get advice tensors.

Each Requant auxiliary has its output's element domain. Rsqrt auxiliaries have
the root's row domain, without the residual's feature axis. Softmax's maximum and
NonNeg auxiliaries have its reciprocal's query-row domain. RFC-0041 splits those
rows by KV head and query block with the attention tile; its exponentials retain
their score domain. Tensor layout order, alignment, causal zero rows and segments
remain RFC-0011/0034/0046/0047's.

| Gadget | Auxiliaries after its graph outputs, in order |
|---|---|
| Requant1, lookup/TableMap, OneHot | None |
| RequantSplit | `z_lo` |
| RequantWide | `z_lo` when `s>0`, then `h0`, then `g` |
| SoftmaxRow | Maximum `c`, then low limbs of `NonNeg(a,sigma+18)`, then low limbs of `NonNeg(b,sigma+18)` |
| RsqrtRow | Input low limbs, root low bytes, lower comparison, upper comparison, as specified below |

For each NonNeg, low limbs are least significant first and the top limb is
virtual. The width `k` and limb width `L` give `ceil(k/L)-1` committed low limbs
and a top lookup of width `k-(ceil(k/L)-1)*L`. A decomposition with one limb has
no committed low limb. General NonNeg and Softmax use the public gadget policy's
limb width; the two fixed Rsqrt decompositions below are explicit specializations.

Other lookups retain RFC-0009's order. In particular, RequantWide's virtual top
range and saturation lookup are not replaced by simply reading its advice list.
Softmax reads Exp, its reciprocal range, then the two NonNeg instances, including
each virtual top. Graph-output order and lookup order are distinct contracts.

### Complete comparison schema

A comparison has a public radix `B=2^L`, public limb count `m>0` and inclusive
coefficient bounds `A_left[j]`, `A_right[j]`. Its input vectors are virtual when
already computed elsewhere; they are not extra commitments.

Its committed arrays are, in this order:

1. left normalized digits;
2. left outgoing normalization carries;
3. right normalized digits;
4. right outgoing normalization carries;
5. difference digits;
6. outgoing subtraction borrows.

Every array has `m` entries, least significant first. Arrays are grouped, not
interleaved by limb. Both final carries and the final borrow remain committed
values constrained to zero; constant sides do not omit arrays.

For either side derive `K[-1]=0`, `K[j]=floor((A[j]+K[j-1])/B)` and
`b[j]=max(1,bit_length(K[j]))`. Digits and differences request Range(L), carries
Range(b[j]), and borrows Range(1). The lookup order is the committed-array order.
The equations and both final-carry checks remain RFC-0009 Lemma 4's.

The integer proof counts the entire allowed carry box `C[j]=2^b[j]-1`, not only
the honest `K[j]`. Require the input bound below `2^40`; each of
`A[j]+C[j-1]` and `(B-1)+B*C[j]`, with `C[-1]=0`, below `2^41` and inside the
centered field range. The Rsqrt specialization below satisfies the stronger
`2^40` bound. A caller must also establish each raw coefficient's public bound
from already bound inputs (SP-7), never advice declarations.

### Exact Rsqrt parameters and decompositions

Let `N=d*2^(2*kappa)`, `1<=e<2^55`, and
`w=max(1,bit_length(floor(sqrt(floor(N/e)))))`. Refuse a root needing more than
32 bits. The minimum width one covers a maximum root of zero: the bracket, not
the lookup alone, pins that root to zero. This does not change `D`.

The root `rho` is the node output. Allocate these auxiliaries next:

1. `u_low[0..3]`, three 16-bit limbs. The virtual top is
   `(u-sum(u_low[j]*2^(16*j)))/2^48`, range-checked to seven bits.
2. `root_low[0..r]`, with `r=ceil(w/8)-1`, in byte order. The virtual root top is
   `(rho-sum(root_low[j]*2^(8*j)))/2^(8*r)`, range-checked to `w-8*r` bits.
3. The lower comparison's six arrays.
4. The upper comparison's six arrays.

The divisions in these virtual affine forms are field inverses. Their complete
55-bit and w-bit boxes are below the field modulus, so the range checks force
the unique integer decomposition. Do not replace the top seven-bit input lookup
with a full 16-bit range.

Both comparisons use byte radix, regardless of `GadgetPolicy.limb_bits`, and

```text
m = ceil((2*w+55)/8).
```

This public count includes every high limb needed for the complete product. It
is never chosen from the actual secret root or input. Keep all `m` columns of
each array even when a particular row's high values are zero.

### Product coefficients and the bracket polynomial

Write the root digits, including its virtual top, as `R_i`, and input limbs as
`U_0,U_1,U_2,U_3`, with the virtual `U_3` of seven bits. In byte radix the lower
product's raw coefficient is

```text
A_j = sum_{i+h+2*k=j} R_i * R_h * U_k.
```

The upper product uses `R'_0=R_0+1`, `R'_i=R_i` otherwise. Its low factor may
equal 256; it is not truncated, wrapped into a byte or redecomposed with its
carry discarded. Coefficients are sums of products, not limbs of a whole product
already reduced modulo the field.

The lower comparison is `rho^2*u <= psi*N`. The upper comparison is
`psi*(N+1) <= (rho+1)^2*u`. Public constant sides are the `m` low bytes of `N` or
`N+1`, multiplied by the real-row selector `psi`; their normalization uses the
same full array schema. Input bounds are the public maximum product coefficients
and the public constant bytes. Derive product bounds with each narrowed root
digit box, `U` boxes of `(16,16,16,7)` bits, and the upper low factor increment.

The original `rho+1` polynomial is retained at the sumcheck's final point.
`rho+psi` gives the same row data only on the required domain, because

```text
u * ((rho+1)^2 - (rho+psi)^2)
    = u * (1-psi) * (2*rho+psi+1).
```

This is zero at Boolean positions when the upstream
`u=psi*(sum(res_i^2)+e)` obligation is proved. It is not an arbitrary-field
polynomial identity; implementations must not substitute its final evaluation.

### Range requests, family interning and counts

Rsqrt requests, in order:

1. Range(w) for the graph root;
2. each input low limb's Range(16), then the virtual top's Range(7);
3. each root low byte's Range(8), then its virtual top's narrowed range;
4. lower comparison requests in array order;
5. upper comparison requests in array order.

Keep repeated requests. The existing base tables retain the independently lowered
order. Walk gadgets in node order; reuse the lowest id of an equal table kind already present, or
append its descriptor on its first request. Equality for interning concerns the
table definition, not its id-dependent digest. Assign canonical ids in that
sequence and regenerate the digests. A required range wider than the public table
policy is a refusal, not permission to select another witness representation.
Both validators compare the complete derived table sequence (RFC-0007 step 5.3).

The standalone root has `4+r+12*m` committed values, including its graph output,
and `6+r+12*m` range requests. Its auxiliary list excludes the graph root. At
`w=32`, these are 187 and 189 (E0 exact schema counts, validated in EXP-0023).
Approximate historical "about 20/30" costs do not override these counts. Model
reports and the soundness calculator must use the derived counts, including
multiplicity columns and actual table sizes; no timing gain is claimed.

### Padding and binding obligations

At a padded row, the upstream selector makes `u=0` and the padded-root zero-check
makes `rho=0`. Their low limbs and virtual tops are zero; both product coefficient
vectors are zero even though the upper low factor is one. Both public comparison
constants are zero. Normalized digits, carries, differences and borrows are zero.
The witness is unique and the layout may skip whole rows proved zero by public
padding rules. It must not skip a real row because a secret witness happens to be
zero.

Every root, auxiliary, lookup key and final comparison claim must bind to the same
node/role and public selector. A checked bracket does not discharge the upstream
input-selector equation or its integer range; those obligations cannot be dropped
by the gadget verifier or driver. General padded witnesses remain RFC-0035's.

The local gadget treats `u` as a virtual tensor with its own multilinear-extension
claim. Its definition must be reduced correctly: the extension of a product is
not in general the product of extensions at the final point. If an implementation
inlines that definition instead, it must derive and charge the resulting degree
and all binding obligations, rather than assume the local gadget degree.

## Security Argument

- **Integer exactness and uniqueness.** Input and root boxes are below Goldilocks.
  Each normalization equation has small integer sides, including the full carry
  lookup boxes. Lemma 0 makes field congruence integer equality; induction with
  final carries zero fixes the complete products. Range-checked subtraction with
  final borrow zero establishes the two integer inequalities. They pin the root
  as RFC-0009 Lemma 7 does. No second auxiliary representation is optional within
  this schema.
- **No high-product truncation.** A root is below `2^w`; its increment is at most
  `2^w`, and `u<2^55`. Both products are below `2^(2*w+55)`. Also `N+1` fits that
  box: for a positive maximum root, `N<e*2^(2*w)`; for a zero maximum, `N<e`.
  The declared `m` is sufficient. At the largest root width the complete product
  may need 119 bits, not one field element.
- **Soundness accounting.** The local pointwise product in its declared virtual
  inputs has degree at most three (four with a sumcheck equality factor); complete
  reductions must also account for the input binding above. The original bracket
  polynomial, transcript patterns and challenges remain. This does not fix a model's security budget by
  assertion. Both sides must supply actual schema-derived rounds, instances,
  lookups, table rows and advice rows to RFC-0014's calculator and reject a preset
  missing its target. No target, grinding requirement or field term is weakened.
- **Secrecy and zero knowledge.** Sizes and zero-row decisions use public inputs
  only. Secret values still live in clearing storage and protocol/2 must mask all
  canonical columns under RFC-0013. This public format decision is not a proof of
  the full protocol/2 simulator.
- **Independence and robustness.** The verifier derives roles, widths, table ids
  and offsets from the specification, without importing prover helpers. Shared
  independent vectors are data only. Reject invalid parameters, overflow, missing
  bindings and resource-limit violations before acceptance or unchecked allocation.

EXP-0023 validates the declared finite E0 arithmetic/schema checks, not an adaptive
Fiat-Shamir proof or complete model implementation. Its original and amendment
plans were committed before their code; exact results and scope are in its report.

## Alternatives Considered

1. **Prover-declared columns or lengths.** Rejected: they do not define a layout
   independently of advice and permit disagreement about what a commitment binds.
2. **Leave representation to each implementation.** Rejected: mathematically
   equivalent witnesses need not have the same column count or offsets.
3. **Keep fifteen comparison limbs for every width and interleave columns.** The
   scalar reference uses this sound representation. Choose the public full-product
   bound and grouped named arrays instead: the specification states exactly what
   each role means, including the smaller-width cases. Migration is explicit;
   passing current scalar tests is not treated as format conformance.
4. **Omit constant-side arrays or final zero carries.** Not chosen: uniform arrays
   keep the generic comparison contract explicit; the final conditions must not
   disappear behind an optimization. This RFC makes no performance tradeoff claim.
5. **Use `rho+psi` in the final-evaluation polynomial.** Not chosen: the registered
   amendment establishes valid-row data equality, not arbitrary-field identity.
   Retain the accepted polynomial to avoid an unnecessary relation change.
6. **Reuse the public policy limb width for every product factor.** Not chosen for
   Rsqrt: triple full 16-bit limb products can exceed the small coefficient bound.
   Byte root factors and 16-bit input limbs meet the exact box requirements without
   reducing a whole product in the field. Other gadgets keep their public policy.

## Drawbacks

- The current scalar fixed-length/interleaved representation needs a public-role
  adapter or refactor before it supplies model advice in this order.
- Exact counts are larger than the old approximate Rsqrt cost; no measured
  full-model cost or proof-size improvement is credited.
- Complete auxiliary families change claim-list table sequences and their
  id-dependent digests. Old incomplete model integration cannot be called complete.
- More virtual/committed-role detail must be implemented twice without sharing code.

## Migration / Rollout

1. Accept this RFC through CONTRIBUTING's RFC process.
2. Generate independent vectors for ordered roles, range requests and family ids;
   adapt standalone gadget metadata by semantic role, preserving secret clearing,
   full products and the original relation polynomial.
3. Implement both complete model layouts for #106, independently, including
   conformance claim-to-offset identities, padding/segments, the 53,248-wide MLP
   zero-row case and the `2^32` maximum at `max_context`.
4. Derive all auxiliary table definitions in compiler/quantizer and verifier;
   regenerate affected conformance artifacts and keep the existing byte schemas.
   No release migration is waived: apply 09 §3 before supporting old artifacts.
5. Integrate #112/#115/#124/#125/#128/#130 against the same roles and exact counts.
   Require all domain, binding and parameter checks before accepting a model proof.

Until those steps are implemented, this decision is not evidence that a current
model proof verifies. Scoring/anchor schemas must be specified when their nodes
join the claim list; they are not silently added here.

## Testing Strategy

Run the pre-registered independent experiment:

```sh
python3 -I research/experiments/EXP-0023-gadget-column-contract/main.py
python3 -I research/tools/check_kb.py
cargo xtask lint
```

Its E0 checks cover all 32 root widths, 4,352 honest rows, 8,704 root alternatives,
13,680 auxiliary alternatives, independent descriptors and carry bounds, table
policies, complete 119-bit products, padded rows and virtual top boxes. Nineteen
distinct faults and an additional `u+P` attack control are caught; five controls
change first rejection behavior rather than accept a bad witness. The amendment
repeats the matrix under the original polynomial and verifies identical columns,
including zero-maximum-root and malformed-parameter boundaries. A separately
pre-registered extension checks twenty additional scalar rows with constants
above Goldilocks and five intentional constant-reduction faults; no production
vulnerability is inferred.

Production tests must additionally compare independent model layouts and complete
table sequences, exercise every source-to-role binding, reject a dropped/moved
column and missing range family at its exact reason, preserve the original
sumcheck final evaluation, verify mask/clearing behavior where applicable, and
use actual derived counts in parameter validation. A component-only arithmetic
pass is not a complete model-proof acceptance test.

## Open Questions

None for this column contract. The implementation and its integration gates remain
the issues named above; this RFC does not assert that those gates have passed.

## References

- [IDEA-0036](../../research/ideas/IDEA-0036-canonical-gadget-columns.md),
  [EXP-0023](../../research/experiments/EXP-0023-gadget-column-contract/README.md),
  [SUR-0014](../../research/surveys/SUR-0014-gadget-column-contract.md).
- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md),
  [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md),
  [RFC-0014](RFC-0014-interaction-and-soundness.md),
  [RFC-0022](RFC-0022-saturating-residual.md), [RFC-0026](RFC-0026-claim-list-layout.md),
  [RFC-0035](RFC-0035-padded-rows.md), [RFC-0041](RFC-0041-execution-plan.md),
  [RFC-0046](RFC-0046-advice-segments.md), [RFC-0047](RFC-0047-advice-order.md).
