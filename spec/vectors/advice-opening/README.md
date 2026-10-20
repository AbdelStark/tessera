# Canonical advice groups and openings

`generate.py` is the independent oracle for RFC-0011/0031/0037. It was written and
frozen before the model opening checker (#116), using only the existing independent
Python lowering, public advice geometry, field and hashing oracles. It reads no Rust
implementation or output.

`opening.json`, version 1, contains:

- All 54 original conformance layout cases: committed row counts per segment,
  descriptor TCE bytes and commitment identifiers. Counts computed from the
  complement of public zero intervals agree with the sum over tensor locations.
  Millions of rows are counted without enumerating them.
- Three complete micro-model openings: `n = 1, 3, 5`, `S = 1, 2, 8`, `B = 32, 3, 5`,
  and 1, 16 and 5 groups, respectively. The 56 original claims include combinations,
  layer-specific tensors, whole attention tensors/auxiliaries and multiplicities.
  Claims are interleaved, so grouping must preserve first appearance and original
  claim order. Each original claim takes one power `gamma^(t+1)`, across all pieces
  and weighted images; powers restart for each group.
- Every combined row and mask, all queried column symbols, canonical siblings and
  SHA-256 digests of the complete row-coefficient vectors in retained-row order.
  The digest input is each Fp2's two canonical little-endian u64 words. SHA-256 here
  only keeps fixture diagnostics compact; it is not a new protocol domain.

The micro-model starts from `tiny-2-servable`'s public metadata. Its architecture is
`d_model=4, heads=2, kv_heads=1, d_head=2, d_ff=6, vocab=8, max_context=8`; it retains
two layers. Its residual has eight bits and the embedding's exponent; norm precision
is zero, `exp_sigma=3`, every PoT shift is 12 and each score scale is 1. The public
table policy is 17 bits, limb width 8 and query block 2. The component's encoding is
`k=64`, rate 1/2 and four mask coefficients. These inputs pass public lowering and
gadget selection. They are not a certified parameter set or a fidelity measurement.
Unmodified conformance geometry remains covered by all 54 original cases.

The witness is public synthetic data. On the row subgroup, row `r` holds
`c(r)+d(r)*X`, with `c(r)=17*r+5` and `d(r)=r*r+3`; mask coefficient `i` is
`(r+1)*(i+1)+7`, all reduced in Goldilocks. Its masked polynomial is evaluated by
Horner's rule, checked on representative rows against independent Lagrange
interpolation. Every segment chain, including an empty segment, contributes to
ColumnMerge, even for S=1. A dense tree produces the root and canonical multiproof.
Queries repeat and are deliberately not sorted; the answer contains sorted distinct
columns. These rows test the commitment component, not an inference statement or
zero knowledge.

Claim values have two independent constructions: Boolean enumeration in the source's
natural coordinates (factored for large one-dimensional multiplicity tables), and
the weighted image sum at the permuted, offset advice points. They must agree before
the fixture is written. Verifier tests check these mappings and original-claim powers
against the existing public geometry, then verify the complete openings and exact
evaluation, mask, symbol and sibling errors. PCS tests reproduce roots, coefficients,
combined rows, columns and siblings. Compiler tests check every original layout count
and descriptor.

Run `python3 spec/vectors/advice-opening/generate.py` to regenerate;
`cargo xtask gen-vectors --check` checks byte-for-byte reproducibility.
