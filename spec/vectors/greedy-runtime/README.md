# Greedy runtime metadata

`generate.py` uses the independent Python model, lowering, range and advice
oracles to derive 141 per-request catalogs and layouts for draft
[RFC-0053](../../../docs/rfcs/RFC-0053-greedy-runtime-boundary.md).
It imports no Rust implementation and reads no observed logit bound.

Cases include empty outputs, one output, ragged vocabulary/time domains, exact
Range reuse, eight appended-Range cases, multiple query blocks/segments, and a
maximum whose time padding removes a whole row. Original model table ids and
model tile locations stay fixed. Each case records complete used-table counts,
maximum/multiplicity locations and point permutations, public zero-row tails,
and every added native descriptor's owner, variables, degree, final count and
original graph projections.

The compiler and independent verifier check all catalog/layout cases. CPU leaf
metadata tests additionally check all 101 cases without model rescale variants,
including every appended table, against both the literal counts and independent
split metadata/digests. Existing model factories cover the rescale variants.
These are metadata and callback-algebra fixtures; they establish no accepting
session, complete runtime proof, soundness budget or performance result.

Regenerate with `python3 -I spec/vectors/greedy-runtime/generate.py`;
`cargo xtask gen-vectors --check` checks its exact bytes.
