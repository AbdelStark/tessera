# Weight-opening vectors

`openings.json` is the independent oracle for the unmasked weight openings of
[RFC-0012](../../../docs/rfcs/RFC-0012-weight-commitment.md#opening-by-the-folding-pass),
[RFC-0040](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md#the-matrices) and
[RFC-0049](../../../docs/rfcs/RFC-0049-weight-proximity.md#the-proximity-row).

`generate.py` was written before the independent verifier's opening checker. It computes
small matrices directly from their public shapes and channel liveness, with Python integers.
It reuses only the independent hash and polynomial oracles in `../hash/` and `../pcs/`:
Lagrange interpolation and Horner evaluation replace the Rust inverse/coset NTTs. Every
combined codeword is checked against the independently encoded rows' linear combination.
No implementation under `crates/` or `verifier/` is called.

Seven cases cover wide inputs, partial output rows, gaps between layers' live rows, sparse
channel parameters, a scalar polynomial and an empty `ChannelParams` matrix. Every case
records the public shape, live rows and their synthetic values, descriptor and commitment id,
root, points, transcript coefficient, claimed value, combined rows, queries, answered columns,
canonical multiproof, row coefficients and combined codeword values at the queries.
Evaluation coefficients index **structural rows**; proximity coefficients index **positions
in the answer**. They differ when whole zero rows were skipped.

Field values are canonical hexadecimal words, and extension elements are `[c0,c1]` with
`X²=7`. No masks or blinding rows are included. An empty matrix has no combined rows or
queries, and a zero evaluation. The fixtures' small public architectures exercise the layout
and opening components; they do not assert claim-list validation or a complete protocol proof.

Run `python3 -I -B spec/vectors/weights-opening/generate.py` to regenerate. The verifier's
`weight_openings` tests consume every case and isolate proof faults. Prover-side differential
coverage was added only after the independent verifier passed these vectors (SP-6, DEP-2).
`crates/tessera-pcs/tests/weights_opening.rs` reproduces every nonempty case through the real
fold pass and checks the empty case has no fold. Its separate production conformance test
verifies 120 real openings against `tv-core`, with authenticated roots bound to `R_W`.

Ten `ordered_faults` cases alter both combined rows using degree-bounded polynomial products.
Each row disagrees at a different authenticated queried column while agreeing at every other
query. The adjusted claim passes the evaluation check; the expected proximity rejection is the
smallest failing column across both rows, independently determined by the oracle.

Each nonempty case also includes `omitted_columns`: fewer column answers with a valid
canonical multiproof for exactly that subset under the same root. The verifier must reject
this as `MerklePath` when more queries were requested; the identical answer is accepted when
its actual query set is restricted to the subset. Authentication alone cannot replace the
answer-count check.

`short_columns` authenticates columns with one symbol too few under a maliciously built
root using the expected descriptor. Both combined rows and the adjusted claim are consistent
with those shortened columns. The derived row-count check must still reject them as
`MerklePath`; binding a root does not make malformed committed column lengths canonical.
