# Gadget final checks

Independent data for the final relations of [RFC-0009](../../../docs/rfcs/RFC-0009-gadgets.md)
and the canonical comparison arrays of [RFC-0050](../../../docs/rfcs/RFC-0050-gadget-advice-columns.md).
`generate.py` uses Python integers, weighted prefixes and direct sums over the cube.
It neither imports nor calls either Rust workspace.

## Comparisons

`comparison.json`, schema version 1, contains 75 cases and 572 integer witnesses:

- Limb widths 1, 2, 7, 8, 16, 31 and 32, including actual normalization carries,
  subtraction borrows, equal operands and zero rows.
- Both root comparisons for every root width from 1 through 32, with public
  maximum coefficient bounds derived from the narrowed digit boxes.
- Root constants whose upper bracket is exactly Goldilocks and whose constants
  exceed Goldilocks. Constant bytes come from the full integer.
- Padded root rows: zero inputs and constants, both full comparisons retained.

Each case records inclusive coefficient bounds, outgoing carry lookup widths,
row-wise inputs, all six arrays in canonical grouped order, and evaluations at a
non-Boolean extension-field point. Arrays are left digits, left carries, right
digits, right carries, differences and borrows; each retains its last element.
The row's `x` and `y` are decimal strings because the full integers can exceed
64 bits. Every field evaluation is a canonical pair of unsigned coefficients.

The oracle obtains normalized digits from the whole integer and each carry from
a weighted prefix. It obtains every borrow by comparing the two low prefixes,
then checks the normalization and subtraction equations separately. It checks
both terminal carries and the terminal borrow, every range box, and exact whole
integer reconstruction. Evaluations sum `eq` over every row without using the
production recurrence.

These fixtures describe local relation arithmetic and its binding inputs. They
are not complete model proofs, lookup proofs or a soundness certification. The
verifier must bind each input and witness evaluation to its canonical role, check
the prescribed range lookups, and prove any nonlinear input reduction.

Regenerate with:

```sh
python3 -I -B spec/vectors/gadget-finals/generate.py
cargo xtask gen-vectors --check
```

The independent verifier consumers are `verifier/crates/tv-core/tests/gadget_comparison.rs`
and `gadget_decomposition.rs`. The producer consumer is
`crates/tessera-prover/tests/gadget_comparison.rs`: it derives actual witness
columns and lookup forms, compares all values and extensions to these fixtures,
and passes the resulting evaluations to the independent affine checks. The
verifier consumers passed before the producer adapters were written.

The final verifier driver must still join these local arithmetic checks to the
canonical lookup, graph and opening obligations. No complete model proof is
accepted by these fixture tests.

## Decompositions

`decomposition.json`, schema version 1, contains 248 cases and 992 integer
witnesses: every NonNeg width 1–62 with limb widths 1, 8, 16 and 32. Zero,
one, the last admissible integer and an interior integer have their low limbs
and their virtual top recorded. The final range width is narrowed independently
of the policy limb width. Direct cube sums give every value and limb evaluation
at a non-Boolean point. This includes the 55-bit `(16,16,16,7)` root input and
all 1–32-bit byte decompositions of roots. These affine relations still owe
lookup proofs and source bindings; the fixture does not replace either.

## Root products

`root-products.json`, schema version 1, contains 35 standalone degree-four
sumcheck batches with 826 coefficient instances: both products for every root
width, plus the constants-at/above-Goldilocks cases. Each has eight rows, four
padded, and reduces a coefficient claim at an incoming non-Boolean point to
claims on the root-byte and input-limb extensions at the sumcheck's final point.

The polynomial is `sum_{i+h+2*k=j} R_i*R_h*U_k`, with `R_0+1` on the upper
side. Every raw coefficient is retained, including the high zero coefficients
needed for full carry normalization. The upper polynomial uses the constant one
at arbitrary field points; substituting the row selector changes its polynomial.
The oracle sums the remaining Boolean vertices in every round and interpolates
with the independent sumcheck oracle. Its transcript oracle checks the official
BLAKE3 vectors first.

The fixture records its context, incoming point and claims, operand tables,
batching challenge, round messages, final evaluations, final point and transcript
state/counter. Incoming claims are fixed under `stage/0/split` before the
`stage/1` batch. These are component proofs with synthetic inputs, not complete
model proofs: the driver must bind the incoming claims to the comparison arrays
and every outgoing digit/input claim to its canonical decomposition and source.

The final batch is a deliberate 32-bit root alternative under admissible public
root parameters (`kappa=31`). Its root and input are at their range-box maxima,
so the products actually require 119 bits. It has a valid product sumcheck but
its lower comparison's final borrow is one. The embedded comparison arrays have
all digits, carries and borrows in range; the affine checker rejects the lower
comparison with the exact gadget id. This is a rejection control, not an accepted
root witness. The CPU producer's honest root differs from the alternative.

`verifier/crates/tv-core/tests/gadget_root_products.rs` verifies these batches,
reconstructs their incoming claims from complete comparison arrays, checks all
operand extensions and transcript states, and rejects the wrong bracket and the
`rho+psi` substitution. `crates/tessera-prover/tests/gadget_root_products.rs`
reproduces every round and final evaluation through the actual CPU sumcheck
engine, feeds them to the independent verifier, and checks all honest root,
auxiliary and comparison columns against the oracle. The product kernel remains
a relation evaluator; complete model scheduling and source binding are separate.

## Requantization bindings

`requant.json`, schema version 1, freezes 119 sites and 183 lookups across five
public micro-model variants before implementing the verifier's binding adapter.
`requant.py` uses the independent public-model, lowering, selection and table
oracles. It derives keys from whole integer digits and evaluates their complete
Boolean tables by direct cube sums. Cases cover single-table and split patterns,
wide patterns with and without a low column, four classes, three real token rows,
non-power-of-two channel domains and non-Boolean extension-field points.

Each site records its graph-node and gadget ids, table ids in request order,
canonical graph/auxiliary sources, source columns, lookup columns and evaluations.
Class vectors use the channel coordinates alone. Range lookups have value zero;
the saturation and final requantization lookups retain their actual table values.
These are binding fixtures, not complete LogUp or model proofs.

`verifier/crates/tv-core/tests/gadget_requant.rs` checks the model-derived ids,
source-point projections, all affine forms and every retained obligation, then
alters each source/key/value in turn. Its frozen cases passed before the producer
adapter was written. `crates/tessera-prover/tests/gadget_requant.rs` compares
actual requantization witness columns and constraint evaluations, and actual SiLU
table outputs, with the same independent data.

## One-hot relations

`onehot.json`, schema version 1, contains six honest and six cheating component
sumcheck batches, frozen before the verifier implementation. `onehot.py` starts
from integer indicator rows and evaluates public selectors and indices by direct
cube sums. Shapes include a singleton vocabulary and token domain, and padding
on both axes. Each alternative violates exactly one of booleanity, vocabulary
membership, row sum or token binding, including nonzero padded token rows.

The row-count and index reductions start at the incoming leaf token claim's
point `a`, with entering claims `psi(a)` and `token(a)`. Their final relations
are `eq(a,t)*h(v,t)` and `eq(a,t)*I(v)*h(v,t)`, where `I` is the public index
column. Booleanity and vocabulary padding use the stage's common zero-check
point and entering claim zero. The row relations have degree two; the zero
checks have degree three. The incoming token value is absorbed before the
stage-one challenges and remains a source-opening obligation at `a`. The batch
uses the existing stage-one labels and degree-three maximum, with no new
message schema. It sends four indicator evaluations in relation order.

A cheating oracle keeps the prescribed entering claims and supplies its actual
final source evaluations, so its component proof fails the final sumcheck check.
Each alternative also has an `omission` proof adapted to a verifier that replaces
just its failed obligation by the public tautology `claim*delta_0`. That proof
passes the omission control with true source claims and fails the full check.
Actual relation sums identify each alternative's single failed obligation. These
are not complete model proofs or a replacement for the final source openings.

`verifier/crates/tv-core/tests/gadget_onehot.rs` checks all 18 batches, their
individual final relations, the honest transcripts, canonical source debts and
exact shape errors. Altering a final evaluation either fails the sumcheck or
leaves a false source claim, including zero-cofactor cases. The producer adapter
in `crates/tessera-prover/tests/gadget_onehot.rs`, written after the independent
checks passed, generates actual OneHot columns and reproduces every CPU round and
final evaluation for six honest batches and six omission controls. The independent
verifier accepts the former and rejects the latter with the exact final-round error.
