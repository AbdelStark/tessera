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

## Model-derived root bindings

`roots.json`, schema version 1, records 45 root sites from nine public micro-model
variants, including a zero maximum, widths 1/8/16/32, a constant above Goldilocks,
and nonzero padded root, residual and feature alternatives.
`roots.py` uses independent lowering/catalog oracles, whole-integer root and
comparison witnesses, and direct cube sums. Each site retains every canonical
source, range request (including repeats and virtual tops), comparison array and
raw product target. The first site of each variant has a degree-four component
batch connecting those targets to root/input digits, plus its padding zero check.

The input is reduced as `u=S+epsilon*psi` only for a graph-derived `SumSquares`
over a committed residual with the canonical zero padding. At Boolean padded
rows `S=0` only once the explicit residual-padding obligation has been discharged.
The component batch includes its degree-three zero check over feature/token
coordinates, with a verifier-derived selector for both axes. Every derived claim
on `S` still needs its graph reduction and the residual's source openings.

All root product target claims come from the already bound comparison digits
and carries, without a new message. Component transcripts absorb the leaf source
evaluations before stage-one challenges. The padded-root variant passes all
range, decomposition, comparison and product relations, and fails the padding
check. Residual and feature alternatives pass the bracket, product and root-padding
checks but fail the residual-padding check. Each additional `omission` proof passes
when only its failed check is disabled. Mixed instance dimensions exercise the
existing front-padding scale and final-point suffix rule.
These fixtures do not define a complete model stage or layer-template stacking.

`wrong_root` is a consistently encoded real-row alternative: every request passes
its actual Range table, but the lower comparison ends with borrow one. The
model-derived binder rejects it with the owning gadget id. Both workspaces check
the range-membership control; the honest CPU root differs from the alternative.

`verifier/crates/tv-core/tests/gadget_root.rs` verifies all 45 model-derived sites,
5,885 requests, retained source/product debts, exact local errors, product/padding
batches and the two alternatives. Coherently changing a witness and its key tests
the comparison checks independently of the source-equality checks. The producer
adapter in `crates/tessera-prover/tests/gadget_root.rs`, written after the verifier
tests passed, checks actual root witness columns and reproduces every CPU round
and final evaluation for six honest batches and three padding-omission controls.

## Softmax model bindings

`softmax.json`, schema version 1, has 14 model-derived cases: eight honest sites
and six isolated alternatives, each with an adapted omission proof. It covers
zero-dimensional scalar relations, three real heads padded to four, padded query
rows, 23-bit exponential precision, and NonNeg limb policies 1, 8, 16 and 32.
The full padded score square is always represented.

Python integer maxima, table values and bracket slacks determine every witness.
The lookup key and forest leaf use different incoming points; each reduces
`C*(c-score)` by summing the actual product over Boolean vertices. The row sum
uses the exponential extension at half on each key coordinate, scaled by the
padded key count. Both NonNeg decompositions include their narrowed virtual top.
The oracle emits mixed-domain degree-three component proofs with front padding.

The alternatives change one obligation: a masked lookup key with the same zero
exponential, a masked forest leaf, a reciprocal, a padded maximum, a padded
reciprocal, or a padded-head maximum. Each omission proof replaces only the
failing relation with a public tautology, passes that weakened control, and fails
the full checker while every canonical final source claim remains true.

`verifier/crates/tv-core/tests/gadget_softmax.rs` consumes every proof and source
claim; its tests passed before the prover's Softmax implementation was read.
`crates/tessera-prover/tests/gadget_softmax.rs` compares the eight honest sites
with actual `SoftmaxRow` witness/lookup generation and reproduces all rounds,
final evaluations and transcript states for eight honest and six omission batches
using the actual CPU proof engine. The latter deliberately assembles the component
relations; it is not a complete model driver, maximum-forest proof or LogUp proof.
The full verifier must still check those earlier proofs and discharge all returned
source claims, including virtual scores through the graph schedule.

## Argmax final relations

`argmax.json`, schema version 1, contains six honest cases and six alternatives.
Each case has a degree-three key/selected-logit batch followed by a degree-two
strict-suffix batch consuming the first batch's actual suffix evaluation. The
second batch binds the suffix to the same next-token indicator role as the
selected-logit batch. It does not use a pointwise product of input extensions.

Integer fixtures cover scalar domains, padded vocabulary and rows, all-negative
real logits beside zero padding, lowest-index ties and logits near both strict
`2^28` boundaries. Every alternative has valid OneHot inputs and valid Range29
keys: a false key for a later tie, a false suffix hiding a later tie, a raised
maximum, a nonzero vocabulary-padding key, a padded-row maximum, or a suffix in
the wrong direction. Each adapted omission proof passes its weakened control
with true source-role claims and fails the full checker at the appropriate stage.

`verifier/crates/tv-core/tests/gadget_argmax.rs` passed before the producer's
Argmax implementation was read. The producer consumer in
`crates/tessera-prover/tests/gadget_argmax.rs` compares actual Argmax and OneHot
columns and reproduces every round, final evaluation and transcript state for
six honest and six omission chains using the CPU proof engine. These are
component proofs. The certificate binder still owes canonical maximum advice,
the Range29 table, the model logits and the recorded next-token OneHot; a typed
role claim alone is not a commitment opening or complete model verification.
