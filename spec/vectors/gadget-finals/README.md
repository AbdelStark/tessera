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

The issue's implementation adds a verifier consumer before comparing with the
prover, and a consumer in the prover workspace before the issue is published.

## Decompositions

`decomposition.json`, schema version 1, contains 248 cases and 992 integer
witnesses: every NonNeg width 1–62 with limb widths 1, 8, 16 and 32. Zero,
one, the last admissible integer and an interior integer have their low limbs
and their virtual top recorded. The final range width is narrowed independently
of the policy limb width. Direct cube sums give every value and limb evaluation
at a non-Boolean point. This includes the 55-bit `(16,16,16,7)` root input and
all 1–32-bit byte decompositions of roots. These affine relations still owe
lookup proofs and source bindings; the fixture does not replace either.
