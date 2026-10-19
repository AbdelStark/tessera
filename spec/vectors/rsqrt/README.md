# Reciprocal-square-root brackets

`generate.py` uses Python integer division and `math.isqrt`, with no production
imports. Run `python3 spec/vectors/rsqrt/generate.py` or `cargo xtask gen-vectors`.

`roots.txt` has eight whitespace-separated integer columns: dimension, precision,
epsilon, selected input u, public real-row selector, root, lower product and upper
product. Its 60 cases include 8B/405B norm parameters, a root of 2^32-1, the
largest 55-bit input and padded rows. `tessera-gadgets/tests/rsqrt.rs` checks the
root and both products through the constraint descriptions. These are per-row
conformance checks, not whole-model proofs (RFC-0009 Lemma 7).
