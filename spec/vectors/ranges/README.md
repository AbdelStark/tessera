# Integer range vectors

`generate.py` applies [RFC-0006 range analysis](../../../docs/rfcs/RFC-0006-compiler-ir.md#range-analysis) to the independent [RFC-0044 lowering vectors](../lowering/README.md). It was written before the verifier range implementation and imports no prover or verifier implementation.

`ranges.json` contains one inclusive integer interval per tensor id, or `null` for an unbounded interval. Python integers supply the arithmetic; an endpoint outside signed i128 makes an interval unbounded. The named relational facts identify the OneHot column-selection bound and the SoftmaxRow mass bound. All 24 lowering cases are included.

Run `python3 spec/vectors/ranges/generate.py` from the repository root. These vectors cover the Llama lowering, while primitive overflow and adversarial-domain checks require separate analyzer tests.
