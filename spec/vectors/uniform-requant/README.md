# Common-family Requant selection

[generate.py](generate.py) derives [cases.json](cases.json) from
[RFC-0076](../../../docs/rfcs/RFC-0076-uniform-requant-families.md), RFC-0006,
RFC-0024 and RFC-0028 using Python integers. It was written and frozen before
either production implementation of RFC-0076, without reading the prover
selector. It shares no code with either workspace or the quantization package.
Split points are enumerated by table cost with the prescribed tie rule.

The cases pin legacy preservation, ordered class shifts, widening, mixed low
presence, the virtual top range, exact request order and rejection precedence.
They cover the four-class 62-bit wrap boundary, singleton groups and `k=1`.
These are selection vectors, not full-proof or global soundness evidence.

Four focused boundary inputs were added after the initial oracle freeze without
changing its algorithm: packed-class wrap, mixed low presence, the minimum Split
envelope and refusal of a Wide envelope beyond 62 bits.
