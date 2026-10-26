# Integer range vectors

`generate.py` applies [RFC-0006 range analysis](../../../docs/rfcs/RFC-0006-compiler-ir.md#range-analysis) to the independent [RFC-0044 lowering vectors](../lowering/README.md). It was written before the verifier range implementation and imports no prover or verifier implementation.

`ranges.json` contains one inclusive integer interval per tensor id, or `null` for an unbounded interval. Python integers supply the arithmetic; an endpoint outside signed i128 makes an interval unbounded. The named relational facts identify the OneHot column-selection bound and the SoftmaxRow mass bound. All 24 lowering cases are included.

Run `python3 spec/vectors/ranges/generate.py` from the repository root. These vectors cover the Llama lowering, while primitive overflow and adversarial-domain checks require separate analyzer tests.

## The anchor circuits

`anchor.json` applies the same analysis, with [RFC-0066 'Ranges'](../../../docs/rfcs/RFC-0066-bits-and-sha256.md#ranges) and RFC-0006 'Committed graph inputs' ([#527](https://github.com/AbdelStark/tessera/issues/527)), to two kinds of graph. That part of `generate.py` was written from RFC-0006 and RFC-0066 alone, by an author who read neither implementation.

**`circuits`.** These are the SHA-256 graphs of [`../anchor-graphs/`](../anchor-graphs/README.md), by `name`, each with:
- `ranges`: one interval per tensor id;
- `checked`: how many `BitDecompose`, `BitFn` and `AssertEqual` nodes passed their domain checks. Every check passes.

The rows of a sum of `k` words are `[0, k]`, and its value `[0, (2^{32} − 1)·k]` exactly. A round's value shifted by `K_t` lies in `[K_t, (2^{32} − 1)·k + K_t]`, inside its `BitDecompose { 32, 3 }`'s width.

**`rules`.** These are small graphs in the format of `../lowering/`, with `inputs`, `nodes`, `constants` and `tensors`. Each breaks one rule once, or none, and has its `ranges` and its `verdict`:
- **The verdict.** It is `null` for a graph the analysis admits, `{"code", "node"}` for the first node in node order whose input leaves its domain, and `{"code", "input"}` for a committed graph input that the proving operations do not cover element by element.
- **The domains.**
  - `BitDecompose`: `[0, 2^{low + high})`, or a committed graph input, read directly or moved by `Reshape`, `Slice`, `Transpose`, `GatherPublic` or `Expand`, whose declared range holds that interval.
  - `BitFn`: `[0, 1]`, and not a committed graph input, read directly or moved.
  - `AssertEqual`: the centered field range, the open interval `(−(P − 1)/2, (P − 1)/2)` of RFC-0006's true-integer rule.
- **The codes.** An unbounded input is `ERR-COMPILE-WRAP`, and one outside its domain `ERR-COMPILE-RANGE`.
- **The constants** are `i64` values, as the IR's are (RFC-0006 'Operation set').
- **The cases.**
  - A byte declared `[0, 255]` under nine bits, read directly, reshaped and gathered.
  - A sum of two bytes under eight.
  - `BitFn` on a sum of bits, and on a committed input read directly and reshaped.
  - `AssertEqual` of an unbounded tensor, and of `[0, (P − 1)/2]`, just outside the centered range.
  - A committed input half read by a `Slice`, and one that nothing reads.
  - Graphs the analysis admits beside them: a committed input declared `[0, 511]` under nine bits; a byte transposed and gathered under eight; `ConstAdd(−1)` of `[1, 256]` under eight; 62 bits of `[0, 2^{62} − 1]`; `BitFn` on a committed input's bits; `AssertEqual` of `[−(P − 3)/2, (P − 3)/2]`, the centered range's edges; and a committed input covered by two `Slice`s.

**Who checks them.** [`crates/tessera-compiler/tests/anchor_vectors.rs`](../../../crates/tessera-compiler/tests/anchor_vectors.rs) builds each circuit with the compiler's builder: `analyze` gives every tensor its interval, and `settle` writes it into the graph. It builds each small graph through the IR's builder, and `analyze` gives its verdict, and its ranges where it returns them.
