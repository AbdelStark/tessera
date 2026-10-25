# RFC-0066: Bits in the IR, and the SHA-256 Circuit

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends:
  - [RFC-0006](RFC-0006-compiler-ir.md#operation-set) 'Operation set': three operations join the closed set, `BitDecompose`, `BitFn` and `AssertEqual`, and `Bitwise` is AES's alone;
  - RFC-0006 'Range analysis': the three operations' ranges and integrality rules;
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations': booleanity, recomposition, bit-padding and equality zero-checks, and `BitFn`'s products;
  - [RFC-0015](RFC-0015-transcript-anchor.md#circuits) 'Circuits' steps 3 and 4: how the SHA-256 circuit is built, its cost, and the binding's check;
  - [RFC-0035](RFC-0035-padded-rows.md#zero-check-points) 'Zero-check points': the new zero-checks run in stage 1.
- Origin: [#143](https://github.com/AbdelStark/tessera/issues/143)

## Summary

RFC-0015 step 3 proves SHA-256 over committed bits: booleanity in the batched zero-check, `Ch`, `Maj`, `Σ` and `σ` as low-degree polynomials, and additions modulo `2^{32}` with committed carries. The IR could not express it:
- it has no committed bit;
- it cannot state an equality between two tensors;
- its interval range analysis bounds `a + b − 2ab` below by `−2`, so a sum of XORed bits fails the true-integer rule.

This RFC adds three operations:
- **`BitDecompose { low, high }`**, a gadget. It commits the bits of a true integer in two parts, `low` result bits and `high` carry bits. Booleanity and recomposition are zero-checks, and the bits are the input's unique binary representation.
- **`BitFn { f }`**, virtual. XOR of two or three bits, AND, `Ch` and `Maj`, each its multilinear polynomial, with range `[0, 1]`.
- **`AssertEqual`**, a relation with no output. Two tensors are equal, checked at a random point.

It then specifies the SHA-256 circuit over them: FIPS 180-4's compression in a bit-major layout, the message assembled from public and committed bits, and the final words bound to the public digest. One 64-byte block commits 6,376 bits beside its message's bits. A frame of `n` tokens hashes `63 + 4n` bytes. The key commitment hashes 53 bytes in one block, and its `BitDecompose` of `k_s`'s bytes is also the range check the AES circuit's key needs (#142).

## Motivation

- **RFC-0015 step 3** asks for bits, and RFC-0006's op table listed SHA-256 under `Bitwise`, whose byte lookups cannot rotate a word by a number of bits that is not a multiple of 8. A byte route would need new table kinds for bit splits and byte carries.
- **An equality the IR can state** binds the computed digest to the public one and the key commitment to the session's.
- **The AES circuit's inputs** (#142) must be bytes, by their producers. `BitDecompose` of a committed byte is that range check.

## Goals

- SHA-256 of any length, its padding public, bound to a public digest, in the IR.
- Every committed bit is the unique value the computation gives (RFC-0009 Lemma 0's rule).
- Committed bits per token at long frames within RFC-0015's estimate of about 1,000. A frame's two blocks at least make short frames cost more per token ('For the anchor').

## Non-Goals

- **`tessera-protocol/2`.** The new tensors follow RFC-0013's rules like any other. A committed bit is advice, its claims masked. `BitFn`'s values are virtual and claimed at stage points. Their zero-knowledge argument is #145's, with every other tensor's.
- **Wiring into a batch's proof.** One circuit per frame and session, its inputs from the log and the session, is #146's, #526's and #163's, as for AES.

## Proposed Design

### `BitDecompose { low, high }`

```rust
BitDecompose { low: u8, high: u8 },   // x → b [.., low], h [.., high]; high may be 0, with no h
```

- **Semantics.** For a true integer `x ∈ [0, 2^{low + high})`, `b_i = ⌊x / 2^i⌋ mod 2` for `i < low`, and `h_j = ⌊x / 2^{low + j}⌋ mod 2` for `j < high`. The outputs have `x`'s shape with one more axis, the bit index, innermost.
- **Kind.** Gadget-backed: both outputs are committed advice, of range `[0, 1]`.
- **Integrality.** `x` is a true integer with range inside `[0, 2^{low + high})`, and `low + high ≤ 62`, so that every sum below stays under `2^{62}`.
  - **A committed graph input** may hold any value of `Fp`, and the relations below prove it lies in `[0, 2^{low + high})` (Lemma 1's converse). That holds when `BitDecompose` reads it directly, or moved unchanged by `Reshape`, `Slice`, `Transpose`, `GatherPublic` or `Expand`, at the positions it reads.
    - Its other readers trust its declared range, so that range must hold all of `[0, 2^{low + high})`; otherwise `ERR-COMPILE-RANGE`.
    - Without that rule, a key byte declared `[0, 255]` under `BitDecompose { 8, 1 }` could hold `256 + k'`, and a `Bitwise8` lookup that takes it as the low operand of `a + 2^8·b` would admit two witnesses.
  - **Elsewhere**, range analysis takes a committed advice input's declared range as given: through any other operation, and at the positions a `Slice` or a gather leaves out, that range is a precondition another operation of the graph must prove. RFC-0006 'Committed graph inputs' states which operations prove it, and range analysis refuses a graph that leaves an element of a committed input unproven (#527).
- **Relations** (RFC-0010 'Relations'):
  - **Booleanity**, one `ZeroCheck` per output: `0 = Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))`, as `OneHot`'s, and the same for `h`.
  - **Recomposition**, one `ZeroCheck`: `0 = Σ_y eq(z, y)·x̃(y) − Σ_{(y, i)} eq(z, y)·w(i)·b̃(y ‖ i) − Σ_{(y, j)} eq(z, y)·2^{low}·w(j)·h̃(y ‖ j)`, with `w(i) = 2^i` on real bits and 0 on padded ones. Its first sum is `x̃(z)`, a claim on `x`. The others are sumchecks of degree 2, `w` public: the verifier evaluates `w̃(r) = Σ_{i < n} 2^i·eq(r, i)`, at most 62 terms, or `Π_k (1 − r_k + r_k·2^{2^k})` when the axis's `n` bits are a power of two.
  - **Padding.** A bit axis pads to the next power of two, 4 for a carry of 3. A padded slot has weight 0 in recomposition, so booleanity alone leaves it 0 or 1, two witnesses. One `ZeroCheck` per output whose `n` bits are not a power of two pins those slots to 0, as `OneHot`'s vocabulary check pins its padded entries (RFC-0009 'OneHot'): `0 = Σ_x eq(z, x)·(1 − LT(x_bit, n))·b̃(x)`, over the bit axis's coordinates `x_bit`, `LT` public.
  - **Padded rows.** The other axes' padded positions hold what the operation computes from their padded inputs, as every gadget's do (RFC-0035 'What a padded row holds').
  - **Completeness there is the graph's.** Range analysis bounds real positions only, so a graph must compute values in `[0, 2^{low + high})` at padded positions too. `BitDecompose(ConstAdd(−1)(p))`, with `p` declared `[1, 256]`, gives `−1` there, which no bits recompose. The SHA-256 circuit's padded inputs are zero bits and zero bytes, and its sums of bits and constants stay within their carries, so its padded positions decompose.
  - **A committed input's padding.** A committed graph input pads to powers of two like every tensor. Nothing computes its padded positions, and booleanity and recomposition hold for any byte there. So each `BitDecompose` that reads a committed graph input, directly or moved, has one padding check on that input when its shape has padded positions, two decompositions of one input two checks: `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·x̃(x)`, RFC-0035's form, `ψ` 1 on its real positions. Everything the circuit computes from them is then determined.

**Lemma 1 (unique bits).** If `x` is an integer in `[0, 2^{low + high})`, exactly one pair `(b, h)` with entries in `Fp` passes booleanity, recomposition and the bit-padding check: `x`'s binary digits, and 0 at padded slots.
- **Proof.**
  - Booleanity makes each entry 0 or 1, as integers.
  - The recomposition `Σ_i 2^i b_i + 2^{low}·Σ_j 2^j h_j` is then an integer below `2^{low + high} ≤ 2^{62} < P`, and equals `x` modulo `P`.
  - Both sides lie in `[0, P)`, so they are equal as integers, and binary representation is unique.
  - The bound is the field's. Over F65537, the test field of calibration (RFC-0002, 07 §4), it is `low + high ≤ 16`, which a calibration circuit must keep, since the IR's rule is Goldilocks's 62.
- **The converse.** For a committed `x` of any value in `Fp`, a passing `(b, h)` forces `x` to equal that integer in `[0, 2^{low + high})`. So `BitDecompose` is its input's range check.

### `BitFn { f }`

```rust
BitFn { f: BoolFn },   // f(a, b[, c]) elementwise; inputs and output of one shape
pub enum BoolFn { Xor, Xor3, And, Ch, Maj }
```

| `f` | Inputs | Polynomial | Degree |
|---|---|---|---|
| `Xor` | `a, b` | `a + b − 2ab` | 2 |
| `Xor3` | `a, b, c` | `a + b + c − 2ab − 2ac − 2bc + 4abc` | 3 |
| `And` | `a, b` | `ab` | 2 |
| `Ch` | `e, f, g` | `ef + g − eg`, FIPS 180-4's `(e ∧ f) ⊕ (¬e ∧ g)` | 2 |
| `Maj` | `a, b, c` | `ab + ac + bc − 2abc` | 3 |

- **Kind.** Virtual. Each is the multilinear polynomial that agrees with its Boolean function on `{0, 1}`, and its relation is `Elementwise` (RFC-0010): `Z̃(r) = Σ_x eq(r, x)·Σ_t w_t·Π_{f ∈ t} I_f(x)`, of degree at most 4 per variable.
- **Range.** Every input's range lies in `[0, 1]`, so it holds 0 or 1 as an integer. The output's range is `[0, 1]`: on Boolean inputs each polynomial is its Boolean function. An input outside `[0, 1]` is `ERR-COMPILE-RANGE`.
  - So is a committed graph input, read directly or moved unchanged: its declared `[0, 1]` proves no bit. The circuit's bits come from `BitDecompose` or are public.
- `¬a` needs no operation: `ConstAdd(1)` of `ConstMul(−1)`, whose interval is `[0, 1]`.

### `AssertEqual`

```rust
AssertEqual,   // a, b → none: a = b elementwise
```

- **Kind.** A relation with no output, as `Argmax`'s certificate has none.
- **Integrality.** Both inputs are true integers with ranges inside the centered field range. Equality in `Fp` is then equality as integers.
- **Relation.** One `ZeroCheck` over the tensors' real positions: `0 = Σ_x eq(z, x)·ψ̃(x)·(ã(x) − b̃(x))`, of degree 3, where `ψ(x) = Π_a LT(x_a, n_a)` is 1 on the positions inside the shape `(n_a)` and 0 on padded ones. The verifier evaluates `ψ̃` in `O(log)` time (RFC-0010 'Multilinear extensions'). The sumcheck leaves a claim on each of `a` and `b`; a public one the verifier evaluates.
  - The selector is needed: a padded position holds what the computation gives there, such as a digest of padded messages, while a public tensor pads with zeros. Without it, an honest prover of `M` messages fails unless `M` is a power of two.

### Where the zero-checks run

Every `ZeroCheck` runs in stage 1 (RFC-0010 'Batching within a stage'), at the point RFC-0035 'Zero-check points' draws before `γ_1`.
- **Order.** The new instances follow stage 1's existing zero-checks, in node order. Each `BitDecompose` contributes:
  1. the padding check of the committed graph input it reads, directly or moved, when that input has padded positions;
  2. its booleanity checks;
  3. its bit-padding checks;
  4. its recomposition.
- **Virtual claims.** A claim one leaves on a virtual tensor, such as recomposition's `x̃(z)` or `AssertEqual`'s `ã(z)`, is a stage-1 claim. RFC-0037 reduces it through the tensor's producer in later stages, as it does any other.
- **The calculator.** Each instance is one entry `n_i` of RFC-0014's `zero_checks`, the variables of its point `z`: `x`'s for recomposition, the output's for booleanity and bit padding, the committed input's for its padding check, and the two tensors' for `AssertEqual`.
- **Messages.** Their messages, labels and batching are open (OQ-0066-1).

### The SHA-256 circuit

FIPS 180-4 §6.2 over `M` messages of one block count, in the IR. It is the circuit's builder's (`tessera_compiler::anchor`, #143's implementation).

**Layout.**
- **A word** of the `M` messages is a bit tensor `[32, M]`: row `i` is the word's bit of weight `2^i`.
- **`ROTR^r`** is `GatherPublic` with row `i` taken from row `(i + r) mod 32`.
- **`SHR^r`** is the same gather, times a public `[32]` mask that is 1 on rows `i < 32 − r`, through `Mul { broadcast: Trailing }`. The masks of `SHR^3` and `SHR^{10}` are the rows of one public `[2, 32]` input.
- **The integer value of rows** `r_i`, a `[32, M]` tensor, is `Σ_i 2^i·r_i`, built in five levels. Each level gathers the even and the odd rows and adds `2^{2^ℓ}` times the odd ones to the even ones, `Add(even, ConstMul(odd))`, halving the rows. Rows in `[0, k]` reach `[0, (2^{2^{ℓ+1}} − 1)·k]` at level `ℓ`, and `[0, (2^{32} − 1)·k]` at the last, exactly.
  - Range analysis bounds a tensor by one interval. `Sum { axis: 0 }` of the rows times the weights `2^i` would get `[0, 32·2^{31}·k]`, 16 times too wide, and fail `BitDecompose`'s width.

**The functions.**
- `Σ0`, `Σ1`, `σ0` and `σ1` are `Xor3` of their three rotations and shifts.
- `Ch` and `Maj` are `BitFn`s.

**The additions.** Each modulo-`2^{32}` sum of `k` words adds their bits row by row (`Add`), into rows in `[0, k]`, takes the rows' value, adds its public constant (`ConstAdd`), then applies `BitDecompose { low: 32, high: c }`. The result word is `b`, transposed to `[32, M]`, and `h` holds the carry. `c` is the least width that bounds the sum:

| Sum | Terms | `c` |
|---|---|---|
| Message schedule `W_t`, `16 ≤ t < 64` | `σ1(W_{t−2}) + W_{t−7} + σ0(W_{t−15}) + W_{t−16}` | 2 |
| Round `e` | `d + h + Σ1(e) + Ch(e, f, g) + K_t + W_t` | 3 |
| Round `a` | `h + Σ1(e) + Ch(e, f, g) + K_t + W_t + Σ0(a) + Maj(a, b, c)` | 3 |
| Chaining | `H_j + (a, …, h)_j` | 1 |

The two round sums share their first rows, `h + Σ1(e) + Ch(e, f, g) + W_t`. Then `e`'s adds `d`, `a`'s adds `Σ0(a)` and `Maj(a, b, c)`, and each adds `K_t` as its constant.

**The message.**
- **Its bytes**, padding included, are public, except those the prover commits: the ciphertext's, a committed key's or salt's.
- **A public byte** enters as public bits.
- **A committed byte** enters through `BitDecompose { low: 8, high: 0 }` of that byte. The circuit takes its committed bytes as one `[K, M]` tensor, `K` per message, in offset order. Over several frames, the anchor gathers that tensor from the AES circuit's `[16, B]` ciphertext with `Transpose`, `Reshape` and `GatherPublic` (#146).
- **The message's bits** are `Xor(message, Mul(gather(c), place))`, a `[512·N, M]` tensor of words, 16 per block:
  - `message` is the public bits of the padded messages, and **0 at every committed byte's bits**. The verifier builds it so, from the public bytes alone.
  - `c` is the committed bits, row `8k + i` holding bit `i` of committed byte `k`. The gather puts each one at its word position, and row 0 at the others.
  - `place` is a public `[512·N]` mask, 1 on the committed bytes' rows. `Mul { broadcast: Trailing }` zeroes the gathered rows elsewhere.

  The supports are disjoint, so XOR is the sum, and the range stays `[0, 1]`.
- **Byte order.** FIPS 180-4's words are big-endian: bit `i` of word `j` is bit `i mod 8` of byte `4j + 3 − ⌊i/8⌋`.

**The output.**
- **Initial values.** The first block starts from a public `[256, M]` input, `H^{(0)}`'s bits for every message.
- **Later blocks** start from the previous block's chaining words. The builder returns the chaining words after every block.
- **The digest.** The last block's chaining words, as bits, are bound by `AssertEqual` to the public digest's bits, 256 per message. The digest is one public `[256, M]` tensor, word `j` on rows `32j` to `32j + 31`, and each word's `AssertEqual` reads its rows through a `Slice`.

**The graph.** Its inputs are, in this order:
- `message`, public bits `[512·N, M]`;
- `start`, public bits `[256, M]`;
- `masks`, public bits `[2, 32]`, the masks of `SHR^3` and `SHR^{10}`;
- when bytes are committed, `committed`, advice declared `u8` in `[0, 255]`, `[K, M]`, and `place`, public bits `[512·N]`.

Each bit input is declared `u1` in `[0, 1]`. Every `GatherPublic`, `ConstMul` and `ConstAdd` has a constant of its own, declared just before its node and never shared, so the constant table holds them in node order. The digest, public bits `[256, M]`, is declared after the circuit's nodes, before its binding's. The nodes are in this order, which fixes the zero-checks' order:
1. each `SHR` mask: a `GatherPublic` of its row of `masks`, then a `Reshape` to `[32]`;
2. when bytes are committed, the message's bits: the bytes' `BitDecompose { 8, 0 }`, a `Transpose` to `[K, 8, M]`, a `Reshape` to `[8K, M]`, the gather, the `Mul` and the `Xor`;
3. the eight initial words, each a `Slice` of `start`;
4. per block:
   - its sixteen words, each a `Slice` of the message's bits;
   - for `t = 16, …, 63`: `σ1(W_{t−2})`, `σ0(W_{t−15})`, then the sum;
   - for each round: `Σ1(e)`, `Ch`, `Σ0(a)`, `Maj`, the shared rows, `e`'s sum, `a`'s sum;
   - the eight chaining sums, `a` to `h`;
5. per digest word, its `Slice` and `AssertEqual` of the computed word, then the digest's rows.

A function is its rotations' gathers, each `SHR` a gather and a `Mul`, in its definition's order, then its `BitFn`. A sum is:
- its rows' `Add`s, left to right;
- the value's five levels, each an even gather, an odd gather, the odd rows' `ConstMul` and the `Add` of the even rows and that product, then a `Reshape` to `[M]`;
- its `ConstAdd` when the constant is not 0;
- the `BitDecompose`, then the `Transpose` of its result bits.

`tessera_compiler::anchor::sha256` builds exactly this graph.

**Committed bits per block.**

| Item | Bits |
|---|---|
| Message schedule: 48 words of 32 + 2 | 1,632 |
| Rounds: 64 × 2 words of 32 + 3 | 4,480 |
| Chaining: 8 words of 32 + 1 | 264 |
| Per block | 6,376 |
| Each committed message byte | 8 |

Bit axes pad to powers of two, 4 for the round carries, so the commitment holds a few more zeros, which the padding zero-check pins.

**For the anchor (RFC-0015 step 3).**
- **A frame** hashes `header ‖ ciphertext`: 41 public bytes and `22 + 4n` committed ones, `⌈(63 + 4n + 9)/64⌉` blocks.
  - Its ciphertext bytes are the AES circuit's outputs, true integers in `[0, 256)`, so their `BitDecompose { 8, 0 }` adds 8 bits per byte.
  - A frame of `n` tokens commits `6,376·⌈(72 + 4n)/64⌉ + 8·(22 + 4n)` bits: 12,928 at `n = 0`, since every frame takes two blocks at least.
  - Per token that is below RFC-0015's estimate of about 1,000 from `n = 20`, 632.5 at `n = 64`, and about `6,376/16 + 32 ≈ 430` at long frames.
- **The key commitment** hashes `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k`, 53 bytes, in one block.
  - `k_s`'s 16 bytes and `salt_k`'s are committed, and their `BitDecompose { 8, 0 }` proves them bytes (Lemma 1).
  - Those `k_s` bytes are the AES circuit's key input, so the precondition #142 states, that its key be bytes by its producer, is met here.
  - The anchor's graph (#146) must range-check every committed byte its lookups read this way, or by another operation RFC-0006 'Committed graph inputs' recognizes: a committed input's declared range is otherwise a claim nothing proves, and range analysis refuses the graph.
  - The digest is bound to the session's public `key_commitment`.
- **The sampler key** (RFC-0061) is hashed the same way, from `"tessera-sampler-key/v1"`.

### Soundness

- **Every committed bit is unique.** By Lemma 1, given its input. Its input is a true integer the range analysis bounds, or a committed byte the lemma ranges, whose padded positions its padding check pins.
- **Every `BitFn` value** is a function of its inputs, virtual and never committed.
- **The computation.** With every committed bit unique and every function exact on Boolean inputs, the circuit's chaining words are FIPS 180-4's. `AssertEqual` makes them the public digest's, except with the zero-checks' error.
- **The zero-checks** err each with probability at most `n_i/|Fp2|`, the calculator's `zero_checks` row.
- **A false carry.** A carry bit other than the true one breaks recomposition, since the result bits then cannot make up the sum, unless a result bit leaves `{0, 1}` and breaks booleanity. Either is a failing `ZeroCheck`: stage 1's sumcheck rejects it, or a later evaluation does, as the prover lies.

### Ranges

RFC-0006 'Range analysis' adds:
- `BitDecompose`'s outputs are `[0, 1]`, and its input must lie in `[0, 2^{low + high})`, or be a committed graph input whose declared range holds that interval;
- `BitFn`'s output is `[0, 1]`, and its inputs must lie in `[0, 1]` and not be committed graph inputs, read directly or moved;
- `AssertEqual` has no output, and its inputs must lie in the centered field range.

## Alternatives Considered

1. **Byte lookups (`Bitwise8`) for SHA-256.** Rejected: rotations by 2, 6, 7, 11, 13, 17, 18, 19, 22 and 25 bits split bytes, which needs new table kinds for shifts and for byte carries. Their cost depends on those tables, and the IR has none of them.
2. **Generic `Add`, `Sub` and `Mul` for the Boolean functions.** Rejected: their interval transfer bounds `a + b − 2ab` below by `−2`, and a sum of such values then fails the true-integer rule at its `BitDecompose`. `BitFn` carries its range.
3. **One `BitDecompose { bits }` output with the carry inside.** Rejected: 35 bits pad to 64, nearly doubling the rounds' committed bits. Two outputs pad 32 and 4.
4. **A `Sha256Compress` gadget op.** Considered. It would hide the circuit in one op, with its own relations and lemma. The three operations here are smaller, each with one lemma or none, and they serve the key commitment and AES's range checks too.

## Drawbacks

- The IR's closed operation set grows by three, and the verifier's lowering must mirror them (#150, #159).
- Stage 1 gains 680 zero-check instances per block: 368 booleanity checks, 128 bit-padding checks for the round carries and 184 recompositions. A batch of frames holds hundreds of blocks, so the schedule should stack instances of one kind (OQ-0066-1).
- A frame's SHA-256 is a long sequential graph: 6,032 nodes per block, 64 per round. The value trees hold 3,864 of them, 21 for each of the block's 184 sums. Its size grows with the block count, not with `M`.

## Migration / Rollout

1. **#143:**
   - the three operations in `tessera-ir`, with their rules;
   - the range analysis;
   - the CPU engine's evaluation, which refuses an integer outside its decomposition's width or a `BitFn` input that is not a bit as `ERR-USAGE`, and an `AssertEqual` whose tensors differ as `ERR-WITNESS-ANCHOR`;
   - the SHA-256 builder in `tessera_compiler::anchor`, tested against NIST CAVP's short and long messages.
2. **The schedule's relations** for the new zero-checks and `BitFn`, with the prover's and the verifier's: follow-up work after #143 (#526), before the anchor's wiring (#146, #163).
3. **#150** checks the anchor's final relations with them.

## Testing Strategy

- **Known answers.** NIST CAVP `SHA256ShortMsg` and `SHA256LongMsg` (`spec/vectors/sha256/`): the circuit's digest bits equal each `MD`, on the CPU engine. The short messages share one graph of two blocks. The long ones, of 3 to 101 blocks, run block by block through the initial-value input. A graph of three blocks checks the chaining inside one graph.
- **An independent interpreter.** The compiler's test interpreter, whose Boolean functions are logic rather than these polynomials, computes FIPS 180-4's `"abc"` and keeps every tensor in its range.
- **Lemma 1.** A wrong carry bit, a wrong result bit and a non-Boolean bit each break a zero-check's identity on the witness. A committed byte of value 256 has no passing bits. A committed byte declared `[0, 255]` under nine bits is refused, read directly, reshaped or gathered, and so is a committed input as a `BitFn` input.
- **Frames.** Committed bytes that cross into a second block give `sha2`'s digest.
- **Binding.** A digest one bit off fails its `AssertEqual`.
- **The key commitment.** The circuit's digest of a session's key matches `tessera_anchor`'s direct `key_commitment`.
- **Ranges.** The range analysis passes the circuit, and refuses a `BitFn` on a non-bit input and a `BitDecompose` of an input that may exceed its width.

## Open Questions

- **OQ-0066-1: the zero-checks' messages.** The schedule's relations (#526) settle:
  - where each claim the new zero-checks leave is sent, and under which label;
  - how recomposition's two sums, over `n_y + log2 low` and `n_y + log2 high` variables, are batched with `x̃(z)`;
  - what `AssertEqual` sends when neither side is public;
  - whether `BitDecompose` takes a `GadgetId` in gadget selection;
  - the reject reason of each failing check;
  - stacking instances of one kind.

  They also amend RFC-0010 'Batching within a stage', RFC-0037 'Stages and their instances' and RFC-0014's counts. Until then, gadget selection refuses the three operations (`ERR-QUANT-ARCH`), so no proof contains them.
- **OQ-0066-2: `GatherPublic`'s relation.** `ROTR`, `SHR` and the message's assembly reduce claims through `GatherPublic`, as AES's `ShiftRows` does. A public-kernel sumcheck `ỹ(r_row ‖ r) = Σ_k G(r_row, k)·x̃(k ‖ r)`, with `G(r, k) = Σ_{j : idx_j = k} eq(r, j)`, of degree 2, is the candidate for #526.

## References

- FIPS 180-4, *Secure Hash Standard*, §4.1.2, §5.1.1, §5.3.3 and §6.2.
- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0059](RFC-0059-frame-headers-and-nonces.md), [RFC-0061](RFC-0061-sampling-randomness.md).
- [#142](https://github.com/AbdelStark/tessera/issues/142), [#143](https://github.com/AbdelStark/tessera/issues/143).
