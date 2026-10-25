# RFC-0066: Bits in the IR, and the SHA-256 Circuit

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Supersedes:
  - [RFC-0006](RFC-0006-compiler-ir.md#operation-set) 'Operation set': three operations join the closed set, `BitDecompose`, `BitFn` and `AssertEqual`, and `Bitwise` is AES's alone;
  - [RFC-0015](RFC-0015-transcript-anchor.md#circuits) 'Circuits' step 3: how the SHA-256 circuit is built.
- Amends:
  - RFC-0006 'Range analysis': the three operations' ranges and integrality rules;
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations': booleanity, recomposition and equality zero-checks, and `BitFn`'s products;
  - [RFC-0035](RFC-0035-padded-rows.md#zero-check-points) 'Zero-check points': the stage of the new zero-checks;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) 'Soundness calculator': each new zero-check is one of `zero_checks`.
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

- **RFC-0015 step 3** asks for bits, and RFC-0006's op table listed SHA-256 under `Bitwise`, whose byte lookups cannot rotate a word by a number of bits that is not a multiple of 8. A byte route would need new table kinds for bit splits and byte carries, at about 660 lookups per token against about 430 committed bits per token here.
- **An equality the IR can state** binds the computed digest to the public one and the key commitment to the session's.
- **The AES circuit's inputs** (#142) must be bytes, by their producers. `BitDecompose` of a committed byte is that range check.

## Goals

- SHA-256 of any length, its padding public, bound to a public digest, in the IR.
- Every committed bit is the unique value the computation gives (RFC-0009 Lemma 0's rule).
- Committed bits per token within RFC-0015's estimate of about 1,000.

## Non-Goals

- **`tessera-protocol/2`.** The new tensors follow RFC-0013's rules like any other. A committed bit is advice, its claims masked. `BitFn`'s values are virtual and claimed at stage points. Their zero-knowledge argument is #145's, with every other tensor's.
- **Wiring into a batch's proof.** One circuit per frame and session, its inputs from the log and the session, is #146's and #163's, as for AES.

## Proposed Design

### `BitDecompose { low, high }`

```rust
BitDecompose { low: u8, high: u8 },   // x → b [.., low], h [.., high]; high may be 0, with no h
```

- **Semantics.** For a true integer `x ∈ [0, 2^{low + high})`, `b_i = ⌊x / 2^i⌋ mod 2` for `i < low`, and `h_j = ⌊x / 2^{low + j}⌋ mod 2` for `j < high`. The outputs have `x`'s shape with one more axis, the bit index, innermost.
- **Kind.** Gadget-backed: both outputs are committed advice, of range `[0, 1]`.
- **Integrality.** `x` is a true integer with range inside `[0, 2^{low + high})`, and `low + high ≤ 62`, so that every sum below stays under `2^{62}`. A committed input passes too, with any declared range, since the relations below prove it.
- **Relations** (RFC-0010 'Relations'):
  - **Booleanity**, one `ZeroCheck` per output: `0 = Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))`, as `OneHot`'s, and the same for `h`.
  - **Recomposition**, one `ZeroCheck`: `0 = Σ_y eq(z, y)·x̃(y) − Σ_{(y, i)} eq(z, y)·w(i)·b̃(y ‖ i) − Σ_{(y, j)} eq(z, y)·2^{low}·w(j)·h̃(y ‖ j)`, with `w(i) = 2^i` on real bits and 0 on padded ones. Its first sum is `x̃(z)`, a claim on `x`. The others are sumchecks of degree 3, `w` public.
  - **Padding.** The bit axes pad to the next power of two. Their padded positions join RFC-0035's padding zero-check, so they are zero.

**Lemma 1 (unique bits).** If `x` is an integer in `[0, 2^{low + high})`, exactly one pair `(b, h)` with entries in `Fp` passes booleanity and recomposition: `x`'s binary digits.
- **Proof.**
  - Booleanity makes each entry 0 or 1, as integers.
  - The recomposition `Σ_i 2^i b_i + 2^{low}·Σ_j 2^j h_j` is then an integer below `2^{low + high} ≤ 2^{62} < P`, and equals `x` modulo `P`.
  - Both sides lie in `[0, P)`, so they are equal as integers, and binary representation is unique.
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
- `¬a` needs no operation: `ConstAdd(1)` of `ConstMul(−1)`, whose interval is `[0, 1]`.

### `AssertEqual`

```rust
AssertEqual,   // a, b → none: a = b elementwise
```

- **Kind.** A relation with no output, as `Argmax`'s certificate has none.
- **Integrality.** Both inputs are true integers with ranges inside the centered field range. Equality in `Fp` is then equality as integers.
- **Relation.** One `ZeroCheck`: `0 = Σ_x eq(z, x)·(ã(x) − b̃(x))`, which is `ã(z) = b̃(z)`: a claim on each at `z`. A public `b` is evaluated by the verifier.

### Where the zero-checks run

RFC-0035 'Zero-check points' draws stage `k`'s point `z` before `γ_k`.
- **Booleanity** runs in stage 1, with `OneHot`'s and the padding zero-checks, since its operands are committed.
- **Recomposition and `AssertEqual`** read a virtual input. Each runs in the first stage after every relation that produces that input's claims has been placed, as any virtual claim does (RFC-0010 'Stages').
- **The calculator.** Each instance is one `n_i` of RFC-0014's `zero_checks`, its variables those of the zero-check's sum.

### The SHA-256 circuit

FIPS 180-4 §6.2 over `M` messages of one block count, in the IR. It is the circuit's builder's (`tessera_compiler::anchor`, #143's implementation).

**Layout.**
- **A word** of the `M` messages is a bit tensor `[32, M]`: row `i` is the word's bit of weight `2^i`.
- **`ROTR^r`** is `GatherPublic` with row `i` taken from row `(i + r) mod 32`.
- **`SHR^r`** is the same gather, times a public `[32]` mask that is 1 on rows `i < 32 − r`, through `Mul { broadcast: Trailing }`.
- **A word's integer value** is `Sum { axis: 0 }` of the word times the public `[32]` weights `2^i`, again through `Mul { broadcast: Trailing }`.

**The functions.**
- `Σ0`, `Σ1`, `σ0` and `σ1` are `Xor3` of their three rotations and shifts.
- `Ch` and `Maj` are `BitFn`s.

**The additions.** Each modulo-`2^{32}` sum is the integer sum of its terms' values and its public constants (`ConstAdd`), then `BitDecompose { low: 32, high: c }`. The result word is `b`, transposed to `[32, M]`, and `h` holds the carry. `c` is the least width that bounds the sum:

| Sum | Terms | `c` |
|---|---|---|
| Message schedule `W_t`, `16 ≤ t < 64` | `σ1(W_{t−2}) + W_{t−7} + σ0(W_{t−15}) + W_{t−16}` | 2 |
| Round `e` | `d + h + Σ1(e) + Ch(e, f, g) + K_t + W_t` | 3 |
| Round `a` | `h + Σ1(e) + Ch(e, f, g) + K_t + W_t + Σ0(a) + Maj(a, b, c)` | 3 |
| Chaining | `H_j + (a, …, h)_j` | 1 |

**The message.**
- **Its bytes**, padding included, are public, except those the prover commits: the ciphertext's, a committed key's or salt's.
- **A public byte** enters as public bits.
- **A committed byte** enters through `BitDecompose { low: 8, high: 0 }` of that byte.
- **A word that mixes the two** is the `Xor` of the public bits and the committed bits, each gathered into the word's positions and masked to zero at the other's. The supports are disjoint, so XOR is the sum, and the range stays `[0, 1]`.
- **Byte order.** FIPS 180-4's words are big-endian: bit `i` of word `j` is bit `i mod 8` of byte `4j + 3 − ⌊i/8⌋`.

**The output.**
- **Initial values.** The first block starts from `H^{(0)}`, public bits.
- **Later blocks** start from the previous block's chaining words.
- **The digest.** The last block's chaining words, as bits, are bound by `AssertEqual` to the public digest's bits, 256 per message.

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
  - Per token that is about `6,376/16 + 32 ≈ 430` committed bits at long frames, against RFC-0015's estimate of about 1,000.
- **The key commitment** hashes `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k`, 53 bytes, in one block.
  - `k_s`'s 16 bytes and `salt_k`'s are committed, and their `BitDecompose { 8, 0 }` proves them bytes (Lemma 1).
  - Those `k_s` bytes are the AES circuit's key input, so the precondition #142 states, that its key be bytes by its producer, is met here.
  - The digest is bound to the session's public `key_commitment`.
- **The sampler key** (RFC-0061) is hashed the same way, from `"tessera-sampler-key/v1"`.

### Soundness

- **Every committed bit is unique.** By Lemma 1, given its input. Its input is a true integer the range analysis bounds, or a committed byte the lemma ranges.
- **Every `BitFn` value** is a function of its inputs, virtual and never committed.
- **The computation.** With every committed bit unique and every function exact on Boolean inputs, the circuit's chaining words are FIPS 180-4's. `AssertEqual` makes them the public digest's, except with the zero-checks' error.
- **The zero-checks** err each with probability at most `n_i/|Fp2|`, the calculator's `zero_checks` row.
- **A false carry.** A carry bit other than the true one breaks recomposition, since the result bits then cannot make up the sum. That is a `ZeroCheck` failure, `Reject(Sumcheck)` at its stage.

### Ranges

RFC-0006 'Range analysis' adds:
- `BitDecompose`'s outputs are `[0, 1]`, and its input must lie in `[0, 2^{low + high})` or be committed;
- `BitFn`'s output is `[0, 1]`, and its inputs must lie in `[0, 1]`;
- `AssertEqual` has no output, and its inputs must lie in the centered field range.

## Alternatives Considered

1. **Byte lookups (`Bitwise8`) for SHA-256.** Rejected: rotations by 2, 6, 7, 11, 13, 17, 18, 19, 22 and 25 bits split bytes, which needs new table kinds for shifts and for byte carries. That would cost about 660 lookups per token, against about 430 committed bits here.
2. **Generic `Add`, `Sub` and `Mul` for the Boolean functions.** Rejected: their interval transfer bounds `a + b − 2ab` below by `−2`, and a sum of such values then fails the true-integer rule at its `BitDecompose`. `BitFn` carries its range.
3. **One `BitDecompose { bits }` output with the carry inside.** Rejected: 35 bits pad to 64, nearly doubling the rounds' committed bits. Two outputs pad 32 and 4.
4. **A `Sha256Compress` gadget op.** Considered. It would hide the circuit in one op, with its own relations and lemma. The three operations here are smaller, each with one lemma or none, and they serve the key commitment and AES's range checks too.

## Drawbacks

- The IR's closed operation set grows by three, and the verifier's lowering must mirror them (#150, #159).
- A frame's SHA-256 is a long sequential graph, 64 rounds per block, of about 40 nodes each.

## Migration / Rollout

1. **#143:**
   - the three operations in `tessera-ir`, with their rules;
   - the range analysis;
   - the CPU engine's evaluation;
   - the SHA-256 builder in `tessera_compiler::anchor`, tested against NIST CAVP's short and long messages.
2. **The schedule's relations** for the new zero-checks and `BitFn`, with the prover's and the verifier's: follow-up work after #143, as the anchor's wiring lands (#146, #163).
3. **#150** checks the anchor's final relations with them.

## Testing Strategy

- **Known answers.** NIST CAVP `SHA256ShortMsg` and `SHA256LongMsg` (`spec/vectors/sha256/`): the circuit's digest bits equal each `MD`, on the CPU engine.
- **Lemma 1.** A wrong carry bit, a wrong result bit and a non-Boolean bit each break a zero-check's identity on the witness. A committed byte of value 256 has no passing bits.
- **The key commitment.** The circuit's digest of a session's key matches `tessera_anchor`'s direct `key_commitment`.
- **Ranges.** The range analysis passes the circuit, and refuses a `BitFn` on a non-bit input and a `BitDecompose` of an input that may exceed its width.

## Open Questions

None.

## References

- FIPS 180-4, *Secure Hash Standard*, §4.1.2, §5.1.1, §5.3.3 and §6.2.
- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0059](RFC-0059-frame-headers-and-nonces.md), [RFC-0061](RFC-0061-sampling-randomness.md).
- [#142](https://github.com/AbdelStark/tessera/issues/142), [#143](https://github.com/AbdelStark/tessera/issues/143).
