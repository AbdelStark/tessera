# RFC-0002: Goldilocks Field and Quadratic Extension

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1
- Amended by: [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains) ('Timing': a stream of RFC-0004 rejecting its words, INV-ARCH-2's one exception, [#567](https://github.com/AbdelStark/tessera/issues/567)), draft [RFC-0073](RFC-0073-weight-mask-expansion.md) ('Timing': the weight mask streams; in effect when it is accepted)

## Summary

Tessera computes over the Goldilocks prime field `Fp`, `P = 2^64 − 2^32 + 1`, and draws every challenge from its quadratic extension `Fp2 = Fp[X]/(X^2 − 7)`, which has about 2^128 elements. This RFC fixes the representation, reduction, inversion, roots of unity, canonical encoding and the trait surface that keeps the rest of the code generic over the field, so that a small test field can be substituted in soundness-calibration tests.

## Motivation

The source fixes Goldilocks with a degree-2 extension as the default (source §10, "Field and prover core") because a 64-bit field needs fewer rescales than a 31-bit field (linear chains up to about 2^60 stay exact), gives a simple RMSNorm, and admits NTTs up to 2^32 points. 128-bit challenges are required for at least 100-bit soundness across tens of thousands of sumcheck rounds (source §2, "Security targets"; §7, "Interaction and soundness"). Every other component depends on these types, so their contracts must be fixed first.

## Goals

- Exact `Fp` and `Fp2` arithmetic, branch-free in the prover, with canonical encodings that reject non-canonical inputs.
- A trait surface that lets the compiler, sumcheck, LogUp and PCS code run over any prime field with a quadratic extension.
- Roots of unity defined by formula, not by copied constants.
- Identical results on CPU and GPU.

## Non-Goals

- Other production fields (M31, BabyBear, Mersenne-61). The traits admit them; implementing them is non-goal NG-4.
- Higher-degree extensions.
- Constant-time guarantees beyond branch-free arithmetic (see Proposed Design, "Timing").

## Proposed Design

### Constants

| Name | Value | Derivation |
|---|---|---|
| `P` | `0xFFFF_FFFF_0000_0001` = 2^64 − 2^32 + 1 | Definition |
| `TWO_ADICITY` | 32 | `P − 1 = 2^32 · (2^32 − 1)` |
| `MULTIPLICATIVE_GENERATOR` | 7 | Generates `Fp^*`; verified by checking `7^((P−1)/q) ≠ 1` for every prime `q` dividing `P − 1` (2, 3, 5, 17, 257, 65537) |
| `ROOT_2_32` | `7^((P−1)/2^32) mod P` | Primitive 2^32-th root of unity |
| `ROOT_2_K` | `ROOT_2_32^(2^(32−k))` | Primitive 2^k-th root, `0 ≤ k ≤ 32` |
| `W` (extension non-residue) | 7 | Smallest quadratic non-residue: `P ≡ 1 (mod 8)` makes 2 a residue; `P ≡ 1 (mod 3)` and `P ≡ 1 (mod 5)` make 3 and 5 residues; 6 = 2·3 is a residue; `P ≡ 6 (mod 7)` with quadratic reciprocity gives `(7/P) = (6/7) = −1` |

Implementations compute `ROOT_2_32` at build time or in a `const fn` and pin its value in `spec/vectors/field/constants.json`, which both workspaces check.

### Traits

```rust
pub trait Field:
    Copy + Eq + core::hash::Hash + Send + Sync + 'static
    + core::ops::Add<Output = Self> + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self> + core::ops::Neg<Output = Self>
    + core::ops::AddAssign + core::ops::SubAssign + core::ops::MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn square(&self) -> Self;
    fn double(&self) -> Self;
    fn inverse(&self) -> Option<Self>;            // None for zero
    fn pow(&self, exp: u64) -> Self;
}

pub trait PrimeField: Field + Ord {
    const MODULUS: u64;
    const TWO_ADICITY: u32;
    fn from_u64_reduced(x: u64) -> Self;          // x mod P
    fn from_i64(x: i64) -> Self;                  // signed embedding: negative x maps to P - |x|
    fn from_canonical_u64(x: u64) -> Option<Self>; // None if x >= P
    fn to_canonical_u64(&self) -> u64;
    fn to_centered_i128(&self) -> i128;           // x if x <= (P-1)/2, else x - P
    fn two_adic_generator(log_n: u32) -> Self;    // primitive 2^log_n-th root; panics if log_n > TWO_ADICITY
}

pub trait QuadraticExtension<B: PrimeField>: Field + From<B> {
    const NON_RESIDUE: B;                          // W
    fn new(c0: B, c1: B) -> Self;
    fn coeffs(&self) -> [B; 2];
    fn mul_base(&self, b: B) -> Self;
    fn conjugate(&self) -> Self;                   // c0 - c1 X
    fn norm(&self) -> B;                           // c0^2 - W c1^2
}

pub fn batch_inverse<F: Field>(xs: &mut [F]) -> Result<(), ZeroElement>; // ZeroElement { index }
```

`Goldilocks` implements `PrimeField`; `GoldilocksExt2` implements `QuadraticExtension<Goldilocks>`. A test-only `F65537` (`P = 2^16 + 1`, two-adicity 16, generator 3) with `F65537Ext2` (non-residue 3) implements the same traits for soundness calibration ([07 §4](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)).

### Representation and reduction

- `Goldilocks(u64)` holds a canonical value in `[0, P)` after every public operation on CPU. GPU kernels may keep values in `[0, 2^64)` between operations and must canonicalize before writing to buffers that are hashed, encoded or copied to the host.
- Multiplication: compute the 128-bit product `x = x_lo + 2^64·x_hi`. Split `x_hi = a + 2^32·b` with 32-bit `a`, `b`. Since `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod P)`, `x ≡ x_lo − b + a·(2^32 − 1) (mod P)`. Evaluate with one wrapping subtraction (adding `P` back on borrow, branch-free) and one wrapping addition (adding `2^32 − 1` on carry, branch-free), then a final conditional subtraction of `P` by mask.
- Addition and subtraction use the same carry-and-mask pattern.
- Inversion uses Fermat's little theorem, `x^(P−2)`, with a fixed addition chain (72 multiplications). Zero returns `None`. `batch_inverse` uses Montgomery's trick and returns `ZeroElement { index }` for the first zero instead of producing garbage.

### Quadratic extension

- `(a0 + a1X)(b0 + b1X) = (a0b0 + 7a1b1) + ((a0 + a1)(b0 + b1) − a0b0 − a1b1)X`: three base multiplications plus a multiplication by 7 computed as `(x << 3) − x` with reduction.
- Inverse: `(a0 − a1X) · N^(−1)` with `N = a0^2 − 7a1^2`. `N = 0` only for zero because 7 is a non-residue.
- Embedding of `Fp` into `Fp2`: `b ↦ (b, 0)`.

### Canonical encoding

- `Fp`: 8 bytes, little-endian, value `< P`. Decoders reject values `≥ P` with `ERR-CODEC-NONCANONICAL`.
- `Fp2`: `c0` then `c1`, 16 bytes.
- Hashing and transcripts consume exactly these bytes (RFC-0003, RFC-0004).

### Signed integers

`from_i64` and `to_centered_i128` are the only bridges between integers and field elements. `to_centered_i128` is meaningful only for values known to be true integers ([03 §1](../spec/03-data-model.md#1-scalar-and-algebraic-types)); callers outside the witness engine and gadget witness generation must not use it on challenge-dependent values.

### Timing

All prover arithmetic is branch-free in the values: conditional corrections use masks or `select`, never `if` on data. This satisfies INV-ARCH-2 ([01 §6](../spec/01-architecture.md#6-concurrency-and-determinism)) at the instruction level. The one exception INV-ARCH-2 admits is not arithmetic on a value: a stream of RFC-0004 rejecting a word of `P` or more ([RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains), draft [RFC-0073](RFC-0073-weight-mask-expansion.md#the-stream)). Variable-time exponentiation is permitted only on public exponents (all exponents in this design are public constants).

### The verifier's implementation

The representation, reduction, inversion chain and timing rules above bind the prover's `tessera-field`. The verifier's `tv-field` implements the same field from this RFC with different algorithms on purpose, so that a structural bug cannot sit in both ([RFC-0001](RFC-0001-workspace-and-verifier-independence.md)): Montgomery form with REDC, schoolbook `Fp2` products, and inversion by square-and-multiply. It handles public data only, so its arithmetic may branch on values (INV-ARCH-2 constrains the prover). Everything observable is shared: the constants, the canonical encodings and every result, which both implementations check against `spec/vectors/field/`.

### Failure modes

| Failure | Handling |
|---|---|
| Non-canonical input bytes | Decoder error, never silent reduction |
| Inverse of zero | `None` or `ZeroElement { index }`; callers propagate a typed error |
| `two_adic_generator(k)` with `k > 32` | Panic: a programming error, caught in tests |
| CPU and GPU disagree | Differential test failure; `ERR-INTERNAL` in production self-checks |

## Alternatives Considered

1. **M31 with a degree-4 extension (QM31), as in Stwo.** Considered because 31-bit symbols halve commitment size and Stwo is mature. Rejected for v1 by the source's default (§10): roughly twice the linear-layer rescales because exact chains overflow 31 bits sooner, and a harder RMSNorm. The traits keep it open (source §9, problem 5).
2. **BabyBear (31-bit) with a degree-4 extension.** Same trade-off as M31, with a better two-adicity; rejected for the same reason.
3. **A 254-bit pairing-friendly scalar field.** Rejected: arithmetic is about 10× slower per element, and the design gains nothing from pairings; it would also tie the system to non-post-quantum commitments.
4. **Montgomery representation for Goldilocks.** Rejected for the prover: the special-form reduction above is faster and keeps values in their natural form, which simplifies debugging and encoding. The verifier uses it for algorithmic diversity (see 'The verifier's implementation').
5. **Cubic or quartic extension for all challenges.** Rejected: the quadratic extension already gives about 128-bit challenges; larger extensions raise the cost of every `Fp2` operation in lookups and sumchecks by 2.25× or more. The one place where 2^-90 matters (LogUp at frontier volume) is handled by grinding and live seeds (RFC-0014).

## Drawbacks

- 64-bit symbols cost twice the commitment bytes of 31-bit fields for 8- to 16-bit values (source §9, problem 3).
- The verifier reimplements this RFC independently (RFC-0001), doubling the code and the test burden.

## Migration / Rollout

First functional code in the repository (v0.1). No migration.

## Testing Strategy

- Property tests over 2^16 random cases per PR: ring axioms, `x · x^(−1) = 1` for non-zero `x`, `(x^2)^(1/2)` is not used, distributivity, `Fp2` multiplication equals schoolbook product modulo `X^2 − 7`, `norm(xy) = norm(x)·norm(y)`.
- Edge cases: `0, 1, P − 1, 2^32 − 1, 2^32, 2^63, P − 2^32` in every binary operation pair.
- Reference vectors: 10,000 random products, sums and inverses computed with arbitrary-precision integers by an independent script (Python `int`), stored in `spec/vectors/field/`, checked by both workspaces and the GPU kernels.
- Roots of unity: `ROOT_2_K^(2^k) = 1` and `ROOT_2_K^(2^(k−1)) = P − 1` for every `k` in `1..=32`.
- Generator check: `7^((P−1)/q) ≠ 1` for each prime factor `q` of `P − 1`.
- Encoding: every value in `[P, 2^64)` sampled at 10,000 points plus the endpoints is rejected; round-trip is the identity on canonical values.
- `batch_inverse` with a zero at each position returns that index.
- The `F65537` test field passes the same suites.

## Open Questions

None. The field is fixed by the source's §10 decision. Any change is a superseding RFC.

## References

- Source §2 (security targets), §7 (prover core, LogUp in a degree-2 extension), §9 problem 5, §10 (field and prover core).
- Goldilocks arithmetic as used in Plonky2 (Polygon Zero, 2022).
- Montgomery, "Speeding the Pollard and elliptic curve methods of factorization", 1987 (batch inversion).
