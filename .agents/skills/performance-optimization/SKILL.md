---
name: performance-optimization
description: Optimize Tessera's prover and verifier kernels using roofline modeling, bit-exact equivalence tests, Criterion benchmarks, and SIMD/CUDA safety gates. Use when optimizing field arithmetic, Poseidon2, Merkle commitments, sumcheck, LogUp/GKR, or Ligero PCS, or when reviewing a performance pull request.
---

# Performance Optimization

Tessera's performance targets are fixed by [08-performance-budget.md](../../../docs/spec/08-performance-budget.md) and [RFC-0018](../../../docs/rfcs/RFC-0018-gpu-prover-runtime.md): decode throughput, peak VRAM, proof size, and verification latency must meet their budgets without ever relaxing a soundness check or introducing floating-point divergence.

Load [kernels-and-roofline.md](kernels-and-roofline.md) for the mathematical and hardware constraints of each hot kernel family.

## Non-Negotiable Invariants

1. **Bit-exact cross-backend equivalence (INV-FIELD-5, INV-TCE-1).** Every SIMD (`aarch64` NEON, `x86_64` AVX2/AVX-512) and GPU (`tessera-gpu`) kernel must produce the exact canonical Goldilocks element (`0 <= x < p`, `p = 2^64 - 2^32 + 1`) or Blake3/Poseidon2 digest produced by the portable scalar reference implementation on every input, including boundary elements (`0`, `1`, `p - 1`, `2^64 - 1`, `2^32 - 1`, `2^32`).
2. **Zero floating point in proved or hashed paths (INV-DM-6).** Never replace integer or finite-field arithmetic with floating-point approximations (`f32`, `f64`, tensor-core FP16/BF16 accumulation without exact CRT reconstruction).
3. **Transcript and proof bit-identity.** Optimizing a prover algorithm (such as sumcheck round evaluation or Merkle tree building) must not change the Fiat–Shamir transcript byte stream, message ordering, or proof format unless an RFC explicitly specifies a protocol change.
4. **Unsafe code containment (`xtask/src/unsafe_code.rs`).** `unsafe` is permitted only in `crates/tessera-field` (SIMD intrinsics) and `crates/tessera-gpu` (CUDA FFI), and only inside modules with `#[allow(unsafe_code)]` where every `unsafe` block has a preceding `// SAFETY:` comment stating the pointer alignment, bounds, and target-feature invariants. The verifier (`verifier/`) forbids `unsafe` unconditionally.
5. **Constant-time / secret-independent execution on secret witnesses.** Prover optimizations on secret witness buffers (`Secret<T>`) must not introduce secret-dependent branches, variable-time division, or secret-indexed cache lookups that leak witness values through side channels or logs ([05 §3](../../../docs/spec/05-observability.md#3-data-classification-and-redaction)).

## Optimization Loop

Follow these six steps in order. Do not write SIMD or CUDA code before completing Steps 1 and 2.

### Step 1 — Establish the Roofline & Baseline

1. Identify the target component and its budget in [08-performance-budget.md](../../../docs/spec/08-performance-budget.md).
2. Compute the theoretical hardware limit for the operation before touching code:
   - **Memory-bound kernels** (e.g., streaming sumcheck round fold, Ligero row encoding, Merkle leaf copy): compute bytes read + written per element (`B_elem`), divide by peak memory bandwidth (`BW`), and compare against measured nanoseconds per element.
   - **Compute-bound kernels** (e.g., Poseidon2 permutation, radix-2 NTT butterflies, small matrix-vector products in L1 cache): count Goldilocks 64x64->128 multiplications and 64-bit additions per output, divide by core issue width, and compare against measured cycles.
3. Run the Criterion benchmark for the target crate and save the baseline:
   ```sh
   cargo bench -p <crate> -- --save-baseline before
   ```
4. If no Criterion benchmark exists for the target function, add one under `crates/<crate>/benches/` before optimizing.

Done when you have recorded the baseline throughput (`ns/op` or `GiB/s`), the arithmetic intensity (`FLOP/byte` or `field_op/byte`), and whether the kernel is compute-bound or memory-bound.

### Step 2 — Lock In the Differential Equivalence Oracle

1. Keep the clean, portable scalar implementation intact (as a reference function or behind the default scalar path).
2. Write a differential test that compares the optimized path against the scalar reference on:
   - Boundary values: `0`, `1`, `p - 1`, `2^32 - 1`, `2^32`, `p - 2^32`, and all combinations of limbs near `2^64 - 1`.
   - For a multi-word accumulator, one start per carry path: each word filled to exactly its rollover point for the addend it receives (`u64::MAX − high` for a middle word that takes a product's high half), so that a carry from the word below ripples through it. All-`MAX` starts and random sums skip that path: dropping `Wide192`'s ripple carry survived every test until such a start was added (#391).
   - Tail lengths not divisible by SIMD lane widths (`1, 2, 3, 4, 7, 8, 15, 16, 31, 33` elements).
   - Random `proptest` inputs across multiple buffer alignments.
3. Run the test and confirm it passes against the current code.

Done when a differential equivalence test exists and exercises boundary field elements and non-multiple-of-lane tail lengths.

### Step 3 — Eliminate Algorithmic & Allocation Waste First

Before dropping to intrinsics, check for algorithmic and memory-hierarchy wins in safe Rust:

1. **Pre-allocation and scratch reuse**: hoist `Vec` allocations out of inner sumcheck rounds, tree layers, and token steps. Pass `&mut [Goldilocks]` scratch slices sized once per layer or token.
2. **Lazy / deferred modular reduction**: when accumulating products $\sum_{i=0}^{k-1} a_i \cdot b_i \pmod p$, accumulate in 128-bit (`u128`) or wide limb registers and reduce mod $p$ once at the end of the inner loop (or every $K$ terms where overflow is impossible—prove the max intermediate value $< 2^{128}$ in a comment).
3. **Batch inversion (Montgomery's trick)**: replace $N$ independent `.inv()` calls ($\approx 64 N$ multiplications) with 1 inversion and $3(N - 1)$ multiplications whenever denominators are known ahead of time (e.g., LogUp denominators $z - \text{key}_i$). Zero checks must still reject or handle zero denominators identically to individual inversions.
4. **Public sparsity only**: use specified closed forms for public wiring/constants. Never branch or shorten a traversal on secret zero values, multiplicities or key distributions. Public-budget chunking can amortize a fixed-capacity count scan without changing its keyed-address loop (INV-ARCH-2).
5. **Cache-friendly traversal**: order multi-dimensional hypercube folds so both halves `(x_0, ...)` and `(x_1, ...)` are read sequentially in contiguous memory lines rather than strided by $2^{v-1}$ when $2^{v-1}$ exceeds L2 cache size.

Done when safe-Rust algorithmic and memory-layout improvements are implemented and benchmarked in isolation.

### Step 4 — Vectorize or Parallelize (When Needed)

If safe-Rust improvements leave a gap against the roofline:

1. **Auto-vectorization check**: inspect generated assembly (`cargo rustc -p <crate> --release -- --emit asm`) to see if LLVM already emits `umull`/`umlal` (NEON) or `vpmuludq` (AVX2). Often a fixed-size chunk loop (`slice.chunks_exact(4)` or `chunks_exact(8)`) with no early exits or bounds checks inside the body is enough for LLVM to vectorize without `unsafe`.
2. **Explicit SIMD (`tessera-field` only)**: if intrinsics are required:
   - Gate behind `#[cfg(target_arch = "aarch64")]` or `#[cfg(target_arch = "x86_64")]` with runtime feature detection (`is_x86_feature_detected!`) where appropriate.
   - Isolate in an opt-in module (`#[allow(unsafe_code)]`) and write a `// SAFETY:` comment on every `unsafe` block proving pointer validity, alignment, and target feature availability.
3. **GPU kernels (`tessera-gpu` only)**: follow [RFC-0018](../../../docs/rfcs/RFC-0018-gpu-prover-runtime.md) and [kernels-and-roofline.md](kernels-and-roofline.md).

Done when the vectorized or parallel kernel compiles cleanly under `cargo clippy --workspace --all-targets --locked -- -D warnings` and `cargo xtask lint`.

### Step 5 — Verify Equivalence, Soundness, and Gates

1. Run the differential equivalence tests and property tests:
   ```sh
   cargo test -p <crate> --locked
   ```
2. Run all repository gates:
   ```sh
   cargo xtask lint
   cargo xtask check-isolation
   cargo xtask check-test-features
   ```
3. Verify that conformance vectors in `spec/vectors/` and end-to-end proof transcripts are bit-for-bit unchanged:
   ```sh
   cargo xtask gen-vectors --check
   ```

Done when all unit tests, differential tests, conformance vectors, and `xtask` gates pass with zero warnings.

### Step 6 — Measure Against Baseline and Document

1. Run Criterion against the saved baseline:
   ```sh
   cargo bench -p <crate> -- --baseline before
   ```
2. Record the before/after numbers (`ns/op`, `GiB/s`, `% change`, and hardware CPU/GPU model) in the commit message and pull request description.
3. If the change fails to improve throughput by at least the measurement noise floor (typically 2–3%) or complicates the code without moving the target metric, revert it. Complexity must buy measurable performance.

Done when before/after benchmark tables are recorded and the optimization shows a statistically significant improvement.
