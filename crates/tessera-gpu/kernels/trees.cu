// SPDX-License-Identifier: Apache-2.0 OR MIT

// The product-tree microkernel of the leaf unit cost c_pi (RFC-0020 'Measurement protocol',
// #64): a forest's levels built bottom-up, and the degree-3 level sumchecks of RFC-0010 'Tree
// levels' in Fp2 = Fp[X]/(X^2 - 7) (RFC-0002).
//
// Level t of a forest of 2^s trees is a table of 2^(s+t) entries whose pairs (2x, 2x + 1) are
// the children of entry x of level t - 1. Level t's sumcheck proves
// sum_x eq(r, x) E(x) O(x) over n = s + t - 1 variables, E(x) = V_t[2x] and O(x) = V_t[2x + 1],
// binding the lowest variable first. A round needs, over the remaining variables x' after the
// current one,
//   S(c) = sum_x' w(x') E(c, x') O(c, x'),  c in {0, 1, 2},
// with w(x') = eq_lo[x' mod 2^lo] * eq_hi[x' >> lo] (eq split in two tables that eq_table
// builds from the level's point) and E(c, x') = E(0, x') + c (E(1, x') - E(0, x')). Blocks
// write partial sums; reduce_sums adds them up. Each round after the first is fused with the
// binding of the round before it, so a table is read once per round.

#include <cstdint>

#include "goldilocks.cuh"

using std::uint32_t;
using std::uint64_t;

struct fp2 {
    uint64_t a, b;  // a + b X, X^2 = 7
};

// Lazily reduced (gl::ladd, gl::lsub, gl::lmul): any 64-bit representatives; f2_canon makes
// a value canonical before it is stored for the host or reduced across blocks.
__device__ __forceinline__ fp2 f2_add(fp2 x, fp2 y) { return {gl::ladd(x.a, y.a), gl::ladd(x.b, y.b)}; }
__device__ __forceinline__ fp2 f2_sub(fp2 x, fp2 y) { return {gl::lsub(x.a, y.a), gl::lsub(x.b, y.b)}; }
__device__ __forceinline__ fp2 f2_mul_base(fp2 x, uint64_t c) { return {gl::lmul(x.a, c), gl::lmul(x.b, c)}; }
__device__ __forceinline__ fp2 f2_canon(fp2 x) { return {gl::canon(x.a), gl::canon(x.b)}; }

__device__ __forceinline__ uint64_t times7(uint64_t x) {
    const uint64_t x2 = gl::ladd(x, x);
    const uint64_t x4 = gl::ladd(x2, x2);
    return gl::ladd(gl::ladd(x4, x2), x);
}

// Karatsuba: three base products.
__device__ __forceinline__ fp2 f2_mul(fp2 x, fp2 y) {
    const uint64_t aa = gl::lmul(x.a, y.a);
    const uint64_t bb = gl::lmul(x.b, y.b);
    const uint64_t cross = gl::lmul(gl::ladd(x.a, x.b), gl::ladd(y.a, y.b));
    return {gl::ladd(aa, times7(bb)), gl::lsub(gl::lsub(cross, aa), bb)};
}

// x0 + c (x1 - x0)
__device__ __forceinline__ fp2 f2_line(fp2 x0, fp2 x1, fp2 c) { return f2_add(x0, f2_mul(c, f2_sub(x1, x0))); }

// x0 + c (x1 - x0) for base-field x0, x1: two base products.
__device__ __forceinline__ fp2 base_line(uint64_t x0, uint64_t x1, fp2 c) {
    const uint64_t d = gl::lsub(x1, x0);
    return {gl::ladd(x0, gl::lmul(c.a, d)), gl::lmul(c.b, d)};
}

// parent[i] = child[2i] child[2i + 1] for i < parents.
extern "C" __global__ void tree_level(const uint64_t *child, uint64_t *parent, uint64_t parents) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < parents;
         i += stride) {
        parent[i] = gl::mul(child[2 * i], child[2 * i + 1]);  // canonical: the next level reads it
    }
}

__device__ __forceinline__ fp2 weight(uint64_t y, const fp2 *eq_lo, uint32_t lo_bits, const fp2 *eq_hi) {
    return f2_mul(eq_lo[y & ((1ull << lo_bits) - 1)], eq_hi[y >> lo_bits]);
}

// Block-wide sum of three fp2 accumulators into out[3 * blockIdx.x ..]. blockDim.x = 256.
__device__ __forceinline__ void block_sums(fp2 acc[3], fp2 *out) {
    __shared__ fp2 s[3][256];
    for (int c = 0; c < 3; ++c) s[c][threadIdx.x] = acc[c];
    __syncthreads();
    for (uint32_t half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            for (int c = 0; c < 3; ++c) s[c][threadIdx.x] = f2_add(s[c][threadIdx.x], s[c][threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int c = 0; c < 3; ++c) out[3 * blockIdx.x + c] = f2_canon(s[c][0]);
    }
}

// The first round of a level, its tables still the base-field level v of 4 * pairs entries:
// E(0, y) = v[4y], O(0, y) = v[4y + 1], E(1, y) = v[4y + 2], O(1, y) = v[4y + 3].
extern "C" __global__ void level_sums_base(const uint64_t *v, uint64_t pairs, const fp2 *eq_lo,
                                           uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials) {
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t y = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; y < pairs; y += stride) {
        const ulonglong4 q = reinterpret_cast<const ulonglong4 *>(v)[y];
        const uint64_t e0 = q.x, o0 = q.y, e1 = q.z, o1 = q.w;
        const uint64_t p0 = gl::lmul(e0, o0);
        const uint64_t p1 = gl::lmul(e1, o1);
        const uint64_t p2 = gl::lmul(gl::lsub(gl::ladd(e1, e1), e0), gl::lsub(gl::ladd(o1, o1), o0));
        const fp2 w = weight(y, eq_lo, lo_bits, eq_hi);
        acc[0] = f2_add(acc[0], f2_mul_base(w, p0));
        acc[1] = f2_add(acc[1], f2_mul_base(w, p1));
        acc[2] = f2_add(acc[2], f2_mul_base(w, p2));
    }
    block_sums(acc, partials);
}

// One pair's contribution to the three sums: w E(c) O(c) for c = 0, 1, 2.
__device__ __forceinline__ void add_pair(fp2 acc[3], fp2 w, fp2 e0, fp2 e1, fp2 o0, fp2 o1) {
    const fp2 e2 = f2_sub(f2_add(e1, e1), e0), o2 = f2_sub(f2_add(o1, o1), o0);
    acc[0] = f2_add(acc[0], f2_mul(w, f2_mul(e0, o0)));
    acc[1] = f2_add(acc[1], f2_mul(w, f2_mul(e1, o1)));
    acc[2] = f2_add(acc[2], f2_mul(w, f2_mul(e2, o2)));
}

// Round 1 fused with the binding of round 0 at c: from the base-field level v (8 * pairs
// entries), writes the bound tables e_out, o_out (2 * pairs entries each) and the sums of round 1
// over its pairs z < pairs, whose entries 2z, 2z + 1 come from v[8z .. 8z + 8).
extern "C" __global__ void bind_sums_base(const uint64_t *v, uint64_t pairs, uint64_t c0, uint64_t c1,
                                          fp2 *e_out, fp2 *o_out, const fp2 *eq_lo, uint32_t lo_bits,
                                          const fp2 *eq_hi, fp2 *partials) {
    const fp2 c = {c0, c1};
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t z = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; z < pairs; z += stride) {
        const ulonglong4 q0 = reinterpret_cast<const ulonglong4 *>(v)[2 * z];
        const ulonglong4 q1 = reinterpret_cast<const ulonglong4 *>(v)[2 * z + 1];
        const fp2 e0 = base_line(q0.x, q0.z, c), o0 = base_line(q0.y, q0.w, c);
        const fp2 e1 = base_line(q1.x, q1.z, c), o1 = base_line(q1.y, q1.w, c);
        e_out[2 * z] = e0;
        e_out[2 * z + 1] = e1;
        o_out[2 * z] = o0;
        o_out[2 * z + 1] = o1;
        add_pair(acc, weight(z, eq_lo, lo_bits, eq_hi), e0, e1, o0, o1);
    }
    block_sums(acc, partials);
}

// Round j + 1 fused with the binding of round j at c, for Fp2 tables e, o of 4 * pairs entries:
// writes the bound tables (2 * pairs entries each) and the sums over the new pairs.
extern "C" __global__ void bind_sums_ext(const fp2 *e, const fp2 *o, uint64_t pairs, uint64_t c0, uint64_t c1,
                                         fp2 *e_out, fp2 *o_out, const fp2 *eq_lo, uint32_t lo_bits,
                                         const fp2 *eq_hi, fp2 *partials) {
    const fp2 c = {c0, c1};
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t z = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; z < pairs; z += stride) {
        const fp2 e0 = f2_line(e[4 * z], e[4 * z + 1], c), e1 = f2_line(e[4 * z + 2], e[4 * z + 3], c);
        const fp2 o0 = f2_line(o[4 * z], o[4 * z + 1], c), o1 = f2_line(o[4 * z + 2], o[4 * z + 3], c);
        e_out[2 * z] = e0;
        e_out[2 * z + 1] = e1;
        o_out[2 * z] = o0;
        o_out[2 * z + 1] = o1;
        add_pair(acc, weight(z, eq_lo, lo_bits, eq_hi), e0, e1, o0, o1);
    }
    block_sums(acc, partials);
}

// out[a] = eq(points[first .. first + m), a) for a < 2^m, points[first] the lowest bit.
extern "C" __global__ void eq_table(const fp2 *points, uint32_t first, uint32_t m, fp2 *out) {
    const uint64_t a = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (a >= (1ull << m)) return;
    fp2 acc = {1, 0};
    for (uint32_t k = 0; k < m; ++k) {
        const fp2 r = points[first + k];
        acc = f2_mul(acc, ((a >> k) & 1) ? r : f2_sub({1, 0}, r));
    }
    out[a] = f2_canon(acc);
}

// out[c] = sum over count blocks of partials[3 b + c]; one block of 256 threads.
extern "C" __global__ void reduce_sums(const fp2 *partials, uint32_t count, fp2 *out) {
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    for (uint32_t b = threadIdx.x; b < count; b += blockDim.x) {
        for (int c = 0; c < 3; ++c) acc[c] = f2_add(acc[c], partials[3 * b + c]);
    }
    block_sums(acc, out);
}

// The last round's binding (or the only round's), to canonical single entries for the host.
// Binds the first variable of a level's base-field tables at c: e_out[y] = E(c, y), o_out[y] =
// O(c, y), for y < pairs.
extern "C" __global__ void bind_base(const uint64_t *v, uint64_t pairs, uint64_t c0, uint64_t c1,
                                     fp2 *e_out, fp2 *o_out) {
    const fp2 c = {c0, c1};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t y = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; y < pairs; y += stride) {
        const ulonglong4 q = reinterpret_cast<const ulonglong4 *>(v)[y];
        e_out[y] = f2_canon(base_line(q.x, q.z, c));
        o_out[y] = f2_canon(base_line(q.y, q.w, c));
    }
}

// Binds the current variable of Fp2 tables at c, into separate output tables.
extern "C" __global__ void bind_ext(const fp2 *e, const fp2 *o, uint64_t pairs, uint64_t c0, uint64_t c1,
                                    fp2 *e_out, fp2 *o_out) {
    const fp2 c = {c0, c1};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t y = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; y < pairs; y += stride) {
        e_out[y] = f2_canon(f2_line(e[2 * y], e[2 * y + 1], c));
        o_out[y] = f2_canon(f2_line(o[2 * y], o[2 * y + 1], c));
    }
}

