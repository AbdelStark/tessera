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

// ---------------------------------------------------------------------------------------------
// Fraction trees (c_l, #63): RFC-0008 'Fraction tree'. Node x of level t is (n, d) = (n_L d_R +
// n_R d_L, d_L d_R) from entries 2x, 2x + 1 of level t + 1. A level's sumcheck proves
// sum_x eq(r, x) g(x), g = Ne Do + No De + lambda De Do, Ne(x) = N[2x], No(x) = N[2x + 1], and
// likewise for D. Levels 1 .. h - 1 are stored as separate N and D tables; the leaves are never
// stored: leaf j is (n_j, alpha - (k_j + beta v_j)), k_j, v_j and the multiplicity behind
// n_j = -m_j from the counter generator, or n_j = 1 for a lookup tree.

struct LeafSpec {
    uint64_t key_seed, value_seed, mult_seed;
    fp2 alpha, beta;
    uint32_t unit;  // numerators 1 (a lookup tree) or -m_j (a table tree)
};

__device__ __forceinline__ fp2 leaf_d(const LeafSpec &l, uint64_t j) {
    const uint64_t k = gl::counter_value(l.key_seed + j);
    const uint64_t v = gl::counter_value(l.value_seed + j);
    return f2_canon({gl::lsub(gl::lsub(l.alpha.a, k), gl::lmul(l.beta.a, v)), gl::lsub(l.alpha.b, gl::lmul(l.beta.b, v))});
}

// -m_j with m_j = counter mod 2^16, canonical.
__device__ __forceinline__ fp2 leaf_n(const LeafSpec &l, uint64_t j) {
    if (l.unit) return {1, 0};
    const uint64_t m = gl::counter_value(l.mult_seed + j) & 0xFFFF;
    return {m == 0 ? 0 : gl::P - m, 0};
}

// Table sources of a level's sumcheck, by pair index i of the current variable's tables:
// ne(i), no(i), de(i), dd(i) are Ne, No, De, Do at entry i.
struct StoredSrc {
    const fp2 *n, *d;
    __device__ fp2 ne(uint64_t i) const { return n[2 * i]; }
    __device__ fp2 no(uint64_t i) const { return n[2 * i + 1]; }
    __device__ fp2 de(uint64_t i) const { return d[2 * i]; }
    __device__ fp2 dd(uint64_t i) const { return d[2 * i + 1]; }
};
struct LeafSrc {
    LeafSpec l;
    __device__ fp2 ne(uint64_t i) const { return leaf_n(l, 2 * i); }
    __device__ fp2 no(uint64_t i) const { return leaf_n(l, 2 * i + 1); }
    __device__ fp2 de(uint64_t i) const { return leaf_d(l, 2 * i); }
    __device__ fp2 dd(uint64_t i) const { return leaf_d(l, 2 * i + 1); }
};
struct BoundSrc {
    const fp2 *bne, *bno, *bde, *bdd;
    uint32_t unit;
    __device__ fp2 ne(uint64_t i) const { return unit ? fp2{1, 0} : bne[i]; }
    __device__ fp2 no(uint64_t i) const { return unit ? fp2{1, 0} : bno[i]; }
    __device__ fp2 de(uint64_t i) const { return bde[i]; }
    __device__ fp2 dd(uint64_t i) const { return bdd[i]; }
};

#define LEAF_ARGS                                                                                    \
    uint64_t key_seed, uint64_t value_seed, uint64_t mult_seed, uint64_t a0, uint64_t a1, uint64_t b0, \
        uint64_t b1, uint32_t unit
#define LEAF_SPEC LeafSpec{key_seed, value_seed, mult_seed, {a0, a1}, {b0, b1}, unit}

__device__ __forceinline__ void frac_node(fp2 nl, fp2 dl, fp2 nr, fp2 dr, fp2 *n, fp2 *d) {
    *n = f2_canon(f2_add(f2_mul(nl, dr), f2_mul(nr, dl)));
    *d = f2_canon(f2_mul(dl, dr));
}

// Level h - 1 from the generated leaves.
extern "C" __global__ void frac_level_leaves(LEAF_ARGS, fp2 *n_out, fp2 *d_out, uint64_t parents) {
    const LeafSpec l = LEAF_SPEC;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t x = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; x < parents; x += stride) {
        frac_node(leaf_n(l, 2 * x), leaf_d(l, 2 * x), leaf_n(l, 2 * x + 1), leaf_d(l, 2 * x + 1), &n_out[x], &d_out[x]);
    }
}

// Level t from level t + 1.
extern "C" __global__ void frac_level(const fp2 *n, const fp2 *d, fp2 *n_out, fp2 *d_out, uint64_t parents) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t x = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; x < parents; x += stride) {
        frac_node(n[2 * x], d[2 * x], n[2 * x + 1], d[2 * x + 1], &n_out[x], &d_out[x]);
    }
}

// The pair (2y, 2y + 1) of the current tables' contribution at c = 0, 1, 2.
__device__ __forceinline__ void frac_pair(fp2 acc[3], fp2 w, fp2 lambda, fp2 ne0, fp2 ne1, fp2 no0, fp2 no1, fp2 de0,
                                          fp2 de1, fp2 dd0, fp2 dd1) {
    const fp2 ne2 = f2_sub(f2_add(ne1, ne1), ne0), no2 = f2_sub(f2_add(no1, no1), no0);
    const fp2 de2 = f2_sub(f2_add(de1, de1), de0), dd2 = f2_sub(f2_add(dd1, dd1), dd0);
    const fp2 nes[3] = {ne0, ne1, ne2}, nos[3] = {no0, no1, no2}, des[3] = {de0, de1, de2}, dds[3] = {dd0, dd1, dd2};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const fp2 g = f2_add(f2_add(f2_mul(nes[c], dds[c]), f2_mul(nos[c], des[c])), f2_mul(lambda, f2_mul(des[c], dds[c])));
        acc[c] = f2_add(acc[c], f2_mul(w, g));
    }
}

template <typename Src>
__device__ __forceinline__ void frac_sums(const Src &src, uint64_t pairs, fp2 lambda, const fp2 *eq_lo,
                                          uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials) {
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t y = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; y < pairs; y += stride) {
        frac_pair(acc, weight(y, eq_lo, lo_bits, eq_hi), lambda, src.ne(2 * y), src.ne(2 * y + 1), src.no(2 * y),
                  src.no(2 * y + 1), src.de(2 * y), src.de(2 * y + 1), src.dd(2 * y), src.dd(2 * y + 1));
    }
    block_sums(acc, partials);
}

// Binds src at c into the four tables (two when unit) of `pairs` pairs, entries 2z and 2z + 1,
// and adds the sums of the new pairs.
template <typename Src>
__device__ __forceinline__ void frac_bind_sums(const Src &src, uint64_t pairs, fp2 c, uint32_t unit, fp2 *bne,
                                               fp2 *bno, fp2 *bde, fp2 *bdd, fp2 lambda, const fp2 *eq_lo,
                                               uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials) {
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t z = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; z < pairs; z += stride) {
        fp2 ne0 = {1, 0}, ne1 = {1, 0}, no0 = {1, 0}, no1 = {1, 0};
        if (!unit) {
            ne0 = f2_line(src.ne(4 * z), src.ne(4 * z + 1), c);
            ne1 = f2_line(src.ne(4 * z + 2), src.ne(4 * z + 3), c);
            no0 = f2_line(src.no(4 * z), src.no(4 * z + 1), c);
            no1 = f2_line(src.no(4 * z + 2), src.no(4 * z + 3), c);
            bne[2 * z] = ne0;
            bne[2 * z + 1] = ne1;
            bno[2 * z] = no0;
            bno[2 * z + 1] = no1;
        }
        const fp2 de0 = f2_line(src.de(4 * z), src.de(4 * z + 1), c), de1 = f2_line(src.de(4 * z + 2), src.de(4 * z + 3), c);
        const fp2 dd0 = f2_line(src.dd(4 * z), src.dd(4 * z + 1), c), dd1 = f2_line(src.dd(4 * z + 2), src.dd(4 * z + 3), c);
        bde[2 * z] = de0;
        bde[2 * z + 1] = de1;
        bdd[2 * z] = dd0;
        bdd[2 * z + 1] = dd1;
        frac_pair(acc, weight(z, eq_lo, lo_bits, eq_hi), lambda, ne0, ne1, no0, no1, de0, de1, dd0, dd1);
    }
    block_sums(acc, partials);
}

// The last binding: the single bound entry of each table, canonical, into out[0..4) as Ne, No,
// De, Do (1 for unit numerators).
template <typename Src>
__device__ __forceinline__ void frac_bind_last(const Src &src, fp2 c, fp2 *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    out[0] = f2_canon(f2_line(src.ne(0), src.ne(1), c));
    out[1] = f2_canon(f2_line(src.no(0), src.no(1), c));
    out[2] = f2_canon(f2_line(src.de(0), src.de(1), c));
    out[3] = f2_canon(f2_line(src.dd(0), src.dd(1), c));
}

#define SUM_ARGS uint64_t pairs, uint64_t l0, uint64_t l1, const fp2 *eq_lo, uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials
#define BIND_ARGS uint64_t c0, uint64_t c1, fp2 *bne, fp2 *bno, fp2 *bde, fp2 *bdd

extern "C" __global__ void frac_sums_stored(const fp2 *n, const fp2 *d, SUM_ARGS) {
    frac_sums(StoredSrc{n, d}, pairs, {l0, l1}, eq_lo, lo_bits, eq_hi, partials);
}
extern "C" __global__ void frac_sums_leaves(LEAF_ARGS, SUM_ARGS) {
    frac_sums(LeafSrc{LEAF_SPEC}, pairs, {l0, l1}, eq_lo, lo_bits, eq_hi, partials);
}
extern "C" __global__ void frac_bind_sums_stored(const fp2 *n, const fp2 *d, BIND_ARGS, SUM_ARGS) {
    frac_bind_sums(StoredSrc{n, d}, pairs, {c0, c1}, 0, bne, bno, bde, bdd, {l0, l1}, eq_lo, lo_bits, eq_hi, partials);
}
extern "C" __global__ void frac_bind_sums_leaves(LEAF_ARGS, BIND_ARGS, SUM_ARGS) {
    frac_bind_sums(LeafSrc{LEAF_SPEC}, pairs, {c0, c1}, unit, bne, bno, bde, bdd, {l0, l1}, eq_lo, lo_bits, eq_hi,
                   partials);
}
extern "C" __global__ void frac_bind_sums_bound(const fp2 *ne, const fp2 *no, const fp2 *de, const fp2 *dd,
                                                uint32_t unit, BIND_ARGS, SUM_ARGS) {
    frac_bind_sums(BoundSrc{ne, no, de, dd, unit}, pairs, {c0, c1}, unit, bne, bno, bde, bdd, {l0, l1}, eq_lo, lo_bits,
                   eq_hi, partials);
}
extern "C" __global__ void frac_last_stored(const fp2 *n, const fp2 *d, uint64_t c0, uint64_t c1, fp2 *out) {
    frac_bind_last(StoredSrc{n, d}, {c0, c1}, out);
}
extern "C" __global__ void frac_last_leaves(LEAF_ARGS, uint64_t c0, uint64_t c1, fp2 *out) {
    frac_bind_last(LeafSrc{LEAF_SPEC}, {c0, c1}, out);
}
extern "C" __global__ void frac_last_bound(const fp2 *ne, const fp2 *no, const fp2 *de, const fp2 *dd, uint32_t unit,
                                           uint64_t c0, uint64_t c1, fp2 *out) {
    frac_bind_last(BoundSrc{ne, no, de, dd, unit}, {c0, c1}, out);
}

// A level on no variable (level 0 of a single tree): its tables' entries, unbound.
template <typename Src>
__device__ __forceinline__ void frac_first(const Src &src, fp2 *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    out[0] = f2_canon(src.ne(0));
    out[1] = f2_canon(src.no(0));
    out[2] = f2_canon(src.de(0));
    out[3] = f2_canon(src.dd(0));
}
extern "C" __global__ void frac_first_stored(const fp2 *n, const fp2 *d, fp2 *out) { frac_first(StoredSrc{n, d}, out); }
extern "C" __global__ void frac_first_leaves(LEAF_ARGS, fp2 *out) { frac_first(LeafSrc{LEAF_SPEC}, out); }

// ---- Window-level rounds (#734): one level instance's tables offloaded by the host prover. ----
//
// Each table t holds 2 * pairs entries; pair p is (t[2p], t[2p + 1]). A round needs
//   S(x) = sum_p w(p) P(line_x(p)),  x in {0, 2, 3},  line_x = t[2p] + x (t[2p + 1] - t[2p]),
// with w(p) = eq_lo[p mod 2^lo_bits] eq_hi[p >> lo_bits]: the same field elements as the host's
// window prover (tessera_sumcheck::prover::window), so the proof does not change.

// line_x at x = 0, 2, 3: a, 2b - a, 3b - 2a.
__device__ __forceinline__ void lines(fp2 a, fp2 b, fp2 out[3]) {
    const fp2 d = f2_sub(b, a);
    out[0] = a;
    out[1] = f2_add(b, d);
    out[2] = f2_add(out[1], d);
}

// N0 D1 + N1 D0 + lambda D0 D1 over [N0, N1, D0, D1].
extern "C" __global__ void lvl_frac_sums(const fp2 *n0, const fp2 *n1, const fp2 *d0, const fp2 *d1,
                                         uint64_t pairs, uint64_t l0, uint64_t l1, const fp2 *eq_lo,
                                         uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials) {
    const fp2 lambda = {l0, l1};
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t p = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < pairs; p += stride) {
        fp2 a[3], b[3], c[3], e[3];
        lines(n0[2 * p], n0[2 * p + 1], a);
        lines(n1[2 * p], n1[2 * p + 1], b);
        lines(d0[2 * p], d0[2 * p + 1], c);
        lines(d1[2 * p], d1[2 * p + 1], e);
        const fp2 w = weight(p, eq_lo, lo_bits, eq_hi);
        for (int z = 0; z < 3; ++z) {
            const fp2 v = f2_add(f2_add(f2_mul(a[z], e[z]), f2_mul(b[z], c[z])), f2_mul(lambda, f2_mul(c[z], e[z])));
            acc[z] = f2_add(acc[z], f2_mul(w, v));
        }
    }
    block_sums(acc, partials);
}

// V0 V1 over [V0, V1].
extern "C" __global__ void lvl_prod_sums(const fp2 *v0, const fp2 *v1, uint64_t pairs, const fp2 *eq_lo,
                                         uint32_t lo_bits, const fp2 *eq_hi, fp2 *partials) {
    fp2 acc[3] = {{0, 0}, {0, 0}, {0, 0}};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t p = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < pairs; p += stride) {
        fp2 a[3], b[3];
        lines(v0[2 * p], v0[2 * p + 1], a);
        lines(v1[2 * p], v1[2 * p + 1], b);
        const fp2 w = weight(p, eq_lo, lo_bits, eq_hi);
        for (int z = 0; z < 3; ++z) acc[z] = f2_add(acc[z], f2_mul(w, f2_mul(a[z], b[z])));
    }
    block_sums(acc, partials);
}

// out[i] = t[2i] + c (t[2i + 1] - t[2i]) for i < half, canonical.
extern "C" __global__ void lvl_bind(const fp2 *t, fp2 *out, uint64_t half, uint64_t c0, uint64_t c1) {
    const fp2 c = {c0, c1};
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < half; i += stride) {
        out[i] = f2_canon(f2_line(t[2 * i], t[2 * i + 1], c));
    }
}
