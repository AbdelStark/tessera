// SPDX-License-Identifier: Apache-2.0 OR MIT

// The encode-and-hash microkernel of the committed-element unit cost c_c (RFC-0020 'Measurement
// protocol', #55): rate-1/2^r coset Reed-Solomon encoding with a vanishing mask (RFC-0011
// 'Encoding') and BLAKE3 chained column digests (RFC-0004 'Column digests').
//
// A row of k = 2^log_k values f(w_k^i) becomes the codeword g(7 w_n^j), j < n = 2k, with
// g = f + (X^k - 1) m and m = sum_{i<P} m_i X^i. Three passes per row:
//
//   1. ntt_pass, DIF, inverse twiddles: the high stages of the size-k inverse NTT, in place.
//   2. encode_low: per tile, the remaining DIF stages (k f's coefficients in bit-reversed order),
//      the spread into bit-reversed size-n order (f_i 7^i / k - m_i 7^i at 2 rev_k(i),
//      m_i 7^(k+i) at 2 rev_k(i) + 1), and the low stages of the size-n DIT forward NTT.
//   3. ntt_pass, DIT, forward twiddles: the high stages, leaving g(7 w_n^j) in natural order.
//
// column_step then advances each column's chain over a block of up to B rows. Every index is
// derived from launch geometry the host checks against buffer lengths (src/commit.rs).

#include <cstdint>

#include "blake3.cuh"
#include "goldilocks.cuh"

using std::uint32_t;
using std::uint64_t;

// out[i] = counter_value(base + i) for i < len: rows and masks generated on the device.
extern "C" __global__ void fill_counter(uint64_t *out, uint64_t len, uint64_t base) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
         i += stride) {
        out[i] = gl::counter_value(base + i);
    }
}

// Shared-memory index of tile entry e: one u64 of padding per 16 keeps the strided accesses of a
// radix round on distinct banks.
__device__ __forceinline__ uint32_t pad(uint32_t e) { return e + (e >> 4); }

// x * w_8^k8 for k8 in 1..3, w_8 = 7^((p-1)/8) = 2^120 (forward) or its inverse 2^72: the 8th
// root of every table built from w_n = 7^((p-1)/n) (RFC-0002). DIT stages use the forward table,
// DIF stages the inverse one, so the direction fixes the root.
template <bool FWD>
__device__ __forceinline__ uint64_t mul_root8(uint64_t x, int k8) {
    if (FWD) {
        return k8 == 1 ? gl::lmul_pow2<120>(x) : k8 == 2 ? gl::lmul_pow2<48>(x) : gl::lmul_pow2<168>(x);
    }
    return k8 == 1 ? gl::lmul_pow2<72>(x) : k8 == 2 ? gl::lmul_pow2<144>(x) : gl::lmul_pow2<24>(x);
}

// Inside a pass, values are lazily reduced (gl::lmul, gl::ladd, gl::lsub: any 64-bit value
// congruent to the exact one); every store to global memory is canonical.
//
// The R stages of one radix-2^R round on the registers v of one group, in place. The group's
// elements are tile entries base + m 2^j, m < 2^R, and the stage of local half-size 2^(j+st)
// pairs m with m + 2^st and multiplies by w0[st] w_{2^(st+1)}^(m mod 2^st), where w0[st] =
// w_{2H}^E for the stage's global half-size H (see tile_round). DIT runs st upward; DIF downward.
template <int R, bool DIT>
__device__ __forceinline__ void radix(uint64_t v[1 << R], const uint64_t w0[R]) {
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const int st = DIT ? q : R - 1 - q;
        const int half = 1 << st;
#pragma unroll
        for (int m = 0; m < (1 << R); ++m) {
            if (m & half) continue;
            const int mm = m & (half - 1);
            const uint64_t w = mm ? mul_root8<DIT>(w0[st], mm << (2 - st)) : w0[st];
            const uint64_t x = v[m];
            const uint64_t y = v[m + half];
            if (DIT) {
                const uint64_t t = gl::lmul(y, w);
                v[m] = gl::ladd(x, t);
                v[m + half] = gl::lsub(x, t);
            } else {
                v[m] = gl::ladd(x, y);
                v[m + half] = gl::lmul(gl::lsub(x, y), w);
            }
        }
    }
}

// w0[st] = w_{2^(a-R+st+1)}^E for st < R, a = j + R + lo_bit, E = ((base mod 2^j) << lo_bit) +
// low0 + g: one table load each (cheaper on the device than squaring the top one). `table`
// holds w^i, i < 2^(L-1), for w of order 2^L = 2^log_table, so w_{2^b}^E = table[E << (L - b)].
template <int R>
__device__ __forceinline__ void group_twiddles(uint64_t w0[R], uint64_t e, uint32_t a,
                                               const uint64_t *table, uint32_t log_table) {
#pragma unroll
    for (int st = 0; st < R; ++st) w0[st] = table[e << (log_table - (a - R + st + 1))];
}

// One radix-2^R round over local bits [j, j+R) of a tile in shared memory: 2^c * G entries,
// entry t * G + g holding the element whose index bits [lo_bit, lo_bit + c) are t and whose low
// bits are low0 + g.
template <int R, bool DIT>
__device__ __forceinline__ void tile_round(uint64_t *s, uint32_t c, uint32_t j, uint32_t log_g,
                                           uint32_t lo_bit, uint64_t low0,
                                           const uint64_t *table, uint32_t log_table) {
    const uint32_t groups = 1u << (c - R + log_g);
    const uint32_t g_mask = (1u << log_g) - 1;
    const uint32_t low_mask = (1u << j) - 1;
    for (uint32_t gid = threadIdx.x; gid < groups; gid += blockDim.x) {
        const uint32_t g = gid & g_mask;
        const uint32_t q = gid >> log_g;
        const uint32_t base = (q & low_mask) | ((q >> j) << (j + R));
        uint64_t v[1 << R];
#pragma unroll
        for (int m = 0; m < (1 << R); ++m) v[m] = s[pad(((base + (m << j)) << log_g) | g)];
        uint64_t w0[R];
        const uint64_t e = (static_cast<uint64_t>(base & low_mask) << lo_bit) + low0 + g;
        group_twiddles<R>(w0, e, j + R + lo_bit, table, log_table);
        radix<R, DIT>(v, w0);
#pragma unroll
        for (int m = 0; m < (1 << R); ++m) s[pad(((base + (m << j)) << log_g) | g)] = v[m];
    }
}

template <bool DIT>
__device__ __forceinline__ void tile_round_any(uint32_t r, uint64_t *s, uint32_t c, uint32_t j,
                                               uint32_t log_g, uint32_t lo_bit, uint64_t low0,
                                               const uint64_t *table, uint32_t log_table) {
    if (r == 3) {
        tile_round<3, DIT>(s, c, j, log_g, lo_bit, low0, table, log_table);
    } else if (r == 2) {
        tile_round<2, DIT>(s, c, j, log_g, lo_bit, low0, table, log_table);
    } else {
        tile_round<1, DIT>(s, c, j, log_g, lo_bit, low0, table, log_table);
    }
}

// All c stages of a tile, in rounds of at most three stages: DIT from local bit `from` upward,
// DIF from bit c downward. A __syncthreads() precedes each round and follows the last.
template <bool DIT>
__device__ __forceinline__ void tile_transform(uint64_t *s, uint32_t from, uint32_t c,
                                               uint32_t log_g, uint32_t lo_bit, uint64_t low0,
                                               const uint64_t *table, uint32_t log_table) {
    if (DIT) {
        for (uint32_t j = from; j < c;) {
            const uint32_t r = c - j < 3 ? c - j : 3;
            __syncthreads();
            tile_round_any<true>(r, s, c, j, log_g, lo_bit, low0, table, log_table);
            j += r;
        }
    } else {
        for (uint32_t top = c; top > 0;) {
            const uint32_t r = top < 3 ? top : 3;
            __syncthreads();
            tile_round_any<false>(r, s, c, top - r, log_g, lo_bit, low0, table, log_table);
            top -= r;
        }
    }
    __syncthreads();
}

// c radix-2 stages over the index bits [lo_bit, lo_bit + c) of rows of 2^log_n elements, row r
// at data + r * row_stride (blockIdx.y = r). A block handles G = 2^log_g consecutive low
// offsets, so a strided pass still reads runs of G elements. DIF (dit = 0) runs half-sizes
// 2^(lo_bit + c - 1) down to 2^lo_bit; DIT runs them upward. `table` holds w^i for i <
// 2^(log_table - 1): w = w_N = 7^((p-1)/N), N = 2^log_table, for DIT, and its inverse for DIF. Dynamic shared memory: pad(2^(c + log_g)) elements.
extern "C" __global__ void ntt_pass(uint64_t *data, uint64_t row_stride, uint32_t log_n,
                                    uint32_t lo_bit, uint32_t c, uint32_t log_g,
                                    const uint64_t *table, uint32_t log_table, uint32_t dit) {
    extern __shared__ uint64_t s[];
    uint64_t *row = data + static_cast<uint64_t>(blockIdx.y) * row_stride;
    const uint32_t low_groups_log = lo_bit - log_g;
    const uint64_t tile = blockIdx.x;
    const uint64_t high = (tile >> low_groups_log) << (lo_bit + c);
    const uint64_t low0 = (tile & ((1ull << low_groups_log) - 1)) << log_g;
    const uint32_t size = 1u << (c + log_g);
    const uint32_t g_mask = (1u << log_g) - 1;
    for (uint32_t e = threadIdx.x; e < size; e += blockDim.x) {
        s[pad(e)] = row[high + (static_cast<uint64_t>(e >> log_g) << lo_bit) + low0 + (e & g_mask)];
    }
    if (dit) {
        tile_transform<true>(s, 0, c, log_g, lo_bit, low0, table, log_table);
    } else {
        tile_transform<false>(s, 0, c, log_g, lo_bit, low0, table, log_table);
    }
    for (uint32_t e = threadIdx.x; e < size; e += blockDim.x) {
        row[high + (static_cast<uint64_t>(e >> log_g) << lo_bit) + low0 + (e & g_mask)] =
            gl::canon(s[pad(e)]);
    }
}

// Pass 2 of the header, for tiles of 2^cx row values (blockIdx.x) of row blockIdx.y.
//   x: rows of k values after the high DIF stages; read only.
//   y: codewords of n = 2k symbols; this tile writes its 2^(cx+1) symbols.
//   inv_table / fwd_table: w_n^-i and w_n^i for i < n/2 (log_table = log_k + 1).
//   scale[t] = 7^rev_k(t) / k for t < k.
//   masks: P coefficients per row, m_0 first; mask_pow[i] = 7^i for i < P; shift_pow_k = 7^k.
// The first DIT round reads its codeword entries straight from the DIF output, so the spread
// costs no pass of its own: entry 2t is k f_rev(t) scale[t] - m 7^i, entry 2t + 1 is m 7^(k+i)
// for i = rev_k(t) < P. The host launches at least 2^(cx+1-R1) / 4 threads, R1 = min(3, cx+1),
// so a thread holds at most four groups of that round. Dynamic shared memory: pad(2^(cx+1)).
extern "C" __global__ void encode_low(const uint64_t *x, uint64_t *y, uint32_t log_k,
                                      uint32_t cx, const uint64_t *inv_table,
                                      const uint64_t *fwd_table, const uint64_t *scale,
                                      const uint64_t *masks, const uint64_t *mask_pow,
                                      uint64_t shift_pow_k, uint32_t mask_coeffs) {
    extern __shared__ uint64_t s[];
    const uint64_t k = 1ull << log_k;
    const uint32_t log_table = log_k + 1;
    const uint32_t tile = 1u << cx;
    const uint64_t x0 = static_cast<uint64_t>(blockIdx.x) << cx;
    const uint64_t *xr = x + static_cast<uint64_t>(blockIdx.y) * k;
    uint64_t *yr = y + static_cast<uint64_t>(blockIdx.y) * 2 * k;
    const uint64_t *mr = masks + static_cast<uint64_t>(blockIdx.y) * mask_coeffs;

    for (uint32_t t = threadIdx.x; t < tile; t += blockDim.x) s[pad(t)] = xr[x0 + t];
    tile_transform<false>(s, 0, cx, 0, 0, 0, inv_table, log_table);

    // The spread and the first DIT round, R1 stages over local bits [0, R1) of the codeword tile.
    const uint32_t c = cx + 1;
    const uint32_t r1 = c < 3 ? c : 3;
    const uint32_t groups = 1u << (c - r1);
    uint64_t held[4][4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const uint32_t gid = threadIdx.x + h * blockDim.x;
        if (gid < groups) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t t = (gid << (r1 - 1)) + i;
                held[h][i] = (i < (1 << (r1 - 1))) ? s[pad(t)] : 0;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const uint32_t gid = threadIdx.x + h * blockDim.x;
        if (gid >= groups) continue;
        uint64_t v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint64_t xt = x0 + (gid << (r1 - 1)) + i;
            uint64_t even = 0, odd = 0;
            if (i < (1 << (r1 - 1))) {
                const uint64_t ri = __brevll(xt) >> (64 - log_k);
                even = gl::lmul(held[h][i], scale[xt]);
                if (ri < mask_coeffs) {
                    const uint64_t scaled = gl::lmul(mr[ri], mask_pow[ri]);
                    even = gl::lsub(even, scaled);
                    odd = gl::lmul(scaled, shift_pow_k);
                }
            }
            v[2 * i] = even;
            v[2 * i + 1] = odd;
        }
        const uint32_t base = gid << r1;
        // The group offset E is 0 in the first round, so w0[st] = 1.
        const uint64_t one[3] = {1, 1, 1};
        if (r1 == 3) {
            radix<3, true>(v, one);
        } else if (r1 == 2) {
            radix<2, true>(v, one);
        } else {
            radix<1, true>(v, one);
        }
        for (uint32_t m = 0; m < (1u << r1); ++m) s[pad(base + m)] = v[m];
    }
    tile_transform<true>(s, r1, c, 0, 0, 0, fwd_table, log_table);
    for (uint32_t t = threadIdx.x; t < 2 * tile; t += blockDim.x) yr[2 * x0 + t] = gl::canon(s[pad(t)]);
}

// d_j[0] = H("column-init", commitment_id || u64(j)) for j < n. params: the column-init key
// (8 words), the column-step key (8 words), commitment_id (8 words); chains: 8 words per column.
extern "C" __global__ void column_init(uint32_t *chains, uint64_t n, const uint32_t *params) {
    const uint64_t j = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) return;
    uint32_t key[8], out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) key[i] = params[i];
    b3::hash_words(
        key, 10,
        [&](int w) -> uint32_t {
            if (w < 8) return params[16 + w];
            return w == 8 ? static_cast<uint32_t>(j) : static_cast<uint32_t>(j >> 32);
        },
        out);
#pragma unroll
    for (int i = 0; i < 8; ++i) chains[8 * j + i] = out[i];
}

// d_j <- H("column-step", d_j || symbols of rows 0..count-1 at column j) for j < n, the rows
// being codewords of n symbols at `codewords`, 1 <= count <= 124.
template <int COUNT>
__device__ __forceinline__ void step_column(uint32_t *chains, const uint64_t *codewords,
                                            uint64_t n, uint64_t j, int count,
                                            const uint32_t *params) {
    uint32_t key[8], cv[8], out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        key[i] = params[8 + i];
        cv[i] = chains[8 * j + i];
    }
    if constexpr (COUNT > 0) {
        uint64_t sym[COUNT];
#pragma unroll
        for (int r = 0; r < COUNT; ++r) sym[r] = codewords[static_cast<uint64_t>(r) * n + j];
        b3::hash_words(
            key, 8 + 2 * COUNT,
            [&](int w) -> uint32_t {
                if (w < 8) return cv[w];
                const uint64_t v = sym[(w - 8) >> 1];
                return (w & 1) ? static_cast<uint32_t>(v >> 32) : static_cast<uint32_t>(v);
            },
            out);
    } else {
        b3::hash_words(
            key, 8 + 2 * count,
            [&](int w) -> uint32_t {
                if (w < 8) return cv[w];
                const uint64_t v = codewords[static_cast<uint64_t>((w - 8) >> 1) * n + j];
                return (w & 1) ? static_cast<uint32_t>(v >> 32) : static_cast<uint32_t>(v);
            },
            out);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) chains[8 * j + i] = out[i];
}

extern "C" __global__ void column_step(uint32_t *chains, const uint64_t *codewords, uint64_t n,
                                       uint32_t count, const uint32_t *params) {
    const uint64_t j = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) return;
    if (count == 32) {
        step_column<32>(chains, codewords, n, j, 32, params);
    } else {
        step_column<0>(chains, codewords, n, j, static_cast<int>(count), params);
    }
}

// P5's column answers (#735): out[r * m + j] = codewords[r * n + columns[j]] for r < rows, j < m.
extern "C" __global__ void gather_columns(const uint64_t *codewords, uint64_t n, std::uint32_t rows,
                                          const uint64_t *columns, std::uint32_t m, uint64_t *out) {
    const uint64_t total = static_cast<uint64_t>(rows) * m;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const uint64_t r = i / m, j = i % m, c = columns[j];
        out[i] = c < n ? codewords[r * n + c] : 0;  // an index past the codeword reads nothing
    }
}

// Rate-1/4 weight codewords from their two rate-1/2 cosets (Device::rate4_encoders):
// out[2i] = even[i], out[2i + 1] = odd[i] for i < total.
extern "C" __global__ void interleave2(const uint64_t *even, const uint64_t *odd, uint64_t total,
                                       uint64_t *out) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        out[2 * i] = even[i];
        out[2 * i + 1] = odd[i];
    }
}
