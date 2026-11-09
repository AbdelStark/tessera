// SPDX-License-Identifier: Apache-2.0 OR MIT

// Goldilocks arithmetic, p = 2^64 - 2^32 + 1 (RFC-0002). Every operation takes and returns
// canonical values in [0, p), so device results compare with the CPU's byte for byte.

#pragma once

#include <cstdint>

namespace gl {

constexpr std::uint64_t P = 0xFFFFFFFF00000001ull;
// 2^64 mod p = 2^32 - 1.
constexpr std::uint64_t EPS = 0xFFFFFFFFull;

__device__ __forceinline__ std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t s = a + b;
    // A carry dropped 2^64 = EPS (mod p); a + b - p < p then needs no further reduction.
    if (s < a) s += EPS;
    if (s >= P) s -= P;
    return s;
}

__device__ __forceinline__ std::uint64_t sub(std::uint64_t a, std::uint64_t b) {
    std::uint64_t d = a - b;
    // A borrow added 2^64 = EPS (mod p): a - b + p < p.
    if (a < b) d -= EPS;
    return d;
}

// hi * 2^64 + lo mod p, with 2^64 = EPS and 2^96 = -1 (mod p).
__device__ __forceinline__ std::uint64_t reduce128(std::uint64_t lo, std::uint64_t hi) {
    const std::uint64_t hh = hi >> 32;
    const std::uint64_t hl = hi & EPS;
    std::uint64_t t0 = lo - hh;
    if (lo < hh) t0 -= EPS;
    const std::uint64_t t1 = hl * EPS;
    std::uint64_t t2 = t0 + t1;
    if (t2 < t1) t2 += EPS;
    if (t2 >= P) t2 -= P;
    return t2;
}

__device__ __forceinline__ std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    return reduce128(a * b, __umul64hi(a, b));
}

// Lazy variants for the transforms: inputs and outputs are any 64-bit values, congruent mod p to
// the exact result; canon() makes one canonical before it leaves the kernel.
__device__ __forceinline__ std::uint64_t canon(std::uint64_t a) { return a >= P ? a - P : a; }

__device__ __forceinline__ std::uint64_t ladd(std::uint64_t a, std::uint64_t b) {
    std::uint64_t s = a + b;
    // Each dropped 2^64 is EPS (mod p); the second carry leaves s below 2 EPS, so no third.
    std::uint64_t c = s < a ? EPS : 0;
    s += c;
    c = s < c ? EPS : 0;
    return s + c;
}

__device__ __forceinline__ std::uint64_t lsub(std::uint64_t a, std::uint64_t b) {
    std::uint64_t d = a - b;
    // Each borrow added 2^64 = EPS (mod p); after a second, d is at least 2^64 - 2 EPS.
    std::uint64_t c = a < b ? EPS : 0;
    const std::uint64_t e = d - c;
    c = d < c ? EPS : 0;
    return e - c;
}

// hi * 2^64 + lo mod p for any 64-bit a * b, without the final canonical subtraction.
__device__ __forceinline__ std::uint64_t lmul(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t lo = a * b;
    const std::uint64_t hi = __umul64hi(a, b);
    const std::uint64_t hh = hi >> 32;
    const std::uint64_t hl = hi & EPS;
    std::uint64_t t0 = lo - hh;
    if (lo < hh) t0 -= EPS;
    const std::uint64_t t1 = hl * EPS;
    std::uint64_t t2 = t0 + t1;
    if (t2 < t1) t2 += EPS;
    return t2;
}

// x * 2^E mod p, lazily, for 0 <= E < 192 (2^96 = -1, so 2^192 = 1): shifts and one reduction
// per 64-bit step instead of a full multiplication. The small roots of unity are such powers.
template <int E>
__device__ __forceinline__ std::uint64_t lmul_pow2(std::uint64_t x) {
    static_assert(E >= 0 && E < 192, "2^E with 0 <= E < 192");
    if constexpr (E >= 96) {
        return lsub(0, lmul_pow2<E - 96>(x));
    } else if constexpr (E >= 64) {
        return lmul_pow2<E - 32>(lmul_pow2<32>(x));
    } else if constexpr (E == 0) {
        return x;
    } else {
        const std::uint64_t lo = x << E;
        const std::uint64_t hi = x >> (64 - E);
        const std::uint64_t hh = hi >> 32;
        const std::uint64_t hl = hi & EPS;
        std::uint64_t t0 = lo - hh;
        if (lo < hh) t0 -= EPS;
        const std::uint64_t t1 = hl * EPS;
        std::uint64_t t2 = t0 + t1;
        if (t2 < t1) t2 += EPS;
        return t2;
    }
}

// SplitMix64 of a 64-bit counter: the counter-based input generator of RFC-0020's protocol,
// reduced to [0, p) by one conditional subtraction (benchmark inputs, not a uniform sampler).
__device__ __forceinline__ std::uint64_t counter_value(std::uint64_t counter) {
    std::uint64_t z = counter * 0x9E3779B97F4A7C15ull + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return z >= P ? z - P : z;
}

}  // namespace gl
