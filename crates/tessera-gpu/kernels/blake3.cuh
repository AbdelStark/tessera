// SPDX-License-Identifier: Apache-2.0 OR MIT

// The BLAKE3 compression function and single-chunk hashing in key-derivation mode (RFC-0004
// 'Hash function and domains'), for inputs of at most one chunk (1,024 bytes). Every message
// word index is a compile-time constant after unrolling, so the state stays in registers.

#pragma once

#include <cstdint>

namespace b3 {

constexpr std::uint32_t CHUNK_START = 1u << 0;
constexpr std::uint32_t CHUNK_END = 1u << 1;
constexpr std::uint32_t ROOT = 1u << 3;
constexpr std::uint32_t DERIVE_KEY_MATERIAL = 1u << 6;

__device__ __forceinline__ std::uint32_t rotr(std::uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

#define B3_G(a, b, c, d, x, y)       \
    a = a + b + (x);                 \
    d = rotr(d ^ a, 16);             \
    c = c + d;                       \
    b = rotr(b ^ c, 12);             \
    a = a + b + (y);                 \
    d = rotr(d ^ a, 8);              \
    c = c + d;                       \
    b = rotr(b ^ c, 7);

// One round with the message words in schedule order s0..s15.
#define B3_ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) \
    B3_G(v0, v4, v8, v12, m[s0], m[s1])                                                \
    B3_G(v1, v5, v9, v13, m[s2], m[s3])                                                \
    B3_G(v2, v6, v10, v14, m[s4], m[s5])                                               \
    B3_G(v3, v7, v11, v15, m[s6], m[s7])                                               \
    B3_G(v0, v5, v10, v15, m[s8], m[s9])                                               \
    B3_G(v1, v6, v11, v12, m[s10], m[s11])                                             \
    B3_G(v2, v7, v8, v13, m[s12], m[s13])                                              \
    B3_G(v3, v4, v9, v14, m[s14], m[s15])

// cv <- the first eight output words of compressing block m (block_len bytes, counter 0).
__device__ __forceinline__ void compress(std::uint32_t cv[8], const std::uint32_t m[16],
                                         std::uint32_t block_len, std::uint32_t flags) {
    std::uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3];
    std::uint32_t v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
    std::uint32_t v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
    std::uint32_t v12 = 0, v13 = 0, v14 = block_len, v15 = flags;
    B3_ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    B3_ROUND(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
    B3_ROUND(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1)
    B3_ROUND(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6)
    B3_ROUND(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4)
    B3_ROUND(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7)
    B3_ROUND(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13)
    cv[0] = v0 ^ v8;
    cv[1] = v1 ^ v9;
    cv[2] = v2 ^ v10;
    cv[3] = v3 ^ v11;
    cv[4] = v4 ^ v12;
    cv[5] = v5 ^ v13;
    cv[6] = v6 ^ v14;
    cv[7] = v7 ^ v15;
}

#undef B3_ROUND
#undef B3_G

// H(domain, input) for an input of `words` little-endian 32-bit words (4 * words bytes, 1 to
// 256 words: one chunk), where `word(i)` yields word i: BLAKE3 keyed by the domain's context key
// with DERIVE_KEY_MATERIAL, the last block zero-padded. `out` receives the digest. Called with a
// constant `words`, the loops unroll and every word stays in a register.
template <typename Word>
__device__ __forceinline__ void hash_words(const std::uint32_t key[8], int words, Word word,
                                           std::uint32_t out[8]) {
    const int blocks = (words + 15) / 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = key[i];
#pragma unroll
    for (int b = 0; b < blocks; ++b) {
        std::uint32_t m[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int w = 16 * b + i;
            m[i] = w < words ? word(w) : 0u;
        }
        const int len = (b == blocks - 1) ? 4 * (words - 16 * b) : 64;
        std::uint32_t flags = DERIVE_KEY_MATERIAL;
        if (b == 0) flags |= CHUNK_START;
        if (b == blocks - 1) flags |= CHUNK_END | ROOT;
        compress(out, m, static_cast<std::uint32_t>(len), flags);
    }
}

}  // namespace b3
