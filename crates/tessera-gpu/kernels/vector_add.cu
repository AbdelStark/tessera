// SPDX-License-Identifier: Apache-2.0 OR MIT

// Template kernel (RFC-0018 'Kernels'): element-wise wrapping addition of two u64 vectors.
// Kernel families follow its layout: one file per family under kernels/, extern "C" entry
// points with explicit lengths, and bounds checks on every index.

#include <cstdint>

extern "C" __global__ void vector_add(std::uint64_t *out, const std::uint64_t *a,
                                      const std::uint64_t *b, std::uint64_t n) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = a[i] + b[i];
    }
}
