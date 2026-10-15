// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CUDA kernels, the multi-GPU runtime and the memory scheduler (RFC-0018).
//!
//! # Features
//!
//! - `gpu` compiles the CUDA kernels under `kernels/` with nvcc from the CUDA toolkit 12.8 or
//!   later, into fatbins for `sm_90` and `sm_100`, and enables `Device`. Without it the crate
//!   is an empty stub, so every other crate builds and tests without a GPU (DEP-5). Setting
//!   `TESSERA_GPU_SKIP_KERNELS=1` builds the host code without CUDA, for type-checking only.

// Unsafe code is denied crate-wide. RFC-0001 permits it only for CUDA FFI and device memory;
// such a module opts in with a local `#[allow(unsafe_code)]` and justifies every block with a
// `// SAFETY:` comment.
#![deny(unsafe_code)]

#[cfg(feature = "gpu")]
mod device;

#[cfg(feature = "gpu")]
pub use device::{Device, DeviceBuffer, GpuError};
