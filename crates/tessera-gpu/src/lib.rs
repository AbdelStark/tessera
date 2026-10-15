// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CUDA kernels, the multi-GPU runtime and the memory scheduler (RFC-0018).

// Unsafe code is denied crate-wide. RFC-0001 permits it only for CUDA FFI and device memory;
// such a module opts in with a local `#[allow(unsafe_code)]` and justifies every block with a
// `// SAFETY:` comment.
#![deny(unsafe_code)]
