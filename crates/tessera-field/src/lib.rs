// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Goldilocks field `Fp`, its quadratic extension `Fp2`, NTT roots and the field traits
//! (RFC-0002).

// Unsafe code is denied crate-wide. RFC-0001 permits it only in optional SIMD backends; such a
// module opts in with a local `#[allow(unsafe_code)]`, justifies every block with a `// SAFETY:`
// comment and is tested against the portable backend.
#![deny(unsafe_code)]
