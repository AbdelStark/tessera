// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Goldilocks field `Fp`, its quadratic extension `Fp2`, NTT roots and the field traits
//! (RFC-0002).
//!
//! - [`Goldilocks`] is `Fp`, `P = 2^64 − 2^32 + 1`, with branch-free arithmetic on canonical
//!   values.
//! - [`F65537`] is the test field of the soundness-calibration tests (07 §4); it never carries a
//!   proof.
//! - [`Field`] and [`PrimeField`] are the interfaces the rest of the prover is generic over.

// Unsafe code is denied crate-wide. RFC-0001 permits it only in optional SIMD backends; such a
// module opts in with a local `#[allow(unsafe_code)]`, justifies every block with a `// SAFETY:`
// comment and is tested against the portable backend.
#![deny(unsafe_code)]

mod f65537;
mod goldilocks;
mod traits;

pub use f65537::F65537;
pub use goldilocks::Goldilocks;
pub use traits::{Field, PrimeField, ZeroElement, batch_inverse};
