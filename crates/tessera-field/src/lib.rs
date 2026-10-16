// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Goldilocks field `Fp`, its quadratic extension `Fp2`, NTT roots and the field traits
//! (RFC-0002).
//!
//! - [`Goldilocks`] is `Fp`, `P = 2^64 − 2^32 + 1`, with branch-free arithmetic on canonical
//!   values.
//! - [`GoldilocksExt2`] is `Fp2 = Fp[X]/(X^2 − 7)`, about 2^128 elements, where every challenge
//!   lives.
//! - [`F65537`] and [`F65537Ext2`] are the test fields of the soundness-calibration tests
//!   (07 §4); they never carry a proof.
//! - [`Field`], [`PrimeField`] and [`QuadraticExtension`] are the interfaces the rest of the
//!   prover is generic over.

// Unsafe code is denied crate-wide. RFC-0001 permits it only in optional SIMD backends; such a
// module opts in with a local `#[allow(unsafe_code)]`, justifies every block with a `// SAFETY:`
// comment and is tested against the portable backend.
#![deny(unsafe_code)]

mod codec;
mod ext2;
mod f65537;
#[doc(hidden)]
pub mod golden;
mod goldilocks;
#[cfg(feature = "codegen-probe")]
pub mod probe;
mod traits;

pub use ext2::{ExtensionBase, F65537Ext2, GoldilocksExt2, QuadraticExt, QuadraticExtension};
pub use f65537::F65537;
pub use goldilocks::Goldilocks;
pub use traits::{Field, PrimeField, ZeroElement, batch_inverse};
