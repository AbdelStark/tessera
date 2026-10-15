// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's own `Fp` and `Fp2` arithmetic, roots of unity and the NTT used to re-encode
//! combined rows, written from RFC-0002 without reference to the prover's field crate.
//!
//! Independence is the point (RFC-0001): a field bug shared with the prover would let an invalid
//! proof pass both. The algorithms therefore differ from the prover's on purpose: Montgomery
//! multiplication instead of the special-form reduction, and schoolbook `Fp2` products instead
//! of Karatsuba. Both are checked against the arbitrary-precision vectors in
//! `spec/vectors/field/`.

#![forbid(unsafe_code)]

mod fp;
mod fp2;
mod ntt;

pub use fp::Fp;
pub use fp2::Fp2;
pub use ntt::{NttError, NttValue, TWO_ADICITY, coset_intt, coset_ntt, intt, ntt, root_of_unity};
