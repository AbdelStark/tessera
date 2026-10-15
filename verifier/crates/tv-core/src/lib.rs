// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation and lowering, range re-derivation, schedule derivation and every proof
//! check, run in the fixed order of RFC-0019.
//!
//! # Features
//!
//! - `test-params` accepts the `test-small` parameter set, which certifies nothing. Tests only: a
//!   release build with it fails to compile.

#![forbid(unsafe_code)]
// RFC-0019 'Dependencies': verification code must not panic on any input.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-params", not(debug_assertions)))]
compile_error!(
    "the `test-params` feature accepts parameter sets that certify nothing; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);
