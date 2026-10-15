// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation and lowering, range re-derivation, schedule derivation and every proof
//! check, run in the fixed order of RFC-0019.

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
