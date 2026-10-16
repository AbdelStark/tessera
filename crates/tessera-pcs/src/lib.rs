// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The streaming advice commitment, the persistent weight commitment, openings and the hiding
//! budget (RFC-0011, RFC-0012).
//!
//! The encoding of RFC-0011 'Encoding' is here so far:
//!
//! - [`NttPlan`] holds the twiddles of one transform size, from `2^0` to `2^32` points, and runs
//!   the NTT, the inverse NTT and their coset variants in place, in natural order on both sides:
//!   `ntt` maps the coefficients of `f` to `f(ω_n^0), …, f(ω_n^{n−1})` with
//!   `ω_n = 7^((P − 1)/n)`, and `coset_ntt` to `f(s·ω_n^j)`.
//! - [`RowEncoder`] encodes rows of `k` values into codewords of `n = 2k` symbols on the coset
//!   `7·L`, adding a vanishing mask of `P` coefficients drawn from a [`MaskSource`]: `encode_row`
//!   writes one codeword into a buffer the caller provides, `encode_rows` a block of them.
//!
//! ```
//! use tessera_field::{Goldilocks, PrimeField};
//! use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};
//!
//! let fp = Goldilocks::from_u64_reduced;
//! let row: Vec<Goldilocks> = (1..=8).map(fp).collect();
//! let encoder = RowEncoder::new(8, 2)?;
//! let mut codeword = vec![Goldilocks::default(); encoder.codeword_len()];
//! let mut masks = [fp(11), fp(12)].into_iter();
//! let mut source = || masks.next().unwrap_or_default();
//! encoder.encode_row(&row, &mut source, &mut codeword)?;
//!
//! // The coefficients of g = f + (X^8 − 1)·m: the top half holds m.
//! let mut g = codeword.clone();
//! NttPlan::new(16)?.coset_intt(&mut g, COSET_SHIFT)?;
//! assert_eq!(&g[8..], &[fp(11), fp(12), fp(0), fp(0), fp(0), fp(0), fp(0), fp(0)]);
//! // On H the mask vanishes: folding g modulo X^8 − 1 gives f, whose values are the row.
//! let mut f: Vec<Goldilocks> = (0..8).map(|i| g[i] + g[i + 8]).collect();
//! NttPlan::new(8)?.ntt(&mut f)?;
//! assert_eq!(f, row);
//! # Ok::<(), tessera_pcs::PcsError>(())
//! ```
//!
//! Rows, mask coefficients and codewords are secret-class (05 §3): nothing here logs, errors
//! carry lengths only ([`PcsError`]), and the `Debug` forms of the plan and the encoder show their
//! sizes. Control flow depends on lengths and indices, never on values (INV-ARCH-2).
//!
//! # Features
//!
//! - `test-rng` seeds masking randomness so that tests are reproducible. Tests only: a release
//!   build with it fails to compile.

#![forbid(unsafe_code)]

// Test-only features cannot reach a release build (RFC-0001 'Cargo features').
#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod encoding;
mod error;
mod ntt;

pub use encoding::{COSET_SHIFT, MaskSource, RowEncoder};
pub use error::PcsError;
pub use ntt::NttPlan;
