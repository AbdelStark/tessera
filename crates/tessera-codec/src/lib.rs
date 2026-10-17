// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tessera Canonical Encoding (TCE), the artifact envelope and the schema registry (RFC-0003).
//!
//! Every value Tessera hashes, transmits or stores has exactly one encoding:
//! - fixed-width little-endian integers;
//! - canonical field elements;
//! - `u64`-length-prefixed sequences;
//! - `u16`-tagged unions;
//! - no optional or unknown fields.
//!
//! [`Encode`] and [`Decode`] cover the primitive and composite types. `#[derive(Encode, Decode)]`
//! covers schema structs and enums. Every decoder rejects any input that is not the encoding of
//! a value, with a stable code and the offset of the first byte the failing check reads
//! (RFC-0003 'Limits').

#![forbid(unsafe_code)]

// The derive macros name this crate by path, which also works inside it.
extern crate self as tessera_codec;

pub mod envelope;
pub mod error;
#[doc(hidden)]
pub mod golden;
pub mod schema;
mod secret;
mod tce;
mod types;

pub use secret::{Public, Secret, SecretBuf};
pub use tce::{
    DEPTH_CEILING, Decode, DecodeError, DecodeLimits, Encode, Reader, Sink, decode, encode,
};
pub use tessera_codec_derive::{Decode, Encode};
pub use types::{Ascii, Bytes, DuplicateKey, Map, NotAscii, Rational};
pub use zeroize::Zeroize;
