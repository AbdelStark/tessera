// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Domain-separated BLAKE3, Merkle trees, chained column digests, the Fiat-Shamir transcript and
//! proof-of-work (RFC-0004).
//!
//! Every hash in Tessera is `H(domain, bytes)`, BLAKE3 in key-derivation mode with a context
//! string per [`Domain`]. On top of it: binary Merkle trees whose batch openings are canonical
//! multiproofs, the chained column digests that let a streaming committer keep 32 bytes of state
//! per column, and the Fiat–Shamir [`Transcript`] every challenge derives from.

#![forbid(unsafe_code)]

mod columns;
mod domain;
mod merkle;
mod transcript;

pub use columns::{ColumnChains, ColumnError};
pub use domain::{Digest, Domain, Hasher, hash};
pub use merkle::{MerkleError, MerkleTree, MultiProof, merkle_node, verify_multiproof};
pub use transcript::{Blake3, Mode, ParamSetId, SessionContext, Transcript, TranscriptHash};
