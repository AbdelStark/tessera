// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Domain-separated BLAKE3, Merkle trees, chained column digests, the Fiat-Shamir transcript and
//! proof-of-work (RFC-0004).
//!
//! Every hash in Tessera is `H(domain, bytes)`, BLAKE3 in key-derivation mode with a context
//! string per [`Domain`]. On top of it: binary Merkle trees whose batch openings are canonical
//! multiproofs, and the chained column digests that let a streaming committer keep 32 bytes of
//! state per column.

#![forbid(unsafe_code)]

mod columns;
mod domain;
mod merkle;

pub use columns::{ColumnChains, ColumnError};
pub use domain::{Digest, Domain, Hasher, hash};
pub use merkle::{MerkleError, MerkleTree, MultiProof, merkle_node, verify_multiproof};
