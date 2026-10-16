// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The table registry and integer-only table generators with pinned digests (RFC-0008).
//!
//! A [`TableSpec`] names a table in the claim list: its [`TableKind`] with the kind's
//! parameters, its row count and its digest (RFC-0008 'Table specifications', 'Generators and
//! digests'). [`ArithmeticTable`] builds one from its kind, or regenerates one that a claim list
//! states and checks it, and implements [`Table`]: the rows that the LogUp prover commits to,
//! and the witness-side lookup of a key.
//!
//! This version holds the arithmetic kinds, `Range`, `Requant`, `Pow2`, `Bitwise8`, `AesSbox`
//! and `Gf256Mul`, whose values are exact integer functions. The transcendental kinds follow,
//! appended to [`TableKind`], and so does the structured evaluation of a table's multilinear
//! extension (RFC-0008 'Structured evaluation').
//!
//! `spec/vectors/tables/` pins every kind's TCE encoding, digest and rows, written by an
//! independent Python oracle that also checks the AES tables against FIPS-197's known-answer
//! tests.

#![forbid(unsafe_code)]

mod error;
mod gf256;
mod spec;
mod table;

pub use error::TableError;
pub use spec::{BitOp, TableId, TableKind, TableSpec};
pub use table::{ArithmeticTable, Table};
