// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The table registry and integer-only table generators with pinned digests (RFC-0008).
//!
//! A [`TableSpec`] names a table in the claim list: its [`TableKind`] with the kind's
//! parameters, its row count and its digest (RFC-0008 'Table specifications', 'Generators and
//! digests'). [`GeneratedTable`] builds one from its kind, or regenerates one that a claim list
//! states and checks it, and implements [`Table`]: the rows that the LogUp prover commits to,
//! and the witness-side lookup of a key.
//!
//! This version holds the arithmetic kinds, `Range`, `Requant`, `Pow2`, `Bitwise8`, `AesSbox`
//! and `Gf256Mul`, whose values are exact integer functions, and the fixed-point kinds `Exp`,
//! `Exp2Frac` and `Silu`, which RFC-0008 'Fixed-point generators' defines to the last integer.
//! RoPE's constants are no lookup table: [`RopeTable`] holds a `Rope` specification and its
//! digest, and [`RopeConstants`] computes its `(C, S)` pairs on demand. The structured evaluation
//! of a table's multilinear extension follows (RFC-0008 'Structured evaluation').
//!
//! `spec/vectors/tables/` pins every kind's TCE encoding, digest and rows, written by an
//! independent Python oracle. It checks the AES tables against FIPS-197's known-answer tests,
//! and every fixed-point value against a 60-digit reference. `cargo xtask lint` keeps floating
//! point out of the generators.

#![forbid(unsafe_code)]

mod error;
mod fixed;
mod gf256;
mod rope;
mod spec;
mod table;

pub use error::TableError;
pub use rope::{RopeConstants, RopeTable};
pub use spec::{BitOp, TableId, TableKind, TableSpec};
pub use table::{GeneratedTable, Table};
