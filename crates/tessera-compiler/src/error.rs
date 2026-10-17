// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). Everything the compiler reads is public, so an error
//! may name nodes, tensors, tables and rules, and never holds a value the model computes.

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_ir::{IrError, NodeId, TableId};

/// Why a graph cannot be compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// A gadget input that must be a true integer has a range, but one the gadget or its table
    /// cannot take (`ERR-COMPILE-RANGE`, RFC-0006 'Range analysis').
    Range {
        /// The gadget's node.
        node: NodeId,
        /// The input, counted from 0.
        input: usize,
        /// What the range breaks.
        rule: &'static str,
    },
    /// A gadget input that must be a true integer has no bound, so it may have wrapped modulo
    /// `P` (`ERR-COMPILE-WRAP`, RFC-0006 'Range analysis').
    Wrap {
        /// The gadget's node.
        node: NodeId,
        /// The input, counted from 0.
        input: usize,
    },
    /// A node names a table that the given specifications lack, or one of a kind its operation
    /// cannot use: a bug in the code that built the graph (`ERR-INTERNAL`).
    Table {
        /// The node.
        node: NodeId,
        /// The table.
        table: TableId,
        /// What is wrong with it.
        rule: &'static str,
    },
    /// The graph breaks a rule of RFC-0006 'Operation set' (`ERR-INTERNAL`).
    Graph(IrError),
}

impl CompileError {
    /// The stable code of 04 §3.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Range { .. } => ErrorCode::CompileRange,
            Self::Wrap { .. } => ErrorCode::CompileWrap,
            Self::Table { .. } | Self::Graph(_) => ErrorCode::Internal,
        }
    }
}

/// For example, `node 12, input 0: a token lies in [0, vocab)`.
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range { node, input, rule } => {
                write!(f, "node {}, input {input}: {rule}", node.0)
            }
            Self::Wrap { node, input } => write!(
                f,
                "node {}, input {input}: no bound proves the value a true integer",
                node.0
            ),
            Self::Table { node, table, rule } => {
                write!(f, "node {}, table {}: {rule}", node.0, table.0)
            }
            Self::Graph(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<IrError> for CompileError {
    fn from(error: IrError) -> Self {
        Self::Graph(error)
    }
}
