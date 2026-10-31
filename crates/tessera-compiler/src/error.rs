// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). Everything the compiler reads is public, so an error
//! may name nodes, tensors, tables and rules, and never holds a value the model computes.

use core::fmt;

use tessera_codec::error::ErrorCode;
use tessera_ir::{IrError, NodeId, TableId, TensorId};
use tessera_tables::TableError;

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
    /// A graph input committed in the advice has an element that no operation proves in its
    /// declared range, which its readers trust (`ERR-COMPILE-RANGE`, RFC-0006 'Committed graph
    /// inputs').
    Unproven {
        /// The input.
        tensor: TensorId,
        /// What the graph lacks.
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
    /// The graph breaks a constraint its profile claims, S1 to S3 of RFC-0005 'Profiles'
    /// (`ERR-COMPILE-PROFILE`).
    Profile {
        /// The node.
        node: NodeId,
        /// The constraint, with what it requires.
        rule: &'static str,
    },
    /// The integer model contradicts its architecture or RFC-0005, or uses what the compiler does
    /// not support yet (`ERR-QUANT-ARCH`, RFC-0006 'Llama lowering').
    Model {
        /// The rule the model breaks.
        rule: &'static str,
    },
    /// A table the model's public constants define is one RFC-0008 refuses (`ERR-QUANT-ARCH`),
    /// or its dense core cannot be allocated (`ERR-RES-MEMORY`).
    ModelTable {
        /// The table's index in the lowering.
        table: TableId,
        /// Why the generator refuses it.
        error: TableError,
    },
    /// The compilation request is malformed, such as a sequence longer than the model's context
    /// (`ERR-USAGE`).
    Usage {
        /// What the request breaks.
        rule: &'static str,
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
    /// The graph has no stage schedule: its layers are not one template, or a claim reaches an
    /// operation RFC-0037 maps no claim through (`ERR-INTERNAL`, RFC-0037 'Layers and kinds').
    Schedule {
        /// The node, when one breaks the rule.
        node: Option<NodeId>,
        /// The rule.
        rule: &'static str,
    },
}

impl CompileError {
    pub(crate) fn table_generation(node: NodeId, table: TableId, error: TableError) -> Self {
        match error {
            TableError::Memory => Self::ModelTable { table, error },
            _ => Self::Table {
                node,
                table,
                rule: "the generator refuses a table selection built",
            },
        }
    }

    /// The stable code of 04 §3.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::ModelTable {
                error: TableError::Memory,
                ..
            } => ErrorCode::ResMemory,
            Self::Range { .. } | Self::Unproven { .. } => ErrorCode::CompileRange,
            Self::Wrap { .. } => ErrorCode::CompileWrap,
            Self::Profile { .. } => ErrorCode::CompileProfile,
            Self::Model { .. } | Self::ModelTable { .. } => ErrorCode::QuantArch,
            Self::Usage { .. } => ErrorCode::Usage,
            Self::Table { .. } | Self::Graph(_) | Self::Schedule { .. } => ErrorCode::Internal,
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
            Self::Unproven { tensor, rule } => write!(f, "tensor {}: {rule}", tensor.0),
            Self::Wrap { node, input } => write!(
                f,
                "node {}, input {input}: no bound proves the value a true integer",
                node.0
            ),
            Self::Profile { node, rule } => write!(f, "node {}: {rule}", node.0),
            Self::Model { rule } => write!(f, "the integer model breaks a rule: {rule}"),
            Self::ModelTable { table, error } => write!(f, "table {}: {error}", table.0),
            Self::Usage { rule } => f.write_str(rule),
            Self::Table { node, table, rule } => {
                write!(f, "node {}, table {}: {rule}", node.0, table.0)
            }
            Self::Graph(error) => error.fmt(f),
            Self::Schedule {
                node: Some(node),
                rule,
            } => write!(f, "node {}: {rule}", node.0),
            Self::Schedule { node: None, rule } => write!(f, "the stage schedule: {rule}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<IrError> for CompileError {
    fn from(error: IrError) -> Self {
        Self::Graph(error)
    }
}

#[cfg(test)]
mod memory_error_tests {
    use super::*;
    #[test]
    fn table_selection_resource_failures_do_not_become_invalid_model_errors() {
        let memory = CompileError::table_generation(NodeId(7), TableId(11), TableError::Memory);
        assert_eq!(memory.code(), ErrorCode::ResMemory);
        assert!(matches!(
            memory,
            CompileError::ModelTable {
                table: TableId(11),
                error: TableError::Memory
            }
        ));
        let invalid = CompileError::table_generation(NodeId(7), TableId(11), TableError::Digest);
        assert_eq!(invalid.code(), ErrorCode::Internal);
        assert!(matches!(
            invalid,
            CompileError::Table {
                node: NodeId(7),
                table: TableId(11),
                rule: "the generator refuses a table selection built"
            }
        ));
    }
}
