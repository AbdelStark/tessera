// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). A graph's structure is public; no error carries a
//! tensor's values.

use core::fmt;

use crate::{NodeId, TensorId, WeightRef};

/// Why a graph, or a node about to join one, breaks a rule of RFC-0006 'Operation set' or 03 §3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrError {
    /// A tensor's description is not one 03 §1 and §3 allow.
    Tensor {
        /// The tensor, or the id it would have taken.
        tensor: TensorId,
        /// The rule it breaks.
        rule: &'static str,
    },
    /// A graph input is neither public nor committed: virtual tensors and lookup outputs have a
    /// producer.
    InputKind {
        /// The tensor.
        tensor: TensorId,
    },
    /// A constant, a set of shift classes or a weight declared against the rules.
    Declaration {
        /// The rule it breaks.
        rule: &'static str,
    },
    /// The weight matrix is declared twice.
    DuplicateWeight {
        /// The matrix.
        weight: WeightRef,
    },
    /// An input names no tensor of the graph.
    UnknownTensor {
        /// The node.
        node: NodeId,
        /// The input, counted from 0.
        input: usize,
    },
    /// An input is produced by the node itself or a later one: a graph's nodes are in
    /// topological order, so this includes every cycle.
    Order {
        /// The node.
        node: NodeId,
        /// The input, counted from 0.
        input: usize,
    },
    /// A tensor is produced by two nodes.
    ProducedTwice {
        /// The tensor.
        tensor: TensorId,
    },
    /// A node names a constant, a set of shift classes or a weight the graph does not declare.
    UnknownReference {
        /// The node.
        node: NodeId,
        /// What it names.
        what: &'static str,
    },
    /// A node breaks a rule of its operation.
    Op {
        /// The node.
        node: NodeId,
        /// The operation, as RFC-0006 writes it.
        op: &'static str,
        /// The rule it breaks.
        rule: &'static str,
    },
    /// A graph's ids are not dense in position order (INV-DM-5).
    Id {
        /// What is out of place.
        rule: &'static str,
    },
}

/// For example, `node 3 (MatMulWeight): the weight's columns match the input's last axis`.
impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor { tensor, rule } => write!(f, "tensor {}: {rule}", tensor.0),
            Self::InputKind { tensor } => write!(
                f,
                "tensor {}: a graph input is public or committed",
                tensor.0
            ),
            Self::Declaration { rule } => write!(f, "declaration: {rule}"),
            Self::DuplicateWeight { weight } => {
                write!(
                    f,
                    "weight {:?} of layer {} is declared twice",
                    weight.ty, weight.layer
                )
            }
            Self::UnknownTensor { node, input } => {
                write!(f, "node {}: input {input} names no tensor", node.0)
            }
            Self::Order { node, input } => write!(
                f,
                "node {}: input {input} is produced by this node or a later one",
                node.0
            ),
            Self::ProducedTwice { tensor } => {
                write!(f, "tensor {} is produced by two nodes", tensor.0)
            }
            Self::UnknownReference { node, what } => {
                write!(f, "node {}: names an undeclared {what}", node.0)
            }
            Self::Op { node, op, rule } => write!(f, "node {} ({op}): {rule}", node.0),
            Self::Id { rule } => write!(f, "ids: {rule}"),
        }
    }
}

impl std::error::Error for IrError {}
