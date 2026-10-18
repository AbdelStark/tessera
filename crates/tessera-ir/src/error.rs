// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1). A graph's structure is public; no error carries a
//! tensor's values.

use core::fmt;

use crate::{NodeId, TensorId, WeightRef};

/// Why a graph, or a node about to join one, breaks a rule of RFC-0006 'Operation set' or 03 §3;
/// or why a model's weights leave the domain the weight commitment takes them in (RFC-0040).
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
    /// A model of no layer. RFC-0007 'Verifier validation' requires `n_layers ≥ 1` in its
    /// architecture consistency check (`Architecture`), as the compiler does; without a layer,
    /// the per-layer weights have no layer count to pad to a power of two (RFC-0040 'The
    /// matrices').
    NoLayers,
    /// A weight entry of −128: quantization rounds weights into `[−127, 127]` (RFC-0005
    /// 'Pipeline', step 5), and range analysis and the compiler's validation take them there
    /// (RFC-0006 'Range analysis', 'Llama lowering').
    WeightOutOfRange {
        /// The weight.
        weight: WeightRef,
    },
    /// A channel's class at or above the shift classes of the site its weight's outputs reach:
    /// each class indexes its site's shifts (03 §2 'Channels', RFC-0006 'Llama lowering').
    ClassOutOfRange {
        /// The weight.
        weight: WeightRef,
        /// The site's shift classes: every class lies below.
        classes: usize,
    },
    /// A multiplier of 0: an affine site's multipliers lie in `[1, 2^16)` (03 §2 'Channels',
    /// RFC-0025, RFC-0006 'Llama lowering').
    MultiplierOutOfRange {
        /// The weight.
        weight: WeightRef,
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
            Self::NoLayers => write!(f, "a model has at least one layer"),
            Self::WeightOutOfRange { weight } => write!(
                f,
                "weight {:?} of layer {} has an entry outside [−127, 127]",
                weight.ty, weight.layer
            ),
            Self::ClassOutOfRange { weight, classes } => write!(
                f,
                "a channel class of weight {:?} of layer {} is not below its site's {classes} \
                 shift classes",
                weight.ty, weight.layer
            ),
            Self::MultiplierOutOfRange { weight } => write!(
                f,
                "a multiplier of weight {:?} of layer {} is 0, outside [1, 2^16)",
                weight.ty, weight.layer
            ),
        }
    }
}

impl std::error::Error for IrError {}
