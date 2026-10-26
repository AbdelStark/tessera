// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public metadata of RFC-0007's independently derived gadget/table choices.
//! The validator and this diagnostic view run the same verifier-owned selection pass.

use crate::claims::GadgetPolicy;
use crate::lowering::{Graph, LoweredTable, NodeId};
use crate::ranges::Interval;
use crate::validation::ClaimsError;
use tv_tables::TableId;

/// The selected RFC-0009 constraint pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// One requantization lookup.
    Requant1,
    /// The exact low/high split.
    RequantSplit {
        /// Number of low key bits.
        split: u8,
    },
    /// The saturation-preserving wide-key decomposition (RFC-0024).
    RequantWide {
        /// Number of low key bits.
        split: u8,
        /// Number of middle key bits.
        middle: u8,
    },
    /// Lookup into the node's base table.
    Lookup,
    /// Causal maximum and reciprocal bracket.
    SoftmaxRow,
    /// Reciprocal-square-root bracket.
    RsqrtRow,
    /// Token indicator constraints.
    OneHot,
}

/// A public gadget decision, in graph-node order.
#[derive(Debug, PartialEq, Eq)]
pub struct Choice {
    /// The node whose constraints use this pattern.
    pub node: NodeId,
    /// Its actual independently chosen pattern and parameters.
    pub pattern: Pattern,
    /// Referenced table families in construction order; repeated ids are retained.
    pub tables: Vec<TableId>,
}

/// The public results of the verifier's range and gadget-selection pass.
#[derive(Debug)]
pub struct Selection {
    /// One independently analyzed interval per tensor.
    pub ranges: Vec<Interval>,
    /// Every base and generated lookup table descriptor, in canonical id order.
    pub tables: Vec<LoweredTable>,
    /// One decision per gadget node, in graph order.
    pub choices: Vec<Choice>,
}

/// Derive gadget choices and table descriptors from a verifier-owned public graph.
///
/// This does not regenerate table values or validate a complete claim list/proof. The normal
/// validator independently compares these same derived definitions to its regenerated tables.
///
/// # Errors
/// The existing RFC-0007 gadget-domain errors, `Memory` for failed allocation, or `Internal`
/// for inconsistent metadata produced by the verifier's own lowering. A graph that holds an
/// operation the lowering never creates (RFC-0066, RFC-0072: `Sub`, `ConstAdd`, `Sum`, `Slice`,
/// `GatherPublic`, `Bitwise`, `BitDecompose`, `BitFn`, `AssertEqual`, a `ConstMul` by a constant
/// per channel or a `TableMap` into an anchor table) is `Internal` before any gadget is numbered,
/// until #150 selects their gadgets. No secret data is read.
pub fn derive(graph: &Graph, policy: GadgetPolicy) -> Result<Selection, ClaimsError> {
    crate::range_checks::catalog(graph, policy)
}
