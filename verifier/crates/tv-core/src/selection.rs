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

impl Selection {
    /// RFC-0076 N1: the selected envelope may be wider than the analyzed input.
    /// Recover it from the independently derived table widths, including the
    /// Wide virtual top range, rather than reusing the original signed width.
    pub(crate) fn key_bits(&self, choice: &Choice) -> Result<u32, ClaimsError> {
        let bits = |id: &TableId| {
            let table = self
                .tables
                .get(usize::try_from(id.0).map_err(|_| ClaimsError::Internal)?)
                .ok_or(ClaimsError::Internal)?;
            match &table.kind {
                tv_tables::TableKind::Requant { b, .. } | tv_tables::TableKind::Silu { b, .. } => {
                    Ok(u32::from(*b))
                }
                _ => Err(ClaimsError::Internal),
            }
        };
        let last = choice.tables.last().ok_or(ClaimsError::Internal)?;
        match choice.pattern {
            Pattern::Lookup | Pattern::Requant1 => bits(last),
            Pattern::RequantSplit { split } => bits(last)?
                .checked_add(u32::from(split))
                .ok_or(ClaimsError::Internal),
            Pattern::RequantWide { split, middle } => {
                let top_table = choice
                    .tables
                    .iter()
                    .rev()
                    .nth(1)
                    .ok_or(ClaimsError::Internal)?;
                bits(top_table)?
                    .checked_sub(1)
                    .and_then(|k| k.checked_add(u32::from(split)))
                    .and_then(|b| b.checked_add(u32::from(middle)))
                    .ok_or(ClaimsError::Internal)
            }
            _ => Err(ClaimsError::Internal),
        }
    }
}
