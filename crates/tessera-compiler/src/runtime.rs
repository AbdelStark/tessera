// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public per-request exact-greedy metadata (draft RFC-0053).

use crate::{CompileError, Lowering, Selection, analyze};
use tessera_ir::{ClaimList, GadgetId, IntRange, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind};

const REQUEST: &str = "the greedy request names a positive prompt within the model sequence";
const METADATA: &str = "runtime selection preserves the model's canonical table catalog";
const RANGE: &str = "greedy logits have a public interval strictly within the signed 28-bit bound";

fn usage(rule: &'static str) -> CompileError {
    CompileError::Usage { rule }
}

/// One maximum source and its virtual lookup column, derived from public data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyCertificate {
    /// Identity after every reserved model node id.
    pub gadget: GadgetId,
    /// Exact physical range table in the complete runtime catalog.
    pub table: TableId,
    /// Positive prompt length; logit time `prompt - 1` predicts its first output.
    pub prompt: u32,
    /// Model time length, including the prompt and every output.
    pub positions: u32,
    /// Unpadded model vocabulary.
    pub vocab: u32,
    /// Inclusive logit interval from public range analysis, never from a witness.
    pub logits: IntRange,
    /// Width of the physical Range table; at most 29 bits.
    pub range_bits: u8,
    /// Padded output-plus-vocabulary lookup variable count.
    pub variables: u32,
}
impl GreedyCertificate {
    /// Both padding axes contribute zero-key lookup requests.
    pub fn entries(self) -> u64 {
        1u64 << self.variables
    }
}

/// Model selection plus the optional implicit Range table, without changing the claim list.
#[derive(Clone, Debug)]
pub struct RuntimePlan {
    selection: Selection,
    certificate: Option<GreedyCertificate>,
}
impl RuntimePlan {
    /// Extend checked model metadata for one request. Empty outputs add no maximum or table.
    ///
    /// # Errors
    /// `Usage` for inconsistent model/request metadata or identity overflow; `Model` for an
    /// unsupported public logit interval or physical table policy; range/table errors propagate.
    pub fn derive(
        lowering: &Lowering,
        selection: &Selection,
        claims: &ClaimList,
        prompt: u32,
        ordinal: u32,
    ) -> Result<Self, CompileError> {
        if selection.tables != claims.tables {
            return Err(usage(METADATA));
        }
        let shape = &lowering
            .graph
            .tensors
            .get(lowering.logits.0 as usize)
            .ok_or_else(|| usage(REQUEST))?
            .shape;
        let [positions, vocab] = shape.as_slice() else {
            return Err(usage(REQUEST));
        };
        if prompt == 0 || prompt > *positions {
            return Err(usage(REQUEST));
        }
        let mut selection = selection.clone();
        let certificate = if prompt < *positions {
            let gadget = GadgetId(
                u32::try_from(lowering.graph.nodes.len())
                    .ok()
                    .and_then(|n| n.checked_add(ordinal))
                    .ok_or_else(|| usage(REQUEST))?,
            );
            let logits = analyze(&lowering.graph, &lowering.tables)?
                .of(lowering.logits)
                .and_then(|r| r.bounded())
                .filter(|r| -(1i128 << 28) < r.lo && r.hi < 1i128 << 28)
                .ok_or(CompileError::Model { rule: RANGE })?;
            let span = u32::try_from(logits.hi - logits.lo)
                .map_err(|_| CompileError::Model { rule: RANGE })?;
            let range_bits = (u32::BITS - span.leading_zeros()).max(1) as u8;
            if range_bits > claims.gadget_policy.max_log2_rows {
                return Err(CompileError::Model {
                    rule: "the greedy physical Range width fits GadgetPolicy.max_log2_rows",
                });
            }
            let kind = TableKind::Range { bits: range_bits };
            let table = match selection.tables.iter().find(|t| t.kind == kind) {
                Some(table) => table.id,
                None => {
                    let id = TableId(
                        u32::try_from(selection.tables.len()).map_err(|_| usage(METADATA))?,
                    );
                    let generated = GeneratedTable::new(id, kind)
                        .map_err(|error| CompileError::ModelTable { table: id, error })?;
                    selection.tables.push(generated.spec().clone());
                    id
                }
            };
            let bits = |n: u32| u32::BITS - n.saturating_sub(1).leading_zeros();
            let variables = bits(*positions - prompt) + bits(*vocab);
            if variables >= u64::BITS {
                return Err(usage(REQUEST));
            }
            Some(GreedyCertificate {
                gadget,
                table,
                prompt,
                positions: *positions,
                vocab: *vocab,
                logits,
                range_bits,
                variables,
            })
        } else {
            None
        };
        Ok(Self {
            selection,
            certificate,
        })
    }
    /// Complete catalog, preserving every model table id, kind and digest.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// A distinct certificate exists exactly when the request has outputs.
    pub const fn certificate(&self) -> Option<GreedyCertificate> {
        self.certificate
    }
}
