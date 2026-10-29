// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-request exact-greedy catalog, leaf and advice metadata (draft RFC-0053).
//! This owns one complete catalog; none of its returned obligations is proof acceptance.

use super::{GadgetError, Plan as Gadgets, greedy::Boundary};
use crate::advice::{Config, Layout};
use crate::claims::GadgetPolicy;
use crate::leaf::{ModelSchedule, split};
use crate::lowering::{Graph, LoweredTable, NodeId, Op};
use crate::validation::{ClaimsError, GadgetId, StatementField};
use tv_field::Fp;
use tv_tables::{Table, TableError, TableId, TableKind};

fn single<T>(value: T) -> Result<Vec<T>, ClaimsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(1)
        .map_err(|_| ClaimsError::Memory)?;
    values.push(value);
    Ok(values)
}
fn bits(n: u32) -> u32 {
    u32::BITS.saturating_sub(n.saturating_sub(1).leading_zeros())
}
fn domain(gadget: GadgetId) -> ClaimsError {
    ClaimsError::StatementMismatch {
        field: StatementField::Gadget { gadget },
    }
}
fn intern(
    tables: &mut Vec<LoweredTable>,
    width: u8,
    limit: u8,
    gadget: GadgetId,
) -> Result<TableId, ClaimsError> {
    if width > limit {
        return Err(domain(gadget));
    }
    if let Some(table) = tables
        .iter()
        .find(|t| t.kind == (TableKind::Range { bits: width }))
    {
        return Ok(table.id);
    }
    let id = TableId(u32::try_from(tables.len()).map_err(|_| ClaimsError::Internal)?);
    tables.try_reserve(1).map_err(|_| ClaimsError::Memory)?;
    tables.push(LoweredTable {
        id,
        kind: TableKind::Range { bits: width },
        log2_rows: width,
    });
    Ok(id)
}

/// One nonempty-output certificate, with identities fixed by the verifier.
#[derive(Clone, Copy, Debug)]
pub struct Certificate {
    gadget: GadgetId,
    ordinal: u32,
    table: TableId,
    boundary: Boundary,
    variables: u32,
    entries: u64,
}
impl Certificate {
    /// Separate maximum identity, after all reserved model node ids.
    pub const fn gadget(self) -> GadgetId {
        self.gadget
    }
    /// Request ordinal `r < R` in batch order (`0` for a single request).
    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
    /// Exact physical Range table in the complete runtime catalog.
    pub const fn table(self) -> TableId {
        self.table
    }
    /// Original-source native reductions; every returned graph/PCS claim is still owed.
    pub const fn boundary(self) -> Boundary {
        self.boundary
    }
    /// Key's padded output-plus-vocabulary variable count, not the model domain's.
    pub const fn variables(self) -> u32 {
        self.variables
    }
    /// Every padded lookup entry; multiplicities include both padded axes.
    pub const fn entries(self) -> u64 {
        self.entries
    }
}

/// Complete public runtime geometry and independently regenerated tables for one sequence or batch
/// (RFC-0053, RFC-0071). Model tables borrow the validated model; at most one implicit Range is
/// added.
#[derive(Debug)]
pub struct Plan<'a> {
    gadgets: Gadgets<'a>,
    model_tables: &'a [Table],
    implicit: Option<Table>,
    certificates: Vec<Certificate>,
}
impl<'a> Plan<'a> {
    /// Derive from the actual-length graph, public prompt/ordinal and validated model tables.
    /// Empty outputs retain the model catalog/layout and owe separate canonical token binding.
    ///
    /// # Errors
    /// Existing selection errors; `StatementMismatch(Gadget)` for an unsupported greedy interval/policy;
    /// `Internal` for request/ordinal or own model-table mismatch; `Memory` on failed reserves.
    pub fn derive(
        graph: &'a Graph,
        policy: GadgetPolicy,
        prompt: u32,
        ordinal: u32,
        model_tables: &'a [Table],
    ) -> Result<Self, ClaimsError> {
        if graph.slots().is_some_and(|s| s.lengths().len() >= 2) {
            return Err(ClaimsError::Internal);
        }
        let n = graph.length();
        if prompt == 0 || prompt > n {
            return Err(ClaimsError::Internal);
        }
        let outputs = n.checked_sub(prompt).ok_or(ClaimsError::Internal)?;
        Self::derive_inner(graph, policy, &[(prompt, outputs)], ordinal, model_tables)
    }

    /// Derive the runtime plan for a slotted batch of `(prompt_len_r, output_len_r)` requests
    /// (RFC-0071 'The greedy certificate and scoring'). Reduces to [`Self::derive`] with
    /// certificate ordinal `0` for a batch of one request.
    ///
    /// # Errors
    /// Existing selection errors; `StatementMismatch(Gadget)` for an unsupported greedy
    /// interval/policy; `Internal` for request/slot or own model-table mismatch; `Memory` on
    /// failed reserves.
    pub fn derive_batch(
        graph: &'a Graph,
        policy: GadgetPolicy,
        requests: &[(u32, u32)],
        model_tables: &'a [Table],
    ) -> Result<Self, ClaimsError> {
        Self::derive_inner(graph, policy, requests, 0, model_tables)
    }

    fn derive_inner(
        graph: &'a Graph,
        policy: GadgetPolicy,
        requests: &[(u32, u32)],
        cert_ordinal: u32,
        model_tables: &'a [Table],
    ) -> Result<Self, ClaimsError> {
        if requests.is_empty() {
            return Err(ClaimsError::Internal);
        }
        if let Some(slots) = graph.slots() {
            if slots.lengths().len() != requests.len() {
                return Err(ClaimsError::Internal);
            }
            for (&(prompt, outputs), &len) in requests.iter().zip(slots.lengths()) {
                if prompt == 0 || prompt.checked_add(outputs) != Some(len) {
                    return Err(ClaimsError::Internal);
                }
            }
        } else {
            let &[(prompt, outputs)] = requests else {
                return Err(ClaimsError::Internal);
            };
            if prompt == 0 || prompt.checked_add(outputs) != Some(graph.length()) {
                return Err(ClaimsError::Internal);
            }
        }
        let mut gadgets = Gadgets::derive(graph, policy)?;
        if model_tables.len() != gadgets.selected.tables.len()
            || model_tables
                .iter()
                .zip(&gadgets.selected.tables)
                .any(|(a, b)| {
                    a.spec().id != b.id
                        || a.spec().kind != b.kind
                        || a.spec().log2_rows != b.log2_rows
                })
        {
            return Err(ClaimsError::Internal);
        }
        let out_count = requests.iter().filter(|&&(_, o)| o > 0).count();
        let mut implicit = None;
        let mut certificates = Vec::new();
        if out_count > 0 {
            certificates
                .try_reserve_exact(out_count)
                .map_err(|_| ClaimsError::Memory)?;
            let gadget = GadgetId(
                u32::try_from(graph.nodes().len())
                    .ok()
                    .and_then(|x| x.checked_add(cert_ordinal))
                    .ok_or(ClaimsError::Internal)?,
            );
            let Op::OneHot { vocab } = graph.nodes().first().ok_or(ClaimsError::Internal)?.op()
            else {
                return Err(ClaimsError::Internal);
            };
            let vocab_bits = bits(*vocab);
            let mut shared_table = None;
            for (r, &(prompt, outputs)) in requests.iter().enumerate() {
                if outputs == 0 {
                    continue;
                }
                let req_ordinal = u32::try_from(r).map_err(|_| ClaimsError::Internal)?;
                let boundary = gadgets
                    .greedy_boundary_in_slot(gadget, req_ordinal, prompt, outputs)
                    .map_err(|e| match e {
                        GadgetError::Memory => ClaimsError::Memory,
                        _ => domain(gadget),
                    })?;
                let table = match shared_table {
                    Some(id) => id,
                    None => {
                        let width = boundary.range_bits();
                        let id = intern(
                            &mut gadgets.selected.tables,
                            width,
                            policy.max_log2_rows,
                            gadget,
                        )?;
                        if usize::try_from(id.0).ok() == Some(model_tables.len()) {
                            implicit = Some(
                                Table::generate(id, TableKind::Range { bits: width }).map_err(
                                    |e| match e {
                                        TableError::Memory => ClaimsError::Memory,
                                        _ => ClaimsError::Internal,
                                    },
                                )?,
                            );
                        }
                        shared_table = Some(id);
                        id
                    }
                };
                let variables = bits(outputs)
                    .checked_add(vocab_bits)
                    .ok_or(ClaimsError::Internal)?;
                certificates.push(Certificate {
                    gadget,
                    ordinal: req_ordinal,
                    table,
                    boundary,
                    variables,
                    entries: 1u64.checked_shl(variables).ok_or(ClaimsError::Internal)?,
                });
            }
        }
        Ok(Self {
            gadgets,
            model_tables,
            implicit,
            certificates,
        })
    }
    /// Model gadgets with the complete, id-preserving runtime table catalog.
    pub fn gadgets(&self) -> &Gadgets<'a> {
        &self.gadgets
    }
    /// All model and implicit table definitions, in canonical id order.
    pub fn catalog(&self) -> &[LoweredTable] {
        &self.gadgets.selected.tables
    }
    /// Actual regenerated table for the leaf verifier; unknown ids have no table.
    pub fn table(&self, id: TableId) -> Option<&Table> {
        if let Some(table) = &self.implicit
            && table.spec().id == id
        {
            return Some(table);
        }
        self.model_tables.get(usize::try_from(id.0).ok()?)
    }
    /// First output-bearing request's certificate (`None` when no request has outputs).
    pub const fn certificate(&self) -> Option<Certificate> {
        self.certificates.as_slice().first().copied()
    }
    /// All output-bearing requests' certificates, in batch order (RFC-0071).
    pub const fn certificates(&self) -> &[Certificate] {
        self.certificates.as_slice()
    }
    /// Canonical runtime advice, including maximum and the complete used-table set.
    ///
    /// # Errors
    /// Existing layout errors (`Layout`, `Memory`, `Internal`), including the full row bound.
    pub fn advice_layout(&self, config: Config) -> Result<Layout<'_>, ClaimsError> {
        Layout::derive_runtime(
            self.gadgets.graph,
            &self.gadgets.selected,
            config,
            self.gadgets.graph.length(),
            self.certificate().map(|c| (c.gadget, c.table)),
        )
    }
    /// Complete lookup/table/forest schedule; shared Range multiplicities count all uses.
    ///
    /// # Errors
    /// `Internal` for own geometry/overflow, `Memory` for failed metadata reservation.
    pub fn leaf_schedule(&self) -> Result<ModelSchedule, ClaimsError> {
        let mut extra = Vec::new();
        extra
            .try_reserve_exact(self.certificates.len())
            .map_err(|_| ClaimsError::Memory)?;
        for c in &self.certificates {
            extra.push((c.table, c.entries));
        }
        ModelSchedule::from_selection(self.gadgets.graph, &self.gadgets.selected, &extra)
    }
    /// Model sites followed by the virtual greedy key columns at the actual LM-head node.
    /// Each key value remains owed to its native reduction; no maximum is a fake graph node.
    ///
    /// # Errors
    /// Existing model-site errors, own geometry `Internal`, failed reserves `Memory`.
    pub fn split_sites(&self) -> Result<Vec<split::Site>, ClaimsError> {
        let mut sites = split::model_sites(self.gadgets.graph, self.gadgets.policy)?;
        if !self.certificates.is_empty() {
            let node = NodeId(
                u32::try_from(
                    self.gadgets
                        .graph
                        .nodes()
                        .len()
                        .checked_sub(1)
                        .ok_or(ClaimsError::Internal)?,
                )
                .map_err(|_| ClaimsError::Internal)?,
            );
            let mut lookups = Vec::new();
            lookups
                .try_reserve_exact(self.certificates.len())
                .map_err(|_| ClaimsError::Memory)?;
            for c in &self.certificates {
                let mut coordinates = Vec::new();
                coordinates
                    .try_reserve_exact(
                        usize::try_from(c.variables).map_err(|_| ClaimsError::Memory)?,
                    )
                    .map_err(|_| ClaimsError::Memory)?;
                coordinates.extend(0..c.variables);
                let read = split::Read {
                    column: split::Column::greedy_key(c.ordinal),
                    coordinates,
                };
                lookups.push(split::Lookup {
                    table: c.table,
                    variables: c.variables,
                    key: split::Affine {
                        constant: Fp::ZERO,
                        terms: single((Fp::ONE, read))?,
                    },
                    value: None,
                    // Outside the lowered graph, the key has no token axis of RFC-0054's rules: a
                    // grouped stage would hold its epilogue block whole in token subset 0.
                    token: None,
                });
            }
            sites.try_reserve(1).map_err(|_| ClaimsError::Memory)?;
            sites.push(split::Site {
                layers: single(split::Layer { node, lookups })?,
            });
        }
        Ok(sites)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_interning_keeps_every_old_id_and_appends_missing_widths() {
        let mut tables = vec![LoweredTable {
            id: TableId(0),
            kind: TableKind::Range { bits: 4 },
            log2_rows: 4,
        }];
        assert!(matches!(
            intern(&mut tables, 4, 3, GadgetId(73)),
            Err(ClaimsError::StatementMismatch {
                field: StatementField::Gadget {
                    gadget: GadgetId(73)
                }
            })
        ));
        assert_eq!(intern(&mut tables, 4, 4, GadgetId(73)).unwrap(), TableId(0));
        assert_eq!(intern(&mut tables, 3, 4, GadgetId(73)).unwrap(), TableId(1));
        assert_eq!(intern(&mut tables, 3, 4, GadgetId(73)).unwrap(), TableId(1));
        assert_eq!(
            tables,
            vec![
                LoweredTable {
                    id: TableId(0),
                    kind: TableKind::Range { bits: 4 },
                    log2_rows: 4
                },
                LoweredTable {
                    id: TableId(1),
                    kind: TableKind::Range { bits: 3 },
                    log2_rows: 3
                }
            ]
        );
    }
}
