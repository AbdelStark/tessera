// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained public source metadata; constructor and tile work remain separate (RFC-0018).

use super::*;
use tessera_gadgets::{Lookup, Virtual};

pub(super) fn array<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

pub(super) fn affine<T>(form: &Affine<Goldilocks, T>) -> Option<usize> {
    array::<(Goldilocks, T)>(form.terms.capacity())
}

pub(super) fn lookups<T>(values: &Vec<Lookup<Goldilocks, T>>) -> Option<usize> {
    values.iter().try_fold(
        array::<Lookup<Goldilocks, T>>(values.capacity())?,
        |sum, value| {
            sum.checked_add(affine(&value.key)?)?
                .checked_add(value.value.as_ref().map_or(Some(0), affine)?)
        },
    )
}

fn constraints(value: &Constraints<Goldilocks>) -> Option<usize> {
    let bytes = array::<Committed>(value.committed.capacity())?
        .checked_add(array::<(Virtual, Affine<Goldilocks>)>(
            value.virtuals.capacity(),
        )?)?
        .checked_add(lookups(&value.lookups)?)?;
    value
        .virtuals
        .iter()
        .try_fold(bytes, |sum, (_, form)| sum.checked_add(affine(form)?))
}

impl Gadgets {
    pub(crate) fn heap_bytes(&self) -> Option<usize> {
        let mut bytes = array::<Option<RequantGadget>>(self.requant.capacity())?;
        for size in [
            array::<Vec<TableId>>(self.requests.capacity())?,
            array::<Option<Prepared>>(self.auxiliary.capacity())?,
            array::<TableId>(self.read.capacity())?,
            array::<TableId>(self.incomplete.capacity())?,
            array::<tessera_compiler::GreedyKeyColumn>(self.key_columns.capacity())?,
            array::<(TableId, u64)>(self.columns.capacity())?,
        ] {
            bytes = bytes.checked_add(size)?;
        }
        for request in &self.requests {
            bytes = bytes.checked_add(array::<TableId>(request.capacity())?)?;
        }
        for value in self.requant.iter().flatten() {
            bytes = bytes
                .checked_add(value.requant.heap_bytes()?)?
                .checked_add(constraints(&value.constraints)?)?
                .checked_add(array::<(TableId, u64)>(value.rows.capacity())?)?;
        }
        self.auxiliary
            .iter()
            .flatten()
            .try_fold(bytes, |sum, value| sum.checked_add(value.heap_bytes()?))
    }
}

impl Gadgets {
    /// Sum requested public metadata, without invoking scalar/witness constructors.
    pub(crate) fn construction_bytes(
        graph: &tessera_ir::Graph,
        selection: &Selection,
        context: TensorId,
    ) -> Result<u64, WitnessError> {
        use crate::memory::{add, array as requested, growing};
        use tessera_gadgets::{GadgetError, RsqrtRow, SoftmaxRow};
        use tessera_tables::{GeneratedTable, TableKind};
        let memory = WitnessError::Memory { tensor: context };
        let nodes = graph.nodes.len() as u64;
        let requests = graph.nodes.iter().try_fold(0u64, |sum, node| {
            add(
                sum,
                selection.reads(node.id).unwrap_or(&[]).len() as u64,
                context,
            )
        })?;
        let reads = selection.gadgets.iter().try_fold(0u64, |sum, &(node, _)| {
            let values = selection.reads(node).ok_or(WitnessError::Internal {
                rule: "every selected gadget has its ordered lookup requests",
            })?;
            add(sum, values.len() as u64, context)
        })?;
        let mut bytes = requested::<Option<RequantGadget>>(nodes, context)?;
        for value in [
            requested::<Option<Prepared>>(nodes, context)?,
            requested::<Vec<TableId>>(nodes, context)?,
            requested::<TableId>(requests, context)?,
            growing::<TableId>(reads, context)?,
            growing::<TableId>(requests, context)?,
        ] {
            bytes = add(bytes, value, context)?;
        }
        // A malformed role may load an Exp/Silu before the constructor rejects
        // its kind. Charge every named payload as well, without deduplicating ids.
        let payload =
            |id: TableId| -> Result<u64, WitnessError> {
                let spec = selection.tables.iter().find(|spec| spec.id == id).ok_or(
                    GadgetError::Table {
                        table: id,
                        rule: "the specifications hold every table the gadget names",
                    },
                )?;
                let core = GeneratedTable::core_storage_len(&spec.kind)
                    .map_err(|error| GadgetError::Spec { table: id, error })?;
                let classes = match &spec.kind {
                    TableKind::Requant { r_by_class, .. } | TableKind::Silu { r_by_class, .. } => {
                        r_by_class.len() as u64
                    }
                    _ => 0,
                };
                let mut out = add(requested::<i64>(core, context)?, classes, context)?;
                if matches!(spec.kind, TableKind::Exp { .. }) {
                    out = add(out, 8, context)?;
                }
                Ok(out)
            };
        for node in &graph.nodes {
            match node.op {
                Op::Requant { .. } => {
                    let (_, gadget) = selection.of(node.id).ok_or(WitnessError::Internal {
                        rule: "the selection gives every requantization its gadget",
                    })?;
                    let bound = Requant::memory().ok_or(memory)?;
                    bytes = add(
                        bytes,
                        add(bound.template_bytes(), bound.constraints_bytes(), context)?,
                        context,
                    )?;
                    let mut table = |id| -> Result<(), WitnessError> {
                        bytes = add(bytes, payload(id)?, context)?;
                        Ok(())
                    };
                    match gadget {
                        Gadget::Requant1 { table: id } => table(id)?,
                        Gadget::RequantSplit { low, high, .. } => {
                            table(low)?;
                            table(high)?;
                        }
                        Gadget::RequantWide {
                            low,
                            mid,
                            top,
                            sign,
                            high,
                            ..
                        } => {
                            if let Some(low) = low {
                                table(low)?;
                            }
                            for id in [mid, top, sign, high] {
                                table(id)?;
                            }
                        }
                        _ => {}
                    }
                    for value in [
                        growing::<&GeneratedTable>(5, context)?,
                        requested::<&GeneratedTable>(5, context)?,
                        requested::<(TableId, u64)>(5, context)?,
                    ] {
                        bytes = add(bytes, value, context)?;
                    }
                }
                Op::RsqrtRow { .. } => {
                    let bound = RsqrtRow::memory().ok_or(memory)?;
                    for value in [
                        bound.template_bytes(),
                        bound.constraints_bytes(),
                        requested::<auxiliary::Root>(1, context)?,
                        growing::<&GeneratedTable>(32, context)?,
                        requested::<&GeneratedTable>(4, context)?,
                        requested::<(TableId, u64)>(32, context)?,
                        requested::<TableId>(189, context)?,
                    ] {
                        bytes = add(bytes, value, context)?;
                    }
                }
                Op::SoftmaxRow { exp_table, .. } => {
                    let bound = SoftmaxRow::memory(exp_table, &selection.tables)?;
                    for value in [
                        bound.template_bytes(),
                        bound.constraints_bytes(),
                        requested::<auxiliary::Softmax>(1, context)?,
                        growing::<&GeneratedTable>(4, context)?,
                        requested::<&GeneratedTable>(2, context)?,
                        requested::<(TableId, u64)>(4, context)?,
                        requested::<TableId>(100, context)?,
                    ] {
                        bytes = add(bytes, value, context)?;
                    }
                }
                _ => {}
            }
        }
        Ok(bytes)
    }
}
