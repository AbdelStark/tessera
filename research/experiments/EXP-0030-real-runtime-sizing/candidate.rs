// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Frozen EXP-0029 public-only adapter helpers, with a normal module boundary.

use std::{collections::BTreeMap, error::Error};
use tessera_compiler::{Gadget, Lowering, Selection};
use tessera_ir::{Op, QuantPoint, Site};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

fn table(tables: &mut Vec<TableSpec>, kind: TableKind) -> Result<TableId, Box<dyn Error>> {
    if let Some(t) = tables
        .iter()
        .filter(|t| t.kind == kind)
        .min_by_key(|t| t.id)
    {
        return Ok(t.id);
    }
    let id = TableId(tables.iter().map(|t| t.id.0 + 1).max().unwrap_or(0));
    let generated = GeneratedTable::new(id, kind)?;
    tables.push(generated.spec().clone());
    Ok(id)
}

pub(super) fn patch(
    lowering: &Lowering,
    selection: &mut Selection,
    plans: &BTreeMap<u32, (u8, u8, u8)>,
) -> Result<usize, Box<dyn Error>> {
    let mut count = 0;
    let mut reads = Vec::new();
    for (id, gadget) in &mut selection.gadgets {
        let node = &lowering.graph.nodes[id.0 as usize];
        let Some((
            QuantPoint::Layer {
                layer,
                site: Site::MlpResidual,
            },
            _,
        )) = lowering
            .sites
            .iter()
            .find(|(_, tensor)| node.outputs.contains(tensor))
        else {
            continue;
        };
        let &(b, s, t) = plans.get(layer).ok_or("a plan per MLP residual layer")?;
        let Op::Requant { shift, lo, hi } = node.op else {
            return Err("a Requant site".into());
        };
        let r = lowering.graph.shift_classes[shift.0 as usize]
            .iter()
            .map(|r| r.checked_sub(s).ok_or("positive reduced shift"))
            .collect::<Result<Vec<_>, _>>()?;
        let k = b
            .checked_sub(s)
            .and_then(|b| b.checked_sub(t))
            .ok_or("positive top")?;
        let low = if s == 0 {
            None
        } else {
            Some(table(&mut selection.tables, TableKind::Range { bits: s })?)
        };
        let mid = table(&mut selection.tables, TableKind::Range { bits: t })?;
        let top = table(&mut selection.tables, TableKind::Range { bits: k })?;
        let sign = table(
            &mut selection.tables,
            TableKind::Requant {
                b: k + 1,
                r_by_class: vec![1],
                lo: -2,
                hi: 1,
            },
        )?;
        let high = table(
            &mut selection.tables,
            TableKind::Requant {
                b: t + 2,
                r_by_class: r,
                lo: i32::try_from(lo)?,
                hi: i32::try_from(hi)?,
            },
        )?;
        *gadget = Gadget::RequantWide {
            split: s,
            middle: t,
            low,
            mid,
            top,
            sign,
            high,
        };
        let ordered = low
            .into_iter()
            .chain([mid, top, sign, high])
            .collect::<Vec<_>>();
        reads.push((*id, ordered));
        count += 1;
    }
    for (node, ordered) in reads {
        selection.reads.insert(node, ordered);
    }
    Ok(count)
}
