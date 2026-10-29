// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Diagnostic metadata adapter. It never exports a claim list, witness or proof.
#![forbid(unsafe_code)]

use std::{collections::BTreeMap, error::Error, fs, path::Path};
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::schedule::LayerMap;
use tessera_compiler::{Gadget, Lowering, PublicModel, Selection, analyze, lower_batch, select};
use tessera_ir::{ClaimList, GadgetPolicy, IntegerModel, Op, QuantPoint, Site, WeightType};
use tessera_prover::LeafLayout;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

const RULE: &str = "a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')";

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

fn patch(
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

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mode = std::env::args().nth(1).ok_or("mode tiny or real")?;
    let source: ClaimList = envelope::open(
        &fs::read(root.join("real-1b-claims.tcl"))?,
        &schema::CLAIM_LIST,
        DecodeLimits::MESSAGE,
    )?;
    let (lowering, mut claims) = if mode == "tiny" {
        let mut model: IntegerModel = envelope::open(
            &fs::read(root.join("../../../spec/vectors/models/tiny-2-affine.d"))?,
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        )?;
        let is_mlp = |point: &QuantPoint| {
            matches!(
                point,
                QuantPoint::Layer {
                    site: Site::MlpResidual,
                    ..
                }
            )
        };
        model
            .quant
            .public
            .shift_class_sets
            .retain(|(point, _)| !is_mlp(point));
        model
            .quant
            .public
            .affine_sites
            .retain(|(point, _)| !is_mlp(point));
        for layer in 0..model.arch.n_layers {
            model.quant.public.affine_sites.push((
                QuantPoint::Layer {
                    layer,
                    site: Site::MlpResidual,
                },
                if layer == 0 { 2 } else { 12 },
            ));
        }
        model
            .quant
            .public
            .affine_sites
            .sort_by_key(|(point, _)| *point);
        for channel in &mut model.quant.channels {
            if channel.ty == WeightType::Down {
                let channels = channel.class.len().max(channel.multiplier.len());
                channel.class.clear();
                channel.multiplier = vec![1; channels];
            }
        }
        let lowering = lower_batch(PublicModel::of(&model), &[2])?;
        let ranges = analyze(&lowering.graph, &lowering.tables)?;
        let selection = select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &GadgetPolicy::default(),
        )?;
        let claims = tessera_compiler::claim_list(
            PublicModel::of(&model),
            &selection,
            &GadgetPolicy::default(),
            source.layout,
            source.params,
        )?;
        (lowering, claims)
    } else if mode == "real" {
        (lower_batch(PublicModel::of_claims(&source), &[2])?, source)
    } else {
        return Err("mode tiny or real".into());
    };
    let ranges = analyze(&lowering.graph, &lowering.tables)?;
    let layers = LayerMap::new(&lowering)?;
    let text = fs::read_to_string(root.join("results/layout-plans.tsv"))?;
    let limits: &[u8] = if mode == "tiny" { &[32] } else { &[30, 31, 32] };
    for &limit in limits {
        let mut plans = BTreeMap::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 6 && parts[0] == mode && parts[1].parse::<u8>()? == limit {
                plans.insert(
                    parts[2].parse::<u32>()?,
                    (parts[3].parse()?, parts[4].parse()?, parts[5].parse()?),
                );
            }
        }
        claims.gadget_policy.max_log2_rows = limit;
        let mut selected = select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &claims.gadget_policy,
        )?;
        claims.tables = selected.tables.clone();
        let baseline = LeafLayout::model(&lowering, &layers, &selected, &claims);
        assert!(
            matches!(&baseline, Err(tessera_prover::leaf::LeafError::Internal { rule }) if *rule == RULE),
            "baseline must reproduce exact refusal"
        );
        let changed = patch(&lowering, &mut selected, &plans)?;
        assert_eq!(changed, plans.len());
        claims.tables = selected.tables.clone();
        // Scratch-only visibility overlay supplies the candidate's ordered lookup cache.
        // All canonical metadata checks and the actual leaf agreement guard remain unchanged.
        // This adapter never runs a witness, runtime prover, verifier or claim-list export.
        let leaf = LeafLayout::model(&lowering, &layers, &selected, &claims)?;
        println!(
            "mode={mode} B={limit} baseline=exact-refusal candidate=layout-ok changed={changed} split_values={}",
            leaf.split_len()
        );
    }
    println!(
        "E0 metadata adapter only; no witness, proof, verifier acceptance or canonical selection adoption"
    );
    Ok(())
}
