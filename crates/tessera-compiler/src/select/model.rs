// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0076 canonical model recipes, materialized once after all sites resolve.
use super::*;
use crate::{Lowering, schedule::LayerMap};

#[derive(Clone, Copy, Debug)]
pub(super) enum RequantPlan {
    Split { bits: u32, split: u32 },
    Wide { bits: u32, split: u32, middle: u32 },
}

struct KeyInput<'a> {
    node: &'a Node,
    bits: u32,
    shifts: &'a [u8],
    lo: i32,
    hi: i32,
}

impl KeyInput<'_> {
    fn refuse(&self) -> CompileError {
        CompileError::Range {
            node: self.node.id,
            input: 0,
            rule: "RFC-0076: a common Requant family meets every integer, rounding and table bound",
        }
    }
    fn split(&self, max: u32) -> Option<RequantPlan> {
        let bits = self.bits.max(2);
        let classes = u32::try_from(self.shifts.len()).ok()?;
        let c = u32::BITS - classes.saturating_sub(1).leading_zeros();
        let a = bits.checked_add(c)?.saturating_sub(max).max(1);
        let d = u32::from(*self.shifts.iter().min()?)
            .checked_sub(1)?
            .min(max)
            .min(bits.checked_sub(1)?);
        let packed = u128::from(classes).checked_shl(bits)?;
        if bits > 62 || packed >= PRIME || a > d {
            return None;
        }
        let split = bits.checked_add(c)?.div_ceil(2).clamp(a, d);
        Some(RequantPlan::Split { bits, split })
    }
    fn wide(&self, max: u32, zero_low: bool) -> Option<RequantPlan> {
        let low = u32::from(*self.shifts.iter().min()?).checked_sub(1)?;
        let split = if zero_low { 0 } else { low.min(max) };
        let most = u32::from(*self.shifts.iter().max()?);
        let magnitude = u64::from(self.lo.unsigned_abs().max(self.hi.unsigned_abs()).max(1));
        let middle = most
            .checked_sub(split)?
            .checked_add(magnitude.checked_next_power_of_two()?.trailing_zeros())?;
        let bits = self.bits.max(split.checked_add(middle)?.checked_add(1)?);
        let top = bits.checked_sub(split)?.checked_sub(middle)?;
        let classes = u32::try_from(self.shifts.len()).ok()?;
        let c = u32::BITS - classes.saturating_sub(1).leading_zeros();
        if bits > 62
            || middle == 0
            || top == 0
            || split > low
            || split > max
            || middle.checked_add(2)?.checked_add(c)? > max
            || top.checked_add(1)? > max
        {
            return None;
        }
        Some(RequantPlan::Wide {
            bits,
            split,
            middle,
        })
    }
    fn kinds(&self, plan: RequantPlan) -> Vec<TableKind> {
        let rq = |b, drop, lo, hi| TableKind::Requant {
            b: narrow(b),
            r_by_class: self.shifts.iter().map(|r| r - narrow(drop)).collect(),
            lo,
            hi,
        };
        match plan {
            RequantPlan::Split { bits, split } => vec![
                TableKind::Range {
                    bits: narrow(split),
                },
                rq(bits - split, split, self.lo, self.hi),
            ],
            RequantPlan::Wide {
                bits,
                split,
                middle,
            } => {
                let top = bits - split - middle;
                let mut kinds = Vec::with_capacity(5);
                if split > 0 {
                    kinds.push(TableKind::Range {
                        bits: narrow(split),
                    });
                }
                kinds.push(TableKind::Range {
                    bits: narrow(middle),
                });
                kinds.push(TableKind::Range { bits: narrow(top) });
                kinds.push(TableKind::Requant {
                    b: narrow(top + 1),
                    r_by_class: vec![1],
                    lo: -2,
                    hi: 1,
                });
                kinds.push(rq(middle + 2, split, self.lo, self.hi));
                kinds
            }
        }
    }
    fn fits(&self, plan: RequantPlan, max: u32) -> bool {
        // Only Range/Requant candidates: their materialized core is empty, so a core
        // allocation failure cannot turn this public family decision into a fallback.
        self.kinds(plan).into_iter().all(|kind| {
            GeneratedTable::new(TableId(0), kind)
                .is_ok_and(|table| u32::from(table.spec().log2_rows) <= max)
        })
    }
}

pub(super) fn preparation_scratch_bytes(
    lowering: &Lowering,
    selected: &Selection,
    layers: &LayerMap,
) -> Option<u64> {
    use crate::memory::{array, btree, grow, sum};
    let nn = lowering.graph.nodes.len() as u64;
    let nt = lowering.graph.tensors.len() as u64;
    let n = u64::from(layers.count);
    let classes = lowering
        .graph
        .shift_classes
        .iter()
        .map(|v| v.len() as u64)
        .max()
        .unwrap_or(0);
    let reads = lowering
        .graph
        .nodes
        .iter()
        .map(|n| selected.reads(n.id).map_or(0, |v| v.len()) as u64)
        .max()
        .unwrap_or(0);
    sum([
        // RFC-0076's old-family and final-plan maps coexist with materialization.
        btree::<NodeId, Gadget>(nn)?,
        btree::<NodeId, RequantPlan>(nn)?,
        layers.construction_peak_bytes(&lowering.graph)?,
        array::<KeyInput<'_>>(n)?,
        2u64.checked_mul(grow::<RequantPlan>(n)?)?,
        grow::<(NodeId, RequantPlan)>(n)?,
        grow::<TableKind>(5)?,
        // All candidates are Range/Requant: no dense core. A candidate class
        // vector, the caller shifts and one cloned spec can coexist.
        array::<u8>(3u64.checked_mul(classes.checked_add(1)?)?)?,
        array::<Option<(NodeId, u32)>>(nt)?,
        grow::<TableId>(reads.checked_add(5)?)?,
        grow::<(NodeId, Gadget)>(nn)?,
    ])
}

fn signature(gadget: Gadget) -> Option<u8> {
    match gadget {
        Gadget::Requant1 { .. } => Some(0),
        Gadget::RequantSplit { .. } => Some(1),
        Gadget::RequantWide { low, .. } => Some(if low.is_some() { 3 } else { 2 }),
        _ => None,
    }
}

/// Select canonical model gadgets for the claim list's validated protocol version.
///
/// RFC-0076 resolves all real-layer sites before assigning auxiliary table ids.
/// Protocol Two and graphs without checkpoints retain the per-node [`select`] rule.
///
/// # Errors
/// The errors of [`select`] and [`LayerMap::new`], `Usage` for an unknown protocol,
/// or `Range` at the first infeasible Wide layer when a site cannot be promoted.
pub fn select_model(
    protocol: u16,
    lowering: &Lowering,
    ranges: &Ranges,
    policy: &GadgetPolicy,
) -> Result<Selection, CompileError> {
    if ![tessera_ir::PROTOCOL_VERSION, tessera_ir::PROTOCOL_VERSION_2].contains(&protocol) {
        return Err(CompileError::Usage {
            rule: "selection uses a supported claim-list protocol",
        });
    }
    if protocol != tessera_ir::PROTOCOL_VERSION || lowering.checkpoints.is_empty() {
        return select(&lowering.graph, ranges, &lowering.tables, policy);
    }
    let max = u32::from(policy.max_log2_rows);
    if !(1..=32).contains(&max) {
        return Err(CompileError::Usage {
            rule: "B_max lies in [1, 32], the most a Range table holds",
        });
    }
    let empty = BTreeMap::new();
    let mut validator = Selector {
        graph: &lowering.graph,
        ranges,
        max,
        limb_bits: policy.limb_bits,
        tables: lowering.tables.clone(),
        reads: Vec::new(),
        producers: None,
        materialize: false,
        plans: &empty,
    };
    let mut old = BTreeMap::new();
    for node in &lowering.graph.nodes {
        if let Some(gadget) = validator.node(node)? {
            old.insert(node.id, gadget);
        }
        validator.reads.clear();
    }
    let layers = LayerMap::new(lowering)?;
    let mut plans = BTreeMap::new();
    for position in 0..layers.width {
        let first = layers.node(position, 0);
        if !matches!(
            lowering.graph.nodes[first.0 as usize].op,
            Op::Requant { .. }
        ) {
            continue;
        }
        let mut inputs = Vec::with_capacity(layers.count as usize);
        for layer in 0..layers.count {
            let node = &lowering.graph.nodes[layers.node(position, layer).0 as usize];
            let Op::Requant { shift, lo, hi } = node.op else {
                return Err(CompileError::Usage {
                    rule: "a layer template preserves Requant nodes",
                });
            };
            let key =
                ranges
                    .of(node.inputs[0])
                    .and_then(Range::bounded)
                    .ok_or(CompileError::Wrap {
                        node: node.id,
                        input: 0,
                    })?;
            inputs.push(KeyInput {
                node,
                bits: signed_width(key.lo, key.hi),
                shifts: &lowering.graph.shift_classes[shift.0 as usize],
                lo: i32::try_from(lo).map_err(|_| CompileError::Range {
                    node: node.id,
                    input: 0,
                    rule: "model saturation fits i32",
                })?,
                hi: i32::try_from(hi).map_err(|_| CompileError::Range {
                    node: node.id,
                    input: 0,
                    rule: "model saturation fits i32",
                })?,
            });
        }
        let chosen = resolve_group(&inputs, &old, max)?;
        plans.extend(chosen);
    }
    select_planned(&lowering.graph, ranges, &lowering.tables, policy, &plans)
}

fn resolve_group(
    inputs: &[KeyInput<'_>],
    old: &BTreeMap<NodeId, Gadget>,
    max: u32,
) -> Result<Vec<(NodeId, RequantPlan)>, CompileError> {
    let first = inputs.first().ok_or(CompileError::Usage {
        rule: "a Requant group is nonempty",
    })?;
    let classed = first.shifts.len() > 1;
    for input in inputs {
        if (input.shifts.len() > 1) != classed || input.lo >= 0 || input.hi < 0 {
            return Err(input.refuse());
        }
    }
    let expected = old.get(&first.node.id).and_then(|g| signature(*g));
    if inputs
        .iter()
        .all(|i| old.get(&i.node.id).and_then(|g| signature(*g)) == expected)
    {
        return Ok(Vec::new());
    }
    let split: Option<Vec<_>> = inputs
        .iter()
        .map(|i| i.split(max).filter(|&p| i.fits(p, max)))
        .collect();
    let chosen = if let Some(plans) = split {
        plans
    } else {
        let zero_low = inputs.iter().any(|i| i.shifts.contains(&1));
        inputs
            .iter()
            .map(|i| {
                i.wide(max, zero_low)
                    .filter(|&p| i.fits(p, max))
                    .ok_or_else(|| i.refuse())
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(inputs
        .iter()
        .zip(chosen)
        .map(|(input, plan)| (input.node.id, plan))
        .collect())
}

impl Selector<'_> {
    pub(super) fn planned_requant(
        &mut self,
        node: &Node,
        shifts: Vec<u8>,
        lo: i64,
        hi: i64,
        plan: RequantPlan,
    ) -> Result<Gadget, CompileError> {
        let input = KeyInput {
            node,
            bits: 0,
            shifts: &shifts,
            lo: i32::try_from(lo).map_err(|_| CompileError::Usage {
                rule: "model saturation fits i32",
            })?,
            hi: i32::try_from(hi).map_err(|_| CompileError::Usage {
                rule: "model saturation fits i32",
            })?,
        };
        let ids = input
            .kinds(plan)
            .into_iter()
            .map(|kind| self.table(node, kind))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match plan {
            RequantPlan::Split { split, .. } => Gadget::RequantSplit {
                split: narrow(split),
                low: ids[0],
                high: ids[1],
            },
            RequantPlan::Wide { split, middle, .. } => {
                let offset = usize::from(split > 0);
                Gadget::RequantWide {
                    split: narrow(split),
                    middle: narrow(middle),
                    low: if split > 0 { Some(ids[0]) } else { None },
                    mid: ids[offset],
                    top: ids[offset + 1],
                    sign: ids[offset + 2],
                    high: ids[offset + 3],
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tessera_ir::{CommitmentClass, GraphBuilder, IntRange, IntType, NodeKind, TensorDecl};

    fn kind(row: &Value) -> TableKind {
        let r = row.as_array().unwrap();
        match r[0].as_str().unwrap() {
            "range" => TableKind::Range {
                bits: r[1].as_u64().unwrap() as u8,
            },
            "requant" => TableKind::Requant {
                b: r[1].as_u64().unwrap() as u8,
                r_by_class: r[2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect(),
                lo: r[3].as_i64().unwrap() as i32,
                hi: r[4].as_i64().unwrap() as i32,
            },
            _ => panic!("oracle table"),
        }
    }
    #[test]
    fn independent_group_oracle_and_complete_read_cache() {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../../spec/vectors/uniform-requant/cases.json"
        ))
        .unwrap();
        let mut count = 0;
        for case in vectors["cases"].as_array().unwrap() {
            let maximum = case["maximum"].as_u64().unwrap() as u8;
            let policy = GadgetPolicy {
                max_log2_rows: maximum,
                ..Default::default()
            };
            let mut builder = GraphBuilder::new();
            let decl = |lo, hi, kind| TensorDecl {
                shape: vec![2],
                ty: IntType::I32,
                range: IntRange { lo, hi },
                kind,
            };
            let advice = NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            };
            for n in case["nodes"].as_array().unwrap() {
                let b = n["bits"].as_u64().unwrap() as u32;
                let half = 1i128 << (b - 1);
                let x = builder.input(decl(-half, half - 1, advice)).unwrap();
                let shifts: Vec<_> = n["shifts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect();
                let mut inputs = vec![x];
                if shifts.len() > 1 {
                    inputs.push(
                        builder
                            .input(decl(
                                0,
                                shifts.len() as i128 - 1,
                                NodeKind::Committed {
                                    commitment: CommitmentClass::Weight,
                                },
                            ))
                            .unwrap(),
                    );
                }
                let shift = builder.shift_classes(shifts).unwrap();
                builder
                    .node(
                        Op::Requant {
                            shift,
                            lo: n["lo"].as_i64().unwrap(),
                            hi: n["hi"].as_i64().unwrap(),
                        },
                        &inputs,
                        vec![decl(0, 0, advice)],
                    )
                    .unwrap();
            }
            let graph = builder.finish();
            let ranges = crate::analyze_fragment(&graph, &[]).unwrap();
            let old = select(&graph, &ranges, &[], &policy);
            let expected = &case["expected"];
            let name = case["name"].as_str().unwrap();
            if expected["error"] == "legacy" {
                let error = old.unwrap_err();
                let node = match error {
                    CompileError::Range { node, .. } | CompileError::Table { node, .. } => node,
                    _ => panic!("wrong legacy error: {error:?}"),
                };
                assert_eq!(node.0, expected["layer"].as_u64().unwrap() as u32, "{name}");
                count += 1;
                continue;
            }
            let old = old.unwrap();
            let old_map = old.gadgets.iter().copied().collect();
            let inputs: Vec<_> = graph
                .nodes
                .iter()
                .map(|node| {
                    let Op::Requant { shift, lo, hi } = node.op else {
                        panic!("Requant only")
                    };
                    let range = ranges.of(node.inputs[0]).and_then(Range::bounded).unwrap();
                    KeyInput {
                        node,
                        bits: signed_width(range.lo, range.hi),
                        shifts: &graph.shift_classes[shift.0 as usize],
                        lo: lo as i32,
                        hi: hi as i32,
                    }
                })
                .collect();
            let plans = resolve_group(&inputs, &old_map, u32::from(maximum));
            if expected["error"].is_string() {
                let error = plans.unwrap_err();
                assert!(
                    matches!(error,CompileError::Range{node,..} if node.0==expected["layer"].as_u64().unwrap() as u32),
                    "{name}: {error:?}"
                );
            } else {
                let plans: BTreeMap<_, _> = plans.unwrap().into_iter().collect();
                let actual = select_planned(&graph, &ranges, &[], &policy, &plans).unwrap();
                for (node, plan) in graph
                    .nodes
                    .iter()
                    .zip(expected["plans"].as_array().unwrap())
                {
                    let keys: Vec<_> = actual
                        .reads(node.id)
                        .unwrap()
                        .iter()
                        .map(|id| actual.tables[id.0 as usize].kind.clone())
                        .collect();
                    let want: Vec<_> = plan["requests"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(kind)
                        .collect();
                    assert_eq!(keys, want, "{name}");
                }
                if expected["promoted"] == false {
                    assert_eq!(actual, old, "{name}");
                }
                // Re-intern the complete request stream to independently pin first use.
                let mut interned = Vec::new();
                for n in &graph.nodes {
                    for id in actual.reads(n.id).unwrap() {
                        let kind = actual.tables[id.0 as usize].kind.clone();
                        let at = if let Some(i) = interned.iter().position(|k| *k == kind) {
                            i
                        } else {
                            interned.push(kind);
                            interned.len() - 1
                        };
                        assert_eq!(id.0 as usize, at, "{name}");
                    }
                }
                assert_eq!(
                    interned.len(),
                    actual.tables.len(),
                    "{name}: no unused temporary tables"
                );
            }
            count += 1;
        }
        assert_eq!(count, 2191);
    }
}
