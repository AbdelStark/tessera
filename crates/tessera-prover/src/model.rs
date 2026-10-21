// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Full current-model transcript geometry, composed from public graph and gadget relations.

use std::collections::{BTreeMap, BTreeSet};

use tessera_compiler::schedule::{Claim, NodeRef, Part, Site, Source};
use tessera_compiler::{
    CompileError, ModelAdviceLayout, PublicModel, StageSchedule, analyze, derive_schedule, lower,
    select,
};
use tessera_ir::{ClaimList, Op, TableId};
use tessera_pcs::{WeightLayout, WeightMatrix as Matrix, WeightShapes};
use tessera_protocol::{
    LeafTree, MatrixCodeword, MatrixPoint, ScheduleSizes, StageSizes, WeightMatrix,
};

const GEOMETRY: &str = "the complete model transcript geometry fits its public domains";

fn internal() -> CompileError {
    CompileError::Schedule {
        node: None,
        rule: GEOMETRY,
    }
}

/// The protocol/1 schedule of one current-model sequence, derived from a claim list.
/// No witness or secret model data is read, and no proof obligations are discharged here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSchedule {
    /// Every fraction tree and Softmax forest, reduction stage, opening point and query domain.
    pub sizes: ScheduleSizes,
    /// All lookup instances in ascending `TableId`: the number of entries after per-axis
    /// padding of each real layer's blocks, before the combined column's row-0 tail (RFC-0045).
    /// This retains repeated requests and shared families; padded layers have no lookup block.
    pub lookup_entries: Vec<(TableId, u64)>,
}

/// Derive the entire current-model transcript's public sizes for `n` sequence positions.
///
/// All canonical lookup requests contribute their own domain (RFC-0050); the Softmax forest
/// stacks padded layers, heads and queries (RFC-0010/0030). Nonlinear gadget definitions and
/// stage-1 zero-checks extend the graph's reduction paths (RFC-0009/0035/0037). The graph schedule
/// supplies the existing symbolic dependencies; this adapter shifts those paths, rather than
/// duplicating its linear-substitution or tensor-coordinate implementation.
///
/// The result feeds `tessera_protocol::TranscriptSchedule::expand` in either protocol/1 mode.
/// Claim lists retain the unassigned prerelease version [`tessera_ir::PROTOCOL_VERSION`] (09 §3).
/// It is metadata, not a proof driver: that driver still proves the named relations (#130).
/// The caller remains responsible for validating the parameter set and checking the proof.
///
/// # Errors
/// [`CompileError::Usage`] for another protocol, sequence lengths refused by the lowering, or
/// a catalog different from the claim list's; other compiler errors retain their public rules.
/// [`CompileError::Schedule`] for geometry that exceeds its integer domains.
pub fn derive_model_schedule(claims: &ClaimList, n: u32) -> Result<ModelSchedule, CompileError> {
    if claims.protocol_version != tessera_ir::PROTOCOL_VERSION {
        return Err(CompileError::Usage {
            rule: "model transcript geometry requires the current claim-list protocol version",
        });
    }
    let lowering = lower(PublicModel::of_claims(claims), n)?;
    let ranges = analyze(&lowering.graph, &lowering.tables)?;
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &claims.gadget_policy,
    )?;
    if selection.tables != claims.tables {
        return Err(CompileError::Usage {
            rule: "the claim list contains the complete selected catalog",
        });
    }
    let advice = ModelAdviceLayout::new(&lowering, &selection, claims)?;
    let graph = derive_schedule(&lowering)?;
    let shape = |t: tessera_ir::TensorId| lowering.graph.tensors[t.0 as usize].shape.as_slice();
    let mut entries = BTreeMap::<TableId, u64>::new();
    for &(id, _) in &selection.gadgets {
        let node = &lowering.graph.nodes[id.0 as usize];
        for &table in selection.reads(id).ok_or_else(internal)? {
            let domain = match node.op {
                Op::SoftmaxRow { exp_table, .. } if table != exp_table => node.outputs[1],
                _ => node.inputs[0],
            };
            let count = 1u64
                .checked_shl(vars(shape(domain))?)
                .ok_or_else(internal)?;
            let total = entries.entry(table).or_default();
            *total = total.checked_add(count).ok_or_else(internal)?;
        }
    }
    let mut trees = Vec::new();
    for (&table, &count) in &entries {
        let height = count
            .checked_next_power_of_two()
            .ok_or_else(internal)?
            .trailing_zeros();
        let spec = selection
            .tables
            .iter()
            .find(|t| t.id == table)
            .ok_or_else(internal)?;
        trees.extend([
            LeafTree::Fraction { height },
            LeafTree::Fraction {
                height: u32::from(spec.log2_rows),
            },
        ]);
    }
    let mut geometry = Geometry {
        graph: &graph,
        stages: Vec::new(),
        seen: BTreeSet::new(),
    };
    // Keep each graph leaf dependency at its actual distance. Masked Softmax keys first need
    // their Elementwise relation, so their score claims reach the graph one stage later.
    for (site, claim) in graph.sites.iter().zip(&graph.stages[0].claims) {
        let shift = match site {
            Site::Gadget { node, .. } => {
                let id = match *node {
                    NodeRef::Node(id) => id,
                    NodeRef::Position(p) => graph.layers.node(p, 0),
                };
                u16::from(matches!(
                    lowering.graph.nodes[id.0 as usize].op,
                    Op::SoftmaxRow { .. }
                ))
            }
            Site::Logits => 0,
        };
        geometry.claim(0, claim, shift)?;
    }
    for (index, instance) in graph
        .stages
        .get(1)
        .ok_or_else(internal)?
        .instances
        .iter()
        .enumerate()
    {
        if instance.source == Source::ZeroCheck {
            geometry.instance(1, index, 0)?;
        }
    }
    // Canonical gadgets have a public domain per relation. Visiting the template once stacks
    // its layers; prologue and epilogue gadgets have no layer axis.
    for &(id, _) in &selection.gadgets {
        let layer = graph.layers.position(id);
        if layer.is_some_and(|(l, _)| l != 0) {
            continue;
        }
        let lambda = if layer.is_some() {
            graph.layers.vars
        } else {
            0
        };
        let node = &lowering.graph.nodes[id.0 as usize];
        match node.op {
            // Its Booleanity instance is already in the native graph zero-checks above.
            Op::OneHot { .. } => {}
            Op::RsqrtRow { .. } => {
                geometry.record(1, vars(shape(node.outputs[0]))? + lambda, true)?;
                // RFC-0050: the simplified u binding also proves every residual padding axis.
                let sum = lowering
                    .graph
                    .producer(node.inputs[0])
                    .ok_or_else(internal)?;
                let residual = lowering.graph.nodes[sum.0 as usize].inputs[0];
                geometry.record(1, vars(shape(residual))? + lambda, true)?;
                let reference = layer.map_or(NodeRef::Node(id), |(_, p)| NodeRef::Position(p));
                let site = graph
                    .sites
                    .iter()
                    .position(|s| {
                        *s == Site::Gadget {
                            node: reference,
                            input: 0,
                        }
                    })
                    .ok_or_else(internal)?;
                // Comparison equations read u's virtual top limb. Since u is the graph's
                // zero-padded SumSquares plus ψ·epsilon, that dependency follows in stage 2.
                geometry.claim(0, &graph.stages[0].claims[site], 1)?;
            }
            Op::SoftmaxRow { .. } => {
                let score = shape(node.inputs[0]);
                let [heads, queries, keys] = score else {
                    return Err(internal());
                };
                geometry.record(1, vars(score)? + lambda, false)?;
                // Reciprocal/slack relations and the maximum/reciprocal padding zero-checks.
                geometry.record(1, vars(shape(node.outputs[1]))? + lambda, true)?;
                // A single leaf has no reduction level; its relation remains in stage 1.
                if *keys > 1 {
                    trees.push(LeafTree::Forest {
                        trees_log: lambda + vars(&[*heads, *queries])?,
                        height: vars(&[*keys])?,
                    });
                }
            }
            Op::Requant { .. } | Op::TableMap { .. } | Op::Bitwise { .. } => {}
            _ => return Err(internal()),
        }
    }
    let channels = lowering
        .classes
        .iter()
        .chain(&lowering.multipliers)
        .map(|(w, _)| (w.ty.matrix(), w.layer))
        .collect();
    let weights = WeightLayout::new(
        claims.layout.weights,
        WeightShapes {
            layers: claims.arch.n_layers,
            shapes: claims.arch.weight_shapes().map_err(CompileError::Graph)?,
            channels,
        },
    )
    .map_err(|_| internal())?;
    let proximity = Matrix::ALL
        .into_iter()
        .filter_map(|matrix| {
            let rows = weights.matrix(matrix).rows();
            (rows > 0).then(|| MatrixPoint {
                matrix: WeightMatrix::ALL[matrix.index()],
                vars: u64::BITS - (rows - 1).leading_zeros(),
            })
        })
        .collect();
    let mut opened = BTreeSet::new();
    for claim in graph
        .pool
        .iter()
        .filter(|c| c.commitment == tessera_ir::CommitmentClass::Weight)
    {
        for atom in &claim.atoms {
            opened.insert(
                graph
                    .weight_type(&lowering.graph, atom.kind)
                    .map_or(Matrix::ChannelParams, |w| w.matrix())
                    .index(),
            );
        }
    }
    let opened = opened
        .into_iter()
        .map(|index| {
            let matrix = weights.matrix(Matrix::ALL[index]);
            let len = 1u64
                .checked_shl(matrix.log2_k() + u32::from(claims.layout.weights.log2_inv_rate))
                .ok_or_else(internal)?;
            Ok(MatrixCodeword {
                matrix: WeightMatrix::ALL[index].id(),
                len,
            })
        })
        .collect::<Result<_, CompileError>>()?;
    Ok(ModelSchedule {
        sizes: ScheduleSizes {
            trees,
            layer_vars: graph.layers.vars,
            stages: geometry.stages,
            advice_vars: advice.layout().row_vars(),
            proximity,
            opened,
        },
        lookup_entries: entries.into_iter().collect(),
    })
}

fn vars(shape: &[u32]) -> Result<u32, CompileError> {
    shape.iter().try_fold(0u32, |sum, &axis| {
        let last = axis.checked_sub(1).ok_or_else(internal)?;
        sum.checked_add(u32::BITS - last.leading_zeros())
            .ok_or_else(internal)
    })
}

/// Follow the compiler's already derived graph edges, with a gadget's extra stage offset.
struct Geometry<'a> {
    graph: &'a StageSchedule,
    stages: Vec<StageSizes>,
    seen: BTreeSet<(usize, usize, u16)>,
}

impl Geometry<'_> {
    fn record(&mut self, index: usize, variables: u32, zero: bool) -> Result<(), CompileError> {
        if index == 0 || index > u16::MAX as usize || variables > 63 {
            return Err(internal());
        }
        self.stages.resize(
            self.stages.len().max(index),
            StageSizes {
                rounds: 0,
                zero_vars: 0,
            },
        );
        let stage = &mut self.stages[index - 1];
        stage.rounds = stage.rounds.max(variables);
        if zero {
            stage.zero_vars = stage.zero_vars.max(variables);
        }
        Ok(())
    }

    fn claim(&mut self, stage: usize, claim: &Claim, shift: u16) -> Result<(), CompileError> {
        for part in &claim.parts {
            if let Part::Instance { instance, .. } = *part {
                self.instance(stage + 1, instance, shift)?;
            }
        }
        Ok(())
    }

    fn instance(&mut self, stage: usize, index: usize, shift: u16) -> Result<(), CompileError> {
        if !self.seen.insert((stage, index, shift)) {
            return Ok(());
        }
        let graph = self.graph;
        let source = graph.stages.get(stage).ok_or_else(internal)?;
        let instance = source.instances.get(index).ok_or_else(internal)?;
        self.record(
            stage + usize::from(shift),
            instance.vars,
            instance.source == Source::ZeroCheck,
        )?;
        for claim in &source.claims[instance.claims.clone()] {
            self.claim(stage, claim, shift)?;
        }
        Ok(())
    }
}
