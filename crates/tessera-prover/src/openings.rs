// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original CPU opening targets (RFC-0037/0045/0052/0053), derived without verifier code.
use crate::leaf::{LeafClaims, LeafLayout, ModelColumn};
use crate::native::ModelNativePlan;
use crate::{StageError, StageProver, WitnessSource};
use core::fmt;
use tessera_compiler::schedule::{Layers, NodeRef};
use tessera_compiler::{AdviceSource, Lowering, ModelAdviceLayout};
use tessera_field::{Field, GoldilocksExt2 as F};
use tessera_ir::CommitmentClass;
use tessera_pcs::{AdviceClaim, AdvicePoint, WeightLayout, WeightMatrix};

pub(crate) fn bad() -> StageError {
    StageError::Internal {
        rule: "INV-SCH-3: original opening sources, points and order follow the runtime schedule",
    }
}
pub(crate) struct OriginalAdvice {
    pub point: Vec<F>,
    pub pieces: Vec<(AdviceSource, F)>,
}
/// Every original advice combination and weight point, in canonical claim order.
/// This producer plan computes opening rows; only the independent verifier judges them.
pub struct ModelOpenings {
    advice: Vec<AdviceClaim>,
    weights: [Vec<Vec<F>>; 10],
}
impl fmt::Debug for ModelOpenings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelOpenings")
            .field("advice", &self.advice.len())
            .field("weights", &self.weights.each_ref().map(Vec::len))
            .finish()
    }
}
impl ModelOpenings {
    /// Derive targets from the completed CPU stage/leaf metadata and original source identities.
    /// Repeated native and same-point weight claims are retained, never deduplicated.
    ///
    /// # Errors
    /// `Internal` for an incomplete/inconsistent schedule or source; failed layout mapping
    /// also returns `Internal`. Public metadata allocations follow the producer's normal policy.
    #[allow(clippy::too_many_arguments)]
    pub fn derive<W: WitnessSource + ?Sized>(
        lowering: &Lowering,
        leaf_layout: &LeafLayout<ModelColumn>,
        leaf: &dyn LeafClaims,
        native: &ModelNativePlan,
        stages: &StageProver<'_, W>,
        advice: &ModelAdviceLayout<'_>,
        weights: &WeightLayout,
    ) -> Result<Self, StageError> {
        if stages.pool().len() != native.schedule().pool.len()
            || stages.pool().iter().any(Option::is_none)
        {
            return Err(bad());
        }
        let mut out = Self {
            advice: Vec::new(),
            weights: core::array::from_fn(|_| Vec::new()),
        };
        let schedule = native.schedule();
        for claim in &schedule.pool {
            let point = stages.point(&claim.point)?;
            if claim.commitment == CommitmentClass::Advice {
                let mut pieces = Vec::new();
                for atom in &claim.atoms {
                    if stages.point(&atom.point)? != point {
                        return Err(bad());
                    }
                    for (layer, weight) in stages.weights(&atom.layers, None)? {
                        let tensor = schedule
                            .layers
                            .tensor(&lowering.graph, atom.kind, layer)
                            .ok_or_else(bad)?;
                        pieces.push((
                            AdviceSource::Tensor(tensor),
                            weight * stages.coefficient(&atom.coefficient, layer)?,
                        ));
                    }
                }
                out.push(advice, OriginalAdvice { point, pieces })?;
            } else {
                let [atom] = claim.atoms.as_slice() else {
                    return Err(bad());
                };
                if !atom.is_unit() || stages.point(&atom.point)? != point {
                    return Err(bad());
                }
                let tensor = schedule
                    .layers
                    .tensor(&lowering.graph, atom.kind, 0)
                    .ok_or_else(bad)?;
                let layer = match &atom.layers {
                    Layers::Eq(p) => stages.point(p)?,
                    Layers::None => Vec::new(),
                    _ => return Err(bad()),
                };
                if let Some((weight, _)) = lowering.graph.weights.iter().find(|(_, t)| *t == tensor)
                {
                    let matrix = weight.ty.matrix();
                    let mut full = point;
                    full.extend(layer);
                    if full.len()
                        != (weights.matrix(matrix).log2_k() + weights.matrix(matrix).row_vars())
                            as usize
                    {
                        return Err(bad());
                    }
                    out.weights[matrix.index()].push(full);
                } else {
                    let (weight, _) = lowering
                        .classes
                        .iter()
                        .chain(&lowering.multipliers)
                        .find(|(_, t)| *t == tensor)
                        .ok_or_else(bad)?;
                    let full = weights
                        .channel_point(weight.ty.matrix(), &point, &layer)
                        .map_err(|_| bad())?;
                    out.weights[WeightMatrix::ChannelParams.index()].push(full);
                }
            }
        }
        // Fixed local gadgets in graph-node order, then native families in registration order.
        let mut fixed = Vec::new();
        for (site_index, site) in leaf_layout.sites().iter().enumerate() {
            for layer in 0..site.layers.len() {
                let node = match site.node {
                    NodeRef::Node(n) => n,
                    NodeRef::Position(p) => schedule.layers.node(p, layer as u32),
                };
                let is_native = matches!(
                    lowering.graph.nodes[node.0 as usize].op,
                    tessera_ir::Op::RsqrtRow { .. } | tessera_ir::Op::SoftmaxRow { .. }
                );
                let order = if is_native {
                    (1, site_index, layer)
                } else {
                    (0, node.0 as usize, 0)
                };
                fixed.push((order, node, site_index, layer));
            }
        }
        fixed.sort_by_key(|&(order, _, _, _)| order);
        for (_, node, s, l) in fixed {
            let mut reads = Vec::new();
            for (r, read) in leaf_layout.reads(s, l).ok_or_else(bad)?.iter().enumerate() {
                let source = match read.term {
                    ModelColumn::Output(i) => AdviceSource::Tensor(
                        *lowering.graph.nodes[node.0 as usize]
                            .outputs
                            .get(usize::from(i))
                            .ok_or_else(bad)?,
                    ),
                    ModelColumn::Auxiliary(a) => AdviceSource::Auxiliary(node, a),
                    _ => continue,
                };
                // Where the leaf stage left the tensor's claim (RFC-0054 'The stage's outputs').
                let point = leaf.read_point(leaf_layout, s, l, r).ok_or_else(bad)?;
                reads.push((source, point));
            }
            reads.sort_by_key(|(source, _)| *source);
            let definition = &lowering.graph.nodes[node.0 as usize];
            if matches!(definition.op, tessera_ir::Op::SoftmaxRow { .. }) {
                let exponential = AdviceSource::Tensor(definition.outputs[0]);
                let reciprocal = AdviceSource::Tensor(definition.outputs[1]);
                let row = reads
                    .iter()
                    .find(|(s, _)| *s == reciprocal)
                    .ok_or_else(bad)?
                    .1
                    .clone();
                out.push(
                    advice,
                    OriginalAdvice {
                        point: row.clone(),
                        pieces: vec![(reciprocal, F::ONE)],
                    },
                )?;
                let n = lowering.graph.tensors[definition.outputs[0].0 as usize]
                    .shape
                    .last()
                    .copied()
                    .ok_or_else(bad)?;
                let mut half = vec![
                    (F::ONE + F::ONE).inverse().ok_or_else(bad)?;
                    n.next_power_of_two().trailing_zeros() as usize
                ];
                half.extend(row);
                out.push(
                    advice,
                    OriginalAdvice {
                        point: half,
                        pieces: vec![(exponential, F::ONE)],
                    },
                )?;
                for (source, point) in reads
                    .iter()
                    .filter(|(s, _)| matches!(s, AdviceSource::Auxiliary(..)))
                {
                    out.push(
                        advice,
                        OriginalAdvice {
                            point: point.clone(),
                            pieces: vec![(*source, F::ONE)],
                        },
                    )?;
                }
                let point = reads
                    .into_iter()
                    .find(|(s, _)| *s == exponential)
                    .ok_or_else(bad)?
                    .1;
                out.push(
                    advice,
                    OriginalAdvice {
                        point,
                        pieces: vec![(exponential, F::ONE)],
                    },
                )?;
            } else {
                for (source, point) in reads {
                    out.push(
                        advice,
                        OriginalAdvice {
                            point,
                            pieces: vec![(source, F::ONE)],
                        },
                    )?;
                }
            }
        }
        for original in native.source_openings(lowering, stages)? {
            out.push(advice, original)?;
        }
        if leaf_layout.instances().len() != leaf.instances() {
            return Err(bad());
        }
        for (k, instance) in leaf_layout.instances().iter().enumerate() {
            let multiplicity = leaf.multiplicity_claim(k).ok_or_else(bad)?;
            out.push(
                advice,
                OriginalAdvice {
                    point: multiplicity.point.clone(),
                    pieces: vec![(AdviceSource::Multiplicity(instance.table), F::ONE)],
                },
            )?;
        }
        Ok(out)
    }
    fn push(
        &mut self,
        layout: &ModelAdviceLayout<'_>,
        original: OriginalAdvice,
    ) -> Result<(), StageError> {
        let mut pieces = Vec::new();
        for (source, coefficient) in original.pieces {
            for image in layout.claim(source, &original.point).map_err(|_| bad())? {
                let image = image.map_err(|_| bad())?;
                pieces.push(AdvicePoint {
                    point: image.point,
                    weight: coefficient * image.weight,
                });
            }
        }
        if pieces.is_empty() {
            return Err(bad());
        }
        self.advice.push(AdviceClaim { pieces });
        Ok(())
    }
    /// Original advice combinations, retaining each claim's own gamma power.
    pub fn advice(&self) -> &[AdviceClaim] {
        &self.advice
    }
    /// Original full matrix points in pool order; `ChannelParams` keeps every reading site.
    pub fn weight_points(&self, matrix: WeightMatrix) -> &[Vec<F>] {
        &self.weights[matrix.index()]
    }
}
