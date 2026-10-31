// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original native source points, separately from their transformed graph projections.
use super::*;
use crate::openings::{OpeningStage, OriginalAdvice, PointConstant};
use tessera_compiler::AdviceSource;
use tessera_compiler::schedule::{Coord, Relation as ScheduledRelation};

impl ModelNativePlan {
    pub(crate) fn source_openings<C: Copy + Eq, S: OpeningStage<C> + ?Sized>(
        &self,
        lowering: &Lowering,
        stages: &S,
    ) -> Result<Vec<OriginalAdvice<C>>, StageError> {
        let mut out = Vec::new();
        for stage in self.schedule.stages.iter().skip(1) {
            let rounds = (0..stage.rounds)
                .map(|round| {
                    stages
                        .challenge(Coord::Round {
                            stage: stage.index,
                            round,
                        })
                        .ok_or_else(bad)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for instance in &stage.instances {
                let ScheduledRelation::Native { tag, .. } = instance.relation else {
                    continue;
                };
                let i = tag.checked_sub(1).ok_or_else(bad)? as usize;
                let slot = self.slots.get(i).ok_or_else(bad)?;
                let family = self.families.get(slot.family).ok_or_else(bad)?;
                let point = rounds
                    .get(
                        rounds
                            .len()
                            .checked_sub(instance.vars as usize)
                            .ok_or_else(bad)?..,
                    )
                    .ok_or_else(bad)?;
                let local = point
                    .len()
                    .checked_sub(family.lambda as usize)
                    .ok_or_else(bad)?;
                let (p, layer) = point.split_at(local);
                let mut push = |point: &[C], column: AuxColumn| {
                    let pieces = family
                        .nodes
                        .iter()
                        .enumerate()
                        .map(|(l, &node)| {
                            (
                                AdviceSource::Auxiliary(node, column),
                                stages.layer_weight(layer, l),
                            )
                        })
                        .collect();
                    out.push(OriginalAdvice {
                        point: point.to_vec(),
                        pieces,
                    });
                };
                match (slot.relation, &family.kind) {
                    (Relation::Product, Kind::Root { digits, .. }) => {
                        for j in 0..digits - 1 {
                            push(p, AuxColumn::RootByte(j as u32));
                        }
                        for j in 0..3 {
                            push(p, AuxColumn::InputLimb(j));
                        }
                    }
                    (Relation::Key | Relation::Forest, Kind::Softmax { .. }) => {
                        let keys = vars(&[family.shape[2]]) as usize;
                        push(p.get(keys..).ok_or_else(bad)?, AuxColumn::Maximum);
                    }
                    (Relation::MaxPad, _) => push(p, AuxColumn::Maximum),
                    (Relation::Reciprocal, Kind::Softmax { .. }) => {
                        let keys = vars(&[family.shape[2]]) as usize;
                        let mut point = vec![stages.constant(PointConstant::Half)?; keys];
                        point.extend_from_slice(p);
                        let pieces = family
                            .nodes
                            .iter()
                            .enumerate()
                            .map(|(l, &node)| {
                                (
                                    AdviceSource::Tensor(
                                        lowering.graph.nodes[node.0 as usize].outputs[0],
                                    ),
                                    stages.layer_weight(layer, l),
                                )
                            })
                            .collect();
                        out.push(OriginalAdvice { point, pieces });
                    }
                    (
                        Relation::Runtime,
                        Kind::Runtime(Runtime::Key(c, _) | Runtime::Selected(c)),
                    ) => {
                        let nv = vars(&[family.shape[1]]) as usize;
                        let time = p.get(nv..).ok_or_else(bad)?;
                        out.push(OriginalAdvice {
                            point: time.to_vec(),
                            pieces: vec![(AdviceSource::GreedyMaximum(c.gadget), F::ONE)],
                        });
                        let zero = stages.constant(PointConstant::Zero)?;
                        let one = stages.constant(PointConstant::One)?;
                        for carry in 0..=time.len() {
                            let mut original = p[..nv].to_vec();
                            original.extend((0..time.len()).map(|j| {
                                if j < carry {
                                    zero
                                } else if j == carry {
                                    one
                                } else {
                                    time[j]
                                }
                            }));
                            out.push(OriginalAdvice {
                                point: original,
                                pieces: vec![(AdviceSource::Tensor(lowering.one_hot), F::ONE)],
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(out)
    }
}
