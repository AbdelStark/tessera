// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual canonical CPU leaf + ALL native relations + ALL graph stages (DEP-2).
//! Original source/opening and token/logit debts remain; this is not a model verdict.
use super::*;
use tessera_codec::SecretBuf;
use tessera_prover::native::{ModelNativePlan, ModelNativeProver};
use tessera_prover::stages::{NativeContext, NativeProver, StageError};
use tessera_sumcheck::{Claim, Instance, Table};
use tv_core::gadgets::model;
use tv_core::stages;

pub fn compare_schedules(plan: &ModelNativePlan, independent: &model::Plan<'_>) {
    let a = plan.schedule();
    let b = independent.schedule();
    assert_eq!(a.stage_count() - 1, b.stages().len());
    assert_eq!(plan.native().len(), independent.native().len());
    for (x, y) in plan.native().iter().zip(independent.native()) {
        assert_eq!(
            (
                x.tag,
                x.node.0,
                x.vars as usize,
                x.degree,
                x.evaluations,
                x.zero
            ),
            (y.tag, y.node.0, y.vars, y.degree, y.evaluations, y.zero)
        );
        assert_eq!(x.outputs.len(), y.outputs.len());
        for (x, y) in x.outputs.iter().zip(&y.outputs) {
            assert_eq!(
                (x.tensor.0, x.from as usize, x.vars as usize),
                (y.tensor.0, y.from, y.vars)
            );
        }
    }
    for (a, b) in a.stages[1..].iter().zip(b.stages()) {
        assert_eq!(
            (a.rounds as usize, a.zero_vars as usize, a.instances.len()),
            (b.rounds, b.zero_vars, b.instances.len())
        );
    }
}

struct Broken<'a> {
    native: ModelNativeProver<'a>,
    tag: u32,
    zero: u32,
    count: usize,
    table: SecretBuf<G>,
    fault: u8,
}
impl NativeProver for Broken<'_> {
    fn prepare(&mut self, ctx: NativeContext<'_>) -> Result<(), StageError> {
        self.native.prepare(ctx)
    }
    fn claim(&self, tag: u32) -> Result<F, StageError> {
        if tag == self.zero && self.fault == 0 {
            Ok(F::ONE)
        } else {
            self.native.claim(tag)
        }
    }
    fn instance(&self, tag: u32) -> Result<Instance<'_, F>, StageError> {
        if tag == self.tag {
            match self.fault {
                1 => return Ok(Instance::new(Vec::new())),
                2 => {
                    return Ok(Instance::new(
                        (0..self.count)
                            .map(|_| Table::base(&self.table[..self.table.len() - 1]))
                            .collect(),
                    ));
                }
                3 => {
                    return Ok(Instance::new(
                        (0..self.count - 1)
                            .map(|_| Table::base(&self.table))
                            .collect(),
                    ));
                }
                _ => {}
            }
        }
        self.native.instance(tag)
    }
    fn claims(&self, tag: u32, point: &[F], values: &[F]) -> Result<Vec<Claim<F>>, StageError> {
        if tag == self.tag && self.fault == 4 {
            return Ok(Vec::new());
        }
        let mut claims = self.native.claims(tag, point, values)?;
        if tag == self.tag && self.fault == 5 {
            claims[0].point.clear();
        }
        Ok(claims)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prove_stages(
    p: &Public,
    graph: &tv_core::lowering::Graph,
    layout: &LeafLayout<ModelColumn>,
    bound: &split::Bound,
    witness: &tessera_witness::Witness,
    actual: &ModelWitness,
    native_plan: &ModelNativePlan,
    independent: &model::Plan<'_>,
    prover: &mut tessera_prover::StageProver<'_, tessera_witness::Witness>,
    points: &[Vec<F>],
    leaf: &[Vec<F>],
    proof: &tessera_prover::ProvenLeaf,
    transcript: &mut Transcript,
    mut verifier_transcript: tv_hash::transcript::Transcript,
    committed: &model_openings::Committed<'_>,
    xi: &[Vec<F>; 10],
    mode: Mode,
    geometry: Option<&tv_core::runtime::Geometry>,
    first: model_openings::First,
) {
    let leaf: Vec<_> = points
        .iter()
        .zip(leaf)
        .map(|(p, v)| stages::LeafValues {
            point: es(p),
            values: es(v),
        })
        .collect();
    let runtime = independent
        .schedule()
        .leaf()
        .iter()
        .all(|l| l.input.is_some());
    let token = independent.schedule().leaf().iter().position(|l| {
        matches!(
            graph.node(l.site).unwrap().op(),
            tv_core::lowering::Op::OneHot { .. }
        )
    });
    let prepared = if runtime {
        independent.bind_runtime(bound)
    } else {
        independent.bind(
            bound,
            model::External {
                tokens: &leaf[token.unwrap()],
                logits: leaf.last().unwrap(),
            },
        )
    }
    .unwrap();
    assert_eq!(prepared.leaf(), leaf);
    let mut native =
        ModelNativeProver::new(native_plan, &p.lowering, actual, witness, proof).unwrap();
    // Exercise transformations with distinct raw values even when a CPU fixture has equal operands.
    for meta in native_plan.native() {
        let point: Vec<_> = (0..meta.vars)
            .map(|j| F::from(G::from_u64_reduced(7 + u64::from(j))))
            .collect();
        let raw: Vec<_> = (0..meta.evaluations)
            .map(|j| F::from(G::from_u64_reduced(31 + j as u64)))
            .collect();
        let layer = vec![E::ONE; independent.schedule().layer_vars()];
        let zero = vec![E::ONE; independent.schedule().stages()[0].zero_vars];
        let checked = stages::NativeChecks::evaluate(
            &prepared,
            meta.tag,
            stages::NativeContext {
                stage: 1,
                layer: &layer,
                zero: &zero,
            },
            &es(&point),
            &es(&raw),
        )
        .unwrap();
        let claims = native.claims(meta.tag, &point, &raw).unwrap();
        assert_eq!(
            es(&claims.iter().map(|c| c.value).collect::<Vec<_>>()),
            checked.graph
        );
    }
    let proofs: Vec<_> = (1..native_plan.schedule().stages.len())
        .map(|k| {
            prover
                .prove_stage_composed(transcript, k as u16, &mut native)
                .unwrap()
        })
        .collect();
    if let Some(g) = geometry {
        assert_eq!(proofs.len(), g.stages().len());
        for (p, f) in proofs.iter().zip(g.stages()) {
            assert_eq!(
                f.sumcheck.instances,
                native_plan.schedule().stages[usize::from(p.stage)]
                    .instances
                    .len()
            );
            assert_eq!(p.stage, f.index);
            assert_eq!(p.sumcheck.rounds.len(), f.sumcheck.rounds);
            assert!(
                p.sumcheck
                    .rounds
                    .iter()
                    .all(|r| r.len() == f.sumcheck.degree)
            );
            assert_eq!(p.sumcheck.final_evals.len(), f.sumcheck.evaluations);
            assert_eq!(
                f.sumcheck.field_count(),
                Some(
                    p.sumcheck.rounds.iter().map(Vec::len).sum::<usize>()
                        + p.sumcheck.final_evals.len()
                )
            );
            assert_eq!(p.splits.len(), f.splits);
        }
    }
    let proofs: Vec<_> = proofs
        .iter()
        .map(|p| stages::StageProof {
            stage: p.stage,
            sumcheck: tv_core::sumcheck::SumcheckProof {
                rounds: p.sumcheck.rounds.iter().map(|r| es(r)).collect(),
                final_evals: es(&p.sumcheck.final_evals),
            },
            splits: es(&p.splits),
        })
        .collect();
    let rope = tv_tables::Table::generate(tv_tables::TableId(0), graph.tables()[0].kind.clone())
        .unwrap()
        .rope()
        .unwrap()
        .clone();
    let accepted = prepared
        .verify(&mut verifier_transcript, &rope, &proofs)
        .unwrap();
    assert_eq!(accepted.graph.pool.len(), prover.pool().len());
    for (claim, value) in accepted.graph.pool.iter().zip(prover.pool()) {
        assert_eq!(claim.value, e(value.unwrap()));
        assert_eq!(claim.value, pool_value(p, graph, witness, claim));
    }
    assert_eq!(accepted.multiplicities, bound.multiplicities);
    if runtime {
        assert!(accepted.external.is_none());
    } else {
        assert_eq!(accepted.external.unwrap().tokens, &leaf[token.unwrap()]);
        assert_eq!(accepted.external.unwrap().logits, leaf.last().unwrap());
    }
    let original = bound
        .columns
        .iter()
        .filter(|c| {
            matches!(
                c.column,
                split::Column::Output(_) | split::Column::Auxiliary(_)
            )
        })
        .count();
    let softmax = p
        .selection
        .gadgets
        .iter()
        .filter(|(id, _)| {
            matches!(
                p.lowering.graph.nodes[id.0 as usize].op,
                tessera_ir::Op::SoftmaxRow { .. }
            )
        })
        .count();
    let products = native_plan
        .native()
        .iter()
        .filter(|n| n.degree == 4)
        .map(|n| n.evaluations - 2)
        .sum::<usize>();
    // Every original read, extra half-key binding, and every repeated native operand is owed.
    let families = native_plan
        .native()
        .iter()
        .filter(|n| n.zero && n.outputs.is_empty())
        .count();
    if !runtime {
        assert_eq!(
            accepted.sources.len(),
            original + softmax + products + 4 * families
        );
    }
    for claim in &accepted.sources {
        let mut sum = E::ZERO;
        assert_eq!(
            claim.sources.len(),
            if claim.layer_point.is_empty() {
                1
            } else {
                p.layers.count as usize
            }
        );
        let mut family = None;
        for (l, source) in claim.sources.iter().enumerate() {
            use tv_core::advice::Source;
            let (value, identity) = match *source {
                Source::Tensor(t) => {
                    let id = tessera_ir::TensorId(t.0);
                    let shape = &p.lowering.graph.tensors[id.0 as usize].shape;
                    let values = witness.tensor(id).unwrap();
                    (
                        tensor_value(values, shape, &claim.point),
                        (p.lowering.graph.producer(id).map(|n| n.0), None),
                    )
                }
                Source::Auxiliary(node, role) => {
                    let (s, l) = layout
                        .sites()
                        .iter()
                        .enumerate()
                        .find_map(|(s, site)| {
                            (0..site.layers.len())
                                .find(|&l| owner(p, site.node, l).0 == node.0)
                                .map(|l| (s, l))
                        })
                        .unwrap();
                    let column = layout
                        .reads(s, l)
                        .unwrap()
                        .iter()
                        .map(|r| r.term)
                        .chain([ModelColumn::Auxiliary(AuxColumn::Maximum)])
                        .find(|&c| matches!(super::role(c),split::Column::Auxiliary(r) if r==role))
                        .unwrap();
                    (
                        mle(actual.values(s, l, column).unwrap(), &claim.point),
                        (Some(node.0), Some(role)),
                    )
                }
                Source::GreedyMaximum(g) => {
                    let (id, m) = actual.greedy_maximum().unwrap();
                    assert_eq!(g.0, id.0);
                    (mle(m, &claim.point), (None, None))
                }
                Source::Multiplicity(_) => {
                    panic!("multiplicities retain their original table points separately")
                }
            };
            // Original role and node position must be the same across the ordered family.
            if !claim.layer_point.is_empty() {
                let node = NodeId(identity.0.unwrap());
                let (layer, position) = p.layers.position(node).unwrap();
                assert_eq!(layer, l as u32);
                if let Some((expected, role)) = family {
                    assert_eq!((position, identity.1), (expected, role));
                } else {
                    family = Some((position, identity.1));
                }
            }
            let weight = if claim.layer_point.is_empty() {
                E::ONE
            } else {
                tv_core::sumcheck::eq(
                    &claim.layer_point,
                    &(0..claim.layer_point.len())
                        .map(|j| E::from(V::from_canonical((l >> j & 1) as u64).unwrap()))
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            sum += weight * value;
        }
        assert_eq!(claim.value, sum, "original source {:?}", claim.sources);
    }
    if p.lowering.graph.tensors[p.lowering.logits.0 as usize].shape[0] == 3 && p.layers.count == 2 {
        let meta = &native_plan.native()[0];
        let zero = native_plan.native().iter().find(|n| n.zero).unwrap().tag;
        for fault in 0..=6 {
            let mut broken = Broken {
                native: ModelNativeProver::new(native_plan, &p.lowering, actual, witness, proof)
                    .unwrap(),
                tag: meta.tag,
                zero,
                count: meta.evaluations,
                table: SecretBuf::filled(G::ZERO, 1 << meta.vars),
                fault,
            };
            let mut own =
                tessera_prover::StageProver::new(&p.lowering, native_plan.schedule(), witness);
            let _leaf = own.leaf(points).unwrap();
            let _atoms = own.leaf_split_values().unwrap();
            let mut t = super::transcript(Mode::NonInteractive);
            let error = if fault == 6 {
                own.prove_stage(&mut t, 1)
            } else {
                own.prove_stage_composed(&mut t, 1, &mut broken)
            }
            .unwrap_err();
            assert!(
                matches!(error, StageError::Internal { .. }),
                "fault {fault}: {error}"
            );
        }
    }
    let round2 = tv_core::protocol::Round2Response {
        logup: tv_core::protocol::LogUpProof {
            roots: es(proof.logup().roots()),
            values: es(proof.logup().values()),
        },
        trees: tv_core::protocol::TreeProof {
            levels: proof
                .logup()
                .levels()
                .iter()
                .map(|p| tv_core::sumcheck::SumcheckProof {
                    rounds: p.proof().rounds.iter().map(|r| es(r)).collect(),
                    final_evals: es(&p.proof().final_evals),
                })
                .collect(),
            groups: Vec::new(),
            table_roots: Vec::new(),
            merge: tv_core::sumcheck::SumcheckProof {
                rounds: Vec::new(),
                final_evals: Vec::new(),
            },
            splits: es(proof.splits()),
        },
        stages: proofs,
        advice_combined: Vec::new(),
        weight_combined: Vec::new(),
        scoring: tv_core::protocol::ScoringClaims {
            requests: vec![tv_core::protocol::FixedBits(0)],
        },
    };
    let cpu_openings = committed.openings(layout, proof, native_plan, prover, witness);
    committed.check(
        graph,
        accepted,
        transcript,
        verifier_transcript,
        xi,
        mode,
        first,
        round2,
        cpu_openings,
    );
}
fn tensor_value(values: &[i64], shape: &[u32], point: &[E]) -> E {
    let bits: Vec<_> = shape
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros())
        .collect();
    let mut padded = SecretBuf::filled(0i64, 1usize << bits.iter().sum::<u32>());
    for (flat, &v) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0, 0);
        for (axis, &n) in shape.iter().enumerate().rev() {
            index |= (rest % n as usize) << shift;
            rest /= n as usize;
            shift += bits[axis];
        }
        padded[index] = v;
    }
    mle(&padded, point)
}

fn pool_value(
    p: &Public,
    graph: &tv_core::lowering::Graph,
    witness: &tessera_witness::Witness,
    claim: &stages::PoolClaim,
) -> E {
    let mut sum = E::ZERO;
    for term in &claim.terms {
        for (l, &w) in term.weights.iter().enumerate() {
            let rep = match term.tensor {
                stages::PoolTensor::Single(t) | stages::PoolTensor::Stacked(t) => t,
            };
            let t = match term.tensor {
                stages::PoolTensor::Single(_) => tessera_ir::TensorId(rep.0),
                stages::PoolTensor::Stacked(_) => {
                    if let Some(id) = graph.tensor(rep).unwrap().producer {
                        let width = graph
                            .tensor(graph.checkpoints()[1])
                            .unwrap()
                            .producer
                            .unwrap()
                            .0
                            - graph
                                .tensor(graph.checkpoints()[0])
                                .unwrap()
                                .producer
                                .unwrap()
                                .0;
                        let output = graph
                            .node(id)
                            .unwrap()
                            .outputs()
                            .iter()
                            .position(|&t| t == rep)
                            .unwrap();
                        tessera_ir::TensorId(
                            graph
                                .node(tv_core::lowering::NodeId(id.0 + l as u32 * width))
                                .unwrap()
                                .outputs()[output]
                                .0,
                        )
                    } else {
                        let role = graph
                            .inputs()
                            .iter()
                            .find(|i| i.tensor == rep)
                            .unwrap()
                            .role;
                        let same = |r| match (role, r) {
                            (
                                tv_core::lowering::Role::Weight(a),
                                tv_core::lowering::Role::Weight(b),
                            )
                            | (
                                tv_core::lowering::Role::Classes(a),
                                tv_core::lowering::Role::Classes(b),
                            )
                            | (
                                tv_core::lowering::Role::Multipliers(a),
                                tv_core::lowering::Role::Multipliers(b),
                            ) => a.ty == b.ty && b.layer == l as u32,
                            _ => false,
                        };
                        tessera_ir::TensorId(
                            graph
                                .inputs()
                                .iter()
                                .find(|i| same(i.role))
                                .unwrap()
                                .tensor
                                .0,
                        )
                    }
                }
            };
            let shape = &p.lowering.graph.tensors[t.0 as usize].shape;
            let bits: Vec<_> = shape
                .iter()
                .map(|n| n.next_power_of_two().trailing_zeros())
                .collect();
            let mut table = SecretBuf::new();
            table.resize(1 << bits.iter().sum::<u32>(), 0i64);
            for (flat, &value) in witness.tensor(t).unwrap().iter().enumerate() {
                let (mut rest, mut index, mut shift) = (flat, 0, 0);
                for (axis, &len) in shape.iter().enumerate().rev() {
                    index |= (rest % len as usize) << shift;
                    rest /= len as usize;
                    shift += bits[axis];
                }
                table[index] = value;
            }
            sum += w * mle(&table, &claim.point);
        }
    }
    sum
}
