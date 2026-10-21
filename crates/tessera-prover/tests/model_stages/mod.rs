// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual canonical CPU leaf + mixed Softmax key/forest + ALL graph stages (DEP-2).
//! This composes components; remaining gadget/opening/greedy debts are not a model verdict.
use super::*;
use tessera_codec::SecretBuf;
use tessera_ir::Op;
use tessera_prover::stages::{NativeContext, NativeProver, StageError};
use tessera_sumcheck::{Claim, Instance, Table, Term};
use tv_core::gadgets::stacked::{Key, SoftmaxStack};
use tv_core::stages::{self, NativeChecks};

pub fn schedules(
    p: &Public,
    g: &tv_core::lowering::Graph,
) -> (tessera_compiler::StageSchedule, tv_core::schedule::Schedule) {
    let node = p
        .lowering
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::SoftmaxRow { .. }))
        .unwrap();
    let vars: u32 = p.lowering.graph.tensors[node.inputs[0].0 as usize]
        .shape
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros())
        .sum();
    let mut native: Vec<_> = [10, 11]
        .into_iter()
        .map(|tag| tessera_compiler::schedule::Native {
            tag,
            node: node.id,
            vars: vars + p.layers.vars,
            // Exercise the full mixed degree-four bound; this key polynomial is degree three.
            degree: if tag == 10 { 4 } else { 3 },
            evaluations: 2,
            zero: false,
            outputs: vec![tessera_compiler::schedule::NativeOutput {
                tensor: node.inputs[0],
                from: 0,
                vars,
            }],
        })
        .collect();
    let key_vars = p.lowering.graph.tensors[node.inputs[0].0 as usize].shape[2]
        .next_power_of_two()
        .trailing_zeros();
    native.push(tessera_compiler::schedule::Native {
        tag: 12,
        node: node.id,
        vars: vars - key_vars + p.layers.vars,
        degree: 3,
        evaluations: 1,
        zero: true,
        outputs: Vec::new(),
    });
    let plan = tessera_compiler::schedule::derive_composed_schedule(
        &p.lowering,
        &[(node.id, Some(0))],
        &native,
    )
    .unwrap();
    // Own verifier graph supplies the site, domain and source; no prover schedule translation.
    let (id, node) = g
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, n)| matches!(n.op(), tv_core::lowering::Op::SoftmaxRow { .. }))
        .unwrap();
    let id = tv_core::lowering::NodeId(id as u32);
    let vars: usize = g
        .tensor(node.inputs()[0])
        .unwrap()
        .shape
        .dims()
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros() as usize)
        .sum();
    let lambda = (g.checkpoints().len() - 1)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let mut native: Vec<_> = [10, 11]
        .into_iter()
        .map(|tag| tv_core::schedule::Native {
            tag,
            node: id,
            vars: vars + lambda,
            // Exercise the full mixed degree-four bound; this key polynomial is degree three.
            degree: if tag == 10 { 4 } else { 3 },
            evaluations: 2,
            zero: false,
            outputs: vec![tv_core::schedule::NativeOutput {
                tensor: node.inputs()[0],
                from: 0,
                vars,
            }],
        })
        .collect();
    let key_vars = g.tensor(node.inputs()[0]).unwrap().shape.dims()[2]
        .next_power_of_two()
        .trailing_zeros() as usize;
    native.push(tv_core::schedule::Native {
        tag: 12,
        node: id,
        vars: vars - key_vars + lambda,
        degree: 3,
        evaluations: 1,
        zero: true,
        outputs: Vec::new(),
    });
    let independent = tv_core::schedule::derive_composed(g, &[(id, Some(0))], &native).unwrap();
    assert_eq!(plan.stage_count() - 1, independent.stages().len());
    for (a, b) in plan.stages[1..].iter().zip(independent.stages()) {
        assert_eq!(a.rounds as usize, b.rounds);
        assert_eq!(a.zero_vars as usize, b.zero_vars);
        assert_eq!(a.instances.len(), b.instances.len());
    }
    (plan, independent)
}

struct Native<'a> {
    stack: SoftmaxStack<'a>,
    lookup: Vec<E>,
    fixed: Vec<(E, E)>,
    forest: Option<(Vec<E>, E)>,
    heads: usize,
    maximum: SecretBuf<G>,
    maximum_rows: SecretBuf<G>,
    padding: Vec<G>,
    scores: SecretBuf<G>,
    causal: Vec<G>,
    equalities: Vec<Vec<F>>,
    targets: Vec<F>,
}
impl Native<'_> {
    fn with_key<R>(&self, tag: u32, layer: &[E], f: impl FnOnce(&Key<'_, '_>) -> R) -> R {
        match tag {
            10 => f(&self.stack.lookup(&self.lookup, layer, &self.fixed).unwrap()),
            11 => {
                let scalar;
                let (point, value) = if let Some((p, v)) = &self.forest {
                    (p.as_slice(), *v)
                } else {
                    scalar = self.lookup.iter().chain(layer).copied().collect::<Vec<_>>();
                    let real = tv_core::selectors::lt(&self.lookup, self.heads as u64).unwrap()
                        * tv_core::selectors::lt(layer, self.stack.sites().len() as u64).unwrap();
                    (scalar.as_slice(), E::ONE - real)
                };
                f(&self.stack.forest(point, value).unwrap())
            }
            _ => panic!("own native tag"),
        }
    }
}
impl NativeProver for Native<'_> {
    fn prepare(&mut self, ctx: NativeContext<'_>) -> Result<(), StageError> {
        assert_eq!(ctx.stage, 1);
        let layer = es(ctx.layer);
        self.targets = [10, 11]
            .into_iter()
            .map(|tag| self.with_key(tag, &layer, |k| from_e(k.instance().unwrap().claim)))
            .collect();
        self.targets.push(F::ZERO);
        let mut lookup = self.lookup.clone();
        lookup.extend(&layer);
        let forest = self
            .forest
            .as_ref()
            .map(|(p, _)| p.clone())
            .unwrap_or_else(|| lookup.clone());
        self.equalities = [&lookup, &forest]
            .into_iter()
            .map(|point| {
                tessera_poly::eq_table(&point.iter().copied().map(from_e).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect();
        let vars = self.maximum_rows.len().trailing_zeros() as usize;
        self.equalities
            .push(tessera_poly::eq_table(&ctx.zero[ctx.zero.len() - vars..]).unwrap());
        Ok(())
    }
    fn claim(&self, tag: u32) -> Result<F, StageError> {
        Ok(self.targets[(tag - 10) as usize])
    }
    fn instance(&self, tag: u32) -> Result<Instance<'_, F>, StageError> {
        if tag == 12 {
            return Ok(Instance::new(vec![
                Table::base(&self.maximum_rows),
                Table::ext(&self.equalities[2]).public(),
                Table::base(&self.padding).public(),
            ]));
        }
        Ok(Instance::sum_of_products(
            vec![
                Table::base(&self.maximum),
                Table::base(&self.scores),
                Table::ext(&self.equalities[(tag - 10) as usize]).public(),
                Table::base(&self.causal).public(),
            ],
            vec![
                Term::new(F::ONE, vec![0, 2, 3]),
                Term::new(-F::ONE, vec![1, 2, 3]),
            ],
        ))
    }
    fn claims(&self, tag: u32, point: &[F], values: &[F]) -> Result<Vec<Claim<F>>, StageError> {
        if tag == 12 {
            return Ok(Vec::new());
        }
        Ok(vec![Claim {
            point: point.to_vec(),
            value: values[1],
        }])
    }
}
impl NativeChecks for Native<'_> {
    fn instance(
        &self,
        tag: u32,
        ctx: stages::NativeContext<'_>,
    ) -> Result<tv_core::sumcheck::Instance, stages::StageError> {
        if tag == 12 {
            return Ok(self.stack.padding_instance().unwrap());
        }
        Ok(self.with_key(tag, ctx.layer, |k| k.instance().unwrap()))
    }
    fn evaluate(
        &self,
        tag: u32,
        ctx: stages::NativeContext<'_>,
        point: &[E],
        values: &[E],
    ) -> Result<stages::NativeFinal, stages::StageError> {
        if tag == 12 {
            let result = self
                .stack
                .padding(
                    tv_core::gadgets::softmax::Padding::Maximum,
                    &ctx.zero[ctx.zero.len() - point.len()..],
                    point,
                    values,
                )
                .unwrap();
            return Ok(stages::NativeFinal {
                value: result.value,
                graph: Vec::new(),
            });
        }
        let result = self.with_key(tag, ctx.layer, |k| k.evaluate(point, values).unwrap());
        Ok(stages::NativeFinal {
            value: result.value,
            graph: vec![result.claims[1].value],
        })
    }
}
struct Broken<'a, 'p> {
    native: &'a Native<'p>,
    fault: u8,
}
impl NativeProver for Broken<'_, '_> {
    fn prepare(&mut self, _: NativeContext<'_>) -> Result<(), StageError> {
        Ok(())
    }
    fn claim(&self, tag: u32) -> Result<F, StageError> {
        if self.fault == 0 && tag == 12 {
            Ok(F::ONE)
        } else {
            NativeProver::claim(self.native, tag)
        }
    }
    fn instance(&self, tag: u32) -> Result<Instance<'_, F>, StageError> {
        if tag == 10 {
            match self.fault {
                1 => return Ok(Instance::new(Vec::new())),
                2 => {
                    return Ok(Instance::new(vec![
                        Table::base(&self.native.scores[..self.native.scores.len() - 1]),
                        Table::base(&self.native.maximum),
                    ]));
                }
                3 => return Ok(Instance::new(vec![Table::base(&self.native.scores)])),
                _ => {}
            }
        }
        NativeProver::instance(self.native, tag)
    }
    fn claims(&self, tag: u32, point: &[F], values: &[F]) -> Result<Vec<Claim<F>>, StageError> {
        if tag == 10 {
            if self.fault == 4 {
                return Ok(Vec::new());
            }
            if self.fault == 5 {
                return Ok(vec![Claim {
                    point: Vec::new(),
                    value: values[1],
                }]);
            }
        }
        NativeProver::claims(self.native, tag, point, values)
    }
}
fn from_e(x: E) -> F {
    F::new(
        G::from_u64_reduced(x.c0.to_canonical()),
        G::from_u64_reduced(x.c1.to_canonical()),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_stages(
    p: &Public,
    graph: &tv_core::lowering::Graph,
    layout: &LeafLayout<ModelColumn>,
    bound: &split::Bound,
    witness: &tessera_witness::Witness,
    source: &impl WitnessSource,
    advice: &ModelAdviceLayout<'_>,
    plan: &tessera_compiler::StageSchedule,
    independent: &tv_core::schedule::Schedule,
    prover: &mut tessera_prover::StageProver<'_, tessera_witness::Witness>,
    points: &[Vec<F>],
    leaf: &[Vec<F>],
    transcript: &mut Transcript,
    verifier_transcript: &mut tv_hash::transcript::Transcript,
) {
    let (_, site) = layout
        .sites()
        .iter()
        .enumerate()
        .find(|(_, s)| {
            matches!(
                p.lowering.graph.nodes[owner(p, s.node, 0).0 as usize].op,
                Op::SoftmaxRow { .. }
            )
        })
        .unwrap();
    let first = owner(p, site.node, 0);
    let gadgets = tv_core::gadgets::Plan::derive(
        graph,
        tv_core::claims::GadgetPolicy {
            max_log2_rows: p.claims.gadget_policy.max_log2_rows,
            limb_bits: p.claims.gadget_policy.limb_bits,
            query_block: p.claims.gadget_policy.query_block,
        },
    )
    .unwrap();
    let stack = gadgets
        .softmax_stack(tv_core::lowering::NodeId(first.0))
        .unwrap();
    assert_eq!(stack.sites().len(), site.layers.len());
    let read = |l, column| {
        bound
            .columns
            .iter()
            .find(|c| c.node.0 == owner(p, site.node, l).0 && c.column == column)
            .unwrap()
    };
    let lookup_point = read(0, split::Column::SoftmaxKey).point.clone();
    let fixed: Vec<_> = (0..site.layers.len())
        .map(|l| {
            let key = read(l, split::Column::SoftmaxKey);
            let value = read(l, split::Column::Output(0));
            assert_eq!(key.point, lookup_point);
            assert_eq!(value.point, lookup_point);
            (key.value, value.value)
        })
        .collect();
    let node = &p.lowering.graph.nodes[first.0 as usize];
    let shape = &p.lowering.graph.tensors[node.inputs[0].0 as usize].shape;
    let (heads, n) = (shape[0] as usize, shape[1] as usize);
    let (hp, np, lp) = (
        heads.next_power_of_two(),
        n.next_power_of_two(),
        site.layers.len().next_power_of_two(),
    );
    let rows = hp * np;
    let cube = rows * np;
    let mut maximum_rows = SecretBuf::new();
    maximum_rows.resize(lp * rows, G::ZERO);
    let mut maximum = SecretBuf::new();
    maximum.resize(lp * cube, G::ZERO);
    let mut scores = SecretBuf::new();
    scores.resize(lp * cube, G::ZERO);
    let mut causal = vec![G::ZERO; lp * cube];
    let mut seen = vec![false; site.layers.len() * heads * n];
    for tile in advice.tiles() {
        let id = match tile.tile {
            tessera_compiler::Tile::Attention { node, .. }
            | tessera_compiler::Tile::Node { node } => node,
            _ => continue,
        };
        let Some(l) = (0..site.layers.len()).find(|&l| owner(p, site.node, l).0 == id) else {
            continue;
        };
        let values = source.tile(tile).unwrap();
        let maximum = values
            .gadget()
            .iter()
            .find(|(r, _)| *r == AuxColumn::Maximum)
            .unwrap();
        let node = &p.lowering.graph.nodes[id as usize];
        let share = tile
            .shares
            .iter()
            .find(|s| s.tensor == node.outputs[1])
            .unwrap();
        for (position, &value) in coordinates(&share.ranges).iter().zip(maximum.1.iter()) {
            let (h, q) = (position[0] as usize, position[1] as usize);
            let at = (l * heads + h) * n + q;
            assert!(!seen[at]);
            seen[at] = true;
            maximum_rows[(l * hp + h) * np + q] = G::from_i64(value);
        }
    }
    assert!(seen.into_iter().all(|v| v));
    for l in 0..site.layers.len() {
        let node = &p.lowering.graph.nodes[owner(p, site.node, l).0 as usize];
        let x = witness.tensor(node.inputs[0]).unwrap();
        for h in 0..hp {
            for q in 0..np {
                for k in 0..np {
                    let at = ((l * hp + h) * np + q) * np + k;
                    maximum[at] = maximum_rows[(l * hp + h) * np + q];
                    if h < heads && q < n && k < n {
                        scores[at] = G::from_i64(x[(h * n + q) * n + k]);
                    }
                    if h < heads && k <= q && q < n {
                        causal[at] = G::ONE;
                    }
                }
            }
        }
    }

    let forest = if n == 1 {
        assert!(bound.forests.is_empty());
        None
    } else {
        assert_eq!(bound.forests.len(), 1);
        Some((
            bound.forests[0].claim.point.clone(),
            bound.forests[0].claim.value,
        ))
    };
    let padding = (0..lp * rows)
        .map(|i| {
            if i / rows < site.layers.len() && i % rows / np < heads && i % np < n {
                G::ZERO
            } else {
                G::ONE
            }
        })
        .collect();
    let mut native = Native {
        stack,
        lookup: lookup_point,
        fixed,
        forest,
        heads,
        maximum,
        maximum_rows,
        padding,
        scores,
        causal,
        equalities: Vec::new(),
        targets: Vec::new(),
    };
    let proofs: Vec<_> = (1..plan.stages.len())
        .map(|k| {
            prover
                .prove_stage_composed(transcript, k as u16, &mut native)
                .unwrap()
        })
        .collect();
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
    let leaf: Vec<_> = points
        .iter()
        .zip(leaf)
        .map(|(p, v)| stages::LeafValues {
            point: es(p),
            values: es(v),
        })
        .collect();
    let rope = tv_tables::Table::generate(tv_tables::TableId(0), graph.tables()[0].kind.clone())
        .unwrap()
        .rope()
        .unwrap()
        .clone();
    let accepted = stages::verify_composed_bound(
        verifier_transcript,
        independent,
        &rope,
        &leaf,
        bound,
        &proofs,
        &native,
    )
    .unwrap();
    assert_eq!(accepted.graph.pool.len(), prover.pool().len());
    for (claim, value) in accepted.graph.pool.iter().zip(prover.pool()) {
        assert_eq!(claim.value, e(value.unwrap()));
        assert_eq!(claim.value, pool_value(p, graph, witness, claim));
    }
    assert_eq!(accepted.native.len(), 3);
    for raw in &accepted.native {
        if raw.tag == 12 {
            assert_eq!(
                raw.values,
                vec![e(tessera_poly::evaluate_base::<F>(
                    &native.maximum_rows,
                    &raw.point.iter().copied().map(from_e).collect::<Vec<_>>()
                )
                .unwrap())]
            );
            continue;
        }
        let result = native.with_key(raw.tag, &raw.layer, |key| {
            key.evaluate(&raw.point, &raw.values).unwrap()
        });
        assert_eq!(result.claims.len(), 2);
        for (i, c) in result.claims.iter().enumerate() {
            let data = if i == 0 {
                &native.maximum_rows
            } else {
                &native.scores
            };
            let point: Vec<_> = c
                .point
                .iter()
                .chain(&c.layer_point)
                .copied()
                .map(from_e)
                .collect();
            assert_eq!(
                c.value,
                e(tessera_poly::evaluate_base::<F>(data, &point).unwrap())
            );
        }
    }
    if n == 3 && site.layers.len() == 2 {
        for fault in 0..=6 {
            let mut p2 = tessera_prover::StageProver::new(&p.lowering, plan, witness);
            let _leaf = p2.leaf(points).unwrap();
            let _atoms = p2.leaf_split_values().unwrap();
            let mut t = super::transcript(Mode::NonInteractive);
            let error = if fault == 6 {
                p2.prove_stage(&mut t, 1)
            } else {
                p2.prove_stage_composed(
                    &mut t,
                    1,
                    &mut Broken {
                        native: &native,
                        fault,
                    },
                )
            }
            .unwrap_err();
            assert!(
                matches!(error, StageError::Internal { .. }),
                "fault {fault}: {error}"
            );
        }
    }
    assert_eq!(
        e(transcript.challenge("test/after-mixed")),
        verifier_transcript
            .challenge_fp2(tv_hash::transcript::Ascii::new("test/after-mixed").unwrap())
            .unwrap()
    );
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
