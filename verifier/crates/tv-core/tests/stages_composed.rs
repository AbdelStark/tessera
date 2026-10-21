// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent mixed stage reference proofs and exact routing/rejection controls.
//! The extra identity relation is a composition oracle, not a complete gadget/model proof.
mod claim_support;
mod stage_support;

use claim_support::{claims_of, with_layers};
use reference::{Fe, Form, Product};
use stage_support::{
    Layout, NativeOracle, Prover, Sources, Witness, eq_at, shape_vars, to_fe, to_fp2,
};
use tv_core::lowering::{self, Graph, NodeId, Op, TensorId};
use tv_core::schedule::{self, Native, NativeOutput, Relation, Schedule, TensorRef};
use tv_core::stages::{
    self, Composed, LeafValues, NativeChecks, NativeContext, NativeFinal, StageError, StageProof,
    SumcheckRejection,
};
use tv_core::sumcheck::{Instance, SumcheckError};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::{Table, TableId, rope::Rope};
use tv_testkit::{SplitMix64, sumcheck as reference};

fn context(mode: Mode) -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [31; 32],
        session_id: [8; 16],
    }
}
fn model(n: u32, layers: u32) -> (Graph, Rope) {
    model_named("tiny-2-servable.d", n, layers)
}
fn model_named(name: &str, n: u32, layers: u32) -> (Graph, Rope) {
    let c = with_layers(&claims_of(name), layers);
    let public = tv_core::validation::check_public(&c).unwrap();
    let g = lowering::lower(&public, n).unwrap();
    let rope = Table::generate(TableId(0), g.tables()[0].kind.clone())
        .unwrap()
        .rope()
        .unwrap()
        .clone();
    (g, rope)
}

#[derive(Clone)]
struct Oracle<'a> {
    graph: &'a Graph,
    witness: &'a Witness,
    tensor: TensorRef,
    target: Vec<Fe>,
    dims: Vec<u32>,
    fault: Option<&'static str>,
}
impl Oracle<'_> {
    fn layout(&self) -> Layout<'_> {
        Layout::new(self.graph)
    }
    fn score_table(&self) -> Vec<Fe> {
        let layout = self.layout();
        let mut out = Vec::new();
        let own = 1 << shape_vars(&self.dims);
        for t in layout.tensors(self.tensor) {
            for v in self.witness.table(t, &self.dims) {
                out.extend([v; 4]);
            }
        }
        let lambda = if matches!(self.tensor, TensorRef::Single(_)) {
            0
        } else {
            layout.lambda
        };
        out.resize((own * 4) << lambda, Fe::ZERO);
        out
    }
    fn residual_table(&self) -> Vec<Fe> {
        let layout = self.layout();
        let dims = layout.dims(self.graph.checkpoints()[0]);
        let own = 1 << shape_vars(&dims);
        let mut out = Vec::new();
        for t in layout.tensors(TensorRef::LayerInput) {
            out.extend(self.witness.table(t, &dims));
        }
        out.resize(own << layout.lambda, Fe::ZERO);
        out
    }
    fn row_dims(&self) -> Vec<u32> {
        self.layout().dims(self.graph.checkpoints()[0])
    }
    fn mask(&self, p: &[Fp2]) -> Fp2 {
        let layout = self.layout();
        let mut from = 0;
        let mut psi = Fp2::ONE;
        for d in self.row_dims().iter().rev() {
            let vars = stage_support::vars(*d);
            psi *= tv_core::selectors::lt(&p[from..from + vars], u64::from(*d)).unwrap();
            from += vars;
        }
        psi * tv_core::selectors::lt(&p[from..], layout.layers as u64).unwrap()
    }
}
impl NativeOracle for Oracle<'_> {
    fn product(&self, p: &Prover<'_>, stage: usize, instance: usize, zero: &[Fe]) -> Product {
        let Relation::Native { tag, .. } = p.schedule.stages()[stage].instances[instance].relation
        else {
            panic!()
        };
        match tag {
            10 => {
                let mut target = self.target.clone();
                if !matches!(self.tensor, TensorRef::Single(_)) {
                    target.extend(&p.sources.layer);
                }
                let table = self.score_table();
                Product {
                    public: vec![(0..table.len()).map(|i| eq_at(&target, i)).collect()],
                    tables: vec![table.clone(), table],
                    lambda: None,
                    form: Form::Difference(-Fe::ONE),
                }
            }
            20 | 21 => {
                let table = self.residual_table();
                let layout = self.layout();
                let dims = self.row_dims();
                let vars = shape_vars(&dims);
                let valid: Vec<_> = (0..table.len())
                    .map(|index| {
                        let mut at = index;
                        let mut yes = index >> vars < layout.layers;
                        for &d in dims.iter().rev() {
                            let width = 1 << stage_support::vars(d);
                            yes &= at % width < d as usize;
                            at /= width;
                        }
                        if yes { Fe::ZERO } else { Fe::ONE }
                    })
                    .collect();
                Product {
                    public: vec![(0..table.len()).map(|i| eq_at(zero, i)).collect(), valid],
                    tables: vec![table],
                    lambda: None,
                    form: Form::Product,
                }
            }
            _ => panic!("unknown reference native tag"),
        }
    }
}
impl NativeChecks for Oracle<'_> {
    fn instance(&self, tag: u32, ctx: NativeContext<'_>) -> Result<Instance, StageError> {
        let layout = self.layout();
        let mut out = match tag {
            10 => {
                let mut target = self.target.clone();
                if !matches!(self.tensor, TensorRef::Single(_)) {
                    target.extend(ctx.layer.iter().copied().map(to_fe));
                }
                Instance {
                    claim: to_fp2(Fe::base(2) * reference::mle(&self.score_table(), &target)),
                    vars: target.len(),
                    evals: 2,
                }
            }
            20 | 21 => Instance {
                claim: Fp2::ZERO,
                vars: shape_vars(&self.row_dims()) + layout.lambda,
                evals: 1,
            },
            _ => return Err(StageError::Internal),
        };
        if self.fault == Some("instance-vars") {
            out.vars += 1;
        }
        if self.fault == Some("instance-evals") {
            out.evals += 1;
        }
        if self.fault == Some("zero-claim") && tag == 20 {
            out.claim = Fp2::ONE;
        }
        Ok(out)
    }
    fn evaluate(
        &self,
        tag: u32,
        ctx: NativeContext<'_>,
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<NativeFinal, StageError> {
        if self.fault == Some("memory") {
            return Err(StageError::Memory);
        }
        let mut out = match tag {
            10 => {
                let mut target: Vec<_> = self.target.iter().copied().map(to_fp2).collect();
                if !matches!(self.tensor, TensorRef::Single(_)) {
                    target.extend(ctx.layer);
                }
                let e = tv_core::sumcheck::eq(&target, point).unwrap();
                let sum = values[0] + values[1];
                NativeFinal {
                    value: e * sum,
                    graph: vec![sum * Fp2::from(Fp::reduce(2).inverse().unwrap())],
                }
            }
            20 | 21 => {
                let zero = &ctx.zero[ctx.zero.len() - point.len()..];
                NativeFinal {
                    value: tv_core::sumcheck::eq(zero, point).unwrap()
                        * (Fp2::ONE - self.mask(point))
                        * values[0],
                    graph: vec![],
                }
            }
            _ => return Err(StageError::Internal),
        };
        if self.fault == Some("output-count") && tag == 10 {
            out.graph.push(Fp2::ZERO);
        }
        if self.fault == Some("output-value") && tag == 10 {
            out.graph[0] += Fp2::ONE;
        }
        Ok(out)
    }
}
fn program(g: &Graph) -> (Vec<schedule::Deferred>, Vec<Native>, TensorId, Vec<u32>) {
    let soft = g
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, n)| matches!(n.op(), Op::SoftmaxRow { .. }))
        .unwrap();
    let soft_node = NodeId(soft.0 as u32);
    let score = soft.1.inputs()[0];
    let dims = g.tensor(score).unwrap().shape.dims().to_vec();
    let root = g
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, n)| matches!(n.op(), Op::RsqrtRow { .. }))
        .unwrap();
    let layout = Layout::new(g);
    let row_dims = layout.dims(g.checkpoints()[0]);
    (
        vec![(soft_node, Some(0))],
        vec![
            Native {
                tag: 10,
                node: soft_node,
                vars: shape_vars(&dims) + 2 + layout.lambda,
                degree: 2,
                evaluations: 2,
                zero: false,
                outputs: vec![NativeOutput {
                    tensor: score,
                    from: 2,
                    vars: shape_vars(&dims),
                }],
            },
            Native {
                tag: 20,
                node: soft_node,
                vars: shape_vars(&row_dims) + layout.lambda,
                degree: 3,
                evaluations: 1,
                zero: true,
                outputs: vec![],
            },
            Native {
                tag: 21,
                node: NodeId(root.0 as u32),
                vars: shape_vars(&row_dims) + layout.lambda,
                degree: 3,
                evaluations: 1,
                zero: true,
                outputs: vec![],
            },
        ],
        score,
        dims,
    )
}
fn prove<'a>(
    g: &'a Graph,
    w: &'a Witness,
    s: &'a Schedule,
    rope: &'a Rope,
    oracle: &'a Oracle<'a>,
    mode: Mode,
) -> (
    Vec<LeafValues>,
    Vec<Fp2>,
    Vec<StageProof>,
    reference::Transcript,
) {
    let mut p = Prover {
        layout: Layout::new(g),
        witness: w,
        schedule: s,
        rope,
        sources: Sources::default(),
        omit_onehot: None,
        native: Some(oracle),
    };
    let mut rng = SplitMix64::new(831);
    let (leaf, splits) = p.leaf(&mut rng);
    let mut t = reference::Transcript::new(&context(mode).to_tce());
    t.absorb("stage/0/split", &reference::tce(&splits));
    let proofs = p.stages(&mut t, None);
    (
        leaf,
        splits.iter().copied().map(to_fp2).collect(),
        proofs,
        t,
    )
}
fn pool_truth(g: &Graph, w: &Witness, s: &Schedule, rope: &Rope, result: &Composed) {
    let p = Prover {
        layout: Layout::new(g),
        witness: w,
        schedule: s,
        rope,
        sources: Sources::default(),
        omit_onehot: None,
        native: None,
    };
    for claim in &result.graph.pool {
        assert_eq!(to_fe(claim.value), p.pool_value(claim));
    }
}
#[test]
fn one_mixed_batch_routes_deferred_native_graph_sources_and_all_original_pool_claims() {
    for (n, layers) in [(1, 1), (1, 3), (3, 2), (3, 3), (5, 4)] {
        let (g, rope) = model(n, layers);
        let (deferred, spec, score, dims) = program(&g);
        let s = schedule::derive_composed(&g, &deferred, &spec).unwrap();
        assert!(
            !s.leaf()
                .iter()
                .any(|l| l.site == deferred[0].0 && l.input == Some(0))
        );
        let st = &s.stages()[0];
        let virtual_at = st
            .instances
            .iter()
            .position(|i| matches!(i.relation, Relation::Native { tag: 10, .. }))
            .unwrap();
        let zero_at = st
            .instances
            .iter()
            .position(|i| matches!(i.relation, Relation::Native { tag: 20, .. }))
            .unwrap();
        assert!(virtual_at < zero_at);
        assert_eq!(st.degree, 3);
        // RFC-0037: the site's graph parts precede its native slots, then later sites.
        for leaf in s.leaf() {
            let indices = match &leaf.claim.route {
                schedule::Route::Instance(i) => vec![*i],
                schedule::Route::Split(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        schedule::Part::Virtual { instance, .. } => Some(*instance),
                        _ => None,
                    })
                    .collect(),
                schedule::Route::Pool => Vec::new(),
            };
            for index in indices {
                assert_eq!(index < virtual_at, leaf.site.0 <= spec[0].node.0);
            }
        }
        let zero_start = st
            .instances
            .iter()
            .position(|i| !i.zero.is_empty())
            .unwrap();
        assert!(
            st.instances[zero_start..]
                .windows(2)
                .all(|w| w[0].node.0 <= w[1].node.0)
        );
        assert_eq!(
            st.rounds,
            st.instances.iter().map(|i| i.vars).max().unwrap()
        );
        assert_eq!(
            st.zero_vars,
            spec[1]
                .vars
                .max(schedule::derive(&g).unwrap().stages()[0].zero_vars)
        );
        let external = &st.instances[virtual_at];
        assert_eq!(external.input_evaluations(), 2);
        assert_eq!(external.evals.len(), 1);
        let w = Witness::new(&g, &rope, 91);
        let target: Vec<_> = (0..shape_vars(&dims) + 2)
            .map(|i| Fe::new(3 + i as u64, 9 + i as u64))
            .collect();
        let oracle = Oracle {
            graph: &g,
            witness: &w,
            tensor: TensorRef::Stacked(score),
            target,
            dims,
            fault: None,
        };
        for mode in [Mode::Live, Mode::NonInteractive] {
            let (leaf, split, proofs, mut pt) = prove(&g, &w, &s, &rope, &oracle, mode);
            let mut vt = Transcript::new(&context(mode));
            let result =
                stages::verify_composed(&mut vt, &s, &rope, &leaf, &split, &proofs, &oracle)
                    .unwrap();
            pool_truth(&g, &w, &s, &rope, &result);
            assert_eq!(result.native.len(), 3);
            for record in &result.native {
                let source = if record.tag == 10 {
                    oracle.score_table()
                } else {
                    oracle.residual_table()
                };
                let expected = reference::mle(
                    &source,
                    &record.point.iter().copied().map(to_fe).collect::<Vec<_>>(),
                );
                assert!(record.values.iter().all(|&v| to_fe(v) == expected));
                assert_eq!(record.layer.len(), Layout::new(&g).lambda);
                assert_eq!(record.zero.len(), s.stages()[0].zero_vars);
            }
            assert_eq!(
                to_fp2(pt.challenge_fp2("stage/999/gamma")),
                vt.challenge_fp2(Ascii::new("stage/999/gamma").unwrap())
                    .unwrap()
            );
            for fault in [
                "instance-vars",
                "instance-evals",
                "zero-claim",
                "output-count",
                "memory",
            ] {
                let bad = Oracle {
                    fault: Some(fault),
                    ..oracle.clone()
                };
                let expected = if fault == "memory" {
                    StageError::Memory
                } else {
                    StageError::Internal
                };
                assert!(
                    matches!(stages::verify_composed(&mut Transcript::new(&context(mode)),&s,&rope,&leaf,&split,&proofs,&bad),Err(e) if e==expected)
                );
            }
            let bad = Oracle {
                fault: Some("output-value"),
                ..oracle.clone()
            };
            assert!(matches!(
                stages::verify_composed(
                    &mut Transcript::new(&context(mode)),
                    &s,
                    &rope,
                    &leaf,
                    &split,
                    &proofs,
                    &bad
                ),
                Err(StageError::Sumcheck { stage: 2, .. })
                    | Err(StageError::Split { stage: 1, .. })
            ));
            assert!(matches!(
                stages::verify(
                    &mut Transcript::new(&context(mode)),
                    &s,
                    &rope,
                    &leaf,
                    &split,
                    &proofs
                ),
                Err(StageError::Internal)
            ));
            let mut bad = proofs.clone();
            bad[0].sumcheck.final_evals[0] += Fp2::ONE;
            assert_eq!(
                stages::verify_composed(
                    &mut Transcript::new(&context(mode)),
                    &s,
                    &rope,
                    &leaf,
                    &split,
                    &bad,
                    &oracle
                )
                .err()
                .unwrap()
                .rejection(),
                Some(SumcheckRejection {
                    stage: 1,
                    round: st.rounds as u32
                })
            );
            let bound = tv_core::leaf::split::Bound {
                columns: vec![],
                multiplicities: vec![],
                forests: vec![],
                atoms: split.clone(),
                point: vec![],
            };
            let mut vt = Transcript::new(&context(mode));
            let mut bytes = (split.len() as u64).to_le_bytes().to_vec();
            for x in &split {
                bytes.extend(x.to_le_bytes());
            }
            vt.absorb(Ascii::new("stage/0/split").unwrap(), &bytes);
            let result =
                stages::verify_composed_bound(&mut vt, &s, &rope, &leaf, &bound, &proofs, &oracle)
                    .unwrap();
            pool_truth(&g, &w, &s, &rope, &result);
            assert_eq!(
                vt.challenge_fp2(Ascii::new("stage/999/gamma").unwrap())
                    .unwrap(),
                to_fp2(reference_next(&g, &w, &s, &rope, &oracle, mode))
            );
        }
    }
}
fn reference_next(
    g: &Graph,
    w: &Witness,
    s: &Schedule,
    rope: &Rope,
    o: &Oracle<'_>,
    mode: Mode,
) -> Fe {
    let (_, _, _, mut t) = prove(g, w, s, rope, o, mode);
    t.challenge_fp2("stage/999/gamma")
}

#[test]
fn malformed_native_descriptors_never_make_a_schedule() {
    let (g, _) = model(3, 3);
    let (deferred, spec, _, _) = program(&g);
    let bad = |deferred: &[schedule::Deferred], native: &[Native]| {
        assert!(schedule::derive_composed(&g, deferred, native).is_err())
    };
    let mut n = spec.clone();
    n[1].tag = n[0].tag;
    bad(&deferred, &n);
    for degree in [0, 8] {
        let mut n = spec.clone();
        n[0].degree = degree;
        bad(&deferred, &n);
    }
    let mut n = spec.clone();
    n[0].vars = 64;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].vars = 0;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].evaluations = 0;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].node = NodeId(u32::MAX);
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].node = NodeId(
        g.nodes()
            .iter()
            .position(|n| matches!(n.op(), Op::ConstMul { .. }))
            .unwrap() as u32,
    );
    bad(&[], std::slice::from_ref(&n[0]));
    let layout = Layout::new(&g);
    let mut n = spec.clone();
    n[0].node = layout.node_at(n[0].node, 1);
    bad(&[], &n);
    let mut n = spec.clone();
    n[0].outputs[0].from = n[0].vars;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].outputs[0].from = usize::MAX;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].outputs[0].vars -= 1;
    bad(&deferred, &n);
    let mut n = spec.clone();
    n[0].outputs[0].tensor = TensorId(u32::MAX);
    bad(&[], &n);
    let mut n = spec.clone();
    n[0].outputs.clear();
    bad(&deferred, &n);
    bad(&[deferred[0], deferred[0]], &spec);
    bad(&[(deferred[0].0, Some(usize::MAX))], &spec);
    bad(&[(NodeId(u32::MAX), Some(0))], &spec);
    bad(&[(layout.node_at(deferred[0].0, 1), Some(0))], &spec);
    let pure = schedule::derive(&g).unwrap();
    assert_eq!(pure, schedule::derive_composed(&g, &[], &[]).unwrap());
}

#[test]
fn single_message_edits_reject_at_the_exact_mixed_stage_and_native_projections_bind() {
    let (g, rope) = model(3, 3);
    let (deferred, spec, score, dims) = program(&g);
    let s = schedule::derive_composed(&g, &deferred, &spec).unwrap();
    let w = Witness::new(&g, &rope, 178);
    let target = (0..shape_vars(&dims) + 2)
        .map(|i| Fe::new(i as u64 + 7, i as u64 + 5))
        .collect();
    let oracle = Oracle {
        graph: &g,
        witness: &w,
        tensor: TensorRef::Stacked(score),
        target,
        dims,
        fault: None,
    };
    let mode = Mode::NonInteractive;
    let (leaf, split, proofs, _) = prove(&g, &w, &s, &rope, &oracle, mode);
    let run = |schedule: &Schedule, p: &[StageProof]| {
        stages::verify_composed(
            &mut Transcript::new(&context(mode)),
            schedule,
            &rope,
            &leaf,
            &split,
            p,
            &oracle,
        )
    };
    for (k, stage) in proofs.iter().enumerate() {
        let mut at = 0;
        for instance in &s.stages()[k].instances {
            if matches!(instance.relation, Relation::Native { .. }) {
                for j in at..at + instance.input_evaluations() {
                    let mut bad = proofs.clone();
                    bad[k].sumcheck.final_evals[j] += Fp2::ONE;
                    assert_eq!(
                        run(&s, &bad).err().unwrap().rejection(),
                        Some(SumcheckRejection {
                            stage: (k + 1) as u16,
                            round: s.stages()[k].rounds as u32
                        })
                    );
                }
            }
            at += instance.input_evaluations();
        }
        for j in 0..stage.sumcheck.rounds.len() {
            for e in 0..stage.sumcheck.rounds[j].len() {
                let mut bad = proofs.clone();
                bad[k].sumcheck.rounds[j][e] += Fp2::ONE;
                assert_eq!(
                    run(&s, &bad).err().unwrap().rejection(),
                    Some(SumcheckRejection {
                        stage: (k + 1) as u16,
                        round: s.stages()[k].rounds as u32
                    })
                );
            }
        }
    }
    let mut n = spec.clone();
    n[0].outputs[0].from = 1;
    let bad = schedule::derive_composed(&g, &deferred, &n).unwrap();
    assert_eq!(
        run(&bad, &proofs).err().unwrap().rejection(),
        Some(SumcheckRejection {
            stage: 2,
            round: bad.stages()[1].rounds as u32
        })
    );
    let mut p = proofs.clone();
    p[0].sumcheck.final_evals.pop();
    assert!(matches!(
        run(&s, &p),
        Err(StageError::Sumcheck {
            stage: 1,
            error: SumcheckError::EvaluationCount { .. }
        })
    ));
    let mut p = proofs.clone();
    p[0].splits.push(Fp2::ZERO);
    assert!(matches!(
        run(&s, &p),
        Err(StageError::SplitCount { stage: 1, .. })
    ));
    let mut p = proofs.clone();
    p[0].stage = 2;
    assert!(matches!(run(&s, &p), Err(StageError::StageId { .. })));
    assert!(matches!(
        run(&s, &proofs[..proofs.len() - 1]),
        Err(StageError::StageCount { .. })
    ));
}

#[test]
fn native_fused_key_splits_and_non_template_logits_bind_to_the_original_graph() {
    let mut c = with_layers(&claims_of("tiny-2-minimal.d"), 3);
    for layer in 0..3 {
        claim_support::make_pot(
            &mut c,
            tv_core::claims::QuantPoint::Layer {
                layer,
                site: tv_core::claims::Site::AttnResidual,
            },
            &[4],
        );
    }
    let g = lowering::lower(&tv_core::validation::check_public(&c).unwrap(), 3).unwrap();
    let rope = Table::generate(TableId(0), g.tables()[0].kind.clone())
        .unwrap()
        .rope()
        .unwrap()
        .clone();
    let w = Witness::new(&g, &rope, 649);
    for logits in [false, true] {
        let (_, mut spec, _, _) = program(&g);
        let (owner, tensor, input, reference) = if logits {
            let (index,node)=g.nodes().iter().enumerate().find(|(_,node)|matches!(node.op(),Op::MatMulWeight{weight} if weight.ty==tv_core::claims::WeightType::LmHead)).unwrap();
            (
                NodeId(index as u32),
                node.outputs()[0],
                None,
                TensorRef::Single(node.outputs()[0]),
            )
        } else {
            let ordinary = schedule::derive(&g).unwrap();
            let split = ordinary
                .leaf()
                .iter()
                .find(|c| c.claim.parts() == 2)
                .unwrap();
            let index = split.site.0 as usize;
            let node = g.node(split.site).unwrap();
            (
                NodeId(index as u32),
                node.inputs()[0],
                Some(0),
                TensorRef::Stacked(node.inputs()[0]),
            )
        };
        let dims = g.tensor(tensor).unwrap().shape.dims().to_vec();
        let lambda = if logits { 0 } else { Layout::new(&g).lambda };
        spec[0].node = owner;
        spec[0].vars = shape_vars(&dims) + 2 + lambda;
        spec[0].degree = 4;
        spec[0].outputs = vec![NativeOutput {
            tensor,
            from: 2,
            vars: shape_vars(&dims),
        }];
        let s = schedule::derive_composed(&g, &[(owner, input)], &spec).unwrap();
        assert_eq!(s.stages()[0].degree, 4);
        assert!(!s.leaf().iter().any(|l| l.site == owner && l.input == input));
        let native = s.stages()[0]
            .instances
            .iter()
            .find(|i| matches!(i.relation, Relation::Native { tag: 10, .. }))
            .unwrap();
        assert_eq!(native.evals[0].parts(), if logits { 0 } else { 2 });
        let oracle = Oracle {
            graph: &g,
            witness: &w,
            tensor: reference,
            target: (0..shape_vars(&dims) + 2)
                .map(|i| Fe::new(i as u64 + 5, i as u64 + 9))
                .collect(),
            dims,
            fault: None,
        };
        for mode in [Mode::Live, Mode::NonInteractive] {
            let (leaf, split, proofs, mut pt) = prove(&g, &w, &s, &rope, &oracle, mode);
            let mut vt = Transcript::new(&context(mode));
            let result =
                stages::verify_composed(&mut vt, &s, &rope, &leaf, &split, &proofs, &oracle)
                    .unwrap();
            pool_truth(&g, &w, &s, &rope, &result);
            assert_eq!(
                to_fp2(pt.challenge_fp2("stage/999/gamma")),
                vt.challenge_fp2(Ascii::new("stage/999/gamma").unwrap())
                    .unwrap()
            );
            if !logits {
                let mut bad = proofs.clone();
                assert!(!bad[0].splits.is_empty());
                for j in 0..bad[0].splits.len() {
                    bad[0].splits[j] += Fp2::ONE;
                    assert_eq!(
                        stages::verify_composed(
                            &mut Transcript::new(&context(mode)),
                            &s,
                            &rope,
                            &leaf,
                            &split,
                            &bad,
                            &oracle
                        )
                        .err()
                        .unwrap(),
                        StageError::Split {
                            stage: 1,
                            round: s.stages()[0].rounds as u32
                        }
                    );
                    bad[0].splits[j] -= Fp2::ONE;
                }
                let bad = Oracle {
                    fault: Some("output-value"),
                    ..oracle.clone()
                };
                assert_eq!(
                    stages::verify_composed(
                        &mut Transcript::new(&context(mode)),
                        &s,
                        &rope,
                        &leaf,
                        &split,
                        &proofs,
                        &bad
                    )
                    .err()
                    .unwrap(),
                    StageError::Split {
                        stage: 1,
                        round: s.stages()[0].rounds as u32
                    }
                );
            }
        }
    }
}
