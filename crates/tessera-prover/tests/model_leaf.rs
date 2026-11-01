// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete canonical CPU model leaf stages, checked by the production independent verifier (DEP-2).
use std::fs;
use std::path::{Path, PathBuf};
use tessera_codec::{Ascii, DecodeLimits, Secret, envelope, schema};
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{
    AuxColumn, CompareColumn, Lowering, ModelAdviceLayout, PublicModel, Selection, analyze,
    claim_list, select_model,
};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Slack, TableColumns};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{ClaimList, GadgetPolicy, IntegerModel, NodeId};
use tessera_protocol::ParamSet;
use tessera_prover::leaf::{LeafLayout, ModelColumn, ModelWitness, prepare_leaf};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::GeneratedTable;
use tessera_witness::{Engine, Resident, WitnessSource};
use tv_core::leaf::{self as tv, split};
use tv_field::{Fp as V, Fp2 as E};

mod batch_leaf;
mod cpu_api;
mod model_groups;
mod model_openings;
mod model_stages;
mod model_witness_memory;

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}
fn model(name: &str) -> IntegerModel {
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(vectors().join("models/models.json")).unwrap())
            .unwrap();
    let entry = index["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == name)
        .unwrap();
    envelope::open(
        &fs::read(
            vectors()
                .join("models")
                .join(entry["file"].as_str().unwrap()),
        )
        .unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap()
}
struct Public {
    lowering: Lowering,
    layers: LayerMap,
    selection: Selection,
    claims: ClaimList,
}
fn public(model: &IntegerModel, n: u32, policy: GadgetPolicy) -> Public {
    let lowering = tessera_compiler::lower(PublicModel::of(model), n).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let selection =
        select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy).unwrap();
    let p = ParamSet::TEST_SMALL;
    let claims = claim_list(
        PublicModel::of(model),
        &selection,
        &policy,
        p.layout,
        ParamSetId(Ascii::new(p.id).unwrap()),
    )
    .unwrap();
    let layers = LayerMap::new(&lowering).unwrap();
    Public {
        lowering,
        layers,
        selection,
        claims,
    }
}
fn verifier(p: &Public, n: u32) -> (tv_core::lowering::Graph, tv_core::claims::ClaimList) {
    let bytes = envelope::seal(&schema::CLAIM_LIST, &p.claims);
    let claims =
        tv_core::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
    let graph =
        tv_core::lowering::lower(&tv_core::validation::check_public(&claims).unwrap(), n).unwrap();
    (graph, claims)
}
fn v(x: G) -> V {
    V::from_canonical(x.to_canonical_u64()).unwrap()
}
fn e(x: F) -> E {
    let [a, b] = x.coeffs();
    E::new(v(a), v(b))
}
fn es(xs: &[F]) -> Vec<E> {
    xs.iter().copied().map(e).collect()
}
fn role(c: ModelColumn) -> split::Column {
    use tv_core::advice::{Auxiliary as A, Comparison as C, ComparisonRole as R};
    match c {
        ModelColumn::Input(i) => split::Column::Input(i),
        ModelColumn::GreedyKey => split::Column::GreedyKey,
        ModelColumn::GreedyKeyOrdinal(r) => split::Column::greedy_key(r),
        ModelColumn::Output(i) => split::Column::Output(i),
        ModelColumn::RootInput => split::Column::RootInput,
        ModelColumn::ExpKey => split::Column::SoftmaxKey,
        ModelColumn::Slack(Slack::A) => split::Column::SoftmaxA,
        ModelColumn::Slack(Slack::B) => split::Column::SoftmaxB,
        ModelColumn::Auxiliary(a) => split::Column::Auxiliary(match a {
            AuxColumn::Low => A::RequantLow,
            AuxColumn::Middle => A::RequantMiddle,
            AuxColumn::Saturation => A::RequantSaturation,
            AuxColumn::Maximum => A::SoftmaxMaximum,
            AuxColumn::InputLimb(j) => A::RootInput(j),
            AuxColumn::RootByte(j) => A::RootLow(j),
            AuxColumn::Slack { second, limb } => {
                if second {
                    A::SoftmaxB(limb)
                } else {
                    A::SoftmaxA(limb)
                }
            }
            AuxColumn::Comparison {
                upper,
                column,
                limb,
            } => A::RootComparison(
                if upper { C::Upper } else { C::Lower },
                match column {
                    CompareColumn::LeftByte => R::LeftDigit,
                    CompareColumn::LeftCarry => R::LeftCarry,
                    CompareColumn::RightByte => R::RightDigit,
                    CompareColumn::RightCarry => R::RightCarry,
                    CompareColumn::Difference => R::Difference,
                    CompareColumn::Borrow => R::Borrow,
                },
                limb,
            ),
        }),
    }
}
fn owner(p: &Public, site: NodeRef, l: usize) -> NodeId {
    match site {
        NodeRef::Node(id) => id,
        NodeRef::Position(i) => p.layers.node(i, l as u32),
    }
}
fn compare(p: &Public, n: u32, layout: &LeafLayout<ModelColumn>) {
    assert_eq!(
        tessera_prover::derive_model_schedule(&p.claims, n)
            .unwrap()
            .lookup_entries,
        layout
            .instances()
            .iter()
            .map(|i| (i.table, i.entries))
            .collect::<Vec<_>>()
    );
    let (graph, claims) = verifier(p, n);
    let sites = split::model_sites(&graph, claims.gadget_policy).unwrap();
    let independent = split::Layout::derive(&graph, claims.gadget_policy).unwrap();
    assert_eq!(layout.split_len(), independent.values());
    assert_eq!(
        layout
            .instances()
            .iter()
            .map(|i| (i.table.0, i.entries))
            .collect::<Vec<_>>(),
        independent
            .instances()
            .map(|(t, c)| (t.0, c))
            .collect::<Vec<_>>()
    );
    assert_eq!(layout.sites().len(), sites.len());
    for (s, (actual, expected)) in layout.sites().iter().zip(sites).enumerate() {
        assert_eq!(actual.layers.len(), expected.layers.len());
        for (l, (a, b)) in actual.layers.iter().zip(expected.layers).enumerate() {
            assert_eq!(owner(p, actual.node, l).0, b.node.0);
            assert_eq!(a.lookups.len(), b.lookups.len());
            for (lookup, other) in a.lookups.iter().zip(b.lookups) {
                assert_eq!(lookup.lookup.table.0, other.table.0);
                assert_eq!(lookup.vars(), other.variables);
                assert_eq!(lookup.lookup.value.is_some(), other.value.is_some());
                for (form, other) in std::iter::once((&lookup.lookup.key, &other.key))
                    .chain(lookup.lookup.value.as_ref().zip(other.value.as_ref()))
                {
                    assert_eq!(v(form.constant), other.constant);
                    assert_eq!(form.terms.len(), other.terms.len());
                    for (&(coefficient, column), (other_coefficient, read)) in
                        form.terms.iter().zip(&other.terms)
                    {
                        assert_eq!(v(coefficient), *other_coefficient);
                        assert_eq!(role(column), read.column);
                        let own = layout
                            .reads(s, l)
                            .unwrap()
                            .iter()
                            .find(|r| r.term == column)
                            .unwrap();
                        assert_eq!(own.coords, read.coordinates);
                    }
                }
            }
        }
    }
}
fn mle(values: &[i64], point: &[E]) -> E {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .enumerate()
        .fold(E::ZERO, |sum, (row, &value)| {
            let weight = point.iter().enumerate().fold(E::ONE, |a, (bit, &r)| {
                a * if row >> bit & 1 == 0 { E::ONE - r } else { r }
            });
            sum + E::from(v(G::from_i64(value))) * weight
        })
}
fn transcript(mode: Mode) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [71; 32],
        session_id: [130; 16],
    })
}
fn vt(mode: Mode) -> tv_hash::transcript::Transcript {
    use tv_hash::transcript as t;
    t::Transcript::new(&t::SessionContext {
        protocol: t::Ascii::new("tessera-protocol/1").unwrap(),
        mode: if mode == Mode::Live {
            t::Mode::Live
        } else {
            t::Mode::NonInteractive
        },
        params: t::ParamSetId(t::Ascii::new("test-small/1").unwrap()),
        statement_digest: [71; 32],
        session_id: [130; 16],
    })
}

fn bounded_model_named(name: &str) -> IntegerModel {
    let mut model = model(name);
    model.arch.d_model = 2;
    model.arch.n_heads = 1;
    model.arch.n_kv_heads = 1;
    model.arch.d_head = 2;
    model.arch.d_ff = 6;
    model.arch.vocab = 8;
    model.arch.max_context = 8;
    model.quant.public.rsqrt_kappa = 2;
    model.quant.public.exp_sigma = 3;
    model.quant.public.residual_bits = 8;
    model.quant.public.residual_exponent = model
        .quant
        .public
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tessera_ir::QuantPoint::Embed)
        .unwrap()
        .1;
    let shifts = model.quant.public.shift_class_sets.clone();
    for (point, values) in &mut model.quant.public.shift_class_sets {
        if let tessera_ir::QuantPoint::Layer { site, .. } = *point {
            *values = shifts
                .iter()
                .find(|(p, _)| *p == tessera_ir::QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1
                .clone();
        }
    }
    let affine = model.quant.public.affine_sites.clone();
    for (point, value) in &mut model.quant.public.affine_sites {
        if let tessera_ir::QuantPoint::Layer { site, .. } = *point {
            *value = affine
                .iter()
                .find(|(p, _)| *p == tessera_ir::QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1;
        }
    }
    if name != "tiny-2-servable" {
        for (point, shifts) in &mut model.quant.public.shift_class_sets {
            let base = if matches!(
                point,
                tessera_ir::QuantPoint::Layer {
                    site: tessera_ir::Site::Silu,
                    ..
                }
            ) {
                12
            } else {
                24
            };
            for (class, shift) in shifts.iter_mut().enumerate() {
                *shift = base + class as i8;
            }
        }
        for (_, shift) in &mut model.quant.public.affine_sites {
            *shift = 24;
        }
    }
    let shapes = model.arch.weight_shapes().unwrap();
    for weight in &mut model.weights {
        let shape = shapes[weight.ty as usize];
        weight.rows = shape.out;
        weight.cols = shape.inputs;
        weight.data = (0..shape.out * shape.inputs)
            .map(|i| (i % 5) as i8 - 2)
            .collect();
    }
    for channel in &mut model.quant.channels {
        let n = shapes[channel.ty as usize].out as usize;
        if let Some(&value) = channel.class.first() {
            channel.class.resize(n, value);
            channel.class.fill(0);
        }
        if let Some(&value) = channel.multiplier.first() {
            channel.multiplier.resize(n, value);
        }
    }
    model
}

fn bounded_model() -> IntegerModel {
    bounded_model_named("tiny-2-servable")
}

#[test]
fn canonical_factory_matches_every_independent_model_form_and_request() {
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        let model = model(name);
        for n in [1, 3, 5] {
            let p = public(&model, n, GadgetPolicy::default());
            let layout =
                LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
            compare(&p, n, &layout);
        }
    }
}

fn check_complete(model: &IntegerModel, n: u32, maximum: u8) {
    check_complete_runtime(model, n, maximum, None);
}
fn check_complete_runtime(model: &IntegerModel, n: u32, maximum: u8, prompt: Option<u32>) {
    let p = public(
        model,
        n,
        GadgetPolicy {
            max_log2_rows: maximum,
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    let runtime = prompt.map(|prompt| {
        tessera_compiler::RuntimePlan::derive(&p.lowering, &p.selection, &p.claims, prompt, 0)
            .unwrap()
    });
    let selection = runtime.as_ref().map_or(&p.selection, |r| r.selection());
    let layout = match &runtime {
        Some(r) => LeafLayout::runtime(&p.lowering, &p.layers, r, &p.claims),
        None => LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims),
    }
    .unwrap();
    if runtime.is_none() {
        compare(&p, n, &layout);
    }
    let engine = Engine::new(&p.lowering).unwrap();
    let mut input: Vec<_> = (0..n).map(|i| (i + 1) % model.arch.vocab).collect();
    if let Some(prompt) = prompt {
        for t in prompt..n {
            let pass = engine.run(model, &Secret::new(input.clone())).unwrap();
            let v = model.arch.vocab as usize;
            let logits =
                &pass.tensor(p.lowering.logits).unwrap()[(t as usize - 1) * v..t as usize * v];
            let max = *logits.iter().max().unwrap();
            input[t as usize] = logits.iter().position(|&l| l == max).unwrap() as u32;
        }
    }
    let tokens = Secret::new(input);
    let witness = engine.run(model, &tokens).unwrap();
    let source = match &runtime {
        Some(r) => Resident::runtime(&engine, r, model, &witness),
        None => Resident::new(&engine, &p.selection, model, &witness),
    }
    .unwrap();
    let advice = match &runtime {
        Some(r) => ModelAdviceLayout::runtime(&p.lowering, r, &p.claims),
        None => ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims),
    }
    .unwrap();
    let actual =
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap();
    if runtime.is_none() {
        source_truth(&p, &witness, &source, &advice, &layout, &actual);
    }
    let (graph, claims) = verifier(&p, n);
    let model_tables = p
        .selection
        .tables
        .iter()
        .map(|spec| {
            tv_tables::Table::from_spec(
                &tv_codec::decode(
                    &tessera_codec::encode(spec),
                    &tv_codec::DecodeLimits::MESSAGE,
                    tv_tables::TableSpec::decode,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let tv_runtime = prompt.map(|prompt| {
        tv_core::gadgets::runtime::Plan::derive(
            &graph,
            claims.gadget_policy,
            prompt,
            0,
            &model_tables,
        )
        .unwrap()
    });
    let schedule = match &tv_runtime {
        Some(r) => r.leaf_schedule(),
        None => tv::ModelSchedule::derive(&graph, claims.gadget_policy),
    }
    .unwrap();
    let independent = match &tv_runtime {
        Some(r) => split::Layout::new(&r.split_sites().unwrap()).unwrap(),
        None => split::Layout::derive(&graph, claims.gadget_policy).unwrap(),
    };
    let tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            let spec = selection.tables.iter().find(|t| t.id == i.table).unwrap();
            assert!(spec.log2_rows <= maximum, "bounded CPU fixture");
            TableColumns::<G>::new(&GeneratedTable::from_spec(spec).unwrap()).unwrap()
        })
        .collect();
    let vt_tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            let bytes =
                tessera_codec::encode(selection.tables.iter().find(|t| t.id == i.table).unwrap());
            tv_tables::Table::from_spec(
                &tv_codec::decode(
                    &bytes,
                    &tv_codec::DecodeLimits::MESSAGE,
                    tv_tables::TableSpec::decode,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect();
    let native_plan = match &runtime {
        Some(r) => tessera_prover::native::ModelNativePlan::derive_runtime(
            &p.lowering,
            r,
            &p.claims,
            &layout,
        ),
        None => tessera_prover::native::ModelNativePlan::derive(
            &p.lowering,
            &p.selection,
            &p.claims,
            &layout,
        ),
    }
    .unwrap();
    // RFC-0056 'Working sets': the compiler derives the same native relations from the lowering
    // alone, a request's included, and so the same composed schedule, which its commitment report
    // sizes.
    {
        use tessera_compiler::schedule::Request;
        let request = match &runtime {
            None => Request::None,
            Some(r) if r.certificate().is_some() => Request::Generate,
            Some(_) => Request::Prefill,
        };
        let mirror = tessera_compiler::schedule::model_natives(&p.lowering, request).unwrap();
        assert_eq!(
            mirror.native.as_slice(),
            native_plan.native(),
            "{request:?}"
        );
        let composed = tessera_compiler::schedule::derive_composed_schedule(
            &p.lowering,
            &mirror.deferred,
            &mirror.native,
        )
        .unwrap();
        assert_eq!(&composed, native_plan.schedule());
    }
    let mut score_buffers = None;
    let score_planning = allocation_counter::measure(|| {
        score_buffers = Some(native_plan.score_input_memory(&p.lowering).unwrap());
    });
    assert_eq!(score_planning.count_total, 0);
    let score_buffers = score_buffers.unwrap();
    assert!(score_buffers.peak_bytes() >= score_buffers.retained_bytes());
    assert!(score_buffers.retained_bytes() > 0);
    let stage_plan = native_plan.schedule();
    let gadgets = tv_core::gadgets::Plan::derive(&graph, claims.gadget_policy).unwrap();
    let independent_native = match &tv_runtime {
        Some(r) => tv_core::gadgets::model::Plan::derive_runtime(r),
        None => tv_core::gadgets::model::Plan::derive(&gadgets),
    }
    .unwrap();
    model_stages::compare_schedules(&native_plan, &independent_native);
    let geometry = match prompt {
        Some(prompt) => tessera_prover::derive_runtime_schedule(&p.claims, n, prompt, 0),
        None => tessera_prover::derive_model_schedule(&p.claims, n),
    }
    .unwrap();
    assert_eq!(geometry.sizes.stages.len(), stage_plan.stages.len() - 1);
    for (a, b) in geometry.sizes.stages.iter().zip(&stage_plan.stages[1..]) {
        assert_eq!((a.rounds, a.zero_vars), (b.rounds, b.zero_vars));
    }
    let trees: Vec<_> = schedule
        .schedule()
        .trees()
        .iter()
        .map(|t| match *t {
            tv::Tree::Lookup { height, .. } | tv::Tree::Table { height, .. } => {
                tessera_protocol::LeafTree::Fraction { height }
            }
            tv::Tree::Forest { height, trees_log } => {
                tessera_protocol::LeafTree::Forest { height, trees_log }
            }
        })
        .collect();
    assert_eq!(geometry.sizes.trees, trees);
    assert_eq!(geometry.sizes.advice_vars, advice.layout().row_vars());
    assert_eq!(
        geometry.lookup_entries,
        layout
            .instances()
            .iter()
            .map(|i| (i.table, i.entries))
            .collect::<Vec<_>>()
    );
    for mode in [Mode::Live, Mode::NonInteractive] {
        tessera_protocol::TranscriptSchedule::expand(
            &geometry.sizes,
            &ParamSet::TEST_SMALL,
            mode,
            tessera_protocol::Protocol::One,
        )
        .unwrap();
    }
    let checkpoints = tessera_witness::Checkpoints::new(&engine, model, &tokens).unwrap();
    let replay = match &runtime {
        Some(r) => tessera_witness::Regenerator::runtime(&engine, r, model, &tokens, &checkpoints),
        None => {
            tessera_witness::Regenerator::new(&engine, &p.selection, model, &tokens, &checkpoints)
        }
    }
    .unwrap();
    let regenerated =
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &replay, &advice).unwrap();
    let committed =
        model_openings::Committed::new(&p, model, &advice, &source, &replay, &claims, prompt);
    for mode in [Mode::Live, Mode::NonInteractive] {
        let runtime_geometry = tv_runtime
            .as_ref()
            .map(|r| tv_core::runtime::Geometry::derive(&claims, r, None).unwrap());
        let (mut t, mut t2, xi, mut first) = committed.start(mode);
        let initial_transcript = first.replay.take().unwrap();
        let forests = actual.forests();
        let borrowed = actual.tensors();
        let lookup = prepare_leaf(&mut t, &layout, &tables, &borrowed, &forests).unwrap();
        let points: Vec<_> = stage_plan.stages[0]
            .claims
            .iter()
            .map(|c| lookup.point()[..c.point.len()].to_vec())
            .collect();
        let mut stages = tessera_prover::StageProver::new(&p.lowering, stage_plan, &witness);
        let graph_leaf = stages.leaf(&points).unwrap();
        let atoms = stages.leaf_split_values().unwrap();
        let proof = lookup.finish(&atoms).unwrap();
        if let Some(g) = &runtime_geometry {
            assert_eq!(g.roots(), proof.logup().roots().len());
            assert_eq!(g.values(), proof.logup().values().len());
            assert_eq!(g.leaf_splits(), proof.splits().len());
            assert_eq!(g.levels().len(), proof.logup().levels().len());
            for (f, p) in g.levels().iter().zip(proof.logup().levels()) {
                assert_eq!(f.rounds, p.proof().rounds.len());
                assert!(p.proof().rounds.iter().all(|r| r.len() == f.degree));
                assert_eq!(f.evaluations, p.proof().final_evals.len());
                assert_eq!(
                    f.field_count(),
                    Some(
                        p.proof().rounds.iter().map(Vec::len).sum::<usize>()
                            + p.proof().final_evals.len()
                    )
                );
            }
        }
        let levels: Vec<_> = proof
            .logup()
            .levels()
            .iter()
            .map(|l| tv_core::sumcheck::SumcheckProof {
                rounds: l.proof().rounds.iter().map(|r| es(r)).collect(),
                final_evals: es(&l.proof().final_evals),
            })
            .collect();
        let leaf = schedule
            .schedule()
            .verify(
                &mut t2,
                tv::Proof {
                    roots: &es(proof.logup().roots()),
                    levels: &levels,
                    values: &es(proof.logup().values()),
                },
                |tree, point| schedule.forest_root(tree, point),
                |i, beta, point| vt_tables[i].eval_mle(point, beta).ok(),
            )
            .unwrap();
        let splits = es(proof.splits());
        let row_zero = |id: tv_tables::TableId| {
            let index = layout.instances().iter().position(|i| i.table.0 == id.0)?;
            let table = &vt_tables[index];
            let (k, v) = table.row(0)?;
            Some((
                k,
                (!matches!(table.spec().kind, tv_tables::TableKind::Range { .. })).then_some(v),
            ))
        };
        let bound = independent
            .bind(&mut t2, leaf.clone(), &splits, atoms.len(), |id| {
                let index = layout.instances().iter().position(|i| i.table.0 == id.0)?;
                let table = &vt_tables[index];
                let (k, v) = table.row(0)?;
                Some((
                    k,
                    (!matches!(table.spec().kind, tv_tables::TableKind::Range { .. })).then_some(v),
                ))
            })
            .unwrap();
        for index in 0..layout.split_len() {
            let mut bad = splits.clone();
            bad[index] += E::ONE;
            assert_eq!(
                independent.bind(&mut vt(mode), leaf.clone(), &bad, atoms.len(), row_zero),
                Err(split::Error::Mismatch),
                "source {index}"
            );
        }
        let mut at = 0;
        for (s, site) in layout.sites().iter().enumerate() {
            for l in 0..site.layers.len() {
                for read in layout.reads(s, l).unwrap() {
                    let claim = &bound.columns[at];
                    at += 1;
                    assert_eq!(claim.node.0, owner(&p, site.node, l).0);
                    assert_eq!(claim.column, role(read.term));
                    assert_eq!(
                        claim.value,
                        mle(actual.values(s, l, read.term).unwrap(), &claim.point)
                    );
                }
            }
        }
        assert_eq!(at, bound.columns.len());
        assert_eq!(bound.forests.len(), actual.forests().len());
        for (forest, site) in bound.forests.iter().zip(actual.forest_sites()) {
            let table = forest_truth(&p, &witness, site);
            assert_eq!(forest.claim.value, mle(&table, &forest.claim.point));
        }
        let mut tr = initial_transcript;
        let other = prepare_leaf(
            &mut tr,
            &layout,
            &tables,
            &regenerated.tensors(),
            &regenerated.forests(),
        )
        .unwrap()
        .finish(&atoms)
        .unwrap();
        assert!(
            other.splits() == proof.splits(),
            "replayed leaf splits differ"
        );
        assert_eq!(other.logup().roots(), proof.logup().roots());
        assert_eq!(other.logup().values(), proof.logup().values());
        for (a, b) in other.logup().levels().iter().zip(proof.logup().levels()) {
            assert_eq!(a.proof(), b.proof());
        }

        model_stages::prove_stages(
            &p,
            model,
            &graph,
            &layout,
            &bound,
            &witness,
            &actual,
            &native_plan,
            &independent_native,
            &mut stages,
            &points,
            &graph_leaf,
            &proof,
            &mut t,
            t2,
            &committed,
            &xi,
            mode,
            runtime_geometry.as_ref(),
            first,
        );
    }
}

#[test]
fn complete_runtime_cpu_proofs_discharge_original_sources_in_both_contexts() {
    let mut coverage = [false; 2];
    for d in [2, 16] {
        let mut model = bounded_model();
        model.arch.vocab = 3;
        model.arch.d_model = d;
        model.arch.n_heads = d / 2;
        model.arch.d_ff = 3 * d;
        if d == 16 {
            model.quant.public.rsqrt_kappa = 0;
        }
        let maximum = if d == 16 { 20 } else { 18 };
        let shapes = model.arch.weight_shapes().unwrap();
        for weight in &mut model.weights {
            let shape = shapes[weight.ty as usize];
            weight.rows = shape.out;
            weight.cols = shape.inputs;
            weight.data.resize((shape.out * shape.inputs) as usize, 0);
        }
        for channel in &mut model.quant.channels {
            let n = shapes[channel.ty as usize].out as usize;
            if !channel.class.is_empty() {
                channel.class.resize(n, 0);
            }
            if let Some(&value) = channel.multiplier.first() {
                channel.multiplier.resize(n, value);
            }
        }
        let p = public(
            &model,
            5,
            GadgetPolicy {
                max_log2_rows: maximum,
                query_block: 2,
                ..GadgetPolicy::default()
            },
        );
        let runtime =
            tessera_compiler::RuntimePlan::derive(&p.lowering, &p.selection, &p.claims, 2, 0)
                .unwrap();
        let c = runtime.certificate().unwrap();
        let reused = (c.table.0 as usize) < p.selection.tables.len();
        coverage[usize::from(reused)] = true;
        if reused {
            let base = LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
            let entries = base
                .instances()
                .iter()
                .find(|i| i.table == c.table)
                .unwrap()
                .entries;
            assert!(
                !(entries + c.entries()).is_power_of_two(),
                "shared table needs a new combined tail"
            );
        }
        let requests: &[_] = if d == 16 {
            &[(5, 2)]
        } else {
            &[(1, 1), (5, 1), (5, 2), (5, 4), (5, 5)]
        };
        for &(n, prompt) in requests {
            check_complete_runtime(&model, n, maximum, Some(prompt));
        }
    }
    assert_eq!(
        coverage,
        [true, true],
        "both appended and reused runtime Range tables"
    );
    let enabled = tv_core::params::ParamSet::accept(
        &tv_core::claims::ParamSetId::new("test-small/1").unwrap(),
    )
    .is_some();
    assert_eq!(
        model_openings::NI_REPLAYS.load(std::sync::atomic::Ordering::SeqCst),
        if enabled { 100 } else { 0 }
    );
    assert_eq!(
        model_openings::LIVE_ACCEPTANCES.load(std::sync::atomic::Ordering::SeqCst),
        if enabled { 6 } else { 0 }
    );
}

// A real leaf/stage fixture kept narrow for allocation fault-injection controls.
#[test]
fn native_memory_counter_uses_real_leaf_claims_in_both_modes() {
    check_complete(&bounded_model_named("tiny-2-servable"), 3, 18);
}

#[test]
fn complete_cpu_leaf_and_mixed_softmax_graph_stages_preserve_all_sources_in_both_modes() {
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        for n in [1, 3, 5] {
            check_complete(&bounded_model_named(name), n, 18);
        }
    }
}

fn layered(mut model: IntegerModel) -> IntegerModel {
    use tessera_ir::{QuantPoint, WeightType};
    model.arch.n_layers = 3;
    fn extend<T: Clone>(values: &mut Vec<(QuantPoint, T)>) {
        let third: Vec<_> = values
            .iter()
            .filter_map(|(p, v)| match *p {
                QuantPoint::Layer { layer: 0, site } => {
                    Some((QuantPoint::Layer { layer: 2, site }, v.clone()))
                }
                _ => None,
            })
            .collect();
        values.extend(third);
        values.sort_by_key(|(p, _)| *p);
    }
    extend(&mut model.quant.public.act_exponents);
    extend(&mut model.quant.public.shift_class_sets);
    extend(&mut model.quant.public.affine_sites);
    model
        .quant
        .public
        .score_scales
        .push((2, model.quant.public.score_scales[0].1));
    model
        .quant
        .silu_in_exponents
        .push((2, model.quant.silu_in_exponents[0].1));
    let mut weights = model
        .weights
        .iter()
        .filter(|w| w.layer == 0 && !matches!(w.ty, WeightType::Embed | WeightType::LmHead))
        .cloned()
        .collect::<Vec<_>>();
    for w in &mut weights {
        w.layer = 2;
    }
    model.weights.extend(weights);
    model.weights.sort_by_key(|w| (w.ty, w.layer));
    let mut channels = model
        .quant
        .channels
        .iter()
        .filter(|c| c.layer == 0 && !matches!(c.ty, WeightType::Embed | WeightType::LmHead))
        .cloned()
        .collect::<Vec<_>>();
    for c in &mut channels {
        c.layer = 2;
    }
    model.quant.channels.extend(channels);
    model.quant.channels.sort_by_key(|c| (c.ty, c.layer));
    model
}
#[test]
fn six_heads_three_kv_heads_and_three_layers_preserve_all_canonical_positions() {
    let mut model = layered(bounded_model());
    model.arch.n_heads = 6;
    model.arch.n_kv_heads = 3;
    model.arch.d_model = 12;
    let shapes = model.arch.weight_shapes().unwrap();
    for w in &mut model.weights {
        let shape = shapes[w.ty as usize];
        w.rows = shape.out;
        w.cols = shape.inputs;
        w.data = (0..shape.out * shape.inputs)
            .map(|i| (i % 5) as i8 - 2)
            .collect();
    }
    for c in &mut model.quant.channels {
        let n = shapes[c.ty as usize].out as usize;
        if let Some(&v) = c.class.first() {
            c.class.resize(n, v);
        }
        if let Some(&v) = c.multiplier.first() {
            c.multiplier.resize(n, v);
        }
    }
    check_complete(&model, 5, 21);
}

/// A coordinate dictionary, independent of the production scatter's flat-index arithmetic.
fn coordinates(ranges: &[std::ops::Range<u32>]) -> Vec<Vec<u32>> {
    let mut rows = vec![vec![]];
    for range in ranges {
        rows = rows
            .into_iter()
            .flat_map(|row| {
                range.clone().map(move |v| {
                    let mut next = row.clone();
                    next.push(v);
                    next
                })
            })
            .collect();
    }
    rows
}
fn vertex(shape: &[u32], row: usize) -> Vec<u32> {
    let mut rest = row;
    let mut result = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        let size = shape[i].next_power_of_two() as usize;
        result[i] = (rest % size) as u32;
        rest /= size;
    }
    assert_eq!(rest, 0);
    result
}
fn flat(shape: &[u32], coordinate: &[u32]) -> usize {
    shape
        .iter()
        .zip(coordinate)
        .fold(0usize, |a, (&d, &x)| a * d as usize + x as usize)
}
fn source_truth(
    p: &Public,
    witness: &tessera_witness::Witness,
    source: &impl WitnessSource,
    advice: &ModelAdviceLayout<'_>,
    layout: &LeafLayout<ModelColumn>,
    actual: &ModelWitness,
) {
    use std::collections::BTreeMap;
    use tessera_ir::{Op, TensorId};
    let mut aux = BTreeMap::<(u32, AuxColumn, Vec<u32>), i64>::new();
    for tile in advice.tiles() {
        let node = match tile.tile {
            tessera_compiler::Tile::Node { node }
            | tessera_compiler::Tile::Attention { node, .. } => node,
            _ => continue,
        };
        let Some(site) = layout
            .sites()
            .iter()
            .find(|s| (0..s.layers.len()).any(|l| owner(p, s.node, l).0 == node))
        else {
            continue;
        };
        let _ = site;
        let values = source.tile(tile).unwrap();
        let definition = &p.lowering.graph.nodes[node as usize];
        let output = if matches!(definition.op, Op::SoftmaxRow { .. }) {
            definition.outputs[1]
        } else {
            definition.outputs[0]
        };
        let rows = coordinates(
            &tile
                .shares
                .iter()
                .find(|s| s.tensor == output)
                .unwrap()
                .ranges,
        );
        for (role, values) in values.gadget() {
            assert_eq!(rows.len(), values.len());
            for (row, &v) in rows.iter().zip(values.iter()) {
                assert!(aux.insert((node, *role, row.clone()), v).is_none());
            }
        }
    }
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            let node = &p.lowering.graph.nodes[owner(p, site.node, l).0 as usize];
            for read in layout.reads(s, l).unwrap() {
                let request = site.layers[l]
                    .lookups
                    .iter()
                    .find(|r| {
                        std::iter::once(&r.lookup.key)
                            .chain(&r.lookup.value)
                            .any(|f| f.terms.iter().any(|(_, c)| *c == read.term))
                    })
                    .unwrap();
                let domain = read
                    .axes
                    .iter()
                    .map(|&a| request.domain[a])
                    .collect::<Vec<_>>();
                let column = actual.values(s, l, read.term).unwrap();
                for (row, &value) in column.iter().enumerate() {
                    let coordinate = vertex(&domain, row);
                    let real = coordinate.iter().zip(&domain).all(|(&c, &d)| c < d);
                    let expected = if !real {
                        if read.term == ModelColumn::ExpKey {
                            (1i64
                                << p.selection
                                    .tables
                                    .iter()
                                    .find(|t| t.id == request.lookup.table)
                                    .unwrap()
                                    .log2_rows)
                                - 1
                        } else {
                            0
                        }
                    } else {
                        let index = flat(&domain, &coordinate);
                        match read.term {
                            ModelColumn::GreedyKey | ModelColumn::GreedyKeyOrdinal(_) => {
                                panic!("model-only witness has no runtime key")
                            }
                            ModelColumn::Input(i) | ModelColumn::Output(i) => {
                                let t: TensorId = if matches!(read.term, ModelColumn::Input(_)) {
                                    node.inputs[i as usize]
                                } else {
                                    node.outputs[i as usize]
                                };
                                witness.tensor(t).unwrap()[index]
                            }
                            ModelColumn::Auxiliary(role) => aux[&(node.id.0, role, coordinate)],
                            ModelColumn::RootInput => {
                                let Op::RsqrtRow { eps_int, .. } = node.op else {
                                    panic!("root")
                                };
                                witness.tensor(node.inputs[0]).unwrap()[index] + eps_int as i64
                            }
                            ModelColumn::ExpKey => {
                                let &[_, n, _] = domain.as_slice() else {
                                    panic!("scores")
                                };
                                let [h, q, k] = coordinate[..] else {
                                    panic!("coordinate")
                                };
                                if k > q {
                                    (1i64
                                        << p.selection
                                            .tables
                                            .iter()
                                            .find(|t| t.id == request.lookup.table)
                                            .unwrap()
                                            .log2_rows)
                                        - 1
                                } else {
                                    let scores = witness.tensor(node.inputs[0]).unwrap();
                                    let base = ((h * n + q) * n) as usize;
                                    *scores[base..=base + q as usize].iter().max().unwrap()
                                        - scores[base + k as usize]
                                }
                            }
                            ModelColumn::Slack(which) => {
                                let Op::SoftmaxRow { sigma, .. } = node.op else {
                                    panic!("softmax")
                                };
                                let n = domain[1] as usize;
                                let e = witness.tensor(node.outputs[0]).unwrap();
                                let sum: i64 = e[index * n..(index + 1) * n].iter().sum();
                                let a = (1i64 << (2 * sigma))
                                    - witness.tensor(node.outputs[1]).unwrap()[index] * sum;
                                if which == Slack::A { a } else { sum - 1 - a }
                            }
                        }
                    };
                    assert_eq!(
                        value, expected,
                        "node {:?}, source {:?}, vertex {row}",
                        node.id, read.term
                    );
                }
            }
        }
    }
}

fn forest_truth(p: &Public, witness: &tessera_witness::Witness, site: NodeRef) -> Vec<i64> {
    let layers = if matches!(site, NodeRef::Position(_)) {
        p.layers.count
    } else {
        1
    };
    let node = &p.lowering.graph.nodes[owner(p, site, 0).0 as usize];
    let shape = &p.lowering.graph.tensors[node.inputs[0].0 as usize].shape;
    let &[heads, n, _] = shape.as_slice() else {
        panic!("score shape")
    };
    let (h, q) = (
        heads.next_power_of_two() as usize,
        n.next_power_of_two() as usize,
    );
    let mut leaves = vec![1; layers.next_power_of_two() as usize * h * q * q];
    for l in 0..layers {
        let node = &p.lowering.graph.nodes[owner(p, site, l as usize).0 as usize];
        let scores = witness.tensor(node.inputs[0]).unwrap();
        for head in 0..heads as usize {
            for query in 0..n as usize {
                let row = &scores[(head * n as usize + query) * n as usize
                    ..(head * n as usize + query + 1) * n as usize];
                let max = *row[..=query].iter().max().unwrap();
                for key in 0..=query {
                    leaves[((l as usize * h + head) * q + query) * q + key] = max - row[key];
                }
            }
        }
    }
    leaves
}

#[test]
fn per_request_domains_determine_blocks_and_duplicate_sources_require_one_point() {
    use tessera_gadgets::{Affine, Lookup};
    use tessera_prover::leaf::{LeafError, Site, SiteGadget, SiteLookup};
    let model = bounded_model();
    let p = public(
        &model,
        3,
        GadgetPolicy {
            max_log2_rows: 18,
            ..GadgetPolicy::default()
        },
    );
    let request = |column: u8, domain: Vec<u32>, axes: Vec<usize>| SiteLookup {
        domain,
        axes: vec![(column, axes)],
        token: None,
        lookup: Lookup {
            table: tessera_tables::TableId(0),
            key: Affine {
                constant: G::ZERO,
                terms: vec![(G::ONE, column)],
            },
            value: None,
        },
    };
    let gadget = SiteGadget {
        lookups: vec![
            request(0, vec![2, 4], vec![0, 1]),
            request(1, vec![2], vec![0]),
            request(2, vec![], vec![]),
        ],
    };
    let site = Site {
        node: NodeRef::Position(0),
        layers: vec![gadget; p.layers.count as usize],
    };
    let layout = LeafLayout::new(&p.layers, vec![site.clone()]).unwrap();
    assert_eq!(
        layout.instances()[0].entries,
        11 * u64::from(p.layers.count)
    );
    assert_eq!(layout.split_len(), 3 * p.layers.count as usize);
    assert_eq!(
        layout.instances()[0]
            .blocks
            .iter()
            .map(|b| b.vars)
            .collect::<Vec<_>>(),
        [3, 3, 1, 1, 0, 0]
    );
    assert_eq!(
        layout
            .reads(0, 0)
            .unwrap()
            .iter()
            .map(|r| r.coords.clone())
            .collect::<Vec<_>>(),
        [vec![0, 1, 2], vec![0], vec![]]
    );
    let mut wrong = site.clone();
    wrong.layers[1].lookups.swap(0, 1);
    assert!(matches!(
        LeafLayout::new(&p.layers, vec![wrong]),
        Err(LeafError::Internal {
            rule: "a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')"
        })
    ));
    let mut wrong = site.clone();
    for layer in &mut wrong.layers {
        layer.lookups = vec![
            request(0, vec![2, 4], vec![0]),
            request(0, vec![4, 2], vec![0]),
        ];
    }
    assert!(matches!(
        LeafLayout::new(&p.layers, vec![wrong]),
        Err(LeafError::Internal {
            rule: "a gadget's lookups read each tensor over one set of axes (RFC-0045 'One claim per input')"
        })
    ));
    let mut zero = site.clone();
    for layer in &mut zero.layers {
        layer.lookups = vec![request(0, vec![0], vec![0])];
    }
    assert!(matches!(
        LeafLayout::new(&p.layers, vec![zero]),
        Err(LeafError::Internal {
            rule: "a domain's axes each hold a position"
        })
    ));
    for domain in [vec![0], vec![1 << 16; 4]] {
        let mut wrong = site.clone();
        for layer in &mut wrong.layers {
            layer.lookups[0].domain = domain.clone();
        }
        assert!(matches!(
            LeafLayout::new(&p.layers, vec![wrong]),
            Err(LeafError::Internal { .. })
        ));
    }
}

#[test]
fn missing_or_misshaped_graph_sources_and_tile_errors_are_explicit_and_redacted() {
    struct Wrong<'a> {
        witness: &'a tessera_witness::Witness,
        id: tessera_ir::TensorId,
        missing: bool,
    }
    impl tessera_prover::WitnessSource for Wrong<'_> {
        fn values(&self, id: tessera_ir::TensorId) -> Option<&[i64]> {
            let values = &self.witness.tensor(id)?[..];
            if id == self.id {
                if self.missing {
                    None
                } else {
                    Some(&values[..values.len() - 1])
                }
            } else {
                Some(values)
            }
        }
    }
    struct NoTiles;
    impl WitnessSource for NoTiles {
        fn tile(
            &self,
            _: &tessera_compiler::SequenceTile,
        ) -> Result<tessera_witness::TileWitness, tessera_witness::WitnessError> {
            Err(tessera_witness::WitnessError::Memory {
                tensor: tessera_ir::TensorId(0),
            })
        }
    }
    let model = bounded_model();
    let p = public(
        &model,
        3,
        GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    let layout = LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
    let engine = Engine::new(&p.lowering).unwrap();
    let tokens = Secret::new(vec![1, 2, 3]);
    let witness = engine.run(&model, &tokens).unwrap();
    let tiles = Resident::new(&engine, &p.selection, &model, &witness).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let node = &p.lowering.graph.nodes[owner(&p, layout.sites()[0].node, 0).0 as usize];
    let id = node.outputs[0];
    for missing in [true, false] {
        let wrong = Wrong {
            witness: &witness,
            id,
            missing,
        };
        assert!(matches!(
            ModelWitness::new(&layout, &p.lowering, &p.layers, &wrong, &tiles, &advice),
            Err(tessera_prover::LeafError::Internal { .. })
        ));
    }
    let error = ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &NoTiles, &advice)
        .unwrap_err();
    assert_eq!(
        error,
        tessera_prover::LeafError::Witness(tessera_witness::WitnessError::Memory {
            tensor: tessera_ir::TensorId(0)
        })
    );
    assert_eq!(error.code(), tessera_codec::error::ErrorCode::ResMemory);
    let mut sites = layout.sites().to_vec();
    for layer in &mut sites[0].layers {
        for lookup in &mut layer.lookups {
            for (_, column) in &mut lookup.lookup.key.terms {
                if matches!(column, ModelColumn::Auxiliary(_)) {
                    *column = ModelColumn::Auxiliary(AuxColumn::Maximum);
                }
            }
        }
    }
    let wrong = LeafLayout::new(&p.layers, sites).unwrap();
    assert!(matches!(
        ModelWitness::new(&wrong, &p.lowering, &p.layers, &witness, &tiles, &advice),
        Err(tessera_prover::LeafError::Internal {
            rule: "the canonical tile contains every requested auxiliary role"
        })
    ));
}

#[test]
fn inconsistent_catalogs_uncovered_sources_and_wrong_tile_tables_are_refused() {
    use tessera_gadgets::{Affine, Lookup};
    use tessera_prover::leaf::{Site, SiteGadget, SiteLookup};
    let model = bounded_model();
    let p = public(
        &model,
        3,
        GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    let mut wrong = p.claims.clone();
    wrong.tables.clear();
    assert!(matches!(
        LeafLayout::model(&p.lowering, &p.layers, &p.selection, &wrong),
        Err(tessera_prover::LeafError::Internal {
            rule: "model leaf sources and requests follow the canonical lowering and selection"
        })
    ));
    let layout = LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
    let engine = Engine::new(&p.lowering).unwrap();
    let tokens = Secret::new(vec![1, 2, 3]);
    let witness = engine.run(&model, &tokens).unwrap();
    let source = Resident::new(&engine, &p.selection, &model, &witness).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let tiled = advice
        .tiles()
        .iter()
        .filter_map(|t| match t.tile {
            tessera_compiler::Tile::Node { node }
            | tessera_compiler::Tile::Attention { node, .. } => Some(node),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let position = p
        .lowering
        .graph
        .nodes
        .iter()
        .find_map(|n| match p.layers.position(n.id) {
            Some((0, pos)) if !tiled.contains(&n.id.0) => Some(pos),
            _ => None,
        })
        .unwrap();
    let gadget = SiteGadget {
        lookups: vec![SiteLookup {
            domain: vec![1],
            axes: vec![],
            token: None,
            lookup: Lookup {
                table: p.selection.tables[0].id,
                key: Affine {
                    constant: G::ZERO,
                    terms: vec![(G::ONE, ModelColumn::Auxiliary(AuxColumn::Maximum))],
                },
                value: None,
            },
        }],
    };
    let uncovered = LeafLayout::new(
        &p.layers,
        vec![Site {
            node: NodeRef::Position(position),
            layers: vec![gadget; p.layers.count as usize],
        }],
    )
    .unwrap();
    assert!(matches!(
        ModelWitness::new(
            &uncovered,
            &p.lowering,
            &p.layers,
            &witness,
            &source,
            &advice
        ),
        Err(tessera_prover::LeafError::Internal {
            rule: "model leaf columns cover every canonical source at its public coordinates"
        })
    ));
    struct WrongTile<'a, S> {
        source: &'a S,
        target: u32,
        other: &'a tessera_compiler::SequenceTile,
    }
    impl<S: WitnessSource> WitnessSource for WrongTile<'_, S> {
        fn tile(
            &self,
            t: &tessera_compiler::SequenceTile,
        ) -> Result<tessera_witness::TileWitness, tessera_witness::WitnessError> {
            if matches!(t.tile,tessera_compiler::Tile::Node {node} if node==self.target) {
                self.source.tile(self.other)
            } else {
                self.source.tile(t)
            }
        }
    }
    let target = layout
        .sites()
        .iter()
        .enumerate()
        .find(|(s, site)| {
            site.layers[0].lookups.len() == 1
                && layout
                    .reads(*s, 0)
                    .unwrap()
                    .iter()
                    .all(|r| matches!(r.term, ModelColumn::Input(_) | ModelColumn::Output(_)))
        })
        .unwrap()
        .1;
    let target = owner(&p, target.node, 0);
    let other_node = p
        .lowering
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, tessera_ir::Op::TableMap { .. }))
        .unwrap()
        .id;
    let other = advice
        .tiles()
        .iter()
        .find(|t| matches!(t.tile,tessera_compiler::Tile::Node {node} if node==other_node.0))
        .unwrap();
    let wrong = WrongTile {
        source: &source,
        target: target.0,
        other,
    };
    assert!(matches!(
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &wrong, &advice),
        Err(tessera_prover::LeafError::Internal {
            rule: "model leaf columns cover every canonical source at its public coordinates"
        })
    ));
}

#[test]
fn runtime_leaf_counts_and_implicit_table_digests_match_independent_metadata() {
    let data: serde_json::Value =
        serde_json::from_slice(&fs::read(vectors().join("greedy-runtime/runtime.json")).unwrap())
            .unwrap();
    let mut cache: Vec<(String, Public)> = Vec::new();
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        // Model rescale variants are covered by the full compiler/verifier geometry oracle.
        if !case["variant"].is_null() {
            continue;
        }
        let key = format!(
            "{}/{}/{}/{}/{}",
            case["file"], case["vocab"], case["custom"], case["n"], case["query_block"]
        );
        let index = cache
            .iter()
            .position(|(k, _)| k == &key)
            .unwrap_or_else(|| {
                let file = case["file"].as_str().unwrap();
                let mut m = model(file.strip_suffix(".d").unwrap());
                if let Some(v) = case["vocab"].as_u64() {
                    m.arch.vocab = v as u32;
                }
                if let Some(c) = case["custom"].as_object() {
                    let u = |k| c[k].as_u64().unwrap() as u32;
                    m.arch.d_model = u("d_model");
                    m.arch.n_heads = u("n_heads");
                    m.arch.n_kv_heads = u("n_kv_heads");
                    m.arch.d_head = u("d_head");
                    m.arch.d_ff = u("d_ff");
                    m.quant.public.rsqrt_kappa = u("rsqrt_kappa") as u8;
                    m.quant.public.exp_sigma = u("exp_sigma") as u8;
                    m.quant.public.residual_bits = u("residual_bits") as u8;
                    m.quant.public.residual_exponent =
                        c["residual_exponent"].as_i64().unwrap() as i8;
                }
                let policy = GadgetPolicy {
                    query_block: case["query_block"].as_u64().unwrap() as u32,
                    ..GadgetPolicy::default()
                };
                cache.push((
                    key.clone(),
                    public(&m, case["n"].as_u64().unwrap() as u32, policy),
                ));
                cache.len() - 1
            });
        let p = &cache[index].1;
        let runtime = tessera_compiler::RuntimePlan::derive(
            &p.lowering,
            &p.selection,
            &p.claims,
            case["prompt"].as_u64().unwrap() as u32,
            case["ordinal"].as_u64().unwrap() as u32,
        )
        .unwrap();
        let layout = LeafLayout::runtime(&p.lowering, &p.layers, &runtime, &p.claims).unwrap();
        assert_eq!(
            serde_json::json!(
                layout
                    .instances()
                    .iter()
                    .map(|i| (i.table.0, i.entries))
                    .collect::<Vec<_>>()
            ),
            case["lookups"]
        );
        let (graph, claims) = verifier(p, case["n"].as_u64().unwrap() as u32);
        let base = tv_core::selection::derive(&graph, claims.gadget_policy).unwrap();
        let tables = base
            .tables
            .iter()
            .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
            .collect::<Vec<_>>();
        let independent = tv_core::gadgets::runtime::Plan::derive(
            &graph,
            claims.gadget_policy,
            case["prompt"].as_u64().unwrap() as u32,
            case["ordinal"].as_u64().unwrap() as u32,
            &tables,
        )
        .unwrap();
        let other = split::Layout::new(&independent.split_sites().unwrap()).unwrap();
        let native = tessera_prover::native::ModelNativePlan::derive_runtime(
            &p.lowering,
            &runtime,
            &p.claims,
            &layout,
        )
        .unwrap();
        let checked_native = tv_core::gadgets::model::Plan::derive_runtime(&independent).unwrap();
        model_stages::compare_schedules(&native, &checked_native);
        assert_eq!(layout.split_len(), other.values());
        assert_eq!(
            layout
                .instances()
                .iter()
                .map(|i| (i.table.0, i.entries))
                .collect::<Vec<_>>(),
            other.instances().map(|(t, n)| (t.0, n)).collect::<Vec<_>>()
        );
        if let Some(c) = runtime.certificate() {
            let spec = &runtime.selection().tables[c.table.0 as usize];
            assert_eq!(
                spec.digest,
                independent
                    .table(tv_tables::TableId(c.table.0))
                    .unwrap()
                    .spec()
                    .digest
            );
            let last = layout.sites().last().unwrap();
            assert_eq!(
                last.node,
                NodeRef::Node(p.lowering.graph.producer(p.lowering.logits).unwrap())
            );
            let [read] = layout.reads(layout.sites().len() - 1, 0).unwrap() else {
                panic!("one virtual key")
            };
            assert_eq!(read.term, ModelColumn::GreedyKey);
            assert_eq!(read.coords.len(), c.variables as usize);
        }
        checked += 1;
    }
    assert_eq!(checked, 101);
}

#[test]
fn retained_public_layout_memory_matches_owned_allocations() {
    use tessera_compiler::RuntimePlan;
    use tessera_prover::{leaf::GroupedLayout, native::ModelNativePlan};
    fn retained<T>(build: impl FnOnce() -> T) -> (T, u64) {
        let mut held = None;
        let observed = allocation_counter::measure(|| {
            held = Some(build());
        });
        (
            held.unwrap(),
            u64::try_from(observed.bytes_current).unwrap(),
        )
    }
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        let model = bounded_model_named(name);
        for n in [1, 3, 5] {
            let p = public(
                &model,
                n,
                GadgetPolicy {
                    max_log2_rows: 18,
                    query_block: 2,
                    ..GadgetPolicy::default()
                },
            );
            let (lowering, lowering_bytes) = retained(|| p.lowering.clone());
            let (selection, selection_bytes) = retained(|| p.selection.clone());
            let (runtime, runtime_bytes) = retained(|| {
                RuntimePlan::derive(&p.lowering, &p.selection, &p.claims, 1, 0).unwrap()
            });
            let (advice, advice_bytes) =
                retained(|| ModelAdviceLayout::runtime(&p.lowering, &runtime, &p.claims).unwrap());
            let (leaf, leaf_bytes) = retained(|| {
                LeafLayout::runtime(&p.lowering, &p.layers, &runtime, &p.claims).unwrap()
            });
            let (native, native_bytes) = retained(|| {
                ModelNativePlan::derive_runtime(&p.lowering, &runtime, &p.claims, &leaf).unwrap()
            });
            let sizing = allocation_counter::measure(|| {
                assert_eq!(
                    lowering.heap_bytes(),
                    Some(lowering_bytes),
                    "lowering {name}/{n}"
                );
                assert!(
                    selection.heap_upper_bound().unwrap() >= selection_bytes,
                    "selection {name}/{n}"
                );
                assert!(
                    runtime.heap_upper_bound().unwrap() >= runtime_bytes,
                    "runtime {name}/{n}"
                );
                assert!(
                    advice.heap_upper_bound().unwrap() >= advice_bytes,
                    "advice {name}/{n}"
                );
                assert_eq!(leaf.heap_bytes(), Some(leaf_bytes), "leaf {name}/{n}");
                assert_eq!(native.heap_bytes(), Some(native_bytes), "native {name}/{n}");
            });
            assert_eq!(sizing.count_total, 0);
            check_geometry_constructor_memory(&p, &runtime, &leaf, &native, None, &advice);
            if n > 1 {
                let (grouped, bytes) = retained(|| {
                    GroupedLayout::new(
                        &leaf,
                        &p.layers,
                        tessera_protocol::LeafGroups {
                            token_bits: 1,
                            layer_bits: 0,
                        },
                    )
                    .unwrap()
                });
                assert_eq!(grouped.heap_bytes(), Some(bytes), "grouped {name}/{n}");
                check_geometry_constructor_memory(
                    &p,
                    &runtime,
                    &leaf,
                    &native,
                    Some(&grouped),
                    &advice,
                );
                for h in 0..grouped.count() {
                    let (group, bytes) = retained(|| grouped.group(h).unwrap());
                    let sizing = allocation_counter::measure(|| {
                        assert_eq!(group.heap_bytes(), Some(bytes), "group {name}/{n}/{h}");
                    });
                    assert_eq!(sizing.count_total, 0);
                }
            }
        }
    }

    // RFC-0071: optional cloned slot lengths and multiple runtime windows.
    let model = model("tiny-2-minimal");
    let mut p = public(&model, 3, GadgetPolicy::default());
    p.lowering = tessera_compiler::lower_batch(PublicModel::of(&model), &[2, 1, 3]).unwrap();
    p.layers = LayerMap::new(&p.lowering).unwrap();
    let ranges = analyze(&p.lowering.graph, &p.lowering.tables).unwrap();
    p.selection = select_model(
        p.claims.protocol_version,
        &p.lowering,
        &ranges,
        &p.claims.gadget_policy,
    )
    .unwrap();
    let (lowering, bytes) = retained(|| p.lowering.clone());
    assert_eq!(lowering.heap_bytes(), Some(bytes));
    let (runtime, bytes) = retained(|| {
        RuntimePlan::derive_batch(
            &p.lowering,
            &p.selection,
            &p.claims,
            &[(1, 1), (1, 0), (1, 2)],
        )
        .unwrap()
    });
    assert!(runtime.heap_upper_bound().unwrap() >= bytes);
    let (advice, bytes) =
        retained(|| ModelAdviceLayout::runtime(&p.lowering, &runtime, &p.claims).unwrap());
    assert!(advice.heap_upper_bound().unwrap() >= bytes);
    let (leaf, bytes) =
        retained(|| LeafLayout::runtime(&p.lowering, &p.layers, &runtime, &p.claims).unwrap());
    assert_eq!(leaf.heap_bytes(), Some(bytes));
    let (native, bytes) = retained(|| {
        ModelNativePlan::derive_runtime(&p.lowering, &runtime, &p.claims, &leaf).unwrap()
    });
    assert_eq!(native.heap_bytes(), Some(bytes));
    check_geometry_constructor_memory(&p, &runtime, &leaf, &native, None, &advice);
}

fn check_geometry_constructor_memory(
    p: &Public,
    runtime: &tessera_compiler::RuntimePlan,
    leaf: &LeafLayout<ModelColumn>,
    native: &tessera_prover::native::ModelNativePlan,
    grouped: Option<&tessera_prover::leaf::GroupedLayout>,
    advice: &ModelAdviceLayout<'_>,
) {
    let weights = tessera_pcs::WeightLayout::new(
        p.claims.layout.weights,
        tessera_pcs::WeightShapes {
            layers: p.claims.arch.n_layers,
            shapes: p.claims.arch.weight_shapes().unwrap(),
            channels: p
                .lowering
                .classes
                .iter()
                .chain(&p.lowering.multipliers)
                .map(|(w, _)| (w.ty.matrix(), w.layer))
                .collect(),
        },
    )
    .unwrap();
    let mut held = None;
    let measured = allocation_counter::measure(|| {
        held = Some(
            tessera_prover::OpeningGeometry::derive(
                &p.lowering,
                &runtime.selection().tables,
                leaf,
                grouped,
                native,
                advice,
                &weights,
            )
            .unwrap(),
        );
    });
    let geometry = held.unwrap();
    let mut bound = None;
    let sizing = allocation_counter::measure(|| {
        bound = Some(
            geometry
                .construction_peak_bytes(&p.lowering, leaf, grouped, native, advice)
                .unwrap(),
        );
    });
    assert_eq!(sizing.count_total, 0);
    assert!(
        measured.bytes_max <= bound.unwrap(),
        "geometry constructor {measured:?}, bound {bound:?}"
    );
}

#[test]
fn public_preparation_bounds_cover_actual_constructors() {
    use tessera_compiler::RuntimePlan;
    fn peak<T>(build: impl FnOnce() -> T) -> (T, u64, u64) {
        let mut held = None;
        let observed = allocation_counter::measure(|| held = Some(build()));
        (
            held.unwrap(),
            observed.bytes_max,
            u64::try_from(observed.bytes_current).unwrap(),
        )
    }
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        let model = bounded_model_named(name);
        for n in [1, 3, 5] {
            let policy = GadgetPolicy {
                max_log2_rows: 18,
                query_block: 2,
                ..GadgetPolicy::default()
            };
            let (lowering, lower_peak, _) =
                peak(|| tessera_compiler::lower(PublicModel::of(&model), n).unwrap());
            let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
            let (layers, layers_peak, _) = peak(|| LayerMap::new(&lowering).unwrap());
            let (selection, selection_peak, selection_retained) = peak(|| {
                select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy).unwrap()
            });
            let p = public(&model, n, policy);
            let cores = Engine::table_memory(&lowering.tables).unwrap().peak_bytes();
            let bound = lowering
                .construction_metadata_bytes(PublicModel::of(&model))
                .unwrap()
                + cores;
            assert!(
                lower_peak <= bound,
                "{name}/{n}: lower {lower_peak} > {bound}"
            );
            let bound = layers.construction_peak_bytes(&lowering.graph).unwrap();
            assert!(
                layers_peak <= bound,
                "{name}/{n}: layers {layers_peak} > {bound}"
            );
            let metadata = selection
                .construction_metadata_bytes(&lowering, &layers)
                .unwrap();
            assert!(
                selection_retained <= metadata,
                "{name}/{n}: selection retained {selection_retained} > {metadata}"
            );
            let cores = Engine::table_memory(&selection.tables)
                .unwrap()
                .peak_bytes();
            let bound = selection
                .construction_metadata_bytes(&lowering, &layers)
                .unwrap()
                + cores;
            assert!(
                selection_peak <= bound,
                "{name}/{n}: selection {selection_peak} > {bound}"
            );
            for prompt in [1, n] {
                let (runtime, runtime_peak, _) = peak(|| {
                    RuntimePlan::derive(&lowering, &selection, &p.claims, prompt, 0).unwrap()
                });
                let (leaf, leaf_peak, leaf_retained) =
                    peak(|| LeafLayout::runtime(&lowering, &layers, &runtime, &p.claims).unwrap());
                let (advice, advice_peak, _) =
                    peak(|| ModelAdviceLayout::runtime(&lowering, &runtime, &p.claims).unwrap());
                let mut advice_bound = None;
                let advice_sizing = allocation_counter::measure(|| {
                    advice_bound =
                        advice.construction_peak_bytes(runtime.selection(), &p.claims, &layers);
                });
                assert_eq!(advice_sizing.bytes_total, 0);
                assert!(
                    advice_peak <= advice_bound.unwrap(),
                    "{name}/{n}/{prompt}: advice {advice_peak} > {advice_bound:?}"
                );
                let (native, native_peak, _) = peak(|| {
                    tessera_prover::native::ModelNativePlan::derive_runtime(
                        &lowering, &runtime, &p.claims, &leaf,
                    )
                    .unwrap()
                });
                let mut native_bound = None;
                let native_sizing = allocation_counter::measure(|| {
                    native_bound = native.construction_peak_bytes(&lowering);
                });
                assert_eq!(native_sizing.bytes_total, 0);
                assert!(
                    native_peak <= native_bound.unwrap(),
                    "{name}/{n}/{prompt}: native {native_peak} > {native_bound:?}"
                );
                let descriptors = tessera_compiler::schedule::model_natives(
                    &lowering,
                    if prompt == n {
                        tessera_compiler::schedule::Request::Prefill
                    } else {
                        tessera_compiler::schedule::Request::Generate
                    },
                )
                .unwrap();
                let (schedule, schedule_peak, _) = peak(|| {
                    tessera_compiler::schedule::derive_composed_schedule(
                        &lowering,
                        &descriptors.deferred,
                        &descriptors.native,
                    )
                    .unwrap()
                });
                let mut schedule_bound = None;
                let schedule_sizing = allocation_counter::measure(|| {
                    schedule_bound = schedule.construction_peak_bytes(&lowering.graph)
                });
                assert_eq!(schedule_sizing.bytes_total, 0);
                assert!(
                    schedule_peak <= schedule_bound.unwrap(),
                    "{name}/{n}/{prompt}: schedule {schedule_peak} > {schedule_bound:?}"
                );
                let mut leaf_bound = None;
                let leaf_sizing = allocation_counter::measure(|| {
                    leaf_bound = Some(
                        leaf.construction_peak_bytes(&lowering, &layers, runtime.selection())
                            .unwrap(),
                    );
                });
                assert_eq!(leaf_sizing.bytes_total, 0);
                assert!(
                    leaf_peak <= leaf_bound.unwrap(),
                    "{name}/{n}/{prompt}: leaf {leaf_peak} > {leaf_bound:?}"
                );
                assert!(leaf_retained <= leaf_bound.unwrap());
                let mut bounds = None;
                let sizing = allocation_counter::measure(|| {
                    bounds = Some((
                        lowering
                            .construction_metadata_bytes(PublicModel::of(&model))
                            .unwrap(),
                        layers.construction_peak_bytes(&lowering.graph).unwrap(),
                        selection
                            .construction_metadata_bytes(&lowering, &layers)
                            .unwrap(),
                        runtime.construction_peak_bytes(&lowering).unwrap(),
                    ));
                });
                assert_eq!(sizing.bytes_total, 0);
                let bound = bounds.unwrap().3;
                assert!(
                    runtime_peak <= bound,
                    "{name}/{n}/{prompt}: runtime {runtime_peak} > {bound}"
                );
            }
        }
    }
}

#[test]
fn batch_runtime_preparation_bound_includes_growing_request_columns() {
    use tessera_compiler::RuntimePlan;
    let model = bounded_model_named("tiny-2-servable");
    let p = public(
        &model,
        3,
        GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    for count in [1, 2, 5, 9, 17] {
        for outputs in [false, true] {
            let requests = (0..count)
                .map(|i| {
                    if outputs && i % 2 == 0 {
                        (1, 2)
                    } else {
                        (3, 0)
                    }
                })
                .collect::<Vec<_>>();
            let lowering =
                tessera_compiler::lower_batch(PublicModel::of(&model), &vec![3; count]).unwrap();
            let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
            let selection = select_model(
                p.claims.protocol_version,
                &lowering,
                &ranges,
                &p.claims.gadget_policy,
            )
            .unwrap();
            assert_eq!(selection.tables, p.claims.tables);
            let mut runtime = None;
            let actual = allocation_counter::measure(|| {
                runtime = Some(
                    RuntimePlan::derive_batch(&lowering, &selection, &p.claims, &requests).unwrap(),
                );
            });
            let runtime = runtime.unwrap();
            let layers = LayerMap::new(&lowering).unwrap();
            let mut advice = None;
            let actual_advice = allocation_counter::measure(|| {
                advice = Some(ModelAdviceLayout::runtime(&lowering, &runtime, &p.claims).unwrap());
            });
            let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, &p.claims).unwrap();
            let mut native = None;
            let actual_native = allocation_counter::measure(|| {
                native = Some(
                    tessera_prover::native::ModelNativePlan::derive_runtime(
                        &lowering, &runtime, &p.claims, &leaf,
                    )
                    .unwrap(),
                );
            });
            let bound_native = native
                .as_ref()
                .unwrap()
                .construction_peak_bytes(&lowering)
                .unwrap();
            assert!(
                actual_native.bytes_max <= bound_native,
                "{count}/{outputs}: native {actual_native:?} > {bound_native}"
            );
            let bound_advice = advice
                .as_ref()
                .unwrap()
                .construction_peak_bytes(runtime.selection(), &p.claims, &layers)
                .unwrap();
            assert!(
                actual_advice.bytes_max <= bound_advice,
                "{count}/{outputs}: advice {actual_advice:?} > {bound_advice}"
            );

            let mut bound = None;
            let sizing =
                allocation_counter::measure(|| bound = runtime.construction_peak_bytes(&lowering));
            assert_eq!(sizing.bytes_total, 0);
            assert!(
                actual.bytes_max <= bound.unwrap(),
                "{count}/{outputs}: {actual:?} {bound:?}"
            );
        }
    }
}

#[test]
fn transcript_bound_covers_staging_growth_and_pow_labels() {
    use tessera_codec::{Encode, Sink};
    struct Pieces<'a>(&'a [u8]);
    impl Encode for Pieces<'_> {
        fn encode_to<S: Sink + ?Sized>(&self, sink: &mut S) {
            for _ in 0..(3 * 65536 / self.0.len() + 3) {
                sink.put(self.0);
            }
        }
    }
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0; 32],
        session_id: [0; 16],
    };
    for width in [1, 3, 7, 31, 4095, 8191, 65535, 65536, 70001] {
        let bytes = vec![0xA5; width];
        let mut transcript = Transcript::new(&context);
        let observed = allocation_counter::measure(|| transcript.absorb("round2", &Pieces(&bytes)));
        assert!(
            observed.bytes_max <= Transcript::work_memory_bytes(9).unwrap(),
            "{width}: {observed:?}"
        );
    }
    for len in [0, 9, 262145] {
        let label = "p".repeat(len);
        let mut transcript = Transcript::new(&context);
        let observed = allocation_counter::measure(|| assert_eq!(transcript.grind(&label, 0), 0));
        assert!(
            observed.bytes_max <= Transcript::work_memory_bytes(len as u64).unwrap(),
            "{len}: {observed:?}"
        );
    }
    assert_eq!(Transcript::work_memory_bytes(u64::MAX), None);
    assert_eq!(Transcript::work_memory_bytes(isize::MAX as u64), None);
}

#[test]
fn streamed_envelope_bound_covers_large_small_write_payloads() {
    use tessera_codec::{Encode, Sink};
    struct Pieces<'a>(&'a [u8]);
    impl Encode for Pieces<'_> {
        fn encode_to<S: Sink + ?Sized>(&self, sink: &mut S) {
            for _ in 0..(3 * 65536 / self.0.len() + 3) {
                sink.put(self.0);
            }
        }
    }
    for width in [1, 7, 31, 4095, 8191, 65535, 65536, 70001] {
        let bytes = vec![0xA5; width];
        let observed = allocation_counter::measure(|| {
            envelope::write(&mut std::io::sink(), &schema::NI_PROOF, &Pieces(&bytes)).unwrap();
        });
        assert!(
            observed.bytes_max <= envelope::write_memory_bytes(),
            "{width}: {observed:?}"
        );
    }
}
