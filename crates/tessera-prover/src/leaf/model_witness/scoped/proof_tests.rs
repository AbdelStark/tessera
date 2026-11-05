// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact grouped-stage transcript/encoding controls for the canonical source factory.

use super::*;
use crate::leaf::grouped::prove_grouped_leaf_source_fixture;
use crate::leaf::{
    GroupedClaims, LeafClaims, MergeMode, grouped_multiplicities, prove_grouped_leaf,
};
use tessera_codec::{Ascii, DecodeLimits, Encode, Secret, envelope, schema};
use tessera_compiler::{PublicModel, analyze, claim_list, select_model};
use tessera_gadgets::TableColumns;
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{GadgetPolicy, IntegerModel, QuantPoint};
use tessera_protocol::{LeafGroups, ParamSet};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::GeneratedTable;
use tessera_witness::{Engine, Resident};

mod failures;
mod runtime_slots;

type F = GoldilocksExt2;

struct Recording {
    inner: Transcript,
    events: Vec<(String, Vec<F>)>,
}
impl Recording {
    fn new() -> Self {
        Self {
            inner: Transcript::new(&SessionContext {
                protocol: Ascii::new("tessera-protocol/1").unwrap(),
                mode: Mode::NonInteractive,
                params: ParamSetId(Ascii::new("test-small/1").unwrap()),
                statement_digest: [71; 32],
                session_id: [130; 16],
            }),
            events: Vec::new(),
        }
    }
}
impl SumcheckTranscript<F> for Recording {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        self.events.push((label.to_owned(), message.to_vec()));
        self.inner.absorb_message(label, message);
    }
    fn challenge(&mut self, label: &str) -> F {
        let c = self.inner.challenge(label);
        self.events.push((label.to_owned(), vec![c]));
        c
    }
}

// Same bounded architecture and public scales as the existing model_groups oracle fixture.
// These fixture changes are not a real-model export or a fidelity result.
pub(crate) fn bounded_model() -> IntegerModel {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/models/tiny-2-servable.d");
    let mut model: IntegerModel = envelope::open(
        &std::fs::read(path).unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
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
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    let shifts = model.quant.public.shift_class_sets.clone();
    for (point, values) in &mut model.quant.public.shift_class_sets {
        if let QuantPoint::Layer { site, .. } = *point {
            *values = shifts
                .iter()
                .find(|(p, _)| *p == QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1
                .clone();
        }
    }
    let affine = model.quant.public.affine_sites.clone();
    for (point, value) in &mut model.quant.public.affine_sites {
        if let QuantPoint::Layer { site, .. } = *point {
            *value = affine
                .iter()
                .find(|(p, _)| *p == QuantPoint::Layer { layer: 0, site })
                .unwrap()
                .1;
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
        if !channel.class.is_empty() {
            channel.class.resize(n, 0);
            // Classed fixture sites have several classes; distinguish their real layers.
            channel.class.fill((channel.layer % 2) as u8);
        }
        if let Some(&v) = channel.multiplier.first() {
            channel.multiplier.resize(n, v);
        }
    }
    model
}

#[allow(clippy::too_many_arguments)]
fn compare_native_stages(
    factory: &ModelSourceFactory<'_>,
    full: &ModelWitness,
    selection: &tessera_compiler::Selection,
    public_claims: &tessera_ir::ClaimList,
    witness: &tessera_witness::Witness,
    reference: &crate::leaf::ProvenGroupedLeaf,
    actual: &crate::leaf::ProvenGroupedLeaf,
    reference_tr: &mut Recording,
    actual_tr: &mut Recording,
    literal: bool,
) {
    use crate::native::{ModelNativePlan, ModelNativeProver};
    use tessera_compiler::schedule::{NodeRef, Site};
    let lowering = factory.lowering;
    let plan = ModelNativePlan::derive(lowering, selection, public_claims, factory.layout).unwrap();
    let reference_claims = GroupedClaims::new(reference, factory.grouped);
    let actual_claims = GroupedClaims::new(actual, factory.grouped);
    let axes = crate::leaf::tokens(lowering).unwrap();
    let points = |claims: &dyn LeafClaims| {
        plan.schedule()
            .sites
            .iter()
            .map(|site| {
                let tensor = match *site {
                    Site::Gadget { node, input } => {
                        let id = match node {
                            NodeRef::Node(id) => id,
                            NodeRef::Position(p) => factory.layers.node(p, 0),
                        };
                        lowering.graph.nodes[id.0 as usize].inputs[input as usize]
                    }
                    Site::Logits => lowering.logits,
                };
                claims
                    .tensor_point(
                        &lowering.graph.tensors[tensor.0 as usize].shape,
                        crate::leaf::token(&axes, tensor),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let mut reference_stage = crate::StageProver::new(lowering, plan.schedule(), witness);
    let mut actual_stage = crate::StageProver::new(lowering, plan.schedule(), witness);
    let _reference_leaf = reference_stage.leaf(&points(&reference_claims)).unwrap();
    let _actual_leaf = actual_stage.leaf(&points(&actual_claims)).unwrap();
    let reference_atoms = reference_stage.leaf_split_values().unwrap();
    let actual_atoms = actual_stage.leaf_split_values().unwrap();
    assert_eq!(reference_atoms, actual_atoms);
    reference_tr.absorb_message("stage/0/split", &reference_atoms);
    actual_tr.absorb_message("stage/0/split", &actual_atoms);
    let mut reference_native = if literal {
        ModelNativeProver::new_literal(&plan, lowering, full, witness, &reference_claims)
    } else {
        ModelNativeProver::new(&plan, lowering, full, witness, &reference_claims)
    }
    .unwrap();
    let mut actual_native =
        ModelNativeProver::new_scoped(&plan, lowering, factory, witness, &actual_claims, literal)
            .unwrap();
    for stage in 1..plan.schedule().stages.len() {
        let reference = reference_stage
            .prove_stage_composed(reference_tr, stage as u16, &mut reference_native)
            .unwrap();
        let actual = actual_stage
            .prove_stage_composed(actual_tr, stage as u16, &mut actual_native)
            .unwrap();
        assert_eq!(
            actual, reference,
            "native/graph stage {stage}, literal {literal}"
        );
    }
    assert_eq!(actual_tr.events, reference_tr.events);
    assert_eq!(actual_stage.pool(), reference_stage.pool());
    assert_eq!(plan.schedule().stages.len() - 1, 2);
}

#[test]
fn scoped_group_driver_matches_resident_transcript_and_encoded_messages() {
    let model = bounded_model();
    let lowering = tessera_compiler::lower(PublicModel::of(&model), 3).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let policy = GadgetPolicy {
        max_log2_rows: 21,
        query_block: 2,
        ..GadgetPolicy::default()
    };
    let selection =
        select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy).unwrap();
    let p = ParamSet::TEST_SMALL;
    let claims = claim_list(
        PublicModel::of(&model),
        &selection,
        &policy,
        p.layout,
        ParamSetId(Ascii::new(p.id).unwrap()),
    )
    .unwrap();
    let layers = LayerMap::new(&lowering).unwrap();
    let stage = LeafLayout::model(&lowering, &layers, &selection, &claims).unwrap();
    let advice = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
    let engine = Engine::new(&lowering).unwrap();
    let witness = engine.run(&model, &Secret::new(vec![1, 2, 3])).unwrap();
    let source = Resident::new(&engine, &selection, &model, &witness).unwrap();
    let full = ModelWitness::new(&stage, &lowering, &layers, &witness, &source, &advice).unwrap();
    assert!(
        full.columns.iter().any(|layers| layers.len() > 1
            && layers[0]
                .iter()
                .zip(&layers[1])
                .any(|(a, b)| a.name == b.name && a.values[..] != b.values[..])),
        "the fixture must distinguish real layer sources"
    );
    assert!(
        full.forests
            .iter()
            .any(|f| f.leaves.contains(&Goldilocks::ZERO) && f.leaves.contains(&Goldilocks::ONE)),
        "the fixture must distinguish real and padded/causal forest leaves"
    );
    let tables: Vec<_> = stage
        .instances()
        .iter()
        .map(|i| {
            let spec = selection.tables.iter().find(|t| t.id == i.table).unwrap();
            TableColumns::<Goldilocks>::new(&GeneratedTable::from_spec(spec).unwrap()).unwrap()
        })
        .collect();
    let tensors = full.tensors();
    let forests = full.stage_forests();
    let mut variants = 0;
    for (gt, gl) in [(0, 1), (1, 0), (1, 1)] {
        let grouped = GroupedLayout::new(
            &stage,
            &layers,
            LeafGroups {
                token_bits: gt,
                layer_bits: gl,
            },
        )
        .unwrap();
        let factory = ModelSourceFactory {
            layout: &stage,
            grouped: &grouped,
            lowering: &lowering,
            layers: &layers,
            witness: &witness,
            source: &source,
            advice: &advice,
        };
        let mut source_bill = None;
        let planned = allocation_counter::measure(|| {
            source_bill = Some(
                ScopedSourceMemory::new(&stage, &lowering, &layers, &grouped, &advice).unwrap(),
            );
        });
        let source_bill = source_bill.unwrap();
        assert_eq!(planned.bytes_current, 0);
        assert!(
            planned.bytes_max <= source_bill.planning,
            "public source planning {planned:?}, {source_bill:?}"
        );
        let callback = engine.source_tile_work_memory().unwrap();
        for h in 0..grouped.count() {
            let bill =
                memory::ScopedGroupMemory::new(&stage, &lowering, &layers, &grouped, h).unwrap();
            let mut input = None;
            let measured = allocation_counter::measure(|| {
                input = Some(
                    factory
                        .group_inputs(h, |id| {
                            tables
                                .iter()
                                .find(|t| t.table() == id)
                                .map(|t| t.tuples())
                                .ok_or(LeafError::Internal {
                                    rule: "fixture table exists",
                                })
                        })
                        .unwrap(),
                );
            });
            assert!(
                measured.bytes_current as u64 <= bill.retained,
                "group {h}: retained {measured:?}, {bill:?}"
            );
            assert!(
                measured.bytes_max <= bill.construction + callback,
                "group {h}: construction {measured:?}, {bill:?}, callback {callback}"
            );
            let input = input.unwrap();
            let forest_polynomials: Vec<_> = factory
                .forest_shapes()
                .unwrap()
                .iter()
                .map(|shape| tessera_sumcheck::MergePolynomial {
                    vars: shape.vars as usize,
                    group_vars: (shape.queries..shape.queries + u32::from(gt))
                        .chain(shape.layers..shape.layers + u32::from(gl))
                        .map(|v| v as usize)
                        .collect(),
                })
                .collect();
            let copied = allocation_counter::measure(|| {
                let mut blocks =
                    Vec::with_capacity(input.layout.reads().len() + input.shares.len());
                for read in input.layout.reads() {
                    blocks.push((
                        read.polynomial,
                        read.group_bits,
                        input
                            .block(
                                read.polynomial,
                                grouped.polynomials().len(),
                                &grouped.polynomials()[read.polynomial],
                                read.group_bits,
                            )
                            .unwrap(),
                    ));
                }
                for (f, polynomial) in forest_polynomials.iter().enumerate() {
                    blocks.push((
                        grouped.polynomials().len() + f,
                        h as u64,
                        input
                            .block(
                                grouped.polynomials().len() + f,
                                grouped.polynomials().len(),
                                polynomial,
                                h as u64,
                            )
                            .unwrap(),
                    ));
                }
            });
            assert!(
                copied.bytes_max <= bill.blocks,
                "group {h}: copies {copied:?}, {bill:?}"
            );
        }
        if gt == 0 && gl == 1 {
            let original_bill =
                memory::original_peak(&stage, &lowering, &layers, &grouped).unwrap();
            let count = grouped.polynomials().len() + factory.forest_shapes().unwrap().len();
            for i in 0..count {
                let measured = allocation_counter::measure(|| {
                    let values = factory.original(i).unwrap();
                    let _copy = crate::leaf::grouped::seal_copy(&values).unwrap();
                });
                assert!(
                    measured.bytes_max <= original_bill + callback,
                    "original {i}: {measured:?}, bill {original_bill}, callback {callback}"
                );
            }
            let native_bill = memory::native_peak(&stage, &lowering, &layers, &advice).unwrap();
            for (s, site) in stage.sites().iter().enumerate() {
                for l in 0..site.layers.len() {
                    for read in stage.reads(s, l).unwrap() {
                        let measured = allocation_counter::measure(|| {
                            let _lease = factory.column(s, l, read.term).unwrap();
                        });
                        assert!(
                            measured.bytes_max <= native_bill + callback,
                            "native {s}/{l}: {measured:?}, bill {native_bill}, callback {callback}"
                        );
                    }
                }
            }
        }
        let input = factory
            .group_inputs(0, |id| {
                tables
                    .iter()
                    .find(|t| t.table() == id)
                    .map(|t| t.tuples())
                    .ok_or(LeafError::Internal {
                        rule: "fixture table exists",
                    })
            })
            .unwrap();
        let (slot, read) = input
            .layout
            .reads()
            .iter()
            .enumerate()
            .find(|(_, r)| {
                let p = &grouped.polynomials()[r.polynomial];
                p.vars > p.group_vars.len()
            })
            .unwrap();
        let mut wrong = read.clone();
        wrong.polynomial = grouped.polynomials().len();
        assert!(matches!(
            input.split(slot, &wrong, &[], &[]),
            Err(LeafError::Internal {
                rule: "a group split keeps its complete source identity"
            })
        ));
        assert!(matches!(
            input.split(slot, read, &[], &[]),
            Err(LeafError::Internal {
                rule: "a scoped tensor's point has its free variables"
            })
        ));
        assert!(matches!(
            input.block(
                read.polynomial,
                grouped.polynomials().len(),
                &grouped.polynomials()[read.polynomial],
                read.group_bits ^ 1
            ),
            Err(LeafError::Internal {
                rule: "a scoped merge block has its polynomial and fixed coordinates"
            })
        ));
        drop(input);
        let counts = grouped_multiplicities(&stage, &grouped, &tables, &tensors).unwrap();
        let counts: Vec<_> = counts.iter().map(|v| &v[..]).collect();
        let mut reference_tr = Recording::new();
        let reference = prove_grouped_leaf(
            &mut reference_tr,
            &stage,
            &grouped,
            &tables,
            &tensors,
            &counts,
            &forests,
            MergeMode::InMemory,
        )
        .unwrap();
        let mut expected = Vec::new();
        reference.to_wire(&[]).encode_to(&mut expected);
        let mut polynomials = grouped.polynomials().to_vec();
        for forest in factory.forest_shapes().unwrap() {
            polynomials.push(tessera_sumcheck::MergePolynomial {
                vars: forest.vars as usize,
                group_vars: (forest.queries..forest.queries + u32::from(gt))
                    .chain(forest.layers..forest.layers + u32::from(gl))
                    .map(|v| v as usize)
                    .collect(),
            });
        }
        let cuts = tessera_sumcheck::largest_table_rounds(&polynomials);
        for mode in core::iter::once(MergeMode::InMemory)
            .chain((0..=cuts).map(|cut| MergeMode::TwoPass { j_star: Some(cut) }))
        {
            let mut actual_tr = Recording::new();
            let actual =
                prove_grouped_leaf_source_fixture(&mut actual_tr, &factory, &tables, &counts, mode)
                    .unwrap();
            let mut bytes = Vec::new();
            actual.to_wire(&[]).encode_to(&mut bytes);
            assert_eq!(
                actual_tr.events, reference_tr.events,
                "grouping ({gt},{gl}) mode {mode:?}"
            );
            assert_eq!(bytes, expected, "grouping ({gt},{gl}) mode {mode:?}");
            assert_eq!(actual.merged(), reference.merged());
            if gt == 1
                && gl == 1
                && matches!(
                    mode,
                    MergeMode::TwoPass {
                        j_star: Some(0 | 1)
                    }
                )
            {
                let mut reference_native_tr = Recording::new();
                let reference_native_leaf = prove_grouped_leaf(
                    &mut reference_native_tr,
                    &stage,
                    &grouped,
                    &tables,
                    &tensors,
                    &counts,
                    &forests,
                    MergeMode::InMemory,
                )
                .unwrap();
                assert_eq!(actual_tr.events, reference_native_tr.events);
                compare_native_stages(
                    &factory,
                    &full,
                    &selection,
                    &claims,
                    &witness,
                    &reference_native_leaf,
                    &actual,
                    &mut reference_native_tr,
                    &mut actual_tr,
                    matches!(mode, MergeMode::TwoPass { j_star: Some(1) }),
                );
            }
            variants += 1;
        }
    }
    assert_eq!(variants, 17);
}
