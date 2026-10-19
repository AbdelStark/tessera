// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedule (RFC-0010 'Stages', RFC-0037) derived from every graph of
//! `spec/vectors/lowering/`: the four conformance models and the two variants at every length.
//! Each property is checked against what the test computes from the graph itself, by its own
//! reading of the RFCs: the stage count of each template (RFC-0010 'Stages', RFC-0025 'Stages'),
//! the leaf stage's claims (RFC-0037 'The layer point'), each instance's variables and degree
//! (RFC-0010 'Relations'), and the invariants INV-SCH-2 to INV-SCH-5. `schedule_vectors.rs`
//! compares the schedules with the prover's, as `spec/vectors/schedule/` records them.

mod claim_support;

use std::collections::BTreeMap;

use claim_support::{
    at, goldens, integer, items, list, make_affine, make_pot, member, string, text, with_layers,
};
use tv_codec::{DecodeLimits, Rational};
use tv_core::claims::{ClaimList, Profile, Site, WeightType};
use tv_core::lowering::{Broadcast, Graph, Kind, NodeId, Op, TensorId, lower};
use tv_core::schedule::{
    Class, Commitment, Layers, Part, RelationKind, Route, Run, Schedule, Source, TensorRef, derive,
};
use tv_core::validation::check_public;
use tv_testkit::{Json, read_vectors};

/// One graph of the lowering vectors and its schedule.
struct Case {
    name: String,
    model: String,
    n: u32,
    profile: Profile,
    forms: Vec<String>,
    graph: Graph,
    schedule: Schedule,
}

fn number<T: TryFrom<i128>>(value: &Json) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(integer(value)).expect("in range")
}

fn site_named(name: &str) -> Site {
    match name {
        "AttnNorm" => Site::AttnNorm,
        "RopeQ" => Site::RopeQ,
        "RopeK" => Site::RopeK,
        "V" => Site::V,
        "AttnOut" => Site::AttnOut,
        "AttnResidual" => Site::AttnResidual,
        "MlpNorm" => Site::MlpNorm,
        "Silu" => Site::Silu,
        "Up" => Site::Up,
        "Product" => Site::Product,
        "MlpResidual" => Site::MlpResidual,
        other => panic!("unknown site {other}"),
    }
}

/// The golden claim list of a model's artifact, with a variant's edits made to its public part.
fn claims_for(case: &Json) -> ClaimList {
    let file = text(case, "file");
    let golden = goldens()
        .into_iter()
        .find(|g| g.model == file)
        .unwrap_or_else(|| panic!("no golden claim list for {file}"));
    let mut claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
    if let Some(variant) = case.get("variant") {
        if let Some(heads) = variant.get("n_kv_heads") {
            claims.arch.n_kv_heads = number(heads);
        }
        for edit in variant.get("rescale").map(items).unwrap_or(&[]) {
            let [layer, site, mode, value] = items(edit) else {
                panic!("a rescale is [layer, site, mode, value]");
            };
            let point = at(number(layer), site_named(string(site)));
            match string(mode) {
                "pot" => {
                    let shifts: Vec<i8> = items(value).iter().map(number).collect();
                    make_pot(&mut claims, point, &shifts);
                }
                "affine" => make_affine(&mut claims, point, number(value)),
                other => panic!("unknown mode {other}"),
            }
        }
        for edit in variant.get("score_scales").map(items).unwrap_or(&[]) {
            let [layer, num, den] = items(edit) else {
                panic!("a score scale is [layer, num, den]");
            };
            let layer: u32 = number(layer);
            let entry = claims
                .quant
                .score_scales
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .unwrap();
            entry.1 = Rational::new(number(num), number(den)).unwrap();
        }
    }
    claims
}

fn cases() -> Vec<Case> {
    let file = read_vectors("lowering/lowering.json");
    list(&file, "cases")
        .iter()
        .map(|case| {
            let n: u32 = number(member(case, "n"));
            let model = text(case, "model").to_owned();
            let name = format!("{model} at n = {n}");
            let claims = claims_for(case);
            let public = check_public(&claims).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let graph = lower(&public, n).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let schedule = derive(&graph).unwrap_or_else(|e| panic!("{name}: {e}"));
            let forms = list(case, "forms")
                .iter()
                .map(|f| string(f).to_owned())
                .collect();
            Case {
                name,
                model,
                n,
                profile: claims.profile,
                forms,
                graph,
                schedule,
            }
        })
        .collect()
}

/// `⌈log2 d⌉`.
fn vars(d: u32) -> usize {
    (u32::BITS - (d - 1).leading_zeros()) as usize
}

fn shape_vars(graph: &Graph, t: TensorId) -> usize {
    graph
        .tensor(t)
        .unwrap()
        .shape
        .dims()
        .iter()
        .map(|&d| vars(d))
        .sum()
}

fn dims(graph: &Graph, t: TensorId) -> Vec<u32> {
    graph.tensor(t).unwrap().shape.dims().to_vec()
}

fn op(graph: &Graph, node: NodeId) -> Op {
    *graph.node(node).unwrap().op()
}

fn is_gadget(op: &Op) -> bool {
    matches!(
        op,
        Op::OneHot { .. }
            | Op::Requant { .. }
            | Op::RsqrtRow { .. }
            | Op::SoftmaxRow { .. }
            | Op::TableMap { .. }
    )
}

/// RFC-0010 'Stages' and RFC-0025 'Stages': `servable` has three stages, `minimal` four, and a
/// `servable` model with an affine RoPE site four; the count is the same at every length.
#[test]
fn every_graph_has_the_stage_count_of_its_template() {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for case in cases() {
        let affine_rope = case
            .forms
            .iter()
            .any(|f| f == "affine RopeQ" || f == "affine RopeK");
        let expected = match case.profile {
            Profile::Servable if !affine_rope => 3,
            Profile::Servable | Profile::Minimal => 4,
        };
        assert_eq!(case.schedule.stage_count(), expected, "{}", case.name);
        // INV-SCH-4: the same count at every length.
        let previous = counts.insert(case.model.clone(), case.schedule.stage_count());
        assert!(
            previous.is_none_or(|count| count == expected),
            "{}",
            case.name
        );
        for (k, stage) in case.schedule.stages().iter().enumerate() {
            assert_eq!(usize::from(stage.index), k + 1, "{}", case.name);
            assert!(
                !stage.instances.is_empty(),
                "{}: stage {}",
                case.name,
                k + 1
            );
        }
    }
    // Two and four layers of `servable` without an affine RoPE site: three stages each.
    assert_eq!(counts["tiny-2-servable"], counts["tiny-4-servable"]);
}

/// RFC-0037 'The layer point': one claim on each input of each gadget, sites in node order, a
/// template position once, then the logits; a template position's claims hold one value per
/// layer, at a point of the input's variables.
#[test]
fn the_leaf_stage_claims_every_gadget_input_and_the_logits() {
    for case in cases() {
        let graph = &case.graph;
        let checkpoints = graph.checkpoints();
        let producer = |c: TensorId| graph.tensor(c).unwrap().producer.unwrap().0;
        let layer0 = producer(checkpoints[0]) + 1..=producer(checkpoints[1]);
        let epilogue = producer(*checkpoints.last().unwrap()) + 1;
        let mut expected = Vec::new();
        for (i, node) in graph.nodes().iter().enumerate() {
            let id = i as u32;
            let inside = id > producer(checkpoints[0]) && id < epilogue;
            if !is_gadget(node.op()) || (inside && !layer0.contains(&id)) {
                continue;
            }
            for (input, &t) in node.inputs().iter().enumerate() {
                expected.push((NodeId(id), Some(input), inside, shape_vars(graph, t)));
            }
        }
        let lm_head = graph
            .nodes()
            .iter()
            .position(|n| {
                matches!(n.op(), Op::MatMulWeight { weight } if weight.ty == WeightType::LmHead)
            })
            .unwrap();
        let logits = graph.nodes()[lm_head].outputs()[0];
        expected.push((
            NodeId(lm_head as u32),
            None,
            false,
            shape_vars(graph, logits),
        ));
        let found: Vec<_> = case
            .schedule
            .leaf()
            .iter()
            .map(|c| (c.site, c.input, c.stacked, c.claim.vars))
            .collect();
        assert_eq!(found, expected, "{}", case.name);
        // A template position's claims are stacked at the layer point ρ_1 in stage 1; the others
        // have no layer index, or read the last layer's output (c_L).
        let rho = [Run {
            source: Source::Layer,
            start: 0,
            len: case.schedule.layer_vars(),
        }];
        for (c, claim) in case.schedule.leaf().iter().enumerate() {
            for atom in &claim.claim.atoms {
                let ok = match (&atom.layers, claim.stacked) {
                    (Layers::Point(p), true) => p.runs() == rho,
                    (Layers::None | Layers::One(_), false) => true,
                    _ => false,
                };
                assert!(ok, "{}: leaf claim {c}: {:?}", case.name, atom.layers);
                let own = Run {
                    source: Source::Leaf(c as u32),
                    start: 0,
                    len: claim.claim.vars,
                };
                assert!(
                    atom.point.runs().iter().all(|r| r.source == own.source),
                    "{}: leaf claim {c}",
                    case.name
                );
            }
        }
    }
}

/// RFC-0010 'Relations' and 'Batching within a stage': each instance's variables and degree, from
/// its node's shapes; a stage's rounds and degree are its instances' largest.
#[test]
fn instances_have_the_variables_and_degrees_of_their_relations() {
    for case in cases() {
        let graph = &case.graph;
        let lambda = case.schedule.layer_vars();
        for stage in case.schedule.stages() {
            let mut most = 0;
            let mut degree = 0;
            for (i, instance) in stage.instances.iter().enumerate() {
                let node = graph.node(instance.node).unwrap();
                let layers = if instance.stacked { lambda } else { 0 };
                let (own, base, evals) = match *node.op() {
                    Op::MatMulWeight { .. } => {
                        let inner = *dims(graph, node.inputs()[1]).last().unwrap();
                        (vars(inner), if layers > 0 { 3 } else { 2 }, 2)
                    }
                    Op::MatMulAct { .. } => {
                        let a = dims(graph, node.inputs()[0]);
                        let lead: usize = a[..a.len() - 2].iter().map(|&d| vars(d)).sum();
                        let k = vars(a[a.len() - 1]);
                        (k + lead, if lead + layers > 0 { 3 } else { 2 }, 2)
                    }
                    Op::Mul { .. } => (shape_vars(graph, node.outputs()[0]), 3, 2),
                    Op::SumSquares { .. } => (shape_vars(graph, node.inputs()[0]), 3, 1),
                    Op::Rotate { .. } => (shape_vars(graph, node.inputs()[0]), 2, 1),
                    Op::ConstMul { .. } => {
                        assert_eq!(instance.relation.kind(), RelationKind::ZeroCheck);
                        (shape_vars(graph, node.outputs()[0]), 2, 2)
                    }
                    other => panic!("{}: an instance of {other:?}", case.name),
                };
                let what = format!("{}: stage {} instance {i}", case.name, stage.index);
                assert_eq!(instance.vars, own + layers, "{what}");
                assert_eq!(instance.degree, base, "{what}");
                assert_eq!(instance.evals.len(), evals, "{what}");
                most = most.max(instance.vars);
                degree = degree.max(instance.degree);
            }
            assert_eq!(stage.rounds, most, "{}: stage {}", case.name, stage.index);
            assert_eq!(stage.degree, degree, "{}: stage {}", case.name, stage.index);
            let zero = stage
                .instances
                .iter()
                .filter(|i| i.relation.kind() == RelationKind::ZeroCheck)
                .map(|i| i.vars)
                .max()
                .unwrap_or(0);
            assert_eq!(stage.zero_vars, zero, "{}", case.name);
            let expected_layer_vars = if stage.index == 1 { lambda } else { 0 };
            assert_eq!(stage.layer_vars, expected_layer_vars, "{}", case.name);
            // In the conformance models every reduction stage is of degree 3.
            assert_eq!(stage.degree, 3, "{}: stage {}", case.name, stage.index);
        }
    }
}

/// Every claim of a stage, in the order its split values and its next instances take them: the
/// leaf stage's, then each stage's evaluations.
fn claims_by_stage(schedule: &Schedule) -> Vec<Vec<&tv_core::schedule::Claim>> {
    let mut out = vec![schedule.leaf().iter().map(|c| &c.claim).collect::<Vec<_>>()];
    for stage in schedule.stages() {
        out.push(
            stage
                .instances
                .iter()
                .flat_map(|i| i.evals.iter())
                .collect(),
        );
    }
    out
}

/// INV-SCH-2: every virtual atom a stage leaves is exactly one instance of the next stage, in
/// order, the atom's tensor and point its claim's, with stage 1's zero-checks after them.
#[test]
fn every_virtual_atom_is_one_instance_of_the_next_stage() {
    for case in cases() {
        let by_stage = claims_by_stage(&case.schedule);
        for (k, claims) in by_stage.iter().enumerate() {
            let mut virtual_atoms = Vec::new();
            for claim in claims {
                match &claim.route {
                    Route::Instance(i) => virtual_atoms.push((*i, &claim.atoms[0])),
                    Route::Pool => {
                        assert!(
                            claim
                                .atoms
                                .iter()
                                .all(|a| matches!(a.class, Class::Committed(_)))
                        );
                    }
                    Route::Split(parts) => {
                        for part in parts {
                            if let Part::Virtual { atom, instance } = part {
                                virtual_atoms.push((*instance, &claim.atoms[*atom]));
                            }
                        }
                    }
                }
            }
            let next = case.schedule.stages().get(k);
            let Some(next) = next else {
                assert!(
                    virtual_atoms.is_empty(),
                    "{}: after the last stage",
                    case.name
                );
                continue;
            };
            for (position, (instance, atom)) in virtual_atoms.iter().enumerate() {
                assert_eq!(*instance, position, "{}: stage {}", case.name, k + 1);
                let target = &next.instances[*instance];
                assert_eq!(target.tensor, atom.tensor, "{}", case.name);
                assert_eq!(target.point, atom.point, "{}", case.name);
                assert_eq!(target.layers, atom.layers, "{}", case.name);
            }
            let zero_checks = next
                .instances
                .iter()
                .filter(|i| i.relation.kind() == RelationKind::ZeroCheck)
                .count();
            assert_eq!(
                next.instances.len(),
                virtual_atoms.len() + zero_checks,
                "{}: stage {}",
                case.name,
                k + 1
            );
            if k > 0 {
                assert_eq!(zero_checks, 0, "{}: zero-checks run in stage 1", case.name);
            }
        }
    }
}

/// INV-SCH-3 and INV-SCH-5: every committed atom reaches the pool in exactly one claim, of one
/// commitment at one point; and every weight type the proof opens has exactly one claim there.
#[test]
fn every_committed_atom_reaches_the_pool_once_and_each_weight_type_once() {
    for case in cases() {
        let mut weight_claims: BTreeMap<String, usize> = BTreeMap::new();
        for claims in claims_by_stage(&case.schedule) {
            for claim in claims {
                let groups: Vec<Vec<usize>> = match &claim.route {
                    Route::Instance(_) => Vec::new(),
                    Route::Pool => vec![(0..claim.atoms.len()).collect()],
                    Route::Split(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            Part::Group { atoms } => Some(atoms.clone()),
                            Part::Virtual { .. } => None,
                        })
                        .collect(),
                };
                let mut seen = vec![0usize; claim.atoms.len()];
                for group in &groups {
                    let lead = &claim.atoms[group[0]];
                    for &a in group {
                        seen[a] += 1;
                        let atom = &claim.atoms[a];
                        assert_eq!(atom.class, lead.class, "{}: one commitment", case.name);
                        assert_eq!(atom.point, lead.point, "{}: one point", case.name);
                    }
                    if let Class::Committed(Commitment::Weight(ty)) = lead.class {
                        *weight_claims.entry(format!("{ty:?}")).or_default() += 1;
                    }
                }
                for (a, atom) in claim.atoms.iter().enumerate() {
                    let expected = usize::from(matches!(atom.class, Class::Committed(_)));
                    assert_eq!(seen[a], expected, "{}: atom {a} of {claim:?}", case.name);
                }
            }
        }
        let expected: BTreeMap<String, usize> = [
            WeightType::Embed,
            WeightType::Q,
            WeightType::K,
            WeightType::V,
            WeightType::O,
            WeightType::Gate,
            WeightType::Up,
            WeightType::Down,
            WeightType::LmHead,
        ]
        .iter()
        .map(|ty| (format!("{ty:?}"), 1))
        .collect();
        assert_eq!(weight_claims, expected, "{}: INV-SCH-5", case.name);
    }
}

/// The residual's fused key `acc + 2^r·X_in` is the only claim the Llama templates split
/// (RFC-0037 'Splits', 'Example'): two values, the virtual key and `X_in`'s group, in each layer
/// of the leaf stage; no reduction stage splits.
#[test]
fn only_the_fused_key_splits_and_in_the_leaf_stage() {
    for case in cases() {
        let fused = case.forms.iter().any(|f| f == "fused key");
        let layers = case.schedule.layers() as usize;
        let expected = if fused { 2 * layers } else { 0 };
        assert_eq!(case.schedule.leaf_splits(), expected, "{}", case.name);
        for stage in case.schedule.stages() {
            assert_eq!(stage.splits, 0, "{}: stage {}", case.name, stage.index);
        }
        for claim in case.schedule.leaf() {
            if let Route::Split(parts) = &claim.claim.route {
                assert!(claim.stacked, "{}", case.name);
                let [Part::Virtual { atom: 0, .. }, Part::Group { atoms }] = parts.as_slice()
                else {
                    panic!("{}: {parts:?}", case.name);
                };
                assert_eq!(atoms, &[1]);
                assert_eq!(claim.claim.atoms[1].tensor, TensorRef::LayerInput);
                let site = case.graph.node(claim.site).unwrap();
                assert!(matches!(site.op(), Op::Requant { .. }), "{}", case.name);
            }
        }
    }
}

/// `tiny-2-servable`'s schedule, written out from the lowering of RFC-0044 by hand: stage 1
/// proves each gadget input's producer, the epilogue's and the logits', then `c_0`'s equality
/// zero-check; stage 2 the relations behind RoPE, the attention product and the embedding.
#[test]
fn tiny_2_servable_has_the_schedule_rfc_0037_gives_it() {
    use RelationKind::*;
    for case in cases().into_iter().filter(|c| c.model == "tiny-2-servable") {
        let kinds = |k: usize| -> Vec<RelationKind> {
            case.schedule.stages()[k]
                .instances
                .iter()
                .map(|i| i.relation.kind())
                .collect()
        };
        assert_eq!(
            kinds(0),
            [
                SumSquares,   // AttnNorm's RsqrtRow input
                Elementwise,  // AttnNorm's rescale
                Rotate,       // RopeQ
                Rotate,       // RopeK
                MatMulWeight, // V
                MatMulAct,    // the scores
                Elementwise,  // AttnOut: R · PV
                MatMulWeight, // AttnResidual's update: O
                SumSquares,   // MlpNorm
                Elementwise,
                MatMulWeight, // Silu: the gate
                MatMulWeight, // Up
                Elementwise,  // Product
                MatMulWeight, // MlpResidual's update: Down
                SumSquares,   // FinalNorm
                Elementwise,
                MatMulWeight, // the logits
                ZeroCheck,    // c_0 = 2^{s_e}·x_0
            ],
            "{}",
            case.name
        );
        // RoPE's projections, the attention product and the embedding.
        assert_eq!(
            kinds(1),
            [MatMulWeight, MatMulWeight, MatMulAct, MatMulWeight],
            "{}",
            case.name
        );
        let stage2 = &case.schedule.stages()[1].instances;
        let weight_of = |i: usize| match op(&case.graph, stage2[i].node) {
            Op::MatMulWeight { weight } => weight.ty,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            [weight_of(0), weight_of(1), weight_of(3)],
            [WeightType::Q, WeightType::K, WeightType::Embed]
        );
        assert!(stage2[0].stacked && stage2[2].stacked && !stage2[3].stacked);
        // The scores' instance is the stacked QKᵀ, the attention product's PV.
        let stage1 = &case.schedule.stages()[0].instances;
        assert_eq!(
            op(&case.graph, stage1[5].node),
            Op::MatMulAct {
                transpose_rhs: true
            }
        );
        assert_eq!(
            op(&case.graph, stage2[2].node),
            Op::MatMulAct {
                transpose_rhs: false
            }
        );
        assert!(matches!(
            op(&case.graph, stage1[1].node),
            Op::Mul {
                broadcast: Broadcast::Trailing
            }
        ));
        // c_0's zero-check reads the committed first checkpoint and the embedding's matmul.
        let zero = &stage1[17];
        assert_eq!(zero.tensor, TensorRef::Single(case.graph.checkpoints()[0]));
        assert_eq!(zero.zero.len(), zero.vars);
        assert_eq!(
            case.schedule.stages()[0].zero_vars,
            vars(case.n) + vars(64),
            "c_0 is [n, d_model]"
        );
    }
}

/// Every stacked kind of an instance or atom is named by layer 0's tensor, and committed kinds
/// keep their tensor's kind: advice, a weight of its type, or `ChannelParams`.
#[test]
fn atoms_are_named_by_layer_zero_and_classed_by_their_tensor() {
    for case in cases() {
        let graph = &case.graph;
        let checkpoints = graph.checkpoints();
        let first_layer_end = graph.tensor(checkpoints[1]).unwrap().producer.unwrap().0;
        for claims in claims_by_stage(&case.schedule) {
            for claim in claims {
                for atom in &claim.atoms {
                    if let TensorRef::Stacked(t) = atom.tensor {
                        let tensor = graph.tensor(t).unwrap();
                        let in_layer0 = tensor.producer.is_none_or(|p| p.0 <= first_layer_end);
                        assert!(in_layer0, "{}: {t:?}", case.name);
                    }
                    match (atom.class, atom.tensor) {
                        (Class::Committed(Commitment::Advice), TensorRef::LayerInput) => {}
                        (Class::Committed(c), TensorRef::Stacked(t) | TensorRef::Single(t)) => {
                            let kind = graph.tensor(t).unwrap().kind;
                            match c {
                                Commitment::Advice => {
                                    assert!(matches!(kind, Kind::Advice | Kind::Lookup { .. }))
                                }
                                Commitment::Weight(_) | Commitment::ChannelParams => {
                                    assert_eq!(kind, Kind::Weight)
                                }
                            }
                        }
                        (Class::Virtual { node }, TensorRef::Stacked(t) | TensorRef::Single(t)) => {
                            assert_eq!(graph.tensor(t).unwrap().producer, Some(node));
                        }
                        other => panic!("{}: {other:?}", case.name),
                    }
                }
            }
        }
    }
}

/// INV-SCH-4 (RFC-0010 'Invariants', RFC-0037 'Testing Strategy'): every golden model's template
/// at 1, 2, 3, 4, 5, 8 and 126 layers has the same stages, of the same relations in the same
/// order, whose variables differ by the layer variables `λ` alone; three and five layers pad the
/// stack with layers of zeros.
#[test]
fn the_stages_do_not_depend_on_the_number_of_layers() {
    for golden in goldens() {
        let claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
        let reference = derive(&lower(&check_public(&claims).unwrap(), 5).unwrap()).unwrap();
        let shape = |schedule: &Schedule| -> Vec<Vec<(RelationKind, bool, usize)>> {
            let lambda = schedule.layer_vars();
            schedule
                .stages()
                .iter()
                .map(|stage| {
                    stage
                        .instances
                        .iter()
                        .map(|i| {
                            let own = i.vars - if i.stacked { lambda } else { 0 };
                            (i.relation.kind(), i.stacked, own)
                        })
                        .collect()
                })
                .collect()
        };
        for layers in [1, 2, 3, 4, 5, 8, 126] {
            let edited = with_layers(&claims, layers);
            let public = check_public(&edited).unwrap_or_else(|e| panic!("{e:?}"));
            let graph = lower(&public, 5).unwrap();
            let schedule = derive(&graph).unwrap_or_else(|e| panic!("{}: {e}", golden.name));
            let what = format!("{} at {layers} layers", golden.name);
            assert_eq!(schedule.layers(), layers, "{what}");
            assert_eq!(schedule.layer_vars(), vars(layers.max(1)), "{what}");
            assert_eq!(schedule.stage_count(), reference.stage_count(), "{what}");
            assert_eq!(shape(&schedule), shape(&reference), "{what}");
            let per_layer = reference.leaf_splits() / reference.layers() as usize;
            assert_eq!(
                schedule.leaf_splits(),
                per_layer * layers as usize,
                "{what}"
            );
        }
    }
}
