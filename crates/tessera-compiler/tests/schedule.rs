// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The stage schedule (RFC-0010 'Stages', RFC-0037) of lowered tiny Llama models: its invariants,
//! and every claim it derives evaluated on an honest witness.
//!
//! The witness is the IR interpreter's (`common::evaluate`) on random weights and tokens. Every
//! tensor is padded per dimension with zeros, as RFC-0035 and RFC-0037 have the Llama templates'
//! padded rows and layers hold, and its extension is evaluated over Goldilocks at random
//! coordinates, `½` for a sum's. On it, every claim must equal the combination of its atoms, which
//! checks each linear map, and each matmul, product and sum of squares must hold at the operand
//! points the schedule gives its evaluations, which checks where each relation leaves its claims.

mod common;

use std::collections::HashMap;

use common::{Seeded, evaluate, llama, regenerate};
use tessera_compiler::schedule::{
    AtomClass, Claim, Coord, Factor, Kind, Layers, NodeRef, Part, Relation, Site as LeafSite,
    Source, Stage, StageSchedule,
};
use tessera_compiler::{CompileError, Lowering, PublicModel, derive_schedule, lower, stage_count};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::{
    CommitmentClass, GraphBuilder, Op, OpClass, Profile, QuantPoint, Site, TensorId, WeightType,
};

type F = Goldilocks;

// ----- Models ---------------------------------------------------------------------------------

/// How a tiny model's sites rescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sites {
    /// One shift per site.
    Pot,
    /// Two classes at the value projection's site and at both residual sites, whose updates then
    /// take two steps (RFC-0022).
    Classed,
    /// Every layer's RoPE query site affine, with multipliers (RFC-0025).
    Affine,
}

/// A tiny Llama model of `layers` layers over `n` positions, with `heads` query heads and `kv` key
/// and value heads of 128 dimensions each, a vocabulary of 300, a context of 16 and `sites`,
/// lowered.
fn lowering(
    profile: Profile,
    layers: u32,
    n: u32,
    (heads, kv): (u32, u32),
    sites: Sites,
) -> Lowering {
    let (mut arch, mut quant, silu) = llama(profile, layers, heads * 128, heads, kv, 512);
    // A short context keeps the RoPE table small.
    arch.vocab = 300;
    arch.max_context = 16;
    for layer in 0..layers {
        let point = |site| QuantPoint::Layer { layer, site };
        match sites {
            Sites::Pot => {}
            Sites::Classed => {
                for site in [Site::V, Site::AttnResidual, Site::MlpResidual] {
                    let set = quant
                        .shift_class_sets
                        .iter_mut()
                        .find(|(p, _)| *p == point(site))
                        .expect("a pot site");
                    let r = set.1[0];
                    set.1 = vec![r, r + 1];
                }
            }
            Sites::Affine => {
                let at = point(Site::RopeQ);
                let shift = quant
                    .shift_class_sets
                    .iter()
                    .find(|(p, _)| *p == at)
                    .map(|(_, shifts)| shifts[0])
                    .expect("a pot site");
                quant.shift_class_sets.retain(|&(p, _)| p != at);
                let place = quant.affine_sites.partition_point(|&(p, _)| p < at);
                quant.affine_sites.insert(place, (at, shift + 15));
            }
        }
    }
    let model = PublicModel {
        arch: &arch,
        profile,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    lower(model, n).expect("the model lowers")
}

/// The models the invariants are checked on: both profiles, every kind of site, 2 to 8 layers,
/// `n` a power of two and not, and one key and value head per query head as well as one per two.
fn models() -> impl Iterator<Item = (Profile, u32, u32, (u32, u32), Sites)> {
    [Profile::Servable, Profile::Minimal]
        .into_iter()
        .flat_map(|profile| {
            [Sites::Pot, Sites::Classed, Sites::Affine]
                .into_iter()
                .map(move |sites| (profile, sites))
        })
        .flat_map(|(profile, sites)| {
            [
                (2, 4, (2, 1)),
                (3, 5, (2, 1)),
                (4, 5, (2, 2)),
                (8, 5, (2, 1)),
            ]
            .into_iter()
            .map(move |(layers, n, heads)| (profile, layers, n, heads, sites))
        })
}

// ----- Invariants -----------------------------------------------------------------------------

/// INV-SCH-4: the stage count is the profile's, 3 for `servable` and 4 for `minimal`, whatever the
/// layers and positions, and the commitment report's; an affine RoPE site adds its `Mul` to RoPE's
/// chain, a fourth stage for `servable` (RFC-0010 'Stages').
#[test]
fn the_stage_count_is_the_profiles_whatever_the_layers() {
    for (profile, layers, n, heads, sites) in models() {
        let lowering = lowering(profile, layers, n, heads, sites);
        let schedule = derive_schedule(&lowering).expect("a schedule");
        let expected = match (profile, sites) {
            (Profile::Servable, Sites::Affine) | (Profile::Minimal, _) => 4,
            (Profile::Servable, _) => 3,
        };
        let label = format!("{profile:?}, {sites:?}, {layers} layers, n = {n}");
        assert_eq!(schedule.stage_count(), expected, "{label}");
        assert_eq!(
            schedule.stage_count(),
            stage_count(&lowering.graph) as usize,
            "{label}: the report's count"
        );
        assert_eq!(schedule.layers.count, layers, "{label}");
    }
}

/// INV-SCH-5: every weight type has exactly one claim in the pool, on it alone, at one point and,
/// for a type of every layer, stacked at a layer point; INV-SCH-3: every committed atom reaches
/// the pool once, with its commitment and point; INV-SCH-2: every virtual atom is one instance of
/// the next stage, in order.
#[test]
fn every_weight_type_has_one_claim_and_every_atom_reaches_its_place() {
    for (profile, layers, n, heads, sites) in models() {
        let lowering = lowering(profile, layers, n, heads, sites);
        let schedule = derive_schedule(&lowering).expect("a schedule");
        let label = format!("{profile:?}, {sites:?}, {layers} layers, n = {n}");
        let graph = &lowering.graph;
        let mut types: Vec<WeightType> = graph.weights.iter().map(|&(w, _)| w.ty).collect();
        types.sort_unstable();
        types.dedup();
        for ty in types {
            let claims: Vec<_> = schedule
                .pool
                .iter()
                .filter(|claim| {
                    claim
                        .atoms
                        .iter()
                        .any(|a| schedule.weight_type(graph, a.kind) == Some(ty))
                })
                .collect();
            let [claim] = claims[..] else {
                panic!("{label}: {ty:?} has {} claims in the pool", claims.len());
            };
            let [atom] = &claim.atoms[..] else {
                panic!("{label}: {ty:?}'s claim combines other tensors");
            };
            assert_eq!(claim.commitment, CommitmentClass::Weight, "{label}: {ty:?}");
            let stacked = !matches!(ty, WeightType::Embed | WeightType::LmHead);
            let layer_point = matches!(&atom.layers, Layers::Eq(rho) if rho.len() == schedule.layers.vars as usize);
            assert_eq!(
                layer_point, stacked,
                "{label}: {ty:?}'s layers {:?}",
                atom.layers
            );
            assert!(atom.is_unit(), "{label}: {ty:?}");
        }
        check_parts(&schedule, &label);
        check_rules(&lowering, &schedule, &label);
    }
}

/// Every atom's place: each committed group's pool claim, each virtual atom's instance.
fn check_parts(schedule: &StageSchedule, label: &str) {
    let mut pool_seen = vec![0; schedule.pool.len()];
    let rho: Vec<Coord> = (0..schedule.layers.vars).map(Coord::Layer).collect();
    for (k, stage) in schedule.stages.iter().enumerate() {
        let next = schedule.stages.get(k + 1);
        let mut instances = 0;
        for (c, claim) in stage.claims.iter().enumerate() {
            let mut placed = vec![0; claim.atoms.len()];
            for (p, part) in claim.parts.iter().enumerate() {
                match part {
                    Part::Instance { atom, instance } => {
                        assert_eq!(
                            *instance, instances,
                            "{label}: stage {k}, claim {c}: in order"
                        );
                        instances += 1;
                        let atom = &claim.atoms[*atom];
                        assert_eq!(atom.class, AtomClass::Virtual, "{label}");
                        let next = next.expect("a next stage");
                        let instance = &next.instances[*instance];
                        assert_eq!(
                            instance.source,
                            Source::Part { claim: c, part: p },
                            "{label}"
                        );
                        assert_eq!(
                            (instance.kind, &instance.point),
                            (atom.kind, &atom.point),
                            "{label}"
                        );
                        let layers = match &atom.layers {
                            Layers::Each => Layers::Eq(rho.clone()),
                            other => other.clone(),
                        };
                        assert_eq!(instance.layers, layers, "{label}");
                    }
                    Part::Pool { atoms, claim: pool } => {
                        pool_seen[*pool] += 1;
                        let entry = &schedule.pool[*pool];
                        assert_eq!((entry.stage as usize, entry.claim), (k, c), "{label}");
                        assert_eq!(entry.atoms.len(), atoms.len(), "{label}");
                        for (&i, pooled) in atoms.iter().zip(&entry.atoms) {
                            let atom = &claim.atoms[i];
                            assert_eq!(
                                atom.class,
                                AtomClass::Committed(entry.commitment),
                                "{label}"
                            );
                            assert_eq!(atom.point, entry.point, "{label}");
                            assert_eq!(
                                (pooled.kind, &pooled.point),
                                (atom.kind, &atom.point),
                                "{label}"
                            );
                        }
                    }
                }
                let atoms = match part {
                    Part::Instance { atom, .. } => vec![*atom],
                    Part::Pool { atoms, .. } => atoms.clone(),
                };
                for i in atoms {
                    placed[i] += 1;
                }
            }
            assert!(
                placed.iter().all(|&n| n == 1),
                "{label}: stage {k}, claim {c}: {placed:?}"
            );
            assert_eq!(claim.split, !ready(claim), "{label}: stage {k}, claim {c}");
        }
        if let Some(next) = next {
            let parts = next
                .instances
                .iter()
                .filter(|i| i.source != Source::ZeroCheck)
                .count();
            assert_eq!(parts, instances, "{label}: stage {}'s instances", k + 1);
        }
    }
    assert!(
        pool_seen.iter().all(|&n| n == 1),
        "{label}: every pool claim once"
    );
}

/// Whether a claim is ready unsplit: one virtual atom of coefficient 1, or committed atoms of one
/// commitment at one point (RFC-0037 'Splits').
fn ready(claim: &Claim) -> bool {
    match &claim.atoms[..] {
        [atom] if atom.class == AtomClass::Virtual => atom.is_unit(),
        atoms => {
            let first = &atoms[0];
            atoms.iter().all(|a| {
                a.class == first.class && a.class != AtomClass::Virtual && a.point == first.point
            })
        }
    }
}

/// The rules the witness cannot see: the leaf stage claims each gadget input and the logits, in
/// node order; an instance takes its stage's last `n_i` challenges and, a zero-check, the zero-check
/// point's last coordinates; its evaluations are its operands', in its relation's order.
fn check_rules(lowering: &Lowering, schedule: &StageSchedule, label: &str) {
    let graph = &lowering.graph;
    let layers = &schedule.layers;
    let gadgets: usize = graph
        .nodes
        .iter()
        .filter(|node| matches!(layers.position(node.id), None | Some((0, _))))
        .filter(|node| node.op.class() == OpClass::Gadget)
        .map(|node| node.inputs.len())
        .sum();
    assert_eq!(
        schedule.sites.len(),
        gadgets + 1,
        "{label}: one site per gadget input"
    );
    assert_eq!(schedule.sites.last(), Some(&LeafSite::Logits), "{label}");
    assert_eq!(
        schedule.stages[0].claims.len(),
        schedule.sites.len(),
        "{label}"
    );
    for stage in &schedule.stages[1..] {
        for (i, instance) in stage.instances.iter().enumerate() {
            let label = format!("{label}, stage {}, instance {i}", stage.index);
            let last: Vec<Coord> = (stage.rounds - instance.vars..stage.rounds)
                .map(|round| Coord::Round {
                    stage: stage.index,
                    round,
                })
                .collect();
            assert_eq!(instance.challenges, last, "{label}: the last challenges");
            if instance.source == Source::ZeroCheck {
                assert_eq!(stage.index, 1, "{label}: zero-checks run in stage 1");
                let mut zero: Vec<Coord> = instance.point.clone();
                if let Layers::Eq(rho) = &instance.layers {
                    zero.extend_from_slice(rho);
                }
                let expected: Vec<Coord> = (stage.zero_vars - instance.vars..stage.zero_vars)
                    .map(Coord::Zero)
                    .collect();
                assert_eq!(
                    zero, expected,
                    "{label}: the zero-check point's last coordinates"
                );
            }
            // The operands in the relation's order: a MatMulWeight's weight first.
            let node = match instance.node {
                NodeRef::Node(id) => id,
                NodeRef::Position(position) => layers.node(position, 0),
            };
            let inputs = &graph.nodes[node.0 as usize].inputs;
            let order: Vec<usize> = match instance.relation {
                Relation::Native { .. } => panic!("this oracle covers graph-only schedules"),
                Relation::MatMulWeight => vec![1, 0],
                Relation::Equality => (0..=inputs.len()).collect(),
                Relation::Booleanity | Relation::OneHotVocabulary | Relation::OneHotCount => {
                    vec![0]
                }
                Relation::OneHotIndex => vec![0, 1],
                Relation::Rotate | Relation::SumSquares | Relation::ScaleChannels => vec![0],
                Relation::MatMulAct { .. } | Relation::Mul(_) => vec![0, 1],
            };
            assert_eq!(
                instance.claims.len(),
                order.len(),
                "{label}: its evaluations"
            );
            for (claim, &operand) in stage.claims[instance.claims.clone()].iter().zip(&order) {
                let tensor = |kind, layer| layers.tensor(graph, kind, layer).expect("a tensor");
                let layer = match claim.layers {
                    Layers::One(layer) => layer,
                    _ => 0,
                };
                let expected = match instance.relation {
                    Relation::Booleanity | Relation::OneHotVocabulary | Relation::OneHotCount => {
                        tensor(instance.kind, 0)
                    }
                    Relation::OneHotIndex => {
                        if operand == 0 {
                            tensor(instance.kind, 0)
                        } else {
                            inputs[0]
                        }
                    }
                    Relation::Equality if operand == 0 => tensor(instance.kind, 0),
                    Relation::Equality => inputs[operand - 1],
                    _ => inputs[operand],
                };
                let input = |kind| tensor(kind, layer);
                // A layer's last checkpoint, read outside the layers, is its last output's.
                let actual = input(claim.kind);
                assert_eq!(actual, expected, "{label}: operand {operand}");
            }
        }
    }
}

/// RFC-0037's example: the residual key `acc + 2^r·X_in` splits into the accumulator's virtual
/// atom and the layer input's committed group, two values; a two-step residual's key
/// `2·(X_in + u)` is one committed group, ready.
#[test]
fn a_residual_key_splits_into_its_accumulator_and_its_stream() {
    for profile in [Profile::Servable, Profile::Minimal] {
        let lowering = lowering(profile, 2, 5, (2, 1), Sites::Pot);
        let schedule = derive_schedule(&lowering).expect("a schedule");
        let leaf = &schedule.stages[0];
        let split: Vec<&Claim> = leaf.claims.iter().filter(|c| c.split).collect();
        assert_eq!(split.len(), 2, "{profile:?}: the two residual sites");
        let attention = split[0];
        assert!(matches!(
            attention.parts[..],
            [Part::Instance { atom: 0, .. }, Part::Pool { .. }]
        ));
        assert_eq!(attention.atoms[1].kind, Kind::LayerInput);
        assert_eq!(attention.atoms[1].coefficient.len(), 1);
        assert!(matches!(
            attention.atoms[1].coefficient[0][..],
            [Factor::Constant(NodeRef::Position(_))]
        ));

        let classed = lowering_classed(profile);
        let schedule = derive_schedule(&classed).expect("a schedule");
        let leaf = &schedule.stages[0];
        assert!(
            leaf.claims.iter().all(|c| !c.split),
            "{profile:?}: two-step residuals are ready"
        );
        assert!(
            leaf.claims
                .iter()
                .any(|c| c.atoms.iter().any(|a| a.kind == Kind::LayerInput))
        );
    }
}

fn lowering_classed(profile: Profile) -> Lowering {
    lowering(profile, 2, 5, (2, 1), Sites::Classed)
}

// ----- Refusals -------------------------------------------------------------------------------

/// A graph whose layers are not one template has no schedule: `ERR-INTERNAL`, with the rule.
#[test]
fn layers_that_are_not_one_template_are_refused() {
    let base = lowering(Profile::Servable, 2, 5, (2, 1), Sites::Pot);
    let schedule = derive_schedule(&base).expect("a schedule");
    let (first, width) = (
        schedule.layers.first as usize,
        schedule.layers.width as usize,
    );
    let refusal = |lowering: &Lowering| match derive_schedule(lowering) {
        Err(error @ CompileError::Schedule { rule, .. }) => {
            assert_eq!(error.code(), tessera_codec::error::ErrorCode::Internal);
            rule
        }
        other => panic!("refused, not {other:?}"),
    };

    // Layer 1's residual scales by another kind of constant.
    let mut other = base.clone();
    let at = (first + width..first + 2 * width)
        .find(|&i| matches!(other.graph.nodes[i].op, Op::ConstMul { .. }))
        .expect("a residual's ConstMul");
    let per_channel = other.graph.constants.len();
    other.graph.constants.push(vec![1; 256]);
    other.graph.nodes[at].op = Op::ConstMul {
        c: tessera_ir::ConstRef(u32::try_from(per_channel).expect("small")),
    };
    assert_eq!(
        refusal(&other),
        "the nodes at one position have one operation"
    );

    // Layer 1's first norm reads layer 0's attention residual, of the checkpoint's shape and kind,
    // in place of its checkpoint.
    let mut other = base.clone();
    let checkpoint = other.checkpoints[1];
    let stream = stream_of(&other, first, width, checkpoint);
    let reader = first + width;
    assert_eq!(other.graph.nodes[reader].inputs[0], checkpoint);
    other.graph.nodes[reader].inputs[0] = stream;
    assert_eq!(
        refusal(&other),
        "a layer reads its own outputs, its checkpoint and graph inputs only"
    );

    // A checkpoint that is no committed advice, and checkpoints whose layers differ in length.
    let mut other = base.clone();
    other.checkpoints[1] = other.graph.nodes[first + width].outputs[0];
    assert_eq!(refusal(&other), "every checkpoint is committed advice");
    let mut other = base.clone();
    let requant = (first + width..first + 2 * width - 1)
        .find(|&i| matches!(other.graph.nodes[i].op, Op::Requant { .. }))
        .expect("a requantization in layer 1");
    other.checkpoints[1] = other.graph.nodes[requant].outputs[0];
    assert_eq!(refusal(&other), "every layer holds as many nodes");

    // The epilogue's final norm reads a tensor inside the last layer in place of its checkpoint.
    let mut other = base.clone();
    let last = *other.checkpoints.last().expect("checkpoints");
    let end = first + 2 * width;
    let reader = (end..other.graph.nodes.len())
        .find(|&i| other.graph.nodes[i].inputs.contains(&last))
        .expect("the final norm");
    other.graph.nodes[reader].inputs[0] = stream_of(&other, first + width, width, last);
    assert_eq!(
        refusal(&other),
        "a layer's outputs are read in the layer, but for its checkpoint"
    );

    // A single affine site would make layer 0 another template than layer 1: the lowering refuses
    // its claim list first (RFC-0038).
    let (arch, mut quant, silu) = llama(Profile::Servable, 2, 256, 2, 1, 512);
    let at = QuantPoint::Layer {
        layer: 0,
        site: Site::RopeQ,
    };
    quant.shift_class_sets.retain(|&(p, _)| p != at);
    quant.affine_sites.push((at, 37));
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    assert_eq!(
        lower(model, 5),
        Err(CompileError::Model {
            rule: "each site is affine in every layer or in none"
        })
    );
}

/// A committed tensor inside the layer of `width` nodes from node `first`, of `like`'s shape and
/// kind, other than its checkpoint.
fn stream_of(lowering: &Lowering, first: usize, width: usize, like: TensorId) -> TensorId {
    let tensors = &lowering.graph.tensors;
    (first..first + width - 1)
        .map(|i| lowering.graph.nodes[i].outputs[0])
        .find(|&t| {
            let (a, b) = (&tensors[t.0 as usize], &tensors[like.0 as usize]);
            a.shape == b.shape && a.kind == b.kind
        })
        .expect("a committed residual stream inside the layer")
}

// ----- Claims on an honest witness ------------------------------------------------------------

/// Every claim of every stage equals the combination of its atoms on an honest witness at random
/// coordinates, and every matmul, product and sum of squares holds at its operands' claim points.
#[test]
fn every_claim_holds_on_an_honest_witness() {
    let mut rng = Seeded(0x0113_5C4E);
    for profile in [Profile::Servable, Profile::Minimal] {
        for sites in [Sites::Pot, Sites::Classed, Sites::Affine] {
            for (layers, heads) in [(2, (2, 1)), (3, (2, 2))] {
                let lowering = lowering(profile, layers, 5, heads, sites);
                let schedule = derive_schedule(&lowering).expect("a schedule");
                let inputs = llama_inputs(&lowering, &mut rng);
                let mut witness = Witness::new(&lowering, &schedule, inputs, &mut rng);
                let label = format!("{profile:?}, {sites:?}, {layers} layers");
                check_witness(&mut witness, &schedule, &label);
            }
        }
    }
}

/// The linear maps the Llama lowering does not use, in a template of their own: two layers that
/// each read their checkpoint `c` twice, through two transposes that are not involutions, and
/// through `Expand` over 3 positions then `Sum` over them, `3c`, scaled by a constant of their own,
/// then subtract and requantize. The two atoms meet on `c` at one point and merge, `Sum`'s `½`
/// meeting `Expand`'s `LT`; the constants differ between the layers.
#[test]
fn every_linear_map_holds_in_a_template_of_its_own() {
    let mut b = GraphBuilder::new();
    let x = b
        .input(common::decl(&[5, 3, 4], common::ADVICE, -100, 100))
        .expect("an input");
    let two = b.constant(vec![2]).expect("a constant");
    let advice = |shape: &[u32]| common::out(shape, common::ADVICE);
    let mut c = common::one(&mut b, Op::ConstMul { c: two }, &[x], advice(&[5, 3, 4]));
    let mut checkpoints = vec![c];
    for layer in 0..2 {
        let turned = common::one(
            &mut b,
            Op::Transpose {
                perm: vec![2, 0, 1],
            },
            &[c],
            common::virt(&[4, 5, 3]),
        );
        let back = common::one(
            &mut b,
            Op::Transpose {
                perm: vec![1, 2, 0],
            },
            &[turned],
            common::virt(&[5, 3, 4]),
        );
        let wide = common::one(
            &mut b,
            Op::Expand { axis: 0, len: 3 },
            &[c],
            common::virt(&[3, 5, 3, 4]),
        );
        let thrice = common::one(
            &mut b,
            Op::Sum { axis: 0 },
            &[wide],
            common::virt(&[5, 3, 4]),
        );
        let k = b.constant(vec![2 + 3 * layer]).expect("a constant");
        let scaled = common::one(
            &mut b,
            Op::ConstMul { c: k },
            &[thrice],
            common::virt(&[5, 3, 4]),
        );
        let difference = common::one(&mut b, Op::Sub, &[back, scaled], common::virt(&[5, 3, 4]));
        // The checkpoint is summed over its axis of 4: its padded positions are `x`'s, a bare
        // committed input's, which nothing pins, so a sum over its axis of 5 is refused
        // (RFC-0072 'Summing relations'). The sum over a padded axis is the `Expand`'s, above.
        let mean = common::one(&mut b, Op::Sum { axis: 2 }, &[c], common::virt(&[5, 3]));
        let spread = common::one(
            &mut b,
            Op::Expand { axis: 2, len: 4 },
            &[mean],
            common::virt(&[5, 3, 4]),
        );
        let key = common::one(
            &mut b,
            Op::Add,
            &[difference, spread],
            common::virt(&[5, 3, 4]),
        );
        // A committed linear output in the layer, which no node reads: a stacked zero-check.
        common::one(&mut b, Op::ConstMul { c: k }, &[c], advice(&[5, 3, 4]));
        let split = common::one(&mut b, Op::Reshape, &[key], common::virt(&[5, 3, 2, 2]));
        let merged = common::one(&mut b, Op::Reshape, &[split], common::virt(&[5, 3, 4]));
        let shift = b.shift_classes(vec![1]).expect("shift classes");
        let requant = Op::Requant {
            shift,
            lo: -1000,
            hi: 1000,
        };
        c = common::one(&mut b, requant, &[merged], advice(&[5, 3, 4]));
        checkpoints.push(c);
    }
    let lowering = Lowering {
        graph: b.finish(),
        tables: Vec::new(),
        tokens: x,
        positions: x,
        one_hot: x,
        logits: c,
        checkpoints,
        sites: Vec::new(),
        classes: Vec::new(),
        multipliers: Vec::new(),
    };
    let schedule = derive_schedule(&lowering).expect("a schedule");
    // The key's claim: the checkpoint at its point, the two products merged, `Sum`'s `½` meeting
    // `Expand`'s `LT`, and the checkpoint at the mean's point, another group: a split.
    let key = &schedule.stages[0].claims[0];
    let [merged, mean] = &key.atoms[..] else {
        panic!("two atoms: {:?}", key.atoms);
    };
    assert_eq!((merged.kind, &merged.point), (Kind::LayerInput, &key.point));
    assert_eq!(merged.coefficient.len(), 2, "merged");
    assert!(merged.coefficient[1].contains(&Factor::Lt {
        coords: vec![Coord::Half; 2],
        len: 3,
    }));
    assert_eq!(mean.kind, Kind::LayerInput);
    assert_eq!(mean.point[..2], [Coord::Half; 2]);
    assert!(key.split, "two groups of one commitment at two points");
    assert!(matches!(
        key.parts[..],
        [Part::Pool { .. }, Part::Pool { .. }]
    ));
    // Stage 1: the zero-checks of `c_0` and of the layers' committed output, of 7 and 8 variables.
    let zero: Vec<u32> = schedule.stages[1]
        .instances
        .iter()
        .map(|i| i.vars)
        .collect();
    assert_eq!(zero, [7, 8]);
    assert_eq!(
        schedule.stage_count(),
        2,
        "the leaf stage and the zero-checks"
    );
    assert_eq!(
        schedule.stage_count(),
        stage_count(&lowering.graph) as usize
    );
    check_parts(&schedule, "the template of every map");
    check_rules(&lowering, &schedule, "the template of every map");
    let mut rng = Seeded(0x0113_11AE);
    let inputs = vec![(x, (0..60).map(|_| rng.range(-100, 100)).collect())];
    let mut witness = Witness::new(&lowering, &schedule, inputs, &mut rng);
    check_witness(&mut witness, &schedule, "the template of every map");
}

/// A relation's output that a stage after the leaf stage reaches through a constant of each layer's
/// own: the stacked claim would weigh the layers by the constants, not by `eq`, so a constant that
/// differs between the layers is refused, and one every layer shares is not.
#[test]
fn a_stacked_relation_output_scaled_per_layer_is_refused() {
    let template = |constants: [i64; 2]| {
        let mut b = GraphBuilder::new();
        let x = b
            .input(common::decl(&[4, 8], common::ADVICE, -100, 100))
            .expect("an input");
        let one = b.constant(vec![1]).expect("a constant");
        let advice = |shape: &[u32]| common::out(shape, common::ADVICE);
        let mut c = common::one(&mut b, Op::ConstMul { c: one }, &[x], advice(&[4, 8]));
        let mut checkpoints = vec![c];
        for constant in constants {
            let mul = Op::Mul {
                broadcast: tessera_ir::Broadcast::Trailing,
            };
            let square = common::one(&mut b, mul.clone(), &[c, c], common::virt(&[4, 8]));
            let k = b.constant(vec![constant]).expect("a constant");
            let scaled = common::one(
                &mut b,
                Op::ConstMul { c: k },
                &[square],
                common::virt(&[4, 8]),
            );
            let cube = common::one(&mut b, mul, &[scaled, c], common::virt(&[4, 8]));
            let shift = b.shift_classes(vec![1]).expect("shift classes");
            let requant = Op::Requant {
                shift,
                lo: -100,
                hi: 100,
            };
            c = common::one(&mut b, requant, &[cube], advice(&[4, 8]));
            checkpoints.push(c);
        }
        Lowering {
            graph: b.finish(),
            tables: Vec::new(),
            tokens: x,
            positions: x,
            one_hot: x,
            logits: c,
            checkpoints,
            sites: Vec::new(),
            classes: Vec::new(),
            multipliers: Vec::new(),
        }
    };
    let lowering = template([3, 3]);
    let shared = derive_schedule(&lowering).expect("one constant in every layer");
    assert_eq!(shared.stage_count(), 3);
    // The scaled square is one virtual atom of coefficient 3: stage 1 sends its value.
    let scaled = &shared.stages[1].claims[0];
    assert!(scaled.split && scaled.parts.len() == 1, "{scaled:?}");
    check_parts(&shared, "one constant");
    check_rules(&lowering, &shared, "one constant");
    let mut rng = Seeded(0x0113_C0DE);
    let inputs = vec![(
        lowering.tokens,
        (0..32).map(|_| rng.range(-100, 100)).collect(),
    )];
    let mut witness = Witness::new(&lowering, &shared, inputs, &mut rng);
    check_witness(&mut witness, &shared, "one constant");
    match derive_schedule(&template([3, 5])) {
        Err(CompileError::Schedule { rule, .. }) => assert_eq!(
            rule,
            "a relation's output stacked over the layers is scaled by one constant in every layer"
        ),
        other => panic!("refused, not {other:?}"),
    }
}

/// Random inputs of a lowered tiny Llama model: tokens, positions `0` to `n − 1`, INT8 weights,
/// channel classes and multipliers.
fn llama_inputs(lowering: &Lowering, rng: &mut Seeded) -> Vec<(TensorId, Vec<i128>)> {
    let graph = &lowering.graph;
    let n = graph.tensors[lowering.positions.0 as usize].shape[0];
    let vocab = graph.tensors[lowering.one_hot.0 as usize].shape[1];
    let len = |t: TensorId| {
        graph.tensors[t.0 as usize]
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
    };
    let mut inputs = vec![
        (
            lowering.tokens,
            (0..n)
                .map(|_| rng.range(0, i128::from(vocab) - 1))
                .collect(),
        ),
        (lowering.positions, (0..i128::from(n)).collect()),
    ];
    for &(_, t) in &graph.weights {
        inputs.push((t, (0..len(t)).map(|_| rng.range(-127, 127)).collect()));
    }
    for &(_, t) in &lowering.classes {
        let hi = graph.tensors[t.0 as usize].range.hi;
        inputs.push((t, (0..len(t)).map(|_| rng.range(0, hi)).collect()));
    }
    for &(_, t) in &lowering.multipliers {
        inputs.push((t, (0..len(t)).map(|_| rng.range(1, 65_535)).collect()));
    }
    inputs
}

/// Checks every claim and instance of `schedule` on `witness`.
fn check_witness(witness: &mut Witness<'_>, schedule: &StageSchedule, label: &str) {
    for stage in &schedule.stages {
        for (c, claim) in stage.claims.iter().enumerate() {
            let label = format!("{label}, stage {}, claim {c}", stage.index);
            witness.check_claim(claim, &label);
            witness.check_pool(claim, &label);
        }
        for i in 0..stage.instances.len() {
            witness.check_instance(
                stage,
                i,
                &format!("{label}, stage {}, instance {i}", stage.index),
            );
        }
    }
}

/// An honest witness, padded per dimension and per layer with zeros, and the values of the
/// schedule's coordinates.
struct Witness<'a> {
    lowering: &'a Lowering,
    schedule: &'a StageSchedule,
    values: Vec<Vec<i128>>,
    tables: HashMap<TensorId, Vec<F>>,
    coords: HashMap<Coord, F>,
    rng: &'a mut Seeded,
}

impl<'a> Witness<'a> {
    /// The witness of `lowering` on `inputs`, every graph input's values.
    fn new(
        lowering: &'a Lowering,
        schedule: &'a StageSchedule,
        inputs: Vec<(TensorId, Vec<i128>)>,
        rng: &'a mut Seeded,
    ) -> Self {
        let values = evaluate(&lowering.graph, &regenerate(&lowering.tables), inputs);
        Self {
            lowering,
            schedule,
            values,
            tables: HashMap::new(),
            coords: HashMap::new(),
            rng,
        }
    }

    /// The value of a coordinate: random, but `½` for a sum's.
    fn coord(&mut self, c: Coord) -> F {
        if c == Coord::Half {
            return F::from_u64_reduced(2).inverse().expect("2 ≠ 0");
        }
        let rng = &mut *self.rng;
        *self
            .coords
            .entry(c)
            .or_insert_with(|| F::from_u64_reduced(rng.next()))
    }

    fn point(&mut self, point: &[Coord]) -> Vec<F> {
        point.iter().map(|&c| self.coord(c)).collect()
    }

    /// `tensor`'s values, padded per dimension with zeros, the last axis the lowest bits.
    fn table(&mut self, tensor: TensorId) -> &[F] {
        let graph = &self.lowering.graph;
        let values = &self.values;
        self.tables.entry(tensor).or_insert_with(|| {
            let shape = &graph.tensors[tensor.0 as usize].shape;
            let vars: Vec<u32> = shape
                .iter()
                .map(|&d| d.next_power_of_two().trailing_zeros())
                .collect();
            let mut table = vec![F::ZERO; 1 << vars.iter().sum::<u32>()];
            for (flat, &v) in values[tensor.0 as usize].iter().enumerate() {
                let (mut rest, mut index, mut shift) = (flat, 0usize, 0);
                for (axis, &len) in shape.iter().enumerate().rev() {
                    index |= (rest % len as usize) << shift;
                    rest /= len as usize;
                    shift += vars[axis];
                }
                table[index] = field(v);
            }
            table
        })
    }

    /// The tensor of `kind` in `layer`, or `None` in a padded layer.
    fn tensor(&self, kind: Kind, layer: u32) -> Option<TensorId> {
        self.schedule
            .layers
            .tensor(&self.lowering.graph, kind, layer)
    }

    /// `X̃(point)` for `kind`'s tensor in `layer`: 0 in a padded layer.
    fn mle(&mut self, kind: Kind, layer: u32, point: &[Coord]) -> F {
        let Some(tensor) = self.tensor(kind, layer) else {
            return F::ZERO;
        };
        let r = self.point(point);
        let mut values = self.table(tensor).to_vec();
        assert_eq!(
            values.len(),
            1 << r.len(),
            "a point of the tensor's variables"
        );
        for &x in &r {
            values = values
                .chunks_exact(2)
                .map(|pair| pair[0] + x * (pair[1] - pair[0]))
                .collect();
        }
        values[0]
    }

    /// The layers a weight runs over, with each one's weight.
    fn weights(&mut self, layers: &Layers, each: Option<u32>) -> Vec<(u32, F)> {
        match layers {
            Layers::None => vec![(0, F::ONE)],
            Layers::Each => vec![(each.expect("a layer"), F::ONE)],
            Layers::One(layer) => vec![(*layer, F::ONE)],
            Layers::Eq(rho) => {
                let rho = self.point(rho);
                (0..1u32 << rho.len())
                    .map(|l| (l, eq_at(&rho, l)))
                    .collect()
            }
        }
    }

    /// A coefficient's value in `layer`.
    fn coefficient(&mut self, products: &[Vec<Factor>], layer: u32) -> F {
        let mut sum = F::ZERO;
        for product in products {
            let mut value = F::ONE;
            for factor in product {
                value *= match factor {
                    Factor::Minus => -F::ONE,
                    Factor::Pow2(j) => F::from_u64_reduced(1 << j),
                    Factor::Constant(node) => self.constant(*node, layer),
                    Factor::Lt { coords, len } => {
                        let r = self.point(coords);
                        (0..*len).map(|a| eq_at(&r, a)).fold(F::ZERO, |s, e| s + e)
                    }
                };
            }
            sum += value;
        }
        sum
    }

    /// The constant of a `ConstMul` by one constant, in `layer` for a template position: 0 in a
    /// padded layer, where every tensor is 0.
    fn constant(&self, node: NodeRef, layer: u32) -> F {
        let layers = &self.schedule.layers;
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(_) if layer >= layers.count => return F::ZERO,
            NodeRef::Position(position) => layers.node(position, layer),
        };
        match self.lowering.graph.nodes[id.0 as usize].op {
            Op::ConstMul { c } => field(i128::from(self.lowering.graph.constants[c.0 as usize][0])),
            _ => panic!("a ConstMul"),
        }
    }

    /// A claim's value against its atoms' combination, per layer for a leaf claim in the layers.
    fn check_claim(&mut self, claim: &Claim, label: &str) {
        let each: Vec<Option<u32>> = match claim.layers {
            Layers::Each => (0..self.schedule.layers.count).map(Some).collect(),
            _ => vec![None],
        };
        for layer in each {
            let mut value = F::ZERO;
            for (l, w) in self.weights(&claim.layers, layer) {
                value += w * self.mle(claim.kind, l, &claim.point);
            }
            let mut atoms = F::ZERO;
            for atom in &claim.atoms {
                for (l, w) in self.weights(&atom.layers, layer) {
                    let c = self.coefficient(&atom.coefficient, l);
                    atoms += w * c * self.mle(atom.kind, l, &atom.point);
                }
            }
            assert_eq!(
                value, atoms,
                "{label}, layer {layer:?}: the claim is its atoms"
            );
        }
    }

    /// Each pool claim a claim's committed groups reach: its value is its group's, stacked at the
    /// layer point `ρ_1` for a leaf claim in the layers.
    fn check_pool(&mut self, claim: &Claim, label: &str) {
        for part in &claim.parts {
            let Part::Pool { atoms, claim: pool } = part else {
                continue;
            };
            let entry = &self.schedule.pool[*pool];
            let mut pooled = F::ZERO;
            for atom in &entry.atoms {
                for (l, w) in self.weights(&atom.layers, None) {
                    let c = self.coefficient(&atom.coefficient, l);
                    pooled += w * c * self.mle(atom.kind, l, &atom.point);
                }
            }
            // The group's value: per layer, stacked by eq(ρ_1, ℓ), for a leaf claim in the layers.
            let stacking: Vec<(Option<u32>, F)> = match claim.layers {
                Layers::Each => {
                    let rho: Vec<Coord> =
                        (0..self.schedule.layers.vars).map(Coord::Layer).collect();
                    let rho = self.point(&rho);
                    (0..self.schedule.layers.count)
                        .map(|l| (Some(l), eq_at(&rho, l)))
                        .collect()
                }
                _ => vec![(None, F::ONE)],
            };
            let mut group = F::ZERO;
            for (layer, stack) in stacking {
                for &i in atoms {
                    let atom = &claim.atoms[i];
                    for (l, w) in self.weights(&atom.layers, layer) {
                        let c = self.coefficient(&atom.coefficient, l);
                        group += stack * w * c * self.mle(atom.kind, l, &atom.point);
                    }
                }
            }
            assert_eq!(pooled, group, "{label}: pool claim {pool} is its group");
        }
    }

    /// An instance's relation on its target, at the points its evaluations' claims give, the
    /// instance's own challenges summed over the hypercube, layer by layer.
    fn check_instance(&mut self, stage: &Stage, index: usize, label: &str) {
        let instance = &stage.instances[index];
        let lambda = match instance.layers {
            Layers::Eq(_) => self.schedule.layers.vars as usize,
            _ => 0,
        };
        let own = &instance.challenges[..instance.challenges.len() - lambda];
        let claims = &stage.claims[instance.claims.clone()];
        let layers: Vec<u32> = match instance.layers {
            Layers::Eq(_) => (0..self.schedule.layers.count).collect(),
            _ => vec![0],
        };
        let lowering = self.lowering;
        let output_shape = self
            .schedule
            .layers
            .tensor(&lowering.graph, instance.kind, 0)
            .map(|t| lowering.graph.tensors[t.0 as usize].shape.clone())
            .expect("the output");
        for layer in layers {
            let partials: Vec<Vec<F>> = claims
                .iter()
                .map(|claim| {
                    let l = match claim.layers {
                        Layers::One(l) => l,
                        Layers::None => 0,
                        _ => layer,
                    };
                    self.partial(claim.kind, l, &claim.point, own)
                })
                .collect();
            let target = self.mle(instance.kind, layer, &instance.point);
            let r = self.point(&instance.point);
            let sum = match instance.relation {
                Relation::Native { .. } => panic!("this oracle covers graph-only schedules"),
                Relation::MatMulWeight => dot(&partials[0], &partials[1]),
                Relation::MatMulAct { .. } => {
                    // eq(r_h, h) over the heads: the target's highest block and the instance's.
                    let heads = output_shape[0].next_power_of_two().trailing_zeros() as usize;
                    let r_h = &r[r.len() - heads..];
                    let low = own.len() - heads;
                    (0..partials[0].len())
                        .map(|f| {
                            eq_at(r_h, u32::try_from(f >> low).expect("small"))
                                * partials[0][f]
                                * partials[1][f]
                        })
                        .fold(F::ZERO, |s, x| s + x)
                }
                Relation::Mul(_) => (0..partials[0].len())
                    .map(|f| {
                        eq_at(&r, u32::try_from(f).expect("small"))
                            * partials[0][f]
                            * partials[1][f]
                    })
                    .fold(F::ZERO, |s, x| s + x),
                Relation::SumSquares => {
                    let low = own.len() - r.len();
                    (0..partials[0].len())
                        .map(|f| {
                            eq_at(&r, u32::try_from(f >> low).expect("small"))
                                * partials[0][f]
                                * partials[0][f]
                        })
                        .fold(F::ZERO, |s, x| s + x)
                }
                Relation::Booleanity => {
                    let residual = partials[0]
                        .iter()
                        .enumerate()
                        .map(|(v, &h)| {
                            eq_at(&r, u32::try_from(v).expect("small")) * h * (F::ONE - h)
                        })
                        .fold(F::ZERO, |s, x| s + x);
                    assert_eq!(residual, F::ZERO, "{label}: original OneHot booleanity");
                    continue;
                }
                Relation::OneHotVocabulary | Relation::OneHotCount | Relation::OneHotIndex => {
                    let width = output_shape[1].next_power_of_two() as usize;
                    let vbits = output_shape[1].next_power_of_two().trailing_zeros() as usize;
                    let scale = F::from_u64_reduced(width as u64).inverse().unwrap();
                    let residual = partials[0]
                        .iter()
                        .enumerate()
                        .map(|(f, &h)| match instance.relation {
                            Relation::OneHotVocabulary => {
                                eq_at(&r, f as u32)
                                    * if f % width >= output_shape[1] as usize {
                                        h
                                    } else {
                                        F::ZERO
                                    }
                            }
                            Relation::OneHotCount => {
                                eq_at(&r[vbits..], (f / width) as u32)
                                    * (h - if f / width < output_shape[0] as usize {
                                        scale
                                    } else {
                                        F::ZERO
                                    })
                            }
                            Relation::OneHotIndex => {
                                eq_at(&r[vbits..], (f / width) as u32)
                                    * (F::from_u64_reduced((f % width) as u64) * h
                                        - scale * partials[1][f])
                            }
                            _ => unreachable!(),
                        })
                        .fold(F::ZERO, |sum, x| sum + x);
                    assert_eq!(
                        residual,
                        F::ZERO,
                        "{label}: original OneHot row/vocabulary relation"
                    );
                    continue;
                }
                Relation::Equality => {
                    // `Y = c·X` everywhere: the zero-checks here are of `ConstMul`s.
                    let c = self.constant(instance.node, layer);
                    let z = r;
                    let residual = (0..partials[0].len())
                        .map(|f| {
                            eq_at(&z, u32::try_from(f).expect("small"))
                                * (partials[0][f] - c * partials[1][f])
                        })
                        .fold(F::ZERO, |s, x| s + x);
                    assert_eq!(residual, F::ZERO, "{label}: the zero-check holds");
                    continue;
                }
                // The kernel's own identity is tessera-sumcheck's (RFC-0037 'Rotate').
                Relation::Rotate | Relation::ScaleChannels => continue,
            };
            assert_eq!(
                sum, target,
                "{label}, layer {layer}: {:?} at its claims' points",
                instance.relation
            );
        }
    }

    /// `X̃` of `kind`'s tensor in `layer` with the coordinates `free` left free: its values at
    /// every assignment of them, `free[0]` the lowest bit.
    fn partial(&mut self, kind: Kind, layer: u32, point: &[Coord], free: &[Coord]) -> Vec<F> {
        let mut out = vec![F::ZERO; 1 << free.len()];
        let Some(tensor) = self.tensor(kind, layer) else {
            return out;
        };
        let fixed: Vec<Option<F>> = point
            .iter()
            .map(|&c| (!free.contains(&c)).then(|| self.coord(c)))
            .collect();
        let slot: Vec<Option<usize>> = point
            .iter()
            .map(|c| free.iter().position(|f| f == c))
            .collect();
        let table = self.table(tensor).to_vec();
        for (b, &v) in table.iter().enumerate() {
            if v == F::ZERO {
                continue;
            }
            let (mut w, mut f) = (F::ONE, 0usize);
            for (j, r) in fixed.iter().enumerate() {
                let bit = (b >> j) & 1;
                match r {
                    Some(r) => w *= if bit == 1 { *r } else { F::ONE - *r },
                    None => f |= bit << slot[j].expect("a free coordinate"),
                }
            }
            out[f] += w * v;
        }
        // A free coordinate the point lacks: the value does not depend on it.
        let present: usize = slot.iter().flatten().map(|&s| 1 << s).sum();
        for f in 0..out.len() {
            out[f] = out[f & present];
        }
        out
    }
}

/// `x` in the field.
fn field(x: i128) -> F {
    let p = i128::from(F::MODULUS);
    F::from_u64_reduced(u64::try_from(x.rem_euclid(p)).expect("reduced"))
}

/// `eq(r, x)` at the vertex `x`, `r[0]` its lowest bit.
fn eq_at(r: &[F], x: u32) -> F {
    r.iter().enumerate().fold(F::ONE, |product, (j, &ri)| {
        product * if (x >> j) & 1 == 1 { ri } else { F::ONE - ri }
    })
}

fn dot(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::ZERO, |s, (&x, &y)| s + x * y)
}

#[test]
fn composed_native_metadata_requires_legal_original_graph_replacements() {
    use tessera_compiler::schedule::{Native, NativeOutput, derive_composed_schedule};
    let low = lowering(Profile::Servable, 3, 5, (2, 1), Sites::Pot);
    let layers = tessera_compiler::schedule::LayerMap::new(&low).unwrap();
    let node = low
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::SoftmaxRow { .. }))
        .unwrap();
    let vars = low.graph.tensors[node.inputs[0].0 as usize]
        .shape
        .iter()
        .map(|n| n.next_power_of_two().trailing_zeros())
        .sum::<u32>();
    let spec = Native {
        tag: 7,
        node: node.id,
        vars: vars + layers.vars,
        degree: 3,
        evaluations: 2,
        zero: false,
        outputs: vec![NativeOutput {
            tensor: node.inputs[0],
            from: 0,
            vars,
        }],
    };
    let deferred = [(node.id, Some(0))];
    assert!(derive_composed_schedule(&low, &deferred, std::slice::from_ref(&spec)).is_ok());
    assert_eq!(
        derive_schedule(&low).unwrap(),
        derive_composed_schedule(&low, &[], &[]).unwrap()
    );
    let bad = |d: &[_], n: &[_]| match derive_composed_schedule(&low, d, n) {
        Err(CompileError::Schedule { .. }) => {}
        Err(e) => panic!("wrong error for {d:?}, {n:?}: {e}"),
        Ok(_) => panic!("accepted malformed {d:?}, {n:?}"),
    };
    bad(&deferred, &[]);
    bad(&[deferred[0], deferred[0]], std::slice::from_ref(&spec));
    bad(&[(node.id, Some(usize::MAX))], std::slice::from_ref(&spec));
    bad(&[(node.id, None)], std::slice::from_ref(&spec));
    bad(&deferred, &[spec.clone(), spec.clone()]);
    let mut forged = low.clone();
    let ordinary = low
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::ConstMul { .. }))
        .unwrap();
    forged.logits = ordinary.outputs[0];
    let fake = Native {
        node: ordinary.id,
        vars: 0,
        outputs: Vec::new(),
        ..spec.clone()
    };
    assert!(matches!(
        derive_composed_schedule(&forged, &[], &[fake]),
        Err(CompileError::Schedule { .. })
    ));
    for fault in 0..=9 {
        let mut n = spec.clone();
        match fault {
            0 => n.degree = 0,
            1 => n.degree = 8,
            2 => n.vars = 64,
            3 => n.vars = 0,
            4 => n.evaluations = 0,
            5 => n.node = tessera_ir::NodeId(u32::MAX),
            6 => {
                n.node = low
                    .graph
                    .nodes
                    .iter()
                    .find(|n| matches!(n.op, Op::ConstMul { .. }))
                    .unwrap()
                    .id
            }
            7 => n.node = layers.node(layers.position(node.id).unwrap().1, 1),
            8 => n.outputs[0].from = u32::MAX,
            9 => n.outputs[0].vars -= 1,
            _ => unreachable!(),
        }
        bad(&[], &[n]);
    }
}

#[test]
fn one_hot_index_is_removed_only_for_the_exact_native_token_binding() {
    use tessera_compiler::schedule::{Native, NativeOutput, derive_composed_schedule};
    let low = lowering(Profile::Servable, 3, 5, (2, 1), Sites::Pot);
    let node = low
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::OneHot { .. }))
        .unwrap();
    let variables = |t: TensorId| {
        low.graph.tensors[t.0 as usize]
            .shape
            .iter()
            .map(|d| d.next_power_of_two().trailing_zeros())
            .sum::<u32>()
    };
    let total = variables(node.outputs[0]);
    let time = variables(node.inputs[0]);
    let native = Native {
        tag: 97,
        node: node.id,
        vars: total,
        degree: 2,
        evaluations: 2,
        zero: true,
        outputs: vec![
            NativeOutput {
                tensor: node.outputs[0],
                from: 0,
                vars: total,
            },
            NativeOutput {
                tensor: node.inputs[0],
                from: total - time,
                vars: time,
            },
        ],
    };
    let plan = derive_composed_schedule(&low, &[(node.id, Some(0))], std::slice::from_ref(&native))
        .unwrap();
    let count = |relation| {
        plan.stages[1]
            .instances
            .iter()
            .filter(|i| i.relation == relation)
            .count()
    };
    assert_eq!(count(Relation::OneHotIndex), 0);
    for relation in [
        Relation::Booleanity,
        Relation::OneHotCount,
        Relation::OneHotVocabulary,
    ] {
        assert_eq!(count(relation), 1);
    }
    for fault in 0..8 {
        let mut bad = native.clone();
        match fault {
            0 => bad.zero = false,
            1 => bad.degree = 3,
            2 => bad.evaluations = 3,
            3 => bad.vars += 1,
            4 => bad.outputs.swap(0, 1),
            5 => bad.outputs[1].from = 0,
            6 => bad.outputs[0].tensor = low.tokens,
            7 => bad.outputs.pop().map(|_| ()).unwrap(),
            _ => unreachable!(),
        }
        assert!(
            matches!(
                derive_composed_schedule(&low, &[(node.id, Some(0))], &[bad]),
                Err(CompileError::Schedule { .. })
            ),
            "malformed token replacement {fault}"
        );
    }
}
