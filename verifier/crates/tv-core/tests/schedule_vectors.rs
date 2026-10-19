// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's stage schedule against `spec/vectors/schedule/schedule.json`, the compiler's
//! derivation of each conformance model's schedule at the lengths 1, 4, 5 and `max_context`
//! (`spec/vectors/schedule/README.md`). The schedule is a function of the graph alone, which the
//! prover and the verifier derive independently and must agree on (INV-SCH-1, RFC-0010
//! 'Invariants'): the verifier lowers each golden claim list from its fields, derives its
//! schedule, and compares it with the case field by field. The stage count, the leaf sites and
//! split values, each stage's rounds, zero-check coordinates and split values, and each
//! instance's relation kind, node, tensor, stacking, variables, point and layer point,
//! coordinate by coordinate, then the pool's claims.

mod claim_support;

use std::fmt::Write as _;

use claim_support::{goldens, integer, list, member, string, text};
use tv_codec::DecodeLimits;
use tv_core::claims::ClaimList;
use tv_core::lowering::lower;
use tv_core::schedule::{
    Class, Commitment, Layers, Part, Point, RelationKind, Route, Schedule, Source, TensorRef,
    derive,
};
use tv_core::validation::check_public;
use tv_testkit::{Json, read_vectors};

/// The differences found so far, reported together at the end.
#[derive(Default)]
struct Differences {
    report: String,
    count: usize,
}

impl Differences {
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, what: &str, expected: T, found: T) {
        if expected != found {
            self.count += 1;
            if self.count <= 40 {
                let _ = writeln!(
                    self.report,
                    "{what}: expected {expected:?}, found {found:?}"
                );
            }
        }
    }
}

fn number(value: &Json) -> i128 {
    integer(value)
}

/// A point's coordinates, written as the vectors write them.
fn coordinates(point: &Point) -> Vec<String> {
    let mut out = Vec::new();
    for run in point.runs() {
        for i in run.start..run.start + run.len {
            out.push(match run.source {
                Source::Leaf(c) => format!("leaf:{c}:{i}"),
                Source::Layer => format!("layer:{i}"),
                Source::Rounds(k) => format!("round:{k}:{i}"),
                Source::Zero(k) => {
                    assert_eq!(k, 1, "zero-checks run in stage 1");
                    format!("zero:{i}")
                }
            });
        }
    }
    out
}

fn strings(value: &Json) -> Vec<String> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|v| string(v).to_owned())
        .collect()
}

fn kind_name(kind: RelationKind) -> &'static str {
    match kind {
        RelationKind::MatMulWeight => "MatMulWeight",
        RelationKind::MatMulAct => "MatMulAct",
        RelationKind::Elementwise => "Elementwise",
        RelationKind::Rotate => "Rotate",
        RelationKind::SumSquares => "SumSquares",
        RelationKind::ZeroCheck => "ZeroCheck",
    }
}

fn class_name(class: Class) -> &'static str {
    match class {
        Class::Committed(Commitment::Advice) => "advice",
        Class::Committed(Commitment::Weight(_) | Commitment::ChannelParams) => "weight",
        Class::Virtual { .. } => "virtual",
    }
}

/// The pool's claims in the order the stages leave them: each ready committed claim and each
/// split's committed group, with its stage, commitment, atoms and point.
fn pool(schedule: &Schedule) -> Vec<(u16, &'static str, usize, Vec<String>)> {
    let mut by_stage: Vec<(u16, Vec<&tv_core::schedule::Claim>)> =
        vec![(0, schedule.leaf().iter().map(|c| &c.claim).collect())];
    for stage in schedule.stages() {
        by_stage.push((
            stage.index,
            stage
                .instances
                .iter()
                .flat_map(|i| i.evals.iter())
                .collect(),
        ));
    }
    let mut out = Vec::new();
    for (stage, claims) in by_stage {
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
            for group in groups {
                let lead = &claim.atoms[group[0]];
                out.push((
                    stage,
                    class_name(lead.class),
                    group.len(),
                    coordinates(&lead.point),
                ));
            }
        }
    }
    out
}

fn compare(case: &Json, schedule: &Schedule, d: &mut Differences) {
    let name = format!(
        "{} at n = {}",
        text(case, "model"),
        number(member(case, "n"))
    );
    d.check(
        &format!("{name}: the stage count"),
        number(member(case, "stage_count")) as usize,
        schedule.stage_count(),
    );
    d.check(
        &format!("{name}: the layers"),
        number(member(case, "layers")) as u32,
        schedule.layers(),
    );
    d.check(
        &format!("{name}: the layer variables"),
        number(member(case, "layer_vars")) as usize,
        schedule.layer_vars(),
    );

    // The leaf stage's sites, in order.
    let sites: Vec<Option<(u32, usize)>> = list(case, "sites")
        .iter()
        .map(|site| {
            if site.get("logits").is_some() {
                None
            } else {
                Some((
                    number(member(site, "node")) as u32,
                    number(member(site, "input")) as usize,
                ))
            }
        })
        .collect();
    let found: Vec<Option<(u32, usize)>> = schedule
        .leaf()
        .iter()
        .map(|c| c.input.map(|input| (c.site.0, input)))
        .collect();
    d.check(&format!("{name}: the leaf sites"), sites, found);

    let stages = list(case, "stages");
    d.check(
        &format!("{name}: the stages listed"),
        stages.len(),
        schedule.stage_count(),
    );
    let leaf = &stages[0];
    d.check(
        &format!("{name}: the leaf stage's split values"),
        number(member(leaf, "splits")) as usize,
        schedule.leaf_splits(),
    );
    for (expected, stage) in stages.iter().skip(1).zip(schedule.stages()) {
        let at = format!("{name}: stage {}", stage.index);
        d.check(
            &format!("{at}: its index"),
            number(member(expected, "index")) as u16,
            stage.index,
        );
        d.check(
            &format!("{at}: its rounds"),
            number(member(expected, "rounds")) as usize,
            stage.rounds,
        );
        d.check(
            &format!("{at}: its zero-check coordinates"),
            number(member(expected, "zero_vars")) as usize,
            stage.zero_vars,
        );
        d.check(
            &format!("{at}: its split values"),
            number(member(expected, "splits")) as usize,
            stage.splits,
        );
        let instances = list(expected, "instances");
        d.check(
            &format!("{at}: its instances"),
            instances.len(),
            stage.instances.len(),
        );
        for (i, (expected, instance)) in instances.iter().zip(&stage.instances).enumerate() {
            let what = |field: &str| format!("{at}: instance {i}: {field}");
            d.check(
                &what("relation"),
                text(expected, "relation"),
                kind_name(instance.relation.kind()),
            );
            d.check(
                &what("node"),
                number(member(expected, "node")) as u32,
                instance.node.0,
            );
            let tensor = match instance.tensor {
                TensorRef::Stacked(t) | TensorRef::Single(t) => Some(t.0),
                TensorRef::LayerInput => None,
            };
            let expected_tensor = member(expected, "tensor");
            d.check(
                &what("tensor"),
                (!expected_tensor.is_null()).then(|| number(expected_tensor) as u32),
                tensor,
            );
            d.check(
                &what("stacked"),
                matches!(member(expected, "stacked"), Json::Bool(true)),
                instance.stacked,
            );
            d.check(
                &what("vars"),
                number(member(expected, "vars")) as usize,
                instance.vars,
            );
            // A zero-check's point is its coordinates of the zero-check point.
            let point = if instance.relation.kind() == RelationKind::ZeroCheck {
                &instance.zero
            } else {
                &instance.point
            };
            d.check(
                &what("point"),
                strings(member(expected, "point")),
                coordinates(point),
            );
            let layers = match &instance.layers {
                Layers::Point(p) => Some(coordinates(p)),
                Layers::None | Layers::One(_) => None,
            };
            let expected_layers = member(expected, "layers");
            d.check(
                &what("layers"),
                (!expected_layers.is_null()).then(|| strings(expected_layers)),
                layers,
            );
        }
    }

    // The pool, in order.
    let expected: Vec<(u16, String, usize, Vec<String>)> = list(case, "pool")
        .iter()
        .map(|claim| {
            (
                number(member(claim, "stage")) as u16,
                text(claim, "commitment").to_owned(),
                number(member(claim, "atoms")) as usize,
                strings(member(claim, "point")),
            )
        })
        .collect();
    let found: Vec<(u16, String, usize, Vec<String>)> = pool(schedule)
        .into_iter()
        .map(|(stage, class, atoms, point)| (stage, class.to_owned(), atoms, point))
        .collect();
    d.check(
        &format!("{name}: the pool's claims"),
        expected.len(),
        found.len(),
    );
    for (i, (expected, found)) in expected.into_iter().zip(found).enumerate() {
        d.check(&format!("{name}: pool claim {i}"), expected, found);
    }
}

/// INV-SCH-1: every case of the compiler's schedules is the verifier's own derivation.
#[test]
fn every_schedule_of_the_vectors_is_reproduced() {
    let file = read_vectors("schedule/schedule.json");
    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 16, "four models at four lengths");
    let mut d = Differences::default();
    for case in cases {
        let model = text(case, "model");
        let n = number(member(case, "n")) as u32;
        let golden = goldens()
            .into_iter()
            .find(|g| g.name == model)
            .unwrap_or_else(|| panic!("no golden claim list for {model}"));
        let claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
        let public = check_public(&claims).unwrap();
        let graph = lower(&public, n).unwrap();
        let schedule = derive(&graph).unwrap_or_else(|e| panic!("{model} at n = {n}: {e}"));
        compare(case, &schedule, &mut d);
    }
    assert!(
        d.count == 0,
        "{} differences from the compiler's schedules; the first:\n{}",
        d.count,
        d.report
    );
}
