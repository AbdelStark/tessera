// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054/0018: count complete lookup-group construction, proof and output.

use super::super::*;
use common::{eq_weights, tv_fp2s, tv_proof};
use tessera_codec::Encode;
use tessera_logup::{
    BoundedLogUpError, LookupForestShape, LookupGroupMemory, LookupGroupShape, LookupInstance,
    ProvenGroup, prove_lookup_group,
};
use tessera_sumcheck::WindowError;
use tv_core::tree as tvt;
use tv_hash::transcript::Ascii;

struct NoAllocationTranscript;
impl SumcheckTranscript<Fp2> for NoAllocationTranscript {
    fn absorb_message(&mut self, _: &str, _: &[Fp2]) {}
    fn challenge(&mut self, _: &str) -> Fp2 {
        Fp2::from(Fp::from_u64_reduced(17))
    }
}

fn absorb(t: &mut tv_hash::transcript::Transcript, label: &str, values: &[Fp2]) {
    let mut bytes = Vec::new();
    values.to_vec().encode_to(&mut bytes);
    t.absorb(Ascii::new(label).unwrap(), &bytes);
}

// The public low-level tv-core driver exposes tree/ labels. Rename only the
// group's tree prefix on the prover; independently check every level and final
// polynomial, including root/leaf joins and all discarded intermediate debts.
fn verify(
    instances: &[LookupInstance<'_, Fp>],
    forests: &[TestForest],
    schedule: &[TreeId],
    alpha: Fp2,
    beta: Fp2,
) {
    let mut rng = SplitMix64::new(0x6750);
    let session = Session::random(&mut rng);
    let mut inner = session.transcript();
    let mut t = Recording {
        inner: &mut inner,
        labels: Vec::new(),
        rename: Some(("group/4294967295/tree/", "tree/")),
    };
    let views: Vec<_> = forests.iter().map(TestForest::forest).collect();
    let proven =
        prove_lookup_group(&mut t, u32::MAX, alpha, beta, instances, &views, schedule).unwrap();
    let mut tv = session.tv_transcript();
    absorb(&mut tv, "group/4294967295/logup/roots", proven.roots());
    let heights: Vec<_> = schedule
        .iter()
        .map(|id| match *id {
            TreeId::Lookup(k) => height(instances[k].lookups.len()),
            TreeId::Forest(k) => forests[k].height,
            TreeId::Table(_) => panic!("no table tree"),
        })
        .collect();
    let tallest = heights.iter().copied().max().unwrap_or(0);
    assert_eq!(proven.levels().len(), tallest);
    let mut roots = proven.roots().chunks_exact(2);
    let mut claims: Vec<_> = schedule
        .iter()
        .map(|id| match id {
            TreeId::Lookup(_) => {
                let root = roots.next().unwrap();
                assert_ne!(root[1], Fp2::ZERO);
                Some(tvt::LevelClaim::Fraction(tvt::FractionClaim {
                    point: vec![],
                    numerator: tv_fp2(root[0]),
                    denominator: tv_fp2(root[1]),
                }))
            }
            _ => None,
        })
        .collect();
    for (level, message) in proven.levels().iter().enumerate() {
        let joins = |p: usize| tallest - heights[p] == level;
        let s = schedule
            .iter()
            .enumerate()
            .filter(|(p, _)| joins(*p))
            .map(|(_, id)| match id {
                TreeId::Forest(k) => forests[*k].trees_log,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let root: Vec<_> = (0..s)
            .map(|c| {
                tv.challenge_fp2(Ascii::new(&format!("tree/{level}/root/{c}")).unwrap())
                    .unwrap()
            })
            .collect();
        for (p, id) in schedule.iter().enumerate().filter(|(p, _)| joins(*p)) {
            if let TreeId::Forest(k) = id {
                let point = root[s - forests[*k].trees_log..].to_vec();
                let value = forests[*k]
                    .roots()
                    .iter()
                    .zip(eq_weights(&point))
                    .fold(tv_field::Fp2::ZERO, |a, (&v, e)| a + v * e);
                claims[p] = Some(tvt::LevelClaim::Product(tvt::TreeClaim { point, value }));
            }
        }
        let present: Vec<_> = (0..schedule.len())
            .filter(|&p| tallest - heights[p] <= level)
            .collect();
        let entering: Vec<_> = present
            .iter()
            .map(|&p| claims[p].clone().unwrap())
            .collect();
        let checked =
            tvt::verify_level(&mut tv, level as u32, &entering, &tv_proof(message.proof()))
                .unwrap();
        for (&p, claim) in present.iter().zip(checked.claims) {
            claims[p] = Some(claim);
        }
    }
    for (id, claim) in schedule.iter().zip(claims) {
        match (id, claim.unwrap()) {
            (TreeId::Lookup(k), tvt::LevelClaim::Fraction(claim)) => {
                let i = &instances[*k];
                let h = height(i.lookups.len());
                let key = mle(&padded(i.lookups.keys, i.padding_key, h), &claim.point);
                let value = i.lookups.values.map_or(tv_field::Fp2::ZERO, |v| {
                    mle(&padded(v, i.padding_value.unwrap(), h), &claim.point)
                });
                assert_eq!(claim.numerator, tv_field::Fp2::ONE);
                assert_eq!(
                    claim.denominator,
                    tv_fp2(alpha) - key - tv_fp2(beta) * value
                );
                let returned = &proven.claims()[*k];
                assert_eq!(tv_fp2s(&returned.key.point), claim.point);
                assert_eq!(tv_fp2(returned.key.value), key);
                assert_eq!(returned.value.is_some(), i.lookups.has_values());
                if let Some(v) = &returned.value {
                    assert_eq!(tv_fp2s(&v.point), claim.point);
                    assert_eq!(tv_fp2(v.value), value);
                }
            }
            (TreeId::Forest(k), tvt::LevelClaim::Product(claim)) => {
                let leaves: Vec<_> = forests[*k]
                    .leaves
                    .iter()
                    .map(|&x| tv_fp(x).into())
                    .collect();
                assert_eq!(claim.value, mle(&leaves, &claim.point));
                let returned = &proven.forest_claims()[*k];
                assert_eq!(tv_fp2s(&returned.point), claim.point);
                assert_eq!(tv_fp2(returned.value), claim.value);
            }
            _ => panic!("tree kind"),
        }
    }
    absorb(&mut tv, "group/4294967295/logup/values", proven.values());
    assert_eq!(*tv.state(), inner.state());
}

fn check(lookups: &[LookupGroupShape], forest_shapes: &[LookupForestShape], reverse: bool) {
    let mut rng = SplitMix64::new(0x6751);
    let columns: Vec<_> = lookups
        .iter()
        .map(|s| (rng.fps(s.rows), s.values.then(|| rng.fps(s.rows))))
        .collect();
    let instances: Vec<_> = columns
        .iter()
        .map(|(k, v)| LookupInstance {
            lookups: Tuples::new(k, v.as_deref()),
            padding_key: Fp::from_u64_reduced(7),
            padding_value: v.as_ref().map(|_| Fp::from_u64_reduced(11)),
        })
        .collect();
    let forests: Vec<_> = forest_shapes
        .iter()
        .map(|s| TestForest::random(&mut rng, s.trees_log, s.height))
        .collect();
    let views: Vec<_> = forests.iter().map(TestForest::forest).collect();
    let mut schedule: Vec<_> = (0..instances.len())
        .map(TreeId::Lookup)
        .chain((0..forests.len()).map(TreeId::Forest))
        .collect();
    if reverse {
        schedule.reverse();
    }
    let mut memory = None;
    let planning = allocation_counter::measure(|| {
        memory = Some(LookupGroupMemory::new::<Fp2>(lookups, forest_shapes, &schedule).unwrap())
    });
    let memory = memory.unwrap();
    assert!(
        planning.bytes_max as u64 <= memory.planning_bytes(),
        "planning {planning:?} > {memory:?}"
    );
    assert_eq!(planning.bytes_current, 0);
    let (alpha, beta) = (
        Fp2::from(Fp::from_u64_reduced(37)),
        Fp2::from(Fp::from_u64_reduced(23)),
    );
    let mut proven: Option<ProvenGroup<Fp2>> = None;
    let run = allocation_counter::measure(|| {
        proven = Some(
            prove_lookup_group(
                &mut NoAllocationTranscript,
                u32::MAX,
                alpha,
                beta,
                &instances,
                &views,
                &schedule,
            )
            .unwrap(),
        )
    });
    assert!(
        run.bytes_max as u64 <= memory.peak_bytes(),
        "complete group {lookups:?}/{forest_shapes:?}: {run:?} > {memory:?}"
    );
    assert!(
        run.bytes_current as u64 <= memory.output_bytes(),
        "returned group {lookups:?}/{forest_shapes:?}: {run:?} > {memory:?}"
    );
    drop(proven);
    verify(&instances, &forests, &schedule, alpha, beta);
}

#[test]
fn public_lookup_group_memory_covers_empty_and_zero_height_groups() {
    check(&[], &[], false);
    for n in [1, 3, 4, 17, 128] {
        for values in [false, true] {
            check(&vec![LookupGroupShape { rows: 1, values }; n], &[], false);
        }
    }
}

#[test]
fn public_lookup_group_memory_covers_padded_values_and_mixed_schedules() {
    for rows in [2, 3, 4, 17, 1023, 1024, 1025, 4096] {
        for values in [false, true] {
            check(&[LookupGroupShape { rows, values }], &[], false);
        }
    }
    let lookups: Vec<_> = (0..17)
        .map(|i| LookupGroupShape {
            rows: 1 << (i % 8),
            values: i % 2 == 0,
        })
        .collect();
    let forests = [
        LookupForestShape {
            height: 1,
            trees_log: 12,
        },
        LookupForestShape {
            height: 7,
            trees_log: 2,
        },
        LookupForestShape {
            height: 3,
            trees_log: 0,
        },
    ];
    for reverse in [false, true] {
        check(&lookups, &forests, reverse);
        check(&[], &forests, reverse);
    }
}

#[test]
fn public_lookup_group_memory_refuses_empty_lookups_and_invalid_schedules() {
    let one = [LookupGroupShape {
        rows: 1,
        values: false,
    }];
    assert_eq!(
        LookupGroupMemory::new::<Fp2>(
            &[LookupGroupShape {
                rows: 0,
                values: false
            }],
            &[],
            &[TreeId::Lookup(0)]
        ),
        Err(BoundedLogUpError::LogUp(LogUpError::NoLookups {
            instance: 0
        }))
    );
    assert_eq!(
        LookupGroupMemory::new::<Fp2>(&one, &[], &[]),
        Err(BoundedLogUpError::LogUp(LogUpError::ScheduleLength {
            len: 0,
            expected: 1
        }))
    );
    for id in [TreeId::Lookup(1), TreeId::Forest(0), TreeId::Table(0)] {
        assert_eq!(
            LookupGroupMemory::new::<Fp2>(&one, &[], &[id]),
            Err(BoundedLogUpError::LogUp(LogUpError::ScheduleTree {
                position: 0
            }))
        );
    }
    assert_eq!(
        LookupGroupMemory::new::<Fp2>(&[one[0]; 2], &[], &[TreeId::Lookup(0); 2]),
        Err(BoundedLogUpError::LogUp(LogUpError::ScheduleTree {
            position: 1
        }))
    );
}

#[test]
fn public_lookup_group_memory_refuses_bad_forest_and_unaddressable_geometry() {
    for shape in [
        LookupForestShape {
            height: 0,
            trees_log: 0,
        },
        LookupForestShape {
            height: 1,
            trees_log: usize::MAX,
        },
    ] {
        assert!(matches!(
            LookupGroupMemory::new::<Fp2>(&[], &[shape], &[TreeId::Forest(0)]),
            Err(BoundedLogUpError::Window(WindowError::Shape(_)))
        ));
    }
    assert_eq!(
        LookupGroupMemory::new::<Fp2>(
            &[LookupGroupShape {
                rows: usize::MAX,
                values: true
            }],
            &[],
            &[TreeId::Lookup(0)]
        ),
        Err(BoundedLogUpError::Window(WindowError::Shape(
            "resident tree variables fit an index"
        )))
    );
    assert_eq!(
        LookupGroupMemory::new::<Fp2>(
            &[LookupGroupShape {
                rows: 1usize << (usize::BITS - 1),
                values: true
            }],
            &[],
            &[TreeId::Lookup(0)]
        ),
        Err(BoundedLogUpError::Window(WindowError::Allocation))
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn public_lookup_group_memory_refuses_aggregate_overflow_without_field_tables() {
    let lookups = vec![
        LookupGroupShape {
            rows: 1usize << 50,
            values: true
        };
        512
    ];
    let schedule: Vec<_> = (0..lookups.len()).map(TreeId::Lookup).collect();
    let run = allocation_counter::measure(|| {
        assert_eq!(
            LookupGroupMemory::new::<Fp2>(&lookups, &[], &schedule),
            Err(BoundedLogUpError::Window(WindowError::Allocation))
        )
    });
    assert!(
        run.bytes_max
            <= 512 * (size_of::<usize>() + size_of::<tessera_sumcheck::ResidentTreeShape>()) as u64
    );
}
