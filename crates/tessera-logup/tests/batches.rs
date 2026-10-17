// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Batches of LogUp instances of different heights in one sumcheck per level (RFC-0008
//! 'Batching'; RFC-0010 'Tree levels'), through the independent verifier.
//!
//! - Batches of 1 to 20 instances verify, their trees of every height from 4 to 24 between them,
//!   each joining the batch at the level where its height matches.
//! - A prover that drops one instance from the levels is rejected at the first level where that
//!   instance is present: the verifier's batch holds every instance, and the proof's final
//!   evaluations do not.

mod common;

use common::{
    Fp, Fp2, Messages, Rejected, Session, SplitMix64, TestInstance, TestTable, prove_and_verify,
    tv_fp2s, tv_proof, verify,
};
use tessera_field::Field;
use tessera_logup::TreeId;
use tessera_sumcheck::{FractionTree, LeafNumerators, SumcheckTranscript, Tree, prove_trees};
use tv_core::sumcheck::SumcheckError;

/// An instance whose lookup tree has height `lookups_log` and whose table tree has height
/// `rows_log`, with random lookups and values when `with_values`.
fn shaped(
    rng: &mut SplitMix64,
    rows_log: u32,
    lookups_log: u32,
    with_values: bool,
) -> TestInstance {
    let table = TestTable::random(rng, 1 << rows_log, with_values);
    TestInstance::random(rng, table, 1 << lookups_log)
}

/// Batches of 1 to 20 instances: every lookup height from 4 to 23 in the batch of 20, a table of
/// `2^24` rows alone, and batches of 2 to 13 instances of mixed heights between. Each instance's
/// two trees join where their heights match, so the levels hold from 1 to 40 trees.
#[test]
fn batches_of_one_to_twenty_instances_of_heights_four_to_twenty_four_verify() {
    let mut rng = SplitMix64::new(0x105_0001);
    let session = Session::random(&mut rng);
    // One instance: a table tree of height 24, beside a lookup tree of height 4.
    let alone = [shaped(&mut rng, 24, 4, false)];
    let proven = prove_and_verify(&session, &alone, &[], &TreeId::paired(1));
    assert_eq!(proven.levels().len(), 24);
    // Twenty instances: lookup trees of heights 4 to 23, one each, and table trees of 4 to 12.
    let twenty: Vec<TestInstance> = (0..20u32)
        .map(|k| shaped(&mut rng, 4 + k % 9, 4 + k, k % 2 == 0))
        .collect();
    let proven = prove_and_verify(&session, &twenty, &[], &TreeId::paired(20));
    assert_eq!(proven.levels().len(), 23);
    assert_eq!(proven.values().len(), 10, "the ten instances with values");
    // Batches of 2, 3, 5, 8 and 13 instances, of heights 4 to 16 at random.
    for m in [2usize, 3, 5, 8, 13] {
        let batch: Vec<TestInstance> = (0..m)
            .map(|_| {
                let (rows_log, lookups_log) = (4 + rng.below(13) as u32, 4 + rng.below(13) as u32);
                let with_values = rng.coin();
                shaped(&mut rng, rows_log, lookups_log, with_values)
            })
            .collect();
        let _ = prove_and_verify(&session, &batch, &[], &TreeId::paired(m));
    }
}

/// The two fraction trees of an instance as RFC-0008 'Fraction tree' builds them: the lookup
/// tree's leaves `(1, α − (a + β·v))`, the table tree's `(−m_j, α − (t_j + β·v_j))`, both padded
/// with row 0.
fn fraction_trees(instance: &TestInstance, alpha: Fp2, beta: Fp2) -> [FractionTree<Fp2>; 2] {
    let table = &instance.table;
    let value = |values: &Option<Vec<Fp>>, x: usize| {
        values
            .as_ref()
            .map_or(Fp2::ZERO, |v| beta * Fp2::from(v[x]))
    };
    let row_0 = Fp2::from(table.keys[0]) + value(&table.values, 0);
    let lookups = instance.keys.len();
    let lookup = FractionTree::new(instance.lookup_height(), LeafNumerators::Unit, |x| {
        match x < lookups {
            true => alpha - Fp2::from(instance.keys[x]) - value(&instance.values, x),
            false => alpha - row_0,
        }
    })
    .expect("a lookup tree");
    let negated: Vec<Fp> = instance.multiplicities.iter().map(|&m| -m).collect();
    let rows = table.rows();
    let table_tree = FractionTree::new(
        instance.table_height(),
        LeafNumerators::Base(&negated),
        |x| match x < rows {
            true => alpha - Fp2::from(table.keys[x]) - value(&table.values, x),
            false => alpha - row_0,
        },
    )
    .expect("a table tree");
    [lookup, table_tree]
}

/// The messages of a prover that keeps instance `dropped` out of the levels: honest challenges,
/// the honest roots of every instance, then the levels of every other tree, proven on the same
/// transcript.
fn dropping(session: &Session, instances: &[TestInstance], dropped: usize) -> Messages {
    let mut transcript = session.transcript();
    let alpha: Fp2 = SumcheckTranscript::<Fp2>::challenge(&mut transcript, "logup/alpha");
    let beta: Fp2 = SumcheckTranscript::<Fp2>::challenge(&mut transcript, "logup/beta");
    let mut roots = Vec::new();
    let mut kept = Vec::new();
    for (k, instance) in instances.iter().enumerate() {
        for tree in fraction_trees(instance, alpha, beta) {
            let (numerator, denominator) = tree.root();
            roots.extend([numerator, denominator]);
            if k != dropped {
                kept.push(Tree::Fraction(tree));
            }
        }
    }
    SumcheckTranscript::<Fp2>::absorb_message(&mut transcript, "logup/roots", &roots);
    let proven = prove_trees(&mut transcript, kept).expect("well-formed trees");
    Messages {
        roots: tv_fp2s(&roots),
        levels: proven
            .levels()
            .iter()
            .map(|level| tv_proof(level.proof()))
            .collect(),
        values: Vec::new(),
    }
}

/// Dropping one instance from the batch is rejected at the first level where its taller tree is
/// present, for an instance that joins at an inner level and for one that joins near the leaves.
/// The levels before it are the honest ones, and the verifier accepts them.
#[test]
fn dropping_one_instance_from_the_batch_is_rejected() {
    let mut rng = SplitMix64::new(0x105_0002);
    let session = Session::random(&mut rng);
    // Lookup heights 9, 6 and 3, table heights 5, 2 and 1: the tallest tree is instance 0's.
    let instances = [
        shaped(&mut rng, 5, 9, false),
        shaped(&mut rng, 2, 6, false),
        shaped(&mut rng, 1, 3, false),
    ];
    let schedule = TreeId::paired(instances.len());
    // The honest stage verifies.
    let _ = prove_and_verify(&session, &instances, &[], &schedule);
    for dropped in [1usize, 2] {
        let messages = dropping(&session, &instances, dropped);
        let tallest = 9;
        let level = tallest - instances[dropped].lookup_height();
        // Present at that level: every tree whose height reaches it, four evaluations each.
        let present = instances
            .iter()
            .flat_map(|i| [i.lookup_height(), i.table_height()])
            .filter(|&h| tallest - h <= level)
            .count();
        let expected = 4 * present;
        assert_eq!(
            verify(&session, &instances, &[], &schedule, &messages).map(|_| ()),
            Err(Rejected::Level(
                level,
                SumcheckError::EvaluationCount {
                    round: level as u32,
                    expected,
                    found: expected - 4,
                }
            )),
            "instance {dropped} dropped"
        );
    }
}
