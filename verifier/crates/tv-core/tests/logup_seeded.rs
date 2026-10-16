// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-core`'s LogUp checks on seeded leaf stages proven by the test kit's reference prover,
//! which shares no code with `tv-core`: shapes and cheating provers the vectors of
//! `spec/vectors/sumcheck/` cannot all hold (RFC-0008 'Testing Strategy').
//!
//! - Batches of one to three instances of 1 to 17 lookups into tables of 1 to 17 rows, with and
//!   without values, beside zero to two product forests, in a random schedule order, verify;
//!   every value the prover knows is what the verifier derives, and every claim left is true.
//! - Sizes `2^k − 1`, `2^k` and `2^k + 1` on both sides, and trees of `2^0` to `2^10` leaves,
//!   verify.
//! - Each cheating prover of RFC-0008 'Testing Strategy' is rejected at the check that catches
//!   it: a lookup outside the table, a wrong value and a multiplicity off by one at the roots;
//!   a zero root denominator on either side at the roots; balanced false roots at the level where
//!   their tree joins; a lookup's leaf numerator set to 0 at the leaf checks; and a table with
//!   one row changed at the table evaluation.
//! - The table tree's padding rows are copies of row 0 (RFC-0008 'Fraction tree'): row 0's count
//!   moved to padding rows verifies; a lookup of `(0, 0)` paid for by a padding multiplicity, the
//!   phantom row, is rejected at the roots, and by a prover that pads with key and value 0 at the
//!   table evaluation.
//! - A leaf stage without LogUp instance absorbs its empty roots and values and verifies, with
//!   forests or with no tree at all.

mod support;

use support::Forest;
use support::from_reference;
use support::logup::{
    FalseClaim, Instance, Kind, LogUpOutcome, LogUpProof, LogUpRun, Lookup, TreeRef, below,
    claim_of, heights, proof_of_stage, random_forest as forest, random_instance as instance,
    random_schedule as schedule, reference_forest, reference_logup, reference_trees, run_logup,
};
use tv_core::logup::LookupError;
use tv_core::sumcheck::SumcheckError;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::SplitMix64;
use tv_testkit::logup::{LeafStage, LogUp, Tamper, prove_leaf_stage};
use tv_testkit::sumcheck::{self as kit, Fe};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x62; 32],
        session_id: *b"logup-gkr seeded",
    }
}

/// A leaf stage, its prover's transcript and its messages.
struct Proven {
    instances: Vec<Instance>,
    forests: Vec<Forest>,
    trees: Vec<TreeRef>,
    stage: LeafStage,
    prover: kit::Transcript,
}

impl Proven {
    /// The reference prover's leaf stage of `instances` as the prover holds them, `logups`,
    /// with `tamper`.
    fn new(
        instances: Vec<Instance>,
        logups: &[LogUp],
        forests: Vec<Forest>,
        trees: Vec<TreeRef>,
        tamper: impl FnMut(Tamper<'_>),
    ) -> Self {
        let reference_forests: Vec<_> = forests.iter().map(reference_forest).collect();
        let mut prover = kit::Transcript::new(&context().to_tce());
        let stage = prove_leaf_stage(
            &mut prover,
            logups,
            &reference_forests,
            &reference_trees(&trees),
            tamper,
        );
        Self {
            instances,
            forests,
            trees,
            stage,
            prover,
        }
    }

    fn honest(instances: Vec<Instance>, forests: Vec<Forest>, trees: Vec<TreeRef>) -> Self {
        let logups: Vec<LogUp> = instances.iter().map(reference_logup).collect();
        Self::new(instances, &logups, forests, trees, |_| {})
    }

    fn proof(&self) -> LogUpProof {
        proof_of_stage(&self.stage)
    }

    fn run(&self, proof: &LogUpProof) -> LogUpRun {
        let mut transcript = Transcript::new(&context());
        run_logup(
            &mut transcript,
            &self.instances,
            &self.forests,
            &self.trees,
            proof,
            |_, _| {},
        )
    }
}

/// The run of an honest stage verifies and reproduces every value the prover knows.
fn assert_reproduces(proven: &Proven, what: &str) {
    let run = proven.run(&proven.proof());
    assert_eq!(run.outcome, LogUpOutcome::Accepted, "{what}");
    let roots = run.roots.as_ref().unwrap();
    assert_eq!(roots.challenges.alpha, from_reference(proven.stage.alpha));
    assert_eq!(roots.challenges.beta, from_reference(proven.stage.beta));
    assert_eq!(run.levels.len(), proven.stage.levels.len(), "{what}");
    for (index, (ran, level)) in run.levels.iter().zip(&proven.stage.levels).enumerate() {
        let what = format!("{what}, level {index}");
        let challenges =
            |values: &[Fe]| -> Vec<Fp2> { values.iter().copied().map(from_reference).collect() };
        assert_eq!(
            ran.root_challenges,
            challenges(&level.root_challenges),
            "{what}"
        );
        let entering: Vec<_> = level
            .entering
            .iter()
            .map(|claim| (claim.tree, claim_of(claim)))
            .collect();
        assert_eq!(ran.entering, entering, "{what}: the claims entering it");
        assert_eq!(
            ran.outcome.gamma,
            from_reference(level.proof.gamma),
            "{what}"
        );
        assert_eq!(
            ran.outcome.lambda,
            level.lambda.map(from_reference),
            "{what}"
        );
        assert_eq!(
            ran.outcome.point,
            challenges(&level.proof.challenges),
            "{what}"
        );
        assert_eq!(ran.outcome.rho, from_reference(level.rho), "{what}");
        let next: Vec<_> = level.next.iter().map(claim_of).collect();
        assert_eq!(ran.outcome.claims, next, "{what}: the next claims");
    }
    let leaves: Vec<_> = proven.stage.leaves.iter().map(claim_of).collect();
    assert_eq!(run.leaves, leaves, "{what}: the leaf claims");
    assert_eq!(
        run.state,
        Some(proven.prover.state()),
        "{what}: the transcripts"
    );
    // A tree of height h joins at level H − h, and every level holds the trees that have joined.
    let heights = heights(&proven.instances, &proven.forests, &proven.trees);
    let tallest = heights.iter().copied().max().unwrap_or(0);
    for (index, ran) in run.levels.iter().enumerate() {
        let present: Vec<usize> = (0..proven.trees.len())
            .filter(|&t| heights[t] > 0 && tallest - heights[t] <= index)
            .collect();
        let entered: Vec<usize> = ran.entering.iter().map(|(t, _)| *t).collect();
        assert_eq!(entered, present, "{what}, level {index}");
        let fractions = present
            .iter()
            .any(|&t| proven.trees[t].kind != Kind::Forest);
        assert_eq!(
            ran.outcome.lambda.is_some(),
            fractions,
            "{what}, level {index}"
        );
    }
}

#[test]
fn seeded_leaf_stages_verify_and_reproduce_every_value_the_prover_knows() {
    let mut rng = SplitMix64::new(0x5eed_0062_0001);
    let mut without_lambda = 0;
    let mut zero_height = 0;
    for case in 0..120 {
        let count = 1 + below(&mut rng, 3);
        let instances: Vec<Instance> = (0..count)
            .map(|_| {
                let (rows, lookups) = (1 + below(&mut rng, 17), 1 + below(&mut rng, 17));
                let has_values = rng.next_u64() & 1 == 0;
                instance(&mut rng, rows, lookups, has_values)
            })
            .collect();
        let forests: Vec<Forest> = (0..below(&mut rng, 3))
            .map(|_| {
                let (s, h) = (below(&mut rng, 3), 1 + below(&mut rng, 5));
                forest(&mut rng, s, h)
            })
            .collect();
        let trees = schedule(&mut rng, instances.len(), forests.len());
        let proven = Proven::honest(instances, forests, trees);
        assert_reproduces(&proven, &format!("case {case}"));
        without_lambda += proven
            .stage
            .levels
            .iter()
            .filter(|level| level.lambda.is_none())
            .count();
        zero_height += heights(&proven.instances, &proven.forests, &proven.trees)
            .iter()
            .zip(&proven.trees)
            .filter(|(h, tree)| **h == 0 && tree.kind != Kind::Forest)
            .count();
    }
    // Forests taller than every fraction tree make levels of forests alone, and single lookups
    // and single rows make trees that join no level.
    assert!(without_lambda > 10, "{without_lambda}");
    assert!(zero_height > 10, "{zero_height}");
}

/// RFC-0008 'Testing Strategy': sizes `2^k − 1`, `2^k` and `2^k + 1` on both sides, and lookup
/// trees of `2^0` to `2^10` leaves beside table trees of `2^0` to `2^8`, joining where their
/// heights match.
#[test]
fn padding_sizes_and_heights_verify() {
    let mut rng = SplitMix64::new(0x5eed_0062_0002);
    let sizes: Vec<usize> = (0..=5)
        .flat_map(|k| [(1usize << k).saturating_sub(1).max(1), 1 << k, (1 << k) + 1])
        .collect();
    for &rows in &sizes {
        for &lookups in &sizes {
            let has_values = (rows + lookups) % 2 == 0;
            let instances = vec![instance(&mut rng, rows, lookups, has_values)];
            let trees = schedule(&mut rng, 1, 0);
            let proven = Proven::honest(instances, Vec::new(), trees);
            assert_reproduces(&proven, &format!("{lookups} lookups into {rows} rows"));
        }
    }
    for (lookups, rows) in [(1 << 10, 1), (1, 1 << 8), (1 << 10, 1 << 8), (1000, 3)] {
        let instances = vec![
            instance(&mut rng, rows, lookups, true),
            instance(&mut rng, 3, 2, false),
        ];
        let forests = vec![forest(&mut rng, 2, 3)];
        let trees = schedule(&mut rng, 2, 1);
        let proven = Proven::honest(instances, forests, trees);
        assert_reproduces(&proven, &format!("{lookups} lookups into {rows} rows"));
    }
}

/// A batch of three instances and a forest for the cheating provers: instance 1 has values.
fn base(rng: &mut SplitMix64) -> (Vec<Instance>, Vec<Forest>, Vec<TreeRef>) {
    let instances = vec![
        instance(rng, 5, 6, false),
        instance(rng, 8, 11, true),
        instance(rng, 3, 2, true),
    ];
    let forests = vec![forest(rng, 1, 2)];
    let trees = schedule(rng, instances.len(), forests.len());
    (instances, forests, trees)
}

/// The run of a cheating prover's stage: the verifier holds `instances`, the prover `logups`,
/// and `tamper` changes what the prover computes before it uses or sends it.
fn cheat(
    instances: &[Instance],
    logups: &[LogUp],
    forests: &[Forest],
    trees: &[TreeRef],
    tamper: impl FnMut(Tamper<'_>),
) -> LogUpRun {
    let proven = Proven::new(
        instances.to_vec(),
        logups,
        forests.to_vec(),
        trees.to_vec(),
        tamper,
    );
    proven.run(&proven.proof())
}

/// The prover's own copy of `instances`.
fn held(instances: &[Instance]) -> Vec<LogUp> {
    instances.iter().map(reference_logup).collect()
}

/// The level at which instance `i`'s tree of `kind` joins.
fn joins_at(
    instances: &[Instance],
    forests: &[Forest],
    trees: &[TreeRef],
    kind: Kind,
    i: usize,
) -> u32 {
    let heights = heights(instances, forests, trees);
    let tallest = heights.iter().copied().max().unwrap();
    let t = trees
        .iter()
        .position(|tree| *tree == TreeRef { kind, index: i })
        .unwrap();
    (tallest - heights[t]) as u32
}

/// RFC-0008 'Testing Strategy', 'Soundness': one lookup outside the table, one wrong value, one
/// multiplicity off by one, each with honest trees over what the prover holds, are rejected at
/// the roots: the lookups' and the rows' fractions do not cancel.
#[test]
fn a_false_lookup_or_multiplicity_with_honest_trees_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x5eed_0062_0003);
    for case in 0..18 {
        let (instances, forests, trees) = base(&mut rng);
        let i = case % 3;
        let mut logups = held(&instances);
        let lookup = below(&mut rng, instances[i].lookups.len());
        let row = below(&mut rng, instances[i].rows.len());
        // The lookup's key moved outside the table, its value wrong, or a row's multiplicity
        // off by one.
        match case / 3 % 3 {
            0 => logups[i].lookups[lookup].0 = logups[i].lookups[lookup].0 + Fe::ONE,
            1 => logups[i].lookups[lookup].1 = logups[i].lookups[lookup].1 + Fe::ONE,
            _ => logups[i].multiplicities[row] = logups[i].multiplicities[row] + Fe::ONE,
        }
        let run = cheat(&instances, &logups, &forests, &trees, |_| {});
        assert_eq!(
            run.outcome,
            LogUpOutcome::Roots(LookupError::Unbalanced { instance: i }),
            "case {case}"
        );
    }
}

/// A zero root denominator in the message, on either side, is rejected at the roots, before
/// any level: the levels' honest proof never runs.
#[test]
fn a_zero_root_denominator_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x5eed_0062_0004);
    let (instances, forests, trees) = base(&mut rng);
    for i in 0..instances.len() {
        for (side, expected) in [
            (1, LookupError::LookupDenominator { instance: i }),
            (3, LookupError::TableDenominator { instance: i }),
        ] {
            // The numerator set to zero too, so that the sum alone would pass.
            let run = cheat(&instances, &held(&instances), &forests, &trees, |tamper| {
                if let Tamper::Roots(roots) = tamper {
                    roots[4 * i + side] = Fe::ZERO;
                    roots[4 * i + side - 1] = Fe::ZERO;
                }
            });
            assert_eq!(
                run.outcome,
                LogUpOutcome::Roots(expected),
                "instance {i}, side {side}"
            );
            assert!(run.levels.is_empty());
        }
    }
}

/// False roots that sum to zero, with honest trees, pass the roots and are rejected at the level
/// where their tree joins, at its final check: the level's sumcheck proves the trees' true sums.
#[test]
fn balanced_false_roots_are_rejected_where_their_tree_joins() {
    let mut rng = SplitMix64::new(0x5eed_0062_0005);
    let (instances, forests, trees) = base(&mut rng);
    for i in 0..instances.len() {
        for (side, kind) in [(0, Kind::Lookup), (2, Kind::Table)] {
            // Scaling a root's numerator and denominator keeps its fraction, and the balance.
            let run = cheat(&instances, &held(&instances), &forests, &trees, |tamper| {
                if let Tamper::Roots(roots) = tamper {
                    roots[4 * i + side] = roots[4 * i + side] * Fe::base(2);
                    roots[4 * i + side + 1] = roots[4 * i + side + 1] * Fe::base(2);
                }
            });
            let level = joins_at(&instances, &forests, &trees, kind, i);
            let height = heights(&instances, &forests, &trees)[trees
                .iter()
                .position(|t| *t == TreeRef { kind, index: i })
                .unwrap()];
            if height == 0 {
                // A single leaf joins no level: its root is its leaf, which the leaf checks
                // see.
                assert!(
                    matches!(run.outcome, LogUpOutcome::Leaves(_)),
                    "{:?}",
                    run.outcome
                );
                continue;
            }
            let LogUpOutcome::Level { level: at, error } = &run.outcome else {
                panic!("instance {i}, {kind:?}: {:?}", run.outcome);
            };
            assert_eq!(*at, level, "instance {i}, {kind:?}");
            assert!(
                matches!(error, SumcheckError::FinalMismatch { .. }),
                "{error}"
            );
        }
    }
}

/// A lookup outside the table whose leaf numerator the prover set to 0, its row's multiplicity
/// lowered by one: the roots balance and every level passes, and the leaf check of the lookup
/// tree's numerators rejects it.
#[test]
fn a_bad_lookup_with_a_zero_numerator_is_rejected_at_the_leaf_numerator() {
    let mut rng = SplitMix64::new(0x5eed_0062_0006);
    for case in 0..9 {
        let (instances, forests, trees) = base(&mut rng);
        let i = case % 3;
        let lookup = below(&mut rng, instances[i].lookups.len());
        let Lookup::Row(row) = instances[i].lookups[lookup] else {
            unreachable!("an honest lookup is a row")
        };
        let run = cheat(
            &instances,
            &held(&instances),
            &forests,
            &trees,
            |tamper| match tamper {
                Tamper::LookupLeaves(at, leaves) if at == i => {
                    leaves[lookup] = (Fe::ZERO, leaves[lookup].1 + Fe::ONE);
                }
                Tamper::TableLeaves(at, leaves) if at == i => {
                    // The row no longer counts the lookup: n = −m_j rises by one.
                    leaves[row].0 = leaves[row].0 + Fe::ONE;
                }
                _ => {}
            },
        );
        assert_eq!(
            run.outcome,
            LogUpOutcome::Leaves(LookupError::LeafNumerator { instance: i }),
            "case {case}"
        );
    }
}

/// A prover that used another table, one row changed, with honest trees over it: the roots
/// balance, every level passes and the lookup tree's numerators are 1, and the table evaluation,
/// which the verifier computes from the true table, rejects it.
#[test]
fn a_table_with_one_row_changed_is_rejected_at_the_table_evaluation() {
    let mut rng = SplitMix64::new(0x5eed_0062_0007);
    for case in 0..9 {
        let (instances, forests, trees) = base(&mut rng);
        let i = case % 3;
        let mut changed = instances.clone();
        let row = below(&mut rng, changed[i].rows.len());
        changed[i].rows[row].0 += Fp::ONE;
        let run = cheat(&instances, &held(&changed), &forests, &trees, |_| {});
        assert_eq!(
            run.outcome,
            LogUpOutcome::Leaves(LookupError::TableEvaluation { instance: i }),
            "case {case}"
        );
    }
}

/// A false value `v` passes the leaf checks, which do not read it, and leaves false claims on
/// the value polynomial and the key, which the stages and openings that take them reject.
#[test]
fn a_false_value_passes_the_leaf_checks_and_leaves_false_claims() {
    let mut rng = SplitMix64::new(0x5eed_0062_0008);
    let (instances, forests, trees) = base(&mut rng);
    // Instances 1 and 2 have values: the message holds two.
    for position in 0..2 {
        let run = cheat(&instances, &held(&instances), &forests, &trees, |tamper| {
            if let Tamper::Values(values) = tamper {
                values[position] = values[position] + Fe::ONE;
            }
        });
        assert_eq!(
            run.outcome,
            LogUpOutcome::False(FalseClaim::Key(1 + position))
        );
    }
}

/// An instance whose table is not a power of two, with row 0 looked up, so that row 0's
/// multiplicity is at least 1 and the table tree has padding rows.
fn padded(rng: &mut SplitMix64) -> Instance {
    let mut instance = instance(rng, 5, 7, true);
    instance.lookups[0] = Lookup::Row(0);
    let recounted = Instance::honest(
        instance.rows.clone(),
        true,
        instance
            .lookups
            .iter()
            .map(|lookup| match *lookup {
                Lookup::Row(j) => j,
                Lookup::Pair(..) => unreachable!("an honest lookup"),
            })
            .collect(),
    );
    assert_eq!(recounted.multiplicities.len(), 8);
    recounted
}

/// RFC-0008 'Fraction tree': a padding row of the table tree is a copy of row 0, so a
/// multiplicity a prover commits there only counts more lookups of row 0. Row 0's count moved,
/// whole or in part, to padding rows verifies, and every claim left is true.
#[test]
fn a_multiplicity_on_a_padding_row_counts_lookups_of_row_zero() {
    let mut rng = SplitMix64::new(0x5eed_0062_0009);
    for case in 0..6 {
        let mut instance = padded(&mut rng);
        let moved = instance.multiplicities[0];
        let to = 5 + case % 3;
        instance.multiplicities[0] -= Fp::ONE;
        instance.multiplicities[to] += Fp::ONE;
        if case >= 3 {
            // All of row 0's count, spread over two padding rows.
            let rest = instance.multiplicities[0];
            instance.multiplicities[0] = Fp::ZERO;
            instance.multiplicities[5 + (case + 1) % 3] += rest;
        }
        assert_eq!(
            instance.multiplicities[0]
                + instance.multiplicities[5]
                + instance.multiplicities[6]
                + instance.multiplicities[7],
            moved
        );
        let trees = schedule(&mut rng, 1, 0);
        let proven = Proven::honest(vec![instance], Vec::new(), trees);
        assert_reproduces(&proven, &format!("case {case}"));
    }
}

/// The phantom-row attack: a lookup of `(0, 0)`, which is no row, paid for by a multiplicity on a
/// padding row. Under RFC-0008's padding the padding row is a copy of row 0, so the payment counts
/// row 0 and the roots do not balance. A prover that pads with a row of key and value 0 instead,
/// `(−m, α)`, balances its roots and passes the levels, and the table evaluation, whose padding
/// rows the verifier computes as copies of row 0, rejects it.
#[test]
fn a_phantom_row_is_rejected() {
    let mut rng = SplitMix64::new(0x5eed_0062_000a);
    for case in 0..4 {
        let instance = padded(&mut rng);
        assert_ne!(instance.rows[0], (Fp::ZERO, Fp::ZERO));
        let mut phantom = instance.clone();
        let at = 1 + case;
        let Lookup::Row(row) = phantom.lookups[at] else {
            unreachable!()
        };
        // The lookup becomes (0, 0); its row loses one count, and a padding row gains it.
        phantom.lookups[at] = Lookup::Pair(Fp::ZERO, Fp::ZERO);
        phantom.multiplicities[row] -= Fp::ONE;
        phantom.multiplicities[6] += Fp::ONE;
        let trees = schedule(&mut rng, 1, 0);
        let run = cheat(
            &[phantom.clone()],
            &held(&[phantom.clone()]),
            &[],
            &trees,
            |_| {},
        );
        assert_eq!(
            run.outcome,
            LogUpOutcome::Roots(LookupError::Unbalanced { instance: 0 }),
            "case {case}"
        );
        // A prover that pads with rows of key and value 0, (−m, α): its table holds them as
        // rows, so its trees are honest over them, and its roots balance.
        let mut zero_padded = phantom.clone();
        zero_padded.rows.resize(8, (Fp::ZERO, Fp::ZERO));
        let run = cheat(&[phantom], &held(&[zero_padded]), &[], &trees, |_| {});
        assert_eq!(
            run.outcome,
            LogUpOutcome::Leaves(LookupError::TableEvaluation { instance: 0 }),
            "case {case}"
        );
    }
}

/// RFC-0008 'Fraction tree' and RFC-0014 row 4: a leaf stage without LogUp instance squeezes `α`
/// and `β`, absorbs an empty roots message, runs its forests' levels with no `λ`, and absorbs an
/// empty values message; with no tree at all, it has no level.
#[test]
fn a_leaf_stage_without_instance_verifies() {
    let mut rng = SplitMix64::new(0x5eed_0062_000b);
    for case in 0..6 {
        let forests: Vec<Forest> = (0..case % 3)
            .map(|_| {
                let (s, h) = (below(&mut rng, 3), 1 + below(&mut rng, 3));
                forest(&mut rng, s, h)
            })
            .collect();
        let trees = schedule(&mut rng, 0, forests.len());
        let proven = Proven::honest(Vec::new(), forests, trees);
        assert!(proven.stage.roots.is_empty() && proven.stage.values.is_empty());
        assert!(
            proven
                .stage
                .levels
                .iter()
                .all(|level| level.lambda.is_none())
        );
        assert_reproduces(&proven, &format!("case {case}"));
    }
}
