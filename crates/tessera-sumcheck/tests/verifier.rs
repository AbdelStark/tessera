// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Honest proofs through the independent verifier (DEP-2): every proof is checked by
//! `tv_core::sumcheck::verify_batch` with the product relation, on a `tv-hash` transcript of the
//! same session, and the verifier's challenges, points and final transcript state must be the
//! prover's (RFC-0010 'Testing Strategy', 07 §3).
//!
//! Tables mix the two kinds: over the prime field, which the prover multiplies there until their
//! first binding, and over the extension.

mod common;

use common::{Session, SplitMix64, TestInstance, TestTable, prove_and_verify};
use tessera_sumcheck::{MAX_DEGREE, Prefix};

/// Whether table `t` of an instance is over the prime field, by pattern: all of them, none, or
/// alternately.
#[derive(Clone, Copy, Debug)]
enum Kinds {
    Base,
    Ext,
    Mixed,
}

impl Kinds {
    const ALL: [Self; 3] = [Self::Base, Self::Ext, Self::Mixed];

    fn base(self, t: usize) -> bool {
        match self {
            Self::Base => true,
            Self::Ext => false,
            Self::Mixed => t.is_multiple_of(2),
        }
    }
}

/// Single products of `d` tables on 0 to 24 variables, each kind in turn below 24, and 24
/// variables over the prime field.
fn single_products(d: usize) {
    let mut rng = SplitMix64::new(0x53_0001 + d as u64);
    let session = Session::random(&mut rng);
    for vars in 0..=24 {
        let kinds = if vars == 24 {
            Kinds::Base
        } else {
            Kinds::ALL[vars % 3]
        };
        let instance = TestInstance::random(&mut rng, vars, d, |t| kinds.base(t));
        let proven = prove_and_verify(&session, Prefix::Stage(vars as u32), d, &[instance]);
        assert_eq!(proven.proof().final_evals.len(), d);
    }
}

#[test]
fn products_of_one_table_verify_on_0_to_24_variables() {
    single_products(1);
}

#[test]
fn products_of_two_tables_verify_on_0_to_24_variables() {
    single_products(2);
}

#[test]
fn products_of_three_tables_verify_on_0_to_24_variables() {
    single_products(3);
}

#[test]
fn products_of_four_tables_verify_on_0_to_24_variables() {
    single_products(4);
}

/// RFC-0010's mixed batch, `n_max = 12`, in every order of its instances and with every kind of
/// table: the padded instances take the last 3 and 7 challenges.
#[test]
fn the_batch_of_3_7_and_12_variables_verifies() {
    let mut rng = SplitMix64::new(0x53_0002);
    let session = Session::random(&mut rng);
    let shapes = [(3, 3), (7, 2), (12, 3)];
    let orders = [[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]];
    for (k, order) in orders.iter().enumerate() {
        for kinds in Kinds::ALL {
            let instances: Vec<TestInstance> = order
                .iter()
                .map(|&i| {
                    let (vars, tables) = shapes[i];
                    TestInstance::random(&mut rng, vars, tables, |t| kinds.base(t + i))
                })
                .collect();
            let proven = prove_and_verify(&session, Prefix::Stage(k as u32), 3, &instances);
            assert_eq!(proven.proof().rounds.len(), 12);
        }
    }
}

/// Instances on no variable: one that is padded through every round, and batches with
/// `n_max = 0`, which have no round at all.
#[test]
fn instances_on_no_variable_verify() {
    let mut rng = SplitMix64::new(0x53_0003);
    let session = Session::random(&mut rng);
    for kinds in Kinds::ALL {
        let base = |t: usize| kinds.base(t);
        let batch = [
            TestInstance::random(&mut rng, 5, 2, base),
            TestInstance::random(&mut rng, 0, 3, base),
            TestInstance::random(&mut rng, 3, 1, base),
            TestInstance::random(&mut rng, 0, 1, base),
        ];
        let proven = prove_and_verify(&session, Prefix::Stage(0), 3, &batch);
        assert_eq!(proven.instance_point(1), Some(&[][..]));

        let constants = [
            TestInstance::random(&mut rng, 0, 2, base),
            TestInstance::random(&mut rng, 0, 1, base),
        ];
        for degree in [2, 3, 7] {
            let proven =
                prove_and_verify(&session, Prefix::Tree(degree as u32), degree, &constants);
            assert!(proven.proof().rounds.is_empty() && proven.point().is_empty());
        }
    }
}

/// Degree 7, every node of RFC-0014's largest degree, with instances of 1 to 7 tables.
#[test]
fn degree_7_batches_with_fewer_tables_than_the_degree_verify() {
    let mut rng = SplitMix64::new(0x53_0004);
    let session = Session::random(&mut rng);
    for kinds in Kinds::ALL {
        let base = |t: usize| kinds.base(t);
        let batch = [
            TestInstance::random(&mut rng, 6, 1, base),
            TestInstance::random(&mut rng, 2, 3, base),
            TestInstance::random(&mut rng, 4, 7, base),
            TestInstance::random(&mut rng, 0, 2, base),
            TestInstance::random(&mut rng, 6, 5, base),
        ];
        let _ = prove_and_verify(&session, Prefix::Stage(7), MAX_DEGREE, &batch);
        let single = [TestInstance::random(&mut rng, 5, 7, base)];
        let _ = prove_and_verify(&session, Prefix::Stage(8), MAX_DEGREE, &single);
    }
}

/// Public factors are left out of the final evaluations and evaluated by the verifier: the
/// shape of a product-tree level, `eq(r_i, x)·V(0 ‖ x)·V(1 ‖ x)` with `V` over the prime field,
/// front-padded, and a public table among sent ones.
#[test]
fn public_factors_verify_without_being_sent() {
    let mut rng = SplitMix64::new(0x53_0005);
    let session = Session::random(&mut rng);
    for level in 0..4u32 {
        let batch: Vec<TestInstance> = [6, 2, 0, 6, 4]
            .into_iter()
            .map(|vars| {
                TestInstance::new(vec![
                    TestTable::eq(&mut rng, vars),
                    TestTable::random(&mut rng, vars, true),
                    TestTable::random(&mut rng, vars, true),
                ])
            })
            .collect();
        let proven = prove_and_verify(&session, Prefix::Tree(level), 3, &batch);
        assert_eq!(proven.proof().final_evals.len(), 2 * batch.len());
    }
    let batch = [
        TestInstance::new(vec![
            TestTable::random(&mut rng, 4, false),
            TestTable::random(&mut rng, 4, true).public(),
            TestTable::random(&mut rng, 4, true),
        ]),
        TestInstance::new(vec![TestTable::eq(&mut rng, 2)]),
    ];
    let proven = prove_and_verify(&session, Prefix::Stage(3), 4, &batch);
    assert_eq!(proven.instance_evals(1), Some(&[][..]));
}

/// Seeded random batches: 1 to 5 instances on 0 to 10 variables, degree 1 to 7, each instance a
/// product of 1 to `d` tables of random kinds, some of them public factors.
#[test]
fn seeded_random_batches_verify() {
    let mut rng = SplitMix64::new(0x53_0006);
    let session = Session::random(&mut rng);
    for case in 0..300u32 {
        let degree = 1 + rng.below(MAX_DEGREE as u64) as usize;
        let count = 1 + rng.below(5) as usize;
        let batch: Vec<TestInstance> = (0..count)
            .map(|_| {
                let vars = rng.below(11) as usize;
                let k = 1 + rng.below(degree as u64) as usize;
                TestInstance::new(
                    (0..k)
                        .map(|_| match rng.below(8) {
                            0 => TestTable::eq(&mut rng, vars),
                            1 => TestTable::random(&mut rng, vars, true).public(),
                            2..=4 => TestTable::random(&mut rng, vars, true),
                            _ => TestTable::random(&mut rng, vars, false),
                        })
                        .collect(),
                )
            })
            .collect();
        let prefix = if rng.coin() {
            Prefix::Stage(case)
        } else {
            Prefix::Tree(case)
        };
        let _ = prove_and_verify(&session, prefix, degree, &batch);
    }
}
