// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sum-of-products instances through the independent verifier (DEP-2): every batch is proven by
//! `prove_batch` and checked by `tv_core::sumcheck::verify_batch` on a `tv-hash` transcript of the
//! same session. The relation the verifier evaluates is written here from the instance's terms,
//! `Σ_t w_t·Π_{k ∈ t} T̃_k`, the sent evaluations taken in table order and the public factors
//! evaluated by the verifier itself; each instance's claim is its relation summed over the
//! hypercube in `tv-field`. The verifier must accept, draw the prover's challenges and end in the
//! prover's state, and each table's evaluation must be its multilinear extension at the
//! instance's point.
//!
//! The cases: a fraction tree's level shape, with numerators over `Fp`, `Fp2` and of 1; shared
//! and repeated factors; tables that no term reads; terms of every degree up to 7; batches that
//! mix sums of products and products, front-padded, instances on no variable included; and 200
//! seeded random batches. A batch of products alone keeps its messages, which `vectors.rs` pins.

mod common;

use common::{Fp, Fp2, Session, SplitMix64, TestTable, TvFp2, tv_fp2, tv_fp2s, tv_mle, tv_proof};
use tessera_field::Field;
use tessera_sumcheck::{Instance, MAX_DEGREE, Prefix, Term, prove_batch};
use tv_core::sumcheck as tv;

/// One weighted product of a test instance, the weight drawn in `Fp2`.
#[derive(Clone)]
struct TestTerm {
    weight: Fp2,
    factors: Vec<usize>,
}

/// A sum-of-products instance, or a product when `terms` is `None`.
#[derive(Clone)]
struct SopInstance {
    tables: Vec<TestTable>,
    terms: Option<Vec<TestTerm>>,
}

impl SopInstance {
    fn instance(&self) -> Instance<'_, Fp2> {
        let tables = self.tables.iter().map(TestTable::table).collect();
        match &self.terms {
            None => Instance::new(tables),
            Some(terms) => Instance::sum_of_products(
                tables,
                terms
                    .iter()
                    .map(|t| Term::new(t.weight, t.factors.clone()))
                    .collect(),
            ),
        }
    }

    fn vars(&self) -> usize {
        self.tables[0].values.len().trailing_zeros() as usize
    }

    fn sent(&self) -> usize {
        self.tables.iter().filter(|t| t.is_sent()).count()
    }

    /// The relation from the tables' values `at(k)`.
    fn relation_of(&self, at: impl Fn(usize) -> TvFp2) -> TvFp2 {
        match &self.terms {
            None => (0..self.tables.len()).fold(TvFp2::ONE, |p, k| p * at(k)),
            Some(terms) => terms.iter().fold(TvFp2::ZERO, |sum, term| {
                sum + tv_fp2(term.weight) * term.factors.iter().fold(TvFp2::ONE, |p, &k| p * at(k))
            }),
        }
    }

    /// The claim: the relation summed over the hypercube, entry by entry.
    fn claim(&self) -> TvFp2 {
        (0..1usize << self.vars()).fold(TvFp2::ZERO, |sum, x| {
            sum + self.relation_of(|k| self.tables[k].values.tv_at(x))
        })
    }

    /// The verifier's relation at `point` from the sent evaluations, in table order, public
    /// factors evaluated from their definition.
    fn relation(&self, point: &[TvFp2], evals: &[TvFp2]) -> Option<TvFp2> {
        if evals.len() != self.sent() {
            return None;
        }
        let mut sent = evals.iter();
        let values: Vec<TvFp2> = self
            .tables
            .iter()
            .map(|table| match &table.kind {
                common::Kind::Sent => sent.next().copied(),
                common::Kind::Eq(r) => tv::eq(&tv_fp2s(r), point),
                common::Kind::Public => Some(tv_mle(&table.values, point)),
            })
            .collect::<Option<_>>()?;
        Some(self.relation_of(|k| values[k]))
    }
}

/// Proves the batch and checks it with `verify_batch`, and each evaluation against its table.
fn prove_and_verify(session: &Session, degree: usize, instances: &[SopInstance]) {
    let borrowed: Vec<Instance<'_, Fp2>> = instances.iter().map(SopInstance::instance).collect();
    let mut transcript = session.transcript();
    let proven = prove_batch(&mut transcript, Prefix::Stage(3), degree, &borrowed)
        .expect("a well-formed batch");
    let claims: Vec<tv::Instance> = instances
        .iter()
        .map(|i| tv::Instance {
            claim: i.claim(),
            vars: i.vars(),
            evals: i.sent(),
        })
        .collect();
    let mut tv_transcript = session.tv_transcript();
    let batch = tv::verify_batch(
        &mut tv_transcript,
        tv::Prefix::Stage(3),
        degree,
        &claims,
        &tv_proof(proven.proof()),
        |i, point, evals| instances.get(i)?.relation(point, evals),
    )
    .expect("the verifier accepts an honest proof");
    assert_eq!(batch.gamma, tv_fp2(*proven.gamma()), "γ");
    assert_eq!(batch.point, tv_fp2s(proven.point()), "the point");
    assert_eq!(*tv_transcript.state(), transcript.state(), "the states");
    for (i, instance) in instances.iter().enumerate() {
        let point = tv_fp2s(proven.instance_point(i).expect("an instance"));
        let evals = proven.instance_evals(i).expect("an instance");
        let sent: Vec<&TestTable> = instance.tables.iter().filter(|t| t.is_sent()).collect();
        assert_eq!(
            evals.len(),
            sent.len(),
            "instance {i}: one evaluation per sent table"
        );
        for (k, (table, &eval)) in sent.iter().zip(evals).enumerate() {
            assert_eq!(
                tv_fp2(eval),
                tv_mle(&table.values, &point),
                "instance {i}: table {k}'s extension at the point"
            );
        }
    }
}

fn term(weight: Fp2, factors: &[usize]) -> TestTerm {
    TestTerm {
        weight,
        factors: factors.to_vec(),
    }
}

/// A fraction tree level's instance: `[eq, N_0, N_1, D_0, D_1]` with the terms
/// `eq·N_0·D_1 + eq·N_1·D_0 + λ·eq·D_0·D_1`, numerators over `Fp` or `Fp2`, or of 1 left out of
/// the terms as a lookup tree's leaves are.
fn fraction_level(rng: &mut SplitMix64, vars: usize, numerators: &str) -> SopInstance {
    let lambda = rng.fp2();
    let (n0, n1, terms) = match numerators {
        "base" | "ext" => {
            let base = numerators == "base";
            (
                TestTable::random(rng, vars, base),
                TestTable::random(rng, vars, base),
                vec![
                    term(Fp2::ONE, &[0, 1, 4]),
                    term(Fp2::ONE, &[0, 2, 3]),
                    term(lambda, &[0, 3, 4]),
                ],
            )
        }
        _ => (
            TestTable::base(vec![Fp::ONE; 1 << vars]),
            TestTable::base(vec![Fp::ONE; 1 << vars]),
            vec![
                term(Fp2::ONE, &[0, 4]),
                term(Fp2::ONE, &[0, 3]),
                term(lambda, &[0, 3, 4]),
            ],
        ),
    };
    SopInstance {
        tables: vec![
            TestTable::eq(rng, vars),
            n0,
            n1,
            TestTable::random(rng, vars, false),
            TestTable::random(rng, vars, false),
        ],
        terms: Some(terms),
    }
}

#[test]
fn fraction_tree_levels_verify() {
    let mut rng = SplitMix64::new(0x60_0001);
    let session = Session::random(&mut rng);
    for vars in [0, 1, 2, 5, 10, 14] {
        for numerators in ["base", "ext", "unit"] {
            let level = fraction_level(&mut rng, vars, numerators);
            prove_and_verify(&session, 3, &[level]);
        }
    }
}

/// Several fraction levels and product forests' levels in one batch, of different sizes, as a
/// leaf stage's level holds them, front-padded.
#[test]
fn fraction_levels_beside_products_verify() {
    let mut rng = SplitMix64::new(0x60_0002);
    let session = Session::random(&mut rng);
    let product = |rng: &mut SplitMix64, vars: usize| SopInstance {
        tables: vec![
            TestTable::eq(rng, vars),
            TestTable::random(rng, vars, true),
            TestTable::random(rng, vars, true),
        ],
        terms: None,
    };
    let instances = vec![
        fraction_level(&mut rng, 7, "ext"),
        product(&mut rng, 3),
        fraction_level(&mut rng, 0, "base"),
        fraction_level(&mut rng, 9, "unit"),
        product(&mut rng, 9),
        fraction_level(&mut rng, 4, "base"),
    ];
    prove_and_verify(&session, 3, &instances);
}

/// Factors shared by some terms only, repeated factors, a term equal to the common factors, and
/// tables no term reads, over both fields and public.
#[test]
fn shared_and_repeated_factors_verify() {
    let mut rng = SplitMix64::new(0x60_0003);
    let session = Session::random(&mut rng);
    for vars in [0, 1, 3, 8] {
        let w = (rng.fp2(), rng.fp2(), rng.fp2());
        let tables = vec![
            TestTable::random(&mut rng, vars, true),
            TestTable::random(&mut rng, vars, false),
            TestTable::random(&mut rng, vars, false).public(),
            TestTable::random(&mut rng, vars, true),
            TestTable::random(&mut rng, vars, false),
        ];
        let cases = [
            // X0²·X1 + w·X0³ + X0²: X0² common, a term left with the product 1.
            vec![
                term(Fp2::ONE, &[0, 0, 1]),
                term(w.0, &[0, 0, 0]),
                term(Fp2::ONE, &[0, 0]),
            ],
            // No common factor; table 4 read by no term.
            vec![term(w.1, &[1, 2]), term(w.2, &[3]), term(w.1, &[2, 3, 1])],
            // One term only.
            vec![term(w.0, &[2, 1, 0, 3])],
            // Every term the same factors, other weights.
            vec![term(w.0, &[1, 3]), term(w.1, &[3, 1]), term(w.2, &[1, 3])],
        ];
        for terms in cases {
            let instance = SopInstance {
                tables: tables.clone(),
                terms: Some(terms),
            };
            prove_and_verify(&session, 4, &[instance]);
        }
    }
}

/// Seeded random batches: 1 to 4 instances on 0 to 9 variables, each a product or a sum of 1 to
/// 4 terms over 1 to 6 tables of either field, some public; degrees 1 to 7.
#[test]
fn seeded_random_batches_verify() {
    let mut rng = SplitMix64::new(0x60_0004);
    let session = Session::random(&mut rng);
    for _ in 0..200 {
        let degree = 1 + rng.below(MAX_DEGREE as u64) as usize;
        let count = 1 + rng.below(4) as usize;
        let instances: Vec<SopInstance> = (0..count)
            .map(|_| {
                let vars = rng.below(10) as usize;
                let products = rng.below(3) == 0;
                let tables_count = match products {
                    true => 1 + rng.below(degree as u64) as usize,
                    false => 1 + rng.below(6) as usize,
                };
                let tables = (0..tables_count)
                    .map(|_| {
                        let base = rng.coin();
                        let table = TestTable::random(&mut rng, vars, base);
                        match rng.below(4) {
                            0 => table.public(),
                            _ => table,
                        }
                    })
                    .collect();
                let terms = (!products).then(|| {
                    (0..1 + rng.below(4))
                        .map(|_| {
                            let factors = (0..1 + rng.below(degree as u64))
                                .map(|_| rng.below(tables_count as u64) as usize)
                                .collect();
                            TestTerm {
                                weight: if rng.coin() { Fp2::ONE } else { rng.fp2() },
                                factors,
                            }
                        })
                        .collect()
                });
                SopInstance { tables, terms }
            })
            .collect();
        prove_and_verify(&session, degree, &instances);
    }
}

/// Every degree from 1 to 7: a term of `d` factors beside lower ones.
#[test]
fn terms_of_every_degree_verify() {
    let mut rng = SplitMix64::new(0x60_0005);
    let session = Session::random(&mut rng);
    for degree in 1..=MAX_DEGREE {
        let tables: Vec<TestTable> = (0..degree)
            .map(|k| TestTable::random(&mut rng, 6, k % 2 == 0))
            .collect();
        let terms = (1..=degree)
            .map(|d| term(rng.fp2(), &(0..d).collect::<Vec<_>>()))
            .collect();
        let instance = SopInstance {
            tables,
            terms: Some(terms),
        };
        let padded = SopInstance {
            tables: vec![TestTable::random(&mut rng, 9, true)],
            terms: None,
        };
        prove_and_verify(&session, degree, &[instance, padded]);
    }
}
