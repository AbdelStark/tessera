// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0013 'Testing Strategy' 'The bound, at the sumcheck level' (#565): `prove_batch` proves
//! batches of degree 8 to 10 at a reduction stage of `tessera-protocol/2`, whose bound is 10
//! (RFC-0013 'Masked extensions' 'Degree'), over Goldilocks; `errors.rs` refuses 11 there and 8 in
//! `tessera-protocol/1` and at a tree level.
//!
//! Each batch is checked by `tv-core`'s independent Newton interpolation, and by a second,
//! test-local arithmetic checker: Lagrange interpolation, explicit multilinear extensions,
//! and direct enumeration of the batched claims.
//!
//! A batch of a degree both protocols allow has the same messages in both: the protocol sets the
//! bound and nothing else.

mod common;

use common::{Fp, Fp2, SplitMix64};
use tessera_codec::Ascii;
use tessera_field::{Field, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    Instance, MAX_DEGREE, Prefix, Protocol, SumcheckProof, SumcheckTranscript, Table, Term,
    prove_batch,
};

/// A session of `protocol`.
fn transcript(protocol: Protocol) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new(protocol.name()).expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x65; 32],
        session_id: [0x13; 16],
    })
}

/// A table over the prime field or over `Fp2`, sent or public.
struct Values {
    entries: Vec<Fp2>,
    base: Option<Vec<Fp>>,
    public: bool,
}

impl Values {
    fn random(rng: &mut SplitMix64, vars: usize, base: bool, public: bool) -> Self {
        let (entries, base) = if base {
            let b = rng.fps(1 << vars);
            (b.iter().map(|&x| Fp2::from(x)).collect(), Some(b))
        } else {
            (rng.fp2s(1 << vars), None)
        };
        Self {
            entries,
            base,
            public,
        }
    }

    fn table(&self) -> Table<'_, Fp2> {
        let table = match &self.base {
            Some(b) => Table::base(b),
            None => Table::ext(&self.entries),
        };
        if self.public { table.public() } else { table }
    }

    /// The multilinear extension at `point`, by enumeration of the hypercube.
    fn mle(&self, point: &[Fp2]) -> Fp2 {
        assert_eq!(self.entries.len(), 1 << point.len());
        self.entries
            .iter()
            .enumerate()
            .fold(Fp2::ZERO, |sum, (x, &value)| {
                let weight = point.iter().enumerate().fold(Fp2::ONE, |w, (i, &r)| {
                    w * if x >> i & 1 == 1 { r } else { Fp2::ONE - r }
                });
                sum + value * weight
            })
    }
}

/// An instance: a product of its tables, or a sum of weighted products over them.
struct Case {
    vars: usize,
    tables: Vec<Values>,
    terms: Option<Vec<(Fp2, Vec<usize>)>>,
}

impl Case {
    fn instance(&self) -> Instance<'_, Fp2> {
        let tables = self.tables.iter().map(Values::table).collect();
        match &self.terms {
            None => Instance::new(tables),
            Some(terms) => Instance::sum_of_products(
                tables,
                terms
                    .iter()
                    .map(|(w, f)| Term::new(*w, f.clone()))
                    .collect(),
            ),
        }
    }

    /// The summand at one vertex, or at a point from the tables' values there.
    fn summand(&self, value: impl FnMut(&Values) -> Fp2) -> Fp2 {
        let values: Vec<Fp2> = self.tables.iter().map(value).collect();
        match &self.terms {
            None => values.iter().fold(Fp2::ONE, |p, &v| p * v),
            Some(terms) => terms.iter().fold(Fp2::ZERO, |sum, (w, factors)| {
                sum + factors.iter().fold(*w, |p, &f| p * values[f])
            }),
        }
    }
}

/// `p(r)` from `p(0), p(1), …, p(d)`, by Lagrange interpolation with field inverses.
fn interpolate(values: &[Fp2], r: Fp2) -> Fp2 {
    let node = |k: usize| Fp2::from(Fp::from_u64_reduced(k as u64));
    (0..values.len()).fold(Fp2::ZERO, |sum, k| {
        let (mut num, mut den) = (Fp2::ONE, Fp2::ONE);
        for m in (0..values.len()).filter(|&m| m != k) {
            num *= r - node(m);
            den *= node(k) - node(m);
        }
        sum + values[k] * num * den.inverse().expect("distinct nodes")
    })
}

/// Checks `proof` of `cases` at degree `degree` under `prefix`, as a verifier does.
fn check(
    protocol: Protocol,
    prefix: u32,
    degree: usize,
    cases: &[Case],
    proof: &SumcheckProof<Fp2>,
) {
    let label = |suffix: &str| format!("stage/{prefix}/{suffix}");
    let mut t = transcript(protocol);
    let n_max = cases.iter().map(|c| c.vars).max().expect("a case");
    let gamma = SumcheckTranscript::<Fp2>::challenge(&mut t, &label("gamma"));
    let (mut claim, mut weight) = (Fp2::ZERO, Fp2::ONE);
    for case in cases {
        let sum = (0..1 << case.vars).fold(Fp2::ZERO, |s, x| s + case.summand(|v| v.entries[x]));
        let scale = Fp2::from(Fp::from_u64_reduced(1 << (n_max - case.vars)));
        claim += weight * scale * sum;
        weight *= gamma;
    }
    assert_eq!(proof.rounds.len(), n_max);
    let mut point = Vec::new();
    for (j, round) in proof.rounds.iter().enumerate() {
        assert_eq!(round.len(), degree, "round {j} holds d values");
        t.absorb_message(&label(&format!("round/{j}")), round);
        let r = SumcheckTranscript::<Fp2>::challenge(&mut t, &label(&format!("r/{j}")));
        let values: Vec<Fp2> = [round[0], claim - round[0]]
            .into_iter()
            .chain(round[1..].iter().copied())
            .collect();
        claim = interpolate(&values, r);
        point.push(r);
    }
    let mut evals = proof.final_evals.iter();
    let (mut batched, mut weight) = (Fp2::ZERO, Fp2::ONE);
    for case in cases {
        let own = &point[n_max - case.vars..];
        for table in case.tables.iter().filter(|t| !t.public) {
            let eval = *evals.next().expect("an evaluation per sent table");
            assert_eq!(eval, table.mle(own), "a table's extension at its point");
        }
        batched += weight * case.summand(|v| v.mle(own));
        weight *= gamma;
    }
    assert!(
        evals.next().is_none(),
        "no evaluation beyond the sent tables"
    );
    assert_eq!(batched, claim, "the final check");
}

/// A batch of degree `degree`: a product of `degree` tables of mixed fields with a public one, a
/// sum of products with a term of `degree` factors beside lower ones, and smaller products,
/// front-padded.
fn batch(rng: &mut SplitMix64, degree: usize) -> Vec<Case> {
    let product = |rng: &mut SplitMix64, vars: usize, k: usize| Case {
        vars,
        tables: (0..k)
            .map(|t| Values::random(rng, vars, t % 2 == 0, t == 1))
            .collect(),
        terms: None,
    };
    let sop = Case {
        vars: 5,
        tables: (0..degree)
            .map(|t| Values::random(rng, 5, t % 3 != 0, false))
            .collect(),
        terms: Some(
            (1..=degree)
                .map(|d| (rng.fp2(), (0..d).collect()))
                .chain([(Fp2::ONE, vec![0; degree])])
                .collect(),
        ),
    };
    vec![
        product(rng, 6, degree),
        sop,
        product(rng, 3, (degree - 2).max(1)),
        product(rng, 0, 2),
    ]
}

/// RFC-0013 'Testing Strategy': `prove_batch` proves batches of degree 8, 9 and 10 under
/// `tessera-protocol/2`'s bound at a reduction stage.
#[test]
fn degree_8_to_10_batches_verify_at_a_reduction_stage_of_protocol_2() {
    let mut rng = SplitMix64::new(0x0565_0010);
    assert_eq!(Protocol::Two.max_degree(Prefix::Stage(1)), MAX_DEGREE);
    for degree in 8..=MAX_DEGREE {
        for stage in [1, 2] {
            let cases = batch(&mut rng, degree);
            let instances: Vec<Instance<'_, Fp2>> = cases.iter().map(Case::instance).collect();
            let mut t = transcript(Protocol::Two);
            let proven = prove_batch(
                &mut t,
                Protocol::Two,
                Prefix::Stage(stage),
                degree,
                &instances,
            )
            .expect("a batch within the bound");
            assert!(proven.proof().rounds.iter().all(|r| r.len() == degree));
            check(Protocol::Two, stage, degree, &cases, proven.proof());
            check_independent_verifier(stage, degree, &cases, proven.proof());
        }
    }
}

/// The protocol sets the bound and nothing else: a batch of degree 1 to 7 has the same messages,
/// and leaves the transcript in the same state, under either protocol's bound.
#[test]
fn a_degree_both_protocols_allow_has_the_same_messages() {
    let mut rng = SplitMix64::new(0x0565_0011);
    for degree in 2..=Protocol::One.max_degree(Prefix::Stage(1)) {
        let cases = batch(&mut rng, degree);
        let instances: Vec<Instance<'_, Fp2>> = cases.iter().map(Case::instance).collect();
        let prove = |protocol| {
            // One session for both, so that only the bound differs.
            let mut t = transcript(Protocol::One);
            let proven = prove_batch(&mut t, protocol, Prefix::Stage(1), degree, &instances)
                .expect("a batch within both bounds");
            (proven.into_proof(), t.state())
        };
        let one = prove(Protocol::One);
        assert_eq!(one.0, prove(Protocol::Two).0, "degree {degree}");
        assert_eq!(one.1, prove(Protocol::Two).1, "degree {degree}");
        check(Protocol::One, 1, degree, &cases, &one.0);
    }
}

/// Actual producer messages through the independently implemented protocol/2 verifier.
fn check_independent_verifier(
    stage: u32,
    degree: usize,
    cases: &[Case],
    proof: &SumcheckProof<Fp2>,
) {
    use tv_core::sumcheck as tv;
    use tv_hash::transcript as hash;
    let context = hash::SessionContext {
        protocol: hash::Ascii::new("tessera-protocol/2").unwrap(),
        mode: hash::Mode::NonInteractive,
        params: hash::ParamSetId(hash::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x65; 32],
        session_id: [0x13; 16],
    };
    let from_tv = |v: tv_field::Fp2| {
        Fp2::new(
            Fp::from_canonical_u64(v.c0.to_canonical()).unwrap(),
            Fp::from_canonical_u64(v.c1.to_canonical()).unwrap(),
        )
    };
    let instances: Vec<_> = cases
        .iter()
        .map(|case| tv::Instance {
            claim: common::tv_fp2(
                (0..1 << case.vars).fold(Fp2::ZERO, |s, x| s + case.summand(|t| t.entries[x])),
            ),
            vars: case.vars,
            evals: case.tables.iter().filter(|t| !t.public).count(),
        })
        .collect();
    let accepted = tv::verify_batch_in(
        &mut hash::Transcript::new(&context),
        tv::Protocol::Two,
        tv::Prefix::Stage(stage),
        degree,
        &instances,
        &common::tv_proof(proof),
        |i, point, values| {
            let case = &cases[i];
            let point: Vec<Fp2> = point.iter().copied().map(from_tv).collect();
            let mut sent = values.iter();
            let value = case.summand(|t| {
                if t.public {
                    t.mle(&point)
                } else {
                    from_tv(*sent.next().expect("one value per sent table"))
                }
            });
            assert!(sent.next().is_none());
            Some(common::tv_fp2(value))
        },
    )
    .expect("the independent verifier accepts the degree-8..10 producer batch");
    assert_eq!(
        accepted.point.len(),
        cases.iter().map(|c| c.vars).max().unwrap()
    );
}
