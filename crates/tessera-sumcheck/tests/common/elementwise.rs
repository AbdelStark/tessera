// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Elementwise relations for the tests, and the verifier's side of them, written from RFC-0010
//! 'Relations', 'Multilinear extensions' and 'Batching within a stage', and RFC-0035 'The padding
//! zero-check', around the verifier's independent batch check, `tv_core::sumcheck::verify_batch`
//! (DEP-2).
//!
//! A case holds its inputs' tables over `Fp`, its terms, and the point of its input claim, or it
//! is a zero-check and takes the stage's zero-check point. Its verifier's side uses `tv-field`,
//! `tv-hash` and `tv_core::sumcheck::eq` alone:
//!
//! - the target, entry by entry: `T[y] = Σ_x Σ_t w_t·Π_{f ∈ t} I_f(x)` over the `2^m` entries `x`
//!   of row `y`, each input read at `x`, at `x ⊕ 2^b` for a partner, and at the bits `k` to
//!   `k + j − 1` of `x` for a broadcast. The input claim is the target's extension at the point;
//!   a zero-check's is 0, whatever its witness.
//! - the zero-check point: challenges squeezed under `stage/{k}/zero/{c}` before `γ`, one per
//!   variable of the largest zero-check, which each zero-check of `n_i` variables takes the last
//!   `n_i` of;
//! - the relation at the instance's point `r'`: `eq(r, r'_high)·Σ_t w_t·Π_{f ∈ t} v_f`, `v_f` the
//!   evaluation the proof sends for an operand and a public table's extension at `r'`;
//! - the claims' points: `r'` for an operand, `r'` with coordinate `b` replaced by `1 − r'_{b+1}`
//!   for a partner, and the coordinates `k` to `k + j − 1` of `r'` for a broadcast.
//!
//! The prover's side runs `Elementwise`, `zero_check_point` and `prove_batch`. None of the
//! verifier's side uses the prover's code.

use tessera_sumcheck::{
    Claim, Elementwise, ElementwiseShape, ElementwiseTables, Input, Instance, Prefix, ProvenBatch,
    Term, prove_batch, zero_check_point,
};
use tv_core::sumcheck as tv;
use tv_hash::transcript::Ascii;

use super::forests::mle;
use super::{Fp, Fp2, Session, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_proof};

/// How a case reads one of its inputs at each vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    /// The operand itself.
    Operand,
    /// The operand at the vertex whose bit `b` is flipped.
    Partner(usize),
    /// An operand of `2^j` entries at the bits `k` to `k + j − 1` of the vertex.
    Broadcast(usize),
    /// A public table, which the verifier evaluates.
    Public,
}

/// One input: how it is read, and its table.
#[derive(Clone)]
pub struct CaseInput {
    pub read: Read,
    pub values: Vec<Fp>,
}

impl CaseInput {
    pub fn new(read: Read, values: Vec<Fp>) -> Self {
        Self { read, values }
    }

    /// Whether the proof sends its evaluation.
    pub fn is_sent(&self) -> bool {
        self.read != Read::Public
    }
}

/// Where a case's input claim is.
#[derive(Clone)]
pub enum At {
    /// A claim on the target at this point.
    Point(Vec<Fp2>),
    /// A zero-check: the claim 0 at the stage's zero-check point.
    Zero,
}

/// How the prover builds a case's relation: through `Elementwise::new` from the case's inputs
/// and terms, or through the constructor of its kind from its inputs alone, so that the terms
/// the verifier's side uses, written from the RFCs, check the constructor's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    New,
    /// `Elementwise::mul` on inputs 0 and 1.
    Mul,
    /// `Elementwise::sum_squares` on input 0.
    SumSquares,
    /// `Elementwise::booleanity` on input 0.
    Booleanity,
    /// `Elementwise::padding` on input 0 and the public input 1.
    Padding,
}

/// A relation on random tables, with its terms and its claim's point.
#[derive(Clone)]
pub struct Case {
    pub build: Build,
    pub shape: ElementwiseShape,
    pub inputs: Vec<CaseInput>,
    /// `(w_t, factors)`, the factors naming inputs.
    pub terms: Vec<(Fp2, Vec<usize>)>,
    pub at: At,
}

impl Case {
    fn input(&self, k: usize) -> Input<'_, Fp2> {
        let input = &self.inputs[k];
        match input.read {
            Read::Operand => Input::Operand(&input.values),
            Read::Partner(bit) => Input::Partner {
                operand: &input.values,
                bit,
            },
            Read::Broadcast(low_vars) => Input::Broadcast {
                operand: &input.values,
                low_vars,
            },
            Read::Public => Input::Public(&input.values),
        }
    }

    /// The prover's relation, as `build` says.
    pub fn relation(&self) -> Elementwise<'_, Fp2> {
        let vars = self.shape.vars;
        let relation = match self.build {
            Build::New => {
                let inputs = (0..self.inputs.len()).map(|k| self.input(k)).collect();
                let terms = self
                    .terms
                    .iter()
                    .map(|(weight, factors)| Term::new(*weight, factors.clone()))
                    .collect();
                Elementwise::new(self.shape, inputs, terms)
            }
            Build::Mul => Elementwise::mul(self.input(0), self.input(1), vars),
            Build::SumSquares => Elementwise::sum_squares(self.input(0), self.shape),
            Build::Booleanity => Elementwise::booleanity(self.input(0), vars),
            Build::Padding => Elementwise::padding(self.input(0), &self.inputs[1].values, vars),
        };
        relation.expect("a well-formed relation")
    }

    pub fn vars(&self) -> usize {
        self.shape.vars
    }

    /// The target's variables, `n − m`.
    pub fn target_vars(&self) -> usize {
        self.shape.vars - self.shape.summed_vars
    }

    /// The evaluations the proof sends: one per input that is not public.
    pub fn sent(&self) -> usize {
        self.inputs.iter().filter(|input| input.is_sent()).count()
    }

    pub fn is_zero_check(&self) -> bool {
        matches!(self.at, At::Zero)
    }

    /// Input `k` at vertex `x`, read as its kind says.
    fn read(&self, k: usize, x: usize) -> TvFp2 {
        let input = &self.inputs[k];
        let entry = match input.read {
            Read::Operand | Read::Public => x,
            Read::Partner(bit) => x ^ (1 << bit),
            Read::Broadcast(low_vars) => (x >> low_vars) % input.values.len(),
        };
        TvFp2::from(tv_fp(input.values[entry]))
    }

    /// The target, entry by entry from the inputs: row `y` sums its `2^m` vertices.
    pub fn target(&self) -> Vec<TvFp2> {
        let width = 1 << self.shape.summed_vars;
        (0..1 << self.target_vars())
            .map(|y| {
                (y * width..(y + 1) * width).fold(TvFp2::ZERO, |sum, x| {
                    sum + self
                        .terms
                        .iter()
                        .fold(TvFp2::ZERO, |term_sum, (w, factors)| {
                            term_sum
                                + factors
                                    .iter()
                                    .fold(tv_fp2(*w), |product, &f| product * self.read(f, x))
                        })
                })
            })
            .collect()
    }

    /// The verifier's input claim: the target's extension at the point, or 0 for a zero-check.
    pub fn tv_claim(&self) -> TvFp2 {
        match &self.at {
            At::Point(point) => extension(&self.target(), &tv_fp2s(point)),
            At::Zero => TvFp2::ZERO,
        }
    }

    pub fn tv_instance(&self) -> tv::Instance {
        tv::Instance {
            claim: self.tv_claim(),
            vars: self.vars(),
            evals: self.sent(),
        }
    }

    /// The point of its input claim: its own, or the last `n` coordinates of the zero-check
    /// point.
    fn input_point<T: Copy>(&self, own: impl FnOnce(&[Fp2]) -> Vec<T>, zero: &[T]) -> Vec<T> {
        match &self.at {
            At::Point(point) => own(point),
            At::Zero => zero[zero.len() - self.vars()..].to_vec(),
        }
    }

    /// The relation at the instance's point `point`, from the evaluations the proof sent, with
    /// the zero-check point `zero`.
    pub fn relation_at(&self, point: &[TvFp2], evals: &[TvFp2], zero: &[TvFp2]) -> Option<TvFp2> {
        if evals.len() != self.sent() || point.len() != self.vars() {
            return None;
        }
        let mut sent = evals.iter();
        let values: Vec<TvFp2> = self
            .inputs
            .iter()
            .map(|input| match input.read {
                Read::Public => mle(&input.values, point),
                _ => *sent.next().expect("one evaluation per sent input"),
            })
            .collect();
        let sum = self.terms.iter().fold(TvFp2::ZERO, |sum, (w, factors)| {
            sum + factors
                .iter()
                .fold(tv_fp2(*w), |product, &f| product * values[f])
        });
        let r = self.input_point(tv_fp2s, zero);
        let eq = tv::eq(&r, &point[self.shape.summed_vars..])?;
        Some(eq * sum)
    }

    /// The points of the claims it leaves, one per sent input, from the instance's point.
    pub fn claim_points(&self, point: &[TvFp2]) -> Vec<Vec<TvFp2>> {
        self.inputs
            .iter()
            .filter_map(|input| match input.read {
                Read::Operand => Some(point.to_vec()),
                Read::Partner(bit) => {
                    let mut flipped = point.to_vec();
                    flipped[bit] = TvFp2::ONE - flipped[bit];
                    Some(flipped)
                }
                Read::Broadcast(low_vars) => {
                    let own = input.values.len().trailing_zeros() as usize;
                    Some(point[low_vars..low_vars + own].to_vec())
                }
                Read::Public => None,
            })
            .collect()
    }

    /// The tables the claims are on, one per sent input.
    pub fn claimed_tables(&self) -> Vec<&[Fp]> {
        self.inputs
            .iter()
            .filter(|input| input.is_sent())
            .map(|input| &input.values[..])
            .collect()
    }
}

/// `Σ_x values[x]·eq(point, x)`, in `tv-field`, the `eq` weights built by doubling as
/// `forests::mle` builds them.
pub fn extension(values: &[TvFp2], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        values.len(),
        1 << point.len(),
        "2^n values at n coordinates"
    );
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    values
        .iter()
        .zip(&weights)
        .fold(TvFp2::ZERO, |sum, (&v, &e)| sum + e * v)
}

/// A proven stage of cases: the batch, each case's claims, the zero-check point, and the
/// transcript's final state.
pub struct ProvenCases {
    pub proven: ProvenBatch<Fp2>,
    pub claims: Vec<Vec<Claim<Fp2>>>,
    pub zero: Vec<Fp2>,
    pub state: [u8; 32],
}

/// The variables of the stage's zero-check point: the largest zero-check's, if any.
fn zero_vars(cases: &[Case]) -> Option<usize> {
    cases
        .iter()
        .filter(|case| case.is_zero_check())
        .map(Case::vars)
        .max()
}

/// Proves stage `stage` on a fresh transcript of `session`: the zero-check point first when
/// the stage holds a zero-check, then each case through its relation's tables and claims.
pub fn prove_cases(session: &Session, stage: u32, degree: usize, cases: &[Case]) -> ProvenCases {
    let mut transcript = session.transcript();
    let zero: Vec<Fp2> = zero_vars(cases)
        .map(|vars| zero_check_point(&mut transcript, stage, vars))
        .unwrap_or_default();
    let tables: Vec<ElementwiseTables<'_, Fp2>> = cases
        .iter()
        .map(|case| {
            let point = case.input_point(<[Fp2]>::to_vec, &zero);
            case.relation()
                .tables(&point)
                .expect("a point of the target's coordinates")
        })
        .collect();
    let instances: Vec<Instance<'_, Fp2>> =
        tables.iter().map(ElementwiseTables::instance).collect();
    let proven = prove_batch(&mut transcript, Prefix::Stage(stage), degree, &instances)
        .expect("a well-formed stage");
    let claims = tables
        .iter()
        .enumerate()
        .map(|(i, tables)| {
            let point = proven.instance_point(i).expect("an instance");
            let evals = proven.instance_evals(i).expect("an instance");
            tables.claims(point, evals).expect("the instance's results")
        })
        .collect();
    ProvenCases {
        proven,
        claims,
        zero,
        state: transcript.state(),
    }
}

/// What the verifier concluded on a stage of cases.
pub struct Verdict {
    pub result: Result<tv::Batch, tv::SumcheckError>,
    pub zero: Vec<TvFp2>,
    pub state: [u8; 32],
}

/// Runs `tv_core::sumcheck::verify_batch` on `proof` with the cases' relations, from the claims
/// `claims`, on a fresh verifier transcript of `session`, after squeezing the zero-check point
/// when the stage holds a zero-check.
pub fn verify_cases(
    session: &Session,
    stage: u32,
    degree: usize,
    cases: &[Case],
    claims: &[tv::Instance],
    proof: &tv::SumcheckProof,
) -> Verdict {
    let mut transcript = session.tv_transcript();
    let zero: Vec<TvFp2> = (0..zero_vars(cases).unwrap_or(0))
        .map(|c| {
            let label = format!("stage/{stage}/zero/{c}");
            transcript
                .challenge_fp2(Ascii::new(&label).expect("ASCII"))
                .expect("a challenge")
        })
        .collect();
    let result = tv::verify_batch(
        &mut transcript,
        tv::Prefix::Stage(stage),
        degree,
        claims,
        proof,
        |i, point, evals| cases.get(i)?.relation_at(point, evals, &zero),
    );
    Verdict {
        result,
        zero,
        state: *transcript.state(),
    }
}

/// The verifier's verdict on a proof of the cases, from their true claims.
pub fn verdict(
    session: &Session,
    stage: u32,
    degree: usize,
    cases: &[Case],
    proof: &tv::SumcheckProof,
) -> Result<(), tv::SumcheckError> {
    let claims: Vec<tv::Instance> = cases.iter().map(Case::tv_instance).collect();
    verify_cases(session, stage, degree, cases, &claims, proof)
        .result
        .map(|_| ())
}

/// Proves the stage and checks it with the independent verifier from the true claims: it
/// accepts; its zero-check point, `γ`, point and final transcript state are the prover's; and
/// every claim is at the point the verifier derives, with the evaluation the verifier read,
/// which is the claimed table's multilinear extension there.
pub fn prove_and_verify_cases(
    session: &Session,
    stage: u32,
    degree: usize,
    cases: &[Case],
) -> ProvenCases {
    let proven = prove_cases(session, stage, degree, cases);
    let proof = proven.proven.proof();
    let n_max = cases.iter().map(Case::vars).max().unwrap_or(0);
    assert_eq!(proof.rounds.len(), n_max, "one message per round");
    let claims: Vec<tv::Instance> = cases.iter().map(Case::tv_instance).collect();
    let verdict = verify_cases(session, stage, degree, cases, &claims, &tv_proof(proof));
    let batch = verdict
        .result
        .expect("the verifier accepts an honest stage");
    assert_eq!(verdict.zero, tv_fp2s(&proven.zero), "the zero-check point");
    assert_eq!(batch.gamma, tv_fp2(*proven.proven.gamma()), "γ");
    assert_eq!(batch.point, tv_fp2s(proven.proven.point()), "the point");
    assert_eq!(
        verdict.state, proven.state,
        "the transcripts end in the same state"
    );
    let mut offset = 0;
    for (k, (case, claims)) in cases.iter().zip(&proven.claims).enumerate() {
        let evals = tv_fp2s(&proof.final_evals[offset..offset + case.sent()]);
        offset += case.sent();
        let point = batch.point_of(case.vars()).expect("an instance's point");
        let expected = case.claim_points(point);
        let tables = case.claimed_tables();
        assert_eq!(
            claims.len(),
            expected.len(),
            "case {k}: a claim per sent input"
        );
        for (n, ((claim, expected), table)) in claims.iter().zip(&expected).zip(&tables).enumerate()
        {
            assert_eq!(
                &tv_fp2s(&claim.point),
                expected,
                "case {k}, claim {n}: the point the verifier derives"
            );
            assert_eq!(
                tv_fp2(claim.value),
                evals[n],
                "case {k}, claim {n}: the evaluation the verifier read"
            );
            assert_eq!(
                tv_fp2(claim.value),
                mle(table, expected),
                "case {k}, claim {n}: the table's extension at the point"
            );
        }
    }
    assert_eq!(offset, proof.final_evals.len(), "every evaluation read");
    proven
}
