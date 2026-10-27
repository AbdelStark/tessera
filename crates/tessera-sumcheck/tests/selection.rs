// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0072's selection relation (`Selection`) through the independent verifier (DEP-2; RFC-0072
//! 'The selection relation', 'Combined selections'). The verifier's side is written here from
//! the RFC, in `tv-field`:
//!
//! - the claim is the extension, at the input point `r_lo ‖ r_a ‖ r_hi`, of the selection's output
//!   computed row by row, `y_j = Σ_t a_t·x_{idx_{t,j}}` with each member 0 past its `len`;
//! - the relation at the instance's point `r'_k` is `K̃(r_a, r'_k)·x̃`, with the kernel's closed
//!   form `K̃(r_a, r'_k) = Σ_t a_t·Σ_{j < len_t} eq(r_a, j)·eq(r'_k, idx_{t,j})`, itself checked
//!   against the kernel's extension by enumeration.
//!
//! A rotation (a gather of 32 rows), even and odd rows, a slice whose `len` is not a power of two,
//! a gather with repeated rows and a combination of selections verify, on axes with variables
//! below and above them; the one claim is `x`'s extension at `r_lo ‖ r'_k ‖ r_hi`. A prover whose
//! `x` differs in a selected entry fails the final check, and one that differs in a row no member
//! selects passes it, the opening's to catch.

mod common;

use common::elementwise::extension;
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, tv_fp, tv_fp2, tv_fp2s, tv_prefix, tv_proof};
use tessera_field::{Field, PrimeField};
use tessera_sumcheck::{
    Prefix, Protocol, Selected, Selection, SumcheckError, SumcheckProof, prove_batch,
};
use tv_core::sumcheck as tv;

/// A case's variables: `x`'s below its selected axis, along it and above it, and the output
/// axis's.
type Shape = (usize, usize, usize, usize);

/// A kernel's member: its coefficient and its rows.
type Member = (Fp2, Vec<usize>);

/// A selection of rows of `x` along an axis of `2^axis` rows, `low` variables below it and
/// `high` above, into an output axis of `2^rows`: its members `(a_t, idx_t)`.
#[derive(Clone)]
struct Case {
    low: usize,
    axis: usize,
    high: usize,
    rows: usize,
    members: Vec<Member>,
    x: Vec<Fp>,
    point: Vec<Fp2>,
}

impl Case {
    fn random(rng: &mut SplitMix64, (low, axis, high, rows): Shape, members: Vec<Member>) -> Self {
        Self {
            low,
            axis,
            high,
            rows,
            members,
            x: rng.fps(1 << (low + axis + high)),
            point: rng.fp2s(low + rows + high),
        }
    }

    fn relation(&self) -> Selection<'_, Fp2> {
        Selection::new(&self.x, self.low, self.axis, self.high).expect("a selection")
    }

    fn kernel(&self) -> Vec<Selected<'_, Fp2>> {
        self.members
            .iter()
            .map(|(coefficient, index)| Selected {
                coefficient: *coefficient,
                index,
            })
            .collect()
    }

    /// The output, row by row: entry `(h, j, l)` is `Σ_t a_t·x[h, idx_{t,j}, l]`, 0 past a
    /// member's rows.
    fn output(&self) -> Vec<TvFp2> {
        let (lo, hi) = (1usize << self.low, 1usize << self.high);
        let mut y = vec![TvFp2::ZERO; lo * (1 << self.rows) * hi];
        for h in 0..hi {
            for j in 0..1usize << self.rows {
                for l in 0..lo {
                    let at = (h << (self.rows + self.low)) | (j << self.low) | l;
                    for (a, index) in &self.members {
                        if let Some(&k) = index.get(j) {
                            let x = self.x[(h << (self.axis + self.low)) | (k << self.low) | l];
                            y[at] += tv_fp2(*a) * TvFp2::from(tv_fp(x));
                        }
                    }
                }
            }
        }
        y
    }

    /// The kernel's closed form `K̃(r_a, r'_k)`.
    fn kernel_at(&self, r_a: &[TvFp2], r_k: &[TvFp2]) -> TvFp2 {
        self.members.iter().fold(TvFp2::ZERO, |sum, (a, index)| {
            let s = index.iter().enumerate().fold(TvFp2::ZERO, |s, (j, &k)| {
                s + vertex(r_a, j) * vertex(r_k, k)
            });
            sum + tv_fp2(*a) * s
        })
    }

    fn prove(&self, session: &Session) -> (SumcheckProof<Fp2>, Vec<Fp2>, Fp2) {
        let tables = self
            .relation()
            .tables(&self.point, self.rows, &self.kernel())
            .expect("a point and a kernel");
        let proven = prove_batch(
            &mut session.transcript(),
            Protocol::One,
            Prefix::Stage(4),
            2,
            &[tables.instance()],
        )
        .expect("a stage");
        let point = proven.point().to_vec();
        let claim = tables
            .claim(&point, &proven.proof().final_evals)
            .expect("its claim");
        assert_eq!(
            claim.point,
            [
                &self.point[..self.low],
                &point[..],
                &self.point[self.low + self.rows..]
            ]
            .concat(),
            "the claim on x is at r_lo ‖ r'_k ‖ r_hi"
        );
        (proven.into_proof(), point, claim.value)
    }

    fn verdict(
        &self,
        session: &Session,
        proof: &SumcheckProof<Fp2>,
    ) -> Result<tv::Batch, tv::SumcheckError> {
        let at = tv_fp2s(&self.point);
        let instance = tv::Instance {
            claim: extension(&self.output(), &at),
            vars: self.axis,
            evals: 1,
        };
        let r_a = at[self.low..self.low + self.rows].to_vec();
        tv::verify_batch(
            &mut session.tv_transcript(),
            tv_prefix(Prefix::Stage(4)),
            2,
            &[instance],
            &tv_proof(proof),
            |_, r_k, evals| Some(self.kernel_at(&r_a, r_k) * *evals.first()?),
        )
    }
}

/// `eq(r, v)` at the vertex `v`, `r[0]` its lowest bit.
fn vertex(r: &[TvFp2], v: usize) -> TvFp2 {
    r.iter().enumerate().fold(TvFp2::ONE, |p, (t, &x)| {
        p * if (v >> t) & 1 == 1 { x } else { TvFp2::ONE - x }
    })
}

fn one() -> Fp2 {
    Fp2::ONE
}

fn constant(c: u64) -> Fp2 {
    Fp2::from(Fp::from_u64_reduced(c))
}

/// RFC-0066's and RFC-0072's selections: `ROTR^r` (a gather of 32 rows of 32), the value levels'
/// even and odd rows, each alone and combined as `even + 2^{2^ℓ}·odd`, a `Slice` of 3 rows from
/// 1 (its `len` not a power of two), a gather into a longer axis with repeated rows, and an
/// assembly of the message's bits from a short axis into a long one.
fn cases() -> Vec<(Shape, Vec<Member>)> {
    let even = |count: usize| (0..count).step_by(2).collect::<Vec<_>>();
    let odd = |count: usize| (1..count).step_by(2).collect::<Vec<_>>();
    vec![
        (
            (1, 5, 0, 5),
            vec![(one(), (0..32).map(|i| (i + 7) % 32).collect())],
        ),
        (
            (0, 5, 2, 5),
            vec![(one(), (0..32).map(|i| (i + 13) % 32).collect())],
        ),
        ((1, 5, 0, 4), vec![(one(), even(32))]),
        ((1, 5, 0, 4), vec![(one(), odd(32))]),
        (
            (1, 5, 0, 4),
            vec![(one(), even(32)), (constant(2), odd(32))],
        ),
        (
            (2, 2, 1, 1),
            vec![(one(), even(4)), (constant(1 << 16), odd(4))],
        ),
        ((2, 3, 1, 2), vec![(one(), vec![1, 2, 3])]),
        ((0, 2, 0, 3), vec![(one(), vec![3, 0, 0, 2, 1, 3])]),
        (
            (1, 3, 2, 5),
            vec![(one(), (0..32).map(|i| (5 * i) % 8).collect())],
        ),
        ((0, 0, 3, 0), vec![(constant(9), vec![0])]),
        (
            (2, 3, 1, 3),
            vec![
                (constant(3), vec![1, 2, 3]),
                (-one(), vec![7, 6, 5, 4, 3, 2]),
            ],
        ),
    ]
}

#[test]
fn an_honest_selection_verifies_and_claims_its_input() {
    let mut rng = SplitMix64::new(0x0526_3001);
    let session = Session::random(&mut rng);
    for (n, (shape, members)) in cases().into_iter().enumerate() {
        let case = Case::random(&mut rng, shape, members);
        let relation = case.relation();
        assert_eq!((relation.degree(), relation.vars()), (2, case.axis), "{n}");
        let (proof, point, value) = case.prove(&session);
        assert_eq!(
            proof.rounds.len(),
            case.axis,
            "{n}: rounds over the axis alone"
        );
        assert!(proof.rounds.iter().all(|r| r.len() == 2), "{n}: degree 2");
        assert_eq!(proof.final_evals.len(), 1, "{n}: one evaluation");
        let batch = case
            .verdict(&session, &proof)
            .unwrap_or_else(|e| panic!("case {n}: {e:?}"));
        assert_eq!(batch.point, tv_fp2s(&point), "{n}");
        let x: Vec<TvFp2> = case.x.iter().map(|&v| TvFp2::from(tv_fp(v))).collect();
        let at = tv_fp2s(
            &[
                &case.point[..case.low],
                &point[..],
                &case.point[case.low + case.rows..],
            ]
            .concat(),
        );
        assert_eq!(tv_fp2(value), extension(&x, &at), "{n}: the claim on x");
    }
}

/// `K̃(r_a, r'_k)`'s closed form, over each member's `len` rows, is the extension of the kernel
/// table `K(j, k) = Σ_t a_t·[j < len_t]·[idx_{t,j} = k]` enumerated over the whole padded cube,
/// `y`'s padded rows among it.
#[test]
fn the_kernel_closed_form_is_the_kernels_extension() {
    let mut rng = SplitMix64::new(0x0526_3002);
    for (shape, members) in cases() {
        let case = Case::random(&mut rng, shape, members);
        let (r_a, r_k) = (tv_fp2s(&rng.fp2s(case.rows)), tv_fp2s(&rng.fp2s(case.axis)));
        let mut table = vec![TvFp2::ZERO; 1 << (case.rows + case.axis)];
        for (a, index) in &case.members {
            for (j, &k) in index.iter().enumerate() {
                table[(j << case.axis) | k] += tv_fp2(*a);
            }
        }
        let enumerated = extension(&table, &[&r_k[..], &r_a[..]].concat());
        assert_eq!(case.kernel_at(&r_a, &r_k), enumerated);
    }
}

#[test]
fn a_changed_selected_entry_fails_the_final_check() {
    let mut rng = SplitMix64::new(0x0526_3003);
    let session = Session::random(&mut rng);
    for (n, (shape, members)) in cases().into_iter().enumerate() {
        let honest = Case::random(&mut rng, shape, members);
        // A selected entry: row idx_{0,0} at every other coordinate 0.
        let k = honest.members[0].1[0];
        let mut wrong = honest.clone();
        wrong.x[k << honest.low] += Fp::ONE;
        let (proof, ..) = wrong.prove(&session);
        assert_eq!(
            honest.verdict(&session, &proof).map(|_| ()),
            Err(tv::SumcheckError::FinalMismatch {
                round: honest.axis as u32
            }),
            "case {n}"
        );
        // A row no member selects: the sumcheck holds, and the claim on x is the opening's.
        let selected: Vec<usize> = honest.members.iter().flat_map(|(_, i)| i.clone()).collect();
        if let Some(free) = (0..1usize << honest.axis).find(|k| !selected.contains(k)) {
            let mut moved = honest.clone();
            moved.x[free << honest.low] += Fp::ONE;
            let (proof, ..) = moved.prove(&session);
            assert!(
                honest.verdict(&session, &proof).is_ok(),
                "case {n}: row {free}"
            );
        }
    }
}

#[test]
fn what_does_not_fit_is_refused() {
    let x = vec![Fp::ZERO; 1 << 4];
    assert_eq!(
        Selection::<Fp2>::new(&x[..8], 1, 2, 1).map(|_| ()),
        Err(SumcheckError::InputLength {
            input: 0,
            len: 8,
            vars: 4
        })
    );
    assert_eq!(
        Selection::<Fp2>::new(&x, usize::MAX, 1, 0).map(|_| ()),
        Err(SumcheckError::RelationShape {
            vars: usize::MAX,
            summed_vars: 0
        })
    );
    let relation = Selection::<Fp2>::new(&x, 1, 2, 1).expect("a selection");
    let rows = [0usize, 3];
    let kernel = [Selected {
        coefficient: Fp2::ONE,
        index: &rows,
    }];
    assert!(matches!(
        relation.tables(&[Fp2::ONE; 3], 2, &kernel),
        Err(SumcheckError::PointLength {
            len: 3,
            expected: 4
        })
    ));
    let long = [0usize, 1, 2];
    let too_many = [
        kernel[0],
        Selected {
            coefficient: Fp2::ONE,
            index: &long,
        },
    ];
    assert!(matches!(
        relation.tables(&[Fp2::ONE; 3], 1, &too_many),
        Err(SumcheckError::SelectionRows {
            member: 1,
            len: 3,
            rows: 1,
            vars: 2
        })
    ));
    let outside = [0usize, 4];
    let beyond = [Selected {
        coefficient: Fp2::ONE,
        index: &outside,
    }];
    assert!(matches!(
        relation.tables(&[Fp2::ONE; 3], 1, &beyond),
        Err(SumcheckError::SelectionRows { member: 0, .. })
    ));
    let tables = relation
        .tables(&[Fp2::ONE; 3], 1, &kernel)
        .expect("a point and a kernel");
    assert_eq!(
        tables.claim(&[Fp2::ONE; 3], &[Fp2::ONE]).map(|_| ()),
        Err(SumcheckError::PointLength {
            len: 3,
            expected: 2
        })
    );
    assert_eq!(
        tables.claim(&[Fp2::ONE; 2], &[Fp2::ONE; 2]).map(|_| ()),
        Err(SumcheckError::EvaluationCount {
            len: 2,
            expected: 1
        })
    );
}
