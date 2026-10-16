// SPDX-License-Identifier: Apache-2.0 OR MIT

//! False matmuls and misrouted claims are rejected (RFC-0010 'Relations', 'Testing Strategy'),
//! each at the exact check where the independent verifier stops:
//!
//! - a product with one entry changed, claimed on the true operands, fails its stage's final
//!   check, `FinalMismatch { round: n_max }`: the claims are not absorbed, so every round check
//!   passes and the change reaches the end. So does a prover whose operand differs from the
//!   product's in one entry.
//! - each round value and each final evaluation of an honest stage of matmuls, changed in turn,
//!   fails the final check;
//! - a claim the relation leaves on an operand, routed to the other operand or placed at another
//!   point, fails the final check of the next stage that consumes it, `common::matmul::next_stage`.

mod common;

use common::matmul::{
    ActCase, Member, WeightCase, extension, next_stage, prove_and_verify_stage, prove_stage,
    verify_stage,
};
use common::{Fp, Session, SplitMix64, TestInstance, TestTable, TvFp, tv_fp2s, tv_proof};
use tessera_field::Field;
use tessera_sumcheck::{Claim, Prefix, RightLayout};
use tv_core::sumcheck::{self as tv, SumcheckError};

fn final_mismatch(rounds: usize) -> Result<(), SumcheckError> {
    Err(SumcheckError::FinalMismatch {
        round: rounds as u32,
    })
}

/// The verifier's verdict on the honest proof of `members`, from their true claims except the
/// claim of member `wrong`, which is its product's extension with entry `entry` changed by one,
/// or unchanged for `None`.
fn verdict_on_a_changed_product(
    session: &Session,
    degree: usize,
    members: &[Member],
    wrong: usize,
    entry: Option<usize>,
) -> Result<(), SumcheckError> {
    let stage = prove_stage(session, Prefix::Stage(6), degree, members);
    let mut claims: Vec<tv::Instance> = members.iter().map(Member::tv_instance).collect();
    let (mut product, point) = match &members[wrong] {
        Member::Weight(case) => (case.product(), tv_fp2s(&case.point)),
        Member::Act(case) => (case.product(), tv_fp2s(&case.point)),
        Member::Product(_) => panic!("a matmul"),
    };
    if let Some(entry) = entry {
        let len = product.len();
        product[entry % len] += TvFp::ONE;
    }
    claims[wrong].claim = extension(&product, &point);
    verify_stage(
        session,
        Prefix::Stage(6),
        degree,
        members,
        &claims,
        &tv_proof(stage.proven.proof()),
    )
    .0
    .map(|_| ())
}

#[test]
fn a_product_with_one_entry_changed_fails_the_final_check() {
    let mut rng = SplitMix64::new(0x57_0101);
    let session = Session::random(&mut rng);
    let cases = [
        (Member::Weight(WeightCase::full(&mut rng, (2, 1, 3, 4))), 3),
        (Member::Weight(WeightCase::full(&mut rng, (0, 6, 5, 8))), 2),
        (
            Member::Act(ActCase::full(&mut rng, (2, 3, 4, 2), RightLayout::Direct)),
            3,
        ),
        (
            Member::Act(ActCase::full(
                &mut rng,
                (0, 6, 8, 5),
                RightLayout::Transposed,
            )),
            2,
        ),
    ];
    for (k, (member, degree)) in cases.iter().enumerate() {
        let alone = std::slice::from_ref(member);
        assert_eq!(
            verdict_on_a_changed_product(&session, *degree, alone, 0, None),
            Ok(()),
            "case {k}: the true product"
        );
        for entry in [0, 5, 77] {
            assert_eq!(
                verdict_on_a_changed_product(&session, *degree, alone, 0, Some(entry)),
                final_mismatch(member.vars()),
                "case {k}, entry {entry}"
            );
        }
    }
    // Within a stage of every kind, the smaller instances front-padded to 8 variables.
    let mut stage: Vec<Member> = cases.into_iter().map(|(member, _)| member).collect();
    stage.push(Member::Product(TestInstance::new(vec![
        TestTable::eq(&mut rng, 5),
        TestTable::random(&mut rng, 5, true),
    ])));
    assert_eq!(stage.iter().map(Member::vars).max(), Some(8));
    for wrong in 0..4 {
        assert_eq!(
            verdict_on_a_changed_product(&session, 3, &stage, wrong, Some(13)),
            final_mismatch(8),
            "instance {wrong}"
        );
    }
}

/// The verdict on a prover whose `W`, `X` or `B` differs from the product's in one entry.
#[test]
fn a_prover_with_one_operand_entry_changed_fails_the_final_check() {
    let mut rng = SplitMix64::new(0x57_0102);
    let session = Session::random(&mut rng);
    let weight = WeightCase::full(&mut rng, (2, 2, 3, 4));
    let act = ActCase::full(&mut rng, (1, 2, 3, 2), RightLayout::Transposed);
    let honest = [Member::Weight(weight.clone()), Member::Act(act.clone())];
    let claims: Vec<tv::Instance> = honest.iter().map(Member::tv_instance).collect();
    for (k, operand, entry) in [(0, 0, 9), (0, 1, 40), (1, 1, 30)] {
        let mut cheating = honest.clone();
        match (&mut cheating[k], operand) {
            (Member::Weight(case), 0) => case.w[entry] += Fp::ONE,
            (Member::Weight(case), _) => case.x[entry] += Fp::ONE,
            (Member::Act(case), _) => case.b[entry] += Fp::ONE,
            (Member::Product(_), _) => unreachable!("matmuls only"),
        }
        let stage = prove_stage(&session, Prefix::Stage(7), 3, &cheating);
        let (verdict, _) = verify_stage(
            &session,
            Prefix::Stage(7),
            3,
            &honest,
            &claims,
            &tv_proof(stage.proven.proof()),
        );
        assert_eq!(
            verdict.map(|_| ()),
            final_mismatch(6),
            "instance {k}, operand {operand}"
        );
    }
}

/// A non-zero change: one, or a random element.
fn delta(rng: &mut SplitMix64) -> common::Fp2 {
    loop {
        let x = if rng.coin() {
            common::Fp2::ONE
        } else {
            rng.fp2()
        };
        if x != common::Fp2::ZERO {
            return x;
        }
    }
}

#[test]
fn every_change_to_a_stage_of_matmuls_is_rejected() {
    let mut rng = SplitMix64::new(0x57_0104);
    let session = Session::random(&mut rng);
    let members = [
        Member::Weight(WeightCase::full(&mut rng, (2, 1, 2, 3))),
        Member::Act(ActCase::full(
            &mut rng,
            (1, 2, 3, 2),
            RightLayout::Transposed,
        )),
        Member::Act(ActCase::full(&mut rng, (0, 3, 4, 1), RightLayout::Direct)),
        Member::Weight(WeightCase::full(&mut rng, (0, 0, 3, 2))),
    ];
    let stage = prove_stage(&session, Prefix::Stage(5), 3, &members);
    let claims: Vec<tv::Instance> = members.iter().map(Member::tv_instance).collect();
    let check = |proof: &tessera_sumcheck::SumcheckProof<common::Fp2>| {
        verify_stage(
            &session,
            Prefix::Stage(5),
            3,
            &members,
            &claims,
            &tv_proof(proof),
        )
        .0
        .map(|_| ())
    };
    let proof = stage.proven.proof();
    assert_eq!(check(proof), Ok(()), "the honest stage verifies");
    let mut changes = 0;
    for j in 0..proof.rounds.len() {
        for m in 0..3 {
            let mut changed = proof.clone();
            changed.rounds[j][m] += delta(&mut rng);
            assert_eq!(check(&changed), final_mismatch(5), "round {j}, value {m}");
            changes += 1;
        }
    }
    for e in 0..proof.final_evals.len() {
        let mut changed = proof.clone();
        changed.final_evals[e] += delta(&mut rng);
        assert_eq!(check(&changed), final_mismatch(5), "final evaluation {e}");
        changes += 1;
    }
    assert_eq!(changes, 5 * 3 + 4 * 2);
}

/// The claim `claim` with its point's coordinates `range` moved to the front.
fn moved_to_front(
    claim: &Claim<common::Fp2>,
    range: core::ops::Range<usize>,
) -> Claim<common::Fp2> {
    let point = &claim.point;
    Claim {
        point: [
            &point[range.clone()],
            &point[..range.start],
            &point[range.end..],
        ]
        .concat(),
        value: claim.value,
    }
}

#[test]
fn misrouted_claims_fail_the_next_stage() {
    let mut rng = SplitMix64::new(0x57_0103);
    let session = Session::random(&mut rng);
    // 2^2 layers of X, 2^2 tokens by 2^4 inputs, by W, 2^3 outputs by 2^4 inputs; A and B: 2^1
    // products of 2^3 × 2^3 by 2^3 × 2^3, so that a claim on A fits B's table; K: 2^2 × 2^3 by
    // 2^3 × 2^2, stored transposed.
    let weight = WeightCase::full(&mut rng, (2, 2, 3, 4));
    let direct = ActCase::full(&mut rng, (1, 3, 3, 3), RightLayout::Direct);
    let transposed = ActCase::full(&mut rng, (0, 2, 3, 2), RightLayout::Transposed);
    let members = [
        Member::Weight(weight.clone()),
        Member::Act(direct.clone()),
        Member::Act(transposed.clone()),
    ];
    let stage = prove_and_verify_stage(&session, Prefix::Stage(8), 3, &members);
    let [w, x] = stage.claims[0].clone().expect("claims on W and X");
    let [a, b] = stage.claims[1].clone().expect("claims on A and B");
    let [a_t, k] = stage.claims[2].clone().expect("claims on A and K");
    let honest: [(&[Fp], &Claim<common::Fp2>); 6] = [
        (&weight.w, &w),
        (&weight.x, &x),
        (&direct.a, &a),
        (&direct.b, &b),
        (&transposed.a, &a_t),
        (&transposed.b, &k),
    ];
    assert_eq!(
        next_stage(&session, Prefix::Stage(9), &honest),
        Ok(()),
        "the honest claims pass the next stage"
    );

    // W's claim is at r_i (4) ‖ r_o (3) ‖ ρ_ℓ (2), X's at r_i (4) ‖ r_t (2) ‖ ρ_ℓ (2), and the
    // input point is r_o (3) ‖ r_t (2) ‖ ρ (2).
    let rho = &weight.point[5..];
    // The activations of each layer's first token: the X of a batch of one token.
    let first_tokens: Vec<Fp> = weight
        .x
        .chunks_exact(4 << 4)
        .flat_map(|layer| layer[..1 << 4].iter().copied())
        .collect();
    let misrouted: [(&str, &[Fp], Claim<common::Fp2>); 11] = [
        (
            "X's claim carrying W's value",
            &weight.x,
            Claim {
                point: x.point.clone(),
                value: w.value,
            },
        ),
        (
            "W's claim at r_o ‖ r_i ‖ ρ_ℓ",
            &weight.w,
            moved_to_front(&w, 4..7),
        ),
        (
            "W's claim at the input's layer coordinate",
            &weight.w,
            Claim {
                point: [&w.point[..7], rho].concat(),
                value: w.value,
            },
        ),
        (
            "X's claim at the input's layer coordinate",
            &weight.x,
            Claim {
                point: [&x.point[..6], rho].concat(),
                value: x.value,
            },
        ),
        (
            "X's claim at r_t ‖ r_i ‖ ρ_ℓ",
            &weight.x,
            moved_to_front(&x, 4..6),
        ),
        (
            "X's claim without r_t, on the first token's activations",
            &first_tokens,
            Claim {
                point: [&x.point[..4], &x.point[6..]].concat(),
                value: x.value,
            },
        ),
        ("A's claim on B", &direct.b, a.clone()),
        ("B's claim on A", &direct.a, b.clone()),
        (
            "A's claim at r_a ‖ r_j ‖ ρ_b",
            &direct.a,
            moved_to_front(&a, 3..6),
        ),
        (
            "B's claim at r_j ‖ r_c ‖ ρ_b",
            &direct.b,
            moved_to_front(&b, 3..6),
        ),
        (
            "K's claim at r_c ‖ r_j ‖ ρ_b",
            &transposed.b,
            moved_to_front(&k, 3..5),
        ),
    ];
    for (what, table, claim) in &misrouted {
        assert_eq!(
            next_stage(&session, Prefix::Stage(9), &[(table, claim)]),
            final_mismatch(claim.point.len()),
            "{what}"
        );
        // Among the honest claims, at the largest of their sizes.
        let mut all = honest.to_vec();
        all.push((table, claim));
        assert_eq!(
            next_stage(&session, Prefix::Stage(9), &all),
            final_mismatch(9),
            "{what}, among the honest claims"
        );
    }
}
