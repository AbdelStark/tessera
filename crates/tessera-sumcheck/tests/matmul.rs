// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Honest matmul relations through the independent verifier (DEP-2; RFC-0010 'Relations',
//! 'Batching within a stage'): every stage is proven by the relations' tables and claims around
//! `prove_batch`, and checked by `tv_core::sumcheck::verify_batch` with the relations written in
//! `common::matmul` from the RFC, from the true products' extensions at the input points. The
//! verifier must accept and draw the prover's challenges, and every claim on an operand must be
//! at the point the verifier derives, with the evaluation the verifier read, which must be the
//! operand's multilinear extension there, summed over the hypercube.

mod common;

use common::matmul::{ActCase, Member, WeightCase, prove_and_verify_stage};
use common::{Session, SplitMix64, TestInstance, TestTable, tv_fp2s};
use tessera_codec::SecretBuf;
use tessera_field::Field;
use tessera_sumcheck::{
    MatMulActShape, MatMulWeightShape, MatMulWeightTables, Operand, Prefix, Protocol, RightLayout,
    SumcheckError, prove_batch,
};

/// The issue's shapes: `2^6×2^8` by `2^8×2^5` as one `MatMulAct`, with `B` stored either way,
/// and stacked over 2^2 products; and as a `MatMulWeight`, `X` of `2^6` tokens by `2^8` inputs
/// times `Wᵀ`, `W` of `2^5` outputs by `2^8` inputs, alone and stacked over 2^2 layers. Each is
/// alone in its stage, of its relation's degree: 2 for one product, 3 stacked.
#[test]
fn the_issue_shapes_verify() {
    let mut rng = SplitMix64::new(0x57_0001);
    let session = Session::random(&mut rng);
    for (k, layout) in [RightLayout::Direct, RightLayout::Transposed]
        .into_iter()
        .enumerate()
    {
        for (stack, degree) in [(0, 2), (2, 3)] {
            let case = ActCase::full(&mut rng, (stack, 6, 8, 5), layout);
            assert_eq!(case.relation().degree(), degree);
            let stage = prove_and_verify_stage(
                &session,
                Prefix::Stage(k as u32),
                degree,
                &[Member::Act(case)],
            );
            assert_eq!(stage.proven.proof().rounds.len(), 8 + stack);
            assert_eq!(stage.proven.proof().final_evals.len(), 2);
        }
    }
    for (layers, degree) in [(0, 2), (2, 3)] {
        let case = WeightCase::full(&mut rng, (layers, 6, 5, 8));
        assert_eq!(case.relation().degree(), degree);
        let stage =
            prove_and_verify_stage(&session, Prefix::Stage(9), degree, &[Member::Weight(case)]);
        assert_eq!(stage.proven.proof().rounds.len(), 8 + layers);
        assert_eq!(stage.proven.proof().final_evals.len(), 2);
    }
}

/// Stacked counts from 2 to 9 layers of `2^3×2^4` weights by 1, 2 or 4 tokens' activations, and 2,
/// 3, 5, 6 and 7 products with `B` stored either way, those that are not powers of two padded
/// with zero layers or products, each in a stage of its relation's degree.
#[test]
fn stacked_layers_padded_with_zero_layers_verify() {
    let mut rng = SplitMix64::new(0x57_0002);
    let session = Session::random(&mut rng);
    for layers in 2..=9usize {
        let layer_vars = layers.next_power_of_two().trailing_zeros() as usize;
        let shape = MatMulWeightShape {
            layer_vars,
            token_vars: layers % 3,
            out_vars: 3,
            in_vars: 4,
        };
        let case = WeightCase::random(&mut rng, shape, layers);
        let degree = case.relation().degree();
        let _ = prove_and_verify_stage(&session, Prefix::Stage(1), degree, &[Member::Weight(case)]);
    }
    for count in [2usize, 3, 5, 6, 7] {
        for layout in [RightLayout::Direct, RightLayout::Transposed] {
            let shape = MatMulActShape {
                stack_vars: count.next_power_of_two().trailing_zeros() as usize,
                row_vars: 3,
                inner_vars: 4,
                col_vars: 2,
            };
            let case = ActCase::random(&mut rng, shape, layout, count);
            let degree = case.relation().degree();
            let _ =
                prove_and_verify_stage(&session, Prefix::Stage(2), degree, &[Member::Act(case)]);
        }
    }
}

/// A single layer or product has no `eq` factor, and proves in a stage of degree 2, which a
/// third table would exceed, for one token and for several; in a stage of degree 3 beside a
/// stacked instance it verifies too.
#[test]
fn a_single_layer_proves_at_degree_2() {
    let mut rng = SplitMix64::new(0x57_0003);
    let session = Session::random(&mut rng);
    let token = WeightCase::full(&mut rng, (0, 0, 4, 5));
    let tokens = WeightCase::full(&mut rng, (0, 3, 4, 5));
    let act = ActCase::full(&mut rng, (0, 3, 4, 3), RightLayout::Transposed);
    let singles = [
        Member::Weight(token.clone()),
        Member::Weight(tokens.clone()),
        Member::Act(act.clone()),
    ];
    let _ = prove_and_verify_stage(&session, Prefix::Stage(3), 2, &singles);
    let mixed = [
        Member::Weight(token),
        Member::Weight(WeightCase::full(&mut rng, (2, 2, 4, 5))),
        Member::Weight(tokens),
        Member::Act(act),
    ];
    let _ = prove_and_verify_stage(&session, Prefix::Stage(4), 3, &mixed);
}

/// `B` stored transposed, as `K` in `QKᵀ`, contracts to the same table as `B` stored as it is:
/// both proofs are the same messages, and the claims on `B` and on `K` have the same value at the
/// same coordinates in the two orders, `r_c ‖ r_j ‖ ρ_b` and `r_j ‖ r_c ‖ ρ_b`.
#[test]
fn transposed_right_operands_verify_with_reordered_claims() {
    let mut rng = SplitMix64::new(0x57_0004);
    let session = Session::random(&mut rng);
    for shape in [
        (0, 6, 8, 5),
        (2, 3, 4, 3),
        (1, 0, 3, 4),
        (1, 3, 0, 2),
        (0, 2, 2, 1),
        (3, 1, 1, 0),
    ] {
        let direct = ActCase::full(&mut rng, shape, RightLayout::Direct);
        let transposed = direct.relaid();
        assert_eq!(transposed.layout, RightLayout::Transposed);
        assert!(
            transposed.product() == direct.product(),
            "the same products"
        );
        let degree = direct.relation().degree();
        let one =
            prove_and_verify_stage(&session, Prefix::Stage(5), degree, &[Member::Act(direct)]);
        let other = prove_and_verify_stage(
            &session,
            Prefix::Stage(5),
            degree,
            &[Member::Act(transposed)],
        );
        assert!(
            one.proven.proof() == other.proven.proof(),
            "{shape:?}: the same messages"
        );
        let ([a, b], [a_t, k]) = (
            one.claims[0].clone().expect("claims"),
            other.claims[0].clone().expect("claims"),
        );
        assert!(a == a_t, "{shape:?}: the same claim on A");
        assert_eq!(b.value, k.value, "{shape:?}: the same value");
        let (_, inner_vars, col_vars) = (shape.0, shape.2, shape.3);
        let reordered = [
            &b.point[col_vars..col_vars + inner_vars],
            &b.point[..col_vars],
            &b.point[col_vars + inner_vars..],
        ]
        .concat();
        assert_eq!(
            tv_fp2s(&k.point),
            tv_fp2s(&reordered),
            "{shape:?}: K's point is B's, reordered"
        );
    }
}

/// Stages of relations of every kind and size, front-padded to the largest, in several orders:
/// two weight types stacked over 2^3 layers with 6 live, for 2^2 tokens and for one; a single
/// layer for one token; weight matrices of one output row, for 2^1 tokens over 2^1 layers and for
/// 2^3 tokens alone; stacked products with `B` either way, and outer products, of no inner
/// variable, over 2^2 products and alone, on no variable at all; and a product of tables with a
/// public `eq`, as the engine's own tests batch. The two weight types' claims share their layer
/// coordinate, the stage's last 3 challenges, as RFC-0012 opens each type at one point.
#[test]
fn stages_of_several_relations_verify() {
    let mut rng = SplitMix64::new(0x57_0005);
    let session = Session::random(&mut rng);
    let weight = |layer_vars, token_vars, out_vars, in_vars| MatMulWeightShape {
        layer_vars,
        token_vars,
        out_vars,
        in_vars,
    };
    let act = |stack_vars, row_vars, inner_vars, col_vars| MatMulActShape {
        stack_vars,
        row_vars,
        inner_vars,
        col_vars,
    };
    let members = vec![
        Member::Weight(WeightCase::random(&mut rng, weight(3, 2, 4, 6), 6)),
        Member::Act(ActCase::random(
            &mut rng,
            act(1, 3, 5, 2),
            RightLayout::Direct,
            2,
        )),
        Member::Act(ActCase::random(
            &mut rng,
            act(2, 2, 3, 3),
            RightLayout::Transposed,
            3,
        )),
        Member::Weight(WeightCase::full(&mut rng, (0, 0, 5, 4))),
        Member::Product(TestInstance::new(vec![
            TestTable::eq(&mut rng, 7),
            TestTable::random(&mut rng, 7, true),
            TestTable::random(&mut rng, 7, false),
        ])),
        Member::Act(ActCase::full(
            &mut rng,
            (0, 4, 0, 3),
            RightLayout::Transposed,
        )),
        Member::Act(ActCase::full(&mut rng, (2, 2, 0, 2), RightLayout::Direct)),
        Member::Weight(WeightCase::full(&mut rng, (1, 1, 0, 3))),
        Member::Weight(WeightCase::full(&mut rng, (0, 3, 0, 2))),
        Member::Weight(WeightCase::random(&mut rng, weight(3, 0, 2, 5), 6)),
    ];
    let n = members.len();
    for (k, rotation) in [0, 3, 5].into_iter().enumerate() {
        let mut order = members.clone();
        order.rotate_left(rotation);
        let stage = prove_and_verify_stage(&session, Prefix::Stage(10 + k as u32), 3, &order);
        assert_eq!(stage.proven.proof().rounds.len(), 9);
        assert_eq!(stage.proven.instances(), n);
        let layers = &stage.proven.point()[9 - 3..];
        let stacked: Vec<&[common::Fp2]> = order
            .iter()
            .zip(&stage.claims)
            .filter(
                |(member, _)| matches!(member, Member::Weight(case) if case.shape.layer_vars == 3),
            )
            .map(|(_, claims)| {
                let w = &claims.as_ref().expect("claims")[0].point;
                &w[w.len() - 3..]
            })
            .collect();
        assert_eq!(stacked.len(), 2, "two stacked weight types");
        assert!(
            stacked.iter().all(|coordinate| *coordinate == layers),
            "one layer coordinate for every stacked weight type"
        );
    }
    let mut reversed = members;
    reversed.reverse();
    let _ = prove_and_verify_stage(&session, Prefix::Tree(3), 4, &reversed);
}

/// Instances on 1 to 24 variables (RFC-0010 'Testing Strategy'), alternately weights and products
/// with `B` stored either way: up to 18 variables stacked over a quarter of them, with one
/// variable to contract on each side, 2 tokens by 2 outputs for the weights; above, a single layer
/// or product with nothing to contract, one token for the weights, so that the operands stay at
/// the instance's size.
#[test]
fn instances_on_1_to_24_variables_verify() {
    let mut rng = SplitMix64::new(0x57_0007);
    let session = Session::random(&mut rng);
    for vars in 1..=24usize {
        let small = vars <= 18;
        let stack = if small { vars / 4 } else { 0 };
        let side = usize::from(small);
        let member = if vars % 2 == 0 {
            Member::Weight(WeightCase::full(
                &mut rng,
                (stack, side, side, vars - stack),
            ))
        } else {
            let layout = if vars % 4 == 1 {
                RightLayout::Direct
            } else {
                RightLayout::Transposed
            };
            Member::Act(ActCase::full(
                &mut rng,
                (stack, side, vars - stack, side),
                layout,
            ))
        };
        let degree = match &member {
            Member::Weight(case) => case.relation().degree(),
            Member::Act(case) => case.relation().degree(),
            Member::Product(_) => unreachable!("matmuls only"),
        };
        assert_eq!(member.vars(), vars);
        let _ = prove_and_verify_stage(&session, Prefix::Stage(vars as u32), degree, &[member]);
    }
}

/// Seeded random stages: 1 to 4 relations of random kinds, shapes of 0 to 4 variables per index,
/// tokens among them, random live counts and layouts, at a degree between the largest relation's
/// and 7.
#[test]
fn seeded_random_stages_verify() {
    let mut rng = SplitMix64::new(0x57_0006);
    let session = Session::random(&mut rng);
    for case in 0..40u32 {
        let count = 1 + rng.below(4) as usize;
        let vars = |rng: &mut SplitMix64| rng.below(5) as usize;
        let members: Vec<Member> = (0..count)
            .map(|_| {
                if rng.coin() {
                    let shape = MatMulWeightShape {
                        layer_vars: vars(&mut rng),
                        token_vars: vars(&mut rng),
                        out_vars: vars(&mut rng),
                        in_vars: vars(&mut rng),
                    };
                    let live = 1 + rng.below(1 << shape.layer_vars) as usize;
                    Member::Weight(WeightCase::random(&mut rng, shape, live))
                } else {
                    let shape = MatMulActShape {
                        stack_vars: vars(&mut rng),
                        row_vars: vars(&mut rng),
                        inner_vars: vars(&mut rng),
                        col_vars: vars(&mut rng),
                    };
                    let layout = if rng.coin() {
                        RightLayout::Direct
                    } else {
                        RightLayout::Transposed
                    };
                    let live = 1 + rng.below(1 << shape.stack_vars) as usize;
                    Member::Act(ActCase::random(&mut rng, shape, layout, live))
                }
            })
            .collect();
        let needed = members
            .iter()
            .map(|member| match member {
                Member::Weight(case) => case.relation().degree(),
                Member::Act(case) => case.relation().degree(),
                Member::Product(_) => 1,
            })
            .max()
            .unwrap_or(2);
        let degree = needed + rng.below((8 - needed) as u64) as usize;
        let _ = prove_and_verify_stage(&session, Prefix::Stage(case), degree, &members);
    }
}

/// `W̃(ℓ, r_o, i) = Σ_o eq(r_o, o)·W[ℓ, o, i]` at `ℓ·2^{n_i} + i`, from the definition: `eq`'s
/// product formula at each output, summed entry by entry.
fn contracted(case: &WeightCase) -> SecretBuf<common::Fp2> {
    let MatMulWeightShape {
        layer_vars,
        out_vars,
        in_vars,
        ..
    } = case.shape;
    let r_o = &case.point[..out_vars];
    let (outs, ins) = (1usize << out_vars, 1usize << in_vars);
    let mut table = SecretBuf::filled(common::Fp2::ZERO, ins << layer_vars);
    for layer in 0..1usize << layer_vars {
        for o in 0..outs {
            let eq = r_o.iter().enumerate().fold(common::Fp2::ONE, |e, (t, &r)| {
                e * if (o >> t) & 1 == 1 {
                    r
                } else {
                    common::Fp2::ONE - r
                }
            });
            for i in 0..ins {
                table[layer * ins + i] +=
                    eq * common::Fp2::from(case.w[(layer * outs + o) * ins + i]);
            }
        }
    }
    table
}

/// RFC-0012 'Opening by the folding pass': the weight commitment's fold pass gives the matmul its
/// weight operand contracted already, `W̃(ℓ, r_o, i)`, so that a proof reads the weights once
/// (INV-WC-4). Built from that table, here from its definition, an instance proves the stage the
/// reference path proves by contracting `W` itself: the same messages, `γ`, point, claims and
/// final transcript state, which the independent verifier accepts. Stacked layers with padded
/// ones, a single layer, one token and one output row are among the shapes.
#[test]
fn a_folded_weight_table_proves_the_stage_contracting_w_proves() {
    let mut rng = SplitMix64::new(0x57_0108);
    let session = Session::random(&mut rng);
    for ((layer_vars, token_vars, out_vars, in_vars), live) in [
        ((0, 0, 4, 5), 1),
        ((0, 3, 5, 4), 1),
        ((2, 3, 4, 3), 3),
        ((1, 2, 0, 3), 2),
        ((3, 1, 2, 4), 5),
        ((2, 0, 3, 0), 4),
    ] {
        let shape = MatMulWeightShape {
            layer_vars,
            token_vars,
            out_vars,
            in_vars,
        };
        let case = WeightCase::random(&mut rng, shape, live);
        let degree = case.relation().degree();
        let prefix = Prefix::Stage(2);
        let reference =
            prove_and_verify_stage(&session, prefix, degree, &[Member::Weight(case.clone())]);
        let tables = MatMulWeightTables::folded(contracted(&case), &case.x, shape, &case.point)
            .expect("a table of the shape");
        let mut transcript = session.transcript();
        let proven = prove_batch(
            &mut transcript,
            Protocol::One,
            prefix,
            degree,
            &[tables.instance()],
        )
        .expect("a stage");
        assert!(
            proven.proof() == reference.proven.proof(),
            "{shape:?}: the same messages"
        );
        assert_eq!(proven.gamma(), reference.proven.gamma(), "{shape:?}");
        assert_eq!(proven.point(), reference.proven.point(), "{shape:?}");
        assert_eq!(transcript.state(), reference.state, "{shape:?}");
        let claims = tables
            .claims(
                proven.instance_point(0).expect("the instance"),
                proven.instance_evals(0).expect("the instance"),
            )
            .expect("its results");
        let [w, x] = reference.claims[0].clone().expect("claims");
        assert!(claims.w == w && claims.x == x, "{shape:?}: the same claims");
    }
}

/// A folded table or activations of another length than the shape gives, and a point of
/// another length, are refused before anything is computed, `W` first.
#[test]
fn a_folded_weight_table_of_another_shape_is_refused() {
    let mut rng = SplitMix64::new(0x57_0109);
    let shape = MatMulWeightShape {
        layer_vars: 2,
        token_vars: 1,
        out_vars: 3,
        in_vars: 4,
    };
    let case = WeightCase::random(&mut rng, shape, 3);
    let table = || contracted(&case);
    let refused = |w: SecretBuf<common::Fp2>, x: &[common::Fp], point: &[common::Fp2]| {
        MatMulWeightTables::folded(w, x, shape, point)
            .map(|_| ())
            .expect_err("refused")
    };
    let mut short = table();
    short.truncate(short.len() - 1);
    assert_eq!(
        refused(short, &case.x[1..], &case.point),
        SumcheckError::OperandLength {
            operand: Operand::W,
            len: (1 << 6) - 1,
            expected: 1 << 6,
        }
    );
    assert_eq!(
        refused(table(), &case.x[1..], &case.point),
        SumcheckError::OperandLength {
            operand: Operand::X,
            len: (1 << 7) - 1,
            expected: 1 << 7,
        }
    );
    assert_eq!(
        refused(table(), &case.x, &case.point[1..]),
        SumcheckError::PointLength {
            len: 5,
            expected: 6,
        }
    );
}
