// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The messages as values: `SumcheckProof` encodes as RFC-0010 defines it, in TCE (RFC-0003),
//! and decodes back; and no debug form of the crate shows a table entry or a leaf, which are
//! secret-class, or a message, challenge or claim value, which are protocol-class (05 §3).

mod common;

use common::{Fp, Fp2, Session, SplitMix64, TestInstance, TestTable, prove_and_verify};
use tessera_codec::{DecodeLimits, decode, encode};
use tessera_field::{Field, PrimeField, QuadraticExtension};
use tessera_sumcheck::{
    Claim, Elementwise, ElementwiseShape, Forest, Input, Instance, Linear, LinearClaim,
    LinearInput, MatMulAct, MatMulActShape, MatMulWeight, MatMulWeightShape, Placement, Prefix,
    RightLayout, SumcheckProof, Table, Term, prove_batch, prove_forests,
};

fn element_bytes(x: Fp2) -> Vec<u8> {
    let [c0, c1] = x.coeffs();
    [
        c0.to_canonical_u64().to_le_bytes(),
        c1.to_canonical_u64().to_le_bytes(),
    ]
    .concat()
}

/// `rounds` as `Vec<Vec<Fp2>>`, then `final_evals` as `Vec<Fp2>`: `u64` lengths, little-endian,
/// and each element `c0 ‖ c1`.
#[test]
fn a_proof_encodes_as_its_rounds_then_its_final_evaluations() {
    let mut rng = SplitMix64::new(0x53_0501);
    let (a, b, c, d, e) = (rng.fp2(), rng.fp2(), rng.fp2(), rng.fp2(), rng.fp2());
    let proof = SumcheckProof {
        rounds: vec![vec![a, b], vec![c, d]],
        final_evals: vec![e],
    };
    let mut expected = 2u64.to_le_bytes().to_vec();
    for round in [[a, b], [c, d]] {
        expected.extend(2u64.to_le_bytes());
        for x in round {
            expected.extend(element_bytes(x));
        }
    }
    expected.extend(1u64.to_le_bytes());
    expected.extend(element_bytes(e));
    assert_eq!(encode(&proof), expected);
    assert_eq!(
        decode::<SumcheckProof<Fp2>>(&expected, DecodeLimits::MESSAGE),
        Ok(proof)
    );
}

#[test]
fn proofs_round_trip_through_their_encoding() {
    let mut rng = SplitMix64::new(0x53_0502);
    let session = Session::random(&mut rng);
    for (degree, shapes) in [
        (3, vec![(3, 3), (7, 2), (12, 3)]),
        (1, vec![(0, 1)]),
        (7, vec![(4, 7), (0, 2)]),
    ] {
        let batch: Vec<TestInstance> = shapes
            .into_iter()
            .map(|(vars, k)| TestInstance::random(&mut rng, vars, k, |t| t % 2 == 1))
            .collect();
        let proven = prove_and_verify(&session, Prefix::Stage(0), degree, &batch);
        let proof = proven.into_proof();
        let bytes = encode(&proof);
        assert_eq!(
            decode::<SumcheckProof<Fp2>>(&bytes, DecodeLimits::MESSAGE),
            Ok(proof)
        );
    }
}

/// Every decimal and hexadecimal form of the coefficients of `values` that is long enough to be
/// recognized.
fn forms(values: &[Fp2]) -> Vec<String> {
    values
        .iter()
        .flat_map(|x| x.coeffs())
        .map(|c| c.to_canonical_u64())
        .filter(|&c| c >= 1 << 20)
        .flat_map(|c| [c.to_string(), format!("{c:x}"), format!("{c:X}")])
        .collect()
}

fn assert_hidden(debug: &str, secrets: &[String], what: &str) {
    for secret in secrets {
        assert!(
            !debug.contains(secret.as_str()),
            "{what} shows {secret}: {debug}"
        );
    }
}

#[test]
fn debug_forms_show_shapes_never_values() {
    let mut rng = SplitMix64::new(0x53_0503);
    let base: Vec<Fp> = (0..8)
        .map(|_| Fp::from_u64_reduced(rng.next_u64() | 1 << 40))
        .collect();
    let ext: Vec<Fp2> = rng.fp2s(8);
    let mut secrets = forms(&ext);
    secrets.extend(forms(
        &base.iter().map(|&b| Fp2::from(b)).collect::<Vec<_>>(),
    ));

    let table = Table::<Fp2>::base(&base);
    assert_eq!(
        format!("{table:?}"),
        "Table { field: \"base\", len: 8, public: false }"
    );
    let instance = Instance::new(vec![table, Table::ext(&ext).public()]);
    let shown = format!("{instance:?}");
    assert_hidden(&shown, &secrets, "an instance");
    assert!(shown.contains("field: \"extension\", len: 8, public: true"));

    let session = Session::random(&mut rng);
    let batch = [
        TestInstance::new(vec![
            TestTable::base(base.clone()),
            TestTable::ext(ext.clone()),
        ]),
        TestInstance::random(&mut rng, 1, 2, |_| true),
    ];
    let proven = prove_and_verify(&session, Prefix::Stage(0), 2, &batch);
    let proof = proven.proof();
    let mut messages: Vec<Fp2> = proof.rounds.concat();
    messages.extend(&proof.final_evals);
    messages.push(*proven.gamma());
    messages.extend(proven.point());
    secrets.extend(forms(&messages));
    assert_eq!(
        format!("{proof:?}"),
        "SumcheckProof { round_lengths: [2, 2, 2], final_evals: 4 }"
    );
    assert_hidden(&format!("{proven:?}"), &secrets, "a proven batch");
    assert_hidden(&format!("{proven:#?}"), &secrets, "a proven batch");
}

#[test]
fn forest_debug_forms_show_shapes_never_values() {
    let mut rng = SplitMix64::new(0x58_0503);
    let leaves: Vec<Fp> = (0..32)
        .map(|_| Fp::from_u64_reduced(rng.next_u64() | 1 << 40))
        .collect();
    let mut secrets = forms(&leaves.iter().map(|&b| Fp2::from(b)).collect::<Vec<_>>());
    let forest = Forest::<Fp2>::new(&leaves, 3);
    assert_eq!(format!("{forest:?}"), "Forest { leaves: 32, height: 3 }");

    let session = Session::random(&mut rng);
    let proven = prove_forests(&mut session.transcript(), &[forest]).expect("a forest");
    let mut values = Vec::new();
    for level in proven.levels() {
        values.extend(level.root_challenges());
        values.push(*level.gamma());
        values.extend(level.proof().rounds.concat());
        values.extend(&level.proof().final_evals);
        values.extend(level.point());
        values.push(*level.rho());
        for claim in level.next_claims() {
            values.extend(&claim.point);
            values.push(claim.value);
        }
    }
    secrets.extend(forms(&values));
    assert_eq!(
        format!("{:?}", proven.leaf_claims()[0]),
        "Claim { vars: 5, .. }"
    );
    assert_hidden(&format!("{proven:?}"), &secrets, "proven forests");
    assert_hidden(&format!("{proven:#?}"), &secrets, "proven forests");
    assert_hidden(
        &format!("{:?}", proven.levels()[1]),
        &secrets,
        "a proven level",
    );
}

/// `Σ_m eq(r, m)·T[(l·M + m)·I + i]` for each `(l, i)`, `M = 2^{|r|}` and `I = inner`: an
/// operand's middle variables contracted with `eq(r, ·)`, from the definition.
fn contracted(table: &[Fp], r: &[Fp2], inner: usize) -> Vec<Fp2> {
    let weights = tessera_poly::eq_table(r).expect("a small table");
    let rows = weights.len();
    (0..table.len() / rows)
        .map(|k| {
            let (layer, column) = (k / inner, k % inner);
            (0..rows).fold(Fp2::ZERO, |sum, row| {
                sum + weights[row] * Fp2::from(table[(layer * rows + row) * inner + column])
            })
        })
        .collect()
}

#[test]
fn matmul_debug_forms_show_shapes_never_values() {
    let mut rng = SplitMix64::new(0x57_0503);
    let mut large = |n: usize| -> Vec<Fp> {
        (0..n)
            .map(|_| Fp::from_u64_reduced(rng.next_u64() | 1 << 40))
            .collect()
    };
    let (w, x, a, b) = (large(64), large(32), large(64), large(32));
    let lift = |values: &[Fp]| values.iter().map(|&v| Fp2::from(v)).collect::<Vec<_>>();
    let mut secrets: Vec<String> = [&w, &x, &a, &b]
        .into_iter()
        .flat_map(|operand| forms(&lift(operand)))
        .collect();
    let session = Session::random(&mut rng);

    // 2^1 layers of 2^1 tokens by 2^3 inputs, by 2^2 outputs; the input point r_o ‖ r_t ‖ ρ.
    let shape = MatMulWeightShape {
        layer_vars: 1,
        token_vars: 1,
        out_vars: 2,
        in_vars: 3,
    };
    let weight = MatMulWeight::<Fp2>::new(&w, &x, shape).expect("a shape");
    let shown =
        "shape: MatMulWeightShape { layer_vars: 1, token_vars: 1, out_vars: 2, in_vars: 3 }, ..";
    assert_eq!(format!("{weight:?}"), format!("MatMulWeight {{ {shown} }}"));
    let point = rng.fp2s(4);
    let tables = weight.tables(&point).expect("a point");
    secrets.extend(forms(&contracted(&w, &point[..2], 8)));
    secrets.extend(forms(&contracted(&x, &point[2..3], 8)));
    assert_eq!(
        format!("{tables:?}"),
        format!("MatMulWeightTables {{ {shown} }}")
    );
    assert_hidden(&format!("{:?}", tables.instance()), &secrets, "an instance");
    let proven = prove_batch(
        &mut session.transcript(),
        Prefix::Stage(0),
        3,
        &[tables.instance()],
    )
    .expect("a stage");
    let claims = tables
        .claims(
            proven.instance_point(0).expect("an instance"),
            proven.instance_evals(0).expect("an instance"),
        )
        .expect("the instance's results");
    secrets.extend(forms(&[claims.w.value, claims.x.value]));
    assert_eq!(
        format!("{claims:?}"),
        "MatMulWeightClaims { w: Claim { vars: 6, .. }, x: Claim { vars: 5, .. } }"
    );

    let shape = MatMulActShape {
        stack_vars: 1,
        row_vars: 2,
        inner_vars: 3,
        col_vars: 1,
    };
    let act = MatMulAct::<Fp2>::new(&a, &b, RightLayout::Transposed, shape).expect("a shape");
    let shown = "shape: MatMulActShape { stack_vars: 1, row_vars: 2, inner_vars: 3, col_vars: 1 }, \
                 layout: Transposed, ..";
    assert_eq!(format!("{act:?}"), format!("MatMulAct {{ {shown} }}"));
    let tables = act.tables(&rng.fp2s(4)).expect("a point");
    assert_eq!(
        format!("{tables:?}"),
        format!("MatMulActTables {{ {shown} }}")
    );
    let proven = prove_batch(
        &mut session.transcript(),
        Prefix::Stage(0),
        3,
        &[tables.instance()],
    )
    .expect("a stage");
    let claims = tables
        .claims(
            proven.instance_point(0).expect("an instance"),
            proven.instance_evals(0).expect("an instance"),
        )
        .expect("the instance's results");
    secrets.extend(forms(&[claims.a.value, claims.b.value]));
    assert_eq!(
        format!("{claims:?}"),
        "MatMulActClaims { a: Claim { vars: 6, .. }, b: Claim { vars: 5, .. } }"
    );
    assert_hidden(&format!("{claims:#?}"), &secrets, "claims");
    assert_hidden(&format!("{tables:#?}"), &secrets, "tables");
    assert_hidden(&format!("{weight:#?}"), &secrets, "a relation");
}

#[test]
fn elementwise_and_linear_debug_forms_show_shapes_never_values() {
    let mut rng = SplitMix64::new(0x99_0503);
    let mut large = |n: usize| -> Vec<Fp> {
        (0..n)
            .map(|_| Fp::from_u64_reduced(rng.next_u64() | 1 << 40))
            .collect()
    };
    let (x, cos, sin, r) = (large(16), large(16), large(16), large(4));
    let weight = Fp2::from(Fp::from_u64_reduced(0x1234_5678_9abc));
    let lift = |values: &[Fp]| values.iter().map(|&v| Fp2::from(v)).collect::<Vec<_>>();
    let mut secrets: Vec<String> = [&x, &cos, &sin, &r]
        .into_iter()
        .flat_map(|values| forms(&lift(values)))
        .collect();
    secrets.extend(forms(&[weight]));
    let session = Session::random(&mut rng);

    let rotate = Elementwise::<Fp2>::rotate(&x, &cos, &sin, 3, 4).expect("a rotation");
    assert_eq!(
        format!("{:?}", rotate.inputs()),
        "[Operand { len: 16 }, Partner { len: 16, bit: 3 }, Public { len: 16 }, \
         Public { len: 16 }]"
    );
    let scaled = Elementwise::<Fp2>::new(
        ElementwiseShape::elementwise(4),
        vec![
            Input::Broadcast {
                operand: &r,
                low_vars: 2,
            },
            Input::Operand(&x),
        ],
        vec![Term::new(weight, vec![0, 1])],
    )
    .expect("a relation");
    assert_eq!(
        format!("{scaled:?}"),
        "Elementwise { shape: ElementwiseShape { vars: 4, summed_vars: 0 }, inputs: \
         [Broadcast { len: 4, low_vars: 2 }, Operand { len: 16 }], terms: \
         [Term { factors: [0, 1], .. }] }"
    );
    for relation in [&rotate, &scaled] {
        let point = rng.fp2s(4);
        secrets.extend(forms(&point));
        let tables = relation.tables(&point).expect("a point");
        assert_hidden(&format!("{tables:#?}"), &secrets, "tables");
        assert_hidden(&format!("{:?}", tables.instance()), &secrets, "an instance");
        let proven = prove_batch(
            &mut session.transcript(),
            Prefix::Stage(1),
            3,
            &[tables.instance()],
        )
        .expect("a stage");
        let claims = tables
            .claims(
                proven.instance_point(0).expect("an instance"),
                proven.instance_evals(0).expect("an instance"),
            )
            .expect("the instance's results");
        let values: Vec<Fp2> = claims.iter().map(|claim| claim.value).collect();
        secrets.extend(forms(&values));
        assert_hidden(&format!("{claims:#?}"), &secrets, "claims");
    }

    // A claim on a linear combination shows its polynomials and numbers of coordinates.
    let point = rng.fp2s(3);
    let value = rng.fp2();
    secrets.extend(forms(&point));
    secrets.extend(forms(&[value]));
    let linear = Linear::new(
        3,
        vec![
            LinearInput {
                coefficient: weight,
                poly: "res",
                placement: Placement::Same,
            },
            LinearInput {
                coefficient: weight,
                poly: "max",
                placement: Placement::Broadcast {
                    low_vars: 1,
                    vars: 2,
                },
            },
        ],
    )
    .expect("a relation");
    let mut claim = LinearClaim::single("next", Claim { point, value });
    claim.substitute(&"next", &linear).expect("a point");
    assert_eq!(
        format!("{claim:?}"),
        "LinearClaim { terms: [LinearTerm { poly: \"res\", vars: 3, .. }, \
         LinearTerm { poly: \"max\", vars: 2, .. }], .. }"
    );
    assert_hidden(&format!("{claim:#?}"), &secrets, "a linear claim");
    assert_hidden(&format!("{linear:#?}"), &secrets, "a linear relation");
}
