// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The messages as values: `SumcheckProof` encodes as RFC-0010 defines it, in TCE (RFC-0003),
//! and decodes back; and no debug form of the crate shows a table entry or a leaf, which are
//! secret-class, or a message, challenge or claim value, which are protocol-class (05 §3).

mod common;

use common::{Fp, Fp2, Session, SplitMix64, TestInstance, TestTable, prove_and_verify};
use tessera_codec::{DecodeLimits, decode, encode};
use tessera_field::{PrimeField, QuadraticExtension};
use tessera_sumcheck::{Forest, Instance, Prefix, SumcheckProof, Table, prove_forests};

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
        "LevelClaim { vars: 5, .. }"
    );
    assert_hidden(&format!("{proven:?}"), &secrets, "proven forests");
    assert_hidden(&format!("{proven:#?}"), &secrets, "proven forests");
    assert_hidden(
        &format!("{:?}", proven.levels()[1]),
        &secrets,
        "a proven level",
    );
}
