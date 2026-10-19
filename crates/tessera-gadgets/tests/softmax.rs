// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemmas 5–6, RFC-0030's masked positions and RFC-0035's padded rows.
//! The oracle checks actual table tuples, product roots and padding zeros over Goldilocks.

mod common;

use common::{Fp, in_table};
use tessera_codec::{Rational, SecretBuf};
use tessera_field::{Field, PrimeField};
use tessera_gadgets::{
    GadgetError, Slack, SoftmaxColumn as Col, SoftmaxRow, SoftmaxTerm as Term,
    SoftmaxValue as Value, SoftmaxWitness,
};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

fn spec(id: u32, kind: TableKind) -> TableSpec {
    GeneratedTable::new(TableId(id), kind)
        .expect("table")
        .spec()
        .clone()
}

fn specs(sigma: u8, l: u8, key_bits: u8) -> Vec<TableSpec> {
    // Scale 1: the last of 64 keys is zero even at σ=31, and scores [-8,8] all fit.
    let mut specs = vec![spec(
        0,
        TableKind::Exp {
            key_bits,
            sigma,
            scale: Rational::new(1, 1).expect("scale"),
        },
    )];
    for kind in SoftmaxRow::range_kinds(sigma, l).expect("ranges") {
        specs.push(spec(specs.len() as u32, kind));
    }
    specs
}

fn gadget(sigma: u8, l: u8, max_context: usize) -> SoftmaxRow {
    SoftmaxRow::new(TableId(0), l, max_context, &specs(sigma, l, 6)).expect("gadget")
}

fn tensors(g: &SoftmaxRow, w: &SoftmaxWitness) -> Vec<(Col, SecretBuf<i64>)> {
    g.constraints::<Fp>()
        .committed
        .iter()
        .map(|&c| (c, w.tensor(c).expect("column").iter().copied().collect()))
        .collect()
}

fn changed(g: &SoftmaxRow, w: &SoftmaxWitness, c: Col, i: usize, value: i64) -> SoftmaxWitness {
    let mut ts = tensors(g, w);
    ts.iter_mut()
        .find(|(held, _)| *held == c)
        .expect("column")
        .1[i] = value;
    SoftmaxWitness::from_tensors(ts)
}

#[derive(Debug, PartialEq, Eq)]
enum Rejected {
    Lookup(TableId),
    Root,
    Padding(usize),
}

fn rejected(g: &SoftmaxRow, scores: &[i64], valid: usize, w: &SoftmaxWitness) -> Vec<Rejected> {
    let e = g.evaluate::<Fp>(scores, valid, w).expect("public shape");
    let tables = g.tables();
    let mut rejected = Vec::new();
    for group in e.lookups.groups() {
        let t = tables
            .iter()
            .find(|t| t.spec().id == group.table())
            .expect("table");
        for i in 0..group.keys().len() {
            if !in_table(t, group.keys()[i], group.values().map(|v| v[i])) {
                let r = Rejected::Lookup(group.table());
                if !rejected.contains(&r) {
                    rejected.push(r);
                }
            }
        }
    }
    if e.leaves.iter().copied().fold(Fp::ONE, |p, x| p * x) != e.root {
        rejected.push(Rejected::Root);
    }
    for (i, &x) in e.padding.iter().enumerate() {
        if x != Fp::ZERO {
            rejected.push(Rejected::Padding(i));
        }
    }
    rejected
}

/// Arbitrary candidate maximum, natural table values, arbitrary R; limbs from each modular
/// slack. This gives a cheating prover every low limb needed to exploit a missing top lookup.
fn candidate(
    g: &SoftmaxRow,
    scores: &[i64],
    valid: usize,
    c: i64,
    r: Option<i64>,
) -> SoftmaxWitness {
    let exp = g.tables()[0];
    let mut e: SecretBuf<i64> = scores
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i >= valid {
                0
            } else {
                exp.lookup(c - x, 0).unwrap_or(0)
            }
        })
        .collect();
    if valid == 0 {
        e.iter_mut().for_each(|x| *x = 0);
    }
    assignment(g, e, valid, c, r)
}

/// Complete the bracket for arbitrary exponentials, including a forged masked value.
fn assignment(
    g: &SoftmaxRow,
    e: SecretBuf<i64>,
    valid: usize,
    c: i64,
    r: Option<i64>,
) -> SoftmaxWitness {
    let s: i64 = e.iter().sum();
    let r = r.unwrap_or_else(|| {
        if valid == 0 {
            0
        } else {
            g.mass_bound() as i64 / s.max(1)
        }
    });
    let psi = i64::from(valid != 0);
    let a = Fp::from_i64(psi) * Fp::from_u64_reduced(g.mass_bound())
        - Fp::from_i64(r) * Fp::from_i64(s);
    let b = Fp::from_i64(s - psi) - a;
    let mut ts = vec![
        (Col::Exponentials, e),
        (Col::Reciprocal, [r].into_iter().collect()),
        (Col::Maximum, [c].into_iter().collect()),
    ];
    // Infer public L from the first committed limb's Range specification, using the descriptor.
    let constraints = g.constraints::<Fp>();
    for (slack, v) in [(Slack::A, a), (Slack::B, b)] {
        let lookups: Vec<_> = constraints
            .lookups
            .iter()
            .filter(|l| {
                l.key
                    .terms
                    .iter()
                    .any(|&(_, t)| matches!(t, Term::Column(Col::Limb(s, _)) if s == slack))
            })
            .collect();
        // The first limb lookup is the committed limb alone.
        let l = lookups.first().map_or(0, |lookup| {
            match g
                .tables()
                .iter()
                .find(|t| t.spec().id == lookup.table)
                .expect("table")
                .spec()
                .kind
            {
                TableKind::Range { bits } => bits,
                _ => panic!("Range"),
            }
        });
        for &col in &constraints.committed {
            if let Col::Limb(s, j) = col
                && s == slack
            {
                let digit =
                    (v.to_canonical_u64() >> (u32::from(j) * u32::from(l))) & ((1u64 << l) - 1);
                ts.push((col, [digit as i64].into_iter().collect()));
            }
        }
    }
    SoftmaxWitness::from_tensors(ts)
}

#[test]
fn all_4913_three_score_rows_have_exactly_the_causal_maximum_and_reciprocal() {
    let g = gadget(4, 4, 4);
    for x in -8..=8 {
        for y in -8..=8 {
            for z in -8..=8 {
                let scores = [x, y, z, 2_097_152]; // masked future score exceeds every causal maximum
                for valid in 1..=3 {
                    let w = g.witness(&scores, valid).expect("honest witness");
                    let c = *scores[..valid].iter().max().unwrap();
                    assert_eq!(w.tensor(Col::Maximum), Some(&[c][..]));
                    assert!(rejected(&g, &scores, valid, &w).is_empty());
                    let e = w.tensor(Col::Exponentials).unwrap();
                    assert!(e[valid..].iter().all(|&x| x == 0));
                    let s: i64 = e.iter().sum();
                    let r = g.mass_bound() as i64 / s;
                    assert_eq!(w.tensor(Col::Reciprocal), Some(&[r][..]));
                    // Try every maximum allowed by these scores, plus both immediate boundaries.
                    for maximum in -9..=9 {
                        let wrong = candidate(&g, &scores, valid, maximum, None);
                        assert_eq!(
                            rejected(&g, &scores, valid, &wrong).is_empty(),
                            maximum == c
                        );
                    }
                    for proposed in 0..32 {
                        let alternative = candidate(&g, &scores, valid, c, Some(proposed));
                        assert_eq!(
                            rejected(&g, &scores, valid, &alternative).is_empty(),
                            proposed == r
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn missing_max_or_mask_checks_each_admit_a_concrete_alternative() {
    let g = gadget(16, 16, 4);
    let scores = [3, 4, 5, 0];
    let w = g.witness(&scores, 2).unwrap();
    // A second-largest score is rejected by the Exp lookup; its product already has a zero.
    assert_eq!(
        rejected(&g, &scores, 2, &candidate(&g, &scores, 2, 3, None)),
        [Rejected::Lookup(TableId(0))]
    );
    // Max+1, or the maximum of a future position: all lookups pass; the root alone refuses.
    assert_eq!(
        rejected(&g, &scores, 2, &candidate(&g, &scores, 2, 5, None)),
        [Rejected::Root]
    );
    // The future score's own key/value is not permitted at the masked lookup, even if nonzero.
    let mut exponentials: SecretBuf<i64> = w
        .tensor(Col::Exponentials)
        .unwrap()
        .iter()
        .copied()
        .collect();
    exponentials[2] = 1;
    let bad = assignment(&g, exponentials, 2, 4, None);
    assert_eq!(
        rejected(&g, &scores, 2, &bad),
        [Rejected::Lookup(TableId(0))]
    );
    let reciprocal = w.tensor(Col::Reciprocal).unwrap()[0];
    for r in [reciprocal - 1, reciprocal + 1] {
        assert_eq!(
            rejected(&g, &scores, 2, &candidate(&g, &scores, 2, 4, Some(r))),
            [Rejected::Lookup(TableId(3))]
        );
    }
    // Every masked leaf is exactly one, including a score equal to the forged maximum.
    let bad = candidate(&g, &scores, 2, 5, None);
    let e = g.evaluate::<Fp>(&scores, 2, &bad).unwrap();
    assert_eq!(&e.leaves[2..], &[Fp::ONE, Fp::ONE]);
}

#[test]
fn padded_rows_have_one_witness_and_root_one() {
    let g = gadget(16, 16, 5);
    let scores = [0; 8];
    let w = g.witness(&scores, 0).unwrap();
    assert!(rejected(&g, &scores, 0, &w).is_empty());
    for &col in &g.constraints::<Fp>().committed {
        assert!(w.tensor(col).unwrap().iter().all(|&x| x == 0));
    }
    assert_eq!(g.evaluate::<Fp>(&scores, 0, &w).unwrap().root, Fp::ONE);
    for (col, index) in [(Col::Maximum, 0), (Col::Reciprocal, 1)] {
        let bad = changed(&g, &w, col, 0, 1);
        assert_eq!(rejected(&g, &scores, 0, &bad), [Rejected::Padding(index)]);
    }
    for &col in &g.constraints::<Fp>().committed {
        if let Col::Limb(..) = col {
            let bad = changed(&g, &w, col, 0, 1);
            assert!(!rejected(&g, &scores, 0, &bad).is_empty());
        }
    }
}

#[test]
fn every_committed_column_and_reciprocal_bound_is_binding() {
    for sigma in [0, 4, 16, 23] {
        let g = gadget(sigma, 16, 4);
        let scores = [3, 1, 3, -2];
        let w = g.witness(&scores, 4).unwrap();
        assert!(rejected(&g, &scores, 4, &w).is_empty());
        for &col in &g.constraints::<Fp>().committed {
            for i in 0..w.tensor(col).unwrap().len() {
                for delta in [-1, 1] {
                    let value = w.tensor(col).unwrap()[i] + delta;
                    assert!(
                        !rejected(&g, &scores, 4, &changed(&g, &w, col, i, value)).is_empty(),
                        "{col:?} sigma={sigma}"
                    );
                }
            }
        }
        let bad = candidate(&g, &scores, 4, 3, Some(1i64 << (sigma + 1)));
        let table = g
            .tables()
            .iter()
            .find(|t| t.spec().kind == TableKind::Range { bits: sigma + 1 })
            .unwrap()
            .spec()
            .id;
        assert!(rejected(&g, &scores, 4, &bad).contains(&Rejected::Lookup(table)));
    }
}

#[test]
fn exact_goldilocks_precision_bound_refuses_the_known_wrap() {
    for (sigma, max, passes) in [
        (23, 1 << 17, true),
        (24, 32768, true),
        (24, 32769, false),
        (31, 2, true),
        (31, 3, false),
        (32, 1, false),
    ] {
        let specs = if sigma < 32 {
            specs(sigma, 16, 6)
        } else {
            vec![spec(
                0,
                TableKind::Exp {
                    key_bits: 6,
                    sigma,
                    scale: Rational::new(1, 1).unwrap(),
                },
            )]
        };
        assert_eq!(SoftmaxRow::new(TableId(0), 16, max, &specs).is_ok(), passes);
    }
    // #376's alternative witness satisfies the modular bracket at σ24 and n33280.
    let sigma = 24;
    let sum = 33_280u64 << sigma;
    let forged = 33_038_714u64;
    let a =
        Fp::from_u64_reduced(1 << 48) - Fp::from_u64_reduced(sum) * Fp::from_u64_reduced(forged);
    assert!(a.to_canonical_u64() < sum);
    assert_ne!(forged, (1u64 << 48) / sum);
    assert!(SoftmaxRow::new(TableId(0), 16, 33_280, &specs(sigma, 16, 6)).is_err());
}

#[test]
fn constructor_and_public_shapes_reject_exact_invalid_domains() {
    let g = gadget(16, 16, 5);
    for (len, valid) in [(0, 0), (3, 2), (16, 1), (4, 5), (8, 6)] {
        assert!(matches!(
            g.witness(&vec![0; len], valid),
            Err(GadgetError::Shape { .. })
        ));
    }
    for value in [-(1i64 << 21) - 1, (1i64 << 21) + 1, i64::MIN, i64::MAX] {
        assert!(matches!(
            g.witness(&[value], 1),
            Err(GadgetError::Input { .. })
        ));
    }
    for value in [-(1i64 << 21), 1i64 << 21] {
        assert!(g.witness(&[value], 1).is_ok());
    }
    assert!(g.witness(&[0; 8], 5).is_ok());
    for (max, l) in [(0, 16), ((1 << 17) + 1, 16), (1, 0), (1, 33)] {
        assert!(matches!(
            SoftmaxRow::new(TableId(0), l, max, &specs(16, 16, 6)),
            Err(GadgetError::Shape { .. })
        ));
    }
    let mut tables = specs(16, 16, 6);
    tables[0].digest[0] ^= 1;
    assert!(matches!(
        SoftmaxRow::new(TableId(0), 16, 5, &tables),
        Err(GadgetError::Spec {
            table: TableId(0),
            ..
        })
    ));
    assert!(matches!(
        SoftmaxRow::new(TableId(0), 16, 5, &specs(16, 16, 1)),
        Err(GadgetError::Table {
            table: TableId(0),
            ..
        })
    ));
    let tables = specs(16, 16, 6);
    assert!(matches!(
        SoftmaxRow::new(TableId(0), 16, 5, &tables[..1]),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        SoftmaxRow::new(TableId(999), 16, 5, &tables),
        Err(GadgetError::Table {
            table: TableId(999),
            ..
        })
    ));
    let w = g.witness(&[0; 8], 5).unwrap();
    for malformed in [
        Vec::new(),
        vec![(Col::Maximum, [0].into_iter().collect())],
        {
            let mut ts = tensors(&g, &w);
            ts[0].1.push(0);
            ts
        },
        {
            let mut ts = tensors(&g, &w);
            ts.push((Col::Maximum, [0].into_iter().collect()));
            ts
        },
    ] {
        assert!(matches!(
            g.evaluate::<Fp>(&[0; 8], 5, &SoftmaxWitness::from_tensors(malformed)),
            Err(GadgetError::Shape { .. })
        ));
    }
    let debug = format!("{w:?} {:?}", g.evaluate::<Fp>(&[0; 8], 5, &w).unwrap());
    assert!(!debug.contains("65536"));
}

#[test]
fn relation_descriptors_are_pointwise_products_and_match_the_evaluated_columns() {
    let g = gadget(16, 16, 4);
    let scores = [3, 4, 5, 0];
    for valid in [0, 1, 2, 4] {
        let w = g.witness(&scores, valid).unwrap();
        let e = g.evaluate::<Fp>(&scores, valid, &w).unwrap();
        let s = w
            .tensor(Col::Exponentials)
            .unwrap()
            .iter()
            .map(|&x| Fp::from_i64(x))
            .fold(Fp::ZERO, |a, b| a + b);
        let r = Fp::from_i64(w.tensor(Col::Reciprocal).unwrap()[0]);
        let c = Fp::from_i64(w.tensor(Col::Maximum).unwrap()[0]);
        let psi = Fp::from_u64_reduced(u64::from(valid > 0));
        let a = psi * Fp::from_u64_reduced(g.mass_bound()) - r * s;
        let b = s - psi - a;
        let terms = |t, i| match t {
            Term::Score => Fp::from_i64(scores[i]),
            Term::Causal => Fp::from_u64_reduced(u64::from(i < valid)),
            Term::RowSelector => psi,
            Term::Column(Col::Maximum) => c,
            Term::Column(Col::Reciprocal) => r,
            Term::Column(Col::Exponentials) => {
                Fp::from_i64(w.tensor(Col::Exponentials).unwrap()[i])
            }
            Term::Virtual(Value::Sum) => s,
            Term::Virtual(Value::Slack(Slack::A)) => a,
            t => panic!("unexpected {t:?}"),
        };
        for relation in g.constraints::<Fp>().relations {
            for i in 0..4 {
                let got = relation.evaluate::<Fp>(|t| terms(t, i));
                let expected = match relation.target {
                    Value::Key => e.lookups.groups()[0].keys()[i],
                    Value::Leaf => e.leaves[i],
                    Value::Sum => Fp::from_i64(w.tensor(Col::Exponentials).unwrap()[i]),
                    Value::Slack(Slack::A) => a,
                    Value::Slack(Slack::B) => b,
                    Value::PaddingMaximum => e.padding[0],
                    Value::PaddingReciprocal => e.padding[1],
                };
                assert_eq!(got, expected);
            }
        }
    }
    // On one bit, C=[1,0], k=[0,2]. MLE(C*k)(1/2)=0, while C~(1/2)*k~(1/2)=1/2.
    let half = Fp::from_u64_reduced(Fp::MODULUS / 2 + 1);
    let pointwise_product_mle = Fp::ZERO;
    assert_ne!(pointwise_product_mle, half * (Fp::ONE));
}

#[test]
fn lookup_and_product_proofs_verify_independently_and_leave_bound_leaf_claims() {
    use common::{Fp2, Messages, Session, Shape, SplitMix64, TvFp2, mle_fp, tv_fp, tv_fp2s};
    use tessera_sumcheck::{Forest, prove_forests};
    use tv_core::{sumcheck, tree};
    let mut rng = SplitMix64::new(0x122_0001);
    let g = gadget(4, 4, 4);
    for valid in [0, 1, 2, 3, 4] {
        let scores = [3, 4, 5, 0];
        let w = g.witness(&scores, valid).unwrap();
        let e = g.evaluate::<Fp>(&scores, valid, &w).unwrap();
        let tables = g.tables();
        let columns = common::table_columns(&tables);
        let multiplicities = e.lookups.multiplicities(&columns).unwrap();
        let instances = e.lookups.instances(&columns, &multiplicities).unwrap();
        let shapes: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(k, i)| Shape::of(i, common::spec_of(&tables, &e.lookups, k)))
            .collect();
        let session = Session::random(&mut rng);
        let proven = common::prove(&session, &instances);
        let checked = common::verify(&session, &shapes, &Messages::of(&proven)).unwrap();
        for (k, (claim, group)) in checked.instances.iter().zip(e.lookups.groups()).enumerate() {
            let n = group.keys().len().next_power_of_two();
            let mut keys: SecretBuf<Fp> = group.keys().iter().copied().collect();
            keys.resize(n, Fp::ZERO); // all tables' row zero has key zero
            assert_eq!(claim.key.value, mle_fp(&keys, &claim.key.point));
            if let Some(value) = &claim.value {
                let row0 = tables
                    .iter()
                    .find(|t| t.spec().id == group.table())
                    .unwrap()
                    .row(0)
                    .unwrap()
                    .1;
                let mut values: SecretBuf<Fp> = group.values().unwrap().iter().copied().collect();
                values.resize(n, row0);
                assert_eq!(value.value, mle_fp(&values, &value.point));
            }
            assert_eq!(
                claim.multiplicity.value,
                mle_fp(&multiplicities[k], &claim.multiplicity.point)
            );
            assert_eq!(group.lengths().iter().sum::<usize>(), group.keys().len());
        }
        // Both real roots 0 and padded root 1 are supplied from public row metadata.
        let mut transcript = session.transcript();
        let proven =
            prove_forests::<Fp2, _>(&mut transcript, &[Forest::new(&e.leaves, 2)]).unwrap();
        let mut verifier = session.tv_transcript();
        let mut claims = vec![tree::TreeClaim {
            point: vec![],
            value: TvFp2::from(tv_fp(e.root)),
        }];
        for (level, p) in proven.levels().iter().enumerate() {
            let proof = sumcheck::SumcheckProof {
                rounds: p.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
                final_evals: tv_fp2s(&p.proof().final_evals),
            };
            claims = tree::verify_product_level(&mut verifier, level as u32, &claims, &proof)
                .unwrap()
                .claims;
        }
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].value, mle_fp(&e.leaves, &claims[0].point));
    }
}

#[test]
fn maximal_safe_precision_rows_refuse_the_modular_wrap_reciprocal() {
    for (sigma, max_context) in [(23, 1 << 17), (24, 1 << 15), (31, 2)] {
        let g = gadget(sigma, 16, max_context);
        let scores = vec![0; max_context];
        let w = g.witness(&scores, max_context).unwrap();
        assert!(rejected(&g, &scores, max_context, &w).is_empty());
        let s = (max_context as u128) << sigma;
        let honest = (u128::from(g.mass_bound()) / s) as i64;
        assert_eq!(w.tensor(Col::Reciprocal), Some(&[honest][..]));
        let wrapped = ((u128::from(Fp::MODULUS) + u128::from(g.mass_bound())) / s) as i64;
        assert_eq!(
            rejected(
                &g,
                &scores,
                max_context,
                &candidate(&g, &scores, max_context, 0, Some(wrapped))
            ),
            [Rejected::Lookup(g.tables()[1].spec().id)]
        );
        for r in [
            honest - 1,
            honest + 1,
            ((u128::from(Fp::MODULUS) + u128::from(g.mass_bound())) / s) as i64,
        ] {
            assert!(
                !rejected(
                    &g,
                    &scores,
                    max_context,
                    &candidate(&g, &scores, max_context, 0, Some(r))
                )
                .is_empty()
            );
        }
    }
}
