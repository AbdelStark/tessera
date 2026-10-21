// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0045 split checks against explicit column/MLE sums, without any prover implementation.
use tv_core::advice::Auxiliary;
use tv_core::leaf::split::{Affine, Bound, Column, Error, Layer, Layout, Lookup, Read, Site};
use tv_core::leaf::{self, ForestClaim};
use tv_core::logup::{Claim, InstanceClaims, LeafClaims};
use tv_core::lowering::NodeId;
use tv_core::tree::TreeClaim;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::TableId;

fn f(x: u64) -> Fp {
    Fp::reduce(x)
}
fn e(x: u64) -> Fp2 {
    Fp2::from(f(x))
}
fn form(constant: u64, terms: &[(u64, Column, u32)]) -> Affine {
    Affine {
        constant: f(constant),
        terms: terms
            .iter()
            .map(|&(coefficient, column, variables)| {
                (
                    f(coefficient),
                    Read {
                        column,
                        coordinates: (0..variables).collect(),
                    },
                )
            })
            .collect(),
    }
}
fn specs() -> Vec<Site> {
    let lookup = |variables, key, value| Lookup {
        table: TableId(11),
        variables,
        key,
        value: Some(value),
    };
    let repeated = || {
        lookup(
            3,
            form(0, &[(1, Column::Input(0), 3), (16, Column::Input(1), 1)]),
            form(0, &[(1, Column::Output(0), 3)]),
        )
    };
    let layer = |node| Layer {
        node: NodeId(node),
        lookups: vec![
            repeated(),
            Lookup {
                table: TableId(7),
                variables: 3,
                key: form(0, &[(1, Column::Auxiliary(Auxiliary::RequantLow), 3)]),
                value: None,
            },
            repeated(),
        ],
    };
    vec![
        Site {
            layers: vec![Layer {
                node: NodeId(0),
                lookups: vec![lookup(
                    2,
                    form(5, &[(1, Column::SoftmaxA, 2)]),
                    form(0, &[(1, Column::Output(0), 2)]),
                )],
            }],
        },
        Site {
            layers: vec![layer(1), layer(5)],
        },
        Site {
            layers: vec![Layer {
                node: NodeId(10),
                lookups: vec![Lookup {
                    table: TableId(7),
                    variables: 0,
                    key: form(0, &[(1, Column::Input(0), 0)]),
                    value: None,
                }],
            }],
        },
    ]
}
fn eq(point: &[Fp2], row: usize) -> Fp2 {
    point
        .iter()
        .enumerate()
        .fold(Fp2::ONE, |weight, (bit, &r)| {
            weight * if row >> bit & 1 == 0 { Fp2::ONE - r } else { r }
        })
}
fn mle(values: &[u64], point: &[Fp2]) -> Fp2 {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .enumerate()
        .fold(Fp2::ZERO, |sum, (row, &v)| sum + e(v) * eq(point, row))
}
fn transcript(mode: Mode) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [45; 32],
        session_id: [37; 16],
    })
}
fn row0(id: TableId) -> Option<(Fp, Option<Fp>)> {
    match id.0 {
        7 => Some((f(0), None)),
        11 => Some((f(3), Some(f(9)))),
        _ => None,
    }
}
fn sigma(point: &[Fp2]) -> Vec<Fp2> {
    point
        .first()
        .copied()
        .into_iter()
        .chain(point.get(1..).unwrap_or(&[]).iter().rev().copied())
        .collect()
}
struct Case {
    layout: Layout,
    leaf: leaf::Verified,
    splits: Vec<Fp2>,
    columns: Vec<(u32, Column, Vec<u64>, u32)>,
}
fn case() -> Case {
    let p: Vec<_> = (0..6).map(|i| Fp2::new(f(i + 2), f(i + 11))).collect();
    let mut columns = vec![
        (0, Column::SoftmaxA, vec![1, 2, 3, 4], 2),
        (0, Column::Output(0), vec![7, 8, 9, 10], 2),
    ];
    let mut keys = vec![];
    let mut values = vec![];
    let mut ranges = vec![];
    for layer in [1u32, 5] {
        let x: Vec<u64> = (0..8).map(|i| i + u64::from(layer)).collect();
        let c = vec![1, 2];
        let y: Vec<u64> = (0..8).map(|i| 9 + i + u64::from(layer)).collect();
        let low: Vec<u64> = (0..8).map(|i| i % 4).collect();
        let key: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v + 16 * c[i % 2])
            .collect();
        // Four equal-sized blocks retain site/layer/request order. No production layout helper
        // computes this oracle column, its offsets, or its row-0 padding.
        keys.extend(&key);
        values.extend(&y);
        keys.extend(&key);
        values.extend(&y);
        ranges.extend(&low);
        columns.extend([
            (layer, Column::Input(0), x, 3),
            (layer, Column::Input(1), c, 1),
            (layer, Column::Output(0), y, 3),
            (layer, Column::Auxiliary(Auxiliary::RequantLow), low, 3),
        ]);
    }
    keys.extend([6, 7, 8, 9]);
    values.extend([7, 8, 9, 10]);
    keys.resize(64, 3);
    values.resize(64, 9);
    ranges.push(7);
    ranges.resize(32, 0);
    columns.push((10, Column::Input(0), vec![7], 0));
    let instance =
        |key: &[u64], value: Option<&[u64]>, h: usize, multiplicity: u64| InstanceClaims {
            key: Claim {
                point: sigma(&p[..h]),
                value: mle(key, &p[..h]),
            },
            value: value.map(|v| Claim {
                point: sigma(&p[..h]),
                value: mle(v, &p[..h]),
            }),
            multiplicity: Claim {
                point: vec![e(19)],
                value: e(multiplicity),
            },
        };
    let leaf = leaf::Verified {
        lookups: LeafClaims {
            instances: vec![
                instance(&ranges, None, 5, 17),
                instance(&keys, Some(&values), 6, 36),
            ],
        },
        forests: vec![ForestClaim {
            tree: 4,
            claim: TreeClaim {
                point: vec![e(23)],
                value: e(29),
            },
        }],
        point: p.clone(),
    };
    let mut splits: Vec<_> = columns
        .iter()
        .map(|(_, _, data, k)| mle(data, &p[..*k as usize]))
        .collect();
    splits.extend([e(101), e(103)]);
    Case {
        layout: Layout::new(&specs()).unwrap(),
        leaf,
        splits,
        columns,
    }
}
fn bind(c: &Case, t: &mut Transcript) -> Result<Bound, Error> {
    c.layout.bind(t, c.leaf.clone(), &c.splits, 2, row0)
}
fn encoded(values: &[Fp2]) -> Vec<u8> {
    let mut out = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        out.extend(value.to_le_bytes())
    }
    out
}
#[test]
fn aligned_blocks_broadcasts_repeated_requests_and_row0_padding_bind_in_both_modes() {
    let c = case();
    assert_eq!(c.layout.values(), 11);
    assert_eq!(
        c.layout.instances().collect::<Vec<_>>(),
        [(TableId(7), 17), (TableId(11), 36)]
    );
    for mode in [Mode::Live, Mode::NonInteractive] {
        let mut t = transcript(mode);
        let bound = bind(&c, &mut t).unwrap();
        assert_eq!(bound.atoms, [e(101), e(103)]);
        assert_eq!(bound.forests, c.leaf.forests);
        assert_eq!(bound.point, c.leaf.point);
        assert_eq!(
            bound.multiplicities,
            c.leaf
                .lookups
                .instances
                .iter()
                .zip([7, 11])
                .map(|(i, t)| (TableId(t), i.multiplicity.clone()))
                .collect::<Vec<_>>()
        );
        assert_eq!(bound.columns.len(), c.columns.len());
        for (claim, (node, column, data, k)) in bound.columns.iter().zip(&c.columns) {
            assert_eq!(claim.node, NodeId(*node));
            assert_eq!(claim.column, *column);
            assert_eq!(claim.point, c.leaf.point[..*k as usize]);
            assert_eq!(claim.value, mle(data, &claim.point));
        }
        let mut expected = transcript(mode);
        expected.absorb(Ascii::new("stage/0/split").unwrap(), &encoded(&c.splits));
        assert_eq!(t.state(), expected.state());
        assert_eq!(t.counter(), 0);
    }
}
#[test]
fn every_lookup_value_is_checked_and_the_atom_tail_is_only_transcript_bound() {
    let mut c = case();
    for index in 0..c.layout.values() {
        c.splits[index] += Fp2::ONE;
        assert_eq!(
            bind(&c, &mut transcript(Mode::Live)),
            Err(Error::Mismatch),
            "source {index}"
        );
        c.splits[index] -= Fp2::ONE;
    }
    let mut t = transcript(Mode::Live);
    let good = bind(&c, &mut t).unwrap();
    let state = *t.state();
    *c.splits.last_mut().unwrap() += Fp2::ONE;
    let mut t = transcript(Mode::Live);
    let bad = bind(&c, &mut t).unwrap();
    assert_eq!(bad.columns, good.columns);
    assert_ne!(*t.state(), state);
    assert_ne!(bad.atoms, good.atoms);
}
#[test]
fn wrong_counts_precede_absorption_and_bad_own_points_are_internal() {
    let c = case();
    for count in [0, c.splits.len() - 1, c.splits.len() + 1] {
        let mut values = c.splits.clone();
        values.resize(count, Fp2::ZERO);
        let mut t = transcript(Mode::Live);
        let state = *t.state();
        assert_eq!(
            c.layout
                .bind(&mut t, c.leaf.clone(), &values, 2, |_| panic!(
                    "no callback"
                )),
            Err(Error::Count {
                expected: c.splits.len(),
                found: count
            })
        );
        assert_eq!(*t.state(), state);
    }
    let mut leaf = c.leaf.clone();
    leaf.lookups.instances[0].key.point.swap(1, 2);
    assert_eq!(
        c.layout
            .bind(&mut transcript(Mode::Live), leaf, &c.splits, 2, row0),
        Err(Error::Internal)
    );
    for kind in 0..4 {
        let mut leaf = c.leaf.clone();
        match kind {
            0 => {
                leaf.lookups.instances.pop();
            }
            1 => {
                leaf.lookups.instances[1]
                    .value
                    .as_mut()
                    .unwrap()
                    .point
                    .swap(1, 2);
            }
            2 => {
                leaf.lookups.instances[1].value = None;
            }
            3 => {
                leaf.point.truncate(3);
            }
            _ => unreachable!(),
        }
        assert_eq!(
            c.layout
                .bind(&mut transcript(Mode::Live), leaf, &c.splits, 2, row0),
            Err(Error::Internal)
        );
    }
    assert_eq!(
        c.layout.bind(
            &mut transcript(Mode::Live),
            c.leaf.clone(),
            &c.splits,
            2,
            |_| None
        ),
        Err(Error::Internal)
    );
    assert_eq!(
        c.layout.bind(
            &mut transcript(Mode::Live),
            c.leaf.clone(),
            &c.splits,
            2,
            |_| Some((f(0), None))
        ),
        Err(Error::Internal)
    );
}
#[test]
fn inconsistent_public_site_descriptors_are_refused() {
    let base = specs();
    let mut cases = vec![vec![Site { layers: vec![] }]];
    let mut bad = base.clone();
    bad.swap(0, 1);
    cases.push(bad);
    let mut bad = base.clone();
    bad[1].layers.swap(0, 1);
    cases.push(bad);
    let mut bad = base.clone();
    bad[1].layers[1].lookups[0].variables = 4;
    cases.push(bad);
    let mut bad = base.clone();
    bad[1].layers[1].lookups[0].key.terms[0]
        .1
        .coordinates
        .swap(0, 1);
    cases.push(bad);
    let mut bad = base.clone();
    bad[0].layers[0].lookups[0].key.terms[0].1.coordinates = vec![0, 0];
    cases.push(bad);
    let mut bad = base.clone();
    bad[0].layers[0].lookups[0].key.terms[0].1.coordinates = vec![3];
    cases.push(bad);
    let mut bad = base.clone();
    bad[0].layers[0].lookups[0].variables = 64;
    cases.push(bad);
    let mut bad = base.clone();
    bad[0].layers[0].lookups[0].key.terms.push((
        f(1),
        Read {
            column: Column::SoftmaxA,
            coordinates: vec![1, 0],
        },
    ));
    cases.push(bad);
    let mut bad = base.clone();
    bad[1].layers[0].lookups[0].value = None;
    cases.push(bad);
    let mut bad = base.clone();
    for request in [0, 2] {
        bad[1].layers[1].lookups[request].key.terms[0].1.column = Column::SoftmaxB;
    }
    cases.push(bad);
    for site in cases {
        assert!(matches!(Layout::new(&site), Err(Error::Internal)));
    }
}

#[test]
fn wrong_public_padding_tuple_changes_the_column_identity() {
    let c = case();
    assert_eq!(
        c.layout.bind(
            &mut transcript(Mode::Live),
            c.leaf.clone(),
            &c.splits,
            2,
            |id| if id.0 == 11 {
                Some((f(0), Some(f(0))))
            } else {
                row0(id)
            }
        ),
        Err(Error::Mismatch)
    );
}

#[test]
fn huge_public_domains_are_counted_without_leaf_allocations_and_overflows_are_refused() {
    let mut sites = specs();
    sites.truncate(1);
    sites[0].layers[0].lookups[0].variables = 63;
    assert_eq!(
        Layout::new(&sites).unwrap().instances().collect::<Vec<_>>(),
        [(TableId(11), 1 << 63)]
    );
    let lookup = sites[0].layers[0].lookups[0].clone();
    sites[0].layers[0].lookups.push(lookup);
    assert!(matches!(Layout::new(&sites), Err(Error::Internal)));
    sites[0].layers[0].lookups[1].variables = 62;
    assert!(matches!(Layout::new(&sites), Err(Error::Internal)));
}
