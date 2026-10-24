// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Literal original-source and matrix-coordinate fixtures, independent of the producer.
mod claim_support;

use tv_core::advice::{self, Auxiliary, Source};
use tv_core::claims::{ClaimList, QuantPoint, Site, WeightType};
use tv_core::gadgets::{
    model::{External, Reduced},
    stacked,
};
use tv_core::lowering::{self, Graph, NodeId, Op, Role, TensorId};
use tv_core::model_openings::{Claims, Error};
use tv_core::schedule::Commitment;
use tv_core::stages::{LeafValues, PoolClaim, PoolTensor, PoolTerm, Verified};
use tv_core::{selection, validation, weights};
use tv_field::{Fp, Fp2};

fn f(x: u64) -> Fp2 {
    Fp2::from(Fp::reduce(x))
}
fn point(n: usize) -> Vec<Fp2> {
    (0..n)
        .map(|i| Fp2::new(Fp::reduce(5 + 3 * i as u64), Fp::reduce(7 + 2 * i as u64)))
        .collect()
}
fn bits(n: u32) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}
fn public(layers: u32, n: u32) -> (ClaimList, Graph, selection::Selection) {
    let mut c = claim_support::with_layers(&claim_support::claims_of("tiny-2-servable.d"), layers);
    c.arch.d_model = 4;
    c.arch.n_heads = 2;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = 8;
    c.arch.max_context = 8;
    c.quant.residual_bits = 8;
    c.quant.residual_exponent = c
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    c.quant.rsqrt_kappa = 0;
    c.quant.exp_sigma = 3;
    for (_, s) in &mut c.quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut c.quant.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    for l in 0..layers {
        claim_support::make_pot(
            &mut c,
            QuantPoint::Layer {
                layer: l,
                site: Site::RopeQ,
            },
            &[12, 13],
        );
        claim_support::make_affine(
            &mut c,
            QuantPoint::Layer {
                layer: l,
                site: Site::Up,
            },
            12,
        );
    }
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = 8;
    c.gadget_policy.query_block = 2;
    c.layout.advice.log2_k = 6;
    c.layout.advice.segments = 3;
    c.layout.weights.log2_k = 4;
    c.layout.weights.mask_coeffs = 0;
    c.layout.weights.blinding_rows = 0;
    let g = lowering::lower(&validation::check_public(&c).unwrap(), n).unwrap();
    let s = selection::derive(&g, c.gadget_policy).unwrap();
    (c, g, s)
}
fn input(g: &Graph, role: Role) -> TensorId {
    g.inputs().iter().find(|i| i.role == role).unwrap().tensor
}
fn parameter(g: &Graph, ty: WeightType, multiplier: bool) -> TensorId {
    g.inputs()
        .iter()
        .find_map(|i| match i.role {
            Role::Classes(w) if w.ty == ty && w.layer == 0 && !multiplier => Some(i.tensor),
            Role::Multipliers(w) if w.ty == ty && w.layer == 0 && multiplier => Some(i.tensor),
            _ => None,
        })
        .unwrap()
}
fn layer_weights(rho: &[Fp2], layers: u32) -> Vec<Fp2> {
    match layers {
        1 => vec![Fp2::ONE],
        3 => vec![
            (Fp2::ONE - rho[0]) * (Fp2::ONE - rho[1]),
            rho[0] * (Fp2::ONE - rho[1]),
            (Fp2::ONE - rho[0]) * rho[1],
        ],
        _ => unreachable!(),
    }
}
fn term(tensor: PoolTensor, weights: Vec<Fp2>, layer_point: Option<Vec<Fp2>>) -> PoolTerm {
    PoolTerm {
        tensor,
        weights,
        layer_point,
    }
}
#[derive(Clone)]
struct Fixture {
    pool: Vec<PoolClaim>,
    sources: Vec<stacked::Claim>,
    multiplicities: Vec<(tv_tables::TableId, tv_core::logup::Claim)>,
    tokens: LeafValues,
    logits: LeafValues,
}
impl Fixture {
    fn reduced(&self) -> Reduced<'_> {
        Reduced {
            graph: Verified {
                pool: self.pool.clone(),
            },
            sources: self.sources.clone(),
            multiplicities: &self.multiplicities,
            external: Some(External {
                tokens: &self.tokens,
                logits: &self.logits,
            }),
        }
    }
}
fn fixture(c: &ClaimList, g: &Graph, s: &selection::Selection) -> Fixture {
    let l = c.arch.n_layers;
    let rho = point(bits(l));
    let weights = layer_weights(&rho, l);
    let cp = g.checkpoints();
    let mut pool = vec![PoolClaim {
        stage: 2,
        commitment: Commitment::Advice,
        point: point(bits(g.length()) + 2),
        terms: vec![
            term(PoolTensor::Single(cp[0]), vec![f(29)], None),
            term(
                PoolTensor::Stacked(cp[1]),
                (0..l).map(|i| f(17 + 2 * u64::from(i))).collect(),
                None,
            ),
            term(PoolTensor::Single(cp[0]), vec![f(37)], None),
        ],
        value: f(31),
    }];
    let q = input(
        g,
        Role::Weight(lowering::WeightRef {
            ty: WeightType::Q,
            layer: 0,
        }),
    );
    pool.push(PoolClaim {
        stage: 1,
        commitment: Commitment::Weight(WeightType::Q),
        point: point(4),
        terms: vec![term(
            PoolTensor::Stacked(q),
            weights.clone(),
            Some(rho.clone()),
        )],
        value: f(47),
    });
    for (ty, multiplier) in [(WeightType::Q, false), (WeightType::Up, true)] {
        let t = parameter(g, ty, multiplier);
        let vars = g
            .tensor(t)
            .unwrap()
            .shape
            .dims()
            .iter()
            .map(|&n| bits(n))
            .sum();
        pool.push(PoolClaim {
            stage: 0,
            commitment: Commitment::ChannelParams,
            point: point(vars),
            terms: vec![term(
                PoolTensor::Stacked(t),
                weights.clone(),
                Some(rho.clone()),
            )],
            value: f(if multiplier { 59 } else { 53 }),
        });
    }
    let soft: Vec<_> = g
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            matches!(n.op(), Op::SoftmaxRow { .. }).then_some((NodeId(i as u32), n))
        })
        .collect();
    let max = stacked::Claim {
        sources: soft
            .iter()
            .map(|(id, _)| Source::Auxiliary(*id, Auxiliary::SoftmaxMaximum))
            .collect(),
        point: point(1 + bits(g.length())),
        layer_point: rho.clone(),
        value: f(61),
    };
    let mut ep = vec![Fp2::from(Fp::reduce(2).inverse().unwrap()); bits(g.length())];
    ep.extend(point(1 + bits(g.length())));
    let sources = vec![
        max.clone(),
        max,
        stacked::Claim {
            sources: soft
                .iter()
                .map(|(_, n)| Source::Tensor(n.outputs()[0]))
                .collect(),
            point: ep,
            layer_point: rho,
            value: f(67),
        },
    ];
    let mut used: Vec<_> = s
        .choices
        .iter()
        .flat_map(|c| c.tables.iter().copied())
        .collect();
    used.sort();
    used.dedup();
    let multiplicities = used
        .into_iter()
        .take(2)
        .map(|id| {
            let spec = s.tables.iter().find(|s| s.id == id).unwrap();
            (
                id,
                tv_core::logup::Claim {
                    point: point(spec.log2_rows as usize),
                    value: f(71 + u64::from(id.0)),
                },
            )
        })
        .collect();
    Fixture {
        pool,
        sources,
        multiplicities,
        tokens: LeafValues {
            point: point(bits(g.length())),
            values: vec![f(73)],
        },
        logits: LeafValues {
            point: point(bits(g.length()) + 3),
            values: vec![f(79)],
        },
    }
}

#[test]
fn original_combinations_duplicates_and_channel_coordinates_are_preserved() {
    for layers in [1, 3] {
        for n in [1, 3, 5] {
            let (c, g, s) = public(layers, n);
            let layout = advice::Layout::derive(
                &g,
                &s,
                advice::Config::new(
                    &c.arch,
                    c.gadget_policy,
                    c.layout.advice.log2_k,
                    c.layout.advice.segments,
                ),
                n,
            )
            .unwrap();
            let w = weights::derive(&c, 0).unwrap();
            let original = fixture(&c, &g, &s);
            let packed = Claims::derive(&g, &c, &layout, &w, original.reduced()).unwrap();
            assert_eq!(
                packed.matrices().collect::<Vec<_>>(),
                [weights::Matrix::Q, weights::Matrix::ChannelParams]
            );
            let a = packed.advice_claims().unwrap();
            assert_eq!(a.len(), 1 + 3 + 2);
            let cp = g.checkpoints();
            let expected: Vec<_> = core::iter::once((Source::Tensor(cp[0]), f(29)))
                .chain(
                    (1..=layers)
                        .map(|l| (Source::Tensor(cp[l as usize]), f(17 + 2 * u64::from(l - 1)))),
                )
                .chain([(Source::Tensor(cp[0]), f(37))])
                .collect();
            assert_eq!(
                a[0].pieces
                    .iter()
                    .map(|p| (p.source, p.coefficient))
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(a[0].value, f(31));
            let rho = point(bits(layers));
            let weights = layer_weights(&rho, layers);
            for (got, source) in a[1..4].iter().zip(&original.sources) {
                assert_eq!(got.point, source.point);
                assert_eq!(got.value, source.value);
                assert_eq!(
                    got.pieces
                        .iter()
                        .map(|p| (p.source, p.coefficient))
                        .collect::<Vec<_>>(),
                    source
                        .sources
                        .iter()
                        .copied()
                        .zip(weights.iter().copied())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(a[1].pieces, a[2].pieces);
            assert_eq!(a[1].value, a[2].value);
            for (got, (id, v)) in a[4..].iter().zip(&original.multiplicities) {
                assert_eq!(got.point, v.point);
                assert_eq!(got.value, v.value);
                assert_eq!(
                    got.pieces,
                    &[advice::opening::Piece {
                        source: Source::Multiplicity(*id),
                        coefficient: Fp2::ONE
                    }]
                );
            }
            let q = packed.weight_claims(weights::Matrix::Q, f(83)).unwrap();
            assert_eq!(q.len(), 1);
            assert_eq!(q[0].column_point, point(4));
            assert_eq!(q[0].row_point, rho);
            assert_eq!(q[0].value, f(47));
            assert_eq!(q[0].gamma, f(83));
            let mut aliases = original.clone();
            let mut repeated = aliases.pool[1].clone();
            repeated.value = f(101);
            aliases.pool.push(repeated);
            let aliased = Claims::derive(&g, &c, &layout, &w, aliases.reduced()).unwrap();
            let q = aliased.weight_claims(weights::Matrix::Q, f(83)).unwrap();
            assert_eq!(q.len(), 2);
            assert_eq!(q[0].column_point, q[1].column_point);
            assert_eq!(q[0].row_point, q[1].row_point);
            assert_eq!((q[0].value, q[1].value), (f(47), f(101)));
            let channel = packed
                .weight_claims(weights::Matrix::ChannelParams, f(89))
                .unwrap();
            assert_eq!(channel.len(), 2);
            for (i, (ty, channel_bits, value)) in [(1, 2, 53), (6, 3, 59)].into_iter().enumerate() {
                let mut full = point(channel_bits);
                full.resize(3, Fp2::ZERO);
                full.extend(&rho);
                full.extend((0..4).map(|j| f((ty >> j) & 1)));
                assert_eq!(channel[i].column_point, &full[..4]);
                assert_eq!(channel[i].row_point, &full[4..]);
                assert_eq!(channel[i].value, f(value));
                assert_eq!(channel[i].gamma, f(89));
            }
            for m in weights::Matrix::ALL {
                if !matches!(m, weights::Matrix::Q | weights::Matrix::ChannelParams) {
                    assert!(packed.weight_claims(m, Fp2::ONE).unwrap().is_empty());
                }
            }
            assert_eq!(packed.external().unwrap().tokens, &original.tokens);
            assert_eq!(packed.external().unwrap().logits, &original.logits);
        }
    }
}

#[test]
fn malformed_original_sources_and_weight_provenance_are_internal_errors() {
    let (c, g, s) = public(3, 3);
    let layout = advice::Layout::derive(
        &g,
        &s,
        advice::Config::new(
            &c.arch,
            c.gadget_policy,
            c.layout.advice.log2_k,
            c.layout.advice.segments,
        ),
        3,
    )
    .unwrap();
    let w = weights::derive(&c, 0).unwrap();
    let original = fixture(&c, &g, &s);
    for fault in 0..31 {
        let mut bad = original.clone();
        match fault {
            0 => bad.pool[0].terms.clear(),
            1 => {
                bad.pool[0].terms[1].weights.pop();
            }
            2 => bad.pool[0].terms[1].tensor = PoolTensor::Stacked(g.checkpoints()[2]),
            3 => bad.pool[0].point.push(Fp2::ONE),
            4 => {
                bad.sources[0].sources.pop();
            }
            5 => bad.sources[0].layer_point.clear(),
            6 => bad.sources[0].sources[0] = Source::Tensor(TensorId(u32::MAX)),
            7 => bad.multiplicities[0].0 = tv_tables::TableId(u32::MAX),
            8 => bad.pool[1].terms[0].layer_point = None,
            9 => bad.pool[1].terms[0].layer_point.as_mut().unwrap()[0] += Fp2::ONE,
            10 => bad.pool[1].terms[0].weights[0] += Fp2::ONE,
            11 => bad.pool[1].commitment = Commitment::Weight(WeightType::K),
            12 => {
                let mut distinct = bad.pool[1].clone();
                distinct.point[0] += Fp2::ONE;
                bad.pool.push(distinct);
            }
            13 => bad.pool[2].terms[0].tensor = bad.pool[1].terms[0].tensor,
            14 => bad.sources[0].sources.clear(),
            15 => bad.sources[0].layer_point.push(Fp2::ONE),
            16 => {
                let term = bad.pool[1].terms[0].clone();
                bad.pool[1].terms.push(term);
            }
            17 => bad.pool[1].point.push(Fp2::ONE),
            18 => bad.pool[2].point.push(Fp2::ONE),
            19 => bad.multiplicities[0].1.point.push(Fp2::ONE),
            20 => bad.pool[0].terms[0].tensor = bad.pool[1].terms[0].tensor,
            21 => bad.pool[1].terms[0].tensor = PoolTensor::Stacked(g.checkpoints()[1]),
            22 => bad.pool[1].terms[0].tensor = PoolTensor::Stacked(TensorId(u32::MAX)),
            23 => {
                bad.pool[1].terms[0].tensor = PoolTensor::Single(input(
                    &g,
                    Role::Weight(lowering::WeightRef {
                        ty: WeightType::Embed,
                        layer: 0,
                    }),
                ));
                bad.pool[1].terms[0].weights = vec![Fp2::ONE];
            }
            24 => {
                bad.pool[1].terms[0].tensor = PoolTensor::Single(input(
                    &g,
                    Role::Weight(lowering::WeightRef {
                        ty: WeightType::Embed,
                        layer: 0,
                    }),
                ));
                bad.pool[1].terms[0].layer_point = None;
            }
            25 => bad.pool[0].terms[0].weights.clear(),
            26 => bad.pool[0].terms[0].tensor = PoolTensor::Single(TensorId(u32::MAX)),
            27 => {
                let source = bad.sources[0].sources[0];
                bad.sources[0].sources.push(source);
            }
            28 => {
                bad.pool[1].terms[0].weights.pop();
            }
            29 => {
                bad.pool[1].terms[0]
                    .layer_point
                    .as_mut()
                    .unwrap()
                    .push(Fp2::ZERO);
                bad.pool[1].point.pop();
            }
            30 => bad.pool[1].commitment = Commitment::Weight(WeightType::O),
            _ => unreachable!(),
        }
        assert!(
            matches!(
                Claims::derive(&g, &c, &layout, &w, bad.reduced()),
                Err(Error::Internal)
            ),
            "fault {fault}"
        );
    }
    let mut swapped = w.clone();
    swapped.swap(1, 4);
    assert!(matches!(
        Claims::derive(&g, &c, &layout, &swapped, original.reduced()),
        Err(Error::Internal)
    ));
}

#[test]
fn unlayered_weight_claims_retain_the_full_original_point() {
    let (c, g, s) = public(3, 3);
    let layout = advice::Layout::derive(
        &g,
        &s,
        advice::Config::new(
            &c.arch,
            c.gadget_policy,
            c.layout.advice.log2_k,
            c.layout.advice.segments,
        ),
        3,
    )
    .unwrap();
    let w = weights::derive(&c, 0).unwrap();
    let mut original = fixture(&c, &g, &s);
    let embed = input(
        &g,
        Role::Weight(lowering::WeightRef {
            ty: WeightType::Embed,
            layer: 0,
        }),
    );
    original.pool[1] = PoolClaim {
        stage: 1,
        commitment: Commitment::Weight(WeightType::Embed),
        point: point(5),
        terms: vec![term(PoolTensor::Single(embed), vec![Fp2::ONE], None)],
        value: f(97),
    };
    let packed = Claims::derive(&g, &c, &layout, &w, original.reduced()).unwrap();
    let claims = packed
        .weight_claims(weights::Matrix::Embed, f(101))
        .unwrap();
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].column_point, &point(5)[..4]);
    assert_eq!(claims[0].row_point, &point(5)[4..]);
    assert_eq!(claims[0].value, f(97));
    assert_eq!(claims[0].gamma, f(101));
    for fault in 0..4 {
        let mut bad = original.clone();
        match fault {
            0 => bad.pool[1].terms[0].weights[0] += Fp2::ONE,
            1 => bad.pool[1].terms[0].layer_point = Some(Vec::new()),
            2 => {
                bad.pool[1].point.pop();
            }
            3 => bad.pool[1].commitment = Commitment::ChannelParams,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                Claims::derive(&g, &c, &layout, &w, bad.reduced()),
                Err(Error::Internal)
            ),
            "single fault {fault}"
        );
    }
}

#[test]
fn complete_runtime_budget_keeps_original_groups_and_checks_preset_name_mode_and_layout() {
    use tv_core::params::ParamSet;
    use tv_core::runtime::{Error as R, Geometry};
    use tv_hash::transcript::Mode;
    let (base, graph, selected) = public(3, 5);
    let tables = selected
        .tables
        .iter()
        .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
        .collect::<Vec<_>>();
    for params in [ParamSet::Live64, ParamSet::Ni100] {
        let mode = if params == ParamSet::Live64 {
            Mode::Live
        } else {
            Mode::NonInteractive
        };
        let mut c = base.clone();
        c.params = tv_core::claims::ParamSetId::new(params.name()).unwrap();
        c.layout = params.layout();
        let runtime =
            tv_core::gadgets::runtime::Plan::derive(&graph, c.gadget_policy, 2, 0, &tables)
                .unwrap();
        let layout = runtime
            .advice_layout(advice::Config::new(
                &c.arch,
                c.gadget_policy,
                c.layout.advice.log2_k,
                c.layout.advice.segments,
            ))
            .unwrap();
        let layouts = weights::derive(&c, 0).unwrap();
        let original = fixture(&c, &graph, &selected);
        let claims = Claims::derive(&graph, &c, &layout, &layouts, original.reduced()).unwrap();
        let views = claims.advice_claims().unwrap();
        let mut descriptor = tv_hash::DomainHasher::new(tv_hash::Domain::CommitmentId);
        descriptor.update(&0u16.to_le_bytes());
        descriptor.update(&[0; 32]);
        descriptor.update(&0u32.to_le_bytes());
        let a = c.layout.advice;
        descriptor.update(&[a.log2_k, a.log2_inv_rate]);
        for n in [a.mask_coeffs, a.row_block, a.segments] {
            descriptor.update(&n.to_le_bytes());
        }
        descriptor.update(&layout.committed_rows().unwrap().to_le_bytes());
        let binding = advice::opening::Commitment::bind(
            &layout,
            a,
            &[0; 32],
            0,
            [0; 32],
            descriptor.finalize(),
        )
        .unwrap();
        let xi = point((layout.nu() - u32::from(a.log2_k)) as usize);
        let opening = binding.prepare(&views, f(37), &xi).unwrap();
        let budget = Geometry::derive(&c, &runtime, None)
            .unwrap()
            .check_budget(params, mode, &opening, &claims, &layouts)
            .unwrap();
        let report = budget.report();
        assert_eq!(report.meets_targets, Some(true));
        assert_eq!(report.params.id, params.name());
        assert_eq!(
            report.statement.claim_groups.iter().sum::<u64>(),
            views.len() as u64
        );
        assert_eq!(
            report
                .statement
                .matrices
                .iter()
                .map(|m| m.matrix)
                .collect::<Vec<_>>(),
            [weights::Matrix::Q, weights::Matrix::ChannelParams]
        );
        let opposite = if mode == Mode::Live {
            Mode::NonInteractive
        } else {
            Mode::Live
        };
        assert_eq!(
            Geometry::derive(&c, &runtime, None)
                .unwrap()
                .check_budget(params, opposite, &opening, &claims, &layouts)
                .err(),
            Some(R::Params)
        );
        let mut renamed = c.clone();
        renamed.params = tv_core::claims::ParamSetId::new("unknown/1").unwrap();
        assert_eq!(
            Geometry::derive(&renamed, &runtime, None)
                .unwrap()
                .check_budget(params, mode, &opening, &claims, &layouts)
                .err(),
            Some(R::Params)
        );
        let mut other_layout = c.clone();
        other_layout.layout.advice.mask_coeffs += 1;
        assert_eq!(
            Geometry::derive(&other_layout, &runtime, None)
                .unwrap()
                .check_budget(params, mode, &opening, &claims, &layouts)
                .err(),
            Some(R::Params)
        );
        let mut misplaced = layouts.clone();
        misplaced.swap(1, 2);
        assert_eq!(
            Geometry::derive(&c, &runtime, None)
                .unwrap()
                .check_budget(params, mode, &opening, &claims, &misplaced)
                .err(),
            Some(R::Internal)
        );
    }
}
