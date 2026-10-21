// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native family checks against independent integer vectors and reference sumchecks.
mod claim_support;
mod support;

use claim_support::{claims_of, integer, items, list, member, number, string};
use support::{extension, from_reference, proof_of};
use tv_core::advice::{Auxiliary, Source};
use tv_core::claims::{ClaimList, QuantPoint, Site};
use tv_core::gadgets::{
    GadgetError, Plan,
    softmax::Padding,
    stacked::{Claim, Layer},
};
use tv_core::lowering::{self, Graph, NodeId};
use tv_core::sumcheck::{self, Prefix, SumcheckError};
use tv_core::validation;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::{Json, read_vectors, sumcheck as reference};

fn f(n: u64) -> Fp2 {
    Fp2::from(Fp::reduce(n))
}
fn field(v: &Json) -> Fp2 {
    let a = items(v);
    Fp2::new(
        Fp::from_canonical(number(&a[0])).unwrap(),
        Fp::from_canonical(number(&a[1])).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    items(v).iter().map(field).collect()
}
fn cols(v: &Json) -> Vec<Vec<Fp>> {
    items(v)
        .iter()
        .map(|r| {
            items(r)
                .iter()
                .map(|x| Fp::reduce(integer(x).rem_euclid(i128::from(Fp::MODULUS)) as u64))
                .collect()
        })
        .collect()
}
fn fp_ext(values: &[Fp2], point: &[Fp2]) -> Fp2 {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .enumerate()
        .fold(Fp2::ZERO, |s, (i, &v)| s + basis(point, i) * v)
}
fn basis(point: &[Fp2], index: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |p, (j, &r)| {
        p * if index >> j & 1 == 0 { Fp2::ONE - r } else { r }
    })
}
fn point(n: usize, seed: u64) -> Vec<Fp2> {
    (0..n)
        .map(|i| {
            Fp2::new(
                Fp::reduce(seed + 3 * i as u64),
                Fp::reduce(seed + 7 * i as u64 + 2),
            )
        })
        .collect()
}
fn bits(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}
#[allow(clippy::too_many_arguments)]
fn models(
    layers: u32,
    n: u32,
    heads: u32,
    dim: u32,
    kappa: u8,
    epsilon: u64,
    sigma: u8,
    limb: u8,
) -> (ClaimList, Graph) {
    let mut c = claims_of("tiny-4-servable.d");
    c.arch.n_layers = layers;
    c.arch.d_model = dim;
    c.arch.d_head = dim / heads;
    c.arch.n_heads = heads;
    c.arch.n_kv_heads = heads;
    c.arch.d_ff = 6;
    c.arch.vocab = 8;
    c.arch.max_context = 8;
    let q = &mut c.quant;
    q.act_exponents
        .retain(|(p, _)| !matches!(p,QuantPoint::Layer{layer,..} if *layer>=layers));
    q.shift_class_sets
        .retain(|(p, _)| !matches!(p,QuantPoint::Layer{layer,..} if *layer>=layers));
    q.affine_sites
        .retain(|(p, _)| !matches!(p,QuantPoint::Layer{layer,..} if *layer>=layers));
    q.score_scales.retain(|(l, _)| *l < layers);
    c.silu_in_exponents.retain(|(l, _)| *l < layers);
    q.residual_bits = 8;
    q.residual_exponent = q
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    q.rsqrt_kappa = kappa;
    q.rms_eps_int = epsilon;
    q.exp_sigma = sigma;
    for (p, shifts) in &mut q.shift_class_sets {
        *shifts = vec![
            if matches!(p,QuantPoint::Layer{site,..} if *site!=Site::Silu) {
                20
            } else {
                12
            },
        ];
    }
    for (l, s) in &mut q.score_scales {
        *s = tv_codec::Rational::new(1, 65536u64 << *l).unwrap();
    }
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = limb;
    let public = validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, n).unwrap();
    (c, graph)
}
fn check_claim(claim: &Claim, sources: &[Source], columns: &[Vec<Fp>]) {
    assert_eq!(claim.sources, sources);
    assert_eq!(sources.len(), columns.len());
    assert_eq!(
        claim.value,
        columns
            .iter()
            .enumerate()
            .fold(Fp2::ZERO, |out, (l, col)| out
                + basis(&claim.layer_point, l) * extension(col, &claim.point))
    );
}
fn context(mode: Mode) -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [7; 32],
        session_id: [3; 16],
    }
}
fn fe(v: Fp2) -> reference::Fe {
    let b = v.to_le_bytes();
    reference::Fe(
        u64::from_le_bytes(b[..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..].try_into().unwrap()),
    )
}
fn tables(v: &[Vec<Fp2>]) -> Vec<Vec<reference::Fe>> {
    v.iter()
        .map(|v| v.iter().copied().map(fe).collect())
        .collect()
}

#[test]
fn masked_key_and_forest_share_one_batch_with_padding_and_original_sources() {
    for (layers, n, heads) in [(1, 1, 1), (2, 3, 3), (3, 1, 3), (3, 3, 3), (4, 5, 2)] {
        let (c, g) = models(layers, n, heads, 2 * heads, 0, 2, 3, 8);
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let nodes: Vec<_> = plan.softmaxes().map(|s| s.unwrap().node()).collect();
        let stack = plan.softmax_stack(nodes[0]).unwrap();
        assert_eq!(stack.sites().len(), layers as usize);
        let (m, h, l) = (
            bits(n as usize),
            bits(heads as usize),
            bits(layers as usize),
        );
        let v = 2 * m + h;
        let rows = 1usize << (m + h);
        let local = point(v, 11);
        let rho = point(l, 19);
        let forest_point = point(v + l, 29);
        let mut maxima = vec![Fp2::ZERO; rows * (1 << l)];
        let mut scores = vec![Fp2::ZERO; rows * (1 << m) * (1 << l)];
        let mut causal = vec![Fp2::ZERO; scores.len()];
        let selected = tv_core::selection::derive(&g, c.gadget_policy).unwrap();
        let mut keys = Vec::new();
        let mut constants = Vec::new();
        for (layer, site) in stack.sites().iter().enumerate() {
            let choice = selected
                .choices
                .iter()
                .find(|s| s.node == site.node())
                .unwrap();
            let tv_tables::TableKind::Exp { key_bits, .. } =
                selected.tables[choice.tables[0].0 as usize].kind
            else {
                panic!()
            };
            let k0 = f((1u64 << key_bits) - 1);
            constants.push(k0);
            let mut k = vec![k0; 1 << v];
            for head in 0..(1 << h) {
                for query in 0..(1 << m) {
                    for key in 0..(1 << m) {
                        let row = head * (1 << m) + query;
                        let idx = (layer * rows + row) * (1 << m) + key;
                        if head < heads as usize && query < n as usize {
                            maxima[layer * rows + row] = f((layer + head + query + 5) as u64);
                            scores[idx] = maxima[layer * rows + row]
                                - f(((query + 2 * key + layer) % 5) as u64);
                        }
                        if head < heads as usize && key <= query && query < n as usize {
                            causal[idx] = Fp2::ONE;
                            k[row * (1 << m) + key] = maxima[layer * rows + row] - scores[idx];
                        }
                    }
                }
            }
            keys.push((fp_ext(&k, &local), f(9 + layer as u64)));
        }
        if layers == 3 {
            assert!(
                constants.windows(2).any(|w| w[0] != w[1]),
                "exercise different K0 per real layer"
            );
        }
        let bound = stack.lookup(&local, &rho, &keys).unwrap();
        assert_eq!(bound.lookups().len(), layers as usize);
        for ((old, site), (key, value)) in bound.lookups().iter().zip(stack.sites()).zip(&keys) {
            assert_eq!(old.lookup().unwrap().key, *key);
            assert_eq!(old.lookup().unwrap().value, *value);
            assert_eq!(
                old.input_claim().unwrap().input.source,
                site.source(1).unwrap()
            );
        }
        let mut target = local.clone();
        target.extend_from_slice(&rho);
        let expanded: Vec<_> = (0..scores.len()).map(|i| maxima[i >> (m)]).collect();
        let forest_col: Vec<_> = causal
            .iter()
            .enumerate()
            .map(|(i, &c)| Fp2::ONE - c + c * (expanded[i] - scores[i]))
            .collect();
        let forest_leaf = fp_ext(&forest_col, &forest_point);
        let forest = stack.forest(&forest_point, forest_leaf).unwrap();
        let mut reference_instances: Vec<_> = [&target, &forest_point]
            .into_iter()
            .map(|p| reference::Product {
                tables: tables(&[expanded.clone(), scores.clone()]),
                public: tables(&[
                    (0..scores.len()).map(|i| basis(p, i)).collect(),
                    causal.clone(),
                ]),
                lambda: None,
                form: reference::Form::Difference(reference::Fe::ONE),
            })
            .collect();
        let z = point(m + h + l, 53);
        let invalid: Vec<_> = (0..maxima.len())
            .map(|i| {
                let layer = i >> (m + h);
                let head = (i >> m) % (1 << h);
                let q = i % (1 << m);
                if layer < layers as usize && head < heads as usize && q < n as usize {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                }
            })
            .collect();
        for _ in 0..2 {
            reference_instances.push(reference::Product {
                tables: tables(&[maxima.clone()]),
                public: tables(&[
                    (0..maxima.len()).map(|i| basis(&z, i)).collect(),
                    invalid.clone(),
                ]),
                lambda: None,
                form: reference::Form::Product,
            });
        }
        let instances = [
            bound.instance().unwrap(),
            forest.instance().unwrap(),
            stack.padding_instance().unwrap(),
            stack.padding_instance().unwrap(),
        ];
        for (i, p) in instances.iter().zip(&reference_instances) {
            assert_eq!(i.claim, from_reference(p.claim()));
            assert_eq!(i.vars, p.vars());
        }
        for mode in [Mode::Live, Mode::NonInteractive] {
            let ctx = context(mode);
            let mut rt = reference::Transcript::new(&ctx.to_tce());
            let proof = proof_of(&reference::prove(
                &mut rt,
                "stage/1",
                3,
                &reference_instances,
                None,
            ));
            let mut vt = Transcript::new(&ctx);
            let batch = sumcheck::verify_batch(
                &mut vt,
                Prefix::Stage(1),
                3,
                &instances,
                &proof,
                |i, p, e| {
                    if i < 2 {
                        [&bound, &forest][i].evaluate(p, e).ok().map(|f| f.value)
                    } else {
                        stack
                            .padding(
                                if i == 2 {
                                    Padding::Maximum
                                } else {
                                    Padding::Reciprocal
                                },
                                &z,
                                p,
                                e,
                            )
                            .ok()
                            .map(|f| f.value)
                    }
                },
            )
            .unwrap();
            let mut offset = 0;
            for key in [&bound, &forest] {
                let own = key
                    .evaluate(&batch.point, &proof.final_evals[offset..offset + 2])
                    .unwrap();
                offset += 2;
                let cm: Vec<_> = stack
                    .sites()
                    .iter()
                    .map(|s| Source::Auxiliary(s.node(), Auxiliary::SoftmaxMaximum))
                    .collect();
                let sx: Vec<_> = stack
                    .sites()
                    .iter()
                    .map(|s| Source::Tensor(g.node(s.node()).unwrap().inputs()[0]))
                    .collect();
                let max_cols = maxima
                    .chunks(rows)
                    .take(layers as usize)
                    .map(|r| {
                        r.iter()
                            .map(|v| Fp::from_canonical(fe(*v).0).unwrap())
                            .collect()
                    })
                    .collect::<Vec<_>>();
                let score_cols = scores
                    .chunks(1 << v)
                    .take(layers as usize)
                    .map(|r| {
                        r.iter()
                            .map(|v| Fp::from_canonical(fe(*v).0).unwrap())
                            .collect()
                    })
                    .collect::<Vec<_>>();
                check_claim(&own.claims[0], &cm, &max_cols);
                check_claim(&own.claims[1], &sx, &score_cols);
                assert_eq!(own.claims[0].point.len(), m + h);
                assert_eq!(own.claims[1].point.len(), v);
            }
            for changed in 0..4 {
                let mut bad = proof.clone();
                bad.final_evals[changed] += Fp2::ONE;
                assert_eq!(
                    sumcheck::verify_batch(
                        &mut Transcript::new(&ctx),
                        Prefix::Stage(1),
                        3,
                        &instances,
                        &bad,
                        |i, p, e| if i < 2 {
                            [&bound, &forest][i].evaluate(p, e).ok().map(|f| f.value)
                        } else {
                            stack
                                .padding(
                                    if i == 2 {
                                        Padding::Maximum
                                    } else {
                                        Padding::Reciprocal
                                    },
                                    &z,
                                    p,
                                    e,
                                )
                                .ok()
                                .map(|f| f.value)
                        }
                    ),
                    Err(SumcheckError::FinalMismatch {
                        round: (v + l) as u32
                    })
                );
            }
        }
        assert!(stack.lookup(&local, &rho, &keys[..keys.len() - 1]).is_err());
        let mut wrong = forest_point.clone();
        wrong.push(Fp2::ONE);
        assert!(matches!(
            stack.forest(&wrong, forest_leaf),
            Err(GadgetError::Internal)
        ));
        if layers > 1 {
            assert!(matches!(
                plan.softmax_stack(nodes[1]),
                Err(GadgetError::Internal)
            ));
        }
    }
}

#[test]
fn frozen_root_products_preserve_complete_coefficients_and_zero_padded_sources() {
    let vectors = read_vectors("gadget-finals/roots.json");
    for case in list(&vectors, "cases").iter().filter(|c| {
        !matches!(
            string(member(c, "name")),
            "padded-root" | "padded-residual" | "padded-feature"
        )
    }) {
        let record = &list(case, "roots")[0];
        for layers in [1, 2, 3, 4] {
            let (c, g) = models(
                layers,
                number(member(case, "n")),
                1,
                number(member(case, "dim")),
                number(member(case, "kappa")),
                number(member(case, "epsilon")),
                3,
                8,
            );
            let plan = Plan::derive(&g, c.gadget_policy).unwrap();
            let node = plan.roots().next().unwrap().unwrap().node();
            let stack = plan.root_stack(node).unwrap();
            let rho = point(bits(layers as usize), 41);
            let local = fields(member(record, "point"));
            let mut full = point(local.len(), 43);
            full.extend(point(rho.len(), 47));
            let sourcecols = cols(member(record, "columns"));
            let keycols = cols(member(record, "key_columns"));
            let rotate = |column: &Vec<Fp>, l: usize| {
                (0..column.len())
                    .map(|i| column[if i < 3 { (i + l) % 3 } else { i }])
                    .collect::<Vec<_>>()
            };
            let sources: Vec<Vec<_>> = (0..layers as usize)
                .map(|l| {
                    sourcecols
                        .iter()
                        .map(|r| extension(&rotate(r, l), &local))
                        .collect()
                })
                .collect();
            let keys: Vec<Vec<_>> = (0..layers as usize)
                .map(|l| {
                    keycols
                        .iter()
                        .map(|r| extension(&rotate(r, l), &local))
                        .collect()
                })
                .collect();
            let zeros = vec![Fp2::ZERO; keys[0].len()];
            let layer_values: Vec<_> = (0..layers as usize)
                .map(|l| Layer {
                    sources: &sources[l],
                    keys: &keys[l],
                    values: &zeros,
                })
                .collect();
            let checked = stack.check(&local, &rho, &layer_values).unwrap();
            assert_eq!(checked.layers().len(), layers as usize);
            for (site, one) in stack.sites().iter().zip(checked.layers()) {
                assert_eq!(
                    one.lookups().map(|l| l.table).collect::<Vec<_>>(),
                    site.tables()
                );
            }
            let mut digits = cols(member(record, "root_digits"));
            let root_count = digits.len();
            digits.extend(cols(member(record, "u_digits")));
            let final_layer = &full[local.len()..];
            let final_local = &full[..local.len()];
            let evals: Vec<_> = digits
                .iter()
                .map(|d| {
                    (0..layers as usize).fold(Fp2::ZERO, |v, l| {
                        v + basis(final_layer, l) * extension(&rotate(d, l), final_local)
                    })
                })
                .collect();
            let products: Vec<_> = checked.products().collect();
            let half = products.len() / 2;
            assert_eq!(
                products.len(),
                2 * usize::from(2 * number::<u8>(member(record, "width")) + 55).div_ceil(8)
            );
            for (index, product) in products.iter().enumerate() {
                let j = index % half;
                let upper = index >= half;
                let mut raw = Fp2::ZERO;
                for a in 0..root_count {
                    for b in 0..root_count {
                        for k in 0..4 {
                            if a + b + 2 * k == j {
                                raw += (evals[a]
                                    + if upper && a == 0 { Fp2::ONE } else { Fp2::ZERO })
                                    * (evals[b]
                                        + if upper && b == 0 { Fp2::ONE } else { Fp2::ZERO })
                                    * evals[root_count + k];
                            }
                        }
                    }
                }
                let final_value = product.evaluate(&full, &evals).unwrap();
                assert_eq!(
                    final_value.value,
                    fp_ext(
                        &(0..1 << local.len())
                            .map(|i| basis(&local, i))
                            .collect::<Vec<_>>(),
                        final_local
                    ) * fp_ext(
                        &(0..1 << rho.len())
                            .map(|i| basis(&rho, i))
                            .collect::<Vec<_>>(),
                        final_layer
                    ) * raw
                );
                let first = product.instance().unwrap();
                assert_eq!(first.vars, full.len());
                assert_eq!(first.evals, evals.len());
                let expected = (0..layers as usize).fold(Fp2::ZERO, |sum, l| {
                    sum + basis(&rho, l)
                        * (0..3).fold(Fp2::ZERO, |sum, t| {
                            let mut raw = Fp2::ZERO;
                            for a in 0..root_count {
                                for b in 0..root_count {
                                    for k in 0..4 {
                                        if a + b + 2 * k == j {
                                            raw += (Fp2::from(digits[a][(t + l) % 3])
                                                + if upper && a == 0 {
                                                    Fp2::ONE
                                                } else {
                                                    Fp2::ZERO
                                                })
                                                * (Fp2::from(digits[b][(t + l) % 3])
                                                    + if upper && b == 0 {
                                                        Fp2::ONE
                                                    } else {
                                                        Fp2::ZERO
                                                    })
                                                * Fp2::from(digits[root_count + k][(t + l) % 3]);
                                        }
                                    }
                                }
                            }
                            sum + basis(&local, t) * raw
                        })
                });
                assert_eq!(first.claim, expected);
                for (i, claim) in final_value.claims.iter().enumerate() {
                    let source_ids: Vec<_> = stack.sites().iter().map(|s| s.sources()[i]).collect();
                    // First two sources are S/rho, then the three u lows and root lows.
                    let col = i;
                    let columns: Vec<_> = (0..layers as usize)
                        .map(|l| rotate(&sourcecols[col], l))
                        .collect();
                    check_claim(claim, &source_ids, &columns);
                }
                let mut short = evals.clone();
                short.pop();
                assert!(matches!(
                    product.evaluate(&full, &short),
                    Err(GadgetError::Rejected { .. })
                ));
            }
            assert!(matches!(
                stack.check(&local, &rho, &layer_values[..layer_values.len() - 1]),
                Err(GadgetError::Internal)
            ));
        }
    }
}

#[test]
fn reciprocal_bracket_uses_original_tables_and_half_key_source_point() {
    let vectors = read_vectors("gadget-finals/softmax.json");
    for case in list(&vectors, "cases").iter().filter(|c| {
        matches!(
            string(member(c, "name")),
            "scalar" | "padded" | "wide" | "bit-limbs"
        )
    }) {
        let (layers, n, heads) = (
            3,
            number::<u32>(member(case, "n")),
            number::<u32>(member(case, "heads")),
        );
        let (c, g) = models(
            layers,
            n,
            heads,
            2 * heads,
            0,
            2,
            number(member(case, "sigma")),
            number(member(case, "limb")),
        );
        let plan = Plan::derive(&g, c.gadget_policy).unwrap();
        let first = plan.softmaxes().next().unwrap().unwrap();
        let stack = plan.softmax_stack(first.node()).unwrap();
        let row = fields(member(case, "row_target"));
        let rho = point(2, 61);
        let m = bits(n as usize);
        let scale = f(1 << m);
        let sources = fields(member(case, "source_values"));
        let keys = fields(member(case, "range_keys"));
        let zero = vec![Fp2::ZERO; keys.len()];
        let input = vec![
            Layer {
                sources: &sources,
                keys: &keys,
                values: &zero
            };
            3
        ];
        let checked = stack.check(&row, &rho, &input).unwrap();
        assert_eq!(checked.layers().len(), 3);
        for (site, c) in stack.sites().iter().zip(checked.layers()) {
            assert_eq!(
                site.range_tables(),
                c.lookups().map(|l| l.table).collect::<Vec<_>>()
            );
        }
        let columns = cols(member(case, "source_columns"));
        let half = Fp::reduce(2).inverse().unwrap();
        let ehalf: Vec<_> = columns[1]
            .iter()
            .map(|&v| v * Fp::reduce(1 << m).inverse().unwrap())
            .collect();
        let r: Vec<_> = columns[0]
            .iter()
            .map(|&v| Fp2::from(v))
            .cycle()
            .take(columns[0].len() * 3)
            .chain(vec![Fp2::ZERO; columns[0].len()])
            .collect();
        let e: Vec<_> = ehalf
            .iter()
            .map(|&v| Fp2::from(v))
            .cycle()
            .take(ehalf.len() * 3)
            .chain(vec![Fp2::ZERO; ehalf.len()])
            .collect();
        let mut target = row.clone();
        target.extend_from_slice(&rho);
        let reference_instance = reference::Product {
            tables: tables(&[r, e]),
            public: tables(&[(0..1 << target.len())
                .map(|i| scale * basis(&target, i))
                .collect()]),
            lambda: None,
            form: reference::Form::Product,
        };
        let instance = checked.instance().unwrap();
        assert_eq!(instance.claim, from_reference(reference_instance.claim()));
        let ctx = context(Mode::NonInteractive);
        let mut rt = reference::Transcript::new(&ctx.to_tce());
        let proof = proof_of(&reference::prove(
            &mut rt,
            "stage/1",
            3,
            &[reference_instance],
            None,
        ));
        let passed = sumcheck::verify_batch(
            &mut Transcript::new(&ctx),
            Prefix::Stage(1),
            3,
            &[instance],
            &proof,
            |_, p, e| checked.evaluate(p, e).ok().map(|v| v.value),
        )
        .unwrap();
        let output = checked.evaluate(&passed.point, &proof.final_evals).unwrap();
        assert_eq!(output.claims.len(), 2);
        assert_eq!(
            output.claims[0].sources,
            stack
                .sites()
                .iter()
                .map(|s| s.source(0).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output.claims[1].sources,
            stack
                .sites()
                .iter()
                .map(|s| s.source(1).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(output.claims[1].point[..m], vec![Fp2::from(half); m]);
        assert_eq!(&output.claims[1].point[m..], output.claims[0].point);
        let exp_col = cols(&Json::Array(vec![
            member(case, "bracket_exp_column").clone(),
        ]))[0]
            .clone();
        check_claim(
            &output.claims[0],
            &output.claims[0].sources,
            &vec![columns[0].clone(); 3],
        );
        check_claim(
            &output.claims[1],
            &output.claims[1].sources,
            &vec![exp_col; 3],
        );
        assert_eq!(output.claims[0].layer_point, passed.point[row.len()..]);
        assert_eq!(output.claims[1].layer_point, output.claims[0].layer_point);
        let mut bad = sources.clone();
        bad[0] += Fp2::ONE;
        let mut input = input.clone();
        input[1].sources = &bad;
        assert!(
            matches!(stack.check(&row,&rho,&input),Err(GadgetError::Rejected{gadget}) if gadget==stack.sites()[1].gadget())
        );
    }
}

#[test]
fn padding_includes_each_boolean_feature_time_head_and_layer_axis() {
    let (c, g) = models(3, 3, 1, 6, 7, 6, 3, 8);
    let plan = Plan::derive(&g, c.gadget_policy).unwrap();
    let root = plan
        .root_stack(plan.roots().next().unwrap().unwrap().node())
        .unwrap();
    for residual in [false, true] {
        let vars = root.padding_instance(residual).unwrap().vars;
        let z = point(vars, 73);
        for index in 0..1 << vars {
            let p: Vec<_> = (0..vars).map(|b| f(((index >> b) & 1) as u64)).collect();
            let time = if residual {
                (index >> 3) & 3
            } else {
                index & 3
            };
            let layer = if residual { index >> 5 } else { index >> 2 };
            let feature_ok = !residual || index & 7 < 6;
            let good = feature_ok && time < 3 && layer < 3;
            let result = root.padding(residual, &z, &p, &[f(7)]).unwrap();
            assert_eq!(
                result.value,
                basis(&z, index) * if good { Fp2::ZERO } else { f(7) }
            );
            assert_eq!(result.claims[0].layer_point, &p[vars - 2..]);
            let expected: Vec<_> = root
                .sites()
                .iter()
                .map(|r| {
                    if residual {
                        let sum = g
                            .tensor(match r.sources()[0] {
                                Source::Tensor(t) => t,
                                _ => panic!(),
                            })
                            .unwrap()
                            .producer
                            .unwrap();
                        Source::Tensor(g.node(sum).unwrap().inputs()[0])
                    } else {
                        r.sources()[1]
                    }
                })
                .collect();
            assert_eq!(result.claims[0].sources, expected);
        }
        assert!(matches!(
            root.padding(residual, &z, &vec![Fp2::ZERO; vars], &[]),
            Err(GadgetError::Rejected { .. })
        ));
    }
    let (c, g) = models(3, 3, 3, 6, 0, 2, 3, 8);
    let plan = Plan::derive(&g, c.gadget_policy).unwrap();
    let soft = plan
        .softmax_stack(plan.softmaxes().next().unwrap().unwrap().node())
        .unwrap();
    let vars = soft.padding_instance().unwrap().vars;
    let z = point(vars, 79);
    for kind in [Padding::Maximum, Padding::Reciprocal] {
        for i in 0..1 << vars {
            let p: Vec<_> = (0..vars).map(|b| f(((i >> b) & 1) as u64)).collect();
            let good = i & 3 < 3 && (i >> 2) & 3 < 3 && i >> 4 < 3;
            let result = soft.padding(kind, &z, &p, &[f(11)]).unwrap();
            assert_eq!(
                result.value,
                basis(&z, i) * if good { Fp2::ZERO } else { f(11) }
            );
            let expected: Vec<_> = soft
                .sites()
                .iter()
                .map(|s| match kind {
                    Padding::Maximum => Source::Auxiliary(s.node(), Auxiliary::SoftmaxMaximum),
                    Padding::Reciprocal => s.source(0).unwrap(),
                })
                .collect();
            assert_eq!(result.claims[0].sources, expected);
        }
    }
    assert!(matches!(
        soft.padding(Padding::Maximum, &z, &vec![Fp2::ZERO; vars + 1], &[f(1)]),
        Err(GadgetError::Internal)
    ));
    assert!(matches!(
        plan.softmax_stack(NodeId(u32::MAX)),
        Err(GadgetError::Internal)
    ));
    assert!(matches!(
        plan.root_stack(soft.sites()[0].node()),
        Err(GadgetError::Internal)
    ));
    assert!(matches!(
        plan.softmax_stack(plan.roots().next().unwrap().unwrap().node()),
        Err(GadgetError::Internal)
    ));
    let final_root = plan.roots().last().unwrap().unwrap().node();
    assert_eq!(plan.root_stack(final_root).unwrap().layer_variables(), 0);
}
