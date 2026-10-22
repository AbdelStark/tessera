// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Model-native fixed bindings and kernels against independent integer column fixtures.
//! These fixtures exercise the component preconditions; they are not complete LogUp/graph proofs.
mod claim_support;
#[allow(dead_code)]
mod split_support;
mod support;

use claim_support::{claims_of, integer, items, list, member, number, string, with_layers};
use std::collections::HashMap;
use support::extension;
use tv_core::advice::{Auxiliary, Comparison, ComparisonRole, Source};
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{
    GadgetError, Plan,
    model::{self, External},
};
use tv_core::leaf::{
    self,
    split::{self, Bound, Column, ColumnClaim},
};
use tv_core::lowering::{self, Graph, NodeId, Op};
use tv_core::stages::{LeafValues, NativeChecks, NativeContext, StageError};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn f(v: u64) -> Fp2 {
    Fp2::from(Fp::reduce(v))
}
fn point(n: usize, seed: u64) -> Vec<Fp2> {
    (0..n)
        .map(|i| {
            Fp2::new(
                Fp::reduce(seed + i as u64 * 3),
                Fp::reduce(seed + i as u64 * 7 + 1),
            )
        })
        .collect()
}
fn bits(n: u32) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}
fn field(v: &Json) -> Fp {
    Fp::reduce(integer(v).rem_euclid(i128::from(Fp::MODULUS)) as u64)
}
fn columns(v: &Json) -> Vec<Vec<Fp>> {
    items(v)
        .iter()
        .map(|v| items(v).iter().map(field).collect())
        .collect()
}
fn resized(v: &[Fp], n: u32, rotation: usize) -> Vec<Fp> {
    (0..n.next_power_of_two() as usize)
        .map(|q| {
            if q < n as usize {
                v[(q + rotation) % 3]
            } else {
                Fp::ZERO
            }
        })
        .collect()
}
fn model(case: &Json, n: u32, layers: u32) -> (ClaimList, Graph) {
    let mut c = with_layers(&claims_of("tiny-2-servable.d"), layers);
    c.arch.d_model = number(member(case, "dim"));
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = c.arch.d_model;
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
    c.quant.rsqrt_kappa = number(member(case, "kappa"));
    c.quant.rms_eps_int = number(member(case, "epsilon"));
    c.quant.exp_sigma = 3;
    for (_, s) in &mut c.quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut c.quant.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = 8;
    let public = tv_core::validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, n).unwrap();
    (c, graph)
}
fn aux(v: &Json) -> Auxiliary {
    let v = items(v);
    let name = string(&v[2]);
    let j = number(&v[3]);
    if name == "root.input" {
        return Auxiliary::RootInput(j);
    }
    if name == "root.low" {
        return Auxiliary::RootLow(j);
    }
    let (side, name) = if let Some(s) = name.strip_prefix("root.lower.") {
        (Comparison::Lower, s)
    } else {
        (Comparison::Upper, name.strip_prefix("root.upper.").unwrap())
    };
    let role = match name {
        "left.digit" => ComparisonRole::LeftDigit,
        "left.carry" => ComparisonRole::LeftCarry,
        "right.digit" => ComparisonRole::RightDigit,
        "right.carry" => ComparisonRole::RightCarry,
        "difference" => ComparisonRole::Difference,
        "borrow" => ComparisonRole::Borrow,
        _ => panic!("role"),
    };
    Auxiliary::RootComparison(side, role, j)
}
struct Fixture {
    bound: Bound,
    tokens: LeafValues,
    logits: LeafValues,
    columns: HashMap<(NodeId, Column), Vec<Fp>>,
}
fn fixture(g: &Graph, c: &ClaimList, record: &Json) -> Fixture {
    let sites = split::model_sites(g, c.gadget_policy).unwrap();
    let plan = Plan::derive(g, c.gadget_policy).unwrap();
    let leaf = leaf::ModelSchedule::derive(g, c.gadget_policy).unwrap();
    let n = g.length();
    let np = n.next_power_of_two() as usize;
    let mut p = point(leaf.schedule().height() as usize, 19);
    let frozen = items(member(record, "point"));
    for (target, value) in p.iter_mut().zip(frozen) {
        let v = items(value);
        *target = Fp2::new(field(&v[0]), field(&v[1]));
    }
    let mut cols = HashMap::new();
    let rootcols = columns(member(record, "columns"));
    for site in &sites {
        for (l, layer) in site.layers.iter().enumerate() {
            let node = g.node(layer.node).unwrap();
            match node.op() {
                Op::RsqrtRow { eps_int, .. } => {
                    let mut expected = vec![
                        Source::Tensor(node.inputs()[0]),
                        Source::Tensor(node.outputs()[0]),
                    ];
                    for s in list(record, "sources").iter().skip(2) {
                        expected.push(Source::Auxiliary(layer.node, aux(s)));
                    }
                    let root = plan
                        .roots()
                        .map(Result::unwrap)
                        .find(|s| s.node() == layer.node)
                        .unwrap();
                    assert_eq!(root.sources(), expected);
                    for (i, column) in rootcols.iter().enumerate() {
                        let column = resized(column, n, l);
                        let role = if i == 0 {
                            Column::RootInput
                        } else if i == 1 {
                            Column::Output(0)
                        } else {
                            Column::Auxiliary(aux(&list(record, "sources")[i]))
                        };
                        let column = if i == 0 {
                            column
                                .iter()
                                .enumerate()
                                .map(|(q, &x)| {
                                    x + if q < n as usize {
                                        Fp::reduce(*eps_int)
                                    } else {
                                        Fp::ZERO
                                    }
                                })
                                .collect()
                        } else {
                            column
                        };
                        cols.insert((layer.node, role), column);
                    }
                }
                Op::SoftmaxRow { sigma, .. } => {
                    let mut maxima = vec![Fp::ZERO; np];
                    let mut scores = vec![Fp::ZERO; np * np];
                    let mut exponentials = vec![Fp::ZERO; np * np];
                    let mut keys = vec![Fp::ZERO; np * np];
                    let mut reciprocal = vec![Fp::ZERO; np];
                    let mut alpha = vec![Fp::ZERO; np];
                    let mut beta = vec![Fp::ZERO; np];
                    let selected = tv_core::selection::derive(g, c.gadget_policy).unwrap();
                    let choice = selected
                        .choices
                        .iter()
                        .find(|s| s.node == layer.node)
                        .unwrap();
                    let tv_tables::TableKind::Exp { key_bits, .. } =
                        selected.tables[choice.tables[0].0 as usize].kind
                    else {
                        panic!()
                    };
                    let last = Fp::reduce((1u64 << key_bits) - 1);
                    for q in 0..np {
                        for k in 0..np {
                            keys[q * np + k] = last;
                            if q < n as usize {
                                maxima[q] = Fp::reduce(3 + q as u64 + l as u64);
                                if k < n as usize {
                                    scores[q * np + k] = maxima[q];
                                }
                            }
                            if q < n as usize && k <= q {
                                keys[q * np + k] = Fp::ZERO;
                                exponentials[q * np + k] = Fp::reduce(1 << sigma);
                            }
                        }
                    }
                    for q in 0..n as usize {
                        let sum = (q as u64 + 1) * (1 << sigma);
                        let a = (1u64 << (2 * sigma)) % sum;
                        let r = (1u64 << (2 * sigma)) / sum;
                        reciprocal[q] = Fp::reduce(r);
                        alpha[q] = Fp::reduce(a);
                        beta[q] = Fp::reduce(sum - 1 - a);
                    }
                    cols.insert((layer.node, Column::SoftmaxKey), keys);
                    cols.insert((layer.node, Column::Output(0)), exponentials);
                    cols.insert((layer.node, Column::Output(1)), reciprocal);
                    cols.insert((layer.node, Column::SoftmaxA), alpha.clone());
                    cols.insert((layer.node, Column::SoftmaxB), beta.clone());
                    for j in 0..2 {
                        cols.insert(
                            (layer.node, Column::Auxiliary(Auxiliary::SoftmaxA(j))),
                            alpha
                                .iter()
                                .map(|v| Fp::reduce((v.to_canonical() >> (8 * j)) & 255))
                                .collect(),
                        );
                        cols.insert(
                            (layer.node, Column::Auxiliary(Auxiliary::SoftmaxB(j))),
                            beta.iter()
                                .map(|v| Fp::reduce((v.to_canonical() >> (8 * j)) & 255))
                                .collect(),
                        );
                    }
                }
                _ => {}
            }
        }
    }
    let mut columns = Vec::new();
    for site in &sites {
        for layer in &site.layers {
            for col in split_support::reads(layer) {
                let read = layer
                    .lookups
                    .iter()
                    .flat_map(|l| [Some(&l.key), l.value.as_ref()])
                    .flatten()
                    .flat_map(|f| &f.terms)
                    .find(|(_, r)| r.column == col)
                    .unwrap()
                    .1
                    .clone();
                let point: Vec<_> = read.coordinates.iter().map(|&i| p[i as usize]).collect();
                let value = cols
                    .get(&(layer.node, col))
                    .map_or(Fp2::ZERO, |v| extension(v, &point));
                columns.push(ColumnClaim {
                    node: layer.node,
                    column: col,
                    point,
                    value,
                });
            }
        }
    }
    let layout = split::Layout::new(&sites).unwrap();
    let selected = tv_core::selection::derive(g, c.gadget_policy).unwrap();
    let multiplicities = layout
        .instances()
        .map(|(id, _)| {
            (
                id,
                tv_core::logup::Claim {
                    point: point(selected.tables[id.0 as usize].log2_rows as usize, 31),
                    value: Fp2::ZERO,
                },
            )
        })
        .collect();
    let forests = leaf
        .schedule()
        .trees()
        .iter()
        .enumerate()
        .filter_map(|(tree, t)| {
            if let leaf::Tree::Forest { height, trees_log } = t {
                let full = point((height + trees_log) as usize, 37);
                let local = &full[..2 * bits(n)];
                let layer = &full[2 * bits(n)..];
                let causal = (0..np * np).fold(Fp2::ZERO, |v, i| {
                    v + if i % np <= i / np && i / np < n as usize {
                        tv_core::sumcheck::eq(
                            local,
                            &(0..local.len())
                                .map(|j| f((i >> j & 1) as u64))
                                .collect::<Vec<_>>(),
                        )
                        .unwrap()
                    } else {
                        Fp2::ZERO
                    }
                });
                let real = (0..c.arch.n_layers).fold(Fp2::ZERO, |s, l| {
                    s + tv_core::sumcheck::eq(
                        layer,
                        &(0..layer.len())
                            .map(|j| f(u64::from(l) >> j & 1))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                });
                Some(leaf::ForestClaim {
                    tree,
                    claim: tv_core::tree::TreeClaim {
                        point: full,
                        value: Fp2::ONE - causal * real,
                    },
                })
            } else {
                None
            }
        })
        .collect();
    Fixture {
        bound: Bound {
            columns,
            multiplicities,
            forests,
            atoms: Vec::new(),
            point: p,
        },
        tokens: LeafValues {
            point: point(bits(n), 41),
            values: vec![f(7)],
        },
        logits: LeafValues {
            point: point(bits(n) + bits(c.arch.vocab), 43),
            values: vec![f(9)],
        },
        columns: cols,
    }
}

#[test]
fn complete_model_slots_and_canonical_leaf_bindings_have_independent_counts_and_points() {
    let vectors = read_vectors("gadget-finals/roots.json");
    for name in [
        "zero-root",
        "width-1",
        "width-8",
        "width-16",
        "width-32",
        "constant-above-P",
    ] {
        let case = list(&vectors, "cases")
            .iter()
            .find(|c| string(member(c, "name")) == name)
            .unwrap();
        let record = &list(case, "roots")[0];
        for (n, layers) in [(1, 1), (3, 3), (5, 3)] {
            let (c, g) = model(case, n, layers);
            let gadgets = Plan::derive(&g, c.gadget_policy).unwrap();
            let model = model::Plan::derive(&gadgets).unwrap();
            let own = fixture(&g, &c, record);
            let prepared = model
                .bind(
                    &own.bound,
                    External {
                        tokens: &own.tokens,
                        logits: &own.logits,
                    },
                )
                .unwrap();
            let root_count = 2 * layers + 1;
            let coefficient = (2 * number::<usize>(member(record, "width")) + 55).div_ceil(8);
            assert_eq!(
                model.native().len(),
                3 * (2 * coefficient + 2) + 5,
                "two template roots and one epilogue root"
            );
            assert_eq!(gadgets.roots().count(), root_count as usize);
            assert_eq!(model.schedule().stages()[0].degree, 4);
            assert_eq!(prepared.leaf().len(), model.schedule().leaf().len());
            assert!(
                model
                    .schedule()
                    .leaf()
                    .iter()
                    .all(|s| !matches!(g.node(s.site).unwrap().op(), Op::SoftmaxRow { .. }))
            );
            assert_eq!(prepared.leaf().last().unwrap(), &own.logits);
            for (leaf, values) in model.schedule().leaf().iter().zip(prepared.leaf()) {
                if let Some(input) = leaf.input
                    && matches!(g.node(leaf.site).unwrap().op(), Op::RsqrtRow { .. })
                {
                    assert_eq!(input, 0);
                    let site = split::model_sites(&g, c.gadget_policy)
                        .unwrap()
                        .into_iter()
                        .find(|s| s.layers[0].node == leaf.site)
                        .unwrap();
                    for (layer, &v) in site.layers.iter().zip(&values.values) {
                        let u = own.columns.get(&(layer.node, Column::RootInput)).unwrap();
                        let eps = number::<u64>(member(case, "epsilon"));
                        let psi = (0..n).fold(Fp2::ZERO, |sum, q| {
                            sum + tv_core::sumcheck::eq(
                                &values.point,
                                &(0..values.point.len())
                                    .map(|j| f(u64::from(q) >> j & 1))
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap()
                        });
                        assert_eq!(v, extension(u, &values.point) - f(eps) * psi);
                    }
                }
            }
            let ctx = NativeContext {
                stage: 1,
                layer: &point(bits(layers), 47),
                zero: &point(model.schedule().stages()[0].zero_vars, 53),
            };
            for meta in model.native() {
                let i = NativeChecks::instance(&prepared, meta.tag, ctx).unwrap();
                assert_eq!((i.vars, i.evals), (meta.vars, meta.evaluations));
                if meta.zero {
                    assert_eq!(i.claim, Fp2::ZERO);
                }
            }
        }
    }
}

fn basis(p: &[Fp2], index: usize) -> Fp2 {
    p.iter().enumerate().fold(Fp2::ONE, |a, (j, &r)| {
        a * if index >> j & 1 == 0 { Fp2::ONE - r } else { r }
    })
}
fn eq(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(Fp2::ONE, |p, (&a, &b)| {
        p * (a * b + (Fp2::ONE - a) * (Fp2::ONE - b))
    })
}
fn selector(p: &[Fp2], axes: &[u32], causal: bool) -> Fp2 {
    assert_eq!(p.len(), axes.iter().map(|&n| bits(n)).sum());
    (0..1usize << p.len()).fold(Fp2::ZERO, |sum, i| {
        let mut at = i;
        let mut indices = Vec::new();
        let mut valid = true;
        for &n in axes {
            let width = 1 << bits(n);
            let v = at % width;
            at /= width;
            valid &= v < n as usize;
            indices.push(v);
        }
        if valid && (!causal || indices[0] <= indices[1]) {
            sum + basis(p, i)
        } else {
            sum
        }
    })
}
fn layered(tables: &[Vec<Fp>], p: &[Fp2]) -> Fp2 {
    let mut table = vec![Fp::ZERO; tables[0].len() * tables.len().next_power_of_two()];
    for (l, v) in tables.iter().enumerate() {
        table[l * v.len()..(l + 1) * v.len()].copy_from_slice(v);
    }
    extension(&table, p)
}
fn literal(v: &[Fp2], j: usize, upper: bool) -> Fp2 {
    let r = v.len() - 4;
    let mut sum = Fp2::ZERO;
    for i in 0..r {
        for h in 0..r {
            for k in 0..4 {
                if i + h + 2 * k == j {
                    sum += (v[i] + if upper && i == 0 { Fp2::ONE } else { Fp2::ZERO })
                        * (v[h] + if upper && h == 0 { Fp2::ONE } else { Fp2::ZERO })
                        * v[r + k];
                }
            }
        }
    }
    sum
}
#[test]
fn every_model_native_target_final_and_graph_source_matches_literal_integer_tables() {
    let vectors = read_vectors("gadget-finals/roots.json");
    for name in ["zero-root", "width-8", "width-32", "constant-above-P"] {
        let case = list(&vectors, "cases")
            .iter()
            .find(|c| string(member(c, "name")) == name)
            .unwrap();
        let record = &list(case, "roots")[0];
        for (n, layers) in [(1, 1), (3, 3), (5, 3)] {
            let (c, g) = model(case, n, layers);
            let gadgets = Plan::derive(&g, c.gadget_policy).unwrap();
            let model = model::Plan::derive(&gadgets).unwrap();
            let own = fixture(&g, &c, record);
            let prepared = model
                .bind(
                    &own.bound,
                    External {
                        tokens: &own.tokens,
                        logits: &own.logits,
                    },
                )
                .unwrap();
            let sites = split::model_sites(&g, c.gadget_policy).unwrap();
            let rho = point(bits(layers), 59);
            let zero = point(model.schedule().stages()[0].zero_vars, 61);
            let ctx = NativeContext {
                stage: 1,
                layer: &rho,
                zero: &zero,
            };
            for meta in model.native() {
                let site = sites
                    .iter()
                    .find(|s| s.layers[0].node == meta.node)
                    .unwrap();
                let count = site.layers.len() as u32;
                let lambda = bits(count);
                let p = point(meta.vars, 67);
                let local = p.len() - lambda;
                let mut target = own.bound.point[..bits(n)].to_vec();
                target.extend(&rho[..lambda]);
                let (incoming, values, expected, graph) = match g.node(meta.node).unwrap().op() {
                    Op::RsqrtRow { dim, .. } => {
                        let mut digits = columns(member(record, "root_digits"));
                        let root_digits = digits.len();
                        digits.extend(columns(member(record, "u_digits")));
                        let by_digit: Vec<Vec<Vec<Fp>>> = digits
                            .iter()
                            .map(|d| (0..count as usize).map(|l| resized(d, n, l)).collect())
                            .collect();
                        let m = (2 * number::<usize>(member(record, "width")) + 55).div_ceil(8);
                        if !meta.zero {
                            let index = model
                                .native()
                                .iter()
                                .filter(|s| s.node == meta.node && !s.zero)
                                .position(|s| s.tag == meta.tag)
                                .unwrap();
                            let upper = index >= m;
                            let j = index % m;
                            let raw: Vec<_> = by_digit.iter().map(|d| layered(d, &p)).collect();
                            assert_eq!(raw.len(), root_digits + 4);
                            let products: Vec<Vec<Fp>> = (0..count as usize)
                                .map(|l| {
                                    (0..n.next_power_of_two() as usize)
                                        .map(|q| {
                                            let v: Vec<_> = by_digit
                                                .iter()
                                                .map(|d| Fp2::from(d[l][q]))
                                                .collect();
                                            literal(&v, j, upper).c0
                                        })
                                        .collect()
                                })
                                .collect();
                            let base = columns(member(record, "columns"));
                            let s: Vec<_> = (0..count as usize)
                                .map(|l| resized(&base[0], n, l))
                                .collect();
                            let r: Vec<_> = (0..count as usize)
                                .map(|l| resized(&base[1], n, l))
                                .collect();
                            (
                                layered(&products, &target),
                                raw.clone(),
                                eq(&target, &p) * literal(&raw, j, upper),
                                vec![layered(&s, &p), layered(&r, &p)],
                            )
                        } else {
                            let residual =
                                meta.outputs[0].tensor != g.node(meta.node).unwrap().outputs()[0];
                            let data: Vec<_> = if residual {
                                let frozen: Vec<_> = items(member(record, "residuals"))
                                    .iter()
                                    .map(field)
                                    .collect();
                                let fp = dim.next_power_of_two() as usize;
                                (0..count as usize)
                                    .map(|l| {
                                        (0..n.next_power_of_two() as usize * fp)
                                            .map(|i| {
                                                if i / fp < n as usize {
                                                    frozen[((i / fp + l) % 3) * fp + i % fp]
                                                } else {
                                                    Fp::ZERO
                                                }
                                            })
                                            .collect()
                                    })
                                    .collect()
                            } else {
                                let base = columns(member(record, "columns"));
                                (0..count as usize)
                                    .map(|l| resized(&base[1], n, l))
                                    .collect()
                            };
                            let raw = layered(&data, &p);
                            let mask = if residual {
                                selector(&p, &[*dim, n, count], false)
                            } else {
                                selector(&p, &[n, count], false)
                            };
                            (
                                Fp2::ZERO,
                                vec![raw],
                                eq(&zero[zero.len() - p.len()..], &p) * (Fp2::ONE - mask) * raw,
                                vec![raw],
                            )
                        }
                    }
                    Op::SoftmaxRow { .. } => {
                        let np = n.next_power_of_two() as usize;
                        let owners: Vec<_> = site.layers.iter().map(|l| l.node).collect();
                        let mut maxima = Vec::new();
                        let mut scores = Vec::new();
                        let mut rs = Vec::new();
                        let mut es = Vec::new();
                        for (l, &owner) in owners.iter().enumerate() {
                            let max: Vec<_> = (0..np)
                                .map(|q| {
                                    if q < n as usize {
                                        Fp::reduce(3 + q as u64 + l as u64)
                                    } else {
                                        Fp::ZERO
                                    }
                                })
                                .collect();
                            maxima.push(max);
                            scores.push(
                                (0..np * np)
                                    .map(|i| {
                                        if i % np < n as usize && i / np < n as usize {
                                            Fp::reduce(3 + (i / np) as u64 + l as u64)
                                        } else {
                                            Fp::ZERO
                                        }
                                    })
                                    .collect(),
                            );
                            rs.push(own.columns[&(owner, Column::Output(1))].clone());
                            let exp = &own.columns[&(owner, Column::Output(0))];
                            let inv = Fp::reduce(np as u64).inverse().unwrap();
                            es.push(
                                (0..np)
                                    .map(|q| {
                                        exp[q * np..(q + 1) * np]
                                            .iter()
                                            .fold(Fp::ZERO, |s, &e| s + e)
                                            * inv
                                    })
                                    .collect(),
                            );
                        }
                        let index = model
                            .native()
                            .iter()
                            .filter(|s| s.node == meta.node && !s.zero)
                            .position(|s| s.tag == meta.tag);
                        if !meta.zero && index.unwrap() < 2 {
                            let row: Vec<_> = p[bits(n)..].to_vec();
                            let raw = vec![layered(&maxima, &row), layered(&scores, &p)];
                            let target = if index == Some(0) {
                                let mut t = own.bound.point[..2 * bits(n)].to_vec();
                                t.extend(&rho);
                                t
                            } else {
                                own.bound
                                    .forests
                                    .first()
                                    .map(|f| f.claim.point.clone())
                                    .unwrap_or_else(|| {
                                        let mut t = own.bound.point[..2 * bits(n)].to_vec();
                                        t.extend(&rho);
                                        t
                                    })
                            };
                            (
                                Fp2::ZERO,
                                raw.clone(),
                                eq(&target, &p)
                                    * selector(&p, &[n, n, count], true)
                                    * (raw[0] - raw[1]),
                                vec![raw[1]],
                            )
                        } else if !meta.zero {
                            let raw = vec![layered(&rs, &p), layered(&es, &p)];
                            let product: Vec<Vec<_>> = rs
                                .iter()
                                .zip(&es)
                                .map(|(r, e)| {
                                    r.iter()
                                        .zip(e)
                                        .map(|(&r, &e)| r * e * Fp::reduce(np as u64))
                                        .collect()
                                })
                                .collect();
                            (
                                layered(&product, &target),
                                raw.clone(),
                                eq(&target, &p) * raw[0] * raw[1] * f(np as u64),
                                vec![raw[0]],
                            )
                        } else {
                            let reciprocal = !meta.outputs.is_empty();
                            let raw = layered(if reciprocal { &rs } else { &maxima }, &p);
                            (
                                Fp2::ZERO,
                                vec![raw],
                                eq(&zero[zero.len() - p.len()..], &p)
                                    * (Fp2::ONE - selector(&p, &[n, count], false))
                                    * raw,
                                if reciprocal { vec![raw] } else { Vec::new() },
                            )
                        }
                    }
                    _ => panic!(),
                };
                assert_eq!(
                    NativeChecks::instance(&prepared, meta.tag, ctx)
                        .unwrap()
                        .claim,
                    incoming,
                    "{name}: tag {} target",
                    meta.tag
                );
                let actual = NativeChecks::evaluate(&prepared, meta.tag, ctx, &p, &values).unwrap();
                assert_eq!(
                    actual.value, expected,
                    "{name}: tag {} final, local {local}",
                    meta.tag
                );
                assert_eq!(
                    actual.graph, graph,
                    "{name}: tag {} original graph sources",
                    meta.tag
                );
            }
        }
    }
}

#[test]
fn bound_identity_point_forest_and_fixed_gadget_failures_have_exact_errors() {
    let v = read_vectors("gadget-finals/roots.json");
    let case = list(&v, "cases")
        .iter()
        .find(|c| string(member(c, "name")) == "width-8")
        .unwrap();
    let record = &list(case, "roots")[0];
    let (c, g) = model(case, 3, 3);
    let gadgets = Plan::derive(&g, c.gadget_policy).unwrap();
    let model = model::Plan::derive(&gadgets).unwrap();
    let own = fixture(&g, &c, record);
    let external = External {
        tokens: &own.tokens,
        logits: &own.logits,
    };
    for fault in 0..16 {
        let mut bad = own.bound.clone();
        match fault {
            0 => {
                bad.columns.pop();
            }
            1 => bad.columns[0].node = NodeId(u32::MAX),
            2 => bad.columns[0].column = Column::Input(255),
            3 => bad.columns[0].point.push(Fp2::ONE),
            4 => bad.point[0] += Fp2::ONE,
            5 => bad.multiplicities[0].0 = tv_tables::TableId(u32::MAX),
            6 => bad.forests[0].tree += 1,
            7 => {
                bad.forests.pop();
            }
            8 => {
                bad.point.pop();
            }
            9 => {
                bad.multiplicities.pop();
            }
            10 => {
                bad.multiplicities[0].1.point.pop();
            }
            11 => {
                bad.forests[0].claim.point.pop();
            }
            12 => {
                bad.columns.push(bad.columns[0].clone());
            }
            13 => {
                bad.columns.swap(0, 1);
            }
            14 => {
                bad.forests.push(bad.forests[0].clone());
            }
            15 => {
                let other = bad.multiplicities[1].clone();
                bad.multiplicities[0] = other;
            }
            _ => unreachable!(),
        }
        assert!(
            matches!(model.bind(&bad, external), Err(GadgetError::Internal)),
            "fault {fault}"
        );
    }
    let mut bad = own.bound.clone();
    let root = gadgets.roots().next().unwrap().unwrap();
    let m = root.product_geometry().coefficients();
    let terminal = Column::Auxiliary(Auxiliary::RootComparison(
        Comparison::Lower,
        ComparisonRole::LeftCarry,
        (m - 1) as u32,
    ));
    bad.columns
        .iter_mut()
        .find(|s| s.node == root.node() && s.column == terminal)
        .unwrap()
        .value += Fp2::ONE;
    assert!(
        matches!(model.bind(&bad,external),Err(GadgetError::Rejected{gadget}) if gadget==root.gadget())
    );
    let prepared = model.bind(&own.bound, external).unwrap();
    let rho = point(bits(3), 79);
    let zero = point(model.schedule().stages()[0].zero_vars, 83);
    let ctx = NativeContext {
        stage: 1,
        layer: &rho,
        zero: &zero,
    };
    let tag = model.native()[0].tag;
    assert_eq!(
        NativeChecks::instance(&prepared, tag + 10000, ctx),
        Err(StageError::Internal)
    );
    assert_eq!(
        NativeChecks::instance(&prepared, tag, NativeContext { stage: 2, ..ctx }),
        Err(StageError::Internal)
    );
    assert_eq!(
        NativeChecks::instance(&prepared, tag, NativeContext { layer: &[], ..ctx }),
        Err(StageError::Internal)
    );
    assert_eq!(
        NativeChecks::instance(&prepared, tag, NativeContext { zero: &[], ..ctx }),
        Err(StageError::Internal)
    );
    for meta in model.native() {
        assert_eq!(
            NativeChecks::instance(&prepared, meta.tag, NativeContext { layer: &[], ..ctx }),
            Err(StageError::Internal)
        );
        for fault in 0..3 {
            let mut p = point(meta.vars, 89);
            let mut raw = vec![Fp2::ZERO; meta.evaluations];
            match fault {
                0 => {
                    p.pop();
                }
                1 => {
                    raw.pop();
                }
                2 => {
                    raw.push(Fp2::ONE);
                }
                _ => unreachable!(),
            }
            let expected = if fault == 0 {
                StageError::Internal
            } else {
                let gadget = gadgets
                    .roots()
                    .map(Result::unwrap)
                    .find(|site| site.node() == meta.node)
                    .map(|site| site.gadget())
                    .or_else(|| {
                        gadgets
                            .softmaxes()
                            .map(Result::unwrap)
                            .find(|site| site.node() == meta.node)
                            .map(|site| site.gadget())
                    })
                    .unwrap();
                StageError::Gadget { gadget }
            };
            assert!(matches!(
                NativeChecks::evaluate(&prepared, meta.tag, ctx, &p, &raw),
                Err(actual) if actual == expected
            ));
        }
    }
    for fault in 0..4 {
        let mut tokens = own.tokens.clone();
        let mut logits = own.logits.clone();
        match fault {
            0 => tokens.point.push(Fp2::ONE),
            1 => tokens.values.push(Fp2::ONE),
            2 => logits.point.push(Fp2::ONE),
            3 => logits.values.push(Fp2::ONE),
            _ => unreachable!(),
        }
        assert!(matches!(
            model.bind(
                &own.bound,
                External {
                    tokens: &tokens,
                    logits: &logits
                }
            ),
            Err(GadgetError::Internal)
        ));
    }
}
