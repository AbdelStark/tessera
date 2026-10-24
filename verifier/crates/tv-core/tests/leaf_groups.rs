// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054's group layouts, against oracles written from the RFC: every stage block entry
//! decoded from its domain's shape and its token axis's element tags; hand-derived coordinates;
//! RFC-0045's split identity and column identity summed over the Boolean cube; the ungrouped
//! stage's own split check; and every refusal.
mod claim_support;
mod group_support;
use group_support::{
    bits, candidates, domain, e, entry_axes, entry_of, eq, f, mle, model, random, tags,
};

use std::collections::{BTreeSet, HashMap};
use tv_core::advice::Auxiliary;
use tv_core::leaf::groups::{Error, Extent, Form, Group, Grouped, Instance, Source, Value};
use tv_core::leaf::split::{
    self, Affine, Column, Layer, Layout, Lookup, Read, Site, TokenAxis, model_sites,
};
use tv_core::leaf::{ForestShape, ModelSchedule, Tree, Verified};
use tv_core::logup::{Claim, InstanceClaims, LeafClaims};
use tv_core::lowering::{Graph, NodeId};
use tv_core::protocol::LeafGroups;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::TableId;
use tv_testkit::SplitMix64;

fn groups(token_bits: u8, layer_bits: u8) -> LeafGroups {
    LeafGroups {
        token_bits,
        layer_bits,
    }
}
fn read(column: Column, coordinates: &[u32]) -> Read {
    Read {
        column,
        coordinates: coordinates.to_vec(),
    }
}
fn affine(constant: u64, terms: &[(u64, Column, &[u32])]) -> Affine {
    Affine {
        constant: f(constant),
        terms: terms
            .iter()
            .map(|&(a, column, coordinates)| (f(a), read(column, coordinates)))
            .collect(),
    }
}
fn lookup(
    table: u32,
    variables: u32,
    token: Option<(u32, u32)>,
    key: Affine,
    value: Option<Affine>,
) -> Lookup {
    Lookup {
        table: TableId(table),
        variables,
        key,
        value,
        token: token.map(|(low, bits)| TokenAxis { low, bits }),
    }
}
fn site(nodes: &[u32], lookups: impl Fn() -> Vec<Lookup>) -> Site {
    Site {
        layers: nodes
            .iter()
            .map(|&node| Layer {
                node: NodeId(node),
                lookups: lookups(),
            })
            .collect(),
    }
}
fn form(constant: u64, terms: &[(u64, usize)]) -> Form {
    Form {
        constant: f(constant),
        terms: terms.iter().map(|&(a, i)| (f(a), i)).collect(),
    }
}
fn value(node: u32, column: Column, polynomial: usize, coordinates: &[u32], bits: u32) -> Value {
    Value {
        node: NodeId(node),
        column,
        polynomial,
        coordinates: coordinates.to_vec(),
        bits,
    }
}
/// `(node, lookup, variables, offset)` of each block of an instance, in column order.
fn blocks(instance: &Instance) -> Vec<(u32, usize, u32, u64)> {
    instance
        .blocks
        .iter()
        .map(|b| (b.node.0, b.lookup, b.variables, b.offset))
        .collect()
}

/// RFC-0045 'The column' over one group's instances: ascending tables, aligned blocks in
/// size order with stable ties, no gaps, the padding exactly the tail.
fn check_columns(group: &Group) {
    assert!(group.instances.windows(2).all(|w| w[0].table < w[1].table));
    for instance in &group.instances {
        assert!(!instance.blocks.is_empty(), "an instance holds a block");
        let mut next = 0u64;
        for (i, block) in instance.blocks.iter().enumerate() {
            assert_eq!(block.offset, next);
            assert_eq!(block.offset % (1 << block.variables), 0);
            if let Some(previous) = i.checked_sub(1).map(|j| &instance.blocks[j]) {
                assert!(previous.variables >= block.variables);
                if previous.variables == block.variables {
                    assert!(
                        (previous.site, previous.layer, previous.lookup)
                            < (block.site, block.layer, block.lookup)
                    );
                }
            }
            next += 1 << block.variables;
        }
        assert_eq!(instance.entries, next);
        assert!(1u64 << instance.height >= next);
        assert!(instance.height == 0 || 1u64 << (instance.height - 1) < next);
    }
}

/// The template's first and end node ids and its layer width, from the checkpoints alone.
fn template(graph: &Graph) -> (u32, u32, u32) {
    let producer = |c| graph.tensor(c).unwrap().producer.unwrap().0;
    let checkpoints = graph.checkpoints();
    let first = producer(checkpoints[0]) + 1;
    let end = producer(*checkpoints.last().unwrap()) + 1;
    (first, end, (end - first) / (checkpoints.len() as u32 - 1))
}
/// The layer subset of a gadget node: its layer's low bits, 0 outside the template.
fn subset(graph: &Graph, node: NodeId, layer_bits: u8) -> u32 {
    let (first, end, width) = template(graph);
    if node.0 >= first && node.0 < end {
        ((node.0 - first) / width) % (1 << layer_bits)
    } else {
        0
    }
}

#[test]
fn every_model_request_carries_its_domains_token_axis_among_its_variables() {
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
    ] {
        for n in [1, 4, 5, 16] {
            let (graph, policy) = model(name, Some(3), n);
            let tags = tags(&graph);
            let mut tokened = 0;
            for site in model_sites(&graph, policy).unwrap() {
                for layer in &site.layers {
                    for (j, request) in layer.lookups.iter().enumerate() {
                        let tensor = domain(&graph, layer.node, j);
                        let dims = graph.tensor(tensor).unwrap().shape.dims().to_vec();
                        // The element tags' axis; with one token, the axis of length 1 the
                        // derivation chose must be one of them, and every choice has no bits.
                        let found = candidates(&graph, &tags, tensor);
                        let expected = found.first().map(|&a| TokenAxis {
                            low: dims[a as usize + 1..].iter().map(|&d| bits(d)).sum(),
                            bits: bits(dims[a as usize]),
                        });
                        if n > 1 {
                            assert_eq!(request.token, expected, "{name} n={n} {:?}", layer.node);
                        } else {
                            assert_eq!(request.token.map(|t| t.bits), Some(0));
                        }
                        tokened += usize::from(request.token.is_some());
                    }
                }
            }
            assert!(tokened > 0);
        }
    }
}

#[test]
fn every_stage_block_entry_lies_in_exactly_one_group() {
    for (name, layers, n) in [
        ("tiny-2-servable", Some(3), 5),
        ("tiny-2-minimal", None, 4),
        ("tiny-2-affine", Some(3), 8),
        ("tiny-4-servable", None, 5),
    ] {
        let (graph, policy) = model(name, layers, n);
        let tags = tags(&graph);
        let sites = model_sites(&graph, policy).unwrap();
        let real = sites.iter().map(|s| s.layers.len()).max().unwrap() as u32;
        for g_t in 0..=bits(n).min(3) as u8 {
            for g_l in 0..=bits(real) as u8 {
                let grouped = Grouped::derive(&graph, policy, groups(g_t, g_l)).unwrap();
                assert_eq!(grouped.count(), 1 << (g_t + g_l));
                let mut covered: HashMap<(usize, u32, usize), Vec<u32>> = HashMap::new();
                for h in 0..grouped.count() {
                    let group = grouped.group(h).unwrap();
                    assert_eq!(
                        (group.index, group.tokens, group.layers),
                        (h, h % (1 << g_t), h >> g_t)
                    );
                    check_columns(&group);
                    for instance in &group.instances {
                        for block in &instance.blocks {
                            let entry = &sites[block.site].layers[block.layer as usize];
                            assert_eq!(entry.node, block.node);
                            assert_eq!(group.layers, subset(&graph, block.node, g_l));
                            let request = &entry.lookups[block.lookup];
                            assert_eq!(request.table, instance.table);
                            // The domain and its token axis from the graph and the element tags.
                            let tensor = domain(&graph, block.node, block.lookup);
                            let dims = graph.tensor(tensor).unwrap().shape.dims().to_vec();
                            let axis = candidates(&graph, &tags, tensor).first().copied();
                            let widths: Vec<u32> = dims.iter().map(|&d| bits(d)).collect();
                            assert_eq!(widths.iter().sum::<u32>(), request.variables);
                            let mut restricted = widths.clone();
                            match axis {
                                Some(a) => restricted[a as usize] -= u32::from(g_t),
                                None => assert_eq!(group.tokens, 0),
                            }
                            assert_eq!(block.variables, restricted.iter().sum::<u32>());
                            let counts = covered
                                .entry((block.site, block.layer, block.lookup))
                                .or_insert_with(|| vec![0; 1 << request.variables]);
                            for r in 0..1u64 << block.variables {
                                let mut index = entry_axes(r, &restricted);
                                if let Some(a) = axis {
                                    let a = a as usize;
                                    index[a] = index[a] << g_t | u64::from(group.tokens);
                                }
                                counts[entry_of(&index, &widths) as usize] += 1;
                            }
                        }
                    }
                }
                let mut stage = 0;
                for (i, site) in sites.iter().enumerate() {
                    for (l, layer) in site.layers.iter().enumerate() {
                        for j in 0..layer.lookups.len() {
                            stage += 1;
                            let counts = &covered[&(i, l as u32, j)];
                            assert!(
                                counts.iter().all(|&c| c == 1),
                                "{name} g=({g_t},{g_l}) site {i} layer {l} lookup {j}"
                            );
                        }
                    }
                }
                assert_eq!(covered.len(), stage);
            }
        }
    }
}

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [54; 32],
        session_id: [45; 16],
    })
}
fn evaluate(form: &Form, splits: &[Fp2]) -> Fp2 {
    form.terms
        .iter()
        .fold(Fp2::from(form.constant), |sum, &(a, i)| {
            sum + Fp2::from(a) * splits[i]
        })
}
/// `LT(p, m)`, summed over the cube.
fn lt(p: &[Fp2], m: u64) -> Fp2 {
    (0..m).fold(Fp2::ZERO, |sum, x| sum + eq(p, x))
}
/// RFC-0045's column claim from an instance's blocks: `Σ_b eq(P[k_b..h], o_b / 2^{k_b}) ·
/// block_b + (1 − LT(P[0..h], m)) · t_0`.
fn column_claim(
    instance: &Instance,
    point: &[Fp2],
    blocks: impl Fn(usize) -> Fp2,
    padding: Fp2,
) -> Fp2 {
    let p = &point[..instance.height as usize];
    let mut sum = (Fp2::ONE - lt(p, instance.entries)) * padding;
    for (i, block) in instance.blocks.iter().enumerate() {
        let k = block.variables as usize;
        sum += eq(&p[k..], block.offset >> k) * blocks(i);
    }
    sum
}
fn sigma(point: &[Fp2]) -> Vec<Fp2> {
    point
        .iter()
        .take(1)
        .chain(point.iter().skip(1).rev())
        .copied()
        .collect()
}

#[test]
fn with_no_groups_group_zero_is_the_stage_layout_its_split_check_accepts() {
    let mut rng = SplitMix64::new(0x0054_0000);
    for (name, layers, n) in [
        ("tiny-2-servable", None, 6),
        ("tiny-2-servable", Some(3), 5),
        ("tiny-2-minimal", None, 1),
        ("tiny-2-affine", Some(3), 4),
        ("tiny-4-servable", None, 5),
    ] {
        let (graph, policy) = model(name, layers, n);
        let layout = Layout::derive(&graph, policy).unwrap();
        let grouped = Grouped::derive(&graph, policy, LeafGroups::NONE).unwrap();
        assert_eq!(grouped.count(), 1);
        let group = grouped.group(0).unwrap();
        check_columns(&group);
        assert_eq!(
            group
                .instances
                .iter()
                .map(|i| (i.table, i.entries))
                .collect::<Vec<_>>(),
            layout.instances().collect::<Vec<_>>()
        );
        assert_eq!(group.split.len(), layout.values());
        assert_eq!(grouped.values(), layout.values());
        for (i, value) in group.split.iter().enumerate() {
            assert_eq!((value.polynomial, value.bits), (i, 0));
        }
        for polynomial in grouped.polynomials() {
            assert!(polynomial.group_variables.is_empty());
            assert_eq!(polynomial.readers, 0..1);
        }
        let schedule = ModelSchedule::derive(&graph, policy).unwrap();
        let shapes: Vec<ForestShape> = schedule.forests().collect();
        assert_eq!(group.forests.len(), shapes.len());
        for (share, shape) in group.forests.iter().zip(&shapes) {
            let Tree::Forest { height, trees_log } = schedule.schedule().trees()[shape.tree] else {
                panic!("not a forest");
            };
            assert_eq!(
                (share.tree, share.height, share.trees_log, share.bits),
                (shape.tree, height, trees_log, 0)
            );
        }
        // RFC-0045's identity from this layout's blocks, offsets and forms, at random split
        // values, is what the stage's own split check accepts.
        let height = group.instances.iter().map(|i| i.height).max().unwrap();
        let point: Vec<Fp2> = (0..height + 2).map(|_| random(&mut rng)).collect();
        let splits: Vec<Fp2> = (0..layout.values()).map(|_| random(&mut rng)).collect();
        let row0: HashMap<TableId, (Fp, Option<Fp>)> = group
            .instances
            .iter()
            .map(|i| {
                let key = f(rng.next_u64());
                (i.table, (key, i.has_values.then(|| f(rng.next_u64()))))
            })
            .collect();
        let instances = group
            .instances
            .iter()
            .map(|instance| {
                let (key0, value0) = row0[&instance.table];
                let at = sigma(&point[..instance.height as usize]);
                let key = column_claim(
                    instance,
                    &point,
                    |b| evaluate(&instance.blocks[b].key, &splits),
                    Fp2::from(key0),
                );
                let value = value0.map(|v0| Claim {
                    point: at.clone(),
                    value: column_claim(
                        instance,
                        &point,
                        |b| evaluate(instance.blocks[b].value.as_ref().unwrap(), &splits),
                        Fp2::from(v0),
                    ),
                });
                InstanceClaims {
                    key: Claim {
                        point: at.clone(),
                        value: key,
                    },
                    value,
                    multiplicity: Claim {
                        point: vec![e(3)],
                        value: e(5),
                    },
                }
            })
            .collect();
        let leaf = Verified {
            lookups: LeafClaims { instances },
            forests: vec![],
            point: point.clone(),
        };
        let bound = layout
            .bind(&mut transcript(), leaf.clone(), &splits, 0, |t| {
                row0.get(&t).copied()
            })
            .unwrap();
        assert_eq!(bound.columns.len(), group.split.len());
        for ((claim, value), &split) in bound.columns.iter().zip(&group.split).zip(&splits) {
            assert_eq!((claim.node, claim.column), (value.node, value.column));
            let at: Vec<Fp2> = value
                .coordinates
                .iter()
                .map(|&c| point[c as usize])
                .collect();
            assert_eq!(claim.point, at);
            assert_eq!(claim.value, split);
        }
        // A control: one value changed fails that check.
        let mut tampered = splits.clone();
        tampered[0] += Fp2::ONE;
        assert_eq!(
            layout
                .bind(&mut transcript(), leaf, &tampered, 0, |t| row0
                    .get(&t)
                    .copied())
                .map(|_| ()),
            Err(split::Error::Mismatch)
        );
    }
}

#[test]
fn groups_differ_by_layer_subset_and_padded_layers_alone_leave_only_forest_trees() {
    let (graph, policy) = model("tiny-2-servable", Some(3), 5);
    let grouped = Grouped::derive(&graph, policy, groups(1, 2)).unwrap();
    assert_eq!(grouped.count(), 8);
    let all: Vec<Group> = (0..8).map(|h| grouped.group(h).unwrap()).collect();
    let nodes = |g: &Group| -> BTreeSet<u32> {
        g.instances
            .iter()
            .flat_map(|i| i.blocks.iter().map(|b| b.node.0))
            .collect()
    };
    // Every gadget node of a layer subset, from the graph's positions alone.
    let (first, end, width) = template(&graph);
    let sites = model_sites(&graph, policy).unwrap();
    let gadgets: BTreeSet<u32> = sites
        .iter()
        .flat_map(|s| s.layers.iter().map(|l| l.node.0))
        .collect();
    let expected = |o: u32| -> BTreeSet<u32> {
        gadgets
            .iter()
            .copied()
            .filter(|&node| {
                if node >= first && node < end {
                    (node - first) / width % 4 == o
                } else {
                    o == 0
                }
            })
            .collect()
    };
    for g in &all {
        assert_eq!(nodes(g), expected(g.layers), "group {}", g.index);
    }
    // Layer subset 0 holds the epilogue's norm beside layer 0.
    assert!(expected(0).len() > expected(1).len());
    assert_eq!(expected(1).len(), expected(2).len());
    assert!(expected(0).iter().any(|&node| node >= end));
    assert_ne!(all[0].instances, all[2].instances);
    // Layer 3 is padding: its groups hold no block and send no value, but hold the forest's
    // trees of their padded layer, whose roots are public.
    for g in &all[6..8] {
        assert!(g.instances.is_empty() && g.split.is_empty());
        assert_eq!(g.forests.len(), 1);
        assert_eq!(g.levels(), g.forests[0].height);
    }
    // Within a layer subset, every block has a token axis: the token subsets differ only in
    // their bits on `S`.
    for o in 0..4 {
        let (a, b) = (&all[2 * o], &all[2 * o + 1]);
        assert_eq!(a.instances, b.instances);
        assert_eq!(a.split.len(), b.split.len());
        for (x, y) in a.split.iter().zip(&b.split) {
            assert_eq!(
                (x.node, x.column, x.polynomial, &x.coordinates),
                (y.node, y.column, y.polynomial, &y.coordinates)
            );
            let fixed = !grouped.polynomials()[x.polynomial]
                .group_variables
                .is_empty();
            assert_eq!((x.bits, y.bits), (0, u32::from(fixed)));
        }
    }
}

/// One layer of two requests: `L0` into a Range table without a token axis reads `A` and `C`
/// over three variables; `L1`, over `[4, 8]` with its token axis on variables 3 and 4, reads
/// `B` and `A`, then `Y`.
fn tokenless() -> Vec<Site> {
    vec![site(&[5], || {
        vec![
            lookup(
                2,
                3,
                None,
                affine(
                    0,
                    &[
                        (1, Column::Input(0), &[0, 1, 2]),
                        (2, Column::Input(1), &[0, 1, 2]),
                    ],
                ),
                None,
            ),
            lookup(
                5,
                5,
                Some((3, 2)),
                affine(
                    7,
                    &[
                        (1, Column::Output(0), &[0, 1, 2, 3, 4]),
                        (4, Column::Input(0), &[0, 1, 2]),
                    ],
                ),
                Some(affine(0, &[(1, Column::Output(1), &[0, 1, 2, 3, 4])])),
            ),
        ]
    })]
}

#[test]
fn a_block_without_a_token_axis_is_whole_in_token_subset_zero_alone() {
    let extent = Extent {
        token_bits: 2,
        layers: 1,
    };
    let (a, c, b, y) = (
        Column::Input(0),
        Column::Input(1),
        Column::Output(0),
        Column::Output(1),
    );
    let grouped = Grouped::new(&tokenless(), &[], extent, groups(1, 0)).unwrap();
    let readers: Vec<_> = grouped
        .polynomials()
        .iter()
        .map(|p| (p.variables, p.group_variables.clone(), p.readers.clone()))
        .collect();
    assert_eq!(
        readers,
        [
            (3, vec![], 0..2),
            (3, vec![], 0..1),
            (5, vec![3], 0..2),
            (5, vec![3], 0..2)
        ],
        "C is read by the tokenless block alone"
    );
    let zero = grouped.group(0).unwrap();
    assert_eq!(
        zero.instances
            .iter()
            .map(|i| (i.table.0, i.entries, i.height, blocks(i)))
            .collect::<Vec<_>>(),
        [
            (2, 8, 3, vec![(5, 0, 3, 0)]),
            (5, 16, 4, vec![(5, 1, 4, 0)])
        ]
    );
    assert_eq!(
        zero.split,
        [
            value(5, a, 0, &[0, 1, 2], 0),
            value(5, c, 1, &[0, 1, 2], 0),
            value(5, b, 2, &[0, 1, 2, 3], 0),
            value(5, y, 3, &[0, 1, 2, 3], 0),
        ]
    );
    assert_eq!(zero.instances[0].blocks[0].key, form(0, &[(1, 0), (2, 1)]));
    assert_eq!(zero.instances[1].blocks[0].key, form(7, &[(1, 2), (4, 0)]));
    assert_eq!(zero.instances[1].blocks[0].value, Some(form(0, &[(1, 3)])));
    // Token subset 1 holds `L1` alone: it sends `A`, `B` and `Y`, the stage's order without
    // `C`, though `L1` reads `B` before `A`. Both token subsets follow the stage's order.
    let one = grouped.group(1).unwrap();
    assert_eq!(
        one.instances
            .iter()
            .map(|i| (i.table.0, i.entries, blocks(i)))
            .collect::<Vec<_>>(),
        [(5, 16, vec![(5, 1, 4, 0)])]
    );
    assert_eq!(
        one.split,
        [
            value(5, a, 0, &[0, 1, 2], 0),
            value(5, b, 2, &[0, 1, 2, 3], 1),
            value(5, y, 3, &[0, 1, 2, 3], 1),
        ]
    );
    assert_eq!(one.instances[0].blocks[0].key, form(7, &[(1, 1), (4, 0)]));
    assert_eq!(one.instances[0].blocks[0].value, Some(form(0, &[(1, 2)])));
    // With two token bits, token subsets 1 to 3 alike.
    let grouped = Grouped::new(&tokenless(), &[], extent, groups(2, 0)).unwrap();
    for h in 1..4 {
        let g = grouped.group(h).unwrap();
        assert_eq!(g.instances.len(), 1);
        assert_eq!(blocks(&g.instances[0]), [(5, 1, 3, 0)]);
        assert_eq!(
            g.split,
            [
                value(5, a, 0, &[0, 1, 2], 0),
                value(5, b, 2, &[0, 1, 2], h),
                value(5, y, 3, &[0, 1, 2], h),
            ]
        );
    }
}

/// A template site of two layers, nodes 10 and 20, each of two requests into one table:
/// `[4, 8]` (channels low, tokens on variables 3 and 4) keyed by `X + 16·K`, `K` the channels'
/// class, valued `Y`; and `[4]`, the tokens alone, keyed `R` and valued `W`. Then a site outside
/// the template, node 30, over `[2, 4, 8]` with its token axis in the middle, keyed `Z`.
fn four_by_eight() -> Vec<Site> {
    let full: &[u32] = &[0, 1, 2, 3, 4];
    vec![
        site(&[10, 20], || {
            vec![
                lookup(
                    1,
                    5,
                    Some((3, 2)),
                    affine(
                        0,
                        &[
                            (1, Column::Input(0), full),
                            (16, Column::Input(1), &[0, 1, 2]),
                        ],
                    ),
                    Some(affine(0, &[(1, Column::Output(0), full)])),
                ),
                lookup(
                    1,
                    2,
                    Some((0, 2)),
                    affine(0, &[(1, Column::SoftmaxA, &[0, 1])]),
                    Some(affine(0, &[(1, Column::SoftmaxB, &[0, 1])])),
                ),
            ]
        }),
        site(&[30], || {
            vec![lookup(
                0,
                6,
                Some((3, 2)),
                affine(
                    0,
                    &[(
                        1,
                        Column::Auxiliary(Auxiliary::RequantLow),
                        &[0, 1, 2, 3, 4, 5],
                    )],
                ),
                None,
            )]
        }),
    ]
}

#[test]
fn hand_derived_coordinates_for_a_four_by_eight_domain() {
    let extent = Extent {
        token_bits: 2,
        layers: 2,
    };
    let (x, k, y) = (Column::Input(0), Column::Input(1), Column::Output(0));
    let (r, w, z) = (
        Column::SoftmaxA,
        Column::SoftmaxB,
        Column::Auxiliary(Auxiliary::RequantLow),
    );
    let layer = |node: u32, first: usize, s: u32, full: &[u32], token: &[u32]| {
        vec![
            value(node, x, first, full, s),
            value(node, k, first + 1, &[0, 1, 2], 0),
            value(node, y, first + 2, full, s),
            value(node, r, first + 3, token, s),
            value(node, w, first + 4, token, s),
        ]
    };
    // One token bit: `X`'s variable 3 is its group variable, and 4 becomes `P_h[3]`; `R`'s
    // variable 0, and 1 becomes `P_h[0]`; `K`, over the channels, has none. `Z` over
    // `[2, 4, 8]`: variable 3 fixed, 4 and 5 at `P_h[3]` and `P_h[4]`.
    let grouped = Grouped::new(&four_by_eight(), &[], extent, groups(1, 0)).unwrap();
    let shapes: Vec<_> = grouped
        .polynomials()
        .iter()
        .map(|p| (p.variables, p.group_variables.clone(), p.readers.clone()))
        .collect();
    let per_layer = [
        (5, vec![3], 0..2),
        (3, vec![], 0..2),
        (5, vec![3], 0..2),
        (2, vec![0], 0..2),
        (2, vec![0], 0..2),
    ];
    let mut expected: Vec<_> = per_layer.iter().chain(&per_layer).cloned().collect();
    expected.push((6, vec![3], 0..2));
    assert_eq!(shapes, expected);
    for s in 0..2 {
        let g = grouped.group(s).unwrap();
        let mut split = layer(10, 0, s, &[0, 1, 2, 3], &[0]);
        split.extend(layer(20, 5, s, &[0, 1, 2, 3], &[0]));
        split.push(value(30, z, 10, &[0, 1, 2, 3, 4], s));
        assert_eq!(g.split, split);
        assert_eq!(
            g.instances
                .iter()
                .map(|i| (i.table.0, i.entries, i.height, blocks(i)))
                .collect::<Vec<_>>(),
            [
                (0, 32, 5, vec![(30, 0, 5, 0)]),
                (
                    1,
                    36,
                    6,
                    vec![
                        (10, 0, 4, 0),
                        (20, 0, 4, 16),
                        (10, 1, 1, 32),
                        (20, 1, 1, 34)
                    ]
                ),
            ]
        );
        let table = &g.instances[1];
        assert_eq!(table.blocks[0].key, form(0, &[(1, 0), (16, 1)]));
        assert_eq!(table.blocks[1].key, form(0, &[(1, 5), (16, 6)]));
        assert_eq!(table.blocks[2].value, Some(form(0, &[(1, 4)])));
        assert_eq!(g.levels(), 6);
    }
    // Two token bits: `X`'s variables 3 and 4 both fixed, to `s`'s two bits; `R` has no
    // variable left outside `S`.
    let grouped = Grouped::new(&four_by_eight(), &[], extent, groups(2, 0)).unwrap();
    assert_eq!(grouped.polynomials()[0].group_variables, [3, 4]);
    assert_eq!(grouped.polynomials()[3].group_variables, [0, 1]);
    for s in 0..4 {
        let g = grouped.group(s).unwrap();
        let mut split = layer(10, 0, s, &[0, 1, 2], &[]);
        split.extend(layer(20, 5, s, &[0, 1, 2], &[]));
        split.push(value(30, z, 10, &[0, 1, 2, 3], s));
        assert_eq!(g.split, split);
        assert_eq!(
            blocks(&g.instances[1]),
            [(10, 0, 3, 0), (20, 0, 3, 8), (10, 1, 0, 16), (20, 1, 0, 17)]
        );
        assert_eq!((g.instances[1].entries, g.instances[1].height), (18, 5));
    }
    // One token bit and one layer bit: layer 1 is layer subset 1's alone, and the site outside
    // the template is layer subset 0's.
    let grouped = Grouped::new(&four_by_eight(), &[], extent, groups(1, 1)).unwrap();
    let readers: Vec<_> = grouped
        .polynomials()
        .iter()
        .map(|p| p.readers.clone())
        .collect();
    let mut expected = vec![0..2; 5];
    expected.extend(vec![2..4; 5]);
    expected.push(0..2);
    assert_eq!(readers, expected);
    for h in 0..4 {
        let (s, o) = (h % 2, h / 2);
        let g = grouped.group(h).unwrap();
        let node = [10, 20][o as usize];
        let mut split = layer(node, 5 * o as usize, s, &[0, 1, 2, 3], &[0]);
        if o == 0 {
            split.push(value(30, z, 10, &[0, 1, 2, 3, 4], s));
        }
        assert_eq!(g.split, split);
        let column = g.instances.iter().find(|i| i.table.0 == 1).unwrap();
        assert_eq!(blocks(column), [(node, 0, 4, 0), (node, 1, 1, 16)]);
        assert_eq!((column.entries, column.height), (18, 5));
        assert_eq!(g.instances.len(), if o == 0 { 2 } else { 1 });
    }
}

/// RFC-0045's split identity in every group, at a random `P_h` and random tensors: a block's
/// extension at `P_h[0..k']`, summed over its restricted entries from the sites' own forms, is
/// its group form at the tensors' extensions at the derived claims. Each instance's column,
/// built entry by entry, also meets RFC-0045's column identity from the offsets.
fn split_identity(sites: &[Site], grouped: &Grouped, rng: &mut SplitMix64) {
    let g_t = u32::from(grouped.groups().token_bits);
    let tensors: HashMap<(NodeId, Column), Vec<Fp2>> = grouped
        .polynomials()
        .iter()
        .filter_map(|p| match p.source {
            Source::Split { node, column } => Some((
                (node, column),
                (0..1u64 << p.variables).map(|_| random(rng)).collect(),
            )),
            Source::Forest { .. } => None,
        })
        .collect();
    let at = |form: &Affine, node: NodeId, entry: u64| {
        form.terms
            .iter()
            .fold(Fp2::from(form.constant), |sum, (a, read)| {
                let index = read
                    .coordinates
                    .iter()
                    .enumerate()
                    .fold(0usize, |i, (bit, &c)| {
                        i | ((entry >> c & 1) as usize) << bit
                    });
                sum + Fp2::from(*a) * tensors[&(node, read.column)][index]
            })
    };
    for h in 0..grouped.count() {
        let group = grouped.group(h).unwrap();
        let levels = group.levels();
        let p: Vec<Fp2> = (0..levels + 1).map(|_| random(rng)).collect();
        let splits: Vec<Fp2> = group
            .split
            .iter()
            .map(|v| {
                let polynomial = &grouped.polynomials()[v.polynomial];
                let mut outside = v.coordinates.iter();
                let point: Vec<Fp2> = (0..polynomial.variables)
                    .map(
                        |var| match polynomial.group_variables.iter().position(|&s| s == var) {
                            Some(j) => e(u64::from(v.bits >> j & 1)),
                            None => p[*outside.next().unwrap() as usize],
                        },
                    )
                    .collect();
                assert!(outside.next().is_none());
                assert!(v.bits < 1 << polynomial.group_variables.len());
                mle(&tensors[&(v.node, v.column)], &point)
            })
            .collect();
        for instance in &group.instances {
            let mut tables = vec![];
            for block in &instance.blocks {
                let request = &sites[block.site].layers[block.layer as usize].lookups[block.lookup];
                let full = |r: u64| match request.token {
                    Some(axis) => {
                        let low = r & ((1 << axis.low) - 1);
                        low | u64::from(group.tokens) << axis.low
                            | (r >> axis.low) << (axis.low + g_t)
                    }
                    None => {
                        assert_eq!(group.tokens, 0);
                        r
                    }
                };
                let k = block.variables as usize;
                let keys: Vec<Fp2> = (0..1u64 << k)
                    .map(|r| at(&request.key, block.node, full(r)))
                    .collect();
                assert_eq!(mle(&keys, &p[..k]), evaluate(&block.key, &splits));
                if let Some(form) = &request.value {
                    let values: Vec<Fp2> = (0..1u64 << k)
                        .map(|r| at(form, block.node, full(r)))
                        .collect();
                    assert_eq!(
                        mle(&values, &p[..k]),
                        evaluate(block.value.as_ref().unwrap(), &splits)
                    );
                }
                tables.push(keys);
            }
            let padding = random(rng);
            let mut column = vec![padding; 1 << instance.height];
            for (block, keys) in instance.blocks.iter().zip(&tables) {
                let o = block.offset as usize;
                column[o..o + keys.len()].copy_from_slice(keys);
            }
            assert_eq!(
                mle(&column, &p[..instance.height as usize]),
                column_claim(
                    instance,
                    &p,
                    |b| mle(&tables[b], &p[..instance.blocks[b].variables as usize]),
                    padding
                )
            );
        }
    }
}

/// One layer whose key reads `X` with its token variables swapped, `X[c, t_1, t_0]`-style: its
/// variable 3 takes the token index's bit 1 and its variable 4 bit 0.
fn permuted() -> Vec<Site> {
    vec![site(&[2], || {
        vec![lookup(
            1,
            5,
            Some((3, 2)),
            affine(3, &[(5, Column::Input(0), &[0, 1, 2, 4, 3])]),
            Some(affine(0, &[(1, Column::Output(0), &[0, 1, 2, 3, 4])])),
        )]
    })]
}

#[test]
fn a_group_block_equals_its_affine_form_at_the_claims_derived_for_it() {
    let mut rng = SplitMix64::new(0x0054_5917);
    for g_t in 0..3 {
        let extent = Extent {
            token_bits: 2,
            layers: 1,
        };
        let grouped = Grouped::new(&permuted(), &[], extent, groups(g_t, 0)).unwrap();
        split_identity(&permuted(), &grouped, &mut rng);
        if g_t == 2 {
            // Token subset 1 sets bit 0: `X`'s variable 4, the second of `S`.
            assert_eq!(grouped.polynomials()[0].group_variables, [3, 4]);
            assert_eq!(grouped.group(1).unwrap().split[0].bits, 0b10);
        }
    }
    for (g_t, g_l) in [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)] {
        let extent = Extent {
            token_bits: 2,
            layers: 2,
        };
        let sites = four_by_eight();
        let grouped = Grouped::new(&sites, &[], extent, groups(g_t, g_l)).unwrap();
        split_identity(&sites, &grouped, &mut rng);
        if g_l == 0 {
            let sites = tokenless();
            let extent = Extent {
                token_bits: 2,
                layers: 1,
            };
            let grouped = Grouped::new(&sites, &[], extent, groups(g_t, 0)).unwrap();
            split_identity(&sites, &grouped, &mut rng);
        }
    }
    for (name, layers, n, g_t, g_l) in [
        ("tiny-2-servable", Some(3), 5, 1, 1),
        ("tiny-2-servable", Some(3), 5, 2, 2),
        ("tiny-2-affine", None, 4, 2, 0),
        ("tiny-2-minimal", None, 7, 3, 1),
    ] {
        let (graph, policy) = model(name, layers, n);
        let sites = model_sites(&graph, policy).unwrap();
        let grouped = Grouped::derive(&graph, policy, groups(g_t, g_l)).unwrap();
        split_identity(&sites, &grouped, &mut rng);
    }
}

#[test]
fn every_polynomial_is_claimed_once_by_exactly_its_readers_in_claim_order() {
    let mut cases: Vec<(Grouped, usize)> = vec![];
    for (name, layers, n, g_t, g_l) in [
        ("tiny-2-servable", Some(3), 5, 1, 2),
        ("tiny-4-servable", None, 6, 2, 1),
        ("tiny-2-minimal", None, 1, 0, 1),
    ] {
        let (graph, policy) = model(name, layers, n);
        let grouped = Grouped::derive(&graph, policy, groups(g_t, g_l)).unwrap();
        let forests = ModelSchedule::derive(&graph, policy)
            .unwrap()
            .forests()
            .count();
        cases.push((grouped, forests));
    }
    let extent = Extent {
        token_bits: 2,
        layers: 1,
    };
    cases.push((
        Grouped::new(&tokenless(), &[], extent, groups(2, 0)).unwrap(),
        0,
    ));
    for (grouped, forests) in cases {
        let polynomials = grouped.polynomials();
        assert_eq!(polynomials.len(), grouped.values() + forests);
        let mut claimed = vec![vec![]; polynomials.len()];
        for h in 0..grouped.count() {
            let group = grouped.group(h).unwrap();
            // The stage's split order, restricted to the tensors the group's blocks read: its
            // polynomials ascending, and exactly the ones its forms name.
            let named: BTreeSet<usize> = group
                .instances
                .iter()
                .flat_map(|i| &i.blocks)
                .flat_map(|b| {
                    b.key
                        .terms
                        .iter()
                        .chain(b.value.iter().flat_map(|v| &v.terms))
                })
                .map(|&(_, at)| group.split[at].polynomial)
                .collect();
            let sent: Vec<usize> = group.split.iter().map(|v| v.polynomial).collect();
            assert_eq!(sent, named.into_iter().collect::<Vec<_>>(), "group {h}");
            let mut seen = BTreeSet::new();
            for value in &group.split {
                assert!(seen.insert(value.polynomial), "one claim per polynomial");
                let p = &polynomials[value.polynomial];
                assert_eq!(
                    p.source,
                    Source::Split {
                        node: value.node,
                        column: value.column
                    }
                );
                assert_eq!(
                    value.coordinates.len() + p.group_variables.len(),
                    p.variables as usize
                );
                claimed[value.polynomial].push(h);
            }
            for share in &group.forests {
                assert!(seen.insert(share.polynomial));
                let p = &polynomials[share.polynomial];
                assert_eq!(share.polynomial, grouped.values() + share.forest);
                assert_eq!(
                    share.variables().unwrap() + p.group_variables.len() as u32,
                    p.variables
                );
                claimed[share.polynomial].push(h);
            }
        }
        for (i, p) in polynomials.iter().enumerate() {
            assert_eq!(
                claimed[i],
                p.readers.clone().collect::<Vec<_>>(),
                "polynomial {i}"
            );
        }
    }
}

#[test]
fn forest_shares_split_queries_then_layers_and_keep_the_forests_public_roots() {
    let mut rng = SplitMix64::new(0x0054_f0e5);
    let (graph, policy) = model("tiny-2-servable", Some(3), 5);
    let schedule = ModelSchedule::derive(&graph, policy).unwrap();
    let shapes: Vec<ForestShape> = schedule.forests().collect();
    // Five keys, five queries, four heads, three layers.
    assert_eq!(shapes.len(), 1);
    let shape = shapes[0];
    assert_eq!(
        (shape.height, shape.queries, shape.heads, shape.layers),
        (3, 5, 4, 3)
    );
    for (g_t, g_l) in [(1u32, 2u32), (2, 1), (3, 0), (0, 2), (3, 2)] {
        let grouped = Grouped::derive(&graph, policy, groups(g_t as u8, g_l as u8)).unwrap();
        let p = &grouped.polynomials()[grouped.values()];
        assert_eq!(
            p.source,
            Source::Forest {
                forest: 0,
                tree: shape.tree
            }
        );
        assert_eq!(p.variables, 3 + 3 + 2 + 2);
        let s: Vec<u32> = (3..3 + g_t).chain(8..8 + g_l).collect();
        assert_eq!(p.group_variables, s);
        assert_eq!(p.readers, 0..grouped.count());
        for h in 0..grouped.count() {
            let share = grouped.group(h).unwrap().forests[0];
            assert_eq!(
                (share.height, share.trees_log, share.bits),
                (3, 7 - g_t - g_l, h)
            );
            let (s, o) = (u64::from(h % (1 << g_t)), u64::from(h >> g_t));
            let point: Vec<Fp2> = (0..share.trees_log).map(|_| random(&mut rng)).collect();
            assert_eq!(share.root_point(&point[1..]), None);
            let full = share.root_point(&point).unwrap();
            // The share's root table, tree by tree: its query index's low bits are `s`, its
            // layer index's `o`; a real query, head and layer has root 0, any padding 1.
            let q_high = 3 - g_t;
            let expected = (0..1u64 << share.trees_log).fold(Fp2::ZERO, |sum, v| {
                let query = (v & ((1 << q_high) - 1)) << g_t | s;
                let head = (v >> q_high) & 3;
                let layer = (v >> (q_high + 2)) << g_l | o;
                let root = if query < 5 && head < 4 && layer < 3 {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                };
                sum + eq(&point, v) * root
            });
            assert_eq!(schedule.forest_root(share.tree, &full), Some(expected));
        }
    }
}

#[test]
fn the_grouping_bounds_are_checked_first_and_in_rfc_order() {
    let extent = Extent {
        token_bits: 2,
        layers: 2,
    };
    let new = |g_t, g_l| Grouped::new(&four_by_eight(), &[], extent, groups(g_t, g_l)).map(|_| ());
    let bits = |token_bits, layer_bits| Error::Bits {
        token_bits,
        layer_bits,
    };
    assert_eq!(new(8, 7), Err(bits(8, 7)));
    assert_eq!(new(15, 0), Err(bits(15, 0)), "the decoding bound first");
    assert_eq!(new(0, 15), Err(bits(0, 15)));
    let tokens = Error::TokenBits {
        token_bits: 3,
        axis_bits: 2,
    };
    assert_eq!(new(3, 0), Err(tokens.clone()));
    assert_eq!(
        new(3, 2),
        Err(tokens.clone()),
        "the token bound before the layers'"
    );
    let layers = Error::LayerBits {
        layer_bits: 2,
        axis_bits: 1,
    };
    assert_eq!(new(2, 2), Err(layers.clone()));
    assert_eq!(new(2, 1), Ok(()));
    for error in [bits(8, 7), tokens, layers] {
        assert!(error.is_rejection());
    }
    assert!(!Error::Internal.is_rejection());
    // On a model: four tokens give `t = 2`, two layers `n_ℓ = 1`; one token gives `t = 0`.
    let (graph, policy) = model("tiny-2-minimal", None, 4);
    let derive = |g_t, g_l| Grouped::derive(&graph, policy, groups(g_t, g_l)).map(|_| ());
    assert_eq!(derive(7, 8), Err(bits(7, 8)));
    assert_eq!(
        derive(3, 0),
        Err(Error::TokenBits {
            token_bits: 3,
            axis_bits: 2
        })
    );
    assert_eq!(
        derive(0, 2),
        Err(Error::LayerBits {
            layer_bits: 2,
            axis_bits: 1
        })
    );
    assert_eq!(derive(2, 1), Ok(()));
    let (graph, policy) = model("tiny-2-minimal", None, 1);
    assert_eq!(
        Grouped::derive(&graph, policy, groups(1, 0)).map(|_| ()),
        Err(Error::TokenBits {
            token_bits: 1,
            axis_bits: 0
        })
    );
}

#[test]
fn lookups_that_would_give_one_tensor_two_layouts_are_refused_at_their_node() {
    let extent = Extent {
        token_bits: 2,
        layers: 1,
    };
    let full: &[u32] = &[0, 1, 2, 3, 4];
    let one = |lookups: Vec<Lookup>| {
        vec![Site {
            layers: vec![Layer {
                node: NodeId(7),
                lookups,
            }],
        }]
    };
    let new = |sites: &[Site], g_t| Grouped::new(sites, &[], extent, groups(g_t, 0)).map(|_| ());
    // `X` read over the token axis by a block that has one, and over the same variables by a
    // block that has none: their group variables differ.
    let both = one(vec![
        lookup(
            1,
            5,
            Some((3, 2)),
            affine(0, &[(1, Column::Input(0), full)]),
            None,
        ),
        lookup(1, 5, None, affine(0, &[(1, Column::Input(0), full)]), None),
    ]);
    assert_eq!(new(&both, 1), Err(Error::Agreement { node: NodeId(7) }));
    assert_eq!(new(&both, 2), Err(Error::Agreement { node: NodeId(7) }));
    // With no token bits no variable is fixed, and the two agree.
    assert_eq!(new(&both, 0), Ok(()));
    // A term over part of its block's token axis, at any grouping.
    let partial = one(vec![lookup(
        1,
        5,
        Some((3, 2)),
        affine(0, &[(1, Column::Input(0), &[0, 1, 2, 3])]),
        None,
    )]);
    for g_t in 0..3 {
        assert_eq!(
            new(&partial, g_t),
            Err(Error::Agreement { node: NodeId(7) })
        );
    }
    // A class read over the channels alone agrees with a block without a token axis.
    let classes = one(vec![
        lookup(
            1,
            5,
            Some((3, 2)),
            affine(0, &[(1, Column::Input(1), &[0, 1, 2])]),
            None,
        ),
        lookup(
            1,
            3,
            None,
            affine(0, &[(1, Column::Input(1), &[0, 1, 2])]),
            None,
        ),
    ]);
    assert_eq!(new(&classes, 2), Ok(()));
}

#[test]
fn own_metadata_that_cannot_be_laid_out_is_internal() {
    let extent = Extent {
        token_bits: 2,
        layers: 2,
    };
    let full: &[u32] = &[0, 1, 2, 3, 4];
    let new = |sites: &[Site], forests: &[ForestShape]| {
        Grouped::new(sites, forests, extent, groups(1, 1)).map(|_| ())
    };
    let with = |lookups: Vec<Lookup>| vec![site(&[3], || lookups.clone())];
    let good = || {
        lookup(
            1,
            5,
            Some((3, 2)),
            affine(0, &[(1, Column::Input(0), full)]),
            None,
        )
    };
    assert_eq!(new(&with(vec![good()]), &[]), Ok(()));
    let internal = Err(Error::Internal);
    // A site of neither one entry nor one per layer, and one of none.
    let three = Extent {
        token_bits: 2,
        layers: 3,
    };
    assert_eq!(
        Grouped::new(&four_by_eight(), &[], three, groups(1, 1)).map(|_| ()),
        internal
    );
    assert_eq!(new(&[Site { layers: vec![] }], &[]), internal);
    // Token axes not of `t` variables, or outside their domain.
    let token = |token| {
        lookup(
            1,
            5,
            Some(token),
            affine(0, &[(1, Column::Input(0), full)]),
            None,
        )
    };
    assert_eq!(new(&with(vec![token((3, 1))]), &[]), internal);
    assert_eq!(new(&with(vec![token((4, 2))]), &[]), internal);
    // Reads outside their domain, repeated coordinates, and two coordinate sets for one tensor.
    let reading = |coordinates: &[u32]| {
        lookup(
            1,
            5,
            Some((3, 2)),
            affine(0, &[(1, Column::Input(0), coordinates)]),
            None,
        )
    };
    assert_eq!(new(&with(vec![reading(&[0, 1, 2, 3, 5])]), &[]), internal);
    assert_eq!(new(&with(vec![reading(&[0, 1, 1, 3, 4])]), &[]), internal);
    assert_eq!(
        new(&with(vec![reading(full), reading(&[1, 0, 2, 3, 4])]), &[]),
        internal
    );
    // A domain past 63 variables; a table read with and without values.
    assert_eq!(
        new(&with(vec![lookup(1, 64, None, affine(0, &[]), None)]), &[]),
        internal
    );
    let valued = lookup(1, 5, Some((3, 2)), affine(0, &[]), Some(affine(0, &[])));
    assert_eq!(new(&with(vec![good(), valued]), &[]), internal);
    // Forests: no height, queries other than the tokens, layers neither 1 nor the template's.
    let forest = |height, queries, layers| ForestShape {
        tree: 0,
        height,
        queries,
        heads: 4,
        layers,
    };
    assert_eq!(new(&with(vec![good()]), &[forest(2, 4, 2)]), Ok(()));
    assert_eq!(new(&with(vec![good()]), &[forest(2, 4, 1)]), Ok(()));
    assert_eq!(new(&with(vec![good()]), &[forest(0, 4, 2)]), internal);
    assert_eq!(new(&with(vec![good()]), &[forest(2, 8, 2)]), internal);
    assert_eq!(new(&with(vec![good()]), &[forest(2, 4, 3)]), internal);
    // A group outside `0..G`.
    let grouped = Grouped::new(&with(vec![good()]), &[], extent, groups(1, 1)).unwrap();
    assert_eq!(grouped.group(4), Err(Error::Internal));
    assert!(grouped.group(3).is_ok());
}

#[test]
fn a_forest_outside_the_template_belongs_to_layer_subset_zero() {
    let extent = Extent {
        token_bits: 2,
        layers: 2,
    };
    let forests = [
        ForestShape {
            tree: 4,
            height: 2,
            queries: 4,
            heads: 3,
            layers: 2,
        },
        ForestShape {
            tree: 5,
            height: 1,
            queries: 3,
            heads: 1,
            layers: 1,
        },
    ];
    let grouped = Grouped::new(&four_by_eight(), &forests, extent, groups(1, 1)).unwrap();
    let base = grouped.values();
    let stacked = &grouped.polynomials()[base];
    assert_eq!(
        (
            stacked.variables,
            stacked.group_variables.clone(),
            stacked.readers.clone()
        ),
        (2 + 2 + 2 + 1, vec![2, 6], 0..4)
    );
    let outside = &grouped.polynomials()[base + 1];
    assert_eq!(
        (
            outside.variables,
            outside.group_variables.clone(),
            outside.readers.clone()
        ),
        (1 + 2, vec![1], 0..2)
    );
    for h in 0..4 {
        let shares = grouped.group(h).unwrap().forests;
        let expected: Vec<(usize, u32, u32)> = if h < 2 {
            vec![(4, 3, h), (5, 1, h)]
        } else {
            vec![(4, 3, h)]
        };
        assert_eq!(
            shares
                .iter()
                .map(|s| (s.tree, s.trees_log, s.bits))
                .collect::<Vec<_>>(),
            expected
        );
        // The outside forest's root point: query bit 0 fixed to `s`, no layer variable.
        if let Some(share) = shares.get(1) {
            let point = [e(9)];
            assert_eq!(
                share.root_point(&point),
                Some(vec![e(u64::from(h & 1)), e(9)])
            );
        }
    }
}
