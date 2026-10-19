// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every location in the expanded, pre-implementation public-model oracle.
mod claim_support;
use claim_support::{apply_variant, claims_of, integer, items, list, member, number, text};
use tv_core::advice::{
    Auxiliary, Comparison, ComparisonRole, Config, Domain, Layout, Source, TileKey,
};
use tv_core::{lowering, selection, validation};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn basis(index: u64, point: &[Fp2]) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |value, (bit, &r)| {
        value
            * if index >> bit & 1 == 0 {
                Fp2::ONE - r
            } else {
                r
            }
    })
}

fn non_boolean(arity: usize) -> Vec<Fp2> {
    (0..arity)
        .map(|i| Fp2::new(Fp::reduce(17 + 13 * i as u64), Fp::reduce(5 + 7 * i as u64)))
        .collect()
}

fn repeat_layer_points<T: Clone>(
    points: &mut Vec<(tv_core::claims::QuantPoint, T)>,
    layers: u32,
    copies: u32,
) {
    let original = points.clone();
    for copy in 1..copies {
        for (point, value) in &original {
            if let tv_core::claims::QuantPoint::Layer { layer, site } = *point {
                points.push((
                    tv_core::claims::QuantPoint::Layer {
                        layer: layer + copy * layers,
                        site,
                    },
                    value.clone(),
                ));
            }
        }
    }
    points.sort_by_key(|(p, _)| *p);
}

fn auxiliary(role: Auxiliary) -> (String, u32) {
    match role {
        Auxiliary::RequantLow => ("requant.low".into(), 0),
        Auxiliary::RequantMiddle => ("requant.middle".into(), 0),
        Auxiliary::RequantSaturation => ("requant.saturation".into(), 0),
        Auxiliary::SoftmaxMaximum => ("softmax.maximum".into(), 0),
        Auxiliary::SoftmaxA(i) => ("softmax.a".into(), i),
        Auxiliary::SoftmaxB(i) => ("softmax.b".into(), i),
        Auxiliary::RootInput(i) => ("root.input".into(), i),
        Auxiliary::RootLow(i) => ("root.low".into(), i),
        Auxiliary::RootComparison(comparison, role, i) => {
            let comparison = match comparison {
                Comparison::Lower => "lower",
                Comparison::Upper => "upper",
            };
            let role = match role {
                ComparisonRole::LeftDigit => "left.digit",
                ComparisonRole::LeftCarry => "left.carry",
                ComparisonRole::RightDigit => "right.digit",
                ComparisonRole::RightCarry => "right.carry",
                ComparisonRole::Difference => "difference",
                ComparisonRole::Borrow => "borrow",
            };
            (format!("root.{comparison}.{role}"), i)
        }
    }
}

fn source(found: Source, expected: &Json) {
    let expected = items(expected);
    match found {
        Source::Tensor(id) => {
            assert_eq!(expected[0].as_str(), Some("tensor"));
            assert_eq!(id.0, number(&expected[1]));
            assert_eq!(expected.len(), 2);
        }
        Source::Multiplicity(id) => {
            assert_eq!(expected[0].as_str(), Some("multiplicity"));
            assert_eq!(id.0, number(&expected[1]));
            assert_eq!(expected.len(), 2);
        }
        Source::Auxiliary(node, role) => {
            assert_eq!(expected[0].as_str(), Some("aux"));
            assert_eq!(node.0, number(&expected[1]));
            let (name, index) = auxiliary(role);
            assert_eq!(expected[2].as_str(), Some(name.as_str()));
            assert_eq!(index, number(&expected[3]));
            assert_eq!(expected.len(), 4);
        }
    }
}

fn key(found: TileKey, expected: &Json) {
    let expected = items(expected);
    match found {
        TileKey::Tokens => assert_eq!(expected[0].as_str(), Some("tokens")),
        TileKey::Node(id) => {
            assert_eq!(expected[0].as_str(), Some("node"));
            assert_eq!(id.0, number(&expected[1]));
        }
        TileKey::Table(id) => {
            assert_eq!(expected[0].as_str(), Some("table"));
            assert_eq!(id.0, number(&expected[1]));
        }
        TileKey::Attention(id, kv, qb) => {
            assert_eq!(expected[0].as_str(), Some("attention"));
            assert_eq!(id.0, number(&expected[1]));
            assert_eq!(kv, number(&expected[2]));
            assert_eq!(qb, number(&expected[3]));
        }
    }
}

#[test]
fn all_model_locations_padding_gaps_and_segments_match_the_independent_oracle() {
    let vectors = read_vectors("advice/layout.json");
    let mut locations = 0;
    let mut cases = 0;
    for case in list(&vectors, "cases") {
        let mut claims = claims_of(text(case, "file"));
        if !matches!(member(case, "variant"), Json::Null) {
            apply_variant(member(case, "variant"), &mut claims);
        }
        let public = validation::check_public(&claims).unwrap();
        // The validator stores the max_context graph; specialize its time axes.
        let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
        let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
        let config = Config {
            log_k: number(member(case, "log2_k")),
            segments: number(member(case, "segments")),
            query_block: number(member(case, "query_block")),
            heads: claims.arch.n_heads,
            kv_heads: claims.arch.n_kv_heads,
            limb_bits: claims.gadget_policy.limb_bits,
        };
        let layout = Layout::derive(&graph, &selected, config, number(member(case, "n"))).unwrap();
        assert_eq!(
            layout.rows(),
            number(member(case, "rows")),
            "{}",
            text(case, "model")
        );
        assert_eq!(layout.nu(), number(member(case, "nu")));
        let expected = list(case, "tiles");
        assert_eq!(layout.tiles().count(), expected.len());
        for (tile, expected) in layout.tiles().zip(expected) {
            key(tile.key(), member(expected, "key"));
            assert_eq!(tile.segment(), number(member(expected, "segment")));
            let range = list(expected, "rows");
            assert_eq!(tile.rows(), (number(&range[0]), number(&range[1])));
            assert_eq!(tile.columns() as usize, list(expected, "values").len());
            for (i, expected) in list(expected, "values").iter().enumerate() {
                let [identity, shape, permutation, base, rows, zero] = items(expected) else {
                    panic!("location")
                };
                let found = tile
                    .location(i as u32)
                    .expect("every allocated source has a location");
                source(found.source, identity);
                assert_eq!(
                    found.domain.shape(),
                    items(shape).iter().map(number).collect::<Vec<u64>>()
                );
                assert_eq!(
                    found.domain.permutation(),
                    items(permutation).iter().map(number).collect::<Vec<u8>>()
                );
                assert_eq!(
                    (found.base, found.domain.rows(), found.zero_at),
                    (number(base), number(rows), number(zero))
                );
                // Every ordinary source's point image preserves a Boolean
                // basis polynomial at a non-Boolean point; values are synthetic.
                if !matches!(tile.key(), TileKey::Attention(..)) {
                    let permutation = items(permutation).iter().map(number).collect::<Vec<u32>>();
                    let point = non_boolean(permutation.len());
                    let image = layout
                        .claim(found.source, &point)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    assert_eq!(image.len(), 1);
                    assert_eq!(image[0].source, found.source);
                    assert_eq!(image[0].tile, tile.key());
                    assert_eq!(image[0].weight, Fp2::ONE);
                    for natural in [0u64, 1, 0x5555_5555_5555_5555, u64::MAX] {
                        let local = permutation
                            .iter()
                            .enumerate()
                            .fold(0, |address, (new, old)| {
                                address | ((natural >> old & 1) << new)
                            });
                        let address = found.base << config.log_k | local;
                        assert_eq!(basis(address, &image[0].point), basis(natural, &point));
                    }
                }
                locations += 1;
            }
            assert!(tile.location(tile.columns()).is_none());
        }
        for (i, expected) in list(case, "segment_rows").iter().enumerate() {
            let range = items(expected);
            assert_eq!(
                layout.segment(i as u32),
                Some((number(&range[0]), number(&range[1])))
            );
        }
        assert!(layout.segment(config.segments).is_none());
        let zeros = list(case, "zero_rows")
            .iter()
            .map(|r| (number::<u64>(&items(r)[0]), number::<u64>(&items(r)[1])))
            .collect::<Vec<_>>();
        for row in 0..layout.rows() {
            assert_eq!(
                layout.is_zero_row(row),
                Some(zeros.iter().any(|&(a, b)| a <= row && row < b)),
                "{} n={} k={} row={row}",
                text(case, "model"),
                integer(member(case, "n")),
                config.log_k
            );
        }
        assert_eq!(layout.is_zero_row(layout.rows()), None);
        cases += 1;
    }
    assert_eq!((cases, locations), (54, 65_620));
}

#[test]
fn whole_attention_claims_reconstruct_at_non_boolean_points_including_padded_kv() {
    for kv_heads in [1, 2, 3] {
        for (query_block, log_k) in [(1, 3), (2, 3), (8, 4), (512, 10)] {
            let mut claims = claims_of("tiny-2-servable.d");
            claims.arch.n_heads = 2 * kv_heads;
            claims.arch.n_kv_heads = kv_heads;
            claims.arch.d_head = 2;
            claims.arch.d_model = 4 * kv_heads;
            claims.quant.rsqrt_kappa = 3;
            let public = validation::check_public(&claims).unwrap();
            let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
            let selection = selection::derive(&graph, claims.gadget_policy).unwrap();
            let config = Config {
                query_block,
                log_k,
                ..Config::new(&claims.arch, claims.gadget_policy, log_k, 3)
            };
            let layout = Layout::derive(&graph, &selection, config, 5).unwrap();
            // Enumerate actual causal cells into their RFC-0034 rows. This
            // catches partial-block zero suffixes even when geometry vectors'
            // larger row widths would put the whole score in one row.
            for share in layout
                .tiles()
                .filter(|t| matches!(t.key(), TileKey::Attention(..)))
            {
                let TileKey::Attention(_, _, qb) = share.key() else {
                    unreachable!()
                };
                let score = share.location(0).unwrap();
                let q = u64::from(query_block.min(8));
                let w = 3u32.min(u32::from(log_k) - 1 - q.ilog2());
                let mut used = vec![false; score.domain.rows() as usize];
                for h in 0..2u64 {
                    for local_query in 0..q {
                        let query = u64::from(qb) * q + local_query;
                        for key in 0..5u64 {
                            if query < 5 && key <= query {
                                let local = (key & ((1 << w) - 1))
                                    | (local_query << w)
                                    | (h << (w + q.ilog2()))
                                    | ((key >> w) << (w + q.ilog2() + 1));
                                used[(local >> log_k) as usize] = true;
                            }
                        }
                    }
                }
                for (row, used) in used.into_iter().enumerate() {
                    assert_eq!(
                        layout.is_zero_row(score.base + row as u64),
                        Some(!used),
                        "Q={q} k={log_k} qb={qb} score row={row}"
                    );
                }
            }
            let tiles = layout
                .tiles()
                .filter(|t| matches!(t.key(), TileKey::Attention(_, 0, 0)))
                .collect::<Vec<_>>();
            for tile in tiles {
                let TileKey::Attention(node, _, _) = tile.key() else {
                    unreachable!()
                };
                for column in 0..tile.columns() {
                    let source = tile.location(column).unwrap().source;
                    let score = column == 0;
                    let heads_bits = (2 * kv_heads).next_power_of_two().ilog2();
                    let point = non_boolean((heads_bits + if score { 6 } else { 3 }) as usize);
                    let q = u64::from(query_block.min(8));
                    let w = 3u32.min(u32::from(log_k) - 1 - q.ilog2());
                    let mut sparse = Vec::new();
                    let mut expected = Fp2::ZERO;
                    for h in 0..u64::from(2 * kv_heads) {
                        for query in 0..5u64 {
                            for key in 0..if score { query + 1 } else { 1 } {
                                let value =
                                    Fp2::new(Fp::reduce(1 + 103 * h + 17 * query + key), Fp::ZERO);
                                let natural = if score {
                                    (h * 8 + query) * 8 + key
                                } else {
                                    h * 8 + query
                                };
                                expected += value * basis(natural, &point);
                                let share = layout
                                    .tile(TileKey::Attention(
                                        node,
                                        (h / 2) as u32,
                                        (query / q) as u32,
                                    ))
                                    .unwrap();
                                let location = share.location(column).unwrap();
                                // Independent RFC-0034 coordinate address, without
                                // consulting the implementation's permutation.
                                let local = if score {
                                    (key & ((1 << w) - 1))
                                        | ((query % q) << w)
                                        | ((h % 2) << (w + q.ilog2()))
                                        | ((key >> w) << (w + q.ilog2() + 1))
                                } else {
                                    (h % 2) * q + query % q
                                };
                                sparse.push(((location.base << log_k) | local, value));
                            }
                        }
                    }
                    let images = layout
                        .claim(source, &point)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    assert_eq!(images.len(), kv_heads as usize * (8 / q) as usize);
                    let reconstructed = images.iter().fold(Fp2::ZERO, |sum, image| {
                        assert_eq!(image.source, source);
                        let opening = sparse.iter().fold(Fp2::ZERO, |value, &(address, entry)| {
                            value + entry * basis(address, &image.point)
                        });
                        sum + image.weight * opening
                    });
                    assert_eq!(
                        reconstructed, expected,
                        "kv={kv_heads} Q={query_block} k={log_k} node={node:?} col={column}"
                    );
                }
            }
            let missing = Source::Auxiliary(lowering::NodeId(u32::MAX), Auxiliary::RequantLow);
            assert!(matches!(
                layout.claim(missing, &[]),
                Err(tv_core::advice::PointError::Source)
            ));
            let tokens = layout
                .tile(TileKey::Tokens)
                .unwrap()
                .location(0)
                .unwrap()
                .source;
            assert!(matches!(
                layout.claim(tokens, &[]),
                Err(tv_core::advice::PointError::Arity)
            ));
            assert!(
                layout
                    .tile(TileKey::Attention(lowering::NodeId(0), kv_heads, 0))
                    .is_none()
            );
        }
    }
}

#[test]
fn mlp_padding_skips_48_rows_but_inner_padded_axes_remain_committed() {
    let domain = Domain::new(&[16_384, 53_248], 22).unwrap();
    assert_eq!((domain.rows(), domain.zero_at()), (256, 208));
    assert_eq!(&domain.permutation()[..14], &(16..30).collect::<Vec<_>>());
    let two_padded = Domain::new(&[3, 5], 1).unwrap();
    assert_eq!((two_padded.rows(), two_padded.zero_at()), (16, 12));
}

#[test]
fn row_width_allows_the_encoding_subgroup_boundary_and_rejects_wider_rows() {
    let claims = claims_of("tiny-2-servable.d");
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, 1).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    let mut config = Config::new(&claims.arch, claims.gadget_policy, 31, 1);
    assert!(Layout::derive(&graph, &selected, config, 1).is_ok());
    for bits in [32, 63, u8::MAX] {
        config.log_k = bits;
        assert_eq!(
            Layout::derive(&graph, &selected, config, 1).unwrap_err(),
            validation::ClaimsError::StatementMismatch {
                field: validation::StatementField::Layout
            }
        );
    }
}

#[test]
fn every_root_width_allocates_each_real_role_and_no_virtual_top() {
    let vectors = read_vectors("advice/layout.json");
    for schema in list(&vectors, "roots") {
        let width: u8 = number(member(schema, "width"));
        let mut claims = claims_of("tiny-2-servable.d");
        claims.arch.n_heads = 1;
        claims.arch.n_kv_heads = 1;
        claims.arch.d_head = 2;
        claims.arch.d_model = 2;
        claims.quant.rsqrt_kappa = width - 1;
        claims.quant.rms_eps_int = 1;
        let public = validation::check_public(&claims).unwrap();
        let graph = lowering::lower(&public, 1).unwrap();
        let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
        let layout = Layout::derive(
            &graph,
            &selected,
            Config::new(&claims.arch, claims.gadget_policy, 10, 1),
            1,
        )
        .unwrap();
        let mut roots = 0;
        for tile in layout.tiles() {
            let TileKey::Node(node) = tile.key() else {
                continue;
            };
            if !matches!(
                graph.node(node).unwrap().op(),
                lowering::Op::RsqrtRow { .. }
            ) {
                continue;
            }
            let output = graph.node(node).unwrap().outputs()[0];
            assert_eq!(
                selected.ranges[output.0 as usize].bounds().unwrap().1,
                (1i128 << width) - 1
            );
            assert_eq!(tile.location(0).unwrap().source, Source::Tensor(output));
            assert_eq!(tile.columns() as usize, list(schema, "roles").len() + 1);
            for (i, expected) in list(schema, "roles").iter().enumerate() {
                let location = tile.location(i as u32 + 1).unwrap();
                let Source::Auxiliary(found, role) = location.source else {
                    panic!("auxiliary role")
                };
                let (name, index) = auxiliary(role);
                assert_eq!(found, node);
                assert_eq!(
                    name,
                    items(expected)[0].as_str().unwrap(),
                    "width={width} role={i}"
                );
                assert_eq!(index, number(&items(expected)[1]));
                assert_eq!(location.domain.shape(), [1]);
                assert_eq!(location.domain.rows(), 1);
                assert_eq!(location.zero_at, 1);
            }
            roots += 1;
        }
        assert_eq!(roots, 2 * claims.arch.n_layers + 1);
    }
}

#[test]
fn max_context_size_is_checked_after_tables_even_when_one_token_fits() {
    let mut claims = claim_support::twin("tiny-2-minimal");
    claims.arch.max_context = 1 << 17;
    claims.arch.d_ff = u32::MAX;
    claims.arch.n_heads = 1;
    claims.arch.n_kv_heads = 1;
    claims.arch.d_head = 2;
    claims.arch.d_model = 2;
    claims.quant.residual_bits = 8;
    claims.quant.residual_exponent = claims
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tv_core::claims::QuantPoint::Embed)
        .unwrap()
        .1;
    let layers = claims.arch.n_layers;
    repeat_layer_points(&mut claims.quant.act_exponents, layers, 16);
    repeat_layer_points(&mut claims.quant.shift_class_sets, layers, 16);
    repeat_layer_points(&mut claims.quant.affine_sites, layers, 16);
    let scales = claims.quant.score_scales.clone();
    let silu = claims.silu_in_exponents.clone();
    for copy in 1..16 {
        claims
            .quant
            .score_scales
            .extend(scales.iter().map(|&(l, v)| (l + copy * layers, v)));
        claims
            .silu_in_exponents
            .extend(silu.iter().map(|&(l, v)| (l + copy * layers, v)));
    }
    claims.arch.n_layers *= 16;
    for layer in 0..claims.arch.n_layers {
        claim_support::make_pot(
            &mut claims,
            tv_core::claims::QuantPoint::Layer {
                layer,
                site: tv_core::claims::Site::MlpResidual,
            },
            &[24, 25, 26, 27],
        );
    }
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    let config = Config::new(&claims.arch, claims.gadget_policy, 22, 8);
    assert!(Layout::derive(&graph, &selected, config, 1).is_ok());
    let rejection = validation::ClaimsError::StatementMismatch {
        field: validation::StatementField::Layout,
    };
    assert_eq!(
        Layout::derive(&graph, &selected, config, claims.arch.max_context).unwrap_err(),
        rejection
    );
    claims.tables = selected
        .tables
        .iter()
        .map(|t| {
            tv_tables::Table::generate(t.id, t.kind.clone())
                .unwrap()
                .spec()
                .clone()
        })
        .collect();
    assert_eq!(validation::validate(&claims).unwrap_err(), rejection);
    claims.tables[0].digest[0] ^= 1;
    assert_eq!(
        validation::validate(&claims).unwrap_err(),
        validation::ClaimsError::StatementMismatch {
            field: validation::StatementField::Table {
                table: claims.tables[0].id
            }
        }
    );
}

#[test]
fn key_row_capacity_is_checked_at_max_context_after_the_complete_table_sequence() {
    let mut claims = claim_support::twin("tiny-2-minimal");
    claims.arch.max_context = 1 << 17;
    claims.arch.n_heads = 128;
    claims.arch.n_kv_heads = 1;
    claims.arch.d_head = 2;
    claims.arch.d_model = 256;
    claims.quant.rsqrt_kappa = 0;
    claims.quant.residual_bits = 8;
    claims.quant.residual_exponent = claims
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tv_core::claims::QuantPoint::Embed)
        .unwrap()
        .1;
    for layer in 0..claims.arch.n_layers {
        claim_support::make_pot(
            &mut claims,
            tv_core::claims::QuantPoint::Layer {
                layer,
                site: tv_core::claims::Site::MlpResidual,
            },
            &[24, 25, 26, 27],
        );
    }
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    claims.tables = selected
        .tables
        .iter()
        .map(|t| {
            tv_tables::Table::generate(t.id, t.kind.clone())
                .unwrap()
                .spec()
                .clone()
        })
        .collect();
    claims.gadget_policy.query_block = 1;
    validation::validate(&claims).unwrap();
    claims.gadget_policy.query_block = 1 << 17;
    let config = Config::new(&claims.arch, claims.gadget_policy, 22, 8);
    assert!(Layout::derive(&graph, &selected, config, 1).is_ok());
    assert_eq!(
        validation::validate(&claims).unwrap_err(),
        validation::ClaimsError::StatementMismatch {
            field: validation::StatementField::Layout
        }
    );
    // A complete, well-formed but noncanonical sequence fails step 5.3 first.
    let orphan = tv_tables::TableId(claims.tables.len() as u32);
    claims.tables.push(
        tv_tables::Table::generate(orphan, tv_tables::TableKind::Range { bits: 2 })
            .unwrap()
            .spec()
            .clone(),
    );
    assert_eq!(
        validation::validate(&claims).unwrap_err(),
        validation::ClaimsError::StatementMismatch {
            field: validation::StatementField::Table { table: orphan }
        }
    );
}
