// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bind every public model source against the pre-implementation advice oracle.
mod common;
use common::public_vectors::{edited, models, read_json, vectors};
use serde_json::{Value, json};
use tessera_codec::Ascii;
use tessera_compiler::{
    AdviceSource, AuxColumn, CompareColumn, Lowering, ModelAdviceLayout, PublicModel, Selection,
    Tile, analyze, claim_list, lower, select,
};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_ir::{
    AdviceLayoutParams, ClaimList, CommitmentLayout, GadgetPolicy, ParamSetId, WeightLayoutParams,
};

fn repeat_points<T: Clone>(points: &mut Vec<(tessera_ir::QuantPoint, T)>, layers: u32) {
    let original = points.clone();
    for copy in 1..16 {
        for (point, value) in &original {
            if let tessera_ir::QuantPoint::Layer { layer, site } = *point {
                points.push((
                    tessera_ir::QuantPoint::Layer {
                        layer: layer + copy * layers,
                        site,
                    },
                    value.clone(),
                ));
            }
        }
    }
    points.sort_by_key(|(point, _)| *point);
}

#[test]
fn emission_rejects_maximum_context_allocation_even_when_one_token_fits() {
    let models = models();
    let model = &models["tiny-2-minimal.d"];
    let mut arch = model.arch.clone();
    let mut quant = model.quant.public.clone();
    arch.n_heads = 1;
    arch.n_kv_heads = 1;
    arch.d_head = 2;
    arch.d_model = 2;
    arch.d_ff = 1 << 31;
    arch.max_context = 1 << 17;
    quant.residual_bits = 8;
    quant.residual_exponent = quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tessera_ir::QuantPoint::Embed)
        .unwrap()
        .1;
    for (point, shifts) in &mut quant.shift_class_sets {
        if matches!(
            point,
            tessera_ir::QuantPoint::Layer {
                site: tessera_ir::Site::MlpResidual,
                ..
            }
        ) {
            *shifts = vec![24, 25, 26, 27];
        }
    }
    let layers = arch.n_layers;
    repeat_points(&mut quant.act_exponents, layers);
    repeat_points(&mut quant.shift_class_sets, layers);
    repeat_points(&mut quant.affine_sites, layers);
    let scales = quant.score_scales.clone();
    let mut silu = model.quant.silu_in_exponents.clone();
    let original = silu.clone();
    for copy in 1..16 {
        quant.score_scales.extend(
            scales
                .iter()
                .map(|&(layer, value)| (layer + copy * layers, value)),
        );
        silu.extend(
            original
                .iter()
                .map(|&(layer, value)| (layer + copy * layers, value)),
        );
    }
    arch.n_layers *= 16;
    let public = PublicModel {
        arch: &arch,
        profile: model.profile,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let policy = GadgetPolicy::default();
    let one = lower(public, 1).unwrap();
    let selection = selected(&one, &policy);
    let raw = ClaimList {
        protocol_version: tessera_ir::PROTOCOL_VERSION,
        arch: arch.clone(),
        profile: model.profile,
        quant: quant.clone(),
        silu_in_exponents: silu.clone(),
        tables: selection.tables.clone(),
        gadget_policy: policy,
        layout: CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: 22,
                log2_inv_rate: 1,
                mask_coeffs: 200,
                row_block: 32,
                segments: 8,
            },
            weights: WeightLayoutParams {
                log2_k: 20,
                log2_inv_rate: 2,
                mask_coeffs: 65_536,
                blinding_rows: 4096,
            },
        },
        params: ParamSetId(Ascii::new("ni-100/1").unwrap()),
    };
    assert!(ModelAdviceLayout::new(&one, &selection, &raw).is_ok());
    let error =
        claim_list(public, &selection, &policy, raw.layout, raw.params.clone()).unwrap_err();
    assert_eq!(
        error,
        tessera_compiler::CompileError::Model {
            rule: "a layout allocates at most 2^32 rows (RFC-0047 'The maximum layout')"
        }
    );
    assert_eq!(error.code(), tessera_codec::error::ErrorCode::QuantArch);
    assert_eq!(tessera_compiler::plan(public, &raw).unwrap_err(), error);
    let maximum = lower(public, arch.max_context).unwrap();
    assert!(matches!(ModelAdviceLayout::new(&maximum,&selection,&raw),Err(e) if e==error));
}

#[test]
fn emission_checks_key_row_capacity_at_maximum_context() {
    let models = models();
    let model = &models["tiny-2-minimal.d"];
    let mut arch = model.arch.clone();
    let mut quant = model.quant.public.clone();
    arch.n_heads = 128;
    arch.n_kv_heads = 1;
    arch.d_head = 2;
    arch.d_model = 256;
    arch.max_context = 1 << 17;
    quant.rsqrt_kappa = 0;
    quant.residual_bits = 8;
    quant.residual_exponent = quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == tessera_ir::QuantPoint::Embed)
        .unwrap()
        .1;
    for (point, shifts) in &mut quant.shift_class_sets {
        if matches!(
            point,
            tessera_ir::QuantPoint::Layer {
                site: tessera_ir::Site::MlpResidual,
                ..
            }
        ) {
            *shifts = vec![24, 25, 26, 27];
        }
    }
    let public = PublicModel {
        arch: &arch,
        profile: model.profile,
        quant: &quant,
        silu_in_exponents: &model.quant.silu_in_exponents,
    };
    let mut policy = GadgetPolicy {
        query_block: 1,
        ..GadgetPolicy::default()
    };
    let one = lower(public, 1).unwrap();
    let selection = selected(&one, &policy);
    let control = claims(public, &selection, &policy, 22, 8);
    policy.query_block = 1 << 17;
    let error = claim_list(
        public,
        &selection,
        &policy,
        control.layout,
        control.params.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        tessera_compiler::CompileError::Model {
            rule: "a row holds one key against a tile's queries and heads: g·min(Q, N_max) ≤ k (RFC-0046)"
        }
    );
}
type Fp2 = GoldilocksExt2;

fn claims(
    public: PublicModel<'_>,
    selected: &Selection,
    policy: &GadgetPolicy,
    log_k: u8,
    segments: u32,
) -> ClaimList {
    claim_list(
        public,
        selected,
        policy,
        CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: log_k,
                log2_inv_rate: 1,
                mask_coeffs: 16,
                row_block: 4,
                segments,
            },
            weights: WeightLayoutParams {
                log2_k: 8,
                log2_inv_rate: 2,
                mask_coeffs: 64,
                blinding_rows: 32,
            },
        },
        ParamSetId(Ascii::new("test-small/1").unwrap()),
    )
    .unwrap()
}
fn selected(lowering: &Lowering, policy: &GadgetPolicy) -> Selection {
    select(
        &lowering.graph,
        &analyze(&lowering.graph, &lowering.tables).unwrap(),
        &lowering.tables,
        policy,
    )
    .unwrap()
}
fn auxiliary(column: AuxColumn) -> (String, u32) {
    match column {
        AuxColumn::Low => ("requant.low".into(), 0),
        AuxColumn::Middle => ("requant.middle".into(), 0),
        AuxColumn::Saturation => ("requant.saturation".into(), 0),
        AuxColumn::Maximum => ("softmax.maximum".into(), 0),
        AuxColumn::Slack { second, limb } => {
            (if second { "softmax.b" } else { "softmax.a" }.into(), limb)
        }
        AuxColumn::InputLimb(limb) => ("root.input".into(), limb),
        AuxColumn::RootByte(byte) => ("root.low".into(), byte),
        AuxColumn::Comparison {
            upper,
            column,
            limb,
        } => {
            let bracket = if upper { "upper" } else { "lower" };
            let role = match column {
                CompareColumn::LeftByte => "left.digit",
                CompareColumn::LeftCarry => "left.carry",
                CompareColumn::RightByte => "right.digit",
                CompareColumn::RightCarry => "right.carry",
                CompareColumn::Difference => "difference",
                CompareColumn::Borrow => "borrow",
            };
            (format!("root.{bracket}.{role}"), limb)
        }
    }
}
fn identity(source: AdviceSource) -> Value {
    match source {
        AdviceSource::Tensor(id) => json!(["tensor", id.0]),
        AdviceSource::Multiplicity(id) => json!(["multiplicity", id.0]),
        AdviceSource::GreedyMaximum(id) => json!(["greedy.maximum", id.0]),
        AdviceSource::Auxiliary(node, column) => {
            let (name, index) = auxiliary(column);
            json!(["aux", node.0, name, index])
        }
    }
}
fn point(arity: usize) -> Vec<Fp2> {
    (0..arity)
        .map(|i| {
            Fp2::new(
                Goldilocks::from_u64_reduced(17 + 13 * i as u64),
                Goldilocks::from_u64_reduced(5 + 7 * i as u64),
            )
        })
        .collect()
}
fn basis(address: u64, point: &[Fp2]) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |out, (bit, &r)| {
        out * if address >> bit & 1 == 0 {
            Fp2::ONE - r
        } else {
            r
        }
    })
}

#[test]
fn inconsistent_source_metadata_and_uncommitted_claims_are_refused() {
    let models = models();
    let model = &models["tiny-2-servable.d"];
    let public = PublicModel::of(model);
    let policy = GadgetPolicy::default();
    let lowering = lower(public, 5).unwrap();
    let selection = selected(&lowering, &policy);
    let claims = claims(public, &selection, &policy, 10, 1);
    let metadata = tessera_compiler::CompileError::Usage {
        rule: "advice metadata comes from the canonical lowering and gadget selection",
    };
    let mut changed = claims.clone();
    changed.tables.pop();
    assert_eq!(
        ModelAdviceLayout::new(&lowering, &selection, &changed).unwrap_err(),
        metadata
    );
    let mut changed = selection.clone();
    let root = changed
        .gadgets
        .iter()
        .find(|(_, g)| matches!(g, tessera_compiler::Gadget::RsqrtRow))
        .unwrap()
        .0;
    changed.gadgets.retain(|(n, _)| *n != root);
    assert_eq!(
        ModelAdviceLayout::new(&lowering, &changed, &claims).unwrap_err(),
        metadata
    );
    let mut changed = selection.clone();
    let range = changed.reads(root).unwrap()[0];
    let table = changed.tables.iter_mut().find(|t| t.id == range).unwrap();
    let tessera_tables::TableKind::Range { bits } = table.kind else {
        unreachable!()
    };
    let bits = if bits <= 8 { 32 } else { 1 };
    table.kind = tessera_tables::TableKind::Range { bits };
    table.log2_rows = bits;
    let mut changed_claims = claims.clone();
    changed_claims.tables = changed.tables.clone();
    assert_eq!(
        ModelAdviceLayout::new(&lowering, &changed, &changed_claims).unwrap_err(),
        metadata
    );
    let mut changed = lowering.clone();
    let one_hot = changed
        .graph
        .nodes
        .iter_mut()
        .find(|n| matches!(n.op, tessera_ir::Op::OneHot { .. }))
        .unwrap();
    one_hot.outputs.push(one_hot.outputs[0]);
    assert_eq!(
        ModelAdviceLayout::new(&changed, &selection, &claims).unwrap_err(),
        metadata
    );
    let mapped = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
    let public_source = AdviceSource::Tensor(lowering.positions);
    assert_eq!(
        mapped.claim::<Fp2>(public_source, &[]).err().unwrap(),
        tessera_compiler::CompileError::Usage {
            rule: "the advice claim names a committed tensor or gadget auxiliary"
        }
    );
    let token = AdviceSource::Tensor(lowering.tokens);
    assert_eq!(
        mapped.claim::<Fp2>(token, &[]).err().unwrap(),
        tessera_compiler::CompileError::Usage {
            rule: "the advice claim point has its source's natural variable count"
        }
    );
}

#[test]
fn every_model_source_location_point_zero_row_and_segment_matches_the_oracle() {
    let models = models();
    let oracle = read_json(&vectors().join("advice/layout.json"));
    let openings = read_json(&vectors().join("advice-opening/opening.json"));
    let mut locations = 0;
    for (case_index, case) in oracle["cases"].as_array().unwrap().iter().enumerate() {
        let model = &models[case["file"].as_str().unwrap()];
        let (arch, quant) = edited(model, &case["variant"]);
        let public = PublicModel {
            arch: &arch,
            profile: model.profile,
            quant: &quant,
            silu_in_exponents: &model.quant.silu_in_exponents,
        };
        let n = case["n"].as_u64().unwrap() as u32;
        let policy = GadgetPolicy {
            query_block: case["query_block"].as_u64().unwrap() as u32,
            ..GadgetPolicy::default()
        };
        let lowering = lower(public, n).unwrap();
        let selection = selected(&lowering, &policy);
        let k = case["log2_k"].as_u64().unwrap() as u8;
        let claims = claims(
            public,
            &selection,
            &policy,
            k,
            case["segments"].as_u64().unwrap() as u32,
        );
        let mapped = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
        let layout = mapped.layout();
        let opening = &openings["layouts"][case_index];
        assert_eq!(opening["case"].as_u64().unwrap() as usize, case_index);
        let counts: Vec<u64> = layout
            .segments()
            .iter()
            .map(|segment| segment.nonzero.iter().map(|run| run.end - run.start).sum())
            .collect();
        assert_eq!(json!(counts), opening["segment_counts"]);
        assert_eq!(layout.rows(), opening["rows"].as_u64().unwrap());
        let hex = |text: &Value| -> Vec<u8> {
            let text = text.as_str().unwrap();
            (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
                .collect()
        };
        let descriptor = tessera_pcs::CommitmentDescriptor::Advice {
            statement_digest: hex(&opening["statement_digest"]).try_into().unwrap(),
            unit: opening["unit"].as_u64().unwrap() as u32,
            layout: AdviceLayoutParams {
                log2_k: k,
                log2_inv_rate: 1,
                mask_coeffs: opening["mask_coeffs"].as_u64().unwrap() as u32,
                row_block: opening["row_block"].as_u64().unwrap() as u32,
                segments: case["segments"].as_u64().unwrap() as u32,
            },
            rows: layout.rows(),
        };
        assert_eq!(
            tessera_codec::encode(&descriptor),
            hex(&opening["descriptor_tce"])
        );
        assert_eq!(
            descriptor.commitment_id().as_slice(),
            hex(&opening["commitment_id"])
        );
        assert_eq!(layout.allocated_rows(), case["rows"].as_u64().unwrap());
        assert_eq!(layout.variables(), case["nu"].as_u64().unwrap() as u32);
        assert_eq!(
            mapped.tiles().len(),
            case["tiles"].as_array().unwrap().len()
        );
        for (tile, expected) in case["tiles"].as_array().unwrap().iter().enumerate() {
            let actual = &mapped.tiles()[tile];
            let key = match actual.tile {
                Tile::Tokens => json!(["tokens"]),
                Tile::Node { node } => json!(["node", node]),
                Tile::Table { table } => json!(["table", table]),
                Tile::GreedyMaximum { gadget } => json!(["greedy.maximum", gadget]),
                Tile::Attention { node, kv_head } => json!([
                    "attention",
                    node,
                    kv_head,
                    actual.block.as_ref().unwrap().index
                ]),
            };
            assert_eq!(key, expected["key"]);
            assert_eq!(
                layout.tiles()[tile].segment,
                expected["segment"].as_u64().unwrap() as u32
            );
            let sources = mapped.sources(tile).unwrap();
            let values = expected["values"].as_array().unwrap();
            assert_eq!(sources.len(), values.len());
            let placements = sources
                .iter()
                .map(|&source| mapped.location(tile, source).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                json!([
                    placements[0].first_row(),
                    placements.last().unwrap().first_row() + placements.last().unwrap().row_count()
                ]),
                expected["rows"]
            );
            for ((&source, placement), value) in sources.iter().zip(placements).zip(values) {
                assert_eq!(identity(source), value[0]);
                let permutation = value[2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(placement.vars() as usize, permutation.len());
                assert_eq!(placement.first_row(), value[3].as_u64().unwrap());
                assert_eq!(placement.row_count(), value[4].as_u64().unwrap());
                let zero_at = value[5].as_u64().unwrap();
                assert_eq!(
                    placement
                        .nonzero()
                        .iter()
                        .map(|r| r.end - r.start)
                        .sum::<u64>(),
                    zero_at
                );
                assert!(placement.nonzero().len() <= 1);
                let natural = point(permutation.len());
                let mut image = permutation.iter().map(|&i| natural[i]).collect::<Vec<_>>();
                for bit in permutation.len() as u32..layout.variables() {
                    let fixed = if bit < u32::from(k) {
                        0
                    } else {
                        placement.first_row() >> (bit - u32::from(k)) & 1
                    };
                    image.push(if fixed == 0 { Fp2::ZERO } else { Fp2::ONE });
                }
                assert_eq!(placement.point(&natural).unwrap(), image);
                if !matches!(source, AdviceSource::Multiplicity(_)) {
                    // Invert the frozen oracle's bit permutation, independently of locate().
                    // Sample each ordinary source's first/middle/last row and column, including
                    // padding. No allocation scales with the source's tensor size.
                    let shape: Vec<u32> = value[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| u32::try_from(d.as_u64().unwrap()).unwrap())
                        .collect();
                    let covered = 1usize << placement.vars().min(u32::from(k));
                    for row in [0, placement.row_count() / 2, placement.row_count() - 1] {
                        for col in [0, covered / 2, covered - 1] {
                            let stored = (u128::from(row) << k) | col as u128;
                            let mut natural_index = 0u128;
                            for (stored_bit, &natural_bit) in permutation.iter().enumerate() {
                                natural_index |= ((stored >> stored_bit) & 1) << natural_bit;
                            }
                            let mut stride = 1usize;
                            let mut flat = 0usize;
                            let mut padding = false;
                            for &len in shape.iter().rev() {
                                let bits = len.next_power_of_two().trailing_zeros();
                                let coordinate = (natural_index & ((1u128 << bits) - 1)) as usize;
                                natural_index >>= bits;
                                padding |= coordinate >= len as usize;
                                flat += coordinate * stride;
                                stride *= len as usize;
                            }
                            assert_eq!(
                                placement
                                    .row_indices(placement.first_row() + row, &shape)
                                    .unwrap()
                                    .nth(col),
                                Some((!padding).then_some(flat))
                            );
                        }
                    }
                }

                if !matches!(actual.tile, Tile::Attention { .. }) {
                    let images = mapped
                        .claim(source, &natural)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    assert_eq!(images.len(), 1);
                    assert_eq!(images[0].source, source);
                    assert_eq!(images[0].tile, tile);
                    assert_eq!(images[0].weight, Fp2::ONE);
                    assert_eq!(images[0].point, image);
                }
                locations += 1;
            }
        }
        let segments = layout
            .segments()
            .iter()
            .map(|s| json!([s.rows.start, s.rows.end]))
            .collect::<Vec<_>>();
        assert_eq!(json!(segments), case["segment_rows"]);
        let zeros = case["zero_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[0].as_u64().unwrap()..r[1].as_u64().unwrap())
            .collect::<Vec<_>>();
        for row in 0..layout.allocated_rows() {
            assert_eq!(
                layout.is_zero_row(row),
                zeros.iter().any(|r| r.contains(&row)),
                "{} n={n} k={k} row={row}",
                case["model"]
            );
        }
        assert!(layout.is_zero_row(layout.allocated_rows()));
        assert!(mapped.sources(mapped.tiles().len()).is_none());
    }
    assert_eq!(locations, 65_620);
}

#[test]
fn whole_attention_claims_reconstruct_over_the_extension_field_with_partial_boxes_and_kv_padding() {
    let models = models();
    let model = &models["tiny-2-servable.d"];
    for kv in [1, 2, 3] {
        for (q, k) in [(1, 3), (2, 3), (8, 4), (512, 10)] {
            let mut arch = model.arch.clone();
            arch.n_heads = 2 * kv;
            arch.n_kv_heads = kv;
            arch.d_head = 2;
            arch.d_model = 4 * kv;
            arch.max_context = 5;
            let mut quant = model.quant.public.clone();
            quant.rsqrt_kappa = 3;
            let public = PublicModel {
                arch: &arch,
                profile: model.profile,
                quant: &quant,
                silu_in_exponents: &model.quant.silu_in_exponents,
            };
            let policy = GadgetPolicy {
                query_block: q,
                ..GadgetPolicy::default()
            };
            let lowering = lower(public, 5).unwrap();
            let selected = selected(&lowering, &policy);
            let claims = claims(public, &selected, &policy, k, 3);
            let mapped = ModelAdviceLayout::new(&lowering, &selected, &claims).unwrap();
            for (first, sequence) in mapped.tiles().iter().enumerate() {
                let Tile::Attention { node, kv_head: 0 } = sequence.tile else {
                    continue;
                };
                if sequence.block.as_ref().unwrap().index != 0 {
                    continue;
                }
                for (column, &source) in mapped.sources(first).unwrap().iter().enumerate() {
                    let score = column == 0;
                    let query_block = u64::from(q.min(8));
                    let w = 3u32.min(u32::from(k) - 1 - query_block.ilog2());
                    let r = point(
                        (2 * kv).next_power_of_two().ilog2() as usize + if score { 6 } else { 3 },
                    );
                    let mut sparse = Vec::new();
                    let mut expected = Fp2::ZERO;
                    for h in 0..u64::from(2 * kv) {
                        for query in 0..5u64 {
                            for key in 0..if score { query + 1 } else { 1 } {
                                let value = Fp2::from(Goldilocks::from_u64_reduced(
                                    1 + 103 * h + 17 * query + key,
                                ));
                                let natural = if score {
                                    (h * 8 + query) * 8 + key
                                } else {
                                    h * 8 + query
                                };
                                expected += value * basis(natural, &r);
                                let (tile, _) = mapped
                                    .tiles()
                                    .iter()
                                    .enumerate()
                                    .find(|(_, t)| {
                                        t.tile
                                            == Tile::Attention {
                                                node,
                                                kv_head: (h / 2) as u32,
                                            }
                                            && t.block.as_ref().unwrap().index
                                                == (query / query_block) as u32
                                    })
                                    .unwrap();
                                let placement = mapped.location(tile, source).unwrap();
                                let local = if score {
                                    (key & ((1 << w) - 1))
                                        | ((query % query_block) << w)
                                        | ((h % 2) << (w + query_block.ilog2()))
                                        | ((key >> w) << (w + query_block.ilog2() + 1))
                                } else {
                                    (h % 2) * query_block + query % query_block
                                };
                                sparse.push(((placement.first_row() << k) | local, value));
                            }
                        }
                    }
                    let images = mapped
                        .claim(source, &r)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    assert_eq!(images.len(), kv as usize * (8 / query_block) as usize);
                    let reconstructed = images.iter().fold(Fp2::ZERO, |sum, image| {
                        assert_eq!(image.source, source);
                        sum + image.weight
                            * sparse.iter().fold(Fp2::ZERO, |value, &(address, entry)| {
                                value + entry * basis(address, &image.point)
                            })
                    });
                    assert_eq!(
                        reconstructed, expected,
                        "KV={kv} Q={q} k={k} column={column}"
                    );
                }
            }
            let token = AdviceSource::Tensor(lowering.tokens);
            assert!(mapped.claim::<Fp2>(token, &[]).is_err());
            assert!(mapped.location(mapped.tiles().len(), token).is_none());
        }
    }
}

#[test]
fn runtime_catalog_and_every_tail_location_match_the_independent_oracle() {
    use tessera_compiler::RuntimePlan;
    let models = models();
    let data = read_json(&vectors().join("greedy-runtime/runtime.json"));
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 141);
    let mut appended = 0;
    for case in cases {
        let model = &models[case["file"].as_str().unwrap()];
        let (mut arch, mut quant) = if case["variant"].is_null() {
            (model.arch.clone(), model.quant.public.clone())
        } else {
            edited(model, &case["variant"])
        };
        if let Some(vocab) = case["vocab"].as_u64() {
            arch.vocab = vocab as u32;
        }
        if let Some(custom) = case["custom"].as_object() {
            let u = |key| custom[key].as_u64().unwrap() as u32;
            arch.d_model = u("d_model");
            arch.n_heads = u("n_heads");
            arch.n_kv_heads = u("n_kv_heads");
            arch.d_head = u("d_head");
            arch.d_ff = u("d_ff");
            quant.rsqrt_kappa = u("rsqrt_kappa") as u8;
            quant.exp_sigma = u("exp_sigma") as u8;
            quant.residual_bits = u("residual_bits") as u8;
            quant.residual_exponent = custom["residual_exponent"].as_i64().unwrap() as i8;
        }
        let public = PublicModel {
            arch: &arch,
            profile: model.profile,
            quant: &quant,
            silu_in_exponents: &model.quant.silu_in_exponents,
        };
        let n = case["n"].as_u64().unwrap() as u32;
        let policy = GadgetPolicy {
            query_block: case["query_block"].as_u64().unwrap() as u32,
            ..GadgetPolicy::default()
        };
        let lowering = lower(public, n).unwrap();
        let selection = selected(&lowering, &policy);
        let claims = claims(
            public,
            &selection,
            &policy,
            case["log2_k"].as_u64().unwrap() as u8,
            case["segments"].as_u64().unwrap() as u32,
        );
        let original = tessera_codec::encode(&claims);
        let runtime = RuntimePlan::derive(
            &lowering,
            &selection,
            &claims,
            case["prompt"].as_u64().unwrap() as u32,
            case["ordinal"].as_u64().unwrap() as u32,
        )
        .unwrap();
        let tables = &runtime.selection().tables;
        assert_eq!(&tables[..selection.tables.len()], selection.tables);
        assert_eq!(tables.len(), case["tables"].as_array().unwrap().len());
        appended += usize::from(tables.len() > selection.tables.len());
        for (actual, expected) in tables.iter().zip(case["tables"].as_array().unwrap()) {
            assert_eq!(u64::from(actual.id.0), expected[0].as_u64().unwrap());
            assert_eq!(u64::from(actual.log2_rows), expected[2].as_u64().unwrap());
            if let Some(range) = expected[1].get("Range") {
                assert_eq!(
                    actual.kind,
                    tessera_tables::TableKind::Range {
                        bits: range["bits"].as_u64().unwrap() as u8
                    }
                );
            }
        }
        if let Some(cert) = runtime.certificate() {
            let expected = &case["certificate"];
            assert_eq!(u64::from(cert.gadget.0), expected["id"].as_u64().unwrap());
            assert_eq!(u64::from(cert.table.0), expected["range"].as_u64().unwrap());
            assert_eq!(
                u64::from(cert.range_bits),
                expected["bits"].as_u64().unwrap()
            );
            assert_eq!(cert.entries(), expected["entries"].as_u64().unwrap());
            assert_eq!(
                json!([cert.logits.lo as i64, cert.logits.hi as i64]),
                expected["logit_bounds"]
            );
        } else {
            assert!(case["certificate"].is_null());
        }
        let mapped = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
        let layout = mapped.layout();
        assert_eq!(layout.allocated_rows(), case["rows"].as_u64().unwrap());
        assert_eq!(u64::from(layout.variables()), case["nu"].as_u64().unwrap());
        let model_tiles = case["model_tiles"].as_u64().unwrap() as usize;
        let base = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
        for tile in 0..model_tiles {
            assert_eq!(mapped.tiles()[tile], base.tiles()[tile]);
            assert_eq!(mapped.sources(tile), base.sources(tile));
            for &source in mapped.sources(tile).unwrap() {
                let a = mapped.location(tile, source).unwrap();
                let b = base.location(tile, source).unwrap();
                assert_eq!(
                    (a.first_row(), a.row_count(), a.vars(), a.nonzero()),
                    (b.first_row(), b.row_count(), b.vars(), b.nonzero())
                );
            }
        }
        let tail = case["tail"].as_array().unwrap();
        assert_eq!(mapped.tiles().len(), model_tiles + tail.len());
        for (tile, expected) in (model_tiles..mapped.tiles().len()).zip(tail) {
            let key = match mapped.tiles()[tile].tile {
                Tile::Table { table } => json!(["table", table]),
                Tile::GreedyMaximum { gadget } => json!(["greedy.maximum", gadget]),
                other => panic!("unexpected runtime tail {other:?}"),
            };
            assert_eq!(key, expected["key"]);
            assert_eq!(
                u64::from(layout.tiles()[tile].segment),
                expected["segment"].as_u64().unwrap()
            );
            let [source] = mapped.sources(tile).unwrap() else {
                panic!("one tail source")
            };
            let place = mapped.location(tile, *source).unwrap();
            let value = &expected["values"][0];
            assert_eq!(identity(*source), value[0]);
            assert_eq!(place.first_row(), value[3].as_u64().unwrap());
            assert_eq!(place.row_count(), value[4].as_u64().unwrap());
            assert_eq!(
                place.nonzero().iter().map(|r| r.end - r.start).sum::<u64>(),
                value[5].as_u64().unwrap()
            );
            let perm = value[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(place.vars() as usize, perm.len());
            let natural = point(perm.len());
            let mut image = perm.iter().map(|&i| natural[i]).collect::<Vec<_>>();
            for bit in perm.len() as u32..layout.variables() {
                let fixed = bit >= layout.params().log2_k
                    && place.first_row() >> (bit - layout.params().log2_k) & 1 == 1;
                image.push(if fixed { Fp2::ONE } else { Fp2::ZERO });
            }
            let actual = mapped
                .claim(*source, &natural)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(actual.len(), 1);
            assert_eq!(
                (
                    actual[0].tile,
                    actual[0].source,
                    actual[0].weight,
                    &actual[0].point
                ),
                (tile, *source, Fp2::ONE, &image)
            );
        }
        assert_eq!(
            json!(
                layout
                    .segments()
                    .iter()
                    .map(|s| [s.rows.start, s.rows.end])
                    .collect::<Vec<_>>()
            ),
            case["segment_rows"]
        );
        assert_eq!(tessera_codec::encode(&claims), original);
    }
    assert_eq!(appended, 8);
}

#[test]
fn runtime_refuses_request_identity_catalog_and_physical_width_faults() {
    use tessera_compiler::{CompileError, RuntimePlan};
    let models = models();
    let public = PublicModel::of(&models["tiny-2-servable.d"]);
    let lowering = lower(public, 5).unwrap();
    let policy = GadgetPolicy::default();
    let selection = selected(&lowering, &policy);
    let claims = claims(public, &selection, &policy, 10, 3);
    let request = CompileError::Usage {
        rule: "the greedy request names a positive prompt within the model sequence",
    };
    for prompt in [0, 6, u32::MAX] {
        assert_eq!(
            RuntimePlan::derive(&lowering, &selection, &claims, prompt, 0).unwrap_err(),
            request
        );
    }
    assert_eq!(
        RuntimePlan::derive(&lowering, &selection, &claims, 1, u32::MAX).unwrap_err(),
        request
    );
    assert!(
        RuntimePlan::derive(&lowering, &selection, &claims, 5, u32::MAX)
            .unwrap()
            .certificate()
            .is_none()
    );
    let mut bad = lowering.clone();
    bad.logits = tessera_ir::TensorId(u32::MAX);
    assert_eq!(
        RuntimePlan::derive(&bad, &selection, &claims, 1, 0).unwrap_err(),
        request
    );
    let mut bad = selection.clone();
    bad.tables.pop();
    assert_eq!(
        RuntimePlan::derive(&lowering, &bad, &claims, 1, 0).unwrap_err(),
        CompileError::Usage {
            rule: "runtime selection preserves the model's canonical table catalog",
        }
    );
    let runtime = RuntimePlan::derive(&lowering, &selection, &claims, 1, 0).unwrap();
    let c = runtime.certificate().unwrap();
    let mut capped = claims.clone();
    capped.gadget_policy.max_log2_rows = c.range_bits - 1;
    assert_eq!(
        RuntimePlan::derive(&lowering, &selection, &capped, 1, 0).unwrap_err(),
        CompileError::Model {
            rule: "the greedy physical Range width fits GadgetPolicy.max_log2_rows",
        }
    );
    capped.gadget_policy.max_log2_rows = c.range_bits;
    assert_eq!(
        RuntimePlan::derive(&lowering, &selection, &capped, 1, 0)
            .unwrap()
            .certificate(),
        Some(c)
    );
    let layout = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
    let source = AdviceSource::GreedyMaximum(c.gadget);
    assert!(matches!(
        layout.claim::<Fp2>(source, &[]),
        Err(CompileError::Usage {
            rule: "the advice claim point has its source's natural variable count",
        })
    ));
    assert!(matches!(
        layout.claim::<Fp2>(
            AdviceSource::GreedyMaximum(tessera_ir::GadgetId(u32::MAX)),
            &[]
        ),
        Err(CompileError::Usage {
            rule: "the advice claim names a committed tensor or gadget auxiliary"
        })
    ));
}

#[test]
fn runtime_derives_the_widest_allowed_range_and_refuses_the_public_logit_bound() {
    use tessera_compiler::{CompileError, RuntimePlan};
    let models = models();
    let base = &models["tiny-2-servable.d"];
    for dimension in [16_384, 32_768] {
        let mut arch = base.arch.clone();
        arch.d_model = dimension;
        arch.n_heads = dimension / 2;
        arch.n_kv_heads = 1;
        arch.d_head = 2;
        arch.d_ff = 3 * dimension;
        arch.max_context = 8;
        let mut quant = base.quant.public.clone();
        quant.rsqrt_kappa = 0;
        quant.exp_sigma = 3;
        quant.residual_bits = 8;
        quant.residual_exponent = quant
            .act_exponents
            .iter()
            .find(|(p, _)| *p == tessera_ir::QuantPoint::Embed)
            .unwrap()
            .1;
        for (point, shifts) in &mut quant.shift_class_sets {
            if matches!(
                point,
                tessera_ir::QuantPoint::Layer {
                    site: tessera_ir::Site::Silu,
                    ..
                }
            ) {
                *shifts = vec![20];
            }
        }
        let public = PublicModel {
            arch: &arch,
            profile: base.profile,
            quant: &quant,
            silu_in_exponents: &base.quant.silu_in_exponents,
        };
        let lowering = lower(public, 2).unwrap();
        let policy = GadgetPolicy::default();
        let selected = selected(&lowering, &policy);
        let claims = claims(public, &selected, &policy, 22, 3);
        let result = RuntimePlan::derive(&lowering, &selected, &claims, 1, 0);
        if dimension == 16_384 {
            let certificate = result.unwrap().certificate().unwrap();
            assert_eq!(certificate.range_bits, 29);
            assert_eq!(certificate.logits.lo, -266_338_304);
            assert_eq!(certificate.logits.hi, 266_338_304);
        } else {
            assert_eq!(
                result.unwrap_err(),
                CompileError::Model {
                    rule: "greedy logits have a public interval strictly within the signed 28-bit bound",
                }
            );
        }
    }
}
