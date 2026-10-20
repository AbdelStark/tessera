// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public micro-model variants exercise canonical opening boundaries without million-row tables.

mod claim_support;
mod openings;

use tv_core::advice::opening as canonical;
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::{advice, lowering, selection, validation};
use tv_field::Fp2;
use tv_testkit::{Json, read_vectors};

use openings::{Case, digest, fp2s, list, member, number};

#[derive(Clone)]
struct Obligations {
    points: Vec<Vec<Fp2>>,
    pieces: Vec<Vec<canonical::Piece>>,
    values: Vec<Fp2>,
}

impl Obligations {
    fn read(case: &Json) -> Self {
        let original = list(case, "claims");
        Self {
            points: original
                .iter()
                .map(|claim| fp2s(member(claim, "point")))
                .collect(),
            pieces: original
                .iter()
                .map(|claim| {
                    list(claim, "pieces")
                        .iter()
                        .map(|piece| canonical::Piece {
                            source: source(member(piece, "source")),
                            coefficient: openings::fp2(member(piece, "coefficient")),
                        })
                        .collect()
                })
                .collect(),
            values: original
                .iter()
                .map(|claim| openings::fp2(member(claim, "value")))
                .collect(),
        }
    }

    fn claims(&self) -> Vec<canonical::Claim<'_>> {
        self.points
            .iter()
            .zip(&self.pieces)
            .zip(&self.values)
            .map(|((point, pieces), &value)| canonical::Claim {
                point,
                pieces,
                value,
            })
            .collect()
    }
}

fn micro_model() -> (ClaimList, lowering::Graph, selection::Selection) {
    let claims = micro_claims();
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    (claims, graph, selected)
}

fn parameters(case: &Json) -> tv_core::claims::AdviceLayoutParams {
    tv_core::claims::AdviceLayoutParams {
        log2_k: number(case, "log2_k") as u8,
        log2_inv_rate: number(case, "log2_inv_rate") as u8,
        mask_coeffs: number(case, "mask_coeffs") as u32,
        row_block: number(case, "row_block") as u32,
        segments: list(case, "segments").len() as u32,
    }
}

fn bind<'a>(layout: &'a advice::Layout<'a>, case: &Json) -> canonical::Commitment<'a> {
    canonical::Commitment::bind(
        layout,
        parameters(case),
        &digest(member(case, "statement_digest")),
        number(case, "unit") as u32,
        digest(member(case, "root")),
        digest(member(case, "commitment_id")),
    )
    .unwrap()
}

fn check(
    prepared: &canonical::Opening<'_>,
    combined: &[tv_core::opening::CombinedRow],
    case: &Case,
) -> Result<(), tv_core::opening::OpeningError> {
    prepared.check_evaluations(combined)?.check_columns(
        &case.queries,
        &case.columns,
        &case.siblings,
    )
}

fn micro_claims() -> ClaimList {
    let mut claims = claim_support::claims_of("tiny-2-servable.d");
    let arch = &mut claims.arch;
    arch.d_model = 4;
    arch.n_heads = 2;
    arch.n_kv_heads = 1;
    arch.d_head = 2;
    arch.d_ff = 6;
    arch.vocab = 8;
    arch.max_context = 8;
    let quant = &mut claims.quant;
    quant.residual_bits = 8;
    quant.residual_exponent = quant
        .act_exponents
        .iter()
        .find(|(point, _)| *point == QuantPoint::Embed)
        .expect("embedding exponent")
        .1;
    quant.rsqrt_kappa = 0;
    quant.exp_sigma = 3;
    for (_, shifts) in &mut quant.shift_class_sets {
        *shifts = vec![12];
    }
    for (_, scale) in &mut quant.score_scales {
        *scale = tv_codec::Rational::new(1, 1).expect("positive scale");
    }
    claims.gadget_policy.max_log2_rows = 17;
    claims.gadget_policy.limb_bits = 8;
    claims.gadget_policy.query_block = 2;
    claims.layout.advice.log2_k = 6;
    claims.layout.advice.mask_coeffs = 4;
    claims.layout.advice.row_block = 3;
    claims.layout.advice.segments = 8;
    claims
}

#[test]
fn the_micro_model_uses_valid_public_lowering_and_gadget_metadata() {
    let claims = micro_claims();
    let public = validation::check_public(&claims).expect("valid public variant");
    let graph = lowering::lower(&public, claims.arch.max_context).expect("lowering");
    let selected = selection::derive(&graph, claims.gadget_policy).expect("selection");
    for n in [1, 3, 5, 8] {
        let config = advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            claims.layout.advice.log2_k,
            claims.layout.advice.segments,
        );
        let layout = advice::Layout::derive(&graph, &selected, config, n).expect("layout");
        assert!(
            layout.rows() < 1 << 16,
            "finite complete model, including tables"
        );
    }
}

#[test]
fn compact_counts_match_locations_at_query_key_and_partial_block_boundaries() {
    let claims = micro_claims();
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    let mut checked = 0;
    for n in 1u32..=8 {
        for log_k in 3..=10 {
            for query in [1, 2, 4, 8] {
                if 1u32 << log_k < 2 * query.min(n.next_power_of_two()) {
                    continue;
                }
                for segments in [1, 2, 8] {
                    let mut config =
                        advice::Config::new(&claims.arch, claims.gadget_policy, log_k, segments);
                    config.query_block = query;
                    let layout = advice::Layout::derive(&graph, &selected, config, n).unwrap();
                    let mut expected = vec![0; segments as usize];
                    for tile in layout.tiles() {
                        for column in 0..tile.columns() {
                            expected[tile.segment() as usize] +=
                                tile.location(column).unwrap().zero_at;
                        }
                    }
                    assert_eq!(layout.committed_rows().unwrap(), expected.iter().sum());
                    for (index, count) in expected.into_iter().enumerate() {
                        assert_eq!(
                            layout.committed_rows_in_segment(index as u32).unwrap(),
                            count
                        );
                    }
                    assert_eq!(
                        layout.committed_rows_in_segment(segments),
                        Err(validation::ClaimsError::Internal)
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 756);
}

fn descriptor(case: &Json, log_k: u8, segments: u32, rows: u64) -> Vec<u8> {
    let mut bytes = vec![0, 0];
    bytes.extend_from_slice(&digest(member(case, "statement_digest")));
    bytes.extend_from_slice(&u32::try_from(number(case, "unit")).unwrap().to_le_bytes());
    bytes.extend([log_k, 1]);
    for value in [
        number(case, "mask_coeffs"),
        number(case, "row_block"),
        u64::from(segments),
    ] {
        bytes.extend_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
    }
    bytes.extend_from_slice(&rows.to_le_bytes());
    bytes
}

fn source(value: &Json) -> advice::Source {
    use advice::{Auxiliary as A, Comparison as C, ComparisonRole as R, Source as S};
    use tv_core::lowering::{NodeId, TensorId};
    let values = value.as_array().unwrap();
    let id = u32::try_from(values[1].as_u64().unwrap()).unwrap();
    match values[0].as_str().unwrap() {
        "tensor" => S::Tensor(TensorId(id)),
        "multiplicity" => S::Multiplicity(tv_tables::TableId(id)),
        "aux" => {
            let index = u32::try_from(values[3].as_u64().unwrap()).unwrap();
            let name = values[2].as_str().unwrap();
            let role = match name {
                "requant.low" => A::RequantLow,
                "requant.middle" => A::RequantMiddle,
                "requant.saturation" => A::RequantSaturation,
                "softmax.maximum" => A::SoftmaxMaximum,
                "softmax.a" => A::SoftmaxA(index),
                "softmax.b" => A::SoftmaxB(index),
                "root.input" => A::RootInput(index),
                "root.low" => A::RootLow(index),
                _ => {
                    let rest = name.strip_prefix("root.").unwrap();
                    let (comparison, role) = rest.split_once('.').unwrap();
                    let comparison = match comparison {
                        "lower" => C::Lower,
                        "upper" => C::Upper,
                        _ => panic!("comparison"),
                    };
                    let role = match role {
                        "left.digit" => R::LeftDigit,
                        "left.carry" => R::LeftCarry,
                        "right.digit" => R::RightDigit,
                        "right.carry" => R::RightCarry,
                        "difference" => R::Difference,
                        "borrow" => R::Borrow,
                        _ => panic!("comparison role"),
                    };
                    A::RootComparison(comparison, role, index)
                }
            };
            S::Auxiliary(NodeId(id), role)
        }
        _ => panic!("source"),
    }
}

#[test]
fn original_claim_pieces_and_powers_map_to_the_independent_group_images() {
    let vectors = read_vectors("advice-opening/opening.json");
    let claims = micro_claims();
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    for case in list(&vectors, "openings") {
        let mut config = advice::Config::new(&claims.arch, claims.gadget_policy, 6, 1);
        config.segments = list(case, "segments").len() as u32;
        let layout =
            advice::Layout::derive(&graph, &selected, config, number(case, "n") as u32).unwrap();
        let originals = list(case, "claims");
        let mut seen = vec![false; originals.len()];
        let mut firsts = Vec::new();
        for group in list(case, "groups") {
            firsts.push(list(group, "claim_indices")[0].as_u64().unwrap());
            let expected = list(group, "claims");
            let column_point = fp2s(member(group, "column_point"));
            let mut power = Fp2::ONE;
            let mut sum = Fp2::ZERO;
            let mut image = 0;
            let mut last = None;
            for index in list(group, "claim_indices") {
                let index = index.as_u64().unwrap() as usize;
                assert!(
                    !seen[index],
                    "one original claim enters one group exactly once"
                );
                assert!(last.is_none_or(|last| index > last), "original claim order");
                last = Some(index);
                seen[index] = true;
                power *= openings::fp2(member(case, "gamma"));
                let claim = &originals[index];
                sum += power * openings::fp2(member(claim, "value"));
                let point = fp2s(member(claim, "point"));
                for piece in list(claim, "pieces") {
                    let coefficient = power * openings::fp2(member(piece, "coefficient"));
                    for mapped in layout
                        .claim(source(member(piece, "source")), &point)
                        .unwrap()
                    {
                        let mapped = mapped.unwrap();
                        assert_eq!(&mapped.point[..6], column_point);
                        assert_eq!(
                            &mapped.point[6..],
                            fp2s(member(&expected[image], "row_point"))
                        );
                        assert_eq!(
                            coefficient * mapped.weight,
                            openings::fp2(member(&expected[image], "gamma"))
                        );
                        image += 1;
                    }
                }
            }
            assert_eq!(image, expected.len());
            assert_eq!(sum, openings::fp2(member(group, "expected")));
        }
        assert!(seen.into_iter().all(|seen| seen));
        assert!(
            firsts.windows(2).all(|pair| pair[0] < pair[1]),
            "groups retain first appearance"
        );
    }
}

#[test]
fn lazy_source_coefficients_match_every_frozen_combination_vector() {
    let vectors = read_vectors("advice-opening/opening.json");
    let claims = micro_claims();
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    for case in list(&vectors, "openings") {
        let mut config = advice::Config::new(&claims.arch, claims.gadget_policy, 6, 1);
        config.segments = list(case, "segments").len() as u32;
        let layout =
            advice::Layout::derive(&graph, &selected, config, number(case, "n") as u32).unwrap();
        let rows: Vec<_> = list(case, "segments")
            .iter()
            .flat_map(|part| part.as_array().unwrap())
            .map(|row| row.as_u64().unwrap())
            .collect();
        let original = list(case, "claims");
        for (group, expected_hash) in list(case, "groups")
            .iter()
            .zip(list(case, "coefficient_sha256"))
        {
            let mut coefficients = vec![Fp2::ZERO; rows.len()];
            let mut power = Fp2::ONE;
            for index in list(group, "claim_indices") {
                power *= openings::fp2(member(case, "gamma"));
                let claim = &original[index.as_u64().unwrap() as usize];
                let point = fp2s(member(claim, "point"));
                for piece in list(claim, "pieces") {
                    let mut images = layout
                        .claim(source(member(piece, "source")), &point)
                        .unwrap();
                    let before = images.row_coefficient(rows[0]).unwrap();
                    let _ = images.next().transpose().unwrap();
                    assert_eq!(
                        images.row_coefficient(rows[0]).unwrap(),
                        before,
                        "cursor-independent weight"
                    );
                    assert_eq!(images.row_coefficient(u64::MAX).unwrap(), Fp2::ZERO);
                    let weight = power * openings::fp2(member(piece, "coefficient"));
                    for (coefficient, &row) in coefficients.iter_mut().zip(&rows) {
                        *coefficient += weight * images.row_coefficient(row).unwrap();
                    }
                }
            }
            let bytes: Vec<_> = coefficients.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(tv_hash::sha256::sha256(&bytes), digest(expected_hash));
        }
    }
}

#[test]
fn every_native_layout_count_and_descriptor_matches_the_independent_oracle() {
    let original = read_vectors("advice/layout.json");
    let vectors = read_vectors("advice-opening/opening.json");
    assert_eq!(list(&vectors, "layouts").len(), 54);
    for expected in list(&vectors, "layouts") {
        let case = &list(&original, "cases")[number(expected, "case") as usize];
        let mut claims = claim_support::claims_of(claim_support::text(case, "file"));
        if !matches!(member(case, "variant"), Json::Null) {
            claim_support::apply_variant(member(case, "variant"), &mut claims);
        }
        let public = validation::check_public(&claims).unwrap();
        let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
        let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
        let log_k = u8::try_from(number(case, "log2_k")).unwrap();
        let segments = u32::try_from(number(case, "segments")).unwrap();
        let config = advice::Config {
            log_k,
            segments,
            query_block: u32::try_from(number(case, "query_block")).unwrap(),
            heads: claims.arch.n_heads,
            kv_heads: claims.arch.n_kv_heads,
            limb_bits: claims.gadget_policy.limb_bits,
        };
        let layout =
            advice::Layout::derive(&graph, &selected, config, number(case, "n") as u32).unwrap();
        // Iterate locations, not the millions of rows that a table can occupy.
        let mut counts = vec![0u64; segments as usize];
        for tile in layout.tiles() {
            for column in 0..tile.columns() {
                counts[tile.segment() as usize] += tile.location(column).unwrap().zero_at;
            }
        }
        let recorded: Vec<_> = list(expected, "segment_counts")
            .iter()
            .map(|n| n.as_u64().unwrap())
            .collect();
        assert_eq!(counts, recorded);
        let compact: Vec<_> = (0..segments)
            .map(|segment| layout.committed_rows_in_segment(segment).unwrap())
            .collect();
        assert_eq!(compact, recorded);
        let rows: u64 = counts.iter().sum();
        assert_eq!(rows, number(expected, "rows"));
        assert_eq!(layout.committed_rows().unwrap(), rows);
        let encoded = descriptor(expected, log_k, segments, rows);
        assert_eq!(
            encoded,
            tv_testkit::hex(claim_support::text(expected, "descriptor_tce"))
        );
        assert_eq!(
            tv_hash::hash(tv_hash::Domain::CommitmentId, &encoded),
            digest(member(expected, "commitment_id"))
        );
        canonical::Commitment::bind(
            &layout,
            tv_core::claims::AdviceLayoutParams {
                log2_k: log_k,
                log2_inv_rate: 1,
                segments,
                mask_coeffs: number(expected, "mask_coeffs") as u32,
                row_block: number(expected, "row_block") as u32,
            },
            &digest(member(expected, "statement_digest")),
            number(expected, "unit") as u32,
            [0; 32],
            digest(member(expected, "commitment_id")),
        )
        .unwrap();
    }
}

#[test]
fn complete_micro_openings_use_the_canonical_rows_and_pass_the_existing_primitive() {
    let vectors = read_vectors("advice-opening/opening.json");
    let counts: Vec<_> = list(&vectors, "openings")
        .iter()
        .map(|case| list(case, "groups").len())
        .collect();
    assert_eq!(counts, [1, 16, 5]);
    let claims = micro_claims();
    let public = validation::check_public(&claims).unwrap();
    let graph = lowering::lower(&public, claims.arch.max_context).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    for expected in list(&vectors, "openings") {
        let case = Case::from_json(expected);
        let mut config = advice::Config::new(&claims.arch, claims.gadget_policy, 6, 1);
        config.segments = u32::try_from(case.segments.len()).unwrap();
        let layout =
            advice::Layout::derive(&graph, &selected, config, number(expected, "n") as u32)
                .unwrap();
        assert_eq!(layout.rows(), number(expected, "allocated_rows"));
        assert_eq!(layout.nu(), 6 + case.row_vars);
        let mut rows = vec![Vec::new(); case.segments.len()];
        for tile in layout.tiles() {
            for index in 0..tile.columns() {
                let location = tile.location(index).unwrap();
                rows[tile.segment() as usize]
                    .extend(location.base..location.base + location.zero_at);
            }
        }
        assert_eq!(rows, case.segments);
        assert_eq!(
            layout.committed_rows().unwrap(),
            rows.iter().map(Vec::len).sum::<usize>() as u64
        );
        assert_eq!(case.check(), Ok(()), "{}", case.name);
        let encoded = descriptor(
            expected,
            6,
            rows.len() as u32,
            rows.iter().map(Vec::len).sum::<usize>() as u64,
        );
        assert_eq!(
            encoded,
            tv_testkit::hex(claim_support::text(expected, "descriptor_tce"))
        );
        assert_eq!(
            tv_hash::hash(tv_hash::Domain::CommitmentId, &encoded),
            case.params.commitment_id
        );
        for (group, expected) in case.groups.iter().zip(list(expected, "groups")) {
            let sum = group
                .claims
                .iter()
                .fold(Fp2::ZERO, |sum, claim| sum + claim.gamma * claim.value);
            assert_eq!(sum, openings::fp2(member(expected, "expected")));
            assert_eq!(group.column_point, fp2s(member(expected, "column_point")));
        }
    }
}

#[test]
fn the_primitive_rejects_each_changed_message_with_the_exact_reason() {
    use tv_core::opening::OpeningError;
    let vectors = read_vectors("advice-opening/opening.json");
    for expected in list(&vectors, "openings") {
        let case = Case::from_json(expected);
        let smallest = u32::try_from(*case.queries.iter().min().unwrap()).unwrap();
        let mut changed = case.clone();
        for value in &mut changed.combined[0].u {
            *value += Fp2::ONE;
        }
        assert_eq!(
            changed.check(),
            Err(OpeningError::Evaluation { group: Some(0) })
        );
        let mut changed = case.clone();
        changed.combined[0].mask[0] += Fp2::ONE;
        assert_eq!(
            changed.check(),
            Err(OpeningError::Proximity { column: smallest })
        );
        let mut changed = case.clone();
        changed.columns[0][0] += tv_field::Fp::ONE;
        assert_eq!(
            changed.check(),
            Err(OpeningError::MerklePath { column: smallest })
        );
        let mut changed = case.clone();
        changed.siblings[0][0] ^= 1;
        assert_eq!(
            changed.check(),
            Err(OpeningError::MerklePath { column: smallest })
        );
    }
}

#[test]
fn canonical_openings_verify_original_values_without_individual_image_values() {
    let vectors = read_vectors("advice-opening/opening.json");
    let (claims, graph, selected) = micro_model();
    for data in list(&vectors, "openings") {
        // Only transport messages are used from Case; its pre-expanded groups are ignored.
        let case = Case::from_json(data);
        let params = parameters(data);
        let config = advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            params.log2_k,
            params.segments,
        );
        let layout =
            advice::Layout::derive(&graph, &selected, config, number(data, "n") as u32).unwrap();
        let context = bind(&layout, data);
        let original = Obligations::read(data);
        let views = original.claims();
        let prepared = context
            .prepare(&views, openings::fp2(member(data, "gamma")), &case.xi_point)
            .unwrap();
        assert_eq!(prepared.group_count(), list(data, "groups").len());
        assert_eq!(
            check(&prepared, &case.combined, &case),
            Ok(()),
            "{}",
            case.name
        );
        // A zero batching challenge is a valid field draw; no original claim takes power zero.
        let zero = context.prepare(&views, Fp2::ZERO, &case.xi_point).unwrap();
        let mut combined = case.combined.clone();
        for row in combined.iter_mut().take(zero.group_count()) {
            row.u.fill(Fp2::ZERO);
            row.mask.fill(Fp2::ZERO);
        }
        assert_eq!(check(&zero, &combined, &case), Ok(()));
        let proximity = context.prepare(&[], Fp2::ONE, &case.xi_point).unwrap();
        assert_eq!(proximity.group_count(), 0);
        assert_eq!(
            check(&proximity, &[case.combined.last().unwrap().clone()], &case),
            Ok(())
        );
    }
}

#[test]
fn canonical_message_errors_and_query_order_match_the_specification() {
    use tv_core::opening::OpeningError as E;
    let vectors = read_vectors("advice-opening/opening.json");
    let (claims, graph, selected) = micro_model();
    for data in list(&vectors, "openings") {
        let case = Case::from_json(data);
        let params = parameters(data);
        let layout = advice::Layout::derive(
            &graph,
            &selected,
            advice::Config::new(
                &claims.arch,
                claims.gadget_policy,
                params.log2_k,
                params.segments,
            ),
            number(data, "n") as u32,
        )
        .unwrap();
        let context = bind(&layout, data);
        let original = Obligations::read(data);
        let views = original.claims();
        let prepared = context
            .prepare(&views, openings::fp2(member(data, "gamma")), &case.xi_point)
            .unwrap();
        let first = *case.queries.iter().min().unwrap() as u32;
        let mut root = case.root;
        root[0] ^= 1;
        let changed_root = canonical::Commitment::bind(
            &layout,
            params,
            &digest(member(data, "statement_digest")),
            number(data, "unit") as u32,
            root,
            case.params.commitment_id,
        )
        .unwrap();
        let wrong = changed_root
            .prepare(&views, openings::fp2(member(data, "gamma")), &case.xi_point)
            .unwrap();
        assert_eq!(
            check(&wrong, &case.combined, &case),
            Err(E::MerklePath { column: first })
        );
        let mut xi = case.xi_point.clone();
        xi[0] += Fp2::ONE;
        let wrong = context
            .prepare(&views, openings::fp2(member(data, "gamma")), &xi)
            .unwrap();
        assert_eq!(
            check(&wrong, &case.combined, &case),
            Err(E::Proximity { column: first })
        );
        let mut combined = case.combined.clone();
        for value in &mut combined[0].u {
            *value += Fp2::ONE;
        }
        assert_eq!(
            check(&prepared, &combined, &case),
            Err(E::Evaluation { group: Some(0) })
        );
        let mut combined = case.combined.clone();
        combined[0].mask[0] += Fp2::ONE;
        assert_eq!(
            check(&prepared, &combined, &case),
            Err(E::Proximity { column: first })
        );
        assert_eq!(
            prepared
                .check_evaluations(&case.combined[..case.combined.len() - 1])
                .err(),
            Some(E::Evaluation { group: None })
        );
        let mut wrong_shape = case.combined.clone();
        wrong_shape[0].mask.push(Fp2::ZERO);
        wrong_shape[0].u[0] += Fp2::ONE;
        assert_eq!(
            prepared.check_evaluations(&wrong_shape).err(),
            Some(E::Evaluation { group: None })
        );
        let mut changed = case.clone();
        changed.columns[0][0] += tv_field::Fp::ONE;
        assert_eq!(
            check(&prepared, &case.combined, &changed),
            Err(E::MerklePath { column: first })
        );
        let mut changed = case.clone();
        changed.siblings[0][0] ^= 1;
        assert_eq!(
            check(&prepared, &combined, &changed),
            Err(E::MerklePath { column: first }),
            "authentication before consistency"
        );
        let mut changed = case.clone();
        changed.columns.swap(0, 1);
        assert_eq!(
            check(&prepared, &case.combined, &changed),
            Err(E::MerklePath { column: first })
        );
        let mut changed = case.clone();
        changed.columns.push(changed.columns[0].clone());
        assert_eq!(
            check(&prepared, &case.combined, &changed),
            Err(E::MerklePath { column: first })
        );
        assert_eq!(
            prepared
                .check_evaluations(&case.combined)
                .unwrap()
                .check_columns(&[], &[], &[]),
            Err(E::NoQueries)
        );
        assert_eq!(
            prepared
                .check_evaluations(&case.combined)
                .unwrap()
                .check_columns(&[128], &[], &[]),
            Err(E::QueryRange { position: 0 })
        );
    }
}

#[test]
fn canonical_binding_refuses_stale_descriptors_and_incompatible_own_parameters() {
    use validation::{ClaimsError as E, StatementField};
    let vectors = read_vectors("advice-opening/opening.json");
    let (claims, graph, selected) = micro_model();
    for data in list(&vectors, "openings") {
        let params = parameters(data);
        let layout = advice::Layout::derive(
            &graph,
            &selected,
            advice::Config::new(
                &claims.arch,
                claims.gadget_policy,
                params.log2_k,
                params.segments,
            ),
            number(data, "n") as u32,
        )
        .unwrap();
        let statement = digest(member(data, "statement_digest"));
        let root = digest(member(data, "root"));
        let id = digest(member(data, "commitment_id"));
        let unit = number(data, "unit") as u32;
        let mismatch = Some(E::StatementMismatch {
            field: StatementField::Layout,
        });
        let mut stale = id;
        stale[0] ^= 1;
        assert_eq!(
            canonical::Commitment::bind(&layout, params, &statement, unit, root, stale).err(),
            mismatch
        );
        let mut other = statement;
        other[0] ^= 1;
        assert_eq!(
            canonical::Commitment::bind(&layout, params, &other, unit, root, id).err(),
            mismatch
        );
        assert_eq!(
            canonical::Commitment::bind(&layout, params, &statement, unit + 1, root, id).err(),
            mismatch
        );
        for bad in [
            tv_core::claims::AdviceLayoutParams {
                row_block: params.row_block + 1,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                mask_coeffs: params.mask_coeffs + 1,
                ..params
            },
        ] {
            assert_eq!(
                canonical::Commitment::bind(&layout, bad, &statement, unit, root, id).err(),
                mismatch
            );
        }
        for bad in [
            tv_core::claims::AdviceLayoutParams {
                log2_k: params.log2_k + 1,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                segments: params.segments + 1,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                log2_inv_rate: 2,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                mask_coeffs: 0,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                mask_coeffs: 65,
                ..params
            },
            tv_core::claims::AdviceLayoutParams {
                row_block: 0,
                ..params
            },
        ] {
            assert_eq!(
                canonical::Commitment::bind(&layout, bad, &statement, unit, root, stale).err(),
                Some(E::Internal)
            );
        }
    }
}

#[test]
fn canonical_preparation_rejects_bad_sources_arities_and_empty_combinations() {
    use tv_core::opening::OpeningError as E;
    let vectors = read_vectors("advice-opening/opening.json");
    let (claims, graph, selected) = micro_model();
    for data in list(&vectors, "openings") {
        let params = parameters(data);
        let layout = advice::Layout::derive(
            &graph,
            &selected,
            advice::Config::new(
                &claims.arch,
                claims.gadget_policy,
                params.log2_k,
                params.segments,
            ),
            number(data, "n") as u32,
        )
        .unwrap();
        let context = bind(&layout, data);
        let original = Obligations::read(data);
        let xi = fp2s(member(data, "xi_point"));
        let gamma = openings::fp2(member(data, "gamma"));
        let mut wrong_xi = xi.clone();
        wrong_xi.push(Fp2::ZERO);
        assert_eq!(
            context.prepare(&original.claims(), gamma, &wrong_xi).err(),
            Some(E::XiPoint {
                length: wrong_xi.len()
            })
        );
        let mut views = original.claims();
        views[0].pieces = &[];
        assert_eq!(context.prepare(&views, gamma, &xi).err(), Some(E::Internal));
        let mut bad = original.clone();
        bad.pieces[0][0].source = advice::Source::Tensor(lowering::TensorId(u32::MAX));
        assert_eq!(
            context.prepare(&bad.claims(), gamma, &xi).err(),
            Some(E::Internal)
        );
        let mut bad = original.clone();
        bad.points[0].push(Fp2::ZERO);
        assert_eq!(
            context.prepare(&bad.claims(), gamma, &xi).err(),
            Some(E::Internal)
        );
    }
}

#[test]
fn original_values_piece_weights_and_claim_order_are_bound() {
    use tv_core::opening::OpeningError as E;
    let vectors = read_vectors("advice-opening/opening.json");
    let data = list(&vectors, "openings")
        .iter()
        .find(|case| number(case, "n") == 3)
        .unwrap();
    let case = Case::from_json(data);
    let (claims, graph, selected) = micro_model();
    let params = parameters(data);
    let layout = advice::Layout::derive(
        &graph,
        &selected,
        advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            params.log2_k,
            params.segments,
        ),
        3,
    )
    .unwrap();
    let context = bind(&layout, data);
    let original = Obligations::read(data);
    let gamma = openings::fp2(member(data, "gamma"));
    for group in [0, 15] {
        let index = list(&list(data, "groups")[group], "claim_indices")[0]
            .as_u64()
            .unwrap() as usize;
        let mut bad = original.clone();
        bad.values[index] += Fp2::ONE;
        let views = bad.claims();
        let prepared = context.prepare(&views, gamma, &case.xi_point).unwrap();
        assert_eq!(
            check(&prepared, &case.combined, &case),
            Err(E::Evaluation { group: Some(group) })
        );
    }
    let mut bad = original.clone();
    bad.pieces[0][0].coefficient += Fp2::ONE;
    let views = bad.claims();
    let prepared = context.prepare(&views, gamma, &case.xi_point).unwrap();
    assert_eq!(
        check(&prepared, &case.combined, &case),
        Err(E::Proximity { column: 1 })
    );
    let first = list(&list(data, "groups")[0], "claim_indices");
    let a = first[0].as_u64().unwrap() as usize;
    let b = first[1].as_u64().unwrap() as usize;
    let mut views = original.claims();
    views.swap(a, b);
    let prepared = context.prepare(&views, gamma, &case.xi_point).unwrap();
    assert_eq!(
        check(&prepared, &case.combined, &case),
        Err(E::Evaluation { group: Some(0) })
    );
    let mut views = original.claims();
    views.remove(b);
    let prepared = context.prepare(&views, gamma, &case.xi_point).unwrap();
    assert_eq!(
        check(&prepared, &case.combined, &case),
        Err(E::Evaluation { group: Some(0) })
    );
}

#[test]
fn one_combination_cannot_mix_distinct_column_sub_points() {
    use tv_core::opening::OpeningError;
    let vectors = read_vectors("advice-opening/opening.json");
    let data = list(&vectors, "openings")
        .iter()
        .find(|case| number(case, "n") == 3)
        .unwrap();
    let (claims, graph, selected) = micro_model();
    let params = parameters(data);
    let layout = advice::Layout::derive(
        &graph,
        &selected,
        advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            params.log2_k,
            params.segments,
        ),
        3,
    )
    .unwrap();
    let context = bind(&layout, data);
    let mut candidates = Vec::new();
    for tile in layout.tiles() {
        for column in 0..tile.columns() {
            let location = tile.location(column).unwrap();
            let arity = if matches!(tile.key(), advice::TileKey::Attention(..)) {
                1 + if column == 0 { 4 } else { 2 }
            } else {
                location
                    .domain
                    .shape()
                    .iter()
                    .map(|n| n.next_power_of_two().ilog2())
                    .sum()
            };
            let point: Vec<_> = (0..arity)
                .map(|i| Fp2::new(tv_field::Fp::reduce(17 + u64::from(i)), tv_field::Fp::ONE))
                .collect();
            let mapped = layout
                .claim(location.source, &point)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            candidates.push((location.source, point, mapped.point[..6].to_vec()));
        }
    }
    let pair = candidates
        .iter()
        .enumerate()
        .find_map(|(i, a)| {
            candidates
                .iter()
                .skip(i + 1)
                .find(|b| a.1.len() == b.1.len() && a.2 != b.2)
                .map(|b| (a, b))
        })
        .expect("equal natural arities can have distinct canonical column points");
    let pieces = [
        canonical::Piece {
            source: pair.0.0,
            coefficient: Fp2::ONE,
        },
        canonical::Piece {
            source: pair.1.0,
            coefficient: Fp2::ONE,
        },
    ];
    let claim = canonical::Claim {
        point: &pair.0.1,
        pieces: &pieces,
        value: Fp2::ZERO,
    };
    let xi = fp2s(member(data, "xi_point"));
    assert_eq!(
        context.prepare(&[claim], Fp2::ONE, &xi).err(),
        Some(OpeningError::Internal)
    );
}

#[test]
fn zero_valued_committed_rows_are_still_authenticated_in_every_segment() {
    use openings::reference::{ClaimSpec, GroupSpec, Matrix, to_fe};
    use tv_testkit::sumcheck::Fe;
    let vectors = read_vectors("advice-opening/opening.json");
    let data = list(&vectors, "openings")
        .iter()
        .find(|case| number(case, "n") == 5)
        .unwrap();
    let original_case = Case::from_json(data);
    let rows = original_case.segments.iter().map(Vec::len).sum();
    let matrix = Matrix {
        params: original_case.params,
        row_vars: original_case.row_vars,
        segments: original_case.segments.clone(),
        values: vec![vec![Fe::ZERO; 64]; rows],
        masks: vec![vec![Fe::ZERO; 4]; rows],
        codewords: vec![vec![Fe::ZERO; 128]; rows],
    };
    let groups: Vec<_> = original_case
        .groups
        .iter()
        .map(|group| GroupSpec {
            column_point: group.column_point.iter().copied().map(to_fe).collect(),
            claims: group
                .claims
                .iter()
                .map(|claim| ClaimSpec {
                    row_point: claim.row_point.iter().copied().map(to_fe).collect(),
                    gamma: to_fe(claim.gamma),
                })
                .collect(),
        })
        .collect();
    // The independent test prover supplies fresh wire messages for an all-zero codeword matrix.
    let xi: Vec<_> = original_case.xi_point.iter().copied().map(to_fe).collect();
    let proof = matrix.open(&groups, &xi, &original_case.queries);
    let (claims, graph, selected) = micro_model();
    let params = parameters(data);
    let layout = advice::Layout::derive(
        &graph,
        &selected,
        advice::Config::new(
            &claims.arch,
            claims.gadget_policy,
            params.log2_k,
            params.segments,
        ),
        5,
    )
    .unwrap();
    let context = canonical::Commitment::bind(
        &layout,
        params,
        &digest(member(data, "statement_digest")),
        number(data, "unit") as u32,
        proof.root,
        original_case.params.commitment_id,
    )
    .unwrap();
    let mut original = Obligations::read(data);
    original.values.fill(Fp2::ZERO);
    let views = original.claims();
    let prepared = context
        .prepare(
            &views,
            openings::fp2(member(data, "gamma")),
            &proof.xi_point,
        )
        .unwrap();
    assert_eq!(check(&prepared, &proof.combined, &proof), Ok(()));
    let mut missing = proof.clone();
    missing.columns[0].pop();
    assert_eq!(
        check(&prepared, &proof.combined, &missing),
        Err(tv_core::opening::OpeningError::MerklePath { column: 1 })
    );
}
