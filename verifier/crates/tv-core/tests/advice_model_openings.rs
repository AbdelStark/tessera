// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public micro-model variants exercise canonical opening boundaries without million-row tables.

mod claim_support;
mod openings;

use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::{advice, lowering, selection, validation};
use tv_field::Fp2;
use tv_testkit::{Json, read_vectors};

use openings::{Case, digest, fp2s, list, member, number};

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
        let rows: u64 = counts.iter().sum();
        assert_eq!(rows, number(expected, "rows"));
        let encoded = descriptor(expected, log_k, segments, rows);
        assert_eq!(
            encoded,
            tv_testkit::hex(claim_support::text(expected, "descriptor_tce"))
        );
        assert_eq!(
            tv_hash::hash(tv_hash::Domain::CommitmentId, &encoded),
            digest(member(expected, "commitment_id"))
        );
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
