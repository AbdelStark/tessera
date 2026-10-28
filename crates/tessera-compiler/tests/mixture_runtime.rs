// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runtime `MixturePlan` derivation, `log2_columns_and_requests` schema and bounds, and
//! `ClaimList` table-ID preservation (#155, RFC-0017, RFC-0026, RFC-0050, RFC-0053, RFC-0058).

use std::fs;
use std::path::Path;

use tessera_codec::{Ascii, DecodeLimits, Rational, encode, envelope, schema};
use tessera_compiler::{
    CdfAdviceColumn, CompareColumn, CompileError, Log2AdviceColumn, MAX_BATCH_TEMPERATURES,
    MixtureDecoding, MixturePlan, MixtureRequestSpec, PublicModel, analyze, claim_list,
    log2_columns_and_requests, lower, select, vocab_exp_scale,
};
use tessera_field::{Field, Goldilocks};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};
use tessera_tables::{GeneratedTable, Table, TableKind};

const TEST_SMALL: CommitmentLayout = CommitmentLayout {
    advice: AdviceLayoutParams {
        log2_k: 10,
        log2_inv_rate: 1,
        mask_coeffs: 16,
        row_block: 4,
        segments: 1,
    },
    weights: WeightLayoutParams {
        log2_k: 8,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    },
};

fn tiny_model() -> IntegerModel {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d");
    let bytes = fs::read(path).expect("tiny-2-servable.d");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).expect("model")
}

#[test]
fn log2_columns_and_requests_enforces_domain_and_matches_rfc_0017_counts_and_order() {
    assert!(matches!(
        log2_columns_and_requests(0, 16),
        Err(CompileError::Model { .. })
    ));
    assert!(matches!(
        log2_columns_and_requests((1 << 18) + 1, 16),
        Err(CompileError::Model { .. })
    ));
    assert!(matches!(
        log2_columns_and_requests(16, 33),
        Err(CompileError::Model { .. })
    ));

    for (vocab, sigma, expected_kz, expected_kq, expected_m) in [
        (1u32, 0u8, 1usize, 2usize, 6usize),
        (16, 15, 2, 3, 6),
        (16, 16, 2, 3, 7),
        (1024, 24, 3, 3, 7),
        (1 << 18, 32, 4, 4, 7),
    ] {
        let (cols, reqs) = log2_columns_and_requests(vocab, sigma).unwrap();
        assert_eq!(
            cols.len(),
            6 + (expected_kz - 1) + (expected_kq - 1) + 3 + 6 * expected_m,
            "committed column count mismatch at vocab={vocab}, sigma={sigma}"
        );
        assert_eq!(
            reqs.len(),
            2 + expected_kz + expected_kq + 4 + 6 * expected_m,
            "lookup request count mismatch at vocab={vocab}, sigma={sigma}"
        );
        assert_eq!(
            &cols[..6],
            &[
                Log2AdviceColumn::MatchBit,
                Log2AdviceColumn::IntExp,
                Log2AdviceColumn::Pow2Val,
                Log2AdviceColumn::FracExp,
                Log2AdviceColumn::Exp2Val,
                Log2AdviceColumn::ScaledSelected,
            ]
        );
        assert_eq!(reqs[0], TableKind::Pow2 { max_exp: 63 });
        assert_eq!(
            reqs[1],
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits: 16
            }
        );

        // Verify RFC-0050 grouped comparison array order at the tail of `cols`.
        let cmp_tail = &cols[cols.len() - 6 * expected_m..];
        let expected_order = [
            CompareColumn::LeftByte,
            CompareColumn::LeftCarry,
            CompareColumn::RightByte,
            CompareColumn::RightCarry,
            CompareColumn::Difference,
            CompareColumn::Borrow,
        ];
        for (grp, &expected_col) in expected_order.iter().enumerate() {
            for limb in 0..expected_m {
                assert_eq!(
                    cmp_tail[grp * expected_m + limb],
                    Log2AdviceColumn::Comparison {
                        column: expected_col,
                        limb: limb as u32,
                    }
                );
            }
        }
    }
}

#[test]
fn mixture_plan_derives_greedy_and_temperature_tables_without_changing_claim_list() {
    let model = tiny_model();
    let public = PublicModel::of(&model);
    let lowering = lower(public, 6).expect("lower");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
    let params = ParamSetId(Ascii::new("test-small/1").unwrap());
    let claims = claim_list(public, &selection, &policy, TEST_SMALL, params).expect("claims");
    let original_claims_bytes = encode(&claims);

    let t07 = Rational::new(7, 10).unwrap();
    let t15 = Rational::new(3, 2).unwrap();
    let t40 = Rational::new(4, 1).unwrap();
    let requests = [
        MixtureRequestSpec {
            prompt: 2,
            positions: 5,
            decoding: MixtureDecoding::Greedy,
        },
        // Zero-output request: validated, consumes no certificate ordinal.
        MixtureRequestSpec {
            prompt: 4,
            positions: 4,
            decoding: MixtureDecoding::Temperature(t07),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 4,
            decoding: MixtureDecoding::Temperature(t07),
        },
        MixtureRequestSpec {
            prompt: 3,
            positions: 6,
            decoding: MixtureDecoding::Temperature(t15),
        },
        MixtureRequestSpec {
            prompt: 2,
            positions: 4,
            decoding: MixtureDecoding::Temperature(t40),
        },
    ];

    let plan = MixturePlan::derive(&lowering, &selection, &claims, &requests, 6, 0).expect("plan");
    assert_eq!(plan.eps_log2(), 6);
    assert_eq!(plan.temperatures().len(), MAX_BATCH_TEMPERATURES);
    assert_eq!(plan.certificates().len(), 4);

    // Every original model table ID, kind and digest is preserved untouched.
    assert_eq!(
        &plan.selection().tables[..claims.tables.len()],
        claims.tables.as_slice()
    );
    assert_eq!(encode(&claims), original_claims_bytes);

    // Greedy request uses Exp_1 (T = 1) and carries argmax_range_table; temperature requests carry
    // bounded CdfAdviceColumn starting with SampledIndicator.
    let c0 = &plan.certificates()[0];
    assert_eq!(c0.request_index, 0);
    assert_eq!(c0.decoding, MixtureDecoding::Greedy);
    assert!(c0.argmax_range_table.is_some());
    assert!(c0.argmax_range_bits.is_some());
    assert!(c0.cdf_columns.is_empty());
    assert!(!c0.log2_columns.is_empty());

    let c1 = &plan.certificates()[1];
    assert_eq!(c1.request_index, 2);
    assert_eq!(c1.decoding, MixtureDecoding::Temperature(t07));
    assert!(c1.argmax_range_table.is_none());
    assert_eq!(c1.cdf_columns[0], CdfAdviceColumn::SampledIndicator);

    // Every active Exp_T table has zero final row Exp_T[K_0] = 0 and matches vocab_exp_scale.
    for entry in plan.temperatures() {
        let expected_scale =
            vocab_exp_scale(claims.quant.logits_exponent, entry.temperature).unwrap();
        assert_eq!(entry.scale, expected_scale);
        let spec = &plan.selection().tables[entry.table.0 as usize];
        let regen = GeneratedTable::from_spec(spec).unwrap();
        assert_eq!(
            regen.row(regen.row_count() - 1).unwrap().1,
            Goldilocks::ZERO
        );
    }

    // Adversarial refusals:
    // 1. eps_log2 > 16
    assert!(matches!(
        MixturePlan::derive(&lowering, &selection, &claims, &requests, 17, 0),
        Err(CompileError::Usage { .. })
    ));
    // 2. > 4 distinct effective temperatures
    let five_temps = [
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Greedy,
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Temperature(t07),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Temperature(t15),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Temperature(t40),
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Temperature(Rational::new(1, 2).unwrap()),
        },
    ];
    assert!(matches!(
        MixturePlan::derive(&lowering, &selection, &claims, &five_temps, 6, 0),
        Err(CompileError::Usage { .. })
    ));
    // 3. Invalid request geometry
    for bad_req in [
        MixtureRequestSpec {
            prompt: 0,
            positions: 2,
            decoding: MixtureDecoding::Greedy,
        },
        MixtureRequestSpec {
            prompt: 5,
            positions: 4,
            decoding: MixtureDecoding::Greedy,
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: claims.arch.max_context + 1,
            decoding: MixtureDecoding::Greedy,
        },
        MixtureRequestSpec {
            prompt: 1,
            positions: 2,
            decoding: MixtureDecoding::Temperature(Rational::new(5, 1).unwrap()),
        },
    ] {
        assert!(matches!(
            MixturePlan::derive(&lowering, &selection, &claims, &[bad_req], 6, 0),
            Err(CompileError::Usage { .. })
        ));
    }
    // 4. max_log2_rows = 15 cannot hold Range { 16 } required by OneSidedLog2.
    let mut narrow_claims = claims.clone();
    narrow_claims.gadget_policy.max_log2_rows = 15;
    assert!(matches!(
        MixturePlan::derive(&lowering, &selection, &narrow_claims, &requests[..1], 6, 0),
        Err(CompileError::Model { .. })
    ));
}
