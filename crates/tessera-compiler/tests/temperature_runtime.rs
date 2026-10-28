// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runtime `Exp_T` derivation, `vocab_exp_scale` cancellation and bounds, and RFC-0058 column
//! and request schemas (`TemperaturePlan`, #152).

use std::fs;
use std::path::Path;

use tessera_codec::{Ascii, DecodeLimits, Rational, encode, envelope, schema};
use tessera_compiler::{
    CdfAdviceColumn, CompareColumn, CompileError, MAX_BATCH_TEMPERATURES, PublicModel,
    TemperaturePlan, TemperatureRequestSpec, analyze, cdf_columns_and_requests, claim_list, lower,
    select, vocab_exp_scale,
};
use tessera_field::{Field, Goldilocks};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};
use tessera_tables::{GeneratedTable, Table};

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
fn vocab_exp_scale_cancels_powers_of_two_and_enforces_temperature_and_rational_bounds() {
    // Exact power-of-two cancellation on positive, zero, and negative logits_exponent.
    assert_eq!(
        vocab_exp_scale(4, Rational::new(8, 3).unwrap()).unwrap(),
        Rational::new(6, 1).unwrap()
    );
    assert_eq!(
        vocab_exp_scale(0, Rational::new(5, 7).unwrap()).unwrap(),
        Rational::new(7, 5).unwrap()
    );
    assert_eq!(
        vocab_exp_scale(-5, Rational::new(3, 16).unwrap()).unwrap(),
        Rational::new(1, 6).unwrap()
    );
    // Exact upper boundary T = 4 is accepted; tiny positive T is accepted.
    assert_eq!(
        vocab_exp_scale(-2, Rational::new(4, 1).unwrap()).unwrap(),
        Rational::new(1, 16).unwrap()
    );
    assert_eq!(
        vocab_exp_scale(-3, Rational::new(1, 1000).unwrap()).unwrap(),
        Rational::new(125, 1).unwrap()
    );

    // T <= 0 or T > 4 is refused as CompileError::Usage, including large-denominator edge cases
    // where 4 * den would overflow u64 if not widened to u128.
    for bad_t in [
        Rational::new(0, 1).unwrap(),
        Rational::new(-1, 2).unwrap(),
        Rational::new(5, 1).unwrap(),
        Rational::new(4001, 1000).unwrap(),
        Rational::new(i64::MAX, (i64::MAX as u64) / 4).unwrap(),
    ] {
        assert!(
            matches!(vocab_exp_scale(0, bad_t), Err(CompileError::Usage { .. })),
            "expected Usage refusal on T = {bad_t:?}"
        );
    }

    // Overflow beyond Rational's signed-64 numerator or unsigned-64 denominator is refused as
    // CompileError::Model.
    assert!(matches!(
        vocab_exp_scale(63, Rational::new(1, 1).unwrap()),
        Err(CompileError::Model { .. })
    ));
    assert!(matches!(
        vocab_exp_scale(-64, Rational::new(1, 1).unwrap()),
        Err(CompileError::Model { .. })
    ));
}

#[test]
fn cdf_columns_and_requests_match_rfc_0058_counts_and_order_across_widths() {
    assert!(matches!(
        cdf_columns_and_requests(0, 16, false),
        Err(CompileError::Model { .. })
    ));
    assert!(matches!(
        cdf_columns_and_requests((1 << 18) + 1, 16, false),
        Err(CompileError::Model { .. })
    ));
    assert!(matches!(
        cdf_columns_and_requests(16, 33, false),
        Err(CompileError::Model { .. })
    ));

    // Check RFC-0058's per-row counts across W = 1 ..= 51:
    // committed scalar columns = 1 + 2*(k - 1) + 12*(k + 2) = 14*k + 23 (79 at k = 4, W = 51);
    // range requests = 2 + 2*k + 12*(k + 2) = 14*k + 26 (82 at k = 4, W = 51).
    for (vocab, sigma, expected_k) in [
        (1u32, 0u8, 1usize),
        (16, 11, 1),
        (16, 12, 2),
        (1024, 24, 3),
        (1 << 18, 32, 4),
    ] {
        let (exact_cols, exact_reqs) = cdf_columns_and_requests(vocab, sigma, false).unwrap();
        assert_eq!(exact_cols.len(), 14 * expected_k + 23);
        assert_eq!(exact_reqs.len(), 14 * expected_k + 26);
        assert_eq!(exact_cols[0], CdfAdviceColumn::DrawLimb);

        let (bounded_cols, bounded_reqs) = cdf_columns_and_requests(vocab, sigma, true).unwrap();
        assert_eq!(bounded_cols.len(), exact_cols.len() + 1);
        assert_eq!(bounded_cols[0], CdfAdviceColumn::SampledIndicator);
        assert_eq!(&bounded_cols[1..], exact_cols.as_slice());
        assert_eq!(bounded_reqs, exact_reqs);

        // Verify RFC-0050 comparison array grouping: lower (false) before upper (true), and
        // LeftByte -> LeftCarry -> RightByte -> RightCarry -> Difference -> Borrow.
        let comp_start = 1 + 2 * (expected_k - 1);
        let m = expected_k + 2;
        let expected_order = [
            CompareColumn::LeftByte,
            CompareColumn::LeftCarry,
            CompareColumn::RightByte,
            CompareColumn::RightCarry,
            CompareColumn::Difference,
            CompareColumn::Borrow,
        ];
        let mut cursor = comp_start;
        for upper in [false, true] {
            for &col_kind in &expected_order {
                for limb in 0..m as u32 {
                    assert_eq!(
                        exact_cols[cursor],
                        CdfAdviceColumn::Comparison {
                            upper,
                            column: col_kind,
                            limb,
                        }
                    );
                    cursor += 1;
                }
            }
        }
        assert_eq!(cursor, exact_cols.len());
    }
}

#[test]
fn temperature_plan_groups_up_to_four_temperatures_preserves_claim_list_and_checks_zero_tail() {
    let model = tiny_model();
    let public = PublicModel::of(&model);
    let lowering = lower(public, 8).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let policy = GadgetPolicy::default();
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).unwrap();
    let params = ParamSetId(Ascii::new("test-small/1").unwrap());
    let claims = claim_list(public, &selection, &policy, TEST_SMALL, params).unwrap();
    let original_claims_bytes = encode(&claims);
    let base_table_count = claims.tables.len();

    let t1 = Rational::new(1, 1).unwrap();
    let t2 = Rational::new(7, 10).unwrap();
    let t3 = Rational::new(3, 2).unwrap();
    let t4 = Rational::new(4, 1).unwrap();

    // 5 requests across 4 distinct temperatures (including one shared temperature and one
    // prompt-only request).
    let requests = [
        TemperatureRequestSpec {
            prompt: 2,
            positions: 6,
            temperature: t1,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 3,
            positions: 5,
            temperature: t2,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 1,
            positions: 4,
            temperature: t1,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 4,
            positions: 4,
            temperature: t3,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 2,
            positions: 5,
            temperature: t4,
            bounded: true,
        },
    ];

    let plan = TemperaturePlan::derive(&lowering, &selection, &claims, &requests, 0).unwrap();
    // ClaimList wire bytes are completely untouched, and runtime selection keeps every model table
    // prefix intact.
    assert_eq!(encode(&claims), original_claims_bytes);
    assert_eq!(
        &plan.selection().tables[..base_table_count],
        claims.tables.as_slice()
    );
    // t3 had only a zero-output request (prompt == positions), so 3 active Exp_T tables are built,
    // with requests 0 and 2 sharing t1's table!
    assert_eq!(plan.temperatures().len(), 3);
    assert_eq!(plan.temperatures()[0].temperature, t1);
    assert_eq!(plan.temperatures()[0].requests, vec![0, 2]);
    assert_eq!(plan.temperatures()[0].output_tokens, 7);
    assert_eq!(plan.certificates().len(), 4);
    assert_eq!(
        plan.certificates()[0].exp_table,
        plan.certificates()[2].exp_table
    );
    assert_eq!(
        plan.certificates()[3].columns[0],
        CdfAdviceColumn::SampledIndicator
    );

    // Every derived Exp_T table regenerates cleanly and maps its last key K_0 to 0 (RFC-0030).
    for entry in plan.temperatures() {
        let spec = &plan.selection().tables[entry.table.0 as usize];
        let regen = GeneratedTable::from_spec(spec).unwrap();
        assert_eq!(
            regen.row(regen.row_count() - 1).unwrap().1,
            Goldilocks::ZERO
        );
    }

    // 5 distinct temperatures exceed MAX_BATCH_TEMPERATURES (4) and are refused as Usage.
    assert_eq!(MAX_BATCH_TEMPERATURES, 4);
    let five_temps: Vec<TemperatureRequestSpec> = [
        Rational::new(1, 2).unwrap(),
        Rational::new(1, 1).unwrap(),
        Rational::new(3, 2).unwrap(),
        Rational::new(2, 1).unwrap(),
        Rational::new(4, 1).unwrap(),
    ]
    .into_iter()
    .map(|temperature| TemperatureRequestSpec {
        prompt: 1,
        positions: 2,
        temperature,
        bounded: false,
    })
    .collect();
    assert!(matches!(
        TemperaturePlan::derive(&lowering, &selection, &claims, &five_temps, 0),
        Err(CompileError::Usage { .. })
    ));

    // Invalid request geometry (prompt == 0, prompt > positions, positions > max_context) refused.
    for bad_req in [
        TemperatureRequestSpec {
            prompt: 0,
            positions: 2,
            temperature: t1,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 5,
            positions: 4,
            temperature: t1,
            bounded: false,
        },
        TemperatureRequestSpec {
            prompt: 1,
            positions: claims.arch.max_context + 1,
            temperature: t1,
            bounded: false,
        },
    ] {
        assert!(matches!(
            TemperaturePlan::derive(&lowering, &selection, &claims, &[bad_req], 0),
            Err(CompileError::Usage { .. })
        ));
    }

    // When public logit spread is narrow (`[-2, 2]` -> `min_key_bits = 3`) and `scale = 1/64`
    // (`logits_exponent = -4`, `T = 4`), `Exp_T[7] != 0`, so `TemperaturePlan::derive` MUST step
    // `key_bits` strictly above `min_key_bits` until `Exp_T[2^{key_bits} - 1] = 0`, and refuse if
    // `max_log2_rows` caps out before that zero tail.
    let mut g = tessera_ir::GraphBuilder::new();
    let tok = g
        .input(tessera_ir::TensorDecl {
            shape: vec![4],
            ty: tessera_ir::IntType {
                bits: 32,
                signed: false,
            },
            range: tessera_ir::IntRange {
                lo: 0,
                hi: i128::from(claims.arch.vocab - 1),
            },
            kind: tessera_ir::NodeKind::Committed {
                commitment: tessera_ir::CommitmentClass::Advice,
            },
        })
        .unwrap();
    let oh = g
        .node(
            tessera_ir::Op::OneHot {
                vocab: claims.arch.vocab,
            },
            &[tok],
            vec![tessera_ir::TensorDecl {
                shape: vec![4, claims.arch.vocab],
                ty: tessera_ir::IntType {
                    bits: 8,
                    signed: false,
                },
                range: tessera_ir::IntRange { lo: 0, hi: 1 },
                kind: tessera_ir::NodeKind::Committed {
                    commitment: tessera_ir::CommitmentClass::Advice,
                },
            }],
        )
        .unwrap()[0];
    let act_in = g
        .input(tessera_ir::TensorDecl {
            shape: vec![4, 1],
            ty: tessera_ir::IntType {
                bits: 8,
                signed: false,
            },
            range: tessera_ir::IntRange { lo: 0, hi: 1 },
            kind: tessera_ir::NodeKind::Public,
        })
        .unwrap();
    let w_ref = tessera_ir::WeightRef {
        layer: 0,
        ty: tessera_ir::WeightType::LmHead,
    };
    let w_in = g.weight(w_ref, claims.arch.vocab, 1).unwrap();
    let narrow_logits = g
        .node(
            tessera_ir::Op::MatMulWeight { weight: w_ref },
            &[act_in, w_in],
            vec![tessera_ir::TensorDecl {
                shape: vec![4, claims.arch.vocab],
                ty: tessera_ir::IntType {
                    bits: 16,
                    signed: true,
                },
                range: tessera_ir::IntRange { lo: -128, hi: 127 },
                kind: tessera_ir::NodeKind::Virtual,
            }],
        )
        .unwrap()[0];
    let narrow_lowering = tessera_compiler::Lowering {
        graph: g.finish(),
        tables: Vec::new(),
        tokens: tok,
        one_hot: oh,
        logits: narrow_logits,
        ..lowering.clone()
    };
    let mut small_scale_claims = claims.clone();
    small_scale_claims.tables.clear();
    small_scale_claims.quant.logits_exponent = -4;
    let mut empty_sel = selection.clone();
    empty_sel.tables.clear();
    let narrow_req = [TemperatureRequestSpec {
        prompt: 1,
        positions: 3,
        temperature: t4,
        bounded: false,
    }];
    let wide_plan = TemperaturePlan::derive(
        &narrow_lowering,
        &empty_sel,
        &small_scale_claims,
        &narrow_req,
        0,
    )
    .unwrap();
    let zero_tail_bits = wide_plan.temperatures()[0].key_bits;
    assert!(
        zero_tail_bits > 8,
        "narrow logit spread (min_key_bits = 8) with scale 1/64 must step key_bits above 8 to reach Exp_T[K_0] = 0"
    );
    let regen_wide = GeneratedTable::from_spec(
        &wide_plan.selection().tables[wide_plan.temperatures()[0].table.0 as usize],
    )
    .unwrap();
    assert_eq!(
        regen_wide.row(regen_wide.row_count() - 1).unwrap().1,
        Goldilocks::ZERO
    );
    let mut cramped_claims = small_scale_claims;
    cramped_claims.gadget_policy.max_log2_rows = zero_tail_bits - 1;
    assert!(matches!(
        TemperaturePlan::derive(
            &narrow_lowering,
            &empty_sel,
            &cramped_claims,
            &narrow_req,
            0
        ),
        Err(CompileError::Model { .. })
    ));
}
