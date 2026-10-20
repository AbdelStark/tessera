// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact request accounting and the same teacher-forcing OneHot, RFC-0017.
use std::cell::Cell;
use tessera_codec::error::{ErrorCode, RequestId, TokenPosition};
use tessera_codec::{DecodeLimits, SecretBuf, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower};
use tessera_field::{Goldilocks as G, PrimeField};
use tessera_gadgets::{OneHot, OneHotWitness};
use tessera_ir::IntegerModel;
use tessera_scoring::{ExactGreedy, GreedyInputs};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};

fn model(n: u32) -> Lowering {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/vectors/models/tiny-2-servable.d"
    ))
    .unwrap();
    let mut model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    // Metadata-only cases use a ragged vocabulary; actual model proofs use the
    // unmodified fixture separately. No secret weights enter this lowering.
    model.arch.vocab = 5;
    lower(PublicModel::of(&model), n).unwrap()
}
fn table() -> GeneratedTable {
    GeneratedTable::new(TableId(700), TableKind::Range { bits: 29 }).unwrap()
}
fn inputs(values: &[i64], token: i64) -> GreedyInputs {
    let h = OneHot::<G>::new(5).unwrap().witness(token, true).unwrap();
    let logits = (0..8)
        .map(|v| G::from_i64(*values.get(v).unwrap_or(&0)))
        .collect();
    GreedyInputs {
        logits,
        next_token: h,
    }
}

#[test]
fn zero_bound_binds_every_output_to_the_original_model_rows() {
    let lowering = model(5);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    let request = scorer.request(RequestId(77), 2, 3).unwrap();
    assert_eq!(request.padded_rows(), 4);
    for i in 0..3 {
        let row = request.row(i).unwrap().unwrap();
        assert_eq!(row.logits, lowering.logits);
        assert_eq!(row.one_hot, lowering.one_hot);
        assert_eq!(row.tokens, lowering.tokens);
        assert_eq!(row.logit_position, 1 + i as u32);
        assert_eq!(row.token_position, 2 + i as u32);
    }
    assert!(request.row(3).unwrap().is_none());
    assert_eq!(request.row(4).unwrap_err().code(), ErrorCode::Usage);
    let rows = [
        inputs(&[-2, -2, -4, -9, -3], 0),
        inputs(&[0, 7, 7, 3, 1], 1),
        inputs(&[0, 0, 0, 0, 1], 4),
    ];
    let certificate = request.witness(rows).unwrap();
    assert_eq!(certificate.uhat(), 0);
    assert_eq!(
        certificate.maxima(),
        &[
            G::from_i64(-2),
            G::from_i64(7),
            G::from_i64(1),
            G::from_i64(0)
        ]
    );
    // The relation borrows the supplied existing indicator, not a replacement.
    let row = inputs(&[0, 7, 7, 3, 1], 1);
    let prepared = certificate.prepare_row(1, &row).unwrap();
    assert!(std::ptr::eq(prepared.next_token(), &row.next_token));
    assert_eq!(prepared.sources().logit_position, 2);
    assert_eq!(prepared.sources().token_position, 3);
}

#[test]
fn a_wrong_output_or_later_tie_is_exact_mismatch_after_all_rows_are_processed() {
    let lowering = model(5);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    for (bad_row, token) in [(0, 1), (1, 2), (2, 3)] {
        let visited = Cell::new(0);
        let rows = (0..3).map(|i| {
            visited.set(visited.get() + 1);
            let values = [-2, -2, -5, -7, -8];
            inputs(&values, if i == bad_row { token } else { 0 })
        });
        let error = scorer
            .request(RequestId(77), 2, 3)
            .unwrap()
            .witness(rows)
            .unwrap_err();
        assert_eq!(visited.get(), 3);
        assert_eq!(error.code(), ErrorCode::WitnessExactMismatch);
        assert_eq!(error.context().request, Some(RequestId(77)));
        assert_eq!(error.context().position, Some(TokenPosition(2 + bad_row)));
        assert!(!format!("{error:?} {error}").contains("-8"));
    }
}

#[test]
fn first_mismatch_is_deterministic_and_discloses_no_logit_or_token_value() {
    let lowering = model(3);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    let error = scorer
        .request(RequestId(77), 1, 2)
        .unwrap()
        .witness([
            inputs(&[111_111_111, -9, -9, -9, -9], 4),
            inputs(&[2, 3, 4, 5, 6], 0),
        ])
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::WitnessExactMismatch);
    assert_eq!(error.context().position, Some(TokenPosition(1)));
    assert!(!format!("{error:?} {error}").contains("111111111"));
    let boundary: tessera_codec::error::TesseraError = error.into();
    assert_eq!(boundary.code, ErrorCode::WitnessExactMismatch);
    assert_eq!(boundary.exit_code(), 20);
    assert_eq!(boundary.context.request, Some(RequestId(77)));
}

#[test]
fn empty_outputs_have_no_certificate_rows_and_still_claim_zero() {
    let lowering = model(3);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    let certificate = scorer
        .request(RequestId(1), 3, 0)
        .unwrap()
        .witness([])
        .unwrap();
    assert_eq!(certificate.uhat(), 0);
    assert!(certificate.maxima().is_empty());
}

#[test]
fn malformed_metadata_or_row_counts_do_not_turn_into_completeness_errors() {
    let lowering = model(3);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    for (p, o) in [(0, 3), (1, 1), (u32::MAX, 1)] {
        assert_eq!(
            scorer.request(RequestId(1), p, o).unwrap_err().code(),
            ErrorCode::Usage
        );
    }
    for rows in [vec![], vec![inputs(&[1, 0, 0, 0, 0], 0)]] {
        assert_eq!(
            scorer
                .request(RequestId(1), 1, 2)
                .unwrap()
                .witness(rows)
                .unwrap_err()
                .code(),
            ErrorCode::Usage
        );
    }
    assert_eq!(
        scorer
            .request(RequestId(1), 2, 1)
            .unwrap()
            .witness([inputs(&[1, 0, 0, 0, 0], 0), inputs(&[1, 0, 0, 0, 0], 0)])
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    let bad = GreedyInputs {
        logits: SecretBuf::filled(G::from_i64(0), 7),
        next_token: OneHot::<G>::new(5).unwrap().witness(0, true).unwrap(),
    };
    assert_eq!(
        scorer
            .request(RequestId(1), 2, 1)
            .unwrap()
            .witness([bad])
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    let bad = GreedyInputs {
        logits: SecretBuf::filled(G::from_i64(0), 8),
        next_token: OneHotWitness::from_column([G::from_i64(1); 8].into_iter().collect()),
    };
    assert_eq!(
        scorer
            .request(RequestId(1), 2, 1)
            .unwrap()
            .witness([bad])
            .unwrap_err()
            .code(),
        ErrorCode::Internal
    );
}

#[test]
fn matching_shapes_cannot_substitute_an_unrelated_tensor_for_the_bound_sources() {
    let lowering = model(3);
    let mut changed = lowering.clone();
    changed.logits = changed.one_hot;
    assert_eq!(
        ExactGreedy::new(&changed, table().spec())
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
    let mut changed = lowering;
    changed.one_hot = changed.logits;
    assert_eq!(
        ExactGreedy::new(&changed, table().spec())
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
}

#[test]
fn only_the_integer_lowest_maximum_passes_for_every_recorded_choice() {
    let lowering = model(2);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    let mut seed = 0x1270_0009_u64;
    for case in 0..65 {
        let mut values = [0i64; 5];
        for x in &mut values {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = (seed >> 32) as i64 % 257 - 128;
        }
        if case == 0 {
            values = [-(1 << 28) + 1, (1 << 28) - 1, (1 << 28) - 1, -1, 0]
        }
        let largest = *values.iter().max().unwrap();
        let expected = values.iter().position(|&x| x == largest).unwrap();
        for choice in 0..5 {
            let result = scorer
                .request(RequestId(2), 1, 1)
                .unwrap()
                .witness([inputs(&values, choice as i64)]);
            if choice == expected {
                assert_eq!(result.unwrap().uhat(), 0)
            } else {
                assert_eq!(result.unwrap_err().code(), ErrorCode::WitnessExactMismatch)
            }
        }
    }
}

#[test]
fn each_one_hot_premise_is_checked_before_a_zero_claim_is_created() {
    let lowering = model(2);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    for h in [[2, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0; 8]] {
        let input = GreedyInputs {
            logits: SecretBuf::filled(G::from_i64(0), 8),
            next_token: OneHotWitness::from_column(h.into_iter().map(G::from_i64).collect()),
        };
        assert_eq!(
            scorer
                .request(RequestId(2), 1, 1)
                .unwrap()
                .witness([input])
                .unwrap_err()
                .code(),
            ErrorCode::Internal
        );
    }
    for bad in [-(1 << 28), 1 << 28] {
        let mut input = inputs(&[0; 5], 0);
        input.logits[0] = G::from_i64(bad);
        assert_eq!(
            scorer
                .request(RequestId(2), 1, 1)
                .unwrap()
                .witness([input])
                .unwrap_err()
                .code(),
            ErrorCode::Internal
        );
    }
}

#[test]
fn replay_retains_the_fixed_certificate_and_cannot_replace_the_indicator() {
    let lowering = model(2);
    let scorer = ExactGreedy::new(&lowering, table().spec()).unwrap();
    let certificate = scorer
        .request(RequestId(2), 1, 1)
        .unwrap()
        .witness([inputs(&[7, 7, 0, 0, 0], 0)])
        .unwrap();
    for changed in [inputs(&[7, 7, 0, 0, 0], 1), inputs(&[8, 7, 0, 0, 0], 0)] {
        let e = certificate.prepare_row(0, &changed).unwrap_err();
        assert_eq!(e.code(), ErrorCode::WitnessExactMismatch);
        assert_eq!(e.context().position, Some(TokenPosition(1)));
    }
    assert_eq!(
        certificate
            .prepare_row(1, &inputs(&[7, 7, 0, 0, 0], 0))
            .unwrap_err()
            .code(),
        ErrorCode::Usage
    );
}
