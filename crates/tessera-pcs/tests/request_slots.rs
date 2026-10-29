// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Multi-request slot layout tests (RFC-0071 'The plan and the layout', RFC-0047).

use tessera_pcs::{
    AdviceLayout, AdviceLayoutCounter, Axis, DenseShape, LayoutParams, Padding, PcsError,
    TensorShape, TileShape,
};

fn sample_params(log2_k: u32, query_block: u32) -> LayoutParams {
    LayoutParams {
        log2_k,
        segments: 1,
        layers: 1,
        group: 2,
        query_block,
        max_context: 16_384,
    }
}

#[test]
fn single_request_batch_layout_matches_single_sequence_layout() {
    let params = sample_params(8, 4);
    for n in [1u32, 3, 4, 5, 11, 16] {
        let slot_len = n.next_power_of_two();
        let attn_single = params.attention(n).unwrap();
        let attn_batch = params.batch_attention(slot_len, &[n]).unwrap();
        assert_eq!(attn_batch, attn_single);

        let tiles = vec![
            TileShape {
                layer: None,
                values: vec![TensorShape::Dense(DenseShape::with_slots(
                    vec![Axis::new(u64::from(n)), Axis::new(64)],
                    Padding::Zero,
                    0,
                    slot_len,
                    vec![n],
                ))],
            },
            TileShape {
                layer: Some(0),
                values: (0..attn_single.blocks)
                    .flat_map(|block| {
                        [
                            TensorShape::Scores { block },
                            TensorShape::Dense(attn_single.row_shape(block, Padding::Zero)),
                        ]
                    })
                    .collect(),
            },
        ];
        let single = AdviceLayout::new(params, n, &tiles).unwrap();
        let batch = AdviceLayout::new_batch(params, slot_len, &[n], &tiles).unwrap();
        assert_eq!(batch, single);
    }
}

#[test]
fn slotted_dense_skips_inter_slot_zero_rows_when_token_axis_is_outermost() {
    // R = 3 (not a power of two), S = 8, lengths = [5, 2, 6], N = 24, padded to 32.
    // With k = 64 (`log2_k = 6`) and channels = 64 (`log2 = 6`), each token `τ` spans one whole
    // row of `[24, 64]`. Real token rows are `0..5`, `8..10`, `16..22`; inter-slot rows `5..8`,
    // `10..16`, `22..24` and axis-tail rows `24..32` are skipped (RFC-0071 'The plan and the
    // layout', RFC-0011 rule 4).
    let params = sample_params(6, 4);
    let lengths = vec![5, 2, 6];
    let slot_len = 8;
    let n_tok = 3 * slot_len;
    let dense_act = DenseShape::with_slots(
        vec![Axis::new(u64::from(n_tok)), Axis::new(64)],
        Padding::Zero,
        0,
        slot_len,
        lengths.clone(),
    );
    assert_eq!(dense_act.order(), &[0, 1]);
    let tiles = vec![TileShape {
        layer: Some(0),
        values: vec![TensorShape::Dense(dense_act)],
    }];
    let layout = AdviceLayout::new_batch(params, slot_len, &lengths, &tiles).unwrap();
    assert_eq!(layout.allocated_rows(), 32);
    assert_eq!(
        layout.tensor(0, 0).unwrap().nonzero(),
        &[0..5, 8..10, 16..22]
    );
    assert_eq!(layout.rows(), 5 + 2 + 6);
}

#[test]
fn power_of_two_requests_commit_slot_padding_when_another_padded_axis_is_outermost() {
    // R = 2 (a power of two), S = 8, lengths = [3, 2], N = 16 (also a power of two!).
    // 1. On `[16, 64]` (no padded axis), `order = [0, 1]` puts the token axis outermost, so
    //    inter-slot padded rows `3..8` and `10..16` ARE skipped.
    // 2. On `[16, 12]` (`12` is padded to `16`), RFC-0047 puts `order = [1, 0]`: the padded
    //    channel axis `12` sits outermost, and with `k = 8` (`log2_k = 3`, 2 rows per channel),
    //    both token halves `0..8` and `8..16` inside `c < 12` are committed (`24` rows), while
    //    channels `12..16` (`8` rows) are skipped (RFC-0071 'The plan and the layout').
    // 3. With `Padding::Constants`, all `32` rows are committed (RFC-0035).
    let params = LayoutParams {
        log2_k: 3,
        segments: 1,
        layers: 1,
        group: 2,
        query_block: 4,
        max_context: 16_384,
    };
    let lengths = vec![3, 2];
    let slot_len = 8;
    let act_pow2 = DenseShape::with_slots(
        vec![Axis::new(16), Axis::new(8)],
        Padding::Zero,
        0,
        slot_len,
        lengths.clone(),
    );
    let act_dff = DenseShape::with_slots(
        vec![Axis::new(16), Axis::new(12)],
        Padding::Zero,
        0,
        slot_len,
        lengths.clone(),
    );
    let act_const = DenseShape::with_slots(
        vec![Axis::new(16), Axis::new(12)],
        Padding::Constants,
        0,
        slot_len,
        lengths.clone(),
    );
    assert_eq!(act_pow2.order(), &[0, 1]);
    assert_eq!(act_dff.order(), &[1, 0]);

    let tiles = vec![TileShape {
        layer: Some(0),
        values: vec![
            TensorShape::Dense(act_pow2),
            TensorShape::Dense(act_dff),
            TensorShape::Dense(act_const),
        ],
    }];
    let layout = AdviceLayout::new_batch(params, slot_len, &lengths, &tiles).unwrap();
    assert_eq!(layout.tensor(0, 0).unwrap().nonzero(), &[0..3, 8..10]);
    let second_first = layout.tensor(0, 1).unwrap().first_row();
    assert_eq!(
        layout.tensor(0, 1).unwrap().nonzero(),
        core::slice::from_ref(&(second_first..second_first + 24))
    );
    let third_first = layout.tensor(0, 2).unwrap().first_row();
    assert_eq!(
        layout.tensor(0, 2).unwrap().nonzero(),
        core::slice::from_ref(&(third_first..third_first + 32))
    );
}

#[test]
fn batch_attention_scores_apply_causal_rule_per_slot_and_skip_padded_slots() {
    // R = 3 (padded to 4 slots), S = 16, lengths = [13, 3, 9], Q = 4 (`4` blocks per slot, `16`
    // total blocks). `log2_k = 5` (`k = 32`), `g = 2`, `Q' = 4` -> `free = 5 - 2 - 1 = 2` ->
    // `w = min(4, 2) = 2` (`4` keys per row, `4` score rows per block).
    let params = sample_params(5, 4);
    let lengths = vec![13, 3, 9];
    let slot_len = 16;
    let attn = params.batch_attention(slot_len, &lengths).unwrap();
    assert_eq!(
        (
            attn.positions,
            attn.padded,
            attn.queries,
            attn.blocks,
            attn.key_bits
        ),
        (48, 16, 4, 16, 2)
    );

    let mut values = Vec::new();
    for block in 0..attn.blocks {
        values.push(TensorShape::Scores { block });
        values.push(TensorShape::Dense(attn.row_shape(block, Padding::Zero)));
    }
    let tiles = vec![TileShape {
        layer: Some(0),
        values,
    }];
    let layout = AdviceLayout::new_batch(params, slot_len, &lengths, &tiles).unwrap();

    // Check counter agreement.
    let mut counter = AdviceLayoutCounter::new_batch(params, slot_len, &lengths).unwrap();
    counter
        .append(
            &[
                TensorShape::Scores { block: 0 },
                TensorShape::Dense(attn.row_shape(0, Padding::Zero)),
            ],
            u64::from(attn.blocks),
        )
        .unwrap();
    assert_eq!(counter.allocated_rows(), layout.allocated_rows());

    // Expected committed score rows per block (out of 4):
    // Slot 0 (n_0 = 13, ceil(13/4) = 4): blocks 0..4 -> 1, 2, 3, 4 rows; held queries 4, 4, 4, 1.
    // Slot 1 (n_1 = 3, ceil(3/4) = 1): block 0 -> 1 row (held 3), blocks 1..4 -> 0 rows (held 0).
    // Slot 2 (n_2 = 9, ceil(9/4) = 3): blocks 0..3 -> 1, 2, 3 rows (held 4, 4, 1), block 3 -> 0.
    // Slot 3 (padded slot): blocks 0..4 -> 0 rows, held 0.
    let expected_score_rows = [1u64, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0];
    let expected_held = [4u32, 4, 4, 1, 3, 0, 0, 0, 4, 4, 1, 0, 0, 0, 0, 0];
    for block in 0..attn.blocks {
        assert_eq!(attn.held_queries(block), expected_held[block as usize]);
        let score_place = layout.tensor(0, (2 * block) as usize).unwrap();
        let row_place = layout.tensor(0, (2 * block + 1) as usize).unwrap();
        let want = expected_score_rows[block as usize];
        if want == 0 {
            assert!(score_place.nonzero().is_empty(), "block {block} scores");
            assert!(row_place.nonzero().is_empty(), "block {block} row_shape");
        } else {
            assert_eq!(
                score_place.nonzero(),
                core::slice::from_ref(&(score_place.first_row()..score_place.first_row() + want))
            );
            assert_eq!(
                row_place.nonzero(),
                core::slice::from_ref(&(row_place.first_row()..row_place.first_row() + 1))
            );
        }
    }
}

#[test]
fn oversized_batch_layout_is_refused_with_layout_error() {
    let params = sample_params(6, 4);
    // 2^20 requests of slot_len = 8192 -> padded_tokens = 2^33 > MAX_AXIS_LEN (2^32).
    let huge = vec![8_192u32; 1 << 20];
    assert!(matches!(
        params.batch_attention(8_192, &huge),
        Err(PcsError::Layout { .. })
    ));
}
