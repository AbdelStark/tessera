// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Invert the preserved scalar scatter, including public padding and RFC-0034/0047 order.
use tessera_pcs::{
    AdviceLayout, Axis, DenseShape, LayoutParams, Padding, PcsError, Placement, TensorShape,
    TileShape,
};

fn params(log2_k: u32) -> LayoutParams {
    LayoutParams {
        log2_k,
        layers: 1,
        segments: 1,
        group: 1,
        query_block: 1,
        max_context: 16,
    }
}
fn dense(shape: &[u32]) -> TensorShape {
    TensorShape::Dense(DenseShape::new(
        shape.iter().map(|&d| Axis::new(u64::from(d))).collect(),
        Padding::Zero,
    ))
}
fn tiles(shape: TensorShape) -> Vec<TileShape> {
    vec![TileShape {
        layer: Some(0),
        values: vec![dense(&[]), shape],
    }]
}

fn compare(place: &Placement, shape: &[u32], log2_k: u32) {
    let width = 1usize << log2_k;
    let mut scattered = vec![None; place.row_count() as usize * width];
    let size: usize = shape.iter().map(|&d| d as usize).product();
    for flat in 0..size {
        let mut coords = vec![0; shape.len()];
        let mut rest = flat;
        for (c, &len) in coords.iter_mut().zip(shape).rev() {
            *c = (rest % len as usize) as u64;
            rest /= len as usize;
        }
        let (row, col) = place.locate(&coords).unwrap();
        let slot = &mut scattered[(row - place.first_row()) as usize * width + col as usize];
        assert_eq!(slot.replace(flat), None, "injective source map");
    }
    for row in place.first_row()..place.first_row() + place.row_count() {
        let start = (row - place.first_row()) as usize * width;
        let iter = place.row_indices(row, shape).unwrap();
        let covered = 1usize << place.vars().min(log2_k);
        assert_eq!(
            iter.len(),
            covered,
            "do not iterate the trailing row padding"
        );
        assert_eq!(iter.collect::<Vec<_>>(), scattered[start..start + covered]);
        assert!(
            scattered[start + covered..start + width]
                .iter()
                .all(Option::is_none)
        );
    }
}

#[test]
fn every_small_dense_shape_matches_scalar_scatter() {
    // All ranks 0..3, all axis lengths 1..5, with each possible row split.
    for rank in 0..=3u32 {
        for encoded in 0..5usize.pow(rank) {
            let mut rest = encoded;
            let shape: Vec<_> = (0..rank)
                .map(|_| {
                    let d = (rest % 5 + 1) as u32;
                    rest /= 5;
                    d
                })
                .collect();
            for log2_k in 0..=5 {
                let layout = AdviceLayout::new(params(log2_k), 1, &tiles(dense(&shape))).unwrap();
                let place = &layout.tensors()[1];
                compare(place, &shape, log2_k);
                let padded: Vec<_> = shape.iter().map(|d| d.next_power_of_two()).collect();
                compare(place, &padded, log2_k);
                if !shape.is_empty() {
                    let mut empty = shape.clone();
                    empty[0] = 0;
                    compare(place, &empty, log2_k);
                }
            }
        }
    }
}

#[test]
fn scores_match_scatter_for_every_key_split_and_partial_block() {
    for n in 1..=9u32 {
        for group in [1u32, 2, 4] {
            for query_block in [1u32, 2, 4] {
                for log2_k in (group * query_block).trailing_zeros()..=7 {
                    let mut p = params(log2_k);
                    p.group = group;
                    p.query_block = query_block;
                    let attention = p.attention(n).unwrap();
                    for block in 0..attention.blocks {
                        let layout =
                            AdviceLayout::new(p, n, &tiles(TensorShape::Scores { block })).unwrap();
                        let place = &layout.tensors()[1];
                        let q = query_block.min(n.next_power_of_two());
                        compare(place, &[group, q, n], log2_k);
                        compare(place, &[group, q, n.next_power_of_two()], log2_k);
                    }
                }
            }
        }
    }
}

#[test]
fn slotted_sources_keep_natural_indices_and_the_public_row_schedule() {
    let lengths = [5u32, 2, 6];
    for channels in [1, 3, 8, 12, 64] {
        for log2_k in [4, 6, 8] {
            let mut p = params(log2_k);
            p.query_block = 4;
            let shape = TensorShape::Dense(DenseShape::with_slots(
                vec![Axis::new(24), Axis::new(channels)],
                Padding::Zero,
                0,
                8,
                lengths.to_vec(),
            ));
            let layout = AdviceLayout::new_batch(p, 8, &lengths, &tiles(shape)).unwrap();
            compare(&layout.tensors()[1], &[24, channels as u32], log2_k);
            if channels == 64 && log2_k == 6 {
                let start = layout.tensors()[1].first_row();
                assert_eq!(
                    layout.tensors()[1].nonzero(),
                    &[
                        start..start + 5,
                        start + 8..start + 10,
                        start + 16..start + 22
                    ]
                );
            }
        }
    }
}

#[test]
fn row_shape_and_source_product_errors_are_typed() {
    let layout = AdviceLayout::new(params(2), 1, &tiles(dense(&[3, 5]))).unwrap();
    let place = &layout.tensors()[1];
    let error = |row, shape: &[u32]| place.row_indices(row, shape).err().unwrap();
    let first = place.first_row();
    assert!(first > 0);
    for row in [first - 1, first + place.row_count(), u64::MAX] {
        assert_eq!(
            error(row, &[3, 5]),
            PcsError::LayoutData {
                rule: "source row lies inside its tensor placement"
            }
        );
    }
    assert_eq!(
        error(first, &[3]),
        PcsError::ElementAxes {
            expected: 2,
            found: 1
        }
    );
    assert_eq!(error(first, &[5, 5]), PcsError::ElementRange { axis: 0 });
    assert_eq!(error(first, &[3, 9]), PcsError::ElementRange { axis: 1 });
    // The largest valid tensor spans 2^63 cells at k=2^31 and 2^32 rows.
    // Construct only metadata and sample its last row; never allocate its entries.
    let big = AdviceLayout::new(
        params(31),
        1,
        &[TileShape {
            layer: Some(0),
            values: vec![dense(&[1 << 31, 1 << 31, 2])],
        }],
    )
    .unwrap();
    let place = &big.tensors()[0];
    let row = place.first_row() + place.row_count() - 1;
    #[cfg(target_pointer_width = "64")]
    {
        let mut last = place.row_indices(row, &[1 << 31, 1 << 31, 2]).unwrap();
        assert_eq!(last.len(), 1usize << 31);
        assert_eq!(last.next(), Some(Some(((1usize << 32) - 1) << 31)));
    }
    #[cfg(target_pointer_width = "32")]
    assert_eq!(
        place.row_indices(row, &[1 << 31, 1 << 31, 2]).err(),
        Some(PcsError::LayoutData {
            rule: "source shape product fits usize"
        })
    );
}

#[test]
fn a_32_bit_axis_uses_widened_indices_and_retains_length_padding() {
    // Both orientations include one zero-bit axis and the largest u32 source dimension.
    // Only public metadata is allocated, even though the tensor spans 2^32 rows.
    for shape in [[u32::MAX, 1], [1, u32::MAX]] {
        let layout = AdviceLayout::new(
            params(0),
            1,
            &[TileShape {
                layer: Some(0),
                values: vec![dense(&shape)],
            }],
        )
        .unwrap();
        let place = &layout.tensors()[0];
        assert_eq!(place.row_count(), 1u64 << 32);
        let row = place.first_row() + u64::from(u32::MAX);
        assert_eq!(
            place.row_indices(row - 1, &shape).unwrap().next(),
            Some(Some(u32::MAX as usize - 1))
        );
        assert_eq!(place.row_indices(row, &shape).unwrap().next(), Some(None));
    }
}

#[test]
fn empty_sources_need_no_addressable_trailing_stride() {
    let layout = AdviceLayout::new(
        params(1),
        1,
        &[TileShape {
            layer: Some(0),
            values: vec![dense(&[1, 1 << 31, 2])],
        }],
    )
    .unwrap();
    let place = &layout.tensors()[0];
    for row in [0, place.row_count() / 2, place.row_count() - 1] {
        let indices = place
            .row_indices(place.first_row() + row, &[0, 1 << 31, 2])
            .unwrap();
        assert_eq!(indices.collect::<Vec<_>>(), vec![None, None]);
    }
}
