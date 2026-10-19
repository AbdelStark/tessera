// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0040's public layouts against the independent oracle, before opening verification.

mod claim_support;

use claim_support::{golden, integer, list, member, text};
use tv_core::weights::{LayoutError, Matrix, derive};
use tv_testkit::{hex, read_vectors};

#[test]
fn all_weight_layouts_and_descriptors_match_the_preexisting_integer_oracle() {
    let vectors = read_vectors("weights/weights.json");
    let mut cases = 0;
    for model in list(&vectors, "models") {
        let name = text(model, "name");
        let mut claims = if name == "synthetic" {
            let mut c = golden("tiny-2-servable");
            let arch = member(model, "arch");
            let get = |key| u32::try_from(integer(member(arch, key))).unwrap();
            c.arch.n_layers = get("n_layers");
            c.arch.d_model = get("d_model");
            c.arch.n_heads = get("n_heads");
            c.arch.n_kv_heads = get("n_kv_heads");
            c.arch.d_head = get("d_head");
            c.arch.d_ff = get("d_ff");
            c.arch.vocab = get("vocab");
            // The synthetic oracle has no ChannelParams. Only public layout inputs are read;
            // this is a layout fixture, not a claim-list validation or protocol acceptance.
            c.quant.shift_class_sets.clear();
            c.quant.affine_sites.clear();
            c
        } else {
            golden(name)
        };
        let derived = derive(&claims, 0).unwrap();
        assert_eq!(derived.len(), 10);
        for ((layout, expected), matrix) in
            derived.iter().zip(list(model, "matrices")).zip(Matrix::ALL)
        {
            assert_eq!(layout.matrix(), matrix, "{name}");
            assert_eq!(
                i128::from(layout.log2_k()),
                integer(member(expected, "log2_k"))
            );
            assert_eq!(
                i128::from(layout.row_vars()),
                integer(member(expected, "row_vars"))
            );
            let rows = list(expected, "committed_rows");
            assert_eq!(
                usize::try_from(layout.committed_rows()).unwrap(),
                rows.len()
            );
            for (position, row) in rows.iter().enumerate() {
                assert_eq!(
                    layout.row_at(position as u64),
                    u64::try_from(integer(row)).ok()
                );
            }
            assert_eq!(layout.row_at(rows.len() as u64), None);
            assert_eq!(layout.row_at(u64::MAX), None);
            let bits = if rows.len() < 2 {
                0
            } else {
                rows.len().next_power_of_two().trailing_zeros()
            };
            assert_eq!(layout.proximity_vars(), bits);
            assert_eq!(
                layout.commitment_id().as_slice(),
                hex(text(expected, "commitment_id"))
            );
            cases += 1;
        }
        let refreshed = derive(&claims, 1).unwrap();
        for (old, new) in derived.iter().zip(refreshed) {
            assert_ne!(
                old.commitment_id(),
                new.commitment_id(),
                "epoch binds every matrix"
            );
        }
        claims.layout.advice.row_block = 1;
        for (old, new) in derived.iter().zip(derive(&claims, 0).unwrap()) {
            assert_ne!(
                old.commitment_id(),
                new.commitment_id(),
                "B binds every matrix"
            );
        }
    }
    assert_eq!(cases, 50);
}

#[test]
fn live_row_positions_skip_structural_gaps_and_keep_partial_output_rows() {
    let mut claims = golden("tiny-2-servable");
    claims.arch.n_layers = 3;
    claims.arch.d_model = 5;
    claims.arch.n_heads = 1;
    claims.arch.n_kv_heads = 1;
    claims.arch.d_head = 5;
    claims.arch.d_ff = 5;
    claims.arch.vocab = 5;
    claims.layout.weights.log2_k = 3;
    claims.quant.shift_class_sets.clear();
    claims.quant.affine_sites.clear();
    let layouts = derive(&claims, 0).unwrap();
    let q = &layouts[1];
    assert_eq!(q.row_vars(), 5); // 8 output rows per padded layer, 4 layers.
    assert_eq!(q.committed_rows(), 15);
    assert_eq!(
        (0..15).map(|p| q.row_at(p).unwrap()).collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20]
    );
    assert_eq!(q.proximity_vars(), 4);
    let cp = &layouts[9];
    assert_eq!(cp.matrix(), Matrix::ChannelParams);
    assert_eq!(cp.committed_rows(), 0);
    assert_eq!(cp.proximity_vars(), 0);
    assert_eq!(cp.row_at(0), None);
}

#[test]
fn invalid_own_layout_sizes_are_inconclusive_without_panics() {
    let claims = golden("tiny-2-servable");
    for kind in 0..5 {
        let mut c = claims.clone();
        match kind {
            0 => c.layout.advice.row_block = 0,
            1 => c.layout.weights.log2_inv_rate = 1,
            2 => c.layout.weights.log2_k = 64,
            3 => c.arch.d_model = 0,
            _ => c.arch.n_layers = 0,
        }
        assert_eq!(derive(&c, 0), Err(LayoutError::Internal));
    }
    assert_eq!(LayoutError::Internal.code(), "ERR-INTERNAL");
    assert_eq!(LayoutError::Memory.code(), "ERR-RES-MEMORY");
}

#[test]
fn the_405b_row_map_is_compact_and_uses_the_documented_message_widths() {
    let mut c = golden("tiny-2-servable");
    c.arch.n_layers = 126;
    c.arch.d_model = 16_384;
    c.arch.n_heads = 128;
    c.arch.n_kv_heads = 8;
    c.arch.d_head = 128;
    c.arch.d_ff = 53_248;
    c.arch.vocab = 128_256;
    c.layout.weights.log2_k = 20;
    c.quant.shift_class_sets.clear();
    c.quant.affine_sites.clear();
    let layouts = derive(&c, 0).unwrap();
    // (row variables, committed rows), from RFC-0040's integer shape formulas.
    let expected = [
        (11, 2_048),
        (15, 32_256),
        (11, 2_016),
        (11, 2_016),
        (15, 32_256),
        (17, 104_832),
        (17, 104_832),
        (17, 129_024),
        (11, 2_004),
    ];
    for (layout, (row_vars, count)) in layouts.iter().zip(expected) {
        assert_eq!(layout.log2_k(), 20);
        assert_eq!(layout.row_vars(), row_vars);
        assert_eq!(layout.committed_rows(), count);
        assert!(layout.row_at(count - 1).is_some());
        assert_eq!(layout.row_at(count), None);
    }
    let gate = &layouts[5];
    assert_eq!(gate.row_at(831), Some(831));
    assert_eq!(gate.row_at(832), Some(1_024));
    assert_eq!(gate.row_at(104_831), Some(128_831));
}

#[test]
fn maximum_codeword_width_is_inclusive_and_oversize_arithmetic_is_inconclusive() {
    let mut c = golden("tiny-2-servable");
    c.arch.n_layers = 1;
    c.arch.d_model = 65_536;
    c.arch.n_heads = 512;
    c.arch.n_kv_heads = 8;
    c.arch.d_head = 128;
    c.arch.d_ff = 65_536;
    c.arch.vocab = 65_536;
    c.quant.shift_class_sets.clear();
    c.quant.affine_sites.clear();
    c.layout.weights.log2_k = 30;
    let at_limit = derive(&c, 0).expect("4k = 2^32 is admitted without materializing rows");
    assert_eq!(at_limit[0].log2_k(), 30);
    assert_eq!(at_limit[0].committed_rows(), 4);
    c.layout.weights.log2_k = 31;
    assert_eq!(derive(&c, 0), Err(LayoutError::Internal));
    c.arch.n_layers = u32::MAX;
    c.arch.d_model = u32::MAX;
    c.arch.n_heads = u32::MAX;
    c.arch.n_kv_heads = u32::MAX;
    c.arch.d_head = u32::MAX;
    c.arch.d_ff = u32::MAX;
    c.arch.vocab = u32::MAX;
    assert_eq!(derive(&c, u32::MAX), Err(LayoutError::Internal));
}

#[test]
fn several_channel_sites_in_one_row_are_committed_once() {
    use claim_support::at;
    use tv_core::claims::Site;
    let mut c = golden("tiny-2-servable");
    c.arch.n_layers = 3;
    c.arch.d_model = 5;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 5;
    c.arch.d_ff = 9;
    c.arch.vocab = 7;
    c.layout.weights.log2_k = 9;
    c.quant.shift_class_sets = vec![
        (at(0, Site::RopeQ), vec![0, 1]),
        (at(2, Site::RopeQ), vec![0, 1]),
    ];
    c.quant.affine_sites = vec![(at(1, Site::MlpResidual), 2)];
    let layouts = derive(&c, 0).unwrap();
    let cp = &layouts[9];
    // Live Q and Down have five outputs: C=8, not Gate's unused width16. The tensor is
    // 16*4*8=512 entries in one row, containing all three sites, without duplicated rows.
    assert_eq!(cp.log2_k(), 9);
    assert_eq!(cp.row_vars(), 0);
    assert_eq!(cp.committed_rows(), 1);
    assert_eq!(cp.row_at(0), Some(0));
    assert_eq!(cp.row_at(1), None);
    c.quant.shift_class_sets = vec![(at(0, Site::RopeQ), vec![0])];
    c.quant.affine_sites.clear();
    let layouts = derive(&c, 0).unwrap();
    assert_eq!(
        layouts[9].committed_rows(),
        0,
        "one shift class needs no secret class vector"
    );
}
