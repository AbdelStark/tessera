// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public selection view preserves the verifier's canonical derivation (RFC-0007).
mod claim_support;
use claim_support::{golden, goldens};
use tv_core::{lowering, ranges, selection, validation};

#[test]
fn derived_catalog_matches_golden_table_definitions_and_independent_ranges() {
    let mut count = 0;
    for g in goldens() {
        let c = golden(&g.name);
        let public = validation::check_public(&c).unwrap();
        for n in [1, 4, 5, c.arch.max_context] {
            let graph = lowering::lower(&public, n).unwrap();
            let derived = selection::derive(&graph, c.gadget_policy).unwrap();
            assert_eq!(derived.ranges, ranges::analyze(&graph).unwrap());
            assert_eq!(derived.tables.len(), c.tables.len());
            for (found, expected) in derived.tables.iter().zip(&c.tables) {
                assert_eq!(found.id, expected.id);
                assert_eq!(found.kind, expected.kind);
                assert_eq!(found.log2_rows, expected.log2_rows);
            }
            let expected: Vec<_> = graph
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    matches!(
                        node.op(),
                        lowering::Op::OneHot { .. }
                            | lowering::Op::Requant { .. }
                            | lowering::Op::TableMap { .. }
                            | lowering::Op::SoftmaxRow { .. }
                            | lowering::Op::RsqrtRow { .. }
                    )
                })
                .map(|(i, _)| lowering::NodeId(i as u32))
                .collect();
            assert_eq!(
                derived.choices.iter().map(|g| g.node).collect::<Vec<_>>(),
                expected
            );
            for choice in &derived.choices {
                for id in &choice.tables {
                    assert_eq!(derived.tables[id.0 as usize].id, *id);
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, 16);
}

#[test]
fn heterogeneous_affine_layers_keep_the_full_leaf_guard() {
    use claim_support::claims_of;
    use tv_core::claims::{QuantPoint, Site};
    use tv_core::sumcheck::Protocol;
    for site in [Site::MlpResidual, Site::Up] {
        for n in [1, 3, 5] {
            for maximum in [30, 31, 32] {
                let mut claims = claims_of("tiny-2-affine.d");
                if site == Site::Up {
                    claims.arch.d_ff = 1;
                }
                for layer in 0..claims.arch.n_layers {
                    let point = QuantPoint::Layer { layer, site };
                    claims.quant.shift_class_sets.retain(|(p, _)| *p != point);
                    claims.quant.affine_sites.retain(|(p, _)| *p != point);
                    claims
                        .quant
                        .affine_sites
                        .push((point, if layer == 0 { 2 } else { 12 }));
                }
                claims.quant.affine_sites.sort_by_key(|(p, _)| *p);
                claims.gadget_policy.max_log2_rows = maximum;
                let public = validation::check_public(&claims).unwrap();
                let graph = lowering::lower(&public, n).unwrap();
                let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
                assert!(!selected.choices.is_empty());
                tv_core::gadgets::Plan::derive(&graph, claims.gadget_policy).unwrap();
                tv_core::leaf::ModelSchedule::derive(&graph, claims.gadget_policy).unwrap();
                tv_core::leaf::split::Layout::derive(&graph, claims.gadget_policy).unwrap();
                // Protocol Two deliberately retains the old incompatible family selection.
                let old = lowering::lower_batch_for(&public, &[n], Protocol::Two).unwrap();
                let old_selection = selection::derive(&old, claims.gadget_policy).unwrap();
                assert_ne!(selected.tables, old_selection.tables);
                assert_eq!(
                    tv_core::leaf::split::Layout::derive(&old, claims.gadget_policy).err(),
                    Some(validation::ClaimsError::Internal)
                );
            }
        }
    }
}

#[test]
fn widened_envelope_binds_zero_to_the_selected_table_rows() {
    use claim_support::claims_of;
    use tv_core::claims::{QuantPoint, Site};
    use tv_core::gadgets::{GadgetError, Plan};
    use tv_core::lowering::Op;
    use tv_core::selection::Pattern;
    use tv_field::{Fp, Fp2};

    let mut claims = claims_of("tiny-2-affine.d");
    claims.arch.d_model = 2;
    claims.arch.n_heads = 1;
    claims.arch.n_kv_heads = 1;
    claims.arch.d_head = 2;
    claims.arch.d_ff = 6;
    claims.arch.vocab = 8;
    claims.arch.max_context = 8;
    claims.quant.residual_bits = 8;
    claims.quant.residual_exponent = claims
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    claims.quant.rsqrt_kappa = 2;
    claims.quant.exp_sigma = 3;
    for (_, shift) in &mut claims.quant.affine_sites {
        *shift = 24;
    }
    for layer in 0..claims.arch.n_layers {
        let point = QuantPoint::Layer {
            layer,
            site: Site::Up,
        };
        claims.quant.shift_class_sets.retain(|(p, _)| *p != point);
        claims.quant.affine_sites.retain(|(p, _)| *p != point);
        claims
            .quant
            .affine_sites
            .push((point, if layer == 0 { 25 } else { 2 }));
    }
    claims.quant.affine_sites.sort_by_key(|(p, _)| *p);
    claims.gadget_policy.max_log2_rows = 30;
    let graph = lowering::lower(&validation::check_public(&claims).unwrap(), 1).unwrap();
    let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
    let choice = selected
        .choices
        .iter()
        .find(|c| {
            let Op::Requant { shift, .. } = graph.node(c.node).unwrap().op() else {
                return false;
            };
            graph.shift_classes()[*shift as usize].as_slice() == [25]
        })
        .unwrap();
    assert_eq!(
        choice.pattern,
        Pattern::RequantWide {
            split: 24,
            middle: 8
        }
    );
    let input = graph.node(choice.node).unwrap().inputs()[0];
    let range = selected.ranges[input.0 as usize];
    let (lo, hi) = range.bounds().unwrap();
    assert!(lo >= -(1i128 << 31) && hi < (1i128 << 31));
    assert!(lo < -(1i128 << 30) || hi >= (1i128 << 30));
    let plan = Plan::derive(&graph, claims.gadget_policy).unwrap();
    let site = plan
        .requantizations()
        .map(Result::unwrap)
        .find(|s| s.node() == choice.node)
        .unwrap();
    // RFC-0076: the selected 33-bit envelope maps integer zero to top key 1,
    // sign key 2 and centered high key 2^9. The old 32-bit offset gives 1/2.
    let point = vec![Fp2::ZERO; 3];
    let values = vec![Fp2::ZERO; 5];
    let f = |n| Fp2::from(Fp::reduce(n));
    let mut lookups = vec![
        (f(0), f(0)),
        (f(0), f(0)),
        (f(1), f(0)),
        (f(2), f(0)),
        (f(1 << 9), f(0)),
    ];
    let checked = site.check(&point, &values, &lookups).unwrap();
    assert_eq!(checked.claims().count(), 5);
    assert_eq!(checked.lookups().count(), 5);
    lookups[2].0 = f(0);
    assert!(
        matches!(site.check(&point, &values, &lookups), Err(GadgetError::Rejected { gadget }) if gadget == site.gadget())
    );
    // The earlier leaf split must prescribe those same rows for zero sources.
    let sites = tv_core::leaf::split::model_sites(&graph, claims.gadget_policy).unwrap();
    let layer = sites
        .iter()
        .flat_map(|s| &s.layers)
        .find(|l| l.node == choice.node)
        .unwrap();
    assert_eq!(layer.lookups.len(), 5);
    assert_eq!(layer.lookups[2].key.constant, Fp::ONE);
    assert_eq!(layer.lookups[3].key.constant, Fp::reduce(2));
    assert_eq!(layer.lookups[4].key.constant, Fp::reduce(1 << 9));
}
