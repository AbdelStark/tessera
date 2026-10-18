// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The residual's route at its boundary, which the lowering vectors do not reach: RFC-0038 'The
//! residual's route' fuses the key when every layer's site has one shift of at most 62, as a
//! `pot` site of one class or an affine site, and takes two steps otherwise. The vectors hold
//! fused keys at shifts 3 and 18 and two steps at 63, so the shift 62 is checked here, in every
//! layer and in all but one, and for both forms of site.

mod claim_support;

use claim_support::{at, golden, make_affine, make_pot};
use tv_core::claims::{ClaimList, Site};
use tv_core::lowering::{Graph, Op, Route, lower};
use tv_core::validation::check_public;

fn lowered(claims: &ClaimList) -> Graph {
    let public = check_public(claims).unwrap_or_else(|e| panic!("{e:?}"));
    lower(&public, 5).unwrap_or_else(|e| panic!("{e:?}"))
}

/// The shift-class sets of the `Requant` nodes that output the checkpoints `c_1` to `c_L`: each
/// layer's last node, its `MlpResidual` site's last `Requant`.
fn residual_sets(graph: &Graph) -> Vec<Vec<u8>> {
    let mut sets = Vec::new();
    for checkpoint in &graph.checkpoints()[1..] {
        let node = graph
            .node(graph.tensor(*checkpoint).unwrap().producer.unwrap())
            .unwrap();
        let Op::Requant { shift, .. } = *node.op() else {
            panic!("a checkpoint is a Requant output");
        };
        sets.push(graph.shift_classes()[shift as usize].as_slice().to_vec());
    }
    sets
}

#[test]
fn a_pot_site_of_one_shift_up_to_62_in_every_layer_fuses_its_key() {
    // `tiny-4-servable` rescales `AttnResidual` with one class in its four layers.
    let mut claims = golden("tiny-4-servable");
    for layer in 0..4 {
        make_pot(&mut claims, at(layer, Site::AttnResidual), &[62]);
    }
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::AttnResidual), Some(Route::FusedKey));
    // The fused key's constant `2^62`, once per layer.
    let powers = graph.constants().iter().filter(|&&c| c == 1 << 62).count();
    assert_eq!(powers, 4);
    // One shift of 63 in one layer sends every layer to two steps, and no `2^r` is created.
    make_pot(&mut claims, at(2, Site::AttnResidual), &[63]);
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::AttnResidual), Some(Route::TwoSteps));
    assert!(!graph.constants().contains(&(1 << 62)));
}

#[test]
fn an_affine_site_of_shift_up_to_62_in_every_layer_fuses_its_key() {
    // `tiny-2-affine`'s `AttnResidual` is affine at 18 in both layers.
    let mut claims = golden("tiny-2-affine");
    for layer in 0..2 {
        make_affine(&mut claims, at(layer, Site::AttnResidual), 62);
    }
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::AttnResidual), Some(Route::FusedKey));
    assert_eq!(
        graph.constants().iter().filter(|&&c| c == 1 << 62).count(),
        2
    );
    make_affine(&mut claims, at(1, Site::AttnResidual), 63);
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::AttnResidual), Some(Route::TwoSteps));
}

#[test]
fn the_last_requant_of_either_route_takes_its_own_set() {
    // The fused key's last `Requant` takes a new set `[r]`, and two steps' a new set `[1]`, each
    // a set of its own even when an earlier one is equal (RFC-0044 'Ids').
    let mut claims = golden("tiny-4-servable");
    for layer in 0..4 {
        make_pot(&mut claims, at(layer, Site::MlpResidual), &[62]);
    }
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::MlpResidual), Some(Route::FusedKey));
    assert_eq!(residual_sets(&graph), vec![vec![62]; 4]);
    // Two classes in every layer, so that the layers stay alike (RFC-0038).
    for layer in 0..4 {
        make_pot(&mut claims, at(layer, Site::MlpResidual), &[2, 3]);
    }
    let graph = lowered(&claims);
    assert_eq!(graph.route(Site::MlpResidual), Some(Route::TwoSteps));
    assert_eq!(residual_sets(&graph), vec![vec![1]; 4]);
}
