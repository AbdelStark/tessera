// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's group layouts against the independent verifier's (DEP-2): `tv-core` derives
//! RFC-0054's groups from its own lowering, and both must agree on every group's instances,
//! blocks, offsets and heights, its split values' polynomials, coordinates and bits, and the
//! merge's tensor polynomials.

use super::*;
use tv_core::leaf::groups::Grouped;

fn tv_groups(token_bits: u8, layer_bits: u8) -> tv_core::protocol::LeafGroups {
    tv_core::protocol::LeafGroups {
        token_bits,
        layer_bits,
    }
}

#[test]
fn the_prover_and_the_verifier_lay_out_every_group_alike() {
    let cases = [
        (bounded_model(), 3, vec![(0, 0), (1, 0), (2, 1)]),
        (
            layered(bounded_model()),
            5,
            vec![(0, 0), (1, 1), (0, 2), (3, 0), (2, 2), (3, 2)],
        ),
    ];
    for (model, n, groupings) in cases {
        let (p, layout) = stage(&model, n);
        let (graph, claims) = verifier(&p, n);
        for (g_t, g_l) in groupings {
            let ours = GroupedLayout::new(&layout, &p.layers, groups(g_t, g_l)).unwrap();
            let theirs = Grouped::derive(&graph, claims.gadget_policy, tv_groups(g_t, g_l))
                .unwrap_or_else(|e| panic!("{g_t}, {g_l}: {e}"));
            assert_eq!(ours.count(), theirs.count() as usize);
            assert_eq!(ours.polynomials().len(), theirs.values());
            for (a, b) in ours.polynomials().iter().zip(theirs.polynomials()) {
                assert_eq!(a.vars, b.variables as usize);
                let group_vars: Vec<usize> =
                    b.group_variables.iter().map(|&v| v as usize).collect();
                assert_eq!(a.group_vars, group_vars);
            }
            for h in 0..ours.count() {
                let a = ours.group(h).unwrap();
                let b = theirs.group(h as u32).unwrap();
                assert_eq!(
                    (a.tokens(), a.layers()),
                    (u64::from(b.tokens), u64::from(b.layers))
                );
                assert_eq!(
                    a.instances().len(),
                    b.instances.len(),
                    "group {h}'s instances"
                );
                for (x, y) in a.instances().iter().zip(&b.instances) {
                    assert_eq!(x.table.0, y.table.0);
                    assert_eq!((x.entries, x.height), (y.entries, y.height));
                    let ours: Vec<_> = x
                        .blocks
                        .iter()
                        .map(|k| (k.site, k.layer as u32, k.lookup, k.vars, k.offset))
                        .collect();
                    let theirs: Vec<_> = y
                        .blocks
                        .iter()
                        .map(|k| (k.site, k.layer, k.lookup, k.variables, k.offset))
                        .collect();
                    assert_eq!(ours, theirs, "group {h}, {:?}", x.table);
                }
                assert_eq!(a.reads().len(), b.split.len(), "group {h}'s split");
                for (x, y) in a.reads().iter().zip(&b.split) {
                    let site = layout.sites()[x.site].node;
                    assert_eq!(owner(&p, site, x.layer).0, y.node.0);
                    assert_eq!(x.polynomial, y.polynomial);
                    assert_eq!(
                        x.coords, y.coordinates,
                        "group {h}, polynomial {}",
                        x.polynomial
                    );
                    assert_eq!(x.group_bits, u64::from(y.bits));
                }
            }
        }
    }
}
