// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054/0018 public layout construction: inputs predate counting, outputs remain live.
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_field::{Field, Goldilocks};
use tessera_gadgets::{Affine, Lookup};
use tessera_ir::NodeId;
use tessera_protocol::LeafGroups;
use tessera_prover::leaf::{GroupedLayout, LeafError, LeafLayout, Site, SiteGadget, SiteLookup};
use tessera_tables::TableId;

fn fixture(reads: usize, domain: &[u32], copies: usize, tables: bool) -> LeafLayout<usize> {
    let lookup = |index| SiteLookup {
        lookup: Lookup {
            table: TableId(if tables { index as u32 } else { 0 }),
            key: Affine {
                constant: Goldilocks::ZERO,
                terms: (0..reads).map(|r| (Goldilocks::ONE, r)).collect(),
            },
            value: None,
        },
        domain: domain.to_vec(),
        axes: Vec::new(),
        token: Some(0),
    };
    LeafLayout::new(
        &LayerMap::none(),
        vec![Site {
            node: NodeRef::Node(NodeId(0)),
            layers: vec![SiteGadget {
                lookups: (0..copies).map(lookup).collect(),
            }],
        }],
    )
    .unwrap()
}

fn construction<T: Copy + Eq>(stage: &LeafLayout<T>, layers: &LayerMap, groups: LeafGroups) {
    let mut bound = None;
    let planning = allocation_counter::measure(|| {
        bound = Some(GroupedLayout::construction_memory(stage, groups).unwrap());
    });
    assert_eq!(planning.count_total, 0);
    let bound = bound.unwrap();
    let mut output = None;
    let measured = allocation_counter::measure(|| {
        output = Some(GroupedLayout::new(stage, layers, groups).unwrap());
    });
    let output = output.unwrap();
    assert_eq!(measured.bytes_current as u64, output.heap_bytes().unwrap());
    assert!(measured.bytes_current as u64 <= bound.output_bytes());
    assert!(
        measured.bytes_max <= bound.peak_bytes(),
        "{measured:?} bound {bound:?}"
    );
    // Counts are public; test both the exceptional zero subset and ordinary groups.
    let mut ids = vec![0, output.count() / 2, output.count() - 1];
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        let mut bound = None;
        let planning = allocation_counter::measure(|| {
            bound = Some(output.group_memory(id).unwrap());
        });
        assert_eq!(planning.count_total, 0);
        let bound = bound.unwrap();
        let mut group = None;
        let measured = allocation_counter::measure(|| group = Some(output.group(id).unwrap()));
        let group = group.unwrap();
        assert_eq!(measured.bytes_current as u64, group.heap_bytes().unwrap());
        assert_eq!(measured.bytes_current as u64, bound.output_bytes());
        assert!(
            measured.bytes_max <= bound.peak_bytes(),
            "group {id}: {measured:?} bound {bound:?}"
        );
        assert_eq!(group.tokens(), id as u64 & ((1 << groups.token_bits) - 1));
        assert_eq!(group.layers(), (id as u64) >> groups.token_bits);
    }
    assert_eq!(
        output.group_memory(output.count()),
        Err(LeafError::Internal {
            rule: "a group of the grouping"
        })
    );
    assert_eq!(
        output.group(output.count()).unwrap_err(),
        LeafError::Internal {
            rule: "a group of the grouping"
        }
    );
}

#[test]
fn scalar_reads_isolate_held_flags_and_preallocated_maps() {
    for reads in [1, 4, 17, 1024, 1025] {
        let stage = fixture(reads, &[1], 1, false);
        let groups = LeafGroups {
            token_bits: 0,
            layer_bits: 0,
        };
        let bound = GroupedLayout::construction_memory(&stage, groups).unwrap();
        let mut output = None;
        let measured = allocation_counter::measure(|| {
            output = Some(GroupedLayout::new(&stage, &LayerMap::none(), groups).unwrap());
        });
        // Rank zero eliminates trial-coordinate scratch; one share has no growth.
        // Every returned owner is live while held's two flags per read remain live.
        assert_eq!(
            measured.bytes_max,
            bound.peak_bytes(),
            "{reads}: {measured:?} bound {bound:?}"
        );
        assert_eq!(
            measured.bytes_current as u64,
            output.as_ref().unwrap().heap_bytes().unwrap()
        );
        construction(&stage, &LayerMap::none(), groups);
    }
}

#[test]
fn coordinates_fixed_axes_and_duplicate_lookup_trials_are_bounded() {
    let domain: Vec<_> = core::iter::once(1 << 14)
        .chain(core::iter::repeat_n(2, 34))
        .collect();
    for reads in [1, 5, 129] {
        let stage = fixture(reads, &domain, 3, false);
        for token_bits in [0, 1, 8, 14] {
            construction(
                &stage,
                &LayerMap::none(),
                LeafGroups {
                    token_bits,
                    layer_bits: 0,
                },
            );
        }
    }
}

#[test]
fn table_index_growth_and_stable_sort_scratch_are_bounded() {
    for copies in [1, 4, 5, 11, 12, 33, 257, 1025] {
        for tables in [false, true] {
            let stage = fixture(3, &[32, 8], copies, tables);
            construction(
                &stage,
                &LayerMap::none(),
                LeafGroups {
                    token_bits: 2,
                    layer_bits: 0,
                },
            );
        }
    }
}

#[test]
fn unsorted_public_block_sizes_exercise_stable_sort_work() {
    let layers = LayerMap::none();
    for count in [33, 129, 1025, 4097] {
        let mut sites = fixture(3, &[32], count, false).sites().to_vec();
        for (index, lookup) in sites[0].layers[0].lookups.iter_mut().enumerate() {
            lookup.domain.push(1 << ((index * 13 + 7) % 17));
            // Scalar reads keep the same points despite unlike public block sizes.
            lookup.axes = (0..3).map(|read| (read, Vec::new())).collect();
        }
        let stage = LeafLayout::new(&layers, sites).unwrap();
        construction(
            &stage,
            &layers,
            LeafGroups {
                token_bits: 2,
                layer_bits: 0,
            },
        );
    }
}

#[test]
fn empty_layout_and_absent_groups_are_bounded() {
    let layers = LayerMap::none();
    let empty: LeafLayout<usize> = LeafLayout::new(&layers, Vec::new()).unwrap();
    construction(
        &empty,
        &layers,
        LeafGroups {
            token_bits: 2,
            layer_bits: 0,
        },
    );
    let mut stage = fixture(3, &[32], 1, false);
    // A separate public fixture has no token axis; only subset zero owns its blocks.
    let mut sites = stage.sites().to_vec();
    sites[0].layers[0].lookups[0].token = None;
    stage = LeafLayout::new(&layers, sites).unwrap();
    construction(
        &stage,
        &layers,
        LeafGroups {
            token_bits: 2,
            layer_bits: 0,
        },
    );
}

#[test]
fn actual_model_layouts_include_layer_groups_and_unequal_subsets() {
    for model in [
        super::model("tiny-2-affine"),
        super::layered(super::model("tiny-2-servable")),
    ] {
        let public = super::public(
            &model,
            5,
            tessera_ir::GadgetPolicy {
                query_block: 2,
                ..Default::default()
            },
        );
        let stage = LeafLayout::model(
            &public.lowering,
            &public.layers,
            &public.selection,
            &public.claims,
        )
        .unwrap();
        for token_bits in [0, 1, 2] {
            for layer_bits in [0, 1] {
                construction(
                    &stage,
                    &public.layers,
                    LeafGroups {
                        token_bits,
                        layer_bits,
                    },
                );
            }
        }
    }
}
