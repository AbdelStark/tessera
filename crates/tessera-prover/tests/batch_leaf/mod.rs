// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-request exact-greedy leaf sources (RFC-0071).
use super::*;

#[test]
fn each_output_request_has_its_own_padded_key_source_in_batch_order() {
    check_batch([(3, 1), (4, 0), (2, 2)], [4, 4, 4]);
}

#[test]
fn unequal_request_lengths_fill_softmax_gap_sources() {
    check_batch([(1, 1), (1, 0), (1, 2)], [2, 1, 3]);
}

fn check_batch(requests: [(u32, u32); 3], lengths: [u32; 3]) {
    let model = model("tiny-2-minimal");
    let mut p = public(&model, 3, GadgetPolicy::default());
    p.lowering = tessera_compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
    p.layers = LayerMap::new(&p.lowering).unwrap();
    let ranges = analyze(&p.lowering.graph, &p.lowering.tables).unwrap();
    p.selection = select_model(
        p.claims.protocol_version,
        &p.lowering,
        &ranges,
        &p.claims.gadget_policy,
    )
    .unwrap();
    let runtime = tessera_compiler::RuntimePlan::derive_batch(
        &p.lowering,
        &p.selection,
        &p.claims,
        &requests,
    )
    .unwrap();
    let layout = LeafLayout::runtime(&p.lowering, &p.layers, &runtime, &p.claims).unwrap();
    let native = tessera_prover::native::ModelNativePlan::derive_runtime(
        &p.lowering,
        &runtime,
        &p.claims,
        &layout,
    )
    .unwrap();
    let planning = allocation_counter::measure(|| {
        let memory = native.score_input_memory(&p.lowering).unwrap();
        assert_eq!((memory.retained_bytes(), memory.peak_bytes()), (0, 0));
        assert_eq!(
            native.memory(&p.lowering, false).unwrap(),
            native.memory(&p.lowering, true).unwrap(),
            "slots force the literal native representation"
        );
    });
    assert_eq!(planning.count_total, 0);
    let runtime_kinds: Vec<_> = native
        .native()
        .iter()
        .filter_map(|n| {
            matches!(
                n.kind,
                NativeKind::Tokens | NativeKind::GreedyKey | NativeKind::GreedySelected
            )
            .then_some(n.kind)
        })
        .collect();
    use tessera_compiler::schedule::NativeKind;
    assert_eq!(
        runtime_kinds,
        [
            NativeKind::Tokens,
            NativeKind::GreedyKey,
            NativeKind::GreedyKey,
            NativeKind::GreedySelected
        ]
    );
    let key_sites: Vec<_> = layout
        .sites()
        .iter()
        .enumerate()
        .filter_map(|(s, _)| {
            layout
                .reads(s, 0)
                .unwrap()
                .iter()
                .any(|r| r.term.is_greedy_key())
                .then_some(s)
        })
        .collect();
    assert_eq!(key_sites.len(), 1, "one LM-head site owns all request keys");
    let engine = Engine::new(&p.lowering)
        .unwrap()
        .threads(std::num::NonZeroUsize::MIN);
    let mut tokens = vec![0; 12];
    let vocab = model.arch.vocab as usize;
    for (r, &(prompt, outputs)) in requests.iter().enumerate() {
        for u in 0..outputs {
            let t = r * 4 + (prompt + u) as usize;
            let w = engine.run(&model, &Secret::new(tokens.clone())).unwrap();
            let row = &w.tensor(p.lowering.logits).unwrap()[(t - 1) * vocab..t * vocab];
            let maximum = row.iter().max().unwrap();
            tokens[t] = row.iter().position(|v| v == maximum).unwrap() as u32;
        }
    }
    let witness = engine.run(&model, &Secret::new(tokens)).unwrap();
    let source = Resident::runtime(&engine, &runtime, &model, &witness).unwrap();
    let advice = ModelAdviceLayout::runtime(&p.lowering, &runtime, &p.claims).unwrap();
    let mut planned = None;
    let planning = allocation_counter::measure(|| {
        planned = Some(ModelWitness::memory(&layout, &p.lowering, &p.layers, &advice).unwrap())
    });
    assert_eq!(planning.count_total, 0);
    let mut actual = None;
    let owned = allocation_counter::measure(|| {
        actual = Some(
            ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap(),
        )
    });
    assert_eq!(
        u64::try_from(owned.bytes_current).unwrap(),
        planned.unwrap().retained_bytes()
    );
    let actual = actual.unwrap();
    for (site, spec) in layout.sites().iter().enumerate() {
        if actual.values(site, 0, ModelColumn::ExpKey).is_none() {
            continue;
        }
        for layer in 0..spec.layers.len() {
            let node = &p.lowering.graph.nodes[owner(&p, spec.node, layer).0 as usize];
            let exp = match node.op {
                tessera_ir::Op::SoftmaxRow { exp_table, .. } => exp_table,
                _ => unreachable!(),
            };
            let last_key = (1i64
                << p.lowering
                    .tables
                    .iter()
                    .find(|t| t.id == exp)
                    .unwrap()
                    .log2_rows)
                - 1;
            for read in layout.reads(site, layer).unwrap() {
                let values = actual.values(site, layer, read.term).unwrap();
                let domain: Vec<_> = read
                    .axes
                    .iter()
                    .map(|&a| {
                        spec.layers[layer]
                            .lookups
                            .iter()
                            .find(|lookup| {
                                core::iter::once(&lookup.lookup.key)
                                    .chain(&lookup.lookup.value)
                                    .any(|form| form.terms.iter().any(|(_, c)| *c == read.term))
                            })
                            .unwrap()
                            .domain[a]
                    })
                    .collect();
                for head in 0..domain[0] as usize {
                    for (r, &len) in lengths.iter().enumerate() {
                        for position in len as usize..4 {
                            let row = (head << 4) | (r * 4 + position);
                            let width = if domain.len() == 3 {
                                domain[2].next_power_of_two() as usize
                            } else {
                                1
                            };
                            for key in 0..width {
                                assert_eq!(
                                    values[row * width + key],
                                    if read.term == ModelColumn::ExpKey {
                                        last_key
                                    } else {
                                        0
                                    },
                                    "gap source {:?}",
                                    read.term
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let maximum_tile = advice
        .tiles()
        .iter()
        .find(|t| matches!(t.tile, tessera_compiler::Tile::GreedyMaximum { .. }))
        .unwrap();
    let tile = source.tile(maximum_tile).unwrap();
    assert_eq!(tile.lookups().count(), 2);
    for ((site, column), lookup) in key_sites
        .iter()
        .cycle()
        .zip(runtime.key_columns())
        .zip(tile.lookups())
    {
        let values = actual
            .values(*site, 0, ModelColumn::greedy_key(column.request_index))
            .unwrap();
        assert_eq!(values.len() as u64, column.entries());
        assert_eq!(
            &values[..],
            lookup
                .keys()
                .iter()
                .map(|&k| i64::from(k))
                .collect::<Vec<_>>()
        );
    }
}
