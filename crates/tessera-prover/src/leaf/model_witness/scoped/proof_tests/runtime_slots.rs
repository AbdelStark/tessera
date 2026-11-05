// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Real runtime/slot canonical sources compared with the resident oracle.

use super::*;

#[test]
fn runtime_and_slot_sources_preserve_request_ordinals_gaps_and_greedy_keys() {
    use tessera_compiler::RuntimePlan;
    let model = bounded_model();
    let mut group_cases = 0;
    let mut greedy_reads = 0;
    for requests in [&[(2, 1)][..], &[(2, 1), (1, 0), (1, 1)][..]] {
        let lengths: Vec<_> = requests.iter().map(|(p, o)| p + o).collect();
        let lowering = tessera_compiler::lower_batch(PublicModel::of(&model), &lengths).unwrap();
        let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
        let policy = GadgetPolicy {
            max_log2_rows: 21,
            query_block: 2,
            ..Default::default()
        };
        let selection =
            select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy).unwrap();
        let p = ParamSet::TEST_SMALL;
        let claims = claim_list(
            PublicModel::of(&model),
            &selection,
            &policy,
            p.layout,
            ParamSetId(Ascii::new(p.id).unwrap()),
        )
        .unwrap();
        let runtime = RuntimePlan::derive_batch(&lowering, &selection, &claims, requests).unwrap();
        let layers = LayerMap::new(&lowering).unwrap();
        let stage = LeafLayout::runtime(&lowering, &layers, &runtime, &claims).unwrap();
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
        let engine = Engine::new(&lowering).unwrap();
        let mut input: Vec<_> = (0..lengths.iter().sum::<u32>())
            .map(|i| (i + 1) % model.arch.vocab)
            .collect();
        let slot_len = lowering
            .graph
            .slots
            .as_ref()
            .map_or(lengths[0], |s| s.slot_len);
        let mut start = 0;
        for (r, &(prompt, outputs)) in requests.iter().enumerate() {
            for t in prompt..prompt + outputs {
                let witness = engine
                    .run_batch(&model, &Secret::new(input.clone()))
                    .unwrap();
                let vocab = model.arch.vocab as usize;
                let row = (r * slot_len as usize + t as usize - 1) * vocab;
                let logits = &witness.tensor(lowering.logits).unwrap()[row..row + vocab];
                let maximum = *logits.iter().max().unwrap();
                input[start + t as usize] =
                    logits.iter().position(|&v| v == maximum).unwrap() as u32;
            }
            start += (prompt + outputs) as usize;
        }
        let witness = engine.run_batch(&model, &Secret::new(input)).unwrap();
        let source = Resident::runtime(&engine, &runtime, &model, &witness).unwrap();
        let full =
            ModelWitness::new(&stage, &lowering, &layers, &witness, &source, &advice).unwrap();
        for (gt, gl) in [(0, 1), (1, 0), (1, 1)] {
            let grouped = GroupedLayout::new(
                &stage,
                &layers,
                LeafGroups {
                    token_bits: gt,
                    layer_bits: gl,
                },
            )
            .unwrap();
            let factory = ModelSourceFactory {
                layout: &stage,
                grouped: &grouped,
                lowering: &lowering,
                layers: &layers,
                witness: &witness,
                source: &source,
                advice: &advice,
            };
            for h in 0..grouped.count() {
                let group = factory.group(h).unwrap();
                group_cases += 1;
                for read in group.layout.reads() {
                    let term = stage.reads(read.site, read.layer).unwrap()[read.read].term;
                    greedy_reads += usize::from(term.is_greedy_key());
                    let original = full.values(read.site, read.layer, term).unwrap();
                    let shape = &grouped.polynomials()[read.polynomial];
                    let expected: Vec<_> = original
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(i, _)| {
                            shape.group_vars.iter().enumerate().all(|(j, &v)| {
                                (i >> v) & 1 == ((read.group_bits >> j) & 1) as usize
                            })
                        })
                        .map(|(_, v)| v)
                        .collect();
                    assert_eq!(
                        group.tensor(read.site, read.layer, read.read).unwrap(),
                        expected,
                        "runtime/slot group {h}, polynomial {}",
                        read.polynomial
                    );
                }
                assert_eq!(group.inner.forests.len(), full.forests.len());
                for (original, actual) in full.forests.iter().zip(&group.inner.forests) {
                    let positions: Vec<_> = (original.height..original.height + gt as usize)
                        .chain(original.layers as usize..original.layers as usize + gl as usize)
                        .collect();
                    let expected: Vec<_> = original
                        .leaves
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(i, _)| {
                            positions
                                .iter()
                                .enumerate()
                                .all(|(j, &v)| (i >> v) & 1 == (h >> j) & 1)
                        })
                        .map(|(_, v)| v)
                        .collect();
                    assert_eq!(
                        &actual.leaves[..],
                        expected,
                        "runtime/slot forest group {h}"
                    );
                }
            }
            let mut index = 0;
            for (s, site) in stage.sites().iter().enumerate() {
                for l in 0..site.layers.len() {
                    for read in stage.reads(s, l).unwrap() {
                        let expected: Vec<_> = full
                            .values(s, l, read.term)
                            .unwrap()
                            .iter()
                            .map(|&v| Goldilocks::from_i64(v))
                            .collect();
                        assert_eq!(&factory.original(index).unwrap()[..], expected);
                        index += 1;
                    }
                }
            }
            for forest in &full.forests {
                assert_eq!(&factory.original(index).unwrap()[..], &forest.leaves[..]);
                index += 1;
            }
            let (id, maximum) = factory.maximum().unwrap();
            let (expected_id, expected_maximum) = full.greedy_maximum().unwrap();
            assert_eq!(id, expected_id);
            assert_eq!(&maximum[..], expected_maximum);
            assert!(index > grouped.polynomials().len());
        }
        if requests.len() > 1 {
            assert!(
                lowering
                    .graph
                    .slots
                    .as_ref()
                    .unwrap()
                    .lengths
                    .iter()
                    .any(|&n| n < slot_len)
            );
            assert!(
                runtime.key_columns().len() > 1,
                "fixture must distinguish request ordinals"
            );
        }
    }
    assert_eq!(group_cases, 16);
    assert!(greedy_reads > 0);
}
