// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Physical owned source storage plus separately measured external tile work (#641).
use super::*;

#[test]
fn public_model_source_memory_matches_owned_buffers_and_separate_tile_overlap() {
    for name in [
        "tiny-2-servable",
        "tiny-2-minimal",
        "tiny-2-affine",
        "tiny-4-servable",
        "three-layer",
    ] {
        for n in [1, 3, 5] {
            check(name, n, None);
        }
    }
    for prompt in [1, 3] {
        check("tiny-2-servable", 3, Some(prompt));
    }
}

fn check(name: &str, n: u32, prompt: Option<u32>) {
    let model = if name == "three-layer" {
        layered(model("tiny-2-servable"))
    } else {
        model(name)
    };
    let p = public(
        &model,
        n,
        GadgetPolicy {
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    let runtime = prompt.map(|prompt| {
        tessera_compiler::RuntimePlan::derive(&p.lowering, &p.selection, &p.claims, prompt, 0)
            .unwrap()
    });
    let layout = match &runtime {
        Some(r) => LeafLayout::runtime(&p.lowering, &p.layers, r, &p.claims),
        None => LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims),
    }
    .unwrap();
    let advice = match &runtime {
        Some(r) => ModelAdviceLayout::runtime(&p.lowering, r, &p.claims),
        None => ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims),
    }
    .unwrap();
    let engine = Engine::new(&p.lowering)
        .unwrap()
        .threads(std::num::NonZeroUsize::MIN);
    let mut input = (0..n)
        .map(|i| (i + 1) % model.arch.vocab)
        .collect::<Vec<_>>();
    if let Some(prompt) = prompt {
        for t in prompt..n {
            let pass = engine.run(&model, &Secret::new(input.clone())).unwrap();
            let v = model.arch.vocab as usize;
            let logits =
                &pass.tensor(p.lowering.logits).unwrap()[(t as usize - 1) * v..t as usize * v];
            let max = *logits.iter().max().unwrap();
            input[t as usize] = logits.iter().position(|&x| x == max).unwrap() as u32;
        }
    }
    let tokens = Secret::new(input);
    let witness = engine.run(&model, &tokens).unwrap();
    let source = match &runtime {
        Some(r) => Resident::runtime(&engine, r, &model, &witness),
        None => Resident::new(&engine, &p.selection, &model, &witness),
    }
    .unwrap();
    let mut plan = None;
    let planning = allocation_counter::measure(|| {
        plan = Some(ModelWitness::memory(&layout, &p.lowering, &p.layers, &advice).unwrap())
    });
    assert_eq!(planning.count_total, 0);
    let plan = plan.unwrap();
    // This is fixture evidence for the external source, not a public tile bound.
    // The production route still needs its own checked canonical tile/gadget plan.
    let mut tile_peak = 0;
    for tile in advice.tiles().iter().filter(|t| {
        matches!(
            t.tile,
            tessera_compiler::Tile::Node { .. }
                | tessera_compiler::Tile::Attention { .. }
                | tessera_compiler::Tile::GreedyMaximum { .. }
        )
    }) {
        let mut values = None;
        let measured = allocation_counter::measure(|| values = Some(source.tile(tile).unwrap()));
        tile_peak = tile_peak.max(measured.bytes_max);
        drop(values);
    }
    let mut actual = None;
    let measured = allocation_counter::measure(|| {
        actual = Some(
            ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap(),
        )
    });
    assert_eq!(
        u64::try_from(measured.bytes_current).unwrap(),
        plan.retained_bytes(),
        "{name} n{n} prompt{prompt:?} {measured:?} {plan:?}"
    );
    assert!(
        measured.bytes_max <= plan.peak_bytes() + tile_peak,
        "{name} n{n} prompt{prompt:?} {measured:?} {plan:?} external {tile_peak}"
    );
    drop(actual);
}

#[test]
fn public_model_source_memory_refuses_missing_owners_without_allocating() {
    let model = model("tiny-2-servable");
    let p = public(&model, 3, GadgetPolicy::default());
    let layout = LeafLayout::model(&p.lowering, &p.layers, &p.selection, &p.claims).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let mut broken = public(&model, 3, GadgetPolicy::default()).lowering;
    broken.graph.nodes.clear();
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(ModelWitness::memory(&layout, &broken, &p.layers, &advice))
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(
        result.unwrap(),
        Err(tessera_prover::LeafError::Internal {
            rule: "a source owner belongs to the lowering"
        })
    );
}
