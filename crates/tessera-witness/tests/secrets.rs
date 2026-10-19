// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Secret-class values reach no debug form and no error (05 §3 OBS-1, OBS-3; INV-ARCH-6).
//!
//! Canaries are planted in the weights and the tokens: the embedding column of a canary token holds
//! 113 in every channel, so that the first checkpoint holds `113·2^{s_e}` wherever the token
//! stands, and the token stands twice among the prompt's.
//!
//! - **Debug forms.** The engine, the first pass, the checkpoints, both sources, and every tile and
//!   lookup they give show the same text as for the model and tokens without the canaries: shapes
//!   and identifiers, nothing a value moves. None names a canary, in decimal or hexadecimal.
//! - **Errors.** A canary token outside the vocabulary, a canary value outside its tensor's analyzed
//!   range, and a tile the plan does not have are refused by errors that name no canary.

mod common;

use common::{conformance, lowered, plan_of};
use tessera_codec::Secret;
use tessera_compiler::{Lowering, Selection, Tile};
use tessera_ir::{IntRange, IntegerModel, QuantPoint, WeightType};
use tessera_witness::{
    Checkpoints, Engine, Regenerator, Resident, Witness, WitnessError, WitnessSource,
};

/// The canary token, inside the vocabulary of every conformance model.
const TOKEN: u32 = 451;
/// The canary weight, in every channel of the canary token's embedding column.
const WEIGHT: i8 = 113;
/// A canary token outside the vocabulary.
const OUTSIDE: u32 = 0x05EC_C4A7;

/// `model` with the canary weight planted in the canary token's embedding column.
fn planted(model: &IntegerModel) -> IntegerModel {
    let mut model = model.clone();
    let embed = model
        .weights
        .iter_mut()
        .find(|w| w.ty == WeightType::Embed)
        .expect("the embedding");
    let vocab = embed.cols as usize;
    for c in 0..embed.rows as usize {
        embed.data[c * vocab + TOKEN as usize] = WEIGHT;
    }
    model
}

/// `c_0` at a canary position: `113·2^{s_e}`, with `s_e = e_Embed − residual_exponent`.
fn first_checkpoint(model: &IntegerModel) -> i64 {
    let quant = &model.quant.public;
    let embed = quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .map(|&(_, e)| i64::from(e))
        .expect("the embedding's exponent");
    let shift = embed - i64::from(quant.residual_exponent);
    i64::from(WEIGHT) << shift
}

/// The forms in which a canary could be printed.
fn forms(value: i64) -> Vec<String> {
    vec![
        value.to_string(),
        format!("{value:x}"),
        format!("{value:X}"),
        format!("{value:#x}"),
    ]
}

fn assert_clean(text: &str, canaries: &[String], what: &str) {
    for canary in canaries {
        assert!(
            !text.contains(canary.as_str()),
            "{what} names the canary {canary}: {text}"
        );
    }
}

/// Every debug form of one sequence's engine, first pass, checkpoints, sources, and tiles under
/// query blocks of one query.
fn debug_forms(
    model: &IntegerModel,
    lowering: &Lowering,
    selection: &Selection,
    tokens: &[u32],
) -> Vec<(String, String)> {
    let engine = Engine::new(lowering).expect("tables");
    let secret = Secret::new(tokens.to_vec());
    let witness: Witness = engine.run(model, &secret).expect("the first pass");
    let store = Checkpoints::new(&engine, model, &secret).expect("checkpoints");
    let regenerator =
        Regenerator::new(&engine, selection, model, &secret, &store).expect("a regenerator");
    let resident = Resident::new(&engine, selection, model, &witness).expect("a resident source");
    let mut forms = vec![
        ("engine".to_owned(), format!("{engine:?}")),
        ("witness".to_owned(), format!("{witness:?}")),
        ("checkpoints".to_owned(), format!("{store:?}")),
        ("regenerator".to_owned(), format!("{regenerator:?}")),
        ("resident".to_owned(), format!("{resident:?}")),
    ];
    let plan = plan_of(model, 1);
    for tile in plan.expand(lowering).expect("the plan expands") {
        let label = format!("{:?} {:?}", tile.tile, tile.block);
        for (source, regenerated) in [
            ("regenerated", regenerator.tile(&tile).expect("a tile")),
            ("resident", resident.tile(&tile).expect("a tile")),
        ] {
            forms.push((format!("{label}, {source}"), format!("{regenerated:?}")));
            for (j, lookup) in regenerated.lookups().enumerate() {
                forms.push((
                    format!("{label}, {source}, lookup {j}"),
                    format!("{lookup:?}"),
                ));
            }
        }
    }
    forms
}

#[test]
fn no_debug_form_names_a_canary() {
    let c = conformance().remove(0);
    let mut tokens = c.sequences[0].tokens.clone();
    let clean_tokens = tokens.clone();
    tokens[0] = TOKEN;
    tokens[3] = TOKEN;
    assert!(!clean_tokens.contains(&TOKEN));
    let model = planted(&c.model);
    let n = u32::try_from(tokens.len()).expect("small");
    let (lowering, selection) = lowered(&model, n);
    let canary = first_checkpoint(&model);
    // The canary is the first checkpoint's value at the canary token's positions.
    {
        let engine = Engine::new(&lowering).expect("tables");
        let store =
            Checkpoints::new(&engine, &model, &Secret::new(tokens.clone())).expect("checkpoints");
        let d = model.arch.d_model as usize;
        let c0 = store.residual(0).expect("c_0");
        assert!(
            c0[..d].iter().all(|&v| i64::from(v) == canary),
            "c_0 holds the canary"
        );
    }
    let mut canaries = forms(canary);
    canaries.push(TOKEN.to_string());
    canaries.push(format!("{TOKEN:x}"));
    let planted_forms = debug_forms(&model, &lowering, &selection, &tokens);
    let clean_forms = debug_forms(&c.model, &lowering, &selection, &clean_tokens);
    assert_eq!(planted_forms.len(), clean_forms.len());
    for ((what, text), (_, clean)) in planted_forms.iter().zip(&clean_forms) {
        assert_eq!(
            text, clean,
            "{what}: the debug form shows what no value moves"
        );
        assert_clean(text, &canaries, what);
    }
}

#[test]
fn no_error_names_a_canary() {
    let c = conformance().remove(0);
    let mut tokens = c.sequences[0].tokens.clone();
    tokens[1] = TOKEN;
    let model = planted(&c.model);
    let n = u32::try_from(tokens.len()).expect("small");
    let (lowering, selection) = lowered(&model, n);
    let engine = Engine::new(&lowering).expect("tables");
    let secret = Secret::new(tokens.clone());
    let store = Checkpoints::new(&engine, &model, &secret).expect("checkpoints");
    let mut canaries = forms(first_checkpoint(&model));
    canaries.extend(forms(i64::from(OUTSIDE)));
    let mut errors: Vec<(String, WitnessError)> = Vec::new();

    // A canary token outside the vocabulary.
    let mut outside = tokens.clone();
    outside[2] = OUTSIDE;
    let outside = Secret::new(outside);
    errors.push((
        "checkpoints, a token outside".into(),
        Checkpoints::new(&engine, &model, &outside).expect_err("refused"),
    ));
    errors.push((
        "regenerator, a token outside".into(),
        Regenerator::new(&engine, &selection, &model, &outside, &store).expect_err("refused"),
    ));

    // The canary value outside its tensor's analyzed range, which debug builds check (INV-DM-2).
    if cfg!(debug_assertions) {
        let mut narrowed = lowering.clone();
        let c0 = narrowed.checkpoints[0];
        narrowed.graph.tensors[c0.0 as usize].range = IntRange { lo: 0, hi: 0 };
        let narrowed_engine = Engine::new(&narrowed).expect("tables");
        errors.push((
            "checkpoints, a value outside its range".into(),
            Checkpoints::new(&narrowed_engine, &model, &secret).expect_err("refused"),
        ));
        let regenerator = Regenerator::new(&narrowed_engine, &selection, &model, &secret, &store)
            .expect("a regenerator over the honest store");
        let plan = plan_of(&model, 512);
        let tile = plan
            .expand(&narrowed)
            .expect("the plan expands")
            .into_iter()
            .find(|t| matches!(t.tile, Tile::Node { node } if narrowed.graph.nodes[node as usize].outputs.contains(&c0)))
            .expect("the first checkpoint's tile");
        errors.push((
            "regenerator, a value outside its range".into(),
            regenerator.tile(&tile).expect_err("refused"),
        ));
    }

    // A tile the plan does not have.
    let regenerator =
        Regenerator::new(&engine, &selection, &model, &secret, &store).expect("a regenerator");
    let plan = plan_of(&model, 512);
    let mut tile = plan.expand(&lowering).expect("the plan expands").remove(0);
    tile.tile = Tile::Node { node: u32::MAX };
    errors.push((
        "regenerator, a tile not the plan's".into(),
        regenerator.tile(&tile).expect_err("refused"),
    ));

    assert!(errors.len() >= 4);
    for (what, error) in &errors {
        assert_clean(&format!("{error}"), &canaries, what);
        assert_clean(&format!("{error:?}"), &canaries, what);
    }
}
