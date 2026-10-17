// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU engine against `D` (RFC-0018 'Witness engine', 07 §5).
//!
//! - **Conformance.** On the tiny conformance models of `spec/vectors/models/`, the engine gives
//!   the logits and the residual at every layer boundary that the independent Python oracle
//!   computed, bit for bit. Its logits leave as a `LogitsFile` that seals and opens back.
//! - **Range soundness.** On 10^3 adversarial runs, weights at ±127 in drawn sign patterns and
//!   tokens at the vocabulary's ends among them, every value stays inside its tensor's analyzed
//!   range (INV-DM-2), which debug builds check.
//! - **Determinism.** One, two, three or eight threads give the same value of every tensor.
//! - **Requests.** A sequence of another length, a token outside the vocabulary, a model without
//!   one of the lowering's weights or with a class its site lacks: each is refused with its code,
//!   as is each malformed input of a graph run on its own.
//! - **Operations.** The operations no lowering emits yet, `Sub`, `Slice`, `Sum`, `ConstAdd` and
//!   `Bitwise`, give RFC-0006's values on a graph built for them; a residual row of zeros takes its
//!   root from ε alone.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use tessera_codec::error::ErrorCode;
use tessera_codec::{DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower, validate};
use tessera_ir::{
    CommitmentClass, Graph, GraphBuilder, IntRange, IntType, IntegerModel, LogitsFile, NodeKind,
    Op, QuantPoint, Site, TableId, TensorDecl, TensorId, WeightType,
};
use tessera_tables::{BitOp, GeneratedTable, Table, TableKind};
use tessera_witness::{Engine, WitnessError};

fn models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models")
}

/// The conformance models, each with its index entry.
fn conformance() -> Vec<(IntegerModel, serde_json::Value)> {
    let text = fs::read_to_string(models().join("models.json")).expect("the index");
    let index: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    index["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|entry| {
            let bytes = fs::read(models().join(entry["file"].as_str().expect("a file")))
                .expect("the artifact");
            let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
            (model, entry.clone())
        })
        .collect()
}

fn ints(value: &serde_json::Value) -> Vec<i64> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .flat_map(|row| row.as_array().expect("a row"))
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

fn tokens(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("a u32"))
        .collect()
}

fn lowered(model: &IntegerModel, n: usize) -> Lowering {
    validate(model).expect("a valid model");
    lower(PublicModel::of(model), u32::try_from(n).expect("a length")).expect("the model lowers")
}

fn run(
    engine: &Engine<'_>,
    model: &IntegerModel,
    tokens: &[u32],
) -> Result<tessera_witness::Witness, WitnessError> {
    engine.run(model, &Secret::new(tokens.to_vec()))
}

fn values(witness: &tessera_witness::Witness, tensor: TensorId) -> Vec<i64> {
    witness.tensor(tensor).expect("the tensor").to_vec()
}

#[test]
fn the_engine_computes_the_conformance_vectors() {
    let mut checked = 0;
    for (model, entry) in conformance() {
        let name = entry["name"].as_str().expect("a name");
        for sequence in entry["sequences"].as_array().expect("sequences") {
            let tokens = tokens(&sequence["tokens"]);
            let lowering = lowered(&model, tokens.len());
            let engine = Engine::new(&lowering).expect("the tables regenerate");
            let witness = run(&engine, &model, &tokens).expect("a witness");
            let logits = witness.logits().expect("a model's logits");
            assert_eq!(
                logits.positions.concat(),
                ints(&sequence["logits"]),
                "{name}: the logits"
            );
            assert_eq!(logits.positions.len(), tokens.len());
            let residuals = sequence["residuals"].as_array().expect("residuals");
            assert_eq!(lowering.checkpoints.len(), residuals.len(), "{name}");
            for (k, (&tensor, expected)) in lowering.checkpoints.iter().zip(residuals).enumerate() {
                assert_eq!(
                    values(&witness, tensor),
                    ints(expected),
                    "{name}: checkpoint {k}"
                );
            }
            // The logits seal as artifact 0x0602 and open back.
            let sealed = envelope::seal(&schema::LOGITS_FILE, &logits);
            let opened: LogitsFile =
                envelope::open(&sealed, &schema::LOGITS_FILE, DecodeLimits::MESSAGE)
                    .expect("a LogitsFile");
            assert_eq!(opened, logits);
            assert!(!format!("{logits:?}").contains(&logits.positions[0][0].to_string()));
            checked += 1;
        }
    }
    assert_eq!(checked, 6, "every sequence of every model");
}

/// SplitMix64, for drawing adversarial inputs reproducibly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// `model` with every weight at ±127 in a drawn pattern: all one sign, signs by row, or signs
/// drawn per entry; and its channel classes drawn anew.
fn adversarial(model: &IntegerModel, rng: &mut Rng) -> IntegerModel {
    let mut model = model.clone();
    for w in &mut model.weights {
        let pattern = rng.below(4);
        let cols = w.cols as usize;
        for (i, x) in w.data.iter_mut().enumerate() {
            let positive = match pattern {
                0 => true,
                1 => false,
                2 => (i / cols).is_multiple_of(2),
                _ => rng.below(2) == 0,
            };
            *x = if positive { 127 } else { -127 };
        }
    }
    let sets = model.quant.public.shift_class_sets.clone();
    for params in &mut model.quant.channels {
        let site = match params.ty {
            WeightType::Q => Site::RopeQ,
            WeightType::K => Site::RopeK,
            WeightType::V => Site::V,
            WeightType::O => Site::AttnResidual,
            WeightType::Gate => Site::Silu,
            WeightType::Up => Site::Up,
            WeightType::Down => Site::MlpResidual,
            WeightType::Embed | WeightType::LmHead => panic!("no site after {:?}", params.ty),
        };
        let point = QuantPoint::Layer {
            layer: params.layer,
            site,
        };
        let (_, shifts) = sets.iter().find(|(p, _)| *p == point).expect("the site");
        let classes = u64::try_from(shifts.len()).expect("small");
        for c in &mut params.class {
            *c = u8::try_from(rng.below(classes)).expect("small");
        }
    }
    model
}

#[test]
fn every_value_stays_in_range_on_adversarial_runs() {
    let mut rng = Rng(0x0082_ADD5_E7A0_0001);
    let mut runs = 0;
    // The two tiny-2 models, each lowered for a few lengths, run on 10^3 adversarial inputs.
    for (model, entry) in conformance().into_iter().take(2) {
        let name = entry["name"].as_str().expect("a name").to_owned();
        let vocab = u64::from(model.arch.vocab);
        for n in [1, 2, 5, 9] {
            let lowering = lowered(&model, n);
            let engine = Engine::new(&lowering).expect("the tables regenerate");
            for _ in 0..125 {
                let variant = adversarial(&model, &mut rng);
                validate(&variant).expect("an adversarial model stays valid");
                let tokens: Vec<u32> = (0..n)
                    .map(|_| match rng.below(4) {
                        0 => 0,
                        1 => model.arch.vocab - 1,
                        _ => u32::try_from(rng.below(vocab)).expect("a token"),
                    })
                    .collect();
                if let Err(error) = run(&engine, &variant, &tokens) {
                    panic!("{name}, n = {n}: {error}");
                }
                runs += 1;
            }
        }
    }
    assert_eq!(runs, 1000);
}

#[test]
fn the_thread_count_changes_no_value() {
    let (model, entry) = conformance().pop().expect("tiny-4");
    let tokens = tokens(&entry["sequences"][1]["tokens"]);
    let lowering = lowered(&model, tokens.len());
    let one = NonZeroUsize::new(1).expect("one");
    let base = Engine::new(&lowering).expect("tables").threads(one);
    let reference = run(&base, &model, &tokens).expect("a witness");
    for threads in [2, 3, 8] {
        let engine = Engine::new(&lowering)
            .expect("tables")
            .threads(NonZeroUsize::new(threads).expect("positive"));
        let witness = run(&engine, &model, &tokens).expect("a witness");
        for t in 0..lowering.graph.tensors.len() {
            let id = TensorId(u32::try_from(t).expect("small"));
            assert_eq!(
                values(&witness, id),
                values(&reference, id),
                "{threads} threads, tensor {t}"
            );
        }
    }
}

#[test]
fn malformed_requests_and_models_are_refused_with_their_codes() {
    let (model, entry) = conformance().remove(0);
    let tokens = tokens(&entry["sequences"][0]["tokens"]);
    let lowering = lowered(&model, tokens.len());
    let engine = Engine::new(&lowering).expect("tables");
    let refused =
        |model: &IntegerModel, tokens: &[u32]| run(&engine, model, tokens).expect_err("refused");
    // A sequence of another length than the lowering's.
    let error = refused(&model, &tokens[1..]);
    assert_eq!(
        error,
        WitnessError::Usage {
            rule: "the sequence has as many tokens as the lowering's positions"
        }
    );
    assert_eq!(error.code(), ErrorCode::Usage);
    // A token outside the vocabulary.
    let mut outside = tokens.clone();
    outside[0] = model.arch.vocab;
    assert_eq!(
        refused(&model, &outside),
        WitnessError::Usage {
            rule: "every token lies inside the vocabulary"
        }
    );
    // A model without one of the lowering's weights.
    let mut missing = model.clone();
    missing.weights.pop();
    let error = refused(&missing, &tokens);
    assert_eq!(
        error,
        WitnessError::Model {
            rule: "the model holds every weight the lowering reads"
        }
    );
    assert_eq!(error.code(), ErrorCode::QuantArch);
    // A weight of another size.
    let mut short = model.clone();
    short.weights[1].data.pop();
    assert_eq!(
        refused(&short, &tokens),
        WitnessError::Model {
            rule: "each weight has the lowering's shape"
        }
    );
    // A channel whose class its site does not have.
    let mut stray = model.clone();
    stray.quant.channels[0].class[0] = 9;
    assert_eq!(
        refused(&stray, &tokens),
        WitnessError::Model {
            rule: "a channel's class indexes its site's shift classes"
        }
    );
    // A model without the classes of a classed site.
    let mut classless = model.clone();
    classless.quant.channels.pop();
    assert_eq!(
        refused(&classless, &tokens),
        WitnessError::Model {
            rule: "the model holds the channel classes of every classed site"
        }
    );
}

#[test]
fn a_value_outside_its_analyzed_range_is_an_internal_error() {
    // Debug builds check INV-DM-2: a lowering whose logits' range misses a value the engine
    // computes is refused at the logits, with ERR-INTERNAL.
    let (model, entry) = conformance().remove(0);
    let tokens = tokens(&entry["sequences"][0]["tokens"]);
    let mut lowering = lowered(&model, tokens.len());
    let logits = lowering.logits;
    lowering.graph.tensors[logits.0 as usize].range = IntRange { lo: 0, hi: 0 };
    let engine = Engine::new(&lowering).expect("tables");
    let error = run(&engine, &model, &tokens).expect_err("refused");
    assert_eq!(error, WitnessError::Range { tensor: logits });
    assert_eq!(error.code(), ErrorCode::Internal);
}

const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

fn decl(shape: &[u32], kind: NodeKind, lo: i128, hi: i128) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange { lo, hi },
        kind,
    }
}

fn buffer(values: &[i64]) -> SecretBuf<i64> {
    let mut b = SecretBuf::with_capacity(values.len());
    for &v in values {
        b.push(v);
    }
    b
}

/// A graph of `Sub`, `Slice`, `Sum`, `ConstAdd` and `Bitwise` over inputs `x` and `z` of shape
/// `[2, 3, 4]` and bytes `a` and `b` of shape `[6]`, with the `Xor` table it reads.
fn operations() -> (
    Graph,
    Vec<tessera_tables::TableSpec>,
    [TensorId; 4],
    [TensorId; 5],
) {
    let mut g = GraphBuilder::new();
    let wide = |shape: &[u32]| decl(shape, NodeKind::Virtual, -(1 << 20), 1 << 20);
    let x = g.input(decl(&[2, 3, 4], ADVICE, -100, 100)).expect("x");
    let z = g.input(decl(&[2, 3, 4], ADVICE, -100, 100)).expect("z");
    let a = g.input(decl(&[6], ADVICE, 0, 255)).expect("a");
    let b = g.input(decl(&[6], ADVICE, 0, 255)).expect("b");
    let c = g
        .constant(vec![1, -2, 3, -4])
        .expect("a constant per channel");
    let mut one = |op: Op, inputs: &[TensorId], output: TensorDecl| {
        g.node(op, inputs, vec![output]).expect("a node")[0]
    };
    let sub = one(Op::Sub, &[x, z], wide(&[2, 3, 4]));
    let slice = one(
        Op::Slice {
            axis: 1,
            start: 1,
            len: 2,
        },
        &[x],
        wide(&[2, 2, 4]),
    );
    let sum = one(Op::Sum { axis: 1 }, &[x], wide(&[2, 4]));
    let add = one(Op::ConstAdd { c }, &[x], wide(&[2, 3, 4]));
    let xor = one(
        Op::Bitwise { table: TableId(0) },
        &[a, b],
        decl(&[6], NodeKind::LookupOutput { table: TableId(0) }, 0, 255),
    );
    let table = GeneratedTable::new(TableId(0), TableKind::Bitwise8 { op: BitOp::Xor })
        .expect("the Xor table");
    (
        g.finish(),
        vec![table.spec().clone()],
        [x, z, a, b],
        [sub, slice, sum, add, xor],
    )
}

#[test]
fn the_operations_no_lowering_emits_yet_follow_rfc_0006() {
    let (graph, tables, [x, z, a, b], [sub, slice, sum, add, xor]) = operations();
    let engine = Engine::for_graph(&graph, &tables).expect("the tables");
    let xs: Vec<i64> = (0..24).map(|i| (i * 37) % 91 - 45).collect();
    let zs: Vec<i64> = (0..24).map(|i| (i * 53) % 97 - 48).collect();
    let (as_, bs) = ([0, 255, 17, 128, 85, 3], [255, 255, 34, 1, 170, 3]);
    let witness = engine
        .evaluate(vec![
            (x, buffer(&xs)),
            (z, buffer(&zs)),
            (a, buffer(&as_)),
            (b, buffer(&bs)),
        ])
        .expect("a witness");
    // Element (o, j, k) of a [2, 3, 4] tensor is at (o · 3 + j) · 4 + k.
    let at = |o: usize, j: usize, k: usize| xs[(o * 3 + j) * 4 + k];
    let expected_sub: Vec<i64> = xs.iter().zip(&zs).map(|(p, q)| p - q).collect();
    let mut expected_slice = Vec::new();
    let mut expected_sum = Vec::new();
    for o in 0..2 {
        for j in 1..3 {
            expected_slice.extend((0..4).map(|k| at(o, j, k)));
        }
        expected_sum.extend((0..4).map(|k| (0..3).map(|j| at(o, j, k)).sum::<i64>()));
    }
    let expected_add: Vec<i64> = xs
        .iter()
        .enumerate()
        .map(|(i, v)| v + [1, -2, 3, -4][i % 4])
        .collect();
    let expected_xor: Vec<i64> = as_.iter().zip(&bs).map(|(p, q)| p ^ q).collect();
    assert_eq!(values(&witness, sub), expected_sub);
    assert_eq!(values(&witness, slice), expected_slice);
    assert_eq!(values(&witness, sum), expected_sum);
    assert_eq!(values(&witness, add), expected_add);
    assert_eq!(values(&witness, xor), expected_xor);
    assert!(
        witness.logits().is_none(),
        "a graph's run has no model's logits"
    );
}

#[test]
fn a_graphs_inputs_are_each_given_once_whole_and_in_range() {
    let (graph, tables, [x, z, a, b], [sub, ..]) = operations();
    let engine = Engine::for_graph(&graph, &tables).expect("the tables");
    let zeros = |n: usize| buffer(&vec![0; n]);
    let all = || vec![(x, zeros(24)), (z, zeros(24)), (a, zeros(6)), (b, zeros(6))];
    let refused = |inputs: Vec<(TensorId, SecretBuf<i64>)>| {
        let error = engine.evaluate(inputs).expect_err("refused");
        assert_eq!(error.code(), ErrorCode::Usage);
        error
    };
    let rule = |error: WitnessError| match error {
        WitnessError::Usage { rule } => rule,
        other => panic!("a usage error, found {other:?}"),
    };
    assert!(engine.evaluate(all()).is_ok());
    let mut missing = all();
    missing.pop();
    assert_eq!(
        rule(refused(missing)),
        "every tensor no node produces is given"
    );
    let mut twice = all();
    twice.push((a, zeros(6)));
    assert_eq!(rule(refused(twice)), "each input is given once");
    let mut short = all();
    short[0] = (x, zeros(23));
    assert_eq!(
        rule(refused(short)),
        "an input holds its shape's number of values"
    );
    let mut outside = all();
    outside[2] = (a, buffer(&[0, 0, 0, 0, 0, 256]));
    assert_eq!(
        rule(refused(outside)),
        "every input value lies in its tensor's range"
    );
    let mut produced = all();
    produced.push((sub, zeros(24)));
    assert_eq!(
        rule(refused(produced)),
        "an input is a tensor no node produces"
    );
    let mut foreign = all();
    foreign.push((TensorId(99), zeros(1)));
    assert_eq!(rule(refused(foreign)), "an input is a tensor of the graph");
    // A model runs only on the engine of its lowering.
    let (model, _) = conformance().remove(0);
    assert_eq!(
        engine
            .run(&model, &Secret::new(vec![0]))
            .expect_err("refused"),
        WitnessError::Usage {
            rule: "a model runs on the engine of its lowering"
        }
    );
}

#[test]
fn a_zero_residual_row_takes_its_root_from_epsilon_alone() {
    // A token whose embedding column is zero gives a residual row of zeros: its sum of squares is
    // 0, and RFC-0005's root is the largest ρ with ρ² e ≤ d · 2^{2κ}.
    let (mut model, entry) = conformance().remove(0);
    let tokens = tokens(&entry["sequences"][0]["tokens"]);
    let token = tokens[1] as usize;
    let embed = model
        .weights
        .iter_mut()
        .find(|w| w.ty == WeightType::Embed)
        .expect("the embedding");
    let vocab = embed.cols as usize;
    for c in 0..embed.rows as usize {
        embed.data[c * vocab + token] = 0;
    }
    let lowering = lowered(&model, tokens.len());
    let engine = Engine::new(&lowering).expect("tables");
    let witness = run(&engine, &model, &tokens).expect("a witness");
    let node = lowering
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::RsqrtRow { .. }))
        .expect("the first norm's root");
    let Op::RsqrtRow {
        kappa,
        eps_int,
        dim,
    } = node.op
    else {
        unreachable!()
    };
    let bound = u128::from(dim) << (2 * u32::from(kappa));
    let roots = values(&witness, node.outputs[0]);
    let mut zero_rows = 0;
    for (p, &t) in tokens.iter().enumerate() {
        if t as usize == token {
            let rho = u128::try_from(roots[p]).expect("a root");
            let e = u128::from(eps_int);
            assert!(
                rho * rho * e <= bound && bound < (rho + 1) * (rho + 1) * e,
                "position {p}"
            );
            zero_rows += 1;
        }
    }
    assert!(zero_rows >= 1);
}
