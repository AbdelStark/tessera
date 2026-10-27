// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment of RFC-0040 against its definitions:
//!
//! - **Positions** (RFC-0040 'The matrices', RFC-0012 'Testing Strategy'). For random small `D`,
//!   every weight and channel parameter is read back from its matrix at the position the RFC
//!   gives, every other entry is 0, and the committed rows are exactly the rows that are not
//!   structurally zero, each rule written out here from the RFC rather than taken from the crate.
//!   The cases cover an input wider than `k_W`, widths and layer counts that are not powers of
//!   two, rows that hold padded outputs, and models with and without `ChannelParams`.
//! - **Binding** (INV-WC-3). `R_W` changes with any one weight or channel parameter, any byte of
//!   the layout, the epoch, and any one root; and the epoch and the layout separate every
//!   matrix's chains (RFC-0031).
//! - **Widths** (RFC-0040 'Testing Strategy'). 405B's and 8B's shapes at `live-64/1` take the
//!   `k_T`, `a_T` and `ChannelParams` rows that RFC-0012 'Layout' states.
//! - **Checks.** Each input the layout and the commitment refuse, with its error.

mod common;

use std::num::{NonZeroU32, NonZeroUsize};

use common::SplitMix64;
use tessera_codec::{Secret, decode, encode};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_pcs::{
    ChannelParamsRef, ChannelValues, CommitmentDescriptor, CommitmentParams, MatrixKind,
    ModelWeights, PcsError, ReferenceCommitment, RowEncoder, RowLayout, WeightCommitment,
    WeightCommitmentPublic, WeightLayout, WeightLayoutParams, WeightMatrix, WeightShape,
    WeightShapes, WeightTensorRef,
};

type Fp = Goldilocks;

/// `test-small/1`'s weight layout (RFC-0026), with `log2 k_W` set by the case.
fn params(log2_k: u8) -> WeightLayoutParams {
    WeightLayoutParams {
        log2_k,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    }
}

/// The next power of two of `n ≥ 1`, from its definition.
fn pad(n: u64) -> u64 {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

fn log2(p: u64) -> u32 {
    assert!(p.is_power_of_two());
    p.trailing_zeros()
}

/// A `ChannelParams`' values, owned: class indices or multipliers.
#[derive(Clone)]
enum Values {
    Classes(Vec<u8>),
    Multipliers(Vec<u16>),
}

/// Each row of a full matrix, `None` when structurally zero.
type Rows = Vec<Option<Vec<Fp>>>;

/// A position case: layers, `(out, in)` of each type, the weights with a `ChannelParams` (with
/// classes or not), and `log2 k_W`.
type Case = (u32, [(u32, u32); 9], Vec<(WeightMatrix, u32, bool)>, u8);

/// A random small `D`: its shapes, its weights and its channel parameters, owned.
struct Model {
    shapes: WeightShapes,
    tensors: Vec<(WeightMatrix, u32, Vec<i8>)>,
    channels: Vec<(WeightMatrix, u32, Values)>,
}

impl Model {
    /// `shapes[t] = (out, in)` for `Embed` to `LmHead`; `channels` the weights that have a
    /// `ChannelParams`, each with classes (`true`) or multipliers.
    fn new(
        rng: &mut SplitMix64,
        layers: u32,
        shapes: [(u32, u32); 9],
        channels: &[(WeightMatrix, u32, bool)],
    ) -> Self {
        let shapes = WeightShapes {
            layers,
            shapes: shapes.map(|(out, inputs)| WeightShape { out, inputs }),
            channels: channels.iter().map(|&(m, l, _)| (m, l)).collect(),
        };
        let mut tensors = Vec::new();
        for matrix in WeightMatrix::WEIGHTS {
            let shape = shapes.shapes[matrix.index()];
            let count = if matrix.per_layer() { layers } else { 1 };
            for layer in 0..count {
                let len = (shape.out * shape.inputs) as usize;
                // Every INT8 value can occur, both ends included.
                let data = (0..len).map(|_| rng.next_u64() as i8).collect();
                tensors.push((matrix, layer, data));
            }
        }
        let channels = channels
            .iter()
            .map(|&(matrix, layer, classes)| {
                let out = shapes.shapes[matrix.index()].out as usize;
                let values = match classes {
                    true => Values::Classes((0..out).map(|_| rng.below(4) as u8).collect()),
                    false => Values::Multipliers(
                        (0..out).map(|_| 1 + rng.below(0xFFFF) as u16).collect(),
                    ),
                };
                (matrix, layer, values)
            })
            .collect();
        Self {
            shapes,
            tensors,
            channels,
        }
    }

    fn weights(&self) -> ModelWeights<'_> {
        ModelWeights {
            tensors: self
                .tensors
                .iter()
                .map(|(matrix, layer, data)| WeightTensorRef {
                    matrix: *matrix,
                    layer: *layer,
                    data,
                })
                .collect(),
            channels: self
                .channels
                .iter()
                .map(|(matrix, layer, values)| ChannelParamsRef {
                    matrix: *matrix,
                    layer: *layer,
                    values: match values {
                        Values::Classes(classes) => ChannelValues::Classes(classes),
                        Values::Multipliers(multipliers) => ChannelValues::Multipliers(multipliers),
                    },
                })
                .collect(),
        }
    }
}

/// The matrices of a model, as RFC-0040 'The matrices' defines them: for each, `log2 k`, the
/// row index's bits, and every row of the full matrix, `None` when structurally zero.
fn defined_matrices(model: &Model, log2_k_w: u32) -> Vec<(u32, u32, Rows)> {
    let k_w = 1u64 << log2_k_w;
    let n_layers = u64::from(model.shapes.layers);
    let l_pad = pad(n_layers);
    let mut out = Vec::new();
    for matrix in WeightMatrix::WEIGHTS {
        let shape = model.shapes.shapes[matrix.index()];
        let (o, i) = (u64::from(shape.out), u64::from(shape.inputs));
        let (out_pad, in_pad) = (pad(o), pad(i));
        let (layers, layers_pad) = match matrix.per_layer() {
            true => (n_layers, l_pad),
            false => (1, 1),
        };
        let k = in_pad.max(k_w.min(out_pad * in_pad));
        let a = log2(k / in_pad);
        let h = log2(out_pad) - a;
        let mut rows = vec![None; (layers_pad << h) as usize];
        // A row is committed unless ℓ ≥ n_layers or ⌊o / 2^a⌋·2^a ≥ out.
        for (r, row) in rows.iter_mut().enumerate() {
            let (layer, o_hi) = (r as u64 >> h, r as u64 & ((1 << h) - 1));
            if layer < layers && (o_hi << a) < o {
                *row = Some(vec![Fp::ZERO; k as usize]);
            }
        }
        for (m, layer, data) in &model.tensors {
            if *m != matrix {
                continue;
            }
            for (index, &w) in data.iter().enumerate() {
                let (oo, ii) = (index as u64 / i, index as u64 % i);
                // W_ℓ[o][i] at row ℓ·2^h + ⌊o / 2^a⌋, column (o mod 2^a)·in_pad + i.
                let r = (u64::from(*layer) << h) + (oo >> a);
                let column = (oo & ((1 << a) - 1)) * in_pad + ii;
                let row = rows[r as usize]
                    .as_mut()
                    .expect("a weight lies in a committed row");
                row[column as usize] = Fp::from_i64(i64::from(w));
            }
        }
        out.push((log2(k), log2(layers_pad) + h, rows));
    }
    // ChannelParams: f = (t·L_pad + ℓ)·C + c, at row ⌊f / k⌋ and column f mod k.
    let c = model
        .channels
        .iter()
        .map(|(m, _, _)| pad(u64::from(model.shapes.shapes[m.index()].out)))
        .max()
        .unwrap_or(1);
    let total = 16 * l_pad * c;
    let k = k_w.min(total);
    let mut rows: Rows = vec![None; (total / k) as usize];
    for (matrix, layer, values) in &model.channels {
        let values: Vec<u64> = match values {
            Values::Classes(classes) => classes.iter().map(|&x| u64::from(x)).collect(),
            Values::Multipliers(multipliers) => multipliers.iter().map(|&x| u64::from(x)).collect(),
        };
        let base = (matrix.index() as u64 * l_pad + u64::from(*layer)) * c;
        for (channel, &value) in values.iter().enumerate() {
            let f = base + channel as u64;
            let row = rows[(f / k) as usize].get_or_insert_with(|| vec![Fp::ZERO; k as usize]);
            row[(f % k) as usize] = Fp::from_u64_reduced(value);
        }
    }
    out.push((log2(k), log2(total / k), rows));
    out
}

/// Checks every matrix of `model` against the definitions: layout, committed rows and entries.
fn check_positions(model: &Model, log2_k_w: u8) {
    let layout = WeightLayout::new(params(log2_k_w), model.shapes.clone()).expect("a layout");
    let weights = model.weights();
    for ((log2_k, row_vars, rows), matrix) in defined_matrices(model, u32::from(log2_k_w))
        .into_iter()
        .zip(WeightMatrix::ALL)
    {
        let found = layout.matrix(matrix);
        assert_eq!(
            (found.log2_k(), found.row_vars()),
            (log2_k, row_vars),
            "{matrix:?}"
        );
        let committed: Vec<u64> = (0..rows.len() as u64)
            .filter(|&r| rows[r as usize].is_some())
            .collect();
        assert_eq!(
            layout.committed_rows(matrix).collect::<Vec<_>>(),
            committed,
            "{matrix:?}: exactly the rows that are not structurally zero"
        );
        assert_eq!(found.rows(), committed.len() as u64, "{matrix:?}: m_T");
        let expected: Vec<Fp> = rows.into_iter().flatten().flatten().collect();
        // Over a buffer that holds garbage: every entry is written.
        let mut values = vec![Fp::ONE; expected.len()];
        layout
            .write_rows(matrix, &weights, &mut values)
            .expect("valid weights");
        assert_eq!(values, expected, "{matrix:?}: every entry at its position");
    }
}

const NO_CHANNELS: &[(WeightMatrix, u32, bool)] = &[];

#[test]
fn every_weight_and_channel_parameter_lies_at_its_documented_position() {
    use WeightMatrix::*;
    let mut rng = SplitMix64::new(0x7E55_E7A0_0102_0001);
    // Llama-like shapes: (out, in) for Embed, Q, K, V, O, Gate, Up, Down, LmHead.
    let llama = |d, q, kv, ff, vocab| {
        [
            (d, vocab),
            (q, d),
            (kv, d),
            (kv, d),
            (d, q),
            (ff, d),
            (ff, d),
            (d, ff),
            (vocab, d),
        ]
    };
    // Widths, layers and vocabulary that are not powers of two; inputs wider than k_W = 16
    // (the embedding's 300 and the down projection's 20); rows that pad outputs; a k_T below
    // k_W; no ChannelParams; classes and multipliers, of every weight type.
    let cases: Vec<Case> = vec![
        (3, llama(12, 12, 6, 20, 300), vec![], 8),
        (3, llama(12, 12, 6, 20, 300), vec![], 4),
        (
            5,
            llama(10, 10, 5, 13, 37),
            vec![
                (Q, 0, true),
                (K, 4, false),
                (Gate, 2, true),
                (Down, 3, false),
            ],
            6,
        ),
        (
            1,
            llama(7, 7, 3, 9, 11),
            vec![
                (Embed, 0, false),
                (V, 0, true),
                (O, 0, true),
                (LmHead, 0, true),
            ],
            3,
        ),
        (
            2,
            llama(16, 16, 8, 24, 64),
            vec![(Q, 0, true), (Q, 1, true), (Up, 1, false), (Down, 0, true)],
            8,
        ),
        // Every weight has a ChannelParams: a row of the tensor holds several weights' blocks.
        (
            2,
            llama(4, 4, 2, 6, 5),
            WeightMatrix::WEIGHTS
                .into_iter()
                .flat_map(|m| {
                    let layers = if m.per_layer() { 2 } else { 1 };
                    (0..layers).map(move |l| (m, l, l == 0))
                })
                .collect(),
            8,
        ),
    ];
    for (layers, shapes, channels, log2_k) in cases {
        let model = Model::new(&mut rng, layers, shapes, &channels);
        check_positions(&model, log2_k);
    }
    // And 50 random small models.
    for _ in 0..50 {
        let dim = |rng: &mut SplitMix64| 1 + rng.below(40) as u32;
        let shapes = core::array::from_fn(|_| (dim(&mut rng), dim(&mut rng)));
        let layers = 1 + rng.below(5) as u32;
        let mut channels: Vec<(WeightMatrix, u32, bool)> = Vec::new();
        for m in WeightMatrix::WEIGHTS {
            let count = if m.per_layer() { layers } else { 1 };
            for l in 0..count {
                if rng.below(3) == 0 {
                    channels.push((m, l, rng.below(2) == 0));
                }
            }
        }
        let model = Model::new(&mut rng, layers, shapes, &channels);
        check_positions(&model, 2 + rng.below(7) as u8);
    }
}

#[test]
fn a_model_with_no_channel_parameters_commits_no_row_of_their_matrix() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0102_0002);
    let model = Model::new(&mut rng, 3, [(5, 6); 9], NO_CHANNELS);
    let layout = WeightLayout::new(params(8), model.shapes.clone()).expect("a layout");
    let channels = layout.matrix(WeightMatrix::ChannelParams);
    // C = 1: k_CP = min(k_W, 16·L_pad·1) = 64, one row, not committed.
    assert_eq!(
        channels.kind(),
        MatrixKind::ChannelParams { channel_width: 1 }
    );
    assert_eq!(
        (channels.log2_k(), channels.row_vars(), channels.rows()),
        (6, 0, 0)
    );
    let commitment = WeightCommitment::commit(
        &layout,
        NonZeroU32::MIN.saturating_add(3),
        0,
        &model.weights(),
    )
    .expect("a commitment");
    // Its leaves are the chains' first digests: H("column-init", commitment_id ‖ u64(j)).
    let id = CommitmentDescriptor::WeightMatrix {
        layout: params(8),
        matrix: WeightMatrix::ChannelParams,
        log2_k: 6,
        row_block: 4,
        rows: 0,
        epoch: 0,
    }
    .commitment_id();
    let matrix = commitment.matrix(WeightMatrix::ChannelParams);
    assert_eq!(matrix.leaves().len(), 256, "n = 4·64");
    for (j, leaf) in matrix.leaves().iter().enumerate() {
        let init = tessera_hash::Hasher::new(tessera_hash::Domain::ColumnInit)
            .update(&id)
            .update(&(j as u64).to_le_bytes())
            .finalize();
        assert_eq!(*leaf, init, "column {j}");
    }
}

/// A small model, its layout and its commitment at epoch 0, `B = 4`.
fn committed(seed: u64) -> (Model, WeightLayout, WeightCommitment) {
    use WeightMatrix::*;
    let mut rng = SplitMix64::new(seed);
    let shapes = [
        (6, 20),
        (6, 6),
        (3, 6),
        (3, 6),
        (6, 6),
        (9, 6),
        (9, 6),
        (6, 9),
        (20, 6),
    ];
    let channels = [(K, 0, true), (Up, 1, false), (LmHead, 0, true)];
    let model = Model::new(&mut rng, 2, shapes, &channels);
    let layout = WeightLayout::new(params(5), model.shapes.clone()).expect("a layout");
    let commitment = WeightCommitment::commit(&layout, block(4), 0, &model.weights()).expect("ok");
    (model, layout, commitment)
}

fn block(b: u32) -> NonZeroU32 {
    NonZeroU32::new(b).expect("non-zero")
}

/// INV-WC-3: changing one weight or one channel parameter changes its matrix's root, no other
/// root, and `R_W`.
#[test]
fn the_weights_root_binds_every_weight_and_channel_parameter() {
    let (model, layout, honest) = committed(0x7E55_E7A0_0102_0003);
    let r_w = honest.weights_root();
    let mut rng = SplitMix64::new(0x7E55_E7A0_0102_0004);
    let changes = model.tensors.len() + model.channels.len();
    for change in 0..changes {
        let mut changed = Model {
            shapes: model.shapes.clone(),
            tensors: model.tensors.clone(),
            channels: model.channels.clone(),
        };
        let matrix = if change < model.tensors.len() {
            let (matrix, _, data) = &mut changed.tensors[change];
            let at = rng.below(data.len());
            data[at] = data[at].wrapping_add(1 + rng.below(255) as i8);
            *matrix
        } else {
            match &mut changed.channels[change - model.tensors.len()].2 {
                Values::Classes(classes) => {
                    let at = rng.below(classes.len());
                    classes[at] ^= 1;
                }
                Values::Multipliers(multipliers) => {
                    let at = rng.below(multipliers.len());
                    multipliers[at] ^= 0x8000;
                }
            }
            WeightMatrix::ChannelParams
        };
        let other = WeightCommitment::commit(&layout, block(4), 0, &changed.weights()).expect("ok");
        for m in WeightMatrix::ALL {
            let same = honest.public().root(m) == other.public().root(m);
            assert_eq!(
                same,
                m != matrix,
                "change {change} in {matrix:?}: root of {m:?}"
            );
        }
        assert_ne!(other.weights_root(), r_w, "change {change}");
    }
}

/// INV-WC-3: `R_W` changes with any byte of the layout, the epoch, or any byte of any root.
#[test]
fn the_weights_root_binds_the_layout_the_roots_and_the_epoch() {
    let (_, _, honest) = committed(0x7E55_E7A0_0102_0005);
    let public = *honest.public();
    let r_w = public.weights_root();
    let bytes = encode(&public);
    assert_eq!(bytes.len(), 10 + 320 + 4, "layout, ten roots, epoch");
    // Every field is an integer or a byte, so a changed byte is another public part: the
    // layout's ten bytes, every byte of every root, the epoch's four.
    for at in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[at] ^= 1 << (at % 8);
        let other: WeightCommitmentPublic =
            decode(&changed, tessera_codec::DecodeLimits::MESSAGE).expect("still a value");
        assert_ne!(other, public);
        assert_ne!(
            other.weights_root(),
            r_w,
            "byte {at} of TCE(WeightCommitmentPublic)"
        );
    }
    // R_W is the hash of exactly these bytes.
    assert_eq!(
        r_w,
        tessera_hash::hash(tessera_hash::Domain::WeightsRoot, &bytes)
    );
}

/// RFC-0031: the epoch and the layout enter every matrix's `commitment_id`, so committing the
/// same weights at another epoch, or under a layout that differs only in `P_W` or `B_W`, which
/// `tessera-protocol/1` does not use, changes every root.
#[test]
fn the_epoch_and_the_layout_separate_every_matrix() {
    let (model, layout, honest) = committed(0x7E55_E7A0_0102_0006);
    let weights = model.weights();
    let next = WeightCommitment::commit(&layout, block(4), 1, &weights).expect("ok");
    let mut others = vec![("epoch 1", next)];
    for (name, changed) in [
        (
            "P_W + 1",
            WeightLayoutParams {
                mask_coeffs: 65,
                ..params(5)
            },
        ),
        (
            "B_W + 1",
            WeightLayoutParams {
                blinding_rows: 33,
                ..params(5)
            },
        ),
    ] {
        let layout = WeightLayout::new(changed, model.shapes.clone()).expect("a layout");
        others.push((
            name,
            WeightCommitment::commit(&layout, block(4), 0, &weights).expect("ok"),
        ));
    }
    // And another row block, which the descriptor binds and the chains use.
    others.push((
        "B = 2",
        WeightCommitment::commit(&layout, block(2), 0, &weights).expect("ok"),
    ));
    for (name, other) in others {
        for m in WeightMatrix::ALL {
            assert_ne!(
                other.public().root(m),
                honest.public().root(m),
                "{name}: {m:?}"
            );
        }
        assert_ne!(other.weights_root(), honest.weights_root(), "{name}");
    }
}

/// RFC-0040 'Encoding, leaves and roots': each matrix is encoded at rate 1/4, `n_T = 4·k_T`
/// columns, with no mask.
#[test]
fn every_matrix_is_committed_at_rate_one_quarter_with_no_mask() {
    let (_, layout, commitment) = committed(0x7E55_E7A0_0102_0007);
    for m in WeightMatrix::ALL {
        let matrix = commitment.matrix(m);
        let k = layout.matrix(m).message_len();
        assert_eq!(
            (matrix.message_len(), matrix.codeword_len()),
            (k, 4 * k),
            "{m:?}"
        );
        assert_eq!(matrix.params().mask_coeffs, 0, "{m:?}");
        assert_eq!(matrix.params().log2_inv_rate, 2, "{m:?}");
    }
}

#[test]
fn the_layout_refuses_shapes_it_cannot_lay_out() {
    use WeightMatrix::*;
    let shapes = |out_q: u32, in_q: u32, channels: Vec<(WeightMatrix, u32)>| WeightShapes {
        layers: 2,
        shapes: core::array::from_fn(|t| match t {
            1 => WeightShape {
                out: out_q,
                inputs: in_q,
            },
            _ => WeightShape { out: 4, inputs: 4 },
        }),
        channels,
    };
    let new = |p, s| WeightLayout::new(p, s).map(|_| ());
    assert_eq!(new(params(8), shapes(4, 4, vec![])), Ok(()));
    assert_eq!(
        new(params(8), shapes(0, 4, vec![])),
        Err(PcsError::EmptyWeight { matrix: Q })
    );
    assert_eq!(
        new(params(8), shapes(4, 0, vec![])),
        Err(PcsError::EmptyWeight { matrix: Q })
    );
    // Rate 1: no code.
    let rate_one = WeightLayoutParams {
        log2_inv_rate: 0,
        ..params(8)
    };
    assert_eq!(
        new(rate_one, shapes(4, 4, vec![])),
        Err(PcsError::InverseRate { log2_inv_rate: 0 })
    );
    // An input of 2^31 needs a message of 2^31, and a codeword of 2^33.
    assert_eq!(
        new(params(8), shapes(4, 1 << 31, vec![])),
        Err(PcsError::TooLong { log_len: 33 })
    );
    // ChannelParams of what is no weight: the channel parameters' matrix, a layer the type does
    // not have, a layer other than 0 for Embed and LmHead; and of one weight twice.
    for (matrix, layer) in [(ChannelParams, 0), (Q, 2), (Embed, 1), (LmHead, 1)] {
        assert_eq!(
            new(params(8), shapes(4, 4, vec![(V, 0), (matrix, layer)])),
            Err(PcsError::WeightPlace {
                matrix,
                layer,
                channels: true
            }),
            "{matrix:?} {layer}"
        );
    }
    assert_eq!(
        new(params(8), shapes(4, 4, vec![(V, 1), (Q, 0), (V, 1)])),
        Err(PcsError::WeightTwice {
            matrix: V,
            layer: 1,
            channels: true
        })
    );
}

#[test]
fn the_commitment_refuses_weights_that_do_not_fit_the_shapes() {
    use WeightMatrix::*;
    let mut rng = SplitMix64::new(0x7E55_E7A0_0102_0008);
    let model = Model::new(
        &mut rng,
        2,
        [(3, 5); 9],
        &[(Gate, 1, true), (Embed, 0, false)],
    );
    let layout = WeightLayout::new(params(4), model.shapes.clone()).expect("a layout");
    let commit = |weights: &ModelWeights<'_>| {
        WeightCommitment::commit(&layout, block(4), 0, weights).map(|_| ())
    };
    assert_eq!(commit(&model.weights()), Ok(()));
    let tensor = |matrix, layer, data: &'static [i8]| WeightTensorRef {
        matrix,
        layer,
        data,
    };
    let fifteen: &'static [i8] = &[1; 15];

    // A tensor with no place: the channel parameters' matrix, a layer past n_layers, a layer
    // other than 0 for LmHead.
    for (matrix, layer) in [(ChannelParams, 0), (O, 2), (LmHead, 1)] {
        let mut weights = model.weights();
        weights.tensors.push(tensor(matrix, layer, fifteen));
        assert_eq!(
            commit(&weights),
            Err(PcsError::WeightPlace {
                matrix,
                layer,
                channels: false
            })
        );
    }
    // A tensor of the wrong length.
    let mut weights = model.weights();
    weights.tensors[3].data = &fifteen[..14];
    let (matrix, layer) = (weights.tensors[3].matrix, weights.tensors[3].layer);
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightLength {
            matrix,
            layer,
            channels: false,
            expected: 15,
            found: 14
        })
    );
    // A tensor twice, and one missing.
    let mut weights = model.weights();
    weights.tensors.push(tensor(Down, 1, fifteen));
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightTwice {
            matrix: Down,
            layer: 1,
            channels: false
        })
    );
    let mut weights = model.weights();
    weights.tensors.retain(|t| (t.matrix, t.layer) != (Up, 0));
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightMissing {
            matrix: Up,
            layer: 0,
            channels: false
        })
    );
    let mut weights = model.weights();
    weights.tensors.retain(|t| t.matrix != LmHead);
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightMissing {
            matrix: LmHead,
            layer: 0,
            channels: false
        })
    );

    // Channel parameters the shapes do not list, of the wrong length, twice, or missing.
    let classes: &'static [u8] = &[0, 1, 2];
    let channels = |matrix, layer, values| ChannelParamsRef {
        matrix,
        layer,
        values,
    };
    let mut weights = model.weights();
    weights
        .channels
        .push(channels(Gate, 0, ChannelValues::Classes(classes)));
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightPlace {
            matrix: Gate,
            layer: 0,
            channels: true
        })
    );
    let mut weights = model.weights();
    weights.channels[0].values = ChannelValues::Classes(&classes[..2]);
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightLength {
            matrix: Gate,
            layer: 1,
            channels: true,
            expected: 3,
            found: 2
        })
    );
    let mut weights = model.weights();
    weights
        .channels
        .push(channels(Embed, 0, ChannelValues::Multipliers(&[7, 8, 9])));
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightTwice {
            matrix: Embed,
            layer: 0,
            channels: true
        })
    );
    // The model lists Gate's then Embed's; with Embed's gone, it is the first missing.
    let mut weights = model.weights();
    weights.channels.remove(1);
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightMissing {
            matrix: Embed,
            layer: 0,
            channels: true
        })
    );
    let mut weights = model.weights();
    weights.channels.remove(0);
    assert_eq!(
        commit(&weights),
        Err(PcsError::WeightMissing {
            matrix: Gate,
            layer: 1,
            channels: true
        })
    );
    // write_rows checks the same, then the buffer's length: Q, 3 × 5 at k_W = 16, has
    // k_T = 16, a_T = 1, so ⌈3/2⌉ = 2 rows in each of its 2 layers.
    let mut out = vec![Fp::ZERO; 3];
    assert_eq!(
        layout.write_rows(Q, &model.weights(), &mut out),
        Err(PcsError::MatrixLength {
            rows: 4,
            row_len: 16,
            found: 3
        })
    );
}

/// RFC-0040: a weight matrix is one chain per column, with no mask; the advice keeps refusing
/// `P = 0` (RFC-0011 'Encoding').
#[test]
fn an_unmasked_commitment_has_one_segment_and_no_mask() {
    let code = |log2_inv_rate, mask_coeffs| CommitmentParams {
        log2_k: 2,
        log2_inv_rate,
        mask_coeffs,
        row_block: NonZeroUsize::new(4).expect("non-zero"),
        commitment_id: [9; 32],
    };
    let rows = || Secret::new(vec![Fp::ONE; 8]);
    let one = || RowLayout::new(2, vec![vec![0, 3]]).expect("a layout");
    assert!(ReferenceCommitment::commit_unmasked(code(2, 0), one(), rows()).is_ok());
    assert!(ReferenceCommitment::commit_unmasked(code(1, 0), one(), rows()).is_ok());
    assert_eq!(
        ReferenceCommitment::commit_unmasked(code(2, 1), one(), rows()).map(|_| ()),
        Err(PcsError::UnexpectedMask { mask_coeffs: 1 })
    );
    for segments in [vec![], vec![vec![0], vec![3]]] {
        let count = segments.len();
        let layout = RowLayout::new(2, segments);
        let layout = match layout {
            Ok(layout) => layout,
            Err(error) => {
                assert_eq!(error, PcsError::NoSegments);
                continue;
            }
        };
        assert_eq!(
            ReferenceCommitment::commit_unmasked(code(2, 0), layout, rows()).map(|_| ()),
            Err(PcsError::Segments { segments: count })
        );
    }
    assert_eq!(
        ReferenceCommitment::commit_unmasked(code(0, 0), one(), rows()).map(|_| ()),
        Err(PcsError::InverseRate { log2_inv_rate: 0 })
    );
    // 2^31 values at rate 1/4 need 2^33 points.
    let mut long = code(2, 0);
    long.log2_k = 31;
    assert_eq!(
        ReferenceCommitment::commit_unmasked(long, one(), rows()).map(|_| ()),
        Err(PcsError::TooLong { log_len: 33 })
    );
    // The advice keeps its checks, at any rate.
    let mut none = || Fp::ZERO;
    let mut advice = |p| ReferenceCommitment::commit(p, one(), rows(), &mut none).map(|_| ());
    assert_eq!(advice(code(1, 0)), Err(PcsError::NoMask));
    assert_eq!(advice(code(2, 0)), Err(PcsError::NoMask));
    assert_eq!(
        advice(code(0, 1)),
        Err(PcsError::InverseRate { log2_inv_rate: 0 })
    );
    // The encoder refuses rate 1 too.
    assert_eq!(
        RowEncoder::new(4, 0, 0).map(|_| ()),
        Err(PcsError::InverseRate { log2_inv_rate: 0 })
    );
    assert_eq!(
        RowEncoder::new(1 << 30, 3, 0).map(|_| ()),
        Err(PcsError::TooLong { log_len: 33 })
    );
}

/// RFC-0040 'Testing Strategy' and RFC-0012 'Layout': at `live-64/1`, `k_W = 2^20`, every
/// matrix of Llama-3.1 405B has `k_T = 2^20`, with `a = 6` for the 16,384-wide inputs, 4 for the
/// down projection's 53,248 inputs padded to 65,536, and 3 for the 2^17-wide embedding. Its
/// `ChannelParams` tensor has `C = 2^16`, Gate's and Up's: a row holds 16 layers' values of one
/// type, and each type with a `ChannelParams` in every layer fills 8 rows. At 8B, with only `O`
/// and `Down` having one, `C = 2^12`, the tensor holds 2^21 entries and a row holds all 32 layers
/// of 8 types.
#[test]
fn the_production_shapes_take_the_widths_rfc_0012_states() {
    use WeightMatrix::*;
    let live_64 = WeightLayoutParams {
        log2_k: 20,
        log2_inv_rate: 2,
        mask_coeffs: 65_536,
        blinding_rows: 4096,
    };
    // (out, in) for Embed to LmHead: d_model 16,384, 128 heads and 8 KV heads of 128, d_ff
    // 53,248, a vocabulary of 128,256, 126 layers.
    let (d, kv, ff, vocab) = (16_384, 1024, 53_248, 128_256);
    let shapes = [
        (d, vocab),
        (d, d),
        (kv, d),
        (kv, d),
        (d, d),
        (ff, d),
        (ff, d),
        (d, ff),
        (vocab, d),
    ]
    .map(|(out, inputs)| WeightShape { out, inputs });
    let per_layer = [Q, K, V, O, Gate, Up, Down];
    let channels: Vec<(WeightMatrix, u32)> = per_layer
        .iter()
        .flat_map(|&m| (0..126).map(move |l| (m, l)))
        .collect();
    let layout = WeightLayout::new(
        live_64,
        WeightShapes {
            layers: 126,
            shapes,
            channels,
        },
    )
    .expect("a layout");
    for (matrix, a) in [
        (Embed, 3),
        (Q, 6),
        (K, 6),
        (V, 6),
        (O, 6),
        (Gate, 6),
        (Up, 6),
        (Down, 4),
        (LmHead, 6),
    ] {
        let found = layout.matrix(matrix);
        assert_eq!(found.log2_k(), 20, "{matrix:?}: k_T = 2^20");
        match found.kind() {
            MatrixKind::Weight { a: found_a, .. } => assert_eq!(found_a, a, "{matrix:?}: a_T"),
            other => panic!("{other:?}"),
        }
    }
    let channels = layout.matrix(ChannelParams);
    assert_eq!(
        channels.kind(),
        MatrixKind::ChannelParams {
            channel_width: 1 << 16
        }
    );
    assert_eq!(channels.log2_k(), 20);
    // Each of the seven types fills 8 rows: rows 8t to 8t + 7.
    let rows: Vec<u64> = layout.committed_rows(ChannelParams).collect();
    let expected: Vec<u64> = per_layer
        .iter()
        .flat_map(|m| (0..8).map(move |r| 8 * m.index() as u64 + r))
        .collect();
    assert_eq!(rows, expected);

    // 8B: 32 layers, d_model 4,096, 32 heads and 8 KV heads of 128, d_ff 14,336, a vocabulary
    // of 128,256; O and Down have a ChannelParams in every layer.
    let (d, kv, ff) = (4096, 1024, 14_336);
    let shapes = [
        (d, vocab),
        (d, d),
        (kv, d),
        (kv, d),
        (d, d),
        (ff, d),
        (ff, d),
        (d, ff),
        (vocab, d),
    ]
    .map(|(out, inputs)| WeightShape { out, inputs });
    let channels: Vec<(WeightMatrix, u32)> = [O, Down]
        .iter()
        .flat_map(|&m| (0..32).map(move |l| (m, l)))
        .collect();
    let layout = WeightLayout::new(
        live_64,
        WeightShapes {
            layers: 32,
            shapes,
            channels,
        },
    )
    .expect("a layout");
    let channels = layout.matrix(ChannelParams);
    assert_eq!(
        channels.kind(),
        MatrixKind::ChannelParams {
            channel_width: 1 << 12
        }
    );
    // 16·32·2^12 = 2^21 entries: two rows of 2^20, the first holding types 0 to 7.
    assert_eq!((channels.log2_k(), channels.row_vars()), (20, 1));
    assert_eq!(
        layout.committed_rows(ChannelParams).collect::<Vec<_>>(),
        [0]
    );
    assert_eq!(layout.check_masked(), Ok(()));
}

/// RFC-0012 'Layout' ('Masked rate and minimum width in `tessera-protocol/2`', #564): a masked
/// weight layout requires `P_W > 0`, `k_W + P_W < n_W`, and `k_T ≥ k_W` (`ρ'_T ≤ ρ'_W`) on every
/// committed matrix (`m_T > 0`), while `ChannelParams` with `m_CP == 0` is exempt.
#[test]
fn check_masked_enforces_nominal_rate_and_minimum_committed_width() {
    use WeightMatrix::*;

    let wide = WeightShapes {
        layers: 1,
        shapes: [WeightShape {
            out: 16,
            inputs: 16,
        }; 9],
        channels: vec![(Q, 0)],
    };
    // Exact boundary at test-small/1 (`k_W = 2^8 = 256`, `n_W = 1024`, `P_W = 64`):
    // every weight matrix has `16 · 16 = 256`, and `ChannelParams` has `16 · 1 · 16 = 256` with
    // `m_CP = 1`.
    let boundary = WeightLayout::new(params(8), wide.clone()).expect("a layout");
    for matrix in WeightMatrix::ALL {
        assert_eq!(boundary.matrix(matrix).log2_k(), 8, "{matrix:?}");
        assert!(boundary.matrix(matrix).rows() > 0, "{matrix:?}");
    }
    assert_eq!(boundary.check_masked(), Ok(()));

    // Wider than k_W (`inputs = 512 > 256`, so `log2 k_T = 9 > 8`) also passes.
    let mut wider = wide.clone();
    wider.shapes[Embed.index()].inputs = 512;
    let wider_layout = WeightLayout::new(params(8), wider).expect("a layout");
    assert_eq!(wider_layout.matrix(Embed).log2_k(), 9);
    assert_eq!(wider_layout.check_masked(), Ok(()));

    // P_W = 0 is refused (`NoMask`), and k_W + P_W ≥ n_W is refused (`MaskedWeightRate`), while
    // k_W + P_W = n_W − 1 passes.
    assert_eq!(
        WeightLayout::new(
            WeightLayoutParams {
                mask_coeffs: 0,
                ..params(8)
            },
            wide.clone()
        )
        .expect("a layout")
        .check_masked(),
        Err(PcsError::NoMask)
    );
    for (mask_coeffs, ok) in [(767, true), (768, false), (769, false)] {
        let layout = WeightLayout::new(
            WeightLayoutParams {
                mask_coeffs,
                ..params(8)
            },
            wide.clone(),
        )
        .expect("a layout");
        let expected = if ok {
            Ok(())
        } else {
            Err(PcsError::MaskedWeightRate {
                log2_k: 8,
                log2_inv_rate: 2,
                mask_coeffs,
            })
        };
        assert_eq!(layout.check_masked(), expected, "P_W = {mask_coeffs}");
    }

    // Nominal target code above 2^32 (`log2_k = 31`, `log2_inv_rate = 2`) is refused (`TooLong`)
    // even when every matrix of D fits below 2^32.
    let overlong_nominal = WeightLayout::new(
        WeightLayoutParams {
            log2_k: 31,
            log2_inv_rate: 2,
            mask_coeffs: 64,
            blinding_rows: 32,
        },
        wide.clone(),
    )
    .expect("every matrix of D has log2_k = 8");
    assert_eq!(
        overlong_nominal.check_masked(),
        Err(PcsError::TooLong { log_len: 33 })
    );

    // Every one of the nine weight matrices narrowed individually to 8 × 16 = 128 (`log2 k_T = 7`)
    // is refused by name, and two narrow matrices report the first in WeightMatrix order.
    for target in WeightMatrix::WEIGHTS {
        let mut shapes = wide.clone();
        // Keep C = 16 on ChannelParams even when Q is narrowed, by attaching ChannelParams to O
        // whenever target == Q.
        if target == Q {
            shapes.channels = vec![(O, 0)];
        }
        shapes.shapes[target.index()] = WeightShape { out: 8, inputs: 16 };
        let layout = WeightLayout::new(params(8), shapes).expect("a layout");
        assert_eq!(
            layout.check_masked(),
            Err(PcsError::NarrowWeight {
                matrix: target,
                log2_k: 7,
                min_log2_k: 8,
            }),
            "{target:?}"
        );
    }
    let mut two_narrow = wide.clone();
    two_narrow.shapes[Down.index()] = WeightShape { out: 8, inputs: 16 };
    two_narrow.shapes[K.index()] = WeightShape { out: 4, inputs: 16 };
    assert_eq!(
        WeightLayout::new(params(8), two_narrow)
            .expect("a layout")
            .check_masked(),
        Err(PcsError::NarrowWeight {
            matrix: K,
            log2_k: 6,
            min_log2_k: 8,
        })
    );

    // ChannelParams: with all nine weight matrices at 8 × 32 = 256 (`log2 k_T = 8`), `C = 8` gives
    // `16 · 1 · 8 = 128 = 2^7 < 256`: refused when `m_CP > 0`, admitted when `m_CP == 0`.
    let narrow_cp_shapes = [WeightShape { out: 8, inputs: 32 }; 9];
    let committed_cp = WeightLayout::new(
        params(8),
        WeightShapes {
            layers: 1,
            shapes: narrow_cp_shapes,
            channels: vec![(Q, 0)],
        },
    )
    .expect("a layout");
    assert_eq!(committed_cp.matrix(ChannelParams).rows(), 1);
    assert_eq!(
        committed_cp.check_masked(),
        Err(PcsError::NarrowWeight {
            matrix: ChannelParams,
            log2_k: 7,
            min_log2_k: 8,
        })
    );

    let uncommitted_cp = WeightLayout::new(
        params(8),
        WeightShapes {
            layers: 1,
            shapes: narrow_cp_shapes,
            channels: Vec::new(),
        },
    )
    .expect("a layout");
    assert_eq!(uncommitted_cp.matrix(ChannelParams).rows(), 0);
    assert_eq!(uncommitted_cp.matrix(ChannelParams).log2_k(), 4);
    assert_eq!(uncommitted_cp.check_masked(), Ok(()));
}
