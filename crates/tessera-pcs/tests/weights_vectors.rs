// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/weights/weights.json`, written by an independent oracle (`generate.py`, Python
//! big integers, from RFC-0040 and the RFCs it builds on), reproduced value for value:
//!
//! - rows encoded at rate 1/4, bit for bit;
//! - for each model, each matrix's layout from the shapes alone: `k_T`, `a_T`, `h_T`, `C`, the
//!   row index's bits and the committed rows;
//! - each matrix's descriptor and `commitment_id`;
//! - each root, from the weights themselves: a conformance model's are read from its artifact at
//!   the offsets the oracle gives, so that this crate needs no `IntegerModel` decoder;
//! - `TCE(WeightCommitmentPublic)`, `R_W` and the `.twr` artifact.
//!
//! RFC-0031's descriptors of `spec/vectors/hash/hash.json` are encoded by this crate's
//! [`CommitmentDescriptor`] too.

mod common;

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use common::{Fp, Replay};
use serde_json::Value;
use tessera_codec::{DecodeLimits, encode, envelope, schema};
use tessera_field::PrimeField;
use tessera_pcs::{
    AdviceLayoutParams, ChannelParamsRef, ChannelValues, CommitmentDescriptor, MatrixKind,
    ModelWeights, RowEncoder, WeightCommitment, WeightLayout, WeightLayoutParams, WeightMatrix,
    WeightShape, WeightShapes, WeightTensorRef,
};

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn load() -> Value {
    let path = vectors().join("weights/weights.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the vector file is JSON")
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digits"))
        .collect()
}

fn digest(value: &Value) -> [u8; 32] {
    hex(value.as_str().expect("a hex digest"))
        .try_into()
        .expect("32 bytes")
}

fn number(value: &Value) -> u64 {
    value.as_u64().expect("a number")
}

fn matrix_named(value: &Value) -> WeightMatrix {
    let name = value.as_str().expect("a matrix name");
    WeightMatrix::ALL
        .into_iter()
        .find(|m| format!("{m:?}") == name)
        .unwrap_or_else(|| panic!("no matrix {name}"))
}

/// The parameter set's layout the file states: `test-small/1`'s, RFC-0026.
fn file_layout(file: &Value) -> WeightLayoutParams {
    let layout = &file["layout"];
    let field = |name: &str| number(&layout[name]);
    let params = WeightLayoutParams {
        log2_k: field("log2_k") as u8,
        log2_inv_rate: field("log2_inv_rate") as u8,
        mask_coeffs: field("mask_coeffs") as u32,
        blinding_rows: field("blinding_rows") as u32,
    };
    assert_eq!(
        params,
        WeightLayoutParams {
            log2_k: 8,
            log2_inv_rate: 2,
            mask_coeffs: 64,
            blinding_rows: 32
        },
        "test-small/1's weight layout (RFC-0026)"
    );
    params
}

/// A model's public shapes, as the oracle derived them from its architecture.
fn shapes(model: &Value) -> WeightShapes {
    let matrices = model["matrices"].as_array().expect("matrices");
    assert_eq!(matrices.len(), 10, "every matrix");
    let shapes = core::array::from_fn(|t| WeightShape {
        out: number(&matrices[t]["out"]) as u32,
        inputs: number(&matrices[t]["in"]) as u32,
    });
    let channels = model["channels"]
        .as_array()
        .expect("channels")
        .iter()
        .map(|c| (matrix_named(&c["matrix"]), number(&c["layer"]) as u32))
        .collect();
    WeightShapes {
        layers: number(&model["arch"]["n_layers"]) as u32,
        shapes,
        channels,
    }
}

/// A `ChannelParams`' values, owned: class indices or multipliers.
enum Values {
    Classes(Vec<u8>),
    Multipliers(Vec<u16>),
}

/// A model's weights and channel parameters: from the JSON for the synthetic model, from the
/// artifact at the stated offsets for a conformance model, the artifact's envelope checked first.
struct Entries {
    tensors: Vec<(WeightMatrix, u32, Vec<i8>)>,
    channels: Vec<(WeightMatrix, u32, Values)>,
}

impl Entries {
    fn of(model: &Value) -> Self {
        let artifact = model.get("model").map(|path| {
            let path = vectors()
                .join("weights")
                .join(path.as_str().expect("a path"));
            let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            envelope::open_payload(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("a sound IntegerModel envelope");
            assert_eq!(
                &bytes[bytes.len() - 32..],
                digest(&model["model_digest"]),
                "the model's digest"
            );
            bytes
        });
        let tensors = model["tensors"]
            .as_array()
            .expect("tensors")
            .iter()
            .map(|t| {
                let len = (number(&t["rows"]) * number(&t["cols"])) as usize;
                let data: Vec<i8> = match &artifact {
                    Some(bytes) => {
                        let at = number(&t["offset"]) as usize;
                        // The vector's length prefix sits just before its first entry.
                        let prefix = u64::from_le_bytes(bytes[at - 8..at].try_into().unwrap());
                        assert_eq!(prefix as usize, len, "the length before {t}");
                        bytes[at..at + len].iter().map(|&b| b as i8).collect()
                    }
                    None => t["data"]
                        .as_array()
                        .expect("data")
                        .iter()
                        .map(|x| x.as_i64().expect("an entry") as i8)
                        .collect(),
                };
                assert_eq!(data.len(), len);
                (matrix_named(&t["matrix"]), number(&t["layer"]) as u32, data)
            })
            .collect();
        let channels = model["channels"]
            .as_array()
            .expect("channels")
            .iter()
            .map(|c| {
                let bytes = artifact
                    .as_ref()
                    .expect("channel parameters live in an artifact");
                let (at, count) = (number(&c["offset"]) as usize, number(&c["count"]) as usize);
                let prefix = u64::from_le_bytes(bytes[at - 8..at].try_into().unwrap());
                assert_eq!(prefix as usize, count, "the count before {c}");
                let values = match c["kind"].as_str() {
                    Some("class") => Values::Classes(bytes[at..at + count].to_vec()),
                    Some("multiplier") => Values::Multipliers(
                        bytes[at..at + 2 * count]
                            .chunks_exact(2)
                            .map(|w| u16::from_le_bytes([w[0], w[1]]))
                            .collect(),
                    ),
                    other => panic!("kind {other:?}"),
                };
                (
                    matrix_named(&c["matrix"]),
                    number(&c["layer"]) as u32,
                    values,
                )
            })
            .collect();
        Self { tensors, channels }
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

/// RFC-0040 'The matrices': each matrix's layout follows from the shapes alone.
#[test]
fn every_matrix_layout_matches_the_oracle() {
    let file = load();
    let params = file_layout(&file);
    let models = file["models"].as_array().expect("models");
    assert_eq!(
        models.len(),
        5,
        "four conformance models and the synthetic one"
    );
    for model in models {
        let name = model["name"].as_str().expect("a name");
        let layout = WeightLayout::new(params, shapes(model)).expect("a valid layout");
        for (expected, matrix) in model["matrices"]
            .as_array()
            .expect("matrices")
            .iter()
            .zip(WeightMatrix::ALL)
        {
            assert_eq!(matrix_named(&expected["matrix"]), matrix);
            let found = layout.matrix(matrix);
            let at = format!("{name} {matrix:?}");
            assert_eq!(
                u64::from(found.log2_k()),
                number(&expected["log2_k"]),
                "{at}: log2 k_T"
            );
            assert_eq!(
                u64::from(found.row_vars()),
                number(&expected["row_vars"]),
                "{at}"
            );
            assert_eq!(found.rows(), number(&expected["rows"]), "{at}: m_T");
            let committed: Vec<u64> = layout.committed_rows(matrix).collect();
            let listed: Vec<u64> = expected["committed_rows"]
                .as_array()
                .expect("committed rows")
                .iter()
                .map(number)
                .collect();
            assert_eq!(committed, listed, "{at}: the committed rows");
            match found.kind() {
                MatrixKind::Weight { layers, a, h, .. } => {
                    assert_eq!(u64::from(layers), number(&expected["layers"]), "{at}");
                    assert_eq!(u64::from(a), number(&expected["a"]), "{at}: a_T");
                    assert_eq!(u64::from(h), number(&expected["h"]), "{at}: h_T");
                }
                MatrixKind::ChannelParams { channel_width } => {
                    assert_eq!(channel_width, number(&expected["channel_width"]), "{at}: C");
                }
            }
        }
    }
}

/// RFC-0031: `commitment_id_T` hashes `TCE(CommitmentDescriptor::WeightMatrix { … })`, with the
/// claim list's layout, the matrix, `log2 k_T`, `B`, `m_T` and the epoch.
#[test]
fn every_descriptor_and_commitment_id_matches_the_oracle() {
    let file = load();
    let params = file_layout(&file);
    let (row_block, epoch) = (
        number(&file["row_block"]) as u32,
        number(&file["epoch"]) as u32,
    );
    assert_eq!(
        (row_block, epoch),
        (4, 0),
        "test-small/1's B and the first epoch"
    );
    for model in file["models"].as_array().expect("models") {
        for expected in model["matrices"].as_array().expect("matrices") {
            let descriptor = CommitmentDescriptor::WeightMatrix {
                layout: params,
                matrix: matrix_named(&expected["matrix"]),
                log2_k: number(&expected["log2_k"]) as u8,
                row_block,
                rows: number(&expected["rows"]),
                epoch,
            };
            let at = format!("{} {}", model["name"], expected["matrix"]);
            assert_eq!(
                encode(&descriptor),
                hex(expected["descriptor"].as_str().expect("hex")),
                "{at}"
            );
            assert_eq!(
                descriptor.commitment_id(),
                digest(&expected["commitment_id"]),
                "{at}"
            );
        }
    }
}

/// RFC-0040 'Encoding, leaves and roots', 'WeightCommitmentPublic': the roots, from the weights,
/// then `TCE(WeightCommitmentPublic)`, `R_W` and the `.twr` artifact.
#[test]
fn every_root_and_weights_root_matches_the_oracle() {
    let file = load();
    let params = file_layout(&file);
    let row_block = NonZeroU32::new(number(&file["row_block"]) as u32).expect("B ≥ 1");
    let epoch = number(&file["epoch"]) as u32;
    for model in file["models"].as_array().expect("models") {
        let name = model["name"].as_str().expect("a name");
        let layout = WeightLayout::new(params, shapes(model)).expect("a valid layout");
        let entries = Entries::of(model);
        let commitment = WeightCommitment::commit(&layout, row_block, epoch, &entries.weights())
            .expect("a valid commitment");
        for (expected, matrix) in model["matrices"]
            .as_array()
            .expect("matrices")
            .iter()
            .zip(WeightMatrix::ALL)
        {
            assert_eq!(
                commitment.matrix(matrix).root(),
                digest(&expected["root"]),
                "{name} {matrix:?}: root"
            );
            assert_eq!(
                commitment.matrix(matrix).codeword_len(),
                4 << number(&expected["log2_k"]),
                "{name} {matrix:?}: n_T = 4·k_T"
            );
        }
        let public = commitment.public();
        assert_eq!(
            encode(public),
            hex(model["public"].as_str().expect("hex")),
            "{name}"
        );
        assert_eq!(
            commitment.weights_root(),
            digest(&model["weights_root"]),
            "{name}: R_W"
        );
        let artifact = fs::read(
            vectors()
                .join("weights")
                .join(model["file"].as_str().expect("a file")),
        )
        .expect("the .twr artifact");
        assert_eq!(
            envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, public),
            artifact,
            "{name}: the .twr artifact"
        );
        assert_eq!(artifact.len() as u64, number(&model["bytes"]));
        assert_eq!(&artifact[artifact.len() - 32..], digest(&model["digest"]));
    }
}

/// RFC-0031 'Testing Strategy': the descriptors of `spec/vectors/hash/hash.json`, the advice
/// ones and the weight matrices', encode as the oracle encodes them.
#[test]
fn the_hash_vectors_descriptors_encode_as_the_oracle_writes_them() {
    let path = vectors().join("hash/hash.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("hash.json")).expect("JSON");
    let cases = file["commitment_ids"].as_array().expect("commitment ids");
    assert_eq!(cases.len(), 4);
    for case in cases {
        let d = &case["descriptor"];
        let layout = &d["layout"];
        let field = |name: &str| number(&layout[name]);
        let descriptor = match d["kind"].as_str() {
            Some("Advice") => CommitmentDescriptor::Advice {
                statement_digest: digest(&d["statement_digest"]),
                unit: number(&d["unit"]) as u32,
                layout: AdviceLayoutParams {
                    log2_k: field("log2_k") as u8,
                    log2_inv_rate: field("log2_inv_rate") as u8,
                    mask_coeffs: field("mask_coeffs") as u32,
                    row_block: field("row_block") as u32,
                    segments: field("segments") as u32,
                },
                rows: number(&d["rows"]),
            },
            Some("WeightMatrix") => CommitmentDescriptor::WeightMatrix {
                layout: WeightLayoutParams {
                    log2_k: field("log2_k") as u8,
                    log2_inv_rate: field("log2_inv_rate") as u8,
                    mask_coeffs: field("mask_coeffs") as u32,
                    blinding_rows: field("blinding_rows") as u32,
                },
                matrix: matrix_named(&d["matrix"]),
                log2_k: number(&d["log2_k"]) as u8,
                row_block: number(&d["row_block"]) as u32,
                rows: number(&d["rows"]),
                epoch: number(&d["epoch"]) as u32,
            },
            other => panic!("kind {other:?}"),
        };
        assert_eq!(
            encode(&descriptor),
            hex(case["tce"].as_str().expect("hex")),
            "{d}"
        );
        assert_eq!(
            descriptor.commitment_id(),
            digest(&case["commitment_id"]),
            "{d}"
        );
    }
}

fn fps(value: &Value) -> Vec<Fp> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|x| {
            let text = x.as_str().expect("values are hex strings");
            let value = u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("valid hex");
            Fp::from_canonical_u64(value).expect("vector values are canonical")
        })
        .collect()
}

/// RFC-0040 'Encoding, leaves and roots': a weight row's codeword has `n = 4k` symbols on
/// `7·L`, with no mask in `tessera-protocol/1`; the masked cases pin where a mask goes at this
/// rate.
#[test]
fn rate_one_quarter_encodings_match_the_oracle() {
    let file = load();
    let cases = file["encoding"].as_array().expect("the encoding cases");
    let mut shapes = Vec::new();
    for case in cases {
        let k = case["k"].as_u64().expect("k") as usize;
        let p = case["mask_coeffs"].as_u64().expect("P") as usize;
        let (row, mask, expected) = (
            fps(&case["row"]),
            fps(&case["mask"]),
            fps(&case["codeword"]),
        );
        assert_eq!((row.len(), mask.len(), expected.len()), (k, p, 4 * k));
        let encoder = RowEncoder::new(k, 2, p).expect("valid parameters");
        assert_eq!(encoder.codeword_len(), 4 * k);
        let mut source = Replay::new(mask);
        let mut codeword = vec![Fp::default(); 4 * k];
        encoder
            .encode_row(&row, &mut source, &mut codeword)
            .expect("matching lengths");
        assert_eq!(source.drawn, p, "P draws, k = {k}");
        assert_eq!(codeword, expected, "codeword, k = {k}, P = {p}");
        // A block of the same row, twice, encodes to the same codeword twice.
        let mut source = Replay::new(fps(&case["mask"]).repeat(2));
        let mut block = vec![Fp::default(); 8 * k];
        encoder
            .encode_rows(&row.repeat(2), &mut source, &mut block)
            .expect("matching lengths");
        assert_eq!(block, expected.repeat(2), "a block, k = {k}, P = {p}");
        shapes.push((k, p));
    }
    // The README promises P = 0 for k = 2^0 to 2^6, then P of 1, k/2 and k for k of 4 and 16.
    let mut promised: Vec<(usize, usize)> = (0..=6).map(|e| (1 << e, 0)).collect();
    promised.extend([(4, 1), (4, 2), (4, 4), (16, 1), (16, 8), (16, 16)]);
    assert_eq!(shapes, promised);
}
