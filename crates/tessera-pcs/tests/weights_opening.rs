// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment opened by the folding pass (RFC-0012 'Opening by the folding pass',
//! RFC-0049), as `tessera-protocol/1` opens it: no mask and no blinding row. A matrix's claim
//! `Ã_T(point) = v` splits its point at `log2 k_T` into a column point and a row point. One read
//! of the matrix's committed rows gives the per-layer folds `F_ℓ` and the proximity row
//! `u_ξ,T = Σ_{j < m_T} eq(ξ*_T, j)·W_T[row_j]`, and the folds give the sumcheck input
//! `W̃(ℓ, r_o, i)` and the evaluation row `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ`.
//!
//! Everything is checked against definitions written here from the RFCs, never against the
//! crate's algorithms: each matrix's committed rows and entries from RFC-0040 'The matrices', the
//! contractions `W̃_ℓ(r_o, i) = Σ_o eq(r_o, o)·W_ℓ[o][i]` straight from the INT8 weights, and an
//! in-test verifier built from the verifier workspace's field, hashing and codec (`tv_field`,
//! `tv_hash`, `tv_codec`) and `tv_core::sumcheck::eq`, which share no code with the prover
//! (DEP-2). `tv_core::opening` checks the advice's openings, whose leaves merge segments and
//! whose rows carry masks; the verifier's own weight opening is #109's.
//!
//! - **Fold equivalence.** `Σ_{o_lo} eq(r_o^lo, o_lo)·F_ℓ(o_lo, i)` equals `W̃_ℓ(r_o, i)` for every
//!   layer, padded layers included, on the conformance models' shapes and on random shapes with
//!   inputs wider than `k_W`; both combined rows equal their definitions.
//! - **One pass** (INV-WC-4). Instrumented readers count every row and every weight a proof
//!   reads: each once, for the folds, `u_T` and `u_ξ,T` of all ten matrices.
//! - **End to end.** A claim on each matrix at a random point opens and verifies against `R_W`;
//!   the in-test verifier refuses a changed column symbol or sibling (`MerklePath`) and another
//!   epoch's public artifact (`ERR-USAGE`), so what it accepts is bound to the trusted root.
//! - **Binding.** Weights changed after the commitment, one entry of every matrix, a padding
//!   position and a `ChannelParams` entry, give combined rows the columns refuse: `Proximity` at
//!   the smallest queried column, while the honest opening of the same claim verifies.
//! - **Checks.** Each input the fold and the readers refuse, with its error.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use common::SplitMix64;
use common::openings::{TvFp, TvFp2, tv_fp2, tv_fp2s};
use serde_json::Value;
use tessera_codec::encode;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::Digest;
use tessera_pcs::{
    ChannelParamsRef, ChannelValues, MatrixRows, ModelWeights, PcsError, WeightCombinedRows,
    WeightCommitment, WeightFold, WeightLayout, WeightLayoutParams, WeightMatrix, WeightShape,
    WeightShapes, WeightTensorRef,
};
use tv_hash::merkle::verify_multiproof;
use tv_hash::{Domain, hash};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// `test-small/1`'s row block `B` and weight queries `τ_W` (RFC-0014 'Parameter sets').
const ROW_BLOCK: u32 = 4;
const QUERIES: usize = 8;

/// `test-small/1`'s weight layout (RFC-0026), with `log2 k_W` set by the case.
fn params(log2_k: u8) -> WeightLayoutParams {
    WeightLayoutParams {
        log2_k,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    }
}

// ----- Models -----------------------------------------------------------------------------------

/// A `ChannelParams`' values, owned: class indices or multipliers.
#[derive(Clone, Debug)]
enum Values {
    Classes(Vec<u8>),
    Multipliers(Vec<u16>),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Self::Classes(v) => v.len(),
            Self::Multipliers(v) => v.len(),
        }
    }

    fn get(&self, c: usize) -> u64 {
        match self {
            Self::Classes(v) => u64::from(v[c]),
            Self::Multipliers(v) => u64::from(v[c]),
        }
    }

    /// Changes value `c` to another value of its kind.
    fn change(&mut self, c: usize) {
        match self {
            Self::Classes(v) => v[c] ^= 1,
            Self::Multipliers(v) => v[c] = if v[c] == 1 { 2 } else { v[c] - 1 },
        }
    }
}

/// A model `D`: its public shapes, its weights and its channel parameters, owned.
#[derive(Clone, Debug)]
struct Model {
    name: String,
    log2_k: u8,
    shapes: WeightShapes,
    tensors: Vec<(WeightMatrix, u32, Vec<i8>)>,
    channels: Vec<(WeightMatrix, u32, Values)>,
}

impl Model {
    /// Random INT8 weights, every value possible, for `shapes[t] = (out, in)`, and random channel
    /// parameters for `channels`, each with classes (`true`) or multipliers.
    fn random(
        name: &str,
        rng: &mut SplitMix64,
        log2_k: u8,
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
            for layer in 0..if matrix.per_layer() { layers } else { 1 } {
                let len = (shape.out * shape.inputs) as usize;
                tensors.push((
                    matrix,
                    layer,
                    (0..len).map(|_| rng.next_u64() as i8).collect(),
                ));
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
            name: name.to_string(),
            log2_k,
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

    fn layout(&self) -> WeightLayout {
        WeightLayout::new(params(self.log2_k), self.shapes.clone()).expect("a valid layout")
    }

    fn tensor(&self, matrix: WeightMatrix, layer: u32) -> &[i8] {
        &self
            .tensors
            .iter()
            .find(|(m, l, _)| (*m, *l) == (matrix, layer))
            .expect("every weight")
            .2
    }

    fn tensor_mut(&mut self, matrix: WeightMatrix, layer: u32) -> &mut Vec<i8> {
        &mut self
            .tensors
            .iter_mut()
            .find(|(m, l, _)| (*m, *l) == (matrix, layer))
            .expect("every weight")
            .2
    }
}

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
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

/// The four conformance models and the synthetic one of `spec/vectors/weights/weights.json`, at
/// `test-small/1`: their shapes as the oracle derived them, and their weights and channel
/// parameters, read from each model's artifact at the offsets the file gives (the synthetic
/// model's are listed).
fn conformance_models() -> Vec<Model> {
    let path = vectors().join("weights/weights.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let file: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(number(&file["layout"]["log2_k"]), 8, "test-small/1's k_W");
    file["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|model| {
            let artifact = model.get("model").map(|path| {
                let path = vectors()
                    .join("weights")
                    .join(path.as_str().expect("a path"));
                fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            });
            let matrices = model["matrices"].as_array().expect("matrices");
            let shapes = WeightShapes {
                layers: number(&model["arch"]["n_layers"]) as u32,
                shapes: core::array::from_fn(|t| WeightShape {
                    out: number(&matrices[t]["out"]) as u32,
                    inputs: number(&matrices[t]["in"]) as u32,
                }),
                channels: model["channels"]
                    .as_array()
                    .expect("channels")
                    .iter()
                    .map(|c| (matrix_named(&c["matrix"]), number(&c["layer"]) as u32))
                    .collect(),
            };
            let tensors = model["tensors"]
                .as_array()
                .expect("tensors")
                .iter()
                .map(|t| {
                    let len = (number(&t["rows"]) * number(&t["cols"])) as usize;
                    let data: Vec<i8> = match &artifact {
                        Some(bytes) => {
                            let at = number(&t["offset"]) as usize;
                            bytes[at..at + len].iter().map(|&b| b as i8).collect()
                        }
                        None => t["data"]
                            .as_array()
                            .expect("data")
                            .iter()
                            .map(|x| x.as_i64().expect("an entry") as i8)
                            .collect(),
                    };
                    (matrix_named(&t["matrix"]), number(&t["layer"]) as u32, data)
                })
                .collect();
            let channels = model["channels"]
                .as_array()
                .expect("channels")
                .iter()
                .map(|c| {
                    let bytes = artifact.as_ref().expect("an artifact");
                    let (at, count) = (number(&c["offset"]) as usize, number(&c["count"]) as usize);
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
            Model {
                name: model["name"].as_str().expect("a name").to_string(),
                log2_k: 8,
                shapes,
                tensors,
                channels,
            }
        })
        .collect()
}

/// Llama-like shapes, `(out, in)` for `Embed`, `Q`, `K`, `V`, `O`, `Gate`, `Up`, `Down` and
/// `LmHead`.
fn llama(d: u32, q: u32, kv: u32, ff: u32, vocab: u32) -> [(u32, u32); 9] {
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
}

/// Random models whose shapes RFC-0040 lays out in every way: inputs wider than `k_W` (`a_T = 0`),
/// a whole layer in one row (`h_T = 0`), widths and layer counts that are not powers of two, rows
/// that hold padded outputs, and channel parameters of every weight type, classes and
/// multipliers, or none.
fn random_models(rng: &mut SplitMix64) -> Vec<Model> {
    use WeightMatrix::*;
    let every: Vec<(WeightMatrix, u32, bool)> = WeightMatrix::WEIGHTS
        .into_iter()
        .flat_map(|m| {
            let layers = if m.per_layer() { 3 } else { 1 };
            (0..layers).map(move |l| (m, l, l % 2 == 0))
        })
        .collect();
    let mut models = vec![
        Model::random("llama-3", rng, 4, 3, llama(12, 12, 6, 20, 300), &[]),
        Model::random(
            "llama-5",
            rng,
            6,
            5,
            llama(10, 10, 5, 13, 37),
            &[
                (Q, 0, true),
                (K, 4, false),
                (Gate, 2, true),
                (Down, 3, false),
            ],
        ),
        Model::random(
            "llama-1",
            rng,
            3,
            1,
            llama(7, 7, 3, 9, 11),
            &[
                (Embed, 0, false),
                (V, 0, true),
                (O, 0, true),
                (LmHead, 0, true),
            ],
        ),
        Model::random("every-channel", rng, 5, 3, llama(4, 4, 2, 6, 5), &every),
    ];
    for i in 0..12 {
        let dim = |rng: &mut SplitMix64| 1 + rng.below(40) as u32;
        let shapes = core::array::from_fn(|_| (dim(rng), dim(rng)));
        let layers = 1 + rng.below(5) as u32;
        let mut channels = Vec::new();
        for matrix in WeightMatrix::WEIGHTS {
            for layer in 0..if matrix.per_layer() { layers } else { 1 } {
                if rng.below(3) == 0 {
                    channels.push((matrix, layer, rng.below(2) == 0));
                }
            }
        }
        let log2_k = 2 + rng.below(7) as u8;
        models.push(Model::random(
            &format!("random-{i}"),
            rng,
            log2_k,
            layers,
            shapes,
            &channels,
        ));
    }
    models
}

// ----- RFC-0040 'The matrices', from the RFC ----------------------------------------------------

/// The next power of two of `n ≥ 1`.
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

/// `⌈log2 m⌉`: the bits of a position below `m ≥ 1`.
fn ceil_log2(m: usize) -> usize {
    log2(pad(m as u64)) as usize
}

/// A weight `w` as its true integer in the centered representation (RFC-0002).
fn centered(w: i64) -> TvFp {
    if w < 0 {
        -TvFp::reduce(w.unsigned_abs())
    } else {
        TvFp::reduce(w as u64)
    }
}

/// One matrix as RFC-0040 'The matrices' defines it. A weight type's row `r` is `(ℓ, o_hi)`,
/// `ℓ = ⌊r / 2^h⌋`, and its column `(o_lo, i)`; `ChannelParams` has one layer, `a = 0` and
/// `h` its row index's bits.
struct Defined {
    matrix: WeightMatrix,
    log2_k: u32,
    /// `a_T`: the output bits a column holds; 0 for `ChannelParams`.
    a: u32,
    /// `h_T`: the row bits below the layer index.
    h: u32,
    /// The layers with weights, `n_layers` or 1.
    layers: u64,
    /// `log2 L_pad`.
    layer_vars: u32,
    /// The committed rows, increasing.
    rows: Vec<u64>,
    /// Their values, row after row.
    values: Vec<TvFp>,
    /// Whether each position of a committed row holds a weight or a channel parameter.
    live: Vec<bool>,
}

impl Defined {
    fn k(&self) -> usize {
        1 << self.log2_k
    }

    fn m(&self) -> usize {
        self.rows.len()
    }

    fn row(&self, j: usize) -> &[TvFp] {
        &self.values[j * self.k()..][..self.k()]
    }

    /// The point's coordinates on the matrix's extension: the column point's then the row
    /// point's.
    fn vars(&self) -> usize {
        (self.log2_k + self.h + self.layer_vars) as usize
    }
}

/// Every matrix of `model`, from RFC-0040 'The matrices'.
fn define(model: &Model) -> Vec<Defined> {
    let k_w = 1u64 << model.log2_k;
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
        let mut rows = Vec::new();
        let mut values = Vec::new();
        let mut live = Vec::new();
        // A row is committed unless ℓ ≥ n_layers or ⌊o / 2^a⌋·2^a ≥ out, in increasing r.
        for layer in 0..layers {
            let data = model.tensor(matrix, layer as u32);
            for o_hi in 0..(1u64 << h) {
                if o_hi << a >= o {
                    continue;
                }
                rows.push((layer << h) + o_hi);
                let mut row = vec![TvFp::ZERO; k as usize];
                let mut held = vec![false; k as usize];
                // W_ℓ[o][i] at column (o mod 2^a)·in_pad + i.
                for o_lo in 0..(1u64 << a) {
                    let oo = (o_hi << a) + o_lo;
                    if oo >= o {
                        break;
                    }
                    for ii in 0..i {
                        let column = (o_lo * in_pad + ii) as usize;
                        row[column] = centered(i64::from(data[(oo * i + ii) as usize]));
                        held[column] = true;
                    }
                }
                values.extend(row);
                live.extend(held);
            }
        }
        out.push(Defined {
            matrix,
            log2_k: log2(k),
            a,
            h,
            layers,
            layer_vars: log2(layers_pad),
            rows,
            values,
            live,
        });
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
    let mut rows: Vec<(u64, Vec<TvFp>, Vec<bool>)> = Vec::new();
    for (matrix, layer, values) in &model.channels {
        let base = (matrix.index() as u64 * l_pad + u64::from(*layer)) * c;
        for channel in 0..values.len() {
            let f = base + channel as u64;
            let (r, column) = (f / k, (f % k) as usize);
            let at = match rows.iter().position(|(row, _, _)| *row == r) {
                Some(at) => at,
                None => {
                    rows.push((r, vec![TvFp::ZERO; k as usize], vec![false; k as usize]));
                    rows.len() - 1
                }
            };
            rows[at].1[column] = TvFp::reduce(values.get(channel));
            rows[at].2[column] = true;
        }
    }
    rows.sort_by_key(|(r, _, _)| *r);
    let row_vars = log2(total / k);
    out.push(Defined {
        matrix: WeightMatrix::ChannelParams,
        log2_k: log2(k),
        a: 0,
        h: row_vars,
        layers: 1,
        layer_vars: 0,
        rows: rows.iter().map(|(r, _, _)| *r).collect(),
        values: rows
            .iter()
            .flat_map(|(_, v, _)| v.iter().copied())
            .collect(),
        live: rows.into_iter().flat_map(|(_, _, l)| l).collect(),
    });
    out
}

// ----- Definitions of the opening, from RFC-0010 and RFC-0012 ------------------------------------

/// `eq(a, b)`, the verifier's (RFC-0010 'Multilinear extensions').
fn eq(a: &[TvFp2], b: &[TvFp2]) -> TvFp2 {
    tv_core::sumcheck::eq(a, b).expect("points of one length")
}

/// The vertex `x` of the hypercube of `vars` variables, bit `t` of `x` its coordinate `t`.
fn vertex(x: u64, vars: usize) -> Vec<TvFp2> {
    (0..vars)
        .map(|t| TvFp2::from(TvFp::reduce((x >> t) & 1)))
        .collect()
}

fn eq_at(point: &[TvFp2], x: u64) -> TvFp2 {
    eq(point, &vertex(x, point.len()))
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)` of a table of `2^{point.len()}` values.
fn extension(table: &[TvFp2], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(table.len(), 1 << point.len());
    table
        .iter()
        .enumerate()
        .fold(TvFp2::ZERO, |sum, (x, &v)| sum + v * eq_at(point, x as u64))
}

/// `W̃_ℓ(r_o, i) = Σ_{o < out} eq(r_o, o)·W_ℓ[o][i]` for `i < in_pad`, straight from the INT8
/// weights: 0 at a padded input.
fn contracted(model: &Model, matrix: WeightMatrix, layer: u32, r_o: &[TvFp2]) -> Vec<TvFp2> {
    let shape = model.shapes.shapes[matrix.index()];
    let (out, inputs) = (shape.out as usize, shape.inputs as usize);
    let data = model.tensor(matrix, layer);
    let mut table = vec![TvFp2::ZERO; pad(inputs as u64) as usize];
    for o in 0..out {
        let e = eq_at(r_o, o as u64);
        for (i, slot) in table.iter_mut().take(inputs).enumerate() {
            *slot += e * centered(i64::from(data[o * inputs + i]));
        }
    }
    table
}

/// The table a fold's point leaves, `Ã_T(i ‖ point ‖ ℓ)` over the free columns `i` (layer high):
/// for a weight type `W̃(ℓ, r_o, i)`, for `ChannelParams` its columns at the row point. From the
/// committed rows: row `j`, `(ℓ_j, f_j) = (⌊r_j / 2^h⌋, r_j mod 2^h)`, adds
/// `eq(point_row, f_j)·Σ_{o_lo} eq(point_col, o_lo)·W[row_j][o_lo·(k/2^a) + i]` to layer `ℓ_j`.
fn bound_table(d: &Defined, point: &[TvFp2]) -> Vec<TvFp2> {
    let (column, row) = point.split_at(d.a as usize);
    let free = d.k() >> d.a;
    let mut table = vec![TvFp2::ZERO; free << d.layer_vars];
    for (j, &r) in d.rows.iter().enumerate() {
        let (layer, f) = (r >> d.h, r & ((1 << d.h) - 1));
        let weight = eq_at(row, f);
        for o_lo in 0..(1usize << d.a) {
            let e = weight * eq_at(column, o_lo as u64);
            for i in 0..free {
                table[layer as usize * free + i] += e * d.row(j)[o_lo * free + i];
            }
        }
    }
    table
}

/// The two combined rows of RFC-0012 and RFC-0049, from the committed rows: `u_T[c] = θ·Σ_j
/// eq(ρ, ℓ_j)·eq(point_row, f_j)·W[row_j][c]` and `u_ξ[c] = Σ_j eq(ξ*, j)·W[row_j][c]`.
fn defined_rows(
    d: &Defined,
    point: &[TvFp2],
    layer_point: &[TvFp2],
    theta: TvFp2,
    xi: &[TvFp2],
) -> (Vec<TvFp2>, Vec<TvFp2>) {
    let row = &point[d.a as usize..];
    let mut u = vec![TvFp2::ZERO; d.k()];
    let mut u_xi = vec![TvFp2::ZERO; d.k()];
    for (j, &r) in d.rows.iter().enumerate() {
        let c_eval = theta * eq_at(layer_point, r >> d.h) * eq_at(row, r & ((1 << d.h) - 1));
        let c_xi = eq_at(xi, j as u64);
        for (c, &x) in d.row(j).iter().enumerate() {
            u[c] += c_eval * x;
            u_xi[c] += c_xi * x;
        }
    }
    (u, u_xi)
}

// ----- The in-test verifier ----------------------------------------------------------------------

/// Why the in-test verifier refuses an opening: RFC-0011's reasons, in its check order
/// (steps 6 and 8, 'Failure modes'), after the verifier's own inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refused {
    /// The public artifact does not hash to the trusted `R_W`, or holds another layout than the
    /// claim list's: the verifier's own inputs disagree, `ERR-USAGE` (RFC-0040 'The verifier's
    /// weight inputs'), before anything of the proof is read.
    Usage,
    /// The combined rows' shape, or a claim the evaluation row does not give: `Evaluation`.
    Evaluation,
    /// Columns of the wrong count or length, or leaves and siblings that do not lead to the root:
    /// `MerklePath` at the smallest queried column.
    MerklePath { column: u64 },
    /// A combined row that disagrees with a queried column: `Proximity` at the smallest.
    Proximity { column: u64 },
}

/// What the verifier knows before an opening: the root it trusts, the epoch's public artifact,
/// the claim list's layout and row block, and the matrix's layout, from the architecture.
struct Verifier<'a> {
    trusted_root: Digest,
    artifact: &'a [u8],
    params: WeightLayoutParams,
    row_block: u32,
    defined: &'a Defined,
}

/// A claim `Ã_T(point) = value` and its opening, as the verifier receives them.
struct Opening<'a> {
    point: &'a [TvFp2],
    value: TvFp2,
    theta: TvFp2,
    xi: &'a [TvFp2],
    rows: &'a WeightCombinedRows,
    queries: &'a [u64],
    columns: &'a [Vec<Fp>],
    siblings: &'a [Digest],
}

impl Verifier<'_> {
    /// `commitment_id_T = H("commitment-id", TCE(CommitmentDescriptor::WeightMatrix { … }))`,
    /// encoded from RFC-0031 and RFC-0003: the variant's `u16` tag 1, the layout's fields, the
    /// matrix's `u16` tag, `log2 k_T`, `B`, `m_T` and the epoch, little-endian.
    fn commitment_id(&self, epoch: u32) -> Digest {
        let d = self.defined;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.push(self.params.log2_k);
        bytes.push(self.params.log2_inv_rate);
        bytes.extend_from_slice(&self.params.mask_coeffs.to_le_bytes());
        bytes.extend_from_slice(&self.params.blinding_rows.to_le_bytes());
        bytes.extend_from_slice(&(d.matrix.index() as u16).to_le_bytes());
        bytes.push(d.log2_k as u8);
        bytes.extend_from_slice(&self.row_block.to_le_bytes());
        bytes.extend_from_slice(&(d.m() as u64).to_le_bytes());
        bytes.extend_from_slice(&epoch.to_le_bytes());
        hash(Domain::CommitmentId, &bytes)
    }

    /// Checks `opening` in RFC-0011's order, with the weights' rules of RFC-0012, RFC-0040 and
    /// RFC-0049: no mask in `tessera-protocol/1`, one chain per column, positions for `ξ*_T`.
    fn verify(&self, opening: &Opening<'_>) -> Result<(), Refused> {
        let d = self.defined;
        // RFC-0040 'The verifier's weight inputs': the artifact hashes to the trusted root, and
        // its layout is the claim list's. A failure is the verifier's own inputs disagreeing.
        let public = tv_codec::decode(
            self.artifact,
            &tv_codec::DecodeLimits::MESSAGE,
            tv_codec::WeightCommitmentPublic::decode,
        )
        .expect("the public artifact decodes");
        let layout = (
            public.layout.log2_k,
            public.layout.log2_inv_rate,
            public.layout.mask_coeffs,
            public.layout.blinding_rows,
        );
        let claimed = (
            self.params.log2_k,
            self.params.log2_inv_rate,
            self.params.mask_coeffs,
            self.params.blinding_rows,
        );
        if public.weights_root(|bytes| hash(Domain::WeightsRoot, bytes)) != self.trusted_root
            || layout != claimed
        {
            return Err(Refused::Usage);
        }
        let root = public.roots[d.matrix.index()];
        let (k, m) = (d.k(), d.m());
        let n = k << self.params.log2_inv_rate;
        assert_eq!(opening.point.len(), d.vars(), "the verifier's own point");
        assert_eq!(opening.xi.len(), ceil_log2(m), "the verifier's own ξ*_T");
        let (column_point, row_point) = opening.point.split_at(d.log2_k as usize);

        // RFC-0011 step 6: shapes, then θ·v = ⟨u_T, eq(column point)⟩.
        let rows = opening.rows;
        let evaluation = tv_fp2s(&rows.evaluation.u);
        let proximity = tv_fp2s(&rows.proximity.u);
        if rows.matrix != d.matrix
            || evaluation.len() != k
            || proximity.len() != k
            || !rows.evaluation.mask.is_empty()
            || !rows.proximity.mask.is_empty()
        {
            return Err(Refused::Evaluation);
        }
        if opening.theta * opening.value != extension(&evaluation, column_point) {
            return Err(Refused::Evaluation);
        }

        // Step 8.1 and 8.2: the queries, then one column of m_T symbols per distinct index,
        // whose leaves, each its chain's last digest, lead to the root.
        let distinct: Vec<u64> = opening
            .queries
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert!(!distinct.is_empty() && distinct.iter().all(|&j| j < n as u64));
        let smallest = distinct[0];
        let merkle = Refused::MerklePath { column: smallest };
        if opening.columns.len() != distinct.len()
            || opening.columns.iter().any(|column| column.len() != m)
        {
            return Err(merkle);
        }
        let commitment_id = self.commitment_id(public.epoch);
        let block = NonZeroU32::new(self.row_block).expect("B ≥ 1");
        let leaves: Vec<(u64, Digest)> = distinct
            .iter()
            .zip(opening.columns)
            .map(|(&j, column)| {
                let symbols: Vec<TvFp> = column
                    .iter()
                    .map(|x| TvFp::from_canonical(x.to_canonical_u64()).expect("canonical"))
                    .collect();
                let leaf = tv_hash::column::column_digest(&commitment_id, j, block, &symbols);
                (j, leaf)
            })
            .collect();
        verify_multiproof(&root, n as u64, &leaves, opening.siblings).map_err(|_| merkle)?;

        // Step 8.3: each combined row against each queried column, the evaluation row's data
        // rows weighted θ·eq(row point, r_j), the proximity row's eq(ξ*_T, j).
        let coefficients = [
            d.rows
                .iter()
                .map(|&r| opening.theta * eq_at(row_point, r))
                .collect::<Vec<_>>(),
            (0..m as u64).map(|j| eq_at(opening.xi, j)).collect(),
        ];
        let codewords = [encode_row(&evaluation, n), encode_row(&proximity, n)];
        for (&j, column) in distinct.iter().zip(opening.columns) {
            for (coefficients, codeword) in coefficients.iter().zip(&codewords) {
                let combined = coefficients
                    .iter()
                    .zip(column)
                    .fold(TvFp2::ZERO, |sum, (&c, x)| {
                        sum + c * TvFp::from_canonical(x.to_canonical_u64()).expect("canonical")
                    });
                if combined != codeword[j as usize] {
                    return Err(Refused::Proximity { column: j });
                }
            }
        }
        Ok(())
    }
}

/// `enc(u)` at `7·ω_n^j` for `j < n`: the polynomial of degree below `k` through `u` on `H`, on
/// the coset `7·L`, with no mask (RFC-0011 'Encoding', RFC-0040).
fn encode_row(u: &[TvFp2], n: usize) -> Vec<TvFp2> {
    let mut codeword = u.to_vec();
    tv_field::intt(&mut codeword).expect("a power of two");
    codeword.resize(n, TvFp2::ZERO);
    tv_field::coset_ntt(&mut codeword, TvFp::SEVEN).expect("a power of two");
    codeword
}

// ----- Openings --------------------------------------------------------------------------------

fn fp2(rng: &mut SplitMix64) -> Fp2 {
    let c0 = rng.fp();
    Fp2::new(c0, rng.fp())
}

fn fp2s(rng: &mut SplitMix64, n: usize) -> Vec<Fp2> {
    (0..n).map(|_| fp2(rng)).collect()
}

/// A random claim on one matrix and the challenges the transcript would give it.
struct Challenge {
    /// The claim's point on the matrix's extension: for a weight type `r_i ‖ r_o ‖ ρ_ℓ`
    /// (RFC-0010), for `ChannelParams` its column point and its row point.
    point: Vec<Fp2>,
    /// `ξ*_T`, `⌈log2 m_T⌉` coordinates (RFC-0049).
    xi: Vec<Fp2>,
    /// `θ_T`.
    theta: Fp2,
    /// `τ_W` column indices, drawn with replacement, one repeated.
    queries: Vec<u64>,
}

impl Challenge {
    fn random(rng: &mut SplitMix64, d: &Defined) -> Self {
        let n = (d.k() as u64) << 2;
        let mut queries: Vec<u64> = (0..QUERIES - 1).map(|_| rng.next_u64() % n).collect();
        queries.push(queries[rng.below(queries.len())]);
        Self {
            point: fp2s(rng, d.vars()),
            xi: fp2s(rng, ceil_log2(d.m())),
            theta: fp2(rng),
            queries,
        }
    }

    /// The point the fold takes, known when the pass starts: the column point's top `a`
    /// coordinates, then the row point below the layer index (`r_o` for a weight type).
    fn fold_point(&self, d: &Defined) -> &[Fp2] {
        &self.point[(d.log2_k - d.a) as usize..(d.log2_k + d.h) as usize]
    }

    /// `ρ_ℓ`, the row point's layer coordinates.
    fn layer_point(&self, d: &Defined) -> &[Fp2] {
        &self.point[(d.log2_k + d.h) as usize..]
    }

    /// Where the claim's value is the fold's table's extension: the free columns, then `ρ_ℓ`.
    fn table_point(&self, d: &Defined) -> Vec<Fp2> {
        let free = &self.point[..(d.log2_k - d.a) as usize];
        [free, self.layer_point(d)].concat()
    }
}

/// A prover's opening of one claim: the fold pass over `rows`, the claim's value from the fold's
/// table, as the sumcheck would give it, the combined rows and the column answers.
#[derive(Clone)]
struct Proven {
    value: TvFp2,
    rows: WeightCombinedRows,
    columns: Vec<Vec<Fp>>,
    siblings: Vec<Digest>,
}

fn prove(
    layout: &WeightLayout,
    commitment: &WeightCommitment,
    d: &Defined,
    rows: &mut dyn MatrixRows,
    challenge: &Challenge,
) -> Proven {
    let fold = WeightFold::new(
        layout,
        d.matrix,
        rows,
        challenge.fold_point(d),
        &challenge.xi,
    )
    .expect("a valid fold");
    let table = tv_fp2s(&fold.sumcheck_input());
    let value = extension(&table, &tv_fp2s(&challenge.table_point(d)));
    let combined = fold
        .combined_rows(challenge.layer_point(d), challenge.theta)
        .expect("a valid layer point");
    let answer = commitment
        .matrix(d.matrix)
        .answer(&challenge.queries)
        .expect("valid queries");
    Proven {
        value,
        rows: combined,
        columns: answer.columns,
        siblings: answer.proof.siblings,
    }
}

fn verify(verifier: &Verifier<'_>, challenge: &Challenge, proven: &Proven) -> Result<(), Refused> {
    verifier.verify(&Opening {
        point: &tv_fp2s(&challenge.point),
        value: proven.value,
        theta: tv_fp2(challenge.theta),
        xi: &tv_fp2s(&challenge.xi),
        rows: &proven.rows,
        queries: &challenge.queries,
        columns: &proven.columns,
        siblings: &proven.siblings,
    })
}

/// A committed model: its layout, its commitment, the public artifact's bytes and `R_W`.
struct Committed {
    layout: WeightLayout,
    commitment: WeightCommitment,
    artifact: Vec<u8>,
    defined: Vec<Defined>,
}

impl Committed {
    fn new(model: &Model) -> Self {
        let layout = model.layout();
        let block = NonZeroU32::new(ROW_BLOCK).expect("B ≥ 1");
        let commitment =
            WeightCommitment::commit(&layout, block, 0, &model.weights()).expect("a commitment");
        let artifact = encode(commitment.public());
        Self {
            layout,
            commitment,
            artifact,
            defined: define(model),
        }
    }

    fn verifier<'a>(&'a self, model: &Model, d: &'a Defined) -> Verifier<'a> {
        Verifier {
            trusted_root: self.commitment.weights_root(),
            artifact: &self.artifact,
            params: params(model.log2_k),
            row_block: ROW_BLOCK,
            defined: d,
        }
    }
}

// ----- Fold equivalence --------------------------------------------------------------------------

/// The fold of every matrix of `model` at random points: its table against the contraction of
/// the INT8 weights and against the committed rows' definition, every layer and every padded
/// one, and both combined rows against theirs.
fn check_fold(model: &Model, rng: &mut SplitMix64) {
    let layout = model.layout();
    let weights = model.weights();
    for d in define(model) {
        let at = format!("{} {:?}", model.name, d.matrix);
        let found = layout.matrix(d.matrix);
        assert_eq!(
            (found.log2_k(), found.row_vars(), found.rows()),
            (d.log2_k, d.layer_vars + d.h, d.m() as u64),
            "{at}: the layout is RFC-0040's"
        );
        if d.m() == 0 {
            continue;
        }
        let challenge = Challenge::random(rng, &d);
        let mut rows = layout.rows(d.matrix, &weights).expect("valid weights");
        let fold = WeightFold::new(
            &layout,
            d.matrix,
            &mut rows,
            challenge.fold_point(&d),
            &challenge.xi,
        )
        .expect("a valid fold");
        let table = tv_fp2s(&fold.sumcheck_input());
        let fold_point = tv_fp2s(challenge.fold_point(&d));
        assert_eq!(
            table,
            bound_table(&d, &fold_point),
            "{at}: the fold's table"
        );
        if d.matrix != WeightMatrix::ChannelParams {
            // Σ_{o_lo} eq(r_o^lo, o_lo)·F_ℓ(o_lo, i) = W̃_ℓ(r_o, i), padded layers zero.
            let in_pad = d.k() >> d.a;
            for (layer, part) in table.chunks_exact(in_pad).enumerate() {
                let expected = match (layer as u64) < d.layers {
                    true => contracted(model, d.matrix, layer as u32, &fold_point),
                    false => vec![TvFp2::ZERO; in_pad],
                };
                assert_eq!(part, expected.as_slice(), "{at}: W̃_{layer}(r_o, ·)");
            }
        }
        let rows = fold
            .combined_rows(challenge.layer_point(&d), challenge.theta)
            .expect("a valid layer point");
        let (u, u_xi) = defined_rows(
            &d,
            &fold_point,
            &tv_fp2s(challenge.layer_point(&d)),
            tv_fp2(challenge.theta),
            &tv_fp2s(&challenge.xi),
        );
        assert_eq!(rows.matrix, d.matrix, "{at}");
        assert_eq!(tv_fp2s(&rows.evaluation.u), u, "{at}: u_T");
        assert_eq!(tv_fp2s(&rows.proximity.u), u_xi, "{at}: u_ξ,T");
        assert!(rows.evaluation.mask.is_empty() && rows.proximity.mask.is_empty());
    }
}

#[test]
fn the_fold_gives_each_layers_contraction_on_the_conformance_models() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0001);
    let models = conformance_models();
    assert_eq!(
        models.len(),
        5,
        "four conformance models and the synthetic one"
    );
    for model in &models {
        check_fold(model, &mut rng);
    }
}

#[test]
fn the_fold_gives_each_layers_contraction_on_random_shapes_wider_than_k_w() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0002);
    let models = random_models(&mut rng);
    // Inputs wider than k_W, so that a row holds one input row (a_T = 0), and whole layers in
    // one row (h_T = 0), both occur.
    let defined: Vec<Vec<Defined>> = models.iter().map(define).collect();
    let weights = defined
        .iter()
        .flatten()
        .filter(|d| d.matrix != WeightMatrix::ChannelParams);
    let (mut wide, mut whole) = (false, false);
    for (model, d) in models.iter().zip(&defined) {
        for d in d.iter().filter(|d| d.matrix != WeightMatrix::ChannelParams) {
            let inputs = model.shapes.shapes[d.matrix.index()].inputs;
            wide |= pad(u64::from(inputs)) > 1 << model.log2_k;
        }
    }
    for d in weights {
        whole |= d.h == 0;
    }
    assert!(wide && whole);
    for model in &models {
        check_fold(model, &mut rng);
    }
}

// ----- One pass ------------------------------------------------------------------------------------

/// The production reader, with a count of each committed row it is asked for.
struct CountedRows<R> {
    inner: R,
    reads: Vec<u32>,
}

impl<R: MatrixRows> MatrixRows for CountedRows<R> {
    fn matrix(&self) -> WeightMatrix {
        self.inner.matrix()
    }

    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError> {
        self.reads[index] += 1;
        self.inner.read_row(index, out)
    }
}

/// A reader written from RFC-0040 here, over the model's own tensors, that counts every weight
/// and channel parameter it copies into a row.
struct WeightCounter<'m> {
    model: &'m Model,
    defined: &'m Defined,
    /// Per weight tensor, in `model.tensors`' order, a count per entry; per `ChannelParams`, in
    /// `model.channels`' order, a count per value.
    tensor_reads: Vec<Vec<u32>>,
    channel_reads: Vec<Vec<u32>>,
}

impl<'m> WeightCounter<'m> {
    fn new(model: &'m Model, defined: &'m Defined) -> Self {
        Self {
            model,
            defined,
            tensor_reads: model.tensors.iter().map(|t| vec![0; t.2.len()]).collect(),
            channel_reads: model.channels.iter().map(|c| vec![0; c.2.len()]).collect(),
        }
    }
}

impl MatrixRows for WeightCounter<'_> {
    fn matrix(&self) -> WeightMatrix {
        self.defined.matrix
    }

    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError> {
        let d = self.defined;
        out.fill(Fp::ZERO);
        let r = d.rows[index];
        let k = d.k() as u64;
        let model = self.model;
        if d.matrix == WeightMatrix::ChannelParams {
            let l_pad = pad(u64::from(model.shapes.layers));
            let c = (k << d.h) / (16 * l_pad);
            for (n, (matrix, layer, values)) in model.channels.iter().enumerate() {
                let base = (matrix.index() as u64 * l_pad + u64::from(*layer)) * c;
                for channel in 0..values.len() {
                    let f = base + channel as u64;
                    if f / k == r {
                        out[(f % k) as usize] = Fp::from_u64_reduced(values.get(channel));
                        self.channel_reads[n][channel] += 1;
                    }
                }
            }
            return Ok(());
        }
        let shape = model.shapes.shapes[d.matrix.index()];
        let (o, i) = (u64::from(shape.out), u64::from(shape.inputs));
        let (layer, o_hi) = (r >> d.h, r & ((1 << d.h) - 1));
        let n = model
            .tensors
            .iter()
            .position(|(m, l, _)| (*m, u64::from(*l)) == (d.matrix, layer))
            .expect("the layer's tensor");
        let in_pad = pad(i);
        for o_lo in 0..(1u64 << d.a) {
            let oo = (o_hi << d.a) + o_lo;
            if oo >= o {
                break;
            }
            for ii in 0..i {
                let at = (oo * i + ii) as usize;
                out[(o_lo * in_pad + ii) as usize] =
                    Fp::from_i64(i64::from(model.tensors[n].2[at]));
                self.tensor_reads[n][at] += 1;
            }
        }
        Ok(())
    }
}

/// INV-WC-4, RFC-0049 'Testing Strategy': a proof reads each weight once. For every matrix, the
/// fold reads each committed row once, in order, and nothing after it reads a weight: the
/// sumcheck input, both combined rows and the column answers come from the fold and from the
/// commitment's codewords. Counted per weight, every INT8 entry and every channel parameter of
/// `D` is read exactly once across the ten matrices, and the counting reader, written from
/// RFC-0040 here, gives the production reader's combined rows.
#[test]
fn a_proof_reads_each_weight_once() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0003);
    let mut models = conformance_models();
    models.truncate(1);
    models.extend(random_models(&mut rng).into_iter().skip(3).take(2));
    for model in &models {
        let committed = Committed::new(model);
        let weights = model.weights();
        let mut counter_reads: (Vec<Vec<u32>>, Vec<Vec<u32>>) = (
            model.tensors.iter().map(|t| vec![0; t.2.len()]).collect(),
            model.channels.iter().map(|c| vec![0; c.2.len()]).collect(),
        );
        for d in &committed.defined {
            let at = format!("{} {:?}", model.name, d.matrix);
            if d.m() == 0 {
                continue;
            }
            let challenge = Challenge::random(&mut rng, d);
            let mut counted = CountedRows {
                inner: committed
                    .layout
                    .rows(d.matrix, &weights)
                    .expect("valid weights"),
                reads: vec![0; d.m()],
            };
            let proven = prove(
                &committed.layout,
                &committed.commitment,
                d,
                &mut counted,
                &challenge,
            );
            assert_eq!(counted.reads, vec![1; d.m()], "{at}: each row once");
            let mut counter = WeightCounter::new(model, d);
            let by_counter = prove(
                &committed.layout,
                &committed.commitment,
                d,
                &mut counter,
                &challenge,
            );
            assert!(
                by_counter.rows == proven.rows,
                "{at}: the same combined rows"
            );
            for (total, reads) in counter_reads.0.iter_mut().zip(&counter.tensor_reads) {
                for (t, r) in total.iter_mut().zip(reads) {
                    *t += r;
                }
            }
            for (total, reads) in counter_reads.1.iter_mut().zip(&counter.channel_reads) {
                for (t, r) in total.iter_mut().zip(reads) {
                    *t += r;
                }
            }
            assert_eq!(
                verify(&committed.verifier(model, d), &challenge, &proven),
                Ok(()),
                "{at}"
            );
        }
        for (reads, (matrix, layer, _)) in counter_reads.0.iter().zip(&model.tensors) {
            assert!(
                reads.iter().all(|&r| r == 1),
                "{}: {matrix:?} of layer {layer}: every weight once",
                model.name
            );
        }
        for (reads, (matrix, layer, _)) in counter_reads.1.iter().zip(&model.channels) {
            assert!(
                reads.iter().all(|&r| r == 1),
                "{}: channel parameters of {matrix:?} of layer {layer}: once",
                model.name
            );
        }
    }
}

// ----- End to end ------------------------------------------------------------------------------

/// A claim on each matrix at a random point opens and verifies against `R_W`: the verifier takes
/// the root it trusts and the epoch's public artifact, never a weight. The claim's value, which
/// the matmul sumcheck would give, is the fold's table at the free columns and `ρ_ℓ`, and equals
/// the weights' own extension at the point.
#[test]
fn a_claim_on_each_matrix_opens_and_verifies_against_the_weights_root() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0004);
    let mut models = conformance_models();
    models.extend(random_models(&mut rng));
    for model in &models {
        let committed = Committed::new(model);
        let weights = model.weights();
        for d in &committed.defined {
            let at = format!("{} {:?}", model.name, d.matrix);
            if d.m() == 0 {
                continue;
            }
            let challenge = Challenge::random(&mut rng, d);
            let mut rows = committed.layout.rows(d.matrix, &weights).expect("weights");
            let proven = prove(
                &committed.layout,
                &committed.commitment,
                d,
                &mut rows,
                &challenge,
            );
            // Ã_T(point) from the committed rows' definition, and for a weight type from the
            // INT8 weights: Σ_ℓ eq(ρ, ℓ)·Σ_i eq(r_i, i)·W̃_ℓ(r_o, i).
            let point = tv_fp2s(&challenge.point);
            let (column_point, row_point) = point.split_at(d.log2_k as usize);
            let by_rows = d.rows.iter().enumerate().fold(TvFp2::ZERO, |sum, (j, &r)| {
                let row: Vec<TvFp2> = d.row(j).iter().map(|&x| TvFp2::from(x)).collect();
                sum + eq_at(row_point, r) * extension(&row, column_point)
            });
            assert_eq!(proven.value, by_rows, "{at}: the claim's value");
            if d.matrix != WeightMatrix::ChannelParams {
                let free = (d.log2_k - d.a) as usize;
                let (r_i, rest) = point.split_at(free);
                let (r_o, rho) = rest.split_at((d.a + d.h) as usize);
                let direct = (0..d.layers).fold(TvFp2::ZERO, |sum, layer| {
                    let table = contracted(model, d.matrix, layer as u32, r_o);
                    sum + eq_at(rho, layer) * extension(&table, r_i)
                });
                assert_eq!(proven.value, direct, "{at}: W̃_T(r_i ‖ r_o ‖ ρ)");
            }
            assert_eq!(
                verify(&committed.verifier(model, d), &challenge, &proven),
                Ok(()),
                "{at}"
            );
        }
    }
}

// ----- Binding -----------------------------------------------------------------------------------

/// The production reader with one position of one row replaced: a prover whose rows differ from
/// the committed ones there.
struct Tampered<R> {
    inner: R,
    row: usize,
    column: usize,
    value: Fp,
}

impl<R: MatrixRows> MatrixRows for Tampered<R> {
    fn matrix(&self) -> WeightMatrix {
        self.inner.matrix()
    }

    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError> {
        self.inner.read_row(index, out)?;
        if index == self.row {
            out[self.column] = self.value;
        }
        Ok(())
    }
}

/// The smallest queried column: where a single changed entry makes every combined row it enters
/// disagree first. A change of `δ` at column `c` of a row with coefficient `w ≠ 0` changes the
/// row's encoding by `w·δ·L_c`, `L_c` the Lagrange polynomial of `ω_k^c` on `H`, which vanishes
/// only on `H`: never at a queried point `7·ω_n^j`.
fn smallest(queries: &[u64]) -> u64 {
    *queries.iter().min().expect("queries")
}

/// Weights changed after the commitment are rejected at the opening (RFC-0012 'Testing
/// Strategy', RFC-0049): one entry of every matrix, a `ChannelParams` entry among them, and a
/// padding position of a committed row. The prover runs honestly on what it reads, so its
/// claim's value matches its evaluation row and step 6 passes; the columns, the commitment's,
/// refuse both combined rows at the smallest queried column. The honest opening of the same
/// claim verifies.
#[test]
fn weights_changed_after_the_commitment_are_rejected_at_the_opening() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0005);
    let mut models = conformance_models();
    models.truncate(2);
    models.extend(random_models(&mut rng).into_iter().take(4));
    let mut padding_cases = 0;
    let mut channel_cases = 0;
    for model in &models {
        let committed = Committed::new(model);
        let weights = model.weights();
        for d in &committed.defined {
            let at = format!("{} {:?}", model.name, d.matrix);
            if d.m() == 0 {
                continue;
            }
            let verifier = committed.verifier(model, d);
            let challenge = Challenge::random(&mut rng, d);
            let honest = prove(
                &committed.layout,
                &committed.commitment,
                d,
                &mut committed.layout.rows(d.matrix, &weights).expect("weights"),
                &challenge,
            );
            assert_eq!(verify(&verifier, &challenge, &honest), Ok(()), "{at}");
            let refused = Refused::Proximity {
                column: smallest(&challenge.queries),
            };

            // One entry of D changed: a weight, or one channel parameter.
            let mut changed = model.clone();
            if d.matrix == WeightMatrix::ChannelParams {
                let n = rng.below(changed.channels.len());
                let c = rng.below(changed.channels[n].2.len());
                changed.channels[n].2.change(c);
                channel_cases += 1;
            } else {
                let layers = if d.matrix.per_layer() {
                    model.shapes.layers
                } else {
                    1
                };
                let tensor = changed.tensor_mut(d.matrix, rng.below(layers as usize) as u32);
                let at = rng.below(tensor.len());
                // A change of 1 to 255 modulo 256: another INT8 value.
                let delta = (1 + rng.below(255)) as u8 as i8;
                tensor[at] = tensor[at].wrapping_add(delta);
            }
            let cheat = prove(
                &committed.layout,
                &committed.commitment,
                d,
                &mut committed
                    .layout
                    .rows(d.matrix, &changed.weights())
                    .expect("weights"),
                &challenge,
            );
            assert_ne!(
                cheat.value, honest.value,
                "{at}: the change shows in the claim"
            );
            assert_eq!(verify(&verifier, &challenge, &cheat), Err(refused), "{at}");

            // A padding position of a committed row set to a non-zero value.
            if let Some(position) = (0..d.live.len()).find(|&p| !d.live[p]) {
                let cheat = prove(
                    &committed.layout,
                    &committed.commitment,
                    d,
                    &mut Tampered {
                        inner: committed.layout.rows(d.matrix, &weights).expect("weights"),
                        row: position / d.k(),
                        column: position % d.k(),
                        value: rng.nonzero(),
                    },
                    &challenge,
                );
                assert_eq!(
                    verify(&verifier, &challenge, &cheat),
                    Err(refused),
                    "{at}: padding at {position}"
                );
                padding_cases += 1;
            }
        }
    }
    assert!(
        padding_cases > 10 && channel_cases > 3,
        "{padding_cases}, {channel_cases}"
    );
}

/// A prover that keeps the honest claim but folds changed weights fails step 6 instead: its
/// evaluation row no longer gives the claim.
#[test]
fn a_claim_on_other_weights_fails_the_evaluation_check() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0006);
    let model = &conformance_models()[0];
    let committed = Committed::new(model);
    let weights = model.weights();
    for d in &committed.defined {
        let verifier = committed.verifier(model, d);
        let challenge = Challenge::random(&mut rng, d);
        let honest = prove(
            &committed.layout,
            &committed.commitment,
            d,
            &mut committed.layout.rows(d.matrix, &weights).expect("weights"),
            &challenge,
        );
        let live = (0..d.live.len())
            .filter(|&p| d.live[p])
            .nth(rng.below(8))
            .expect("live");
        // Another value than the committed one.
        let committed_value = d.values[live].to_canonical();
        let value =
            Fp::from_u64_reduced(committed_value) + Fp::from_u64_reduced(1 + rng.below(100) as u64);
        let mut cheat = prove(
            &committed.layout,
            &committed.commitment,
            d,
            &mut Tampered {
                inner: committed.layout.rows(d.matrix, &weights).expect("weights"),
                row: live / d.k(),
                column: live % d.k(),
                value,
            },
            &challenge,
        );
        cheat.value = honest.value;
        assert_eq!(
            verify(&verifier, &challenge, &cheat),
            Err(Refused::Evaluation),
            "{} {:?}",
            model.name,
            d.matrix
        );
    }
}

/// The in-test verifier binds what it accepts to the root it trusts, as #109's verifier must: one
/// changed symbol of an answered column, and one changed byte of a multiproof sibling, are
/// `MerklePath` at the smallest queried column; the public artifact of another epoch, whose
/// roots differ, does not hash to the trusted `R_W` and is `ERR-USAGE` (RFC-0040 'The verifier's
/// weight inputs'). So the openings it accepts verify against `R_W`, not against any root.
#[test]
fn the_in_test_verifier_binds_every_column_to_the_trusted_weights_root() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0008);
    let model = &conformance_models()[0];
    let committed = Committed::new(model);
    let weights = model.weights();
    let block = NonZeroU32::new(ROW_BLOCK).expect("B ≥ 1");
    let next_epoch =
        WeightCommitment::commit(&committed.layout, block, 1, &weights).expect("a commitment");
    assert_ne!(
        next_epoch.weights_root(),
        committed.commitment.weights_root()
    );
    let other_artifact = encode(next_epoch.public());
    for d in &committed.defined {
        let at = format!("{} {:?}", model.name, d.matrix);
        let verifier = committed.verifier(model, d);
        let challenge = Challenge::random(&mut rng, d);
        let honest = prove(
            &committed.layout,
            &committed.commitment,
            d,
            &mut committed.layout.rows(d.matrix, &weights).expect("weights"),
            &challenge,
        );
        assert_eq!(verify(&verifier, &challenge, &honest), Ok(()), "{at}");
        let merkle = Refused::MerklePath {
            column: smallest(&challenge.queries),
        };
        let mut symbol = honest.clone();
        let column = rng.below(symbol.columns.len());
        let entry = rng.below(symbol.columns[column].len());
        symbol.columns[column][entry] += Fp::ONE;
        assert_eq!(verify(&verifier, &challenge, &symbol), Err(merkle), "{at}");
        let mut sibling = honest.clone();
        let at_sibling = rng.below(sibling.siblings.len());
        sibling.siblings[at_sibling][rng.below(32)] ^= 1;
        assert_eq!(verify(&verifier, &challenge, &sibling), Err(merkle), "{at}");
        let elsewhere = Verifier {
            artifact: &other_artifact,
            ..committed.verifier(model, d)
        };
        assert_eq!(
            verify(&elsewhere, &challenge, &honest),
            Err(Refused::Usage),
            "{at}"
        );
    }
}

// ----- Checks ----------------------------------------------------------------------------------

/// Each input the fold refuses, with its error, before it reads a row: a point or `ξ*_T` of the
/// wrong length, a reader of another matrix, and a matrix with no committed row, which has no
/// combined rows and no `ξ*_T` (RFC-0049); then a layer point of the wrong length. The readers
/// refuse a row past `m_T` and a buffer of the wrong length.
#[test]
fn the_fold_and_the_readers_refuse_what_they_cannot_open() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0108_0007);
    let models = conformance_models();
    let model = &models[0];
    let layout = model.layout();
    let weights = model.weights();
    let defined = define(model);
    let d = &defined[WeightMatrix::Q.index()];
    let challenge = Challenge::random(&mut rng, d);
    let fold_point = challenge.fold_point(d);
    let rows = || layout.rows(WeightMatrix::Q, &weights).expect("weights");
    let refused = |point: &[Fp2], xi: &[Fp2], reader: &mut dyn MatrixRows| {
        let mut counted = CountedRows {
            inner: reader,
            reads: vec![0; d.m()],
        };
        let error = WeightFold::new(&layout, WeightMatrix::Q, &mut counted, point, xi)
            .map(|_| ())
            .expect_err("refused");
        assert!(
            counted.reads.iter().all(|&r| r == 0),
            "{error}: nothing read"
        );
        error
    };
    let longer = [fold_point, &[Fp2::ONE]].concat();
    assert_eq!(
        refused(&longer, &challenge.xi, &mut rows()),
        PcsError::FoldPoint {
            matrix: WeightMatrix::Q,
            expected: fold_point.len(),
            found: fold_point.len() + 1,
        }
    );
    assert_eq!(
        refused(fold_point, &challenge.xi[1..], &mut rows()),
        PcsError::XiPoint {
            expected: challenge.xi.len(),
            found: challenge.xi.len() - 1,
        }
    );
    let mut other = layout.rows(WeightMatrix::O, &weights).expect("weights");
    assert_eq!(
        refused(fold_point, &challenge.xi, &mut other),
        PcsError::OtherMatrix {
            expected: WeightMatrix::Q,
            found: WeightMatrix::O,
        }
    );
    let fold = WeightFold::new(
        &layout,
        WeightMatrix::Q,
        &mut rows(),
        fold_point,
        &challenge.xi,
    )
    .expect("a fold");
    assert_eq!(fold.matrix(), WeightMatrix::Q);
    let layer_point = challenge.layer_point(d);
    assert_eq!(
        fold.combined_rows(&layer_point[1..], challenge.theta)
            .map(|_| ()),
        Err(PcsError::LayerPoint {
            matrix: WeightMatrix::Q,
            expected: layer_point.len(),
            found: layer_point.len() - 1,
        })
    );
    // The readers: a row past m_T, and a buffer that is not k_T values.
    let mut buffer = vec![Fp::ZERO; d.k()];
    assert_eq!(
        rows().read_row(d.m(), &mut buffer),
        Err(PcsError::RowIndex {
            matrix: WeightMatrix::Q,
            index: d.m(),
            rows: d.m() as u64,
        })
    );
    let mut short = vec![Fp::ZERO; d.k() - 1];
    assert_eq!(
        rows().read_row(0, &mut short),
        Err(PcsError::RowLength {
            expected: d.k(),
            found: d.k() - 1,
        })
    );
    // The synthetic model has no ChannelParams: its matrix has no committed row.
    let synthetic = models.last().expect("the synthetic model");
    assert!(synthetic.channels.is_empty());
    let layout = synthetic.layout();
    let weights = synthetic.weights();
    let mut empty = layout
        .rows(WeightMatrix::ChannelParams, &weights)
        .expect("weights");
    let row_vars = layout.matrix(WeightMatrix::ChannelParams).row_vars() as usize;
    assert_eq!(
        WeightFold::new(
            &layout,
            WeightMatrix::ChannelParams,
            &mut empty,
            &fp2s(&mut rng, row_vars),
            &[],
        )
        .map(|_| ()),
        Err(PcsError::NoRows {
            matrix: WeightMatrix::ChannelParams
        })
    );
    // Weights the layout refuses are refused before any reader exists.
    let mut missing = synthetic.weights();
    missing.tensors.pop();
    assert_eq!(
        layout.rows(WeightMatrix::Q, &missing).map(|_| ()),
        Err(PcsError::WeightMissing {
            matrix: WeightMatrix::LmHead,
            layer: 0,
            channels: false,
        })
    );
}
