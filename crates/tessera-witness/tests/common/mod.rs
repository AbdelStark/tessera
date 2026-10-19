// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support for the checkpoint and regeneration tests: the conformance models of
//! `spec/vectors/models/` with the independent oracle's residuals, a model's execution plan under a
//! query block, a sequence's lowering with its gadgets, and drawn tokens.

#![allow(dead_code)] // each test binary uses part of it

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use tessera_codec::{Ascii, DecodeLimits, envelope, schema};
use tessera_compiler::{
    ExecutionPlan, Lowering, PublicModel, Selection, analyze, claim_list, lower, plan, select,
    validate,
};
use tessera_ir::{
    AdviceLayoutParams, CommitmentLayout, GadgetPolicy, IntegerModel, ParamSetId,
    WeightLayoutParams,
};

/// `test-small/1`'s layout (RFC-0014 'Parameter sets').
pub const TEST_SMALL: CommitmentLayout = CommitmentLayout {
    advice: AdviceLayoutParams {
        log2_k: 10,
        log2_inv_rate: 1,
        mask_coeffs: 16,
        row_block: 4,
        segments: 1,
    },
    weights: WeightLayoutParams {
        log2_k: 8,
        log2_inv_rate: 2,
        mask_coeffs: 64,
        blinding_rows: 32,
    },
};

/// Query blocks: RFC-0006's default, which holds the whole context of every conformance model, and
/// two that split it, into blocks of four queries and of one (RFC-0041 'Tiles').
pub const QUERY_BLOCKS: [u32; 3] = [512, 4, 1];

/// One sequence of a conformance model: its tokens and the residual the oracle computed at each
/// layer boundary, `c_0` first, each flat and row-major.
pub struct Sequence {
    pub tokens: Vec<u32>,
    pub residuals: Vec<Vec<i64>>,
}

/// A conformance model (07 §5) with its name and its sequences.
pub struct Conformance {
    pub name: String,
    pub model: IntegerModel,
    pub sequences: Vec<Sequence>,
}

pub fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models")
}

fn json(path: PathBuf) -> serde_json::Value {
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("JSON")
}

fn integers(value: &serde_json::Value) -> Vec<i64> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .flat_map(|row| row.as_array().expect("a row"))
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

/// Every conformance model, in the index's order.
pub fn conformance() -> Vec<Conformance> {
    let index = json(models_dir().join("models.json"));
    let models: Vec<Conformance> = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .map(|entry| {
            let file = entry["file"].as_str().expect("a file");
            let bytes = fs::read(models_dir().join(file)).expect("the artifact");
            let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
                .expect("an IntegerModel");
            validate(&model).expect("a valid model");
            let sequences = entry["sequences"]
                .as_array()
                .expect("sequences")
                .iter()
                .map(|s| Sequence {
                    tokens: s["tokens"]
                        .as_array()
                        .expect("tokens")
                        .iter()
                        .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("a u32"))
                        .collect(),
                    residuals: s["residuals"]
                        .as_array()
                        .expect("residuals")
                        .iter()
                        .map(integers)
                        .collect(),
                })
                .collect();
            Conformance {
                name: entry["name"].as_str().expect("a name").to_owned(),
                model,
                sequences,
            }
        })
        .collect();
    assert_eq!(models.len(), 4, "every conformance model");
    models
}

/// The default gadget policy with query blocks of `query_block` positions.
pub fn policy(query_block: u32) -> GadgetPolicy {
    GadgetPolicy {
        query_block,
        ..GadgetPolicy::default()
    }
}

/// `model`'s execution plan for `test-small/1`, under the default gadget policy with query blocks
/// of `query_block` positions (RFC-0041 'The plan').
pub fn plan_of(model: &IntegerModel, query_block: u32) -> ExecutionPlan {
    let public = PublicModel::of(model);
    let lowering = lower(public, 1).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = policy(query_block);
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("every gadget fits");
    let params = ParamSetId(Ascii::new("test-small/1").expect("ASCII"));
    let claims = claim_list(public, &selection, &policy, TEST_SMALL, params);
    plan(public, &claims).expect("a plan")
}

/// A sequence of `n` tokens of `model`: its lowering and the gadgets the default policy selects.
/// The query block does not enter the selection, only the tiles.
pub fn lowered(model: &IntegerModel, n: u32) -> (Lowering, Selection) {
    let lowering = lower(PublicModel::of(model), n).expect("the model lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .expect("every gadget fits");
    (lowering, selection)
}

/// SplitMix64, for drawing tokens reproducibly.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `n` tokens below `vocab`, the vocabulary's ends among them.
    pub fn tokens(&mut self, vocab: u32, n: u32) -> Vec<u32> {
        (0..n)
            .map(|_| match self.next() % 4 {
                0 => 0,
                1 => vocab - 1,
                _ => u32::try_from(self.next() % u64::from(vocab)).expect("a token"),
            })
            .collect()
    }
}

/// The row-major index, in a tensor of `shape`, of every element whose coordinate on each axis lies
/// in that axis's range, in row-major order.
pub fn indices(shape: &[u32], ranges: &[Range<u32>]) -> Vec<usize> {
    assert_eq!(ranges.len(), shape.len(), "one range per axis");
    let mut indices = vec![0usize];
    for (range, &len) in ranges.iter().zip(shape) {
        assert!(
            range.start <= range.end && range.end <= len,
            "{range:?} lies in an axis of {len}"
        );
        indices = indices
            .iter()
            .flat_map(|&outer| {
                range
                    .clone()
                    .map(move |i| outer * len as usize + i as usize)
            })
            .collect();
    }
    indices
}

/// The table counts this foundation cannot produce: root and merged row-gadget auxiliaries.
pub fn unsupported_table(selection: &Selection, tile: &tessera_compiler::SequenceTile) -> bool {
    let tessera_compiler::Tile::Table { table } = tile.tile else {
        return false;
    };
    selection.gadgets.iter().any(|(node, gadget)| match gadget {
        tessera_compiler::Gadget::RsqrtRow => selection
            .reads(*node)
            .unwrap()
            .contains(&tessera_ir::TableId(table)),
        tessera_compiler::Gadget::SoftmaxRow {
            reciprocal,
            limb,
            top,
            ..
        } => [Some(*reciprocal), *limb, Some(*top)].contains(&Some(tessera_ir::TableId(table))),
        tessera_compiler::Gadget::Argmax { table: id } => id.0 == table,
        _ => false,
    })
}

/// Assert a public refusal instead of permitting absent or partial multiplicities.
pub fn assert_unsupported(
    result: Result<tessera_witness::TileWitness, tessera_witness::WitnessError>,
) {
    assert_eq!(
        result.expect_err("auxiliary table regeneration is unsupported"),
        tessera_witness::WitnessError::Usage {
            rule: "auxiliary RsqrtRow, SoftmaxRow and Argmax multiplicities are not implemented"
        }
    );
}
