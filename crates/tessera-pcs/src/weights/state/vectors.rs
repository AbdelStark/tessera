// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The state's encoder against the golden vectors (RFC-0040 'WeightCommitmentState', 03 §7).
//! `cargo xtask gen-vectors` lays out each state case of
//! `spec/vectors/codec/weight-commitment.json` itself, from RFC-0040's field list. Here the
//! state's own encoder seals the case's value, pinned to the case's seed through
//! [`WeightCommitmentState::with_seed`], and [`WeightCommitmentState::sealed`] must give the
//! case's bytes, byte for byte, so that the encoder and the vectors cannot drift apart. A state
//! pinned to another seed also seals exactly as [`WeightCommitmentState::write`] writes it, its
//! seed after the public part.
//!
//! The test sits inside the crate because both constructors exist only in its own test build
//! and, for other crates, under `test-rng`, which this crate's benchmarks must not enable.

use std::fs;
use std::path::Path;

use serde_json::Value;

use super::{OpenedColumns, TypeBudget, WeightCommitmentState};
use crate::{WeightCommitmentPublic, WeightLayoutParams};

/// The public part (334 bytes) comes first in the state's payload, then the seed (32), after
/// the envelope's 20-byte header.
const SEED_AT: usize = 20 + 334;

fn vectors() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/codec/weight-commitment.json");
    serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON")
}

fn hex(value: &Value) -> Vec<u8> {
    let text = value.as_str().expect("hex");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

/// A number of the vectors' JSON, written as a decimal string.
fn int(value: &Value) -> u64 {
    value
        .as_str()
        .expect("a decimal string")
        .parse()
        .expect("a number")
}

fn bytes_of(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .expect("bytes")
        .iter()
        .map(|b| u8::try_from(int(b)).expect("a byte"))
        .collect()
}

fn public_of(value: &Value) -> WeightCommitmentPublic {
    let layout = &value["layout"];
    WeightCommitmentPublic {
        layout: WeightLayoutParams {
            log2_k: u8::try_from(int(&layout["log2_k"])).expect("a u8"),
            log2_inv_rate: u8::try_from(int(&layout["log2_inv_rate"])).expect("a u8"),
            mask_coeffs: u32::try_from(int(&layout["mask_coeffs"])).expect("a u32"),
            blinding_rows: u32::try_from(int(&layout["blinding_rows"])).expect("a u32"),
        },
        roots: core::array::from_fn(|t| bytes_of(&value["roots"][t]).try_into().expect("32 bytes")),
        epoch: u32::try_from(int(&value["epoch"])).expect("a u32"),
    }
}

fn budget_of(value: &Value) -> [TypeBudget; 10] {
    core::array::from_fn(|t| {
        let budget = &value["budget"][t];
        let columns = budget["opened_columns"].as_array().expect("columns");
        TypeBudget {
            opened_columns: OpenedColumns::new(
                columns
                    .iter()
                    .map(|c| u32::try_from(int(c)).expect("a u32"))
                    .collect(),
            )
            .expect("strictly increasing"),
            next_blinding_row: u32::try_from(int(&budget["next_blinding_row"])).expect("a u32"),
        }
    })
}

#[test]
fn each_state_vector_is_what_the_encoder_seals_from_its_value_and_seed() {
    let mut states = 0;
    for case in vectors()["cases"].as_array().expect("cases") {
        if case["type"] != "WeightCommitmentState" {
            continue;
        }
        let value = &case["value"];
        let pinned = WeightCommitmentState::with_seed(
            public_of(&value["public"]),
            bytes_of(&value["masks_seed"]).try_into().expect("32 bytes"),
            budget_of(value),
        );
        assert_eq!(pinned.sealed()[..], hex(&case["hex"]), "{}", case["name"]);
        states += 1;
    }
    assert_eq!(states, 2, "the vectors' two states");
}

/// `write` sets mode 0600, which only Unix systems have.
#[cfg(unix)]
#[test]
fn a_pinned_state_seals_as_it_is_written_with_its_seed_after_the_public_part() {
    let dir = std::env::temp_dir().join(format!("tessera-pcs-pinned-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    let seed: [u8; 32] = core::array::from_fn(|i| 3 * i as u8 + 1);
    let public = WeightCommitmentPublic {
        layout: WeightLayoutParams {
            log2_k: 8,
            log2_inv_rate: 2,
            mask_coeffs: 64,
            blinding_rows: 32,
        },
        roots: core::array::from_fn(|t| [t as u8 + 1; 32]),
        epoch: 5,
    };
    let state = WeightCommitmentState::with_seed(
        public,
        seed,
        core::array::from_fn(|t| TypeBudget {
            opened_columns: OpenedColumns::new(vec![t as u32, 1000]).expect("increasing"),
            next_blinding_row: t as u32,
        }),
    );
    let path = dir.join("pinned.twc");
    state.write(&path).expect("written");
    let sealed = state.sealed();
    assert_eq!(sealed[..], fs::read(&path).expect("the state"));
    assert_eq!(sealed[SEED_AT..SEED_AT + 32], seed);
    let _ = fs::remove_dir_all(&dir);
}
