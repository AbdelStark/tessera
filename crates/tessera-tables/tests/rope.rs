// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The RoPE constants against `spec/vectors/tables/` (RFC-0008 'Fixed-point generators').
//!
//! Every pair of the test tables and the digest of Llama-3.1's 8K-context table are the oracle's,
//! itself checked against 40-digit references. Sampled pairs of the full 131,072-position table
//! are the oracle's, and 10^6 more lie within one unit of an f64 implementation of Hugging Face's
//! Llama-3 RoPE. Parameters outside RFC-0008's rules are refused, and so is a lookup kind.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tessera_codec::{DecodeLimits, decode};
use tessera_tables::{
    GeneratedTable, RopeConstants, RopeTable, TableError, TableId, TableKind, TableSpec,
};

fn document() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/tables/tables.json");
    serde_json::from_str(&fs::read_to_string(path).expect("the vectors")).expect("JSON")
}

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn number(value: &Value) -> i64 {
    value.as_i64().expect("an integer")
}

/// Llama-3.1's RoPE, `θ = 500000`, `factor 8`, `low 1`, `high 4`, `L = 8192`, 14-bit constants, at
/// `positions` positions of `d_head = 128`.
fn llama(max_context: u32) -> TableKind {
    TableKind::Rope {
        theta: 500_000,
        factor: 8,
        low_freq_factor: 1,
        high_freq_factor: 4,
        original_max: 8192,
        bits: 14,
        max_context,
        d_head: 128,
    }
}

#[test]
fn every_rope_vector_regenerates_with_its_digest_and_pairs() {
    let cases: Vec<Value> = document()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|case| case["kind"].get("Rope").is_some())
        .cloned()
        .collect();
    assert_eq!(cases.len(), 3);
    for case in &cases {
        let name = case["name"].as_str().expect("a name");
        let spec: TableSpec = decode(
            &bytes(case["spec"].as_str().expect("hex")),
            DecodeLimits::MESSAGE,
        )
        .expect("the specification decodes");
        let table = RopeTable::from_spec(&spec).expect(name);
        let frequencies: Vec<u128> = case["frequencies"]
            .as_array()
            .expect("frequencies")
            .iter()
            .map(|f| {
                let hex = f.as_str().expect("hex").trim_start_matches("0x");
                u128::from_str_radix(hex, 16).expect("a Q64 frequency")
            })
            .collect();
        assert_eq!(
            table.constants().frequencies(),
            frequencies,
            "{name}: ω_i in Q64"
        );
        let half = match spec.kind {
            TableKind::Rope { d_head, .. } => u32::from(d_head / 2),
            _ => unreachable!("a Rope case"),
        };
        if let Some(values) = case["values"].as_array() {
            for (j, pair) in (0u32..).zip(values) {
                let expected = (number(&pair[0]), number(&pair[1]));
                let (position, frequency) = (j / half, j % half);
                assert_eq!(
                    table.constants().at(position, frequency),
                    Some(expected),
                    "{name}: ({position}, {frequency})"
                );
            }
        }
        // One field wrong: refused like a lookup table's.
        let mut digest = spec.clone();
        digest.digest[0] ^= 1;
        assert_eq!(RopeTable::from_spec(&digest), Err(TableError::Digest));
        let mut size = spec.clone();
        size.log2_rows += 1;
        assert!(matches!(
            RopeTable::from_spec(&size),
            Err(TableError::Log2Rows { .. })
        ));
    }
}

#[test]
fn sampled_pairs_of_the_full_table_are_the_oracles() {
    let full = &document()["rope_llama31_131072"];
    let constants = RopeConstants::new(&llama(131_072)).expect("Llama-3.1's RoPE");
    let samples = full["samples"].as_array().expect("samples");
    assert_eq!(samples.len(), 4008);
    for sample in samples {
        let position = u32::try_from(number(&sample["position"])).expect("a position");
        let frequency = u32::try_from(number(&sample["frequency"])).expect("a frequency");
        let expected = (
            number(&sample["constants"][0]),
            number(&sample["constants"][1]),
        );
        assert_eq!(
            constants.at(position, frequency),
            Some(expected),
            "({position}, {frequency})"
        );
    }
    assert_eq!(constants.at(131_072, 0), None, "past the positions");
    assert_eq!(constants.at(0, 64), None, "past the frequencies");
}

/// Hugging Face's Llama-3 RoPE in f64 (`_compute_llama3_parameters`), the reference of the 10^6
/// sampled pairs: an f64 angle errs by about `10^-11` rad at position `2^17`, far below the
/// `2^-14` of one unit.
fn reference(position: u32, frequency: u32) -> (f64, f64) {
    let (theta, d, l, factor, low, high) = (500_000f64, 128f64, 8192f64, 8f64, 1f64, 4f64);
    let omega = theta.powf(-2.0 * f64::from(frequency) / d);
    let wavelen = 2.0 * std::f64::consts::PI / omega;
    let scaled = if wavelen < l / high {
        omega
    } else if wavelen > l / low {
        omega / factor
    } else {
        let smooth = (l / wavelen - low) / (high - low);
        (1.0 - smooth) * omega / factor + smooth * omega
    };
    let angle = f64::from(position) * scaled;
    (16384.0 * angle.cos(), 16384.0 * angle.sin())
}

#[test]
fn a_million_sampled_pairs_lie_within_one_unit_of_the_f64_reference() {
    let constants = RopeConstants::new(&llama(131_072)).expect("Llama-3.1's RoPE");
    let mut state = 0x7E55_E7A0_0071_0004u64;
    let mut next = || {
        // SplitMix64.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut worst = 0f64;
    for _ in 0..1_000_000 {
        let draw = next();
        let (position, frequency) = ((draw >> 32) as u32 % 131_072, (draw as u32) % 64);
        let (c, s) = constants.at(position, frequency).expect("inside the table");
        let (rc, rs) = reference(position, frequency);
        let error = (c as f64 - rc).abs().max((s as f64 - rs).abs());
        assert!(
            error <= 1.0,
            "({position}, {frequency}): {c}, {s} against {rc}, {rs}"
        );
        worst = worst.max(error);
    }
    assert!(worst > 0.0, "the comparison ran");
}

#[test]
fn rope_parameters_outside_each_rule_are_refused() {
    let refused = |rule| Err(TableError::Parameter { kind: "Rope", rule });
    let with = |change: fn(&mut TableKind)| {
        let mut kind = llama(16);
        change(&mut kind);
        RopeConstants::new(&kind).map(|_| ())
    };
    macro_rules! set {
        ($field:ident = $value:expr) => {
            |k: &mut TableKind| {
                if let TableKind::Rope { $field, .. } = k {
                    *$field = $value;
                }
            }
        };
    }
    assert_eq!(with(set!(theta = 1)), refused("θ ≥ 2"));
    let factors = "factor ≥ 1 and 1 ≤ low_freq_factor < high_freq_factor";
    assert_eq!(with(set!(factor = 0)), refused(factors));
    assert_eq!(with(set!(low_freq_factor = 0)), refused(factors));
    assert_eq!(with(set!(low_freq_factor = 4)), refused(factors));
    assert_eq!(with(set!(original_max = 0)), refused("original_max ≥ 1"));
    for bits in [0, 31] {
        let mut kind = llama(16);
        if let TableKind::Rope { bits: b, .. } = &mut kind {
            *b = bits;
        }
        assert_eq!(
            RopeConstants::new(&kind).map(|_| ()),
            refused("1 ≤ bits ≤ 30")
        );
    }
    assert_eq!(
        with(set!(d_head = 0)),
        refused("an even d_head of at least 2")
    );
    assert_eq!(
        with(set!(d_head = 127)),
        refused("an even d_head of at least 2")
    );
    let pairs = "1 ≤ max_context and max_context · d_head / 2 ≤ 2^24";
    assert_eq!(with(set!(max_context = 0)), refused(pairs));
    assert_eq!(with(set!(max_context = (1 << 18) + 1)), refused(pairs));
    assert_eq!(with(set!(max_context = 1 << 18)), Ok(()), "2^24 pairs");
    assert_eq!(with(set!(theta = 2)), Ok(()));
    // Each kind has its builder.
    assert_eq!(
        GeneratedTable::new(TableId(0), llama(16)).map(|_| ()),
        Err(TableError::Parameter {
            kind: "Rope",
            rule: "built as a RopeTable: it is no lookup table",
        })
    );
    assert_eq!(
        RopeTable::new(TableId(0), TableKind::AesSbox).map(|_| ()),
        Err(TableError::Parameter {
            kind: "lookup",
            rule: "built as a GeneratedTable: only Rope is a RopeTable",
        })
    );
}
