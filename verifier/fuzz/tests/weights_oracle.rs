// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fuzz targets' `WeightCommitmentPublic` (RFC-0040, schema `0x0301`) on the artifacts
//! `seeds.py` seeds them with, so that `cargo test` shows those seeds reach an accepted payload:
//! each `.twr` of `spec/vectors/weights/` opens under the `envelope` target's `0x0301` reader, and
//! the echo of its payload, encoded from RFC-0040's field list, is the payload, three levels deep
//! and with no sequence.

use std::fs;
use std::path::Path;

use tv_codec::{DecodeLimits, Envelope, Expected};
use tv_fuzz::{TYPES, decode_as};
use tv_testkit::blake3;

#[test]
fn every_weight_artifact_opens_and_its_payload_echoes_itself() {
    let selector = TYPES
        .iter()
        .position(|&name| name == "WeightCommitmentPublic")
        .expect("the type");
    let reader = Expected {
        schema_id: 0x0301,
        version: 1,
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/weights");
    let mut artifacts = 0;
    for entry in fs::read_dir(&dir).expect("spec/vectors/weights") {
        let path = entry.expect("an entry").path();
        if path.extension().is_none_or(|extension| extension != "twr") {
            continue;
        }
        let bytes = fs::read(&path).expect("an artifact");
        let envelope = Envelope::open(&bytes, reader, &DecodeLimits::MESSAGE, |signed| {
            blake3::derive_key("tessera/v1/envelope", signed)
        })
        .expect("an envelope of schema 0x0301");
        let echo = envelope
            .decode_payload(|c, _version| decode_as(selector, c))
            .expect("a WeightCommitmentPublic");
        assert_eq!(echo.bytes, envelope.payload(), "{}", path.display());
        assert_eq!((echo.depth, echo.longest), (3, 0), "{}", path.display());
        artifacts += 1;
    }
    assert_eq!(artifacts, 5, "the .twr artifacts of spec/vectors/weights/");
}
