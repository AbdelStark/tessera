// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One public conformance fixture for crate-private allocation and reachability tests.

use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_compiler::{Lowering, PublicModel, lower, validate};
use tessera_ir::IntegerModel;

pub(crate) fn fixture(positions: u32) -> (IntegerModel, Lowering) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/models/tiny-2-servable.d");
    let bytes = std::fs::read(path).expect("public conformance model");
    let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("canonical conformance model");
    validate(&model).expect("valid conformance model");
    let lowering = lower(PublicModel::of(&model), positions).expect("fixture lowering");
    (model, lowering)
}
