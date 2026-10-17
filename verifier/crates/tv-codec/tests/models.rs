// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tiny conformance models of `spec/vectors/models/` (07 §5), as the verifier reads them: each
//! artifact opens as an `IntegerModel` envelope under its limits, its digest checked by the test
//! kit's reference BLAKE3 and equal to the index's. The verifier never decodes the weights; the
//! models are secret-class artifacts it only ever frames (RFC-0003 'Envelope').

use std::fs;

use tv_codec::{DIGEST_CONTEXT, DecodeLimits, Envelope, Expected};
use tv_testkit::{blake3, hex, read_vectors, vectors_dir};

const INTEGER_MODEL: Expected = Expected {
    schema_id: 0x0101,
    version: 1,
};

#[test]
fn every_conformance_model_opens_with_the_index_digest() {
    let index = read_vectors("models/models.json");
    assert_eq!(index.get("schema").and_then(|s| s.as_str()), Some("0x0101"));
    let models = index
        .get("models")
        .and_then(|m| m.as_array())
        .expect("models");
    assert_eq!(models.len(), 3, "every conformance model");
    for entry in models {
        let field = |key: &str| entry.get(key).and_then(|v| v.as_str()).expect(key);
        let path = vectors_dir().join("models").join(field("file"));
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let envelope = Envelope::open(
            &bytes,
            INTEGER_MODEL,
            &DecodeLimits::INTEGER_MODEL,
            |signed| blake3::derive_key(DIGEST_CONTEXT, signed),
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", field("name")));
        let header = envelope.header();
        assert_eq!(header.schema_version, 1);
        assert_eq!(
            header.payload_len,
            u64::try_from(bytes.len() - 52).expect("a length")
        );
        assert_eq!(&bytes[bytes.len() - 32..], hex(field("digest")).as_slice());
        let size = entry.get("bytes").and_then(|b| b.as_u64()).expect("bytes");
        assert_eq!(u64::try_from(bytes.len()).expect("a length"), size);
    }
}
