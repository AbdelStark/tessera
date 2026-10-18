// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `WeightCommitmentPublic` (`0x0301`, RFC-0040) as the verifier reads it, against the shared
//! vectors: the golden envelopes of `spec/vectors/codec/weight-commitment.json`, and each
//! model's `.twr` artifact in `spec/vectors/weights/`, which an independent oracle wrote. The
//! verifier decodes each with its own decoder and recomputes `R_W` with the test kit's reference
//! BLAKE3; it never sees a weight, and refuses the secret state (`0x0302`).

use tv_codec::{
    DIGEST_CONTEXT, DecodeError, DecodeLimits, Envelope, ErrorCode, WEIGHTS_ROOT_CONTEXT,
    WeightCommitmentPublic, WeightLayoutParams, decode,
};
use tv_testkit::{Json, blake3, hex, read_vectors, vectors_dir};

fn text<'a>(json: &'a Json, key: &str) -> &'a str {
    json.get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("{key}"))
}

fn list<'a>(json: &'a Json, key: &str) -> &'a [Json] {
    json.get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("{key}"))
}

/// A decimal string of the codec vectors' values.
fn decimal(json: &Json) -> u64 {
    json.as_str()
        .and_then(|s| s.parse().ok())
        .expect("a decimal string")
}

fn open(bytes: &[u8]) -> Result<WeightCommitmentPublic, DecodeError> {
    let envelope = Envelope::open(
        bytes,
        WeightCommitmentPublic::EXPECTED,
        &DecodeLimits::MESSAGE,
        |signed| blake3::derive_key(DIGEST_CONTEXT, signed),
    )?;
    envelope.decode_payload(|c, _| WeightCommitmentPublic::decode(c))
}

fn weights_root(public: &WeightCommitmentPublic) -> [u8; 32] {
    public.weights_root(|bytes| blake3::derive_key(WEIGHTS_ROOT_CONTEXT, bytes))
}

#[test]
fn the_golden_public_artifacts_decode_to_their_values_and_weights_roots() {
    assert_eq!(WEIGHTS_ROOT_CONTEXT, "tessera/v1/weights-root");
    let file = read_vectors("codec/weight-commitment.json");
    assert_eq!(text(&file, "limits"), "message");
    let cases = list(&file, "cases");
    let mut publics = 0;
    for case in cases {
        let bytes = hex(text(case, "hex"));
        if text(case, "type") != "WeightCommitmentPublic" {
            // The verifier never reads the secret state: its reader refuses the schema.
            assert_eq!(
                open(&bytes).map(|_| ()),
                Err(DecodeError::new(ErrorCode::Magic, 6)),
                "{}",
                text(case, "name")
            );
            continue;
        }
        let public = open(&bytes).unwrap_or_else(|e| panic!("{}: {e}", text(case, "name")));
        let value = case.get("value").expect("value");
        let layout = value.get("layout").expect("layout");
        let field = |key: &str| decimal(layout.get(key).expect(key));
        assert_eq!(
            public.layout,
            WeightLayoutParams {
                log2_k: u8::try_from(field("log2_k")).expect("a u8"),
                log2_inv_rate: u8::try_from(field("log2_inv_rate")).expect("a u8"),
                mask_coeffs: u32::try_from(field("mask_coeffs")).expect("a u32"),
                blinding_rows: u32::try_from(field("blinding_rows")).expect("a u32"),
            }
        );
        let roots = list(value, "roots");
        assert_eq!(roots.len(), 10);
        for (root, expected) in public.roots.iter().zip(roots) {
            let expected: Vec<u8> = expected
                .as_array()
                .expect("a root")
                .iter()
                .map(|b| u8::try_from(decimal(b)).expect("a byte"))
                .collect();
            assert_eq!(root.as_slice(), expected.as_slice());
        }
        assert_eq!(
            u64::from(public.epoch),
            decimal(value.get("epoch").expect("epoch"))
        );
        // The value encodes back to the payload, and hashes to the file's R_W.
        assert_eq!(public.encode(), bytes[20..bytes.len() - 32].to_vec());
        assert_eq!(
            weights_root(&public).to_vec(),
            hex(text(case, "weights_root"))
        );
        publics += 1;
    }
    assert_eq!(publics, 2, "every public case");
}

#[test]
fn every_models_twr_artifact_gives_its_roots_and_weights_root() {
    let file = read_vectors("weights/weights.json");
    let models = list(&file, "models");
    assert_eq!(models.len(), 5);
    let layout = file.get("layout").expect("layout");
    let field = |key: &str| layout.get(key).and_then(Json::as_u64).expect(key);
    for model in models {
        let name = text(model, "name");
        let path = vectors_dir().join("weights").join(text(model, "file"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            &bytes[bytes.len() - 32..],
            hex(text(model, "digest")).as_slice()
        );
        let public = open(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (
                u64::from(public.layout.log2_k),
                u64::from(public.layout.log2_inv_rate),
                u64::from(public.layout.mask_coeffs),
                u64::from(public.layout.blinding_rows),
            ),
            (
                field("log2_k"),
                field("log2_inv_rate"),
                field("mask_coeffs"),
                field("blinding_rows")
            ),
            "{name}"
        );
        assert_eq!(public.epoch, 0, "{name}: the first epoch");
        let matrices = list(model, "matrices");
        assert_eq!(matrices.len(), 10);
        for (root, matrix) in public.roots.iter().zip(matrices) {
            assert_eq!(
                root.as_slice(),
                hex(text(matrix, "root")).as_slice(),
                "{name} {}",
                text(matrix, "matrix")
            );
        }
        assert_eq!(public.encode(), hex(text(model, "public")), "{name}");
        assert_eq!(
            weights_root(&public).to_vec(),
            hex(text(model, "weights_root")),
            "{name}: R_W"
        );
    }
}

#[test]
fn a_payload_cut_short_or_too_long_or_too_deep_is_refused_where_rfc_0003_says() {
    let file = read_vectors("weights/weights.json");
    let payload = hex(text(&list(&file, "models")[0], "public"));
    assert_eq!(payload.len(), 334);
    let read = |bytes: &[u8], limits: &DecodeLimits| {
        decode(bytes, limits, WeightCommitmentPublic::decode).map(|_| ())
    };
    assert_eq!(read(&payload, &DecodeLimits::MESSAGE), Ok(()));
    // Each prefix is cut short at the field it ends in: log2_k (0), log2_inv_rate (1),
    // mask_coeffs (2), blinding_rows (6), the roots, checked as a whole (10), the epoch (330).
    for cut in 0..payload.len() {
        let at = match cut {
            0 => 0,
            1 => 1,
            2..6 => 2,
            6..10 => 6,
            10..330 => 10,
            _ => 330,
        };
        assert_eq!(
            read(&payload[..cut], &DecodeLimits::MESSAGE),
            Err(DecodeError::new(ErrorCode::Truncated, at)),
            "a prefix of {cut} bytes"
        );
    }
    // A byte past the epoch is left over.
    let mut longer = payload.clone();
    longer.push(0);
    assert_eq!(
        read(&longer, &DecodeLimits::MESSAGE),
        Err(DecodeError::new(ErrorCode::Noncanonical, 334))
    );
    // Depth: the struct is one level, the layout and the roots two, each digest three.
    let depth = |max_depth| DecodeLimits {
        max_depth,
        ..DecodeLimits::MESSAGE
    };
    assert_eq!(read(&payload, &depth(3)), Ok(()));
    assert_eq!(
        read(&payload, &depth(2)),
        Err(DecodeError::new(ErrorCode::Limit, 10))
    );
    assert_eq!(
        read(&payload, &depth(1)),
        Err(DecodeError::new(ErrorCode::Limit, 0))
    );
    assert_eq!(
        read(&payload, &depth(0)),
        Err(DecodeError::new(ErrorCode::Limit, 0))
    );
}

#[test]
fn the_weights_root_changes_with_every_byte_of_the_artifact() {
    // INV-WC-3: R_W binds the layout, every root and the epoch.
    let file = read_vectors("weights/weights.json");
    let payload = hex(text(&list(&file, "models")[3], "public"));
    let public = decode(
        &payload,
        &DecodeLimits::MESSAGE,
        WeightCommitmentPublic::decode,
    )
    .expect("ok");
    let honest = weights_root(&public);
    for at in 0..payload.len() {
        let mut changed = payload.clone();
        changed[at] ^= 0x10;
        let other = decode(
            &changed,
            &DecodeLimits::MESSAGE,
            WeightCommitmentPublic::decode,
        )
        .expect("ok");
        assert_ne!(weights_root(&other), honest, "byte {at}");
    }
}
