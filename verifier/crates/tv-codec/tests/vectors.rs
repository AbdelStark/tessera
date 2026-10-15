// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-codec` against the shared conformance vectors of `spec/vectors/codec/` (RFC-0003 'Testing
//! Strategy'), which the prover's codec checks with its own decoders: agreement on these is
//! agreement through the specification (SP-6).

mod support;

use support::{Ty, code_named, decode_value, decode_whole, encoded, limits_of, list, same, text};
use tv_codec::{DIGEST_CONTEXT, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected};
use tv_testkit::{Json, blake3, hex, read_vectors};

/// The envelope digest: BLAKE3 `derive_key("tessera/v1/envelope")` over the header and payload,
/// from the test kit's reference BLAKE3.
fn envelope_digest(signed: &[u8]) -> [u8; 32] {
    blake3::derive_key("tessera/v1/envelope", signed)
}

#[test]
fn golden_values_decode_render_and_re_encode() {
    let file = read_vectors("codec/golden.json");
    let limits = limits_of(file.get("limits").expect("limits"));
    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 51, "every golden case");
    for case in cases {
        let name = text(case, "type");
        let ty = Ty::parse(name);
        let bytes = hex(text(case, "hex"));
        let expected = case.get("value").expect("value");

        let value = decode_whole(&ty, &bytes, &limits)
            .unwrap_or_else(|e| panic!("{name} {}: {e}", text(case, "hex")));
        assert!(
            same(&value, expected),
            "{name}: decoded {value:?}, expected {expected:?}"
        );
        // INV-TCE-1 and INV-TCE-2: the value encodes back to exactly these bytes.
        assert_eq!(
            encoded(&ty, expected),
            bytes,
            "{name}: encoding of the value"
        );

        // Every proper prefix is cut short inside a value (RFC-0003 'Testing Strategy').
        for cut in 0..bytes.len() {
            let error =
                decode_whole(&ty, &bytes[..cut], &limits).expect_err("a proper prefix decodes");
            assert_eq!(
                error.code(),
                ErrorCode::Truncated,
                "{name}, prefix of {cut} bytes: {error}"
            );
            assert!(
                error.offset() <= cut as u64,
                "{name}, prefix of {cut}: {error}"
            );
        }
    }
}

#[test]
fn noncanonical_inputs_fail_with_their_code_and_limit_cases_decode() {
    let file = read_vectors("codec/noncanonical.json");
    let limits = limits_of(file.get("limits").expect("limits"));

    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 43, "every rejected case");
    for case in cases {
        let name = text(case, "type");
        let bytes = hex(text(case, "hex"));
        let code = code_named(text(case, "code"));
        match decode_whole(&Ty::parse(name), &bytes, &limits) {
            Ok(value) => panic!("{name} {}: accepted as {value:?}", text(case, "why")),
            Err(error) => {
                assert_eq!(
                    error.code(),
                    code,
                    "{name}: {} gave {error}",
                    text(case, "why")
                );
                assert!(error.offset() <= bytes.len() as u64, "{name}: {error}");
            }
        }
    }

    let accepted = list(&file, "accepted");
    assert_eq!(accepted.len(), 11, "every case at a limit");
    for case in accepted {
        let name = text(case, "type");
        let ty = Ty::parse(name);
        let bytes = hex(text(case, "hex"));
        let value = decode_whole(&ty, &bytes, &limits)
            .unwrap_or_else(|e| panic!("{name} {}: {e}", text(case, "why")));
        assert_eq!(encoded(&ty, &value), bytes, "{name}: re-encoding");
    }
}

/// The limits an envelope case reads under: the file's, or the RFC-0003 defaults it names.
fn case_limits(case: &Json, file_limits: &DecodeLimits) -> DecodeLimits {
    match case
        .get("limits")
        .map(|l| l.as_str().expect("a limits name"))
    {
        None => *file_limits,
        Some("message") => DecodeLimits::MESSAGE,
        Some("integer-model") => DecodeLimits::INTEGER_MODEL,
        Some(other) => panic!("unknown limits {other:?}"),
    }
}

/// Opens `bytes` as an envelope of `expected`, then decodes its payload as `ty`.
fn read_envelope(
    bytes: &[u8],
    expected: Expected,
    limits: &DecodeLimits,
    ty: &Ty,
) -> Result<(Vec<u8>, Json), DecodeError> {
    let envelope = Envelope::open(bytes, expected, limits, envelope_digest)?;
    let value = envelope.decode_payload(|c| decode_value(c, ty))?;
    Ok((envelope.payload().to_vec(), value))
}

#[test]
fn envelopes_give_their_payload_or_the_first_fault_in_rfc_order() {
    assert_eq!(DIGEST_CONTEXT, "tessera/v1/envelope");
    let file = read_vectors("codec/envelope.json");
    let file_limits = limits_of(file.get("limits").expect("limits"));

    let named = file.get("named_limits").expect("named_limits");
    assert_eq!(
        limits_of(named.get("message").expect("message")),
        DecodeLimits::MESSAGE
    );
    assert_eq!(
        limits_of(named.get("integer-model").expect("integer-model")),
        DecodeLimits::INTEGER_MODEL
    );

    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 52, "every envelope case");
    let (mut opened, mut rejected) = (0, 0);
    for case in cases {
        let why = text(case, "why");
        let ty = Ty::parse(text(case, "type"));
        let bytes = hex(text(case, "hex"));
        let field = |key: &str| case.get(key).and_then(Json::as_u64);
        let expected = Expected {
            schema_id: u16::try_from(field("schema").expect("schema")).expect("a u16 schema"),
            version: u16::try_from(field("version").expect("version")).expect("a u16 version"),
        };
        let limits = case_limits(case, &file_limits);

        let outcome = read_envelope(&bytes, expected, &limits, &ty);
        match case.get("payload") {
            Some(payload) => {
                let payload = hex(payload.as_str().expect("payload hex"));
                let (found, value) = outcome.unwrap_or_else(|e| panic!("{why}: {e}"));
                assert_eq!(found, payload, "{why}: payload");
                assert_eq!(encoded(&ty, &value), payload, "{why}: re-encoded payload");
                opened += 1;
            }
            None => {
                let code = code_named(text(case, "code"));
                let offset = field("offset").expect("offset");
                let error = outcome.expect_err(why);
                assert_eq!(
                    (error.code(), error.offset()),
                    (code, offset),
                    "{why}: {error}"
                );
                rejected += 1;
            }
        }
    }
    assert_eq!((opened, rejected), (5, 47));
}

/// The six codec codes, spelled as 04 §3 spells them and in its order, from the file both
/// workspaces check.
#[test]
fn codec_error_identifiers_are_the_specification_s() {
    let file = read_vectors("errors/codes.json");
    let codec: Vec<&str> = list(&file, "codes")
        .iter()
        .map(|entry| text(entry, "code"))
        .filter(|code| code.starts_with("ERR-CODEC-"))
        .collect();
    let ours: Vec<&str> = ErrorCode::ALL.iter().map(|code| code.id()).collect();
    assert_eq!(ours, codec);
}
