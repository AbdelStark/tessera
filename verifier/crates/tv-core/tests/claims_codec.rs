// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list's decoding (RFC-0007 'Schema', RFC-0025, RFC-0026, RFC-0027, RFC-0042, RFC-0003)
//! against the golden lists of `spec/vectors/claims/`, written from the compiler's emission:
//! each opens, re-encodes to its own payload (INV-DM-1) and hashes to its vector's
//! `claim_list_digest`. A claim list is the verifier's own input, so one that fails to decode is
//! `Inconclusive(ERR-CODEC-*)`, never a rejection (RFC-0043 'Malformed'): every failure below is
//! checked for its code and offset, and as `load`'s verdict class.

mod claim_support;

use claim_support::{goldens, list, member, payload, seal, seal_as, text};
use tv_codec::{DecodeError, DecodeLimits, ErrorCode};
use tv_core::claims::{ClaimList, Profile, QuantPoint, Site};
use tv_core::validation::{ClaimsError, load};
use tv_hash::{Domain, hash};
use tv_testkit::{SplitMix64, hex, read_vectors};

/// The offset, in the envelope, of a field `at` bytes into the payload.
const fn field(at: u64) -> u64 {
    20 + at
}

/// The decoding error `ClaimList::open` gives for `envelope`.
fn open_error(envelope: &[u8], limits: &DecodeLimits) -> DecodeError {
    match ClaimList::open(envelope, limits) {
        Err(e) => e,
        Ok(_) => panic!("the claim list decoded"),
    }
}

/// `load` must give the same failure as `Inconclusive`, with the decoder's code.
fn assert_inconclusive(envelope: &[u8], limits: &DecodeLimits, expected: DecodeError) {
    assert_eq!(open_error(envelope, limits), expected);
    match load(envelope, limits) {
        Err(error @ ClaimsError::Decode(found)) => {
            assert_eq!(found, expected);
            assert!(!error.is_rejection(), "{error:?} is not a rejection");
            assert_eq!(error.code(), expected.code().id());
        }
        other => panic!("expected Inconclusive({expected}), got {other:?}"),
    }
}

/// The golden list `tiny-2-servable`'s envelope and payload.
fn servable() -> (Vec<u8>, Vec<u8>) {
    let golden = goldens()
        .into_iter()
        .find(|g| g.name == "tiny-2-servable")
        .expect("tiny-2-servable");
    let payload = payload(&golden.envelope).to_vec();
    (golden.envelope, payload)
}

#[test]
fn every_golden_claim_list_opens_re_encodes_and_hashes_to_its_digest() {
    let goldens = goldens();
    assert_eq!(goldens.len(), 4, "every conformance model");
    for golden in &goldens {
        let claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE)
            .unwrap_or_else(|e| panic!("{}: {e}", golden.name));
        let bytes = claims.to_tce();
        assert_eq!(
            bytes,
            payload(&golden.envelope),
            "{}: TCE(ClaimList)",
            golden.name
        );
        assert_eq!(claims.digest(), golden.digest, "{}: digest", golden.name);
        assert_eq!(
            claims.digest(),
            hash(Domain::ClaimList, &bytes),
            "{}: H(\"claim-list\", TCE(ClaimList))",
            golden.name
        );
        // Each list is the model it names: what RFC-0007 'Schema' says each one holds.
        assert_eq!(claims.params.as_str(), "test-small/1", "{}", golden.name);
        assert_eq!(claims.protocol_version, 0, "{}", golden.name);
        let profile = if golden.name.contains("minimal") {
            Profile::Minimal
        } else {
            Profile::Servable
        };
        assert_eq!(claims.profile, profile, "{}", golden.name);
        assert_eq!(
            claims.quant.act_exponents.first().map(|(p, _)| *p),
            Some(QuantPoint::Embed)
        );
        assert_eq!(
            claims.quant.act_exponents.last().map(|(p, _)| *p),
            Some(QuantPoint::FinalNorm)
        );
        assert!(golden.model.ends_with(".d"), "{}", golden.model);
    }
}

#[test]
fn the_affine_list_states_its_affine_sites_and_the_minimal_list_its_profile() {
    let goldens = goldens();
    let open = |name: &str| {
        let golden = goldens.iter().find(|g| g.name == name).expect(name);
        ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).expect(name)
    };
    let affine = open("tiny-2-affine");
    let sites: Vec<(QuantPoint, i8)> = affine.quant.affine_sites.clone();
    assert_eq!(sites.len(), 8);
    assert!(sites.contains(&(
        QuantPoint::Layer {
            layer: 1,
            site: Site::Up
        },
        24
    )));
    let minimal = open("tiny-2-minimal");
    assert!(minimal.quant.act_exponents.iter().all(|(p, _)| !matches!(
        p,
        QuantPoint::Layer {
            site: Site::AttnOut | Site::Up | Site::Product,
            ..
        }
    )));
}

#[test]
fn every_prefix_of_a_golden_envelope_is_cut_short() {
    let (envelope, _) = servable();
    for cut in 0..envelope.len() {
        let error = open_error(&envelope[..cut], &DecodeLimits::MESSAGE);
        assert_eq!(error.code(), ErrorCode::Truncated, "cut at {cut}");
        // RFC-0003 'Envelope': a header field cut short at its offset, then the rest at 12.
        let expected = match cut {
            0..4 => 0,
            4..6 => 4,
            6..8 => 6,
            8..10 => 8,
            10..12 => 10,
            _ => 12,
        };
        assert_eq!(error.offset(), expected, "cut at {cut}");
    }
}

#[test]
fn every_prefix_of_a_golden_payload_resealed_is_cut_short_where_its_value_starts() {
    let (_, payload) = servable();
    for cut in 0..payload.len() {
        let envelope = seal(&payload[..cut]);
        let error = open_error(&envelope, &DecodeLimits::MESSAGE);
        assert_eq!(error.code(), ErrorCode::Truncated, "payload cut at {cut}");
        let cut = u64::try_from(cut).expect("a length");
        assert!(
            error.offset() <= field(cut),
            "payload cut at {cut}: offset {}",
            error.offset()
        );
    }
}

#[test]
fn a_payload_cut_short_fails_at_the_value_it_cuts() {
    let (_, payload) = servable();
    let len = payload.len();
    // In the protocol version, at the payload's first byte.
    let cut = |at: usize| open_error(&seal(&payload[..at]), &DecodeLimits::MESSAGE);
    assert_eq!(cut(1), DecodeError::new(ErrorCode::Truncated, field(0)));
    // In `n_layers`, the architecture's second field, after its family's tag.
    assert_eq!(cut(6), DecodeError::new(ErrorCode::Truncated, field(4)));
    // In the profile's tag.
    assert_eq!(cut(68), DecodeError::new(ErrorCode::Truncated, field(67)));
    // Inside the parameter set's name: its length promises more than is left, at the length.
    let name_at = u64::try_from(len - 8 - 12).expect("a length");
    assert_eq!(
        cut(len - 1),
        DecodeError::new(ErrorCode::Truncated, field(name_at))
    );
}

#[test]
fn a_non_canonical_payload_fails_at_its_value() {
    let (_, payload) = servable();
    let len = payload.len();
    let with = |at: usize, bytes: &[u8]| {
        let mut changed = payload.clone();
        changed[at..at + bytes.len()].copy_from_slice(bytes);
        seal(&changed)
    };
    let limits = DecodeLimits::MESSAGE;
    let noncanonical = |at: usize| DecodeError::new(ErrorCode::Noncanonical, field(at as u64));
    // `ArchFamily` has one variant, `LlamaV3`.
    assert_inconclusive(&with(2, &1u16.to_le_bytes()), &limits, noncanonical(2));
    // `rms_eps`, `1/100000`, written as `2/200000`, which is not in lowest terms.
    let mut rational = 2i64.to_le_bytes().to_vec();
    rational.extend_from_slice(&200_000u64.to_le_bytes());
    assert_inconclusive(&with(51, &rational), &limits, noncanonical(51));
    // A zero denominator.
    assert_inconclusive(&with(59, &0u64.to_le_bytes()), &limits, noncanonical(51));
    // `Profile` has two variants.
    assert_inconclusive(&with(67, &2u16.to_le_bytes()), &limits, noncanonical(67));
    // `QuantPoint` has three variants: the first exponent's point.
    assert_inconclusive(&with(77, &3u16.to_le_bytes()), &limits, noncanonical(77));
    // `Site` has eleven: the second exponent's point is `(0, AttnNorm)`, its site at 86.
    assert_inconclusive(&with(86, &11u16.to_le_bytes()), &limits, noncanonical(86));
    // A byte of the parameter set's name outside printable ASCII.
    assert_inconclusive(&with(len - 12, &[0x7f]), &limits, noncanonical(len - 12));
    // A byte after the claim list (INV-TCE-4).
    let mut longer = payload.clone();
    longer.push(0);
    assert_inconclusive(&seal(&longer), &limits, noncanonical(len));
}

#[test]
fn a_table_specification_that_is_not_canonical_fails_at_its_value() {
    let (_, payload) = servable();
    // The first table follows `silu_in_exponents`: find its tag by re-encoding the prefix.
    let claims = ClaimList::open(&seal(&payload), &DecodeLimits::MESSAGE).expect("golden");
    let tables_at = payload.len()
        - claims
            .tables
            .iter()
            .map(|t| t.to_tce().len())
            .sum::<usize>()
        - (1 + 4 + 1) // gadget policy
        - (1 + 1 + 4 + 4 + 4) // advice layout
        - (1 + 1 + 4 + 4) // weights layout
        - (8 + 12); // the parameter set
    // The first table is `Rope`: its id, then its kind's tag, which names no kind at 10.
    let kind_at = tables_at + 4;
    assert_eq!(
        &payload[tables_at - 8..tables_at],
        &u64::try_from(claims.tables.len()).unwrap().to_le_bytes()
    );
    let mut changed = payload.clone();
    changed[kind_at..kind_at + 2].copy_from_slice(&10u16.to_le_bytes());
    assert_inconclusive(
        &seal(&changed),
        &DecodeLimits::MESSAGE,
        DecodeError::new(ErrorCode::Noncanonical, field(kind_at as u64)),
    );
}

#[test]
fn an_envelope_that_is_not_a_claim_list_fails_in_its_header() {
    let (envelope, payload) = servable();
    let limits = DecodeLimits::MESSAGE;
    // Another schema: `IntegerModel`.
    assert_inconclusive(
        &seal_as(0x0101, 1, &payload),
        &limits,
        DecodeError::new(ErrorCode::Magic, 6),
    );
    // Schema versions 0 and 2: a reader of version 1 accepts version 1 only (RFC-0003).
    for version in [0, 2] {
        assert_inconclusive(
            &seal_as(0x0201, version, &payload),
            &limits,
            DecodeError::new(ErrorCode::Version, 8),
        );
    }
    // A digest that is not the envelope's.
    let mut tampered = envelope.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    let digest_at = field(u64::try_from(payload.len()).unwrap());
    assert_inconclusive(
        &tampered,
        &limits,
        DecodeError::new(ErrorCode::Digest, digest_at),
    );
    // A payload byte changed without resealing fails the digest too, before any decoding.
    let mut changed = envelope.clone();
    changed[87] = 2;
    assert_inconclusive(
        &changed,
        &limits,
        DecodeError::new(ErrorCode::Digest, digest_at),
    );
}

#[test]
fn the_codec_vectors_claim_list_envelopes_fail_as_they_say() {
    let file = read_vectors("codec/envelope.json");
    let mut seen = 0;
    for case in list(&file, "cases") {
        if member(case, "schema").as_u64() != Some(0x0201) {
            continue;
        }
        let envelope = hex(text(case, "hex"));
        let code = text(case, "code");
        let offset = member(case, "offset").as_u64().expect("an offset");
        let error = open_error(&envelope, &DecodeLimits::MESSAGE);
        assert_eq!(error.code().id(), code, "{}", text(case, "why"));
        assert_eq!(error.offset(), offset, "{}", text(case, "why"));
        seen += 1;
    }
    assert_eq!(seen, 2, "the version cases of schema 0x0201");
}

#[test]
fn the_decode_limits_bound_the_claim_list() {
    let (envelope, _) = servable();
    let len = u64::try_from(envelope.len()).unwrap();
    // The whole envelope: one byte over the limit is refused at its length.
    let tight = DecodeLimits {
        max_total_bytes: len - 1,
        ..DecodeLimits::MESSAGE
    };
    assert_inconclusive(&envelope, &tight, DecodeError::new(ErrorCode::Limit, 12));
    let exact = DecodeLimits {
        max_total_bytes: len,
        ..DecodeLimits::MESSAGE
    };
    assert!(ClaimList::open(&envelope, &exact).is_ok());
    // A sequence longer than `max_vec_len`: `act_exponents`, the first sequence, of 20 points.
    let short = DecodeLimits {
        max_vec_len: 19,
        ..DecodeLimits::MESSAGE
    };
    assert_inconclusive(
        &envelope,
        &short,
        DecodeError::new(ErrorCode::Limit, field(69)),
    );
    // Depth: the claim list is level 1, its architecture 2 and the family's enum 3; the first
    // exponent's tuple is level 4, its point 5, and the second point's site 6.
    let depth = |max_depth| DecodeLimits {
        max_depth,
        ..DecodeLimits::MESSAGE
    };
    assert_inconclusive(
        &envelope,
        &depth(2),
        DecodeError::new(ErrorCode::Limit, field(2)),
    );
    assert_inconclusive(
        &envelope,
        &depth(3),
        DecodeError::new(ErrorCode::Limit, field(77)),
    );
    assert_inconclusive(
        &envelope,
        &depth(5),
        DecodeError::new(ErrorCode::Limit, field(86)),
    );
    assert!(
        ClaimList::open(&envelope, &depth(6)).is_ok(),
        "level 6 is the deepest"
    );
}

#[test]
fn seeded_changes_of_a_golden_payload_never_panic() {
    // Each change is resealed so that the payload is decoded, then validated: every input
    // yields a value or a typed error (RFC-0019 'Robustness').
    let (_, payload) = servable();
    let mut rng = SplitMix64::new(0x0107_c0de);
    let (mut decoded, mut failed) = (0, 0);
    for _ in 0..600 {
        let mut changed = payload.clone();
        for _ in 0..1 + rng.next_u64() % 3 {
            let at = usize::try_from(rng.next_u64() % (changed.len() as u64)).unwrap();
            changed[at] = u8::try_from(rng.next_u64() % 256).unwrap();
        }
        if rng.next_u64().is_multiple_of(8) {
            let cut = usize::try_from(rng.next_u64() % (changed.len() as u64)).unwrap();
            changed.truncate(cut);
        }
        match load(&seal(&changed), &DecodeLimits::MESSAGE) {
            Err(ClaimsError::Decode(_)) => failed += 1,
            _ => decoded += 1,
        }
    }
    assert!(
        decoded > 50 && failed > 50,
        "{decoded} decoded, {failed} failed"
    );
}
