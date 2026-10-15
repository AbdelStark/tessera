// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hostile inputs (RFC-0019 'Robustness'): seeded random bytes, and seeded mutations of every
//! input of `spec/vectors/codec/`, through the decoder of every type the vectors use and through
//! the envelope, under the vectors' limits.
//!
//! A rejection must be a fault of the input, pointing inside it; a panic, or a fault of the
//! reader, is reported with its input. An accepted input is judged by the test support's oracles,
//! written from RFC-0003 without `tv-codec`: it must be the canonical encoding of the value it
//! decoded to (INV-TCE-2), the value must be within the limits, and each proper prefix of the
//! input must be cut short. A decoder that accepts a second encoding of a value, or a value past
//! a limit, fails here.

mod support;

use std::panic::{AssertUnwindSafe, catch_unwind};

use support::{
    Ty, accepted_soundly, decode_value, decode_whole, limits_of, list, text, truncated_at,
};
use tv_codec::{DecodeError, DecodeLimits, Envelope, EnvelopeHeader, ErrorCode, Expected};
use tv_testkit::{Json, SplitMix64, blake3, hex, read_vectors};

/// Bytes that sit on a boundary of some check: `bool`, tags, `Ascii`, signs.
const EDGE_BYTES: [u8; 10] = [0x00, 0x01, 0x02, 0x03, 0x1f, 0x20, 0x7e, 0x7f, 0x80, 0xff];

/// `u64` values that sit on a boundary of some check: lengths at and past the vectors' limits,
/// the field modulus, and the extremes.
const EDGE_WORDS: [u64; 14] = [
    0,
    1,
    2,
    16,
    17,
    256,
    257,
    65_536,
    1 << 34,
    (1 << 34) + 1,
    0xFFFF_FFFF_0000_0000,
    0xFFFF_FFFF_0000_0001,
    1 << 63,
    u64::MAX,
];

struct Rng(SplitMix64);

impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        (self.0.next_u64() % bound as u64) as usize
    }

    fn pick<'t, T>(&mut self, items: &'t [T]) -> &'t T {
        &items[self.below(items.len())]
    }

    fn byte(&mut self) -> u8 {
        match self.below(4) {
            0 => *self.pick(&EDGE_BYTES),
            1 => self.below(4) as u8,
            _ => self.0.next_u64() as u8,
        }
    }

    /// Random bytes, biased towards small values so that lengths and tags are often plausible.
    fn bytes(&mut self, max_len: usize) -> Vec<u8> {
        let len = self.below(max_len + 1);
        (0..len).map(|_| self.byte()).collect()
    }

    /// Applies one to four mutations, some of them taking bytes from `donor`.
    fn mutate(&mut self, input: &mut Vec<u8>, donor: &[u8]) {
        for _ in 0..1 + self.below(4) {
            let at = self.below(input.len() + 1);
            match self.below(9) {
                0 if at < input.len() => input[at] ^= 1 << self.below(8),
                1 if at < input.len() => input[at] = *self.pick(&EDGE_BYTES),
                2 if at < input.len() => input[at] = self.0.next_u64() as u8,
                3 => input.insert(at, self.byte()),
                4 if at < input.len() => {
                    input.remove(at);
                }
                5 => input.truncate(at),
                6 => {
                    let word = self.pick(&EDGE_WORDS).to_le_bytes();
                    let end = (at + 8).min(input.len());
                    input.splice(at..end, word);
                }
                7 => {
                    let half = (self.below(5) as u16).wrapping_sub(1).to_le_bytes();
                    let end = (at + 2).min(input.len());
                    input.splice(at..end, half);
                }
                _ if !donor.is_empty() => {
                    let from = self.below(donor.len());
                    let len = self.below(donor.len() - from + 1);
                    let end = (at + self.below(len + 1)).min(input.len());
                    input.splice(at..end, donor[from..from + len].iter().copied());
                }
                _ => {}
            }
        }
    }
}

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every type and input of the vector files, with the limits each file reads under.
struct Corpus {
    types: Vec<(String, Ty)>,
    limits: Vec<DecodeLimits>,
    /// (type index, limits index, bytes)
    inputs: Vec<(usize, usize, Vec<u8>)>,
    envelopes: Vec<EnvelopeCase>,
}

struct EnvelopeCase {
    bytes: Vec<u8>,
    expected: Expected,
    limits: DecodeLimits,
    ty: Ty,
}

impl Corpus {
    fn load() -> Self {
        let mut corpus = Corpus {
            types: Vec::new(),
            limits: Vec::new(),
            inputs: Vec::new(),
            envelopes: Vec::new(),
        };
        for (file, keys) in [
            ("codec/golden.json", &["cases"][..]),
            ("codec/noncanonical.json", &["cases", "accepted"][..]),
            ("codec/envelope.json", &["cases"][..]),
        ] {
            let json = read_vectors(file);
            let limits = limits_of(json.get("limits").expect("limits"));
            corpus.limits.push(limits);
            let limits_index = corpus.limits.len() - 1;
            for key in keys {
                for case in list(&json, key) {
                    let ty = corpus.type_index(text(case, "type"));
                    let bytes = hex(text(case, "hex"));
                    if file.ends_with("envelope.json") {
                        corpus.envelopes.push(envelope_case(case, &limits, bytes));
                    } else {
                        corpus.inputs.push((ty, limits_index, bytes));
                    }
                }
            }
        }
        corpus.limits.push(DecodeLimits::MESSAGE);
        corpus
    }

    fn type_index(&mut self, name: &str) -> usize {
        if let Some(index) = self.types.iter().position(|(n, _)| n == name) {
            return index;
        }
        self.types.push((name.to_owned(), Ty::parse(name)));
        self.types.len() - 1
    }
}

fn envelope_case(case: &Json, file_limits: &DecodeLimits, bytes: Vec<u8>) -> EnvelopeCase {
    let field =
        |key: &str| u16::try_from(case.get(key).and_then(Json::as_u64).expect(key)).expect("a u16");
    let limits = match case.get("limits").and_then(Json::as_str) {
        Some("message") => DecodeLimits::MESSAGE,
        Some("integer-model") => DecodeLimits::INTEGER_MODEL,
        _ => *file_limits,
    };
    EnvelopeCase {
        bytes,
        expected: Expected {
            schema_id: field("schema"),
            version: field("version"),
        },
        limits,
        ty: Ty::parse(text(case, "type")),
    }
}

#[derive(Default)]
struct Tally {
    inputs: usize,
    accepted: usize,
}

/// Decodes `input` as `ty`, failing the test on a panic or on a broken invariant.
fn check(name: &str, ty: &Ty, input: &[u8], limits: &DecodeLimits, tally: &mut Tally) {
    tally.inputs += 1;
    let outcome = catch_unwind(AssertUnwindSafe(|| decode_whole(ty, input, limits)));
    let Ok(outcome) = outcome else {
        panic!(
            "decoding {name} under {limits:?} panicked on {}",
            hex_of(input)
        );
    };
    match outcome {
        Err(error) => assert!(
            error.code().is_input_fault() && error.offset() <= input.len() as u64,
            "{name}: {error} on {}",
            hex_of(input)
        ),
        Ok(value) => {
            tally.accepted += 1;
            if let Err(why) = accepted_soundly(ty, &value, input, limits) {
                panic!(
                    "{name} under {limits:?} accepted {} as {value:?}: {why}",
                    hex_of(input)
                );
            }
            for cut in 0..input.len() {
                let error = decode_whole(ty, &input[..cut], limits)
                    .expect_err("a proper prefix of an encoding decodes");
                assert_eq!(
                    (error.code(), error.offset()),
                    (ErrorCode::Truncated, truncated_at(ty, &value, cut as u64)),
                    "{name}, the first {cut} bytes of {}",
                    hex_of(input)
                );
            }
        }
    }
}

#[test]
fn random_and_mutated_inputs_never_panic_and_accepted_ones_are_canonical() {
    let corpus = Corpus::load();
    assert_eq!(corpus.types.len(), 45, "every type the vectors use");
    assert_eq!(corpus.inputs.len(), 105, "every non-envelope vector input");
    let mut rng = Rng(SplitMix64::new(0x7465_7373_6572_6131));
    let (mut random, mut mutated) = (Tally::default(), Tally::default());

    for (name, ty) in &corpus.types {
        for _ in 0..10_000 {
            let input = rng.bytes(96);
            let limits = *rng.pick(&corpus.limits);
            check(name, ty, &input, &limits, &mut random);
        }
    }

    for (ty_index, limits_index, original) in &corpus.inputs {
        let (name, ty) = &corpus.types[*ty_index];
        for _ in 0..3_000 {
            let donor = &rng.pick(&corpus.inputs).2;
            let mut input = original.clone();
            rng.mutate(&mut input, donor);
            check(
                name,
                ty,
                &input,
                &corpus.limits[*limits_index],
                &mut mutated,
            );
            let (other_name, other) = rng.pick(&corpus.types);
            let limits = *rng.pick(&corpus.limits);
            check(other_name, other, &input, &limits, &mut mutated);
        }
    }

    assert_eq!((random.inputs, mutated.inputs), (450_000, 630_000));
    // The generators reach the paths past every check, not only the early rejections.
    assert!(
        random.accepted > 1_000,
        "random inputs accepted: {}",
        random.accepted
    );
    assert!(
        mutated.accepted > 20_000,
        "mutations accepted: {}",
        mutated.accepted
    );
}

/// Opens `bytes` and decodes its payload as `ty`.
fn read_envelope(
    case: &EnvelopeCase,
    bytes: &[u8],
) -> Result<(EnvelopeHeader, Vec<u8>, Json), DecodeError> {
    let envelope = Envelope::open(bytes, case.expected, &case.limits, |signed| {
        blake3::derive_key("tessera/v1/envelope", signed)
    })?;
    let value = envelope.decode_payload(|c, _version| decode_value(c, &case.ty))?;
    Ok((envelope.header(), envelope.payload().to_vec(), value))
}

/// The envelope a writer of schema version `version` emits for `payload`, from RFC-0003
/// 'Envelope': the header, the payload and the digest of both.
fn sealed(schema_id: u16, version: u16, payload: &[u8]) -> Vec<u8> {
    let mut bytes = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &schema_id.to_le_bytes(),
        &version.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    bytes.extend(blake3::derive_key("tessera/v1/envelope", &bytes));
    bytes
}

/// Makes a mutated envelope's framing and digest sound again, so that its header checks and
/// its payload decoder are reached: the declared length matches the input, then the digest is
/// recomputed.
fn reseal(bytes: &mut [u8], fix_length: bool) {
    if bytes.len() < 52 {
        return;
    }
    if fix_length {
        let payload_len = (bytes.len() - 52) as u64;
        bytes[12..20].copy_from_slice(&payload_len.to_le_bytes());
    }
    let signed = bytes.len() - 32;
    let digest = blake3::derive_key("tessera/v1/envelope", &bytes[..signed]);
    bytes[signed..].copy_from_slice(&digest);
}

#[test]
fn mutated_envelopes_never_panic_and_opened_ones_are_canonical() {
    let corpus = Corpus::load();
    assert_eq!(corpus.envelopes.len(), 52, "every envelope vector");
    let mut rng = Rng(SplitMix64::new(0x656e_7665_6c6f_7065));
    let (mut tally, mut decoded_payloads) = (Tally::default(), 0);

    for case in &corpus.envelopes {
        for round in 0..10_000 {
            let donor = &rng.pick(&corpus.envelopes).bytes;
            let mut bytes = case.bytes.clone();
            rng.mutate(&mut bytes, donor);
            match round % 3 {
                0 => {}
                1 => reseal(&mut bytes, false),
                _ => reseal(&mut bytes, true),
            }
            tally.inputs += 1;
            let outcome = catch_unwind(AssertUnwindSafe(|| read_envelope(case, &bytes)));
            let Ok(outcome) = outcome else {
                panic!(
                    "an envelope of {:?} panicked on {}",
                    case.ty,
                    hex_of(&bytes)
                );
            };
            match outcome {
                Err(error) => {
                    assert!(
                        error.code().is_input_fault() && error.offset() <= bytes.len() as u64,
                        "{error} on {}",
                        hex_of(&bytes)
                    );
                    // An envelope that opens failed in its payload, and the error points into
                    // the payload: from its first byte, 20, to its end.
                    let opened = Envelope::open(&bytes, case.expected, &case.limits, |s| {
                        blake3::derive_key("tessera/v1/envelope", s)
                    });
                    if let Ok(envelope) = opened {
                        let end = 20 + envelope.header().payload_len;
                        assert!(
                            (20..=end).contains(&error.offset()),
                            "{error} outside the payload of {}",
                            hex_of(&bytes)
                        );
                        decoded_payloads += 1;
                    }
                }
                Ok((header, payload, value)) => {
                    tally.accepted += 1;
                    // A reader accepts the version its writers emit and the one before, never 0;
                    // what it opens is byte for byte the envelope such a writer emits.
                    let (found, v) = (header.schema_version, case.expected.version);
                    assert!(
                        found != 0 && (found == v || found.checked_add(1) == Some(v)),
                        "version {found} opened by a reader of version {v}: {}",
                        hex_of(&bytes)
                    );
                    assert_eq!(
                        sealed(case.expected.schema_id, found, &payload),
                        bytes,
                        "not the envelope a writer emits: {}",
                        hex_of(&bytes)
                    );
                    if let Err(why) = accepted_soundly(&case.ty, &value, &payload, &case.limits) {
                        panic!("payload of {} as {value:?}: {why}", hex_of(&bytes));
                    }
                    assert!(bytes.len() as u64 <= case.limits.max_total_bytes);
                }
            }
        }
    }

    assert_eq!(tally.inputs, 520_000);
    assert!(
        tally.accepted > 5_000,
        "envelopes opened: {}",
        tally.accepted
    );
    assert!(
        decoded_payloads > 5_000,
        "payload rejections: {decoded_payloads}"
    );
}
