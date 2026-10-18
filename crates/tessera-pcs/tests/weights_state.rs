// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment's artifacts (RFC-0040 'WeightCommitmentPublic', 'WeightCommitmentState'):
//!
//! - the state has no TCE of its own, and its seed field is private: no code outside its file
//!   can encode it or copy the seed out (checked at compile time here, and by the `compile_fail`
//!   examples of its documentation); release builds cannot enable `test-rng`, the only feature
//!   that gives another crate a state with a chosen seed or its envelope in memory (`cargo xtask
//!   check-test-features`);
//! - a new state holds a seed from the operating system's CSPRNG, no opened column and no
//!   burned blinding row;
//! - the state is written with mode 0600 by atomic rename, the public part with the default
//!   mode; each reads back, and a state read back writes the same bytes, seed included;
//! - reading a state checks its file in the order of RFC-0003 'Envelope';
//! - `opened_columns` is a set: an unsorted or repeated index is `ERR-CODEC-NONCANONICAL` at its
//!   first byte, and cannot be built;
//! - the codec vectors of `spec/vectors/codec/weight-commitment.json` decode to their values and
//!   write back to their bytes. The crate's own tests (`src/weights/state/vectors.rs`) seal each
//!   state's value, pinned to its seed, to its bytes too.

#![cfg(unix)]

use std::fs;
use std::marker::PhantomData;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::envelope::{self, ReadError};
use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, DecodeError, DecodeLimits, Encode, encode, schema};
use tessera_pcs::{
    OpenedColumns, PcsError, TypeBudget, WeightCommitmentPublic, WeightCommitmentState,
    WeightLayoutParams,
};

fn public(epoch: u32) -> WeightCommitmentPublic {
    WeightCommitmentPublic {
        layout: WeightLayoutParams {
            log2_k: 8,
            log2_inv_rate: 2,
            mask_coeffs: 64,
            blinding_rows: 32,
        },
        roots: core::array::from_fn(|t| [t as u8 + 1; 32]),
        epoch,
    }
}

fn directory(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weights-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn mode(path: &Path) -> u32 {
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

/// The public part (334 bytes) comes first in the state's payload, then the seed (32), after
/// the envelope's 20-byte header.
const SEED_AT: usize = 20 + 334;

/// The bytes `write` writes for `state`, read back from `dir/name`.
fn written(state: &WeightCommitmentState, dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    state.write(&path).expect("written");
    fs::read(&path).expect("the state")
}

/// An envelope of schema `0x0302` around `payload`, as a writer seals one: the header, the
/// payload, and `H("envelope", header ‖ payload)`.
fn sealed(payload: &[u8]) -> Vec<u8> {
    let header = envelope::Header {
        schema: schema::WEIGHT_COMMITMENT_STATE.id,
        version: schema::WEIGHT_COMMITMENT_STATE.version,
        payload_len: payload.len() as u64,
    };
    let mut bytes = header.to_bytes().to_vec();
    bytes.extend(payload);
    let digest = tessera_hash::hash(tessera_hash::Domain::Envelope, &bytes);
    bytes.extend(digest);
    bytes
}

/// A state's payload built field by field: `public(0)`, `seed`, then ten budgets, the first
/// one's opened columns `columns` as given, sorted or not, the others empty.
fn payload(seed: [u8; 32], columns: &[u32]) -> Vec<u8> {
    let mut out = encode(&public(0));
    out.extend(seed);
    out.extend((columns.len() as u64).to_le_bytes());
    for column in columns {
        out.extend(column.to_le_bytes());
    }
    out.extend(0u32.to_le_bytes());
    for _ in 1..10 {
        out.extend(0u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
    }
    out
}

/// The seed has no public way out (05 §3, AGENTS.md invariant 6): each assertion fails to
/// compile when the state gains an `Encode` or a `Decode` impl; the public part and a budget,
/// which have both, show the probe tells them apart.
#[test]
// The probe is a type, never constructed, and its constants and trait are used by the assertions
// alone, which the dead-code lint of Rust 1.88, the minimum supported version, does not count.
#[allow(dead_code)]
fn the_state_has_no_encoding_of_its_own() {
    struct Probe<T: ?Sized>(PhantomData<T>);
    trait Neither {
        const ENCODES: bool = false;
        const DECODES: bool = false;
    }
    impl<T: ?Sized> Neither for Probe<T> {}
    impl<T: ?Sized + Encode> Probe<T> {
        const ENCODES: bool = true;
    }
    impl<T: Decode> Probe<T> {
        const DECODES: bool = true;
    }
    const _: () = assert!(!Probe::<WeightCommitmentState>::ENCODES);
    const _: () = assert!(!Probe::<WeightCommitmentState>::DECODES);
    const _: () = assert!(Probe::<WeightCommitmentPublic>::ENCODES);
    const _: () = assert!(Probe::<WeightCommitmentPublic>::DECODES);
    const _: () = assert!(Probe::<TypeBudget>::ENCODES && Probe::<TypeBudget>::DECODES);
}

#[test]
fn a_new_state_has_a_fresh_seed_and_an_untouched_budget() {
    let dir = directory("new");
    let first = WeightCommitmentState::new(public(0)).expect("the CSPRNG");
    let second = WeightCommitmentState::new(public(0)).expect("the CSPRNG");
    assert_eq!(first.public, public(0));
    for state in [&first, &second] {
        assert!(
            state
                .budget
                .iter()
                .all(|b| b.opened_columns.columns().is_empty())
        );
        assert!(state.budget.iter().all(|b| b.next_blinding_row == 0));
    }
    let (a, b) = (
        written(&first, &dir, "first.twc"),
        written(&second, &dir, "second.twc"),
    );
    assert_eq!(a.len(), 20 + 334 + 32 + 10 * 12 + 32);
    // Two draws of 256 bits agree, or are all zeros, with probability 2^-256.
    let seed = |bytes: &[u8]| bytes[SEED_AT..SEED_AT + 32].to_vec();
    assert_ne!(seed(&a), seed(&b), "each state draws its own seed");
    assert_ne!(seed(&a), vec![0; 32]);
    // Apart from the seeds, and so the digests, the files agree.
    assert_eq!(a[..SEED_AT], b[..SEED_AT]);
    assert_eq!(a[SEED_AT + 32..a.len() - 32], b[SEED_AT + 32..b.len() - 32]);
    // The debug form shows no seed byte.
    let shown = format!("{first:?} {first:#?}");
    let bytes: String = seed(&a)[..8].iter().map(|x| format!("{x}, ")).collect();
    assert!(
        !shown.contains("masks_seed") && !shown.contains(&bytes),
        "{shown}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_state_is_written_with_mode_0600_and_the_public_part_with_the_default_mode() {
    let dir = directory("files");
    let plain = dir.join("plain");
    fs::write(&plain, b"x").expect("a plain file");
    let default = mode(&plain);
    assert_ne!(default, 0o600);

    let mut state = WeightCommitmentState::new(public(3)).expect("the CSPRNG");
    state.budget = core::array::from_fn(|t| TypeBudget {
        opened_columns: OpenedColumns::new((0..t as u32).map(|c| 3 * c + 1).collect())
            .expect("increasing"),
        next_blinding_row: 2 * t as u32,
    });
    let path = dir.join("weights.twc");
    let digest = state.write(&path).expect("written");
    assert_eq!(mode(&path), 0o600);
    let bytes = fs::read(&path).expect("the state");
    assert_eq!(&bytes[bytes.len() - 32..], digest);
    let read = WeightCommitmentState::read(&path).expect("a valid state");
    assert_eq!((read.public, &read.budget), (state.public, &state.budget));
    // Read back, it writes the same bytes, its seed among them (INV-TCE-1, INV-TCE-2).
    assert_eq!(written(&read, &dir, "again.twc"), bytes);
    assert_eq!(
        WeightCommitmentState::open(&bytes).map(|s| s.public),
        Ok(public(3))
    );

    // Overwriting keeps the mode, and leaves no other file.
    let next = WeightCommitmentState::new(public(4)).expect("the CSPRNG");
    next.write(&path).expect("written");
    assert_eq!(mode(&path), 0o600);
    let read = WeightCommitmentState::read(&path).expect("a valid state");
    assert_eq!(read.public, public(4));

    let twr = dir.join("weights.twr");
    let digest = public(3).write(&twr).expect("written");
    assert_eq!(mode(&twr), default);
    let bytes = fs::read(&twr).expect("the public part");
    assert_eq!(&bytes[bytes.len() - 32..], digest);
    let read: WeightCommitmentPublic = envelope::open(
        &bytes,
        &schema::WEIGHT_COMMITMENT_PUBLIC,
        DecodeLimits::MESSAGE,
    )
    .expect("a valid public part");
    assert_eq!(read, public(3));
    assert_eq!(read.weights_root(), public(3).weights_root());

    let mut names: Vec<String> = fs::read_dir(&dir)
        .expect("the directory")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert_eq!(names, ["again.twc", "plain", "weights.twc", "weights.twr"]);
    let _ = fs::remove_dir_all(&dir);
}

/// `read` checks the file as RFC-0003 'Envelope' orders, then opens it: each fault of the input
/// is the error `open` gives the same bytes, and a missing file is the operating system's.
#[test]
fn reading_a_state_checks_its_file_in_rfc_0003_order() {
    let dir = directory("read");
    let honest = sealed(&payload([9; 32], &[2, 5]));
    let len = honest.len() as u64;
    let digest_at = len - 32;
    let mut flipped = honest.clone();
    flipped[100] ^= 1;
    let mut longer = honest.clone();
    longer.push(0);
    let twr = envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, &public(0));
    let error = |code, offset| DecodeError { code, offset };
    for (name, bytes, expected) in [
        ("honest", honest.clone(), None),
        (
            "a payload byte changed",
            flipped,
            Some(error(ErrorCode::CodecDigest, digest_at)),
        ),
        (
            "the last byte missing",
            honest[..honest.len() - 1].to_vec(),
            Some(error(ErrorCode::CodecTruncated, 12)),
        ),
        (
            "a byte past the digest",
            longer,
            Some(error(ErrorCode::CodecNoncanonical, len)),
        ),
        (
            "a public artifact",
            twr,
            Some(error(ErrorCode::CodecMagic, 6)),
        ),
    ] {
        let path = dir.join(format!("{}.twc", name.replace(' ', "-")));
        fs::write(&path, &bytes).expect("a file");
        let read = WeightCommitmentState::read(&path);
        match expected {
            None => {
                let state = read.expect("a valid state");
                assert_eq!(state.budget[0].opened_columns.columns(), [2, 5]);
                assert_eq!(written(&state, &dir, "rewritten.twc"), bytes);
            }
            Some(error) => {
                match read {
                    Err(ReadError::Invalid(found)) => assert_eq!(found, error, "{name}"),
                    other => panic!("{name}: {other:?}"),
                }
                assert_eq!(
                    WeightCommitmentState::open(&bytes).map(|_| ()),
                    Err(error),
                    "{name}"
                );
            }
        }
    }
    match WeightCommitmentState::read(&dir.join("absent.twc")) {
        Err(ReadError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        other => panic!("a missing file: {other:?}"),
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn opened_columns_that_do_not_increase_strictly_are_noncanonical() {
    let dir = directory("columns");
    // Accepted: no column, one, and strictly increasing ones, among them 1 and 256, which a
    // map's byte order would sort the other way (RFC-0040 orders the indices as integers).
    for columns in [&[][..], &[7], &[0, 1, 2], &[1, 256, 65_536, u32::MAX]] {
        let bytes = sealed(&payload([1; 32], columns));
        let state = WeightCommitmentState::open(&bytes).expect("a valid state");
        assert_eq!(state.budget[0].opened_columns.columns(), columns);
        assert_eq!(written(&state, &dir, "state.twc"), bytes, "{columns:?}");
    }
    // Rejected at the first index not above the one before it: unsorted, repeated.
    let first = SEED_AT + 32 + 8;
    for (columns, bad) in [
        (&[5, 3][..], 1),
        (&[2, 2][..], 1),
        (&[0, 1, 1][..], 2),
        (&[1, 4, 9, 8, 10][..], 3),
        (&[256, 1][..], 1),
    ] {
        assert_eq!(
            WeightCommitmentState::open(&sealed(&payload([1; 32], columns))).map(|_| ()),
            Err(DecodeError {
                code: ErrorCode::CodecNoncanonical,
                offset: (first + 4 * bad) as u64,
            }),
            "{columns:?}"
        );
        assert_eq!(
            OpenedColumns::new(columns.to_vec()),
            Err(PcsError::ColumnOrder { position: bad }),
            "{columns:?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

fn codec_vectors() -> Value {
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
        .map(|b| int(b) as u8)
        .collect()
}

fn public_of(value: &Value) -> WeightCommitmentPublic {
    WeightCommitmentPublic {
        layout: WeightLayoutParams {
            log2_k: int(&value["layout"]["log2_k"]) as u8,
            log2_inv_rate: int(&value["layout"]["log2_inv_rate"]) as u8,
            mask_coeffs: int(&value["layout"]["mask_coeffs"]) as u32,
            blinding_rows: int(&value["layout"]["blinding_rows"]) as u32,
        },
        roots: core::array::from_fn(|t| bytes_of(&value["roots"][t]).try_into().expect("32 bytes")),
        epoch: int(&value["epoch"]) as u32,
    }
}

fn budget_of(value: &Value) -> [TypeBudget; 10] {
    core::array::from_fn(|t| {
        let budget = &value["budget"][t];
        TypeBudget {
            opened_columns: OpenedColumns::new(
                budget["opened_columns"]
                    .as_array()
                    .expect("columns")
                    .iter()
                    .map(|c| int(c) as u32)
                    .collect(),
            )
            .expect("increasing"),
            next_blinding_row: int(&budget["next_blinding_row"]) as u32,
        }
    })
}

/// `spec/vectors/codec/weight-commitment.json` (03 §7): each case opens and decodes to its
/// value, each public part's `R_W` is the file's, each state writes back to its bytes, seed
/// included, and each rejected state fails with its code at its offset.
#[test]
fn the_codec_vectors_decode_to_their_values_and_back() {
    let dir = directory("vectors");
    let file = codec_vectors();
    assert_eq!(file["limits"], "message");
    let cases = file["cases"].as_array().expect("cases");
    let (mut publics, mut states) = (0, 0);
    for case in cases {
        let bytes = hex(&case["hex"]);
        let value = &case["value"];
        match case["type"].as_str() {
            Some("WeightCommitmentPublic") => {
                assert_eq!(case["schema"], 0x0301);
                let read: WeightCommitmentPublic = envelope::open(
                    &bytes,
                    &schema::WEIGHT_COMMITMENT_PUBLIC,
                    DecodeLimits::MESSAGE,
                )
                .expect("a valid public part");
                assert_eq!(read, public_of(value), "{}", case["name"]);
                assert_eq!(
                    envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, &read),
                    bytes
                );
                assert_eq!(read.weights_root().to_vec(), hex(&case["weights_root"]));
                publics += 1;
            }
            Some("WeightCommitmentState") => {
                assert_eq!(case["schema"], 0x0302);
                let read = WeightCommitmentState::open(&bytes).expect("a valid state");
                assert_eq!(read.public, public_of(&value["public"]), "{}", case["name"]);
                assert_eq!(read.budget, budget_of(value), "{}", case["name"]);
                let rewritten = written(&read, &dir, "state.twc");
                assert_eq!(rewritten, bytes, "{}", case["name"]);
                assert_eq!(
                    rewritten[SEED_AT..SEED_AT + 32].to_vec(),
                    bytes_of(&value["masks_seed"]),
                    "{}: the seed",
                    case["name"]
                );
                states += 1;
            }
            other => panic!("type {other:?}"),
        }
    }
    assert_eq!((publics, states), (2, 2));
    let rejected = file["rejected"].as_array().expect("rejected");
    assert_eq!(rejected.len(), 3);
    for case in rejected {
        assert_eq!(case["code"], "ERR-CODEC-NONCANONICAL");
        assert_eq!(
            WeightCommitmentState::open(&hex(&case["hex"])).map(|_| ()),
            Err(DecodeError {
                code: ErrorCode::CodecNoncanonical,
                offset: case["offset"].as_u64().expect("an offset"),
            }),
            "{}",
            case["why"]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}
