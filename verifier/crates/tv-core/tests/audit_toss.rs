// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The pending toss (RFC-0067 'The pending toss'): the `VerifierToss` artifact, schema `0x0702`,
//! `.tvt`, and its file, `TossFile`. Each test breaks one rule and asserts the exact error, the
//! adversarial ones first: a toss read back after a restart reveals the sample a reveal without
//! the restart derives; a second commitment onto a toss, a tampered toss, a toss of another
//! manifest and a toss behind a link are refused, and nothing is replaced; the toss is kept until
//! the ledger records its sample, and a stop between the record and the removal changes nothing;
//! the file is mode 0600 and is overwritten with zeros before it is removed; no `Debug` shows the
//! share. The layout is read from RFC-0067 and RFC-0003, and tampered envelopes are resealed with
//! the test kit's BLAKE3, not with `tv-core`'s encoder.

mod anchor_support;

use anchor_support::{GENESIS_NS, PERIOD_NS, Scenario, carry, chain, end, inbound, open, outbound};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tv_codec::DecodeLimits;
use tv_core::anchor::{Failure, VerifiedLog, Window};
use tv_core::audit::{
    AuditPolicy, AuditSample, AuditToss, CoinTossTranscript, LedgerFile, RateBucket, RequestKey,
    TOSS_LIMITS, TossFile, VerifierToss, read_ledger,
};
use tv_core::claims::ParamSetId;
use tv_core::verdict::{ErrorCode as Code, VerifierError};
use tv_testkit::blake3::derive_key;

// ----- Helpers ---------------------------------------------------------------------------------

const fn err(code: Code) -> VerifierError {
    VerifierError { code }
}

fn inconclusive(code: Code) -> Failure {
    Failure::Inconclusive(err(code))
}

/// One bucket to `2^32 − 1` tokens at `threshold`.
fn policy(threshold: u64) -> AuditPolicy {
    AuditPolicy::new(
        86_400_000_000_000,
        vec![RateBucket {
            max_tokens: u32::MAX,
            threshold,
        }],
        16_384,
        262_144,
        604_800_000_000_000,
        ParamSetId::new("live-64/1").unwrap(),
    )
    .unwrap()
}

/// Rate 1/2: which requests the sample selects depends on both shares.
fn halves() -> AuditPolicy {
    policy(1 << 63)
}

/// A log of session 1's requests 1 to 8, each of one inbound and one outbound frame, ended.
fn requests() -> Scenario {
    let mut items = vec![open(1)];
    for r in 1..=8 {
        items.push(inbound(1, r, 0, 2).item());
        items.push(outbound(1, r, 0, 3).item());
        items.push(end(1, r));
    }
    Scenario::one(items)
}

fn scratch(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tessera-toss-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    fs::canonicalize(dir).unwrap()
}

/// The names in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// RFC-0003's envelope of `schema` and `version` around `payload`, its digest the test kit's
/// BLAKE3: magic, envelope version 1, the schema and its version, flags 0, the length, the
/// payload, then `H("envelope", everything before)`.
fn envelope(schema: u16, version: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &schema.to_le_bytes(),
        &version.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    let digest = derive_key("tessera/v1/envelope", &out);
    out.extend_from_slice(&digest);
    out
}

/// A `.tvt` envelope, schema `0x0702` version 1.
fn seal(payload: &[u8]) -> Vec<u8> {
    envelope(0x0702, 1, payload)
}

/// The payload of a sealed toss: past the 20 bytes of the header, before the 32 of the digest.
fn payload(tvt: &[u8]) -> &[u8] {
    &tvt[20..tvt.len() - 32]
}

/// The share, the payload's last 32 bytes (RFC-0067: `manifest_digest`, `policy`, `commitment`,
/// then `share`).
fn share_of(tvt: &[u8]) -> [u8; 32] {
    let p = payload(tvt);
    p[p.len() - 32..].try_into().unwrap()
}

/// `H("coin-commit", share)`, with the test kit's BLAKE3.
fn coin_commit(share: &[u8]) -> [u8; 32] {
    derive_key("tessera/v1/coin-commit", share)
}

/// Temporary files left in `dir`.
fn leftovers(dir: &Path) -> Vec<String> {
    names(dir)
        .into_iter()
        .filter(|n| n.ends_with(".tmp"))
        .collect()
}

const PROVER: [u8; 32] = [0x5a; 32];

// ----- Restart ---------------------------------------------------------------------------------

/// Issue #529's criterion: a verifier that stops after committing reveals after a restart, from
/// the files alone, and derives the sample that a reveal without the restart derives, which is
/// RFC-0016's sample over the shares the file holds at RFC-0067's offsets.
#[test]
fn a_toss_read_back_after_a_restart_reveals_the_sample_a_reveal_without_it_derives() {
    let scenario = requests();
    let window = scenario.window();
    let first = scratch("restart-kept");
    let second = scratch("restart-read");
    let path = first.join("toss.tvt");
    // The toss is on disk before its commitment can be read.
    let mut kept = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    let commitment = *kept.commitment();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(kept.path(), path);
    // The same file, to be read back after the restart: a copy in a directory of its own, with
    // a ledger of its own, so that the two reveals stay apart.
    let copy = second.join("toss.tvt");
    fs::write(&copy, &bytes).unwrap();
    // Without a restart: the value that committed reveals.
    let mut ledger = LedgerFile::create(&first.join("ledger.tvl")).unwrap();
    let without = kept.reveal(&window, PROVER, &mut ledger).unwrap();
    drop(kept);
    drop(ledger);
    // The restart: nothing but the files.
    let mut read = TossFile::open(&copy).unwrap();
    assert_eq!(read.commitment(), &commitment);
    assert_eq!(read.manifest_digest(), scenario.period.manifest_digest());
    assert_eq!(read.policy(), &halves());
    let mut other = LedgerFile::create(&second.join("ledger.tvl")).unwrap();
    let after = read.reveal(&window, PROVER, &mut other).unwrap();
    assert_eq!(after, without);
    assert_eq!(after.seal().unwrap(), without.seal().unwrap());
    assert!(other.ledger().is_derived(&after));
    // RFC-0016's sample over the file's own shares.
    let share = share_of(&bytes);
    assert_eq!(commitment, coin_commit(&share));
    let expected = AuditSample::derive(
        &window,
        halves(),
        CoinTossTranscript {
            verifier_commitment: commitment,
            prover_share: PROVER,
            verifier_share: share,
        },
    )
    .unwrap();
    assert_eq!(after, expected);
    read.clear().unwrap();
    drop(other);
    fs::remove_dir_all(&first).unwrap();
    fs::remove_dir_all(&second).unwrap();
}

// ----- Refusals --------------------------------------------------------------------------------

/// RFC-0067 'Before the commitment leaves': a toss is never silently replaced. A second
/// commitment onto it is `ERR-USAGE` and leaves its bytes, and so is one onto any name that
/// holds anything: a link that leads nowhere, a directory, another file.
#[test]
fn a_second_commitment_onto_a_toss_is_refused_and_leaves_it_as_it_was() {
    let scenario = requests();
    let manifest = scenario.period.manifest();
    let dir = scratch("second");
    let path = dir.join("toss.tvt");
    let first = TossFile::commit(&path, manifest, halves()).unwrap();
    let bytes = fs::read(&path).unwrap();
    for _ in 0..2 {
        assert_eq!(
            TossFile::commit(&path, manifest, halves()).err(),
            Some(err(Code::Usage))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
    assert_eq!(
        TossFile::open(&path).unwrap().commitment(),
        first.commitment()
    );
    assert!(leftovers(&dir).is_empty());
    #[cfg(unix)]
    {
        let dangling = dir.join("dangling.tvt");
        std::os::unix::fs::symlink(dir.join("nowhere"), &dangling).unwrap();
        assert_eq!(
            TossFile::commit(&dangling, manifest, halves()).err(),
            Some(err(Code::Usage))
        );
        assert_eq!(fs::read_link(&dangling).unwrap(), dir.join("nowhere"));
        assert!(!dir.join("nowhere").exists());
    }
    let sub = dir.join("sub.tvt");
    fs::create_dir(&sub).unwrap();
    assert_eq!(
        TossFile::commit(&sub, manifest, halves()).err(),
        Some(err(Code::Usage))
    );
    assert!(sub.is_dir());
    let other = dir.join("other.tvt");
    fs::write(&other, b"not a toss").unwrap();
    assert_eq!(
        TossFile::commit(&other, manifest, halves()).err(),
        Some(err(Code::Usage))
    );
    assert_eq!(fs::read(&other).unwrap(), b"not a toss");
    // A path with no file name.
    assert_eq!(
        TossFile::commit(&dir.join(".."), manifest, halves()).err(),
        Some(err(Code::Usage))
    );
    assert!(leftovers(&dir).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'What is refused, not repaired': one byte of the share, or of the commitment,
/// flipped and the envelope resealed, no longer opens the commitment, `ERR-PROTO-COINTOSS`; a
/// byte flipped without resealing breaks the envelope, `ERR-CODEC-DIGEST`; another schema or
/// version is `ERR-CODEC-MAGIC` or `ERR-CODEC-VERSION`. Each is refused by the file's reader and
/// by both decoders, the file is left as it was, and a commitment onto it is refused.
#[test]
fn a_tampered_toss_is_refused_and_never_replaced() {
    let scenario = requests();
    let manifest = scenario.period.manifest();
    let dir = scratch("tampered");
    let path = dir.join("toss.tvt");
    drop(TossFile::commit(&path, manifest, halves()).unwrap());
    let bytes = fs::read(&path).unwrap();
    let p = payload(&bytes).to_vec();
    let n = p.len();
    let flipped = |at: usize| {
        let mut q = p.clone();
        q[at] ^= 1;
        q
    };
    let refused = |tvt: &[u8], code: Code, what: &str| {
        fs::write(&path, tvt).unwrap();
        assert_eq!(TossFile::open(&path).err(), Some(err(code)), "{what}");
        assert_eq!(
            VerifierToss::open(tvt, &TOSS_LIMITS).err(),
            Some(err(code)),
            "{what}"
        );
        // Refused, not repaired: the file is as it was, and no commitment replaces it.
        assert_eq!(fs::read(&path).unwrap(), tvt, "{what}");
        assert_eq!(
            TossFile::commit(&path, manifest, halves()).err(),
            Some(err(Code::Usage)),
            "{what}"
        );
        assert_eq!(fs::read(&path).unwrap(), tvt, "{what}");
    };
    for (what, at) in [
        ("the share's first byte", n - 32),
        ("the share's last byte", n - 1),
        ("the commitment", n - 64),
    ] {
        let changed = flipped(at);
        refused(&seal(&changed), Code::ProtocolCoinToss, what);
        assert_eq!(
            VerifierToss::decode_payload(&changed, &DecodeLimits::MESSAGE).err(),
            Some(err(Code::ProtocolCoinToss)),
            "{what}"
        );
    }
    for (what, at) in [
        ("a byte of the share", 20 + n - 1),
        ("a byte of the digest", bytes.len() - 1),
    ] {
        let mut broken = bytes.clone();
        broken[at] ^= 1;
        refused(&broken, Code::CodecDigest, what);
    }
    refused(
        &envelope(0x0403, 1, &p),
        Code::CodecMagic,
        "a sample's schema",
    );
    refused(&envelope(0x0702, 2, &p), Code::CodecVersion, "version 2");
    // The untouched toss still opens.
    fs::write(&path, &bytes).unwrap();
    assert!(TossFile::open(&path).is_ok());
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'The reveal': a toss over another manifest than the window's is `ERR-USAGE`, read
/// back or not, before anything is derived: the ledger and the toss are left as they were. Over
/// its own manifest it reveals.
#[test]
fn a_toss_of_another_manifest_is_refused_and_records_nothing() {
    let built = chain(&[
        (
            GENESIS_NS,
            GENESIS_NS + PERIOD_NS,
            vec![open(1), inbound(1, 1, 0, 1).item(), end(1, 1)],
        ),
        (
            GENESIS_NS + PERIOD_NS,
            GENESIS_NS + 2 * PERIOD_NS,
            vec![carry(1), inbound(1, 2, 0, 1).item(), end(1, 2)],
        ),
    ]);
    let logs: Vec<VerifiedLog> = built.iter().map(|b| b.verified()).collect();
    let first = Window::bind(&logs[0], &[], None).unwrap();
    let second = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    let dir = scratch("manifest");
    let path = dir.join("toss.tvt");
    let ledger_path = dir.join("ledger.tvl");
    let mut pending = TossFile::commit(&path, &built[1].checked(), policy(u64::MAX)).unwrap();
    let bytes = fs::read(&path).unwrap();
    let mut ledger = LedgerFile::create(&ledger_path).unwrap();
    let empty = fs::read(&ledger_path).unwrap();
    assert_eq!(
        pending.reveal(&first, PROVER, &mut ledger).err(),
        Some(inconclusive(Code::Usage))
    );
    assert_eq!(
        TossFile::open(&path)
            .unwrap()
            .reveal(&first, PROVER, &mut ledger)
            .err(),
        Some(inconclusive(Code::Usage))
    );
    assert!(ledger.ledger().derived().is_empty());
    assert_eq!(fs::read(&ledger_path).unwrap(), empty);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    // Over its own manifest, the same value reveals.
    let sample = pending.reveal(&second, PROVER, &mut ledger).unwrap();
    assert_eq!(sample.manifest_digest(), &built[1].identity);
    assert_eq!(
        sample.selected(),
        [RequestKey {
            session_id: [1; 16],
            request_id: 2
        }]
    );
    assert!(read_ledger(&ledger_path).unwrap().is_derived(&sample));
    pending.clear().unwrap();
    drop(ledger);
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'The reveal' reads the file without following a link, under the ledger's path
/// rules: a symbolic link to a toss, a dangling one, a toss of two names and a directory are
/// `ERR-USAGE`, and a lost toss `ERR-RES-IO`.
#[cfg(unix)]
#[test]
fn a_toss_behind_a_link_or_of_two_names_is_refused() {
    let scenario = requests();
    let dir = scratch("links");
    let path = dir.join("toss.tvt");
    drop(TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap());
    let link = dir.join("link.tvt");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert_eq!(TossFile::open(&link).err(), Some(err(Code::Usage)));
    let dangling = dir.join("dangling.tvt");
    std::os::unix::fs::symlink(dir.join("nowhere"), &dangling).unwrap();
    assert_eq!(TossFile::open(&dangling).err(), Some(err(Code::Usage)));
    let hard = dir.join("hard.tvt");
    fs::hard_link(&path, &hard).unwrap();
    for name in [&path, &hard] {
        assert_eq!(TossFile::open(name).err(), Some(err(Code::Usage)));
    }
    fs::remove_file(&hard).unwrap();
    assert!(TossFile::open(&path).is_ok());
    assert_eq!(TossFile::open(&dir).err(), Some(err(Code::Usage)));
    assert_eq!(
        TossFile::open(&dir.join("lost.tvt")).err(),
        Some(err(Code::Io))
    );
    fs::remove_dir_all(&dir).unwrap();
}

/// A toss past [`TOSS_LIMITS`] is refused before anything is written, `ERR-CODEC-LIMIT`, and a
/// file past them before it is read.
#[test]
fn a_toss_past_its_limits_is_refused_before_anything_is_written() {
    let scenario = requests();
    let dir = scratch("limits");
    let path = dir.join("toss.tvt");
    // 90,000 buckets of 12 bytes: past 1 MiB.
    let buckets = (1..=90_000u32)
        .map(|max_tokens| RateBucket {
            max_tokens,
            threshold: 1,
        })
        .collect();
    let large =
        AuditPolicy::new(1, buckets, 1, 2, 1, ParamSetId::new("live-64/1").unwrap()).unwrap();
    assert_eq!(
        TossFile::commit(&path, scenario.period.manifest(), large).err(),
        Some(err(Code::CodecLimit))
    );
    assert!(names(&dir).is_empty());
    let file = fs::File::create(&path).unwrap();
    file.set_len(TOSS_LIMITS.max_total_bytes + 1).unwrap();
    drop(file);
    assert_eq!(TossFile::open(&path).err(), Some(err(Code::CodecLimit)));
    fs::remove_dir_all(&dir).unwrap();
}

// ----- The file --------------------------------------------------------------------------------

/// RFC-0067 'Before the commitment leaves': mode 0600, one name, nothing left beside it.
#[cfg(unix)]
#[test]
fn the_toss_is_created_with_mode_0600_and_one_name() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let scenario = requests();
    let dir = scratch("mode");
    let path = dir.join("toss.tvt");
    drop(TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap());
    let meta = fs::symlink_metadata(&path).unwrap();
    assert!(meta.file_type().is_file());
    assert_eq!(meta.permissions().mode() & 0o777, 0o600);
    assert_eq!(meta.nlink(), 1);
    assert_eq!(names(&dir), ["toss.tvt"]);
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'The encoding': `manifest_digest`, `policy`, `commitment`, then `share`, 32 bytes,
/// TCE in RFC-0003's envelope, schema `0x0702` version 1, built here field by field. The readers'
/// value encodes and seals back to the bytes (INV-DM-1); every proper prefix of the payload is
/// cut short and a byte past it is refused.
#[test]
fn a_toss_is_its_fields_in_order_in_an_envelope_of_schema_0x0702() {
    let scenario = requests();
    let dir = scratch("layout");
    let path = dir.join("toss.tvt");
    let pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    let bytes = fs::read(&path).unwrap();
    let share = share_of(&bytes);
    let mut expected = scenario.period.manifest_digest().to_vec();
    expected.extend(86_400_000_000_000u64.to_le_bytes());
    expected.extend(1u64.to_le_bytes());
    expected.extend(u32::MAX.to_le_bytes());
    expected.extend((1u64 << 63).to_le_bytes());
    expected.extend(16_384u32.to_le_bytes());
    expected.extend(262_144u32.to_le_bytes());
    expected.extend(604_800_000_000_000u64.to_le_bytes());
    expected.extend(9u64.to_le_bytes());
    expected.extend(b"live-64/1");
    expected.extend(coin_commit(&share));
    expected.extend(share);
    assert_eq!(bytes, seal(&expected));
    assert_eq!(pending.commitment(), &coin_commit(&share));
    let toss = VerifierToss::open(&bytes, &TOSS_LIMITS).unwrap();
    assert_eq!(toss.manifest_digest(), scenario.period.manifest_digest());
    assert_eq!(toss.policy(), &halves());
    assert_eq!(toss.commitment(), &coin_commit(&share));
    assert_eq!(toss.seal().unwrap().as_slice(), bytes);
    assert_eq!(toss.encode_payload().unwrap().as_slice(), expected);
    let decoded = VerifierToss::decode_payload(&expected, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(decoded.seal().unwrap().as_slice(), bytes);
    for cut in 0..expected.len() {
        assert_eq!(
            VerifierToss::decode_payload(&expected[..cut], &DecodeLimits::MESSAGE).err(),
            Some(err(Code::CodecTruncated)),
            "{cut}"
        );
    }
    let mut longer = expected.clone();
    longer.push(0);
    assert_eq!(
        VerifierToss::decode_payload(&longer, &DecodeLimits::MESSAGE).err(),
        Some(err(Code::CodecNoncanonical))
    );
    assert_eq!(
        VerifierToss::open(&seal(&longer), &TOSS_LIMITS).err(),
        Some(err(Code::CodecNoncanonical))
    );
    // Two draws, two shares.
    let again = dir.join("again.tvt");
    let other = TossFile::commit(&again, scenario.period.manifest(), halves()).unwrap();
    assert_ne!(other.commitment(), pending.commitment());
    fs::remove_dir_all(&dir).unwrap();
}

// ----- Cleared ---------------------------------------------------------------------------------

/// RFC-0067 'Cleared after the record': once the ledger records the sample, the toss is cleared
/// and its file gone, nothing left beside it. A stop between the record and the removal leaves
/// the file; read back, the toss is refused against another prover share by the ledger, and
/// against the same share gives the same sample, the ledger not written again, one `derived`
/// entry; then it is cleared.
#[test]
fn a_toss_is_cleared_after_its_record_and_a_stop_between_the_two_changes_nothing() {
    let scenario = requests();
    let window = scenario.window();
    let dir = scratch("cleared");
    let path = dir.join("toss.tvt");
    let ledger_path = dir.join("ledger.tvl");
    let mut ledger = LedgerFile::create(&ledger_path).unwrap();
    let mut pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    let sample = pending.reveal(&window, PROVER, &mut ledger).unwrap();
    let recorded = fs::read(&ledger_path).unwrap();
    assert_eq!(
        read_ledger(&ledger_path).unwrap().derived(),
        [(*scenario.period.manifest_digest(), *sample.identity())]
    );
    // The stop: the value is gone before its clearing, the file is not.
    drop(pending);
    let bytes = fs::read(&path).unwrap();
    // Read back and revealed against another share: the ledger holds the period's sample.
    assert_eq!(
        TossFile::open(&path)
            .unwrap()
            .reveal(&window, [0xee; 32], &mut ledger)
            .err(),
        Some(inconclusive(Code::ProtocolOrder))
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    // Against the same share: the same sample, and nothing is written.
    let mut again = TossFile::open(&path).unwrap();
    assert_eq!(again.reveal(&window, PROVER, &mut ledger).unwrap(), sample);
    assert_eq!(fs::read(&ledger_path).unwrap(), recorded);
    assert_eq!(ledger.ledger().derived().len(), 1);
    again.clear().unwrap();
    assert!(fs::symlink_metadata(&path).is_err());
    assert_eq!(names(&dir), ["ledger.tvl"]);
    // A cleared toss is lost: nothing can read it back.
    assert_eq!(TossFile::open(&path).err(), Some(err(Code::Io)));
    drop(ledger);
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'Cleared after the record': the toss is kept until the ledger records its sample.
/// Cleared before any reveal, or after a reveal whose ledger write failed, it is refused,
/// `ERR-USAGE`, and the file is left as it was; once the ledger takes the record, it clears.
#[cfg(unix)]
#[test]
fn a_toss_is_kept_until_the_ledger_records_its_sample() {
    let scenario = requests();
    let window = scenario.window();
    let dir = scratch("kept");
    let path = dir.join("toss.tvt");
    let ledger_path = dir.join("ledger.tvl");
    let pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(pending.clear().err(), Some(err(Code::Usage)));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    // A second name makes the ledger's write fail: no record, and the toss stays.
    let mut ledger = LedgerFile::create(&ledger_path).unwrap();
    let second = dir.join("second.tvl");
    fs::hard_link(&ledger_path, &second).unwrap();
    let mut pending = TossFile::open(&path).unwrap();
    assert_eq!(
        pending.reveal(&window, PROVER, &mut ledger).err(),
        Some(inconclusive(Code::Usage))
    );
    assert!(ledger.ledger().derived().is_empty());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(pending.clear().err(), Some(err(Code::Usage)));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    // The ledger written, the toss clears.
    fs::remove_file(&second).unwrap();
    let mut pending = TossFile::open(&path).unwrap();
    let sample = pending.reveal(&window, PROVER, &mut ledger).unwrap();
    assert!(read_ledger(&ledger_path).unwrap().is_derived(&sample));
    pending.clear().unwrap();
    assert!(fs::symlink_metadata(&path).is_err());
    drop(ledger);
    fs::remove_dir_all(&dir).unwrap();
}

/// RFC-0067 'Cleared after the record': the file is overwritten with zeros before its removal.
/// A second name made after the toss was read keeps the inode in sight: once the toss is
/// cleared, its every byte is zero.
#[cfg(unix)]
#[test]
fn the_toss_is_overwritten_with_zeros_before_it_is_removed() {
    let scenario = requests();
    let window = scenario.window();
    let dir = scratch("zeros");
    let path = dir.join("toss.tvt");
    let mut ledger = LedgerFile::create(&dir.join("ledger.tvl")).unwrap();
    let mut pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    pending.reveal(&window, PROVER, &mut ledger).unwrap();
    let size = fs::metadata(&path).unwrap().len();
    let witness = dir.join("witness");
    fs::hard_link(&path, &witness).unwrap();
    pending.clear().unwrap();
    assert!(fs::symlink_metadata(&path).is_err());
    let left = fs::read(&witness).unwrap();
    assert_eq!(left.len() as u64, size);
    assert!(left.iter().all(|&b| b == 0));
    drop(ledger);
    fs::remove_dir_all(&dir).unwrap();
}

/// The clearing overwrites only the file the toss was read from: a file put at its path, or a
/// link put there to the toss, is refused, `ERR-RES-IO`, and neither it nor the toss is changed.
#[cfg(unix)]
#[test]
fn a_toss_replaced_before_its_clearing_is_not_overwritten() {
    let scenario = requests();
    let window = scenario.window();
    let dir = scratch("replaced");
    let path = dir.join("toss.tvt");
    let moved = dir.join("moved.tvt");
    let mut ledger = LedgerFile::create(&dir.join("ledger.tvl")).unwrap();
    let mut pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    pending.reveal(&window, PROVER, &mut ledger).unwrap();
    let bytes = fs::read(&path).unwrap();
    fs::rename(&path, &moved).unwrap();
    fs::write(&path, b"decoy").unwrap();
    assert_eq!(pending.clear().err(), Some(err(Code::Io)));
    assert_eq!(fs::read(&path).unwrap(), b"decoy");
    assert_eq!(fs::read(&moved).unwrap(), bytes);
    // Read back from where it went, then a link to it put at that path.
    let mut pending = TossFile::open(&moved).unwrap();
    pending.reveal(&window, PROVER, &mut ledger).unwrap();
    let away = dir.join("away.tvt");
    fs::rename(&moved, &away).unwrap();
    std::os::unix::fs::symlink(&away, &moved).unwrap();
    assert_eq!(pending.clear().err(), Some(err(Code::Io)));
    assert_eq!(fs::read(&away).unwrap(), bytes);
    drop(ledger);
    fs::remove_dir_all(&dir).unwrap();
}

// ----- Secrecy ---------------------------------------------------------------------------------

/// The decimal numbers of `text`, in order, whatever separates them.
fn numbers(text: &str) -> Vec<u64> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// 05 §3: the share is secret-class until the reveal. No `Debug` of the toss, the file, the
/// in-memory toss or an error shows it, as hexadecimal of either case or as its 32 bytes in
/// order, however formatted, while each shows the commitment.
#[test]
fn no_debug_shows_the_share() {
    let scenario = requests();
    let dir = scratch("secret");
    let path = dir.join("toss.tvt");
    let pending = TossFile::commit(&path, scenario.period.manifest(), halves()).unwrap();
    let bytes = fs::read(&path).unwrap();
    let share = share_of(&bytes);
    let hex: String = share.iter().map(|b| format!("{b:02x}")).collect();
    let share_bytes: Vec<u64> = share.iter().map(|&b| u64::from(b)).collect();
    let commitment: Vec<u64> = pending.commitment().iter().map(|&b| u64::from(b)).collect();
    let toss = VerifierToss::open(&bytes, &TOSS_LIMITS).unwrap();
    let audit = AuditToss::from(VerifierToss::open(&bytes, &TOSS_LIMITS).unwrap());
    let shown = [
        format!("{pending:?}"),
        format!("{pending:#?}"),
        format!("{toss:?}"),
        format!("{toss:#?}"),
        format!("{audit:?}"),
        format!("{audit:#?}"),
        format!(
            "{:?}",
            TossFile::commit(&path, scenario.period.manifest(), halves())
        ),
    ];
    for text in &shown {
        assert!(!text.to_lowercase().contains(&hex), "{text}");
        assert!(
            !numbers(text).windows(32).any(|w| w == share_bytes),
            "{text}"
        );
    }
    for text in &shown[..6] {
        assert!(numbers(text).windows(32).any(|w| w == commitment), "{text}");
    }
    fs::remove_dir_all(&dir).unwrap();
}
