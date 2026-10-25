// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A differential test against the independent verifier (DEP-2): random two-period logs, signed
//! and sealed as the recorder writes them, go through `tessera-anchor`'s checks and `tv-core`'s,
//! which were written apart from the same RFCs (RFC-0062 'The check', RFC-0063 'Which requests a
//! period draws from'). The two must agree:
//! - on whether each period's log is its manifest's, a byte changed in it included;
//! - on the period's candidates and their lengths, every field.
//!
//! `tv-core` checks each manifest's signature with `tv-hash`'s strict Ed25519 (RFC-0060
//! 'Signatures'), which `tests/ed25519.rs` compares with `ed25519-dalek`'s `verify_strict`.

use tessera_anchor::recorder::encode_log;
use tessera_anchor::{
    DigestLogEntry, Direction, HeapKey, LogManifest, ManifestBody, VerifiedLog, candidates, key_id,
    merkle, public_key,
};
use tessera_codec::{Secret, envelope, schema};
use tv_codec::DecodeLimits;
use tv_core::anchor as tv;

/// SplitMix64, the inputs' generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn seed() -> Secret<HeapKey<32>> {
    Secret::new(HeapKey::from_box(Box::new([0x61; 32])))
}

/// Up to `n` random entries numbered from `first_seq`, over three sessions and four requests.
/// Frames mostly have lengths `22 + 4n`, sometimes one no frame has.
fn entries(rng: &mut Rng, first_seq: u64, n: u64, t_ns: u64) -> Vec<DigestLogEntry> {
    let sessions = [[1u8; 16], [2; 16], [3; 16]];
    let requests = [0u64, 1, 7, u64::MAX];
    let count = rng.below(n + 1);
    (0..count)
        .map(|i| {
            let seq = first_seq + i;
            let session_id = sessions[rng.below(3) as usize];
            let request_id = requests[rng.below(4) as usize];
            match rng.below(8) {
                0..=2 => {
                    let length = if rng.below(8) == 0 {
                        [0u32, 21, 23, 25][rng.below(4) as usize]
                    } else {
                        22 + 4 * rng.below(40) as u32
                    };
                    DigestLogEntry::Frame {
                        seq,
                        t_ns,
                        session_id,
                        request_id,
                        direction: if rng.below(2) == 0 {
                            Direction::In
                        } else {
                            Direction::Out
                        },
                        frame_index: rng.below(3) as u32,
                        length,
                        frame_seq: rng.below(1 << 20) as u32,
                        sha256: [rng.below(256) as u8; 32],
                    }
                }
                3 | 4 => DigestLogEntry::RequestEnd {
                    seq,
                    t_ns,
                    session_id,
                    request_id,
                },
                5 => DigestLogEntry::SessionOpen {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: [9; 32],
                },
                6 => DigestLogEntry::SessionCarry {
                    seq,
                    t_ns,
                    session_id,
                    key_commitment: [9; 32],
                },
                _ => DigestLogEntry::SessionClose {
                    seq,
                    t_ns,
                    session_id,
                },
            }
        })
        .collect()
}

/// The signed manifest over `entries`, following `prev` when there is one.
fn manifest(entries: &[DigestLogEntry], first_seq: u64, prev: Option<&LogManifest>) -> LogManifest {
    let leaves: Vec<_> = entries.iter().map(merkle::leaf_hash).collect();
    let (start, prev_manifest) = match prev {
        Some(p) => (p.body.period_end_ns, p.identity()),
        None => (1_000, [0; 32]),
    };
    ManifestBody {
        recorder_key_id: key_id(&public_key(&seed())),
        log_id: [5; 16],
        period_start_ns: start,
        period_end_ns: start + 1_000,
        first_seq,
        entry_count: entries.len() as u64,
        entries_root: merkle::root(&leaves),
        prev_manifest,
    }
    .sign(&seed())
    .expect("a signature")
}

/// The verifier's checked manifest of `m`.
fn checked(m: &LogManifest) -> tv::CheckedManifest {
    let sealed = envelope::seal(&schema::LOG_MANIFEST, m);
    tv::LogManifest::open(&sealed, &DecodeLimits::MESSAGE)
        .expect("a manifest envelope")
        .check(&tv::RecorderKeys::new(vec![public_key(&seed())]))
        .expect("a valid manifest")
}

#[test]
fn both_workspaces_check_logs_and_draw_candidates_alike() {
    let mut rng = Rng(0x0150_0001);
    let (mut agreed_logs, mut refused, mut drawn) = (0, 0, 0);
    for case in 0..400 {
        // The previous period, in three cases of four, then the period.
        let previous = (rng.below(4) != 0).then(|| {
            let e = entries(&mut rng, 0, 12, 1_500);
            let m = manifest(&e, 0, None);
            (e, m)
        });
        let first = previous.as_ref().map_or(0, |(e, _)| e.len() as u64);
        let current_entries = entries(&mut rng, first, 12, 2_500);
        let current = manifest(&current_entries, first, previous.as_ref().map(|(_, m)| m));

        // The prover's side.
        let p_prev = previous
            .as_ref()
            .map(|(e, m)| VerifiedLog::check(m, e.clone()).expect("the previous log"));
        let p_cur = VerifiedLog::check(&current, current_entries.clone()).expect("the log");
        let p_candidates = candidates(p_prev.as_ref(), &p_cur);

        // The verifier's side, from the bytes the recorder writes.
        let v_prev = previous.as_ref().map(|(e, m)| {
            tv::VerifiedLog::check(&checked(m), &encode_log(e), &DecodeLimits::MESSAGE)
                .expect("the verifier's previous log")
        });
        let v_cur = tv::VerifiedLog::check(
            &checked(&current),
            &encode_log(&current_entries),
            &DecodeLimits::MESSAGE,
        )
        .expect("the verifier's log");
        let window = tv::Window::bind(&v_cur, &[], v_prev.as_ref()).expect("bound");
        let v_candidates = window.candidates().expect("candidates");

        let p: Vec<_> = p_candidates
            .iter()
            .map(|c| (c.session_id, c.request_id, c.tokens, c.ended))
            .collect();
        let v: Vec<_> = v_candidates
            .iter()
            .map(|c| (c.session_id, c.request_id, c.tokens, c.ended))
            .collect();
        assert_eq!(p, v, "case {case}: the candidates");
        drawn += p.len();

        // One byte of the period's log changed: both refuse it, or both accept what decodes to
        // the same entries, which cannot happen for a byte inside an entry.
        let mut bytes = encode_log(&current_entries);
        if !bytes.is_empty() {
            let at = rng.below(bytes.len() as u64) as usize;
            bytes[at] ^= 1 << rng.below(8);
            let prover_accepts = tessera_anchor::recorder::decode_log(&bytes)
                .ok()
                .and_then(|e| VerifiedLog::check(&current, e).ok())
                .is_some();
            let verifier_accepts =
                tv::VerifiedLog::check(&checked(&current), &bytes, &DecodeLimits::MESSAGE).is_ok();
            assert_eq!(prover_accepts, verifier_accepts, "case {case}: byte {at}");
            assert!(!verifier_accepts, "case {case}: a changed log is accepted");
            refused += 1;
        }
        agreed_logs += 1;
    }
    // The comparison covered something: the counts are this generator's, fixed by its seed.
    assert_eq!(agreed_logs, 400);
    assert!(refused > 300, "{refused} changed logs");
    assert!(drawn > 200, "{drawn} candidates");
}
