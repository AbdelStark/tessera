// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete NI driver robustness plus known-false, single-check proof controls.

use std::{sync::OnceLock, time::Duration};
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic, decode};
use tv_core::{
    ErrorCode, RejectReason, ResourceLimits, Verdict, VerifierConfig,
    claims::ClaimList,
    params::ParamSet,
    protocol::{FixedBits, NiProof, PublicStatement},
    validation::StatementField,
    verdict::CommitmentClass,
};
use tv_field::Fp2;
use tv_hash::{Domain, hash};

/// Read public fixture data at runtime, as the isolated driver tests do.
/// No Rust source outside this package is compiled or included (SP-6).
pub fn fixture_bytes(tamper: bool, name: &str) -> Vec<u8> {
    let base = if tamper {
        tv_testkit::vectors_dir().join("tamper")
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/tv-core/tests/fixtures/driver")
    };
    std::fs::read(base.join(name)).unwrap()
}

/// Recorded public producer-integration data; not an independent conformance oracle.
/// Its test-small parameters certify nothing.
pub fn golden() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| fixture_bytes(false, "ni.tnp"))
}

/// The public tamper corpus base, with its own immutable caller inputs.
pub fn tamper() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| fixture_bytes(true, "ni.tnp"))
}

struct Fixture {
    claims: ClaimList,
    statement: PublicStatement,
    config: VerifierConfig,
    lookup_rejection: RejectReason,
}

fn fixture(which: usize) -> &'static Fixture {
    static FIXTURE: OnceLock<[Fixture; 2]> = OnceLock::new();
    &FIXTURE.get_or_init(|| {
        std::array::from_fn(|i| {
            let limits = DecodeLimits::MESSAGE;
            let claims_bytes = fixture_bytes(i != 0, "claims.tcl");
            let claims = ClaimList::open(&claims_bytes, &limits).unwrap();
            let statement = NiProof::open(if i == 0 { golden() } else { tamper() }, &limits)
                .unwrap()
                .round1
                .statement;
            let weights = if i == 0 {
                decode(
                    &fixture_bytes(false, "weights.tce"),
                    &limits,
                    WeightCommitmentPublic::decode,
                )
                .unwrap()
            } else {
                Envelope::open(
                    &fixture_bytes(true, "weights.twr"),
                    WeightCommitmentPublic::EXPECTED,
                    &limits,
                    |b| hash(Domain::Envelope, b),
                )
                .unwrap()
                .decode_payload(|c, _| WeightCommitmentPublic::decode(c))
                .unwrap()
            };
            let validated = tv_core::validation::validate(&claims).unwrap();
            let schedule =
                tv_core::leaf::ModelSchedule::derive(validated.graph(), claims.gadget_policy)
                    .unwrap();
            let first_lookup = schedule.lookups()[0].0;
            Fixture {
                claims,
                statement,
                lookup_rejection: RejectReason::Lookup {
                    table: first_lookup,
                },
                config: VerifierConfig {
                    params: ParamSet::TestSmall,
                    weights,
                    accept_bound: FixedBits(0),
                    limits: ResourceLimits {
                        max_artifact_bytes: 2 << 20,
                        max_memory_bytes: 64 << 20,
                        max_wall_time: Duration::from_secs(5),
                    },
                },
            }
        })
    })[which]
}

/// Header low nibble: raw NI envelope (0), raw payload (1), known valid (2), wrong
/// layout (3), zero lookup root (4), wrong stage final (5), wrong advice row (6),
/// zero memory (7), or expired deadline (8). Bit six selects the tamper base; raw modes use bits four/five for memory.
/// Arbitrary raw acceptance is not automatically a forgery: another nonce may be valid.
/// Mode nine replays a declared NI tamper case by two-byte index. Known-false controls require exact rejection. Returns true on Accept.
pub fn check(data: &[u8]) -> bool {
    let Some((&header, bytes)) = data.split_first() else {
        return false;
    };
    let mode = (header & 15) % 10;
    if mode == 9 {
        return tamper_case(bytes);
    }
    let f = fixture(usize::from(header & 64 != 0));
    let mut config = f.config.clone();
    let mut expected = None;
    let verdict = if mode == 0 {
        config.limits.max_memory_bytes =
            [0, 64 << 10, 1 << 20, 64 << 20][usize::from(header >> 4) % 4];
        tv_core::verify_non_interactive_bytes(&f.claims, &f.statement, bytes, &config)
    } else {
        let limits = DecodeLimits {
            max_total_bytes: 2 << 20,
            max_vec_len: 1 << 18,
            max_depth: 32,
        };
        let mut proof = if mode == 1 {
            match decode(bytes, &limits, NiProof::decode) {
                Ok(value) => value,
                Err(error) => {
                    assert!(error.offset() <= bytes.len() as u64);
                    assert!(
                        error.code().is_input_fault()
                            || error.code() == tv_codec::ErrorCode::Memory
                    );
                    return false;
                }
            }
        } else {
            NiProof::open(
                if header & 64 == 0 { golden() } else { tamper() },
                &DecodeLimits::MESSAGE,
            )
            .unwrap()
        };
        match mode {
            3 => {
                proof.round1.advice.layout_digest[0] ^= 1;
                expected = Some(RejectReason::StatementMismatch {
                    field: StatementField::LayoutDigest,
                });
            }
            4 => {
                // Zero both n and d: preserve the cross-multiplied identity while
                // violating only the lookup side's nonzero denominator (RFC-0008).
                proof.round2.logup.roots[0] = Fp2::ZERO;
                proof.round2.logup.roots[1] = Fp2::ZERO;
                expected = Some(f.lookup_rejection.clone());
            }
            5 => {
                let stage = &mut proof.round2.stages[0];
                stage.sumcheck.final_evals[0] += Fp2::ONE;
                expected = Some(RejectReason::Sumcheck {
                    stage: stage.stage,
                    round: stage.sumcheck.rounds.len() as u32,
                });
            }
            6 => {
                proof.round2.advice_combined[0].u[0] += Fp2::ONE;
                expected = Some(RejectReason::Evaluation {
                    commitment: CommitmentClass::Advice,
                });
            }
            7 => config.limits.max_memory_bytes = 0,
            8 => config.limits.max_wall_time = Duration::ZERO,
            _ => {}
        }
        tv_core::verify_non_interactive(&f.claims, &f.statement, &proof, &config)
    };
    if let Some(reason) = expected {
        assert!(
            matches!(&verdict, Verdict::Reject(got) if got == &reason),
            "{verdict:?}, expected {reason:?}"
        );
    } else if mode == 7 || mode == 8 {
        let code = if mode == 7 {
            ErrorCode::Memory
        } else {
            ErrorCode::Time
        };
        assert!(
            matches!(&verdict, Verdict::Inconclusive(error) if error.code == code),
            "{verdict:?}"
        );
    } else if mode == 2 {
        assert!(matches!(&verdict, Verdict::Accept(_)), "{verdict:?}");
    }
    match verdict {
        Verdict::Accept(report) => {
            assert_eq!(report.claim_list_digest, f.claims.digest());
            assert_eq!(report.statement_digest, f.statement.digest());
            assert_eq!(report.weights_root, f.statement.weights_root);
            assert_eq!(report.manifest_digest, f.statement.manifest_digest);
            assert_eq!(report.batch_uhat.0, 0);
            assert_eq!(report.requests.len(), f.statement.batch.requests.len());
            for ((id, bound), request) in report.requests.iter().zip(&f.statement.batch.requests) {
                assert_eq!(*id, request.request_id);
                assert_eq!(bound.0, 0);
            }
            true
        }
        Verdict::Inconclusive(error) => {
            assert_ne!(error.code, ErrorCode::Internal);
            false
        }
        Verdict::Reject(_) => false,
    }
}

/// Number of declared NI mutation cases replayed by the complete-driver oracle.
pub fn tamper_cases() -> Vec<usize> {
    let corpus = tv_testkit::read_vectors("tamper/protocol-1.json");
    corpus
        .get("mutations")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            c.get("edits")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e.get("target").unwrap().as_str() == Some("ni.tnp"))
                .then_some(i)
        })
        .collect()
}

fn tamper_case(input: &[u8]) -> bool {
    use tv_testkit::{Json, hex};
    static CORPUS: OnceLock<Json> = OnceLock::new();
    let corpus = CORPUS.get_or_init(|| tv_testkit::read_vectors("tamper/protocol-1.json"));
    let cases = corpus.get("mutations").unwrap().as_array().unwrap();
    let selector = usize::from(u16::from_le_bytes([
        input.first().copied().unwrap_or(0),
        input.get(1).copied().unwrap_or(0),
    ])) % cases.len();
    let case = &cases[selector];
    let edits = case.get("edits").unwrap().as_array().unwrap();
    if edits
        .iter()
        .any(|e| e.get("target").unwrap().as_str() != Some("ni.tnp"))
    {
        return false;
    }
    let reseal = case.get("reseal") == Some(&Json::Bool(true));
    let mut bytes = if reseal {
        tamper()[20..tamper().len() - 32].to_vec()
    } else {
        tamper().to_vec()
    };
    for edit in edits {
        let offset = usize::try_from(edit.get("offset").unwrap().as_u64().unwrap()).unwrap();
        if let Some(shift) = edit.get("shift_fp2") {
            let count = usize::try_from(shift.get("count").unwrap().as_u64().unwrap()).unwrap();
            let delta = shift.get("delta").unwrap().as_array().unwrap();
            const P: u128 = 0xffff_ffff_0000_0001;
            for (i, word) in bytes[offset..offset + count * 16]
                .chunks_exact_mut(8)
                .enumerate()
            {
                let x = u64::from_le_bytes(word.try_into().unwrap());
                let d = delta[i % 2].as_u64().unwrap();
                word.copy_from_slice(&(((u128::from(x) + u128::from(d)) % P) as u64).to_le_bytes());
            }
        } else {
            let before = hex(edit.get("before").unwrap().as_str().unwrap());
            let after = hex(edit.get("after").unwrap().as_str().unwrap());
            assert_eq!(&bytes[offset..offset + before.len()], before);
            bytes.splice(offset..offset + before.len(), after);
        }
    }
    if reseal {
        let mut sealed = tamper()[..12].to_vec();
        sealed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        sealed.extend_from_slice(&bytes);
        sealed.extend_from_slice(&hash(Domain::Envelope, &sealed));
        bytes = sealed;
    }
    let f = fixture(1);
    let mut config = f.config.clone();
    config.limits = ResourceLimits::default();
    let verdict = tv_core::verify_non_interactive_bytes(&f.claims, &f.statement, &bytes, &config);
    let report = Json::parse(&verdict.to_json().unwrap()).unwrap();
    assert_eq!(
        report.get("verdict"),
        case.get("expected"),
        "tamper case {selector}"
    );
    assert!(
        !matches!(verdict, Verdict::Accept(_)),
        "a known-false tamper case accepted"
    );
    false
}
