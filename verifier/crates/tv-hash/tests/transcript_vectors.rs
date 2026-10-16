// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-hash`'s transcript against `spec/vectors/transcript/transcript.json` (RFC-0004
//! 'Testing Strategy'), written by an independent Python oracle: every session's context and
//! TCE encoding, and every step's value, state and counter.
//!
//! The vectors play the prover's side too. A `grind` step is replayed as the verifier sees it, a
//! `check_grind` of the nonce found, and the nonce is checked to be the smallest that passes, as
//! the honest prover's must be. A failed `check_grind` ends the transcript in this crate, so the
//! replay rebuilds it from the steps before, and checks that the failure changed nothing.

mod support;

use support::{digest_hex, list, number, text};
use tv_field::Fp;
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::{Json, hex, read_vectors};

/// The session context of a case.
fn context_of(context: &Json) -> SessionContext<'_> {
    SessionContext {
        protocol: Ascii::new(text(context, "protocol")).unwrap(),
        mode: match text(context, "mode") {
            "Live" => Mode::Live,
            "NonInteractive" => Mode::NonInteractive,
            other => panic!("unknown mode {other}"),
        },
        params: ParamSetId(Ascii::new(text(context, "params")).unwrap()),
        statement_digest: digest_hex(text(context, "statement_digest")),
        session_id: hex(text(context, "session_id")).try_into().unwrap(),
    }
}

fn label(step: &Json) -> Ascii<'_> {
    Ascii::new(text(step, "label")).unwrap()
}

/// A canonical value from a decimal string.
fn fp(value: &Json) -> Fp {
    Fp::from_canonical(value.as_str().unwrap().parse().unwrap()).unwrap()
}

/// Plays `steps` on `transcript` as the verifier does, every check passing. A `check_grind` that
/// the vectors show failing is skipped: it leaves the state unchanged.
fn replay(mut transcript: Transcript, steps: &[Json]) -> Transcript {
    for step in steps {
        match text(step, "op") {
            "absorb" => transcript.absorb(label(step), &hex(text(step, "tce"))),
            "squeeze" => {
                transcript.squeeze(label(step));
            }
            "challenge_fp" => {
                transcript.challenge_fp(label(step)).unwrap();
            }
            "challenge_fp2" => {
                transcript.challenge_fp2(label(step)).unwrap();
            }
            "challenge_indices" => {
                let log_n = number(step, "log_n") as u32;
                let count = number(step, "count") as u32;
                transcript
                    .challenge_indices(label(step), log_n, count)
                    .unwrap();
            }
            "grind" | "check_grind" => {
                if step.get("valid") != Some(&Json::Bool(false)) {
                    let bits = number(step, "bits") as u8;
                    transcript = transcript
                        .check_grind(label(step), bits, number(step, "nonce"))
                        .unwrap();
                }
            }
            op => panic!("unknown op {op}"),
        }
    }
    transcript
}

/// How many nonces below `nonce` pass the proof of work after `steps`.
fn valid_nonces_below(start: &dyn Fn() -> Transcript, steps: &[Json], step: &Json) -> u64 {
    let bits = number(step, "bits") as u8;
    (0..number(step, "nonce"))
        .filter(|&nonce| {
            replay(start(), steps)
                .check_grind(label(step), bits, nonce)
                .is_ok()
        })
        .count() as u64
}

#[test]
fn every_session_and_every_step_of_the_vectors() {
    let file = read_vectors("transcript/transcript.json");
    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 6, "every session of the vectors");
    let mut steps_checked = 0;
    let mut rejections_reached = 0;
    for case in cases {
        let name = text(case, "name");
        let context = context_of(case.get("context").unwrap());
        assert_eq!(
            context.to_tce(),
            hex(text(case, "context_tce")),
            "{name}: TCE(SessionContext)"
        );
        let start = || Transcript::new(&context);
        let mut transcript = start();
        assert_eq!(
            *transcript.state(),
            digest_hex(text(case, "state")),
            "{name}"
        );
        assert_eq!(transcript.counter(), 0);

        let steps = list(case, "steps");
        for (at, step) in steps.iter().enumerate() {
            let op = text(step, "op");
            let before = transcript.counter();
            let what = format!("{name}, step {at} ({op} {:?})", text(step, "label"));
            match op {
                "absorb" => transcript.absorb(label(step), &hex(text(step, "tce"))),
                "squeeze" => {
                    let out = transcript.squeeze(label(step));
                    assert_eq!(out, digest_hex(text(step, "out")), "{what}");
                }
                "challenge_fp" => {
                    let value = transcript.challenge_fp(label(step)).unwrap();
                    assert_eq!(value, fp(step.get("value").unwrap()), "{what}");
                    // `rejected` counts the words at or above P passed over: replayed, the first
                    // squeeze has that many before the value.
                    let rejected = number(step, "rejected");
                    let out = replay(start(), &steps[..at]).squeeze(label(step));
                    let words: Vec<u64> = out
                        .chunks(8)
                        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                        .collect();
                    let skipped = words.iter().take_while(|&&w| w >= Fp::MODULUS).count();
                    assert_eq!(skipped as u64, rejected, "{what}");
                    assert_eq!(words[skipped], value.to_canonical(), "{what}");
                    rejections_reached += u64::from(rejected > 0);
                }
                "challenge_fp2" => {
                    let value = transcript.challenge_fp2(label(step)).unwrap();
                    let expected = list(step, "value");
                    assert_eq!(value.c0, fp(&expected[0]), "{what}");
                    assert_eq!(value.c1, fp(&expected[1]), "{what}");
                    // Each rejected squeeze is one more squeeze.
                    let rejected = number(step, "rejected");
                    assert_eq!(transcript.counter() - before, 1 + rejected, "{what}");
                    rejections_reached += u64::from(rejected > 0);
                }
                "challenge_indices" => {
                    let log_n = number(step, "log_n") as u32;
                    let count = number(step, "count") as u32;
                    let indices = transcript
                        .challenge_indices(label(step), log_n, count)
                        .unwrap();
                    let expected: Vec<u64> = list(step, "value")
                        .iter()
                        .map(|v| v.as_str().unwrap().parse().unwrap())
                        .collect();
                    assert_eq!(indices, expected, "{what}");
                    assert_eq!(transcript.counter() - before, u64::from(count.div_ceil(4)));
                }
                "grind" => {
                    let bits = number(step, "bits") as u8;
                    let nonce = number(step, "nonce");
                    transcript = transcript
                        .check_grind(label(step), bits, nonce)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    // The honest prover's nonce is the smallest that passes.
                    assert_eq!(valid_nonces_below(&start, &steps[..at], step), 0, "{what}");
                }
                "check_grind" => {
                    let bits = number(step, "bits") as u8;
                    let nonce = number(step, "nonce");
                    let valid = step.get("valid") == Some(&Json::Bool(true));
                    match transcript.check_grind(label(step), bits, nonce) {
                        Ok(next) => {
                            assert!(valid, "{what}: an invalid nonce passed");
                            transcript = next;
                            // A valid nonce that is not the smallest is accepted.
                            assert_eq!(valid_nonces_below(&start, &steps[..at], step), 1, "{what}");
                        }
                        Err(failed) => {
                            assert!(!valid, "{what}: a valid nonce failed: {failed}");
                            assert_eq!(failed.required(), bits);
                            assert!(failed.found() < u32::from(bits));
                            // The transcript is gone; the one rebuilt from the steps before is
                            // the state the vector records: the failure absorbed nothing.
                            transcript = replay(start(), &steps[..at]);
                        }
                    }
                }
                other => panic!("unknown op {other}"),
            }
            assert_eq!(
                *transcript.state(),
                digest_hex(text(step, "state")),
                "{what}: state"
            );
            assert_eq!(
                transcript.counter(),
                number(step, "counter"),
                "{what}: counter"
            );
            steps_checked += 1;
        }
        // Replaying the whole case from scratch reaches the same transcript.
        assert_eq!(replay(start(), steps), transcript, "{name}");
    }
    assert_eq!(steps_checked, 33, "every step of the vectors");
    assert_eq!(
        rejections_reached, 3,
        "the vectors reach each rejection branch"
    );
}

/// Labels and values cannot be confused: a label is TCE `Ascii`, its length then its bytes.
#[test]
fn label_and_value_boundaries_change_the_state() {
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::Live,
        params: ParamSetId(Ascii::new("live-64/1").unwrap()),
        statement_digest: [1; 32],
        session_id: [2; 16],
    };
    let mut a = Transcript::new(&context);
    let mut b = Transcript::new(&context);
    a.absorb(Ascii::new("ab").unwrap(), b"c");
    b.absorb(Ascii::new("a").unwrap(), b"bc");
    assert_ne!(a.state(), b.state());

    // Every field of the context reaches the state.
    let base = Transcript::new(&context);
    let changed = [
        SessionContext {
            protocol: Ascii::new("tessera-protocol/2").unwrap(),
            ..context
        },
        SessionContext {
            mode: Mode::NonInteractive,
            ..context
        },
        SessionContext {
            params: ParamSetId(Ascii::new("ni-100/1").unwrap()),
            ..context
        },
        SessionContext {
            statement_digest: [3; 32],
            ..context
        },
        SessionContext {
            session_id: [4; 16],
            ..context
        },
    ];
    for other in &changed {
        assert_ne!(Transcript::new(other).state(), base.state(), "{other:?}");
    }
}
