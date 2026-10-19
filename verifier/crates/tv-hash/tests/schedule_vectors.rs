// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-hash`'s transcript against `spec/vectors/transcript-schedule/schedule.json` (RFC-0014
//! 'Transcript schedule' and 'Grinding'), written by an independent Python oracle: every case's
//! context and its TCE encoding, then every step the oracle played, under its label, as the
//! verifier checks it, and the state and counter it ends in:
//!
//! - an absorb takes the step's bytes; a live session's `seed` is the coin toss of the case's
//!   shares, `H("coin-toss", s_V ‖ s_P)`;
//! - a challenge, or each coordinate of a proximity point, is drawn in `Fp2`; the
//!   non-interactive seed is squeezed; the column indices are drawn below `2^log_n`;
//! - a proof of work's nonce passes `check_grind` with the step's bits, and is the smallest that
//!   does, as the honest prover's must be.
//!
//! The labels and their order are the file's. Expanding the schedule from a claim list is the
//! verifier's own derivation (#114, #128), which these vectors will check too.

mod support;

use support::{digest_hex, list, number, text};
use tv_field::Fp;
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_hash::{Domain, hash};
use tv_testkit::{Json, hex, read_vectors};

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

/// Plays `steps` on a new transcript of `context`, every check passing.
fn replay(context: &SessionContext<'_>, steps: &[Json]) -> Transcript {
    let mut transcript = Transcript::new(context);
    for step in steps {
        match text(step, "op") {
            "absorb" => transcript.absorb(label(step), &hex(text(step, "tce"))),
            "digest" => {
                transcript.squeeze(label(step));
            }
            "fp2" => {
                for _ in 0..number(step, "count") {
                    transcript.challenge_fp2(label(step)).unwrap();
                }
            }
            "indices" => {
                let (log_n, count) = (number(step, "log_n"), number(step, "count"));
                transcript
                    .challenge_indices(label(step), log_n as u32, count as u32)
                    .unwrap();
            }
            "grind" => {
                let bits = number(step, "bits") as u8;
                transcript = transcript
                    .check_grind(label(step), bits, number(step, "nonce"))
                    .unwrap();
            }
            op => panic!("unknown op {op}"),
        }
    }
    transcript
}

#[test]
fn every_step_of_every_schedule() {
    let file = read_vectors("transcript-schedule/schedule.json");
    let cases: Vec<&Json> = list(&file, "cases")
        .iter()
        .chain(list(&file, "models"))
        .collect();
    assert!(cases.len() >= 16, "the cases and four models in two modes");
    let (mut steps_checked, mut proofs_of_work) = (0, 0);
    for case in cases {
        let name = text(case, "name");
        let context = context_of(case.get("context").unwrap());
        assert_eq!(context.to_tce(), hex(text(case, "context_tce")), "{name}");
        let steps = list(case, "steps");
        let replayed = number(case, "replayed") as usize;
        assert!(replayed <= steps.len(), "{name}");
        let mut transcript = Transcript::new(&context);
        for (at, step) in steps[..replayed].iter().enumerate() {
            let what = format!("{name}, step {at} ({})", text(step, "label"));
            match text(step, "op") {
                "absorb" => {
                    let tce = hex(text(step, "tce"));
                    if text(step, "label") == "seed" {
                        let coin = case.get("coin_toss").unwrap();
                        let shares = [
                            hex(text(coin, "verifier_share")),
                            hex(text(coin, "prover_share")),
                        ]
                        .concat();
                        assert_eq!(tce, hash(Domain::CoinToss, &shares), "{what}");
                    }
                    transcript.absorb(label(step), &tce);
                }
                "digest" => {
                    let out = transcript.squeeze(label(step));
                    assert_eq!(out, digest_hex(text(step, "out")), "{what}");
                }
                "fp2" => {
                    let values = list(step, "values");
                    assert_eq!(values.len() as u64, number(step, "count"), "{what}");
                    for value in values {
                        let drawn = transcript.challenge_fp2(label(step)).unwrap();
                        let pair = value.as_array().unwrap();
                        assert_eq!((drawn.c0, drawn.c1), (fp(&pair[0]), fp(&pair[1])), "{what}");
                    }
                }
                "indices" => {
                    let (log_n, count) = (number(step, "log_n"), number(step, "count"));
                    let indices = transcript
                        .challenge_indices(label(step), log_n as u32, count as u32)
                        .unwrap();
                    let expected: Vec<u64> = list(step, "values")
                        .iter()
                        .map(|v| v.as_str().unwrap().parse().unwrap())
                        .collect();
                    assert_eq!(indices, expected, "{what}");
                    assert!(indices.iter().all(|&i| i < 1 << log_n), "{what}");
                }
                "grind" => {
                    let bits = number(step, "bits") as u8;
                    let nonce = number(step, "nonce");
                    // The honest prover's nonce is the smallest that passes.
                    for smaller in 0..nonce {
                        let before = replay(&context, &steps[..at]);
                        assert!(
                            before.check_grind(label(step), bits, smaller).is_err(),
                            "{what}: nonce {smaller} passes too"
                        );
                    }
                    transcript = transcript
                        .check_grind(label(step), bits, nonce)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    proofs_of_work += u64::from(bits > 0);
                }
                op => panic!("{what}: unknown op {op}"),
            }
            steps_checked += 1;
        }
        assert_eq!(
            *transcript.state(),
            digest_hex(text(case, "state")),
            "{name}: state"
        );
        assert_eq!(transcript.counter(), number(case, "counter"), "{name}");
    }
    assert!(steps_checked > 7000, "{steps_checked} steps");
    assert!(proofs_of_work >= 4, "proofs of work of more than 0 bits");
}
