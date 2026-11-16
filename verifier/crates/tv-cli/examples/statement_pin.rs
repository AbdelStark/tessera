// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Prints `STATEMENT_DIGEST WEIGHTS_ROOT` for `tessera-verify verify`, built from public inputs
//! only: the claim list, the public weight commitment and the request's public fields, never
//! from the proof (07 §6). The MVP scripts' request: id 1, session `11`×16, manifest `13`×32,
//! exact greedy.
//!
//! ```text
//! cargo run --release -p tv-cli --example statement_pin -- CLAIMS.tcl WEIGHTS.twr PROMPT OUTPUT
//! ```

// A command-line tool reports on stdout and stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::ExitCode;
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
use tv_core::{claims::ClaimList, protocol::*};
use tv_hash::{Domain, hash};

fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [claims, weights, prompt, output] = args.as_slice() else {
        eprintln!("usage: statement_pin CLAIMS.tcl WEIGHTS.twr PROMPT_LEN OUTPUT_LEN");
        return ExitCode::from(2);
    };
    let (Ok(prompt_len), Ok(output_len)) = (prompt.parse(), output.parse()) else {
        eprintln!("statement_pin: lengths are integers");
        return ExitCode::from(2);
    };
    let limits = DecodeLimits::MESSAGE;
    let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
    let run = || -> Result<String, String> {
        let claims =
            ClaimList::open(&read(claims)?, &limits).map_err(|e| format!("claims: {e:?}"))?;
        let bytes = read(weights)?;
        let weights = Envelope::open(&bytes, WeightCommitmentPublic::EXPECTED, &limits, |b| {
            hash(Domain::Envelope, b)
        })
        .and_then(|e| e.decode_payload(|c, _| WeightCommitmentPublic::decode(c)))
        .map_err(|e| format!("weights: {e:?}"))?;
        let weights_root = weights.weights_root(|b| hash(Domain::WeightsRoot, b));
        let statement = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root,
            weights_epoch: weights.epoch,
            manifest_digest: [0x13; 32],
            batch: BatchDescriptor {
                requests: vec![RequestRef {
                    request_id: 1,
                    session_id: [0x11; 16],
                    prompt_len,
                    output_len,
                }],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        Ok(format!(
            "{} {}",
            hex(&statement.digest()),
            hex(&weights_root)
        ))
    };
    match run() {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("statement_pin: {e}");
            ExitCode::FAILURE
        }
    }
}
