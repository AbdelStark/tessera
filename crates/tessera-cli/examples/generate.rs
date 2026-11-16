// SPDX-License-Identifier: Apache-2.0 OR MIT

//! D's exact-greedy continuation of a prompt (MVP workload, #727, #741): the CPU witness engine,
//! the bit-exact oracle the prover checks its tokens against, run once per new token on the
//! whole sequence. Writes the TokenFile `tessera prove --reference-tokens` reads and prints the
//! prompt length, then the continuation as a JSON list.
//!
//! ```text
//! cargo run --release -p tessera-cli --example generate -- MODEL.d '[128000, 791]' NEW OUT.ttk [THREADS]
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;

use tessera_codec::{DecodeLimits, Secret, envelope, schema};
use tessera_compiler::PublicModel;
use tessera_ir::{IntegerModel, TokenFile};
use tessera_witness::Engine;

fn run(args: &[String]) -> Result<(usize, Vec<u32>), String> {
    let [model, prompt, new, out, rest @ ..] = args else {
        return Err("usage: generate MODEL.d PROMPT_IDS_JSON NEW OUT.ttk [THREADS]".into());
    };
    let threads = rest
        .first()
        .map(|t| t.parse::<usize>().map_err(|e| e.to_string()))
        .transpose()?
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(NonZeroUsize::get)
        })
        .and_then(NonZeroUsize::new)
        .unwrap_or(NonZeroUsize::MIN);
    let new: usize = new.parse().map_err(|_| "NEW is a count")?;
    let mut ids: Vec<u32> = prompt
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|w| !w.trim().is_empty())
        .map(|w| w.trim().parse().map_err(|_| format!("not a token id: {w}")))
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("the prompt is nonempty".into());
    }
    let bytes = std::fs::read(model).map_err(|e| format!("{model}: {e}"))?;
    let model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
            .map_err(|e| format!("{model}: {e:?}"))?;
    let prompt = ids.len();
    for step in 0..new {
        let started = std::time::Instant::now();
        let n = u32::try_from(ids.len()).map_err(|_| "sequence length")?;
        let lowering = tessera_compiler::lower(PublicModel::of(&model), n)
            .map_err(|e| format!("lowering: {e:?}"))?;
        let engine = Engine::new(&lowering)
            .map_err(|e| format!("engine: {e:?}"))?
            .threads(threads);
        let witness = engine
            .run(&model, &Secret::new(ids.clone()))
            .map_err(|e| format!("witness: {e:?}"))?;
        let logits = witness.logits().ok_or("a model run has logits")?;
        let last = logits.positions.last().ok_or("a run has a position")?;
        // RFC-0009 Lemma 9: the earliest maximum.
        let (best, _) =
            last.iter().enumerate().fold(
                (0usize, i64::MIN),
                |(b, m), (i, &v)| if v > m { (i, v) } else { (b, m) },
            );
        ids.push(u32::try_from(best).map_err(|_| "token id")?);
        eprintln!(
            "generate: token {}/{new} in {:.1} s",
            step + 1,
            started.elapsed().as_secs_f64()
        );
    }
    envelope::write_file(
        Path::new(out),
        &schema::TOKEN_FILE,
        &TokenFile {
            sequences: vec![ids.clone()],
        },
    )
    .map_err(|e| format!("{out}: {e}"))?;
    Ok((prompt, ids.split_off(prompt)))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok((prompt, continuation)) => {
            let list: Vec<String> = continuation.iter().map(u32::to_string).collect();
            println!("{prompt}");
            println!("[{}]", list.join(", "));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("generate: {e}");
            ExitCode::FAILURE
        }
    }
}
