// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compare public compiler/verifier constraint systems and locate the first divergence.
mod compiler;
mod fixtures;
mod report;
#[cfg(test)]
mod tests;
mod verifier;
use std::io::Write;

fn run() -> Result<bool, String> {
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    let mut model = None;
    let mut context = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(args.next().ok_or("--model needs a name")?),
            "--n" => {
                context = Some(
                    args.next()
                        .ok_or("--n needs a length")?
                        .parse::<u32>()
                        .map_err(|_| "--n needs an integer length")?,
                )
            }
            "--help" => {
                writeln!(out, "constraint-digest [--model NAME] [--n LENGTH]\nCompare public compiler/verifier metadata and report the first divergent constraint.\nDefault: conformance models, variants and 8B/405B shapes, both profiles, four lengths.").map_err(|e|e.to_string())?;
                return Ok(true);
            }
            _ => return Err(format!("unrecognized argument {arg}; use --help")),
        }
    }
    let mut count = 0;
    for item in fixtures::models()
        .into_iter()
        .filter(|m| model.as_ref().is_none_or(|name| *name == m.name))
    {
        let native: tessera_ir::ClaimList = tessera_codec::envelope::open(
            &item.envelope,
            &tessera_codec::schema::CLAIM_LIST,
            tessera_codec::DecodeLimits::MESSAGE,
        )
        .map_err(|e| format!("compiler input: {e}"))?;
        let tv = tv_core::claims::ClaimList::open(&item.envelope, &tv_codec::DecodeLimits::MESSAGE)
            .map_err(|e| format!("verifier input: {e}"))?;
        for n in context.map_or(item.contexts, |n| vec![n]) {
            if n == 0 || n > native.arch.max_context {
                return Err(format!(
                    "{}: length must be in 1..={}",
                    item.name, native.arch.max_context
                ));
            }
            let expected = compiler::snapshot(&native, n);
            let found = verifier::snapshot(&tv, n);
            let a = compiler::digest(&expected);
            let b = verifier::digest(&found);
            let hex = |v: &[u8; 32]| v.iter().map(|x| format!("{x:02x}")).collect::<String>();
            writeln!(
                out,
                "{} n={n}: compiler={} verifier={}",
                item.name,
                hex(&a),
                hex(&b)
            )
            .map_err(|e| e.to_string())?;
            if let Some(d) = report::difference(&expected, &found) {
                let show = |v: Option<serde_json::Value>| {
                    v.map_or_else(|| "<missing>".to_owned(), |v| v.to_string())
                };
                writeln!(
                    err,
                    "{} n={n}: first divergent constraint at {}: compiler={}, verifier={}",
                    item.name,
                    d.path,
                    show(d.compiler),
                    show(d.verifier)
                )
                .map_err(|e| e.to_string())?;
                return Ok(false);
            }
            if a != b {
                return Err(format!(
                    "{} n={n}: equal constraint trees have unequal digests; diagnostic encoder/hash inconsistency",
                    item.name
                ));
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err("no model matched --model".to_owned());
    }
    writeln!(out, "{count} constraint systems agree").map_err(|e| e.to_string())?;
    Ok(true)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            let _ = writeln!(std::io::stderr().lock(), "{e}");
            std::process::exit(2);
        }
    }
}
