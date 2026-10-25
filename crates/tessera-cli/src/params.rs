// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera params check`: a parameter set's soundness report (RFC-0014 'Soundness calculator').
//!
//! It prints every row of RFC-0014's soundness table with its bits, a lower bound rounded down,
//! its target and whether it reaches it, then the minimum. With `--json` it prints the
//! `tessera.soundness/1` report (`spec/schemas/soundness-v1.schema.json`), whose integers are
//! exact: the bits in FixedBits, units of `2^-8` bits. When any row misses its target
//! (`meets_targets == Some(false)`), it still prints the report and exits 10 (02 §2.4, RFC-0051);
//! when every row meets its target or the parameter set has no targets (`test-small`), it exits 0.
//!
//! The sizes are the 405B 16K-token, 16K-context statement,
//! `StatementSizes::llama_405b_16k`. `--lookups` and `--table-rows` replace its LogUp
//! instances with one instance of `N` lookups and `T` rows; the leaf stage's levels then follow
//! the tallest tree, `⌈log2 max(N, T)⌉` levels when above the default's 42 (RFC-0014).

use std::io::{self, Write as _};
use std::process::ExitCode;

use clap::Args;
use serde_json::{Map, Value, json};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_protocol::{
    Commitment, ErrorBound, FixedBits, LookupSizes, ParamMode, ParamSet, ProofOfWork, Rounds,
    RowKind, SoundnessReport, SoundnessRow, StatementSizes,
};

use crate::{PARAM_SETS, io_error};

/// The report's schema identifier (03 §5).
const SCHEMA: &str = "tessera.soundness/1";

/// What the default sizes are, for the command's help.
const DEFAULTS: &str = "\
The sizes are the 405B 16K-token, 16K-context statement under tessera-protocol/1 (RFC-0014 \
'Soundness calculator'): Llama-3.1-405B, servable, one batch of 16,384 tokens at 16,384 \
positions of context. Each size comes from RFC-0014, the compiler's 405B commitment report or \
the architecture; where none gives a count, a bound:
  rounds         10,000 of degree 7, RFC-0014's \"about 10^4 rounds\" of \"degree d <= 7\"
  stages         2 reduction stages batching at most 64 instances each
  tree levels    42, the Exp lookup tree's height, each batching at most 31 instances
  zero-checks    8 of at most 42 variables
  forests        the SoftmaxRow forest of 2^28 trees
  LogUp          the report's 15 instances at 16K tokens, Exp's over the padded score tensor:
                 2^42 lookups into 2^22 rows (RFC-0030)
  claim groups   16 of at most 64 claims
  advice rows    2^21, a 21-bit row index
  matrices       the ten weight matrices of RFC-0040 at k_W = 2^20

--lookups N and --table-rows T replace the LogUp instances with one of N lookups and T rows; \
a flag left out takes the Exp instance's. The bits shown are lower bounds, rounded down; the \
passes and fails are exact.";

/// `tessera params check`.
#[derive(Debug, Args)]
#[command(after_help = DEFAULTS)]
pub(crate) struct CheckArgs {
    /// The parameter set (RFC-0014 'Parameter sets').
    #[arg(long, value_name = "PARAMSET", value_parser = PARAM_SETS)]
    params: String,
    /// Count one LogUp instance of N lookups [default: 2^42, the Exp instance's].
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    lookups: Option<u64>,
    /// Give that instance's table T rows [default: 2^22, the Exp table's].
    #[arg(long, value_name = "T", value_parser = clap::value_parser!(u64).range(1..))]
    table_rows: Option<u64>,
}

/// Exit status when a soundness row misses its target (02 §2.4, RFC-0051).
const EXIT_TARGET_MISSED: u8 = 10;

/// Runs `tessera params check`; with `json`, the report is the result printed on stdout.
pub(crate) fn check(args: &CheckArgs, json: bool) -> Result<ExitCode, TesseraError> {
    let Some(params) = ParamSet::ALL
        .into_iter()
        .find(|set| set.name() == args.params)
    else {
        return Err(TesseraError::new(ErrorCode::Usage)
            .with_source(format!("{} is not a parameter set", args.params)));
    };
    let mut sizes = StatementSizes::llama_405b_16k();
    let replaced = args.lookups.is_some() || args.table_rows.is_some();
    if replaced {
        let largest = sizes
            .lookups
            .iter()
            .copied()
            .max_by_key(|l| u128::from(l.lookups) + u128::from(l.rows))
            .unwrap_or(LookupSizes {
                lookups: 1,
                rows: 1,
            });
        let instance = LookupSizes {
            lookups: args.lookups.unwrap_or(largest.lookups),
            rows: args.table_rows.unwrap_or(largest.rows),
        };
        sizes.lookups = vec![instance];
        // RFC-0014: the leaf stage's levels follow the tallest tree, ⌈log2 max(N, T)⌉ when above
        // the default's, with their rounds and the merge's n_max (RFC-0054 'Soundness').
        let tallest = instance.lookups.max(instance.rows);
        let levels = u64::BITS - (tallest - 1).leading_zeros();
        if levels as usize > sizes.tree_levels.len() {
            let instances = sizes.tree_levels.iter().copied().max().unwrap_or(0);
            sizes.tree_levels = vec![instances; levels as usize];
            if sizes.groups.grouped() {
                sizes.groups.level_rounds = vec![Rounds {
                    count: u64::from(levels) * u64::from(levels - 1) / 2,
                    degree: 3,
                }];
                sizes.groups.merge_rounds = sizes.groups.merge_rounds.max(levels);
            }
        }
    }
    let report = SoundnessReport::new(params, &sizes)
        .map_err(|error| TesseraError::from(error).with_source(error))?;
    let shown = match json {
        true => {
            let text = serde_json::to_string_pretty(&report_json(&report)).map_err(|_| {
                TesseraError::new(ErrorCode::Internal).with_source("the report has no JSON form")
            })?;
            format!("{text}\n")
        }
        false => table(&report, replaced),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))?;
    match report.meets_targets() {
        Some(false) => Ok(ExitCode::from(EXIT_TARGET_MISSED)),
        Some(true) | None => Ok(ExitCode::SUCCESS),
    }
}

fn commitment(c: Commitment) -> &'static str {
    match c {
        Commitment::Advice => "advice",
        Commitment::Weights(matrix) => matrix.name(),
    }
}

/// A row's name in the report, and its commitment's.
fn names(kind: RowKind) -> (&'static str, Option<&'static str>) {
    match kind {
        RowKind::Rounds => ("rounds", None),
        RowKind::Union => ("union", None),
        RowKind::Forests => ("forests", None),
        RowKind::LogUp => ("logup", None),
        RowKind::Openings => ("openings", None),
        RowKind::Queries(c) => ("queries", Some(commitment(c))),
        RowKind::Field(c) => ("field", Some(commitment(c))),
        RowKind::Linking => ("linking", None),
        RowKind::Hash => ("hash", None),
    }
}

fn mode(mode: ParamMode) -> &'static str {
    match mode {
        ParamMode::Live => "live",
        ParamMode::NonInteractive => "non-interactive",
        ParamMode::Either => "either",
    }
}

/// The `tessera.soundness/1` report.
fn report_json(report: &SoundnessReport) -> Value {
    let p = &report.params;
    let (advice, weights) = (p.layout.advice, p.layout.weights);
    let s = &report.sizes;
    json!({
        "schema": SCHEMA,
        "params": {
            "id": p.id,
            "mode": mode(p.mode),
            "release": p.release,
            "advice": {
                "log2_k": advice.log2_k,
                "log2_inv_rate": advice.log2_inv_rate,
                "mask_coeffs": advice.mask_coeffs,
                "row_block": advice.row_block,
                "segments": advice.segments,
            },
            "weights": {
                "log2_k": weights.log2_k,
                "log2_inv_rate": weights.log2_inv_rate,
                "mask_coeffs": weights.mask_coeffs,
                "blinding_rows": weights.blinding_rows,
            },
            "advice_queries": p.advice_queries,
            "weight_queries": p.weight_queries,
            "seed_pow": p.seed_pow,
            "query_pow": p.query_pow,
            "targets": p.targets.map(|t| json!({"bits": t.bits, "one_shot": t.one_shot})),
        },
        "statement": {
            "protocol": s.protocol,
            "rounds": s.rounds.iter()
                .map(|r| json!({"count": r.count, "degree": r.degree}))
                .collect::<Vec<_>>(),
            "stages": s.stages,
            "tree_levels": s.tree_levels,
            "zero_checks": s.zero_checks,
            "layer_vars": s.layer_vars,
            "forests": s.forests,
            "lookups": s.lookups.iter()
                .map(|l| json!({"lookups": l.lookups, "rows": l.rows}))
                .collect::<Vec<_>>(),
            "claim_groups": s.claim_groups,
            "advice_rows": s.advice_rows,
            "matrices": s.matrices.iter()
                .map(|m| json!({
                    "matrix": m.matrix.name(),
                    "width": m.width,
                    "rows": m.rows,
                    "variables": m.variables,
                    "claims": m.claims,
                }))
                .collect::<Vec<_>>(),
            "groups": {
                "count": s.groups.count,
                "level_rounds": s.groups.level_rounds.iter()
                    .map(|r| json!({"count": r.count, "degree": r.degree}))
                    .collect::<Vec<_>>(),
                "merge_rounds": s.groups.merge_rounds,
                "merge_claims": s.groups.merge_claims,
            },
        },
        "rows": report.rows.iter().map(row_json).collect::<Vec<_>>(),
        "minimum": report.minimum(),
        "meets_targets": report.meets_targets(),
    })
}

fn row_json(row: &SoundnessRow) -> Value {
    let mut object = Map::new();
    let (name, commitment) = names(row.kind);
    object.insert("row".into(), name.into());
    if let Some(commitment) = commitment {
        object.insert("commitment".into(), commitment.into());
    }
    match row.bound {
        // A numerator reaches 2^128, past what JSON numbers hold exactly: a decimal string.
        ErrorBound::Field { numerator } => {
            object.insert("numerator".into(), numerator.to_string().into());
        }
        ErrorBound::Queries {
            codeword,
            message,
            mask,
            queries,
        } => {
            object.insert("codeword".into(), codeword.into());
            object.insert("message".into(), message.into());
            object.insert("mask".into(), mask.into());
            object.insert("queries".into(), queries.into());
        }
        ErrorBound::Hash => {}
    }
    if let Some(theta) = row.theta {
        object.insert("theta".into(), theta.into());
    }
    if let Some(per_query) = row.per_query {
        object.insert("per_query_bits".into(), per_query.0.into());
    }
    let covered = row.covered_by.map(|pow| match pow {
        ProofOfWork::Seed => "seed-pow",
        ProofOfWork::Query => "query-pow",
    });
    object.insert("bits".into(), json!(row.bits.map(|b| b.0)));
    object.insert("grinding".into(), row.grinding.into());
    object.insert("covered_by".into(), json!(covered));
    object.insert("target".into(), json!(row.target));
    object.insert("margin".into(), json!(row.margin()));
    object.insert("pass".into(), json!(row.pass));
    Value::Object(object)
}

/// A row's name in the table.
fn label(kind: RowKind) -> String {
    match kind {
        RowKind::Rounds => "sumcheck and tree rounds".to_owned(),
        RowKind::Union => "all rounds, union bound".to_owned(),
        RowKind::Forests => "product-forest root claims".to_owned(),
        RowKind::LogUp => "LogUp identity".to_owned(),
        RowKind::Openings => "opening evaluation checks".to_owned(),
        RowKind::Queries(c) => format!("proximity queries, {}", commitment(c)),
        RowKind::Field(c) => format!("proximity field term, {}", commitment(c)),
        RowKind::Linking => "weight linking".to_owned(),
        RowKind::Hash => "hash collisions".to_owned(),
    }
}

/// Bits with two decimals, rounded down, so the table never shows more than the bound.
fn shown(bits: FixedBits) -> String {
    format!("{}.{:02}", bits.0 / 256, bits.0 % 256 * 100 / 256)
}

/// The report as a table, one line per row, then the minimum and the verdict.
fn table(report: &SoundnessReport, replaced: bool) -> String {
    let p = &report.params;
    let modes = match p.mode {
        ParamMode::Live => "live sessions",
        ParamMode::NonInteractive => "non-interactive proofs",
        ParamMode::Either => "either mode",
    };
    let accepted = match p.release {
        true => "accepted by a release verifier",
        false => "accepted only with the test-params feature",
    };
    let mut text = format!("{}: {modes}, {accepted}\n", p.id);
    let instance = match (replaced, report.sizes.lookups.first()) {
        (true, Some(l)) => format!(
            ", with one LogUp instance of {} lookups and {} rows",
            l.lookups, l.rows
        ),
        _ => String::new(),
    };
    text.push_str(&format!(
        "statement: tessera-protocol/{}, the 405B 16K-token, 16K-context statement{instance}\n\n",
        report.sizes.protocol
    ));
    text.push_str(&format!(
        "{:<36} {:>8}  {:>6}  {:<14} {}\n",
        "row", "bits", "target", "grinding", "pass"
    ));
    for row in &report.rows {
        let bits = row.bits.map_or_else(|| "none".to_owned(), shown);
        let target = row.target.map_or_else(|| "-".to_owned(), |t| t.to_string());
        let grinding = match row.covered_by {
            Some(ProofOfWork::Seed) => format!("seed-pow +{}", row.grinding),
            Some(ProofOfWork::Query) => format!("query-pow +{}", row.grinding),
            None => "-".to_owned(),
        };
        let pass = match row.pass {
            Some(true) => "yes",
            Some(false) => "NO",
            None => "-",
        };
        text.push_str(&format!(
            "{:<36} {bits:>8}  {target:>6}  {grinding:<14} {pass}\n",
            label(row.kind)
        ));
    }
    text.push('\n');
    if let Some(row) = report.minimum().map(|i| &report.rows[i]) {
        let bits = row.bits.map_or_else(|| "none".to_owned(), shown);
        let target = row
            .target
            .map_or_else(String::new, |t| format!(", target {t}"));
        text.push_str(&format!(
            "minimum: {}: {bits} bits{target}\n",
            label(row.kind)
        ));
    }
    let missed = report
        .rows
        .iter()
        .filter(|row| row.pass == Some(false))
        .count();
    let verdict = match (report.meets_targets(), missed) {
        (None, _) => format!("{} has no targets: it certifies nothing", p.id),
        (Some(true), _) => "every row meets its target".to_owned(),
        (Some(false), 1) => "1 row misses its target".to_owned(),
        (Some(false), n) => format!("{n} rows miss their targets"),
    };
    text.push_str(&verdict);
    text.push('\n');
    text
}
