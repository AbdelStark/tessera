// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{CommandOutput, files::Inputs, options::Options};
use std::fmt::Write;
use std::path::Path;
use tv_codec::DecodeLimits;
use tv_core::{
    claims::ClaimList,
    validation::StatementField,
    verdict::{ErrorCode, RejectReason, Verdict},
};
use tv_tables::{Table, TableError};

pub(crate) fn run(args: &[String], json: bool) -> CommandOutput {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        return CommandOutput {exit_code:0,stdout:"tessera-verify tables --claims CLAIMS.tcl [--json] [--max-table-rows N]\nRegenerate and check the table catalog; this accepts no model proof.\nDefault total table-work allowance: 16777216 padded rows.\n".into(),stderr:String::new()};
    }
    let result = (|| {
        let opts = Options::parse(args, &["--claims", "--max-table-rows"])?;
        let allowance = opts.number("--max-table-rows", 1 << 24)?;
        let mut inputs = Inputs {
            remaining: 1 << 30,
            max_artifact: 1 << 30,
        };
        let bytes = inputs.read(Path::new(opts.required("--claims")?))?;
        let limits = DecodeLimits {
            max_total_bytes: 1 << 30,
            ..DecodeLimits::MESSAGE
        };
        let claims = ClaimList::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))?;
        let rows = claims.tables.iter().try_fold(0u64, |n, t| {
            n.checked_add(1u64.checked_shl(u32::from(t.log2_rows))?)
        });
        if rows.is_none_or(|n| n > allowance) {
            return Err(ErrorCode::Time);
        }
        let mut output = if json {
            String::from("{\"schema\":\"tessera.table-digests/1\",\"tables\":[")
        } else {
            String::new()
        };
        for (i, spec) in claims.tables.iter().enumerate() {
            let table = if usize::try_from(spec.id.0).ok() != Some(i) {
                None
            } else {
                match Table::from_spec(spec) {
                    Ok(table) => Some(table),
                    Err(TableError::Memory) => return Err(ErrorCode::Memory),
                    Err(e) if e.is_rejection() => None,
                    Err(_) => return Err(ErrorCode::Internal),
                }
            };
            let Some(table) = table else {
                let verdict = Verdict::Reject(RejectReason::StatementMismatch {
                    field: StatementField::Table { table: spec.id },
                });
                let text = if json {
                    verdict.to_json().map_err(|e| e.code)?
                } else {
                    format!("REJ-STATEMENT: table {}\n", spec.id.0)
                };
                return Ok(CommandOutput {
                    exit_code: 10,
                    stdout: text,
                    stderr: String::new(),
                });
            };
            output.try_reserve(160).map_err(|_| ErrorCode::Memory)?;
            if json {
                if i > 0 {
                    output.push(',');
                }
                write!(
                    output,
                    "{{\"id\":{},\"log2_rows\":{},\"digest\":\"",
                    spec.id.0,
                    table.spec().log2_rows
                )
                .map_err(|_| ErrorCode::Internal)?;
            } else {
                write!(output, "{} ", spec.id.0).map_err(|_| ErrorCode::Internal)?;
            }
            for byte in table.spec().digest {
                write!(output, "{byte:02x}").map_err(|_| ErrorCode::Internal)?;
            }
            output.push_str(if json { "\"}" } else { "\n" });
        }
        if json {
            output.push_str("]}\n");
        }
        Ok(CommandOutput {
            exit_code: 0,
            stdout: output,
            stderr: String::new(),
        })
    })();
    result.unwrap_or_else(|e: ErrorCode| {
        crate::command_error(
            "tables",
            "table catalog could not be evaluated",
            json,
            e.id(),
            e.exit_code(),
        )
    })
}
