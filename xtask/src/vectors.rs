// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask gen-vectors`: writes the codec conformance vectors under `spec/vectors/codec/`
//! from the cases `tessera-codec` and `tessera-field` define (RFC-0001 'Shared artifacts that
//! are not code'). The verifier and the Python package check the same files with their own
//! decoders.
//!
//! `--check` compares instead of writing. A test runs the comparison, so vectors that no longer
//! match the cases fail `cargo test`.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::{Value, json};
use tessera_codec::DecodeLimits;
use tessera_codec::golden::{self, Boundary, Golden, Rejected};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn limits(limits: DecodeLimits) -> Value {
    json!({
        "max_total_bytes": limits.max_total_bytes,
        "max_vec_len": limits.max_vec_len,
        "max_depth": limits.max_depth,
    })
}

/// Every vector file, by path from the repository root, with its contents.
pub(crate) fn render() -> Vec<(&'static str, String)> {
    let cases: Vec<Golden> = golden::cases()
        .into_iter()
        .chain(tessera_field::golden::cases())
        .collect();
    let rejected: Vec<Rejected> = golden::corpus()
        .into_iter()
        .chain(tessera_field::golden::corpus())
        .collect();
    let boundaries: Vec<Boundary> = golden::boundaries()
        .into_iter()
        .chain(tessera_field::golden::boundaries())
        .collect();
    let golden = json!({
        "description": "TCE golden vectors (RFC-0003), written by `cargo xtask gen-vectors`. \
            Decoding `hex` as `type` under `limits` succeeds, gives `value`, and encodes back to \
            `hex`; every proper prefix of `hex` fails with ERR-CODEC-TRUNCATED. Values: integers \
            and field elements are decimal strings, `Fp2` is [c0, c1], `Option` is null or \
            {\"some\": value}, an enum value is {\"Variant\": fields}, a map is its entries in \
            canonical order as [key, value] pairs. `Fp` is Goldilocks, `Fp2` is \
            Fp[X]/(X^2 - 7), `Digest` is [u8; 32] and `Bytes` is Vec<u8>.",
        "limits": limits(golden::GOLDEN_LIMITS),
        "schemas": golden::SCHEMAS
            .iter()
            .map(|(name, schema)| ((*name).to_owned(), Value::from(*schema)))
            .collect::<serde_json::Map<_, _>>(),
        "cases": cases
            .iter()
            .map(|c| {
                let value: Value = serde_json::from_str(&c.value.json()).unwrap_or(Value::Null);
                json!({ "type": c.ty, "value": value, "hex": hex(&c.bytes) })
            })
            .collect::<Vec<_>>(),
    });
    let corpus = json!({
        "description": "TCE non-canonical corpus (RFC-0003 'Failure modes', 'Limits'), written \
            by `cargo xtask gen-vectors`. Under `limits`, decoding each `cases` entry as `type` \
            fails with `code`, each for one reason; decoding each `accepted` entry, which sits \
            exactly at a limit, succeeds and encodes back to `hex`.",
        "limits": limits(golden::CORPUS_LIMITS),
        "accepted": boundaries
            .iter()
            .map(|b| json!({ "type": b.ty, "hex": hex(&b.bytes), "why": b.why }))
            .collect::<Vec<_>>(),
        "cases": rejected
            .iter()
            .map(|r| json!({
                "type": r.ty,
                "hex": hex(&r.bytes),
                "code": r.code.as_str(),
                "why": r.why,
            }))
            .collect::<Vec<_>>(),
    });
    let envelope = json!({
        "description": "Envelope vectors (RFC-0003 'Envelope'), written by `cargo xtask \
            gen-vectors`. A reader expects schema `schema` whose writers emit version `version`, \
            so it accepts `version` and `version - 1` but never 0. It reads under `limits`, or \
            under the RFC-0003 defaults a case names, which every implementation takes from its \
            own constants. It checks `hex` in the order RFC-0003 fixes, then decodes the payload \
            as `type`, which must fill it. It returns `payload`, or fails with `code` at `offset`, \
            counted from the start of the envelope. A rejection has one fault, except the cases \
            that pin which of two faults is reported.",
        "limits": limits(golden::ENVELOPE_LIMITS),
        "named_limits": {
            "message": limits(DecodeLimits {
                max_total_bytes: golden::RFC_MESSAGE_TOTAL,
                max_vec_len: golden::RFC_MESSAGE_VEC_LEN,
                max_depth: golden::RFC_MESSAGE_DEPTH,
            }),
            "integer-model": limits(DecodeLimits {
                max_total_bytes: golden::RFC_INTEGER_MODEL_TOTAL,
                max_vec_len: golden::RFC_MESSAGE_VEC_LEN,
                max_depth: golden::RFC_MESSAGE_DEPTH,
            }),
        },
        "cases": golden::envelopes()
            .iter()
            .map(|c| {
                let mut case = json!({
                    "schema": c.expect.id.0,
                    "version": c.expect.version,
                    "type": c.ty,
                    "hex": hex(&c.bytes),
                });
                if let Some(name) = c.limits.name() {
                    case["limits"] = Value::from(name);
                }
                match &c.outcome {
                    Ok(payload) => case["payload"] = Value::from(hex(payload)),
                    Err((code, offset)) => {
                        case["code"] = Value::from(code.as_str());
                        case["offset"] = Value::from(*offset);
                    }
                }
                case["why"] = Value::from(c.why);
                case
            })
            .collect::<Vec<_>>(),
    });
    let pretty = |value: &Value| {
        let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
        text.push('\n');
        text
    };
    vec![
        ("spec/vectors/codec/golden.json", pretty(&golden)),
        ("spec/vectors/codec/noncanonical.json", pretty(&corpus)),
        ("spec/vectors/codec/envelope.json", pretty(&envelope)),
    ]
}

/// Writes the vectors, or with `check`, reports the files that differ from what would be
/// written; returns whether everything matched or was written.
pub(crate) fn run(repo: &Path, check: bool) -> io::Result<bool> {
    let mut current = true;
    for (path, contents) in render() {
        let target = repo.join(path);
        if check {
            if fs::read_to_string(&target).ok().as_deref() != Some(contents.as_str()) {
                println!("FAIL  {path} is out of date: run `cargo xtask gen-vectors`");
                current = false;
            } else {
                println!("ok    {path}");
            }
        } else {
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::write(&target, contents)?;
            println!("wrote {path}");
        }
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use crate::repo;

    #[test]
    fn codec_vectors_are_current() {
        for (path, contents) in super::render() {
            let committed = std::fs::read_to_string(repo::root().join(path))
                .unwrap_or_else(|e| panic!("{path}: {e}; run `cargo xtask gen-vectors`"));
            assert!(
                committed == contents,
                "{path} is out of date: run `cargo xtask gen-vectors`"
            );
        }
    }
}
