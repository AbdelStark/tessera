// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask gen-vectors`: writes the conformance vectors under `spec/vectors/` (RFC-0001
//! 'Shared artifacts that are not code'):
//!
//! - the codec vectors under `spec/vectors/codec/`, from the cases `tessera-codec` and
//!   `tessera-field` define, which the verifier and the Python package check with their own
//!   decoders;
//! - the files of every set with an independent Python oracle, `spec/vectors/<set>/generate.py`,
//!   by running the oracle.
//!
//! `--check` compares instead of writing: the codec vectors with what would be written, and the
//! oracles' files with what the oracles write when run on a copy of `spec/vectors/`, so a check
//! leaves the tree untouched. Tests run both comparisons, so vectors that no longer match their
//! cases or their oracle fail `cargo test`. The oracles need `python3`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::repo::copy_dir;

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
            fails with `code` at `offset`, the first byte the failing check reads, counted from \
            the start of the input; each fails for one reason, or for the pair whose order \
            RFC-0003 'Limits' fixes. Decoding each `accepted` entry, which sits exactly at a \
            limit, succeeds and encodes back to `hex`.",
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
                "offset": r.offset,
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
    let sets = oracles(repo)?;
    if check {
        let stale = stale_oracle_files(repo)?;
        for message in &stale {
            println!("FAIL  {message}");
        }
        if stale.is_empty() {
            println!(
                "ok    the oracles of {} reproduce their files",
                sets.join(", ")
            );
        }
        current &= stale.is_empty();
    } else {
        for set in &sets {
            run_oracle(repo, set)?.map_err(io::Error::other)?;
            println!("wrote spec/vectors/{set}/ with its oracle");
        }
    }
    Ok(current)
}

/// The sets with a Python oracle, each `spec/vectors/<set>/generate.py`, by name.
pub(crate) fn oracles(root: &Path) -> io::Result<Vec<String>> {
    let mut sets = Vec::new();
    for entry in fs::read_dir(root.join("spec/vectors"))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.path().join("generate.py").is_file() {
            sets.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    sets.sort();
    Ok(sets)
}

/// Runs the oracle of `set` in the tree at `root`, which writes that set's files. A failure to
/// start `python3` is an error; an oracle that fails comes back as its message.
fn run_oracle(root: &Path, set: &str) -> io::Result<Result<(), String>> {
    let script = root.join("spec/vectors").join(set).join("generate.py");
    let output = Command::new("python3")
        // Isolated from `PYTHON*` variables, the user's site packages and the working directory,
        // and writing no bytecode next to the oracles.
        .args(["-I", "-B"])
        .arg(&script)
        .current_dir(root)
        .output()
        .map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("cannot run python3, which the vector oracles need: {e}"),
            )
        })?;
    if output.status.success() {
        Ok(Ok(()))
    } else {
        Ok(Err(format!(
            "spec/vectors/{set}/generate.py failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

/// Every file under `dir`, by path relative to it, with its bytes.
fn tree(dir: &Path) -> io::Result<BTreeMap<PathBuf, Vec<u8>>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(rel) = pending.pop() {
        for entry in fs::read_dir(dir.join(&rel))? {
            let entry = entry?;
            let path = rel.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                pending.push(path);
            } else {
                files.insert(path, fs::read(entry.path())?);
            }
        }
    }
    Ok(files)
}

/// Runs every oracle on a copy of `spec/vectors/` under `root`, and returns what differs from the
/// tree as messages: an oracle that fails, and each file the oracles write differently, add or
/// remove. The tree itself is never written.
pub(crate) fn stale_oracle_files(root: &Path) -> io::Result<Vec<String>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let copy = std::env::temp_dir().join(format!("tessera-gen-vectors-{}-{n}", std::process::id()));
    // A stale directory can only come from a crashed earlier run with the same pid.
    let _ = fs::remove_dir_all(&copy);
    let outcome = compare_oracle_files(root, &copy);
    let _ = fs::remove_dir_all(&copy);
    outcome
}

fn compare_oracle_files(root: &Path, copy: &Path) -> io::Result<Vec<String>> {
    let vectors = root.join("spec/vectors");
    copy_dir(&vectors, &copy.join("spec/vectors"))?;
    let mut stale = Vec::new();
    for set in oracles(root)? {
        if let Err(message) = run_oracle(copy, &set)? {
            stale.push(message);
        }
    }
    let before = tree(&vectors)?;
    let after = tree(&copy.join("spec/vectors"))?;
    let paths: BTreeSet<&PathBuf> = before.keys().chain(after.keys()).collect();
    for path in paths {
        let shown = Path::new("spec/vectors").join(path).display().to_string();
        let set = path
            .components()
            .next()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_default();
        let rerun = format!("run `python3 spec/vectors/{set}/generate.py`");
        match (before.get(path), after.get(path)) {
            (Some(old), Some(new)) if old == new => {}
            (Some(_), Some(_)) => {
                stale.push(format!("{shown} is not what its oracle writes: {rerun}"))
            }
            (None, Some(_)) => stale.push(format!(
                "{shown} is written by its oracle but missing: {rerun}"
            )),
            (Some(_), None) => stale.push(format!("{shown} is removed by its oracle: {rerun}")),
            (None, None) => {}
        }
    }
    Ok(stale)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::repo;
    use crate::testing::Fixture;

    /// Every oracle reproduces its files byte for byte. The listing must find the oracles, or
    /// the comparison would pass with none.
    #[test]
    fn vector_oracles_reproduce_their_files() {
        let root = repo::root();
        let sets = super::oracles(&root).expect("list the oracles");
        for set in ["field", "hash", "transcript"] {
            assert!(sets.iter().any(|s| s == set), "{set} not among {sets:?}");
        }
        let stale = super::stale_oracle_files(&root).expect("run the oracles");
        assert!(stale.is_empty(), "{}", stale.join("\n"));
    }

    /// The check reports a stale file, a file missing from the tree and a failing oracle, leaves
    /// the tree as it was, and passes once the oracles have rewritten their files.
    #[test]
    fn stale_and_failing_oracles_are_reported() {
        let fixture = Fixture::new("vectors");
        let writes = |name: &str, text: &str| {
            format!(
                "from pathlib import Path\n(Path(__file__).resolve().parent / {name:?}).write_text({text:?})\n"
            )
        };
        fixture.write(
            "spec/vectors/current/generate.py",
            &writes("out.json", "1\n"),
        );
        fixture.write("spec/vectors/current/out.json", "1\n");
        fixture.write("spec/vectors/stale/generate.py", &writes("out.json", "2\n"));
        fixture.write("spec/vectors/stale/out.json", "old\n");
        fixture.write("spec/vectors/fresh/generate.py", &writes("new.json", "3\n"));
        fixture.write("spec/vectors/data/plain.json", "4\n");
        fixture.write(
            "spec/vectors/broken/generate.py",
            "raise SystemExit('no such case')\n",
        );
        let root = fixture.path();
        assert_eq!(
            super::oracles(root).unwrap(),
            ["broken", "current", "fresh", "stale"]
        );
        let stale = super::stale_oracle_files(root).unwrap();
        assert_eq!(stale.len(), 3, "{stale:#?}");
        assert!(stale[0].starts_with("spec/vectors/broken/generate.py failed"));
        assert!(stale[0].ends_with("no such case"), "{}", stale[0]);
        assert_eq!(
            stale[1],
            "spec/vectors/fresh/new.json is written by its oracle but missing: \
             run `python3 spec/vectors/fresh/generate.py`"
        );
        assert_eq!(
            stale[2],
            "spec/vectors/stale/out.json is not what its oracle writes: \
             run `python3 spec/vectors/stale/generate.py`"
        );
        assert_eq!(
            fs::read_to_string(root.join("spec/vectors/stale/out.json")).unwrap(),
            "old\n"
        );
        assert!(!root.join("spec/vectors/fresh/new.json").exists());

        // Writing stops at the failing oracle, and once it is fixed rewrites every set, after
        // which the check passes.
        let error = super::run(root, false).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("spec/vectors/broken/generate.py failed"),
            "{error}"
        );
        fixture.write("spec/vectors/broken/generate.py", "\n");
        assert!(super::run(root, false).unwrap());
        assert_eq!(
            fs::read_to_string(root.join("spec/vectors/stale/out.json")).unwrap(),
            "2\n"
        );
        assert_eq!(
            super::stale_oracle_files(root).unwrap(),
            Vec::<String>::new()
        );
        assert!(super::run(root, true).unwrap());
    }

    #[test]
    fn codec_vectors_are_current() {
        for (path, contents) in super::render() {
            let committed = fs::read_to_string(repo::root().join(path))
                .unwrap_or_else(|e| panic!("{path}: {e}; run `cargo xtask gen-vectors`"));
            assert!(
                committed == contents,
                "{path} is out of date: run `cargo xtask gen-vectors`"
            );
        }
    }
}
