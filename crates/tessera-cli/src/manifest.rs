// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The run manifest of 05 §6: `run-manifest.json`, `tessera.run-manifest/1`
//! (`spec/schemas/run-manifest-v1.schema.json`).
//!
//! `compile`, `commit-weights`, `prove` and `bench model --out` write one at [`crate::run`]'s
//! exit point, successful or not, in a directory of the operator's own: the output directory
//! that holds the private plan, beside the secret weight state, or beside the cost model's
//! output. `--run-manifest PATH` writes it elsewhere.
//!
//! The manifest is private-class (05 §3). Its timings depend on secret data, so it stays on the
//! prover's side of TB-1 (05 §4), never in a Live exchange directory or a report the verifier
//! reads. It holds no secret-class value: it names a secret-class artifact by schema and path,
//! never by digest, since a digest of secret data is derived secret data (OBS-6). Every other
//! artifact is named with its envelope digest. Public seeds are the coin-toss seeds a Live
//! transcript absorbs, never a per-proof secret seed.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value, json};
use tessera_codec::error::TesseraError;
use tessera_codec::metrics::{self, Sample};
use tessera_codec::schema::{self, Class, Schema};

/// What a command records as it runs.
struct Run {
    directory: Option<PathBuf>,
    params: Option<String>,
    protocol: Option<&'static str>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
    seeds: Vec<Value>,
}

static RUN: Mutex<Run> = Mutex::new(Run {
    directory: None,
    params: None,
    protocol: None,
    inputs: Vec::new(),
    outputs: Vec::new(),
    seeds: Vec::new(),
});

fn with<T>(f: impl FnOnce(&mut Run) -> T) -> T {
    f(&mut RUN.lock().unwrap_or_else(|e| e.into_inner()))
}

fn class(class: Class) -> &'static str {
    match class {
        Class::Public => "public",
        Class::Protocol => "protocol",
        Class::Private => "private",
        Class::Secret => "secret",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// An artifact entry: a secret-class artifact without its digest (OBS-6).
fn artifact(path: &Path, schema: &Schema, digest: Option<&[u8]>) -> Value {
    let digest = match schema.class {
        Class::Secret => None,
        _ => digest.map(hex),
    };
    json!({
        "path": path.display().to_string(),
        "schema": schema.name,
        "class": class(schema.class),
        "digest": digest,
    })
}

/// The envelope digest of the file at `path`: its last 32 bytes (RFC-0003 'Envelope').
fn envelope_digest(path: &Path) -> Option<[u8; 32]> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    file.seek(SeekFrom::End(-32)).ok()?;
    let mut digest = [0u8; 32];
    file.read_exact(&mut digest).ok()?;
    Some(digest)
}

/// Where the manifest goes unless `--run-manifest` says: `run-manifest.json` in `dir`.
pub(crate) fn directory(dir: &Path) {
    with(|run| run.directory = Some(dir.to_path_buf()));
}

/// Where the manifest goes: beside `path`, in its directory.
pub(crate) fn beside(path: &Path) {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    directory(dir.unwrap_or(Path::new(".")));
}

/// The run's parameter set and protocol.
pub(crate) fn params(id: &str, protocol: Option<&'static str>) {
    with(|run| {
        run.params = Some(id.to_owned());
        run.protocol = protocol;
    });
}

/// An artifact the run read and checked.
pub(crate) fn input(path: &Path, schema: &'static Schema) {
    let entry = artifact(path, schema, envelope_digest(path).as_ref().map(|d| &d[..]));
    with(|run| run.inputs.push(entry));
}

/// A TCE artifact the run wrote, recognized by its header (RFC-0003 'Envelope').
pub(crate) fn output_envelope(path: &Path, bytes: &[u8]) {
    let id = (bytes.len() >= 52 && &bytes[..4] == b"TSRA")
        .then(|| tessera_codec::error::SchemaId(u16::from_le_bytes([bytes[6], bytes[7]])));
    let Some(schema) = id.and_then(schema::by_id) else {
        return;
    };
    let entry = artifact(path, schema, Some(&bytes[bytes.len() - 32..]));
    with(|run| run.outputs.push(entry));
}

/// A TCE artifact the run wrote through its own writer, read back for its digest.
pub(crate) fn output_file(path: &Path, schema: &'static Schema) {
    let entry = artifact(path, schema, envelope_digest(path).as_ref().map(|d| &d[..]));
    with(|run| run.outputs.push(entry));
}

/// A JSON report the run wrote: public, named by its schema identifier.
pub(crate) fn output_report(path: &Path, report: &'static str) {
    let entry = json!({
        "path": path.display().to_string(),
        "schema": report,
        "class": "public",
        "digest": null,
    });
    with(|run| run.outputs.push(entry));
}

/// A public seed: one a protocol transcript absorbs.
pub(crate) fn seed(name: &'static str, digest: &[u8; 32]) {
    let entry = json!({"name": name, "digest": hex(digest)});
    with(|run| run.seeds.push(entry));
}

/// RFC-0020 'Continuous measurement': unit costs from this run's own metrics, recorded
/// before the snapshot (05 §4).
/// - `commit`: P1's commitment time per committed advice element.
/// - `lookup` and `product_leaf`: the leaf stage proves every lookup tree, table tree and
///   product forest in one batched sumcheck per level, so its time is shared per leaf: per
///   lookup, per table row and per product leaf alike.
/// - `glue`: the reduction stages' time as a fraction of commitment and leaf time, the factor
///   `1 + glue` that the cost model applies to them.
fn derive_unit_costs(samples: &[Sample]) -> Derived {
    let sum = |name: &str, label: Option<(&str, &str)>| -> f64 {
        samples
            .iter()
            .filter(|s| s.name == name)
            .filter(|s| label.is_none_or(|(k, v)| s.labels.iter().any(|(a, b)| *a == k && b == v)))
            .map(|s| match s.value {
                metrics::Value::Counter(n) => n as f64,
                metrics::Value::Histogram { sum, .. } => sum,
                metrics::Value::Gauge(g) => g,
            })
            .sum()
    };
    let commit = sum("tessera_phase_seconds", Some(("phase", "round1")));
    let advice = sum(
        "tessera_committed_elements_total",
        Some(("class", "advice")),
    );
    let leaf = sum("tessera_stage_seconds", Some(("stage", "0")));
    let leaves = sum("tessera_lookups_total", None)
        + sum("tessera_table_rows_total", None)
        + sum("tessera_product_leaves_total", None);
    let stages = sum("tessera_stage_seconds", None) - leaf;
    Derived {
        commit_ps: (commit > 0.0 && advice > 0.0).then(|| commit * 1e12 / advice),
        leaf_ps: (leaf > 0.0 && leaves > 0.0).then(|| leaf * 1e12 / leaves),
        glue: (commit + leaf > 0.0 && stages > 0.0).then(|| stages / (commit + leaf)),
    }
}

/// A run's unit costs, those its metrics determine.
#[derive(Debug, PartialEq)]
struct Derived {
    commit_ps: Option<f64>,
    leaf_ps: Option<f64>,
    glue: Option<f64>,
}

fn record(derived: &Derived) {
    if let Some(ps) = derived.commit_ps {
        metrics::unit_cost(metrics::UnitCost::Commit, ps);
    }
    if let Some(ps) = derived.leaf_ps {
        metrics::unit_cost(metrics::UnitCost::Lookup, ps);
        metrics::unit_cost(metrics::UnitCost::ProductLeaf, ps);
    }
    if let Some(glue) = derived.glue {
        metrics::glue_fraction(glue);
    }
}

fn snapshot_json(samples: &[Sample]) -> Vec<Value> {
    samples
        .iter()
        .map(|s| {
            let labels: Map<String, Value> = s
                .labels
                .iter()
                .map(|(k, v)| ((*k).to_owned(), json!(v)))
                .collect();
            let mut entry = json!({"name": s.name, "labels": labels});
            let fields = match s.value {
                metrics::Value::Counter(n) => json!({"type": "counter", "value": n}),
                metrics::Value::Gauge(g) => json!({"type": "gauge", "value": g}),
                metrics::Value::Histogram {
                    count,
                    sum,
                    min,
                    max,
                } => {
                    json!({"type": "histogram", "count": count, "sum": sum, "min": min, "max": max})
                }
            };
            for (k, v) in fields.as_object().into_iter().flatten() {
                entry[k] = v.clone();
            }
            entry
        })
        .collect()
}

/// The host fields of 05 §6, from what the operating system reports without privileges.
fn host() -> Value {
    let cores = std::thread::available_parallelism().map(|n| n.get()).ok();
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_model": cpu_model(),
        "cores": cores,
        "memory_bytes": memory_bytes(),
        "gpus": [],
        "ecc": null,
    })
}

fn sysctl(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// The value of the first `key:` line of a `/proc` file's text.
fn proc_field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        line.split_once(':')
            .filter(|(k, _)| k.trim() == key)
            .map(|(_, v)| v.trim())
    })
}

/// A `/proc` size such as `16384 kB`, in bytes.
fn kib(value: &str) -> Option<u64> {
    value
        .strip_suffix("kB")?
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn cpu_model() -> Option<String> {
    match std::env::consts::OS {
        "linux" => std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|t| proc_field(&t, "model name").map(str::to_owned)),
        "macos" => sysctl("machdep.cpu.brand_string"),
        _ => None,
    }
}

fn memory_bytes() -> Option<u64> {
    match std::env::consts::OS {
        "linux" => std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|t| proc_field(&t, "MemTotal").and_then(kib)),
        "macos" => sysctl("hw.memsize").and_then(|s| s.parse().ok()),
        _ => None,
    }
}

/// The process's peak resident memory, where `/proc` reports it (Linux).
fn host_memory_peak() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    proc_field(&text, "VmHWM").and_then(kib)
}

/// The manifest of a run of `command` with `arguments`, its outcome `error` (`None` on success).
pub(crate) fn render(
    command: &str,
    arguments: &[String],
    started: SystemTime,
    elapsed: Duration,
    error: Option<&TesseraError>,
) -> Value {
    if let Some(peak) = host_memory_peak() {
        metrics::host_memory_peak(peak);
    }
    record(&derive_unit_costs(&metrics::snapshot()));
    let samples = metrics::snapshot();
    let mut features = Vec::<&str>::new();
    if cfg!(feature = "test-rng") {
        features.push("test-rng");
    }
    if cfg!(feature = "otel") {
        features.push("otel");
    }
    let commit = option_env!("TESSERA_GIT_COMMIT").filter(|c| !c.is_empty());
    let rustc = option_env!("TESSERA_RUSTC_VERSION").filter(|c| !c.is_empty());
    with(|run| {
        json!({
            "schema": "tessera.run-manifest/1",
            "tool": {
                "name": "tessera",
                "version": env!("CARGO_PKG_VERSION"),
                "git_commit": commit,
                "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                "features": features,
            },
            "toolchain": {
                "rustc": rustc,
                "cuda_runtime": null,
                "cuda_driver": null,
                "python": null,
                "torch": null,
            },
            "host": host(),
            "command": command,
            "arguments": arguments,
            "params": run.params,
            "protocol": run.protocol,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "public_seeds": run.seeds,
            "started_at": crate::log::rfc3339(started),
            "elapsed_seconds": elapsed.as_secs_f64(),
            "metrics": snapshot_json(&samples),
            "exit_code": error.map_or(0, TesseraError::exit_code),
            "error_code": error.map(|e| e.code.as_str()),
        })
    })
}

/// Where this run's manifest goes: `explicit`, else the directory its command named.
pub(crate) fn path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(|| with(|run| run.directory.as_ref().map(|d| d.join("run-manifest.json"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_fields_and_sizes_parse() {
        let status = "Name:\ttessera\nVmPeak:\t  2048 kB\nVmHWM:\t    1536 kB\n";
        assert_eq!(proc_field(status, "VmHWM").and_then(kib), Some(1536 * 1024));
        assert_eq!(proc_field(status, "Missing"), None);
        assert_eq!(kib("12 MB"), None);
        let cpu = "processor\t: 0\nmodel name\t: AMD EPYC 9654 96-Core Processor\n";
        assert_eq!(
            proc_field(cpu, "model name"),
            Some("AMD EPYC 9654 96-Core Processor")
        );
    }

    #[test]
    fn unit_costs_divide_phase_time_by_counts() {
        let sample = |name, labels: &[(&'static str, &str)], value| Sample {
            name,
            labels: labels.iter().map(|(k, v)| (*k, (*v).to_owned())).collect(),
            value,
        };
        let seconds = |x| metrics::Value::Histogram {
            count: 1,
            sum: x,
            min: x,
            max: x,
        };
        let count = metrics::Value::Counter;
        let samples = [
            sample(
                "tessera_phase_seconds",
                &[("phase", "round1")],
                seconds(1.0),
            ),
            sample(
                "tessera_phase_seconds",
                &[("phase", "witness")],
                seconds(8.0),
            ),
            sample(
                "tessera_committed_elements_total",
                &[("class", "advice")],
                count(1_000_000),
            ),
            sample(
                "tessera_committed_elements_total",
                &[("class", "mask")],
                count(7),
            ),
            sample("tessera_stage_seconds", &[("stage", "0")], seconds(2.0)),
            sample("tessera_stage_seconds", &[("stage", "1")], seconds(0.25)),
            sample("tessera_stage_seconds", &[("stage", "2")], seconds(0.5)),
            sample("tessera_lookups_total", &[("table", "1")], count(600_000)),
            sample("tessera_lookups_total", &[("table", "2")], count(400_000)),
            sample(
                "tessera_table_rows_total",
                &[("table", "1")],
                count(2_000_000),
            ),
            sample("tessera_product_leaves_total", &[], count(1_000_000)),
        ];
        // 1 s over 10^6 elements; 2 s over 4·10^6 leaves; 0.75 s of stages over 3 s.
        assert_eq!(
            derive_unit_costs(&samples),
            Derived {
                commit_ps: Some(1e6),
                leaf_ps: Some(5e5),
                glue: Some(0.25)
            }
        );
        assert_eq!(
            derive_unit_costs(&samples[1..2]),
            Derived {
                commit_ps: None,
                leaf_ps: None,
                glue: None
            }
        );
    }

    #[test]
    fn secret_artifacts_carry_no_digest() {
        let digest = [7u8; 32];
        let secret = artifact(Path::new("m.d"), &schema::INTEGER_MODEL, Some(&digest));
        assert_eq!(secret["digest"], Value::Null);
        assert_eq!(secret["class"], "secret");
        let public = artifact(Path::new("c.tcl"), &schema::CLAIM_LIST, Some(&digest));
        assert_eq!(public["digest"], hex(&digest));
        let private = artifact(Path::new("p.tep"), &schema::EXECUTION_PLAN, Some(&digest));
        assert_eq!(private["class"], "private");
        assert_eq!(private["digest"], hex(&digest));
    }
}
