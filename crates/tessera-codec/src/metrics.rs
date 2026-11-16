// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The in-process metrics registry of 05 §4, which the run manifest snapshots (05 §6).
//!
//! Every metric has a fixed name, and its labels are enum variants, table identifiers or stage
//! numbers, so a sample can carry no secret-class value (05 §3 OBS-1). Recording takes a lock
//! once per phase or count, never per element. Phase and stage times depend on secret data,
//! so they and the unit costs derived from them stay on the prover's side of TB-1 (05 §4).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

/// A phase of a proof (`tessera_phase_seconds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// The witness pass.
    Witness,
    /// P1: the advice commitment, after the witness.
    Round1,
    /// The leaf stage and every reduction stage.
    Stage,
    /// The opening pass: combined rows of advice and weights.
    Opening,
    /// P5: column answers.
    Round3,
}

/// A pass of RFC-0018 'Pass structure' over the witness (`tessera_passes_total`,
/// `tessera_pass_seconds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pass {
    /// The witness pass and the advice commitment.
    P1,
    /// The leaf stage: its group pass when it is in token groups (RFC-0054).
    P2,
    /// A grouped leaf stage's merge pass (RFC-0054).
    P2b,
    /// One reduction stage's pass: its window's grid when it is windowed (RFC-0055).
    P3,
    /// A windowed reduction stage's fold pass (RFC-0055).
    P3b,
    /// The opening pass: the combined rows.
    P4,
    /// The column answers.
    P5,
}

/// A class of committed elements (`tessera_committed_elements_total`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Committed {
    /// Advice rows' entries.
    Advice,
    /// Weight rows' entries.
    Weight,
    /// Mask coefficients.
    Mask,
}

/// A part of a proof (`tessera_proof_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofPart {
    /// `Round1`.
    Round1,
    /// The tree levels, splits and stages.
    Sumcheck,
    /// The combined rows.
    Openings,
    /// `Round3`'s columns and multiproofs.
    Columns,
}

/// A unit cost (`tessera_unit_cost_ps`, RFC-0020).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnitCost {
    /// `c_c`, per committed element.
    Commit,
    /// `c_ℓ`, per lookup.
    Lookup,
    /// `c_π`, per product leaf.
    ProductLeaf,
}

/// The name of a label's value, as 05 §4 spells it.
fn name<T: core::fmt::Debug>(value: T) -> String {
    let text = format!("{value:?}");
    let mut out = String::new();
    for (i, c) in text.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// One recorded value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A counter's total.
    Counter(u64),
    /// A gauge's last value.
    Gauge(f64),
    /// A histogram's observations: how many, and their sum, least and greatest, in seconds.
    Histogram {
        /// Observations.
        count: u64,
        /// Their sum.
        sum: f64,
        /// The least.
        min: f64,
        /// The greatest.
        max: f64,
    },
}

/// One metric and label set with its value.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// The metric, `tessera_…`.
    pub name: &'static str,
    /// Its labels, by name.
    pub labels: Vec<(&'static str, String)>,
    /// Its value.
    pub value: Value,
}

type Key = (&'static str, Vec<(&'static str, String)>);

static REGISTRY: Mutex<BTreeMap<Key, Value>> = Mutex::new(BTreeMap::new());

fn record(
    name: &'static str,
    labels: Vec<(&'static str, String)>,
    update: impl FnOnce(&mut Option<Value>),
) {
    // A poisoned lock still holds the registry; metrics never stop a proof.
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let mut slot = registry.remove(&(name, labels.clone()));
    update(&mut slot);
    if let Some(value) = slot {
        registry.insert((name, labels), value);
    }
}

fn add(name: &'static str, labels: Vec<(&'static str, String)>, n: u64) {
    record(name, labels, |slot| {
        let total = match slot {
            Some(Value::Counter(t)) => t.saturating_add(n),
            _ => n,
        };
        *slot = Some(Value::Counter(total));
    });
}

fn set(name: &'static str, labels: Vec<(&'static str, String)>, value: f64) {
    record(name, labels, |slot| *slot = Some(Value::Gauge(value)));
}

fn observe(name: &'static str, labels: Vec<(&'static str, String)>, elapsed: Duration) {
    let x = elapsed.as_secs_f64();
    record(name, labels, |slot| {
        *slot = Some(match slot.take() {
            Some(Value::Histogram {
                count,
                sum,
                min,
                max,
            }) => Value::Histogram {
                count: count + 1,
                sum: sum + x,
                min: min.min(x),
                max: max.max(x),
            },
            _ => Value::Histogram {
                count: 1,
                sum: x,
                min: x,
                max: x,
            },
        });
    });
}

/// `tessera_phase_seconds{phase}`.
pub fn phase(phase: Phase, elapsed: Duration) {
    observe(
        "tessera_phase_seconds",
        vec![("phase", name(phase))],
        elapsed,
    );
}

/// `tessera_passes_total{pass}` and `tessera_pass_seconds{pass}`: one more pass, and its time
/// (RFC-0018 'Pass structure': the run manifest records every pass and its time). The device
/// traffic and memory peak since the previous pass ended are the pass's own:
/// `tessera_pass_device_bytes{pass}` and `tessera_pass_device_memory_peak_bytes{pass}`.
pub fn pass(pass: Pass, elapsed: Duration) {
    add("tessera_passes_total", vec![("pass", name(pass))], 1);
    observe("tessera_pass_seconds", vec![("pass", name(pass))], elapsed);
    let (bytes, peak) = {
        let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let taken = (window.bytes, window.peak);
        window.bytes = 0;
        window.peak = window.used;
        taken
    };
    if bytes > 0 {
        add(
            "tessera_pass_device_bytes",
            vec![("pass", name(pass))],
            bytes,
        );
    }
    if peak > 0 {
        record(
            "tessera_pass_device_memory_peak_bytes",
            vec![("pass", name(pass))],
            |slot| *slot = Some(Value::Gauge(gauge_max(slot, peak))),
        );
    }
}

/// A direction of a host–device copy (`tessera_device_bytes_total`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
}

/// Device traffic and memory since the last pass ended, which `pass` charges to that pass.
struct Window {
    bytes: u64,
    peak: u64,
    used: u64,
}

static WINDOW: Mutex<Window> = Mutex::new(Window {
    bytes: 0,
    peak: 0,
    used: 0,
});

fn gauge_max(slot: &Option<Value>, value: u64) -> f64 {
    match slot {
        Some(Value::Gauge(g)) => g.max(value as f64),
        _ => value as f64,
    }
}

/// `tessera_device_bytes_total{direction}`: one host–device copy of `bytes`.
pub fn device_transfer(direction: Direction, bytes: u64) {
    add(
        "tessera_device_bytes_total",
        vec![("direction", name(direction))],
        bytes,
    );
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    window.bytes = window.bytes.saturating_add(bytes);
}

/// `tessera_device_memory_peak_bytes{device}`: device `device` has `used` bytes allocated now.
pub fn device_memory(device: usize, used: u64) {
    record(
        "tessera_device_memory_peak_bytes",
        vec![("device", device.to_string())],
        |slot| *slot = Some(Value::Gauge(gauge_max(slot, used))),
    );
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    window.used = used;
    window.peak = window.peak.max(used);
}

/// `tessera_stage_seconds{stage}`.
pub fn stage(stage: u16, elapsed: Duration) {
    observe(
        "tessera_stage_seconds",
        vec![("stage", stage.to_string())],
        elapsed,
    );
}

/// `tessera_committed_elements_total{class}`.
pub fn committed(class: Committed, elements: u64) {
    add(
        "tessera_committed_elements_total",
        vec![("class", name(class))],
        elements,
    );
}

/// `tessera_lookups_total{table}`, by table identifier.
pub fn lookups(table: u32, lookups: u64) {
    add(
        "tessera_lookups_total",
        vec![("table", table.to_string())],
        lookups,
    );
}

/// `tessera_table_rows_total{table}`: the rows of a table's fraction tree, each a leaf priced as a
/// lookup (RFC-0020's table term).
pub fn table_rows(table: u32, rows: u64) {
    add(
        "tessera_table_rows_total",
        vec![("table", table.to_string())],
        rows,
    );
}

/// `tessera_product_leaves_total`.
pub fn product_leaves(leaves: u64) {
    add("tessera_product_leaves_total", Vec::new(), leaves);
}

/// `tessera_host_memory_peak_bytes`.
pub fn host_memory_peak(bytes: u64) {
    set("tessera_host_memory_peak_bytes", Vec::new(), bytes as f64);
}

/// `tessera_proof_bytes{part}`.
pub fn proof_bytes(part: ProofPart, bytes: u64) {
    set(
        "tessera_proof_bytes",
        vec![("part", name(part))],
        bytes as f64,
    );
}

/// `tessera_unit_cost_ps{kind}`.
pub fn unit_cost(kind: UnitCost, picoseconds: f64) {
    set(
        "tessera_unit_cost_ps",
        vec![("kind", name(kind))],
        picoseconds,
    );
}

/// `tessera_glue_fraction`: RFC-0020's glue, the sumcheck work beyond commitment and tree leaves.
pub fn glue_fraction(fraction: f64) {
    set("tessera_glue_fraction", Vec::new(), fraction);
}

/// Every recorded sample, by name then labels.
pub fn snapshot() -> Vec<Sample> {
    let registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry
        .iter()
        .map(|((name, labels), value)| Sample {
            name,
            labels: labels.clone(),
            value: value.clone(),
        })
        .collect()
}

/// Forgets every sample: a run starts empty.
pub fn reset() {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).clear();
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    (window.bytes, window.peak, window.used) = (0, 0, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_accumulate_by_name_and_labels() {
        // One test owns the process-wide registry.
        reset();
        committed(Committed::Advice, 4);
        committed(Committed::Advice, 6);
        committed(Committed::Weight, 1);
        lookups(7, 3);
        // Powers of two, so that the sums are exact.
        stage(1, Duration::from_millis(500));
        stage(1, Duration::from_millis(250));
        unit_cost(UnitCost::ProductLeaf, 2.5);
        unit_cost(UnitCost::ProductLeaf, 3.5);
        pass(Pass::P3, Duration::from_millis(500));
        pass(Pass::P3, Duration::from_millis(250));
        pass(Pass::P2b, Duration::from_millis(125));
        pass(Pass::P3b, Duration::from_millis(125));
        let samples = snapshot();
        let get = |n: &str, l: &[(&str, &str)]| {
            samples
                .iter()
                .find(|s| {
                    s.name == n
                        && s.labels
                            .iter()
                            .map(|(k, v)| (*k, v.as_str()))
                            .eq(l.iter().copied())
                })
                .map(|s| s.value.clone())
        };
        assert_eq!(
            get("tessera_committed_elements_total", &[("class", "advice")]),
            Some(Value::Counter(10))
        );
        assert_eq!(
            get("tessera_committed_elements_total", &[("class", "weight")]),
            Some(Value::Counter(1))
        );
        assert_eq!(
            get("tessera_lookups_total", &[("table", "7")]),
            Some(Value::Counter(3))
        );
        assert_eq!(
            get("tessera_stage_seconds", &[("stage", "1")]),
            Some(Value::Histogram {
                count: 2,
                sum: 0.75,
                min: 0.25,
                max: 0.5
            })
        );
        assert_eq!(
            get("tessera_unit_cost_ps", &[("kind", "product_leaf")]),
            Some(Value::Gauge(3.5))
        );
        // A pass counts once per call and keeps its times, by 05 §4's label.
        assert_eq!(
            get("tessera_passes_total", &[("pass", "p3")]),
            Some(Value::Counter(2))
        );
        assert_eq!(
            get("tessera_pass_seconds", &[("pass", "p3")]),
            Some(Value::Histogram {
                count: 2,
                sum: 0.75,
                min: 0.25,
                max: 0.5
            })
        );
        assert_eq!(
            get("tessera_passes_total", &[("pass", "p2b")]),
            Some(Value::Counter(1))
        );
        assert_eq!(
            get("tessera_passes_total", &[("pass", "p3b")]),
            Some(Value::Counter(1))
        );
        assert_eq!(samples.len(), 11);

        // Device traffic and memory go to the pass that ends next, and only to it.
        reset();
        device_memory(0, 100);
        device_transfer(Direction::HostToDevice, 64);
        device_memory(0, 300);
        device_memory(0, 200);
        device_transfer(Direction::DeviceToHost, 8);
        pass(Pass::P2, Duration::ZERO);
        device_transfer(Direction::HostToDevice, 16);
        device_memory(0, 250);
        pass(Pass::P3, Duration::ZERO);
        pass(Pass::P3, Duration::ZERO);
        pass(Pass::P4, Duration::ZERO);
        let samples = snapshot();
        let get = |n: &str, l: (&str, &str)| {
            samples
                .iter()
                .find(|s| s.name == n && s.labels == [(l.0, l.1.to_owned())])
                .map(|s| s.value.clone())
        };
        let p = |pass| ("pass", pass);
        assert_eq!(
            get("tessera_pass_device_bytes", p("p2")),
            Some(Value::Counter(72))
        );
        assert_eq!(
            get("tessera_pass_device_bytes", p("p3")),
            Some(Value::Counter(16))
        );
        assert_eq!(get("tessera_pass_device_bytes", p("p4")), None);
        let peak = "tessera_pass_device_memory_peak_bytes";
        assert_eq!(get(peak, p("p2")), Some(Value::Gauge(300.0)));
        // P3 starts with what P2 left allocated, 200 bytes, and grows to 250.
        assert_eq!(get(peak, p("p3")), Some(Value::Gauge(250.0)));
        // Memory still allocated through a pass counts toward its peak.
        assert_eq!(get(peak, p("p4")), Some(Value::Gauge(250.0)));
        assert_eq!(
            get("tessera_device_memory_peak_bytes", ("device", "0")),
            Some(Value::Gauge(300.0))
        );
        assert_eq!(
            get(
                "tessera_device_bytes_total",
                ("direction", "host_to_device")
            ),
            Some(Value::Counter(80))
        );
        reset();
        assert!(snapshot().is_empty());
    }
}
