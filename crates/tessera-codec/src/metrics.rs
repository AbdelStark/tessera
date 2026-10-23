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
        assert_eq!(samples.len(), 5);
        reset();
        assert!(snapshot().is_empty());
    }
}
