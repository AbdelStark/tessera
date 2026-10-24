// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent public log events and check measurements, from 05 §2/§4/§5.
use crate::escape_json;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tv_core::driver::CheckEvent;

pub(crate) struct Observation {
    pub(crate) started: SystemTime,
    pub(crate) events: Vec<(SystemTime, CheckEvent)>,
    pub(crate) proof_digest: Option<String>,
    pub(crate) inputs: Vec<Artifact>,
    pub(crate) params: Option<String>,
}
pub(crate) struct Artifact {
    pub(crate) path: String,
    pub(crate) schema: &'static str,
    pub(crate) class: &'static str,
    pub(crate) digest: String,
}
impl Observation {
    pub(crate) fn new() -> Self {
        Self {
            started: SystemTime::now(),
            events: Vec::new(),
            proof_digest: None,
            inputs: Vec::new(),
            params: None,
        }
    }
    pub(crate) fn record(&mut self, event: CheckEvent) {
        self.events.push((SystemTime::now(), event));
    }
    pub(crate) fn artifact(
        &mut self,
        path: &str,
        schema: &'static str,
        class: &'static str,
        bytes: &[u8],
    ) {
        // Called only after canonical envelope validation, so its final bytes are the digest.
        let digest = hex(&bytes[bytes.len() - 32..]);
        if schema == "NiProof" {
            self.proof_digest = Some(digest.clone());
        }
        self.inputs.push(Artifact {
            path: path.into(),
            schema,
            class,
            digest,
        });
    }
    pub(crate) fn logs(&self, exit_code: i32) -> String {
        let mut out = String::new();
        out.push_str(&self.event(self.started, "verify_started", &[], "", "info"));
        let mut active = Vec::new();
        for &(ts, event) in &self.events {
            match event {
                CheckEvent::Started { kind } => {
                    active.push(kind);
                    out.push_str(&self.event(ts, "check_started", &active, "", "info"));
                }
                CheckEvent::Finished {
                    duration,
                    completed,
                    ..
                } => {
                    let fields = format!(
                        "\"duration_ns\":{},\"outcome\":\"{}\"",
                        nanos(duration),
                        if completed { "completed" } else { "error" }
                    );
                    out.push_str(&self.event(
                        ts,
                        "check_finished",
                        &active,
                        &fields,
                        if completed { "info" } else { "error" },
                    ));
                    active.pop();
                }
            }
        }
        out.push_str(&self.event(
            SystemTime::now(),
            "verify_finished",
            &[],
            &format!("\"exit_code\":{exit_code}"),
            if exit_code == 0 { "info" } else { "error" },
        ));
        out
    }
    fn event(
        &self,
        ts: SystemTime,
        event: &str,
        kinds: &[&str],
        fields: &str,
        level: &str,
    ) -> String {
        let proof = self
            .proof_digest
            .as_ref()
            .map_or(String::new(), |d| format!("\"proof_digest\":\"{d}\""));
        let check = kinds
            .iter()
            .map(|k| format!(",{{\"name\":\"check\",\"fields\":{{\"kind\":\"{k}\"}}}}"))
            .collect::<String>();
        format!(
            "{{\"ts\":\"{}\",\"level\":\"{level}\",\"target\":\"tv_cli::verify\",\"event\":\"tessera.verifier.{event}\",\"span_path\":[{{\"name\":\"verify\",\"fields\":{{{proof}}}}}{check}],\"fields\":{{{fields}}}}}\n",
            timestamp(ts)
        )
    }
    pub(crate) fn metrics(&self) -> String {
        self.events.iter().filter_map(|(_, event)| match event {
            CheckEvent::Finished { kind, duration, .. } => {
                let seconds = duration.as_secs_f64();
                Some(format!("{{\"name\":\"tessera_verifier_seconds\",\"labels\":{{\"check\":\"{kind}\"}},\"type\":\"histogram\",\"count\":1,\"sum\":{seconds},\"min\":{seconds},\"max\":{seconds}}}"))
            }
            _ => None,
        }).collect::<Vec<_>>().join(",")
    }
}
pub(crate) fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub(crate) fn quoted(s: &str) -> String {
    format!("\"{}\"", escape_json(s))
}
pub(crate) fn optional(s: Option<&str>) -> String {
    s.map_or("null".into(), quoted)
}

// Gregorian civil date from Unix days, split into 400-year eras; no timezone dependency.
pub(crate) fn timestamp(time: SystemTime) -> String {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs();
    let days = (seconds / 86400) as i64 + 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60,
        elapsed.subsec_micros()
    )
}
